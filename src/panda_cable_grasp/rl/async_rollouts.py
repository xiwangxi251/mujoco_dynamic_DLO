"""Readiness-driven PPO with fixed worker quotas and live aligned callbacks.

Each full batch contains n_steps from every worker. Actions share one policy;
only scheduling changes. SB3 workers still return terminal observations and
reset information. Setters issued by callbacks run between that worker's steps.
"""
from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
from multiprocessing.connection import wait
from pathlib import Path
import time

from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

VERSION = 'async_fixed_quota_v2'


def _copy_obs(obs, indices=None):
    if isinstance(obs, dict):
        return {k: _copy_obs(v, indices) for k, v in obs.items()}
    return np.array(obs if indices is None else obs[indices], copy=True)


def _stack_obs(rows):
    if isinstance(rows[0], dict):
        return {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    return np.stack(rows)


def _finite(obs):
    return all(_finite(v) for v in obs.values()) if isinstance(obs, dict) else np.isfinite(obs).all()


def policy_digest(policy):
    digest = hashlib.sha256()
    for name, value in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FixedQuotaVecEnv(SubprocVecEnv):
    """Standard SB3 workers, with queued setters during independent sampling."""
    def __init__(self, *args, **kwargs):
        self.collecting = False
        self.sampler_failed = False
        super().__init__(*args, **kwargs)
        self.control_queue = [deque() for _ in range(self.num_envs)]

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if not self.collecting:
            return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)
        if not method_name.startswith('set_'):
            raise RuntimeError('Blocking worker query during async sampling; use --sampler sync: '+method_name)
        selected = list(self._get_indices(indices))
        for i in selected:
            self.control_queue[i].append(deepcopy((method_name, method_args, method_kwargs)))
        return [None for _ in selected]

    def _guard_query(self):
        if self.collecting:
            raise RuntimeError('Blocking worker query during async sampling; use --sampler sync')

    def get_attr(self, *args, **kwargs):
        self._guard_query()
        return super().get_attr(*args, **kwargs)

    def set_attr(self, *args, **kwargs):
        self._guard_query()
        return super().set_attr(*args, **kwargs)

    def env_is_wrapped(self, *args, **kwargs):
        self._guard_query()
        return super().env_is_wrapped(*args, **kwargs)

    def has_attr(self, *args, **kwargs):
        self._guard_query()
        return super().has_attr(*args, **kwargs)

    def close(self):
        if self.closed:
            return
        if self.collecting or self.sampler_failed:
            # A failed worker cannot participate in SubprocVecEnv's blocking close.
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for process in self.processes:
                process.join(5)
                if process.is_alive():
                    process.kill(); process.join(5)
            for remote in self.remotes:
                remote.close()
            self.closed = True
        else:
            super().close()


class SampleCheckpointCallback(BaseCallback):
    """Checkpoint intervals count returned transitions, not readiness events."""
    def __init__(self, interval, directory):
        super().__init__()
        self.interval = int(interval)
        self.directory = Path(directory)

    def _on_training_start(self):
        self.next_sample = (self.num_timesteps//self.interval+1)*self.interval

    def _on_step(self):
        if self.num_timesteps >= self.next_sample:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.model.save(self.directory/f'ppo_cable_{self.num_timesteps}_steps')
            self.next_sample = (self.num_timesteps//self.interval+1)*self.interval
        return True


def worker_gae(rewards, values, dones, worker_ids, last_values, gamma, gae_lambda):
    """GAE along each worker's chronological chain, including partial tail quotas."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    following = np.array(last_values, copy=True)
    tails = np.zeros(len(last_values), dtype=np.float32)
    for j in reversed(range(len(rewards))):
        i = worker_ids[j]
        continuation = 1.-float(dones[j])
        delta = rewards[j]+gamma*following[i]*continuation-values[j]
        advantages[j] = delta+gamma*gae_lambda*continuation*tails[i]
        following[i] = values[j]
        tails[i] = advantages[j]
    return advantages


class FixedQuotaPPO(PPO):
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if not isinstance(env, VecMonitor) or not isinstance(env.venv, FixedQuotaVecEnv):
            raise TypeError('Fixed-quota PPO requires VecMonitor(FixedQuotaVecEnv)')
        if self.use_sde or not isinstance(self.action_space, spaces.Box):
            raise ValueError('Fixed-quota PPO requires Box actions and use_sde=False')
        raw = env.venv
        raw.reset_infos = list(raw.reset_infos)
        count = self.n_envs
        remaining = int(self._total_timesteps-self.num_timesteps)
        target = min(n_rollout_steps*count, remaining)
        # Avoid leaving a singleton optimization batch at the end of the budget.
        if remaining-target == 1 and target > 2:
            target -= 1
        if target < 2:
            raise ValueError('PPO needs at least two remaining samples')
        quotas = np.full(count, target//count, dtype=np.int64)
        quotas[:target % count] += 1
        self.rollout_buffer = buffer = self.rollout_buffer_class(target,
            self.observation_space, self.action_space, device=self.device,
            gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=1,
            **self.rollout_buffer_kwargs)
        worker_ids = np.empty(target, dtype=np.int64)
        terminal = np.empty(target, dtype=bool)
        latest_obs = _copy_obs(self._last_obs)
        latest_starts = self._last_episode_starts.copy()
        completed = np.zeros(count, dtype=np.int64)
        pending = {}
        received = 0
        keep_going = True
        maximum_lead = 0
        scene_counts = Counter()
        self.policy.set_training_mode(False)
        frozen_policy = policy_digest(self.policy)
        callback.on_rollout_start()
        started = last_progress = time.monotonic()
        raw.collecting = True
        try:
            while True:
                available = []
                for i, connection in enumerate(raw.remotes):
                    if connection in pending:
                        continue
                    if raw.control_queue[i]:
                        connection.send(('env_method', raw.control_queue[i].popleft()))
                        pending[connection] = ('control', i, None)
                    elif keep_going and completed[i] < quotas[i]:
                        available.append(i)
                if available:
                    old_obs = _copy_obs(latest_obs, available)
                    starts = latest_starts[available].copy()
                    with torch.no_grad():
                        acts, vals, probs = self.policy(obs_as_tensor(old_obs, self.device))
                    actions = acts.cpu().numpy()
                    vals = vals.cpu().numpy().flatten()
                    probs = probs.cpu().numpy().flatten()
                    clipped = (self.policy.unscale_action(actions) if self.policy.squash_output
                               else np.clip(actions, self.action_space.low, self.action_space.high))
                    for j, i in enumerate(available):
                        connection = raw.remotes[i]
                        pending[connection] = ('step', i, (_copy_obs(old_obs, j),
                            actions[j].copy(), starts[j], vals[j], probs[j]))
                        connection.send(('step', clipped[j]))
                if not pending:
                    break
                ready = wait(list(pending), timeout=1.)
                if not ready:
                    if any(not p.is_alive() for p in raw.processes):
                        raise RuntimeError('Async simulation worker exited')
                rows = []
                for connection in ready:
                    kind, i, data = pending.pop(connection)
                    result = connection.recv()
                    if kind == 'control':
                        continue
                    new_obs, reward, done, info, reset_info = result
                    if not _finite(new_obs) or not np.isfinite(reward):
                        raise FloatingPointError('Nonfinite async transition')
                    raw.reset_infos[i] = reset_info
                    if isinstance(latest_obs, dict):
                        for key in latest_obs:
                            latest_obs[key][i] = new_obs[key]
                    else:
                        latest_obs[i] = new_obs
                    latest_starts[i] = done
                    env.episode_returns[i] += reward
                    env.episode_lengths[i] += 1
                    info = dict(info)
                    if done:
                        episode = dict(r=env.episode_returns[i], l=env.episode_lengths[i],
                                       t=round(time.time()-env.t_start, 6))
                        episode.update({k: info[k] for k in env.info_keywords})
                        info['episode'] = episode
                        env.episode_count += 1
                        env.episode_returns[i] = 0; env.episode_lengths[i] = 0
                        if env.results_writer:
                            env.results_writer.write_row(episode)
                    rows.append((i, data, new_obs, float(reward), bool(done), info))
                    completed[i] += 1
                if rows:
                    indices = np.array([r[0] for r in rows], dtype=np.int64)
                    actions = np.stack([r[1][1] for r in rows])
                    new_obs = _stack_obs([r[2] for r in rows])
                    rewards = np.array([r[3] for r in rows], dtype=np.float32)
                    dones = np.array([r[4] for r in rows], dtype=bool)
                    infos = [r[5] for r in rows]
                    self.num_timesteps += len(rows)
                    self._last_obs = _copy_obs(latest_obs)
                    self._last_episode_starts = latest_starts.copy()
                    callback.update_locals(dict(actions=actions, rewards=rewards,
                        new_obs=new_obs, dones=dones, infos=infos, worker_indices=indices))
                    # Once stopped, drain issued steps and count them; never train this batch.
                    keep_going = bool(callback.on_step()) and keep_going
                    self._update_info_buffer(infos, dones)
                    for j, (i, data, _, _, done, info) in enumerate(rows):
                        obs, action, start, value, log_prob = data
                        reward = rewards[j]
                        if done and info.get('TimeLimit.truncated', False):
                            with torch.no_grad():
                                terminal_obs = self.policy.obs_to_tensor(info['terminal_observation'])[0]
                                reward += self.gamma*float(self.policy.predict_values(terminal_obs)[0].item())
                        # One transition per slot; chronology is tracked by worker_ids.
                        if isinstance(buffer.observations, dict):
                            for key in obs:
                                buffer.observations[key][received, 0] = obs[key]
                        else:
                            buffer.observations[received, 0] = obs
                        buffer.actions[received, 0] = action
                        buffer.episode_starts[received, 0] = start
                        buffer.values[received, 0] = value
                        buffer.log_probs[received, 0] = log_prob
                        buffer.rewards[received, 0] = reward
                        worker_ids[received] = i; terminal[received] = done
                        scene_counts[(info.get('scenario_name', 'unknown'),
                            info.get('rollout_episode_phase'), info.get('rollout_episode_difficulty'))] += 1
                        received += 1
                    maximum_lead = max(maximum_lead, int(completed.max()-completed.min()))
                now = time.monotonic()
                if now-last_progress >= 10:
                    self._write_progress(completed, quotas, now-started)
                    last_progress = now
            self._write_progress(completed, quotas, time.monotonic()-started)
            if policy_digest(self.policy) != frozen_policy:
                raise RuntimeError('Policy changed during fixed-policy sampling')
        except BaseException:
            raw.sampler_failed = True
            raise
        finally:
            raw.collecting = False
        self.async_rollout_receipt = dict(version=VERSION, workers=count,
            per_worker_steps=completed.tolist(), per_worker_quotas=quotas.tolist(),
            transitions=received, policy_unchanged=True, no_pending_steps=True,
            maximum_worker_lead=maximum_lead, stopped_early=not keep_going,
            sampling_seconds=time.monotonic()-started,
            scene_samples=[dict(scenario=k[0], phase=k[1], difficulty=k[2], samples=v)
                           for k, v in scene_counts.items()], callback_order='ready_aligned_worker_indices')
        if not keep_going:
            return False
        assert received == target and np.array_equal(completed, quotas)
        with torch.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(latest_obs, self.device)).cpu().numpy().flatten()
        buffer.advantages[:, 0] = worker_gae(buffer.rewards[:, 0], buffer.values[:, 0],
            terminal, worker_ids, last_values, self.gamma, self.gae_lambda)
        buffer.returns[:] = buffer.advantages+buffer.values
        buffer.pos = target; buffer.full = True
        callback.update_locals(dict(rollout_buffer=buffer))
        callback.on_rollout_end()
        return True

    def _write_progress(self, completed, quotas, seconds):
        path = getattr(self, 'sampler_progress_path', None)
        if path is not None:
            path = Path(path)
            record = dict(updated_at=time.time(), version=VERSION, timesteps=self.num_timesteps,
                returned_unoptimized_samples=int(completed.sum()), target_samples=int(quotas.sum()),
                per_worker_steps=completed.tolist(), sampling_seconds=seconds,
                formal_optimizer_progress=False)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(record), encoding='utf-8')
            temporary.replace(path)
