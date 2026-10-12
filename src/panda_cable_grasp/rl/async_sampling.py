"""On-policy, fixed-total-batch PPO with independently ready simulation workers.

Terminal transitions are returned BEFORE reset. Resets may overlap policy updates,
but actions never do: every dispatched action is drained into its original batch.
GAE follows each worker's own chronological chain, not arrival order.
"""
import multiprocessing as mp
from multiprocessing.connection import wait
import time
from collections import Counter

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv, CloudpickleWrapper
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.monitor import ResultsWriter
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

VERSION = 'ready_worker_fixed_batch_v1'


class SampleCheckpointCallback(BaseCallback):
    """Checkpoint by actual returned samples, independent of event batch size."""
    def __init__(self, interval, directory, starting_steps):
        super().__init__()
        self.interval = int(interval)
        self.directory = Path(directory)
        self.next_sample = (int(starting_steps)//self.interval+1)*self.interval

    def _on_step(self):
        if self.num_timesteps >= self.next_sample:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.model.save(self.directory/f'ppo_cable_{self.num_timesteps}_steps')
            self.next_sample = (self.num_timesteps//self.interval+1)*self.interval
        return True


def _worker(remote, parent, factory):
    parent.close()
    env = _patch_env(factory.var())
    try:
        while True:
            command, data = remote.recv()
            if command == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info = dict(info)
                info['TimeLimit.truncated'] = truncated and not terminated
                if done:
                    info['terminal_observation'] = obs
                # No implicit reset here: it is a separate asynchronous command.
                remote.send((obs, reward, done, info))
            elif command == 'reset':
                seed, options = data
                obs, info = env.reset(seed=seed, **({'options': options} if options else {}))
                status = (env.get_wrapper_attr('future_prefetch_status')()
                          if env.has_wrapper_attr('future_prefetch_status') else None)
                remote.send((obs, info, status))
            elif command == 'method_nowait':
                name, args, kwargs = data
                env.get_wrapper_attr(name)(*args, **kwargs)
            elif command == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif command == 'env_method':
                name, args, kwargs = data
                remote.send(env.get_wrapper_attr(name)(*args, **kwargs))
            elif command == 'get_attr':
                remote.send(env.get_wrapper_attr(data))
            elif command == 'set_attr':
                env.set_wrapper_attr(data[0], data[1]); remote.send(None)
            elif command == 'is_wrapped':
                from stable_baselines3.common.env_util import is_wrapped
                remote.send(is_wrapped(env, data))
            elif command == 'close':
                break
            else:
                raise RuntimeError('Unknown async worker command: '+command)
    except EOFError:
        pass
    finally:
        env.close()
        remote.close()


class ReadyWorkerVecEnv(VecEnv):
    """VecEnv control interface plus partial, independently ready step collection."""
    def __init__(self, factories, filename=None, info_keywords=(), start_method='spawn'):
        context = mp.get_context(start_method)
        self.remotes, peers = zip(*[context.Pipe() for _ in factories])
        self.processes = []
        self.closed = False
        for remote, peer, factory in zip(self.remotes, peers, factories):
            process = context.Process(target=_worker,
                args=(peer, remote, CloudpickleWrapper(factory)), daemon=True)
            process.start(); self.processes.append(process); peer.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError('Ready-worker PPO currently supports flat Box observations only')
        self.pending = {}
        self.ready = set()
        self.obs = np.zeros((len(factories), *observation_space.shape), dtype=observation_space.dtype)
        self.starts = np.ones(len(factories), dtype=bool)
        self.prefetch = [dict(hits=0, misses=0, errors=0, wait_seconds=0., pending_pid=None)
                         for _ in factories]
        self.episode_returns = np.zeros(len(factories), dtype=np.float64)
        self.episode_lengths = np.zeros(len(factories), dtype=np.int64)
        self.info_keywords = info_keywords
        self.t_start = time.time()
        self.results_writer = ResultsWriter(filename, header={'t_start': self.t_start, 'env_id': None},
                                             extra_keys=info_keywords) if filename else None
        super().__init__(len(factories), observation_space, action_space)

    def reset(self):
        if self.pending:
            raise RuntimeError('Explicit full reset requires no outstanding commands')
        self.ready.clear()
        for i in range(self.num_envs):
            self.remotes[i].send(('reset', (self._seeds[i], self._options[i])))
            self.pending[i] = 'reset'
        while self.pending:
            assert not self.receive(), 'No actions may be pending in full reset'
        self._reset_seeds(); self._reset_options()
        self.episode_returns[:] = 0; self.episode_lengths[:] = 0
        return self.obs.copy()

    def dispatch(self, indices, actions):
        for i, action in zip(indices, actions, strict=True):
            assert i in self.ready and i not in self.pending
            self.ready.remove(i)
            self.remotes[i].send(('step', action)); self.pending[i] = 'step'

    def receive(self, timeout=60.):
        if not self.pending:
            return []
        selected = wait([self.remotes[i] for i in self.pending], timeout=timeout)
        if not selected:
            dead = [i for i in self.pending if not self.processes[i].is_alive()]
            if dead:
                raise RuntimeError('Async simulation worker exited: '+str(dead))
            return []
        arrivals = []
        for i in list(self.pending):
            if self.remotes[i] not in selected:
                continue
            kind = self.pending.pop(i)
            result = self.remotes[i].recv()
            if kind == 'reset':
                obs, info, prefetch = result
                self.obs[i] = obs; self.starts[i] = True; self.reset_infos[i] = info
                if prefetch is not None:
                    self.prefetch[i] = prefetch
                self.ready.add(i)
            else:
                obs, reward, done, info = result
                self.episode_returns[i] += reward; self.episode_lengths[i] += 1
                if done:
                    episode = dict(r=float(self.episode_returns[i]), l=int(self.episode_lengths[i]),
                                   t=round(time.time()-self.t_start, 6))
                    episode.update({key: info[key] for key in self.info_keywords})
                    info['episode'] = episode
                    if self.results_writer:
                        self.results_writer.write_row(episode)
                    self.episode_returns[i] = 0; self.episode_lengths[i] = 0
                    # Launch reset only AFTER emitting the terminal transition.
                    self.remotes[i].send(('reset', (None, {}))); self.pending[i] = 'reset'
                else:
                    self.obs[i] = obs; self.starts[i] = False; self.ready.add(i)
                arrivals.append((i, obs, reward, done, info))
        return arrivals

    def step_async(self, actions):
        if self.pending or len(self.ready) != self.num_envs:
            raise RuntimeError('Use independent dispatch/receive during async collection')
        self.dispatch(list(range(self.num_envs)), actions)

    def step_wait(self):
        results = {}
        while len(results) != self.num_envs:
            for row in self.receive(): results[row[0]] = row[1:]
        while self.pending:
            assert not self.receive()
        rows = [results[i] for i in range(self.num_envs)]
        return self.obs.copy(), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]), [r[3] for r in rows]

    def _query(self, command, payload, indices):
        indices = list(self._get_indices(indices))
        if self.pending:
            raise RuntimeError('Blocking RPC prohibited while async resets are pending: '+command)
        for i in indices: self.remotes[i].send((command, payload))
        return [self.remotes[i].recv() for i in indices]

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        selected = list(self._get_indices(indices))
        if method_name == 'future_prefetch_status':
            return [dict(self.prefetch[i]) for i in selected]
        if method_name in ('set_rollout_curriculum', 'set_penalty_ramp_progress'):
            for i in selected:
                self.remotes[i].send(('method_nowait', (method_name, method_args, method_kwargs)))
            # Pipe ordering ensures commands run before the next dispatched action.
            # An already resetting episode keeps its original phase/difficulty tags.
            return [None for _ in selected]
        return self._query('env_method', (method_name, method_args, method_kwargs), selected)

    def get_attr(self, attr_name, indices=None):
        return self._query('get_attr', attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        self._query('set_attr', (attr_name, value), indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self._query('is_wrapped', wrapper_class, indices)

    def close(self):
        if self.closed: return
        # Formal final budget drains every action; only reset work can remain.
        for remote in self.remotes:
            try: remote.send(('close', None))
            except (BrokenPipeError, EOFError, OSError): pass
        until = time.monotonic()+150
        for process in self.processes:
            process.join(max(0., until-time.monotonic()))
        for process in self.processes:
            if process.is_alive(): process.terminate(); process.join(5)
            if process.is_alive(): process.kill(); process.join(5)
        for remote in self.remotes: remote.close()
        if self.results_writer: self.results_writer.close()
        self.closed = True


def worker_gae(rewards, values, next_values, dones, worker_ids, gamma, gae_lambda):
    """Compute independent chronological GAE chains in interleaved arrival data."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    tails = {}
    for j in reversed(range(len(rewards))):
        worker = int(worker_ids[j])
        continuation = 1.-float(dones[j])
        delta = rewards[j]+gamma*next_values[j]*continuation-values[j]
        advantages[j] = delta+gamma*gae_lambda*continuation*tails.get(worker, 0.)
        tails[worker] = advantages[j]
    return advantages


class ReadyWorkerPPO(PPO):
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if not isinstance(env, ReadyWorkerVecEnv) or self.use_sde:
            raise ValueError('Ready-worker collector requires ReadyWorkerVecEnv and use_sde=False')
        target = min(n_rollout_steps*self.n_envs, self._total_timesteps-self.num_timesteps)
        if target < 2:
            raise ValueError('Final PPO batch must contain at least two samples')
        self.rollout_buffer = buffer = RolloutBuffer(target, self.observation_space, self.action_space,
            self.device, self.gae_lambda, self.gamma, n_envs=1)
        worker_ids = np.empty(target, dtype=np.int32)
        next_values = np.empty(target, dtype=np.float32)
        terminal = np.empty(target, dtype=bool)
        self.policy.set_training_mode(False)
        callback.on_rollout_start()
        dispatched = received = 0
        inflight = {}
        scene_counts = Counter(); worker_counts = Counter()
        batch_started = time.monotonic()
        while received < target:
            available = sorted(env.ready)[:target-dispatched]
            if available:
                old_obs = env.obs[available].copy()
                starts = env.starts[available].copy()
                with torch.no_grad():
                    acts, vals, probs = self.policy(obs_as_tensor(old_obs, self.device))
                actions = acts.cpu().numpy(); vals = vals.cpu().numpy().flatten(); probs = probs.cpu().numpy().flatten()
                clipped = (self.policy.unscale_action(actions) if self.policy.squash_output
                           else np.clip(actions, self.action_space.low, self.action_space.high))
                for j, i in enumerate(available):
                    inflight[i] = (old_obs[j], actions[j], starts[j], vals[j], probs[j])
                env.dispatch(available, clipped); dispatched += len(available)
            arrivals = env.receive()
            if not arrivals: continue
            indices = [row[0] for row in arrivals]
            new_obs = np.stack([row[1] for row in arrivals])
            rewards = np.array([row[2] for row in arrivals], dtype=np.float32)
            dones = np.array([row[3] for row in arrivals], dtype=bool)
            infos = [row[4] for row in arrivals]
            if not np.isfinite(new_obs).all() or not np.isfinite(rewards).all():
                raise FloatingPointError('Nonfinite async transition')
            with torch.no_grad():
                following = self.policy.predict_values(obs_as_tensor(new_obs, self.device)).cpu().numpy().flatten()
            self.num_timesteps += len(arrivals)
            # Callbacks see only the transitions actually returned in this event.
            callback.update_locals(locals())
            if not callback.on_step():
                raise RuntimeError('Early callback stop requires a separately accounted drain; refusing to discard actions')
            self._update_info_buffer(infos, dones)
            for j, i in enumerate(indices):
                obs, action, start, value, log_prob = inflight.pop(i)
                reward = rewards[j]
                if dones[j] and infos[j].get('TimeLimit.truncated', False):
                    reward += self.gamma*following[j]
                p = received+j
                buffer.observations[p, 0] = obs; buffer.actions[p, 0] = action
                buffer.episode_starts[p, 0] = start; buffer.values[p, 0] = value
                buffer.log_probs[p, 0] = log_prob; buffer.rewards[p, 0] = reward
                worker_ids[p] = i; next_values[p] = following[j]; terminal[p] = dones[j]
                worker_counts[i] += 1
                key = (infos[j].get('scenario_name', 'unknown'), infos[j].get('rollout_episode_phase'),
                       infos[j].get('rollout_episode_difficulty'))
                scene_counts[key] += 1
            received += len(arrivals)
        assert received == dispatched == target and not inflight
        assert all(kind == 'reset' for kind in env.pending.values())
        buffer.advantages[:, 0] = worker_gae(buffer.rewards[:, 0], buffer.values[:, 0],
            next_values, terminal, worker_ids, self.gamma, self.gae_lambda)
        buffer.returns[:] = buffer.advantages+buffer.values
        buffer.pos = target; buffer.full = True
        self._last_obs = env.obs.copy(); self._last_episode_starts = env.starts.copy()
        self.async_sampling_record = dict(version=VERSION, samples=target,
            worker_samples={str(k): v for k, v in sorted(worker_counts.items())},
            scene_samples=[dict(scenario=k[0], phase=k[1], difficulty=k[2], samples=v)
                           for k, v in scene_counts.items()],
            resets_pending_at_update=len(env.pending), actions_pending_at_update=0,
            policy_fixed_during_batch=True, sampling_seconds=time.monotonic()-batch_started)
        callback.update_locals(locals()); callback.on_rollout_end()
        return True
