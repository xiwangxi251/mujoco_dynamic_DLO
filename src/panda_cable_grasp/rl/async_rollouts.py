"""C PPO: readiness-driven sampling, fixed per-environment quotas, one policy.

Callbacks consume completed transitions in the original time-major/env-major
order after collection. They are statistics/after-update curriculum callbacks;
callbacks that change the environment or policy during sampling are unsupported.
"""
import hashlib
from multiprocessing.connection import wait
import time

import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


def policy_digest(policy):
    digest = hashlib.sha256()
    for name, value in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def monitor_transition(monitor, index, reward, done, info):
    """Same episode accounting as VecMonitor, for one ready environment."""
    monitor.episode_returns[index] += reward
    monitor.episode_lengths[index] += 1
    info = dict(info)
    if done:
        episode = dict(r=monitor.episode_returns[index], l=monitor.episode_lengths[index],
                       t=round(time.time()-monitor.t_start, 6))
        episode.update({key: info[key] for key in monitor.info_keywords})
        info['episode'] = episode
        monitor.episode_count += 1
        monitor.episode_returns[index] = 0
        monitor.episode_lengths[index] = 0
        if monitor.results_writer:
            monitor.results_writer.write_row(episode)
    return info


def collect_fixed_quota(model, env, callback, rollout_buffer, n_rollout_steps):
    if not isinstance(env, VecMonitor) or not isinstance(env.venv, SubprocVecEnv):
        raise TypeError('Async C sampler requires VecMonitor(SubprocVecEnv)')
    if model.use_sde:
        raise ValueError('Async C sampler does not change SDE noise semantics')
    raw = env.venv
    assert not raw.waiting and model._last_obs is not None
    raw.reset_infos = list(raw.reset_infos)
    count = env.num_envs
    remaining = model._total_timesteps-model.num_timesteps
    if remaining % count:
        raise ValueError('C async budget must be divisible by worker count')
    quota = min(n_rollout_steps, remaining//count)
    assert quota > 0
    if rollout_buffer.buffer_size != quota:
        model.rollout_buffer = model.rollout_buffer_class(quota, model.observation_space, model.action_space,
            device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=count,
            **model.rollout_buffer_kwargs)
        rollout_buffer = model.rollout_buffer
    model.policy.set_training_mode(False)
    frozen_policy = policy_digest(model.policy)
    rollout_buffer.reset()
    callback.on_rollout_start()
    latest_obs = np.array(model._last_obs, copy=True)
    latest_starts = np.array(model._last_episode_starts, copy=True)
    completed = np.zeros(count, dtype=np.int64)
    transitions = [[] for _ in range(count)]
    pending = {}
    started = time.monotonic()
    last_progress = started
    maximum_pending = 0
    maximum_worker_lead = 0

    def dispatch(indices):
        nonlocal maximum_pending
        with torch.no_grad():
            actions, values, log_probs = model.policy(obs_as_tensor(latest_obs[indices], model.device))
        actions = actions.cpu().numpy()
        clipped = model.policy.unscale_action(actions) if model.policy.squash_output else np.clip(
            actions, model.action_space.low, model.action_space.high)
        for j, i in enumerate(indices):
            assert completed[i] < quota and raw.remotes[i] not in pending
            pending[raw.remotes[i]] = (i, latest_obs[i].copy(), actions[j].copy(),
                bool(latest_starts[i]), float(values[j].item()), float(log_probs[j].item()))
            raw.remotes[i].send(('step', clipped[j]))
        maximum_pending = max(maximum_pending, len(pending))

    dispatch(list(range(count)))
    while pending:
        ready = wait(list(pending), timeout=120)
        if not ready:
            raise TimeoutError('No async environment completed a step in 120 seconds')
        continue_indices = []
        for connection in ready:
            i, old_obs, action, episode_start, value, log_prob = pending.pop(connection)
            new_obs, reward, done, info, reset_info = connection.recv()
            if not np.isfinite(new_obs).all() or not np.isfinite(reward):
                raise FloatingPointError('Nonfinite async environment transition')
            raw.reset_infos[i] = reset_info
            info = monitor_transition(env, i, reward, done, info)
            transitions[i].append((old_obs, action, episode_start, value, log_prob,
                                   np.array(new_obs, copy=True), float(reward), bool(done), info))
            latest_obs[i], latest_starts[i] = new_obs, done
            completed[i] += 1
            maximum_worker_lead = max(maximum_worker_lead, int(completed.max()-completed.min()))
            if completed[i] < quota:
                continue_indices.append(i)
        if continue_indices:
            dispatch(continue_indices)
        now = time.monotonic()
        if now-last_progress >= 10:
            _progress(model, completed, quota, frozen_policy, now-started)
            last_progress = now
    sampling_seconds = time.monotonic()-started
    assert np.all(completed == quota) and policy_digest(model.policy) == frozen_policy
    _progress(model, completed, quota, frozen_policy, sampling_seconds)

    # Restore the same rectangular trajectory layout and callback event order.
    # The fast environment's quota never displaces samples from a slow scene.
    for step in range(quota):
        row = [transitions[i][step] for i in range(count)]
        old_obs = np.stack([x[0] for x in row])
        actions = np.stack([x[1] for x in row])
        episode_starts = np.asarray([x[2] for x in row], dtype=bool)
        values = torch.tensor([x[3] for x in row], device=model.device)
        log_probs = torch.tensor([x[4] for x in row], device=model.device)
        new_obs = np.stack([x[5] for x in row])
        rewards = np.asarray([x[6] for x in row], dtype=np.float32)
        dones = np.asarray([x[7] for x in row], dtype=bool)
        infos = [x[8] for x in row]
        model.num_timesteps += count
        callback.update_locals(dict(actions=actions, rewards=rewards, new_obs=new_obs, dones=dones, infos=infos))
        if not callback.on_step():
            # All workers already advanced; a stopped rollout must never be optimized.
            model._last_obs, model._last_episode_starts = latest_obs, latest_starts
            return False
        model._update_info_buffer(infos, dones)
        for i, done in enumerate(dones):
            if done and infos[i].get('TimeLimit.truncated', False) and infos[i].get('terminal_observation') is not None:
                with torch.no_grad():
                    terminal_value = model.policy.predict_values(model.policy.obs_to_tensor(infos[i]['terminal_observation'])[0])[0]
                rewards[i] += model.gamma*float(terminal_value.item())
        rollout_buffer.add(old_obs, actions, rewards, episode_starts, values, log_probs)
    model._last_obs, model._last_episode_starts = latest_obs, latest_starts
    with torch.no_grad():
        last_values = model.policy.predict_values(obs_as_tensor(latest_obs, model.device))
    rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=latest_starts)
    model.async_rollout_receipt = dict(sampler='async_fixed_quota_v1', workers=count,
        per_worker_steps=completed.tolist(), transitions=count*quota, policy_sha256=frozen_policy,
        policy_unchanged=True, sampling_seconds=sampling_seconds, max_pending=maximum_pending,
        callback_order='time_major_env_major', no_pending_steps=True)
    model.async_rollout_receipt['maximum_worker_lead'] = maximum_worker_lead
    callback.on_rollout_end()
    return True


def _progress(model, completed, quota, digest, seconds):
    cb = getattr(model, '_speed_callback', None)
    if cb is not None:
        from .speed_curriculum import atomic
        atomic(cb.output/'sampler_progress.json', dict(updated_at=time.time(),
            sampler='async_fixed_quota_v1', model_global_steps_at_rollout_start=model.num_timesteps,
            sampled_unoptimized_transitions=int(completed.sum()), target_transitions=len(completed)*quota,
            per_worker_steps=completed.tolist(), policy_sha256=digest, sampling_seconds=seconds,
            formal_optimized_progress=False))
