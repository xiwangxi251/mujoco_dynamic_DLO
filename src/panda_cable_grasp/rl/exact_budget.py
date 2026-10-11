"""Finish a scratch PPO budget exactly, including a partial vector step."""
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


def _partial_step(env, actions):
    """Step only the counted workers. All other workers remain untouched."""
    monitor = env if isinstance(env, VecMonitor) else None
    raw = env.venv if monitor else env
    if not isinstance(raw, SubprocVecEnv):
        raise TypeError("Final partial step requires a spawned SubprocVecEnv")
    n = len(actions)
    for remote, action in zip(raw.remotes[:n], actions):
        remote.send(("step", action))
    results = [remote.recv() for remote in raw.remotes[:n]]
    obs, rewards, dones, infos, resets = zip(*results)
    raw.reset_infos = list(raw.reset_infos)
    raw.reset_infos[:n] = resets
    rewards, dones, infos = np.asarray(rewards), np.asarray(dones), list(infos)
    if monitor:
        monitor.episode_returns[:n] += rewards
        monitor.episode_lengths[:n] += 1
        for i, done in enumerate(dones):
            if done:
                info = dict(infos[i])
                episode = dict(r=monitor.episode_returns[i], l=monitor.episode_lengths[i],
                               t=round(time.time()-monitor.t_start, 6))
                episode.update({k: info[k] for k in monitor.info_keywords})
                info["episode"] = episode
                infos[i] = info
                monitor.episode_count += 1
                monitor.episode_returns[i] = 0
                monitor.episode_lengths[i] = 0
                if monitor.results_writer:
                    monitor.results_writer.write_row(episode)
    return np.stack(obs), rewards, dones, infos


def install_exact_budget():
    if getattr(PPO, "_exact_budget_installed", False):
        return
    original = PPO.collect_rollouts

    def collect(self, env, callback, rollout_buffer, n_rollout_steps):
        remaining = self._total_timesteps-self.num_timesteps
        steps = min(n_rollout_steps, remaining//self.n_envs)
        n = self.n_envs if steps else remaining
        size = steps if steps else 1
        if rollout_buffer.buffer_size != size or rollout_buffer.n_envs != n:
            self.rollout_buffer = self.rollout_buffer_class(size, self.observation_space, self.action_space,
                device=self.device, gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=n,
                **self.rollout_buffer_kwargs)
            rollout_buffer = self.rollout_buffer
        if steps:
            return original(self, env, callback, rollout_buffer, steps)
        assert 1 < n < self.n_envs, "Scratch sample budget must leave at least two samples for the final PPO batch"
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        old_obs = self._last_obs[:n]
        with torch.no_grad():
            actions, values, log_probs = self.policy(obs_as_tensor(old_obs, self.device))
        actions = actions.cpu().numpy()
        clipped = np.clip(actions, self.action_space.low, self.action_space.high)
        new_obs, rewards, dones, infos = _partial_step(env, clipped)
        self.num_timesteps += n
        callback.update_locals(locals())
        if not callback.on_step():
            return False
        self._update_info_buffer(infos, dones)
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None and infos[i].get("TimeLimit.truncated", False):
                with torch.no_grad():
                    terminal_value = self.policy.predict_values(self.policy.obs_to_tensor(infos[i]["terminal_observation"])[0])[0]
                rewards[i] += self.gamma*terminal_value
        rollout_buffer.add(old_obs, actions, rewards, self._last_episode_starts[:n], values, log_probs)
        self._last_obs[:n] = new_obs
        self._last_episode_starts[:n] = dones
        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    PPO.collect_rollouts = collect
    PPO._exact_budget_installed = True
