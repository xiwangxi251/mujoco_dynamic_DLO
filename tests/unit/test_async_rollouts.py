import time
import types

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from panda_cable_grasp.rl.speed_curriculum import SpeedPPO


class CounterEnv(gym.Env):
    def __init__(self, index, delay=0):
        self.index, self.delay, self.step_count, self.total = index, delay, 0, 0
        self.observation_space = gym.spaces.Box(-10000, 10000, (2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.array([self.index, 0], dtype=np.float32), {}

    def step(self, action):
        time.sleep(self.delay)
        self.step_count += 1
        self.total += 1
        terminated = self.index == 0 and self.step_count == 3
        truncated = self.index == 1 and self.step_count == 4
        return np.array([self.index, self.step_count], dtype=np.float32), float(self.step_count), terminated, truncated, {'index': self.index}


class Capture(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.update_counts = []

    def _on_step(self):
        self.rows.append((self.num_timesteps, self.locals['dones'].copy(),
                          [dict(x.get('episode', {})) for x in self.locals['infos']]))
        self.update_counts.append(self.model._n_updates)
        return True


def make_model(async_enabled, delays=(0, 0)):
    env = VecMonitor(SubprocVecEnv([lambda: CounterEnv(0, delays[0]), lambda: CounterEnv(1, delays[1])], start_method='spawn'))
    m = SpeedPPO('MlpPolicy', env, n_steps=8, batch_size=4, n_epochs=1, seed=123,
                 policy_kwargs=dict(net_arch=[8]), device='cpu')
    m.async_rollouts_enabled = async_enabled
    # Deterministic actions isolate scheduling from global RNG draw ordering.
    def forward(policy, obs, deterministic=False):
        return torch.zeros((len(obs), 1)), policy.predict_values(obs), torch.zeros(len(obs))
    m.policy.forward = types.MethodType(forward, m.policy)
    return m, env


def collect(async_enabled, delays=(0, 0)):
    model, env = make_model(async_enabled, delays)
    try:
        cb = Capture()
        model._setup_learn(16)
        cb.init_callback(model)
        cb.on_training_start({}, {})
        assert model.collect_rollouts(env, cb, model.rollout_buffer, 8)
        b = model.rollout_buffer
        result = {key: np.array(getattr(b, key), copy=True) for key in
                  ('observations', 'actions', 'rewards', 'episode_starts', 'values', 'log_probs', 'advantages', 'returns')}
        result.update(rows=cb.rows, updates=cb.update_counts, last_obs=model._last_obs.copy(),
                      last_starts=model._last_episode_starts.copy(),
                      worker_totals=env.get_attr('total'), receipt=getattr(model, 'async_rollout_receipt', None))
        return result
    finally:
        env.close()


def test_async_matches_sync_trajectories_timeout_bootstrap_and_gae():
    sync, asynchronous = collect(False), collect(True, (.02, 0))
    for key in ('observations', 'actions', 'rewards', 'episode_starts', 'values', 'log_probs', 'advantages', 'returns', 'last_obs', 'last_starts'):
        np.testing.assert_allclose(asynchronous[key], sync[key], atol=1e-6)
    assert asynchronous['worker_totals'] == sync['worker_totals'] == [8, 8]
    assert asynchronous['updates'] == [0]*8
    assert [x[0] for x in asynchronous['rows']] == list(range(2, 17, 2))
    for a, s in zip(asynchronous['rows'], sync['rows']):
        np.testing.assert_array_equal(a[1], s[1])
        assert [(x.get('r'), x.get('l')) for x in a[2]] == [(x.get('r'), x.get('l')) for x in s[2]]
    assert asynchronous['receipt']['policy_unchanged']
    assert asynchronous['receipt']['per_worker_steps'] == [8, 8]
    assert asynchronous['receipt']['no_pending_steps']
    assert asynchronous['receipt']['maximum_worker_lead'] > 1


def test_exact_short_final_rollout_and_finite_update_reload(tmp_path):
    model, env = make_model(True)
    try:
        cb = Capture()
        model.learn(total_timesteps=20, callback=cb)
        assert model.num_timesteps == 20 and model._n_updates == 2
        assert model.rollout_buffer.buffer_size == 2
        assert env.get_attr('total') == [10, 10]
        assert model.async_rollout_receipt['per_worker_steps'] == [2, 2]
        assert all(torch.isfinite(x).all() for x in model.policy.parameters())
        model.save(tmp_path/'model')
        restored = SpeedPPO.load(tmp_path/'model.zip', device='cpu')
        assert restored.async_rollouts_enabled and restored.num_timesteps == 20
        assert restored._n_updates == 2 and restored.policy.optimizer.state
    finally:
        env.close()


def test_reject_nondivisible_budget_before_advancing_workers():
    model, env = make_model(True)
    try:
        cb = Capture()
        model._setup_learn(17)
        cb.init_callback(model)
        with pytest.raises(ValueError, match='divisible'):
            model.collect_rollouts(env, cb, model.rollout_buffer, 8)
        assert env.get_attr('total') == [0, 0]
    finally:
        env.close()
