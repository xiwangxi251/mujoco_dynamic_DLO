import types
import time
import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from panda_cable_grasp.rl.async_rollouts import (
    FixedQuotaPPO, FixedQuotaVecEnv, SampleCheckpointCallback,
)


class CounterEnv(gym.Env):
    def __init__(self, index, delay=0., dictionary=False):
        self.index, self.delay, self.dictionary = index, delay, dictionary
        box = gym.spaces.Box(-10000, 10000, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({'state': box}) if dictionary else box
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.total = self.step_count = self.controls = 0

    def observation(self):
        obs = np.array([self.index, self.step_count], dtype=np.float32)
        return {'state': obs} if self.dictionary else obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self.observation(), {}

    def step(self, action):
        time.sleep(self.delay)
        self.step_count += 1
        self.total += 1
        terminated = self.index == 0 and self.step_count == 3
        truncated = self.index == 1 and self.step_count == 4
        return self.observation(), float(self.step_count), terminated, truncated, dict(
            index=self.index, action=float(action[0]), scenario_name=f'scene{self.index}',
            rollout_episode_phase=0, rollout_episode_difficulty=0.)

    def set_test_control(self, value):
        self.controls = value


class Capture(BaseCallback):
    def __init__(self, stop=False, control=False):
        super().__init__()
        self.samples = 0
        self.stop, self.control = stop, control
        self.receipts = []
        self.locals_aligned = []

    def _on_step(self):
        ids = self.locals.get('worker_indices', np.arange(self.training_env.num_envs))
        assert len(ids) == len(self.locals['actions']) == len(self.locals['infos'])
        for j, info in enumerate(self.locals['infos']):
            assert info['index'] == ids[j]
            assert info['action'] == pytest.approx(float(np.clip(self.locals['actions'][j, 0], -1, 1)))
        self.samples += len(ids)
        self.locals_aligned.append(ids.copy())
        assert self.num_timesteps == self.samples
        if self.control and self.n_calls == 1:
            self.training_env.env_method('set_test_control', 7)
        return not self.stop

    def _on_rollout_end(self):
        receipt = getattr(self.model, 'async_rollout_receipt', None)
        if receipt:
            self.receipts.append(dict(receipt))
        if self.control:
            assert self.training_env.get_attr('controls') == [7]*self.training_env.num_envs


def model_env(asynchronous, delays=(0., 0.), dictionary=False):
    torch.set_num_threads(1)
    cls = FixedQuotaVecEnv if asynchronous else SubprocVecEnv
    factories = [lambda i=i, d=d: CounterEnv(i, d, dictionary) for i, d in enumerate(delays)]
    env = VecMonitor(cls(factories, start_method='spawn'))
    model_cls = FixedQuotaPPO if asynchronous else PPO
    model = model_cls('MultiInputPolicy' if dictionary else 'MlpPolicy', env,
        n_steps=8, batch_size=4, n_epochs=1, seed=123, device='cpu',
        policy_kwargs=dict(net_arch=[8]))
    return model, env


def collect(asynchronous, delays=(0., 0.)):
    model, env = model_env(asynchronous, delays)
    # Deterministic actions isolate scheduling and compare the actual GAE chains.
    def forward(policy, obs, deterministic=False):
        return torch.zeros((len(obs), 1)), policy.predict_values(obs), torch.zeros(len(obs))
    model.policy.forward = types.MethodType(forward, model.policy)
    try:
        cb = Capture()
        model._setup_learn(16)
        cb.init_callback(model); cb.on_training_start({}, {})
        assert model.collect_rollouts(env, cb, model.rollout_buffer, 8)
        b = model.rollout_buffer
        # Standard SB3 is time-major, async is arrival-major; compare each worker's chain.
        obs = b.observations.reshape(-1, 2)
        ix = np.argsort(obs[:, 0], kind='stable')
        result = {key: np.array(getattr(b, key)).reshape(len(obs), -1)[ix] for key in (
            'observations', 'actions', 'rewards', 'episode_starts', 'values', 'log_probs',
            'advantages', 'returns')}
        result['totals'] = env.get_attr('total')
        result['receipt'] = getattr(model, 'async_rollout_receipt', None)
        return result
    finally:
        env.close()


def test_fixed_quotas_match_sync_gae_terminals_and_timeout():
    sync = collect(False)
    asynchronous = collect(True, (.001, .025))
    for key in sync.keys()-{'totals', 'receipt'}:
        np.testing.assert_allclose(asynchronous[key], sync[key], atol=1e-6)
    assert asynchronous['totals'] == sync['totals'] == [8, 8]
    assert asynchronous['receipt']['maximum_worker_lead'] > 1
    assert asynchronous['receipt']['policy_unchanged']
    assert asynchronous['receipt']['no_pending_steps']


@pytest.mark.parametrize('budget,batches', [(37, [16, 16, 5]), (33, [16, 15, 2])])
def test_exact_tail_finite_updates_and_reload(tmp_path, budget, batches):
    model, env = model_env(True)
    cb = Capture(control=True)
    try:
        model.learn(budget, callback=cb)
        assert model.num_timesteps == sum(env.get_attr('total')) == budget
        assert [r['transitions'] for r in cb.receipts] == batches
        for r in cb.receipts:
            assert r['per_worker_steps'] == r['per_worker_quotas']
            assert max(r['per_worker_steps'])-min(r['per_worker_steps']) <= 1
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        model.save(tmp_path/'model')
        loaded = FixedQuotaPPO.load(tmp_path/'model', device='cpu')
        assert loaded.num_timesteps == budget and loaded._n_updates == len(batches)
        assert loaded.policy.optimizer.state
        # Evaluation and ordinary PPO loading do not require the training subclass.
        plain = PPO.load(tmp_path/'model', device='cpu')
        obs = np.array([0., 0.], dtype=np.float32)
        np.testing.assert_array_equal(loaded.predict(obs, deterministic=True)[0],
                                      plain.predict(obs, deterministic=True)[0])
    finally:
        env.close()


def test_early_stop_drains_and_counts_issued_steps_without_update():
    model, env = model_env(True, (.001, .03))
    cb = Capture(stop=True)
    try:
        model.learn(16, callback=cb)
        assert model.num_timesteps == sum(env.get_attr('total')) == cb.samples
        assert 0 < model.num_timesteps < 16 and model._n_updates == 0
        assert model.async_rollout_receipt['stopped_early']
        assert model.async_rollout_receipt['no_pending_steps']
    finally:
        env.close()


def test_dict_observations_and_sample_checkpoint_interval(tmp_path):
    model, env = model_env(True, (.001, .02), dictionary=True)
    cb = Capture()
    checkpoint = SampleCheckpointCallback(5, tmp_path)
    try:
        model.learn(21, callback=CallbackList([cb, checkpoint]))
        assert model.num_timesteps == sum(env.get_attr('total')) == 21
        saved = sorted(tmp_path.glob('ppo_cable_*_steps.zip'))
        steps = sorted(int(p.stem.split('_')[2]) for p in saved)
        assert len(saved) == 4 and all(5*i <= n <= 5*i+1 for i, n in enumerate(steps, 1))
        assert isinstance(model.rollout_buffer.observations, dict)
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    finally:
        env.close()


def test_default_cli_and_sync_opt_out(monkeypatch):
    from panda_cable_grasp.rl.train import parse_args
    monkeypatch.setattr('sys.argv', ['train'])
    assert parse_args().sampler == 'async_fixed_quota'
    monkeypatch.setattr('sys.argv', ['train', '--sampler', 'sync'])
    assert parse_args().sampler == 'sync'


def test_tail_smaller_than_worker_count():
    model, env = model_env(True, (0., 0., 0.))
    try:
        model.learn(2)
        assert model.num_timesteps == 2 and env.get_attr('total') == [1, 1, 0]
        assert model.async_rollout_receipt['per_worker_steps'] == [1, 1, 0]
    finally:
        env.close()


def test_arm_diagnostics_keep_worker_identity_across_partial_arrivals(tmp_path):
    from panda_cable_grasp.rl.arm_diagnostics import ArmSpeedDiagnostics
    diagnostic = ArmSpeedDiagnostics(tmp_path)
    diagnostic.num_timesteps = 3
    for worker, done in [(7, False), (2, False), (7, True)]:
        info = dict(reward_test=1., actual_hand_linear_velocity=np.zeros(3),
            actual_hand_angular_velocity=np.zeros(3), actual_arm_velocity=np.zeros(6),
            commanded_world_translation=np.zeros(3))
        diagnostic.locals = dict(actions=np.zeros((1, 5)), infos=[info], dones=[done],
                                 worker_indices=np.array([worker]))
        assert diagnostic._on_step()
    record = json.loads((tmp_path/'arm_episode_diagnostics.jsonl').read_text())
    assert record['worker'] == 7 and record['reward_component_sums']['reward_test'] == 2.
    assert diagnostic.reward_sums[2]['reward_test'] == 1.
