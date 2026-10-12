import tempfile
import time
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from panda_cable_grasp.rl.async_sampling import ReadyWorkerVecEnv, ReadyWorkerPPO, worker_gae


class SlowResetEnv(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, (2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    render_mode = None
    def __init__(self, slow=False):
        self.slow = slow; self.resets = 0; self.steps = 0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1; self.steps = 0
        if self.slow and self.resets > 1: time.sleep(2.)
        return np.array([float(self.slow), 0.], dtype=np.float32), {}
    def step(self, action):
        self.steps += 1; time.sleep(.002)
        return np.array([float(self.slow), self.steps], dtype=np.float32), 1., False, self.steps >= (2 if self.slow else 7), dict(
            scenario_name='slow' if self.slow else 'fast', rollout_episode_phase=0, rollout_episode_difficulty=0.)
    def set_rollout_curriculum(self, *args): pass
    def set_penalty_ramp_progress(self, *args): pass


class AuditCallback(BaseCallback):
    def __init__(self):
        super().__init__(); self.records = []; self.timings = []; self.episodes = []
    def _on_step(self):
        self.episodes.extend(info['episode'] for info in self.locals['infos'] if 'episode' in info)
        return True
    def _on_rollout_start(self): self.start = time.monotonic()
    def _on_rollout_end(self):
        self.timings.append(time.monotonic()-self.start)
        self.records.append(dict(self.model.async_sampling_record))
        assert all(kind == 'reset' for kind in self.training_env.pending.values())
        assert self.model.async_sampling_record['actions_pending_at_update'] == 0
        self.training_env.env_method('set_rollout_curriculum', 1)
        self.training_env.env_method('set_penalty_ramp_progress', self.num_timesteps)
        buffer = self.model.rollout_buffer
        with torch.no_grad():
            values, probs, _ = self.model.policy.evaluate_actions(
                torch.as_tensor(buffer.observations[:, 0]), torch.as_tensor(buffer.actions[:, 0]))
        np.testing.assert_allclose(probs.numpy(), buffer.log_probs[:, 0], atol=1e-6)
        np.testing.assert_allclose(values.numpy().flatten(), buffer.values[:, 0], atol=1e-6)


class AsyncSamplingTests(unittest.TestCase):
    def test_gae_matches_independent_sb3_chains(self):
        # Two interleaved chains, including a terminal and a timeout reward bootstrap.
        workers = np.array([0, 1, 0, 1, 0, 1])
        rewards = np.array([1., 10., 2., 20.+.9*8., 3., 30.])
        values = np.array([.1, 1., .2, 2., .3, 3.])
        next_values = np.array([.2, 2., .3, 8., .4, 4.])
        dones = np.array([False, False, True, True, False, False])
        actual = worker_gae(rewards, values, next_values, dones, workers, .9, .8)
        space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        for worker in (0, 1):
            ix = np.where(workers == worker)[0]
            buffer = RolloutBuffer(3, space, space, gamma=.9, gae_lambda=.8, n_envs=1)
            for j, k in enumerate(ix):
                buffer.add(np.zeros((1, 1)), np.zeros((1, 1)), rewards[k:k+1],
                    np.array([j == 0 or dones[ix[j-1]]]), torch.tensor([values[k]]), torch.tensor([0.]))
            buffer.compute_returns_and_advantage(torch.tensor([next_values[ix[-1]]]), dones[ix[-1:]])
            np.testing.assert_allclose(actual[ix], buffer.advantages[:, 0], rtol=1e-6)

    def test_slow_reset_does_not_hold_batch_and_exact_partial_reload(self):
        torch.set_num_threads(1)
        env = ReadyWorkerVecEnv([lambda: SlowResetEnv(False), lambda: SlowResetEnv(True)])
        callback = AuditCallback()
        try:
            model = ReadyWorkerPPO('MlpPolicy', env, n_steps=8, batch_size=4, n_epochs=1,
                policy_kwargs={'net_arch': dict(pi=[8], vf=[8])}, device='cpu', seed=5)
            model.learn(37, callback=callback)
            self.assertEqual(model.num_timesteps, 37)
            self.assertEqual([r['samples'] for r in callback.records], [16, 16, 5])
            self.assertTrue(all(t < 1.5 for t in callback.timings), callback.timings)
            self.assertGreater(callback.records[0]['worker_samples']['0'], callback.records[0]['worker_samples'].get('1', 0))
            self.assertTrue(any(r['resets_pending_at_update'] > 0 for r in callback.records))
            self.assertTrue(callback.episodes)
            self.assertTrue(all(ep['r'] == ep['l'] for ep in callback.episodes))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'model.zip'; model.save(path)
                loaded = ReadyWorkerPPO.load(path, env=env, force_reset=False, device='cpu')
                self.assertEqual(loaded.num_timesteps, 37)
                for key, value in model.policy.state_dict().items():
                    self.assertTrue(torch.equal(value, loaded.policy.state_dict()[key]))
        finally: env.close()


if __name__ == '__main__': unittest.main()
