from __future__ import annotations

import numpy as np

from rl.train_rl import StrictSuccessEvalCallback


class _FakeModel:
    def predict(self, observations, *, deterministic):
        assert deterministic
        return np.zeros((len(observations), 1), dtype=np.float32), None


class _FakeEvalVecEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.seed_calls: list[int] = []
        self._seed = 0

    def seed(self, seed: int):
        self._seed = seed
        self.seed_calls.append(seed)
        return [seed + rank for rank in range(self.num_envs)]

    def reset(self):
        self._steps = np.zeros(self.num_envs, dtype=np.int64)
        return np.zeros((self.num_envs, 1), dtype=np.float32)

    def step(self, actions):
        self._steps += 1
        seeds = self._seed + np.arange(self.num_envs)
        episode_lengths = seeds % 3 + 1
        dones = self._steps >= episode_lengths
        infos = []
        for seed in seeds:
            infos.append({
                "success": seed % 2 == 0,
                "ever_pinched": seed % 3 == 0,
                "ever_aligned_pinch": seed % 4 == 0,
                "lift_attempt": seed % 5 == 0,
                "loaded_lift": seed % 6 == 0,
                "ever_grasped": seed % 2 == 1,
                "physical_slip_after_secured_count": int(seed % 7 == 0),
            })
        return (
            np.zeros((self.num_envs, 1), dtype=np.float32),
            np.ones(self.num_envs, dtype=np.float32),
            dones,
            infos,
        )


def test_parallel_strict_eval_preserves_episode_count_and_seed_order(tmp_path):
    env = _FakeEvalVecEnv(num_envs=2)
    callback = StrictSuccessEvalCallback(
        env,
        tmp_path,
        eval_freq=100,
        episodes=5,
        confirmation_episodes=5,
        seed=10,
    )
    callback.model = _FakeModel()

    metrics = callback._evaluate(5)

    assert env.seed_calls == [10, 12, 14]
    assert metrics["success_rate"] == 3 / 5
    assert metrics["pinch_rate"] == 1 / 5
    assert metrics["aligned_pinch_rate"] == 1 / 5
    assert metrics["lift_attempt_rate"] == 1 / 5
    assert metrics["loaded_lift_rate"] == 1 / 5
    assert metrics["grasp_rate"] == 2 / 5
    assert metrics["slip_rate"] == 1 / 5
    assert metrics["mean_return"] == 2.2
    assert metrics["mean_length"] == 2.2
