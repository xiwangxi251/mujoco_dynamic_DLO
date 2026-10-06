from __future__ import annotations

import numpy as np
import pytest

from panda_cable_grasp.rl.train import StrictSuccessEvalCallback


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
        self._seeds = [seed + rank for rank in range(self.num_envs)]
        return self._seeds

    def reset(self):
        self._steps = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_seeds = np.array(self._seeds)
        if hasattr(self, "allowed_seeds"):
            assert all(seed in self.allowed_seeds for seed in self._episode_seeds)
        return np.zeros((self.num_envs, 1), dtype=np.float32)

    def step(self, actions):
        self._steps += 1
        seeds = self._episode_seeds
        episode_lengths = seeds % 3 + 1
        dones = self._steps >= episode_lengths
        infos = []
        for seed in seeds:
            infos.append({
                "success": seed % 2 == 0,
                "strict_success": seed % 4 == 0,
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
    assert metrics["strict_success_rate"] == 1 / 5
    assert metrics["pinch_rate"] == 1 / 5
    assert metrics["aligned_pinch_rate"] == 1 / 5
    assert metrics["lift_attempt_rate"] == 1 / 5
    assert metrics["loaded_lift_rate"] == 1 / 5
    assert metrics["grasp_rate"] == 2 / 5
    assert metrics["slip_rate"] == 1 / 5
    assert metrics["mean_return"] == 2.2
    assert metrics["mean_length"] == 2.2


def test_bank_pool_confirmation_uses_remaining_seeds_and_wraps_inactive_workers(tmp_path):
    import json
    env = _FakeEvalVecEnv(num_envs=13)
    env.allowed_seeds = set(range(20260804, 20260854))
    original_step = env.step

    def step(actions):
        observations, rewards, dones, infos = original_step(actions)
        for seed, info in zip(env._episode_seeds, infos):
            info.update(requested_seed=int(seed), actual_seed=int(seed), workspace_mask={})
        return observations, rewards, dones, infos

    env.step = step
    callback = StrictSuccessEvalCallback(
        env, tmp_path, eval_freq=100, episodes=26,
        confirmation_episodes=26, seed=20260804, seed_pool_size=50,
    )
    callback.model = _FakeModel()
    assert callback.confirmation_episodes == 24
    assert callback.confirmation_seed_offset == 26
    callback._evaluate(26)
    callback._evaluate(callback.confirmation_episodes, seed_offset=callback.confirmation_seed_offset)
    rows = [json.loads(line) for line in (tmp_path / "workspace_eval_episodes.jsonl").read_text().splitlines()]
    routine = {row["actual_seed"] for row in rows if row["seed_offset"] == 0}
    confirmation = {row["actual_seed"] for row in rows if row["seed_offset"] == 26}
    assert len(routine) == 26 and len(confirmation) == 24
    assert not routine & confirmation
    assert routine | confirmation == env.allowed_seeds
    assert env.seed_calls == [20260804, 20260817, 20260830, 20260843]


def test_bank_pool_requires_unused_confirmation_seeds(tmp_path):
    with pytest.raises(ValueError, match="leave seeds for confirmation"):
        StrictSuccessEvalCallback(
            _FakeEvalVecEnv(13), tmp_path, eval_freq=100, episodes=26,
            confirmation_episodes=26, seed=20260804, seed_pool_size=26,
        )
