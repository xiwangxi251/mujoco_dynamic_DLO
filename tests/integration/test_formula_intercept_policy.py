from __future__ import annotations

import unittest

import numpy as np

from cable_grasp_env import CableGraspEnv
from dynamic_grasp_policy import DynamicCableGraspPolicy, Phase
from experiment_scenarios import get_scenario
from motion_diagnostics import env_config_for_scenario

from panda_cable_grasp.expert.formula_intercept_policy import (
    FormulaInterceptConfig,
    FormulaInterceptExpert,
)


class FormulaInterceptExpertTests(unittest.TestCase):
    @staticmethod
    def make(name: str, seed: int = 20260804) -> CableGraspEnv:
        return CableGraspEnv(env_config_for_scenario(
            get_scenario(name),
            seed=seed,
            episode_seconds=0.1,
            camera_observation_enabled=False,
        ))

    def test_config_rejects_invalid_search_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_horizons"):
            FormulaInterceptConfig(candidate_horizons=())
        with self.assertRaisesRegex(ValueError, "failed_segment_radius"):
            FormulaInterceptConfig(failed_segment_radius=-1)

    def test_static_prediction_is_exactly_the_baseline_prediction(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=20260804)
            baseline = DynamicCableGraspPolicy(env)
            expert = FormulaInterceptExpert(env)
            baseline.phase = Phase.APPROACH
            expert.phase = Phase.APPROACH
            baseline.filtered_target = env.target_position()
            expert.filtered_target = env.target_position()
            self.assertTrue(np.array_equal(
                expert._predicted_segment(), baseline._predicted_segment()
            ))
        finally:
            env.close()

    def test_rigid_formula_prediction_is_read_only_and_moves_forward(self) -> None:
        env = self.make("id_rigid_l1_nominal")
        try:
            env.reset(randomize=False, seed=20260804)
            expert = FormulaInterceptExpert(env)
            time_before = float(env.data.time)
            state_before = env.data.qpos.copy()
            current = expert.predict_nodes(0.0)
            future = expert.predict_nodes(0.5)
            self.assertGreater(float(np.mean(future[:, 1] - current[:, 1])), 0.0)
            self.assertEqual(float(env.data.time), time_before)
            self.assertTrue(np.array_equal(env.data.qpos, state_before))
        finally:
            env.close()

    def test_shape_shadow_rollout_is_read_only(self) -> None:
        env = self.make("id_shape_nominal_current")
        try:
            env.reset(randomize=False, seed=20260804)
            expert = FormulaInterceptExpert(env)
            time_before = float(env.data.time)
            qpos_before = env.data.qpos.copy()
            qvel_before = env.data.qvel.copy()
            ctrl_before = env.data.ctrl.copy()
            diagnostics_before = env._last_shape_acceleration.copy()
            current = expert.predict_nodes(0.0)
            future = expert.predict_nodes(0.5)
            self.assertGreater(float(np.linalg.norm(future - current)), 0.0)
            self.assertEqual(float(env.data.time), time_before)
            self.assertTrue(np.array_equal(env.data.qpos, qpos_before))
            self.assertTrue(np.array_equal(env.data.qvel, qvel_before))
            self.assertTrue(np.array_equal(env.data.ctrl, ctrl_before))
            self.assertTrue(np.array_equal(
                env._last_shape_acceleration, diagnostics_before
            ))
            self.assertEqual(expert.shadow_rollouts, 1)
        finally:
            env.close()

    def test_dynamic_replan_selects_an_internal_finite_candidate(self) -> None:
        env = self.make("id_combined_l1_nominal")
        try:
            env.reset(randomize=False, seed=20260804)
            expert = FormulaInterceptExpert(env)
            expert._replan_intercept()
            margin = expert.expert_config.endpoint_margin_nodes
            self.assertGreaterEqual(expert.expert_segment_index, margin)
            self.assertLess(
                expert.expert_segment_index,
                len(env.cable_ids) - margin - 1,
            )
            self.assertTrue(np.isfinite(expert.expert_score))
            self.assertIn(
                expert.expert_horizon,
                expert.expert_config.candidate_horizons,
            )
        finally:
            env.close()

    def test_shape_replan_preserves_episode_target(self) -> None:
        env = self.make("id_shape_nominal_current")
        try:
            env.reset(randomize=True, seed=20260804)
            expert = FormulaInterceptExpert(env)
            expert._replan_intercept()
            target_index = env.cable_ids.index(env.target_body_id)
            self.assertEqual(expert.expert_segment_index, target_index)
            self.assertEqual(
                expert.expert_horizon,
                expert.config.approach_prediction_horizon,
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
