from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from panda_cable_grasp.env import CableGraspEnv
from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
from panda_cable_grasp.policies import DynamicCableGraspPolicy, Phase
from panda_cable_grasp.scenarios import get_scenario

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
            expert = FormulaInterceptExpert(env, FormulaInterceptConfig(
                dynamic_portfolio_enabled=False,
            ))
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
            expert = FormulaInterceptExpert(env, FormulaInterceptConfig(
                dynamic_portfolio_enabled=False,
            ))
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
            expert = FormulaInterceptExpert(env, FormulaInterceptConfig(
                dynamic_portfolio_enabled=False,
            ))
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

    def test_shape_replan_searches_a_valid_internal_segment(self) -> None:
        env = self.make("id_shape_nominal_current")
        try:
            env.reset(randomize=True, seed=20260804)
            expert = FormulaInterceptExpert(env, FormulaInterceptConfig(
                shape_use_scripted_fallback=False,
                search_shape_all_segments=True,
                dynamic_portfolio_enabled=False,
            ))
            expert._replan_intercept()
            margin = expert.expert_config.endpoint_margin_nodes
            self.assertGreaterEqual(expert.expert_segment_index, margin)
            self.assertLess(
                expert.expert_segment_index,
                len(env.cable_ids) - margin - 1,
            )
            self.assertIn(
                expert.expert_horizon,
                expert.expert_config.candidate_horizons,
            )
        finally:
            env.close()

    def test_shape_scripted_fallback_matches_baseline_prediction(self) -> None:
        env = self.make("id_shape_nominal_current")
        try:
            env.reset(randomize=True, seed=20260804)
            baseline = DynamicCableGraspPolicy(env)
            expert = FormulaInterceptExpert(env, FormulaInterceptConfig(
                dynamic_portfolio_enabled=False,
            ))
            baseline.phase = Phase.APPROACH
            expert.phase = Phase.APPROACH
            baseline.filtered_target = env.target_position()
            expert.filtered_target = env.target_position()
            self.assertTrue(np.array_equal(
                expert._predicted_segment(), baseline._predicted_segment()
            ))
        finally:
            env.close()

    def test_intercept_timeout_records_failed_segment(self) -> None:
        env = self.make("id_rigid_l1_nominal")
        try:
            env.reset(randomize=False, seed=20260804)
            expert = FormulaInterceptExpert(env, FormulaInterceptConfig(
                dynamic_portfolio_enabled=False,
            ))
            expert.phase = Phase.INTERCEPT
            expert.locked_segment_index = 12
            expert.locked_segment_alpha = 0.5
            expert._begin_vertical_recovery(env.hand_position.copy())
            self.assertEqual(expert.failed_segment_indices, [12])
            self.assertEqual(expert.phase, Phase.RECOVER)
        finally:
            env.close()

    def test_early_closure_distance_is_combined_only(self) -> None:
        combined = self.make("id_combined_l1_nominal")
        rigid = self.make("id_rigid_l1_nominal")
        try:
            combined.reset(randomize=False, seed=20260804)
            rigid.reset(randomize=False, seed=20260804)
            combined_expert = FormulaInterceptExpert(
                combined,
                FormulaInterceptConfig(
                    close_capture_distance=0.012,
                    combined_close_capture_distance=0.018,
                    dynamic_portfolio_enabled=False,
                ),
            )
            rigid_expert = FormulaInterceptExpert(
                rigid,
                FormulaInterceptConfig(
                    close_capture_distance=0.012,
                    combined_close_capture_distance=0.018,
                    dynamic_portfolio_enabled=False,
                ),
            )
            self.assertEqual(combined_expert._close_capture_distance(), 0.018)
            self.assertEqual(rigid_expert._close_capture_distance(), 0.012)
        finally:
            combined.close()
            rigid.close()

    def test_portfolio_selects_scripted_after_formula_preview_failure(self) -> None:
        env = self.make("id_rigid_l1_nominal")
        try:
            outcomes = [
                (False, False, False, 0.05),
                (True, True, True, 0.01),
            ]
            with patch.object(
                FormulaInterceptExpert,
                "_preview_controller",
                side_effect=outcomes,
            ) as preview:
                expert = FormulaInterceptExpert(env)
                self.assertEqual(preview.call_count, 0)
                env.reset(randomize=False, seed=20260804)
                expert.reset()
            self.assertTrue(expert.episode_use_scripted)
            self.assertFalse(expert.portfolio_formula_success)
            self.assertTrue(expert.portfolio_scripted_success)
            self.assertEqual(preview.call_count, 2)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
