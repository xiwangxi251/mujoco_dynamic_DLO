from __future__ import annotations

import unittest

import numpy as np

from panda_cable_grasp.rl.environment import RLCableGraspEnv
from panda_cable_grasp.rl.train import (
    MotionCurriculumCallback,
    RL_L1_CURRICULUM_STAGES,
    RL_L1_CURRICULUM_TRAINING_MIXES,
    build_l1_curriculum_training_mixes,
)


class RLBaselineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = RLCableGraspEnv(seed=17, episode_seconds=0.20)

    def tearDown(self) -> None:
        self.env.close()

    def test_observation_and_action_contract(self) -> None:
        observation, info = self.env.reset(seed=17)
        self.assertEqual(observation.shape, (99,))
        self.assertEqual(self.env.observation_space.shape, (99,))
        self.assertEqual(self.env.action_space.shape, (5,))
        self.assertEqual(len(self.env.OBSERVATION_NAMES), 99)
        self.assertNotIn("target_body_id", self.env.OBSERVATION_NAMES)
        self.assertIn("nearest_graspable_distance", info)

    def test_five_dimensional_action_uses_base_frame_ik_and_hysteretic_gripper(self) -> None:
        self.env.reset(seed=18)
        _, _, _, _, info = self.env.step(np.array([
            0.0, 0.0, 1.0, 0.4, -1.0,
        ], dtype=np.float32))
        self.assertEqual(info["mujoco_action"].shape, (8,))
        self.assertGreaterEqual(info["ik_velocity_scale"], 0.0)
        self.assertLessEqual(info["ik_velocity_scale"], 1.0)
        self.assertEqual(info["gripper_switch_count"], 1)
        self.assertLess(info["reward_gripper_switch"], 0.0)
        np.testing.assert_allclose(
            info["commanded_world_translation"], [0.0, 0.0, 0.01]
        )

    def test_gripper_command_uses_hysteretic_deadband(self) -> None:
        self.env.reset(seed=24)
        low, high = self.env.model.actuator_ctrlrange[7]
        closed = self.env._convert_action(np.array([0, 0, 0, 0, -1.0]))[7]
        closed_deadband = self.env._convert_action(
            np.array([0, 0, 0, 0, 0.0])
        )[7]
        opened = self.env._convert_action(np.array([0, 0, 0, 0, 1.0]))[7]
        open_deadband = self.env._convert_action(
            np.array([0, 0, 0, 0, 0.0])
        )[7]
        self.assertAlmostEqual(closed, low)
        self.assertAlmostEqual(closed_deadband, low)
        self.assertAlmostEqual(opened, high)
        self.assertAlmostEqual(open_deadband, high)

    def test_public_task_success_controls_terminal_reward_and_termination(self) -> None:
        self.env.reset(seed=30)
        base_info = self.env.base_env.info()
        base_info["success"] = True
        base_info["success_now"] = True
        self.env.base_env.step = lambda action: ({}, 0.0, True, False, base_info)
        grasp_status = self.env._empty_grasp_status()
        grasp_status["rl_hold_success"] = False
        self.env._update_grasp_status = lambda info: grasp_status

        _, reward, terminated, truncated, info = self.env.step(
            np.zeros(5, dtype=np.float32)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertTrue(info["task_success"])
        self.assertFalse(info["strict_success"])
        self.assertFalse(info["policy_internal_success"])
        self.assertEqual(info["reward_success"], self.env.rl_config.reward_success)
        self.assertGreater(reward, self.env.rl_config.reward_success - 0.01)

    def test_capture_corridor_rewards_close_without_forcing_it_early(self) -> None:
        self.env.reset(seed=28)
        hand_position = self.env._grasp_center_position()
        hand_rotation = self.env.data.xmat[
            self.env.base_env.hand_id
        ].reshape(3, 3)

        def world_from_hand(local: np.ndarray) -> np.ndarray:
            return hand_position + hand_rotation @ local

        self.env._nearest_graspable_segment = lambda point: (
            world_from_hand(np.zeros(3)),
            0.0,
            np.array([1.0, 0.0, 0.0]),
            5,
            0.5,
        )
        self.env._alignment_terms = lambda distance, tangent: (1.0, 1.0)
        grasp_status = {
            "pinch_confirmed": False,
            "secured_grasp": False,
            "grasp_lift_delta": 0.0,
            "strict_success_hold": 0.0,
        }

        self.env._gripper_closed = False
        self.env._gripper_switch_event = False
        _, open_components = self.env._reward(
            np.zeros(5), False, {"lifted_fraction": 0.0}, grasp_status
        )
        self.assertTrue(self.env._capture_ready)
        self.assertLess(open_components["reward_capture_ready_open"], 0.0)

        self.env._gripper_closed = True
        self.env._gripper_switch_event = True
        _, close_components = self.env._reward(
            np.zeros(5), False, {"lifted_fraction": 0.0}, grasp_status
        )
        self.assertGreater(close_components["reward_capture_close"], 0.0)
        self.assertTrue(self.env._capture_close_event)

        _, repeated_components = self.env._reward(
            np.zeros(5), False, {"lifted_fraction": 0.0}, grasp_status
        )
        self.assertEqual(repeated_components["reward_capture_close"], 0.0)

        # This point is still close and aligned, but lies near the fingertip
        # instead of being seated deeply enough between the pads.
        shallow_point = world_from_hand(np.array([0.0, 0.0, 0.0075]))
        self.env._nearest_graspable_segment = lambda point: (
            shallow_point, 0.0075, np.array([1.0, 0.0, 0.0]), 5, 0.5
        )
        self.env._gripper_switch_event = True
        _, premature_components = self.env._reward(
            np.zeros(5), False, {"lifted_fraction": 0.0}, grasp_status
        )
        self.assertFalse(self.env._capture_ready)
        self.assertLess(
            premature_components["reward_premature_close_event"], 0.0
        )
        self.assertLess(premature_components["reward_premature_close"], 0.0)

        accumulated_event_penalty = premature_components[
            "reward_premature_close_event"
        ]
        for _ in range(100):
            _, capped_components = self.env._reward(
                np.zeros(5), False, {"lifted_fraction": 0.0}, grasp_status
            )
            accumulated_event_penalty += capped_components[
                "reward_premature_close_event"
            ]
        self.assertAlmostEqual(
            accumulated_event_penalty,
            -self.env.rl_config.premature_close_episode_penalty_cap,
        )
        self.assertAlmostEqual(
            self.env._premature_close_penalty_total,
            self.env.rl_config.premature_close_episode_penalty_cap,
        )
        self.assertEqual(capped_components["reward_premature_close_event"], 0.0)

    def test_capture_corridor_rejects_shallow_and_off_center_cable(self) -> None:
        self.env.reset(seed=29)
        hand_position = self.env._grasp_center_position()
        hand_rotation = self.env.data.xmat[
            self.env.base_env.hand_id
        ].reshape(3, 3)

        def terms(local: np.ndarray) -> tuple[np.ndarray, float, bool]:
            return self.env._capture_corridor_terms(
                hand_position + hand_rotation @ local
            )

        _, centered_depth, centered = terms(np.zeros(3))
        _, shallow_depth, shallow = terms(np.array([0.0, 0.0, 0.0075]))
        _, _, off_center = terms(np.array([0.0, 0.015, 0.0]))

        self.assertTrue(centered)
        self.assertFalse(shallow)
        self.assertFalse(off_center)
        self.assertGreater(centered_depth, shallow_depth)

    def test_alignment_scores_planar_closing_axis_perpendicular_to_cable(self) -> None:
        identity = np.eye(3)
        perpendicular = self.env._planar_perpendicular_alignment(
            identity, np.array([1.0, 0.0, 0.0])
        )
        parallel = self.env._planar_perpendicular_alignment(
            identity, np.array([0.0, 1.0, 0.0])
        )
        self.assertAlmostEqual(perpendicular, 1.0)
        self.assertAlmostEqual(parallel, 0.0)

    def test_pinch_disables_reach_and_alignment_rewards(self) -> None:
        self.env.reset(seed=25)
        grasp_status = {
            "pinch_confirmed": True,
            "secured_grasp": False,
            "grasp_lift_delta": 0.0,
            "strict_success_hold": 0.0,
        }
        _, components = self.env._reward(
            np.zeros(5), False, {"lifted_fraction": 0.0}, grasp_status
        )
        self.assertEqual(components["reward_reach_progress"], 0.0)
        self.assertEqual(components["reward_alignment_progress"], 0.0)

    def test_lift_shaping_penalizes_downward_motion_and_cannot_be_farmed(self) -> None:
        self.env.reset(seed=31)
        grasp_status = {
            "pinch_confirmed": True,
            "secured_grasp": False,
            "grasp_lift_delta": 0.04,
            "strict_success_hold": 0.0,
        }
        _, rising = self.env._reward(
            np.zeros(5), False, {"lifted_fraction": 0.0}, grasp_status
        )

        grasp_status["grasp_lift_delta"] = 0.01
        _, falling = self.env._reward(
            np.zeros(5), False, {"lifted_fraction": 0.0}, grasp_status
        )

        grasp_status["grasp_lift_delta"] = 0.04
        _, rising_again = self.env._reward(
            np.zeros(5), False, {"lifted_fraction": 0.0}, grasp_status
        )

        self.assertGreater(rising["reward_lift_progress"], 0.0)
        self.assertLess(falling["reward_lift_progress"], 0.0)
        self.assertAlmostEqual(
            rising["reward_lift_progress"]
            + falling["reward_lift_progress"]
            + rising_again["reward_lift_progress"],
            rising["reward_lift_progress"],
        )

    def test_aligned_pinch_reward_only_fires_on_pinch_transition(self) -> None:
        self.env.reset(seed=27)
        self.env._alignment_terms = lambda distance, tangent: (1.0, 1.0)
        self.env._previous_pinch_confirmed = True
        grasp_status = {
            "pinch_confirmed": True,
            "secured_grasp": False,
            "grasp_lift_delta": 0.0,
            "strict_success_hold": 0.0,
        }
        _, components = self.env._reward(
            np.zeros(5), False, {"lifted_fraction": 0.0}, grasp_status
        )
        self.assertEqual(components["reward_new_aligned_pinch"], 0.0)
        self.assertFalse(self.env._aligned_pinch_event)

    def test_unloaded_pinch_penalties_are_delayed_and_capped(self) -> None:
        self.env.reset(seed=26)
        unloaded = {
            "pinch_confirmed": True,
            "grasp_lift_delta": 0.0,
        }
        rewards = [
            self.env._update_pinch_session(np.zeros(5), unloaded)[0]
            for _ in range(30)
        ]
        self.assertTrue(all(value == 0.0 for value in rewards[:20]))
        self.assertLess(rewards[-1], 0.0)
        released = dict(unloaded, pinch_confirmed=False)
        _, failed_reward = self.env._update_pinch_session(
            np.zeros(5), released
        )
        self.assertLess(failed_reward, 0.0)
        self.assertLessEqual(
            self.env._unloaded_pinch_penalty_total,
            self.env.rl_config.unloaded_pinch_episode_penalty_cap,
        )

    def test_dlo_samples_are_uniform_in_arc_length(self) -> None:
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        sampled = self.env._arc_length_samples(positions, positions, 4)
        np.testing.assert_allclose(sampled[:, 0], [0.0, 1.0, 2.0, 3.0])

    def test_lift_reference_is_fixed_at_episode_reset(self) -> None:
        self.env.reset(seed=19)
        reference = self.env._episode_initial_cable_z.copy()
        self.env._locked_body_id = self.env.base_env.cable_ids[10]
        np.testing.assert_array_equal(self.env._episode_initial_cable_z, reference)

    def test_l1_curriculum_switches_scenes_and_motion_intensity(self) -> None:
        curriculum_env = RLCableGraspEnv(
            seed=21,
            episode_seconds=0.20,
            scenario_names=(
                "id_static",
                "id_shape_nominal_current",
                "id_rigid_l1_nominal",
                "id_combined_l1_nominal",
            ),
        )
        try:
            curriculum_env.set_training_scenarios(("id_static",))
            curriculum_env.set_motion_difficulty(0.0)
            _, info = curriculum_env.reset(seed=21)
            self.assertEqual(info["scenario_name"], "id_static")

            curriculum_env.set_training_scenarios(
                ("id_shape_nominal_current",)
            )
            _, info = curriculum_env.reset(seed=22)
            self.assertEqual(info["scenario_name"], "id_shape_nominal_current")
            self.assertAlmostEqual(
                curriculum_env.base_env.config.disturbance_strength, 0.75
            )
            self.assertAlmostEqual(
                curriculum_env.base_env.config.motion_frequency_scale, 2.0 / 3.0
            )

            curriculum_env.set_motion_difficulty(1.0)
            curriculum_env.reset(seed=23)
            self.assertAlmostEqual(
                curriculum_env.base_env.config.disturbance_strength, 1.5
            )
            self.assertAlmostEqual(
                curriculum_env.base_env.config.motion_frequency_scale, 1.0
            )
        finally:
            curriculum_env.close()

    def test_curriculum_training_distribution_supports_weighted_rehearsal(
        self,
    ) -> None:
        curriculum_env = RLCableGraspEnv(
            seed=24,
            episode_seconds=0.20,
            scenario_names=("id_static", "id_shape_nominal_current"),
        )
        try:
            curriculum_env.set_training_scenario_distribution(
                ("id_static", "id_shape_nominal_current"),
                (0.20, 0.80),
            )
            self.assertEqual(
                curriculum_env._active_scenario_probabilities,
                (0.20, 0.80),
            )
            selections = [
                curriculum_env._select_training_scenario().name
                for _ in range(2_000)
            ]
            static_rate = selections.count("id_static") / len(selections)
            self.assertAlmostEqual(static_rate, 0.20, delta=0.04)

            curriculum_env.set_training_scenarios(
                ("id_static", "id_shape_nominal_current")
            )
            self.assertEqual(
                curriculum_env._active_scenario_probabilities,
                (0.50, 0.50),
            )
        finally:
            curriculum_env.close()

    def test_curriculum_uses_strict_eval_thresholds_and_discrete_intensities(
        self,
    ) -> None:
        callback = MotionCurriculumCallback(RL_L1_CURRICULUM_STAGES)
        self.assertEqual(
            callback.phases,
            (
                (0, 0.0),
                (1, 1.0 / 3.0), (1, 2.0 / 3.0), (1, 1.0),
                (2, 1.0 / 3.0), (2, 2.0 / 3.0), (2, 1.0),
            ),
        )
        self.assertTrue(callback.evaluation_qualifies({
            "aligned_pinch_rate": 0.50,
            "grasp_rate": 0.20,
            "success_rate": 0.10,
        }))
        self.assertFalse(callback.evaluation_qualifies({
            "aligned_pinch_rate": 0.49,
            "grasp_rate": 0.20,
            "success_rate": 0.10,
        }))

    def test_curriculum_rehearses_easy_scenes_but_evaluates_current_stage(
        self,
    ) -> None:
        callback = MotionCurriculumCallback(RL_L1_CURRICULUM_STAGES)
        self.assertEqual(
            callback.training_mixes,
            RL_L1_CURRICULUM_TRAINING_MIXES,
        )

        callback.phase_index = 1
        self.assertEqual(
            callback.current_training_mix,
            (
                ("id_static", 0.20),
                ("id_shape_nominal_current", 0.40),
                ("id_rigid_l1_nominal", 0.40),
            ),
        )
        self.assertEqual(
            callback.current_scenarios,
            ("id_shape_nominal_current", "id_rigid_l1_nominal"),
        )

        callback.phase_index = 4
        self.assertEqual(
            callback.current_training_mix,
            (
                ("id_static", 0.10),
                ("id_shape_nominal_current", 0.10),
                ("id_rigid_l1_nominal", 0.10),
                ("id_combined_l1_nominal", 0.70),
            ),
        )
        self.assertEqual(callback.current_scenarios, ("id_combined_l1_nominal",))

        evaluation_env = RLCableGraspEnv(
            seed=25,
            episode_seconds=0.20,
            scenario_names=(
                "id_static",
                "id_shape_nominal_current",
                "id_rigid_l1_nominal",
                "id_combined_l1_nominal",
            ),
        )
        try:
            callback.configure_evaluation_env(evaluation_env)
            self.assertEqual(
                evaluation_env._active_scenario_names,
                ("id_combined_l1_nominal",),
            )
            self.assertEqual(
                evaluation_env._active_scenario_probabilities,
                (1.0,),
            )
        finally:
            evaluation_env.close()

    def test_curriculum_replay_fraction_validation(self) -> None:
        with self.assertRaises(ValueError):
            build_l1_curriculum_training_mixes(
                stage2_static_replay=0.60,
                stage2_component_replay=0.40,
            )


if __name__ == "__main__":
    unittest.main()
