from __future__ import annotations

import unittest

import numpy as np

from panda_cable_grasp.rl.environment import RLCableGraspEnv
from panda_cable_grasp.rl.train import (
    MotionCurriculumCallback,
    RL_L1_CURRICULUM_STAGES,
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
        self.assertEqual(open_components["reward_capture_ready_open"], 0.0)

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

    def test_curriculum_uses_strict_eval_thresholds_and_discrete_intensities(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
