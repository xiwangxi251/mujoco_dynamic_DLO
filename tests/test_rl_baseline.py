from __future__ import annotations

import unittest

import numpy as np

from rl.rl_cable_env import RLCableGraspEnv


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

    def test_five_dimensional_action_uses_ik_and_gripper_hysteresis(self) -> None:
        self.env.reset(seed=18)
        _, _, _, _, info = self.env.step(np.array([
            0.2, -0.1, 0.3, 0.4, -1.0,
        ], dtype=np.float32))
        self.assertEqual(info["mujoco_action"].shape, (8,))
        self.assertGreaterEqual(info["ik_velocity_scale"], 0.0)
        self.assertLessEqual(info["ik_velocity_scale"], 1.0)
        self.assertEqual(info["gripper_switch_count"], 1)
        self.assertLess(info["reward_gripper_switch"], 0.0)

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


if __name__ == "__main__":
    unittest.main()
