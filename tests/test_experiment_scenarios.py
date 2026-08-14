from __future__ import annotations

import math
import unittest

import numpy as np

from cable_grasp_env import CableGraspEnv, EnvConfig, GraspState
from experiment_scenarios import (
    DEFAULT_SCENARIO,
    get_scenario,
    list_scenarios,
    list_suite_scenarios,
)
from motion_diagnostics import MotionTracker, env_config_for_scenario


class ScenarioRegistryTests(unittest.TestCase):
    def test_suites_are_complete_and_default_is_legacy_compatible_shape(self) -> None:
        core = {item.name for item in list_suite_scenarios("core")}
        sweep = {item.name for item in list_suite_scenarios("motion_sweep")}
        ood = {item.name for item in list_suite_scenarios("ood")}
        paper = {item.name for item in list_suite_scenarios("paper")}
        self.assertFalse(core & sweep)
        self.assertFalse(core & ood)
        self.assertFalse(sweep & ood)
        self.assertEqual(core | sweep | ood, paper)
        self.assertEqual(paper, {item.name for item in list_scenarios()})
        self.assertEqual(DEFAULT_SCENARIO.motion_type.value, "shape")
        self.assertEqual(DEFAULT_SCENARIO.disturbance_strength, 1.5)
        self.assertEqual(
            DEFAULT_SCENARIO.to_env_overrides()["motion_profile_version"],
            "factorized_v1",
        )


class EnvironmentScenarioTests(unittest.TestCase):
    @staticmethod
    def make(name: str, seed: int = 1234) -> CableGraspEnv:
        scenario = get_scenario(name)
        return CableGraspEnv(env_config_for_scenario(
            scenario, seed=seed, episode_seconds=0.1,
        ))

    def test_constructor_initialization_is_not_counted_as_a_trial(self) -> None:
        env = self.make("id_static")
        try:
            self.assertEqual(env.trial_index, 0)
            _, first = env.reset(seed=1001)
            self.assertEqual(env.trial_index, 1)
            self.assertEqual(first["trial"], 1)
            _, second = env.reset(seed=1002)
            self.assertEqual(env.trial_index, 2)
            self.assertEqual(second["trial"], 2)
        finally:
            del env

    def test_seed_reproduces_initial_scene_and_motion_profile(self) -> None:
        env = self.make("ood_dynamics_high_stochastic")
        try:
            _, first = env.reset(seed=9001)
            _, second = env.reset(seed=9001)
            keys = (
                "initial_cable_dx", "initial_cable_dy", "target_body_id",
                "disturbance_phase", "disturbance_spatial_phase",
                "motion_profile_hash",
            )
            self.assertEqual(
                tuple(first[key] for key in keys),
                tuple(second[key] for key in keys),
            )
        finally:
            del env

    def test_core_force_components_switch_exactly(self) -> None:
        expected = {
            "id_static": (False, False),
            "id_rigid_nominal": (False, True),
            "id_shape_nominal_current": (True, False),
            "id_combined_nominal": (True, True),
        }
        for name, (shape_expected, rigid_expected) in expected.items():
            with self.subTest(name=name):
                env = self.make(name)
                try:
                    env.reset(seed=17)
                    env.data.xfrc_applied[:] = 0.0
                    env._apply_cable_disturbance()
                    shape = np.linalg.norm(env._last_shape_acceleration) > 1e-10
                    rigid = np.linalg.norm(
                        env._last_rigid_translation_acceleration
                    ) + np.linalg.norm(
                        env._last_rigid_rotation_acceleration
                    ) > 1e-10
                    self.assertEqual(shape, shape_expected)
                    self.assertEqual(rigid, rigid_expected)
                    if name == "id_static":
                        self.assertTrue(np.array_equal(
                            env.data.xfrc_applied[env.cable_ids],
                            np.zeros((len(env.cable_ids), 6)),
                        ))
                finally:
                    del env

    def test_factorized_shape_has_zero_net_force_and_torque(self) -> None:
        env = self.make("id_shape_nominal_current")
        try:
            env.reset(seed=41)
            env._apply_cable_disturbance()
            acceleration = env._last_shape_acceleration
            force = env.cable_mass[:, None] * acceleration
            positions = env.data.xpos[env.cable_ids]
            center = np.average(positions, axis=0, weights=env.cable_mass)
            torque = np.cross(positions - center, force).sum(axis=0)
            self.assertLess(np.linalg.norm(force.sum(axis=0)), 1e-12)
            self.assertLess(np.linalg.norm(torque), 1e-12)
        finally:
            del env

    def test_legacy_default_preserves_original_shape_formula(self) -> None:
        env = CableGraspEnv(EnvConfig(seed=71, episode_seconds=0.1))
        try:
            env.reset(seed=71)
            t = env.phase_offset + env.data.time
            p = env.spatial_phase
            lateral = (
                2.30 * np.sin(env._lateral_space[0] - 3.4 * t + p + 0.65 * np.sin(1.9 * t))
                + 2.00 * np.sin(env._lateral_space[1] + 2.8 * t - 0.4 * p + 0.55 * np.sin(2.7 * t + p))
                + 1.60 * np.sin(env._lateral_space[2] - 4.6 * t + 0.7)
                + 1.20 * np.sin(env._lateral_space[3] + 5.2 * t + 0.35 * p)
            )
            longitudinal = (
                0.55 * np.sin(env._longitudinal_space[0] + 2.3 * t + 0.2 * p)
                + 0.45 * np.sin(env._longitudinal_space[1] - 3.1 * t)
            )
            vertical = (
                0.75 * np.sin(env._vertical_space[0] - 2.5 * t + 0.5 * p)
                + 0.55 * np.sin(env._vertical_space[1] + 3.4 * t)
            )
            lateral -= lateral.mean()
            longitudinal -= longitudinal.mean()
            vertical -= vertical.mean()
            expected = 1.5 * np.column_stack((
                5.0 * longitudinal, 19.0 * lateral, 9.0 * vertical,
            ))
            env._apply_cable_disturbance()
            self.assertTrue(np.array_equal(expected, env._last_shape_acceleration))
        finally:
            del env

    def test_combined_is_sum_of_matching_shape_and_rigid_fields(self) -> None:
        environments = {
            name: self.make(name, seed=81)
            for name in (
                "id_shape_nominal_current", "id_rigid_nominal",
                "id_combined_nominal",
            )
        }
        try:
            for env in environments.values():
                env.reset(seed=81)
                env._apply_cable_disturbance()
            shape = environments["id_shape_nominal_current"]
            rigid = environments["id_rigid_nominal"]
            combined = environments["id_combined_nominal"]
            expected = (
                shape._last_shape_acceleration
                + rigid._last_rigid_translation_acceleration
                + rigid._last_rigid_rotation_acceleration
            )
            actual = (
                combined._last_shape_acceleration
                + combined._last_rigid_translation_acceleration
                + combined._last_rigid_rotation_acceleration
            )
            self.assertTrue(np.allclose(expected, actual, rtol=0.0, atol=1e-12))
        finally:
            for env in environments.values():
                del env

    def test_disturbance_does_not_read_grasp_state(self) -> None:
        env = self.make("id_combined_nominal")
        try:
            env.reset(seed=52)
            env._apply_cable_disturbance()
            before = (
                env._last_shape_acceleration.copy(),
                env._last_rigid_translation_acceleration.copy(),
                env._last_rigid_rotation_acceleration.copy(),
            )
            env.grasp_state = GraspState(
                body_id=env.target_body_id,
                candidate_time=0.0,
                bilateral_confirmed=True,
                last_bilateral_time=0.0,
                lost_contact_time=0.0,
            )
            env._apply_cable_disturbance()
            after = (
                env._last_shape_acceleration,
                env._last_rigid_translation_acceleration,
                env._last_rigid_rotation_acceleration,
            )
            for left, right in zip(before, after, strict=True):
                self.assertTrue(np.array_equal(left, right))
        finally:
            del env

    def test_ood_length_and_material_change_compiled_physics(self) -> None:
        nominal = self.make("id_combined_nominal")
        short = self.make("ood_length_short")
        soft = self.make("ood_material_soft")
        try:
            nominal_span = np.linalg.norm(
                nominal.data.xpos[nominal.cable_ids[-1]]
                - nominal.data.xpos[nominal.cable_ids[0]]
            )
            short_span = np.linalg.norm(
                short.data.xpos[short.cable_ids[-1]]
                - short.data.xpos[short.cable_ids[0]]
            )
            self.assertAlmostEqual(short_span / nominal_span, 0.8, delta=0.01)
            self.assertAlmostEqual(
                soft.cable_mass.sum() / nominal.cable_mass.sum(), 0.8, delta=0.02
            )
            nominal_damping = nominal.model.dof_damping[
                nominal.model.jnt_dofadr[10]
            ]
            soft_damping = soft.model.dof_damping[soft.model.jnt_dofadr[10]]
            self.assertAlmostEqual(
                soft_damping / nominal_damping, 0.72, delta=1e-6
            )
        finally:
            del nominal, short, soft


class MotionMetricTests(unittest.TestCase):
    def test_tracker_removes_synthetic_rigid_motion(self) -> None:
        x = np.linspace(-0.4, 0.4, 40)
        reference = np.column_stack((x, 0.03 * np.sin(5.0 * x), np.zeros_like(x)))
        angle = 0.73
        rotation = np.array([
            [math.cos(angle), math.sin(angle)],
            [-math.sin(angle), math.cos(angle)],
        ])
        current = reference.copy()
        current[:, :2] = reference[:, :2] @ rotation + np.array([0.2, -0.1])
        tracker = MotionTracker(reference, np.ones(40))
        info = {
            "shape_acceleration_rms": 0.0,
            "rigid_translation_acceleration_rms": 1.0,
            "rigid_rotation_acceleration_rms": 1.0,
            "boundary_acceleration_rms": 0.0,
        }
        tracker.record(0.1, current, info)
        summary = tracker.summary()
        self.assertLess(float(summary["shape_change_rms_m"]), 1e-12)
        self.assertAlmostEqual(
            float(summary["rigid_rotation_max_rad"]), angle, delta=1e-12
        )


if __name__ == "__main__":
    unittest.main()
