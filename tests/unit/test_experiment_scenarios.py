from __future__ import annotations

import math
import unittest

import mujoco
import numpy as np

from panda_cable_grasp.policies import DynamicCableGraspPolicy, Phase, PolicyConfig
from panda_cable_grasp.dynamicvla.adapter import (
    DynamicVLATaskSpaceAdapter,
    make_dynamicvla_observation,
)
from panda_cable_grasp.env.environment import (
    CableGraspEnv,
    EnvConfig,
    GraspState,
    RIGID_MOTION_START_TIME,
    RIGID_MOTION_START_Y,
    RIGID_MOTION_TRAVEL,
    RIGID_MOTION_ROTATION,
    rotation_to_quat,
)
from panda_cable_grasp.scenarios import (
    DEFAULT_SCENARIO,
    get_scenario,
    list_scenarios,
    list_suite_scenarios,
)
from panda_cable_grasp.evaluation.motion_diagnostics import (
    MotionTracker,
    env_config_for_scenario,
)


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
        self.assertEqual(
            paper, {item.name for item in list_scenarios()}
        )
        self.assertEqual(len(core), 6)
        self.assertEqual(DEFAULT_SCENARIO.motion_type.value, "shape")
        self.assertEqual(DEFAULT_SCENARIO.disturbance_strength, 1.5)
        self.assertEqual(DEFAULT_SCENARIO.shape_motion_scale, 0.4695)
        self.assertEqual(
            DEFAULT_SCENARIO.to_env_overrides()["shape_motion_scale"], 0.4695
        )
        self.assertEqual(
            DEFAULT_SCENARIO.to_env_overrides()["motion_profile_version"],
            "factorized_v2",
        )

    def test_all_registered_rigid_motion_is_explicit_l1_or_l2_id(self) -> None:
        scenarios = list_scenarios()
        rigid_scenarios = [
            item for item in scenarios
            if item.motion_type.value in {"rigid", "combined"}
        ]
        self.assertTrue(rigid_scenarios)
        self.assertTrue(all(
            item.motion_profile_version in {
                "rigid_level1_single_pass_v2",
                "rigid_level2_single_pass_v2",
            }
            for item in rigid_scenarios
        ))
        id_names = {item.name for item in list_scenarios("id")}
        expected = {
            f"id_{motion}_l{trajectory}_{level}"
            for motion in ("rigid", "combined")
            for trajectory in (1, 2)
            for level in ("low", "nominal", "high")
        }
        self.assertTrue(expected <= id_names)
        self.assertFalse(any(name.startswith("pilot_") for name in id_names))
        self.assertNotIn("id_rigid_nominal", id_names)
        self.assertNotIn("id_combined_nominal", id_names)

    def test_environment_rejects_removed_quasiperiodic_rigid_path(self) -> None:
        with self.assertRaises(ValueError):
            EnvConfig(
                motion_mode="rigid",
                motion_profile_version="factorized_v2",
            )
        with self.assertRaises(ValueError):
            EnvConfig(
                motion_mode="combined",
                motion_profile_version="factorized_v2",
            )


class EnvironmentScenarioTests(unittest.TestCase):
    @staticmethod
    def make(name: str, seed: int = 1234) -> CableGraspEnv:
        scenario = get_scenario(name)
        return CableGraspEnv(env_config_for_scenario(
            scenario,
            seed=seed,
            episode_seconds=0.1,
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

    def test_dynamicvla_opst_camera_is_the_only_fixed_sensor_camera(self) -> None:
        defaults = EnvConfig()
        self.assertEqual(defaults.dynamicvla_camera_width, 480)
        self.assertEqual(defaults.dynamicvla_camera_height, 360)
        self.assertEqual(defaults.dynamicvla_opst_camera_pos, (1.0, 0.0, 0.6))
        self.assertEqual(
            defaults.dynamicvla_opst_camera_quat,
            (0.6123724357, 0.3535533906, 0.3535533906, 0.6123724357),
        )
        config = EnvConfig(
            seed=12,
            episode_seconds=0.1,
            dynamicvla_cameras_enabled=True,
            dynamicvla_camera_width=160,
            dynamicvla_camera_height=120,
        )
        env = CableGraspEnv(config)
        try:
            observation, _ = env.reset(seed=12)
            self.assertNotIn("camera_rgb", observation)
            self.assertEqual(
                mujoco.mj_name2id(
                    env.model, mujoco.mjtObj.mjOBJ_CAMERA, "global_camera"
                ),
                -1,
            )
            self.assertEqual(
                int(env.model.cam_bodyid[env.dynamicvla_opst_camera_id]), 0
            )
            camera_position = env.data.cam_xpos[
                env.dynamicvla_opst_camera_id
            ].copy()
            frames = env.dynamicvla_camera_rgb()
            self.assertEqual(frames["opst_cam"].shape, (120, 160, 3))
            self.assertEqual(frames["wrist_cam"].shape, (120, 160, 3))
            self.assertEqual(frames["opst_cam"].dtype, np.uint8)
            self.assertGreater(float(frames["opst_cam"].std()), 1.0)
            frames["opst_cam"][:] = 0
            self.assertGreater(
                float(env.dynamicvla_camera_rgb()["opst_cam"].std()), 1.0
            )
            next_observation, _, _, _, _ = env.step(env.ready_ctrl)
            self.assertNotIn("camera_rgb", next_observation)
            self.assertGreater(next_observation["time"], observation["time"])
            self.assertTrue(np.array_equal(
                env.data.cam_xpos[env.dynamicvla_opst_camera_id], camera_position
            ))
        finally:
            env.close()

    def test_dynamicvla_camera_rig_and_task_space_adapter(self) -> None:
        env = CableGraspEnv(EnvConfig(
            seed=13,
            episode_seconds=0.1,
            dynamicvla_cameras_enabled=True,
        ))
        try:
            env.reset(seed=13)
            self.assertEqual(
                int(env.model.cam_bodyid[env.dynamicvla_opst_camera_id]), 0
            )
            self.assertEqual(
                int(env.model.cam_bodyid[env.dynamicvla_wrist_camera_id]),
                env.hand_id,
            )
            observation = make_dynamicvla_observation(
                env, "Pick up the orange cable.", index=0
            )
            self.assertEqual(
                observation["observation.images.opst_cam"].shape,
                (1, 360, 480, 3),
            )
            self.assertEqual(
                observation["observation.images.wrist_cam"].shape,
                (1, 360, 480, 3),
            )
            self.assertGreater(
                float(observation["observation.images.opst_cam"].std()), 1.0
            )
            self.assertGreater(
                float(observation["observation.images.wrist_cam"].std()), 1.0
            )
            self.assertEqual(
                observation["observation.state"]["end_effector"]["pos"].shape,
                (1, 3),
            )

            adapter = DynamicVLATaskSpaceAdapter(env)
            adapter.set_model_action(np.array([
                99.0, -99.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0,
            ]))
            action = adapter.action()
            diagnostics = adapter.diagnostics()
            self.assertEqual(action.shape, (8,))
            self.assertTrue(np.all(np.isfinite(action)))
            self.assertTrue(diagnostics["position_clipped"])
            self.assertEqual(action[-1], 0.0)
        finally:
            env.close()

    def test_environment_limits_arm_and_gripper_commands(self) -> None:
        env = self.make("id_static")
        try:
            observation, _ = env.reset(randomize=False, seed=1003)
            requested = env.model.actuator_ctrlrange[:, 1].copy()
            requested[3] = env.model.actuator_ctrlrange[3, 0]
            requested[7] = 0.0
            _, _, _, _, info = env.step(requested)

            control_dt = env.model.opt.timestep * env.config.frame_skip
            applied_velocity = info["applied_arm_velocity"]
            self.assertTrue(np.all(
                np.abs(applied_velocity)
                <= np.asarray(env.config.arm_joint_velocity_limits) + 1e-12
            ))
            self.assertFalse(env.config.arm_acceleration_limit_enabled)
            self.assertTrue(np.allclose(
                info["applied_action"][:7],
                observation["arm_qpos"] + applied_velocity * control_dt,
                rtol=0.0,
                atol=1e-12,
            ))
            self.assertFalse(env.config.hand_cartesian_velocity_limit_enabled)
            finger_target_change = (
                255.0 - info["applied_action"][7]
            ) * env._gripper_ctrl_to_finger_position
            self.assertLessEqual(
                finger_target_change / control_dt,
                env.config.gripper_finger_velocity_limit + 1e-12,
            )
            self.assertTrue(info["motion_limit_active"])
            self.assertFalse(info["motion_limit_flags"]["acceleration"])
            self.assertTrue(info["motion_limit_flags"]["gripper_velocity"])
        finally:
            del env

    def test_default_speed_limits_match_dynamicvla_panda(self) -> None:
        env = self.make("id_static")
        try:
            self.assertEqual(
                env.config.robot_motion_limit_profile, "dynamicvla_panda_v4"
            )
            self.assertTrue(np.array_equal(
                np.asarray(env.config.arm_joint_velocity_limits),
                np.asarray([2.175] * 4 + [2.61] * 3),
            ))
            self.assertEqual(env.config.gripper_finger_velocity_limit, 0.20)
            self.assertEqual(env.config.low_level_velocity_guard_fraction, 1.0)
            self.assertFalse(env.config.arm_acceleration_limit_enabled)
            self.assertFalse(env.config.hand_cartesian_velocity_limit_enabled)
        finally:
            del env

    def test_reset_clears_motion_limiter_history(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1004)
            requested = env.model.actuator_ctrlrange[:, 1].copy()
            requested[7] = 0.0
            env.step(requested)
            _, info = env.reset(randomize=False, seed=1004)
            self.assertEqual(info["motion_limit_active_ratio"], 0.0)
            self.assertTrue(np.array_equal(
                info["applied_arm_velocity"], np.zeros(7)
            ))
            self.assertTrue(np.array_equal(
                info["applied_action"], env.ready_ctrl
            ))
        finally:
            del env

    def test_low_level_guard_brakes_without_clipping_state_velocity(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1006)
            action = env.ready_ctrl.copy()
            action[0] += 0.2
            original_velocity = 1.1 * env._arm_velocity_limits[0]
            env.data.qvel[env.arm_dof_adr[0]] = original_velocity
            guarded, active = env._velocity_guarded_action(action)
            self.assertTrue(active)
            self.assertEqual(
                guarded[0], env.data.qpos[env.arm_qpos_adr[0]]
            )
            self.assertEqual(
                env.data.qvel[env.arm_dof_adr[0]], original_velocity
            )
        finally:
            del env

    def test_low_level_guard_does_not_derate_dynamicvla_limit(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1007)
            action = env.ready_ctrl.copy()
            action[0] += 0.2
            env.data.qvel[env.arm_dof_adr[0]] = (
                0.99 * env._arm_velocity_limits[0]
            )
            original_jacobian = env._hand_jacobian
            env._hand_jacobian = lambda: (
                np.zeros((3, env.model.nv)),
                np.zeros((3, env.model.nv)),
            )
            try:
                guarded, active = env._velocity_guarded_action(action)
            finally:
                env._hand_jacobian = original_jacobian
            self.assertFalse(active)
            self.assertGreater(
                guarded[0], env.data.qpos[env.arm_qpos_adr[0]]
            )
        finally:
            del env

    def test_low_level_guard_brakes_on_cartesian_speed(self) -> None:
        env = self.make("id_static")
        try:
            env.config.hand_cartesian_velocity_limit_enabled = True
            env.reset(randomize=False, seed=1008)
            action = env.ready_ctrl.copy()
            action[:7] += 0.05
            original_velocity = np.zeros(7)
            original_velocity[4] = env._arm_velocity_limits[4] * 0.70
            env.data.qvel[env.arm_dof_adr] = original_velocity

            original_jacobian = env._hand_jacobian
            env._hand_jacobian = lambda: (
                np.zeros((3, env.model.nv)),
                np.vstack((
                    np.zeros(env.model.nv),
                    np.zeros(env.model.nv),
                    np.eye(1, env.model.nv, env.arm_dof_adr[4])[0] * 2.0,
                )),
            )
            try:
                guarded, active = env._velocity_guarded_action(action)
            finally:
                env._hand_jacobian = original_jacobian

            self.assertTrue(active)
            self.assertTrue(np.array_equal(
                guarded[:7], env.data.qpos[env.arm_qpos_adr]
            ))
            self.assertTrue(np.array_equal(
                env.data.qvel[env.arm_dof_adr], original_velocity
            ))
        finally:
            del env

    def test_unconfirmed_grasp_tolerates_brief_candidate_dropout(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1007)
            env.grasp_state = GraspState(
                body_id=env.target_body_id,
                candidate_time=float(env.data.time),
                bilateral_confirmed=False,
                last_bilateral_time=float(env.data.time),
                lost_contact_time=0.0,
            )
            env._physical_grasp_candidate = lambda: None
            env._update_physical_grasp_state(gripper_closed=True)
            self.assertIsNotNone(env.grasp_state)

            steps = math.ceil(
                env.config.grasp_candidate_gap_seconds
                / env.model.opt.timestep
            )
            for _ in range(steps):
                env._update_physical_grasp_state(gripper_closed=True)
            self.assertIsNone(env.grasp_state)
        finally:
            del env

    def test_confirmed_grasp_retains_bilateral_finger_body_contact(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1008)
            env.grasp_state = GraspState(
                body_id=env.target_body_id,
                candidate_time=float(env.data.time),
                bilateral_confirmed=True,
                last_bilateral_time=float(env.data.time),
                lost_contact_time=0.0,
            )
            env._physical_grasp_candidate = lambda: None
            env._finger_body_contact_pairs = lambda cable_body=None: [
                (env.target_body_id, env.left_finger_id),
                (env.target_body_id, env.right_finger_id),
            ]
            steps = math.ceil(
                1.5 * env.config.grasp_loss_seconds / env.model.opt.timestep
            )
            for _ in range(steps):
                env._update_physical_grasp_state(gripper_closed=True)
            self.assertIsNotNone(env.grasp_state)
            self.assertEqual(env.grasp_state.lost_contact_time, 0.0)
        finally:
            del env

    def test_confirmed_grasp_still_clears_without_bilateral_finger_contact(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1009)
            env.grasp_state = GraspState(
                body_id=env.target_body_id,
                candidate_time=float(env.data.time),
                bilateral_confirmed=True,
                last_bilateral_time=float(env.data.time),
                lost_contact_time=0.0,
            )
            env._physical_grasp_candidate = lambda: None
            env._finger_body_contact_pairs = lambda cable_body=None: [
                (env.target_body_id, env.left_finger_id),
            ]
            steps = math.ceil(
                env.config.grasp_loss_seconds / env.model.opt.timestep
            ) + 1
            for _ in range(steps):
                env._update_physical_grasp_state(gripper_closed=True)
            self.assertIsNone(env.grasp_state)
            self.assertEqual(
                env.last_grasp_break["reason"], "lost_physical_pad_contact"
            )
        finally:
            del env

    def test_curved_grasp_uses_two_node_contact_radius(self) -> None:
        config = EnvConfig(grasp_contact_index_radius=2)
        self.assertEqual(config.episode_seconds, 15.0)
        self.assertEqual(config.grasp_contact_index_radius, 2)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            EnvConfig(grasp_contact_index_radius=-1)

    def test_scripted_policy_uses_measured_motion_delay_compensation(self) -> None:
        config = PolicyConfig()
        self.assertEqual(config.prediction_horizon, 0.30)
        self.assertEqual(config.approach_prediction_horizon, 0.30)
        self.assertEqual(config.close_prediction_horizon, 0.12)
        self.assertEqual(config.target_filter_alpha, 0.10)
        self.assertEqual(config.intercept_y_limits, (-0.43, 0.43))
        self.assertEqual(config.approach_fast_distance, 0.12)
        self.assertEqual(config.approach_linear_velocity_limit, 0.90)
        self.assertEqual(config.intercept_linear_velocity_limit, 0.95)
        self.assertEqual(config.precision_linear_velocity_limit, 0.65)
        self.assertEqual(config.approach_orientation_gain, 0.65)
        self.assertEqual(config.precision_orientation_gain, 1.0)
        self.assertEqual(config.policy_joint_velocity_fraction, 1.0)
        self.assertEqual(config.ik_target_horizon, 0.11)
        with self.assertRaisesRegex(ValueError, "prediction_horizon"):
            PolicyConfig(prediction_horizon=-0.01)
        with self.assertRaisesRegex(ValueError, "target_filter_alpha"):
            PolicyConfig(target_filter_alpha=0.0)
        with self.assertRaisesRegex(ValueError, "policy_joint_velocity_fraction"):
            PolicyConfig(policy_joint_velocity_fraction=1.01)
        with self.assertRaisesRegex(ValueError, "intercept_y_limits"):
            PolicyConfig(intercept_y_limits=(0.5, -0.5))

    def test_prediction_uses_total_target_velocity_for_every_scenario(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1010)
            policy = DynamicCableGraspPolicy(env)
            policy.phase = Phase.APPROACH
            position = np.array([0.50, 0.00, 0.05])
            policy.filtered_target = position.copy()
            env.target_position = lambda: position.copy()
            env.target_velocity = lambda: np.array([0.30, 0.20, 0.00])
            predicted = policy._predicted_segment()
            expected_unfiltered_offset = np.array([0.09, 0.06, 0.0])
            self.assertTrue(np.allclose(
                predicted,
                position + 0.10 * expected_unfiltered_offset,
                rtol=0.0,
                atol=1e-12,
            ))
        finally:
            del env

    def test_environment_rejects_nonfinite_actions(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1005)
            action = env.ready_ctrl.copy()
            action[0] = math.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                env.step(action)
        finally:
            del env

    def test_approach_requires_position_and_safe_tilt_but_allows_yaw_error(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1008)
            policy = DynamicCableGraspPolicy(env)
            policy.phase = Phase.APPROACH
            policy.phase_start = float(env.data.time)
            policy._predicted_segment = lambda prediction_horizon=None: (
                env.hand_position - np.array([0.0, 0.0, 0.20])
            )

            # reset目标与当前姿态只有90度平面内偏航；夹爪仍然竖直，可以进入截获。
            policy.action()
            self.assertIs(policy.phase, Phase.INTERCEPT)

            policy.phase = Phase.APPROACH
            policy.phase_start = float(env.data.time)
            policy.desired_approach_axis = np.array([1.0, 0.0, 0.0])
            policy.action()
            self.assertIs(policy.phase, Phase.APPROACH)
        finally:
            del env

    def test_approach_accepts_bounded_motion_lag_but_rejects_large_error(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1012)
            policy = DynamicCableGraspPolicy(env)
            rotation = env.data.xmat[env.hand_id].reshape(3, 3)
            policy.desired_quat = rotation_to_quat(rotation)
            policy.desired_approach_axis = rotation[:, 2].copy()
            policy.phase = Phase.APPROACH
            policy.phase_start = float(env.data.time)

            policy._predicted_segment = lambda prediction_horizon=None: (
                env.hand_position
                + np.array([0.15, 0.0, -0.20])
            )
            policy.action()
            self.assertIs(policy.phase, Phase.APPROACH)

            policy._predicted_segment = lambda prediction_horizon=None: (
                env.hand_position
                + np.array([
                    0.9 * policy.config.approach_position_tolerance,
                    0.0,
                    -0.20,
                ])
            )
            policy.action()
            self.assertIs(policy.phase, Phase.INTERCEPT)
        finally:
            del env

    def test_approach_timeout_recovers_instead_of_descending(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1009)
            policy = DynamicCableGraspPolicy(env)
            policy.phase = Phase.APPROACH
            policy.phase_start = (
                float(env.data.time) - policy.config.approach_timeout - 0.1
            )
            policy.action()
            self.assertIs(policy.phase, Phase.RECOVER)
            self.assertGreater(policy.recover_goal[2], policy.recover_start[2])
        finally:
            del env

    def test_far_approach_prioritizes_translation_without_skipping_safety_gate(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1011)
            policy = DynamicCableGraspPolicy(env)
            policy.phase = Phase.APPROACH
            policy.phase_start = float(env.data.time)
            far_target = env.hand_position + np.array([0.0, 0.30, -0.10])
            policy._predicted_segment = lambda prediction_horizon=None: (
                far_target - np.array([0.0, 0.0, 0.20])
            )

            action = policy.action()
            self.assertIs(policy.phase, Phase.APPROACH)
            self.assertTrue(np.all(np.isfinite(action)))

            control_dt = env.model.opt.timestep * env.config.frame_skip
            requested_velocity = (
                action[:7] - env.data.qpos[env.arm_qpos_adr]
            ) / policy.config.ik_target_horizon
            self.assertLessEqual(
                np.max(
                    np.abs(requested_velocity)
                    / np.asarray(env.config.arm_joint_velocity_limits)
                ),
                policy.config.policy_joint_velocity_fraction + 1e-9,
            )
            self.assertGreater(policy.config.ik_target_horizon, control_dt)
        finally:
            del env

    def test_intercept_rejects_unsafe_gripper_orientation(self) -> None:
        env = self.make("id_static")
        try:
            env.reset(randomize=False, seed=1010)
            policy = DynamicCableGraspPolicy(env)
            policy.phase = Phase.INTERCEPT
            policy.phase_start = float(env.data.time)
            # 将安全接近轴改为水平，模拟夹爪横倒；必须撤离而不是继续下探。
            policy.desired_approach_axis = np.array([1.0, 0.0, 0.0])
            policy.action()
            self.assertIs(policy.phase, Phase.RECOVER)
        finally:
            del env

    def test_seed_reproduces_initial_scene_and_motion_profile(self) -> None:
        env = self.make("ood_combined_l1_dynamics_high_stochastic")
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
            "id_rigid_l1_nominal": (False, True),
            "id_shape_nominal_current": (True, False),
            "id_combined_l1_nominal": (True, True),
        }
        for name, (shape_expected, rigid_expected) in expected.items():
            with self.subTest(name=name):
                env = self.make(name)
                try:
                    env.reset(seed=17)
                    if env.config.motion_mode in {"rigid", "combined"}:
                        env.data.time = RIGID_MOTION_START_TIME + 0.2
                    env.data.xfrc_applied[:] = 0.0
                    env._apply_cable_disturbance()
                    shape = np.linalg.norm(env._last_shape_acceleration) > 1e-10
                    rigid = np.linalg.norm(
                        env._last_rigid_translation_acceleration
                    ) + np.linalg.norm(
                        env._last_rigid_rotation_acceleration
                    ) + np.linalg.norm(
                        env._last_rigid_shape_hold_acceleration
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

    def test_level1_target_is_straight_and_constant_speed_between_turns(self) -> None:
        env = self.make("id_rigid_l1_nominal")
        try:
            env.reset(seed=42)
            dt = 1e-4
            time = RIGID_MOTION_START_TIME + 0.20
            before = env._rigid_motion_target(time - dt)
            after = env._rigid_motion_target(time + dt)
            velocity = (after - before) / (2.0 * dt)
            self.assertAlmostEqual(velocity[0], 0.0, delta=1e-10)
            self.assertAlmostEqual(
                np.linalg.norm(velocity),
                env.config.rigid_motion_nominal_speed,
                delta=1e-8,
            )
            times = np.linspace(
                RIGID_MOTION_START_TIME,
                RIGID_MOTION_START_TIME + env._rigid_motion_duration(),
                101,
            )
            positions = np.array([env._rigid_motion_target(t) for t in times])
            self.assertTrue(np.all(np.diff(positions[:, 1]) >= -1e-12))
            self.assertTrue(np.allclose(positions[0], [0.0, 0.0]))
            self.assertTrue(np.allclose(
                positions[-1], [0.0, RIGID_MOTION_TRAVEL]
            ))
            initial_com = np.average(
                env.data.xpos[env.cable_ids, :2],
                axis=0,
                weights=env.cable_mass,
            )
            self.assertAlmostEqual(initial_com[1], RIGID_MOTION_START_Y)
        finally:
            del env

    def test_level2_target_has_curvature_and_matches_nominal_speed_scale(self) -> None:
        env = self.make("id_rigid_l2_nominal")
        try:
            env.reset(seed=43)
            dt = 1e-3
            positions = np.array([
                env._rigid_motion_target(
                    RIGID_MOTION_START_TIME + 0.30 + offset * dt
                )
                for offset in (-1, 0, 1)
            ])
            velocity = (positions[2] - positions[0]) / (2.0 * dt)
            acceleration = (positions[2] - 2.0 * positions[1] + positions[0]) / (dt * dt)
            cross = velocity[0] * acceleration[1] - velocity[1] * acceleration[0]
            self.assertGreater(abs(cross), 1e-3)
            self.assertTrue(np.allclose(
                env._rigid_motion_target(
                    RIGID_MOTION_START_TIME + env._rigid_motion_duration() + 1.0
                ),
                [0.0, RIGID_MOTION_TRAVEL],
            ))
            low = self.make("id_rigid_l2_low")
            high = self.make("id_rigid_l2_high")
            try:
                self.assertAlmostEqual(
                    high.config.motion_frequency_scale
                    / low.config.motion_frequency_scale,
                    2.25,
                )
            finally:
                del low, high
        finally:
            del env

    def test_long_l1_l2_path_terminates_on_real_com_boundary(self) -> None:
        level1 = self.make("id_rigid_l1_high")
        level2 = self.make("id_rigid_l2_high")
        try:
            self.assertEqual(RIGID_MOTION_TRAVEL, 1.4)
            self.assertGreater(level1._rigid_motion_duration(), 3.6)
            self.assertGreater(level2._rigid_motion_duration(), 3.6)
            end_time = (
                RIGID_MOTION_START_TIME + level1._rigid_motion_duration()
            )
            nominal_end = level1._rigid_motion_target(end_time)
            self.assertTrue(np.array_equal(
                nominal_end, level1._rigid_motion_target(end_time + 1.0)
            ))
            self.assertEqual(
                level1.info()["rigid_motion_control"],
                "actual_progress_velocity_v1",
            )

            level1.reset(seed=43)
            level1.config.episode_seconds = 30.0
            level1.data.time = (
                RIGID_MOTION_START_TIME + level1._rigid_motion_duration()
            )
            _, _, _, truncated, info = level1.step(level1.ready_ctrl)
            self.assertTrue(info["rigid_motion_nominal_finished"])
            self.assertFalse(info["rigid_motion_finished"])
            self.assertFalse(info["rigid_motion_released"])
            self.assertFalse(truncated)
            self.assertIsNone(info["termination_reason"])

            delta_y = (
                level1.config.rigid_motion_exit_y
                + 0.01
                - level1.rigid_motion_com_y
            )
            level1.data.qpos[level1.cable_free_qadr + 1] += delta_y
            level1.data.qvel[
                level1.cable_free_dadr:level1.cable_free_dadr + 6
            ] = 0.0
            mujoco.mj_forward(level1.model, level1.data)
            _, _, _, truncated, info = level1.step(level1.ready_ctrl)
            self.assertTrue(info["rigid_motion_finished"])
            self.assertTrue(truncated)
            self.assertEqual(
                info["termination_reason"], "rigid_motion_boundary_crossed"
            )

            level1.rigid_motion_released = True
            _, _, _, truncated, info = level1.step(level1.ready_ctrl)
            self.assertFalse(truncated)
            self.assertIsNone(info["termination_reason"])
        finally:
            del level1, level2

    def test_rigid_motion_uses_paired_curved_shape_and_rotation(self) -> None:
        level1 = self.make("id_rigid_l1_nominal", seed=45)
        level2 = self.make("id_rigid_l2_nominal", seed=45)
        try:
            _, info1 = level1.reset(seed=45)
            _, info2 = level2.reset(seed=45)
            reference1 = level1._rigid_reference_xy
            reference2 = level2._rigid_reference_xy
            centered1 = reference1 - np.average(
                reference1, axis=0, weights=level1.cable_mass
            )
            centered2 = reference2 - np.average(
                reference2, axis=0, weights=level2.cable_mass
            )
            self.assertGreater(np.linalg.svd(centered1)[1][1], 0.01)
            self.assertTrue(np.array_equal(centered1, centered2))
            segment_lengths = np.linalg.norm(np.diff(reference1, axis=0), axis=1)
            self.assertLess(np.ptp(segment_lengths), 1e-7)
            self.assertEqual(
                info1["rigid_initial_shape_family"],
                info2["rigid_initial_shape_family"],
            )
            self.assertIn(info1["rigid_initial_shape_family"], {"c", "s", "spline"})
            self.assertEqual(
                info1["rigid_motion_rotation_sign"],
                info2["rigid_motion_rotation_sign"],
            )
            start = RIGID_MOTION_START_TIME
            end = start + level1._rigid_motion_duration()
            self.assertEqual(level1._rigid_motion_rotation_target(start), 0.0)
            self.assertAlmostEqual(
                abs(level1._rigid_motion_rotation_target(end)),
                RIGID_MOTION_ROTATION,
            )
            halfway = level1._rigid_motion_target(
                start + 0.5 * level1._rigid_motion_duration()
            )
            level1.data.qpos[level1.cable_free_qadr:level1.cable_free_qadr + 2] += (
                halfway
            )
            mujoco.mj_forward(level1.model, level1.data)
            level1.data.time = start + 0.5 * level1._rigid_motion_duration()
            level1._apply_cable_disturbance()
            self.assertGreater(
                np.linalg.norm(level1._last_rigid_rotation_acceleration), 0.0
            )
        finally:
            del level1, level2

    def test_all_new_scenarios_share_curved_initial_distribution(self) -> None:
        names = (
            "id_static", "id_rigid_l1_nominal", "id_shape_nominal_current",
            "id_combined_l1_nominal", "id_rigid_l2_nominal",
            "id_combined_l2_nominal",
        )
        environments = [self.make(name, seed=46) for name in names]
        try:
            centered = []
            for env in environments:
                _, info = env.reset(seed=46)
                reference = env._rigid_reference_xy.copy()
                reference -= np.average(
                    reference, axis=0, weights=env.cable_mass
                )
                self.assertGreater(np.linalg.svd(reference)[1][1], 0.01)
                self.assertIn(info["initial_shape_family"], {"c", "s", "spline"})
                centered.append(reference.copy())
            for reference in centered[1:]:
                self.assertTrue(np.allclose(
                    centered[0], reference, rtol=0.0, atol=1e-12
                ))

            legacy = CableGraspEnv(EnvConfig(
                seed=46,
                episode_seconds=0.1,
            ))
            try:
                observation, info = legacy.reset(seed=46)
                legacy_xy = observation["cable_positions"][:, :2]
                legacy_xy -= legacy_xy.mean(axis=0)
                self.assertLess(np.linalg.svd(legacy_xy)[1][1], 1e-10)
                self.assertEqual(info["initial_shape_family"], "none")
            finally:
                del legacy
        finally:
            for env in environments:
                del env

    def test_rigid_shape_hold_has_zero_net_force_and_torque(self) -> None:
        env = self.make("id_rigid_l1_nominal", seed=47)
        try:
            env.reset(seed=47)
            address = int(env.cable_ball_qadr[len(env.cable_ball_qadr) // 2])
            quaternion = env.data.qpos[address:address + 4]
            angle = 2.0 * math.atan2(float(quaternion[3]), float(quaternion[0]))
            angle += math.radians(3.0)
            quaternion[:] = [math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)]
            mujoco.mj_forward(env.model, env.data)
            env.data.time = RIGID_MOTION_START_TIME + 0.3
            env._apply_cable_disturbance()
            acceleration = env._last_rigid_shape_hold_acceleration
            force = env.cable_mass[:, None] * acceleration
            positions = env.data.xpos[env.cable_ids]
            center = np.average(positions, axis=0, weights=env.cable_mass)
            torque = np.cross(positions - center, force).sum(axis=0)
            self.assertGreater(np.linalg.norm(acceleration), 0.0)
            self.assertLess(np.linalg.norm(force.sum(axis=0)), 1e-10)
            self.assertLess(abs(float(torque[2])), 1e-10)
        finally:
            del env

    def test_rigid_motion_waits_for_confirmed_grasp_before_releasing_drive(self) -> None:
        env = self.make("id_rigid_l1_nominal")
        try:
            env.reset(seed=47)
            env._last_contact_count = 1
            env.grasp_state = GraspState(
                body_id=env.target_body_id,
                candidate_time=float(env.data.time),
                bilateral_confirmed=False,
                last_bilateral_time=float(env.data.time),
                lost_contact_time=0.0,
            )
            env._update_rigid_motion_release_state()
            self.assertFalse(env.rigid_motion_released)

            env.grasp_state.bilateral_confirmed = True
            env._update_rigid_motion_release_state()
            self.assertTrue(env.rigid_motion_released)

            # 该状态需要锁存，防止抓取后的短暂接触抖动重新启动整体驱动。
            env.grasp_state = None
            env._update_rigid_motion_release_state()
            self.assertTrue(env.rigid_motion_released)
        finally:
            del env

    def test_rigid_motion_releases_environment_drive_after_confirmed_grasp(self) -> None:
        env = self.make("id_rigid_l1_nominal")
        try:
            env.reset(seed=44)
            env.data.time = RIGID_MOTION_START_TIME + 0.2
            env._apply_cable_disturbance()
            self.assertGreater(
                np.linalg.norm(env._last_rigid_translation_acceleration), 0.0
            )
            env.rigid_motion_released = True
            env.data.xfrc_applied[:] = 0.0
            env._apply_cable_disturbance()
            self.assertTrue(np.array_equal(
                env._last_rigid_translation_acceleration,
                np.zeros_like(env._last_rigid_translation_acceleration),
            ))
            self.assertTrue(np.array_equal(
                env._last_rigid_rotation_acceleration,
                np.zeros_like(env._last_rigid_rotation_acceleration),
            ))
            self.assertTrue(np.array_equal(
                env._last_rigid_shape_hold_acceleration,
                np.zeros_like(env._last_rigid_shape_hold_acceleration),
            ))
        finally:
            del env

    def test_rigid_motion_does_not_accumulate_catch_up_error_while_blocked(self) -> None:
        environments = (
            self.make("id_rigid_l1_nominal", seed=48),
            self.make("id_rigid_l2_nominal", seed=48),
        )
        try:
            for env in environments:
                env.reset(seed=48)
                stopped_velocity = np.zeros((len(env.cable_ids), 2))
                early_translation, early_rotation = env._rigid_motion_acceleration(
                    RIGID_MOTION_START_TIME + 0.1, stopped_velocity,
                )
                late_translation, late_rotation = env._rigid_motion_acceleration(
                    RIGID_MOTION_START_TIME + 10.0, stopped_velocity,
                )

                # A ten-second obstruction cannot build a larger release command.
                self.assertTrue(np.array_equal(
                    early_translation, late_translation,
                ))
                self.assertTrue(np.array_equal(early_rotation, late_rotation))
                self.assertLessEqual(
                    np.linalg.norm(late_translation[0, :2]),
                    env.config.rigid_translation_max_acceleration + 1e-12,
                )

            level1 = environments[0]
            nominal_velocity = np.broadcast_to(
                [0.0, level1._rigid_motion_speed()],
                (len(level1.cable_ids), 2),
            ).copy()
            translation, _ = level1._rigid_motion_acceleration(
                RIGID_MOTION_START_TIME + 10.0, nominal_velocity,
            )
            # At nominal forward speed there is no hidden longitudinal pull.
            self.assertAlmostEqual(float(translation[0, 1]), 0.0, places=12)

            # With the environment's 50 Hz command interval, releasing a fully
            # blocked cable approaches nominal speed monotonically and cannot
            # overshoot it even when table friction disappears completely.
            control_dt = level1.model.opt.timestep * level1.config.frame_skip
            simulated_speed = 0.0
            for _ in range(20):
                simulated_velocity = np.broadcast_to(
                    [0.0, simulated_speed], (len(level1.cable_ids), 2),
                ).copy()
                translation, _ = level1._rigid_motion_acceleration(
                    RIGID_MOTION_START_TIME + 10.0, simulated_velocity,
                )
                simulated_speed += float(translation[0, 1]) * control_dt
                self.assertLessEqual(
                    simulated_speed, level1._rigid_motion_speed() + 1e-12,
                )
        finally:
            for env in environments:
                del env

    def test_combined_motion_releases_only_rigid_drive_after_confirmed_grasp(self) -> None:
        env = self.make("id_combined_l1_nominal")
        try:
            env.reset(seed=46)
            env.data.time = RIGID_MOTION_START_TIME + 0.2
            env._apply_cable_disturbance()
            self.assertGreater(np.linalg.norm(env._last_shape_acceleration), 0.0)
            self.assertGreater(
                np.linalg.norm(env._last_rigid_translation_acceleration), 0.0
            )
            self.assertTrue(np.array_equal(
                env._last_rigid_shape_hold_acceleration,
                np.zeros_like(env._last_rigid_shape_hold_acceleration),
            ))
            env.rigid_motion_released = True
            env._apply_cable_disturbance()
            self.assertGreater(np.linalg.norm(env._last_shape_acceleration), 0.0)
            self.assertTrue(np.array_equal(
                env._last_rigid_translation_acceleration,
                np.zeros_like(env._last_rigid_translation_acceleration),
            ))
            self.assertTrue(np.array_equal(
                env._last_rigid_rotation_acceleration,
                np.zeros_like(env._last_rigid_rotation_acceleration),
            ))
        finally:
            del env

    def test_legacy_default_preserves_original_shape_formula(self) -> None:
        env = CableGraspEnv(EnvConfig(
            seed=71,
            episode_seconds=0.1,
        ))
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
                "id_shape_nominal_current", "id_rigid_l1_nominal",
                "id_combined_l1_nominal",
            )
        }
        try:
            for env in environments.values():
                env.reset(seed=81)
                env._apply_cable_disturbance()
            shape = environments["id_shape_nominal_current"]
            rigid = environments["id_rigid_l1_nominal"]
            combined = environments["id_combined_l1_nominal"]
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

    def test_environment_force_has_no_table_boundary_component(self) -> None:
        env = self.make("id_combined_l1_nominal", seed=82)
        try:
            env.reset(seed=82)
            env.data.xfrc_applied[:] = 0.0
            env._apply_cable_disturbance()
            expected = (
                env._last_shape_acceleration
                + env._last_rigid_translation_acceleration
                + env._last_rigid_rotation_acceleration
                + env._last_rigid_shape_hold_acceleration
            )
            applied = (
                env.data.xfrc_applied[env.cable_ids, :3]
                / env.cable_mass[:, None]
            )
            self.assertTrue(np.allclose(
                applied, expected, rtol=0.0, atol=1e-12
            ))
            self.assertNotIn("boundary_acceleration_rms", env.info())
        finally:
            del env

    def test_table_is_enlarged_without_edge_barrier(self) -> None:
        env = self.make("id_static")
        try:
            self.assertTrue(np.array_equal(
                env.model.geom_size[env.table_geom_id, :2], [1.20, 1.20]
            ))
            self.assertEqual(
                mujoco.mj_name2id(
                    env.model, mujoco.mjtObj.mjOBJ_GEOM, "table_edge"
                ),
                -1,
            )
        finally:
            del env

    def test_disturbance_does_not_read_grasp_state(self) -> None:
        env = self.make("id_combined_l1_nominal")
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
        nominal = self.make("id_combined_l1_nominal")
        short = self.make("ood_combined_l1_length_short")
        soft = self.make("ood_combined_l1_material_soft")
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
        }
        tracker.record(0.1, current, info)
        summary = tracker.summary()
        self.assertLess(float(summary["shape_change_rms_m"]), 1e-12)
        self.assertAlmostEqual(
            float(summary["rigid_rotation_max_rad"]), angle, delta=1e-12
        )


if __name__ == "__main__":
    unittest.main()
