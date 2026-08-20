"""供强化学习使用的 Gymnasium 接口。

该文件只负责把观测和动作转换成固定长度向量，并计算训练奖励。线缆扰动、
接触、纯摩擦夹持和成功判定仍完全由 CableGraspEnv 决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from runtime_config import configure_mujoco_runtime

configure_mujoco_runtime()

import gymnasium as gym
import mujoco
import numpy as np

from cable_grasp_env import CableGraspEnv, EnvConfig, quat_error, rotation_to_quat
from experiment_scenarios import ScenarioConfig, get_scenario


@dataclass
class RLConfig:
    """RL接口参数，不改变底层线缆物理。"""

    cable_sample_count: int = 14
    cable_position_scale: float = 0.50
    cable_velocity_scale: float = 1.0
    translation_delta_scale: float = 0.010
    yaw_delta_scale: float = 0.020
    ik_damping: float = 0.05
    graspable_end_fraction: float = 0.25
    alignment_gate_distance: float = 0.10
    gripper_close_threshold: float = -0.35
    gripper_open_threshold: float = 0.35
    gripper_closed_ctrl: float = 0.0
    secured_lift_delta: float = 0.03
    secured_confirm_seconds: float = 0.10
    secured_contact_loss_grace_seconds: float = 0.06
    lift_credit_cap: float = 0.12
    cable_lift_credit_cap: float = 0.30
    reward_reach_progress: float = 6.0
    reward_alignment_progress: float = 0.5
    reward_new_contact: float = 0.15
    reward_new_pinch: float = 0.75
    reward_new_secured_grasp: float = 3.0
    reward_lift_progress: float = 20.0
    reward_cable_lift_progress: float = 3.0
    reward_strict_hold_progress: float = 0.8
    reward_success: float = 25.0
    reward_slip: float = -3.0
    reward_active_open_after_secured: float = -8.0
    reward_gripper_switch: float = -0.02
    reward_time_step: float = -0.002
    reward_action_magnitude: float = -0.0005
    reward_action_rate: float = -0.01
    reward_post_grasp_arm_action_magnitude: float = -0.02
    reward_post_grasp_arm_action_rate: float = -0.05

    def __post_init__(self) -> None:
        if self.cable_sample_count != 14:
            raise ValueError("cable_sample_count must be 14 for the 99-D baseline")
        for name in (
            "cable_position_scale", "cable_velocity_scale",
            "translation_delta_scale", "yaw_delta_scale", "ik_damping",
            "alignment_gate_distance", "secured_lift_delta",
            "secured_confirm_seconds", "secured_contact_loss_grace_seconds",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.graspable_end_fraction < 0.5:
            raise ValueError("graspable_end_fraction must be in [0, 0.5)")
        if not self.gripper_close_threshold < self.gripper_open_threshold:
            raise ValueError("gripper hysteresis thresholds must be increasing")


class RLCableGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """Panda dynamic-cable grasping with a conventional task-space RL API.

    The policy observes the whole DLO through 14 arc-length samples and receives
    no privileged target segment.  Five-dimensional TCP actions are converted
    by damped IK; the shared environment enforces the final robot speed limits.
    """

    metadata = {"render_modes": []}
    # 实际夹持中心由底层环境统一定义，RL观测不再维护重复的局部坐标常量。

    OBSERVATION_NAMES = (
        *(
            f"dlo_point_{point:02d}_{quantity}_{axis}"
            for point in range(14)
            for quantity in ("position_tcp", "velocity_tcp")
            for axis in ("x", "y", "z")
        ),
        *(f"arm_qpos_{i}" for i in range(1, 8)),
        *(f"arm_qvel_{i}" for i in range(1, 8)),
        "gripper_aperture",
    )
    ACTION_NAMES = (
        "delta_tcp_x", "delta_tcp_y", "delta_tcp_z",
        "delta_tcp_yaw", "gripper",
    )

    def __init__(
        self,
        *,
        seed: int = 20260804,
        disturbance_strength: float = 1.5,
        episode_seconds: float = 15.0,
        env_config: EnvConfig | None = None,
        scenario_names: Sequence[str] | None = None,
        rl_config: RLConfig | None = None,
    ):
        super().__init__()
        self.rl_config = rl_config or RLConfig()
        self._scenario_configs: tuple[ScenarioConfig, ...] = tuple(
            get_scenario(name) for name in (scenario_names or ())
        )
        self._scenario_by_name = {
            scenario.name: scenario for scenario in self._scenario_configs
        }
        self._active_scenario_names = tuple(self._scenario_by_name)
        self._final_disturbance_scale = float(disturbance_strength) / 1.5
        self._motion_difficulty = 1.0
        if self._scenario_configs and env_config is not None:
            raise ValueError("env_config and scenario_names are mutually exclusive")
        if self._scenario_configs:
            # 一个 Gym 环境只编译一次模型；训练分布可切换运动规律，但不能在
            # episode 间偷换需要重新编译的线缆材质/长度。
            physical_scales = {
                (
                    scenario.cable_length_scale,
                    scenario.cable_density_scale,
                    scenario.cable_stiffness_scale,
                    scenario.cable_damping_scale,
                    scenario.cable_friction_scale,
                )
                for scenario in self._scenario_configs
            }
            if len(physical_scales) != 1:
                raise ValueError(
                    "one RLCableGraspEnv cannot sample scenarios with different "
                    "compiled cable properties"
                )
            first_scenario = self._scenario_configs[0]
            first_overrides = first_scenario.to_env_overrides()
            first_overrides["disturbance_strength"] = (
                first_scenario.disturbance_strength
                * self._final_disturbance_scale
            )
            env_config = EnvConfig(
                seed=seed,
                episode_seconds=episode_seconds,
                scenario_name=first_scenario.name,
                scenario_id=first_scenario.scenario_id,
                scenario_split=first_scenario.split.value,
                camera_observation_enabled=False,
                **first_overrides,
            )
        self.base_env = CableGraspEnv(
            env_config
            if env_config is not None
            else EnvConfig(
                seed=seed,
                disturbance_strength=disturbance_strength,
                episode_seconds=episode_seconds,
                camera_observation_enabled=False,
            )
        )
        # The wrapper converts task-space actions to joint targets, while the
        # shared environment remains the final authority on robot capability.
        self.base_env.config.hand_cartesian_velocity_limit_enabled = True
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -10.0, 10.0, shape=(len(self.OBSERVATION_NAMES),), dtype=np.float32
        )
        self._previous_distance = 0.0
        self._episode_return = 0.0
        self._episode_steps = 0
        self._episode_initial_cable_z = np.zeros(len(self.base_env.cable_ids))
        self._secured_candidate_hold = 0.0
        self._secured_contact_loss_hold = 0.0
        self._strict_success_hold = 0.0
        self._ever_secured_grasp = False
        self._last_grasp_status = self._empty_grasp_status()
        self._locked_body_id: int | None = None
        self._gripper_closed = False
        self._previous_action = np.zeros(5)
        self._previous_alignment_score: float | None = None
        self._nearest_graspable_distance = 0.0
        self._nearest_graspable_segment_index = -1
        self._alignment_score = 0.0
        self._desired_hand_rotation = np.eye(3)
        self._last_ik_velocity_scale = 1.0
        self._gripper_switch_event = False
        self._gripper_switch_count = 0
        self._max_contacting_fingers = 0
        self._pinch_rewarded = False
        self._secured_rewarded = False
        self._lift_credit_high_water = 0.0
        self._cable_lift_credit_high_water = 0.0
        self._secured_session_active = False
        self._last_grasp_break_key: tuple[float, str] | None = None
        self._active_open_after_secured_event = False
        self._physical_slip_after_secured_event = False
        self._open_during_contact_loss_event = False
        self._active_open_after_secured_count = 0
        self._physical_slip_after_secured_count = 0
        self._open_during_contact_loss_count = 0
        self._current_grasp_loss_reason: str | None = None
        self._current_grasp_loss_causal_class: str | None = None
        self._contact_loss_penalty_applied = False
        self._active_open_penalty_applied = False
        self._strict_hold_credit_high_water = 0.0
        self._previous_secured_grasp = False
        self._post_grasp_arm_action_magnitude = 0.0
        self._post_grasp_arm_action_rate = 0.0

        joint_range = self.base_env.model.jnt_range[self.base_env.arm_joint_ids]
        self._arm_center = joint_range.mean(axis=1)
        self._arm_half_range = np.maximum(0.5 * np.ptp(joint_range, axis=1), 1e-6)

    def _select_training_scenario(
        self, explicit_seed: int | None = None,
    ) -> ScenarioConfig | None:
        if not self._scenario_configs:
            return None
        active_configs = tuple(
            self._scenario_by_name[name]
            for name in self._active_scenario_names
        )
        # 训练中的普通reset随机采样；严格评估传连续显式seed时按模循环，保证
        # core/ID场景近似等次数覆盖，而不是小样本恰好漏掉某种运动类型。
        index = (
            int(explicit_seed) % len(active_configs)
            if explicit_seed is not None
            else int(self.np_random.integers(0, len(active_configs)))
        )
        scenario = active_configs[index]
        overrides = scenario.to_env_overrides()
        amplitude_scale = 0.5 + (
            self._final_disturbance_scale - 0.5
        ) * self._motion_difficulty
        frequency_scale = (2.0 + self._motion_difficulty) / 3.0
        for name, value in overrides.items():
            if name.startswith("cable_"):
                # 已在构造时验证并编译；重复赋值仅会掩盖意外的模型/配置差异。
                continue
            if name == "disturbance_strength":
                value = scenario.disturbance_strength * amplitude_scale
            elif name == "motion_frequency_scale":
                value = scenario.frequency_scale * frequency_scale
            setattr(self.base_env.config, name, value)
        self.base_env.config.scenario_name = scenario.name
        self.base_env.config.scenario_id = scenario.scenario_id
        self.base_env.config.scenario_split = scenario.split.value
        return scenario

    @property
    def model(self) -> mujoco.MjModel:
        return self.base_env.model

    @property
    def data(self) -> mujoco.MjData:
        return self.base_env.data

    def _hand_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        jacp, jacr = self.base_env._hand_jacobian()
        return (
            (jacp @ self.data.qvel).copy(),
            (jacr @ self.data.qvel).copy(),
        )

    def _grasp_center_position(self) -> np.ndarray:
        """返回底层环境统一定义的实际两指夹持中心。"""
        return self.base_env.hand_position.copy()

    def _cable_body_velocities(self) -> np.ndarray:
        return np.asarray([
            self.base_env.body_linear_velocity(body_id)
            for body_id in self.base_env.cable_ids
        ])

    @staticmethod
    def _arc_length_samples(
        positions: np.ndarray,
        values: np.ndarray,
        count: int,
    ) -> np.ndarray:
        """Interpolate ``values`` at uniformly spaced cable arc lengths."""
        segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        if cumulative[-1] <= 1e-9:
            indices = np.rint(np.linspace(0, len(values) - 1, count)).astype(int)
            return values[indices].copy()
        queries = np.linspace(0.0, cumulative[-1], count)
        result = np.empty((count, values.shape[1]), dtype=float)
        for dimension in range(values.shape[1]):
            result[:, dimension] = np.interp(
                queries, cumulative, values[:, dimension]
            )
        return result

    def _sample_cable_state(self) -> tuple[np.ndarray, np.ndarray]:
        positions = self.data.xpos[self.base_env.cable_ids].copy()
        velocities = self._cable_body_velocities()
        return (
            self._arc_length_samples(
                positions, positions, self.rl_config.cable_sample_count
            ),
            self._arc_length_samples(
                positions, velocities, self.rl_config.cable_sample_count
            ),
        )

    def _nearest_graspable_segment(
        self, point: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, int, float]:
        """Return the closest point and a smooth tangent on the cable middle."""
        positions = self.data.xpos[self.base_env.cable_ids]
        node_count = len(positions)
        margin = max(
            1, int(np.floor(node_count * self.rl_config.graspable_end_fraction))
        )
        first = min(margin, node_count - 2)
        stop = max(first + 1, node_count - margin - 1)
        segment_indices = np.arange(first, stop)
        starts = positions[segment_indices]
        vectors = positions[segment_indices + 1] - starts
        lengths_squared = np.sum(vectors * vectors, axis=1)
        alpha = np.sum((point - starts) * vectors, axis=1) / np.maximum(
            lengths_squared, 1e-12
        )
        alpha = np.clip(alpha, 0.0, 1.0)
        projected = starts + alpha[:, None] * vectors
        distances = np.linalg.norm(projected - point, axis=1)
        selected = int(np.argmin(distances))
        segment_index = int(segment_indices[selected])
        selected_alpha = float(alpha[selected])

        node_tangents = np.empty_like(positions)
        node_tangents[0] = positions[1] - positions[0]
        node_tangents[-1] = positions[-1] - positions[-2]
        node_tangents[1:-1] = positions[2:] - positions[:-2]
        tangent = (
            (1.0 - selected_alpha) * node_tangents[segment_index]
            + selected_alpha * node_tangents[segment_index + 1]
        )
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-9:
            tangent = vectors[selected]
            tangent_norm = max(float(np.linalg.norm(tangent)), 1e-9)
        tangent = tangent / tangent_norm
        return (
            projected[selected].copy(),
            float(distances[selected]),
            tangent,
            segment_index,
            selected_alpha,
        )

    def _observation(self) -> np.ndarray:
        """Return 14-point TCP-relative DLO state and robot proprioception."""
        cable_positions, cable_velocities = self._sample_cable_state()
        hand_position = self._grasp_center_position()
        hand_rotation = self.data.xmat[self.base_env.hand_id].reshape(3, 3)
        hand_linear_velocity, _ = self._hand_velocity()

        # Row-vector form of R_tcp^T @ vector_world.  Velocities are relative
        # to the moving TCP as well as expressed in the TCP frame.
        relative_positions = (
            (cable_positions - hand_position) @ hand_rotation
            / self.rl_config.cable_position_scale
        )
        relative_velocities = (
            (cable_velocities - hand_linear_velocity) @ hand_rotation
            / self.rl_config.cable_velocity_scale
        )
        dlo_state = np.concatenate(
            (relative_positions, relative_velocities), axis=1
        ).reshape(-1)

        arm_qpos = self.data.qpos[self.base_env.arm_qpos_adr]
        arm_qvel = self.data.qvel[self.base_env.arm_dof_adr]
        finger_qpos = self.data.qpos[self.base_env.finger_qpos_adr]
        finger_ranges = self.model.jnt_range[self.base_env.finger_joint_ids]
        maximum_aperture = max(float(np.sum(finger_ranges[:, 1])), 1e-6)
        aperture = float(np.sum(finger_qpos))
        observation = np.concatenate([
            dlo_state,
            (arm_qpos - self._arm_center) / self._arm_half_range,
            arm_qvel / np.asarray(
                self.base_env.config.arm_joint_velocity_limits, dtype=float
            ),
            np.array([2.0 * aperture / maximum_aperture - 1.0]),
        ]).astype(np.float32)
        return np.clip(observation, -10.0, 10.0)

    @staticmethod
    def _rotation_about_z(angle: float) -> np.ndarray:
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        return np.array([
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        """Map a 5-D TCP-local delta action to a velocity-safe joint target."""
        normalized = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if normalized.shape != (5,):
            raise ValueError(f"Expected RL action shape (5,), got {normalized.shape}")

        control_dt = float(
            self.model.opt.timestep * max(1, self.base_env.config.frame_skip)
        )
        hand_rotation = self.data.xmat[self.base_env.hand_id].reshape(3, 3)
        local_direction = normalized[:3]
        direction_norm = float(np.linalg.norm(local_direction))
        if direction_norm > 1.0:
            local_direction = local_direction / direction_norm
        local_translation = (
            self.rl_config.translation_delta_scale * local_direction
        )
        linear_velocity = hand_rotation @ local_translation / control_dt

        yaw_delta = self.rl_config.yaw_delta_scale * float(normalized[3])
        self._desired_hand_rotation = (
            self._desired_hand_rotation @ self._rotation_about_z(yaw_delta)
        )
        current_quat = rotation_to_quat(hand_rotation)
        desired_quat = rotation_to_quat(self._desired_hand_rotation)
        orientation_error = quat_error(current_quat, desired_quat)
        angular_velocity = orientation_error / control_dt
        angular_speed = float(np.linalg.norm(angular_velocity))
        angular_limit = float(self.base_env.config.hand_angular_velocity_limit)
        if angular_speed > angular_limit:
            angular_velocity *= angular_limit / angular_speed

        jacp, jacr = self.base_env._hand_jacobian()
        jacobian = np.vstack((jacp, jacr))[:, self.base_env.arm_dof_adr]
        twist = np.concatenate((linear_velocity, angular_velocity))
        damping = self.rl_config.ik_damping
        q_velocity = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6),
            twist,
        )
        velocity_limits = np.asarray(
            self.base_env.config.arm_joint_velocity_limits, dtype=float
        )
        velocity_scale = min(
            1.0,
            float(np.min(
                velocity_limits / np.maximum(np.abs(q_velocity), 1e-12)
            )),
        )
        q_velocity *= velocity_scale
        self._last_ik_velocity_scale = velocity_scale

        previous_target = self.base_env._last_applied_action[:7]
        target_qpos = previous_target + control_dt * q_velocity
        joint_range = self.model.jnt_range[self.base_env.arm_joint_ids]
        mujoco_action = np.empty(8)
        mujoco_action[:7] = np.clip(
            target_qpos, joint_range[:, 0], joint_range[:, 1]
        )

        previous_gripper_closed = self._gripper_closed
        if normalized[4] <= self.rl_config.gripper_close_threshold:
            self._gripper_closed = True
        elif normalized[4] >= self.rl_config.gripper_open_threshold:
            self._gripper_closed = False
        self._gripper_switch_event = bool(
            self._gripper_closed != previous_gripper_closed
        )
        self._gripper_switch_count += int(self._gripper_switch_event)
        gripper_range = self.model.actuator_ctrlrange[7]
        mujoco_action[7] = (
            np.clip(
                self.rl_config.gripper_closed_ctrl,
                gripper_range[0],
                gripper_range[1],
            )
            if self._gripper_closed
            else gripper_range[1]
        )
        return mujoco_action

    @staticmethod
    def _empty_grasp_status() -> dict[str, float | bool]:
        return {
            "pinch_confirmed": False,
            "raw_bilateral_contact": False,
            "raw_secured_candidate": False,
            "secured_grasp": False,
            "secured_hysteresis_active": False,
            "secured_contact_loss_seconds": 0.0,
            "grasp_lift_delta": 0.0,
            "strict_success_qualification": False,
            "strict_success_hold": 0.0,
            "rl_hold_success": False,
            "strict_success": False,
        }

    def _update_grasp_status(self, info: dict) -> dict[str, float | bool]:
        """区分双侧夹持接触和已经实际承载抬起的有效抓取。"""
        state = self.base_env.grasp_state
        action_seconds = float(
            self.model.opt.timestep * max(1, self.base_env.config.frame_skip)
        )
        if state is None:
            self._secured_candidate_hold = 0.0
            self._secured_contact_loss_hold = 0.0
            self._strict_success_hold = 0.0
            self._last_grasp_status = self._empty_grasp_status()
            return self._last_grasp_status

        pinch_confirmed = bool(self.base_env.grasp_confirmed)
        pairs = self.base_env._finger_contact_pairs()
        contacting_fingers = {finger for _, finger in pairs}
        raw_bilateral = bool(
            self.base_env.left_finger_id in contacting_fingers
            and self.base_env.right_finger_id in contacting_fingers
        )

        if pinch_confirmed:
            self._locked_body_id = state.body_id

        body_id = state.body_id
        body_index = self.base_env.cable_index[body_id]
        lift_delta = float(
            self.data.xpos[body_id, 2]
            - self._episode_initial_cable_z[body_index]
        )

        aperture = float(np.sum(self.data.qpos[self.base_env.finger_qpos_adr]))
        distance = float(np.linalg.norm(
            self.data.xpos[body_id] - self._grasp_center_position()
        ))
        secured_geometry_valid = bool(
            pinch_confirmed
            and aperture <= self.base_env.config.max_grasp_aperture
            and distance <= self.base_env.config.max_pad_distance
            and lift_delta >= self.rl_config.secured_lift_delta
        )
        raw_secured_candidate = bool(secured_geometry_valid and raw_bilateral)
        secured_was_confirmed = bool(
            self._secured_candidate_hold >= self.rl_config.secured_confirm_seconds
        )
        if raw_secured_candidate:
            self._secured_candidate_hold += action_seconds
            self._secured_contact_loss_hold = 0.0
        elif (
            secured_was_confirmed
            and self._gripper_closed
            and secured_geometry_valid
            and not raw_bilateral
        ):
            # 训练态只吸收极短的碰撞求解抖动；主动张爪不会获得这段宽限。
            self._secured_contact_loss_hold += action_seconds
            if (
                self._secured_contact_loss_hold
                > self.rl_config.secured_contact_loss_grace_seconds
            ):
                self._secured_candidate_hold = 0.0
        else:
            self._secured_candidate_hold = 0.0
            self._secured_contact_loss_hold = 0.0
        secured_grasp = bool(
            self._secured_candidate_hold >= self.rl_config.secured_confirm_seconds
        )
        secured_hysteresis_active = bool(
            secured_grasp and not raw_secured_candidate
        )
        self._ever_secured_grasp = self._ever_secured_grasp or secured_grasp
        self._secured_session_active = self._secured_session_active or secured_grasp

        # 严格成功计时仍要求当前原始双侧接触；训练态滞回不能延长0.80 s计时。
        strict_qualification = bool(
            secured_grasp
            and pinch_confirmed
            and raw_bilateral
            and self.data.xpos[body_id, 2] > 0.14
            and float(info["lifted_fraction"]) >= 0.18
            and distance <= self.base_env.config.max_pad_distance
        )
        if strict_qualification:
            self._strict_success_hold += action_seconds
        else:
            self._strict_success_hold = 0.0
        rl_hold_success = bool(
            self._strict_success_hold >= self.base_env.config.success_hold_seconds
        )

        self._last_grasp_status = {
            "pinch_confirmed": pinch_confirmed,
            "raw_bilateral_contact": raw_bilateral,
            "raw_secured_candidate": raw_secured_candidate,
            "secured_grasp": secured_grasp,
            "secured_hysteresis_active": secured_hysteresis_active,
            "secured_contact_loss_seconds": self._secured_contact_loss_hold,
            "grasp_lift_delta": lift_delta,
            "strict_success_qualification": strict_qualification,
            "strict_success_hold": self._strict_success_hold,
            "rl_hold_success": rl_hold_success,
            # 最终 strict_success 还会在 step() 中与底层 500 Hz 连续判定取交集。
            "strict_success": rl_hold_success,
        }
        return self._last_grasp_status

    def _classify_secured_grasp_break(self) -> tuple[bool, bool, bool]:
        """把一次已稳定抓取后的断开分类为主动张爪或物理滑脱。"""
        self._active_open_after_secured_event = False
        self._physical_slip_after_secured_event = False
        self._open_during_contact_loss_event = False
        self._current_grasp_loss_reason = None
        self._current_grasp_loss_causal_class = None

        break_info = self.base_env.last_grasp_break
        if break_info is None:
            return False, False, False

        reason = str(break_info.get("reason", "unknown"))
        break_key = (float(break_info.get("time", self.data.time)), reason)
        if break_key == self._last_grasp_break_key:
            return False, False, False
        self._last_grasp_break_key = break_key

        # 未达到secured的候选接触断开不属于抓后失败，但仍消费该底层事件。
        if not self._secured_session_active:
            return False, False, False

        self._current_grasp_loss_reason = reason
        causal_class = str(break_info.get("causal_class", ""))
        if not causal_class:
            if reason == "gripper_command_open":
                causal_class = (
                    "open_during_contact_loss"
                    if float(break_info.get("no_contact_time", 0.0)) > 0.0
                    else "active_open"
                )
            elif reason == "lost_physical_pad_contact":
                causal_class = "physical_slip"
            else:
                causal_class = "other"
        self._current_grasp_loss_causal_class = causal_class
        if causal_class == "active_open":
            self._active_open_after_secured_event = True
            self._active_open_after_secured_count += 1
        elif causal_class == "physical_slip":
            self._physical_slip_after_secured_event = True
            self._physical_slip_after_secured_count += 1
        elif causal_class == "open_during_contact_loss":
            self._open_during_contact_loss_event = True
            self._open_during_contact_loss_count += 1
        self._secured_session_active = False
        return (
            self._active_open_after_secured_event,
            self._physical_slip_after_secured_event,
            self._open_during_contact_loss_event,
        )

    def _reward(
        self,
        action: np.ndarray,
        success: bool,
        info: dict,
        grasp_status: dict[str, float | bool],
    ) -> tuple[float, dict[str, float]]:
        """用事件和状态增量奖励任务进展，避免靠长期停留反复刷正奖励。"""
        _, distance, tangent, segment_index, _ = self._nearest_graspable_segment(
            self._grasp_center_position()
        )
        progress = self._previous_distance - distance
        self._previous_distance = distance

        hand_rotation = self.data.xmat[self.base_env.hand_id].reshape(3, 3)
        alignment_score = float(abs(np.dot(hand_rotation[:, 0], tangent)))
        alignment_gate = bool(
            distance < self.rl_config.alignment_gate_distance
            and not bool(grasp_status["secured_grasp"])
        )
        if alignment_gate and self._previous_alignment_score is not None:
            alignment_progress = alignment_score - self._previous_alignment_score
        else:
            alignment_progress = 0.0
        self._previous_alignment_score = alignment_score if alignment_gate else None
        self._nearest_graspable_distance = distance
        self._nearest_graspable_segment_index = segment_index
        self._alignment_score = alignment_score

        pairs = self.base_env._finger_contact_pairs()
        contacting_fingers = len({finger for _, finger in pairs})
        new_contact_count = max(0, contacting_fingers - self._max_contacting_fingers)
        self._max_contacting_fingers = max(
            self._max_contacting_fingers, contacting_fingers
        )

        pinch_confirmed = bool(grasp_status["pinch_confirmed"])
        secured_grasp = bool(grasp_status["secured_grasp"])
        new_pinch = pinch_confirmed and not self._pinch_rewarded
        new_secured = secured_grasp and not self._secured_rewarded
        self._pinch_rewarded = self._pinch_rewarded or pinch_confirmed
        self._secured_rewarded = self._secured_rewarded or secured_grasp

        # 两种抬升奖励都只发放整回合首次达到的新高度，并在安全任务范围封顶。
        # 因而掉落、重抓、再次抬到旧高度不会重复获得正奖励。
        capped_lift = float(np.clip(
            float(grasp_status["grasp_lift_delta"]),
            0.0,
            self.rl_config.lift_credit_cap,
        ))
        previous_lift_high_water = self._lift_credit_high_water
        if pinch_confirmed:
            self._lift_credit_high_water = max(
                self._lift_credit_high_water, capped_lift
            )
        lift_progress = self._lift_credit_high_water - previous_lift_high_water

        lifted_fraction = float(info["lifted_fraction"])
        capped_lifted_fraction = float(np.clip(
            lifted_fraction,
            0.0,
            self.rl_config.cable_lift_credit_cap,
        ))
        previous_cable_high_water = self._cable_lift_credit_high_water
        # 未夹持时也推进基准但不发奖励，防止把扰动造成的自然抬升归功于策略。
        self._cable_lift_credit_high_water = max(
            self._cable_lift_credit_high_water, capped_lifted_fraction
        )
        cable_lift_progress = (
            self._cable_lift_credit_high_water - previous_cable_high_water
            if pinch_confirmed
            else 0.0
        )

        active_open, physical_slip, _ = (
            self._classify_secured_grasp_break()
        )
        # 多次重抓/断开仍完整计入诊断。外力滑脱与歧义张爪共用一次较轻惩罚；
        # 明确主动张爪另有一次较重惩罚，避免先制造低成本滑脱后免费张爪。
        penalize_active_open = bool(
            active_open and not self._active_open_penalty_applied
        )
        penalize_contact_loss = bool(
            physical_slip and not self._contact_loss_penalty_applied
        )
        if penalize_active_open:
            self._active_open_penalty_applied = True
        if penalize_contact_loss:
            self._contact_loss_penalty_applied = True
        penalize_physical_slip = bool(penalize_contact_loss and physical_slip)

        strict_progress = float(np.clip(
            float(grasp_status["strict_success_hold"])
            / self.base_env.config.success_hold_seconds,
            0.0,
            1.0,
        ))
        previous_strict_high_water = self._strict_hold_credit_high_water
        self._strict_hold_credit_high_water = max(
            self._strict_hold_credit_high_water, strict_progress
        )
        strict_progress_credit = (
            self._strict_hold_credit_high_water - previous_strict_high_water
        )

        action_rate = float(np.mean(np.square(action - self._previous_action)))
        arm_action_magnitude = float(np.mean(np.square(action[:4])))
        arm_action_rate = float(np.mean(np.square(
            action[:4] - self._previous_action[:4]
        )))
        post_grasp_control = bool(secured_grasp or self._previous_secured_grasp)
        self._post_grasp_arm_action_magnitude = (
            arm_action_magnitude if post_grasp_control else 0.0
        )
        self._post_grasp_arm_action_rate = (
            arm_action_rate if post_grasp_control else 0.0
        )
        self._previous_action = action.copy()
        self._previous_secured_grasp = secured_grasp

        components = {
            "reward_reach_progress": self.rl_config.reward_reach_progress * progress,
            "reward_alignment_progress": (
                self.rl_config.reward_alignment_progress * alignment_progress
            ),
            "reward_new_contact": self.rl_config.reward_new_contact * new_contact_count,
            "reward_new_pinch": self.rl_config.reward_new_pinch * float(new_pinch),
            "reward_new_secured_grasp": (
                self.rl_config.reward_new_secured_grasp * float(new_secured)
            ),
            "reward_lift_progress": self.rl_config.reward_lift_progress * lift_progress,
            "reward_cable_lift_progress": (
                self.rl_config.reward_cable_lift_progress * cable_lift_progress
            ),
            "reward_strict_hold": (
                self.rl_config.reward_strict_hold_progress
                * strict_progress_credit
            ),
            "reward_success": self.rl_config.reward_success * float(success),
            "reward_physical_slip": (
                self.rl_config.reward_slip * float(penalize_physical_slip)
            ),
            "reward_active_open_after_secured": (
                self.rl_config.reward_active_open_after_secured
                * float(penalize_active_open)
            ),
            "reward_gripper_switch": (
                self.rl_config.reward_gripper_switch
                * float(self._gripper_switch_event)
            ),
            "reward_time": self.rl_config.reward_time_step,
            "reward_action_magnitude": (
                self.rl_config.reward_action_magnitude
                * float(np.mean(np.square(action)))
            ),
            "reward_action_rate": self.rl_config.reward_action_rate * action_rate,
            "reward_post_grasp_arm_action_magnitude": (
                self.rl_config.reward_post_grasp_arm_action_magnitude
                * self._post_grasp_arm_action_magnitude
            ),
            "reward_post_grasp_arm_action_rate": (
                self.rl_config.reward_post_grasp_arm_action_rate
                * self._post_grasp_arm_action_rate
            ),
        }
        return float(sum(components.values())), components

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._select_training_scenario(seed)
        # 底层环境只使用自己的 Generator；把显式 seed 透传到底层，既保证并行
        # 环境中的场景可复现，也让评估 CSV/manifest 能准确记录该轮场景。
        _, info = self.base_env.reset(randomize=True, seed=seed)
        self._episode_return = 0.0
        self._episode_steps = 0
        self._episode_initial_cable_z = self.data.xpos[
            self.base_env.cable_ids, 2
        ].copy()
        self._secured_candidate_hold = 0.0
        self._secured_contact_loss_hold = 0.0
        self._strict_success_hold = 0.0
        self._ever_secured_grasp = False
        self._last_grasp_status = self._empty_grasp_status()
        self._locked_body_id = None
        self._gripper_closed = False
        self._previous_action = np.zeros(5)
        self._previous_alignment_score = None
        self._nearest_graspable_distance = 0.0
        self._nearest_graspable_segment_index = -1
        self._alignment_score = 0.0
        self._desired_hand_rotation = self.data.xmat[
            self.base_env.hand_id
        ].reshape(3, 3).copy()
        self._last_ik_velocity_scale = 1.0
        self._gripper_switch_event = False
        self._gripper_switch_count = 0
        self._max_contacting_fingers = 0
        self._pinch_rewarded = False
        self._secured_rewarded = False
        self._lift_credit_high_water = 0.0
        self._cable_lift_credit_high_water = float(np.clip(
            float(info["lifted_fraction"]),
            0.0,
            self.rl_config.cable_lift_credit_cap,
        ))
        self._secured_session_active = False
        self._last_grasp_break_key = None
        self._active_open_after_secured_event = False
        self._physical_slip_after_secured_event = False
        self._open_during_contact_loss_event = False
        self._active_open_after_secured_count = 0
        self._physical_slip_after_secured_count = 0
        self._open_during_contact_loss_count = 0
        self._current_grasp_loss_reason = None
        self._current_grasp_loss_causal_class = None
        self._contact_loss_penalty_applied = False
        self._active_open_penalty_applied = False
        self._strict_hold_credit_high_water = 0.0
        self._previous_secured_grasp = False
        self._post_grasp_arm_action_magnitude = 0.0
        self._post_grasp_arm_action_rate = 0.0
        (
            _, self._previous_distance, tangent,
            self._nearest_graspable_segment_index, _,
        ) = self._nearest_graspable_segment(self._grasp_center_position())
        self._nearest_graspable_distance = self._previous_distance
        hand_rotation = self.data.xmat[self.base_env.hand_id].reshape(3, 3)
        self._alignment_score = float(abs(np.dot(hand_rotation[:, 0], tangent)))
        return self._observation(), self._augment_info(info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        normalized_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        mujoco_action = self._convert_action(normalized_action)
        _, _, base_success, truncated, info = self.base_env.step(mujoco_action)
        grasp_status = self._update_grasp_status(info)
        # RL 的原始双侧接触/secured进度在50 Hz动作边界检查，底层的高度、
        # 整线离桌比例和中心距离连续性在500 Hz物理子步检查。当前两个0.80 s
        # 窗口必须同时达标；使用非粘性的success_now，历史成功不能兜底。
        success = bool(
            grasp_status["rl_hold_success"]
            and info.get("success_now", False)
        )
        grasp_status["strict_success"] = success
        reward, reward_components = self._reward(
            normalized_action, success, info, grasp_status
        )

        self._episode_return += reward
        self._episode_steps += 1
        info = self._augment_info(info, base_success=base_success)
        info.update(reward_components)
        info["episode_return"] = self._episode_return
        info["episode_steps"] = self._episode_steps
        info["mujoco_action"] = mujoco_action.copy()
        return self._observation(), reward, bool(success), bool(truncated), info

    def _augment_info(self, info: dict, *, base_success: bool | None = None) -> dict:
        result = dict(info)
        result["target_distance"] = self._nearest_graspable_distance
        result["nearest_graspable_distance"] = self._nearest_graspable_distance
        result["nearest_graspable_segment_index"] = (
            self._nearest_graspable_segment_index
        )
        result["alignment_score"] = self._alignment_score
        result["alignment_reward_active"] = bool(
            self._previous_alignment_score is not None
        )
        result["rl_grasped_body_id"] = self._locked_body_id
        result["ik_velocity_scale"] = self._last_ik_velocity_scale
        result["gripper_switch_event"] = self._gripper_switch_event
        result["gripper_switch_count"] = self._gripper_switch_count
        result["motion_curriculum_difficulty"] = self._motion_difficulty
        result["motion_curriculum_scenarios"] = self._active_scenario_names
        result["raw_finger_contacts"] = len(self.base_env.finger_contacts())
        result["unique_contacting_fingers"] = len({
            finger for _, finger in self.base_env._finger_contact_pairs()
        })
        result["base_success"] = bool(
            info.get("success", False) if base_success is None else base_success
        )
        result["base_success_now"] = bool(info.get("success_now", False))
        result.update(self._last_grasp_status)
        # RL中的“抓取”只统计已经离桌并持续承载的有效抓取；单纯闭爪接触另记为pinch。
        result["ever_pinched"] = self.base_env.last_grasped_body_id is not None
        result["ever_grasped"] = self._ever_secured_grasp
        result["secured_session_active"] = self._secured_session_active
        result["active_open_after_secured_event"] = (
            self._active_open_after_secured_event
        )
        result["physical_slip_after_secured_event"] = (
            self._physical_slip_after_secured_event
        )
        result["open_during_contact_loss_after_secured_event"] = (
            self._open_during_contact_loss_event
        )
        result["active_open_after_secured_count"] = (
            self._active_open_after_secured_count
        )
        result["physical_slip_after_secured_count"] = (
            self._physical_slip_after_secured_count
        )
        result["open_during_contact_loss_after_secured_count"] = (
            self._open_during_contact_loss_count
        )
        result["grasp_loss_reason"] = self._current_grasp_loss_reason
        result["grasp_loss_causal_class"] = self._current_grasp_loss_causal_class
        result["contact_loss_penalty_applied"] = (
            self._contact_loss_penalty_applied
        )
        result["active_open_penalty_applied"] = (
            self._active_open_penalty_applied
        )
        result["grasp_loss_penalty_applied"] = bool(
            self._contact_loss_penalty_applied
            or self._active_open_penalty_applied
        )
        result["lift_credit_high_water"] = self._lift_credit_high_water
        result["cable_lift_credit_high_water"] = (
            self._cable_lift_credit_high_water
        )
        result["strict_hold_credit_high_water"] = (
            self._strict_hold_credit_high_water
        )
        result["post_grasp_arm_action_magnitude"] = (
            self._post_grasp_arm_action_magnitude
        )
        result["post_grasp_arm_action_rate"] = self._post_grasp_arm_action_rate
        result["success"] = bool(self._last_grasp_status["strict_success"])
        return result

    def set_disturbance_strength(self, strength: float) -> None:
        """Set the final shape-force level used at curriculum difficulty 1."""
        strength = float(strength)
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("disturbance strength must be finite and non-negative")
        if self._scenario_configs:
            self._final_disturbance_scale = strength / 1.5
        else:
            self.base_env.config.disturbance_strength = strength

    def set_training_scenarios(self, scenario_names: Sequence[str]) -> None:
        """Restrict future resets to a curriculum subset of compiled scenarios."""
        names = tuple(str(name) for name in scenario_names)
        if not names:
            raise ValueError("training scenario subset cannot be empty")
        unknown = set(names) - set(self._scenario_by_name)
        if unknown:
            raise ValueError(
                "curriculum scenarios were not compiled into this environment: "
                + ", ".join(sorted(unknown))
            )
        self._active_scenario_names = names

    def set_motion_difficulty(self, difficulty: float) -> None:
        """Set low-to-nominal motion difficulty for subsequent episode resets."""
        difficulty = float(difficulty)
        if not np.isfinite(difficulty):
            raise ValueError("motion difficulty must be finite")
        self._motion_difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def close(self) -> None:
        # 原生MuJoCo数据没有额外线程或窗口；保留Gymnasium标准接口。
        return None
