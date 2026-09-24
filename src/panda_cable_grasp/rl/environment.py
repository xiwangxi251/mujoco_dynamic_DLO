"""供强化学习使用的 Gymnasium 接口。

该文件只负责把观测和动作转换成固定长度向量，并计算训练奖励。线缆扰动、
接触、纯摩擦夹持和成功判定仍完全由 CableGraspEnv 决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from ..runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import gymnasium as gym
import mujoco
import numpy as np

from ..env.environment import CableGraspEnv, EnvConfig
from ..env.kinematics import quat_error, rotation_to_quat
from ..scenarios.registry import ScenarioConfig, get_scenario


@dataclass
class RLConfig:
    """RL接口参数，不改变底层线缆物理。"""

    cable_sample_count: int = 14
    cable_position_scale: float = 0.50
    cable_velocity_scale: float = 1.0
    translation_delta_scale: float = 0.010
    yaw_delta_scale: float = 0.020
    ik_damping: float = 0.05
    # Task-space IK singularity avoidance is soft: it only becomes active near
    # a rank-deficient Jacobian or a joint limit, and never terminates an episode.
    singularity_avoidance_enabled: bool = True
    singularity_soft_sigma: float = 0.040
    singularity_hard_sigma: float = 0.008
    singularity_max_damping: float = 0.16
    singularity_posture_gain: float = 0.25
    singularity_joint_limit_gain: float = 1.00
    singularity_joint_limit_buffer_ratio: float = 0.04
    singularity_nullspace_velocity_limit: float = 0.75
    singularity_joint_margin_ratio: float = 0.15
    graspable_end_fraction: float = 0.25
    gripper_close_threshold: float = -0.35
    gripper_open_threshold: float = 0.35
    gripper_closed_ctrl: float = 0.0
    alignment_distance_scale: float = 0.08
    aligned_pinch_threshold: float = 0.90
    capture_ready_distance: float = 0.040
    capture_ready_alignment: float = 0.90
    capture_longitudinal_tolerance: float = 0.012
    capture_lateral_tolerance: float = 0.010
    capture_pad_tip_offset: float = 0.0085
    capture_min_insertion_depth: float = 0.0065
    capture_max_insertion_depth: float = 0.0175
    secured_lift_delta: float = 0.03
    secured_confirm_seconds: float = 0.10
    secured_contact_loss_grace_seconds: float = 0.06
    lift_credit_cap: float = 0.12
    cable_lift_credit_cap: float = 0.30
    reward_reach_progress: float = 6.0
    pinch_minimum_lift: float = 0.005
    pinch_stall_grace_seconds: float = 0.40
    pinch_stall_step_penalty: float = -0.005
    pinch_stall_penalty_cap: float = 0.50
    failed_unloaded_pinch_penalty: float = -0.50
    unloaded_pinch_episode_penalty_cap: float = 1.00
    reward_alignment_progress: float = 2.0
    reward_new_contact: float = 0.15
    reward_capture_close: float = 0.75
    reward_capture_ready_open_step: float = -0.001
    reward_premature_close_event: float = -0.04
    reward_premature_close_step: float = -0.001
    premature_close_episode_penalty_cap: float = 0.50
    reward_new_pinch: float = 0.25
    reward_new_aligned_pinch: float = 1.0
    reward_new_secured_grasp: float = 4.0
    reward_lift_progress: float = 40.0
    reward_cable_lift_progress: float = 3.0
    reward_strict_hold_progress: float = 2.0
    reward_success: float = 25.0
    reward_slip: float = -3.0
    reward_active_open_after_secured: float = -8.0
    reward_gripper_switch: float = -0.02
    reward_time_step: float = -0.0005
    reward_action_magnitude: float = -0.0005
    reward_action_rate: float = -0.01
    reward_geometric_safety_violation: float = -5.0
    # Optional late-stage safety shaping.  The table filter remains the hard
    # execution backstop; these bounded penalties teach the policy not to rely
    # on it and not to settle in poorly conditioned arm postures.
    safety_reward_enabled: bool = False
    reward_table_filter_step: float = -0.02
    reward_table_contact_step: float = -0.05
    table_safety_penalty_cap: float = 2.0
    reward_singularity_step: float = -0.01
    singularity_penalty_soft_sigma: float = 0.020
    singularity_penalty_hard_sigma: float = 0.005
    singularity_penalty_joint_margin_ratio: float = 0.05
    singularity_safety_penalty_cap: float = 1.5
    # This is per worker.  With 24 workers, 5k control steps is about 120k
    # global samples, so resumed PPO sees the new objective gradually.
    safety_penalty_ramp_env_steps: int = 5_000

    def __post_init__(self) -> None:
        if self.cable_sample_count != 14:
            raise ValueError("cable_sample_count must be 14 for the 99-D baseline")
        for name in (
            "cable_position_scale", "cable_velocity_scale",
            "translation_delta_scale", "yaw_delta_scale", "ik_damping",
            "singularity_soft_sigma", "singularity_hard_sigma",
            "singularity_max_damping", "singularity_posture_gain",
            "singularity_joint_limit_gain",
            "singularity_joint_limit_buffer_ratio",
            "singularity_nullspace_velocity_limit",
            "singularity_joint_margin_ratio",
            "alignment_distance_scale",
            "capture_ready_distance", "capture_longitudinal_tolerance",
            "capture_lateral_tolerance", "capture_pad_tip_offset",
            "capture_min_insertion_depth", "capture_max_insertion_depth",
            "secured_lift_delta", "pinch_minimum_lift",
            "pinch_stall_grace_seconds",
            "secured_confirm_seconds", "secured_contact_loss_grace_seconds",
            "singularity_penalty_soft_sigma",
            "singularity_penalty_hard_sigma",
            "singularity_penalty_joint_margin_ratio",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.graspable_end_fraction < 0.5:
            raise ValueError("graspable_end_fraction must be in [0, 0.5)")
        if not 0.0 <= self.aligned_pinch_threshold <= 1.0:
            raise ValueError("aligned_pinch_threshold must be in [0, 1]")
        if not 0.0 <= self.capture_ready_alignment <= 1.0:
            raise ValueError("capture_ready_alignment must be in [0, 1]")
        if self.capture_min_insertion_depth >= self.capture_max_insertion_depth:
            raise ValueError(
                "capture_min_insertion_depth must be less than "
                "capture_max_insertion_depth"
            )
        if self.singularity_hard_sigma >= self.singularity_soft_sigma:
            raise ValueError(
                "singularity_hard_sigma must be less than singularity_soft_sigma"
            )
        if self.singularity_penalty_hard_sigma >= self.singularity_penalty_soft_sigma:
            raise ValueError(
                "singularity_penalty_hard_sigma must be less than "
                "singularity_penalty_soft_sigma"
            )
        if self.singularity_joint_limit_buffer_ratio >= 0.5:
            raise ValueError(
                "singularity_joint_limit_buffer_ratio must be less than 0.5"
            )
        if not self.gripper_close_threshold < self.gripper_open_threshold:
            raise ValueError("gripper hysteresis thresholds must be increasing")
        if self.pinch_stall_step_penalty > 0.0:
            raise ValueError("pinch_stall_step_penalty must be non-positive")
        if self.failed_unloaded_pinch_penalty > 0.0:
            raise ValueError("failed_unloaded_pinch_penalty must be non-positive")
        for name in (
            "reward_capture_ready_open_step", "reward_premature_close_event",
            "reward_premature_close_step",
            "reward_geometric_safety_violation",
            "reward_table_filter_step", "reward_table_contact_step",
            "reward_singularity_step",
        ):
            if float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be non-positive")
        for name in (
            "pinch_stall_penalty_cap",
            "unloaded_pinch_episode_penalty_cap",
            "premature_close_episode_penalty_cap",
            "table_safety_penalty_cap", "singularity_safety_penalty_cap",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.singularity_penalty_joint_margin_ratio >= 0.5:
            raise ValueError(
                "singularity_penalty_joint_margin_ratio must be less than 0.5"
            )
        if int(self.safety_penalty_ramp_env_steps) < 0:
            raise ValueError("safety_penalty_ramp_env_steps must be non-negative")


TASK_SPACE_ACTION_MODE = "task_space"
VERTICAL_DOWN_TASK_SPACE_ACTION_MODE = "task_space_vertical_down"
JOINT_TARGET_ACTION_MODE = "joint_target"
ACTION_MODES = (
    TASK_SPACE_ACTION_MODE,
    VERTICAL_DOWN_TASK_SPACE_ACTION_MODE,
    JOINT_TARGET_ACTION_MODE,
)


def action_interface_version(action_mode: str) -> str:
    """Return the checkpoint interface marker for an action parameterization."""
    if action_mode == TASK_SPACE_ACTION_MODE:
        return "baseline_v4"
    if action_mode == VERTICAL_DOWN_TASK_SPACE_ACTION_MODE:
        return "baseline_v6_task_space_vertical_down"
    if action_mode == JOINT_TARGET_ACTION_MODE:
        return "baseline_v5_joint_target"
    raise ValueError(f"unsupported RL action mode: {action_mode!r}")


class RLCableGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """Dynamic-cable grasping with a conventional task-space RL API.

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
        "delta_base_x", "delta_base_y", "delta_base_z",
        "delta_world_yaw", "gripper_hysteresis",
    )

    def __init__(
        self,
        *,
        robot: str = "panda",
        seed: int = 20260804,
        disturbance_strength: float = 1.5,
        episode_seconds: float = 15.0,
        dynamicvla_cameras_enabled: bool = False,
        env_config: EnvConfig | None = None,
        scenario_names: Sequence[str] | None = None,
        rl_config: RLConfig | None = None,
        geometric_safety_enabled: bool | None = None,
        table_finger_collision_filter_enabled: bool = True,
        frame_skip: int | None = None,
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
        self._active_scenario_probabilities = (
            tuple(
                1.0 / len(self._active_scenario_names)
                for _ in self._active_scenario_names
            )
            if self._active_scenario_names
            else ()
        )
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
            if frame_skip is not None:
                first_overrides["frame_skip"] = int(frame_skip)
            first_overrides["disturbance_strength"] = (
                first_scenario.disturbance_strength
                * self._final_disturbance_scale
            )
            env_config = EnvConfig(
                robot=robot,
                seed=seed,
                episode_seconds=episode_seconds,
                scenario_name=first_scenario.name,
                scenario_id=first_scenario.scenario_id,
                scenario_split=first_scenario.split.value,
                dynamicvla_cameras_enabled=dynamicvla_cameras_enabled,
                **first_overrides,
            )
        self.base_env = CableGraspEnv(
            env_config
            if env_config is not None
            else EnvConfig(
                robot=robot,
                seed=seed,
                disturbance_strength=disturbance_strength,
                episode_seconds=episode_seconds,
                dynamicvla_cameras_enabled=dynamicvla_cameras_enabled,
            )
        )
        if geometric_safety_enabled is not None:
            self.base_env.config.geometric_safety_enabled = bool(
                geometric_safety_enabled
            )
        self.table_finger_collision_filter_enabled = bool(
            table_finger_collision_filter_enabled
        )
        # The PPO training/evaluation interface uses the shared target limiter
        # plus the configured joint acceleration limiter.
        self.base_env.config.arm_acceleration_limit_enabled = True
        # The wrapper converts task-space actions to joint targets, while the
        # shared environment remains the final authority on robot capability.
        # Keep the historical joint-speed command limit as the only arm speed
        # guard for this configuration; do not add the newer Cartesian cap.
        self.base_env.config.hand_cartesian_velocity_limit_enabled = False
        self.action_mode = TASK_SPACE_ACTION_MODE
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -10.0, 10.0, shape=(len(self.OBSERVATION_NAMES),), dtype=np.float32
        )
        self._previous_distance = 0.0
        self._episode_return = 0.0
        self._episode_steps = 0
        self._total_env_steps = 0
        self._table_safety_penalty_total = 0.0
        self._singularity_safety_penalty_total = 0.0
        self._last_table_filter_reward = 0.0
        self._last_table_contact_reward = 0.0
        self._last_singularity_safety_reward = 0.0
        self._last_singularity_penalty_severity = 0.0
        self._last_safety_penalty_scale = 0.0
        self._episode_initial_cable_z = np.zeros(len(self.base_env.cable_ids))
        self._secured_candidate_hold = 0.0
        self._secured_contact_loss_hold = 0.0
        self._strict_success_hold = 0.0
        self._ever_secured_grasp = False
        self._last_grasp_status = self._empty_grasp_status()
        self._locked_body_id: int | None = None
        self._gripper_closed = False
        self._previous_action = np.zeros(5)
        self._previous_alignment_potential: float | None = None
        self._nearest_graspable_distance = 0.0
        self._nearest_graspable_segment_index = -1
        self._alignment_score = 0.0
        self._alignment_potential = 0.0
        self._desired_hand_rotation = np.eye(3)
        self._last_ik_velocity_scale = 1.0
        self._last_ik_singularity_sigma_min = float("inf")
        self._last_ik_singularity_condition = 1.0
        self._last_ik_singularity_damping = self.rl_config.ik_damping
        self._last_ik_singularity_severity = 0.0
        self._last_ik_singularity_guard_active = False
        self._last_ik_singularity_guard_count = 0
        self._last_ik_singularity_target_clipped = False
        self._last_ik_singularity_target_clip_count = 0
        self._last_ik_joint_margin_ratio = 1.0
        self._gripper_switch_event = False
        self._gripper_switch_count = 0
        self._max_contacting_fingers = 0
        self._pinch_rewarded = False
        self._aligned_pinch_rewarded = False
        self._aligned_pinch_event = False
        self._ever_aligned_pinch = False
        self._alignment_score_at_first_pinch: float | None = None
        self._capture_ready = False
        self._capture_local_offset = np.full(3, np.nan)
        self._capture_insertion_depth = float("nan")
        self._capture_close_event = False
        self._capture_close_rewarded = False
        self._premature_close_event = False
        self._premature_close_penalty_total = 0.0
        self._secured_rewarded = False
        self._lift_credit_high_water = 0.0
        self._cable_lift_credit_high_water = 0.0
        # Reversible shaping state: upward motion is positive, downward motion
        # is negative. The high-water values remain diagnostics only.
        self._lift_credit_previous = 0.0
        self._cable_lift_credit_previous = 0.0
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
        self._previous_pinch_confirmed = False
        self._pinch_session_elapsed = 0.0
        self._pinch_session_lift_high_water = 0.0
        self._pinch_session_lift_peak_global = 0.0
        self._pinch_session_stall_penalty = 0.0
        self._unloaded_pinch_penalty_total = 0.0
        self._failed_unloaded_pinch_count = 0
        self._failed_unloaded_pinch_event = False
        self._lift_attempt = False
        self._loaded_lift = False
        self._pinch_session_start_hand_z: float | None = None
        self._post_pinch_world_z_action_sum = 0.0
        self._post_pinch_world_z_action_count = 0
        self._last_commanded_world_translation = np.zeros(3)

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
        probabilities = np.asarray(
            self._active_scenario_probabilities, dtype=float
        )
        # 训练中的普通 reset 按课程权重采样；严格评估使用均匀分布并传连续
        # 显式 seed，此时按模循环，保证小样本也能近似等次数覆盖目标场景。
        if explicit_seed is not None and np.allclose(
            probabilities, 1.0 / len(active_configs)
        ):
            index = int(explicit_seed) % len(active_configs)
        elif explicit_seed is not None:
            seeded_rng = np.random.default_rng(int(explicit_seed))
            index = int(seeded_rng.choice(len(active_configs), p=probabilities))
        else:
            index = int(
                self.np_random.choice(len(active_configs), p=probabilities)
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
        finger_ranges = self.model.jnt_range[self.base_env.finger_joint_ids]
        maximum_aperture = max(
            float(np.sum(np.max(np.abs(finger_ranges), axis=1))), 1e-6
        )
        aperture = self.base_env.finger_aperture
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

    @staticmethod
    def _planar_perpendicular_alignment(
        hand_rotation: np.ndarray,
        tangent: np.ndarray,
        closing_axis_local: np.ndarray | None = None,
    ) -> float:
        """Return 1 when the finger closing axis is perpendicular to the DLO.

        The selected robot supplies its local jaw-closing axis.  Grasp alignment
        is therefore a table-plane relationship between that axis and the local
        cable tangent; using a 3-D dot product lets tool tilt contaminate the
        yaw objective.
        """
        if closing_axis_local is None:
            closing_axis_local = np.array([0.0, 1.0, 0.0])
        closing_axis = (
            np.asarray(hand_rotation, dtype=float)
            @ np.asarray(closing_axis_local, dtype=float)
        )[:2]
        cable_axis = np.asarray(tangent, dtype=float)[:2]
        closing_norm = float(np.linalg.norm(closing_axis))
        cable_norm = float(np.linalg.norm(cable_axis))
        if closing_norm <= 1e-9 or cable_norm <= 1e-9:
            return 0.0
        closing_axis = closing_axis / closing_norm
        cable_axis = cable_axis / cable_norm
        return float(abs(
            closing_axis[0] * cable_axis[1]
            - closing_axis[1] * cable_axis[0]
        ))

    def _alignment_terms(
        self, distance: float, tangent: np.ndarray,
    ) -> tuple[float, float]:
        hand_rotation = self.data.xmat[self.base_env.hand_id].reshape(3, 3)
        score = self._planar_perpendicular_alignment(
            hand_rotation,
            tangent,
            self.base_env.gripper_closing_axis_local,
        )
        distance_scale = self.rl_config.alignment_distance_scale
        proximity = float(np.exp(-np.square(distance / distance_scale)))
        return score, proximity * score

    def _capture_corridor_terms(
        self, cable_point: np.ndarray,
    ) -> tuple[np.ndarray, float, bool]:
        """Measure whether the cable centerline is inside the usable pad region.

        The robot-specific local jaw axes identify the closing direction and the
        root-to-tip direction.  Distance and yaw alignment alone
        cannot distinguish a cable at the fingertips from one seated between
        the pads, so closure readiness also requires lateral centering and a
        minimum insertion depth past the fingertip edge.
        """
        hand_rotation = self.data.xmat[self.base_env.hand_id].reshape(3, 3)
        local_offset = (
            np.asarray(cable_point, dtype=float) - self._grasp_center_position()
        ) @ hand_rotation
        longitudinal = float(
            local_offset @ self.base_env.gripper_longitudinal_axis_local
        )
        lateral = float(
            local_offset @ self.base_env.gripper_lateral_axis_local
        )
        insertion_depth = float(self.rl_config.capture_pad_tip_offset - longitudinal)
        inside = bool(
            abs(lateral)
            <= self.rl_config.capture_longitudinal_tolerance
            and abs(float(local_offset @ self.base_env.gripper_closing_axis_local))
            <= self.rl_config.capture_lateral_tolerance
            and self.rl_config.capture_min_insertion_depth
            <= insertion_depth
            <= self.rl_config.capture_max_insertion_depth
        )
        return local_offset, insertion_depth, inside

    def _singularity_metrics(
        self, jacobian: np.ndarray,
    ) -> tuple[float, float, float, float, bool]:
        """Measure IK conditioning and return an adaptive damping decision.

        This is deliberately a soft, action-decoder guard.  It does not alter
        episode termination or reward; it only makes the local IK solution more
        conservative when the hand Jacobian loses rank or a joint approaches a
        hard limit.
        """
        singular_values = np.linalg.svd(
            np.asarray(jacobian, dtype=float), compute_uv=False
        )
        if singular_values.size and np.all(np.isfinite(singular_values)):
            sigma_min = float(singular_values[-1])
            sigma_max = float(singular_values[0])
        else:
            sigma_min = 0.0
            sigma_max = 0.0
        condition = sigma_max / max(sigma_min, 1e-12)

        joint_range = np.asarray(
            self.model.jnt_range[self.base_env.arm_joint_ids], dtype=float
        )
        joint_span = np.maximum(joint_range[:, 1] - joint_range[:, 0], 1e-6)
        arm_qpos = np.asarray(
            self.data.qpos[self.base_env.arm_qpos_adr], dtype=float
        )
        lower_margin = (arm_qpos - joint_range[:, 0]) / joint_span
        upper_margin = (joint_range[:, 1] - arm_qpos) / joint_span
        joint_margin_ratio = float(
            np.clip(np.min(np.minimum(lower_margin, upper_margin)), 0.0, 0.5)
        )

        if self.rl_config.singularity_avoidance_enabled:
            sigma_span = max(
                self.rl_config.singularity_soft_sigma
                - self.rl_config.singularity_hard_sigma,
                1e-9,
            )
            sigma_severity = float(np.clip(
                (self.rl_config.singularity_soft_sigma - sigma_min)
                / sigma_span,
                0.0,
                1.0,
            ))
            margin_target = self.rl_config.singularity_joint_margin_ratio
            limit_severity = float(np.clip(
                (margin_target - joint_margin_ratio)
                / max(margin_target, 1e-9),
                0.0,
                1.0,
            ))
            severity = max(sigma_severity, limit_severity)
        else:
            severity = 0.0

        base_damping = float(self.rl_config.ik_damping)
        max_damping = max(base_damping, float(
            self.rl_config.singularity_max_damping
        ))
        damping = base_damping + severity * (max_damping - base_damping)
        active = bool(
            self.rl_config.singularity_avoidance_enabled and severity > 0.0
        )
        self._last_ik_singularity_sigma_min = sigma_min
        self._last_ik_singularity_condition = condition
        self._last_ik_singularity_damping = damping
        self._last_ik_singularity_severity = severity
        self._last_ik_singularity_guard_active = active
        self._last_ik_joint_margin_ratio = joint_margin_ratio
        if active:
            self._last_ik_singularity_guard_count += 1
        return sigma_min, condition, severity, damping, active

    def _apply_singularity_nullspace(
        self,
        jacobian: np.ndarray,
        q_velocity: np.ndarray,
        damping: float,
        severity: float,
    ) -> np.ndarray:
        """Add a bounded null-space velocity away from singular postures."""
        if (
            not self.rl_config.singularity_avoidance_enabled
            or severity <= 0.0
        ):
            return np.asarray(q_velocity, dtype=float)

        joint_range = np.asarray(
            self.model.jnt_range[self.base_env.arm_joint_ids], dtype=float
        )
        joint_span = np.maximum(joint_range[:, 1] - joint_range[:, 0], 1e-6)
        arm_qpos = np.asarray(
            self.data.qpos[self.base_env.arm_qpos_adr], dtype=float
        )
        center = 0.5 * (joint_range[:, 0] + joint_range[:, 1])
        half_range = np.maximum(0.5 * joint_span, 1e-6)
        normalized_center = np.clip(
            (arm_qpos - center) / half_range, -1.0, 1.0
        )
        lower_margin = (arm_qpos - joint_range[:, 0]) / joint_span
        upper_margin = (joint_range[:, 1] - arm_qpos) / joint_span
        margin = np.minimum(lower_margin, upper_margin)
        limit_activation = np.clip(
            (self.rl_config.singularity_joint_margin_ratio - margin)
            / max(self.rl_config.singularity_joint_margin_ratio, 1e-9),
            0.0,
            1.0,
        )
        posture_bias = (
            -self.rl_config.singularity_posture_gain
            * severity
            * normalized_center
        )
        limit_bias = (
            -self.rl_config.singularity_joint_limit_gain
            * limit_activation
            * normalized_center
        )
        bias = np.clip(
            posture_bias + limit_bias,
            -self.rl_config.singularity_nullspace_velocity_limit,
            self.rl_config.singularity_nullspace_velocity_limit,
        )
        identity = np.eye(jacobian.shape[1])
        damped_pinv = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(jacobian.shape[0]),
            np.eye(jacobian.shape[0]),
        )
        nullspace_projector = identity - damped_pinv @ jacobian
        return np.asarray(q_velocity, dtype=float) + nullspace_projector @ bias

    def _limit_ik_joint_velocity(self, q_velocity: np.ndarray) -> np.ndarray:
        """Apply the legacy task-space decoder joint-velocity scaling."""
        velocity_limits = np.asarray(
            self.base_env.config.arm_joint_velocity_limits, dtype=float
        )
        velocity_scale = min(
            1.0,
            float(np.min(
                velocity_limits / np.maximum(np.abs(q_velocity), 1e-12)
            )),
        )
        q_velocity = q_velocity * velocity_scale
        self._last_ik_velocity_scale = velocity_scale
        return q_velocity

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        """Map a 5-D base-frame delta action to a velocity-safe joint target."""
        normalized = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if normalized.shape != (5,):
            raise ValueError(f"Expected RL action shape (5,), got {normalized.shape}")

        control_dt = float(
            self.model.opt.timestep * max(1, self.base_env.config.frame_skip)
        )
        hand_rotation = self.data.xmat[self.base_env.hand_id].reshape(3, 3)
        base_direction = normalized[:3]
        direction_norm = float(np.linalg.norm(base_direction))
        if direction_norm > 1.0:
            base_direction = base_direction / direction_norm
        world_translation = (
            self.rl_config.translation_delta_scale * base_direction
        )
        self._last_commanded_world_translation = world_translation.copy()
        linear_velocity = world_translation / control_dt

        yaw_delta = self.rl_config.yaw_delta_scale * float(normalized[3])
        self._desired_hand_rotation = (
            self._rotation_about_z(yaw_delta) @ self._desired_hand_rotation
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
        _, _, singularity_severity, damping, singularity_active = (
            self._singularity_metrics(jacobian)
        )
        q_velocity = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6),
            twist,
        )
        q_velocity = self._apply_singularity_nullspace(
            jacobian, q_velocity, damping, singularity_severity
        )
        q_velocity = self._limit_ik_joint_velocity(q_velocity)

        previous_target = self.base_env._last_applied_action[:7]
        target_qpos = previous_target + control_dt * q_velocity
        joint_range = self.model.jnt_range[self.base_env.arm_joint_ids]
        if singularity_active:
            # Keep a small margin from the hard stop only while the guard is
            # active.  This is a command buffer, not an episode termination.
            buffer = self.rl_config.singularity_joint_limit_buffer_ratio
            joint_span = np.maximum(
                joint_range[:, 1] - joint_range[:, 0], 1e-6
            )
            safe_lower = joint_range[:, 0] + buffer * joint_span
            safe_upper = joint_range[:, 1] - buffer * joint_span
            guarded_target = np.clip(target_qpos, safe_lower, safe_upper)
            self._last_ik_singularity_target_clipped = bool(
                np.any(np.abs(guarded_target - target_qpos) > 1e-10)
            )
            self._last_ik_singularity_target_clip_count += int(
                self._last_ik_singularity_target_clipped
            )
            target_qpos = guarded_target
        else:
            self._last_ik_singularity_target_clipped = False
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
        gripper_range = self.model.actuator_ctrlrange[
            self.base_env.gripper_actuator_id
        ]
        closed_ctrl = float(np.clip(
            self.rl_config.gripper_closed_ctrl,
            gripper_range[0], gripper_range[1],
        ))
        mujoco_action[self.base_env.gripper_actuator_id] = (
            closed_ctrl if self._gripper_closed else float(gripper_range[1])
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

        aperture = self.base_env.finger_aperture
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
            # 最终 strict_success 还会在 step() 中与底层 500 Hz 当前判定取交集；
            # 它只作为训练漏斗诊断，不再定义任务终止。
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

    def _update_pinch_session(
        self,
        action: np.ndarray,
        grasp_status: dict[str, float | bool],
    ) -> tuple[float, float]:
        """Penalize an unloaded pinch without making avoiding pinch optimal."""
        pinch_now = bool(grasp_status["pinch_confirmed"])
        lift_delta = max(0.0, float(grasp_status["grasp_lift_delta"]))
        action_seconds = float(
            self.model.opt.timestep * max(1, self.base_env.config.frame_skip)
        )
        self._failed_unloaded_pinch_event = False
        stall_reward = 0.0
        failed_reward = 0.0

        if pinch_now:
            if not self._previous_pinch_confirmed:
                self._pinch_session_elapsed = 0.0
                self._pinch_session_lift_high_water = 0.0
                self._pinch_session_stall_penalty = 0.0
                self._pinch_session_start_hand_z = float(
                    self._grasp_center_position()[2]
                )
            self._pinch_session_elapsed += action_seconds
            self._pinch_session_lift_high_water = max(
                self._pinch_session_lift_high_water, lift_delta
            )
            self._pinch_session_lift_peak_global = max(
                self._pinch_session_lift_peak_global,
                self._pinch_session_lift_high_water,
            )
            self._loaded_lift = bool(
                self._loaded_lift
                or self._pinch_session_lift_high_water
                >= self.rl_config.pinch_minimum_lift
            )
            if self.action_mode in (
                TASK_SPACE_ACTION_MODE,
                VERTICAL_DOWN_TASK_SPACE_ACTION_MODE,
            ):
                self._post_pinch_world_z_action_sum += float(action[2])
                self._post_pinch_world_z_action_count += 1
            if self._pinch_session_start_hand_z is not None:
                self._lift_attempt = bool(
                    self._lift_attempt
                    or float(self._grasp_center_position()[2])
                    - self._pinch_session_start_hand_z >= 0.01
                )

            stalled = bool(
                self._pinch_session_elapsed
                > self.rl_config.pinch_stall_grace_seconds + 1e-9
                and self._pinch_session_lift_high_water
                < self.rl_config.pinch_minimum_lift
            )
            if stalled:
                session_remaining = max(
                    0.0,
                    self.rl_config.pinch_stall_penalty_cap
                    - self._pinch_session_stall_penalty,
                )
                episode_remaining = max(
                    0.0,
                    self.rl_config.unloaded_pinch_episode_penalty_cap
                    - self._unloaded_pinch_penalty_total,
                )
                magnitude = min(
                    abs(self.rl_config.pinch_stall_step_penalty),
                    session_remaining,
                    episode_remaining,
                )
                self._pinch_session_stall_penalty += magnitude
                self._unloaded_pinch_penalty_total += magnitude
                stall_reward = -magnitude
        elif self._previous_pinch_confirmed:
            if (
                self._pinch_session_lift_high_water
                < self.rl_config.pinch_minimum_lift
            ):
                self._failed_unloaded_pinch_event = True
                self._failed_unloaded_pinch_count += 1
                episode_remaining = max(
                    0.0,
                    self.rl_config.unloaded_pinch_episode_penalty_cap
                    - self._unloaded_pinch_penalty_total,
                )
                magnitude = min(
                    abs(self.rl_config.failed_unloaded_pinch_penalty),
                    episode_remaining,
                )
                self._unloaded_pinch_penalty_total += magnitude
                failed_reward = -magnitude
            self._pinch_session_elapsed = 0.0
            self._pinch_session_lift_high_water = 0.0
            self._pinch_session_stall_penalty = 0.0
            self._pinch_session_start_hand_z = None

        self._previous_pinch_confirmed = pinch_now
        return stall_reward, failed_reward

    def _reward(
        self,
        action: np.ndarray,
        success: bool,
        info: dict,
        grasp_status: dict[str, float | bool],
    ) -> tuple[float, dict[str, float]]:
        """用事件和状态增量奖励任务进展，避免靠长期停留反复刷正奖励。"""
        nearest_point, distance, tangent, segment_index, _ = (
            self._nearest_graspable_segment(self._grasp_center_position())
        )
        pinch_confirmed = bool(grasp_status["pinch_confirmed"])
        pregrasp = not pinch_confirmed
        progress = self._previous_distance - distance if pregrasp else 0.0
        self._previous_distance = distance

        alignment_score, alignment_potential = self._alignment_terms(
            distance, tangent
        )
        if pregrasp and self._previous_alignment_potential is not None:
            alignment_progress = (
                alignment_potential - self._previous_alignment_potential
            )
        else:
            alignment_progress = 0.0
        self._previous_alignment_potential = (
            alignment_potential if pregrasp else None
        )
        self._nearest_graspable_distance = distance
        self._nearest_graspable_segment_index = segment_index
        self._alignment_score = alignment_score
        self._alignment_potential = alignment_potential
        (
            self._capture_local_offset,
            self._capture_insertion_depth,
            inside_capture_corridor,
        ) = self._capture_corridor_terms(nearest_point)
        self._capture_ready = bool(
            pregrasp
            and distance <= self.rl_config.capture_ready_distance
            and alignment_score >= self.rl_config.capture_ready_alignment
            and inside_capture_corridor
        )
        self._capture_close_event = bool(
            self._capture_ready
            and self._gripper_closed
            and self._gripper_switch_event
            and not self._capture_close_rewarded
        )
        self._capture_close_rewarded = bool(
            self._capture_close_rewarded or self._capture_close_event
        )
        premature_close = bool(
            pregrasp and self._gripper_closed and not self._capture_ready
        )
        self._premature_close_event = bool(
            premature_close and self._gripper_switch_event
        )
        premature_close_event_reward = 0.0
        if self._premature_close_event:
            remaining_penalty = max(
                0.0,
                self.rl_config.premature_close_episode_penalty_cap
                - self._premature_close_penalty_total,
            )
            applied_penalty = min(
                -self.rl_config.reward_premature_close_event,
                remaining_penalty,
            )
            self._premature_close_penalty_total += applied_penalty
            premature_close_event_reward = -applied_penalty

        pairs = self.base_env._finger_contact_pairs()
        contacting_fingers = len({finger for _, finger in pairs})
        new_contact_count = max(0, contacting_fingers - self._max_contacting_fingers)
        self._max_contacting_fingers = max(
            self._max_contacting_fingers, contacting_fingers
        )

        secured_grasp = bool(grasp_status["secured_grasp"])
        new_pinch = pinch_confirmed and not self._pinch_rewarded
        pinch_transition = bool(
            pinch_confirmed and not self._previous_pinch_confirmed
        )
        self._aligned_pinch_event = bool(
            pinch_transition
            and not self._aligned_pinch_rewarded
            and alignment_score >= self.rl_config.aligned_pinch_threshold
        )
        if new_pinch and self._alignment_score_at_first_pinch is None:
            self._alignment_score_at_first_pinch = alignment_score
        self._ever_aligned_pinch = bool(
            self._ever_aligned_pinch or self._aligned_pinch_event
        )
        self._aligned_pinch_rewarded = bool(
            self._aligned_pinch_rewarded or self._aligned_pinch_event
        )
        new_secured = secured_grasp and not self._secured_rewarded
        self._pinch_rewarded = self._pinch_rewarded or pinch_confirmed
        self._secured_rewarded = self._secured_rewarded or secured_grasp

        # 抬升塑形使用可逆势能：上升给正奖励，下降给对称负奖励。
        # high-water 仍保留为诊断指标，不能单独作为奖励基准。
        capped_lift = float(np.clip(
            float(grasp_status["grasp_lift_delta"]),
            0.0,
            self.rl_config.lift_credit_cap,
        ))
        previous_lift_credit = self._lift_credit_previous
        current_lift_credit = capped_lift if pinch_confirmed else 0.0
        lift_progress = current_lift_credit - previous_lift_credit
        self._lift_credit_previous = current_lift_credit
        if pinch_confirmed:
            self._lift_credit_high_water = max(
                self._lift_credit_high_water, capped_lift
            )

        lifted_fraction = float(info["lifted_fraction"])
        capped_lifted_fraction = float(np.clip(
            lifted_fraction,
            0.0,
            self.rl_config.cable_lift_credit_cap,
        ))
        previous_cable_credit = self._cable_lift_credit_previous
        # 未夹持时也推进基准但不发奖励，防止把扰动造成的自然抬升归功于策略。
        self._cable_lift_credit_high_water = max(
            self._cable_lift_credit_high_water, capped_lifted_fraction
        )
        cable_lift_progress = (
            capped_lifted_fraction - previous_cable_credit
            if pinch_confirmed
            else 0.0
        )
        self._cable_lift_credit_previous = capped_lifted_fraction

        active_open, physical_slip, _ = (
            self._classify_secured_grasp_break()
        )
        stall_reward, failed_unloaded_reward = self._update_pinch_session(
            action, grasp_status
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
        self._previous_action = action.copy()

        safety_components = self._safety_reward_components()

        components = {
            "reward_reach_progress": self.rl_config.reward_reach_progress * progress,
            "reward_alignment_progress": (
                self.rl_config.reward_alignment_progress * alignment_progress
            ),
            "reward_new_contact": self.rl_config.reward_new_contact * new_contact_count,
            "reward_capture_close": (
                self.rl_config.reward_capture_close
                * float(self._capture_close_event)
            ),
            "reward_capture_ready_open": (
                self.rl_config.reward_capture_ready_open_step
                * float(self._capture_ready and not self._gripper_closed)
            ),
            "reward_premature_close_event": premature_close_event_reward,
            "reward_premature_close": (
                self.rl_config.reward_premature_close_step
                * float(premature_close)
            ),
            "reward_new_pinch": self.rl_config.reward_new_pinch * float(new_pinch),
            "reward_new_aligned_pinch": (
                self.rl_config.reward_new_aligned_pinch
                * float(self._aligned_pinch_event)
            ),
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
            "reward_pinch_stall": stall_reward,
            "reward_failed_unloaded_pinch": failed_unloaded_reward,
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
            "reward_geometric_safety_violation": (
                self.rl_config.reward_geometric_safety_violation
                * float(
                    info.get("termination_reason")
                    == "geometric_safety_violation"
                )
            ),
            **safety_components,
        }
        return float(sum(components.values())), components

    def _safety_reward_components(self) -> dict[str, float]:
        """Return bounded shaping terms without changing termination semantics."""
        self._last_table_filter_reward = 0.0
        self._last_table_contact_reward = 0.0
        self._last_singularity_safety_reward = 0.0
        self._last_singularity_penalty_severity = 0.0
        if not self.rl_config.safety_reward_enabled:
            self._last_safety_penalty_scale = 0.0
            return {
                "reward_table_filter": 0.0,
                "reward_table_contact": 0.0,
                "reward_singularity_safety": 0.0,
            }

        ramp_steps = int(self.rl_config.safety_penalty_ramp_env_steps)
        scale = (
            1.0
            if ramp_steps == 0
            else min(1.0, float(self._total_env_steps) / float(ramp_steps))
        )
        self._last_safety_penalty_scale = scale

        table_remaining = max(
            0.0,
            self.rl_config.table_safety_penalty_cap
            - self._table_safety_penalty_total,
        )
        if bool(getattr(self, "_table_finger_filter_active", False)):
            magnitude = min(
                -self.rl_config.reward_table_filter_step * scale,
                table_remaining,
            )
            self._last_table_filter_reward = -magnitude
            self._table_safety_penalty_total += magnitude
            table_remaining -= magnitude

        # Recompute after the physical step.  Only the two finger collision
        # geoms are considered; normal cable-table contact is intentionally
        # excluded.
        distance_fn = getattr(self, "_table_finger_min_distance", None)
        actual_distance = float("inf")
        if callable(distance_fn):
            actual_distance = float(distance_fn(self.data))
            self._table_finger_actual_min_distance = actual_distance
        if actual_distance < 0.0 and table_remaining > 0.0:
            magnitude = min(
                -self.rl_config.reward_table_contact_step * scale,
                table_remaining,
            )
            self._last_table_contact_reward = -magnitude
            self._table_safety_penalty_total += magnitude

        sigma_span = max(
            self.rl_config.singularity_penalty_soft_sigma
            - self.rl_config.singularity_penalty_hard_sigma,
            1e-9,
        )
        sigma_severity = float(np.clip(
            (self.rl_config.singularity_penalty_soft_sigma
             - self._last_ik_singularity_sigma_min) / sigma_span,
            0.0,
            1.0,
        ))
        margin_target = self.rl_config.singularity_penalty_joint_margin_ratio
        margin_severity = float(np.clip(
            (margin_target - self._last_ik_joint_margin_ratio)
            / max(margin_target, 1e-9),
            0.0,
            1.0,
        ))
        severity = max(sigma_severity, margin_severity)
        self._last_singularity_penalty_severity = severity
        singularity_remaining = max(
            0.0,
            self.rl_config.singularity_safety_penalty_cap
            - self._singularity_safety_penalty_total,
        )
        if severity > 0.0 and singularity_remaining > 0.0:
            magnitude = min(
                -self.rl_config.reward_singularity_step
                * severity * severity * scale,
                singularity_remaining,
            )
            self._last_singularity_safety_reward = -magnitude
            self._singularity_safety_penalty_total += magnitude

        return {
            "reward_table_filter": self._last_table_filter_reward,
            "reward_table_contact": self._last_table_contact_reward,
            "reward_singularity_safety": self._last_singularity_safety_reward,
        }

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
        self._table_safety_penalty_total = 0.0
        self._singularity_safety_penalty_total = 0.0
        self._last_table_filter_reward = 0.0
        self._last_table_contact_reward = 0.0
        self._last_singularity_safety_reward = 0.0
        self._last_singularity_penalty_severity = 0.0
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
        self._previous_alignment_potential = None
        self._nearest_graspable_distance = 0.0
        self._nearest_graspable_segment_index = -1
        self._alignment_score = 0.0
        self._alignment_potential = 0.0
        self._desired_hand_rotation = self.data.xmat[
            self.base_env.hand_id
        ].reshape(3, 3).copy()
        self._last_ik_velocity_scale = 1.0
        self._last_ik_singularity_sigma_min = float("inf")
        self._last_ik_singularity_condition = 1.0
        self._last_ik_singularity_damping = self.rl_config.ik_damping
        self._last_ik_singularity_severity = 0.0
        self._last_ik_singularity_guard_active = False
        self._last_ik_singularity_guard_count = 0
        self._last_ik_singularity_target_clipped = False
        self._last_ik_singularity_target_clip_count = 0
        self._last_ik_joint_margin_ratio = 1.0
        self._gripper_switch_event = False
        self._gripper_switch_count = 0
        self._max_contacting_fingers = 0
        self._pinch_rewarded = False
        self._aligned_pinch_rewarded = False
        self._aligned_pinch_event = False
        self._ever_aligned_pinch = False
        self._alignment_score_at_first_pinch = None
        self._capture_ready = False
        self._capture_local_offset = np.full(3, np.nan)
        self._capture_insertion_depth = float("nan")
        self._capture_close_event = False
        self._capture_close_rewarded = False
        self._premature_close_event = False
        self._premature_close_penalty_total = 0.0
        self._secured_rewarded = False
        self._lift_credit_high_water = 0.0
        self._cable_lift_credit_high_water = float(np.clip(
            float(info["lifted_fraction"]),
            0.0,
            self.rl_config.cable_lift_credit_cap,
        ))
        self._lift_credit_previous = 0.0
        self._cable_lift_credit_previous = self._cable_lift_credit_high_water
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
        self._previous_pinch_confirmed = False
        self._pinch_session_elapsed = 0.0
        self._pinch_session_lift_high_water = 0.0
        self._pinch_session_lift_peak_global = 0.0
        self._pinch_session_stall_penalty = 0.0
        self._unloaded_pinch_penalty_total = 0.0
        self._failed_unloaded_pinch_count = 0
        self._failed_unloaded_pinch_event = False
        self._lift_attempt = False
        self._loaded_lift = False
        self._pinch_session_start_hand_z = None
        self._post_pinch_world_z_action_sum = 0.0
        self._post_pinch_world_z_action_count = 0
        self._last_commanded_world_translation = np.zeros(3)
        (
            nearest_point, self._previous_distance, tangent,
            self._nearest_graspable_segment_index, _,
        ) = self._nearest_graspable_segment(self._grasp_center_position())
        self._nearest_graspable_distance = self._previous_distance
        self._alignment_score, self._alignment_potential = self._alignment_terms(
            self._previous_distance, tangent
        )
        (
            self._capture_local_offset,
            self._capture_insertion_depth,
            _,
        ) = self._capture_corridor_terms(nearest_point)
        self._previous_alignment_potential = self._alignment_potential
        return self._observation(), self._augment_info(info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        normalized_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        mujoco_action = self._convert_action(normalized_action)
        _, _, base_success, truncated, info = self.base_env.step(mujoco_action)
        self._total_env_steps += 1
        grasp_status = self._update_grasp_status(info)
        # RL 的原始双侧接触/secured进度在50 Hz动作边界检查，作为训练漏斗
        # 和过程奖励诊断。跨方法共享的底层任务成功是唯一终止条件与终点奖励
        # 来源；PPO不能用额外条件改变任务定义。
        strict_success = bool(
            grasp_status["rl_hold_success"]
            and info.get("success_now", False)
        )
        grasp_status["strict_success"] = strict_success
        task_success = bool(base_success)
        reward, reward_components = self._reward(
            normalized_action, task_success, info, grasp_status
        )

        self._episode_return += reward
        self._episode_steps += 1
        info = self._augment_info(info, base_success=base_success)
        info.update(reward_components)
        info["episode_return"] = self._episode_return
        info["episode_steps"] = self._episode_steps
        info["mujoco_action"] = mujoco_action.copy()
        return self._observation(), reward, task_success, bool(truncated), info

    def _augment_info(self, info: dict, *, base_success: bool | None = None) -> dict:
        result = dict(info)
        result["target_distance"] = self._nearest_graspable_distance
        result["nearest_graspable_distance"] = self._nearest_graspable_distance
        result["nearest_graspable_segment_index"] = (
            self._nearest_graspable_segment_index
        )
        result["alignment_score"] = self._alignment_score
        result["alignment_potential"] = self._alignment_potential
        result["alignment_reward_active"] = bool(
            self._previous_alignment_potential is not None
        )
        result["rl_grasped_body_id"] = self._locked_body_id
        result["ik_velocity_scale"] = self._last_ik_velocity_scale
        result["ik_singularity_sigma_min"] = (
            self._last_ik_singularity_sigma_min
        )
        result["ik_singularity_condition"] = (
            self._last_ik_singularity_condition
        )
        result["ik_singularity_damping"] = self._last_ik_singularity_damping
        result["ik_singularity_severity"] = (
            self._last_ik_singularity_severity
        )
        result["ik_singularity_guard_active"] = (
            self._last_ik_singularity_guard_active
        )
        result["ik_singularity_guard_count"] = (
            self._last_ik_singularity_guard_count
        )
        result["ik_singularity_target_clipped"] = (
            self._last_ik_singularity_target_clipped
        )
        result["ik_singularity_target_clip_count"] = (
            self._last_ik_singularity_target_clip_count
        )
        result["ik_joint_margin_ratio"] = self._last_ik_joint_margin_ratio
        result["safety_reward_enabled"] = self.rl_config.safety_reward_enabled
        result["safety_penalty_scale"] = self._last_safety_penalty_scale
        result["table_safety_penalty_total"] = self._table_safety_penalty_total
        result["singularity_safety_penalty_total"] = (
            self._singularity_safety_penalty_total
        )
        result["singularity_penalty_severity"] = (
            self._last_singularity_penalty_severity
        )
        result["gripper_switch_event"] = self._gripper_switch_event
        result["gripper_switch_count"] = self._gripper_switch_count
        result["premature_close_event"] = self._premature_close_event
        result["premature_close_penalty_total"] = (
            self._premature_close_penalty_total
        )
        result["commanded_world_translation"] = (
            self._last_commanded_world_translation.copy()
        )
        result["motion_curriculum_difficulty"] = self._motion_difficulty
        result["motion_curriculum_scenarios"] = self._active_scenario_names
        result["motion_curriculum_scenario_probabilities"] = (
            self._active_scenario_probabilities
        )
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
        result["aligned_pinch_event"] = self._aligned_pinch_event
        result["ever_aligned_pinch"] = self._ever_aligned_pinch
        result["alignment_score_at_first_pinch"] = (
            self._alignment_score_at_first_pinch
        )
        result["gripper_closed"] = self._gripper_closed
        result["capture_ready"] = self._capture_ready
        result["capture_local_x"] = float(self._capture_local_offset[0])
        result["capture_local_y"] = float(self._capture_local_offset[1])
        result["capture_local_z"] = float(self._capture_local_offset[2])
        result["capture_insertion_depth"] = self._capture_insertion_depth
        result["capture_close_event"] = self._capture_close_event
        result["pinch_session_elapsed"] = self._pinch_session_elapsed
        result["pinch_session_lift_high_water"] = (
            self._pinch_session_lift_high_water
        )
        result["pinch_session_lift_peak"] = self._pinch_session_lift_peak_global
        result["failed_unloaded_pinch_event"] = (
            self._failed_unloaded_pinch_event
        )
        result["failed_unloaded_pinch_count"] = (
            self._failed_unloaded_pinch_count
        )
        result["unloaded_pinch_penalty_total"] = (
            self._unloaded_pinch_penalty_total
        )
        result["lift_attempt"] = self._lift_attempt
        result["loaded_lift"] = self._loaded_lift
        result["post_pinch_world_z_action_mean"] = (
            self._post_pinch_world_z_action_sum
            / self._post_pinch_world_z_action_count
            if self._post_pinch_world_z_action_count
            else 0.0
        )
        strict_success = bool(self._last_grasp_status["strict_success"])
        task_success = bool(result["base_success"])
        result["strict_success"] = strict_success
        result["policy_internal_success"] = strict_success
        result["task_success"] = task_success
        result["success"] = task_success
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
        self.set_training_scenario_distribution(
            names, tuple(1.0 for _ in names)
        )

    def set_training_scenario_distribution(
        self,
        scenario_names: Sequence[str],
        probabilities: Sequence[float],
    ) -> None:
        """Set a weighted scenario distribution used by subsequent resets."""
        names = tuple(str(name) for name in scenario_names)
        if not names:
            raise ValueError("training scenario subset cannot be empty")
        if len(set(names)) != len(names):
            raise ValueError("training scenario names must be unique")
        unknown = set(names) - set(self._scenario_by_name)
        if unknown:
            raise ValueError(
                "curriculum scenarios were not compiled into this environment: "
                + ", ".join(sorted(unknown))
            )
        weights = np.asarray(tuple(probabilities), dtype=float)
        if weights.shape != (len(names),):
            raise ValueError(
                "scenario probabilities must match the number of scenarios"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(
                "scenario probabilities must be finite and non-negative"
            )
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("scenario probabilities must have positive mass")
        self._active_scenario_names = names
        self._active_scenario_probabilities = tuple(
            float(value) for value in weights / total
        )

    def set_motion_difficulty(self, difficulty: float) -> None:
        """Set low-to-nominal motion difficulty for subsequent episode resets."""
        difficulty = float(difficulty)
        if not np.isfinite(difficulty):
            raise ValueError("motion difficulty must be finite")
        self._motion_difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def close(self) -> None:
        # 原生MuJoCo数据没有额外线程或窗口；保留Gymnasium标准接口。
        return None


class VerticalDownRLCableGraspEnv(RLCableGraspEnv):
    """5D task-space IK with a world-vertical-down tool approach axis.

    The common action limiter and guarded execution path remain active.  Only
    the original task-space decoder's per-joint velocity rescaling is removed.
    The gripper's yaw is retained around world Z.
    """

    # A small positive margin prevents the predicted finger meshes from
    # entering the tabletop while leaving the cable-table contact untouched.
    TABLE_FINGER_COLLISION_SAFETY_MARGIN = 0.0002
    TABLE_FINGER_COLLISION_SEARCH_STEPS = 8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_mode = VERTICAL_DOWN_TASK_SPACE_ACTION_MODE
        self._table_finger_probe_data = mujoco.MjData(self.model)
        self._table_finger_geom_ids = tuple(
            int(geom_id) for geom_id in self.base_env.finger_collision_geom_ids
        )
        self._table_geom_id = int(self.base_env.table_geom_id)
        self._table_finger_filter_count = 0
        self._table_finger_filter_active = False
        self._table_finger_filter_alpha = 1.0
        self._table_finger_actual_min_distance = float("inf")
        self._table_finger_predicted_min_distance = float("inf")

    def _table_finger_min_distance(self, data: mujoco.MjData) -> float:
        minimum = float("inf")
        for contact in data.contact[:data.ncon]:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            is_pair = (
                geom1 == self._table_geom_id
                and geom2 in self._table_finger_geom_ids
            ) or (
                geom2 == self._table_geom_id
                and geom1 in self._table_finger_geom_ids
            )
            if is_pair:
                minimum = min(minimum, float(contact.dist))
        return minimum

    def _probe_table_finger_distance(self, target_qpos: np.ndarray) -> float:
        probe = self._table_finger_probe_data
        probe.qpos[:] = self.data.qpos
        probe.qpos[self.base_env.arm_qpos_adr] = np.asarray(
            target_qpos, dtype=float
        )
        mujoco.mj_forward(self.model, probe)
        return self._table_finger_min_distance(probe)

    def _table_finger_recovery_target(self) -> np.ndarray:
        """Generate a small upward escape target when contact already exists."""
        jacp, _ = self.base_env._hand_jacobian()
        jacobian = jacp[:, self.base_env.arm_dof_adr]
        upward_delta = np.array([0.0, 0.0, 0.01], dtype=float)
        damping = float(self.rl_config.ik_damping)
        q_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(3),
            upward_delta,
        )
        joint_range = self.model.jnt_range[self.base_env.arm_joint_ids]
        return np.clip(
            self.data.qpos[self.base_env.arm_qpos_adr] + q_delta,
            joint_range[:, 0],
            joint_range[:, 1],
        )

    def _project_table_safe_target(self, candidate_qpos: np.ndarray) -> np.ndarray:
        margin = float(self.TABLE_FINGER_COLLISION_SAFETY_MARGIN)
        self._table_finger_filter_active = False
        self._table_finger_filter_alpha = 1.0
        self._table_finger_actual_min_distance = self._table_finger_min_distance(
            self.data
        )
        candidate_qpos = np.asarray(candidate_qpos, dtype=float)
        candidate_distance = self._probe_table_finger_distance(candidate_qpos)
        self._table_finger_predicted_min_distance = candidate_distance
        if candidate_distance >= margin:
            return candidate_qpos

        # A target-only projection cannot undo contact already accumulated by
        # PD tracking and inertia.  In that case issue a small upward escape
        # target first; cable-table contacts are not part of this filter.
        if self._table_finger_actual_min_distance < margin:
            recovery_qpos = self._table_finger_recovery_target()
            recovery_distance = self._probe_table_finger_distance(recovery_qpos)
            if recovery_distance > self._table_finger_actual_min_distance + 1e-8:
                self._table_finger_filter_active = True
                self._table_finger_filter_count += 1
                self._table_finger_filter_alpha = 0.0
                self._table_finger_predicted_min_distance = recovery_distance
                return recovery_qpos

        joint_range = self.model.jnt_range[self.base_env.arm_joint_ids]
        start_qpos = np.clip(
            np.asarray(self.base_env._last_applied_action[:7], dtype=float),
            joint_range[:, 0],
            joint_range[:, 1],
        )
        start_distance = self._probe_table_finger_distance(start_qpos)

        # If the current target is already in contact, prioritize a target that
        # increases clearance so the filter cannot trap the arm against the
        # tabletop.  Otherwise hold the last safe target.
        if start_distance < margin:
            if candidate_distance > start_distance + 1e-8:
                self._table_finger_filter_active = True
                self._table_finger_filter_count += 1
                self._table_finger_filter_alpha = 1.0
                return candidate_qpos
            self._table_finger_filter_active = True
            self._table_finger_filter_count += 1
            self._table_finger_filter_alpha = 0.0
            self._table_finger_predicted_min_distance = start_distance
            return start_qpos

        # Binary-search the largest fraction of the original target motion
        # that stays outside the small tabletop clearance margin.
        low = 0.0
        high = 1.0
        safe_qpos = start_qpos
        for _ in range(self.TABLE_FINGER_COLLISION_SEARCH_STEPS):
            alpha = 0.5 * (low + high)
            probe_qpos = start_qpos + alpha * (candidate_qpos - start_qpos)
            probe_distance = self._probe_table_finger_distance(probe_qpos)
            if probe_distance >= margin:
                low = alpha
                safe_qpos = probe_qpos
            else:
                high = alpha
        self._table_finger_filter_active = True
        self._table_finger_filter_count += 1
        self._table_finger_filter_alpha = low
        self._table_finger_predicted_min_distance = (
            self._probe_table_finger_distance(safe_qpos)
        )
        return safe_qpos

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        mujoco_action = super()._convert_action(action)
        if (
            not self.table_finger_collision_filter_enabled
            or self.base_env.config.geometric_safety_enabled
        ):
            return mujoco_action
        mujoco_action[:7] = self._project_table_safe_target(mujoco_action[:7])
        return mujoco_action

    def _augment_info(self, info: dict, *, base_success: bool | None = None) -> dict:
        result = super()._augment_info(info, base_success=base_success)
        result["table_finger_collision_avoidance_enabled"] = (
            self.table_finger_collision_filter_enabled
        )
        result["table_finger_collision_filter_active"] = (
            self._table_finger_filter_active
        )
        result["table_finger_collision_filter_count"] = (
            self._table_finger_filter_count
        )
        result["table_finger_collision_filter_alpha"] = (
            self._table_finger_filter_alpha
        )
        result["table_finger_actual_min_distance"] = (
            float(self._table_finger_actual_min_distance)
            if np.isfinite(self._table_finger_actual_min_distance)
            else None
        )
        result["table_finger_predicted_min_distance"] = (
            float(self._table_finger_predicted_min_distance)
            if np.isfinite(self._table_finger_predicted_min_distance)
            else None
        )
        result["table_finger_collision_safety_margin"] = (
            self.TABLE_FINGER_COLLISION_SAFETY_MARGIN
        )
        return result

    def _vertical_down_rotation(self) -> np.ndarray:
        current = self.data.xmat[self.base_env.hand_id].reshape(3, 3)
        approach_world = np.array([0.0, 0.0, -1.0], dtype=float)
        closing_world = current @ np.asarray(
            self.base_env.gripper_closing_axis_local, dtype=float
        )
        closing_world[2] = 0.0
        closing_norm = float(np.linalg.norm(closing_world))
        if closing_norm < 1e-8:
            closing_world = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            closing_world /= closing_norm
        # Local axes are x=approach, y=lateral, z=closing, so y=z cross x.
        lateral_world = np.cross(closing_world, approach_world)
        lateral_world /= max(float(np.linalg.norm(lateral_world)), 1e-8)
        return np.column_stack((approach_world, lateral_world, closing_world))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        observation, info = super().reset(seed=seed, options=options)
        self._desired_hand_rotation = self._vertical_down_rotation()
        self._table_finger_filter_count = 0
        self._table_finger_filter_active = False
        self._table_finger_filter_alpha = 1.0
        self._table_finger_actual_min_distance = float("inf")
        self._table_finger_predicted_min_distance = float("inf")
        return observation, info

    def _limit_ik_joint_velocity(self, q_velocity: np.ndarray) -> np.ndarray:
        # Keep the shared action limiter and velocity guard; remove only the
        # original task-space decoder's per-joint velocity scaling.
        self._last_ik_velocity_scale = 1.0
        return np.asarray(q_velocity, dtype=float)


class JointTargetRLCableGraspEnv(RLCableGraspEnv):
    """RL environment with direct seven-joint target deltas plus gripper.

    The seven arm outputs are normalized per-control-cycle target deltas.  They
    are mapped directly to the arm joint position targets and then passed
    through the shared target-velocity limiter and PD controller.  No task-
    space IK is used in this mode.
    """

    ACTION_NAMES = (
        *(f"joint_target_delta_{i}" for i in range(1, 8)),
        "gripper_hysteresis",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_mode = JOINT_TARGET_ACTION_MODE
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self._previous_action = np.zeros(8, dtype=float)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        observation, info = super().reset(seed=seed, options=options)
        self._previous_action = np.zeros(8, dtype=float)
        return observation, info

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        normalized = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if normalized.shape != (8,):
            raise ValueError(
                f"Expected joint-target RL action shape (8,), got {normalized.shape}"
            )

        control_dt = float(
            self.model.opt.timestep * max(1, self.base_env.config.frame_skip)
        )
        previous_target = self.base_env._last_applied_action[:7]
        velocity_limits = np.asarray(
            self.base_env.config.arm_joint_velocity_limits, dtype=float
        )
        target_delta = normalized[:7] * velocity_limits * control_dt
        joint_range = self.model.jnt_range[self.base_env.arm_joint_ids]
        target_qpos = np.clip(
            previous_target + target_delta,
            joint_range[:, 0],
            joint_range[:, 1],
        )
        mujoco_action = np.empty(8, dtype=float)
        mujoco_action[:7] = target_qpos

        previous_gripper_closed = self._gripper_closed
        if normalized[7] <= self.rl_config.gripper_close_threshold:
            self._gripper_closed = True
        elif normalized[7] >= self.rl_config.gripper_open_threshold:
            self._gripper_closed = False
        self._gripper_switch_event = bool(
            self._gripper_closed != previous_gripper_closed
        )
        self._gripper_switch_count += int(self._gripper_switch_event)
        gripper_range = self.model.actuator_ctrlrange[
            self.base_env.gripper_actuator_id
        ]
        closed_ctrl = float(np.clip(
            self.rl_config.gripper_closed_ctrl,
            gripper_range[0],
            gripper_range[1],
        ))
        mujoco_action[self.base_env.gripper_actuator_id] = (
            closed_ctrl if self._gripper_closed else float(gripper_range[1])
        )
        self._last_ik_velocity_scale = 1.0
        self._last_commanded_world_translation[:] = 0.0
        return mujoco_action


def make_rl_env(*, action_mode: str = TASK_SPACE_ACTION_MODE, **kwargs):
    """Construct the requested RL action interface without mixing checkpoints."""
    if action_mode == TASK_SPACE_ACTION_MODE:
        return RLCableGraspEnv(**kwargs)
    if action_mode == VERTICAL_DOWN_TASK_SPACE_ACTION_MODE:
        return VerticalDownRLCableGraspEnv(**kwargs)
    if action_mode == JOINT_TARGET_ACTION_MODE:
        return JointTargetRLCableGraspEnv(**kwargs)
    raise ValueError(f"unsupported RL action mode: {action_mode!r}")
