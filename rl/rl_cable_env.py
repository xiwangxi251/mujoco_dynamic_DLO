"""供强化学习使用的 Gymnasium 接口。

该文件只负责把观测和动作转换成固定长度向量，并计算训练奖励。线缆扰动、
接触、纯摩擦夹持和成功判定仍完全由 CableGraspEnv 决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import gymnasium as gym
import mujoco
import numpy as np

from cable_grasp_env import CableGraspEnv, EnvConfig, rotation_to_quat
from experiment_scenarios import ScenarioConfig, get_scenario


@dataclass
class RLConfig:
    """RL接口参数，不改变底层线缆物理。"""

    joint_delta_scale: float = 0.04  # 每个50 Hz动作允许改变的关节目标，单位rad
    target_velocity_scale: float = 1.0
    arm_velocity_scale: float = 3.0
    finger_velocity_scale: float = 0.5
    hand_linear_velocity_scale: float = 1.0
    hand_angular_velocity_scale: float = 4.0
    normal_force_scale: float = 20.0
    gripper_close_threshold: float = -0.35
    gripper_open_threshold: float = 0.35
    gripper_closed_ctrl: float = 20.0
    secured_lift_delta: float = 0.03
    secured_confirm_seconds: float = 0.10
    secured_contact_loss_grace_seconds: float = 0.06
    lift_credit_cap: float = 0.12
    cable_lift_credit_cap: float = 0.30
    reward_reach_progress: float = 6.0
    reward_new_contact: float = 0.15
    reward_new_pinch: float = 0.75
    reward_new_secured_grasp: float = 3.0
    reward_lift_progress: float = 20.0
    reward_cable_lift_progress: float = 3.0
    reward_strict_hold_progress: float = 0.8
    reward_success: float = 25.0
    reward_slip: float = -3.0
    reward_active_open_after_secured: float = -8.0
    reward_open_during_contact_loss: float = -3.0
    reward_time_step: float = -0.002
    reward_action_magnitude: float = -0.0005
    reward_action_rate: float = -0.01
    reward_post_grasp_arm_action_magnitude: float = -0.02
    reward_post_grasp_arm_action_rate: float = -0.05


class RLCableGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """Panda动态线缆抓取的连续控制环境。

    观测不包含完整线缆形状，只给策略目标段状态与机器人本体状态。动作不经过
    脚本状态机：策略可在任意时刻移动任意关节并开合夹爪。
    """

    metadata = {"render_modes": []}
    # 实际夹持中心由底层环境统一定义，RL观测不再维护重复的局部坐标常量。

    OBSERVATION_NAMES = (
        "target_pos_x", "target_pos_y", "target_pos_z",
        "target_vel_x", "target_vel_y", "target_vel_z",
        "target_relative_x", "target_relative_y", "target_relative_z",
        *(f"arm_qpos_{i}" for i in range(1, 8)),
        *(f"arm_qvel_{i}" for i in range(1, 8)),
        "left_finger_qpos", "right_finger_qpos",
        "left_finger_qvel", "right_finger_qvel",
        "hand_pos_x", "hand_pos_y", "hand_pos_z",
        "hand_quat_w", "hand_quat_x", "hand_quat_y", "hand_quat_z",
        "hand_linear_vel_x", "hand_linear_vel_y", "hand_linear_vel_z",
        "hand_angular_vel_x", "hand_angular_vel_y", "hand_angular_vel_z",
        "left_pad_contact", "right_pad_contact",
        "left_pad_normal_force", "right_pad_normal_force",
        "pinch_confirmed", "secured_grasp",
        "grasp_lift_delta", "strict_success_progress",
    )

    def __init__(
        self,
        *,
        seed: int = 20260804,
        disturbance_strength: float = 1.5,
        episode_seconds: float = 28.0,
        env_config: EnvConfig | None = None,
        scenario_names: Sequence[str] | None = None,
        rl_config: RLConfig | None = None,
    ):
        super().__init__()
        self.rl_config = rl_config or RLConfig()
        self._scenario_configs: tuple[ScenarioConfig, ...] = tuple(
            get_scenario(name) for name in (scenario_names or ())
        )
        self._scenario_strength_scale = float(disturbance_strength) / 1.5
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
                first_scenario.disturbance_strength * self._scenario_strength_scale
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
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -10.0, 10.0, shape=(len(self.OBSERVATION_NAMES),), dtype=np.float32
        )
        self._previous_distance = 0.0
        self._episode_return = 0.0
        self._episode_steps = 0
        self._pinch_reference_z: np.ndarray | None = None
        self._secured_candidate_hold = 0.0
        self._secured_contact_loss_hold = 0.0
        self._strict_success_hold = 0.0
        self._ever_secured_grasp = False
        self._last_grasp_status = self._empty_grasp_status()
        self._locked_body_id: int | None = None
        self._gripper_closed = False
        self._previous_action = np.zeros(8)
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
        # 训练中的普通reset随机采样；严格评估传连续显式seed时按模循环，保证
        # core/ID场景近似等次数覆盖，而不是小样本恰好漏掉某种运动类型。
        index = (
            int(explicit_seed) % len(self._scenario_configs)
            if explicit_seed is not None
            else int(self.np_random.integers(0, len(self._scenario_configs)))
        )
        scenario = self._scenario_configs[index]
        overrides = scenario.to_env_overrides()
        for name, value in overrides.items():
            if name.startswith("cable_"):
                # 已在构造时验证并编译；重复赋值仅会掩盖意外的模型/配置差异。
                continue
            if name == "disturbance_strength":
                value = scenario.disturbance_strength * self._scenario_strength_scale
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
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_env.hand_id,
            velocity,
            0,
        )
        return velocity[3:].copy(), velocity[:3].copy()

    def _grasp_center_position(self) -> np.ndarray:
        """返回底层环境统一定义的实际两指夹持中心。"""
        return self.base_env.hand_position.copy()

    def _task_target_body_id(self) -> int:
        """夹持形成后使用实际夹持线段，不再追逐原随机参考节点。"""
        state = self.base_env.grasp_state
        if state is not None and self.base_env.grasp_confirmed:
            return state.body_id
        if self._locked_body_id is not None:
            return self._locked_body_id
        return self.base_env.target_body_id

    def _task_target_position(self) -> np.ndarray:
        return self.data.xpos[self._task_target_body_id()].copy()

    def _task_target_velocity(self) -> np.ndarray:
        return self.base_env.body_linear_velocity(self._task_target_body_id())

    def _observation(self) -> np.ndarray:
        """返回经过固定尺度归一化的目标状态和机器人本体状态。"""
        target_position = self._task_target_position()
        target_velocity = self._task_target_velocity()
        hand_position = self._grasp_center_position()
        hand_rotation = self.data.xmat[self.base_env.hand_id].reshape(3, 3)
        hand_quat = rotation_to_quat(hand_rotation)
        hand_linear_velocity, hand_angular_velocity = self._hand_velocity()

        arm_qpos = self.data.qpos[self.base_env.arm_qpos_adr]
        arm_qvel = self.data.qvel[self.base_env.arm_dof_adr]
        finger_qpos = self.data.qpos[self.base_env.finger_qpos_adr]
        finger_dof_adr = self.model.jnt_dofadr[self.base_env.finger_joint_ids]
        finger_qvel = self.data.qvel[finger_dof_adr]
        contact_pairs = self.base_env._finger_contact_pairs()
        contact_fingers = {finger for _, finger in contact_pairs}
        normal_forces = self.base_env.finger_normal_forces()
        strict_progress = float(np.clip(
            float(self._last_grasp_status["strict_success_hold"])
            / self.base_env.config.success_hold_seconds,
            0.0,
            1.0,
        ))

        # 归一化只改变网络输入尺度，不改变MuJoCo中的物理状态。
        target_position_normalized = (
            target_position - np.array([0.55, 0.0, 0.20])
        ) / np.array([0.40, 0.50, 0.40])
        target_relative = (target_position - hand_position) / 0.50
        hand_position_normalized = (
            hand_position - np.array([0.55, 0.0, 0.25])
        ) / np.array([0.45, 0.55, 0.45])

        observation = np.concatenate([
            target_position_normalized,
            target_velocity / self.rl_config.target_velocity_scale,
            target_relative,
            (arm_qpos - self._arm_center) / self._arm_half_range,
            arm_qvel / self.rl_config.arm_velocity_scale,
            2.0 * finger_qpos / 0.04 - 1.0,
            finger_qvel / self.rl_config.finger_velocity_scale,
            hand_position_normalized,
            hand_quat,
            hand_linear_velocity / self.rl_config.hand_linear_velocity_scale,
            hand_angular_velocity / self.rl_config.hand_angular_velocity_scale,
            np.array([
                float(self.base_env.left_finger_id in contact_fingers),
                float(self.base_env.right_finger_id in contact_fingers),
                normal_forces[self.base_env.left_finger_id]
                / self.rl_config.normal_force_scale,
                normal_forces[self.base_env.right_finger_id]
                / self.rl_config.normal_force_scale,
                float(bool(self._last_grasp_status["pinch_confirmed"])),
                float(bool(self._last_grasp_status["secured_grasp"])),
                float(self._last_grasp_status["grasp_lift_delta"]) / 0.15,
                strict_progress,
            ]),
        ]).astype(np.float32)
        return np.clip(observation, -10.0, 10.0)

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        """把[-1,1]策略动作映射为MuJoCo合法的执行器命令。"""
        normalized = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if normalized.shape != (8,):
            raise ValueError(f"Expected RL action shape (8,), got {normalized.shape}")

        current_qpos = self.data.qpos[self.base_env.arm_qpos_adr]
        target_qpos = current_qpos + normalized[:7] * self.rl_config.joint_delta_scale
        joint_range = self.model.jnt_range[self.base_env.arm_joint_ids]

        mujoco_action = np.empty(8)
        mujoco_action[:7] = np.clip(target_qpos, joint_range[:, 0], joint_range[:, 1])
        # 使用开合滞回，避免连续动作在单一阈值附近抖动并反复清除抓取状态。
        gripper_range = self.model.actuator_ctrlrange[7]
        if normalized[7] <= self.rl_config.gripper_close_threshold:
            self._gripper_closed = True
        elif normalized[7] >= self.rl_config.gripper_open_threshold:
            self._gripper_closed = False
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
            self._pinch_reference_z = None
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

        if pinch_confirmed and self._pinch_reference_z is None:
            self._pinch_reference_z = self.data.xpos[self.base_env.cable_ids, 2].copy()
            self._locked_body_id = state.body_id

        body_id = state.body_id
        body_index = self.base_env.cable_index[body_id]
        lift_delta = 0.0
        if self._pinch_reference_z is not None:
            lift_delta = float(
                self.data.xpos[body_id, 2] - self._pinch_reference_z[body_index]
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
        distance = float(np.linalg.norm(
            self._task_target_position() - self._grasp_center_position()
        ))
        progress = self._previous_distance - distance
        self._previous_distance = distance

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

        active_open, physical_slip, open_during_contact_loss = (
            self._classify_secured_grasp_break()
        )
        # 多次重抓/断开仍完整计入诊断。外力滑脱与歧义张爪共用一次较轻惩罚；
        # 明确主动张爪另有一次较重惩罚，避免先制造低成本滑脱后免费张爪。
        penalize_active_open = bool(
            active_open and not self._active_open_penalty_applied
        )
        penalize_contact_loss = bool(
            (physical_slip or open_during_contact_loss)
            and not self._contact_loss_penalty_applied
        )
        if penalize_active_open:
            self._active_open_penalty_applied = True
        if penalize_contact_loss:
            self._contact_loss_penalty_applied = True
        penalize_physical_slip = bool(penalize_contact_loss and physical_slip)
        penalize_ambiguous_open = bool(
            penalize_contact_loss and open_during_contact_loss
        )

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
        arm_action_magnitude = float(np.mean(np.square(action[:7])))
        arm_action_rate = float(np.mean(np.square(
            action[:7] - self._previous_action[:7]
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
            "reward_open_during_contact_loss": (
                self.rl_config.reward_open_during_contact_loss
                * float(penalize_ambiguous_open)
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
        self._pinch_reference_z = None
        self._secured_candidate_hold = 0.0
        self._secured_contact_loss_hold = 0.0
        self._strict_success_hold = 0.0
        self._ever_secured_grasp = False
        self._last_grasp_status = self._empty_grasp_status()
        self._locked_body_id = None
        self._gripper_closed = False
        self._previous_action = np.zeros(8)
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
        self._previous_distance = float(np.linalg.norm(
            self._task_target_position() - self._grasp_center_position()
        ))
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
        result["target_distance"] = float(np.linalg.norm(
            self._task_target_position() - self._grasp_center_position()
        ))
        result["rl_target_body_id"] = self._task_target_body_id()
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
        """供训练课程回调调整强度；正式评估始终显式使用完整强度。"""
        strength = float(strength)
        if self._scenario_configs:
            self._scenario_strength_scale = strength / 1.5
            active_name = self.base_env.config.scenario_name
            scenario = next(
                (
                    item for item in self._scenario_configs
                    if item.name == active_name
                ),
                self._scenario_configs[0],
            )
            self.base_env.config.disturbance_strength = (
                scenario.disturbance_strength * self._scenario_strength_scale
            )
        else:
            self.base_env.config.disturbance_strength = strength

    def close(self) -> None:
        # 原生MuJoCo数据没有额外线程或窗口；保留Gymnasium标准接口。
        return None
