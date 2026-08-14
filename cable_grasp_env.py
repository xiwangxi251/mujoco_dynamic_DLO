"""环境端
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
MENAGERIE = ROOT.parent / "mujoco_menagerie"
XML_PATH = MENAGERIE / "franka_emika_panda" / "panda_cable_grasp.xml"


@dataclass
class EnvConfig:
    """环境参数"""

    seed: int = 20260804
    episode_seconds: float = 28.0       # 每次随机试验最多运行多少仿真秒
    disturbance_strength: float = 1.5   # 线缆外力倍率

    # 场景标记
    scenario_name: str = "legacy_shape_current"
    scenario_id: str | None = None
    scenario_split: str = "legacy"

    # 线缆运动形式
    motion_mode: str = "shape"          # static / rigid / shape / combined
    motion_profile_version: str = "legacy_v1"  # legacy_v1 / factorized_v1
    motion_regularity: str = "quasiperiodic"  # regular / quasiperiodic / stochastic
    motion_frequency_scale: float = 1.0
    shape_motion_scale: float = 1.0
    rigid_translation_scale: float = 1.0
    rigid_rotation_scale: float = 1.0

    # 线缆 OOD 参数相对 XML 标称值缩放（改变长度、质量、弹性、阻尼和摩擦）
    cable_length_scale: float = 1.0
    cable_density_scale: float = 1.0
    cable_stiffness_scale: float = 1.0
    cable_damping_scale: float = 1.0
    cable_friction_scale: float = 1.0

    # 抓取判断参数
    success_hold_seconds: float = 0.80  # 成功条件必须连续保持的时间
    grasp_confirm_seconds: float = 0.06 # 双侧内指垫接触保持多久才确认抓取
    grasp_loss_seconds: float = 0.35    # 双指接触短暂中断的容忍时间
    max_grasp_aperture: float = 0.034  # 28 mm线缆被真正夹紧时允许的最大开口
    max_pad_distance: float = 0.055    # 接触线段中心到指垫中心的最大距离
    min_pad_normal_force: float = 0.20 # 每侧内指垫所需最小法向力，单位N

    # 仿真和夹爪参数
    frame_skip: int = 10                # 一个50 Hz动作对应10个500 Hz物理步
    gripper_force_scale: float = 5.0    # 提高闭爪位置伺服刚度；执行器最大力范围保持不变
    pad_friction: tuple[float, float, float] = (4.0, 0.10, 0.05)

    # 桌面软边界
    boundary_margin: float = 0.16       # 距桌边多远开始调整环境扰动力
    boundary_stiffness: float = 180.0   # 软边界回正加速度系数，单位1/s²
    boundary_damping: float = 28.0      # 只衰减朝桌外运动的速度，单位1/s

    # 合法性检查
    def __post_init__(self) -> None:
        if self.motion_mode not in {"static", "rigid", "shape", "combined"}:
            raise ValueError(f"unsupported motion_mode: {self.motion_mode!r}")
        if self.motion_regularity not in {
            "regular", "quasiperiodic", "stochastic",
        }:
            raise ValueError(
                f"unsupported motion_regularity: {self.motion_regularity!r}"
            )
        if self.motion_profile_version not in {"legacy_v1", "factorized_v1"}:
            raise ValueError(
                "unsupported motion_profile_version: "
                f"{self.motion_profile_version!r}"
            )
        nonnegative = (
            "disturbance_strength",
            "motion_frequency_scale",
            "shape_motion_scale",
            "rigid_translation_scale",
            "rigid_rotation_scale",
        )
        positive = (
            "episode_seconds",
            "cable_length_scale",
            "cable_density_scale",
            "cable_stiffness_scale",
            "cable_damping_scale",
            "cable_friction_scale",
        )
        for name in (*nonnegative, *positive):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in nonnegative and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            if name in positive and value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be positive")


@dataclass
class GraspState:
    """抓取状态"""

    body_id: int
    candidate_time: float
    bilateral_confirmed: bool  # 双侧夹取确认（60 ms 的连续双侧夹持确）
    last_bilateral_time: float # 上次双侧确认时间
    lost_contact_time: float  # 连续失去双侧接触多时间


def id_of(model: mujoco.MjModel, obj: int, name: str) -> int:
    result = mujoco.mj_name2id(model, obj, name)
    if result < 0:
        raise RuntimeError(f"Missing {name!r} in model")
    return result

class CableGraspEnv:
    """具有独立线缆力场、提供 reset/step 接口的小型环境。

    动作是8维向量:前7维为 Panda 关节位置执行器目标,第8维为夹爪命令。
    255表示张开,较小数值表示闭合。
    观测采用字典形式
    """
    # 夹爪指垫中心
    GRASP_CENTER_LOCAL = np.array([0.0, 0.0, 0.1029])
    # 机器臂初始状态
    READY_ARM_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.model = self._load_model(self.config)

        # 控制夹爪力度
        grip_scale = self.config.gripper_force_scale
        self.model.actuator_gainprm[7, 0] *= grip_scale
        self.model.actuator_biasprm[7, 1] *= grip_scale
        self.model.actuator_biasprm[7, 2] *= math.sqrt(grip_scale)
        self.data = mujoco.MjData(self.model)

        # 批量查找各类对象 ID、地址
        self.home_id = id_of(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.arm_joint_ids = np.array([
            id_of(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ], dtype=int)
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_ids].copy()
        self.arm_dof_adr = self.model.jnt_dofadr[self.arm_joint_ids].copy()
        self.hand_id = id_of(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.table_geom_id = id_of(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        table_center = self.model.geom_pos[self.table_geom_id, :2]
        table_half_size = self.model.geom_size[self.table_geom_id, :2]
        self.table_xy_min = table_center - table_half_size
        self.table_xy_max = table_center + table_half_size
        self.left_finger_id = id_of(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        self.right_finger_id = id_of(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )
        self.finger_ids = {self.left_finger_id, self.right_finger_id}

        # 提升指垫的摩擦
        self.pad_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (
                int(self.model.geom_bodyid[geom_id]) in self.finger_ids
                and self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
                and self.model.geom_contype[geom_id] != 0
            )
        }
        if len(self.pad_geom_ids) != 10:
            raise RuntimeError(
                f"Expected 10 Panda pad collision boxes, found {len(self.pad_geom_ids)}"
            )
        for geom_id in self.pad_geom_ids:
            self.model.geom_priority[geom_id] = 1
            self.model.geom_condim[geom_id] = 6
            self.model.geom_friction[geom_id] = self.config.pad_friction
            self.model.geom_solref[geom_id] = [0.004, 1.0]
            self.model.geom_solimp[geom_id] = [0.98, 0.995, 0.0005, 0.5, 2.0]

        # 获取指关节，线缆 ID、地址
        self.finger_joint_ids = np.array([
            id_of(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
            id_of(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
        ], dtype=int)
        self.finger_qpos_adr = self.model.jnt_qposadr[self.finger_joint_ids].copy()
        self.cable_ids = self._cable_bodies()
        self.cable_set = set(self.cable_ids)
        self.cable_index = {body_id: index for index, body_id in enumerate(self.cable_ids)}
        self.cable_geom_ids = np.array([
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in self.cable_set
        ], dtype=int)
        self.cable_radius = float(self.model.geom_size[self.cable_geom_ids, 0].max())
        self.cable_free_qadr = self._find_cable_free_qpos_address()
        self.cable_free_joint_id = int(np.flatnonzero(
            self.model.jnt_qposadr == self.cable_free_qadr
        )[0])
        self.cable_free_dadr = int(
            self.model.jnt_dofadr[self.cable_free_joint_id]
        )
        self.cable_mass = self.model.body_mass[self.cable_ids].copy()
        self.cable_s = np.linspace(0.0, 1.0, len(self.cable_ids))

        # 预计算相位矩阵，用于之后的扰动外力控制
        self._lateral_space = math.pi * np.outer(
            np.array([3.3, 7.7, 13.1, 19.3]), self.cable_s
        )
        self._longitudinal_space = math.pi * np.outer(
            np.array([5.1, 11.6]), self.cable_s
        )
        self._vertical_space = math.pi * np.outer(
            np.array([4.6, 10.4]), self.cable_s
        )

        # 设置机器臂状态
        self.ready_qpos = self.model.key_qpos[self.home_id, :9].copy()
        self.ready_qpos[:7] = self.READY_ARM_QPOS
        self.ready_ctrl = self.model.key_ctrl[self.home_id, :8].copy()
        self.ready_ctrl[:7] = self.READY_ARM_QPOS
        self.ready_ctrl[7] = 255.0

        # 任务状态参数全部初始化
        self.trial_index = 0

        self.phase_offset = 0.0
        self.spatial_phase = 0.0
        self._stochastic_shape_frequency = np.ones(8)
        self._stochastic_shape_direction = np.ones(8)
        self._stochastic_shape_phase = np.zeros(8)
        self._stochastic_rigid_frequency = np.ones((3, 4))
        self._stochastic_rigid_phase = np.zeros((3, 4))
        self._stochastic_rigid_weight = np.ones((3, 4)) / 2.0

        node_count = len(self.cable_ids)
        self._last_shape_acceleration = np.zeros((node_count, 3))
        self._last_rigid_translation_acceleration = np.zeros((node_count, 3))
        self._last_rigid_rotation_acceleration = np.zeros((node_count, 3))
        self._last_boundary_acceleration = np.zeros((node_count, 3))
        self.motion_profile_hash = ""
        self._rigid_reference_xy = np.zeros((node_count, 2))
        self._rigid_reference_com_xy = np.zeros(2)

        self.target_body_id = self.cable_ids[len(self.cable_ids) // 2]

        self.grasp_state: GraspState | None = None
        self.success_hold = 0.0
        self.success_now = False
        self.ever_success = False
        self.success_snapshot: dict | None = None

        self.last_grasped_body_id: int | None = None
        self.last_grasp_break: dict | None = None
        self.grasp_break_history: list[dict] = []
        self.ever_bilateral_candidate = False
        self.ever_confirmed_grasp = False
        self.episode_seed: int | None = None
        self.initial_cable_translation = np.zeros(2)
        self._last_contact_count = 0
        self.reset()
        # 上面的 reset 只用于让刚构造的对象拥有完整、可查询的初始物理状态，
        # 不是调用方实际运行的 episode。首次显式 reset 应编号为 trial=1。
        self.trial_index = 0

    # -------------------------------------------------------------------------
    # 1. 生命周期主接口：重置环境、执行动作
    # -------------------------------------------------------------------------

    def reset(
        self,
        *,
        randomize: bool = True,
        seed: int | None = None,
    ) -> tuple[dict, dict]:
        """重置机器人、线缆、随机相位、目标线段和所有成功判定状态。"""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.episode_seed = int(seed)
        else:
            # 只有调用方明确给出的 seed 才能作为可复现实验元数据；连续训练
            # 中的普通 reset 继续使用现有 Generator，但不能沿用上一轮的 seed 标签。
            self.episode_seed = None

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = self.ready_qpos
        self.data.ctrl[:8] = self.ready_ctrl
        mujoco.mj_forward(self.model, self.data)

        if randomize:
            # 每轮改变线缆初始平移、波形相位和目标段，但固定 seed 时序列可复现。
            dx = self.rng.uniform(-0.10, 0.10)
            dy = self.rng.uniform(-0.13, 0.13)
            # 长线缆 OOD 的初始端点仍放在桌面内，避免“长度变化”被初始坠桌混淆。
            # 标称 0.8 m 线缆的可行区间比原采样区间宽，因而旧默认数值不变。
            cable_xy = self.data.xpos[self.cable_ids, :2]
            lower = self.table_xy_min + self.cable_radius - cable_xy.min(axis=0)
            upper = self.table_xy_max - self.cable_radius - cable_xy.max(axis=0)
            if np.any(lower > upper):
                raise ValueError(
                    "configured cable does not fit on the table at reset: "
                    f"length_scale={self.config.cable_length_scale}"
                )
            dx = float(np.clip(dx, lower[0], upper[0]))
            dy = float(np.clip(dy, lower[1], upper[1]))
            self.initial_cable_translation[:] = [dx, dy]
            self.data.qpos[self.cable_free_qadr:self.cable_free_qadr + 3] += [dx, dy, 0.0]
            self.phase_offset = self.rng.uniform(0.0, 2.0 * math.pi)
            self.spatial_phase = self.rng.uniform(0.0, 2.0 * math.pi)
            # 避开看起来近似固定的两端，但也不总是选择几何中心。
            lo = len(self.cable_ids) // 4
            hi = len(self.cable_ids) - lo
            target_index = int(self.rng.integers(lo, hi))
            self.target_body_id = self.cable_ids[target_index]
        else:
            self.initial_cable_translation[:] = 0.0
            self.phase_offset = 0.0
            self.spatial_phase = 0.0
            self.target_body_id = self.cable_ids[len(self.cable_ids) // 2]

        self._reset_stochastic_motion(randomize=randomize)
        profile_header = (
            f"{self.config.motion_profile_version}|{self.config.motion_mode}|"
            f"{self.config.motion_regularity}|"
            f"{self.config.disturbance_strength:.17g}|"
            f"{self.config.motion_frequency_scale:.17g}|"
            f"{self.phase_offset:.17g}|{self.spatial_phase:.17g}"
        ).encode("ascii")
        profile_bytes = b"".join((
            profile_header,
            self._stochastic_shape_frequency.tobytes(),
            self._stochastic_shape_direction.tobytes(),
            self._stochastic_shape_phase.tobytes(),
            self._stochastic_rigid_frequency.tobytes(),
            self._stochastic_rigid_phase.tobytes(),
            self._stochastic_rigid_weight.tobytes(),
        ))
        self.motion_profile_hash = hashlib.sha256(profile_bytes).hexdigest()

        self.grasp_state = None
        self.success_hold = 0.0
        self.success_now = False
        self.ever_success = False
        self.success_snapshot = None
        self.last_grasped_body_id = None
        self.last_grasp_break = None
        self.grasp_break_history = []
        self.ever_bilateral_candidate = False
        self.ever_confirmed_grasp = False
        self._last_contact_count = 0
        self._last_shape_acceleration[:] = 0.0
        self._last_rigid_translation_acceleration[:] = 0.0
        self._last_rigid_rotation_acceleration[:] = 0.0
        self._last_boundary_acceleration[:] = 0.0
        self.trial_index += 1
        mujoco.mj_forward(self.model, self.data)
        self._rigid_reference_xy[:] = self.data.xpos[self.cable_ids, :2]
        self._rigid_reference_com_xy[:] = np.average(
            self._rigid_reference_xy, axis=0, weights=self.cable_mass
        )
        return self.observation(), self.info()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """一个50 Hz控制动作内部执行十个500 Hz物理子步。"""
        # action裁剪。
        action = np.asarray(action, dtype=float)
        if action.shape != (8,):
            raise ValueError(f"Expected action shape (8,), got {action.shape}")
        clipped_action = np.clip(
            action,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

        last_qualification = False
        truncated = False
        gripper_closed = bool(clipped_action[7] < 100.0)

        for _ in range(max(1, self.config.frame_skip)):
            # 扰动线缆
            self.data.xfrc_applied[:] = 0.0
            self._apply_cable_disturbance()
            # 执行动作
            self.data.ctrl[:] = clipped_action
            mujoco.mj_step(self.model, self.data)

            # 更新抓取候选
            self._last_contact_count = len(self._finger_contact_pairs())
            self._update_physical_grasp_state(gripper_closed)

            # 任务成功所要求的几何条件检查
            last_qualification = self._success_qualification(gripper_closed)
            if last_qualification:
                self.success_hold += self.model.opt.timestep
            else:
                self.success_hold = 0.0
            self.success_now = bool(
                self.success_hold >= self.config.success_hold_seconds
            )
            if self.success_now and not self.ever_success:
                self.ever_success = True
                current_info = self.info()
                self.success_snapshot = {
                    "time": float(self.data.time),
                    "grasped_body_id": (
                        current_info["grasped_body_id"]
                        if current_info["grasped_body_id"] is not None
                        else self.target_body_id
                    ),
                    "finger_contact_count": len(self.finger_contacts()),
                    "finger_aperture": current_info["finger_aperture"],
                    "bilateral_grasp": current_info["bilateral_grasp"],
                    "grasp_error": current_info["grasp_error"],
                    "lifted_fraction": current_info["lifted_fraction"],
                    "max_z": current_info["max_z"],
                    "success_hold": self.success_hold,
                }
            truncated = self.data.time >= self.config.episode_seconds
            if truncated:
                break

        success = self.ever_success
        info = self.info()
        reward = float(last_qualification) + 5.0 * float(success)
        return self.observation(), reward, success, truncated, info

    # -------------------------------------------------------------------------
    # 2. 对外观测与诊断接口
    # -------------------------------------------------------------------------

    def observation(self) -> dict:
        return {
            "time": float(self.data.time),
            "arm_qpos": self.data.qpos[self.arm_qpos_adr].copy(),
            "hand_position": self.hand_position.copy(),
            "target_body_id": self.target_body_id,
            "target_position": self.target_position(),
            "target_velocity": self.target_velocity(),
            "cable_positions": self.data.xpos[self.cable_ids].copy(),
            "grasped_body_id": None if self.grasp_state is None else self.grasp_state.body_id,
        }

    def info(self) -> dict:
        cable_positions = self.data.xpos[self.cable_ids]
        grasp_error = math.inf
        grasped_body = self.last_grasped_body_id
        if self.grasp_state is not None:
            grasped_body = self.grasp_state.body_id
        if grasped_body is not None:
            grasp_error = self._current_grasp_error()
            if self.grasp_state is None:
                grasp_error = float(np.linalg.norm(
                    self.data.xpos[grasped_body] - self.pad_center_position
                ))
        normal_forces = self.finger_normal_forces()
        intended_acceleration = (
            self._last_shape_acceleration
            + self._last_rigid_translation_acceleration
            + self._last_rigid_rotation_acceleration
        )
        rms = lambda values: float(np.sqrt(np.mean(np.square(values))))
        return {
            "trial": self.trial_index,
            "episode_seed": self.episode_seed,
            "scenario_name": self.config.scenario_name,
            "scenario_id": self.config.scenario_id,
            "scenario_split": self.config.scenario_split,
            "motion_mode": self.config.motion_mode,
            "motion_profile_version": self.config.motion_profile_version,
            "motion_regularity": self.config.motion_regularity,
            "motion_frequency_scale": self.config.motion_frequency_scale,
            "disturbance_strength": self.config.disturbance_strength,
            "shape_motion_scale": self.config.shape_motion_scale,
            "rigid_translation_scale": self.config.rigid_translation_scale,
            "rigid_rotation_scale": self.config.rigid_rotation_scale,
            "motion_profile_hash": self.motion_profile_hash,
            "cable_length_scale": self.config.cable_length_scale,
            "cable_density_scale": self.config.cable_density_scale,
            "cable_stiffness_scale": self.config.cable_stiffness_scale,
            "cable_damping_scale": self.config.cable_damping_scale,
            "cable_friction_scale": self.config.cable_friction_scale,
            "shape_acceleration_rms": rms(self._last_shape_acceleration),
            "rigid_translation_acceleration_rms": rms(
                self._last_rigid_translation_acceleration
            ),
            "rigid_rotation_acceleration_rms": rms(
                self._last_rigid_rotation_acceleration
            ),
            "intended_disturbance_acceleration_rms": rms(intended_acceleration),
            "boundary_acceleration_rms": rms(self._last_boundary_acceleration),
            "initial_cable_dx": float(self.initial_cable_translation[0]),
            "initial_cable_dy": float(self.initial_cable_translation[1]),
            "disturbance_phase": float(self.phase_offset),
            "disturbance_spatial_phase": float(self.spatial_phase),
            "target_body_id": self.target_body_id,
            "grasped_body_id": grasped_body,
            "bilateral_grasp": self.grasp_confirmed,
            "ever_bilateral_candidate": self.ever_bilateral_candidate,
            "ever_confirmed_grasp": self.ever_confirmed_grasp,
            "finger_aperture": float(np.sum(self.data.qpos[self.finger_qpos_adr])),
            "finger_contact_count": self._last_contact_count,
            "left_finger_normal_force": normal_forces[self.left_finger_id],
            "right_finger_normal_force": normal_forces[self.right_finger_id],
            "grasp_error": grasp_error,
            "cable_com": cable_positions.mean(axis=0).copy(),
            "max_z": float(cable_positions[:, 2].max()),
            "lifted_fraction": float(np.mean(cable_positions[:, 2] > 0.055)),
            "success_hold": self.success_hold,
            "success_now": self.success_now,
            "success": self.ever_success,
            "grasp_break_count": len(self.grasp_break_history),
            "active_open_break_count": sum(
                event["reason"] == "gripper_command_open"
                for event in self.grasp_break_history
            ),
            "physical_slip_break_count": sum(
                event["reason"] == "lost_physical_pad_contact"
                for event in self.grasp_break_history
            ),
            "active_open_after_confirmed_count": sum(
                event.get("causal_class") == "active_open"
                and bool(event["bilateral_confirmed"])
                for event in self.grasp_break_history
            ),
            "physical_slip_after_confirmed_count": sum(
                event.get("causal_class") == "physical_slip"
                and bool(event["bilateral_confirmed"])
                for event in self.grasp_break_history
            ),
            "open_during_contact_loss_count": sum(
                event.get("causal_class") == "open_during_contact_loss"
                for event in self.grasp_break_history
            ),
            "open_during_contact_loss_after_confirmed_count": sum(
                event.get("causal_class") == "open_during_contact_loss"
                and bool(event["bilateral_confirmed"])
                for event in self.grasp_break_history
            ),
            "last_grasp_break_reason": (
                None if self.last_grasp_break is None
                else self.last_grasp_break["reason"]
            ),
            "last_grasp_break_causal_class": (
                None if self.last_grasp_break is None
                else self.last_grasp_break["causal_class"]
            ),
        }

    # -------------------------------------------------------------------------
    # 3. 抓取状态与任务成功核心逻辑
    # -------------------------------------------------------------------------

    @property
    def grasp_confirmed(self) -> bool:
        return (
            self.grasp_state is not None
            and self.grasp_state.bilateral_confirmed
            and self.data.time - self.grasp_state.last_bilateral_time
            <= self.config.grasp_loss_seconds
        )

    def _update_physical_grasp_state(self, gripper_closed: bool) -> None:
        """只根据内指垫接触、法向力和开口更新抓取状态。"""
        if not gripper_closed:
            self._clear_grasp_with_reason("gripper_command_open")
            return

        candidate = self._physical_grasp_candidate()
        timestep = float(self.model.opt.timestep)
        if candidate is None:
            if self.grasp_state is None:
                return
            # 首次确认仍要求两侧法向力达到阈值；但确认后的高速动态接触中，
            # 求解器法向力可能短暂低于阈值，而真实碰撞仍同时存在于左右指垫。
            # 这种情况不是滑脱，不能把它累计成“无接触”并清除抓取状态。
            raw_pairs = self._finger_contact_pairs()
            raw_fingers = {finger for _, finger in raw_pairs}
            raw_bilateral = (
                self.left_finger_id in raw_fingers
                and self.right_finger_id in raw_fingers
            )
            if self.grasp_state.bilateral_confirmed and raw_bilateral:
                self.grasp_state.last_bilateral_time = float(self.data.time)
                self.grasp_state.lost_contact_time = 0.0
                return
            self.grasp_state.lost_contact_time += timestep
            if (
                not self.grasp_state.bilateral_confirmed
                or self.grasp_state.lost_contact_time >= self.config.grasp_loss_seconds
            ):
                self._clear_grasp_with_reason("lost_physical_pad_contact")
            return

        body_id = candidate
        self.ever_bilateral_candidate = True
        if self.grasp_state is None:
            self.grasp_state = GraspState(
                body_id=body_id,
                candidate_time=float(self.data.time),
                bilateral_confirmed=False,
                last_bilateral_time=float(self.data.time),
                lost_contact_time=0.0,
            )
            return

        state = self.grasp_state
        # 线缆可以在高摩擦指垫间发生有限滑动；只要双侧真实接触仍在，就更新当前段。
        state.body_id = body_id
        state.last_bilateral_time = float(self.data.time)
        state.lost_contact_time = 0.0
        if (
            not state.bilateral_confirmed
            and self.data.time - state.candidate_time >= self.config.grasp_confirm_seconds
        ):
            state.bilateral_confirmed = True
            self.last_grasped_body_id = state.body_id
            self.ever_confirmed_grasp = True

    def _physical_grasp_candidate(
        self,
    ) -> int | None:
        """查找当前被两侧内指垫真实夹紧的局部线段。"""
        aperture = float(np.sum(self.data.qpos[self.finger_qpos_adr]))
        if aperture > self.config.max_grasp_aperture:
            return None
        samples = self._pad_contact_samples()
        if not samples:
            return None

        contacted_bodies = {body_id for body_id, _, _ in samples}
        candidates: list[tuple[float, int]] = []
        for body_id in contacted_bodies:
            index = self.cable_index[body_id]
            neighborhood = set(
                self.cable_ids[max(0, index - 1):min(len(self.cable_ids), index + 2)]
            )
            left_force = sum(
                force for cable, finger, force in samples
                if cable in neighborhood and finger == self.left_finger_id
            )
            right_force = sum(
                force for cable, finger, force in samples
                if cable in neighborhood and finger == self.right_finger_id
            )
            if (
                left_force < self.config.min_pad_normal_force
                or right_force < self.config.min_pad_normal_force
            ):
                continue
            local_bodies = [cable for cable in contacted_bodies if cable in neighborhood]
            center_body = min(
                local_bodies,
                key=lambda cable: np.linalg.norm(
                    self.data.xpos[cable] - self.pad_center_position
                ),
            )
            distance = float(np.linalg.norm(
                self.data.xpos[center_body] - self.pad_center_position
            ))
            if distance <= self.config.max_pad_distance:
                candidates.append((distance, center_body))
        if not candidates:
            return None
        _, body_id = min(candidates)
        return body_id

    def _success_qualification(self, gripper_closed: bool) -> bool:
        """主要根据线缆是否被举起且仍随夹爪运动判断成功。"""

        if (
            not gripper_closed
            or self.grasp_state is None
            or not self.grasp_confirmed
        ):
            return False
        body_id = self.grasp_state.body_id
        cable_z = self.data.xpos[self.cable_ids, 2]
        lifted_fraction = float(np.mean(cable_z > 0.055))
        distance = float(np.linalg.norm(
            self.data.xpos[body_id] - self.pad_center_position
        ))
        return (
            self.data.xpos[body_id, 2] > 0.14
            and lifted_fraction >= 0.18
            and distance <= 1.5 * self.config.max_pad_distance
        )
    def _clear_grasp_with_reason(
        self,
        reason: str,
    ) -> None:
        """清除抓取状态，同时冻结清除前的诊断数据。"""
        if self.grasp_state is None:
            return
        state = self.grasp_state
        all_pairs = self._finger_contact_pairs()
        grasp_error = self._current_grasp_error()
        contacting_fingers = {finger for _, finger in all_pairs}
        unique_nodes = {body for body, _ in all_pairs}
        if reason == "gripper_command_open":
            # 若张爪前接触已经连续丢失，因果无法诚实地归为“策略主动放掉”；
            # 单独保留歧义类，避免系统性低估环境外力/物理滑脱的影响。
            causal_class = (
                "open_during_contact_loss"
                if state.lost_contact_time > 0.0
                else "active_open"
            )
        elif reason == "lost_physical_pad_contact":
            causal_class = "physical_slip"
        else:
            causal_class = "other"
        self.last_grasp_break = {
            "reason": reason,
            "causal_class": causal_class,
            "time": float(self.data.time),
            "bilateral_confirmed": bool(state.bilateral_confirmed),
            "raw_contact_count": len(all_pairs),
            "unique_contact_nodes": len(unique_nodes),
            "contacting_finger_count": len(contacting_fingers),
            "lost_bilateral_seconds": max(
                0.0, float(self.data.time - state.last_bilateral_time)
            ),
            "no_contact_time": float(state.lost_contact_time),
            "grasp_error": float(grasp_error),
            "finger_aperture": float(np.sum(self.data.qpos[self.finger_qpos_adr])),
            "gripper_ctrl": float(self.data.ctrl[7]),
            "ever_success": bool(self.ever_success),
            "success_hold": float(self.success_hold),
        }
        self.grasp_break_history.append(self.last_grasp_break.copy())
        self.grasp_state = None

    # -------------------------------------------------------------------------
    # 4. 线缆运动与环境扰动核心逻辑
    # -------------------------------------------------------------------------

    def _apply_cable_disturbance(self) -> None:
        """按场景组合静止、整体运动和局部形变三个可审计分量。"""

        elapsed_time = float(self.data.time)
        t = (
            self.phase_offset
            + self.config.motion_frequency_scale * elapsed_time
        )
        p = self.spatial_phase

        shape = np.zeros((len(self.cable_ids), 3))
        translation = np.zeros_like(shape)
        rotation = np.zeros_like(shape)

        if self.config.motion_mode in {"shape", "combined"}:
            shape = self._shape_acceleration(t, p)
        if self.config.motion_mode in {"rigid", "combined"}:
            translation, rotation = self._rigid_acceleration(elapsed_time, p)

        intended = shape + translation + rotation
        confined = (
            intended
            if self.config.motion_mode == "static"
            else self._confine_environment_acceleration(intended)
        )

        # 记录统计
        self._last_shape_acceleration[:] = shape
        self._last_rigid_translation_acceleration[:] = translation
        self._last_rigid_rotation_acceleration[:] = rotation
        self._last_boundary_acceleration[:] = confined - intended
        # 施加外力
        self.data.xfrc_applied[self.cable_ids, :3] += (
            self.cable_mass[:, None] * confined
        )

    def _shape_acceleration(self, t: float, p: float) -> np.ndarray:
        """返回去除净平动与净转动的局部形变加速度。"""

        if self.config.motion_regularity == "regular":
            # 所有时间频率均为 1.2 rad/s 的整数倍，整段轨迹严格周期重复。
            lateral = (
                2.30 * np.sin(self._lateral_space[0] - 2.4 * t + p)
                + 2.00 * np.sin(self._lateral_space[1] + 3.6 * t - 0.4 * p)
                + 1.60 * np.sin(self._lateral_space[2] - 4.8 * t + 0.7)
                + 1.20 * np.sin(self._lateral_space[3] + 6.0 * t + 0.35 * p)
            )
            longitudinal = (
                0.55 * np.sin(self._longitudinal_space[0] + 2.4 * t + 0.2 * p)
                + 0.45 * np.sin(self._longitudinal_space[1] - 4.8 * t)
            )
            vertical = (
                0.75 * np.sin(self._vertical_space[0] - 3.6 * t + 0.5 * p)
                + 0.55 * np.sin(self._vertical_space[1] + 6.0 * t)
            )
        elif self.config.motion_regularity == "quasiperiodic":
            # 频率固定，但互不整除，并且还有相位调制。
            lateral = (
                2.30 * np.sin(
                    self._lateral_space[0] - 3.4 * t + p
                    + 0.65 * np.sin(1.9 * t)
                )
                + 2.00 * np.sin(
                    self._lateral_space[1] + 2.8 * t - 0.4 * p
                    + 0.55 * np.sin(2.7 * t + p)
                )
                + 1.60 * np.sin(self._lateral_space[2] - 4.6 * t + 0.7)
                + 1.20 * np.sin(self._lateral_space[3] + 5.2 * t + 0.35 * p)
            )
            longitudinal = (
                0.55 * np.sin(self._longitudinal_space[0] + 2.3 * t + 0.2 * p)
                + 0.45 * np.sin(self._longitudinal_space[1] - 3.1 * t)
            )
            vertical = (
                0.75 * np.sin(self._vertical_space[0] - 2.5 * t + 0.5 * p)
                + 0.55 * np.sin(self._vertical_space[1] + 3.4 * t)
            )
        else:
            # 每个 episode 随机生成一组频率、方向和相位，但 episode 内仍然是平滑、连续、确定的正弦运动。
            frequency = self._stochastic_shape_frequency
            direction = self._stochastic_shape_direction
            phase = self._stochastic_shape_phase
            lateral = sum(
                coefficient * np.sin(
                    self._lateral_space[index]
                    + direction[index] * frequency[index] * t
                    + phase[index] + (0.25 * index - 0.3) * p
                )
                for index, coefficient in enumerate((2.30, 2.00, 1.60, 1.20))
            )
            longitudinal = sum(
                coefficient * np.sin(
                    self._longitudinal_space[index]
                    + direction[index + 4] * frequency[index + 4] * t
                    + phase[index + 4] + 0.2 * p
                )
                for index, coefficient in enumerate((0.55, 0.45))
            )
            vertical = sum(
                coefficient * np.sin(
                    self._vertical_space[index]
                    + direction[index + 6] * frequency[index + 6] * t
                    + phase[index + 6] + 0.4 * p
                )
                for index, coefficient in enumerate((0.75, 0.55))
            )

        # legacy_v1 减去每轴均值，使总和为0
        if self.config.motion_profile_version == "legacy_v1":
            lateral -= lateral.mean()
            longitudinal -= longitudinal.mean()
            vertical -= vertical.mean()
            return (
                self.config.disturbance_strength
                * self.config.shape_motion_scale
                * np.column_stack((
                    5.0 * longitudinal,
                    19.0 * lateral,
                    9.0 * vertical,
                ))
            )

        # 先去平移，再投影掉当前构型的刚体旋转子空间。仅去算术均值会残留
        # 净力矩，把“shape”场景污染成整体旋转；这里用质量内积做严格投影。
        raw = (
            self.config.disturbance_strength
            * self.config.shape_motion_scale
            * np.column_stack((
                5.0 * longitudinal,
                19.0 * lateral,
                9.0 * vertical,
            ))
        )
        raw -= np.average(raw, axis=0, weights=self.cable_mass)
        positions = self.data.xpos[self.cable_ids]
        center = np.average(positions, axis=0, weights=self.cable_mass)
        relative = positions - center
        inertia = (
            np.eye(3) * np.sum(self.cable_mass * np.sum(relative * relative, axis=1))
            - np.einsum(
                "n,ni,nj->ij", self.cable_mass, relative, relative,
                optimize=True,
            )
        )
        torque = np.sum(
            np.cross(relative, self.cable_mass[:, None] * raw), axis=0
        )
        angular_acceleration = np.linalg.pinv(inertia, rcond=1e-10) @ torque
        shape = raw - np.cross(
            np.broadcast_to(angular_acceleration, relative.shape), relative
        )
        # 浮点尾差再次去除，保证净力为数值零；不逐时刻归一化，从而保持旧环境
        # 的幅度随相位自然起伏，而所有新模式共享同一个强度定义。
        shape -= np.average(shape, axis=0, weights=self.cable_mass)
        return shape

    def _rigid_acceleration(
        self, time_value: float, p: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """用外力跟踪有界整体轨迹；不写 qpos、不创建隐藏约束。"""

        h = 1e-3
        desired = self._rigid_motion_target(time_value, p)
        before = self._rigid_motion_target(time_value - h, p)
        after = self._rigid_motion_target(time_value + h, p)
        desired_velocity = (after - before) / (2.0 * h)
        desired_acceleration = (after - 2.0 * desired + before) / (h * h)

        current_com, current_yaw = self._current_rigid_pose()
        # composite 的根 free joint 直接给出整体平动/转动速度；使用它保持该
        # 力场是当前物理状态的纯函数，且避免每个500 Hz子步查询40个body速度。
        current_velocity = self.data.qvel[
            self.cable_free_dadr:self.cable_free_dadr + 2
        ]
        current_yaw_rate = float(self.data.qvel[self.cable_free_dadr + 5])

        current_offset = current_com - self._rigid_reference_com_xy
        translation_xy = (
            desired_acceleration[:2]
            + 30.0 * (desired[:2] - current_offset)
            + 8.0 * (desired_velocity[:2] - current_velocity)
        )
        translation_norm = float(np.linalg.norm(translation_xy))
        if translation_norm > 8.0:
            translation_xy *= 8.0 / translation_norm
        translation_xy *= self.config.rigid_translation_scale
        translation = np.broadcast_to(
            np.r_[translation_xy, 0.0], (len(self.cable_ids), 3)
        ).copy()

        angular_acceleration = (
            desired_acceleration[2]
            + 24.0 * (desired[2] - current_yaw)
            + 7.0 * (desired_velocity[2] - current_yaw_rate)
        )
        angular_acceleration = float(np.clip(angular_acceleration, -18.0, 18.0))
        angular_acceleration *= self.config.rigid_rotation_scale
        angular_velocity = desired_velocity[2] * self.config.rigid_rotation_scale
        positions = self.data.xpos[self.cable_ids]
        center = np.average(positions, axis=0, weights=self.cable_mass)
        relative = positions - center
        alpha = np.array([0.0, 0.0, angular_acceleration])
        omega = np.array([0.0, 0.0, angular_velocity])
        rotation = (
            np.cross(np.broadcast_to(alpha, relative.shape), relative)
            + np.cross(
                np.broadcast_to(omega, relative.shape),
                np.cross(np.broadcast_to(omega, relative.shape), relative),
            )
        )
        rotation -= np.average(rotation, axis=0, weights=self.cable_mass)
        return translation, rotation

    def _rigid_motion_target(self, time_value: float, p: float) -> np.ndarray:
        """给出从静止平滑启动的有界 [x, y, yaw] 整体轨迹。"""

        t = float(time_value)
        frequency_scale = self.config.motion_frequency_scale
        phase = self.phase_offset
        envelope = 1.0 - math.exp(-((t / 0.35) ** 2))
        if self.config.motion_regularity == "regular":
            omega = 1.20 * frequency_scale
            signals = np.array([
                math.sin(omega * t + phase),
                math.sin(omega * t + phase + 0.5 * math.pi),
                math.sin(0.75 * omega * t + 0.5 * p),
            ])
        elif self.config.motion_regularity == "quasiperiodic":
            signals = np.array([
                0.70 * math.sin(0.85 * frequency_scale * t + phase)
                + 0.30 * math.sin(1.45 * frequency_scale * t - 0.2 * p),
                0.65 * math.sin(0.70 * frequency_scale * t - 0.3 * phase)
                + 0.35 * math.sin(1.30 * frequency_scale * t + p),
                0.70 * math.sin(0.60 * frequency_scale * t + 0.4 * p)
                + 0.30 * math.sin(1.10 * frequency_scale * t - phase),
            ])
        else:
            signals = np.sum(
                self._stochastic_rigid_weight
                * np.sin(
                    self._stochastic_rigid_frequency * frequency_scale * t
                    + self._stochastic_rigid_phase
                ),
                axis=1,
            )
            normalizer = np.maximum(
                np.sum(np.abs(self._stochastic_rigid_weight), axis=1), 1e-9
            )
            signals /= normalizer

        level_scale = self.config.disturbance_strength / 1.5
        return envelope * level_scale * np.array([
            0.045 * signals[0],
            0.065 * signals[1],
            0.45 * signals[2],
        ])

    def _current_rigid_pose(self) -> tuple[np.ndarray, float]:
        current_xy = self.data.xpos[self.cable_ids, :2]
        current_com = np.average(current_xy, axis=0, weights=self.cable_mass)
        reference = self._rigid_reference_xy - self._rigid_reference_com_xy
        current = current_xy - current_com
        root_mass = np.sqrt(self.cable_mass[:, None])
        covariance = (reference * root_mass).T @ (current * root_mass)
        left, _, right_t = np.linalg.svd(covariance)
        rotation = left @ right_t
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right_t
        yaw = math.atan2(float(rotation[0, 1]), float(rotation[0, 0]))
        return current_com, yaw

    def _confine_environment_acceleration(
        self, acceleration: np.ndarray
    ) -> np.ndarray:
        """只修正环境力场，降低线缆自行越过桌边的概率。

        这里不裁剪位置、不覆盖机器人动作，也不创建实体围栏。机械臂接触力由
        MuJoCo 在之后单独计算，所以仍能克服这段有限软力把线缆推出桌外。
        """
        result = acceleration.copy()
        position = self.data.xpos[self.cable_ids, :2]
        margin = self.config.boundary_margin
        soft_min = self.table_xy_min + margin
        soft_max = self.table_xy_max - margin

        # 以下位置、权重、反射和回正计算全部按Nx2数组批量完成。
        low_penetration = np.maximum(soft_min - position, 0.0)
        high_penetration = np.maximum(position - soft_max, 0.0)
        low_weight = np.clip(low_penetration / margin, 0.0, 1.0)
        high_weight = np.clip(high_penetration / margin, 0.0, 1.0)
        horizontal = result[:, :2]

        low_outward = (low_penetration > 0.0) & (horizontal < 0.0)
        high_outward = (high_penetration > 0.0) & (horizontal > 0.0)
        low_factor = 1.0 - 2.0 * low_weight
        high_factor = 1.0 - 2.0 * high_weight
        horizontal[low_outward] *= low_factor[low_outward]
        horizontal[high_outward] *= high_factor[high_outward]
        horizontal += self.config.boundary_stiffness * (
            low_penetration - high_penetration
        )

        # 通常只有少数节点进入缓冲带；仅对这些节点查询精确世界坐标速度。
        near_indices = np.flatnonzero(
            np.any((low_penetration > 0.0) | (high_penetration > 0.0), axis=1)
        )
        if near_indices.size:
            velocity = np.zeros((near_indices.size, 2))
            for slot, index in enumerate(near_indices):
                velocity[slot] = self.body_linear_velocity(self.cable_ids[index])[:2]
            low_outward_velocity = (
                (low_penetration[near_indices] > 0.0) & (velocity < 0.0)
            )
            high_outward_velocity = (
                (high_penetration[near_indices] > 0.0) & (velocity > 0.0)
            )
            damping_velocity = np.where(
                low_outward_velocity | high_outward_velocity, velocity, 0.0
            )
            horizontal[near_indices] -= self.config.boundary_damping * damping_velocity
        return result

    def _reset_stochastic_motion(self, *, randomize: bool) -> None:
        """为一个 episode 冻结平滑带限随机运动谱。

        随机数只在 reset 时生成；物理子步中只计算这些固定正弦基，因此同一
        seed 与场景对所有方法完全可复现，也不会引入与控制频率相关的白噪声。
        """

        if self.config.motion_regularity != "stochastic":
            self._stochastic_shape_frequency[:] = 1.0
            self._stochastic_shape_direction[:] = 1.0
            self._stochastic_shape_phase[:] = 0.0
            self._stochastic_rigid_frequency[:] = 1.0
            self._stochastic_rigid_phase[:] = 0.0
            self._stochastic_rigid_weight[:] = 0.5
            return

        generator = self.rng if randomize else np.random.default_rng(0)
        base_shape_frequency = np.array([
            3.4, 2.8, 4.6, 5.2, 2.3, 3.1, 2.5, 3.4,
        ])
        self._stochastic_shape_frequency[:] = (
            base_shape_frequency * generator.uniform(0.65, 1.35, 8)
        )
        self._stochastic_shape_direction[:] = generator.choice((-1.0, 1.0), 8)
        self._stochastic_shape_phase[:] = generator.uniform(0.0, 2.0 * math.pi, 8)

        base_rigid_frequency = np.array([0.75, 1.20, 1.85, 2.70])
        self._stochastic_rigid_frequency[:] = (
            base_rigid_frequency[None, :] * generator.uniform(0.70, 1.30, (3, 4))
        )
        self._stochastic_rigid_phase[:] = generator.uniform(
            0.0, 2.0 * math.pi, (3, 4)
        )
        weights = generator.normal(size=(3, 4))
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        self._stochastic_rigid_weight[:] = weights / np.maximum(norms, 1e-9)

    # -------------------------------------------------------------------------
    # 5. 常用状态查询接口
    # -------------------------------------------------------------------------

    @property
    def hand_position(self) -> np.ndarray:
        """返回实际两指夹持中心的世界坐标。"""
        rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        return self.data.xpos[self.hand_id] + rotation @ self.GRASP_CENTER_LOCAL

    @property
    def pad_center_position(self) -> np.ndarray:
        """返回两块主内指垫之间的几何中心，与IK控制点完全相同。"""
        return self.hand_position

    def body_linear_velocity(self, body_id: int) -> np.ndarray:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
        )
        return velocity[3:].copy()

    def target_position(self) -> np.ndarray:
        return self.data.xpos[self.target_body_id].copy()

    def target_velocity(self) -> np.ndarray:
        return self.body_linear_velocity(self.target_body_id)

    def finger_contacts(self, cable_body: int | None = None) -> list[int]:
        return [body for body, _ in self._finger_contact_pairs(cable_body)]

    def finger_normal_forces(self) -> dict[int, float]:
        """汇总当前线缆对左右内指垫的法向力，单位N。"""
        result = {self.left_finger_id: 0.0, self.right_finger_id: 0.0}
        for _, finger_id, normal_force in self._pad_contact_samples():
            result[finger_id] += normal_force
        return result

    # -------------------------------------------------------------------------
    # 6. 底层接触、几何与模型工具
    # -------------------------------------------------------------------------

    def _current_grasp_error(self) -> float:
        if self.grasp_state is None:
            return math.inf
        return float(np.linalg.norm(
            self.data.xpos[self.grasp_state.body_id] - self.pad_center_position
        ))
    def _pad_contact_samples(self) -> list[tuple[int, int, float]]:
        """读取真实内指垫接触及其法向力，不修改任何物理状态。"""
        samples: list[tuple[int, int, float]] = []
        contact_force = np.zeros(6)
        for contact_id, contact in enumerate(self.data.contact[:self.data.ncon]):
            pad_geom = None
            cable_body = None
            if contact.geom1 in self.pad_geom_ids:
                other_body = int(self.model.geom_bodyid[contact.geom2])
                if other_body in self.cable_set:
                    pad_geom, cable_body = int(contact.geom1), other_body
            elif contact.geom2 in self.pad_geom_ids:
                other_body = int(self.model.geom_bodyid[contact.geom1])
                if other_body in self.cable_set:
                    pad_geom, cable_body = int(contact.geom2), other_body
            if pad_geom is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_id, contact_force)
            finger_id = int(self.model.geom_bodyid[pad_geom])
            samples.append((
                cable_body,
                finger_id,
                max(0.0, float(contact_force[0])),
            ))
        return samples
    def _finger_contact_pairs(self, cable_body: int | None = None) -> list[tuple[int, int]]:
        """返回内指垫box与线缆的（线缆body，手指body）接触。"""
        return [
            (body_id, finger_id)
            for body_id, finger_id, _ in self._pad_contact_samples()
            if cable_body is None or body_id == cable_body
        ]
    @staticmethod
    def _load_model(config: EnvConfig) -> mujoco.MjModel:
        plugin = Path(mujoco.__file__).resolve().parent / "plugin" / "elasticity.dll"
        if plugin.exists():
            mujoco.mj_loadPluginLibrary(str(plugin))

        physical_scales = (
            config.cable_length_scale,
            config.cable_density_scale,
            config.cable_stiffness_scale,
            config.cable_damping_scale,
            config.cable_friction_scale,
        )
        if all(math.isclose(value, 1.0) for value in physical_scales):
            return mujoco.MjModel.from_xml_path(str(XML_PATH))

        # 修改线缆 OOD 属性
        spec = mujoco.MjSpec.from_file(str(XML_PATH))
        # 长度
        for body in spec.bodies:
            if body.name.startswith("cableB") and body.name != "cableB_first":
                body.pos[:] *= config.cable_length_scale
        # 长度，密度，摩擦
        for geom in spec.geoms:
            if not geom.name.startswith("cableG"):
                continue
            geom.fromto[3:] *= config.cable_length_scale
            geom.density *= config.cable_density_scale
            geom.friction[:] *= config.cable_friction_scale
        # 阻尼
        for joint in spec.joints:
            if joint.name.startswith("cableJ_") and joint.name != "cableJ_first":
                joint.damping[:] *= config.cable_damping_scale
        # 弹性刚度
        for plugin_spec in spec.plugins:
            plugin_config = dict(plugin_spec.config)
            if "bend" not in plugin_config or "twist" not in plugin_config:
                continue
            plugin_config["bend"] = str(
                float(plugin_config["bend"]) * config.cable_stiffness_scale
            )
            plugin_config["twist"] = str(
                float(plugin_config["twist"]) * config.cable_stiffness_scale
            )
            plugin_spec.config = plugin_config
        return spec.compile()

    def _cable_bodies(self) -> list[int]:
        result: list[int] = []
        for body_id in range(1, self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or ""
            if name.startswith("cable"):
                result.append(body_id)
        if not result:
            raise RuntimeError("No cable bodies were generated by the composite")
        return result

    def _find_cable_free_qpos_address(self) -> int:
        first_body = self.cable_ids[0]
        for joint_id in range(self.model.njnt):
            if (self.model.jnt_bodyid[joint_id] == first_body and
                    self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE):
                return int(self.model.jnt_qposadr[joint_id])
        raise RuntimeError("Cable root free joint was not found")


def rotation_to_quat(rotation: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
    return quat


def quat_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """计算世界坐标系中的小角度姿态误差。"""
    inverse = np.zeros(4)
    mujoco.mju_negQuat(inverse, current)
    delta = np.zeros(4)
    mujoco.mju_mulQuat(delta, desired, inverse)
    if delta[0] < 0:
        delta *= -1
    return 2.0 * delta[1:4]


def point_jacobian(model: mujoco.MjModel, data: mujoco.MjData,
                   body_id: int, local_offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jac_pos, jac_rot, body_id)
    offset_world = data.xmat[body_id].reshape(3, 3) @ local_offset
    jac_pos += np.cross(jac_rot.T, offset_world).T
    return jac_pos, jac_rot
