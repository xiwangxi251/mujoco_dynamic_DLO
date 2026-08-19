"""环境端
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path

from runtime_config import configure_mujoco_runtime

configure_mujoco_runtime()

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "panda_cable_grasp.xml"
PANDA_XML_PATH = ROOT / "models" / "panda.xml"
MENAGERIE_ENV_VAR = "MUJOCO_MENAGERIE_PATH"


def resolve_menagerie_panda_dir() -> Path:
    """Locate official Panda mesh assets without assuming a workstation path."""

    configured = os.environ.get(MENAGERIE_ENV_VAR)
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend((
        ROOT / "mujoco_menagerie",
        ROOT / "third_party" / "mujoco_menagerie",
        ROOT.parent / "mujoco_menagerie",
    ))

    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        panda_dir = (
            candidate
            if candidate.name == "franka_emika_panda"
            else candidate / "franka_emika_panda"
        )
        checked.append(panda_dir)
        if (panda_dir / "assets").is_dir():
            return panda_dir

    locations = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "MuJoCo Menagerie Panda assets were not found. Clone "
        "https://github.com/google-deepmind/mujoco_menagerie.git and set "
        f"{MENAGERIE_ENV_VAR} to its root directory. Checked:\n  - {locations}"
    )


@lru_cache(maxsize=4)
def _panda_assets(panda_dir_text: str) -> dict[str, bytes]:
    assets_dir = Path(panda_dir_text) / "assets"
    return {
        path.relative_to(assets_dir).as_posix(): path.read_bytes()
        for path in assets_dir.rglob("*")
        if path.is_file()
    }

RIGID_MOTION_START_TIME = 0.80
RIGID_MOTION_START_Y = -0.70
RIGID_MOTION_TRAVEL = 1.40
RIGID_MOTION_ROTATION = math.radians(24.0)
RIGID_MOTION_PROFILES = {
    "rigid_level1_single_pass_v2",
    "rigid_level2_single_pass_v2",
}
RIGID_MOTION_L2_CONTROL = RIGID_MOTION_TRAVEL * np.array([
    [0.00, 0.00],
    [0.16, 0.30],
    [-0.16, 0.70],
    [0.00, 1.00],
])


def _cubic_bezier(control: np.ndarray, u: float | np.ndarray) -> np.ndarray:
    """Evaluate a planar cubic Bezier curve for scalar or vector ``u``."""

    values = np.asarray(u, dtype=float)
    one_minus = 1.0 - values
    return (
        one_minus[..., None] ** 3 * control[0]
        + 3.0 * one_minus[..., None] ** 2 * values[..., None] * control[1]
        + 3.0 * one_minus[..., None] * values[..., None] ** 2 * control[2]
        + values[..., None] ** 3 * control[3]
    )


_L2_ARC_SAMPLES = _cubic_bezier(
    RIGID_MOTION_L2_CONTROL, np.linspace(0.0, 1.0, 1001)
)
RIGID_MOTION_L2_ARC_LENGTH = float(np.linalg.norm(
    np.diff(_L2_ARC_SAMPLES, axis=0), axis=1
).sum())


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
    # legacy_v1 / factorized_v1/v2 / rigid_level{1,2}_single_pass_v2
    motion_profile_version: str = "legacy_v1"
    motion_regularity: str = "quasiperiodic"  # regular / quasiperiodic / stochastic
    motion_frequency_scale: float = 1.0
    shape_motion_scale: float = 1.0
    rigid_translation_scale: float = 1.0
    rigid_rotation_scale: float = 1.0
    rigid_motion_nominal_speed: float = 0.25 # L1/L2标称平均速度，单位m/s
    # L1/L2只在线缆质量中心实际越过这条世界坐标Y线后结束。
    rigid_motion_exit_y: float = 0.70
    # 整体运动按物体的实际路径进度推进，不追赶绝对时间目标。位置增益只修正
    # 横向轨迹误差，纵向始终使用标称速度，避免受阻后补路程式加速。
    rigid_path_position_gain: float = 20.0
    rigid_velocity_gain: float = 32.0
    rigid_translation_max_acceleration: float = 8.0
    rigid_shape_stiffness: float = 1000.0    # rigid中保持初始平面构型，单位1/s²
    rigid_shape_damping: float = 68.0        # rigid相对运动阻尼，单位1/s
    rigid_shape_max_acceleration: float = 45.0

    # 线缆 OOD 参数相对 XML 标称值缩放（改变长度、质量、弹性、阻尼和摩擦）
    cable_length_scale: float = 1.0
    cable_density_scale: float = 1.0
    cable_stiffness_scale: float = 1.0
    cable_damping_scale: float = 1.0
    cable_friction_scale: float = 1.0
    table_half_size: tuple[float, float] = (1.20, 1.20)

    # 抓取判断参数
    success_hold_seconds: float = 0.80  # 成功条件必须连续保持的时间
    grasp_confirm_seconds: float = 0.06 # 双侧内指垫接触保持多久才确认抓取
    grasp_candidate_gap_seconds: float = 0.02  # 确认前容忍求解器短暂接触/力波动
    grasp_contact_index_radius: int = 2  # 弯曲线缆双侧接触允许跨越的离散节点数
    grasp_loss_seconds: float = 0.35    # 双指接触短暂中断的容忍时间
    max_grasp_aperture: float = 0.034  # 28 mm线缆被真正夹紧时允许的最大开口
    max_pad_distance: float = 0.055    # 接触线段中心到指垫中心的最大距离
    min_pad_normal_force: float = 0.20 # 每侧内指垫所需最小法向力，单位N

    # 仿真和夹爪参数
    frame_skip: int = 10                # 一个50 Hz动作对应10个500 Hz物理步
    gripper_force_scale: float = 5.0    # 提高闭爪位置伺服刚度；执行器最大力范围保持不变
    pad_friction: tuple[float, float, float] = (4.0, 0.10, 0.05)

    # 固定全局相机。相机位于桌面一侧的斜上方，覆盖整条线缆及L1/L2运动范围；
    # MuJoCo相机沿自身-Z轴观察，外参定义在世界坐标系中。
    camera_observation_enabled: bool = True
    global_camera_name: str = "global_camera"
    global_camera_width: int = 320
    global_camera_height: int = 240
    global_camera_fovy: float = 45.0
    global_camera_pos: tuple[float, float, float] = (
        0.50, -1.42658348, 2.01596494,
    )
    global_camera_quat: tuple[float, float, float, float] = (
        0.9537169497, 0.3007058028, 0.0, 0.0,
    )

    # 环境统一限制机器人能力，脚本、RL与后续VLA都不能绕过；数值为Panda官方上限的80%。
    robot_motion_limit_profile: str = "panda_eval_80_v3"
    arm_joint_velocity_limits: tuple[float, ...] = (
        1.74, 1.74, 1.74, 1.74, 2.09, 2.09, 2.09,
    )
    arm_joint_acceleration_limits: tuple[float, ...] = (
        12.0, 6.0, 8.0, 10.0, 12.0, 16.0, 16.0,
    )
    hand_linear_velocity_limit: float = 1.0
    hand_angular_velocity_limit: float = 2.0
    gripper_finger_velocity_limit: float = 0.20
    arm_position_tracking_error_limit: float = 0.03
    low_level_velocity_guard_fraction: float = 0.65

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
        if self.motion_profile_version not in {
            "legacy_v1", "factorized_v1", "factorized_v2",
            "rigid_level1_single_pass_v2", "rigid_level2_single_pass_v2",
        }:
            raise ValueError(
                "unsupported motion_profile_version: "
                f"{self.motion_profile_version!r}"
            )
        uses_rigid_trajectory = self.motion_profile_version in RIGID_MOTION_PROFILES
        if self.motion_mode in {"rigid", "combined"} and not uses_rigid_trajectory:
            raise ValueError(
                "rigid and combined motion must explicitly select Level-1 or Level-2"
            )
        if uses_rigid_trajectory and self.motion_mode not in {"rigid", "combined"}:
            raise ValueError(
                "Level-1/Level-2 profiles require rigid or combined motion"
            )
        nonnegative = (
            "disturbance_strength",
            "motion_frequency_scale",
            "shape_motion_scale",
            "rigid_translation_scale",
            "rigid_rotation_scale",
            "rigid_motion_nominal_speed",
            "rigid_path_position_gain",
            "rigid_velocity_gain",
            "rigid_shape_stiffness",
            "rigid_shape_damping",
            "rigid_shape_max_acceleration",
        )
        positive = (
            "episode_seconds",
            "cable_length_scale",
            "cable_density_scale",
            "cable_stiffness_scale",
            "cable_damping_scale",
            "cable_friction_scale",
            "rigid_translation_max_acceleration",
        )
        for name in (*nonnegative, *positive):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in nonnegative and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            if name in positive and value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.rigid_motion_exit_y):
            raise ValueError("rigid_motion_exit_y must be finite")
        if not (
            RIGID_MOTION_START_Y
            < self.rigid_motion_exit_y
            <= RIGID_MOTION_START_Y + RIGID_MOTION_TRAVEL
        ):
            raise ValueError(
                "rigid_motion_exit_y must be greater than the L1/L2 start "
                "and no greater than the nominal endpoint"
            )
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        if not isinstance(self.camera_observation_enabled, bool):
            raise ValueError("camera_observation_enabled must be boolean")
        if not self.global_camera_name.strip():
            raise ValueError("global_camera_name must be non-empty")
        for name in ("global_camera_width", "global_camera_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.global_camera_fovy < 180.0:
            raise ValueError("global_camera_fovy must be in (0, 180) degrees")
        camera_pos = np.asarray(self.global_camera_pos, dtype=float)
        camera_quat = np.asarray(self.global_camera_quat, dtype=float)
        if camera_pos.shape != (3,) or not np.all(np.isfinite(camera_pos)):
            raise ValueError("global_camera_pos must contain 3 finite values")
        if (
            camera_quat.shape != (4,)
            or not np.all(np.isfinite(camera_quat))
            or not math.isclose(float(np.linalg.norm(camera_quat)), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("global_camera_quat must be a normalized quaternion")
        if self.grasp_candidate_gap_seconds < 0.0:
            raise ValueError("grasp_candidate_gap_seconds must be non-negative")
        if (
            isinstance(self.grasp_contact_index_radius, bool)
            or not isinstance(self.grasp_contact_index_radius, int)
            or self.grasp_contact_index_radius < 0
        ):
            raise ValueError(
                "grasp_contact_index_radius must be a non-negative integer"
            )
        for name in (
            "arm_joint_velocity_limits", "arm_joint_acceleration_limits",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (7,):
                raise ValueError(f"{name} must contain exactly 7 values")
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"{name} values must be finite and positive")
        table_half_size = np.asarray(self.table_half_size, dtype=float)
        if (
            table_half_size.shape != (2,)
            or not np.all(np.isfinite(table_half_size))
            or np.any(table_half_size <= 0.0)
        ):
            raise ValueError(
                "table_half_size must contain exactly 2 finite positive values"
            )
        for name in (
            "hand_linear_velocity_limit", "hand_angular_velocity_limit",
            "gripper_finger_velocity_limit", "arm_position_tracking_error_limit",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.low_level_velocity_guard_fraction <= 1.0:
            raise ValueError(
                "low_level_velocity_guard_fraction must be in (0, 1]"
            )


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
        self.model.vis.global_.offwidth = max(
            int(self.model.vis.global_.offwidth), self.config.global_camera_width
        )
        self.model.vis.global_.offheight = max(
            int(self.model.vis.global_.offheight), self.config.global_camera_height
        )

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
        self.global_camera_id = id_of(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.global_camera_name
        )
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
        self.cable_ball_joint_ids = np.array([
            joint_id
            for joint_id in range(self.model.njnt)
            if (
                int(self.model.jnt_bodyid[joint_id]) in self.cable_set
                and self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL
            )
        ], dtype=int)
        self.cable_ball_qadr = self.model.jnt_qposadr[
            self.cable_ball_joint_ids
        ].copy()
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

        self._arm_velocity_limits = np.asarray(
            self.config.arm_joint_velocity_limits, dtype=float
        )
        self._arm_acceleration_limits = np.asarray(
            self.config.arm_joint_acceleration_limits, dtype=float
        )
        self._hand_jacp = np.zeros((3, self.model.nv))
        self._hand_jacr = np.zeros((3, self.model.nv))
        gripper_bias = -float(self.model.actuator_biasprm[7, 1])
        self._gripper_ctrl_to_finger_position = (
            float(self.model.actuator_gainprm[7, 0]) / gripper_bias
            if gripper_bias > 0.0 else 0.0
        )
        if self._gripper_ctrl_to_finger_position <= 0.0:
            raise RuntimeError("Unable to derive gripper control-to-position scale")

        # 任务状态参数全部初始化
        self.trial_index = 0

        self.phase_offset = 0.0
        self.spatial_phase = 0.0
        self._stochastic_shape_frequency = np.ones(8)
        self._stochastic_shape_direction = np.ones(8)
        self._stochastic_shape_phase = np.zeros(8)
        node_count = len(self.cable_ids)
        self._last_shape_acceleration = np.zeros((node_count, 3))
        self._last_rigid_translation_acceleration = np.zeros((node_count, 3))
        self._last_rigid_rotation_acceleration = np.zeros((node_count, 3))
        self._last_rigid_shape_hold_acceleration = np.zeros((node_count, 3))
        self.motion_profile_hash = ""
        self._rigid_reference_xy = np.zeros((node_count, 2))
        self._rigid_reference_com_xy = np.zeros(2)
        self._rigid_initial_shape_family = "none"
        self._rigid_initial_tangent_angles = np.zeros(node_count - 1)
        self._rigid_motion_rotation_sign = 1.0

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
        self.rigid_motion_released = False
        self.last_termination_reason: str | None = None
        self.episode_seed: int | None = None
        self.initial_cable_translation = np.zeros(2)
        self._last_contact_count = 0
        self._previous_arm_command_velocity = np.zeros(7)
        self._last_requested_action = self.ready_ctrl.copy()
        self._last_applied_action = self.ready_ctrl.copy()
        self._last_requested_arm_velocity = np.zeros(7)
        self._last_applied_arm_velocity = np.zeros(7)
        self._last_commanded_hand_velocity = np.zeros(6)
        self._last_motion_limit_flags = {
            "acceleration": False,
            "joint_velocity": False,
            "cartesian_velocity": False,
            "gripper_velocity": False,
        }
        self._motion_limit_steps = 0
        self._motion_limit_active_steps = 0
        self._acceleration_limit_steps = 0
        self._joint_velocity_limit_steps = 0
        self._cartesian_velocity_limit_steps = 0
        self._gripper_velocity_limit_steps = 0
        self._physics_steps = 0
        self._velocity_guard_steps = 0
        self._actual_velocity_exceedance_steps = 0
        self._max_abs_actual_arm_velocity = np.zeros(7)
        self._max_actual_hand_linear_speed = 0.0
        self._max_actual_hand_angular_speed = 0.0
        self._camera_renderer: mujoco.Renderer | None = None
        self._camera_frame_time: float | None = None
        self._camera_frame: np.ndarray | None = None
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

        uses_rigid_motion = self.config.motion_profile_version in RIGID_MOTION_PROFILES
        uses_curved_initial_shape = (
            self.config.motion_profile_version == "factorized_v2"
            or uses_rigid_motion
        )
        base_cable_xy = self.data.xpos[self.cable_ids, :2].copy()
        base_com_xy = np.average(base_cable_xy, axis=0, weights=self.cable_mass)
        if randomize:
            # 每轮改变线缆初始位置、运动相位和目标段；固定seed时完整场景可复现。
            dx = float(self.rng.uniform(-0.10, 0.10))
            dy = float(self.rng.uniform(-0.13, 0.13))
            self.phase_offset = self.rng.uniform(0.0, 2.0 * math.pi)
            self.spatial_phase = self.rng.uniform(0.0, 2.0 * math.pi)
            lo = len(self.cable_ids) // 4
            hi = len(self.cable_ids) - lo
            target_index = int(self.rng.integers(lo, hi))
            self.target_body_id = self.cable_ids[target_index]
        else:
            dx = 0.0
            dy = 0.0
            self.phase_offset = 0.0
            self.spatial_phase = 0.0
            self.target_body_id = self.cable_ids[len(self.cable_ids) // 2]

        if uses_curved_initial_shape:
            # 所有新实验场景共享随机弯曲初始分布；相同seed可配对比较。
            # L1/L2另外固定从同一Y入口开始，并预检完整SE(2)扫掠范围。
            if uses_rigid_motion:
                dy = RIGID_MOTION_START_Y - float(base_com_xy[1])
            self.initial_cable_translation[:] = [dx, dy]
            self._set_curved_initial_shape(
                desired_com_xy=base_com_xy + np.array([dx, dy]),
                randomize=randomize,
                check_rigid_motion_sweep=uses_rigid_motion,
            )
        else:
            # 长线缆OOD的初始端点仍放在桌面内，避免长度变化被初始坠桌混淆。
            lower = self.table_xy_min + self.cable_radius - base_cable_xy.min(axis=0)
            upper = self.table_xy_max - self.cable_radius - base_cable_xy.max(axis=0)
            if np.any(lower > upper):
                raise ValueError(
                    "configured cable does not fit on the table at reset: "
                    f"length_scale={self.config.cable_length_scale}"
                )
            dx = float(np.clip(dx, lower[0], upper[0]))
            dy = float(np.clip(dy, lower[1], upper[1]))
            self.initial_cable_translation[:] = [dx, dy]
            self.data.qpos[
                self.cable_free_qadr:self.cable_free_qadr + 3
            ] += [dx, dy, 0.0]
            self._rigid_initial_shape_family = "none"
            self._rigid_initial_tangent_angles[:] = 0.0
            self._rigid_motion_rotation_sign = 1.0

        self._reset_stochastic_motion(randomize=randomize)
        profile_header = (
            f"{self.config.motion_profile_version}|{self.config.motion_mode}|"
            f"{self.config.motion_regularity}|"
            f"{self.config.disturbance_strength:.17g}|"
            f"{self.config.motion_frequency_scale:.17g}|"
            f"{self.phase_offset:.17g}|{self.spatial_phase:.17g}"
        ).encode("ascii")
        mujoco.mj_forward(self.model, self.data)
        self._rigid_reference_xy[:] = self.data.xpos[self.cable_ids, :2]
        self._rigid_reference_com_xy[:] = np.average(
            self._rigid_reference_xy, axis=0, weights=self.cable_mass
        )
        profile_bytes = b"".join((
            profile_header,
            self._stochastic_shape_frequency.tobytes(),
            self._stochastic_shape_direction.tobytes(),
            self._stochastic_shape_phase.tobytes(),
            self._rigid_reference_xy.tobytes(),
            self._rigid_initial_shape_family.encode("ascii"),
            np.asarray([self._rigid_motion_rotation_sign]).tobytes(),
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
        self.rigid_motion_released = False
        self.last_termination_reason = None
        self._last_contact_count = 0
        self._last_shape_acceleration[:] = 0.0
        self._last_rigid_translation_acceleration[:] = 0.0
        self._last_rigid_rotation_acceleration[:] = 0.0
        self._last_rigid_shape_hold_acceleration[:] = 0.0
        self._previous_arm_command_velocity[:] = 0.0
        self._last_requested_action[:] = self.ready_ctrl
        self._last_applied_action[:] = self.ready_ctrl
        self._last_requested_arm_velocity[:] = 0.0
        self._last_applied_arm_velocity[:] = 0.0
        self._last_commanded_hand_velocity[:] = 0.0
        for name in self._last_motion_limit_flags:
            self._last_motion_limit_flags[name] = False
        self._motion_limit_steps = 0
        self._motion_limit_active_steps = 0
        self._acceleration_limit_steps = 0
        self._joint_velocity_limit_steps = 0
        self._cartesian_velocity_limit_steps = 0
        self._gripper_velocity_limit_steps = 0
        self._physics_steps = 0
        self._velocity_guard_steps = 0
        self._actual_velocity_exceedance_steps = 0
        self._max_abs_actual_arm_velocity[:] = 0.0
        self._max_actual_hand_linear_speed = 0.0
        self._max_actual_hand_angular_speed = 0.0
        self._camera_frame_time = None
        self._camera_frame = None
        self.trial_index += 1
        return self.observation(), self.info()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """一个50 Hz控制动作内部执行十个500 Hz物理子步。"""
        # action裁剪。
        action = np.asarray(action, dtype=float)
        if action.shape != (8,):
            raise ValueError(f"Expected action shape (8,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action values must be finite")
        clipped_action = np.clip(
            action,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        applied_action = self._limit_robot_action(clipped_action)

        last_qualification = False
        truncated = False
        self.last_termination_reason = None
        gripper_closed = bool(applied_action[7] < 100.0)

        for _ in range(max(1, self.config.frame_skip)):
            # 扰动线缆
            self.data.xfrc_applied[:] = 0.0
            self._apply_cable_disturbance()
            # 执行动作
            guarded_action, velocity_guard_active = (
                self._velocity_guarded_action(applied_action)
            )
            self.data.ctrl[:] = guarded_action
            mujoco.mj_step(self.model, self.data)
            self._record_actual_robot_velocity(velocity_guard_active)

            # 更新抓取候选
            self._last_contact_count = len(self._finger_contact_pairs())
            self._update_physical_grasp_state(gripper_closed)
            self._update_rigid_motion_release_state()

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
            rigid_motion_boundary_crossed = (
                self.rigid_motion_finished
                and not self.rigid_motion_released
            )
            if rigid_motion_boundary_crossed:
                self.last_termination_reason = "rigid_motion_boundary_crossed"
            elif self.data.time >= self.config.episode_seconds:
                self.last_termination_reason = "episode_time_limit"
            truncated = self.last_termination_reason is not None
            if truncated:
                break

        success = self.ever_success
        info = self.info()
        reward = float(last_qualification) + 5.0 * float(success)
        return self.observation(), reward, success, truncated, info

    def _limit_robot_action(self, requested_action: np.ndarray) -> np.ndarray:
        """在进入执行器前，按统一的50 Hz周期限制机器人运动命令。"""

        control_dt = float(
            self.model.opt.timestep * max(1, self.config.frame_skip)
        )
        previous_position_target = self._last_applied_action[:7]
        requested_velocity = (
            requested_action[:7] - previous_position_target
        ) / control_dt

        # Keep the requested multi-joint direction intact while limiting its
        # change.  Independent component clipping can rotate a resolved-rate
        # IK command substantially at phase changes (for example, turning a
        # requested descent into an upward end-effector transient).
        velocity_delta = (
            requested_velocity - self._previous_arm_command_velocity
        )
        allowed_delta = self._arm_acceleration_limits * control_dt
        acceleration_scale = min(
            1.0,
            float(np.min(
                allowed_delta / np.maximum(np.abs(velocity_delta), 1e-12)
            )),
        )
        acceleration_limited_velocity = (
            self._previous_arm_command_velocity
            + acceleration_scale * velocity_delta
        )
        acceleration_limited = not np.allclose(
            acceleration_limited_velocity, requested_velocity,
            rtol=0.0, atol=1e-12,
        )

        joint_limited_velocity = np.clip(
            acceleration_limited_velocity,
            -self._arm_velocity_limits,
            self._arm_velocity_limits,
        )
        joint_velocity_limited = not np.allclose(
            joint_limited_velocity, acceleration_limited_velocity,
            rtol=0.0, atol=1e-12,
        )

        joint_range = self.model.jnt_range[self.arm_joint_ids]
        position_target = np.clip(
            previous_position_target + joint_limited_velocity * control_dt,
            joint_range[:, 0], joint_range[:, 1],
        )
        bounded_velocity = (
            position_target - previous_position_target
        ) / control_dt

        jacp, jacr = self._hand_jacobian()
        linear_velocity = jacp[:, self.arm_dof_adr] @ bounded_velocity
        angular_velocity = jacr[:, self.arm_dof_adr] @ bounded_velocity
        linear_speed = float(np.linalg.norm(linear_velocity))
        angular_speed = float(np.linalg.norm(angular_velocity))
        cartesian_scale = min(
            1.0,
            self.config.hand_linear_velocity_limit / max(linear_speed, 1e-12),
            self.config.hand_angular_velocity_limit / max(angular_speed, 1e-12),
        )
        cartesian_velocity_limited = cartesian_scale < 1.0 - 1e-12
        applied_velocity = bounded_velocity * cartesian_scale
        position_target = (
            previous_position_target + applied_velocity * control_dt
        )

        max_gripper_delta = (
            self.config.gripper_finger_velocity_limit
            * control_dt
            / self._gripper_ctrl_to_finger_position
        )
        previous_gripper_command = float(self._last_applied_action[7])
        gripper_command = float(np.clip(
            requested_action[7],
            previous_gripper_command - max_gripper_delta,
            previous_gripper_command + max_gripper_delta,
        ))
        gripper_velocity_limited = not math.isclose(
            gripper_command, float(requested_action[7]),
            rel_tol=0.0, abs_tol=1e-12,
        )

        applied_action = requested_action.copy()
        applied_action[:7] = position_target
        applied_action[7] = gripper_command

        commanded_linear_velocity = (
            jacp[:, self.arm_dof_adr] @ applied_velocity
        )
        commanded_angular_velocity = (
            jacr[:, self.arm_dof_adr] @ applied_velocity
        )
        flags = {
            "acceleration": acceleration_limited,
            "joint_velocity": joint_velocity_limited,
            "cartesian_velocity": cartesian_velocity_limited,
            "gripper_velocity": gripper_velocity_limited,
        }
        self._previous_arm_command_velocity[:] = applied_velocity
        self._last_requested_action[:] = requested_action
        self._last_applied_action[:] = applied_action
        self._last_requested_arm_velocity[:] = requested_velocity
        self._last_applied_arm_velocity[:] = applied_velocity
        self._last_commanded_hand_velocity[:3] = commanded_linear_velocity
        self._last_commanded_hand_velocity[3:] = commanded_angular_velocity
        self._last_motion_limit_flags = flags
        self._motion_limit_steps += 1
        self._motion_limit_active_steps += int(any(flags.values()))
        self._acceleration_limit_steps += int(acceleration_limited)
        self._joint_velocity_limit_steps += int(joint_velocity_limited)
        self._cartesian_velocity_limit_steps += int(cartesian_velocity_limited)
        self._gripper_velocity_limit_steps += int(gripper_velocity_limited)
        return applied_action

    def _hand_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_jac(
            self.model, self.data,
            self._hand_jacp, self._hand_jacr,
            self.hand_position, self.hand_id,
        )
        return self._hand_jacp, self._hand_jacr

    def _velocity_guarded_action(
        self, applied_action: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """关节接近速度上限后停止同向驱动。

        这里只改变执行器目标，由原有PD阻尼和力矩上限制动；不直接裁剪
        ``data.qvel``或其他物理状态。
        """

        guarded_action = applied_action.copy()
        current_qpos = self.data.qpos[self.arm_qpos_adr]
        current_qvel = self.data.qvel[self.arm_dof_adr]
        guarded_action[:7] = np.clip(
            guarded_action[:7],
            current_qpos - self.config.arm_position_tracking_error_limit,
            current_qpos + self.config.arm_position_tracking_error_limit,
        )
        guard_limits = (
            self.config.low_level_velocity_guard_fraction
            * self._arm_velocity_limits
        )
        positive = (
            current_qvel >= guard_limits
        ) & (guarded_action[:7] > current_qpos)
        negative = (
            current_qvel <= -guard_limits
        ) & (guarded_action[:7] < current_qpos)
        active = positive | negative
        guarded_action[:7][active] = current_qpos[active]

        # Joint-wise limits do not guarantee a Cartesian angular-speed limit:
        # several sub-limit joint velocities can add constructively through the
        # Jacobian.  Apply a whole-arm damping command before either Cartesian
        # speed reaches its limit.  This changes only the actuator target; the
        # simulator remains responsible for the physical deceleration.
        jacp, jacr = self._hand_jacobian()
        hand_linear_speed = float(np.linalg.norm(jacp @ self.data.qvel))
        hand_angular_speed = float(np.linalg.norm(jacr @ self.data.qvel))
        cartesian_guard_active = bool(
            hand_linear_speed
            >= self.config.low_level_velocity_guard_fraction
            * self.config.hand_linear_velocity_limit
            or hand_angular_speed
            >= self.config.low_level_velocity_guard_fraction
            * self.config.hand_angular_velocity_limit
        )
        if cartesian_guard_active:
            guarded_action[:7] = current_qpos
        return guarded_action, bool(np.any(active) or cartesian_guard_active)

    def _record_actual_robot_velocity(self, velocity_guard_active: bool) -> None:
        actual_arm_velocity = self.data.qvel[self.arm_dof_adr]
        jacp, jacr = self._hand_jacobian()
        actual_linear_speed = float(np.linalg.norm(jacp @ self.data.qvel))
        actual_angular_speed = float(np.linalg.norm(jacr @ self.data.qvel))
        self._physics_steps += 1
        self._velocity_guard_steps += int(velocity_guard_active)
        self._actual_velocity_exceedance_steps += int(np.any(
            np.abs(actual_arm_velocity) > 1.05 * self._arm_velocity_limits
        ))
        self._max_abs_actual_arm_velocity[:] = np.maximum(
            self._max_abs_actual_arm_velocity,
            np.abs(actual_arm_velocity),
        )
        self._max_actual_hand_linear_speed = max(
            self._max_actual_hand_linear_speed, actual_linear_speed
        )
        self._max_actual_hand_angular_speed = max(
            self._max_actual_hand_angular_speed, actual_angular_speed
        )

    # -------------------------------------------------------------------------
    # 2. 对外观测与诊断接口
    # -------------------------------------------------------------------------

    def camera_rgb(self) -> np.ndarray:
        """返回固定全局RGB相机图像，形状为(H, W, 3)、类型为uint8。"""
        if not self.config.camera_observation_enabled:
            raise RuntimeError("camera observation is disabled for this environment")
        current_time = float(self.data.time)
        if self._camera_frame is None or self._camera_frame_time != current_time:
            if self._camera_renderer is None:
                self._camera_renderer = mujoco.Renderer(
                    self.model,
                    height=self.config.global_camera_height,
                    width=self.config.global_camera_width,
                )
            self._camera_renderer.update_scene(
                self.data, camera=self.config.global_camera_name
            )
            self._camera_frame = self._camera_renderer.render().copy()
            self._camera_frame_time = current_time
        return self._camera_frame.copy()

    def observation(self) -> dict:
        observation = {
            "time": float(self.data.time),
            "arm_qpos": self.data.qpos[self.arm_qpos_adr].copy(),
            "hand_position": self.hand_position.copy(),
            "target_body_id": self.target_body_id,
            "target_position": self.target_position(),
            "target_velocity": self.target_velocity(),
            "cable_positions": self.data.xpos[self.cable_ids].copy(),
            "grasped_body_id": None if self.grasp_state is None else self.grasp_state.body_id,
        }
        if self.config.camera_observation_enabled:
            observation["camera_rgb"] = self.camera_rgb()
        return observation

    def close(self) -> None:
        """释放惰性创建的离屏相机渲染器。"""
        renderer = getattr(self, "_camera_renderer", None)
        if renderer is not None:
            renderer.close()
            self._camera_renderer = None
        self._camera_frame = None
        self._camera_frame_time = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # 构造失败或解释器关闭期间不传播清理异常。
            pass

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
            + self._last_rigid_shape_hold_acceleration
        )
        actual_arm_velocity = self.data.qvel[self.arm_dof_adr].copy()
        jacp, jacr = self._hand_jacobian()
        actual_hand_linear_velocity = jacp @ self.data.qvel
        actual_hand_angular_velocity = jacr @ self.data.qvel
        motion_limit_denominator = max(1, self._motion_limit_steps)
        physics_step_denominator = max(1, self._physics_steps)
        rms = lambda values: float(np.sqrt(np.mean(np.square(values))))
        uses_rigid_motion = self.config.motion_profile_version in RIGID_MOTION_PROFILES
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
            "rigid_motion_duration": (
                self._rigid_motion_duration() if uses_rigid_motion else None
            ),
            "rigid_motion_nominal_finished": self.rigid_motion_nominal_finished,
            "rigid_motion_exit_y": (
                self.config.rigid_motion_exit_y if uses_rigid_motion else None
            ),
            "rigid_motion_control": (
                "actual_progress_velocity_v1" if uses_rigid_motion else None
            ),
            "rigid_path_position_gain": self.config.rigid_path_position_gain,
            "rigid_velocity_gain": self.config.rigid_velocity_gain,
            "rigid_translation_max_acceleration": (
                self.config.rigid_translation_max_acceleration
            ),
            "rigid_motion_com_y": (
                self.rigid_motion_com_y if uses_rigid_motion else None
            ),
            "rigid_motion_finished": self.rigid_motion_finished,
            "rigid_motion_active": bool(
                uses_rigid_motion
                and not self.rigid_motion_finished
                and not self.rigid_motion_released
            ),
            "rigid_motion_released": self.rigid_motion_released,
            "termination_reason": self.last_termination_reason,
            "initial_shape_family": self._rigid_initial_shape_family,
            "rigid_initial_shape_family": self._rigid_initial_shape_family,
            "rigid_motion_rotation_sign": self._rigid_motion_rotation_sign,
            "rigid_motion_rotation_total_rad": (
                self._rigid_motion_rotation_target(
                    RIGID_MOTION_START_TIME + self._rigid_motion_duration()
                ) if uses_rigid_motion else None
            ),
            "cable_length_scale": self.config.cable_length_scale,
            "cable_density_scale": self.config.cable_density_scale,
            "cable_stiffness_scale": self.config.cable_stiffness_scale,
            "cable_damping_scale": self.config.cable_damping_scale,
            "cable_friction_scale": self.config.cable_friction_scale,
            "robot_motion_limit_profile": self.config.robot_motion_limit_profile,
            "arm_joint_velocity_limits": self._arm_velocity_limits.copy(),
            "arm_joint_acceleration_limits": self._arm_acceleration_limits.copy(),
            "hand_linear_velocity_limit": (
                self.config.hand_linear_velocity_limit
            ),
            "hand_angular_velocity_limit": (
                self.config.hand_angular_velocity_limit
            ),
            "gripper_finger_velocity_limit": (
                self.config.gripper_finger_velocity_limit
            ),
            "arm_position_tracking_error_limit": (
                self.config.arm_position_tracking_error_limit
            ),
            "low_level_velocity_guard_fraction": (
                self.config.low_level_velocity_guard_fraction
            ),
            "requested_action": self._last_requested_action.copy(),
            "applied_action": self._last_applied_action.copy(),
            "requested_arm_velocity": (
                self._last_requested_arm_velocity.copy()
            ),
            "applied_arm_velocity": self._last_applied_arm_velocity.copy(),
            "actual_arm_velocity": actual_arm_velocity,
            "commanded_hand_linear_velocity": (
                self._last_commanded_hand_velocity[:3].copy()
            ),
            "commanded_hand_angular_velocity": (
                self._last_commanded_hand_velocity[3:].copy()
            ),
            "actual_hand_linear_velocity": actual_hand_linear_velocity.copy(),
            "actual_hand_angular_velocity": actual_hand_angular_velocity.copy(),
            "max_abs_actual_arm_velocity": (
                self._max_abs_actual_arm_velocity.copy()
            ),
            "max_actual_hand_linear_speed": (
                self._max_actual_hand_linear_speed
            ),
            "max_actual_hand_angular_speed": (
                self._max_actual_hand_angular_speed
            ),
            "motion_limit_active": any(self._last_motion_limit_flags.values()),
            "motion_limit_flags": self._last_motion_limit_flags.copy(),
            "motion_limit_active_ratio": (
                self._motion_limit_active_steps / motion_limit_denominator
            ),
            "acceleration_limit_ratio": (
                self._acceleration_limit_steps / motion_limit_denominator
            ),
            "joint_velocity_limit_ratio": (
                self._joint_velocity_limit_steps / motion_limit_denominator
            ),
            "cartesian_velocity_limit_ratio": (
                self._cartesian_velocity_limit_steps / motion_limit_denominator
            ),
            "gripper_velocity_limit_ratio": (
                self._gripper_velocity_limit_steps / motion_limit_denominator
            ),
            "low_level_velocity_guard_ratio": (
                self._velocity_guard_steps / physics_step_denominator
            ),
            "actual_joint_velocity_exceedance_ratio": (
                self._actual_velocity_exceedance_steps
                / physics_step_denominator
            ),
            "shape_acceleration_rms": rms(self._last_shape_acceleration),
            "rigid_translation_acceleration_rms": rms(
                self._last_rigid_translation_acceleration
            ),
            "rigid_rotation_acceleration_rms": rms(
                self._last_rigid_rotation_acceleration
            ),
            "rigid_shape_hold_acceleration_rms": rms(
                self._last_rigid_shape_hold_acceleration
            ),
            "intended_disturbance_acceleration_rms": rms(intended_acceleration),
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

    def _update_rigid_motion_release_state(self) -> None:
        """稳定双侧抓取确认后，锁存L1/L2整体运动的撤除状态。"""
        if (
            self.config.motion_profile_version in RIGID_MOTION_PROFILES
            and self.grasp_confirmed
        ):
            self.rigid_motion_released = True

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
                (
                    not self.grasp_state.bilateral_confirmed
                    and self.grasp_state.lost_contact_time
                    >= self.config.grasp_candidate_gap_seconds
                )
                or (
                    self.grasp_state.bilateral_confirmed
                    and self.grasp_state.lost_contact_time
                    >= self.config.grasp_loss_seconds
                )
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
        radius = self.config.grasp_contact_index_radius
        for body_id in contacted_bodies:
            index = self.cable_index[body_id]
            neighborhood = set(
                self.cable_ids[
                    max(0, index - radius):
                    min(len(self.cable_ids), index + radius + 1)
                ]
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

    @staticmethod
    def _planar_rotation(angle: float) -> np.ndarray:
        """返回供行向量右乘的二维旋转矩阵。"""

        cosine = math.cos(angle)
        sine = math.sin(angle)
        return np.array([[cosine, sine], [-sine, cosine]])

    def _sample_initial_tangents(self, randomize: bool) -> np.ndarray:
        """生成长度不变、无自交的C/S/样条型平面初始构型。"""

        segment_s = np.linspace(0.0, 1.0, len(self.cable_ids) - 1)
        if not randomize:
            family = "c"
            tangents = math.radians(55.0) * (segment_s - 0.5)
            rotation_sign = 1.0
        else:
            family = str(self.rng.choice(("c", "s", "spline")))
            curve_sign = float(self.rng.choice((-1.0, 1.0)))
            if family == "c":
                amplitude = math.radians(self.rng.uniform(42.0, 70.0))
                tangents = curve_sign * amplitude * (segment_s - 0.5)
            elif family == "s":
                amplitude = math.radians(self.rng.uniform(24.0, 40.0))
                tangents = curve_sign * amplitude * np.sin(
                    2.0 * math.pi * segment_s
                )
            else:
                first = self.rng.uniform(0.55, 1.0)
                second = self.rng.uniform(-0.45, 0.45)
                phase = self.rng.uniform(-0.6, 0.6)
                tangents = (
                    first * np.sin(math.pi * segment_s + phase)
                    + second * np.sin(2.0 * math.pi * segment_s - phase)
                )
                tangents -= tangents.mean()
                maximum = float(np.max(np.abs(tangents)))
                tangents *= math.radians(self.rng.uniform(30.0, 46.0)) / maximum
                tangents *= curve_sign
            rotation_sign = float(self.rng.choice((-1.0, 1.0)))

        self._rigid_initial_shape_family = family
        self._rigid_motion_rotation_sign = rotation_sign
        self._rigid_initial_tangent_angles[:] = tangents
        return tangents

    def _set_curved_initial_shape(
        self,
        *,
        desired_com_xy: np.ndarray,
        randomize: bool,
        check_rigid_motion_sweep: bool,
    ) -> None:
        """直接设置球关节得到弯曲构型，并确保所需扫掠范围位于桌内。"""

        tangents = self._sample_initial_tangents(randomize)
        root_quaternion = self.data.qpos[
            self.cable_free_qadr + 3:self.cable_free_qadr + 7
        ]
        root_quaternion[:] = [
            math.cos(0.5 * tangents[0]), 0.0, 0.0,
            math.sin(0.5 * tangents[0]),
        ]
        for index, qpos_address in enumerate(self.cable_ball_qadr):
            angle = (
                tangents[index + 1] - tangents[index]
                if index + 1 < tangents.size
                else 0.0
            )
            self.data.qpos[qpos_address:qpos_address + 4] = [
                math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle),
            ]
        mujoco.mj_forward(self.model, self.data)

        current_xy = self.data.xpos[self.cable_ids, :2]
        current_com = np.average(current_xy, axis=0, weights=self.cable_mass)
        relative = current_xy - current_com
        margin = self.cable_radius + 0.01
        minimum_offset = np.full(2, math.inf)
        maximum_offset = np.full(2, -math.inf)
        if check_rigid_motion_sweep:
            duration = self._rigid_motion_duration()
            for time_value in np.linspace(
                RIGID_MOTION_START_TIME,
                RIGID_MOTION_START_TIME + duration,
                101,
            ):
                path = self._rigid_motion_target(float(time_value))
                yaw = self._rigid_motion_rotation_target(float(time_value))
                swept = path + relative @ self._planar_rotation(yaw)
                minimum_offset = np.minimum(minimum_offset, swept.min(axis=0))
                maximum_offset = np.maximum(maximum_offset, swept.max(axis=0))
        else:
            minimum_offset[:] = relative.min(axis=0)
            maximum_offset[:] = relative.max(axis=0)

        feasible_min = self.table_xy_min + margin - minimum_offset
        feasible_max = self.table_xy_max - margin - maximum_offset
        if np.any(feasible_min > feasible_max):
            raise ValueError("sampled curved cable does not fit on the table")
        start_com = np.clip(desired_com_xy, feasible_min, feasible_max)
        if check_rigid_motion_sweep and not math.isclose(
            float(start_com[1]), RIGID_MOTION_START_Y, abs_tol=1e-12
        ):
            raise ValueError(
                "sampled L1/L2 curve does not fit the Y sweep on the table"
            )
        self.initial_cable_translation[:] += start_com - desired_com_xy
        self.data.qpos[
            self.cable_free_qadr:self.cable_free_qadr + 2
        ] += start_com - current_com
        self.data.qvel[
            self.cable_free_dadr:self.cable_free_dadr + 6
        ] = 0.0
        mujoco.mj_forward(self.model, self.data)

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
        shape_hold = np.zeros_like(shape)

        if self.config.motion_mode in {"shape", "combined"}:
            shape = self._shape_acceleration(t, p)
        if self.config.motion_mode in {"rigid", "combined"}:
            # All rigid motion is now an explicit L1/L2 single pass.  It stops
            # after confirmed bilateral grasp; combined shape motion continues.
            if not self.rigid_motion_released:
                velocity_xy = np.array([
                    self.body_linear_velocity(body_id)[:2]
                    for body_id in self.cable_ids
                ])
                translation, rotation = self._rigid_motion_acceleration(
                    elapsed_time, velocity_xy
                )
                if self.config.motion_mode == "rigid":
                    shape_hold = self._rigid_shape_hold_acceleration(
                        elapsed_time, velocity_xy
                    )

        intended = shape + translation + rotation + shape_hold

        # 记录统计
        self._last_shape_acceleration[:] = shape
        self._last_rigid_translation_acceleration[:] = translation
        self._last_rigid_rotation_acceleration[:] = rotation
        self._last_rigid_shape_hold_acceleration[:] = shape_hold
        # 施加外力
        self.data.xfrc_applied[self.cable_ids, :3] += (
            self.cable_mass[:, None] * intended
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

    def _rigid_motion_acceleration(
        self, time_value: float, velocity_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return actual-progress L1/L2 translation and rotation fields.

        The controller never stores longitudinal path error against wall-clock
        time. Contact may therefore slow the cable, but releasing that contact
        only restores the nominal velocity instead of commanding a catch-up
        burst.
        """

        current_xy = self.data.xpos[self.cable_ids, :2]
        current_com = np.average(current_xy, axis=0, weights=self.cable_mass)
        current_velocity = np.average(
            velocity_xy, axis=0, weights=self.cable_mass
        )
        current_offset = current_com - self._rigid_reference_com_xy
        path_point, tangent, progress, progress_metric = (
            self._rigid_motion_path_state(current_offset)
        )

        if time_value < RIGID_MOTION_START_TIME:
            acceleration_xy = np.zeros(2)
        else:
            path_error = path_point - current_offset
            # Correct only normal displacement. Deliberately discard the
            # tangential component: that component is the old catch-up error.
            cross_track_error = path_error - tangent * np.dot(path_error, tangent)
            desired_velocity = self._rigid_motion_speed() * tangent
            acceleration_xy = (
                self.config.rigid_path_position_gain * cross_track_error
                + self.config.rigid_velocity_gain
                * (desired_velocity - current_velocity)
            )
        norm = float(np.linalg.norm(acceleration_xy))
        maximum = self.config.rigid_translation_max_acceleration
        if norm > maximum:
            acceleration_xy *= maximum / norm
        translation = np.broadcast_to(
            np.r_[acceleration_xy, 0.0], (len(self.cable_ids), 3)
        ).copy()

        desired_yaw, desired_yaw_rate = self._rigid_motion_rotation_state(
            progress, progress_metric, tangent, current_velocity,
        )
        if time_value < RIGID_MOTION_START_TIME:
            desired_yaw = 0.0
            desired_yaw_rate = 0.0
        _, current_yaw = self._current_rigid_pose()
        velocity_relative = velocity_xy - current_velocity
        current_relative_xy = self.data.xpos[self.cable_ids, :2] - current_com
        planar_inertia = float(np.sum(
            self.cable_mass
            * np.sum(current_relative_xy * current_relative_xy, axis=1)
        ))
        current_yaw_rate = float(np.sum(
            self.cable_mass * (
                current_relative_xy[:, 0] * velocity_relative[:, 1]
                - current_relative_xy[:, 1] * velocity_relative[:, 0]
            )
        ) / max(planar_inertia, 1e-12))
        angular_acceleration = (
            32.0 * (desired_yaw - current_yaw)
            + 9.0 * (desired_yaw_rate - current_yaw_rate)
        )
        angular_acceleration = float(np.clip(
            angular_acceleration, -18.0, 18.0
        ))
        positions = self.data.xpos[self.cable_ids]
        center = np.average(positions, axis=0, weights=self.cable_mass)
        relative = positions - center
        alpha = np.array([0.0, 0.0, angular_acceleration])
        omega = np.array([0.0, 0.0, desired_yaw_rate])
        rotation = (
            np.cross(np.broadcast_to(alpha, relative.shape), relative)
            + np.cross(
                np.broadcast_to(omega, relative.shape),
                np.cross(np.broadcast_to(omega, relative.shape), relative),
            )
        )
        rotation -= np.average(rotation, axis=0, weights=self.cable_mass)
        return translation, rotation

    def _rigid_motion_speed(self) -> float:
        return (
            self.config.rigid_motion_nominal_speed
            * self.config.motion_frequency_scale
        )

    def _rigid_motion_path_state(
        self, current_offset: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Project the real cable COM onto L1/L2 and return local path state.

        The final scalar is metres per unit progress. Past the nominal end the
        final tangent remains active until the real COM crosses the exit line.
        """

        if self.config.motion_profile_version == "rigid_level1_single_pass_v2":
            distance = float(np.clip(
                current_offset[1], 0.0, RIGID_MOTION_TRAVEL,
            ))
            return (
                np.array([0.0, distance]),
                np.array([0.0, 1.0]),
                distance / RIGID_MOTION_TRAVEL,
                RIGID_MOTION_TRAVEL,
            )

        x_sign = 1.0 if math.cos(self.phase_offset) >= 0.0 else -1.0
        reflection = np.array([x_sign, 1.0])
        samples = _L2_ARC_SAMPLES * reflection
        sample_index = int(np.argmin(np.sum(
            (samples - current_offset) ** 2, axis=1,
        )))
        progress = sample_index / float(len(samples) - 1)
        control = RIGID_MOTION_L2_CONTROL * reflection
        one_minus = 1.0 - progress
        derivative = (
            3.0 * one_minus ** 2 * (control[1] - control[0])
            + 6.0 * one_minus * progress * (control[2] - control[1])
            + 3.0 * progress ** 2 * (control[3] - control[2])
        )
        metric = float(np.linalg.norm(derivative))
        tangent = derivative / max(metric, 1e-12)
        return samples[sample_index], tangent, progress, metric

    def _rigid_motion_rotation_from_progress(self, progress: float) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        return (
            self._rigid_motion_rotation_sign
            * self.config.rigid_rotation_scale
            * RIGID_MOTION_ROTATION
            * smooth_progress
        )

    def _rigid_motion_rotation_state(
        self,
        progress: float,
        progress_metric: float,
        tangent: np.ndarray,
        current_velocity: np.ndarray,
    ) -> tuple[float, float]:
        """Return yaw and yaw rate driven by real, not scheduled, progress."""

        clipped_progress = float(np.clip(progress, 0.0, 1.0))
        forward_speed = max(float(np.dot(current_velocity, tangent)), 0.0)
        progress_rate = forward_speed / max(progress_metric, 1e-12)
        smooth_derivative = 6.0 * clipped_progress * (1.0 - clipped_progress)
        yaw_rate = (
            self._rigid_motion_rotation_sign
            * self.config.rigid_rotation_scale
            * RIGID_MOTION_ROTATION
            * smooth_derivative
            * progress_rate
        )
        return self._rigid_motion_rotation_from_progress(progress), yaw_rate

    def _rigid_motion_target(self, time_value: float) -> np.ndarray:
        """返回同起终点、无折返的L1直线或L2三次曲线质心位移。"""

        elapsed = max(0.0, float(time_value) - RIGID_MOTION_START_TIME)
        speed = (
            self.config.rigid_motion_nominal_speed
            * self.config.motion_frequency_scale
        )
        x_sign = 1.0 if math.cos(self.phase_offset) >= 0.0 else -1.0

        if self.config.motion_profile_version == "rigid_level1_single_pass_v2":
            progress = min(speed * elapsed, RIGID_MOTION_TRAVEL)
            return np.array([0.0, progress])

        duration = RIGID_MOTION_L2_ARC_LENGTH / speed
        u = float(np.clip(elapsed / duration, 0.0, 1.0))
        control = RIGID_MOTION_L2_CONTROL * np.array([x_sign, 1.0])
        return _cubic_bezier(control, u)

    def _rigid_motion_rotation_target(self, time_value: float) -> float:
        """Return the nominal scheduled yaw for diagnostics only."""

        elapsed = max(0.0, float(time_value) - RIGID_MOTION_START_TIME)
        progress = float(np.clip(elapsed / self._rigid_motion_duration(), 0.0, 1.0))
        return self._rigid_motion_rotation_from_progress(progress)

    def _rigid_motion_duration(self) -> float:
        speed = (
            self.config.rigid_motion_nominal_speed
            * self.config.motion_frequency_scale
        )
        path_length = (
            RIGID_MOTION_TRAVEL
            if self.config.motion_profile_version == "rigid_level1_single_pass_v2"
            else RIGID_MOTION_L2_ARC_LENGTH
        )
        return path_length / speed

    @property
    def rigid_motion_nominal_finished(self) -> bool:
        return bool(
            self.config.motion_profile_version in RIGID_MOTION_PROFILES
            and self.data.time
            >= RIGID_MOTION_START_TIME + self._rigid_motion_duration()
        )

    @property
    def rigid_motion_com_y(self) -> float:
        return float(np.average(
            self.data.xpos[self.cable_ids, 1], weights=self.cable_mass,
        ))

    @property
    def rigid_motion_finished(self) -> bool:
        """Return whether the real cable COM crossed the L1/L2 exit line."""

        return bool(
            self.config.motion_profile_version in RIGID_MOTION_PROFILES
            and self.rigid_motion_com_y >= self.config.rigid_motion_exit_y
        )

    def _rigid_shape_hold_acceleration(
        self, time_value: float, velocity_xy: np.ndarray,
    ) -> np.ndarray:
        """保持随目标转角旋转的初始构型；不产生净平移或净转矩。"""

        current_xy = self.data.xpos[self.cable_ids, :2]
        current_com = np.average(current_xy, axis=0, weights=self.cable_mass)
        reference_relative = self._rigid_reference_xy - self._rigid_reference_com_xy
        com_velocity = np.average(velocity_xy, axis=0, weights=self.cable_mass)
        current_offset = current_com - self._rigid_reference_com_xy
        _, tangent, progress, progress_metric = self._rigid_motion_path_state(
            current_offset
        )
        target_yaw, target_yaw_rate = self._rigid_motion_rotation_state(
            progress, progress_metric, tangent, com_velocity,
        )
        if time_value < RIGID_MOTION_START_TIME:
            target_yaw = 0.0
            target_yaw_rate = 0.0
        target_relative = reference_relative @ self._planar_rotation(target_yaw)
        current_relative = current_xy - current_com
        position_error = target_relative - current_relative

        relative_velocity = velocity_xy - com_velocity
        target_relative_velocity = target_yaw_rate * np.column_stack((
            -target_relative[:, 1], target_relative[:, 0],
        ))
        correction_xy = (
            self.config.rigid_shape_stiffness * position_error
            - self.config.rigid_shape_damping
            * (relative_velocity - target_relative_velocity)
        )
        correction_xy -= np.average(
            correction_xy, axis=0, weights=self.cable_mass
        )
        inertia = float(np.sum(
            self.cable_mass * np.sum(current_relative * current_relative, axis=1)
        ))
        torque = float(np.sum(
            self.cable_mass * (
                current_relative[:, 0] * correction_xy[:, 1]
                - current_relative[:, 1] * correction_xy[:, 0]
            )
        ))
        if inertia > 1e-12:
            angular_component = torque / inertia
            correction_xy -= angular_component * np.column_stack((
                -current_relative[:, 1], current_relative[:, 0],
            ))
        maximum_norm = float(np.max(np.linalg.norm(correction_xy, axis=1)))
        if maximum_norm > self.config.rigid_shape_max_acceleration:
            correction_xy *= (
                self.config.rigid_shape_max_acceleration / maximum_norm
            )
        result = np.zeros((len(self.cable_ids), 3))
        result[:, :2] = correction_xy
        return result

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

    def _reset_stochastic_motion(self, *, randomize: bool) -> None:
        """为一个 episode 冻结平滑带限随机运动谱。

        随机数只在 reset 时生成；物理子步中只计算这些固定正弦基，因此同一
        seed 与场景对所有方法完全可复现，也不会引入与控制频率相关的白噪声。
        """

        if self.config.motion_regularity != "stochastic":
            self._stochastic_shape_frequency[:] = 1.0
            self._stochastic_shape_direction[:] = 1.0
            self._stochastic_shape_phase[:] = 0.0
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
        # The wheel uses platform-specific library names (.dll/.so/.dylib).
        # Loading the whole bundled plugin directory avoids encoding any one OS.
        plugin_dir = Path(mujoco.__file__).resolve().parent / "plugin"
        if not plugin_dir.is_dir():
            raise FileNotFoundError(
                f"MuJoCo bundled plugin directory was not found: {plugin_dir}"
            )
        mujoco.mj_loadAllPluginLibraries(str(plugin_dir))

        panda_dir = resolve_menagerie_panda_dir()
        if not PANDA_XML_PATH.is_file():
            raise FileNotFoundError(
                f"Repository-owned Panda model definition is missing: {PANDA_XML_PATH}"
            )

        # 所有场景都从同一源模型编译，在编译阶段扩大真实碰撞桌面并删除旧的
        # 单侧实体挡板。Panda XML由仓库固定，外部Menagerie只提供官方mesh资产，
        # 因此服务器无需修改或复制Menagerie文件，抓取几何也不会随机器变化。
        spec = mujoco.MjSpec.from_file(
            str(XML_PATH),
            include={"panda.xml": PANDA_XML_PATH.read_bytes()},
            assets=_panda_assets(str(panda_dir)),
        )
        table = next(geom for geom in spec.geoms if geom.name == "table")
        table.size[:2] = config.table_half_size
        for geom in list(spec.geoms):
            if geom.name == "table_edge":
                spec.delete(geom)
        spec.stat.extent = max(float(spec.stat.extent), 1.80)

        # 相机挂在world body上，因此机械臂运动不会改变外参。编译前注入MjSpec，
        # 确保保存出的.mjb、观测和视频使用同一个固定视觉传感器。
        global_camera = spec.worldbody.add_camera()
        global_camera.name = config.global_camera_name
        global_camera.pos[:] = config.global_camera_pos
        global_camera.quat[:] = config.global_camera_quat
        global_camera.fovy = config.global_camera_fovy
        global_camera.resolution[:] = (
            config.global_camera_width, config.global_camera_height,
        )

        # 修改线缆 OOD 属性
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
