"""Franka Panda 抓取持续运动线缆的 MuJoCo 环境。

物理、任务状态和成功判定都放在这里。机械臂动作生成被有意放在环境之外，
因此策略动作不能暗中改变线缆扰动。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
MENAGERIE = ROOT.parent / "mujoco_menagerie"
# 注意：程序真正加载的是 Menagerie 目录中的 XML；项目目录还有一个同步副本。
XML_PATH = MENAGERIE / "franka_emika_panda" / "panda_cable_grasp.xml"


@dataclass
class EnvConfig:
    """环境参数。这里只决定物理和任务，不决定机械臂采用什么抓取方法。"""

    seed: int = 20260804
    episode_seconds: float = 28.0       # 每次随机试验最多运行多少仿真秒
    disturbance_strength: float = 1.5   # 线缆外力倍率，与策略动作无关
    success_hold_seconds: float = 0.55  # 成功条件必须连续保持的时间
    grasp_confirm_seconds: float = 0.06 # 双侧内指垫接触保持多久才确认抓取
    grasp_loss_seconds: float = 0.10    # 接触短暂抖动的容忍时间
    max_grasp_aperture: float = 0.034  # 28 mm线缆被真正夹紧时允许的最大开口
    max_pad_distance: float = 0.055    # 接触线段中心到指垫中心的最大距离
    min_pad_normal_force: float = 0.20 # 每侧内指垫所需最小法向力，单位N
    frame_skip: int = 10                # 一个50 Hz动作对应10个500 Hz物理步
    gripper_force_scale: float = 5.0    # 适度增强夹紧力，避免位置伺服把圆柱从指垫间挤射出去
    boundary_margin: float = 0.16       # 距桌边多远开始调整环境扰动力
    boundary_stiffness: float = 180.0   # 软边界回正加速度系数，单位1/s²
    boundary_damping: float = 28.0      # 只衰减朝桌外运动的速度，单位1/s


@dataclass
class GraspState:
    """真实内指垫接触的短期状态；不产生任何约束或外力。"""

    body_id: int
    capture_contact_count: int
    capture_finger_count: int
    candidate_time: float
    bilateral_confirmed: bool
    last_bilateral_time: float
    lost_contact_time: float
    left_normal_force: float
    right_normal_force: float


def id_of(model: mujoco.MjModel, obj: int, name: str) -> int:
    result = mujoco.mj_name2id(model, obj, name)
    if result < 0:
        raise RuntimeError(f"Missing {name!r} in model")
    return result


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


class CableGraspEnv:
    """具有独立线缆力场、提供 reset/step 接口的小型环境。

    动作是8维向量：前7维为 Panda 关节位置执行器目标，第8维为夹爪命令。
    255表示张开，较小数值表示闭合。观测采用字典形式，便于检查，也不绑定某个
    特定强化学习库。
    """

    # 手部 body 原点不在指尖；所有末端位置和 Jacobian 都使用这个局部偏移点。
    HAND_LOCAL_POINT = np.array([0.0, 0.0, 0.145])
    # 由 finger body 的0.0584 m基座偏移和主指垫0.0445 m局部位置相加得到。
    PAD_CENTER_LOCAL = np.array([0.0, 0.0, 0.1029])
    # Menagerie 原始的弯臂、夹爪朝下 home 姿态。配合模型原始 qpos0，
    # 夹爪大约位于 [0.55, 0, 0.48]；全零关节配置则是这里不需要的竖直姿态。
    READY_ARM_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.model = self._load_model()
        # Menagerie 的 actuator8 原本比较柔和。细线缆受到持续扰动时可能把手指撑开，
        # 因此等比例放大增益和位置偏置：保持开合目标位置不变，只提高夹持力。
        grip_scale = self.config.gripper_force_scale
        self.model.actuator_gainprm[7, 0] *= grip_scale
        self.model.actuator_biasprm[7, 1] *= grip_scale
        self.model.actuator_biasprm[7, 2] *= math.sqrt(grip_scale)
        self.data = mujoco.MjData(self.model)

        # joint id 用来查模型属性；qpos address 才是 data.qpos 中的数组下标。
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
        # Menagerie每根手指有一个整体mesh和五个内指垫box。碰撞仍全部保留，
        # 但抓取判断只统计box，避免把手指外侧碰撞误判成夹紧。
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
        # 指垫的尺寸、位置、姿态和可视模型全部保持Menagerie原样。这里只修改材料接触参数，
        # 因此策略必须真正把28 mm线缆对准官方17 mm主指垫，环境不会用大碰撞盒或凹槽兜底。
        for geom_id in self.pad_geom_ids:
            # 指垫优先级高于普通线缆geom，使用高摩擦且较硬的橡胶接触。
            self.model.geom_priority[geom_id] = 1
            # condim=6同时启用滑动、绕法线扭转和滚动摩擦。这里模拟高摩擦橡胶指垫；
            # 由于priority=1，参数只在该指垫参与接触时覆盖普通线缆参数。
            self.model.geom_condim[geom_id] = 6
            self.model.geom_friction[geom_id] = [6.0, 0.35, 0.12]
            self.model.geom_solref[geom_id] = [0.002, 1.0]
            self.model.geom_solimp[geom_id] = [0.98, 0.995, 0.0005, 0.5, 2.0]
        self.finger_joint_ids = np.array([
            id_of(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
            id_of(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
        ], dtype=int)
        self.finger_qpos_adr = self.model.jnt_qposadr[self.finger_joint_ids].copy()
        self.cable_ids = self._cable_bodies()
        self.cable_set = set(self.cable_ids)
        self.cable_index = {body_id: index for index, body_id in enumerate(self.cable_ids)}
        self.cable_free_qadr = self._find_cable_free_qpos_address()
        # 缓存每步不变的节点质量和空间相位，避免500 Hz循环中反复创建相同数组。
        self.cable_mass = self.model.body_mass[self.cable_ids].copy()
        self.cable_s = np.linspace(0.0, 1.0, len(self.cable_ids))
        self._lateral_space = math.pi * np.outer(
            np.array([3.3, 7.7, 13.1, 19.3]), self.cable_s
        )
        self._longitudinal_space = math.pi * np.outer(
            np.array([5.1, 11.6]), self.cable_s
        )
        self._vertical_space = math.pi * np.outer(
            np.array([4.6, 10.4]), self.cable_s
        )

        self.ready_qpos = self.model.key_qpos[self.home_id, :9].copy()
        self.ready_qpos[:7] = self.READY_ARM_QPOS
        self.ready_ctrl = self.model.key_ctrl[self.home_id, :8].copy()
        self.ready_ctrl[:7] = self.READY_ARM_QPOS
        self.ready_ctrl[7] = 255.0

        # 不修改 model.qpos0：转动关节运动学把它当作参考配置，若将其作为重置书签会
        # 改变几何关系。GUI 回调会直接执行完整的任务重置。

        self.trial_index = 0
        self.phase_offset = 0.0
        self.spatial_phase = 0.0
        self.target_body_id = self.cable_ids[len(self.cable_ids) // 2]
        self.grasp_state: GraspState | None = None
        self.success_hold = 0.0
        self.ever_success = False
        self.success_snapshot: dict | None = None
        self.last_grasped_body_id: int | None = None
        # 保存最近一次抓取状态被清除前的现场。策略随后即使主动张开夹爪，
        # 这里记录的仍是张开前的数据。
        self.last_grasp_break: dict | None = None
        self._last_contact_count = 0
        self.reset()

    @staticmethod
    def _load_model() -> mujoco.MjModel:
        plugin = Path(mujoco.__file__).resolve().parent / "plugin" / "elasticity.dll"
        if plugin.exists():
            mujoco.mj_loadPluginLibrary(str(plugin))
        return mujoco.MjModel.from_xml_path(str(XML_PATH))

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

    def reset(self, *, randomize: bool = True) -> tuple[dict, dict]:
        """重置机器人、线缆、随机相位、目标线段和所有成功判定状态。"""

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = self.ready_qpos
        self.data.ctrl[:8] = self.ready_ctrl

        if randomize:
            # 每轮改变线缆初始平移、波形相位和目标段，但固定 seed 时序列可复现。
            dx = self.rng.uniform(-0.10, 0.10)
            dy = self.rng.uniform(-0.13, 0.13)
            self.data.qpos[self.cable_free_qadr:self.cable_free_qadr + 3] += [dx, dy, 0.0]
            self.phase_offset = self.rng.uniform(0.0, 2.0 * math.pi)
            self.spatial_phase = self.rng.uniform(0.0, 2.0 * math.pi)
            # 避开看起来近似固定的两端，但也不总是选择几何中心。
            lo = len(self.cable_ids) // 4
            hi = len(self.cable_ids) - lo
            target_index = int(self.rng.integers(lo, hi))
            self.target_body_id = self.cable_ids[target_index]
        else:
            self.phase_offset = 0.0
            self.spatial_phase = 0.0
            self.target_body_id = self.cable_ids[len(self.cable_ids) // 2]

        self.grasp_state = None
        self.success_hold = 0.0
        self.ever_success = False
        self.success_snapshot = None
        self.last_grasped_body_id = None
        self.last_grasp_break = None
        self._last_contact_count = 0
        self.trial_index += 1
        mujoco.mj_forward(self.model, self.data)
        return self.observation(), self.info()

    @property
    def hand_position(self) -> np.ndarray:
        rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        return self.data.xpos[self.hand_id] + rotation @ self.HAND_LOCAL_POINT

    @property
    def pad_center_position(self) -> np.ndarray:
        """返回两块主内指垫之间的几何中心。"""
        rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        return self.data.xpos[self.hand_id] + rotation @ self.PAD_CENTER_LOCAL

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

    def _finger_contact_pairs(self, cable_body: int | None = None) -> list[tuple[int, int]]:
        """返回内指垫box与线缆的（线缆body，手指body）接触。"""
        return [
            (body_id, finger_id)
            for body_id, finger_id, _ in self._pad_contact_samples()
            if cable_body is None or body_id == cable_body
        ]

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

    def finger_normal_forces(self) -> dict[int, float]:
        """汇总当前线缆对左右内指垫的法向力，单位N。"""
        result = {self.left_finger_id: 0.0, self.right_finger_id: 0.0}
        for _, finger_id, normal_force in self._pad_contact_samples():
            result[finger_id] += normal_force
        return result

    def _physical_grasp_candidate(
        self,
    ) -> tuple[int, int, float, float] | None:
        """查找当前被两侧内指垫真实夹紧的局部线段。"""
        aperture = float(np.sum(self.data.qpos[self.finger_qpos_adr]))
        if aperture > self.config.max_grasp_aperture:
            return None
        samples = self._pad_contact_samples()
        if not samples:
            return None

        contacted_bodies = {body_id for body_id, _, _ in samples}
        candidates: list[tuple[float, int, int, float, float]] = []
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
                contact_count = sum(1 for cable, _, _ in samples if cable in neighborhood)
                candidates.append((
                    distance, center_body, contact_count, left_force, right_force
                ))
        if not candidates:
            return None
        _, body_id, contact_count, left_force, right_force = min(candidates)
        return body_id, contact_count, left_force, right_force

    def _clear_grasp_with_reason(
        self,
        reason: str,
        *,
        pairs: list[tuple[int, int]] | None = None,
        grasp_error: float | None = None,
    ) -> None:
        """清除抓取状态，同时冻结清除前的诊断数据。"""
        if self.grasp_state is None:
            return
        state = self.grasp_state
        all_pairs = self._finger_contact_pairs() if pairs is None else pairs
        if grasp_error is None:
            grasp_error = self._current_grasp_error()
        contacting_fingers = {finger for _, finger in all_pairs}
        unique_nodes = {body for body, _ in all_pairs}
        self.last_grasp_break = {
            "reason": reason,
            "time": float(self.data.time),
            "body_id": state.body_id,
            "bilateral_confirmed": state.bilateral_confirmed,
            "raw_contact_count": len(all_pairs),
            "unique_contact_nodes": len(unique_nodes),
            "contacting_finger_count": len(contacting_fingers),
            "lost_bilateral_seconds": max(
                0.0, float(self.data.time - state.last_bilateral_time)
            ),
            "one_sided_contact_time": 0.0,
            "no_contact_time": float(state.lost_contact_time),
            "large_error_time": 0.0,
            "outside_gripper_time": 0.0,
            "grasp_error": float(grasp_error),
            "grasp_break_distance": float(self.config.max_pad_distance),
            "finger_aperture": float(np.sum(self.data.qpos[self.finger_qpos_adr])),
            "gripper_ctrl": float(self.data.ctrl[7]),
            "left_normal_force": float(state.left_normal_force),
            "right_normal_force": float(state.right_normal_force),
            "ever_success": bool(self.ever_success),
            "success_hold": float(self.success_hold),
        }
        self.grasp_state = None

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
            self.grasp_state.lost_contact_time += timestep
            if (
                not self.grasp_state.bilateral_confirmed
                or self.grasp_state.lost_contact_time >= self.config.grasp_loss_seconds
            ):
                self._clear_grasp_with_reason("lost_physical_pad_contact")
            return

        body_id, contact_count, left_force, right_force = candidate
        if self.grasp_state is None:
            self.grasp_state = GraspState(
                body_id=body_id,
                capture_contact_count=contact_count,
                capture_finger_count=2,
                candidate_time=float(self.data.time),
                bilateral_confirmed=False,
                last_bilateral_time=float(self.data.time),
                lost_contact_time=0.0,
                left_normal_force=left_force,
                right_normal_force=right_force,
            )
            return

        state = self.grasp_state
        # 线缆可以在高摩擦指垫间发生有限滑动；只要双侧真实接触仍在，就更新当前段。
        state.body_id = body_id
        state.capture_contact_count = contact_count
        state.last_bilateral_time = float(self.data.time)
        state.lost_contact_time = 0.0
        state.left_normal_force = left_force
        state.right_normal_force = right_force
        if (
            not state.bilateral_confirmed
            and self.data.time - state.candidate_time >= self.config.grasp_confirm_seconds
        ):
            state.bilateral_confirmed = True
            self.last_grasped_body_id = state.body_id

    def _apply_cable_disturbance(self) -> None:
        """对所有机器人动作施加完全相同的行波形变力场。"""
        # s是沿线缆从0到1的节点位置；t是仿真时间；p是每轮随机空间相位。
        t = self.phase_offset + self.data.time
        p = self.spatial_phase

        # 多个互不整除的行波使内部各段持续运动。去掉均值后主要产生弯折，避免整条线缆
        # 像刚体一样持续加速。快速相位调制和更强的短波产生局部曲率与加速度，
        # 而不是仅让线缆整体平移或滚动；频率互不整除也能避免单次任务内周期性重复。
        lateral = (
            2.30 * np.sin(
                self._lateral_space[0] - 3.4 * t + p + 0.65 * np.sin(1.9 * t)
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
        # 去掉各方向平均值，尽量产生局部弯折，而不是给整条线一个净推力。
        lateral -= lateral.mean()
        longitudinal -= longitudinal.mean()
        vertical -= vertical.mean()

        # 一次生成全部节点的Nx3加速度矩阵，避免逐节点构造小型NumPy数组。
        strength = self.config.disturbance_strength
        acceleration = strength * np.column_stack((
            5.0 * longitudinal,
            19.0 * lateral,
            9.0 * vertical,
        ))
        acceleration = self._confine_environment_acceleration(acceleration)
        self.data.xfrc_applied[self.cable_ids, :3] += self.cable_mass[:, None] * acceleration

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

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """一个50 Hz控制动作内部执行十个500 Hz物理子步。"""
        # action[0:7]是Panda七个关节的位置目标，action[7]是夹爪命令。
        # 环境只按执行器合法范围裁剪，不根据策略阶段冻结或覆盖动作。
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
            # 线缆外力只有持续环境扰动；抓取完全由MuJoCo碰撞、夹紧力和摩擦产生。
            self.data.xfrc_applied[:] = 0.0
            self._apply_cable_disturbance()  # 扰动强度永远不随夹爪命令缩放

            # 真正把当前方法/模型给出的动作交给MuJoCo执行。
            self.data.ctrl[:] = clipped_action
            mujoco.mj_step(self.model, self.data)
            self._last_contact_count = len(self._finger_contact_pairs())
            self._update_physical_grasp_state(gripper_closed)

            grasp_error = self._current_grasp_error()
            last_qualification = self._success_qualification(gripper_closed, grasp_error)
            if last_qualification:
                self.success_hold += self.model.opt.timestep
            else:
                self.success_hold = 0.0
            success_now = self.success_hold >= self.config.success_hold_seconds
            if success_now and not self.ever_success:
                self.ever_success = True
                current_info = self.info()
                self.success_snapshot = {
                    "time": float(self.data.time),
                    "grasped_body_id": current_info["grasped_body_id"],
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

    def _current_grasp_error(self) -> float:
        if self.grasp_state is None:
            return math.inf
        return float(np.linalg.norm(
            self.data.xpos[self.grasp_state.body_id] - self.pad_center_position
        ))

    def _success_qualification(self, gripper_closed: bool, grasp_error: float) -> bool:
        """单个物理时刻是否满足成功条件；还需连续保持0.55秒才最终成功。"""

        if not gripper_closed or self.grasp_state is None or not self.grasp_confirmed:
            return False
        candidate = self._physical_grasp_candidate()
        if candidate is None:
            return False
        body_id = candidate[0]
        cable_z = self.data.xpos[self.cable_ids, 2]
        lifted_fraction = float(np.mean(cable_z > 0.055))
        aperture = float(np.sum(self.data.qpos[self.finger_qpos_adr]))
        return (
            self.data.xpos[body_id, 2] > 0.14
            and lifted_fraction >= 0.18
            and grasp_error <= self.config.max_pad_distance
            and 0.0 <= aperture <= self.config.max_grasp_aperture
        )

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
        grasped_body = None
        if self.grasp_state is not None:
            grasped_body = self.grasp_state.body_id
            grasp_error = self._current_grasp_error()
        normal_forces = self.finger_normal_forces()
        return {
            "trial": self.trial_index,
            "target_body_id": self.target_body_id,
            "grasped_body_id": grasped_body,
            "capture_contact_count": (
                0 if self.grasp_state is None else self.grasp_state.capture_contact_count
            ),
            "capture_finger_count": (
                0 if self.grasp_state is None else self.grasp_state.capture_finger_count
            ),
            "bilateral_grasp": self.grasp_confirmed,
            "grasp_patch_size": (
                0 if self.grasp_state is None else 1
            ),
            "finger_aperture": float(np.sum(self.data.qpos[self.finger_qpos_adr])),
            "grasp_attempt_eligible_count": 0,
            "finger_contact_count": self._last_contact_count,
            "left_finger_normal_force": normal_forces[self.left_finger_id],
            "right_finger_normal_force": normal_forces[self.right_finger_id],
            "grasp_error": grasp_error,
            "cable_com": cable_positions.mean(axis=0).copy(),
            "max_z": float(cable_positions[:, 2].max()),
            "lifted_fraction": float(np.mean(cable_positions[:, 2] > 0.055)),
            "success_hold": self.success_hold,
            "success": self.ever_success,
        }
