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
    grasp_break_distance: float = 0.025 # 夹持代理均方根误差阈值
    grasp_break_hold_seconds: float = 0.15 # 超阈值必须持续一段时间才确认滑脱
    frame_skip: int = 10                # 一个50 Hz动作对应10个500 Hz物理步
    gripper_force_scale: float = 1.5    # Panda夹爪执行器增益倍率
    boundary_margin: float = 0.16       # 距桌边多远开始调整环境扰动力
    boundary_stiffness: float = 180.0   # 软边界回正加速度系数，单位1/s²
    boundary_damping: float = 28.0      # 只衰减朝桌外运动的速度，单位1/s


@dataclass
class GraspState:
    """一次已候选/已确认抓取的环境内部状态。"""

    body_id: int
    hand_local_offset: np.ndarray
    capture_contact_count: int
    capture_finger_count: int
    candidate_time: float
    bilateral_confirmed: bool
    patch_body_ids: tuple[int, ...]
    patch_local_offsets: np.ndarray
    last_bilateral_time: float
    one_sided_contact_time: float
    no_contact_time: float
    large_error_time: float
    outside_gripper_time: float


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
        self.finger_ids = {
            id_of(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
            id_of(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
        }
        self.finger_joint_ids = np.array([
            id_of(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
            id_of(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
        ], dtype=int)
        self.finger_qpos_adr = self.model.jnt_qposadr[self.finger_joint_ids].copy()
        self.cable_ids = self._cable_bodies()
        self.cable_set = set(self.cable_ids)
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
        self._gripper_was_closed = False
        self._capture_eligible: set[int] = set()
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
        self._gripper_was_closed = False
        self._capture_eligible.clear()
        self.trial_index += 1
        mujoco.mj_forward(self.model, self.data)
        return self.observation(), self.info()

    @property
    def hand_position(self) -> np.ndarray:
        rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        return self.data.xpos[self.hand_id] + rotation @ self.HAND_LOCAL_POINT

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
        """返回当前每个指垫接触对应的（线缆 body，手指 body）。"""
        contacts: list[tuple[int, int]] = []
        for contact in self.data.contact[:self.data.ncon]:
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            cable = None
            if body1 in self.cable_set and body2 in self.finger_ids:
                cable = body1
            elif body2 in self.cable_set and body1 in self.finger_ids:
                cable = body2
            if cable is not None and (cable_body is None or cable == cable_body):
                finger = body2 if body1 == cable else body1
                contacts.append((cable, finger))
        return contacts

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
            "one_sided_contact_time": float(state.one_sided_contact_time),
            "no_contact_time": float(state.no_contact_time),
            "large_error_time": float(state.large_error_time),
            "outside_gripper_time": float(state.outside_gripper_time),
            "grasp_error": float(grasp_error),
            "grasp_break_distance": float(self.config.grasp_break_distance),
            "finger_aperture": float(np.sum(self.data.qpos[self.finger_qpos_adr])),
            "gripper_ctrl": float(self.data.ctrl[7]),
            "ever_success": bool(self.ever_success),
            "success_hold": float(self.success_hold),
        }
        if state.bilateral_confirmed:
            # 已确认抓取一旦真正断裂，本次闭爪周期不能再次自动吸附；必须先张开再闭合。
            self._capture_eligible.clear()
        self.grasp_state = None

    def _try_capture_grasp(self) -> None:
        """只有同时发生双指接触时才创建抓取候选。"""
        # 必须由左右两个手指同时接触同一离散段或相邻段，单侧接触不算抓住。
        pairs = self._finger_contact_pairs()
        self._last_contact_count = len(pairs)
        contacted_bodies = {body for body, _ in pairs}
        candidates = set()
        for body_id in contacted_bodies:
            index = self.cable_ids.index(body_id)
            neighborhood = set(
                self.cable_ids[max(0, index - 1):min(len(self.cable_ids), index + 2)]
            )
            fingers = {finger for body, finger in pairs if body in neighborhood}
            if fingers == self.finger_ids:
                candidates.add(body_id)
        # 抓取资格在夹爪命令由张开转为闭合时采样。闭合夹爪之后横扫碰到的线缆仍会
        # 正常碰撞和运动，但不能激活承重代理或任务成功。
        # 再与“闭爪开始时已在指间”的集合取交集，排除闭爪横扫后补挤进去的线段。
        candidates.intersection_update(self._capture_eligible)
        if not candidates:
            return
        reference = self.hand_position
        body_id = min(
            candidates,
            key=lambda candidate: np.linalg.norm(self.data.xpos[candidate] - reference),
        )
        rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        measured_local = rotation.T @ (self.data.xpos[body_id] - reference)
        # 将线缆放在两指之间；保持沿指垫方向的位置和接触深度连续，避免激活代理时跳变。
        centred_local = measured_local.copy()
        centred_local[1] = 0.0
        centred_local[2] = float(np.clip(centred_local[2], -0.036, -0.018))
        self.grasp_state = GraspState(
            body_id=body_id,
            hand_local_offset=centred_local,
            capture_contact_count=sum(1 for cable, _ in pairs if cable == body_id),
            capture_finger_count=2,
            candidate_time=float(self.data.time),
            bilateral_confirmed=False,
            patch_body_ids=(body_id,),
            patch_local_offsets=centred_local[None, :],
            last_bilateral_time=float(self.data.time),
            one_sided_contact_time=0.0,
            no_contact_time=0.0,
            large_error_time=0.0,
            outside_gripper_time=0.0,
        )
        # 此时先不施加柔性承重约束。双侧物理接触必须持续一小段时间，避免夹爪经过线缆时
        # 一次短暂碰撞就触发抬升。

    @property
    def grasp_confirmed(self) -> bool:
        return self.grasp_state is not None and self.grasp_state.bilateral_confirmed

    def _update_grasp_confirmation(self) -> None:
        """只有持续存在的真实双指接触才能确认抓取。"""
        if self.grasp_state is None:
            return
        state = self.grasp_state
        index = self.cable_ids.index(state.body_id)
        neighborhood = set(self.cable_ids[max(0, index - 1):min(len(self.cable_ids), index + 2)])
        pairs = [pair for pair in self._finger_contact_pairs() if pair[0] in neighborhood]
        contacting_fingers = {finger for _, finger in pairs}
        if contacting_fingers == self.finger_ids:
            state.last_bilateral_time = float(self.data.time)
            state.one_sided_contact_time = 0.0
            state.no_contact_time = 0.0
            if (
                not state.bilateral_confirmed
                and self.data.time - state.candidate_time >= 0.08
            ):
                # 双侧接触连续保持0.08秒后，才允许建立可承重的局部夹持代理。
                state.bilateral_confirmed = True
                self._initialize_grasp_patch(state)
                self.last_grasped_body_id = state.body_id
        else:
            if not state.bilateral_confirmed:
                # 确认前失去双侧接触就是抓空，不提供弱引导或不可见的磁吸效果。
                self._clear_grasp_with_reason(
                    "lost_contact_before_confirmation", pairs=pairs
                )
            elif len(contacting_fingers) == 1:
                # 弹性夹持建立后，线缆可能因夹爪开口略大于直径而只与一侧指垫接触。
                # 单侧接触只作诊断，不再撤销仍然稳定的夹持代理。
                state.one_sided_contact_time += self.model.opt.timestep
                state.no_contact_time = 0.0
            else:
                state.no_contact_time += self.model.opt.timestep
                state.one_sided_contact_time = 0.0

    def _initialize_grasp_patch(self, state: GraspState) -> None:
        """在指垫宽度方向建立包含三个节点、约40 mm的局部夹持段。"""
        # 使用中心节点及左右邻居表示约40 mm夹持宽度，避免只固定一个点时自由旋转/穿模。
        index = self.cable_ids.index(state.body_id)
        lo = max(0, index - 1)
        hi = min(len(self.cable_ids), index + 2)
        patch_ids = tuple(self.cable_ids[lo:hi])
        rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        reference = self.hand_position
        offsets = np.array([
            rotation.T @ (self.data.xpos[body_id] - reference)
            for body_id in patch_ids
        ])
        # Panda 指垫中心位于 HAND_LOCAL_POINT 后方约42 mm。将局部夹持段对齐到这里，
        # 并沿手坐标系 X 轴（线缆切向）排列。保留线缆20 mm静止节距，因为接触瞬间
        # 相邻节点中心可能暂时挤在一起。
        center_slot = patch_ids.index(state.body_id)
        tangent_sign = float(np.sign(offsets[-1, 0] - offsets[0, 0]))
        if tangent_sign == 0.0:
            tangent_sign = 1.0
        # 主指垫中心位于局部 X=0，宽度约17 mm；将中心线缆胶囊对齐到该位置。
        center_x = 0.0
        for slot in range(len(patch_ids)):
            offsets[slot, 0] = center_x + tangent_sign * (slot - center_slot) * 0.020
        offsets[:, 1] = 0.0
        offsets[:, 2] = -0.042
        state.patch_body_ids = patch_ids
        state.patch_local_offsets = offsets
        state.hand_local_offset = offsets[center_slot].copy()

    def _apply_compliant_grasp(self) -> float:
        """对已确认的三个局部节点施加有限弹簧阻尼力；这不是纯接触/FEM指垫。"""

        if self.grasp_state is None:
            return math.inf
        state = self.grasp_state
        reference = self.hand_position
        rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        hand_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.hand_id,
            hand_velocity,
            0,
        )
        angular_velocity = hand_velocity[:3]
        origin_velocity = hand_velocity[3:]
        total_mass = float(np.sum(self.model.body_mass[self.cable_ids]))
        body_ids = state.patch_body_ids
        local_offsets = state.patch_local_offsets
        # 手坐标系各向异性刚度：夹持法向较硬，沿线方向较软，仍允许滑移和断开。
        stiffness = np.array([350.0, 1000.0, 600.0])
        damping = np.array([5.0, 12.0, 9.0])

        squared_errors = []
        for body_id, local_offset in zip(body_ids, local_offsets):
            target = reference + rotation @ local_offset
            error = target - self.data.xpos[body_id]
            squared_errors.append(float(error @ error))
            target_arm = rotation @ (self.HAND_LOCAL_POINT + local_offset)
            target_velocity = origin_velocity + np.cross(angular_velocity, target_arm)
            relative_velocity = target_velocity - self.body_linear_velocity(body_id)
            error_local = rotation.T @ error
            velocity_local = rotation.T @ relative_velocity
            force = rotation @ (stiffness * error_local + damping * velocity_local)
            force[2] += total_mass * 9.81 / len(body_ids)
            self.data.xfrc_applied[body_id, :3] += np.clip(force, -8.0, 8.0)

        error_norm = math.sqrt(sum(squared_errors) / len(squared_errors))
        timestep = float(self.model.opt.timestep)
        if error_norm > self.config.grasp_break_distance:
            state.large_error_time += timestep
        else:
            state.large_error_time = 0.0

        # 使用线缆中心相对夹爪的真实位置检查是否已经离开两指区域。线缆半径为14 mm；
        # 再留6 mm数值余量，避免软接触表面的小幅振动被当成滑脱。
        center_local = rotation.T @ (self.data.xpos[state.body_id] - reference)
        aperture = float(np.sum(self.data.qpos[self.finger_qpos_adr]))
        inside_gripper = (
            abs(center_local[0]) <= 0.050
            and abs(center_local[1]) <= aperture / 2.0 + 0.014 + 0.006
            and -0.070 <= center_local[2] <= -0.014
        )
        if inside_gripper:
            state.outside_gripper_time = 0.0
        else:
            state.outside_gripper_time += timestep

        # 瞬时误差或瞬时越界不立即撤销代理；必须连续保持0.15秒。
        if state.large_error_time >= self.config.grasp_break_hold_seconds:
            self._clear_grasp_with_reason(
                "grasp_proxy_error_exceeded", grasp_error=error_norm
            )
        elif state.outside_gripper_time >= self.config.grasp_break_hold_seconds:
            self._clear_grasp_with_reason(
                "cable_exited_gripper", grasp_error=error_norm
            )
        return error_norm

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

    def _begin_grasp_attempt(self) -> None:
        """在闭爪开始时记录已经位于张开夹爪内部的线缆节点。"""
        # 新的闭爪边沿代表新的抓取尝试，不能沿用上一次尝试的断裂原因。
        self.last_grasp_break = None
        # 这是任务/代理资格快照，不会改写机械臂动作，也不会改变普通碰撞。
        rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        reference = self.hand_position
        eligible: set[int] = set()
        for body_id in self.cable_ids:
            local = rotation.T @ (self.data.xpos[body_id] - reference)
            if (
                abs(local[0]) <= 0.040
                and abs(local[1]) <= 0.050
                and -0.070 <= local[2] <= -0.010
            ):
                eligible.add(body_id)
        # 即使线缆穿过同一指垫区域，MuJoCo 也可能把接触报告在相邻20 mm胶囊上，
        # 因此将左右各一个邻居也加入资格集合。
        expanded = set(eligible)
        for body_id in eligible:
            index = self.cable_ids.index(body_id)
            expanded.update(
                self.cable_ids[max(0, index - 1):min(len(self.cable_ids), index + 2)]
            )
        self._capture_eligible = expanded

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

        # 只在“张开 -> 闭合”的边沿记录一次抓取资格；一直闭着横扫不会刷新资格。
        gripper_closed = bool(clipped_action[7] < 100.0)
        if gripper_closed and not self._gripper_was_closed:
            self._begin_grasp_attempt()
        elif not gripper_closed:
            self._capture_eligible.clear()

        for _ in range(max(1, self.config.frame_skip)):
            # 每个物理子步先清空外力，再重新计算线缆扰动和已确认的夹持力。
            self.data.xfrc_applied[:] = 0.0
            self._apply_cable_disturbance()  # 扰动强度永远不随夹爪命令缩放

            if not gripper_closed:
                self._clear_grasp_with_reason("gripper_command_open")
            elif self.grasp_state is None:
                self._try_capture_grasp()
            if gripper_closed and self.grasp_confirmed:
                self._apply_compliant_grasp()

            # 真正把当前方法/模型给出的动作交给MuJoCo执行。
            self.data.ctrl[:] = clipped_action
            mujoco.mj_step(self.model, self.data)
            self._update_grasp_confirmation()

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

        self._gripper_was_closed = gripper_closed

        success = self.ever_success
        info = self.info()
        reward = float(last_qualification) + 5.0 * float(success)
        return self.observation(), reward, success, truncated, info

    def _current_grasp_error(self) -> float:
        if self.grasp_state is None:
            return math.inf
        rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        errors = []
        for body_id, offset in zip(
            self.grasp_state.patch_body_ids,
            self.grasp_state.patch_local_offsets,
        ):
            expected = self.hand_position + rotation @ offset
            errors.append(float(np.sum((expected - self.data.xpos[body_id]) ** 2)))
        return math.sqrt(sum(errors) / len(errors))

    def _success_qualification(self, gripper_closed: bool, grasp_error: float) -> bool:
        """单个物理时刻是否满足成功条件；还需连续保持0.55秒才最终成功。"""

        if not gripper_closed or self.grasp_state is None:
            return False
        body_id = self.grasp_state.body_id
        cable_z = self.data.xpos[self.cable_ids, 2]
        lifted_fraction = float(np.mean(cable_z > 0.055))
        return (
            self.data.xpos[body_id, 2] > 0.14
            and lifted_fraction >= 0.18
            and grasp_error < 0.045
            and np.linalg.norm(self.data.xpos[body_id] - self.hand_position) < 0.085
            and self.grasp_state.bilateral_confirmed
            and self.grasp_state.capture_finger_count == 2
            and float(np.sum(self.data.qpos[self.finger_qpos_adr])) > 0.018
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
            rotation = self.data.xmat[self.hand_id].reshape(3, 3)
            grasp_error = self._current_grasp_error()
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
                0 if self.grasp_state is None else len(self.grasp_state.patch_body_ids)
            ),
            "finger_aperture": float(np.sum(self.data.qpos[self.finger_qpos_adr])),
            "grasp_attempt_eligible_count": len(self._capture_eligible),
            "finger_contact_count": self._last_contact_count,
            "grasp_error": grasp_error,
            "cable_com": cable_positions.mean(axis=0).copy(),
            "max_z": float(cable_positions[:, 2].max()),
            "lifted_fraction": float(np.mean(cable_positions[:, 2] > 0.055)),
            "success_hold": self.success_hold,
            "success": self.ever_success,
        }
