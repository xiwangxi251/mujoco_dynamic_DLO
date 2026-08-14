"""动态线缆环境使用的反应式截获脚本策略。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math

import numpy as np

from cable_grasp_env import (
    CableGraspEnv,
    point_jacobian,
    quat_error,
    rotation_to_quat,
)


class Phase(Enum):
    """简单脚本基线的阶段；不是环境状态，也不会约束其他模型。"""

    SETTLE = auto()
    APPROACH = auto()
    INTERCEPT = auto()
    CLOSE = auto()
    RECOVER = auto()
    LIFT = auto()
    CARRY = auto()
    HOLD = auto()
    FAILURE_OBSERVE = auto()
    RELEASE = auto()
    DONE = auto()


@dataclass
class PolicyConfig:
    """脚本基线的时间参数。论文方法可完全不用这个类。"""

    prediction_horizon: float = 0.22
    settle_seconds: float = 0.8
    approach_timeout: float = 6.0
    intercept_timeout: float = 9.0
    close_timeout: float = 0.8
    close_capture_distance: float = 0.018
    close_contact_grace: float = 0.35
    lift_seconds: float = 2.4
    lift_distance: float = 0.22
    minimum_post_lift_rise: float = 0.14
    carry_seconds: float = 1.0
    carry_side_distance: float = 0.08
    carry_up_distance: float = 0.02
    hold_seconds: float = 0.0
    failure_observe_seconds: float = 0.6
    release_seconds: float = 1.2
    max_retries: int = 2


class DynamicCableGraspPolicy:
    """预测、截获并抓取一个观测到的运动线缆段。

    该类只读取环境状态并输出执行器命令，绝不缩放外力或修改线缆物理。
    """

    # actuator8 将255映射为每根手指张开40 mm，0为完全闭合。20对0的20-seed
    # 成对消融没有显示减小挤压的稳定收益，反而增加了确认抓取后的终局物理滑脱，
    # 因此脚本基线恢复为完全闭合；最终开口仍由真实碰撞和执行器力范围决定。
    HOLD_GRIPPER_CTRL = 0.0

    def __init__(self, env: CableGraspEnv, config: PolicyConfig | None = None):
        self.env = env
        self.config = config or PolicyConfig()
        self.phase = Phase.SETTLE
        self.phase_start = 0.0
        self.retry_count = 0
        self.finished = False
        self.result = "running"
        self.failure_diagnostics: dict | None = None
        self.filtered_target = np.zeros(3)
        self.locked_segment_index: int | None = None
        self.locked_segment_alpha = 0.0
        self.last_close_contact_time = -math.inf
        self.lift_start = np.zeros(3)
        self.lift_goal = np.zeros(3)
        self.carry_start = np.zeros(3)
        self.carry_goal = np.zeros(3)
        self.hold_position = np.zeros(3)
        self.recover_start = np.zeros(3)
        self.recover_goal = np.zeros(3)
        self.failure_hold_position = np.zeros(3)
        self.max_recover_xy_drift = 0.0
        self.last_desired = np.zeros(3)
        self.desired_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.SETTLE
        self.phase_start = float(self.env.data.time)
        self.retry_count = 0
        self.finished = False
        self.result = "running"
        self.failure_diagnostics = None
        self.filtered_target = self.env.target_position()
        self.locked_segment_index = None
        self.locked_segment_alpha = 0.0
        self.last_close_contact_time = -math.inf
        self.last_desired = self.env.hand_position.copy()
        self.recover_start = self.last_desired.copy()
        self.recover_goal = self.last_desired.copy()
        self.failure_hold_position = self.last_desired.copy()
        self.hold_position = self.last_desired.copy()
        self.max_recover_xy_drift = 0.0
        rotation = self.env.data.xmat[self.env.hand_id].reshape(3, 3)
        # 线缆切向主要沿世界坐标 X，因此把平行夹爪开合方向旋转到 Y。
        z_rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.desired_quat = rotation_to_quat(z_rotation @ rotation)

    @property
    def phase_time(self) -> float:
        return float(self.env.data.time - self.phase_start)

    def _transition(self, phase: Phase) -> None:
        self.phase = phase
        self.phase_start = float(self.env.data.time)

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _predicted_segment(self) -> np.ndarray:
        """用当前位置加短时速度外推，得到脚本要追踪的目标点。"""

        if self.locked_segment_index is None:
            position = self.env.target_position()
            velocity = self.env.target_velocity()
        else:
            index = self.locked_segment_index
            alpha = self.locked_segment_alpha
            body0 = self.env.cable_ids[index]
            body1 = self.env.cable_ids[index + 1]
            position = (
                (1.0 - alpha) * self.env.data.xpos[body0]
                + alpha * self.env.data.xpos[body1]
            )
            velocity = (
                (1.0 - alpha) * self.env.body_linear_velocity(body0)
                + alpha * self.env.body_linear_velocity(body1)
            )
        velocity = np.clip(velocity, -0.8, 0.8)
        predicted = position + self.config.prediction_horizon * velocity
        predicted[0] = np.clip(predicted[0], 0.30, 0.82)
        predicted[1] = np.clip(predicted[1], -0.43, 0.43)
        predicted[2] = np.clip(predicted[2], 0.010, 0.18)
        # 对预测值而不是原始测量做低通滤波，既保留快速横向跟踪，也避免把接触抖动送入IK。
        self.filtered_target += 0.10 * (predicted - self.filtered_target)
        return self.filtered_target.copy()

    def _nearest_cable_point(self, point: np.ndarray) -> tuple[np.ndarray, float, int, float]:
        """返回整条线缆中心线上距给定点最近的位置及其线段参数。"""
        positions = self.env.data.xpos[self.env.cable_ids]
        starts = positions[:-1]
        vectors = positions[1:] - starts
        lengths_squared = np.sum(vectors * vectors, axis=1)
        alpha = np.sum((point - starts) * vectors, axis=1) / np.maximum(
            lengths_squared, 1e-12
        )
        alpha = np.clip(alpha, 0.0, 1.0)
        projected = starts + alpha[:, None] * vectors
        distances = np.linalg.norm(projected - point, axis=1)
        index = int(np.argmin(distances))
        return projected[index].copy(), float(distances[index]), index, float(alpha[index])

    def _lock_segment_near(self, point: np.ndarray) -> np.ndarray:
        """锁定当前进入夹持中心的线段，闭爪后不再追逐原随机目标节点。"""
        nearest, _, index, alpha = self._nearest_cable_point(point)
        self.locked_segment_index = index
        self.locked_segment_alpha = alpha
        self.filtered_target = nearest.copy()
        return nearest

    def _ik_action(self, desired_position: np.ndarray, gripper: float) -> np.ndarray:
        """用阻尼最小二乘逆运动学生成7个关节目标和1个夹爪命令。"""

        model = self.env.model
        data = self.env.data
        point_jac, jac_rot = point_jacobian(
            model, data, self.env.hand_id, self.env.GRASP_CENTER_LOCAL
        )
        rotation = data.xmat[self.env.hand_id].reshape(3, 3)
        position_error = desired_position - self.env.hand_position
        orientation_error = quat_error(rotation_to_quat(rotation), self.desired_quat)
        task_velocity = np.concatenate([
            8.0 * position_error,
            0.9 * orientation_error,
        ])
        # 任务空间误差(位置3维+旋转3维)通过6x7 Jacobian映射为关节速度。
        jacobian = np.vstack([point_jac, jac_rot])[:, self.env.arm_dof_adr]
        damping = 0.045
        q_velocity = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), task_velocity
        )
        q_velocity = np.clip(q_velocity, -3.0, 3.0)
        q_current = data.qpos[self.env.arm_qpos_adr]
        # 位置执行器接收一个短时前瞻目标。关节范围使用 joint id 查询，只有访问
        # data.qpos 时才使用 qpos address。
        q_target = q_current + 0.11 * q_velocity
        q_target = np.clip(
            q_target,
            model.jnt_range[self.env.arm_joint_ids, 0],
            model.jnt_range[self.env.arm_joint_ids, 1],
        )
        action = np.empty(8)
        action[:7] = q_target
        action[7] = gripper
        self.last_desired = desired_position.copy()
        return action

    def _home_action(self) -> np.ndarray:
        return self.env.ready_ctrl.copy()

    def _begin_vertical_recovery(self, hand: np.ndarray) -> None:
        """张开夹爪并竖直撤离，然后才开始下一次横向跟踪。"""
        self.locked_segment_index = None
        self.locked_segment_alpha = 0.0
        self.recover_start = hand.copy()
        self.recover_goal = hand + np.array([0.0, 0.0, 0.18])
        self._transition(Phase.RECOVER)

    def action(self) -> np.ndarray:
        """推进反应式状态机，并返回一个控制周期的动作。"""
        # 无论处于哪个阶段，策略只读取环境状态并返回动作，不直接修改线缆物理。
        hand = self.env.hand_position
        # 一旦真实指垫接触产生候选，就锁定实际进入夹爪的局部线段；否则继续追踪
        # 回合开始时选择的参考目标。这样不会夹到线后仍被远处目标节点拉走。
        if self.phase is Phase.CLOSE and self.env.grasp_state is not None:
            self._lock_segment_near(self.env.data.xpos[self.env.grasp_state.body_id])
        target = self._predicted_segment()

        if self.phase is Phase.SETTLE:
            # 先让线缆自然运动0.8秒，机械臂保持ready姿态。
            if self.phase_time >= self.config.settle_seconds:
                self._transition(Phase.APPROACH)
            return self._home_action()

        if self.phase is Phase.APPROACH:
            # 让实际两指夹持中心移动到预测线段上方20 cm，夹爪保持张开。
            desired = target + np.array([0.0, 0.0, 0.20])
            if np.linalg.norm(hand - desired) < 0.035 or self.phase_time > self.config.approach_timeout:
                self._transition(Phase.INTERCEPT)
            return self._ik_action(desired, 255.0)

        if self.phase is Phase.INTERCEPT:
            # 实际两指夹持中心直接追踪目标线缆段中心，不再使用旧虚拟点的z补偿。
            desired = target.copy()
            nearest, nearest_distance, _, _ = self._nearest_cable_point(hand)
            # 原参考节点可能已因弯折远离，但只要任意真实线缆中心线进入夹持中心，
            # 就应闭爪并锁定该线段，不能继续等待远处节点。
            if nearest_distance < self.config.close_capture_distance:
                desired = self._lock_segment_near(nearest)
                self._transition(Phase.CLOSE)
                self.last_close_contact_time = float(self.env.data.time)
                return self._ik_action(desired, self.HOLD_GRIPPER_CTRL)
            elif self.phase_time > self.config.intercept_timeout:
                self._begin_vertical_recovery(hand)
                return self._ik_action(self.recover_start, 255.0)
            return self._ik_action(desired, 255.0)

        if self.phase is Phase.CLOSE:
            # 闭爪时继续追踪已经进入夹持中心的局部线段；真实接触存在时延长确认窗口，
            # 避免固定0.8秒超时在夹爪仍夹着线缆时主动张开。
            desired = target.copy()
            if self.env.finger_contacts():
                self.last_close_contact_time = float(self.env.data.time)
            if self.env.grasp_confirmed:
                self._lock_segment_near(hand)
                self.lift_start = hand.copy()
                self.lift_goal = hand + np.array([0.0, 0.0, self.config.lift_distance])
                self._transition(Phase.LIFT)
            elif (
                self.phase_time > self.config.close_timeout
                and self.env.data.time - self.last_close_contact_time
                > self.config.close_contact_grace
            ):
                if self.retry_count < self.config.max_retries:
                    self.retry_count += 1
                    self._begin_vertical_recovery(hand)
                    return self._ik_action(hand, 255.0)
                else:
                    self.result = "failed_no_contact"
                    self._transition(Phase.RELEASE)
                    return self._ik_action(hand, 255.0)
            return self._ik_action(desired, self.HOLD_GRIPPER_CTRL)

        if self.phase is Phase.RECOVER:
            # 抓空后张开夹爪并抬高，再重新进入APPROACH。
            self.max_recover_xy_drift = max(
                self.max_recover_xy_drift,
                float(np.linalg.norm(hand[:2] - self.recover_start[:2])),
            )
            blend = self._smoothstep(self.phase_time / 1.0)
            desired = (1.0 - blend) * self.recover_start + blend * self.recover_goal
            if self.phase_time > 1.0:
                self.filtered_target = self.env.target_position()
                self._transition(Phase.APPROACH)
            return self._ik_action(desired, 255.0)

        if self.phase is Phase.LIFT:
            # 把夹持点抬高22 cm；若环境报告抓取断开则保持闭爪记录失败现场。
            if self.env.grasp_state is None:
                # 保存环境记录的断裂现场；后续不主动张开，以免掩盖最初原因。
                self.failure_diagnostics = (
                    None if self.env.last_grasp_break is None
                    else self.env.last_grasp_break.copy()
                )
                self.result = "failed_grasp_broke_on_lift"
                self.failure_hold_position = hand.copy()
                self._transition(Phase.FAILURE_OBSERVE)
                return self._ik_action(hand, self.HOLD_GRIPPER_CTRL)
            blend = self._smoothstep(self.phase_time / self.config.lift_seconds)
            desired = (1.0 - blend) * self.lift_start + blend * self.lift_goal
            if self.phase_time >= self.config.lift_seconds:
                self.carry_start = hand.copy()
                side = (
                    self.config.carry_side_distance
                    if hand[1] <= 0.0
                    else -self.config.carry_side_distance
                )
                self.carry_goal = hand + np.array([
                    0.0,
                    side,
                    self.config.carry_up_distance,
                ])
                self._transition(Phase.CARRY)
            return self._ik_action(desired, self.HOLD_GRIPPER_CTRL)

        if self.phase is Phase.CARRY:
            # 抬升后朝桌面中心侧移并小幅上抬，用来验证抓取不是瞬时接触。
            if self.env.grasp_state is None:
                self.failure_diagnostics = (
                    None if self.env.last_grasp_break is None
                    else self.env.last_grasp_break.copy()
                )
                self.result = "failed_grasp_broke_on_carry"
                self.failure_hold_position = hand.copy()
                self._transition(Phase.FAILURE_OBSERVE)
                return self._ik_action(hand, self.HOLD_GRIPPER_CTRL)
            blend = self._smoothstep(self.phase_time / self.config.carry_seconds)
            desired = (1.0 - blend) * self.carry_start + blend * self.carry_goal
            desired[2] = max(
                desired[2],
                self.lift_start[2] + self.config.minimum_post_lift_rise,
            )
            if self.phase_time >= self.config.carry_seconds:
                # HOLD 应保持实际已经到达的位置，不能继续追逐存在IK残差的旧目标。
                self.hold_position = hand.copy()
                self._transition(Phase.HOLD)
            return self._ik_action(desired, self.HOLD_GRIPPER_CTRL)

        if self.phase is Phase.HOLD:
            # 在目标位置保持，环境成功条件需要连续满足规定时长。
            if self.env.grasp_state is None:
                self.failure_diagnostics = (
                    None if self.env.last_grasp_break is None
                    else self.env.last_grasp_break.copy()
                )
                self.result = "failed_grasp_broke_on_hold"
                self.failure_hold_position = hand.copy()
                self._transition(Phase.FAILURE_OBSERVE)
            elif self.phase_time >= self.config.hold_seconds:
                if self.env.success_hold >= self.env.config.success_hold_seconds:
                    self.result = "success"
                    # 成功录像停在闭爪保持状态；不再主动释放后把正常掉落误看成滑脱。
                    self._transition(Phase.DONE)
                    self.finished = True
                    # 调用方仍会执行action()返回值一次；原地闭爪可避免最后一步继续
                    # 追踪而离开刚刚满足的当前举升状态。
                    return self._ik_action(hand.copy(), self.HOLD_GRIPPER_CTRL)
                # 物理抓取仍在但旧成功条件尚未满足时继续闭爪保持；回合上限负责
                # 最终停止。不能因为固定计时器到点就主动放开一个仍在夹持的线缆。
            desired = self.hold_position.copy()
            desired[2] = max(
                desired[2],
                self.lift_start[2] + self.config.minimum_post_lift_rise,
            )
            return self._ik_action(desired, self.HOLD_GRIPPER_CTRL)

        if self.phase is Phase.FAILURE_OBSERVE:
            # 真正判定滑脱后保持闭爪并结束本轮，不再主动张开掩盖最初的滑脱原因。
            if self.phase_time >= self.config.failure_observe_seconds:
                self._transition(Phase.DONE)
                self.finished = True
            return self._ik_action(self.failure_hold_position, self.HOLD_GRIPPER_CTRL)

        if self.phase is Phase.RELEASE:
            # 张开夹爪并保留当前手部目标，让线缆自然落下。
            desired = self.last_desired.copy()
            if self.phase_time >= self.config.release_seconds:
                self._transition(Phase.DONE)
                self.finished = True
            return self._ik_action(desired, 255.0)

        self.finished = True
        return self._home_action()

    def summary(self) -> str:
        info = self.env.info()
        decisive = self.env.success_snapshot or info
        error = decisive["grasp_error"]
        error_text = "none" if not math.isfinite(error) else f"{error:.4f}m"
        summary = (
            f"trial={info['trial']} result={self.result} sim_time={self.env.data.time:.3f}s "
            f"target_body={info['target_body_id']} grasped_body={decisive['grasped_body_id']} "
            f"bilateral={decisive['bilateral_grasp']} "
            f"aperture={1000.0 * decisive['finger_aperture']:.1f}mm "
            f"contacts_at_success={decisive['finger_contact_count']} grasp_error={error_text} "
            f"recover_xy_drift={1000.0 * self.max_recover_xy_drift:.1f}mm "
            f"lifted_fraction={decisive['lifted_fraction']:.2f} max_z={decisive['max_z']:.3f}m "
            f"success_hold={decisive['success_hold']:.3f}s"
        )
        diagnostics = self.break_diagnostics_text()
        return summary if diagnostics is None else f"{summary}\n  {diagnostics}"

    def break_diagnostics_text(self) -> str | None:
        """格式化主动张开夹爪之前保存的抓取断裂现场。"""
        event = self.failure_diagnostics
        if event is None:
            if "grasp_broke" not in self.result:
                return None
            return "grasp_break reason=missing_break_record"
        error = event["grasp_error"]
        error_text = "none" if not math.isfinite(error) else f"{error:.4f}m"
        return (
            f"grasp_break reason={event['reason']} break_time={event['time']:.3f}s "
            f"contacts_before_open={event['raw_contact_count']} "
            f"unique_nodes={event['unique_contact_nodes']} "
            f"contacting_fingers={event['contacting_finger_count']}/2 "
            f"lost_bilateral={event['lost_bilateral_seconds']:.3f}s "
            f"no_contact={event['no_contact_time']:.3f}s "
            f"grasp_error={error_text} "
            f"aperture={1000.0 * event['finger_aperture']:.1f}mm "
            f"ctrl_before_open={event['gripper_ctrl']:.1f} "
            f"success_before_break={event['ever_success']} "
            f"success_hold={event['success_hold']:.3f}s"
        )
