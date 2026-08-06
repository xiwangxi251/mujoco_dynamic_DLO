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
    lift_seconds: float = 2.4
    carry_seconds: float = 2.5
    hold_seconds: float = 2.0
    failure_observe_seconds: float = 0.6
    release_seconds: float = 1.2
    max_retries: int = 2


class DynamicCableGraspPolicy:
    """预测、截获并抓取一个观测到的运动线缆段。

    该类只读取环境状态并输出执行器命令，绝不缩放外力或修改线缆物理。
    """

    # actuator8 将255映射为每根手指张开40 mm，将0映射为完全闭合。
    # 这里请求完全闭合，让真实的线缆—指垫碰撞决定最终可见间隙，而不是预设开口。
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
        self.lift_start = np.zeros(3)
        self.lift_goal = np.zeros(3)
        self.carry_start = np.zeros(3)
        self.carry_goal = np.zeros(3)
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
        self.last_desired = self.env.hand_position.copy()
        self.recover_start = self.last_desired.copy()
        self.recover_goal = self.last_desired.copy()
        self.failure_hold_position = self.last_desired.copy()
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

        position = self.env.target_position()
        velocity = np.clip(self.env.target_velocity(), -0.8, 0.8)
        predicted = position + self.config.prediction_horizon * velocity
        predicted[0] = np.clip(predicted[0], 0.30, 0.82)
        predicted[1] = np.clip(predicted[1], -0.43, 0.43)
        predicted[2] = np.clip(predicted[2], 0.010, 0.18)
        # 对预测值而不是原始测量做低通滤波，既保留快速横向跟踪，也避免把接触抖动送入IK。
        self.filtered_target += 0.10 * (predicted - self.filtered_target)
        return self.filtered_target.copy()

    def _ik_action(self, desired_position: np.ndarray, gripper: float) -> np.ndarray:
        """用阻尼最小二乘逆运动学生成7个关节目标和1个夹爪命令。"""

        model = self.env.model
        data = self.env.data
        point_jac, jac_rot = point_jacobian(
            model, data, self.env.hand_id, self.env.HAND_LOCAL_POINT
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
        self.recover_start = hand.copy()
        self.recover_goal = hand + np.array([0.0, 0.0, 0.18])
        self._transition(Phase.RECOVER)

    def action(self) -> np.ndarray:
        """推进反应式状态机，并返回一个控制周期的动作。"""
        # 无论处于哪个阶段，策略只读取环境状态并返回动作，不直接修改线缆物理。
        target = self._predicted_segment()
        hand = self.env.hand_position

        if self.phase is Phase.SETTLE:
            # 先让线缆自然运动0.8秒，机械臂保持ready姿态。
            if self.phase_time >= self.config.settle_seconds:
                self._transition(Phase.APPROACH)
            return self._home_action()

        if self.phase is Phase.APPROACH:
            # 移动到预测线段上方20 cm，夹爪保持张开。
            desired = target + np.array([0.0, 0.0, 0.20])
            if np.linalg.norm(hand - desired) < 0.035 or self.phase_time > self.config.approach_timeout:
                self._transition(Phase.INTERCEPT)
            return self._ik_action(desired, 255.0)

        if self.phase is Phase.INTERCEPT:
            # HAND_LOCAL_POINT 位于指尖中点；让线缆位于其上方约36 mm，
            # 可以把线缆胶囊放进两块指垫之间。
            desired = target + np.array([0.0, 0.0, -0.036])
            desired[2] = max(desired[2], -0.025)
            distance = np.linalg.norm(hand - desired)
            if distance < 0.018:
                self._transition(Phase.CLOSE)
                return self._ik_action(desired, self.HOLD_GRIPPER_CTRL)
            elif self.phase_time > self.config.intercept_timeout:
                self._begin_vertical_recovery(hand)
                return self._ik_action(self.recover_start, 255.0)
            return self._ik_action(desired, 255.0)

        if self.phase is Phase.CLOSE:
            # 该脚本基线在闭爪期间仍可自由横向跟踪。抓取是否具备资格、是否属于闭爪横扫
            # 补抓，由环境而不是该方法决定。
            desired = target + np.array([0.0, 0.0, -0.036])
            desired[2] = max(desired[2], -0.025)
            if self.env.grasp_confirmed:
                self.lift_start = hand.copy()
                self.lift_goal = hand + np.array([0.0, 0.0, 0.30])
                self._transition(Phase.LIFT)
            elif self.phase_time > self.config.close_timeout:
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
                self._transition(Phase.APPROACH)
            return self._ik_action(desired, 255.0)

        if self.phase is Phase.LIFT:
            # 把夹持点抬高30 cm；若环境报告抓取断开则立即失败释放。
            if self.env.grasp_state is None:
                # 在发送张开命令之前保存环境记录的断裂现场。否则本次动作执行后
                # contacts=0 只说明夹爪已主动打开，不能解释最初的断裂原因。
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
                side = 0.22 if self.env.target_position()[1] <= 0.0 else -0.22
                self.carry_goal = hand + np.array([0.17, side, 0.05])
                self._transition(Phase.CARRY)
            return self._ik_action(desired, self.HOLD_GRIPPER_CTRL)

        if self.phase is Phase.CARRY:
            # 抬升后向侧方搬运，用来验证抓取不是瞬时接触。
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
            if self.phase_time >= self.config.carry_seconds:
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
                self.result = "success" if self.env.ever_success else "failed_success_conditions"
                self._transition(Phase.RELEASE)
            return self._ik_action(self.carry_goal, self.HOLD_GRIPPER_CTRL)

        if self.phase is Phase.FAILURE_OBSERVE:
            # 真正判定滑脱后先保持闭爪，保留0.6秒可观察窗口；不要让主动张开掩盖原因。
            if self.phase_time >= self.config.failure_observe_seconds:
                self._transition(Phase.RELEASE)
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
            f"one_sided={event['one_sided_contact_time']:.3f}s "
            f"no_contact={event['no_contact_time']:.3f}s "
            f"error_hold={event['large_error_time']:.3f}s "
            f"outside_hold={event['outside_gripper_time']:.3f}s "
            f"grasp_error={error_text}/{event['grasp_break_distance']:.4f}m "
            f"aperture={1000.0 * event['finger_aperture']:.1f}mm "
            f"ctrl_before_open={event['gripper_ctrl']:.1f} "
            f"success_before_break={event['ever_success']} "
            f"success_hold={event['success_hold']:.3f}s"
        )
