"""动态线缆环境使用的反应式截获脚本策略。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math

import numpy as np

from ..env.environment import CableGraspEnv
from ..env.kinematics import point_jacobian, quat_error, rotation_to_quat


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

    # 所有场景直接使用目标节点总速度预测，不按场景或运动分量采用不同算法。
    # 0.30 s作为当前统一测试值；闭爪时仍使用更保守的0.12 s短时预测。
    prediction_horizon: float = 0.30
    approach_prediction_horizon: float = 0.30
    close_prediction_horizon: float = 0.12
    target_filter_alpha: float = 0.10
    intercept_x_limits: tuple[float, float] = (0.30, 0.82)
    intercept_y_limits: tuple[float, float] = (-0.43, 0.43)
    intercept_z_limits: tuple[float, float] = (0.010, 0.18)
    settle_seconds: float = 0.8
    approach_timeout: float = 6.0
    # APPROACH只负责到达线缆上方；匀速L1目标存在约5 cm稳态跟踪滞后，随后由
    # INTERCEPT完成精确下降。该门限仍远小于旧故障中的20--27 cm强制下降误差。
    approach_position_tolerance: float = 0.065
    approach_tilt_tolerance: float = math.radians(25.0)
    intercept_tilt_limit: float = math.radians(35.0)
    intercept_singularity_limit: float = 0.045
    # APPROACH远离目标时优先追赶位置；进入目标上方后再恢复完整姿态权重。
    # 最终速度仍由环境端统一限幅，较长的IK目标时域只避免策略命令过弱。
    approach_fast_distance: float = 0.12
    approach_linear_velocity_limit: float = 0.90
    intercept_linear_velocity_limit: float = 0.95
    precision_linear_velocity_limit: float = 0.65
    approach_orientation_gain: float = 0.65
    precision_orientation_gain: float = 1.0
    # Experimental ablation: make the complete vertical grasp orientation the
    # primary IK task and solve translation only in its nullspace.
    strict_vertical_gripper: bool = False
    strict_vertical_tolerance: float = math.radians(5.0)
    policy_joint_velocity_fraction: float = 1.0
    ik_target_horizon: float = 0.11
    intercept_timeout: float = 9.0
    close_timeout: float = 0.8
    close_hard_timeout: float = 3.0
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
    # Formal evaluation gives every method the same episode time budget.  A
    # fixed retry cap made scripted/expert give up while PPO and DynamicVLA
    # could still act, so retries are unlimited by default and bounded by the
    # environment time limit.  Set this to 0 for single-attempt diagnostics.
    max_retries: int | None = None
    retry_min_remaining_seconds: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "prediction_horizon", "approach_prediction_horizon",
            "close_prediction_horizon",
            "approach_position_tolerance",
            "approach_tilt_tolerance", "intercept_tilt_limit",
            "intercept_singularity_limit", "approach_fast_distance",
            "approach_linear_velocity_limit", "intercept_linear_velocity_limit",
            "precision_linear_velocity_limit", "policy_joint_velocity_fraction",
            "approach_orientation_gain", "precision_orientation_gain",
            "ik_target_horizon", "strict_vertical_tolerance",
            "retry_min_remaining_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not math.isfinite(self.target_filter_alpha)
            or not 0.0 < self.target_filter_alpha <= 1.0
        ):
            raise ValueError("target_filter_alpha must be in (0, 1]")
        if not 0.0 < self.policy_joint_velocity_fraction <= 1.0:
            raise ValueError("policy_joint_velocity_fraction must be in (0, 1]")
        if not isinstance(self.strict_vertical_gripper, bool):
            raise ValueError("strict_vertical_gripper must be boolean")
        if (
            self.max_retries is not None
            and (
                isinstance(self.max_retries, bool)
                or not isinstance(self.max_retries, int)
                or self.max_retries < 0
            )
        ):
            raise ValueError("max_retries must be a non-negative integer or None")
        for name in (
            "intercept_x_limits", "intercept_y_limits", "intercept_z_limits",
        ):
            limits = np.asarray(getattr(self, name), dtype=float)
            if (
                limits.shape != (2,)
                or not np.all(np.isfinite(limits))
                or limits[0] >= limits[1]
            ):
                raise ValueError(
                    f"{name} must contain two finite increasing values"
                )


class DynamicCableGraspPolicy:
    """预测、截获并抓取一个观测到的运动线缆段。

    该类只读取环境状态并输出执行器命令，绝不缩放外力或修改线缆物理。
    """

    # Panda actuator8 and NERO's gripper actuator both use the environment's
    # native open/closed command values.  The environment resolves the scale.
    # 成对消融没有显示减小挤压的稳定收益，反而增加了确认抓取后的终局物理滑脱，
    # 因此脚本基线恢复为完全闭合；最终开口仍由真实碰撞和执行器力范围决定。
    HOLD_GRIPPER_CTRL = 0.0
    # Columns are the desired hand-local x/y/z axes in world coordinates.
    # Local z points straight down and local y is the finger closing axis.
    VERTICAL_GRASP_ROTATION = np.diag([-1.0, 1.0, -1.0])

    def __init__(self, env: CableGraspEnv, config: PolicyConfig | None = None):
        self.env = env
        self.config = config or PolicyConfig()
        self.phase = Phase.SETTLE
        self.phase_start = 0.0
        self.retry_count = 0
        self.attempt_failure_count = 0
        self.last_attempt_failure: str | None = None
        self._retry_after_failure_observe = False
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
        self.tilt_error_sum = 0.0
        self.tilt_error_samples = 0
        self.max_tilt_error = 0.0
        self.grasp_tilt_error_sum = 0.0
        self.grasp_tilt_error_samples = 0
        self.max_grasp_tilt_error = 0.0
        self.last_desired = np.zeros(3)
        self.desired_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.desired_approach_axis = np.array([0.0, 0.0, 1.0])
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.SETTLE
        self.phase_start = float(self.env.data.time)
        self.retry_count = 0
        self.attempt_failure_count = 0
        self.last_attempt_failure = None
        self._retry_after_failure_observe = False
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
        self.tilt_error_sum = 0.0
        self.tilt_error_samples = 0
        self.max_tilt_error = 0.0
        self.grasp_tilt_error_sum = 0.0
        self.grasp_tilt_error_samples = 0
        self.max_grasp_tilt_error = 0.0
        rotation = self.env.data.xmat[self.env.hand_id].reshape(3, 3)
        # 线缆切向主要沿世界坐标 X，因此把平行夹爪开合方向旋转到 Y。
        z_rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        if self.env.robot == "nero":
            # NERO's jaw closing and tool approach axes differ from Panda's
            # link frame, so use the calibrated robot-frame convention.
            desired_rotation = self.env.vertical_grasp_rotation()
        else:
            desired_rotation = (
                self.VERTICAL_GRASP_ROTATION.copy()
                if self.config.strict_vertical_gripper
                else z_rotation @ rotation
            )
        self.desired_quat = rotation_to_quat(desired_rotation)
        self.desired_approach_axis = (
            desired_rotation @ self.env.gripper_approach_axis_local
        )

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

    def _predicted_segment(
        self, prediction_horizon: float | None = None,
    ) -> np.ndarray:
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
        if prediction_horizon is None:
            if self.phase in {Phase.SETTLE, Phase.APPROACH}:
                prediction_horizon = self.config.approach_prediction_horizon
            elif self.phase is Phase.CLOSE:
                prediction_horizon = self.config.close_prediction_horizon
            else:
                prediction_horizon = self.config.prediction_horizon
        predicted = position + prediction_horizon * velocity
        predicted[0] = np.clip(predicted[0], *self.config.intercept_x_limits)
        predicted[1] = np.clip(predicted[1], *self.config.intercept_y_limits)
        predicted[2] = np.clip(predicted[2], *self.config.intercept_z_limits)
        # 对预测值而不是原始测量做低通滤波，既保留快速横向跟踪，也避免把接触抖动送入IK。
        self.filtered_target += self.config.target_filter_alpha * (
            predicted - self.filtered_target
        )
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

    @staticmethod
    def _limit_vector_norm(vector: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= limit:
            return vector
        return vector * (limit / norm)

    def _orientation_error(self) -> tuple[np.ndarray, float]:
        rotation = self.env.data.xmat[self.env.hand_id].reshape(3, 3)
        error = quat_error(rotation_to_quat(rotation), self.desired_quat)
        # quat_error返回2*sin(theta/2)*axis；这里恢复真实转角用于阶段门控。
        angle = 2.0 * math.asin(float(np.clip(0.5 * np.linalg.norm(error), 0.0, 1.0)))
        return error, angle

    def _tilt_error(self) -> float:
        """返回夹爪接近轴相对安全竖直方向的倾斜角，不把平面内偏航算作横倒。"""

        rotation = self.env.data.xmat[self.env.hand_id].reshape(3, 3)
        actual_axis = rotation @ self.env.gripper_approach_axis_local
        cosine = float(np.clip(
            np.dot(actual_axis, self.desired_approach_axis), -1.0, 1.0
        ))
        return math.acos(cosine)

    def _task_jacobian(self) -> np.ndarray:
        point_jac, jac_rot = point_jacobian(
            self.env.model,
            self.env.data,
            self.env.hand_id,
            self.env.GRASP_CENTER_LOCAL,
        )
        return np.vstack([point_jac, jac_rot])[:, self.env.arm_dof_adr]

    def _minimum_task_singular_value(self) -> float:
        return float(np.linalg.svd(self._task_jacobian(), compute_uv=False)[-1])

    def _retry_available(self) -> bool:
        """Return whether another complete attempt still fits this episode."""

        remaining = float(self.env.config.episode_seconds - self.env.data.time)
        within_count_limit = (
            self.config.max_retries is None
            or self.retry_count < self.config.max_retries
        )
        return bool(
            within_count_limit
            and remaining >= self.config.retry_min_remaining_seconds
        )

    def _retry_from_unsafe_pose(self, hand: np.ndarray) -> np.ndarray:
        """Record a failed attempt, then recover whenever budget remains."""

        self.attempt_failure_count += 1
        self.last_attempt_failure = "failed_no_contact"
        if self._retry_available():
            self.retry_count += 1
            self.result = "running"
            self._begin_vertical_recovery(hand)
            return self._ik_action(self.recover_start, self.env.gripper_open_ctrl)
        self.result = "failed_no_contact"
        self._transition(Phase.RELEASE)
        return self._ik_action(hand, self.env.gripper_open_ctrl)

    def _observe_grasp_break(
        self,
        hand: np.ndarray,
        failure_result: str,
    ) -> np.ndarray:
        """Freeze one confirmed break, then retry if the episode permits it."""

        self.failure_diagnostics = (
            None if self.env.last_grasp_break is None
            else self.env.last_grasp_break.copy()
        )
        self.attempt_failure_count += 1
        self.last_attempt_failure = failure_result
        self.failure_hold_position = hand.copy()
        self._retry_after_failure_observe = self._retry_available()
        if self._retry_after_failure_observe:
            self.retry_count += 1
            self.result = "running"
        else:
            self.result = failure_result
        self._transition(Phase.FAILURE_OBSERVE)
        return self._ik_action(hand, self.HOLD_GRIPPER_CTRL)

    def _ik_action(self, desired_position: np.ndarray, gripper: float) -> np.ndarray:
        """Use the selected hierarchical damped IK ordering."""

        model = self.env.model
        data = self.env.data
        position_error = desired_position - self.env.hand_position
        orientation_error, _ = self._orientation_error()
        position_error_norm = float(np.linalg.norm(position_error))
        fast_approach = bool(
            self.phase is Phase.APPROACH
            and position_error_norm > self.config.approach_fast_distance
        )
        linear_velocity_limit = (
            self.config.approach_linear_velocity_limit
            if fast_approach
            else (
                self.config.intercept_linear_velocity_limit
                if self.phase is Phase.INTERCEPT
                else self.config.precision_linear_velocity_limit
            )
        )
        linear_velocity = self._limit_vector_norm(
            6.0 * position_error, linear_velocity_limit
        )
        if (
            self.config.strict_vertical_gripper
            and self._tilt_error() > self.config.strict_vertical_tolerance
        ):
            # Do not trade tilt for progress. Once outside the five-degree
            # operational envelope, hold translation until orientation recovers.
            linear_velocity = np.zeros(3)
        angular_velocity = self._limit_vector_norm(2.5 * orientation_error, 1.40)

        jacobian = self._task_jacobian()
        position_jacobian = jacobian[:3]
        if self.config.strict_vertical_gripper:
            # The full vertical grasp pose is primary. Translation receives the
            # remaining four-DOF nullspace and may lag instead of tilting the hand.
            # An exact pseudoinverse is intentional here: damping the primary
            # inverse couples a requested world-z yaw into roll/pitch residuals,
            # which defeats the strict vertical constraint.
            orientation_jacobian = jacobian[3:]
            orientation_pseudoinverse = np.linalg.pinv(
                orientation_jacobian, rcond=1e-5
            )
            orientation_velocity = orientation_pseudoinverse @ angular_velocity
            orientation_nullspace = (
                np.eye(7)
                - orientation_pseudoinverse @ orientation_jacobian
            )
            secondary_position_jacobian = (
                position_jacobian @ orientation_nullspace
            )
            position_damping = 0.04
            position_inverse = np.linalg.solve(
                secondary_position_jacobian @ secondary_position_jacobian.T
                + position_damping**2 * np.eye(3),
                np.eye(3),
            )
            position_residual = (
                linear_velocity - position_jacobian @ orientation_velocity
            )
            position_velocity = (
                secondary_position_jacobian.T
                @ position_inverse
                @ position_residual
            )
            q_velocity = orientation_velocity + position_velocity
        else:
            # Baseline: position is primary and orientation uses only its nullspace.
            position_damping = 0.04
            position_inverse = np.linalg.solve(
                position_jacobian @ position_jacobian.T
                + position_damping**2 * np.eye(3),
                np.eye(3),
            )
            position_pseudoinverse = position_jacobian.T @ position_inverse
            position_velocity = position_pseudoinverse @ linear_velocity
            position_nullspace = np.eye(7) - np.linalg.pinv(
                position_jacobian, rcond=1e-5
            ) @ position_jacobian

            orientation_gain = (
                self.config.approach_orientation_gain
                if fast_approach
                else self.config.precision_orientation_gain
            )
            orientation_jacobian = jacobian[3:] @ position_nullspace
            orientation_sigma_min = float(
                np.linalg.svd(orientation_jacobian, compute_uv=False)[-1]
            )
            orientation_singularity = float(np.clip(
                (0.10 - orientation_sigma_min) / 0.10, 0.0, 1.0
            ))
            orientation_damping = (
                0.04 + 0.12 * orientation_singularity * orientation_singularity
            )
            orientation_inverse = np.linalg.solve(
                orientation_jacobian @ orientation_jacobian.T
                + orientation_damping**2 * np.eye(3),
                np.eye(3),
            )
            orientation_residual = (
                angular_velocity - jacobian[3:] @ position_velocity
            )
            orientation_velocity = (
                orientation_jacobian.T
                @ orientation_inverse
                @ orientation_residual
            )
            q_velocity = (
                position_velocity + orientation_gain * orientation_velocity
            )

        # 最后的冗余自由度才用于回到ready姿态，避免肘部任意翻转和逼近关节限位；
        # 使用完整任务的精确零空间，不能污染上面的手部位置和姿态任务。
        q_current = data.qpos[self.env.arm_qpos_adr]
        task_nullspace = np.eye(7) - np.linalg.pinv(
            jacobian, rcond=1e-5
        ) @ jacobian
        q_velocity += task_nullspace @ (
            0.8 * (self.env.ready_qpos[:7] - q_current)
        )

        # 整体缩放而非逐关节裁剪，保留IK求出的多关节运动方向。
        velocity_limits = self.config.policy_joint_velocity_fraction * np.asarray(
            self.env.config.arm_joint_velocity_limits, dtype=float
        )
        velocity_scale = min(
            1.0,
            float(np.min(
                velocity_limits / np.maximum(np.abs(q_velocity), 1e-12)
            )),
        )
        q_velocity *= velocity_scale
        # 位置执行器接收一个短时前瞻目标。关节范围使用 joint id 查询，只有访问
        # data.qpos 时才使用 qpos address。
        q_target = q_current + self.config.ik_target_horizon * q_velocity
        q_target = np.clip(
            q_target,
            model.jnt_range[self.env.arm_joint_ids, 0],
            model.jnt_range[self.env.arm_joint_ids, 1],
        )
        action = np.empty(8)
        action[:7] = q_target
        action[self.env.gripper_actuator_id] = gripper
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

    def _close_capture_distance(self) -> float:
        """Distance at which descent transitions to gripper closure."""

        return self.config.close_capture_distance

    def action(self) -> np.ndarray:
        """推进反应式状态机，并返回一个控制周期的动作。"""
        # 无论处于哪个阶段，策略只读取环境状态并返回动作，不直接修改线缆物理。
        tilt_error = self._tilt_error()
        self.tilt_error_sum += tilt_error
        self.tilt_error_samples += 1
        self.max_tilt_error = max(self.max_tilt_error, tilt_error)
        if self.phase in {Phase.INTERCEPT, Phase.CLOSE}:
            self.grasp_tilt_error_sum += tilt_error
            self.grasp_tilt_error_samples += 1
            self.max_grasp_tilt_error = max(
                self.max_grasp_tilt_error, tilt_error
            )
        hand = self.env.hand_position
        # 一旦真实指垫接触产生候选，就锁定实际进入夹爪的局部线段；否则继续追踪
        # 回合开始时选择的参考目标。这样不会夹到线后仍被远处目标节点拉走。
        if self.phase is Phase.CLOSE and self.env.grasp_state is not None:
            self._lock_segment_near(self.env.data.xpos[self.env.grasp_state.body_id])
        close_contacts = (
            self.env.finger_contacts() if self.phase is Phase.CLOSE else []
        )
        if close_contacts:
            # 单侧指垫先接触时也应保持当前线段位于指间；这里仅改变机器人目标，
            # 不会撤掉环境运动驱动，也不把单侧接触当作确认抓取。
            self.filtered_target = self._lock_segment_near(hand)
        target = self._predicted_segment(
            prediction_horizon=0.0 if close_contacts else None
        )

        if self.phase is Phase.SETTLE:
            # 线缆自然运动期间保持夹持中心位置，同时先完成夹爪朝向对齐。
            if self.phase_time >= self.config.settle_seconds:
                self._transition(Phase.APPROACH)
            return self._ik_action(self.last_desired, self.env.gripper_open_ctrl)

        if self.phase is Phase.APPROACH:
            # 让实际两指夹持中心移动到预测线段上方20 cm，夹爪保持张开。
            desired = target + np.array([0.0, 0.0, 0.20])
            tilt_angle = self._tilt_error()
            position_ready = (
                np.linalg.norm(hand - desired)
                < self.config.approach_position_tolerance
            )
            tilt_tolerance = (
                self.config.strict_vertical_tolerance
                if self.config.strict_vertical_gripper
                else self.config.approach_tilt_tolerance
            )
            tilt_ready = tilt_angle < tilt_tolerance
            if position_ready and tilt_ready:
                # Select one material segment when descent begins.  Re-selecting
                # the globally nearest point every control step makes the goal
                # jump between adjacent folds in a deforming cable.
                self._lock_segment_near(hand)
                self._transition(Phase.INTERCEPT)
            elif self.phase_time > self.config.approach_timeout:
                return self._retry_from_unsafe_pose(hand)
            return self._ik_action(desired, self.env.gripper_open_ctrl)

        if self.phase is Phase.INTERCEPT:
            # 实际两指夹持中心直接追踪目标线缆段中心，不再使用旧虚拟点的z补偿。
            desired = target.copy()
            tilt_angle = self._tilt_error()
            if (
                tilt_angle > self.config.intercept_tilt_limit
                or self._minimum_task_singular_value()
                < self.config.intercept_singularity_limit
            ):
                return self._retry_from_unsafe_pose(hand)
            nearest, nearest_distance, _, _ = self._nearest_cable_point(hand)
            # 截获期间持续追踪进入该阶段时锁定的材料线段，避免在相邻弯折间
            # 跳变；但闭爪触发仍以任意真实线缆中心线进入夹持区域为准。
            vertical_ready = (
                not self.config.strict_vertical_gripper
                or tilt_angle < self.config.strict_vertical_tolerance
            )
            if (
                nearest_distance < self._close_capture_distance()
                and vertical_ready
            ):
                desired = self._lock_segment_near(nearest)
                self.filtered_target = desired.copy()
                self._transition(Phase.CLOSE)
                self.last_close_contact_time = float(self.env.data.time)
                return self._ik_action(desired, self.HOLD_GRIPPER_CTRL)
            elif self.phase_time > self.config.intercept_timeout:
                return self._retry_from_unsafe_pose(hand)
            return self._ik_action(desired, self.env.gripper_open_ctrl)

        if self.phase is Phase.CLOSE:
            # 闭爪时继续追踪已经进入夹持中心的局部线段；真实接触存在时延长确认窗口，
            # 避免固定0.8秒超时在夹爪仍夹着线缆时主动张开。
            desired = target.copy()
            close_contacts = self.env.finger_contacts()
            if close_contacts:
                self.last_close_contact_time = float(self.env.data.time)
            if self.env.grasp_confirmed:
                self._lock_segment_near(hand)
                self.lift_start = hand.copy()
                self.lift_goal = hand + np.array([0.0, 0.0, self.config.lift_distance])
                self._transition(Phase.LIFT)
            else:
                # Never actively open while the simulator still reports pad
                # contact or an in-progress grasp candidate.  Previously the
                # unconditional hard timeout overrode this evidence at 3 s,
                # producing the visible "grasp, open, lift away" failure.
                contact_evidence = bool(
                    close_contacts or self.env.grasp_state is not None
                )
                close_timed_out = (
                    (
                        self.phase_time > self.config.close_hard_timeout
                        and not contact_evidence
                    )
                    or (
                        self.phase_time > self.config.close_timeout
                        and self.env.data.time - self.last_close_contact_time
                        > self.config.close_contact_grace
                        and not contact_evidence
                    )
                )
            if not self.env.grasp_confirmed and close_timed_out:
                return self._retry_from_unsafe_pose(hand)
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
            return self._ik_action(desired, self.env.gripper_open_ctrl)

        if self.phase is Phase.LIFT:
            # 把夹持点抬高22 cm；若环境报告抓取断开则保持闭爪记录失败现场。
            if self.env.grasp_state is None:
                return self._observe_grasp_break(
                    hand, "failed_grasp_broke_on_lift"
                )
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
                return self._observe_grasp_break(
                    hand, "failed_grasp_broke_on_carry"
                )
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
                return self._observe_grasp_break(
                    hand, "failed_grasp_broke_on_hold"
                )
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
            # 先闭爪保留失败现场；环境已保存 break history，因此之后张爪重试
            # 不会覆盖第一次滑脱的物理因果记录。
            if self.phase_time >= self.config.failure_observe_seconds:
                if self._retry_after_failure_observe:
                    self._retry_after_failure_observe = False
                    self._begin_vertical_recovery(hand)
                    return self._ik_action(
                        self.recover_start, self.env.gripper_open_ctrl
                    )
                self._transition(Phase.DONE)
                self.finished = True
            return self._ik_action(self.failure_hold_position, self.HOLD_GRIPPER_CTRL)

        if self.phase is Phase.RELEASE:
            # 张开夹爪并保留当前手部目标，让线缆自然落下。
            desired = self.last_desired.copy()
            if self.phase_time >= self.config.release_seconds:
                self._transition(Phase.DONE)
                self.finished = True
            return self._ik_action(desired, self.env.gripper_open_ctrl)

        self.finished = True
        return self._home_action()

    def policy_info(self) -> dict[str, float | int | bool]:
        """Return orientation-ablation diagnostics for benchmark rows."""

        return {
            "strict_vertical_gripper": self.config.strict_vertical_gripper,
            "mean_tilt_error_rad": (
                self.tilt_error_sum / max(self.tilt_error_samples, 1)
            ),
            "max_tilt_error_rad": self.max_tilt_error,
            "grasp_mean_tilt_error_rad": (
                self.grasp_tilt_error_sum
                / max(self.grasp_tilt_error_samples, 1)
            ),
            "grasp_max_tilt_error_rad": self.max_grasp_tilt_error,
            "grasp_tilt_samples": self.grasp_tilt_error_samples,
            "policy_retry_count": self.retry_count,
            "policy_attempt_failure_count": self.attempt_failure_count,
        }

    def summary(self) -> str:
        info = self.env.info()
        decisive = self.env.success_snapshot or info
        task_result = "success" if self.env.ever_success else self.result
        policy_result_text = (
            "" if task_result == self.result else f" policy_result={self.result}"
        )
        error = decisive["grasp_error"]
        error_text = "none" if not math.isfinite(error) else f"{error:.4f}m"
        summary = (
            f"trial={info['trial']} result={task_result}{policy_result_text} "
            f"sim_time={self.env.data.time:.3f}s "
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
