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
    # Optional target-point ablation.  The default retains the historical
    # middle-node baseline; middle_angle selects a nearby material node whose
    # tangent crosses the planned jaw-closing axis at a favorable angle.
    # middle_geometry also favors a long straight, isolated, reachable section.
    target_selection_mode: str = "middle"
    target_middle_fraction: float = 0.20
    target_min_crossing_angle_deg: float = 60.0
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
    # INTERCEPT exceeding intercept_tilt_limit used to abort the whole attempt.
    # A retry re-homes vertically and restarts APPROACH, which costs most of a
    # 15 s budget and, in modes that keep the same material segment, can loop
    # until timeout.  With realignment enabled the hand instead holds position
    # while orientation runs as the primary IK task, and only the far larger
    # abort limit still discards the attempt.  Off by default so previously
    # frozen runs stay bit-identical.
    intercept_tilt_realign: bool = False
    intercept_tilt_realign_timeout: float = 1.2
    intercept_tilt_abort_limit: float = math.radians(75.0)
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
    # NERO can accumulate a large orientation error while lifting a confirmed
    # grasp.  This optional mode keeps the normal position-priority IK for
    # approach/intercept, then makes orientation primary only after grasp
    # confirmation so the lift/carry motion does not twist the cable out.
    nero_post_grasp_orientation_priority: bool = False
    # Optional NERO-only ablation: make orientation primary during the final
    # descent/intercept, while retaining position priority during the longer
    # approach where exact pose tracking would otherwise prevent contact.
    nero_intercept_orientation_priority: bool = False
    # Optional NERO-only ablation: regulate the two tilt components of the
    # vertical grasp frame while approaching and after grasp confirmation.
    # Translation remains available in the orientation nullspace; unlike
    # strict_vertical_gripper this does not halt translation when tilt is high.
    nero_approach_orientation_priority: bool = False
    # NERO-only alternative to full-pose priority: regulate only the two tilt
    # components of the approach axis in the position-task nullspace, leaving
    # the horizontal yaw free for dynamic tracking.
    nero_tilt_only_orientation_control: bool = False
    # Optional NERO-only weighted 6-D IK.  The baseline uses a strict
    # position-primary/nullspace decomposition; this mode lets a small pose
    # weight trade a little translation error for better orientation retention.
    nero_weighted_pose_ik: bool = False
    nero_pose_orientation_weight: float = 0.25
    # NERO's stock vertical convention leaves the finger tips pointing away
    # from the table.  This optional 180-degree flip about the jaw axis keeps
    # the jaw closing direction unchanged while pointing the tips downward.
    nero_finger_tips_down: bool = False
    # Rotate the NERO grasp frame in the horizontal plane while preserving the
    # downward approach axis.  This is a task-space pose ablation; zero keeps
    # the calibrated baseline unchanged.
    nero_grasp_yaw_offset_deg: float = 0.0
    policy_joint_velocity_fraction: float = 1.0
    ik_target_horizon: float = 0.11
    intercept_timeout: float = 9.0
    close_timeout: float = 0.8
    close_hard_timeout: float = 3.0
    close_invalid_contact_timeout: float = 1.6
    close_capture_distance: float = 0.018
    close_contact_grace: float = 0.35
    # middle_geometry locks one material segment when descent starts and then
    # refuses to close unless the nearest point still belongs to that segment,
    # so a hand resting on an adjacent fold never triggers closure.  The bound
    # was hard-coded; exposing it leaves frozen runs bit-identical while making
    # the fold-adjacency guard measurable rather than assumed.
    intercept_segment_tolerance: int = 4
    # A folded cable puts two material-distant strands inside the pad travel at
    # once.  Two 28 mm strands need >= 56 mm of aperture, so the 40 mm grasp
    # gate can never confirm and the episode is lost before the fingers move.
    # The isotropic table-plane clearance used by middle_geometry scores a
    # harmless neighbour offset along the jaw's long axis the same as a fatal
    # one sitting in the closing gap.  These options replace it with clearance
    # measured along the world closing axis, which is what the pads sweep.
    corridor_clearance: bool = False
    # Material-coordinate separation above which a node counts as a different
    # strand rather than as part of the segment being pinched.
    corridor_index_gap: int = 6
    # Half aperture plus one cable radius: a strand closer than this along the
    # closing axis is trapped together with the intended one.
    corridor_half_gap: float = 0.032
    # Pad reach along the approach axis.  Nodes deeper than this cannot be
    # pinched regardless of their lateral offset.
    corridor_depth: float = 0.060
    # Pad half length along the jaw's long axis.  A strand beyond it passes
    # between the fingers instead of being trapped, so it must not veto an
    # otherwise clean grasp site.
    corridor_span: float = 0.050
    # Score weight; large enough to dominate bend and midpoint preferences.
    corridor_weight: float = 6.0
    # The corridor test above runs once at settle time and excludes material
    # neighbours, so it stays blind to a tight fold: there the two legs sit a
    # few nodes apart in material coordinate yet tens of millimetres apart in
    # space.  Re-measuring the pad box live, at the moment the fingers are
    # about to close, is what the recorded episodes say works -- 96% of
    # never-bilateral attempts are already blocked at jaw entry versus 46% of
    # successful ones, while at hover the two groups are indistinguishable.
    close_box_gate: bool = False
    box_aperture_limit: float = 0.040
    # Pad box half extents along the jaw's long axis and the approach axis.
    box_lat_half: float = 0.030
    box_dep_half: float = 0.030
    # Drop already-attempted sites rather than merely penalising them: the soft
    # -3.0 is smaller than a corridor bonus of up to +6.0, so a retry keeps
    # walking back to the same clean-looking site.
    retry_hard_blacklist: bool = False
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
            "intercept_tilt_abort_limit", "intercept_tilt_realign_timeout",
            "intercept_singularity_limit", "approach_fast_distance",
            "approach_linear_velocity_limit", "intercept_linear_velocity_limit",
            "precision_linear_velocity_limit", "policy_joint_velocity_fraction",
            "approach_orientation_gain", "precision_orientation_gain",
            "ik_target_horizon", "strict_vertical_tolerance",
            "retry_min_remaining_seconds", "nero_pose_orientation_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not math.isfinite(self.target_filter_alpha)
            or not 0.0 < self.target_filter_alpha <= 1.0
        ):
            raise ValueError("target_filter_alpha must be in (0, 1]")
        if self.target_selection_mode not in {"middle", "middle_angle", "middle_geometry"}:
            raise ValueError("target_selection_mode must be middle, middle_angle, or middle_geometry")
        if not 0.0 < self.target_middle_fraction < 0.5:
            raise ValueError("target_middle_fraction must be in (0, 0.5)")
        if not 0.0 <= self.target_min_crossing_angle_deg <= 90.0:
            raise ValueError("target_min_crossing_angle_deg must be in [0, 90]")
        if not 0.0 < self.policy_joint_velocity_fraction <= 1.0:
            raise ValueError("policy_joint_velocity_fraction must be in (0, 1]")
        if not isinstance(self.strict_vertical_gripper, bool):
            raise ValueError("strict_vertical_gripper must be boolean")
        if not isinstance(self.nero_post_grasp_orientation_priority, bool):
            raise ValueError("nero_post_grasp_orientation_priority must be boolean")
        if not isinstance(self.nero_intercept_orientation_priority, bool):
            raise ValueError("nero_intercept_orientation_priority must be boolean")
        if not isinstance(self.nero_approach_orientation_priority, bool):
            raise ValueError("nero_approach_orientation_priority must be boolean")
        if not isinstance(self.nero_tilt_only_orientation_control, bool):
            raise ValueError("nero_tilt_only_orientation_control must be boolean")
        if not isinstance(self.nero_weighted_pose_ik, bool):
            raise ValueError("nero_weighted_pose_ik must be boolean")
        if not isinstance(self.nero_finger_tips_down, bool):
            raise ValueError("nero_finger_tips_down must be boolean")
        if not isinstance(self.intercept_tilt_realign, bool):
            raise ValueError("intercept_tilt_realign must be boolean")
        if self.intercept_tilt_abort_limit < self.intercept_tilt_limit:
            raise ValueError(
                "intercept_tilt_abort_limit must be >= intercept_tilt_limit"
            )
        if (
            not isinstance(self.intercept_segment_tolerance, int)
            or isinstance(self.intercept_segment_tolerance, bool)
            or self.intercept_segment_tolerance < 0
        ):
            raise ValueError(
                "intercept_segment_tolerance must be a non-negative integer"
            )
        if not isinstance(self.corridor_clearance, bool):
            raise ValueError("corridor_clearance must be boolean")
        if (
            not isinstance(self.corridor_index_gap, int)
            or isinstance(self.corridor_index_gap, bool)
            or self.corridor_index_gap < 1
        ):
            raise ValueError("corridor_index_gap must be a positive integer")
        for name in (
            "corridor_half_gap", "corridor_depth", "corridor_span",
            "corridor_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.nero_grasp_yaw_offset_deg):
            raise ValueError("nero_grasp_yaw_offset_deg must be finite")
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
        self.realign_count = 0
        self._tilt_realign_active = False
        self._realign_start_time: float | None = None
        # CLOSE-gate triage counters.  Nothing here feeds back into control;
        # they only record which of the three gates kept refusing to close, so
        # a frozen run can be diagnosed without replaying every episode.
        self.min_hand_cable_distance = float("inf")
        self.min_distance_when_vertical_ok = float("inf")
        self.close_gate_blocked_distance = 0
        self.close_gate_blocked_vertical = 0
        self.close_gate_blocked_segment = 0
        self.max_segment_index_gap = 0
        self.finished = False
        self.result = "running"
        self.failure_diagnostics: dict | None = None
        self.filtered_target = np.zeros(3)
        self.selected_target_body_id = self.env.target_body_id
        self.selected_target_index = self._policy_node_ids().index(self.env.target_body_id)
        self.selected_target_crossing_angle_deg = float("nan")
        self.selected_target_corridor_clearance = float("nan")
        self.locked_segment_index: int | None = None
        self.locked_segment_alpha = 0.0
        self.locked_node_ids: list[int] | None = None
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

    def _nero_grasp_rotation(self, yaw_offset_deg: float) -> np.ndarray:
        """Return one NERO vertical grasp frame with a horizontal yaw offset."""

        return self._nero_rotation_at_yaw(math.radians(yaw_offset_deg))

    def _nero_rotation_at_yaw(self, yaw_rad: float) -> np.ndarray:
        """NERO vertical grasp frame rotated by an absolute planar yaw."""

        desired_rotation = self.env.vertical_grasp_rotation()
        if self.config.nero_finger_tips_down:
            desired_rotation = np.diag([-1.0, 1.0, -1.0]) @ desired_rotation
        if abs(yaw_rad) > 1e-12:
            yaw_rotation = np.array([
                [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
                [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
                [0.0, 0.0, 1.0],
            ])
            desired_rotation = yaw_rotation @ desired_rotation
        return desired_rotation

    def _initial_target(self) -> np.ndarray:
        """Return the target used to initialise or restart target filtering.

        The ordinary scripted baseline is privileged and therefore uses the
        simulator target. Vision-only subclasses override this hook so the
        shared phase machine never needs to read cable ground truth.
        """
        return self.env.data.xpos[self.selected_target_body_id].copy()

    def _corridor_clearance(
        self,
        positions: np.ndarray,
        index: int,
        closing: np.ndarray,
        approach: np.ndarray,
        lateral: np.ndarray,
    ) -> float:
        """Closing-axis offset to the nearest strand the pads could also trap.

        The jaw sweeps a box, not a disc: a node only competes with the grasp
        when it lies inside the pad reach along the approach axis and along the
        jaw's long axis.  Material neighbours are excluded because they belong
        to the segment being pinched rather than to a second strand.  Nodes
        outside the box report the box depth so the caller can compare them
        against ``corridor_half_gap`` without a separate flag.
        """

        gap = self.config.corridor_index_gap
        far = np.abs(np.arange(len(positions)) - index) >= gap
        if not bool(np.any(far)):
            return self.config.corridor_depth
        delta = positions - positions[index]
        inside = (
            far
            & (np.abs(delta @ approach) <= self.config.corridor_depth)
            & (np.abs(delta @ lateral) <= self.config.corridor_span)
        )
        if not bool(np.any(inside)):
            return self.config.corridor_depth
        return float(np.min(np.abs(delta[inside] @ closing)))

    def _jaw_box_blocked(self) -> bool:
        """Whether two strands inside the live pad box exceed the aperture.

        Unlike :meth:`_corridor_clearance` this uses no material-coordinate
        rule, because the fatal configuration is a tight fold whose legs are
        adjacent in index but far apart in space.  The closing axis is bounded
        by physics rather than by the box: a node can only be trapped while its
        offset along that axis stays within half the aperture plus one cable
        radius, so anything further away cannot be pinched and must not veto
        the site.
        """

        aperture = self.config.box_aperture_limit
        radius = float(self.env.cable_radius)
        rotation = self.env.data.xmat[self.env.hand_id].reshape(3, 3)
        pad = (self.env.data.xpos[self.env.hand_id]
               + rotation @ self.env.GRASP_CENTER_LOCAL)
        delta = self.env.data.xpos[self._policy_node_ids()] - pad
        inside = (
            (np.abs(delta @ (rotation @ self.env.gripper_closing_axis_local))
             <= aperture / 2.0 + radius)
            & (np.abs(delta @ (rotation @ self.env.gripper_lateral_axis_local))
               <= self.config.box_lat_half)
            & (np.abs(delta @ (rotation @ self.env.gripper_approach_axis_local))
               <= self.config.box_dep_half)
        )
        if int(np.count_nonzero(inside)) < 2:
            return False
        closing = (delta @ (rotation @ self.env.gripper_closing_axis_local))[inside]
        return bool(float(closing.max() - closing.min()) + 2.0 * radius > aperture)

    def _choose_initial_target(self, desired_rotation: np.ndarray) -> None:
        """Pick one near-middle node without changing the environment's task."""

        node_ids = self._policy_node_ids()
        self.selected_target_body_id = self.env.target_body_id
        self.selected_target_index = node_ids.index(self.selected_target_body_id)
        self.selected_target_crossing_angle_deg = float("nan")
        self.selected_target_corridor_clearance = float("nan")
        if self.config.target_selection_mode == "middle":
            return

        positions = self.env.data.xpos[node_ids, :2]
        closing = (
            desired_rotation @ self.env.gripper_closing_axis_local
        )[:2]
        closing /= max(float(np.linalg.norm(closing)), 1e-12)
        middle = len(node_ids) // 2
        geometry_mode = self.config.target_selection_mode == "middle_geometry"
        # The corridor test needs the full 3D jaw frame; middle and
        # middle_angle never score geometry so they keep the planar path.
        corridor = bool(self.config.corridor_clearance and geometry_mode)
        if corridor:
            def unit(axis: np.ndarray) -> np.ndarray:
                return axis / max(float(np.linalg.norm(axis)), 1e-12)

            positions3 = self.env.data.xpos[node_ids]
            closing3 = unit(
                desired_rotation @ self.env.gripper_closing_axis_local
            )
            approach3 = unit(
                desired_rotation @ self.env.gripper_approach_axis_local
            )
            lateral3 = unit(
                desired_rotation @ self.env.gripper_lateral_axis_local
            )
        fraction = (max(self.config.target_middle_fraction, 0.30)
                    if geometry_mode
                    else self.config.target_middle_fraction)
        radius = max(1, int(round(fraction * len(node_ids))))
        candidates: list[tuple[int, float, float, float]] = []
        for index in range(max(1, middle - radius), min(len(node_ids) - 1, middle + radius + 1)):
            tangent = positions[index + 1] - positions[index - 1]
            length = float(np.linalg.norm(tangent))
            if length < 1e-8:
                continue
            tangent /= length
            angle = math.degrees(math.acos(float(np.clip(
                abs(np.dot(tangent, closing)), 0.0, 1.0
            ))))
            quality = 0.0
            jaw_clearance = float("nan")
            if geometry_mode:
                span = min(3, index, len(node_ids) - 1 - index)
                before = positions[index] - positions[index - span]
                after = positions[index + span] - positions[index]
                before /= max(float(np.linalg.norm(before)), 1e-12)
                after /= max(float(np.linalg.norm(after)), 1e-12)
                bend = math.degrees(math.acos(float(np.clip(
                    np.dot(before, after), -1.0, 1.0
                ))))
                other = [j for j in range(len(node_ids)) if abs(j - index) >= 6]
                clearance = (min(float(np.linalg.norm(positions[index] - positions[j]))
                                 for j in other) if other else 0.12)
                reach = float(np.linalg.norm(positions[index] - self.env.hand_position[:2]))
                # A clear, locally straight section is more useful than one
                # whose node happens to lie one step closer to the midpoint.
                quality = (
                    -2.0 * min(bend / 45.0, 2.0)
                    + 1.0 * min(clearance / 0.10, 1.0)
                    - 0.10 * abs(index - middle)
                    - 1.5 * max(reach - 0.25, 0.0) / 0.10
                )
                if hasattr(self, "attempted_target_indices") and index in self.attempted_target_indices:
                    quality -= 3.0
                if corridor:
                    jaw_clearance = self._corridor_clearance(
                        positions3, index, closing3, approach3, lateral3
                    )
                    quality += self.config.corridor_weight * min(
                        jaw_clearance / self.config.corridor_half_gap, 1.0
                    )
            candidates.append((index, angle, quality, jaw_clearance))
        if not candidates:
            return
        if self.config.retry_hard_blacklist and self.attempted_target_indices:
            fresh = [
                candidate for candidate in candidates
                if candidate[0] not in self.attempted_target_indices
            ]
            if fresh:
                candidates = fresh
        if corridor:
            # A site whose closing corridor is blocked cannot be rescued by a
            # better approach, so drop those sites outright whenever the window
            # still offers at least one clean alternative.
            clean = [
                candidate for candidate in candidates
                if candidate[3] >= self.config.corridor_half_gap
            ]
            if clean:
                candidates = clean
        favorable = [
            candidate for candidate in candidates
            if candidate[1] >= self.config.target_min_crossing_angle_deg
        ]
        if favorable:
            # Among acceptable crossings, move as little as possible from the
            # baseline midpoint.  Angle breaks equal-distance ties.
            if geometry_mode:
                index, angle, _, jaw_clearance = max(favorable, key=lambda item: (
                    item[2], item[1], -abs(item[0] - middle), -item[0]
                ))
            else:
                index, angle, _, jaw_clearance = min(favorable, key=lambda item: (
                    abs(item[0] - middle), -item[1], item[0]
                ))
        else:
            index, angle, _, jaw_clearance = max(candidates, key=lambda item: (
                item[1], item[2], -abs(item[0] - middle), -item[0]
            ))
        self.selected_target_index = index
        self.selected_target_body_id = node_ids[index]
        self.selected_target_crossing_angle_deg = angle
        self.selected_target_corridor_clearance = jaw_clearance

    def reset(self) -> None:
        self.phase = Phase.SETTLE
        self.phase_start = float(self.env.data.time)
        self.retry_count = 0
        self.attempted_target_indices: set[int] = set()
        self.attempt_failure_count = 0
        self.last_attempt_failure = None
        self._retry_after_failure_observe = False
        self.realign_count = 0
        self._tilt_realign_active = False
        self._realign_start_time = None
        self.min_hand_cable_distance = float("inf")
        self.min_distance_when_vertical_ok = float("inf")
        self.close_gate_blocked_distance = 0
        self.close_gate_blocked_vertical = 0
        self.close_gate_blocked_segment = 0
        self.close_gate_blocked_box = 0
        self.max_segment_index_gap = 0
        self.finished = False
        self.result = "running"
        self.failure_diagnostics = None
        self.filtered_target = np.zeros(3)
        self.locked_segment_index = None
        self.locked_segment_alpha = 0.0
        self.locked_node_ids = None
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
            desired_rotation = self._nero_grasp_rotation(
                self.config.nero_grasp_yaw_offset_deg
            )
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
        self._choose_initial_target(desired_rotation)
        self.filtered_target = self._initial_target()

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
            position, velocity = self._tracked_cable_point()
        else:
            index = self.locked_segment_index
            alpha = self.locked_segment_alpha
            node_ids = self.locked_node_ids or self._policy_node_ids()
            body0 = node_ids[index]
            body1 = node_ids[index + 1]
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

    def _tracked_cable_point(self) -> tuple[np.ndarray, np.ndarray]:
        """Source position/velocity while no material segment is locked.

        Defaults to the environment-assigned target node.  Subclasses may
        override to track a selected cable segment instead.
        """

        return (
            self.env.data.xpos[self.selected_target_body_id].copy(),
            self.env.body_linear_velocity(self.selected_target_body_id),
        )

    def _select_intercept_segment(self, point: np.ndarray) -> np.ndarray:
        """Choose the material segment locked when descent begins.

        Defaults to the segment nearest the hand.  Subclasses may override
        to implement predictive segment selection.
        """

        if self.config.target_selection_mode != "middle_geometry":
            return self._lock_segment_near(point)
        node_ids = self._policy_node_ids()
        positions = self.env.data.xpos[node_ids]
        lo = max(0, self.selected_target_index - 2)
        hi = min(len(node_ids) - 2, self.selected_target_index + 1)
        candidates = []
        for index in range(lo, hi + 1):
            vector = positions[index + 1] - positions[index]
            alpha = float(np.clip(
                np.dot(point - positions[index], vector)
                / max(float(np.dot(vector, vector)), 1e-12), 0.0, 1.0
            ))
            candidate = positions[index] + alpha * vector
            candidates.append((float(np.linalg.norm(candidate - point)), index, alpha, candidate))
        _, index, alpha, nearest = min(candidates, key=lambda item: item[0])
        self.locked_segment_index = index
        self.locked_segment_alpha = alpha
        self.locked_node_ids = node_ids
        self.filtered_target = nearest.copy()
        return nearest

    def _policy_node_ids(self) -> list[int]:
        """脚本策略当前关注的对象节点集合（多对象时为目标对象）。"""

        env = self.env
        objects = getattr(env, "_objects", None)
        if (
            objects is not None
            and getattr(env.config, "n_objects", 1) > 1
        ):
            return list(objects[env._target_object].ids)
        return list(env.cable_ids)

    def _nearest_cable_point(self, point: np.ndarray) -> tuple[np.ndarray, float, int, float]:
        """返回目标对象中心线上距给定点最近的位置及其线段参数。"""
        positions = self.env.data.xpos[self._policy_node_ids()]
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
        node_ids = self._policy_node_ids()
        nearest, _, index, alpha = self._nearest_cable_point(point)
        self.locked_segment_index = index
        self.locked_segment_alpha = alpha
        self.locked_node_ids = node_ids
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

    def _hover_target(self, target: np.ndarray) -> np.ndarray:
        """悬停点：目标 XY 上方 +0.20 高度。"""
        hover = target.copy()
        hover[2] += 0.20
        return hover

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
        if self.config.target_selection_mode == "middle_geometry":
            self.attempted_target_indices.add(self.selected_target_index)
        if self._retry_available():
            self.retry_count += 1
            self.result = "running"
            self._begin_vertical_recovery(hand)
            return self._ik_action(self.recover_start, self.env.gripper_open_ctrl)
        self.result = "failed_no_contact"
        self._transition(Phase.RELEASE)
        return self._ik_action(hand, self.env.gripper_open_ctrl)

    def _realign_tilt(self, hand: np.ndarray) -> np.ndarray:
        """Hold position and recover the approach axis instead of retrying.

        Requesting the current hand position makes the position error vanish, so
        the orientation task can run as primary without the hand drifting away
        from the segment it was descending on.
        """

        self.realign_count += 1
        self._tilt_realign_active = True
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
        post_grasp_orientation_priority = bool(
            self.config.nero_post_grasp_orientation_priority
            and self.env.robot == "nero"
            and self.phase in {Phase.LIFT, Phase.CARRY, Phase.HOLD}
        )
        intercept_orientation_priority = bool(
            self.config.nero_intercept_orientation_priority
            and self.env.robot == "nero"
            and self.phase is Phase.INTERCEPT
        )
        approach_orientation_priority = bool(
            self.config.nero_approach_orientation_priority
            and self.env.robot == "nero"
            and self.phase is Phase.APPROACH
        )
        tilt_only_orientation_control = bool(
            self.config.nero_tilt_only_orientation_control
            and self.env.robot == "nero"
            and self.phase in {
                Phase.APPROACH,
                Phase.INTERCEPT,
                Phase.LIFT,
                Phase.CARRY,
                Phase.HOLD,
            }
        )
        orientation_primary = (
            self.config.strict_vertical_gripper
            or post_grasp_orientation_priority
            or intercept_orientation_priority
            or approach_orientation_priority
            or self._tilt_realign_active
        )
        # Re-alignment deliberately bypasses the downweighted pose solve: the
        # position error is already zeroed, so a 0.35 orientation weight would
        # only slow the tilt recovery this branch exists to perform.
        if self.env.robot == "nero" and not self._tilt_realign_active and (
            self.config.nero_weighted_pose_ik
            or self.config.target_selection_mode == "middle_geometry"
        ):
            pose_weight = (
                0.35 if self.config.target_selection_mode == "middle_geometry"
                and not self.config.nero_weighted_pose_ik
                else self.config.nero_pose_orientation_weight
            )
            weighted_jacobian = np.vstack([
                position_jacobian,
                pose_weight * jacobian[3:],
            ])
            weighted_velocity = np.concatenate([
                linear_velocity,
                pose_weight * angular_velocity,
            ])
            pose_damping = 0.04
            pose_inverse = np.linalg.solve(
                weighted_jacobian @ weighted_jacobian.T
                + pose_damping**2 * np.eye(6),
                np.eye(6),
            )
            q_velocity = weighted_jacobian.T @ pose_inverse @ weighted_velocity
        elif tilt_only_orientation_control:
            # For NERO, preserve the safety-critical downward approach axis
            # without forcing a complete world-frame yaw during the dynamic
            # chase. The tilt controller lives in the position nullspace.
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
            actual_axis = data.xmat[self.env.hand_id].reshape(3, 3) @ (
                self.env.gripper_approach_axis_local
            )
            desired_axis = self.desired_approach_axis
            tilt_projection = np.eye(3) - np.outer(actual_axis, actual_axis)
            tilt_jacobian = (
                tilt_projection @ jacobian[3:] @ position_nullspace
            )
            tilt_velocity_target = self._limit_vector_norm(
                2.5 * np.cross(actual_axis, desired_axis), 1.40
            )
            tilt_residual = (
                tilt_velocity_target
                - tilt_projection @ jacobian[3:] @ position_velocity
            )
            tilt_damping = 0.04
            tilt_inverse = np.linalg.solve(
                tilt_jacobian @ tilt_jacobian.T
                + tilt_damping**2 * np.eye(3),
                np.eye(3),
            )
            tilt_velocity = (
                tilt_jacobian.T @ tilt_inverse @ tilt_residual
            )
            orientation_gain = (
                self.config.approach_orientation_gain
                if fast_approach
                else self.config.precision_orientation_gain
            )
            q_velocity = position_velocity + orientation_gain * tilt_velocity
        elif orientation_primary:
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
        self.locked_node_ids = None
        self.recover_start = hand.copy()
        self.recover_goal = hand + np.array([0.0, 0.0, 0.18])
        self._transition(Phase.RECOVER)

    def _close_capture_distance(self) -> float:
        """Distance at which descent transitions to gripper closure.

        触发距离随目标对象厚度放宽：跨骑更粗的对象时手部参考点天然离
        目标节点更远，旧线缆标定值会让粗对象永远达不到闭爪阈值。
        """

        objects = getattr(self.env, "_objects", None)
        if objects:
            index = min(self.env._target_object, len(objects) - 1)
            extra = max(0.0, float(objects[index].radius) - 0.014)
            return self.config.close_capture_distance + extra
        return self.config.close_capture_distance

    def action(self) -> np.ndarray:
        """推进反应式状态机，并返回一个控制周期的动作。"""
        # 无论处于哪个阶段，策略只读取环境状态并返回动作，不直接修改线缆物理。
        self._tilt_realign_active = False
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
            desired = self._hover_target(target)
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
                self._select_intercept_segment(hand)
                self._transition(Phase.INTERCEPT)
            elif self.phase_time > self.config.approach_timeout:
                return self._retry_from_unsafe_pose(hand)
            return self._ik_action(desired, self.env.gripper_open_ctrl)

        if self.phase is Phase.INTERCEPT:
            # 实际两指夹持中心直接追踪目标线缆段中心，不再使用旧虚拟点的z补偿。
            desired = target.copy()
            tilt_angle = self._tilt_error()
            abort_limit = (
                self.config.intercept_tilt_abort_limit
                if self.config.intercept_tilt_realign
                else self.config.intercept_tilt_limit
            )
            if (
                tilt_angle > abort_limit
                or self._minimum_task_singular_value()
                < self.config.intercept_singularity_limit
            ):
                self._realign_start_time = None
                return self._retry_from_unsafe_pose(hand)
            if (
                self.config.intercept_tilt_realign
                and tilt_angle > self.config.intercept_tilt_limit
                and self.phase_time <= self.config.intercept_timeout
            ):
                now = float(self.env.data.time)
                if self._realign_start_time is None:
                    self._realign_start_time = now
                elif (
                    now - self._realign_start_time
                    > self.config.intercept_tilt_realign_timeout
                ):
                    # The approach axis is not recoverable at this location, so
                    # only now pay for a retry.
                    self._realign_start_time = None
                    return self._retry_from_unsafe_pose(hand)
                return self._realign_tilt(hand)
            self._realign_start_time = None
            nearest, nearest_distance, nearest_index, _ = (
                self._nearest_cable_point(hand)
            )
            # 截获期间持续追踪进入该阶段时锁定的材料线段，避免在相邻弯折间
            # 跳变；但闭爪触发仍以任意真实线缆中心线进入夹持区域为准。
            vertical_ready = (
                not self.config.strict_vertical_gripper
                or tilt_angle < self.config.strict_vertical_tolerance
            )
            intended_segment_ready = True
            if (
                self.config.target_selection_mode == "middle_geometry"
                and self.locked_segment_index is not None
            ):
                intended_segment_ready = (
                    abs(self._nearest_cable_point(hand)[2]
                        - self.locked_segment_index)
                    <= self.config.intercept_segment_tolerance
                )
            if vertical_ready:
                self.min_distance_when_vertical_ok = min(
                    self.min_distance_when_vertical_ok,
                    float(nearest_distance),
                )
            self.min_hand_cable_distance = min(
                self.min_hand_cable_distance, float(nearest_distance)
            )
            if self.locked_segment_index is not None:
                self.max_segment_index_gap = max(
                    self.max_segment_index_gap,
                    abs(int(nearest_index) - int(self.locked_segment_index)),
                )
            if nearest_distance >= self._close_capture_distance():
                self.close_gate_blocked_distance += 1
            elif not vertical_ready:
                self.close_gate_blocked_vertical += 1
            elif not intended_segment_ready:
                self.close_gate_blocked_segment += 1
            if (
                nearest_distance < self._close_capture_distance()
                and vertical_ready
                and intended_segment_ready
            ):
                if self.config.close_box_gate and self._jaw_box_blocked():
                    # Two strands already sit inside the pad travel, so closing
                    # here can only stall above the aperture.  Recovering costs
                    # a second or two; a stalled close costs the episode.
                    self.close_gate_blocked_box += 1
                    return self._retry_from_unsafe_pose(hand)
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
                    (self.config.target_selection_mode == "middle_geometry"
                     and self.phase_time > self.config.close_invalid_contact_timeout)
                    or
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
                if self.config.target_selection_mode == "middle_geometry":
                    self._choose_initial_target(self._nero_grasp_rotation(
                        self.config.nero_grasp_yaw_offset_deg
                    ) if self.env.robot == "nero" else self.VERTICAL_GRASP_ROTATION)
                self.filtered_target = self._initial_target()
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

    def policy_info(self) -> dict[str, float | int | bool | None]:
        """Return orientation-ablation diagnostics for benchmark rows."""

        return {
            "policy_selected_target_body_id": self.selected_target_body_id,
            "policy_selected_target_index": self.selected_target_index,
            "policy_selected_target_crossing_angle_deg": (
                self.selected_target_crossing_angle_deg
            ),
            "policy_selected_target_corridor_clearance": (
                self.selected_target_corridor_clearance
            ),
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
            "policy_tilt_realign_count": self.realign_count,
            "policy_min_hand_cable_distance": (
                None
                if math.isinf(self.min_hand_cable_distance)
                else float(self.min_hand_cable_distance)
            ),
            "policy_min_distance_when_vertical_ok": (
                None
                if math.isinf(self.min_distance_when_vertical_ok)
                else float(self.min_distance_when_vertical_ok)
            ),
            "policy_close_gate_blocked_distance": (
                self.close_gate_blocked_distance
            ),
            "policy_close_gate_blocked_vertical": (
                self.close_gate_blocked_vertical
            ),
            "policy_close_gate_blocked_segment": (
                self.close_gate_blocked_segment
            ),
            "policy_max_segment_index_gap": self.max_segment_index_gap,
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
