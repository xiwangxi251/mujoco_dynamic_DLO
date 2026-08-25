"""Formula-aware privileged interception expert.

This policy is intentionally separate from the evaluation baseline. It may
read simulator-only cable state and the environment's motion equations so it
can serve as a teacher for data collection. A learned student must not receive
these privileged quantities at evaluation time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math

import numpy as np

from ..env.environment import CableGraspEnv, RIGID_MOTION_PROFILES
from ..env.kinematics import rotation_to_quat
from ..policies.scripted import DynamicCableGraspPolicy, Phase, PolicyConfig

from .shadow_rollout import ShadowCableRollout, ShadowTrajectory


@dataclass
class FormulaInterceptConfig(PolicyConfig):
    """Parameters for future-trajectory candidate search."""

    candidate_horizons: tuple[float, ...] = (0.35, 0.55, 0.75, 0.95)
    replan_interval: float = 0.20
    endpoint_margin_nodes: int = 4
    # This is an effective candidate-ranking speed, not the instantaneous IK
    # velocity limit. Paired ablations retained 0.62 m/s because lowering it
    # changed otherwise successful rigid/shape candidate choices.
    assumed_reach_speed: float = 0.62
    combined_close_capture_distance: float = 0.018
    intercept_prediction_horizon: float = 0.30
    shadow_rollout_extra_time: float = 0.35
    candidate_height_limit: float = 0.10
    candidate_table_margin: float = 0.06
    reach_lateness_weight: float = 5.0
    travel_weight: float = 0.65
    target_speed_weight: float = 0.35
    curvature_weight: float = 0.12
    boundary_weight: float = 0.08
    failed_segment_penalty: float = 2.0
    failed_segment_radius: int = 3
    align_gripper_to_tangent: bool = False
    search_shape_all_segments: bool = False
    record_intercept_failures: bool = True
    shape_use_scripted_fallback: bool = True
    dynamic_portfolio_enabled: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        horizons = np.asarray(self.candidate_horizons, dtype=float)
        if (
            horizons.ndim != 1
            or horizons.size == 0
            or not np.all(np.isfinite(horizons))
            or np.any(horizons <= 0.0)
        ):
            raise ValueError("candidate_horizons must be finite and positive")
        for name in (
            "replan_interval", "assumed_reach_speed",
            "intercept_prediction_horizon", "candidate_height_limit",
            "combined_close_capture_distance",
            "shadow_rollout_extra_time",
            "candidate_table_margin", "reach_lateness_weight", "travel_weight",
            "target_speed_weight", "curvature_weight", "boundary_weight",
            "failed_segment_penalty",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.endpoint_margin_nodes < 1:
            raise ValueError("endpoint_margin_nodes must be positive")
        if self.failed_segment_radius < 0:
            raise ValueError("failed_segment_radius must be non-negative")
        for name in (
            "align_gripper_to_tangent", "search_shape_all_segments",
            "record_intercept_failures",
            "shape_use_scripted_fallback", "dynamic_portfolio_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


class FormulaInterceptExpert(DynamicCableGraspPolicy):
    """Choose a future low-cost cable segment and intercept it receding-horizon."""

    def __init__(
        self,
        env,
        config: FormulaInterceptConfig | None = None,
    ) -> None:
        self.expert_config = config or FormulaInterceptConfig()
        self.expert_segment_index = 0
        self.expert_segment_alpha = 0.5
        self.expert_horizon = self.expert_config.candidate_horizons[0]
        self.expert_score = math.inf
        self.expert_replans = 0
        self.last_replan_time = -math.inf
        self._base_grasp_rotation = np.eye(3)
        self.failed_segment_indices: list[int] = []
        self.shadow_oracle = ShadowCableRollout(env)
        self.shadow_trajectory: ShadowTrajectory | None = None
        self.shadow_rollouts = 0
        self.shadow_physics_steps = 0
        self._shadow_release_state = False
        self.episode_use_scripted = False
        self.portfolio_formula_success: bool | None = None
        self.portfolio_scripted_success: bool | None = None
        super().__init__(env, self.expert_config)

    def reset(self) -> None:
        super().reset()
        self.episode_use_scripted = False
        self.portfolio_formula_success = None
        self.portfolio_scripted_success = None
        self.expert_segment_index = max(0, len(self.env.cable_ids) // 2 - 1)
        self.expert_segment_alpha = 0.5
        if self.env.config.motion_mode in {"static", "shape"}:
            target_index = self.env.cable_ids.index(self.env.target_body_id)
            self.expert_segment_index = min(
                target_index, len(self.env.cable_ids) - 2
            )
            self.expert_segment_alpha = float(
                target_index == len(self.env.cable_ids) - 1
            )
            self.expert_horizon = self.config.approach_prediction_horizon
        self.expert_horizon = self.expert_config.candidate_horizons[0]
        self.expert_score = math.inf
        self.expert_replans = 0
        self.last_replan_time = -math.inf
        self.failed_segment_indices = []
        self.shadow_trajectory = None
        self.shadow_rollouts = 0
        self.shadow_physics_steps = 0
        self._shadow_release_state = bool(self.env.rigid_motion_released)
        current_rotation = self.env.data.xmat[self.env.hand_id].reshape(3, 3)
        z_rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self._base_grasp_rotation = (
            self.VERTICAL_GRASP_ROTATION.copy()
            if self.config.strict_vertical_gripper
            else z_rotation @ current_rotation
        )
        if (
            self.expert_config.dynamic_portfolio_enabled
            and self.env.episode_seed is not None
            and self.env.config.motion_mode in {"rigid", "combined"}
        ):
            self._select_portfolio_controller()

    def _rigid_future_nodes(self, horizon: float) -> tuple[np.ndarray, np.ndarray]:
        """Return future nodes and the current rigid velocity field."""

        positions = self.env.data.xpos[self.env.cable_ids].copy()
        velocities = np.asarray([
            self.env.body_linear_velocity(body_id)
            for body_id in self.env.cable_ids
        ])
        current_com = np.average(
            positions[:, :2], axis=0, weights=self.env.cable_mass
        )
        com_velocity = np.average(
            velocities[:, :2], axis=0, weights=self.env.cable_mass
        )
        relative_xy = positions[:, :2] - current_com

        current_offset = current_com - self.env._rigid_reference_com_xy
        offset = current_offset.copy()
        speed = self.env._rigid_motion_speed()
        remaining = float(horizon)
        while remaining > 1e-12:
            dt = min(0.04, remaining)
            _, tangent, _, _ = self.env._rigid_motion_path_state(offset)
            offset += speed * dt * tangent
            remaining -= dt
        _, tangent, future_progress, _ = self.env._rigid_motion_path_state(offset)
        _, current_tangent, current_progress, current_metric = (
            self.env._rigid_motion_path_state(current_offset)
        )
        current_yaw, yaw_rate = self.env._rigid_motion_rotation_state(
            current_progress, current_metric, current_tangent, com_velocity
        )
        future_yaw = self.env._rigid_motion_rotation_from_progress(
            future_progress
        )
        yaw_delta = future_yaw - current_yaw
        cosine = math.cos(yaw_delta)
        sine = math.sin(yaw_delta)
        rotation = np.array([[cosine, sine], [-sine, cosine]])

        future = positions.copy()
        future[:, :2] = (
            current_com + (offset - current_offset) + relative_xy @ rotation
        )
        rigid_velocity = np.zeros_like(velocities)
        rigid_velocity[:, :2] = (
            com_velocity
            + yaw_rate * np.column_stack((-relative_xy[:, 1], relative_xy[:, 0]))
        )
        return future, rigid_velocity

    def _refresh_shadow_trajectory(self, minimum_duration: float) -> None:
        duration = max(
            float(minimum_duration),
            max(self.expert_config.candidate_horizons)
            + self.expert_config.shadow_rollout_extra_time,
        )
        self.shadow_trajectory = self.shadow_oracle.rollout(duration)
        self.shadow_rollouts += 1
        self.shadow_physics_steps += len(self.shadow_trajectory.positions) - 1
        self._shadow_release_state = bool(self.env.rigid_motion_released)

    def _shadow_node_states(
        self, horizon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        current_time = float(self.env.data.time)
        required_elapsed = horizon
        if self.shadow_trajectory is not None:
            required_elapsed += current_time - self.shadow_trajectory.origin_time
        cache_invalid = (
            self.shadow_trajectory is None
            or current_time < self.shadow_trajectory.origin_time
            or required_elapsed > self.shadow_trajectory.duration
            or self._shadow_release_state != bool(self.env.rigid_motion_released)
        )
        if cache_invalid:
            self._refresh_shadow_trajectory(
                horizon + self.expert_config.shadow_rollout_extra_time
            )
            required_elapsed = horizon
        assert self.shadow_trajectory is not None
        return self.shadow_trajectory.sample(required_elapsed)

    def predict_nodes(self, horizon: float) -> np.ndarray:
        """Predict every cable node using privileged motion equations."""

        horizon = max(0.0, float(horizon))
        positions = self.env.data.xpos[self.env.cable_ids].copy()
        if horizon == 0.0:
            return positions
        if self.env.config.motion_mode in {"shape", "combined"}:
            return self._shadow_node_states(horizon)[0]
        velocities = np.asarray([
            self.env.body_linear_velocity(body_id)
            for body_id in self.env.cable_ids
        ])
        velocities = np.clip(velocities, -0.8, 0.8)
        uses_rigid = (
            self.env.config.motion_profile_version in RIGID_MOTION_PROFILES
            and not self.env.rigid_motion_released
        )
        if uses_rigid:
            predicted, rigid_velocity = self._rigid_future_nodes(horizon)
        else:
            predicted = positions.copy()
            rigid_velocity = np.zeros_like(velocities)

        if not uses_rigid:
            predicted += horizon * velocities
        return predicted

    def _predicted_node_velocities(self, horizon: float) -> np.ndarray:
        if self.env.config.motion_mode in {"shape", "combined"}:
            return self._shadow_node_states(horizon)[1]
        dt = 0.03
        before_horizon = max(0.0, horizon - dt)
        before = self.predict_nodes(before_horizon)
        after = self.predict_nodes(horizon + dt)
        return (after - before) / (horizon + dt - before_horizon)

    def _candidate_cost(
        self,
        nodes: np.ndarray,
        node_velocities: np.ndarray,
        index: int,
        horizon: float,
        alpha: float = 0.5,
    ) -> float:
        point = (1.0 - alpha) * nodes[index] + alpha * nodes[index + 1]
        if point[2] > self.expert_config.candidate_height_limit:
            return math.inf
        lower = self.env.table_xy_min + self.expert_config.candidate_table_margin
        upper = self.env.table_xy_max - self.expert_config.candidate_table_margin
        if np.any(point[:2] <= lower) or np.any(point[:2] >= upper):
            return math.inf

        approach_point = point + np.array([0.0, 0.0, 0.20])
        travel = float(np.linalg.norm(approach_point - self.env.hand_position))
        reach_time = travel / self.expert_config.assumed_reach_speed
        lateness = max(0.0, reach_time - horizon)
        velocity = (
            (1.0 - alpha) * node_velocities[index]
            + alpha * node_velocities[index + 1]
        )

        previous_tangent = nodes[index] - nodes[index - 1]
        next_tangent = nodes[index + 2] - nodes[index + 1]
        previous_tangent /= max(np.linalg.norm(previous_tangent), 1e-12)
        next_tangent /= max(np.linalg.norm(next_tangent), 1e-12)
        curvature = math.acos(float(np.clip(
            np.dot(previous_tangent, next_tangent), -1.0, 1.0
        )))
        boundary_clearance = float(np.min(np.r_[
            point[:2] - lower,
            upper - point[:2],
        ]))
        retry_penalty = self.expert_config.failed_segment_penalty * sum(
            abs(index - failed_index)
            <= self.expert_config.failed_segment_radius
            for failed_index in self.failed_segment_indices
        )
        return (
            self.expert_config.reach_lateness_weight * lateness
            + self.expert_config.travel_weight * travel
            + self.expert_config.target_speed_weight * np.linalg.norm(velocity)
            + self.expert_config.curvature_weight * curvature
            + self.expert_config.boundary_weight
            / max(boundary_clearance, 0.01)
            + 0.04 * horizon
            + retry_penalty
        )

    def _set_orientation_from_tangent(self, tangent: np.ndarray) -> None:
        angle = math.atan2(float(tangent[1]), float(tangent[0]))
        angle = (angle + 0.5 * math.pi) % math.pi - 0.5 * math.pi
        cosine = math.cos(angle)
        sine = math.sin(angle)
        yaw = np.array([
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ])
        desired_rotation = yaw @ self._base_grasp_rotation
        self.desired_quat = rotation_to_quat(desired_rotation)
        self.desired_approach_axis = desired_rotation[:, 2].copy()

    def _replan_intercept(self) -> None:
        if self.env.config.motion_mode in {"shape", "combined"}:
            self._refresh_shadow_trajectory(
                max(self.expert_config.candidate_horizons)
                + self.expert_config.shadow_rollout_extra_time
            )
        if (
            self.env.config.motion_mode == "shape"
            and not self.expert_config.search_shape_all_segments
        ):
            target_index = self.env.cable_ids.index(self.env.target_body_id)
            self.expert_segment_index = min(
                target_index, len(self.env.cable_ids) - 2
            )
            self.expert_segment_alpha = float(
                target_index == len(self.env.cable_ids) - 1
            )
            self.expert_horizon = self.config.approach_prediction_horizon
            nodes = self.predict_nodes(self.expert_horizon)
            velocities = self._predicted_node_velocities(self.expert_horizon)
            self.expert_score = self._candidate_cost(
                nodes,
                velocities,
                self.expert_segment_index,
                self.expert_horizon,
                self.expert_segment_alpha,
            )
            if self.expert_config.align_gripper_to_tangent:
                tangent = (
                    nodes[self.expert_segment_index + 1]
                    - nodes[self.expert_segment_index]
                )
                self._set_orientation_from_tangent(tangent)
            self.expert_replans += 1
            self.last_replan_time = float(self.env.data.time)
            return
        margin = self.expert_config.endpoint_margin_nodes
        best: tuple[float, int, float, np.ndarray] | None = None
        for horizon in self.expert_config.candidate_horizons:
            nodes = self.predict_nodes(horizon)
            node_velocities = self._predicted_node_velocities(horizon)
            for index in range(margin, len(nodes) - margin - 1):
                cost = self._candidate_cost(
                    nodes, node_velocities, index, horizon
                )
                if not math.isfinite(cost):
                    continue
                tangent = nodes[index + 1] - nodes[index]
                if best is None or cost < best[0]:
                    best = (cost, index, horizon, tangent)
        if best is None:
            return
        self.expert_score, self.expert_segment_index, self.expert_horizon, tangent = best
        if self.expert_config.align_gripper_to_tangent:
            self._set_orientation_from_tangent(tangent)
        self.expert_replans += 1
        self.last_replan_time = float(self.env.data.time)

    def _locked_segment_position(self, horizon: float) -> np.ndarray:
        if self.locked_segment_index is None:
            raise RuntimeError("no material segment is locked")
        nodes = self.predict_nodes(horizon)
        index = self.locked_segment_index
        alpha = self.locked_segment_alpha
        return (1.0 - alpha) * nodes[index] + alpha * nodes[index + 1]

    def _uses_scripted_controller(self) -> bool:
        return bool(
            self.episode_use_scripted
            or (
                self.env.config.motion_mode == "shape"
                and self.expert_config.shape_use_scripted_fallback
            )
        )

    def _preview_controller(self, use_scripted: bool) -> tuple[bool, bool, bool, float]:
        """Run one controller in an isolated same-seed environment."""

        if self.env.episode_seed is None:
            raise RuntimeError("portfolio preview requires a deterministic episode seed")
        preview_env = CableGraspEnv(copy.deepcopy(self.env.config))
        try:
            seed = self.env.episode_seed
            preview_env.reset(seed=seed)
            if use_scripted:
                # Preserve the experimental base-policy configuration. In
                # particular, a strict-orientation Expert portfolio must not
                # compare against or silently select the position-first baseline.
                policy: DynamicCableGraspPolicy = DynamicCableGraspPolicy(
                    preview_env, self.expert_config
                )
            else:
                preview_config = replace(
                    self.expert_config,
                    dynamic_portfolio_enabled=False,
                    shape_use_scripted_fallback=False,
                    combined_close_capture_distance=0.018,
                )
                policy = FormulaInterceptExpert(preview_env, preview_config)
            min_distance = math.inf
            while (
                not policy.finished
                and preview_env.data.time < preview_env.config.episode_seconds
            ):
                action = policy.action()
                _, _, _, truncated, _ = preview_env.step(action)
                cable = preview_env.data.xpos[preview_env.cable_ids]
                min_distance = min(
                    min_distance,
                    float(np.min(np.linalg.norm(
                        cable - preview_env.hand_position,
                        axis=1,
                    ))),
                )
                if preview_env.ever_success or truncated:
                    break
            return (
                bool(preview_env.ever_success),
                bool(preview_env.ever_confirmed_grasp),
                bool(preview_env.ever_bilateral_candidate),
                float(min_distance),
            )
        finally:
            preview_env.close()

    @staticmethod
    def _preview_rank(outcome: tuple[bool, bool, bool, float]) -> tuple[int, int, int, float]:
        success, confirmed, bilateral, min_distance = outcome
        return (
            int(success), int(confirmed), int(bilateral), -float(min_distance),
        )

    def _select_portfolio_controller(self) -> None:
        """Use privileged same-seed rollouts to select the live controller."""

        formula = self._preview_controller(use_scripted=False)
        self.portfolio_formula_success = formula[0]
        if formula[0]:
            self.episode_use_scripted = False
            return
        scripted = self._preview_controller(use_scripted=True)
        self.portfolio_scripted_success = scripted[0]
        self.episode_use_scripted = (
            scripted[0]
            or self._preview_rank(scripted) > self._preview_rank(formula)
        )

    def _selected_segment(self, horizon: float) -> np.ndarray:
        nodes = self.predict_nodes(horizon)
        index = self.expert_segment_index
        alpha = self.expert_segment_alpha
        return (1.0 - alpha) * nodes[index] + alpha * nodes[index + 1]

    def _predicted_segment(
        self, prediction_horizon: float | None = None,
    ) -> np.ndarray:
        if self.env.config.motion_mode == "static" or self._uses_scripted_controller():
            return super()._predicted_segment(prediction_horizon)
        if self.locked_segment_index is not None:
            if prediction_horizon is None:
                prediction_horizon = (
                    self.config.close_prediction_horizon
                    if self.phase is Phase.CLOSE
                    else self.expert_config.intercept_prediction_horizon
                )
            point = self._locked_segment_position(prediction_horizon)
        else:
            horizon = (
                self.expert_horizon
                if prediction_horizon is None else prediction_horizon
            )
            point = self._selected_segment(horizon)
        point = point.copy()
        point[0] = np.clip(point[0], *self.config.intercept_x_limits)
        point[1] = np.clip(point[1], *self.config.intercept_y_limits)
        point[2] = np.clip(point[2], *self.config.intercept_z_limits)
        self.filtered_target += self.config.target_filter_alpha * (
            point - self.filtered_target
        )
        return self.filtered_target.copy()

    def action(self) -> np.ndarray:
        if (
            self.phase in {Phase.SETTLE, Phase.APPROACH}
            and self.env.config.motion_mode != "static"
            and not self._uses_scripted_controller()
            and self.env.data.time - self.last_replan_time
            >= self.expert_config.replan_interval
        ):
            self._replan_intercept()
        return super().action()

    def _begin_vertical_recovery(self, hand: np.ndarray) -> None:
        if (
            (
                self.phase is Phase.CLOSE
                or (
                    self.phase is Phase.INTERCEPT
                    and self.expert_config.record_intercept_failures
                )
            )
            and self.env.config.motion_mode != "static"
            and not self._uses_scripted_controller()
        ):
            attempted = (
                self.locked_segment_index
                if self.locked_segment_index is not None
                else self.expert_segment_index
            )
            self.failed_segment_indices.append(int(attempted))
        super()._begin_vertical_recovery(hand)
        self.last_replan_time = -math.inf

    def _close_capture_distance(self) -> float:
        if self._uses_scripted_controller():
            return super()._close_capture_distance()
        if self.env.config.motion_mode == "combined":
            return self.expert_config.combined_close_capture_distance
        return super()._close_capture_distance()

    def expert_info(self) -> dict[str, float | int | bool | str | None]:
        return {
            "expert_segment_index": int(self.expert_segment_index),
            "expert_prediction_horizon": float(self.expert_horizon),
            "expert_candidate_score": float(self.expert_score),
            "expert_replans": int(self.expert_replans),
            "expert_failed_segment_count": len(self.failed_segment_indices),
            "expert_shadow_rollouts": int(self.shadow_rollouts),
            "expert_shadow_physics_steps": int(self.shadow_physics_steps),
            "expert_selected_controller": (
                "scripted"
                if (
                    self.env.config.motion_mode == "static"
                    or self._uses_scripted_controller()
                )
                else "formula"
            ),
            "expert_portfolio_formula_success": self.portfolio_formula_success,
            "expert_portfolio_scripted_success": self.portfolio_scripted_success,
        }
