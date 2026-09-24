"""Stress-test the RL task-space IK while the open gripper touches the table.

The diagnostic uses the exact 5-D action conversion from ``RLCableGraspEnv``.
It finds a conservative far pose with the reset orientation, approaches that
pose above the table, lowers the fingertips to the table, and then commands an
inward Cartesian sweep while maintaining the same fingertip height.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from panda_cable_grasp.runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import cv2
import mujoco
import numpy as np

from panda_cable_grasp.env.environment import EnvConfig
from panda_cable_grasp.env.kinematics import quat_error, rotation_to_quat
from panda_cable_grasp.rl.environment import RLCableGraspEnv


def _tilt_degrees(rotation: np.ndarray) -> float:
    approach_axis = np.asarray(rotation, dtype=float)[:, 2]
    cosine = float(np.clip(np.dot(approach_axis, [0.0, 0.0, -1.0]), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _orientation_error_degrees(
    current_rotation: np.ndarray, desired_rotation: np.ndarray,
) -> float:
    error = quat_error(
        rotation_to_quat(current_rotation), rotation_to_quat(desired_rotation)
    )
    sine_half = float(np.clip(0.5 * np.linalg.norm(error), 0.0, 1.0))
    return math.degrees(2.0 * math.asin(sine_half))


def _pad_minimum_z(env: RLCableGraspEnv) -> float:
    minimum = math.inf
    for geom_id in env.base_env.pad_geom_ids:
        if env.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        rotation = env.data.geom_xmat[geom_id].reshape(3, 3)
        half_size = env.model.geom_size[geom_id]
        vertical_extent = float(np.dot(np.abs(rotation[2]), half_size))
        minimum = min(
            minimum,
            float(env.data.geom_xpos[geom_id, 2]) - vertical_extent,
        )
    if not math.isfinite(minimum):
        raise RuntimeError("no box-shaped fingertip pad geometry was found")
    return minimum


def _finger_table_contacts(env: RLCableGraspEnv) -> int:
    finger_bodies = {env.base_env.left_finger_id, env.base_env.right_finger_id}
    table_geom = env.base_env.table_geom_id
    count = 0
    for contact in env.data.contact[:env.data.ncon]:
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(env.model.geom_bodyid[geom1])
        body2 = int(env.model.geom_bodyid[geom2])
        if (
            (geom1 == table_geom and body2 in finger_bodies)
            or (geom2 == table_geom and body1 in finger_bodies)
        ):
            count += 1
    return count


def _solve_pose_ik(
    env: RLCableGraspEnv,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    initial_qpos: np.ndarray,
    *,
    iterations: int = 500,
) -> tuple[np.ndarray, float, float, float]:
    """Solve a pose only to select a conservative diagnostic endpoint."""

    qpos = np.asarray(initial_qpos, dtype=float).copy()
    joint_range = env.model.jnt_range[env.base_env.arm_joint_ids]
    for _ in range(iterations):
        env.data.qpos[env.base_env.arm_qpos_adr] = qpos
        env.data.qvel[env.base_env.arm_dof_adr] = 0.0
        mujoco.mj_forward(env.model, env.data)
        position_error = target_position - env.base_env.hand_position
        current_rotation = env.data.xmat[env.base_env.hand_id].reshape(3, 3)
        rotation_error = quat_error(
            rotation_to_quat(current_rotation), rotation_to_quat(target_rotation)
        )
        if np.linalg.norm(position_error) < 2e-4 and np.linalg.norm(rotation_error) < 2e-4:
            break
        jacp, jacr = env.base_env._hand_jacobian()
        jacobian = np.vstack((jacp, jacr))[:, env.base_env.arm_dof_adr]
        error = np.concatenate((position_error, rotation_error))
        damping = env.rl_config.ik_damping
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), error
        )
        maximum = float(np.max(np.abs(delta)))
        if maximum > 0.08:
            delta *= 0.08 / maximum
        qpos = np.clip(qpos + delta, joint_range[:, 0], joint_range[:, 1])

    env.data.qpos[env.base_env.arm_qpos_adr] = qpos
    env.data.qvel[env.base_env.arm_dof_adr] = 0.0
    mujoco.mj_forward(env.model, env.data)
    final_position = env.base_env.hand_position
    final_rotation = env.data.xmat[env.base_env.hand_id].reshape(3, 3)
    position_error = float(np.linalg.norm(target_position - final_position))
    orientation_error = _orientation_error_degrees(final_rotation, target_rotation)
    margin = float(np.min(np.minimum(
        qpos - joint_range[:, 0], joint_range[:, 1] - qpos
    )))
    return qpos, position_error, orientation_error, margin


def _select_far_target(
    env: RLCableGraspEnv,
    target_z: float,
    sweep_angle_radians: float,
    target_rotation: np.ndarray,
    reach_margin: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    ready_qpos = env.data.qpos[env.base_env.arm_qpos_adr].copy()
    feasible: list[tuple[float, np.ndarray, float, float, float]] = []
    guess = ready_qpos
    try:
        for radius in np.arange(0.55, 0.901, 0.01):
            target = np.array([
                radius * math.cos(sweep_angle_radians),
                radius * math.sin(sweep_angle_radians),
                target_z,
            ])
            guess, position_error, orientation_error, joint_margin = _solve_pose_ik(
                env, target, target_rotation, guess
            )
            if (
                position_error <= 0.005
                and orientation_error <= 1.0
                and joint_margin >= 0.04
            ):
                feasible.append((
                    float(radius), guess.copy(), position_error,
                    orientation_error, joint_margin,
                ))
        if not feasible:
            raise RuntimeError("no conservative vertical ground pose was found")

        scanned_radius = feasible[-1][0]
        selected_radius = max(feasible[0][0], scanned_radius - reach_margin)
        selected_target = np.array([
            selected_radius * math.cos(sweep_angle_radians),
            selected_radius * math.sin(sweep_angle_radians),
            target_z,
        ])
        selected_qpos, position_error, orientation_error, joint_margin = _solve_pose_ik(
            env, selected_target, target_rotation, feasible[-1][1]
        )
        return selected_target, selected_qpos, {
            "scanned_maximum_radius_m": scanned_radius,
            "selected_radius_m": selected_radius,
            "selection_margin_m": scanned_radius - selected_radius,
            "offline_position_error_m": position_error,
            "offline_orientation_error_deg": orientation_error,
            "offline_minimum_joint_margin_rad": joint_margin,
        }
    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(env.model, env.data)


def _camera(
    *, lookat: np.ndarray, distance: float, azimuth: float, elevation: float,
) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = float(distance)
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    return camera


def _draw_text(
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int] = (14, 26),
    color: tuple[int, int, int] = (245, 245, 245),
) -> None:
    x, y = origin
    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        for line in lines
    ]
    box_width = max((size[0] for size in sizes), default=0) + 18
    box_height = 24 * len(lines) + 8
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (max(0, x - 8), max(0, y - 21)),
        (min(image.shape[1] - 1, x - 8 + box_width), min(image.shape[0] - 1, y - 21 + box_height)),
        (8, 12, 18),
        -1,
    )
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0.0, image)
    for line in lines:
        cv2.putText(
            image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            color, 1, cv2.LINE_AA,
        )
        y += 24


def _draw_chart(
    panel: np.ndarray,
    rows: list[dict[str, float | str | int]],
    *,
    start_time: float,
) -> None:
    height, width = panel.shape[:2]
    margin_left, margin_right = 58, 18
    top, bottom = 28, height - 35
    cv2.rectangle(panel, (margin_left, top), (width - margin_right, bottom), (90, 90, 90), 1)
    if len(rows) < 2:
        return
    times = np.asarray([float(row["time_s"]) - start_time for row in rows])
    duration = max(float(times[-1]), 1e-6)
    x_values = margin_left + (width - margin_left - margin_right) * times / duration
    traces = (
        ("tilt deg", "tilt_deg", (50, 210, 255), 10.0),
        ("lateral mm", "lateral_drift_mm", (80, 255, 90), 20.0),
        ("height err mm", "height_error_mm", (255, 150, 70), 20.0),
    )
    for index, (label, key, color, scale) in enumerate(traces):
        values = np.asarray([float(row[key]) for row in rows])
        clipped = np.clip(values, -scale, scale)
        y_values = (top + bottom) / 2.0 - clipped / scale * (bottom - top) / 2.0
        points = np.column_stack((x_values, y_values)).astype(np.int32)
        cv2.polylines(panel, [points], False, color, 2, cv2.LINE_AA)
        cv2.putText(
            panel, f"{label} (+/-{scale:g})", (margin_left + index * 230, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
        )
    cv2.line(
        panel, (margin_left, int((top + bottom) / 2)),
        (width - margin_right, int((top + bottom) / 2)), (120, 120, 120), 1,
    )
    cv2.putText(
        panel, f"0 s", (margin_left, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA,
    )
    cv2.putText(
        panel, f"{duration:.1f} s", (width - 80, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA,
    )


def _write_video_frame(
    renderer: mujoco.Renderer,
    writer: cv2.VideoWriter,
    env: RLCableGraspEnv,
    cameras: tuple[mujoco.MjvCamera, mujoco.MjvCamera],
    rows: list[dict[str, float | str | int]],
    current: dict[str, float | str | int],
    target: np.ndarray,
    start_time: float,
) -> np.ndarray:
    views: list[np.ndarray] = []
    for camera in cameras:
        renderer.update_scene(env.data, camera=camera)
        rgb = renderer.render().copy()
        views.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    frame = np.concatenate(views, axis=1)
    phase = str(current["phase"])
    if phase == "rotate_closing_axis":
        experiment_label = "SETUP: ROTATE OPEN GRIPPER PERPENDICULAR TO RADIAL MOTION"
    elif phase == "verified_inner_pose_hold":
        experiment_label = "SETUP: VERIFIED INNER VERTICAL POSE; GRIPPER OPEN"
    elif phase == "outward_hover_sweep":
        experiment_label = "A: FAST RL DELTA OUTWARD TO 0.80 M"
    elif phase in {"move_to_far_hover", "vertical_descent", "far_hover_hold"}:
        experiment_label = "A: RL DELTA APPROACH TO FAR HOVER TARGET"
    elif phase == "verified_far_pose_hold":
        experiment_label = "SETUP: OFFLINE-VERIFIED FAR VERTICAL POSE"
    elif phase in {"inward_hover_sweep", "inward_hover_hold"}:
        experiment_label = "B: RL INWARD HOVER SWEEP (NO TABLE CONTACT)"
    else:
        experiment_label = "INITIAL SETTLE"
    _draw_text(frame, [
        experiment_label,
        f"phase: {phase}",
        f"time: {float(current['time_s']) - start_time:5.2f} s",
        (
            "TCP xyz: "
            f"{float(current['tcp_x_m']):+.3f}, "
            f"{float(current['tcp_y_m']):+.3f}, "
            f"{float(current['tcp_z_m']):+.3f} m"
        ),
        f"target xyz: {target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f} m",
    ])
    _draw_text(frame, [
        f"tilt from down: {float(current['tilt_deg']):6.3f} deg",
        f"orientation error: {float(current['orientation_error_deg']):6.3f} deg",
        f"pad clearance: {float(current['pad_clearance_mm']):+7.2f} mm",
        (
            "TABLE CONTACT: YES" if int(current["finger_table_contacts"]) > 0
            else "TABLE CONTACT: NO"
        ),
        f"finger opening: {float(current['finger_opening_mm']):6.2f} mm",
        f"closing-axis perp error: {float(current['closing_axis_perp_error_deg']):5.2f} deg",
        f"lateral drift: {float(current['lateral_drift_mm']):+7.2f} mm",
        f"IK scale: {float(current['ik_velocity_scale']):.3f}",
    ], origin=(655, 26))
    panel = np.full((210, frame.shape[1], 3), 24, dtype=np.uint8)
    _draw_chart(panel, rows, start_time=start_time)
    composite = np.concatenate((frame, panel), axis=0)
    writer.write(composite)
    return composite


def run(args: argparse.Namespace) -> Path:
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    config = EnvConfig(
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        motion_mode="static",
        motion_profile_version="factorized_v2",
        disturbance_strength=0.0,
    )
    env = RLCableGraspEnv(env_config=config)
    renderer: mujoco.Renderer | None = None
    writer: cv2.VideoWriter | None = None
    try:
        env.reset(seed=args.seed)
        initial_rotation = env.data.xmat[env.base_env.hand_id].reshape(3, 3).copy()
        initial_tilt = _tilt_degrees(initial_rotation)
        table_top_z = float(
            env.model.geom_pos[env.base_env.table_geom_id, 2]
            + env.model.geom_size[env.base_env.table_geom_id, 2]
        )
        pad_offset_below_tcp = float(env.base_env.hand_position[2] - _pad_minimum_z(env))
        contact_tcp_z = table_top_z + pad_offset_below_tcp + args.pad_clearance
        sweep_angle = math.radians(args.sweep_angle_deg)
        radial = np.array([math.cos(sweep_angle), math.sin(sweep_angle), 0.0])
        tangent = np.array([-radial[1], radial[0]])
        initial_closing_axis = initial_rotation[:2, 1]
        if float(np.dot(initial_closing_axis, tangent)) < 0.0:
            tangent = -tangent
        yaw_to_perpendicular = math.atan2(
            initial_closing_axis[0] * tangent[1]
            - initial_closing_axis[1] * tangent[0],
            float(np.dot(initial_closing_axis, tangent)),
        )
        target_rotation = (
            env._rotation_about_z(yaw_to_perpendicular) @ initial_rotation
        )
        far_target, far_qpos, reach_diagnostics = _select_far_target(
            env, contact_tcp_z, sweep_angle, target_rotation, args.reach_margin
        )
        far_radius = float(np.linalg.norm(far_target[:2]))
        inward_radius = max(args.minimum_inward_radius, far_radius - args.inward_distance)
        inward_target = np.array([
            inward_radius * math.cos(sweep_angle),
            inward_radius * math.sin(sweep_angle),
            contact_tcp_z,
        ])
        inner_qpos, inner_position_error, inner_orientation_error, inner_joint_margin = (
            _solve_pose_ik(
                env, inward_target, target_rotation, far_qpos,
            )
        )
        reach_diagnostics.update({
            "inner_offline_position_error_m": inner_position_error,
            "inner_offline_orientation_error_deg": inner_orientation_error,
            "inner_offline_minimum_joint_margin_rad": inner_joint_margin,
        })
        hover_target = far_target.copy()
        hover_target[2] = max(args.hover_height, contact_tcp_z + 0.10)

        width, height = args.view_width, args.view_height
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        side_azimuth = math.degrees(math.atan2(radial[1], radial[0])) + 90.0
        lookat = np.array([
            0.5 * (far_target[0] + inward_target[0]),
            0.5 * (far_target[1] + inward_target[1]),
            0.18,
        ])
        cameras = (
            _camera(
                lookat=lookat, distance=1.25,
                azimuth=side_azimuth, elevation=-12.0,
            ),
            _camera(
                lookat=lookat, distance=1.35,
                azimuth=side_azimuth + 42.0, elevation=-28.0,
            ),
        )
        video_path = run_dir / "rl_vertical_hover_sweep.mp4"
        frame_size = (2 * width, height + 210)
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
            args.video_fps, frame_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {video_path}")

        rows: list[dict[str, float | str | int]] = []
        start_time = float(env.data.time)
        next_frame_time = start_time
        last_frame: np.ndarray | None = None
        maximum_tilt_frame: np.ndarray | None = None
        maximum_recorded_inward_tilt = -math.inf
        path_origin = far_target[:2].copy()
        inward_direction = -radial[:2]

        def sample(phase: str, target: np.ndarray, info: dict) -> None:
            nonlocal next_frame_time, last_frame
            nonlocal maximum_tilt_frame, maximum_recorded_inward_tilt
            position = env.base_env.hand_position.copy()
            rotation = env.data.xmat[env.base_env.hand_id].reshape(3, 3).copy()
            displacement = position[:2] - path_origin
            lateral_axis = np.array([-inward_direction[1], inward_direction[0]])
            row: dict[str, float | str | int] = {
                "time_s": float(env.data.time),
                "phase": phase,
                "tcp_x_m": float(position[0]),
                "tcp_y_m": float(position[1]),
                "tcp_z_m": float(position[2]),
                "target_x_m": float(target[0]),
                "target_y_m": float(target[1]),
                "target_z_m": float(target[2]),
                "position_error_mm": 1000.0 * float(np.linalg.norm(target - position)),
                "height_error_mm": 1000.0 * float(position[2] - contact_tcp_z),
                "lateral_drift_mm": 1000.0 * float(np.dot(displacement, lateral_axis)),
                "tilt_deg": _tilt_degrees(rotation),
                "orientation_error_deg": _orientation_error_degrees(
                    rotation, target_rotation
                ),
                "pad_clearance_mm": 1000.0 * (
                    _pad_minimum_z(env) - table_top_z
                ),
                "finger_table_contacts": _finger_table_contacts(env),
                "finger_opening_mm": 1000.0 * float(
                    np.sum(env.data.qpos[env.base_env.finger_qpos_adr])
                ),
                "closing_axis_perp_error_deg": math.degrees(math.asin(np.clip(
                    abs(float(np.dot(
                        rotation[:2, 1] / max(np.linalg.norm(rotation[:2, 1]), 1e-12),
                        radial[:2],
                    ))),
                    0.0, 1.0,
                ))),
                "ik_velocity_scale": float(info.get("ik_velocity_scale", 1.0)),
            }
            rows.append(row)
            if float(env.data.time) + 1e-12 >= next_frame_time:
                last_frame = _write_video_frame(
                    renderer, writer, env, cameras, rows, row, target, start_time
                )
                if (
                    phase == "inward_hover_sweep"
                    and float(row["tilt_deg"]) > maximum_recorded_inward_tilt
                ):
                    maximum_recorded_inward_tilt = float(row["tilt_deg"])
                    maximum_tilt_frame = last_frame.copy()
                next_frame_time += 1.0 / args.video_fps

        def action_towards(target: np.ndarray, feedforward: np.ndarray | None = None) -> np.ndarray:
            error = target - env.base_env.hand_position
            delta = args.position_gain * error
            if feedforward is not None:
                delta += feedforward * (
                    env.model.opt.timestep * env.base_env.config.frame_skip
                )
            delta /= env.rl_config.translation_delta_scale
            norm = float(np.linalg.norm(delta))
            if norm > 1.0:
                delta /= norm
            current_closing_axis = env._desired_hand_rotation[:2, 1]
            desired_closing_axis = target_rotation[:2, 1]
            yaw_error = math.atan2(
                current_closing_axis[0] * desired_closing_axis[1]
                - current_closing_axis[1] * desired_closing_axis[0],
                float(np.dot(current_closing_axis, desired_closing_axis)),
            )
            yaw_action = float(np.clip(
                yaw_error / env.rl_config.yaw_delta_scale, -1.0, 1.0
            ))
            # +1 is explicitly the open command; this test never requests closure.
            return np.r_[delta, yaw_action, 1.0].astype(np.float32)

        def execute(phase: str, duration: float, target_at_time, velocity_at_time=None) -> None:
            steps = int(math.ceil(duration / (
                env.model.opt.timestep * env.base_env.config.frame_skip
            )))
            for index in range(steps):
                alpha = min(1.0, (index + 1) / max(steps, 1))
                target = np.asarray(target_at_time(alpha), dtype=float)
                velocity = (
                    None if velocity_at_time is None
                    else np.asarray(velocity_at_time(alpha), dtype=float)
                )
                action = action_towards(target, velocity)
                _, _, terminated, truncated, info = env.step(action)
                sample(phase, target, info)
                if terminated or truncated:
                    raise RuntimeError(
                        f"episode ended during {phase}: "
                        f"reason={info.get('termination_reason')}"
                    )

        def install_verified_pose(qpos: np.ndarray) -> None:
            env.data.qpos[env.base_env.arm_qpos_adr] = qpos
            env.data.qpos[env.base_env.finger_qpos_adr] = 0.04
            env.data.qvel[:] = 0.0
            env.data.ctrl[:7] = qpos
            env.data.ctrl[7] = float(env.model.actuator_ctrlrange[7, 1])
            env.base_env._last_requested_action[:7] = qpos
            env.base_env._last_applied_action[:7] = qpos
            env.base_env._last_requested_action[7] = env.data.ctrl[7]
            env.base_env._last_applied_action[7] = env.data.ctrl[7]
            env.base_env._previous_arm_command_velocity[:] = 0.0
            env._desired_hand_rotation = target_rotation.copy()
            env._previous_action[:] = 0.0
            env._gripper_closed = False
            mujoco.mj_forward(env.model, env.data)

        if args.radial_only:
            # Isolate exactly the requested radial motion.  Start from a
            # verified inner pose, then use the public RL delta action at its
            # maximum translation command first outward and then inward.
            install_verified_pose(inner_qpos)
            execute("verified_inner_pose_hold", 0.50, lambda _: inward_target)
            radial_duration = args.inward_distance / args.radial_speed
            outward_velocity = args.radial_speed * radial
            execute(
                "outward_hover_sweep", radial_duration,
                lambda alpha: (1.0 - alpha) * inward_target + alpha * far_target,
                lambda _: outward_velocity,
            )
            execute("far_hover_hold", args.far_hold_seconds, lambda _: far_target)
        else:
            ready = env.base_env.hand_position.copy()
            execute("settle", 0.40, lambda _: ready)
            execute("rotate_closing_axis", args.rotate_seconds, lambda _: ready)
            execute("move_to_far_hover", args.approach_seconds, lambda _: hover_target)
            descent_start = hover_target.copy()
            execute(
                "vertical_descent", args.descent_seconds,
                lambda alpha: (1.0 - alpha) * descent_start + alpha * far_target,
            )
            execute("far_hover_hold", 0.60, lambda _: far_target)

            # Separate approach quality from the inward-sweep question.
            install_verified_pose(far_qpos)
            execute("verified_far_pose_hold", 0.80, lambda _: far_target)

        if not args.radial_only:
            inward_duration = args.inward_distance / args.inward_speed
            inward_velocity = args.inward_speed * inward_direction
            inward_target_at_time = (
                lambda alpha: (1.0 - alpha) * far_target + alpha * inward_target
            )
            inward_feedforward = lambda _: np.r_[inward_velocity, 0.0]
        else:
            inward_duration = radial_duration
            inward_velocity = -args.radial_speed * radial
            inward_target_at_time = (
                lambda alpha: (1.0 - alpha) * far_target + alpha * inward_target
            )
            inward_feedforward = lambda _: inward_velocity

        execute(
            "inward_hover_sweep", inward_duration,
            inward_target_at_time, inward_feedforward,
        )
        execute("inward_hover_hold", 0.60, lambda _: inward_target)

        if last_frame is not None:
            cv2.imwrite(str(run_dir / "final_frame.png"), last_frame)
        if maximum_tilt_frame is not None:
            cv2.imwrite(str(run_dir / "maximum_tilt_frame.png"), maximum_tilt_frame)

        csv_path = run_dir / "metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(rows)

        inward_rows = [row for row in rows if row["phase"] == "inward_hover_sweep"]
        if args.radial_only:
            approach_rows = [
                row for row in rows
                if row["phase"] in {"outward_hover_sweep", "far_hover_hold"}
            ]
            far_hold_rows = [
                row for row in rows if row["phase"] == "far_hover_hold"
            ]
        else:
            approach_rows = [
                row for row in rows
                if row["phase"] in {
                    "move_to_far_hover", "vertical_descent", "far_hover_hold"
                }
            ]
            far_hold_rows = [
                row for row in rows if row["phase"] == "verified_far_pose_hold"
            ]
        contact_rows = [
            row for row in inward_rows
            if int(row["finger_table_contacts"]) > 0
            or float(row["pad_clearance_mm"]) <= 1.0
        ]
        physical_contact_rows = [
            row for row in inward_rows
            if int(row["finger_table_contacts"]) > 0
        ]
        near_table_rows = [
            row for row in inward_rows
            if float(row["pad_clearance_mm"]) <= 1.0
        ]
        summary = {
            "seed": args.seed,
            "control_frequency_hz": 1.0 / (
                env.model.opt.timestep * env.base_env.config.frame_skip
            ),
            "video_fps": args.video_fps,
            "initial_tilt_deg": initial_tilt,
            "table_top_z_m": table_top_z,
            "pad_offset_below_tcp_m": pad_offset_below_tcp,
            "commanded_contact_tcp_z_m": contact_tcp_z,
            "far_target_m": far_target.tolist(),
            "inward_target_m": inward_target.tolist(),
            "inward_distance_m": args.inward_distance,
            "inward_speed_m_s": (
                args.radial_speed if args.radial_only else args.inward_speed
            ),
            "yaw_to_perpendicular_deg": math.degrees(yaw_to_perpendicular),
            "reach_margin_m": args.reach_margin,
            "reach_selection": reach_diagnostics,
            "approach_final_position_error_mm": float(
                approach_rows[-1]["position_error_mm"]
            ),
            "approach_final_tilt_deg": float(approach_rows[-1]["tilt_deg"]),
            "verified_far_hold_final_position_error_mm": float(
                far_hold_rows[-1]["position_error_mm"]
            ),
            "verified_far_hold_final_tilt_deg": float(
                far_hold_rows[-1]["tilt_deg"]
            ),
            "verified_far_hold_min_pad_clearance_mm": min(
                float(row["pad_clearance_mm"]) for row in far_hold_rows
            ),
            "inward_sample_count": len(inward_rows),
            "inward_physical_contact_sample_fraction": (
                len(physical_contact_rows) / len(inward_rows) if inward_rows else 0.0
            ),
            "inward_near_table_sample_fraction": (
                len(near_table_rows) / len(inward_rows) if inward_rows else 0.0
            ),
            "inward_contact_or_near_sample_fraction": (
                len(contact_rows) / len(inward_rows) if inward_rows else 0.0
            ),
            "inward_max_tilt_deg": max(float(row["tilt_deg"]) for row in inward_rows),
            "inward_rms_tilt_deg": float(np.sqrt(np.mean([
                float(row["tilt_deg"]) ** 2 for row in inward_rows
            ]))),
            "inward_max_abs_lateral_drift_mm": max(
                abs(float(row["lateral_drift_mm"])) for row in inward_rows
            ),
            "inward_max_abs_height_error_mm": max(
                abs(float(row["height_error_mm"])) for row in inward_rows
            ),
            "inward_min_pad_clearance_mm": min(
                float(row["pad_clearance_mm"]) for row in inward_rows
            ),
            "inward_max_pad_clearance_mm": max(
                float(row["pad_clearance_mm"]) for row in inward_rows
            ),
            "inward_max_position_error_mm": max(
                float(row["position_error_mm"]) for row in inward_rows
            ),
            "inward_max_closing_axis_perp_error_deg": max(
                float(row["closing_axis_perp_error_deg"]) for row in inward_rows
            ),
            "inward_final_closing_axis_perp_error_deg": float(
                inward_rows[-1]["closing_axis_perp_error_deg"]
            ),
            "all_phase_physical_contact_sample_fraction": sum(
                int(row["finger_table_contacts"]) > 0 for row in rows
            ) / len(rows),
            "all_phase_min_pad_clearance_mm": min(
                float(row["pad_clearance_mm"]) for row in rows
            ),
            "all_phase_min_finger_opening_mm": min(
                float(row["finger_opening_mm"]) for row in rows
            ),
            "post_rotation_max_closing_axis_perp_error_deg": max(
                float(row["closing_axis_perp_error_deg"])
                for row in rows
                if row["phase"] not in {"settle", "rotate_closing_axis"}
            ),
            "files": {
                "video": video_path.name,
                "metrics": csv_path.name,
                "final_frame": "final_frame.png",
                "maximum_tilt_frame": "maximum_tilt_frame.png",
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"output_dir={run_dir.resolve()}")
        return run_dir
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPOSITORY_ROOT / "outputs" / "rl_vertical_ground_sweep",
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--sweep-angle-deg", type=float, default=25.0)
    parser.add_argument("--pad-clearance", type=float, default=0.0)
    parser.add_argument("--reach-margin", type=float, default=0.04)
    parser.add_argument("--hover-height", type=float, default=0.15)
    parser.add_argument("--inward-distance", type=float, default=0.25)
    parser.add_argument("--minimum-inward-radius", type=float, default=0.35)
    parser.add_argument("--inward-speed", type=float, default=0.05)
    parser.add_argument("--rotate-seconds", type=float, default=1.60)
    parser.add_argument("--approach-seconds", type=float, default=2.00)
    parser.add_argument("--descent-seconds", type=float, default=1.20)
    parser.add_argument("--radial-only", action="store_true")
    parser.add_argument("--radial-speed", type=float, default=0.15)
    parser.add_argument("--far-hold-seconds", type=float, default=1.50)
    parser.add_argument("--position-gain", type=float, default=0.35)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--view-width", type=int, default=640)
    parser.add_argument("--view-height", type=int, default=480)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
