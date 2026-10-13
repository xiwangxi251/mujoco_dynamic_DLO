"""Material-node shape amplitude and speed metrics for range-motion runs."""
from __future__ import annotations

import numpy as np


def _proper_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return R such that row vectors in source @ R best match target."""
    u, _, vt = np.linalg.svd(source.T @ target)
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    return u @ correction @ vt


def _rotation_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stable shortest proper rotation; used for nearly straight cables."""
    a = a / max(float(np.linalg.norm(a)), 1e-15)
    b = b / max(float(np.linalg.norm(b)), 1e-15)
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(a, np.array([1., 0., 0.]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0., 1., 0.]))
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = cross / sine
    skew = np.array([[0., -k[2], k[1]], [k[2], 0., -k[0]], [-k[1], k[0], 0.]])
    return np.eye(3) + skew * sine + (skew @ skew) * (1.0 - cosine)


def aligned_shape_positions(positions: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remove each frame's mass-weighted translation and proper rigid rotation.

    Frames are aligned to the first frame. When the reference is effectively a
    line, its principal axis determines the observable rotation; roll around
    that axis is underdetermined and is fixed by the shortest rotation.
    """
    xyz = np.asarray(positions, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] != 3 or xyz.shape[1] != len(mass):
        raise ValueError("positions must have shape (frames, nodes, 3)")
    if len(xyz) < 2 or not np.isfinite(xyz).all() or np.any(mass <= 0):
        raise ValueError("need finite multi-frame positions and positive masses")
    w = mass / mass.sum()
    centers = np.einsum("tni,n->ti", xyz, w)
    centered = xyz - centers[:, None, :]
    reference = centered[0]
    weighted = reference * np.sqrt(w[:, None])
    singular = np.linalg.svd(weighted, compute_uv=False)
    near_line = len(singular) < 2 or singular[1] <= max(singular[0], 1e-15) * 1e-6
    if near_line:
        _, _, vt = np.linalg.svd(weighted, full_matrices=False)
        axis = vt[0]
        aligned = np.empty_like(centered)
        aligned[0] = centered[0]
        for i in range(1, len(centered)):
            frame_weighted = centered[i] * np.sqrt(w[:, None])
            _, _, fvt = np.linalg.svd(frame_weighted, full_matrices=False)
            frame_axis = fvt[0]
            if np.dot(frame_axis, axis) < 0:
                frame_axis = -frame_axis
            rotation = _rotation_from_vectors(frame_axis, axis)
            # _rotation_from_vectors uses column-vector convention; positions
            # are stored as row vectors throughout this module.
            aligned[i] = centered[i] @ rotation.T
    else:
        aligned = np.empty_like(centered)
        for i, frame in enumerate(centered):
            rotation = _proper_rotation(frame * np.sqrt(w[:, None]), weighted)
            aligned[i] = frame @ rotation
    return aligned, {"near_line_alignment": bool(near_line), "reference_singular_values": singular.tolist()}


def measure_range_motion(times: np.ndarray, positions: np.ndarray, masses: np.ndarray,
                         node_velocities: np.ndarray | None = None,
                         window: tuple[float, float] = (0.82, 7.42)) -> dict:
    """Compute time-weighted A (mm), V (m/s), and whole-object speed.

    A is the mass/time weighted RMS of aligned node positions after subtracting
    each node's time mean. V is mean mass-weighted residual node speed after
    least-squares rigid translation and angular velocity removal. Velocities
    use simulator node velocities for acceptance; finite-difference velocities
    remain a separately labelled diagnostic. Integration clips exactly to the
    fixed physical window.
    """
    time = np.asarray(times, dtype=np.float64)
    xyz = np.asarray(positions, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    if time.ndim != 1 or len(time) != len(xyz) or np.any(np.diff(time) <= 0):
        raise ValueError("times must be strictly increasing and match positions")
    start, end = map(float, window)
    if time[0] > start + 1e-9 or time[-1] < end - 1e-9:
        raise ValueError("trajectory does not cover the complete fixed window")
    if len(time) < 3:
        raise ValueError("insufficient samples in fixed window")
    finite_difference_velocity = np.gradient(xyz, time, axis=0, edge_order=2)
    actual_velocity = (finite_difference_velocity if node_velocities is None
                       else np.asarray(node_velocities, dtype=np.float64))
    if actual_velocity.shape != xyz.shape or not np.isfinite(actual_velocity).all():
        raise ValueError("node_velocities must be finite and match positions")
    query_time = np.r_[start, time[(time > start) & (time < end)], end]
    query_xyz = np.stack([
        np.column_stack([np.interp(query_time, time, xyz[:, node, axis]) for axis in range(3)])
        for node in range(xyz.shape[1])
    ], axis=1)
    query_velocity = np.stack([
        np.column_stack([np.interp(query_time, time, actual_velocity[:, node, axis]) for axis in range(3)])
        for node in range(xyz.shape[1])
    ], axis=1)
    query_fd_velocity = np.stack([
        np.column_stack([np.interp(query_time, time, finite_difference_velocity[:, node, axis]) for axis in range(3)])
        for node in range(xyz.shape[1])
    ], axis=1)
    time, xyz = query_time, query_xyz
    aligned, align_meta = aligned_shape_positions(xyz, mass)
    # Trapezoidal weights make the metric a physical-time average.
    dt = np.diff(time)
    tw = np.empty_like(time)
    tw[0], tw[-1] = dt[0] / 2, dt[-1] / 2
    if len(time) > 2:
        tw[1:-1] = (dt[:-1] + dt[1:]) / 2
    tw /= tw.sum()
    w = mass / mass.sum()
    mean_shape = np.einsum("t,tni->ni", tw, aligned)
    deviation = aligned - mean_shape[None, :, :]
    amp_m = np.sqrt(np.einsum("t,n,tni,tni->", tw, w, deviation, deviation))

    com = np.einsum("tni,n->ti", xyz, w)
    vcom = np.einsum("tni,n->ti", query_velocity, w)
    residual_velocity = np.empty_like(query_velocity)
    whole_rigid_velocity = np.empty_like(query_velocity)
    omega_values = np.empty((len(time), 3))
    for k in range(len(time)):
        relative = xyz[k] - com[k]
        rel_v = query_velocity[k] - vcom[k]
        inertia = (np.eye(3) * np.sum(w * np.sum(relative * relative, axis=1))
                   - np.einsum("n,ni,nj->ij", w, relative, relative))
        angular_momentum = np.sum(np.cross(relative, w[:, None] * rel_v), axis=0)
        omega = np.linalg.pinv(inertia, rcond=1e-10) @ angular_momentum
        omega_values[k] = omega
        whole_rigid_velocity[k] = vcom[k] + np.cross(
            np.broadcast_to(omega, relative.shape), relative
        )
        residual_velocity[k] = rel_v - np.cross(np.broadcast_to(omega, relative.shape), relative)
    node_shape_speed = np.linalg.norm(residual_velocity, axis=2)
    v_shape = float(np.einsum("t,n,tn->", tw, w, node_shape_speed))
    fd_residual = np.empty_like(query_fd_velocity)
    for k in range(len(time)):
        fd_vcom = np.einsum("ni,n->i", query_fd_velocity[k], w)
        fd_rel_v = query_fd_velocity[k] - fd_vcom
        relative = xyz[k] - com[k]
        inertia = (np.eye(3) * np.sum(w * np.sum(relative * relative, axis=1))
                   - np.einsum("n,ni,nj->ij", w, relative, relative))
        fd_momentum = np.sum(np.cross(relative, w[:, None] * fd_rel_v), axis=0)
        fd_omega = np.linalg.pinv(inertia, rcond=1e-10) @ fd_momentum
        fd_residual[k] = fd_rel_v - np.cross(np.broadcast_to(fd_omega, relative.shape), relative)
    fd_shape_speed = float(np.einsum("t,n,tn->", tw, w, np.linalg.norm(fd_residual, axis=2)))
    whole_rigid_node_speed = np.linalg.norm(whole_rigid_velocity, axis=2)
    whole_object_speed = float(np.einsum("t,n,tn->", tw, w, whole_rigid_node_speed))
    return {
        "window_seconds": [start, end], "samples": int(len(time)),
        "shape_amplitude_rms_mm": float(1000 * amp_m),
        "shape_speed_m_s": v_shape,
        "shape_speed_from_position_difference_m_s": fd_shape_speed,
        "velocity_source": "simulator node velocities" if node_velocities is not None else "finite-difference diagnostic only",
        "whole_object_rigid_node_speed_m_s": whole_object_speed,
        "whole_object_com_speed_m_s": float(np.dot(tw, np.linalg.norm(vcom, axis=1))),
        "whole_object_com_velocity_m_s": np.einsum("t,ti->i", tw, vcom).tolist(),
        "whole_object_rotation_speed_rad_s": float(np.dot(tw, np.linalg.norm(omega_values, axis=1))),
        **align_meta,
    }
