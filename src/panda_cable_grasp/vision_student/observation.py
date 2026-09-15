"""Shared observation construction for the 127-D visual student policy.

The module is deliberately independent of MuJoCo and TrackDLO.  TrackDLO
produces a 45-node camera/world curve and a visibility index set; the caller
converts the curve to TCP coordinates before calling ``build_student_observation``.
"""

from __future__ import annotations

import numpy as np


NUM_TRACK_NODES = 45
NUM_POLICY_POINTS = 14
POSITION_SCALE_M = 0.50
STUDENT_OBSERVATION_DIM = 127


def _as_points(points: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.shape != (NUM_TRACK_NODES, 3):
        raise ValueError(f"{name} must have shape (45, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def resample_polyline_arclength(
    points: np.ndarray,
    sample_count: int = NUM_POLICY_POINTS,
) -> np.ndarray:
    """Uniformly resample an ordered polyline by normalized arc length."""

    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}")
    if len(array) < 2 or sample_count < 2:
        raise ValueError("at least two input points and two output samples are required")
    if not np.isfinite(array).all():
        raise ValueError("points must contain only finite values")
    segment_lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(arc[-1])
    if total <= 1e-9:
        return np.repeat(array[:1], sample_count, axis=0)
    targets = np.linspace(0.0, total, sample_count)
    result = np.empty((sample_count, 3), dtype=np.float32)
    for axis in range(3):
        result[:, axis] = np.interp(targets, arc, array[:, axis])
    return result


def map_visibility_to_samples(
    visible_nodes: np.ndarray,
    *,
    source_count: int = NUM_TRACK_NODES,
    sample_count: int = NUM_POLICY_POINTS,
    support_radius_nodes: float = 1.0,
) -> np.ndarray:
    """Map TrackDLO visible node indices to a binary policy-point mask.

    A policy point is visible when a TrackDLO visible node lies within the
    requested material-coordinate support radius.  This preserves uncertainty
    explicitly instead of zeroing the corresponding predicted position.
    """

    if source_count < 2 or sample_count < 2:
        raise ValueError("source_count and sample_count must be at least two")
    indices = np.asarray(visible_nodes, dtype=np.int64).reshape(-1)
    indices = indices[(indices >= 0) & (indices < source_count)]
    if support_radius_nodes < 0.0:
        raise ValueError("support_radius_nodes must be non-negative")
    sample_positions = np.linspace(0.0, source_count - 1, sample_count)
    if not len(indices):
        return np.zeros(sample_count, dtype=np.float32)
    distances = np.min(
        np.abs(sample_positions[:, None] - indices[None, :]), axis=1
    )
    return (distances <= float(support_radius_nodes)).astype(np.float32)


def stabilize_polyline_direction(
    points: np.ndarray,
    previous_points: np.ndarray | None,
) -> np.ndarray:
    """Choose the endpoint ordering closest to the previous frame."""

    current = np.asarray(points, dtype=np.float32)
    if current.ndim != 2 or current.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {current.shape}")
    if previous_points is None:
        return current.copy()
    previous = np.asarray(previous_points, dtype=np.float32)
    if previous.shape != current.shape:
        raise ValueError("previous_points must have the same shape as points")
    direct = float(np.mean(np.sum((current - previous) ** 2, axis=1)))
    reversed_error = float(np.mean(np.sum((current[::-1] - previous) ** 2, axis=1)))
    return current[::-1].copy() if reversed_error < direct else current.copy()


def build_student_observation(
    current_points_14_tcp: np.ndarray,
    previous_points_14_tcp: np.ndarray,
    current_visibility_14: np.ndarray,
    previous_visibility_14: np.ndarray,
    arm_qpos: np.ndarray,
    arm_qvel: np.ndarray,
    gripper_aperture: float,
    *,
    position_scale_m: float = POSITION_SCALE_M,
) -> np.ndarray:
    """Build the fixed 127-D observation consumed by the student actor."""

    current = np.asarray(current_points_14_tcp, dtype=np.float32)
    previous = np.asarray(previous_points_14_tcp, dtype=np.float32)
    if current.shape != (NUM_POLICY_POINTS, 3):
        raise ValueError(f"current_points_14_tcp must have shape (14, 3), got {current.shape}")
    if previous.shape != current.shape:
        raise ValueError("previous_points_14_tcp must have shape (14, 3)")
    current_mask = np.asarray(current_visibility_14, dtype=np.float32).reshape(-1)
    previous_mask = np.asarray(previous_visibility_14, dtype=np.float32).reshape(-1)
    if current_mask.shape != (NUM_POLICY_POINTS,) or previous_mask.shape != (NUM_POLICY_POINTS,):
        raise ValueError("visibility masks must have shape (14,)")
    qpos = np.asarray(arm_qpos, dtype=np.float32).reshape(-1)
    qvel = np.asarray(arm_qvel, dtype=np.float32).reshape(-1)
    if qpos.shape != (7,) or qvel.shape != (7,):
        raise ValueError("arm_qpos and arm_qvel must have shape (7,)")
    if not np.isfinite(position_scale_m) or position_scale_m <= 0.0:
        raise ValueError("position_scale_m must be finite and positive")
    observation = np.concatenate(
        (
            (current / float(position_scale_m)).reshape(-1),
            (previous / float(position_scale_m)).reshape(-1),
            np.clip(current_mask, 0.0, 1.0),
            np.clip(previous_mask, 0.0, 1.0),
            qpos,
            qvel,
            np.asarray([gripper_aperture], dtype=np.float32),
        )
    ).astype(np.float32, copy=False)
    if observation.shape != (STUDENT_OBSERVATION_DIM,):
        raise AssertionError(f"unexpected student observation shape: {observation.shape}")
    if not np.isfinite(observation).all():
        raise ValueError("student observation must contain only finite values")
    # Match the teacher RL wrapper's final observation safety range.  This is
    # especially important for replayed states around contact impulses, where
    # a normalized joint velocity can briefly exceed the nominal range.
    return np.clip(observation, -10.0, 10.0).astype(np.float32, copy=False)
