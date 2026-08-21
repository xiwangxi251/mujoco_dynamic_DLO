"""Pure MuJoCo kinematics helpers shared by scripted and RL controllers."""

from __future__ import annotations

import mujoco
import numpy as np


def rotation_to_quat(rotation: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
    return quat


def quat_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Return the shortest small-angle orientation error in world axes."""
    inverse = np.zeros(4)
    mujoco.mju_negQuat(inverse, current)
    delta = np.zeros(4)
    mujoco.mju_mulQuat(delta, desired, inverse)
    if delta[0] < 0:
        delta *= -1
    return 2.0 * delta[1:4]


def point_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    local_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return translational and rotational Jacobians at a body-fixed point."""
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jac_pos, jac_rot, body_id)
    offset_world = data.xmat[body_id].reshape(3, 3) @ local_offset
    jac_pos += np.cross(jac_rot.T, offset_world).T
    return jac_pos, jac_rot


__all__ = ["point_jacobian", "quat_error", "rotation_to_quat"]

