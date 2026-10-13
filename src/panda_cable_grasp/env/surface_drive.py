"""Live velocity feedback on table-supported nodes, with no path catch-up."""
from __future__ import annotations

import numpy as np


def surface_accelerations(positions, velocities, mass, grounded, target_velocity,
                          target_yaw_rate, *, velocity_gain, max_acceleration,
                          yaw_velocity_gain, yaw_feedforward=0., reference=None,
                          shape_stiffness=0., shape_damping=0., max_shape_acceleration=45.):
    """Only supported geometry/velocities enter feedback or shape restoration.

    There is no absolute position/yaw reference error. Airborne nodes receive
    zero whole-motion acceleration and cannot influence table velocity feedback.
    Forces are never reallocated to preserve the whole cable's original wrench.
    """
    translation = np.zeros_like(positions)
    rotation = np.zeros_like(positions)
    hold = np.zeros_like(positions)
    if not np.any(grounded):
        return translation, rotation, hold
    m = mass[grounded]
    xy = positions[grounded, :2]
    velocity = velocities[grounded, :2]
    center = np.average(xy, axis=0, weights=m)
    relative = xy-center
    actual_velocity = np.average(velocity, axis=0, weights=m)
    a = velocity_gain*(target_velocity-actual_velocity)
    a *= min(1., max_acceleration/max(float(np.linalg.norm(a)), 1e-12))
    translation[grounded, :2] = a
    tangent = np.column_stack((-relative[:, 1], relative[:, 0]))
    inertia = float(np.sum(m*np.sum(relative**2, axis=1)))
    relative_velocity = velocity-actual_velocity
    actual_rate = float(np.sum(m*np.sum(tangent*relative_velocity, axis=1))/max(inertia, 1e-12))
    alpha = float(np.clip(yaw_feedforward+yaw_velocity_gain*(target_yaw_rate-actual_rate), -18., 18.))
    spin = alpha*tangent-actual_rate**2*relative
    spin -= np.average(spin, axis=0, weights=m)
    rotation[grounded, :2] = spin
    if reference is not None and len(m)>1:
        ref = reference[grounded]
        ref = ref-np.average(ref, axis=0, weights=m)
        left, _, right = np.linalg.svd((ref*np.sqrt(m[:, None])).T@(relative*np.sqrt(m[:, None])))
        orient = left@right
        if np.linalg.det(orient)<0:
            left[:, -1] *= -1
            orient = left@right
        target = ref@orient  # restore shape in its CURRENT orientation, never an old yaw
        target_velocity_relative = actual_rate*np.column_stack((-target[:, 1], target[:, 0]))
        correction = shape_stiffness*(target-relative)-shape_damping*(relative_velocity-target_velocity_relative)
        correction -= np.average(correction, axis=0, weights=m)
        if inertia>1e-12:
            torque = float(np.sum(m*np.sum(tangent*correction, axis=1)))
            correction -= torque/inertia*tangent
        peak = float(np.linalg.norm(correction, axis=1).max())
        correction *= min(1., max_shape_acceleration/max(peak, 1e-12))
        hold[grounded, :2] = correction
    return translation, rotation, hold
