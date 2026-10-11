"""Opt-in enhanced full-six-dimensional DLS; legacy decoders stay independent."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class Full6ControlProfile:
    name: str = "enhanced6_bounded_dls_v1"
    linear_gain: float = 6.0
    angular_gain: float = 10.0
    linear_velocity_limit: float = 0.90
    angular_velocity_limit: float = 1.40
    damping: float = 0.04
    ready_posture_gain: float = 0.8
    target_horizon: float = 0.11


ENHANCED6_BOUNDED_DLS_V1 = Full6ControlProfile()
CONTROLLER_PROFILES = ("legacy", ENHANCED6_BOUNDED_DLS_V1.name)


def configure_profile(config, profile):
    if profile not in CONTROLLER_PROFILES:
        raise ValueError(f"unknown controller profile: {profile}")
    if profile != "legacy":
        if config.robot != "nero":
            raise ValueError("enhanced6_bounded_dls_v1 requires NERO")
        config.frame_skip = 10
        # The serialized arm switch also governs profile reconfiguration.
        config.arm_acceleration_limit_enabled = False
        config.hand_cartesian_velocity_limit_enabled = config.arm_velocity_limits_enabled
        config.hand_linear_velocity_limit = 0.90
        config.hand_angular_velocity_limit = 2.0
    config.controller_profile = profile
    return config


def limit_norm(value, maximum):
    value = np.asarray(value, dtype=float)
    return value * min(1.0, maximum / max(float(np.linalg.norm(value)), 1e-12))


def absolute_pose_twist(position_error, orientation_error, profile=ENHANCED6_BOUNDED_DLS_V1):
    return (limit_norm(profile.linear_gain * np.asarray(position_error), profile.linear_velocity_limit),
            limit_norm(profile.angular_gain * np.asarray(orientation_error), profile.angular_velocity_limit))


def delta_pose_twist(translation_delta, policy_dt, orientation_error, profile=ENHANCED6_BOUNDED_DLS_V1,
                     *, velocity_limits_enabled=True):
    if policy_dt <= 0:
        raise ValueError("policy_dt must be positive")
    linear = np.asarray(translation_delta, dtype=float) / policy_dt
    angular = profile.angular_gain * np.asarray(orientation_error, dtype=float)
    if velocity_limits_enabled:
        linear = limit_norm(linear, profile.linear_velocity_limit)
        angular = limit_norm(angular, profile.angular_velocity_limit)
    return linear, angular


def solve_bounded_full6(jacobian, linear_velocity, angular_velocity, q_current,
                       ready_q, joint_velocity_limits, joint_lower, joint_upper,
                       profile=ENHANCED6_BOUNDED_DLS_V1, *, velocity_limits_enabled=True):
    """Minimize ||Jv-twist||² + damping²||v-ready_bias||² under joint bounds.

    Both translation and rotation participate in the same six-dimensional task.
    Position bounds constrain q_actual + 0.11*v inside the solve, rather than
    dropping components after an unconstrained solution hits a joint limit.
    """
    j = np.asarray(jacobian, dtype=float)
    q = np.asarray(q_current, dtype=float)
    limits = np.asarray(joint_velocity_limits, dtype=float)
    lo, hi = np.asarray(joint_lower, dtype=float), np.asarray(joint_upper, dtype=float)
    twist = np.r_[linear_velocity, angular_velocity].astype(float)
    if velocity_limits_enabled:
        twist = np.r_[limit_norm(linear_velocity, profile.linear_velocity_limit),
                      limit_norm(angular_velocity, profile.angular_velocity_limit)]
    if j.shape != (6, 7) or any(a.shape != (7,) for a in (q, limits, lo, hi, np.asarray(ready_q))):
        raise ValueError("Expected a 6x7 Jacobian and seven joint values")
    if not all(np.all(np.isfinite(a)) for a in (j, q, limits, lo, hi, twist, ready_q)):
        raise ValueError("Full6 inputs must be finite")
    if np.any(limits <= 0) or np.any(lo >= hi):
        raise ValueError("Invalid joint bounds")
    lower, upper = (lo-q) / profile.target_horizon, (hi-q) / profile.target_horizon
    if velocity_limits_enabled:
        lower, upper = np.maximum(-limits, lower), np.minimum(limits, upper)
    if np.any(lower > upper):
        raise RuntimeError("Joint position/velocity bounds infeasible at actual state")
    bias = (np.eye(7)-np.linalg.pinv(j, rcond=1e-5) @ j) @ (
        profile.ready_posture_gain * (np.asarray(ready_q)-q))
    a = np.vstack((j, profile.damping*np.eye(7)))
    b = np.r_[twist, profile.damping*bias]
    # Eliminate equality bounds so the least-squares API also handles a joint
    # exactly one horizon of travel outside a soft MuJoCo joint range.
    fixed = lower == upper
    velocity = lower.copy()
    free = ~fixed
    if np.any(free):
        solution = lsq_linear(a[:, free], b-a[:, fixed] @ velocity[fixed],
                              bounds=(lower[free], upper[free]),
                              method="bvls", tol=1e-10, max_iter=100)
        if not solution.success or not np.all(np.isfinite(solution.x)):
            raise RuntimeError(f"Bounded Full6 DLS failed: {solution.message}")
        velocity[free] = solution.x
    target = np.clip(q + profile.target_horizon*velocity, lo, hi)
    return velocity, target, 1.0


def compose_bounded_pd_gravity(requested_pd, gravity_feedforward, preexisting_applied, force_range):
    low, high = np.asarray(force_range).T
    pd = np.clip(requested_pd, low, high)
    external = np.asarray(preexisting_applied)
    total = np.clip(pd + external + gravity_feedforward, low, high)
    return pd, total, total-pd-external


def add_controller_arguments(parser):
    import argparse
    parser.add_argument("--controller-profile", choices=CONTROLLER_PROFILES, default="legacy")
    parser.add_argument("--table-execution-guard", action=argparse.BooleanOptionalAction, default=False,
                        help="enable 500 Hz near-table command projection (NERO enhanced6)")
