"""DynamicVLA task-space action adapter for the MuJoCo Panda environment.

DynamicVLA's DOM checkpoint consumes two RGB streams plus a 6-D absolute
end-effector state and emits ``xyz + quaternion(wxyz) + gripper``.  The local
environment intentionally keeps its native actuator interface (seven joint
position targets plus one gripper command), so this module is the only place
where the model's task-space command is converted to robot controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.environment import CableGraspEnv
from ..env.kinematics import point_jacobian, quat_error, rotation_to_quat


@dataclass(frozen=True)
class DynamicVLAAdapterConfig:
    """Safety and IK parameters independent of DynamicVLA's learned policy."""

    workspace_x: tuple[float, float] = (0.20, 0.85)
    workspace_y: tuple[float, float] = (-0.55, 0.55)
    workspace_z: tuple[float, float] = (0.005, 0.70)
    linear_gain: float = 6.0
    angular_gain: float = 2.5
    linear_velocity_limit: float = 0.90
    angular_velocity_limit: float = 1.40
    position_damping: float = 0.04
    orientation_damping_min: float = 0.04
    orientation_damping_max: float = 0.16
    ik_target_horizon: float = 0.11
    nullspace_gain: float = 0.8
    gripper_threshold: float = 0.0
    gripper_open_ctrl: float = 255.0
    gripper_close_ctrl: float = 0.0


class DynamicVLATaskSpaceAdapter:
    """Convert absolute DynamicVLA pose commands into native environment actions.

    The adapter does not modify physics or bypass the environment's unified
    joint, Cartesian, acceleration, or gripper speed limits.  Workspace
    clipping is explicit and logged so out-of-distribution model outputs do not
    command unreachable poses or drive the arm below the table.
    """

    def __init__(
        self,
        env: CableGraspEnv,
        config: DynamicVLAAdapterConfig | None = None,
    ) -> None:
        self.env = env
        self.config = config or DynamicVLAAdapterConfig()
        self.has_model_action = False
        self.last_raw_action = np.full(8, np.nan, dtype=float)
        self.last_pose_command = np.zeros(7, dtype=float)
        self.last_joint_action = env.ready_ctrl.copy()
        self.last_position_clipped = False
        self.last_quaternion_repaired = False
        self.reset()

    def reset(self) -> None:
        rotation = self.env.data.xmat[self.env.hand_id].reshape(3, 3)
        self.has_model_action = False
        self.last_raw_action[:] = np.nan
        self.last_pose_command[:3] = self.env.hand_position
        self.last_pose_command[3:] = rotation_to_quat(rotation)
        self.last_joint_action[:] = self.env.ready_ctrl
        self.last_position_clipped = False
        self.last_quaternion_repaired = False

    @staticmethod
    def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector if norm <= limit else vector * (limit / norm)

    def set_model_action(self, action: np.ndarray) -> None:
        """Accept one official server-format action ``[xyz, quat_wxyz, grip]``."""

        value = np.asarray(action, dtype=float)
        if value.shape == (1, 8):
            value = value[0]
        if value.shape != (8,):
            raise ValueError(f"Expected DynamicVLA action shape (8,), got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("DynamicVLA action contains non-finite values")

        self.last_raw_action[:] = value
        lower = np.array([
            self.config.workspace_x[0],
            self.config.workspace_y[0],
            self.config.workspace_z[0],
        ])
        upper = np.array([
            self.config.workspace_x[1],
            self.config.workspace_y[1],
            self.config.workspace_z[1],
        ])
        position = np.clip(value[:3], lower, upper)
        self.last_position_clipped = not np.allclose(
            position, value[:3], rtol=0.0, atol=1e-12
        )

        quaternion = value[3:7].copy()
        quaternion_norm = float(np.linalg.norm(quaternion))
        self.last_quaternion_repaired = quaternion_norm < 1e-8
        if self.last_quaternion_repaired:
            quaternion = self.last_pose_command[3:].copy()
        else:
            quaternion /= quaternion_norm
        # q and -q represent the same orientation.  Keeping the nearest sign
        # makes command logs continuous without changing the desired pose.
        if float(np.dot(quaternion, self.last_pose_command[3:])) < 0.0:
            quaternion *= -1.0

        self.last_pose_command[:3] = position
        self.last_pose_command[3:] = quaternion
        self._gripper_ctrl = (
            self.config.gripper_open_ctrl
            if value[-1] > self.config.gripper_threshold
            else self.config.gripper_close_ctrl
        )
        self.has_model_action = True

    def action(self) -> np.ndarray:
        """Return one 8-D actuator target; hold ready pose before first output."""

        if not self.has_model_action:
            self.last_joint_action[:] = self.env.ready_ctrl
            return self.last_joint_action.copy()

        desired_position = self.last_pose_command[:3]
        desired_quaternion = self.last_pose_command[3:]
        current_rotation = self.env.data.xmat[self.env.hand_id].reshape(3, 3)
        current_quaternion = rotation_to_quat(current_rotation)
        position_error = desired_position - self.env.hand_position
        orientation_error = quat_error(current_quaternion, desired_quaternion)
        linear_velocity = self._limit_norm(
            self.config.linear_gain * position_error,
            self.config.linear_velocity_limit,
        )
        angular_velocity = self._limit_norm(
            self.config.angular_gain * orientation_error,
            self.config.angular_velocity_limit,
        )

        jac_pos, jac_rot = point_jacobian(
            self.env.model,
            self.env.data,
            self.env.hand_id,
            self.env.GRASP_CENTER_LOCAL,
        )
        jacobian = np.vstack([jac_pos, jac_rot])[:, self.env.arm_dof_adr]

        # Position is primary. Orientation is solved only in its nullspace; this
        # is the same posture-safe hierarchy used by the scripted baseline and
        # prevents a pose command from pushing the fingers sideways into table.
        position_jacobian = jacobian[:3]
        position_inverse = np.linalg.solve(
            position_jacobian @ position_jacobian.T
            + self.config.position_damping**2 * np.eye(3),
            np.eye(3),
        )
        position_pseudoinverse = position_jacobian.T @ position_inverse
        position_velocity = position_pseudoinverse @ linear_velocity
        position_nullspace = (
            np.eye(7)
            - np.linalg.pinv(position_jacobian, rcond=1e-5) @ position_jacobian
        )

        orientation_jacobian = jacobian[3:] @ position_nullspace
        sigma_min = float(np.linalg.svd(orientation_jacobian, compute_uv=False)[-1])
        singularity = float(np.clip((0.10 - sigma_min) / 0.10, 0.0, 1.0))
        damping = (
            self.config.orientation_damping_min
            + (self.config.orientation_damping_max - self.config.orientation_damping_min)
            * singularity**2
        )
        orientation_inverse = np.linalg.solve(
            orientation_jacobian @ orientation_jacobian.T
            + damping**2 * np.eye(3),
            np.eye(3),
        )
        orientation_velocity = (
            orientation_jacobian.T
            @ orientation_inverse
            @ (angular_velocity - jacobian[3:] @ position_velocity)
        )
        q_velocity = position_velocity + orientation_velocity

        q_current = self.env.data.qpos[self.env.arm_qpos_adr]
        task_nullspace = (
            np.eye(7) - np.linalg.pinv(jacobian, rcond=1e-5) @ jacobian
        )
        q_velocity += task_nullspace @ (
            self.config.nullspace_gain * (self.env.ready_qpos[:7] - q_current)
        )
        velocity_limits = np.asarray(
            self.env.config.arm_joint_velocity_limits, dtype=float
        )
        velocity_scale = min(
            1.0,
            float(np.min(
                velocity_limits / np.maximum(np.abs(q_velocity), 1e-12)
            )),
        )
        q_velocity *= velocity_scale
        q_target = q_current + self.config.ik_target_horizon * q_velocity
        q_target = np.clip(
            q_target,
            self.env.model.jnt_range[self.env.arm_joint_ids, 0],
            self.env.model.jnt_range[self.env.arm_joint_ids, 1],
        )

        self.last_joint_action[:7] = q_target
        self.last_joint_action[7] = self._gripper_ctrl
        return self.last_joint_action.copy()

    def diagnostics(self) -> dict:
        return {
            "has_model_action": self.has_model_action,
            "raw_action": self.last_raw_action.copy(),
            "pose_command": self.last_pose_command.copy(),
            "joint_action": self.last_joint_action.copy(),
            "position_clipped": self.last_position_clipped,
            "quaternion_repaired": self.last_quaternion_repaired,
        }


def make_dynamicvla_observation(
    env: CableGraspEnv,
    instruction: str | None,
    index: int,
    dt_scale: float = 1.0,
) -> dict:
    """Build one observation using the unmodified official client schema."""

    images = env.dynamicvla_camera_rgb()
    rotation = env.data.xmat[env.hand_id].reshape(3, 3)
    observation = {
        "dt_scale": float(max(1.0, dt_scale)),
        "index": int(index),
        "observation.state": {
            "end_effector": {
                "pos": env.hand_position[None, :].astype(np.float32),
                "quat": rotation_to_quat(rotation)[None, :].astype(np.float32),
            },
        },
        "observation.images.opst_cam": images["opst_cam"][None, ...],
        "observation.images.wrist_cam": images["wrist_cam"][None, ...],
    }
    if instruction is not None:
        observation["task"] = instruction
    return observation
