"""Full 6D JIK execution and workspace-conditioned material-node sampling.

Loaded with the previous target-node PPO's frozen package. Reward methods,
contacts, cable physics and task success filters are inherited verbatim.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from panda_cable_grasp.env.environment import CableGraspEnv, XML_PATH, _robot_assets
from panda_cable_grasp.env.kinematics import quat_error, rotation_to_quat
from panda_cable_grasp.rl import environment as rl
from panda_cable_grasp.rl.target_node import TargetNodeTask

PROFILE = dict(name="full6_dls_gravity_servo_gain10_v1", damping=0.04,
               angular_gain=10.0, angular_cap=1.4, linear_cap=0.9,
               target_horizon=0.11, ready_gain=0.8,
               arm_kp_scale=4.0, arm_kv_scale=2.0,
               gravity_feedforward_hz=500, arm_acceleration_limit_enabled=False)
_GRAVITY_MODELS = {}


def robot_model(spec):
    root = ET.fromstring(XML_PATH.read_text(encoding="utf-8"))
    for tag in ("extension", "contact"):
        for item in root.findall(tag):
            root.remove(item)
    world = root.find("worldbody")
    for tag in ("composite", "site"):
        for item in world.findall(tag):
            world.remove(item)
    mj_spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"),
        include={"robot.xml": spec.xml_path.read_bytes()},
        assets=_robot_assets(str(spec.asset_dir)))
    next(body for body in mj_spec.bodies if body.name == spec.base_body_name).pos[:] += spec.base_offset
    return mj_spec.compile()


class EnhancedCableEnv(CableGraspEnv):
    def __init__(self, config):
        super().__init__(config)
        self.arm_actuator_ids = np.array([self.model.actuator(n).id for n in self.robot_spec.arm_joint_names])
        ids = self.arm_actuator_ids
        self.original_kp = self.model.actuator_gainprm[ids, 0].copy()
        self.original_kv = -self.model.actuator_biasprm[ids, 2].copy()
        self.original_torque_bounds = self.model.actuator_forcerange[ids].copy()
        self.model.actuator_gainprm[ids, 0] *= PROFILE["arm_kp_scale"]
        self.model.actuator_biasprm[ids, 1] *= PROFILE["arm_kp_scale"]
        self.model.actuator_biasprm[ids, 2] *= PROFILE["arm_kv_scale"]
        key = str(self.robot_spec.xml_path)
        if key not in _GRAVITY_MODELS:
            _GRAVITY_MODELS[key] = robot_model(self.robot_spec)
        self.gravity_model = _GRAVITY_MODELS[key]
        self.gravity_data = mujoco.MjData(self.gravity_model)
        names = (*self.robot_spec.arm_joint_names, *self.robot_spec.finger_joint_names)
        self.gravity_qadr = np.array([self.gravity_model.jnt_qposadr[self.gravity_model.joint(n).id] for n in names])
        self.scene_qadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in names])
        self.gravity_dadr = np.array([self.gravity_model.jnt_dofadr[self.gravity_model.joint(n).id] for n in self.robot_spec.arm_joint_names])
        for name in (self.robot_spec.base_body_name, *(f"link{i}" for i in range(1, 8)), self.robot_spec.left_finger_body_name, self.robot_spec.right_finger_body_name):
            a, b = self.model.body(name).id, self.gravity_model.body(name).id
            for attr in ("body_mass", "body_inertia", "body_pos", "body_quat"):
                assert np.allclose(getattr(self.model, attr)[a], getattr(self.gravity_model, attr)[b])
        self.ff_peak = np.zeros(7)
        self.total_torque_peak = np.zeros(7)
        self.ff_saturation_steps = 0
        self.ff_steps = 0

    def reset(self, **kwargs):
        # Keep the original motion law and sample its physically evolved
        # in-workspace entry state, rather than teaching PPO an entry wait.
        self.config.episode_seconds = 15.
        observation, info = super().reset(**kwargs)
        self.workspace_prefix_seconds = 0.
        if hasattr(self, "workspace_sampling_mask"):
            steps = 0
            while not np.any(self.workspace_sampling_mask.contains(self.data.xpos[self.cable_ids])) and steps < 400:
                observation, _, done, truncated, info = super().step(self.ready_ctrl.copy())
                steps += 1
                if done or truncated:
                    break
            self.workspace_prefix_seconds = float(self.data.time)
            if self.grasp_confirmed:
                raise NoWorkspaceTarget("Workspace reset prefix unexpectedly grasped the cable")
            self.config.episode_seconds = self.workspace_prefix_seconds + 15.
            info = self.info()
        info["workspace_prefix_seconds"] = self.workspace_prefix_seconds
        return observation, info

    def _physics_step(self):
        self.gravity_data.qpos[self.gravity_qadr] = self.data.qpos[self.scene_qadr]
        self.gravity_data.qvel[:] = 0
        mujoco.mj_forward(self.gravity_model, self.gravity_data)
        ff = self.gravity_data.qfrc_bias[self.gravity_dadr]
        ids = self.arm_actuator_ids
        pd = (self.model.actuator_gainprm[ids, 0] * self.data.ctrl[ids]
              + self.model.actuator_biasprm[ids, 0]
              + self.model.actuator_biasprm[ids, 1] * self.data.qpos[self.arm_qpos_adr]
              + self.model.actuator_biasprm[ids, 2] * self.data.qvel[self.arm_dof_adr])
        bounds = self.model.actuator_forcerange[ids]
        pd = np.clip(pd, bounds[:, 0], bounds[:, 1])
        external = self.data.qfrc_applied[self.arm_dof_adr].copy()
        total = np.clip(pd + external + ff, bounds[:, 0], bounds[:, 1])
        self.ff_saturation_steps += int(np.any(np.abs(total - pd - external - ff) > 1e-9))
        applied = total - pd - external
        self.data.qfrc_applied[self.arm_dof_adr] = external + applied
        self.ff_peak = np.maximum(self.ff_peak, np.abs(applied))
        self.total_torque_peak = np.maximum(self.total_torque_peak, np.abs(total))
        self.ff_steps += 1
        try:
            return super()._physics_step()
        finally:
            self.data.qfrc_applied[self.arm_dof_adr] = external


def limit_norm(value, maximum):
    value = np.asarray(value, dtype=float)
    return value * min(1.0, maximum / max(float(np.linalg.norm(value)), 1e-12))


def solve_full6(base, linear, angular):
    jp, jr = base._hand_jacobian()
    jacobian = np.vstack((jp, jr))[:, base.arm_dof_adr]
    twist = np.r_[limit_norm(linear, PROFILE["linear_cap"]),
                  limit_norm(angular, PROFILE["angular_cap"])]
    qdot = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + PROFILE["damping"] ** 2 * np.eye(6), twist)
    q = base.data.qpos[base.arm_qpos_adr]
    nullspace = np.eye(7) - np.linalg.pinv(jacobian, rcond=1e-5) @ jacobian
    qdot += nullspace @ (PROFILE["ready_gain"] * (base.ready_qpos[:7] - q))
    scale = min(1.0, float(np.min(base._arm_velocity_limits / np.maximum(np.abs(qdot), 1e-12))))
    qdot *= scale
    ranges = base.model.jnt_range[base.arm_joint_ids]
    target = np.clip(q + PROFILE["target_horizon"] * qdot, ranges[:, 0], ranges[:, 1])
    return target, qdot, scale, jacobian, twist


class Full6TargetEnv(rl.VerticalDownRLCableGraspEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_env.config.arm_acceleration_limit_enabled = False
        self.base_env.config.hand_cartesian_velocity_limit_enabled = True
        self.base_env.config.hand_linear_velocity_limit = PROFILE["linear_cap"]
        self.base_env.config.hand_angular_velocity_limit = 2.0
        assert np.isclose(self.model.opt.timestep * self.base_env.config.frame_skip, .02)

    def _convert_action(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        if action.shape != (5,):
            raise ValueError("Expected five PPO actions")
        direction = limit_norm(action[:3], 1.0)
        delta = self.rl_config.translation_delta_scale * direction
        self._last_commanded_world_translation = delta.copy()
        self._desired_hand_rotation = self._rotation_about_z(self.rl_config.yaw_delta_scale * action[3]) @ self._desired_hand_rotation
        error = quat_error(rotation_to_quat(self.data.xmat[self.base_env.hand_id].reshape(3, 3)), rotation_to_quat(self._desired_hand_rotation))
        target, self.last_qdot, self._last_ik_velocity_scale, jacobian, self.last_twist = solve_full6(self.base_env, delta / .02, PROFILE["angular_gain"] * error)
        self._singularity_metrics(jacobian)  # Existing diagnostics/reward inputs.
        self._last_ik_singularity_damping = PROFILE["damping"]
        self._last_ik_singularity_target_clipped = False
        previous = self._gripper_closed
        if action[4] <= self.rl_config.gripper_close_threshold:
            self._gripper_closed = True
        elif action[4] >= self.rl_config.gripper_open_threshold:
            self._gripper_closed = False
        self._gripper_switch_event = self._gripper_closed != previous
        self._gripper_switch_count += int(self._gripper_switch_event)
        result = np.r_[target, self.rl_config.gripper_closed_ctrl if self._gripper_closed else self.base_env.gripper_open_ctrl]
        if self.table_finger_collision_filter_enabled and not self.base_env.config.geometric_safety_enabled:
            result[:7] = self._project_table_safe_target(result[:7])
        return result


class NoWorkspaceTarget(ValueError):
    pass


class WorkspaceMask:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path) as file:
            self.xs, self.ys, self.mask = file["xs"], file["ys"], file["interior"]
        self.step = float(self.xs[1] - self.xs[0])
        if not np.any(self.mask):
            raise ValueError("Measured workspace has no interior cells")

    def contains(self, positions):
        positions = np.atleast_2d(positions)
        ix = np.rint((positions[:, 0] - self.xs[0]) / self.step).astype(int)
        iy = np.rint((positions[:, 1] - self.ys[0]) / self.step).astype(int)
        valid = (ix >= 0) & (ix < len(self.xs)) & (iy >= 0) & (iy < len(self.ys))
        # The scan descends from 10 cm to the near-floor grasp level. Legacy
        # scenes initialize the cable at 4.5 cm before it settles on the table.
        valid &= (positions[:, 2] >= -.005) & (positions[:, 2] <= .10)
        result = np.zeros(len(positions), dtype=bool)
        result[valid] = self.mask[iy[valid], ix[valid]]
        return result


class WorkspaceTargetNode(TargetNodeTask):
    def __init__(self, wrapper, seed, workspace):
        super().__init__(wrapper, seed)
        self.workspace = workspace
        self.candidates = np.array([], dtype=int)

    def reset(self, seed, options):
        self.candidates = np.flatnonzero(self.workspace.contains(self.base.data.xpos[self.base.cable_ids]))
        if not len(self.candidates):
            positions = self.base.data.xpos[self.base.cable_ids]
            raise NoWorkspaceTarget(f"No eligible material node: scene={self.base.config.scenario_name}; min_xyz={positions.min(axis=0).tolist()}; max_xyz={positions.max(axis=0).tolist()}")
        fixed = (options or {}).get("goal_node_index", self.fixed_index)
        if fixed is not None and fixed not in self.candidates:
            raise NoWorkspaceTarget(f"Explicit goal {fixed} is outside workspace")
        if seed is not None:
            self.rng = np.random.default_rng(int(seed) ^ 0x607A19)
        goal = int(fixed) if fixed is not None else int(self.rng.choice(self.candidates))
        super().reset(seed, dict(options or {}, goal_node_index=goal))

    def info(self):
        info = super().info()
        info.update(workspace_candidate_count=len(self.candidates),
                    workspace_goal_current_inside=bool(self.workspace.contains(self.base.data.xpos[self.base.cable_ids[self.index]])[0]))
        return info


# Only this process's isolated snapshot uses the enhanced execution class.
rl.CableGraspEnv = EnhancedCableEnv
