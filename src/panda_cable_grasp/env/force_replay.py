"""Compile a no-intervention force tape, then play it without live feedback.

The v2 tracking servo runs ONLY in the isolated compiler. Its output is sampled
at physics rate in world coordinates for each material node. Actual contact,
pose, velocity and grasp state never feed back into the compiled sequence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib

import mujoco
import numpy as np

FORCE_COMPILER_PARAMETERS = dict(
    replay_position_gain=1600., replay_velocity_gain=80.,
    replay_yaw_position_gain=320., replay_yaw_velocity_gain=36.,
    rigid_translation_max_acceleration=20.,
)


@dataclass(frozen=True)
class ForceReplayTape:
    forces: np.ndarray  # (physics steps, material nodes, xyz), newtons, world frame
    timestep: float
    start_time: float
    sha256: str
    final_qpos: np.ndarray | None = None
    final_qvel: np.ndarray | None = None
    max_node_force_n: float = 0.

    def index(self, time: float) -> int:
        # Recover an integer physics tick despite floating point mj_step time.
        return int(np.floor((time-self.start_time)/self.timestep + 1e-7))

    def sample(self, time: float) -> np.ndarray:
        index = self.index(time)
        if index < 0 or index >= len(self.forces):
            raise ValueError("force replay requested outside its compiled episode")
        return self.forces[index].copy()


def prepare_force_replay(env):
    """Freeze the exact reset state; do not reset/re-randomize a replica.

    A private model/data excludes robot collision surfaces, keeping compilation
    independent of the ready arm, while retaining cable/table physics. All array
    diagnostics written by the teacher and gravity data are private too. No
    camera, grasp scoring, policy or wall-clock pause participates in compilation.
    """
    teacher = copy.copy(env)
    teacher.__dict__ = {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in env.__dict__.items()
    }
    teacher.model = copy.copy(env.model)
    teacher.data = mujoco.MjData(teacher.model)
    mujoco.mj_copyData(teacher.data, teacher.model, env.data)
    teacher.config = replace(
        env.config, replay_controller_version="continuous_ff_v2",
        rigid_motion_contact_only=False,
        **FORCE_COMPILER_PARAMETERS,
    )
    robot_geoms = list(env.geometric_safety_geom_ids)
    teacher.model.geom_contype[robot_geoms] = 0
    teacher.model.geom_conaffinity[robot_geoms] = 0
    if env._gravity_data is not None:
        teacher._gravity_data = mujoco.MjData(env._gravity_model)
    teacher._pad_contact_cache_active = False
    teacher._pad_contact_cache = None
    teacher._dynamicvla_camera_renderer = None
    teacher._dynamicvla_camera_frames = {}
    teacher.grasp_state = None
    teacher.rigid_motion_suspended = False
    teacher._replay_force_tape = None
    teacher._replay_elapsed = 0.
    teacher.data.ctrl[:] = teacher.ready_ctrl
    mujoco.mj_forward(teacher.model, teacher.data)
    return teacher


def compile_force_replay(teacher) -> ForceReplayTape:
    """Compile before first physics tick, from the frozen private teacher.

    Deferring this work avoids compiling the constructor's provisional reset
    when a caller immediately performs its seeded reset. Live state is never
    used here, even if the real cable was moved before the first action.
    """
    dt = float(teacher.model.opt.timestep)
    start = float(teacher.data.time)
    count = int(np.ceil((teacher.config.episode_seconds-start)/dt-1e-10))
    if count < 1:
        raise ValueError("force replay requires a positive remaining episode")
    forces = np.empty((count, len(teacher.cable_ids), 3))
    for index in range(count):
        teacher.data.xfrc_applied[:] = 0.
        teacher._apply_cable_disturbance()
        # In the compiler ONLY, preserve the intended rigid wrench using the
        # table-supported nodes. Dropping free-node forces without this
        # calibration loses translation/rotation even without a robot.
        force = supported_force_distribution(
            teacher.data.xpos[teacher.cable_ids],
            teacher._last_ungated_scene_force,
            teacher._cable_table_contact_mask(),
        )
        forces[index] = force
        teacher.data.xfrc_applied[teacher.cable_ids, :3] = force
        teacher.data.ctrl[:] = teacher.ready_ctrl
        teacher._physics_step()
    if not np.isfinite(forces).all():
        raise ValueError("compiled force replay contains nonfinite forces")
    digest = hashlib.sha256(forces.tobytes()).hexdigest()
    forces.setflags(write=False)
    qpos=teacher.data.qpos.copy(); qvel=teacher.data.qvel.copy()
    qpos.setflags(write=False); qvel.setflags(write=False)
    maximum = float(np.linalg.norm(forces, axis=2).max())
    return ForceReplayTape(forces, dt, start, digest, qpos, qvel, maximum)


def supported_force_distribution(positions, forces, grounded):
    """Minimum-change grounded horizontal forces preserving the rigid wrench.

    Only the private no-intervention compiler may call this function. Playback
    never reallocates forces when the actual cable is moved or lifted.
    """
    result = np.zeros_like(forces)
    indices = np.flatnonzero(grounded)
    if not len(indices):
        return result
    relative = positions[:, :2] - positions[:, :2].mean(axis=0)
    # Preserve the planar resultant. Trying to exactly recover tiny pitch/roll
    # moments from nearly collinear supports makes the allocation ill-conditioned.
    matrix = np.zeros((3, 2*len(indices)))
    matrix[0, 0::2] = 1.
    matrix[1, 1::2] = 1.
    matrix[2, 0::2] = -relative[indices, 1]
    matrix[2, 1::2] = relative[indices, 0]
    wrench = np.r_[forces[:, :2].sum(axis=0), np.sum(
        relative[:, 0]*forces[:, 1]-relative[:, 1]*forces[:, 0],
    )]
    original = forces[indices, :2].ravel()
    correction = np.linalg.lstsq(matrix, wrench-matrix@original, rcond=1e-5)[0]
    result[indices, :2] = (original+correction).reshape(-1, 2)
    return result
