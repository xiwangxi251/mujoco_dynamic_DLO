"""Read-only MuJoCo rollout oracle for privileged data collection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class ShadowTrajectory:
    """Cable node states sampled from a cloned MuJoCo state."""

    origin_time: float
    timestep: float
    positions: np.ndarray
    velocities: np.ndarray

    @property
    def duration(self) -> float:
        return self.timestep * (len(self.positions) - 1)

    def sample(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        elapsed = float(np.clip(elapsed, 0.0, self.duration))
        frame = elapsed / self.timestep
        lower = min(int(math.floor(frame)), len(self.positions) - 1)
        upper = min(lower + 1, len(self.positions) - 1)
        alpha = frame - lower
        positions = (
            (1.0 - alpha) * self.positions[lower]
            + alpha * self.positions[upper]
        )
        velocities = (
            (1.0 - alpha) * self.velocities[lower]
            + alpha * self.velocities[upper]
        )
        return positions.copy(), velocities.copy()


class ShadowCableRollout:
    """Predict cable motion without advancing or modifying the live episode."""

    def __init__(self, env) -> None:
        self.env = env
        self.data = mujoco.MjData(env.model)

    def rollout(self, duration: float) -> ShadowTrajectory:
        duration = max(0.0, float(duration))
        timestep = float(self.env.model.opt.timestep)
        steps = int(math.ceil(duration / timestep))
        live_data = self.env.data
        mujoco.mj_copyData(self.data, self.env.model, live_data)

        # The oracle predicts exogenous cable motion. Holding the arm at its
        # current configuration prevents a stale live command from creating a
        # fictitious future robot/cable interaction inside the cloned state.
        held_arm_qpos = self.data.qpos[:7].copy()
        held_gripper_qpos = self.data.qpos[7:9].copy()
        self.data.ctrl[:7] = held_arm_qpos

        positions = np.empty(
            (steps + 1, len(self.env.cable_ids), 3), dtype=float
        )
        velocities = np.empty_like(positions)
        positions[0] = self.data.xpos[self.env.cable_ids]
        velocities[0] = self._node_velocities()

        diagnostic_names = (
            "_last_shape_acceleration",
            "_last_rigid_translation_acceleration",
            "_last_rigid_rotation_acceleration",
            "_last_rigid_shape_hold_acceleration",
        )
        saved_diagnostics = {
            name: getattr(self.env, name).copy() for name in diagnostic_names
        }
        try:
            self.env.data = self.data
            for step in range(1, steps + 1):
                self.data.qpos[:7] = held_arm_qpos
                self.data.qpos[7:9] = held_gripper_qpos
                self.data.qvel[:9] = 0.0
                self.data.xfrc_applied[:] = 0.0
                self.env._apply_cable_disturbance()
                mujoco.mj_step(self.env.model, self.data)
                positions[step] = self.data.xpos[self.env.cable_ids]
                velocities[step] = self._node_velocities()
        finally:
            self.env.data = live_data
            for name, value in saved_diagnostics.items():
                getattr(self.env, name)[:] = value

        return ShadowTrajectory(
            origin_time=float(live_data.time),
            timestep=timestep,
            positions=positions,
            velocities=velocities,
        )

    def _node_velocities(self) -> np.ndarray:
        return np.asarray([
            self.env.body_linear_velocity(body_id)
            for body_id in self.env.cable_ids
        ])
