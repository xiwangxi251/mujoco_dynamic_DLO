"""Experimental command-only near-table guard, independent of policy/reward.

This is a servo velocity projection prototype, not a certified CBF. It never
edits simulation qpos/qvel, contacts, actuator gains or torque limits.
"""
from dataclasses import asdict, dataclass
import mujoco
import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class GuardConfig:
    margin_m: float = .001
    distance_gain: float = 5.
    braking_acceleration_m_s2: float = 2.
    response_delay_s: float = .02
    speed_cap_m_s: float = .9
    velocity_feedback_gain: float = 2.
    activation_distance_m: float = .15
    projection_sweeps: int = 12


class TableGeometry:
    def __init__(self, base):
        self.base = base
        m = base.model
        self.table = m.geom('table').id
        assert m.geom_type[self.table] == mujoco.mjtGeom.mjGEOM_BOX
        assert np.allclose(base.data.geom_xmat[self.table].reshape(3, 3), np.eye(3))
        self.top = float(base.data.geom_xpos[self.table, 2] + m.geom_size[self.table, 2])
        self.xy = base.data.geom_xpos[self.table, :2].copy()
        self.half = m.geom_size[self.table, :2].copy()
        bodies = {base.hand_id, base.left_finger_id, base.right_finger_id}
        self.meshes = []
        seen = set()
        for g in range(m.ngeom):
            if m.geom_bodyid[g] not in bodies or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = int(m.geom_dataid[g])
            key = (int(m.geom_bodyid[g]), mesh, tuple(m.geom_pos[g]), tuple(m.geom_quat[g]))
            if key in seen:
                continue
            seen.add(key)
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            vertices = np.asarray(m.mesh_vert[start:start+count], dtype=float)
            vertices = vertices[ConvexHull(vertices).vertices]
            self.meshes.append((g, vertices))
        assert len(self.meshes) == 5, len(self.meshes)
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros_like(self.jp)

    def points(self):
        b = self.base
        rows = []
        for g, vertices in self.meshes:
            world = vertices @ b.data.geom_xmat[g].reshape(3, 3).T + b.data.geom_xpos[g]
            valid = np.all(np.abs(world[:, :2]-self.xy) <= self.half, axis=1)
            if not np.any(valid):
                continue
            indices = np.flatnonzero(valid)
            point = world[indices[np.argmin(world[indices, 2])]]
            mujoco.mj_jac(b.model, b.data, self.jp, self.jr, point, int(b.model.geom_bodyid[g]))
            rows.append((g, float(point[2]-self.top), self.jp[2, b.arm_dof_adr].copy(),
                         float(self.jp[2] @ b.data.qvel)))
        return rows


class TableExecutionGuard:
    def __init__(self, base, geometry=None, config=GuardConfig()):
        self.base, self.geometry, self.config = base, geometry or TableGeometry(base), config
        ids = base.arm_actuator_ids
        self.kp = base.model.actuator_gainprm[ids, 0].copy()
        self.kv = -base.model.actuator_biasprm[ids, 2].copy()
        assert np.all(self.kp > 0) and np.all(self.kv > 0)
        self.steps = self.interventions = self.feedback_steps = self.residual_steps = 0
        self.last_active = False
        self.last_min_clearance = np.inf

    def project(self, requested, points=None):
        b, c = self.base, self.config
        points = self.geometry.points() if points is None else points
        self.steps += 1
        self.last_min_clearance = min((r[1] for r in points), default=np.inf)
        q = b.data.qpos[b.arm_qpos_adr]
        # Servo's velocity equilibrium Kp*(target-q) = Kv*qdot. This allows
        # filtering the final target after the original command limiter.
        reference = self.kp/self.kv*(requested[:7]-q)
        normals, bounds = [], []
        feedback = False
        for geom, distance, normal, actual_v in points:
            if distance > c.activation_distance_m:
                continue
            clearance = max(0., distance-c.margin_m)
            a, tau = c.braking_acceleration_m_s2, c.response_delay_s
            cap = min(c.speed_cap_m_s, c.distance_gain*clearance,
                      np.sqrt((a*tau)**2+2*a*clearance)-a*tau)
            lower = -cap
            if actual_v < -cap:
                lower += c.velocity_feedback_gain*(-cap-actual_v)
                feedback = True
            normals.append(normal)
            bounds.append(lower)
        if not normals:
            self.last_active = False
            return requested.copy()
        A, lower = np.asarray(normals), np.asarray(bounds)
        # Dykstra projection onto point-normal halfspaces. Safety residuals
        # are counted and reported; no silent assertion of feasibility.
        velocity = reference.copy()
        correction = np.zeros_like(A)
        norm2 = np.sum(A*A, axis=1)+1e-12
        for _ in range(c.projection_sweeps):
            for i in range(len(A)):
                candidate = velocity+correction[i]
                projected = candidate + max(0., lower[i]-A[i]@candidate)/norm2[i]*A[i]
                correction[i] = candidate-projected
                velocity = projected
            if np.min(A@velocity-lower) >= -1e-7:
                break
        target = requested.copy()
        ranges = b.model.jnt_range[b.arm_joint_ids]
        target[:7] = np.clip(q + self.kv/self.kp*velocity, ranges[:, 0], ranges[:, 1])
        actual_projection = self.kp/self.kv*(target[:7]-q)
        self.residual_steps += int(np.min(A@actual_projection-lower) < -1e-5)
        self.last_active = bool(np.max(np.abs(target-requested)) > 1e-9)
        self.interventions += int(self.last_active)
        self.feedback_steps += int(feedback and self.last_active)
        return target

    def summary(self):
        return dict(config=asdict(self.config), guard_steps=self.steps,
                    intervention_steps=self.interventions,
                    intervention_ratio=self.interventions/max(1, self.steps),
                    velocity_feedback_steps=self.feedback_steps,
                    projection_residual_steps=self.residual_steps,
                    monitored_mesh_count=len(self.geometry.meshes),
                    coverage='link7, flange, palm and both fingers; lower-arm links not protected')
