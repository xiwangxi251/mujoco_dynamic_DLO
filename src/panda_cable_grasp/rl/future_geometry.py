"""Fixed first-three-second natural-motion oracle for ordinary free-action PPO.

The branch is conditional on absence of robot contact, never the realized
post-contact future. It does not consume live physics time or RNG.
"""
import copy
import time

import gymnasium as gym
import mujoco
import numpy as np

VERSION = 'initial_future3s_geometry_v1'
FRAME_TIMES = np.arange(1, 31, dtype=float) / 10
PROTOCOL = dict(version=VERSION, frame_times_s=FRAME_TIMES.tolist(), samples=14,
    per_frame_fields='14 material point xyz, relative time / 3, valid mask',
    observation_dim=1419, horizon='fixed episode [0,3]; not rolling t+3',
    coordinates='current TCP frame; original cable position scale',
    contact_stop='first robot-cable physics substep; latched until reset',
    future='no-robot-contact natural motion; includes hidden drive phase',
    actor_critic='same complete observation', actions='unchanged ordinary 5-D')


def robot_bodies(env):
    root = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.robot_spec.base_body_name)
    if root < 0:
        raise ValueError('Robot base body missing')
    ids = {root}
    for body in range(root + 1, env.model.nbody):
        if int(env.model.body_parentid[body]) in ids:
            ids.add(body)
    return ids


def robot_cable_contact(env):
    robot = robot_bodies(env)
    for contact in env.data.contact:
        a, b = (int(env.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        if (a in env.cable_set and b in robot) or (b in env.cable_set and a in robot):
            return True
    return False


def material_samples(initial_positions, count=14):
    lengths = np.linalg.norm(np.diff(initial_positions, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    if cumulative[-1] <= 1e-9:
        locations = np.linspace(0, len(initial_positions)-1, count)
        segments = np.minimum(np.floor(locations).astype(int), len(initial_positions)-2)
        return segments, locations-segments
    queries = np.linspace(0, cumulative[-1], count)
    segments = np.clip(np.searchsorted(cumulative, queries, side='right')-1, 0, len(lengths)-1)
    alpha = (queries-cumulative[segments]) / np.maximum(lengths[segments], 1e-12)
    return segments, np.clip(alpha, 0, 1)


def natural_branch(env):
    """Copy every drive mutable and full physical model, disable robot contact only."""
    branch = copy.copy(env)
    branch.__dict__ = env.__dict__.copy()
    for name, value in env.__dict__.items():
        if isinstance(value, (np.ndarray, np.random.Generator, dict, list, set)):
            branch.__dict__[name] = copy.deepcopy(value)
    branch.config = copy.deepcopy(env.config)
    branch.model = copy.deepcopy(env.model)
    branch.data = mujoco.MjData(branch.model)
    state_type = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, state_type))
    mujoco.mj_getState(env.model, env.data, state, state_type)
    mujoco.mj_setState(branch.model, branch.data, state, state_type)
    robot = robot_bodies(env)
    geoms = np.array([int(body) in robot for body in env.model.geom_bodyid])
    branch.model.geom_contype[geoms] = 0
    branch.model.geom_conaffinity[geoms] = 0
    # Explicit contact pairs bypass contype/conaffinity filtering.
    for pair in range(branch.model.npair):
        if geoms[branch.model.pair_geom1[pair]] or geoms[branch.model.pair_geom2[pair]]:
            branch.model.pair_margin[pair] = -1e6
            branch.model.pair_gap[pair] = 0
    # Preserve active cable-only replay welds and all original cable/table dynamics.
    # Only existing robot-related equality constraints are deactivated.
    for eq in range(branch.model.neq):
        if branch.model.eq_type[eq] in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD):
            if int(branch.model.eq_obj1id[eq]) in robot or int(branch.model.eq_obj2id[eq]) in robot:
                branch.data.eq_active[eq] = False
    mujoco.mj_forward(branch.model, branch.data)
    return branch


def precompute(env, node_slice):
    started = time.perf_counter()
    if abs(float(env.data.time)) > 1e-9:
        raise ValueError('Initial fixed-horizon oracle requires episode physics time zero')
    branch = natural_branch(env)
    initial = branch.data.xpos[branch.cable_ids].copy()[node_slice]
    segments, alpha = material_samples(initial)
    qadr = np.r_[branch.arm_qpos_adr, branch.finger_qpos_adr]
    dadr = np.r_[branch.arm_dof_adr, branch.finger_dof_adr]
    held = branch.data.qpos[qadr].copy()
    step_time = float(branch.model.opt.timestep)
    ticks = np.rint(FRAME_TIMES / step_time).astype(int)
    if not np.allclose(ticks*step_time, FRAME_TIMES, atol=1e-10):
        raise ValueError('Oracle frame times must align with physical substeps')
    frames = []
    next_frame = 0
    for tick in range(1, ticks[-1]+1):
        branch.data.qpos[qadr] = held
        branch.data.qvel[dadr] = 0
        branch.data.xfrc_applied[:] = 0
        branch._apply_cable_disturbance()
        mujoco.mj_step(branch.model, branch.data)
        if tick == ticks[next_frame]:
            branch.data.qpos[qadr] = held
            branch.data.qvel[dadr] = 0
            mujoco.mj_forward(branch.model, branch.data)
            xyz = branch.data.xpos[branch.cable_ids].copy()[node_slice]
            frames.append(xyz[segments]*(1-alpha[:, None]) + xyz[segments+1]*alpha[:, None])
            next_frame += 1
            if next_frame == len(ticks):
                break
    frames = np.asarray(frames)
    if not np.isfinite(frames).all():
        raise FloatingPointError('Nonfinite future trajectory')
    return frames, time.perf_counter()-started, (segments, alpha)


def encode(frames, current_time, position, rotation, scale, enabled):
    result = np.zeros((30, 44), dtype=np.float32)
    valid = (FRAME_TIMES > current_time+1e-9) & enabled
    if np.any(valid):
        result[valid, :42] = (((frames[valid]-position) @ rotation)/scale).reshape(-1, 42)
        result[valid, 42] = (FRAME_TIMES[valid]-current_time)/3
        result[valid, 43] = 1
    return np.clip(result.reshape(-1), -10, 10)


class FutureGeometryObservation(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if env.observation_space.shape != (99,) or env.rl_config.cable_sample_count != 14:
            raise ValueError('Future3s requires ordinary state99 and 14 cable samples')
        self.observation_space = gym.spaces.Box(-10, 10, shape=(1419,), dtype=np.float32)
        self.enabled = False
        self.frames = np.zeros((30, 14, 3))
        self.first_contact_time = None

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.env, name)

    def _observation(self, current):
        env = self.env
        extra = encode(self.frames, float(env.data.time), env._grasp_center_position(),
            env.data.xmat[env.base_env.hand_id].reshape(3, 3),
            env.rl_config.cable_position_scale, self.enabled)
        return np.concatenate((current, extra)).astype(np.float32)

    def _info(self, info):
        return dict(info, future_geometry_version=VERSION,
            future_geometry_enabled=self.enabled,
            future_valid_frames=int(np.count_nonzero(FRAME_TIMES > float(self.env.data.time)+1e-9)) if self.enabled else 0,
            future_first_contact_time=self.first_contact_time,
            future_precompute_seconds=self.precompute_seconds)

    def reset(self, **kwargs):
        current, info = self.env.reset(**kwargs)
        self.enabled = True
        self.first_contact_time = None
        self.frames, self.precompute_seconds, self.material_mapping = precompute(self.env.base_env, self.env._target_object_slice())
        base = self.env.base_env
        if getattr(base, '_future_contact_owner', None) is not self:
            original = base._update_physical_grasp_state
            def hook(closed):
                if self.enabled and robot_cable_contact(base):
                    self.enabled = False
                    self.first_contact_time = float(base.data.time)
                return original(closed)
            base._update_physical_grasp_state = hook
            base._future_contact_owner = self
        if robot_cable_contact(base):
            self.enabled = False
            self.first_contact_time = float(base.data.time)
        return self._observation(current), self._info(info)

    def step(self, action):
        current, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(current), reward, terminated, truncated, self._info(info)
