"""Opt-in material-node task; no changes to simulated forces or grasp contacts."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

GOAL_OBSERVATION_NAMES = (
    'goal_material_u_signed', 'goal_tcp_x', 'goal_tcp_y', 'goal_tcp_z',
    'goal_tcp_vx', 'goal_tcp_vy', 'goal_tcp_vz',
    'goal_tcp_tangent_x', 'goal_tcp_tangent_y', 'goal_tcp_tangent_z',
)


class TargetNodeTask:
    def __init__(self, wrapper, seed: int):
        self.wrapper = wrapper
        self.base = wrapper.base_env
        if len(self.base._objects) != 1:
            raise ValueError("target-node task currently supports one cable")
        self.rng = np.random.default_rng(seed ^ 0x607A19)
        self.index = len(self.base.cable_ids) // 2
        self.any_hold = 0.0
        self.any_success = False
        self.previous_match = False
        self.previous_secured = False
        self.was_pinched = False
        self.wrong_count = 0
        self.wrong_penalty = 0.0
        self.fixed_index = None
        self.base.task_success_filter = self.qualify

    def reset(self, seed, options):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed) ^ 0x607A19)
        count = len(self.base.cable_ids)
        config = self.wrapper.rl_config
        first = math.ceil((count - 1) * config.target_u_min)
        last = math.floor((count - 1) * config.target_u_max)
        if first > last:
            raise ValueError("target u range contains no material nodes")
        fixed = (options or {}).get("goal_node_index", self.fixed_index)
        if fixed is not None:
            if isinstance(fixed, bool) or not isinstance(fixed, (int, np.integer)):
                raise ValueError("goal_node_index must be an integer")
            if not 0 <= fixed < count:
                raise ValueError("goal_node_index outside the cable")
            self.index = int(fixed)
        else:
            self.index = int(self.rng.integers(first, last + 1))
        self.any_hold = 0.0
        self.any_success = False
        self.previous_match = self.previous_secured = self.was_pinched = False
        self.wrong_count = 0
        self.wrong_penalty = 0.0

    def actual_index(self):
        state = self.base.grasp_state
        return None if state is None else self.base.cable_index[state.body_id]

    def matches(self):
        actual = self.actual_index()
        return (actual is not None and
                abs(actual - self.index) <= self.wrapper.rl_config.target_node_tolerance)

    def qualify(self, physical_qualification):
        # Called once at each physics substep. Any-node success is measured in
        # parallel, while the base timer receives the continuous goal predicate.
        dt = self.base.model.opt.timestep
        self.any_hold = self.any_hold + dt if physical_qualification else 0.0
        self.any_success |= self.any_hold + 1e-12 >= self.base.config.success_hold_seconds
        return bool(physical_qualification and self.matches())

    def tangent(self, index):
        ids = self.base.cable_ids
        lo, hi = max(0, index - 1), min(len(ids) - 1, index + 1)
        vector = self.base.data.xpos[ids[hi]] - self.base.data.xpos[ids[lo]]
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def observation(self):
        env = self.wrapper
        body = self.base.cable_ids[self.index]
        rotation = self.base.data.xmat[self.base.hand_id].reshape(3, 3)
        hand_velocity, _ = env._hand_velocity()
        config = env.rl_config
        return np.concatenate([
            [2.0 * self.index / (len(self.base.cable_ids) - 1) - 1.0],
            (self.base.data.xpos[body] - env._grasp_center_position()) @ rotation
            / config.cable_position_scale,
            (self.base.body_linear_velocity(body) - hand_velocity) @ rotation
            / config.cable_velocity_scale,
            self.tangent(self.index) @ rotation,
        ]).astype(np.float32)

    def nearest(self, point):
        ids = self.base.cable_ids
        tolerance = self.wrapper.rl_config.target_node_tolerance
        first = max(0, self.index - tolerance)
        last = min(len(ids) - 1, self.index + tolerance)
        if first == last:
            position = self.base.data.xpos[ids[self.index]].copy()
            return position, float(np.linalg.norm(position - point)), self.tangent(self.index), self.index, 0.0
        indices = np.arange(first, last)
        starts = self.base.data.xpos[np.asarray(ids)[indices]]
        vectors = self.base.data.xpos[np.asarray(ids)[indices + 1]] - starts
        alpha = np.clip(np.sum((point - starts) * vectors, axis=1)
                        / np.maximum(np.sum(vectors ** 2, axis=1), 1e-12), 0.0, 1.0)
        projections = starts + alpha[:, None] * vectors
        distances = np.linalg.norm(projections - point, axis=1)
        chosen = int(np.argmin(distances))
        index = int(indices[chosen])
        tangent = ((1.0 - alpha[chosen]) * self.tangent(index)
                   + alpha[chosen] * self.tangent(index + 1))
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
        return projections[chosen].copy(), float(distances[chosen]), tangent, index, float(alpha[chosen])

    def finish_reward(self, components, status):
        pinched = bool(status['pinch_confirmed'])
        matched = self.matches()
        wrong_event = pinched and not matched and not self.was_pinched
        if wrong_event:
            self.wrong_count += 1
        penalty = min(0.5, max(0.0, 2.0 - self.wrong_penalty)) if wrong_event else 0.0
        self.wrong_penalty += penalty
        components['reward_wrong_target_grasp'] = -penalty
        self.previous_match = pinched and matched
        self.previous_secured = bool(status['secured_grasp']) and matched
        self.was_pinched = pinched

    def info(self):
        actual = self.actual_index()
        return {
            'target_node_task': True, 'goal_node_index': self.index,
            'goal_material_u': self.index / (len(self.base.cable_ids) - 1),
            'goal_node_tolerance': self.wrapper.rl_config.target_node_tolerance,
            'actual_grasp_node_index': actual,
            'goal_node_error': None if actual is None else abs(actual - self.index),
            'goal_match': bool(self.base.grasp_confirmed and self.matches()),
            'goal_success': bool(self.base.ever_success),
            'any_node_success': bool(self.any_success),
            'wrong_target_grasp_count': self.wrong_count,
        }


def warm_start_policy(destination, checkpoint: Path):
    """Expand the existing 99-D MLP, keeping a fresh optimizer and schedules."""
    from stable_baselines3 import PPO
    from .evaluate import checkpoint_action_mode, checkpoint_interface_version
    from .environment import action_interface_version

    checkpoint = Path(checkpoint)
    mode = destination.get_env().get_attr('action_mode')[0]
    if (checkpoint_action_mode(checkpoint) != mode
            or checkpoint_interface_version(checkpoint) != action_interface_version(mode)):
        raise ValueError('warm-start checkpoint action interface differs')
    source = PPO.load(checkpoint, device=destination.device, custom_objects={
        'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0, 'clip_range': 0.2,
    })
    if source.observation_space.shape != (99,) or destination.observation_space.shape != (109,):
        raise ValueError('warm-start requires state99 source and target-node109 destination')
    if (source.action_space.shape != destination.action_space.shape
            or source.policy.net_arch != destination.policy.net_arch
            or source.policy.activation_fn != destination.policy.activation_fn):
        raise ValueError('warm-start requires matching policy architecture and actions')
    old, new = source.policy.state_dict(), destination.policy.state_dict()
    if old.keys() != new.keys():
        raise ValueError('warm-start policy state keys differ')
    expanded = {'mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight'}
    for key in new:
        if key in expanded:
            if old[key].shape != (new[key].shape[0], 99) or new[key].shape[1] != 109:
                raise ValueError('unexpected first-layer shape: ' + key)
            new[key].zero_()
            new[key][:, :99] = old[key]
        elif old[key].shape == new[key].shape:
            new[key] = old[key].clone()
        else:
            raise ValueError('incompatible warm-start parameter: ' + key)
    destination.policy.load_state_dict(new, strict=True)
    return {'source': str(checkpoint.resolve()),
            'source_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'source_timesteps': int(source.num_timesteps),
            'optimizer_transferred': False, 'new_goal_columns': 'zero initialized'}
