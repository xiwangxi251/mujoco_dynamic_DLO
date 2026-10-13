"""Warm-started speed curriculum with changes only after a PPO update."""
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .environment import VerticalDownRLCableGraspEnv
from ..env.control import scaled_full6_profile, configure_profile
from .rollout_curriculum import (Windows, FAST_PROTOCOL, GROUPS, PHASES,
                                MIXES, ALL_SCENES, CONSOLIDATION_MIX, difficulties)

VERSION = 'warm_speed_curriculum_v1'
LEVELS = (1., 1.25, 1.5, 1.75, 2.)
PRETRAINING_STEPS = 16997280
ADDITIONAL_BUDGET = 20000000


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def schedule_value(initial, final, added):
    return initial + (final-initial)*np.clip(added/ADDITIONAL_BUDGET, 0., 1.)


class SpeedState:
    """Exact, nonoverlapping windows; only episodes after the minimum count."""
    def __init__(self, saved=None):
        saved = saved or {}
        if saved and saved.get('version') != VERSION:
            raise ValueError('Incompatible speed curriculum state')
        self.level_index = int(saved.get('level_index', 0))
        self.level_start_added = int(saved.get('level_start_added', 0))
        self.mode = saved.get('mode', 'speed')
        self.scene_phase = int(saved.get('scene_phase', 0))
        self.window = list(saved.get('window', []))
        self.consecutive = int(saved.get('consecutive', 0))
        self.pending = bool(saved.get('pending', False))
        self.generation = int(saved.get('generation', 0))
        self.stall_notified = bool(saved.get('stall_notified', False))
        if not 0 <= self.level_index < len(LEVELS) or self.mode not in ('speed', 'scenes'):
            raise ValueError('Invalid speed stage')
        if not 0 <= self.scene_phase <= 14 or len(self.window) >= 100:
            raise ValueError('Invalid scene/window state')

    @property
    def scale(self):
        return LEVELS[self.level_index]

    def record(self):
        return dict(version=VERSION, **vars(self))

    def add(self, success, added, episode_start_added):
        # A pre-minimum episode must not supply evidence to a post-minimum gate.
        if (self.mode != 'speed' or self.pending
                or episode_start_added < self.level_start_added + 200000):
            return None
        self.window.append(int(bool(success)))
        if len(self.window) < 100:
            return None
        count = sum(self.window)
        self.window.clear()
        self.consecutive = self.consecutive + 1 if count >= 80 else 0
        self.pending = self.consecutive >= 2
        return dict(successes=count, episodes=100, passed=count >= 80,
                    consecutive=self.consecutive, promote=self.pending,
                    added_steps=added, scale=self.scale)

    def advance(self, added):
        if not self.pending:
            raise RuntimeError('No completed gate to advance')
        if self.mode == 'speed':
            if self.level_index < len(LEVELS)-1:
                self.level_index += 1
            else:
                self.mode, self.scene_phase = 'scenes', 1
        else:
            self.scene_phase = min(14, self.scene_phase+1)
        self.generation += 1
        self.level_start_added = added
        self.window, self.consecutive, self.pending = [], 0, False
        self.stall_notified = False


class SpeedEnvironment(VerticalDownRLCableGraspEnv):
    """Keep execution and action scales consistent through every reset."""
    def configure_speed_stage(self, state):
        s = SpeedState(state)
        self.rl_config = replace(self.rl_config, arm_velocity_limits_enabled=True,
            arm_velocity_limit_scale=s.scale,
            translation_delta_scale=.01*s.scale, yaw_delta_scale=.02*s.scale)
        self.base_env.config.arm_velocity_limits_enabled = True
        self.base_env.config.arm_velocity_limit_scale = s.scale
        configure_profile(self.base_env.config, 'enhanced6_bounded_dls_v1')
        self.base_env._arm_velocity_limits = np.asarray(
            self.base_env.config.arm_joint_velocity_limits)*s.scale
        self._full6_control_profile = scaled_full6_profile(s.scale)
        self.speed_generation = s.generation
        if s.mode == 'speed':
            names, weights, phase = GROUPS[0], (1.,), 0
        elif s.scene_phase == 14:
            names, weights, phase = tuple(n for n, _ in CONSOLIDATION_MIX), tuple(p for _, p in CONSOLIDATION_MIX), 14
        else:
            phase = s.scene_phase
            mix = MIXES[PHASES[phase][0]]
            names, weights = tuple(n for n, _ in mix), tuple(p for _, p in mix)
        self.set_rollout_curriculum(phase, names, weights, difficulties(phase))
        return self.execution_configuration()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        linear, angular = self._hand_velocity()
        info.update(speed_generation=self.speed_generation,
                    arm_velocity_limit_scale=self.rl_config.arm_velocity_limit_scale,
                    actual_tcp_linear_speed=float(np.linalg.norm(linear)),
                    actual_tcp_angular_speed=float(np.linalg.norm(angular)),
                    actual_joint_speed=float(np.max(np.abs(self.data.qvel[self.base_env.arm_dof_adr]))))
        return obs, reward, terminated, truncated, info


def verify_workers(audits, state, original_compiled=None):
    s = SpeedState(state)
    if not audits:
        raise ValueError('No worker audit')
    for a in audits:
        r, e, f = a['rl_config'], a['env_config'], a['effective_full6_profile']
        assert r['translation_delta_scale'] == .01*s.scale
        assert r['yaw_delta_scale'] == .02*s.scale
        assert r['arm_velocity_limits_enabled'] and r['arm_velocity_limit_scale'] == s.scale
        assert e['arm_velocity_limits_enabled'] and e['arm_velocity_limit_scale'] == s.scale
        assert e['hand_cartesian_velocity_limit_enabled'] and not e['arm_acceleration_limit_enabled']
        assert e['hand_linear_velocity_limit'] == .9*s.scale and e['hand_angular_velocity_limit'] == 2*s.scale
        assert f == asdict(scaled_full6_profile(s.scale))
        np.testing.assert_allclose(a['effective_arm_joint_velocity_limits'], np.asarray(e['arm_joint_velocity_limits'])*s.scale)
        assert e['gripper_finger_velocity_limit'] == .2 and e['table_execution_guard_enabled']
        assert not e['geometric_safety_enabled']
        assert a['physics_timestep'] == .002 and e['frame_skip'] == 10
        assert a['observation_shape'] == [99] and a['action_shape'] == [5]
        assert a['compiled_hashes'] == (original_compiled or audits[0]['compiled_hashes'])


class SpeedPPO(PPO):
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if getattr(self, 'async_rollouts_enabled', False):
            from .async_rollouts import collect_fixed_quota
            return collect_fixed_quota(self, env, callback, rollout_buffer, n_rollout_steps)
        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['_speed_callback']

    def train(self):
        super().train()
        if hasattr(self, '_speed_callback'):
            self._speed_callback.after_update()


class SpeedCallback(BaseCallback):
    def __init__(self, output, state, configuration):
        super().__init__()
        self.output = Path(output)
        self.state = SpeedState(state)
        self.configuration = configuration
        self.last_checkpoint = 0
        self.episode_count = 0
        self.recent = []
        self.scene_windows = None
        self.speeds = {k: [] for k in ('tcp_linear', 'tcp_angular', 'joint')}

    @property
    def added(self):
        return int(self.model.num_timesteps)-PRETRAINING_STEPS

    def _scene_windows(self, saved=None):
        phase = self.state.scene_phase
        targets = ALL_SCENES if phase == 14 else GROUPS[PHASES[phase][0]]
        replay = () if phase == 14 else tuple(n for group in GROUPS[:PHASES[phase][0]] for n in group)
        self.scene_windows = Windows(targets, replay, saved, FAST_PROTOCOL)

    def _on_training_start(self):
        self.model._speed_callback = self
        self.starting_updates = int(self.model._n_updates)
        self.episode_start = np.full(self.model.n_envs, self.added, dtype=np.int64)
        self.last_checkpoint = self.added
        if self.state.mode == 'scenes':
            self._scene_windows(getattr(self.model, 'c_scene_windows', None))
        self._set_metadata()
        self._audit()
        self.started = time.monotonic()
        self.invocation_start_added = self.added
        self._persist()
        self._write_health()
        self.file = (self.output/'c_episodes.csv').open('a', newline='', encoding='utf-8')
        columns = ('episode', 'model_global_steps', 'c_added_steps', 'speed_scale', 'generation',
                   'mode', 'scene_phase', 'scenario', 'success', 'reward', 'length', 'duration_seconds',
                   'eligible_speed_gate', 'eligible_scene_gate', 'ever_grasped', 'loaded_lift')
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if self.file.tell() == 0:
            self.writer.writeheader()

    def _set_metadata(self):
        self.model.c_speed_state = self.state.record()
        self.model.c_scene_windows = self.scene_windows.state() if self.scene_windows else None
        self.model.original151_pretraining_steps = PRETRAINING_STEPS
        self.model.c_added_steps = self.added
        self.model.ordinary_evaluation_phase = 0 if self.state.mode == 'speed' else self.state.scene_phase
        phase = self.model.ordinary_evaluation_phase
        self.model.ordinary_evaluation_difficulty = 0. if phase == 14 else PHASES[phase][1]
        self.model.ordinary_evaluation_scenarios = ALL_SCENES if phase == 14 else GROUPS[PHASES[phase][0]]
        self.configuration['rl_config'].update(arm_velocity_limits_enabled=True,
            arm_velocity_limit_scale=self.state.scale, translation_delta_scale=.01*self.state.scale,
            yaw_delta_scale=.02*self.state.scale)
        self.configuration.update(c_speed_state=self.state.record(), model_global_steps=self.model.num_timesteps,
                                  c_added_steps=self.added)

    def _audit(self):
        audits = self.training_env.env_method('execution_configuration')
        verify_workers(audits, self.state.record(), getattr(self, 'original_compiled', None))
        for a in audits:
            assert a['rl_config'] == self.configuration['rl_config']
        self.original_compiled = audits[0]['compiled_hashes']
        atomic(self.output/'worker_configuration_audit.json', audits)
        self.workers_verified = len(audits)

    def _persist(self):
        self._set_metadata()
        atomic(self.output/'training_config.json', self.configuration)
        atomic(self.output/'speed_curriculum_status.json', dict(self.state.record(), updated_at=time.time(),
            c_added_steps=self.added, model_global_steps=int(self.model.num_timesteps),
            recent_episodes=len(self.recent), recent_successes=sum(x['success'] for x in self.recent),
            recent_reward_mean=float(np.mean([x['reward'] for x in self.recent])) if self.recent else None,
            scene_windows=self.scene_windows.state() if self.scene_windows else None,
            minimum_added_per_speed=200000, windows_required=2, episodes_per_window=100,
            success_threshold=.8, speed_curriculum_complete=self.state.mode == 'scenes'))

    def _write_health(self):
        tensors = list(self.model.policy.parameters())
        for state in self.model.policy.optimizer.state.values():
            tensors.extend(v for v in state.values() if __import__('torch').is_tensor(v))
        finite = all(bool(__import__('torch').isfinite(t).all()) for t in tensors)
        metrics = {k: float(v) for k, v in self.logger.name_to_value.items()
                   if k.startswith('train/') and isinstance(v, (int, float, np.number))}
        if not finite or not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Nonfinite PPO update')
        health = dict(updated_at=time.time(), timesteps=int(self.model.num_timesteps), c_added_steps=self.added,
            starting_updates=self.starting_updates, updates=int(self.model._n_updates),
            completed_episodes=self.episode_count, parameters_finite=finite, worker_ready=True,
            workers_verified=self.workers_verified,
            normal_training=int(self.model._n_updates)>self.starting_updates and bool(metrics),
            optimization_metrics=metrics, scale=self.state.scale, mode=self.state.mode,
            scene_phase=self.state.scene_phase, phase=self.model.ordinary_evaluation_phase,
            rollout_fps=getattr(self, 'rollout_fps', None),
            overall_fps=(self.added-self.invocation_start_added)/max(time.monotonic()-self.started, .001),
            actual_speed_measurement='50Hz control boundary; not physics peak')
        health['actual_speeds'] = {k: dict(mean=float(np.mean(v)), peak=float(np.max(v))) if v else None
                                  for k, v in self.speeds.items()}
        health['sampler'] = 'async_fixed_quota_v1' if getattr(self.model, 'async_rollouts_enabled', False) else 'synchronous_vector_steps'
        health['async_rollout_receipt'] = getattr(self.model, 'async_rollout_receipt', None)
        atomic(self.output/'training_health.json', health)

    def _on_rollout_start(self):
        if not hasattr(self, 'invocation_start_added'):
            self.invocation_start_added = self.added
        self.sampling_start, self.sampling_added = time.monotonic(), self.added
        self.speeds = {k: [] for k in self.speeds}

    def _on_step(self):
        for key in ('rewards', 'new_obs'):
            if not np.all(np.isfinite(self.locals[key])):
                raise FloatingPointError('Nonfinite '+key)
        for i, (done, info) in enumerate(zip(self.locals['dones'], self.locals['infos'])):
            for k in self.speeds:
                v = info['actual_'+k+'_speed']
                if not np.isfinite(v):
                    raise FloatingPointError('Nonfinite actual speed')
                self.speeds[k].append(v)
            if not done:
                continue
            e = info['episode']
            success = int(bool(e['success']))
            eligible = info['speed_generation'] == self.state.generation
            eligible_speed = eligible and self.state.mode == 'speed' and self.episode_start[i] >= self.state.level_start_added+200000
            eligible_scene = eligible and self.state.mode == 'scenes' and int(info['rollout_episode_phase']) == self.state.scene_phase
            result = dict(success=success, loaded=int(bool(e['loaded_lift'])), secured=int(bool(e['ever_grasped'])), attempt=int(bool(e['lift_attempt'])))
            if eligible and self.state.mode == 'speed':
                decision = self.state.add(success, self.added, int(self.episode_start[i]))
                if decision:
                    self._event('speed_window', decision)
            elif eligible_scene:
                name = info['scenario_name']
                if abs(float(info['rollout_episode_difficulty'])-difficulties(self.state.scene_phase)[name]) < 1e-9:
                    self.scene_windows.add(name, result)
            self.episode_count += 1
            self.writer.writerow(dict(episode=self.episode_count, model_global_steps=self.num_timesteps,
                c_added_steps=self.added, speed_scale=self.state.scale, generation=self.state.generation,
                mode=self.state.mode, scene_phase=self.model.ordinary_evaluation_phase, scenario=info['scenario_name'],
                success=success, reward=e['r'], length=e['l'], duration_seconds=e['l']*.02,
                eligible_speed_gate=int(eligible_speed), eligible_scene_gate=int(eligible_scene),
                ever_grasped=result['secured'], loaded_lift=result['loaded']))
            self.recent = (self.recent+[dict(success=success, reward=e['r'])])[-100:]
            self.episode_start[i] = self.added
        self.file.flush()
        self._set_metadata()
        return True

    def _event(self, kind, value):
        record = dict(event=kind, c_added_steps=self.added, model_global_steps=self.num_timesteps, **value)
        with (self.output/'curriculum_events.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)

    def _on_rollout_end(self):
        self.rollout_fps = (self.added-self.sampling_added)/max(time.monotonic()-self.sampling_start, .001)
        if self.state.mode == 'scenes':
            decision = self.scene_windows.decide()
            if decision:
                self._event('scene_window', dict(phase=self.state.scene_phase, **decision))
                self.state.pending = decision['promote'] and self.state.scene_phase < 14
        self._set_metadata()

    def save_checkpoint(self, label):
        self._persist()
        directory = self.output/'checkpoints'/label
        directory.mkdir(parents=True, exist_ok=True)
        atomic(directory/'training_config.json', self.configuration)
        self.model.save(directory/'model')
        atomic(directory/'checkpoint_receipt.json', dict(c_added_steps=self.added,
            model_global_steps=int(self.model.num_timesteps), speed_state=self.state.record(),
            optimizer_epochs=int(self.model._n_updates)))

    def after_update(self):
        # SB3 has already optimized the old rollout. No old-state sample enters
        # a buffer computed using reset observations from the next stage.
        if self.state.pending and self.added < self.configuration['invocation_target_added']:
            before = self.state.record()
            self.save_checkpoint(f'c_{self.added}_before_transition')
            self.state.advance(self.added)
            self.scene_windows = None
            if self.state.mode == 'scenes':
                self._scene_windows()
            self.training_env.env_method('configure_speed_stage', self.state.record())
            self._set_metadata()
            self.model._last_obs = self.training_env.reset()
            self.model._last_episode_starts = np.ones(self.model.n_envs, dtype=bool)
            self.episode_start[:] = self.added
            self.recent.clear()
            self._audit()
            self.save_checkpoint(f'c_{self.added}_after_transition')
            self._event('transition_after_update', dict(previous=before, current=self.state.record(),
                                                       optimizer_epochs=int(self.model._n_updates)))
        if self.added-self.last_checkpoint >= 200000:
            self.save_checkpoint(f'c_{self.added}')
            self.last_checkpoint = self.added
        if self.state.mode == 'speed' and self.added-self.state.level_start_added >= 1000000 and not self.state.stall_notified:
            self.state.stall_notified = True
            self._event('speed_stage_needs_attention', dict(scale=self.state.scale, advance_forced=False))
        self._persist()
        self._write_health()

    def _on_training_end(self):
        self._persist()
        self._write_health()
        self.file.close()
