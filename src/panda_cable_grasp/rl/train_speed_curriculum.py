"""C-group entry point: resume a frozen ordinary policy, retain its optimizer."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from .environment import RLConfig
from .exact_budget import install_exact_budget
from .metrics import TrainingMetricsCallback, plot_training_curves
from .penalty_ramp import apply_progress
from .speed_curriculum import (SpeedEnvironment, SpeedState, SpeedPPO, SpeedCallback,
    PRETRAINING_STEPS, ADDITIONAL_BUDGET, VERSION, schedule_value, atomic, verify_workers)


def worker(rank, config, state):
    def initialize():
        env = SpeedEnvironment(robot='nero', seed=config['seed']+rank,
            disturbance_strength=config['disturbance'], episode_seconds=config['episode_seconds'],
            dynamicvla_cameras_enabled=False, scenario_names=config['training_scenario_names'],
            rl_config=RLConfig(**config['rl_config']),
            geometric_safety_enabled=config['geometric_safety_enabled'],
            table_finger_collision_filter_enabled=config['table_finger_collision_filter_enabled'],
            controller_profile=config['controller_profile'],
            table_execution_guard_enabled=config['table_execution_guard_enabled'],
            replay_bank_dir=config['replay_bank_dir'], replay_seed_fallback=config['replay_seed_fallback'])
        env.base_env.config.contact_read_cache_enabled = config['contact_read_cache_enabled']
        env.configure_speed_stage(state)
        env.set_penalty_ramp_progress(PRETRAINING_STEPS, 0, config['penalty_ramp_schedule'])
        return env
    return initialize


class EntropySchedule(BaseCallback):
    def _on_rollout_start(self):
        schedule = self.model.c_adaptation_schedule
        self.model.ent_coef = float(schedule_value(schedule['entropy_initial'], schedule['entropy_final'],
                                                 self.model.num_timesteps-PRETRAINING_STEPS))

    def _on_step(self):
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--additional', type=int, default=ADDITIONAL_BUDGET)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.resume or Path(plan['checkpoint'])
    if not args.resume:
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == plan['checkpoint_sha256']
    torch.set_num_threads(1)
    set_random_seed(plan['configuration']['seed'])
    # Read metadata before the first worker reset. Retain all original history
    # separately; the original phase 14 must not select the C startup scene.
    model = SpeedPPO.load(checkpoint, device='cpu', n_steps=1024, batch_size=512, n_epochs=5,
                          target_kl=None, seed=plan['configuration']['seed'])
    if not args.resume:
        assert model.num_timesteps == PRETRAINING_STEPS and model._n_updates == 1715
        assert model.observation_space.shape == (99,) and model.action_space.shape == (5,)
        assert model.ordinary_evaluation_phase == 14
        model.c_original151_history = dict(steps=model.num_timesteps, phase=14,
            rollout_gate_state=getattr(model, 'rollout_gate_state', None),
            global_schedule=getattr(model, 'training_global_schedule', None),
            checkpoint_sha256=plan['checkpoint_sha256'])
        model.c_adaptation_schedule = dict(version='c_added_samples_v1', total_added=ADDITIONAL_BUDGET,
            learning_rate_initial=float(model.policy.optimizer.param_groups[0]['lr']),
            learning_rate_final=3e-5, entropy_initial=float(model.ent_coef), entropy_final=.001)
        model.rollout_gate_state = {}
        state = SpeedState().record()
    else:
        assert model.c_adaptation_schedule['version'] == 'c_added_samples_v1'
        state = SpeedState(model.c_speed_state).record()
    start = int(model.num_timesteps)
    added = start-PRETRAINING_STEPS
    assert 0 <= added < ADDITIONAL_BUDGET and 0 < args.additional <= ADDITIONAL_BUDGET-added
    configuration = dict(plan['configuration'])
    configuration['rl_config'] = dict(configuration['rl_config'])
    configuration['workers'] = args.workers
    s = SpeedState(state)
    configuration['rl_config'].update(arm_velocity_limits_enabled=True,
        arm_velocity_limit_scale=s.scale, translation_delta_scale=.01*s.scale, yaw_delta_scale=.02*s.scale)
    accepted = {f.name for f in fields(RLConfig)}
    assert not (set(configuration['rl_config'])-accepted)
    env = VecMonitor(SubprocVecEnv([worker(i, configuration, state) for i in range(args.workers)], start_method='spawn'),
        filename=str(args.output/'monitor.csv'), info_keywords=('success', 'strict_success', 'ever_pinched',
            'ever_aligned_pinch', 'lift_attempt', 'loaded_lift', 'ever_grasped', 'lifted_fraction'))
    try:
        history = getattr(model, 'c_original151_history', None)
        adaptation_schedule = model.c_adaptation_schedule
        model = SpeedPPO.load(checkpoint, env=env, device='cpu', n_steps=1024, batch_size=512,
                              n_epochs=5, target_kl=None, seed=configuration['seed'])
        model.c_original151_history = history
        model.c_adaptation_schedule = adaptation_schedule
        model.async_rollouts_enabled = bool(plan.get('async_rollouts_enabled', False))
        model.rollout_gate_state = {}
        assert model.n_envs == args.workers
        env.reset()
        audits = env.env_method('execution_configuration')
        verify_workers(audits, state)
        policy_finite = all(bool(torch.isfinite(t).all()) for t in model.policy.parameters())
        assert policy_finite and model.policy.optimizer.state
        model.penalty_ramp_schedule = configuration['penalty_ramp_schedule']
        apply_progress(env, model, model.penalty_ramp_schedule, training_workers=args.workers)
        cumulative_target = start+args.additional
        schedule = model.c_adaptation_schedule
        def learning_rate(progress):
            global_steps = cumulative_target*(1-progress)
            return float(schedule_value(schedule['learning_rate_initial'], schedule['learning_rate_final'],
                                        global_steps-PRETRAINING_STEPS))
        model.learning_rate = model.lr_schedule = learning_rate
        model.verbose = 1
        model.tensorboard_log = str(args.output/'tensorboard')
        configuration.update(algorithm='PPO', starting_timesteps=start,
            original151_pretraining_steps=PRETRAINING_STEPS, additional_budget=ADDITIONAL_BUDGET,
            invocation_target_added=added+args.additional, invocation_additional=args.additional,
            cumulative_target_timesteps=cumulative_target, c_adaptation_schedule=schedule,
            c_speed_state=state, source_commit=plan['source_commit'], configuration_sha256=plan['configuration_sha256'],
            starting_checkpoint_sha256=plan['checkpoint_sha256'], reset_num_timesteps=False,
            original_151_config=plan['original_151_config'], statistics_source='stochastic training episodes',
            independent_evaluation_enabled=False, rollout_gate_profile='fast after speed adaptation',
            rollout_sampler='async_fixed_quota_v1' if model.async_rollouts_enabled else 'synchronous_vector_steps',
            speed_levels=[1, 1.25, 1.5, 1.75, 2], minimum_added_per_speed=200000,
            success_threshold=.8, required_nonoverlapping_windows=2, episodes_per_speed_window=100)
        atomic(args.output/'training_config.json', configuration)
        atomic(args.output/'input_load_receipt.json', dict(input_steps=start, optimizer_epochs=int(model._n_updates),
            loaded_optimizer=True, parameters_finite=policy_finite,
            learning_rate=float(model.policy.optimizer.param_groups[0]['lr']), entropy=float(model.ent_coef),
            original_phase=14 if not args.resume else None, migrated_phase=0 if s.mode=='speed' else s.scene_phase,
            workers=args.workers, identity=plan['configuration_sha256'], c_initial_state=state))
        cb = SpeedCallback(args.output, state, configuration)
        callbacks = CallbackList([cb, EntropySchedule(), TrainingMetricsCallback(args.output/'training_metrics.csv', window=100)])
        install_exact_budget()
        model.learn(total_timesteps=args.additional, callback=callbacks, reset_num_timesteps=False, progress_bar=False)
        assert model.num_timesteps == cumulative_target
        model.save(args.output/'final_model')
        plot_training_curves(args.output/'training_metrics.csv', args.output, window=100)
    finally:
        env.close()


if __name__ == '__main__':
    main()
