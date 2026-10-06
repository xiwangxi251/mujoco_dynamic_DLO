import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3 import PPO

from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
from panda_cable_grasp.rl.penalty_ramp import resolve_schedule, apply_progress
from panda_cable_grasp.rl.workspace import WorkspaceMask, DEFAULT_TARGET_WORKSPACE, WORKSPACE_STAGES
from panda_cable_grasp.rl.evaluate import parse_args, configure_target_evaluation


@pytest.fixture
def env():
    instance = make_rl_env(robot='nero', action_mode='task_space_vertical_down',
        scenario_names=['id_static'], rl_config=RLConfig(target_node_task=True,
            target_reward_profile='tolerance_progress_v6', safety_reward_enabled=True))
    instance.reset(seed=81)
    yield instance
    instance.close()


def test_checkpoint_restores_ramp_across_worker_change_and_episode_reset(env, tmp_path):
    model = PPO('MlpPolicy', env, n_steps=8, batch_size=4, n_epochs=1, device='cpu')
    model.num_timesteps = 120000
    model.penalty_ramp_schedule = resolve_schedule(model, env.rl_config, 24)
    model.target_evaluation_stage = 2
    model.save(tmp_path/'policy')
    loaded = PPO.load(tmp_path/'policy', device='cpu')
    schedule = resolve_schedule(loaded, env.rl_config, 48)
    assert schedule['reference_workers'] == 24
    assert schedule['height_full_samples'] == 240000
    assert loaded.target_evaluation_stage == 2
    apply_progress(env, loaded, schedule, training_workers=48)
    assert env._penalty_ramp_scale('height') == .5
    assert env._penalty_ramp_scale('safety') == 1.
    env.step(np.zeros(5))
    before = env._penalty_ramp_scale('height')
    assert before == pytest.approx((120000+48)/240000)
    env.reset(seed=82)
    assert env._penalty_ramp_scale('height') == before
    # A recreated worker resumes the exact same global progress.
    loaded.num_timesteps += 48
    apply_progress(env, loaded, schedule, training_workers=48)
    assert env._penalty_ramp_scale('height') == before


def test_old_checkpoint_migrates_using_saved_parallelism_and_eval_is_fixed(env):
    model = SimpleNamespace(num_timesteps=60000)
    schedule = resolve_schedule(model, env.rl_config, 48, {'workers': 24})
    apply_progress(env, model, schedule)
    env.step(np.zeros(5))
    assert env._penalty_ramp_scale('height') == .25
    env.reset(seed=83)
    env.step(np.zeros(5))
    assert env._penalty_ramp_scale('height') == .25
    # Changing evaluation history does not change an identical safety event.
    env._table_finger_filter_active = True
    env._table_finger_min_distance = lambda data: float('inf')
    env._last_ik_singularity_sigma_min = env._last_ik_joint_margin_ratio = 1.
    env._table_safety_penalty_total = 0.
    assert env._safety_reward_components()['reward_table_filter'] == pytest.approx(-.01)


def saved_args(tmp_path, monkeypatch, extra=()):
    names = ['a','b','c','d','e','f']
    config = dict(target_node_task=True, workspace_scene_family='cable_x075_v1',
        target_workspace=WorkspaceMask(DEFAULT_TARGET_WORKSPACE).manifest(),
        workspace_curriculum=True, training_scenario_names=names, workers=48,
        rl_config=asdict(RLConfig(target_node_task=True)),
        controller_profile='enhanced6_bounded_dls_v1', table_execution_guard_enabled=True,
        table_finger_collision_filter_enabled=False)
    (tmp_path/'training_config.json').write_text(json.dumps(config))
    (tmp_path/'evaluation').mkdir()
    (tmp_path/'evaluation/workspace_curriculum.json').write_text(json.dumps({'stage': 5}))
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--model', str(tmp_path/'policy.zip'), *extra])
    return parse_args(), config


def test_saved_stage_defaults_use_checkpoint_stage_not_later_run_stage(tmp_path, monkeypatch):
    args, config = saved_args(tmp_path, monkeypatch)
    assert WorkspaceMask(args.target_workspace).manifest()['xy_bounds_m'] == [.60,.84,-.22,.30]
    calls = {}
    env = SimpleNamespace(goal=SimpleNamespace(), rl_config=RLConfig(),
        set_penalty_ramp_progress=lambda *values: calls.update(ramp=values),
        set_training_scenario_distribution=lambda names, weights: calls.update(names=names, weights=weights),
        set_motion_difficulty=lambda value: calls.update(difficulty=value))
    model = SimpleNamespace(num_timesteps=100000, target_evaluation_stage=2)
    configure_target_evaluation(env, args, model)
    assert calls['weights'] == WORKSPACE_STAGES[2][1]
    assert calls['difficulty'] == WORKSPACE_STAGES[2][2]
    assert args.resolved_evaluation_stage['source'] == 'checkpoint'
    assert calls['ramp'][1] == 0


@pytest.mark.parametrize('extra', [('--evaluation-protocol', 'six-scenes'), ('--scenario', 'id_static')])
def test_explicit_scenario_protocol_does_not_apply_saved_curriculum(tmp_path, monkeypatch, extra):
    args, _ = saved_args(tmp_path, monkeypatch, extra)
    assert not args.saved_stage_evaluation


def test_unavailable_workspace_is_not_silently_replaced(tmp_path, monkeypatch):
    args, config = saved_args(tmp_path, monkeypatch)
    config['target_workspace'].update(path=str(tmp_path/'absent.npz'), sha256='different')
    (tmp_path/'training_config.json').write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        parse_args()


def test_archived_checkpoint_uses_frozen_curriculum_before_mutable_run(tmp_path, monkeypatch):
    from panda_cable_grasp.rl.target_defaults import saved_curriculum_stage
    args, saved = saved_args(tmp_path, monkeypatch)
    (tmp_path/'workspace_curriculum.json').write_text(json.dumps({'stage': 1}))
    assert saved_curriculum_stage(args.model, saved) == (1, 'archived_curriculum')
