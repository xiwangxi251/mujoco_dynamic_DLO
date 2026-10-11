import json
import numpy as np
import pytest
from unittest.mock import patch
from panda_cable_grasp.rl.train import parse_args, make_configured_env
from panda_cable_grasp.rl.evaluate import parse_args as evaluation_args
from panda_cable_grasp.rl.workspace import WORKSPACE_SCENE_FAMILIES


def test_target_training_enables_validated_stack_and_new_seven_scenes_by_default():
    with patch('sys.argv', ['train', '--target-node-task']):
        args = parse_args()
    assert args.robot == 'nero'
    assert args.action_mode == 'task_space_vertical_down'
    assert args.controller_profile == 'enhanced6_bounded_dls_v1'
    assert args.table_execution_guard
    assert args.disable_table_finger_collision_filter
    assert args.target_reward_profile == 'tolerance_progress_v6'
    from panda_cable_grasp.rl.workspace import WorkspaceMask
    mask=WorkspaceMask(args.target_workspace)
    assert mask.manifest()['kind']=='annular_sector'
    assert mask.manifest()['center_xy_m']==[.35,0.]
    assert mask.manifest()['radius_range_m']==[.28,.56]
    assert mask.manifest()['angle_range_deg']==[-30.,70.]
    assert mask.manifest()['provenance']['selected_passing_samples']==88
    assert args.workspace_scene_family == 'cable_x075_v1'
    assert tuple(args.training_scenario_names) == tuple(args.eval_scenario_names) == WORKSPACE_SCENE_FAMILIES['cable_x075_v1']
    assert len(args.training_scenario_names) == 7


def test_non_target_training_shares_execution_defaults_without_changing_task():
    with patch('sys.argv', ['train']):
        args = parse_args()
    assert (args.robot, args.action_mode, args.controller_profile) == (
        'nero', 'task_space_vertical_down', 'enhanced6_bounded_dls_v1')
    assert args.table_execution_guard
    assert args.disable_table_finger_collision_filter
    assert args.execution_defaults_source == 'fresh_default:nero'
    assert not args.target_node_task
    assert args.target_workspace is None
    assert args.workspace_scene_family is None
    assert args.target_reward_profile == 'legacy'
    assert args.ordinary_reward_profile == 'legacy'


@pytest.mark.parametrize('target', [False, True])
def test_explicit_legacy_execution_and_individual_overrides(target):
    argv = ['train', '--execution-profile', 'legacy'] + (['--target-node-task'] if target else [])
    with patch('sys.argv', argv):
        args = parse_args()
    assert (args.robot, args.action_mode, args.controller_profile) == ('panda', 'task_space', 'legacy')
    assert not args.table_execution_guard and not args.disable_table_finger_collision_filter
    assert args.execution_defaults_source == 'explicit_profile:legacy'
    with patch('sys.argv', argv + ['--robot', 'nero', '--action-mode', 'task_space_vertical_down',
            '--controller-profile', 'enhanced6_bounded_dls_v1', '--table-execution-guard',
            '--disable-table-finger-collision-filter']):
        args = parse_args()
    assert args.robot == 'nero' and args.action_mode == 'task_space_vertical_down'
    assert args.controller_profile == 'enhanced6_bounded_dls_v1'
    assert args.table_execution_guard and args.disable_table_finger_collision_filter


def test_incompatible_partial_override_fails_before_environment_creation(capsys):
    with patch('sys.argv', ['train', '--robot', 'panda']), pytest.raises(SystemExit):
        parse_args()
    assert '--execution-profile legacy' in capsys.readouterr().err


@pytest.mark.parametrize('profile,version', [
    ('legacy', 'baseline_v4'), ('nero', 'baseline_v6_task_space_vertical_down')])
def test_resume_restores_saved_execution_with_individual_overrides(tmp_path, profile, version):
    from panda_cable_grasp.rl.target_defaults import PPO_EXECUTION_PROFILES
    execution = PPO_EXECUTION_PROFILES[profile]
    checkpoint = tmp_path/'checkpoints/checkpoint.zip'
    checkpoint.parent.mkdir()
    checkpoint.touch()
    (tmp_path/'training_config.json').write_text(json.dumps(dict(
        robot=execution['robot'], action_mode=execution['action_mode'],
        controller_profile=execution['controller_profile'],
        table_execution_guard_enabled=execution['table_execution_guard'],
        table_finger_collision_filter_enabled=not execution['disable_table_finger_collision_filter'],
        rl_interface_version=version)))
    argv = ['train', '--resume', str(checkpoint)]
    with patch('sys.argv', argv):
        args = parse_args()
    for key, value in execution.items():
        assert getattr(args, key) == value
    assert args.execution_defaults_source == 'checkpoint'
    with patch('sys.argv', argv + ['--no-table-execution-guard', '--enable-table-finger-collision-filter']):
        args = parse_args()
    assert not args.table_execution_guard and not args.disable_table_finger_collision_filter
    assert args.robot == execution['robot'] and args.action_mode == execution['action_mode']


def test_resume_with_missing_settings_requires_explicit_choice(tmp_path, capsys):
    checkpoint = tmp_path/'checkpoint.zip'
    checkpoint.touch()
    (tmp_path/'training_config.json').write_text(json.dumps(dict(rl_interface_version='baseline_v4')))
    argv = ['train', '--resume', str(checkpoint)]
    with patch('sys.argv', argv), pytest.raises(SystemExit):
        parse_args()
    assert 'Resume checkpoint lacks execution settings' in capsys.readouterr().err
    with patch('sys.argv', argv + ['--execution-profile', 'legacy']):
        assert parse_args().robot == 'panda'
    with patch('sys.argv', argv + ['--execution-profile', 'nero']), pytest.raises(SystemExit):
        parse_args()
    assert 'different RL action interface' in capsys.readouterr().err


def test_non_target_defaults_reach_physical_execution(tmp_path):
    with patch('sys.argv', ['train', '--training-scenarios', 'id_static',
                           '--curriculum-stage-steps', '0', '--output', str(tmp_path/'run')]):
        args = parse_args()
    env = make_configured_env(args, seed=81, scenario_names=args.training_scenario_names)
    try:
        observation, _ = env.reset(seed=81)
        assert observation.shape == (99,) and env.action_space.shape == (5,)
        with patch.object(env, '_probe_table_finger_distance', side_effect=AssertionError('old filter called')):
            observation, reward, _, _, info = env.step(np.zeros(5))
        assert np.isfinite(observation).all() and np.isfinite(reward)
        assert info['table_execution_guard_enabled']
        assert info['table_execution_guard']['guard_steps'] == 10
        assert not info['table_finger_collision_avoidance_enabled']
    finally:
        env.close()


def test_defaults_reach_physical_execution_without_guard_or_family_flags(tmp_path):
    mask = tmp_path/'mask.npz'
    np.savez(mask, xs=[0.,1.5], ys=[-1.,1.], interior=np.ones((2,2),bool))
    with patch('sys.argv', ['train','--target-node-task','--target-workspace',str(mask),
                           '--workspace-curriculum','--episode-seconds','.04',
                           '--output',str(tmp_path/'run')]):
        args = parse_args()
    env = make_configured_env(args,seed=81,scenario_names=args.training_scenario_names)
    try:
        _, initial = env.reset(seed=81)
        assert initial['scenario_name'] == 'dev_static_x075_v1'
        _, _, _, _, info = env.step(np.zeros(5))
        assert info['table_execution_guard_enabled']
        assert info['table_execution_guard']['guard_steps'] == 10
        assert not info['table_finger_collision_avoidance_enabled']
        # The old complete-scene prediction path must never run, while the
        # new 500 Hz execution guard stays active.
        with patch.object(env, '_probe_table_finger_distance', side_effect=AssertionError('old filter called')):
            env.step(np.zeros(5))
        assert env.base_env._table_execution_guard.steps == 20
    finally:
        env.close()


def test_explicit_historical_target_settings_remain_available():
    with patch('sys.argv', ['train', '--target-node-task', '--robot', 'panda',
                           '--action-mode', 'task_space', '--controller-profile', 'legacy',
                           '--no-table-execution-guard', '--target-reward-profile', 'legacy',
                           '--enable-table-finger-collision-filter',
                           '--workspace-scene-family', 'legacy_ppo_six']):
        args = parse_args()
    assert args.controller_profile == 'legacy' and not args.table_execution_guard
    assert not args.disable_table_finger_collision_filter
    assert args.target_reward_profile == 'legacy'
    assert tuple(args.training_scenario_names) == WORKSPACE_SCENE_FAMILIES['legacy_ppo_six']


def test_explicit_scenario_does_not_get_replaced_by_default_family():
    with patch('sys.argv', ['train', '--target-node-task', '--training-scenarios', 'id_static',
                           '--curriculum-stage-steps', '0']):
        args = parse_args()
    assert args.workspace_scene_family is None
    assert args.training_scenario_names == ['id_static']


def test_evaluation_inherits_saved_new_defaults_and_preserves_historical_settings(tmp_path):
    model = tmp_path/'checkpoint.zip'
    for family,guard,filter_enabled in [('cable_x075_v1',True,False), ('legacy_ppo_six',False,True)]:
        (tmp_path/'training_config.json').write_text(json.dumps(dict(target_node_task=True,
            robot='nero', controller_profile='enhanced6_bounded_dls_v1',
            table_execution_guard_enabled=guard, workspace_scene_family=family,
            table_finger_collision_filter_enabled=filter_enabled)))
        with patch('sys.argv', ['eval', '--model', str(model)]):
            args = evaluation_args()
        assert args.table_execution_guard == guard
        assert args.disable_table_finger_collision_filter == (not filter_enabled)
        assert args.robot == 'nero' and args.controller_profile == 'enhanced6_bounded_dls_v1'
        assert tuple(args.scenario_names) == WORKSPACE_SCENE_FAMILIES[family]
    with patch('sys.argv', ['eval','--model',str(model),'--no-table-execution-guard',
                           '--workspace-scene-family','cable_x075_v1']):
        args = evaluation_args()
    assert not args.table_execution_guard and args.workspace_scene_family == 'cable_x075_v1'
    with patch('sys.argv', ['eval','--model',str(model),'--disable-table-finger-collision-filter']):
        assert evaluation_args().disable_table_finger_collision_filter


def test_saved_workspace_relocation_preserves_both_sector_and_legacy_rectangle(tmp_path):
    from panda_cable_grasp.env.workspace_geometry import DEFAULT_TARGET_WORKSPACE, LEGACY_RECTANGLE_WORKSPACE
    from panda_cable_grasp.rl.workspace import WorkspaceMask
    for original in (DEFAULT_TARGET_WORKSPACE, LEGACY_RECTANGLE_WORKSPACE):
        workspace=WorkspaceMask(original).manifest()
        workspace['path']='/unavailable_server/original_workspace.npz'
        (tmp_path/'training_config.json').write_text(json.dumps(dict(target_node_task=True,
            robot='nero',target_workspace=workspace,workspace_curriculum=False,
            controller_profile='enhanced6_bounded_dls_v1',table_execution_guard_enabled=True)))
        with patch('sys.argv',['eval','--model',str(tmp_path/'checkpoint.zip')]):
            args=evaluation_args()
        assert args.target_workspace==original
        assert WorkspaceMask(args.target_workspace).sha256==workspace['sha256']
