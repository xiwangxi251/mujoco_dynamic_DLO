import json
import numpy as np
from unittest.mock import patch
from panda_cable_grasp.rl.train import parse_args, make_configured_env
from panda_cable_grasp.rl.evaluate import parse_args as evaluation_args
from panda_cable_grasp.rl.workspace import WORKSPACE_SCENE_FAMILIES


def test_target_training_enables_validated_stack_and_new_six_scenes_by_default():
    with patch('sys.argv', ['train', '--target-node-task']):
        args = parse_args()
    assert args.robot == 'nero'
    assert args.action_mode == 'task_space_vertical_down'
    assert args.controller_profile == 'enhanced6_bounded_dls_v1'
    assert args.table_execution_guard
    assert args.target_reward_profile == 'tolerance_progress_v2'
    assert args.workspace_scene_family == 'cable_x075_v1'
    assert tuple(args.training_scenario_names) == tuple(args.eval_scenario_names) == WORKSPACE_SCENE_FAMILIES['cable_x075_v1']


def test_non_target_training_retains_existing_defaults():
    with patch('sys.argv', ['train']):
        args = parse_args()
    assert (args.robot, args.action_mode, args.controller_profile) == ('panda', 'task_space', 'legacy')
    assert not args.table_execution_guard
    assert args.workspace_scene_family is None
    assert args.target_reward_profile == 'legacy'


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
    finally:
        env.close()


def test_explicit_historical_target_settings_remain_available():
    with patch('sys.argv', ['train', '--target-node-task', '--robot', 'panda',
                           '--action-mode', 'task_space', '--controller-profile', 'legacy',
                           '--no-table-execution-guard', '--target-reward-profile', 'legacy',
                           '--workspace-scene-family', 'legacy_ppo_six']):
        args = parse_args()
    assert args.controller_profile == 'legacy' and not args.table_execution_guard
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
    for family,guard in [('cable_x075_v1',True), ('legacy_ppo_six',False)]:
        (tmp_path/'training_config.json').write_text(json.dumps(dict(target_node_task=True,
            robot='nero', controller_profile='enhanced6_bounded_dls_v1',
            table_execution_guard_enabled=guard, workspace_scene_family=family)))
        with patch('sys.argv', ['eval', '--model', str(model)]):
            args = evaluation_args()
        assert args.table_execution_guard == guard
        assert args.robot == 'nero' and args.controller_profile == 'enhanced6_bounded_dls_v1'
        assert tuple(args.scenario_names) == WORKSPACE_SCENE_FAMILIES[family]
    with patch('sys.argv', ['eval','--model',str(model),'--no-table-execution-guard',
                           '--workspace-scene-family','cable_x075_v1']):
        args = evaluation_args()
    assert not args.table_execution_guard and args.workspace_scene_family == 'cable_x075_v1'
