"""Workspace eligibility, material identity and curriculum integration regressions."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig, GraspState
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
from panda_cable_grasp.rl.train import parse_args, make_configured_env, StrictSuccessEvalCallback
from panda_cable_grasp.rl.evaluate import parse_args as eval_args, _episode_row, EpisodeDiagnostics
from panda_cable_grasp.rl.workspace import (
    WorkspaceMask, NoWorkspaceTarget, WorkspaceCurriculumCallback,
    WORKSPACE_STAGES, WORKSPACE_SCENE_FAMILIES, prepare_workspace_entry, workspace_stages,
)


@pytest.fixture
def mask_path(tmp_path):
    path = tmp_path / 'workspace.npz'
    np.savez(path, xs=np.arange(0., 1.51, .1), ys=np.arange(-1., 1.01, .1),
             interior=np.ones((21, 16), bool))
    return path


def make(mask_path, **kwargs):
    return make_rl_env(robot='nero', action_mode='task_space_vertical_down',
                       scenario_names=['dev_static_x075_v1'], episode_seconds=.04,
                       target_workspace=mask_path, workspace_entry_seconds=0.,
                       rl_config=RLConfig(target_node_task=True), **kwargs)


def test_mask_uses_xy_grid_and_height_band(mask_path):
    mask = WorkspaceMask(mask_path)
    np.testing.assert_array_equal(mask.contains([[.75, 0, .01], [2, 0, .01],
                                               [.75, 0, .11], [.75, 0, -.006]]),
                                  [True, False, False, False])
    assert len(mask.manifest()['sha256']) == 64


def test_legacy_rectangle_includes_exact_corners_and_rejects_outside():
    from panda_cable_grasp.env.workspace_geometry import LEGACY_RECTANGLE_WORKSPACE
    mask=WorkspaceMask(LEGACY_RECTANGLE_WORKSPACE)
    corners=[[x,y,z] for x in (.60,.84) for y in (-.22,.30) for z in (-.005,.10)]
    assert np.all(mask.contains(corners))
    assert not np.any(mask.contains([[.60-1e-6,0,.01],[.84+1e-6,0,.01],
                                     [.70,-.22-1e-6,.01],[.70,.30+1e-6,.01],
                                     [.70,0,-.005-1e-6],[.70,0,.10+1e-6]]))
    assert np.all(mask.mask) and mask.mask.shape==(13,6)


def test_default_sector_includes_curved_boundaries_and_excludes_hole_and_bbox_corners():
    from panda_cable_grasp.rl.workspace import DEFAULT_TARGET_WORKSPACE
    mask=WorkspaceMask(DEFAULT_TARGET_WORKSPACE)
    def point(radius,angle,z=.014):
        a=np.radians(angle)
        return [.35+radius*np.cos(a),radius*np.sin(a),z]
    assert np.all(mask.contains([point(r,a,z) for r in (.28,.56) for a in (-30.,0.,70.) for z in (-.005,.10)]))
    outside=[point(.28-1e-6,0),point(.56+1e-6,0),point(.4,-30.-1e-6),
             point(.4,70.+1e-6),point(.4,0,-.005-1e-6),point(.4,0,.10+1e-6),
             [.35,0,.014],[.9,.5,.014],[.5,-.25,.014]]
    assert not np.any(mask.contains(outside))
    assert np.all(mask.contains([point(.4,35),point(.56,0)]))
    assert mask.manifest()['membership']=='analytic'


@pytest.mark.parametrize('axes,shape', [([0, .1, .3], (2, 3)), ([0, 0], (2, 2)),
                                      ([0, .1], (3, 2))])
def test_invalid_grid_is_rejected(tmp_path, axes, shape):
    path = tmp_path / 'bad.npz'
    np.savez(path, xs=axes, ys=[0, .1], interior=np.ones(shape))
    with pytest.raises(ValueError):
        WorkspaceMask(path)


def test_all_nodes_eligible_and_goal_identity_does_not_follow_spatial_position(mask_path):
    env = make(mask_path)
    try:
        observation, info = env.reset(seed=81, options={'goal_node_index': 0})
        assert observation.shape == (109,) and env.action_space.shape == (5,)
        assert info['workspace_candidate_count'] == len(env.base_env.cable_ids)
        assert info['workspace_goal_current_inside']
        env.data.xpos[env.base_env.cable_ids[0]] = [2., 0., .01]
        assert env.goal.index == 0
        assert not env.goal.info()['workspace_goal_current_inside']
        env.reset(seed=81, options={'goal_node_index': 39})
        assert env.goal.index == 39
    finally:
        env.close()


@pytest.mark.parametrize('index', [True, 1.5, -1, 40])
def test_invalid_explicit_node_is_not_coerced_or_resampled(mask_path, index):
    env = make(mask_path)
    try:
        with pytest.raises(ValueError):
            env.reset(seed=81, options={'goal_node_index': index})
    finally:
        env.close()


def test_requested_and_fallback_seed_are_recorded(mask_path):
    env = make(mask_path, workspace_reset_attempts=2, workspace_start_protocol="physical_entry_v1")
    calls = []

    def entry(base, workspace, maximum_seconds, episode_seconds):
        calls.append(base.episode_seed)
        if len(calls) == 1:
            raise NoWorkspaceTarget('test rejection')
        return prepare_workspace_entry(base, workspace, maximum_seconds, episode_seconds)

    try:
        with patch('panda_cable_grasp.rl.workspace.prepare_workspace_entry', side_effect=entry):
            _, info = env.reset(seed=81)
        assert calls == [81, 100084]
        assert info['requested_seed'] == 81 and info['actual_seed'] == 100084
        assert info['workspace_reset_rejections'] == 1
        assert info['episode_seed'] == info['actual_seed']
    finally:
        env.close()


def test_entry_wait_advances_physics_and_preserves_full_policy_duration(mask_path):
    data = SimpleNamespace(xpos=np.array([[2., 0., .01]]), time=0.)
    base = SimpleNamespace(data=data, cable_ids=[0], config=SimpleNamespace(
        frame_skip=10, episode_seconds=15.), model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)),
        ready_ctrl=np.zeros(8), grasp_confirmed=False)

    def step(action):
        np.testing.assert_array_equal(action, base.ready_ctrl)
        data.time += .02
        if data.time >= .04:
            data.xpos[0, 0] = .75
        return None, 0., False, False, {}

    base.step = Mock(side_effect=step)
    assert prepare_workspace_entry(base, WorkspaceMask(mask_path), .08, 15.) == .04
    assert base.step.call_count == 2
    assert base.config.episode_seconds == 15.04
    data.xpos[0, 0] = 2.
    with pytest.raises(NoWorkspaceTarget):
        prepare_workspace_entry(base, WorkspaceMask(mask_path), 0., 15.)


def test_first_confirmed_grasp_is_latched_at_physics_frequency(mask_path):
    env = make(mask_path)
    try:
        env.reset(seed=81, options={'goal_node_index': 20})
        for index in (35, 20):
            env.base_env.grasp_state = GraspState(
                body_id=env.base_env.cable_ids[index], candidate_time=0., bilateral_confirmed=True,
                last_bilateral_time=0., lost_contact_time=0.)
            env.goal.qualify(True)
        assert env.goal.first_grasp_index == 35
        assert env.goal.first_grasp_correct is False
        assert env.goal.matches()  # Regrasp success remains possible, rewards are unchanged.
    finally:
        env.close()


def attach(callback):
    vec = Mock()
    callback.model = SimpleNamespace(get_env=lambda: vec, logger=Mock())
    return vec


def test_curriculum_two_of_three_and_saved_history(tmp_path):
    names = WORKSPACE_SCENE_FAMILIES['cable_x075_v1']
    path = tmp_path / 'state.json'
    callback = WorkspaceCurriculumCallback(names, path)
    attach(callback)
    passed = dict(first_grasp_correct_rate=.60, success_rate=.40)
    failed = dict(first_grasp_correct_rate=.59, success_rate=.90)
    callback.observe_evaluation(passed)
    callback.observe_evaluation(failed)
    assert callback.stage_index == 0
    resumed = WorkspaceCurriculumCallback(names, path, resume=True)
    attach(resumed)
    resumed.observe_evaluation(passed)
    assert resumed.stage_index == 1 and resumed.history == []
    assert json.loads(path.read_text())['last_evaluation']['stage'] == 0
    assert WorkspaceCurriculumCallback(names, path).stage_index == 0
    for stage, (_, weights, difficulty) in enumerate(workspace_stages(names)):
        resumed.restore_phase(stage)
        assert abs(sum(weights)-1.) < 1e-12
        vec = Mock()
        resumed.configure_evaluation_env(vec)
        vec.env_method.assert_any_call('set_training_scenario_distribution', tuple(names), weights)
        vec.env_method.assert_any_call('set_motion_difficulty', difficulty)


def test_expanded_family_preserves_legacy_curriculum_and_rejects_silent_resume(tmp_path):
    from panda_cable_grasp.scenarios.registry import CABLE_X075_REPLAY_SCENARIO_NAMES
    names = WORKSPACE_SCENE_FAMILIES['cable_x075_v1']
    assert len(WORKSPACE_SCENE_FAMILIES['legacy_ppo_six']) == 6
    assert workspace_stages(names[:6]) == WORKSPACE_STAGES
    stages = workspace_stages(names)
    for old, new in zip(WORKSPACE_STAGES, stages):
        assert new[1][:5] == old[1][:5]
        assert sum(new[1][5:]) == old[1][5]
        assert len(new[1]) == len(names)
        assert new[2] == old[2]
    assert stages[-1][1][-2:] == (.15, .15)
    path = tmp_path / 'state.json'
    path.write_text(json.dumps(dict(scene_names=names[:6], stage=5, history=[])))
    with pytest.raises(ValueError, match='scene family changed'):
        WorkspaceCurriculumCallback(names, path, resume=True)
    path.write_text(json.dumps(dict(scene_names=CABLE_X075_REPLAY_SCENARIO_NAMES, stage=5, history=[])))
    archived = WorkspaceCurriculumCallback(CABLE_X075_REPLAY_SCENARIO_NAMES, path, resume=True)
    assert archived.scene_names[-1] == 'dev_rigid_replay_midshape_v1_x075_v1'
    assert archived.stages == stages
    with pytest.raises(ValueError, match='scene family changed'):
        WorkspaceCurriculumCallback(names, path, resume=True)


def test_formal_training_factory_and_strict_eval_use_workspace(mask_path, tmp_path):
    argv = ['train', '--target-node-task', '--robot', 'nero', '--action-mode',
            'task_space_vertical_down', '--target-workspace', str(mask_path),
            '--workspace-scene-family', 'cable_x075_v1', '--workspace-curriculum',
            '--workspace-entry-seconds', '0', '--output', str(tmp_path / 'run'),
            '--controller-profile', 'enhanced6_bounded_dls_v1', '--table-execution-guard',
            '--episode-seconds', '.04']
    with patch('sys.argv', argv):
        args = parse_args()
    assert args.curriculum_stage_steps == 0
    env = make_configured_env(args, seed=81, scenario_names=args.training_scenario_names)
    vec = DummyVecEnv([lambda: env])
    try:
        model = PPO('MlpPolicy', vec, n_steps=2, batch_size=2, device='cpu')
        callback = StrictSuccessEvalCallback(vec, tmp_path, eval_freq=2, episodes=1,
                                             confirmation_episodes=1, seed=81)
        callback.model = model
        metrics = callback._evaluate(1)
        assert metrics['first_grasp_correct_rate'] == 0.
        record = json.loads((tmp_path / 'workspace_eval_episodes.jsonl').read_text().splitlines()[0])
        assert record['actual_seed'] == record['requested_seed'] == 81
        assert record['scenario_name'] == 'dev_static_x075_v1'
        assert env.base_env.config.controller_profile == 'enhanced6_bounded_dls_v1'
        assert env.base_env.config.table_execution_guard_enabled
    finally:
        vec.close()
    args.resume = tmp_path / 'checkpoint.zip'
    state_path = args.output / 'evaluation/workspace_curriculum.json'
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(dict(stage=3, history=[True],
                                         scene_names=args.training_scenario_names)))
    resumed = make_configured_env(args, seed=81, scenario_names=args.training_scenario_names)
    try:
        assert resumed._motion_difficulty == 1.
        np.testing.assert_allclose(resumed._active_scenario_probabilities, workspace_stages(args.training_scenario_names)[3][1])
    finally:
        resumed.close()


def test_unselected_workspace_keeps_existing_target_sampling():
    env = make_rl_env(robot='nero', action_mode='task_space_vertical_down',
                      scenario_names=['id_static'], rl_config=RLConfig(target_node_task=True))
    try:
        _, info = env.reset(seed=81)
        assert env._workspace is None
        assert 7 <= info['goal_node_index'] <= 32
        assert 'workspace_mask' not in info
    finally:
        env.close()


def test_replay_settings_are_applied_before_constructor_reset(tmp_path):
    def construct(config):
        assert config.replay_bank_dir == str(tmp_path)
        assert config.replay_seed_fallback == 'error'
        return CableGraspEnv(config)

    with patch('panda_cable_grasp.rl.environment.CableGraspEnv', side_effect=construct):
        env = make_rl_env(env_config=EnvConfig(robot='nero', replay_seed_fallback='error'),
                          replay_bank_dir=tmp_path)
    env.close()


def test_evaluation_family_resolves_seven_new_scenes():
    with patch('sys.argv', ['eval', '--model', 'model.zip',
                            '--workspace-scene-family', 'cable_x075_v1']):
        args = eval_args()
    assert tuple(args.scenario_names) == WORKSPACE_SCENE_FAMILIES['cable_x075_v1']
    assert len(args.scenario_names) == 7
    assert args.distribution is None


def test_eval_csv_accepts_recorded_conditioned_seed_but_rejects_silent_mismatch(mask_path):
    env = make(mask_path)
    try:
        _, info = env.reset(seed=81)
        initial = dict(info, requested_seed=80, actual_seed=81, workspace_reset_rejections=1)
        kwargs = dict(episode=1, episode_seed=80, success=False, failure_type='never_pinch',
                      policy_internal_success=False, policy_failure_type='never_pinch',
                      task_success=False, task_failure_type='never_pinch', terminated=False,
                      truncated=True, episode_return=0., steps=2, sim_time=.04, wall_time=.1,
                      initial_info=initial, final_info=initial, diagnostics=EpisodeDiagnostics(),
                      break_events=[])
        row = _episode_row(**kwargs)
        assert row['seed'] == row['requested_seed'] == 80
        assert row['actual_episode_seed'] == 81
        initial.pop('workspace_mask')
        with pytest.raises(RuntimeError, match='seed mismatch'):
            _episode_row(**kwargs)
    finally:
        env.close()


def test_initial_policy_reset_does_not_advance_or_fallback_when_all_nodes_outside(tmp_path):
    path = tmp_path / 'outside.npz'
    np.savez(path, xs=[10., 10.1], ys=[10., 10.1], interior=np.ones((2, 2), bool))
    env = make(path, workspace_reset_attempts=50)
    try:
        with patch('panda_cable_grasp.rl.workspace.prepare_workspace_entry',
                   side_effect=AssertionError('hidden entry simulation')), \
             patch.object(env.base_env, 'step', side_effect=AssertionError('hidden control step')):
            observation, info = env.reset(seed=81, options={'goal_node_index': 39})
        assert observation.shape == (109,) and env.action_space.shape == (5,)
        assert env.data.time == 0.
        assert info['requested_seed'] == info['actual_seed'] == info['episode_seed'] == 81
        assert info['workspace_reset_attempts'] == 1 and info['workspace_reset_rejections'] == 0
        assert info['workspace_prefix_seconds'] == 0.
        assert info['workspace_start_protocol'] == 'policy_from_initial_v2'
        assert info['workspace_candidate_count'] == 40 and env.goal.index == 39
        assert not info['workspace_goal_ever_inside'] and info['workspace_goal_first_entry_seconds'] is None
        np.testing.assert_array_equal(env._episode_initial_cable_z,
                                      env.data.xpos[env.base_env.cable_ids, 2])
        for _ in range(2):
            _, reward, terminated, truncated, info = env.step(np.zeros(5))
            assert np.isfinite(reward)
            assert env.goal.index == 39
        assert truncated and not info['success']
        assert not info['workspace_goal_ever_inside'] and info['workspace_inside_fraction'] == 0.
        assert info['episode_steps'] == 2
    finally:
        env.close()


def test_workspace_entry_is_latched_without_switching_goal(mask_path):
    env = make(mask_path)
    try:
        env.reset(seed=81, options={'goal_node_index': 39})
        env.goal.ever_inside = False
        env.goal.first_entry_seconds = None
        with patch.object(env.goal.workspace, 'contains', return_value=np.array([False])):
            _, _, _, _, outside = env.step(np.zeros(5))
        assert not outside['workspace_goal_ever_inside']
        with patch.object(env.goal.workspace, 'contains', return_value=np.array([True])):
            _, _, _, _, entered = env.step(np.zeros(5))
        assert entered['workspace_goal_ever_inside']
        assert entered['workspace_goal_first_entry_seconds'] == pytest.approx(.04)
        with patch.object(env.goal.workspace, 'contains', return_value=np.array([False])):
            assert not env.goal.observe_workspace_goal()
            assert env.goal.info()['workspace_goal_ever_inside']
        assert env.goal.index == 39
    finally:
        env.close()


def test_initial_start_and_episode_budget_defaults_preserve_explicit_legacy_options():
    with patch('sys.argv', ['train', '--target-node-task']):
        args = parse_args()
    assert args.workspace_start_protocol == 'policy_from_initial_v2' and args.episode_seconds == 15.
    with patch('sys.argv', ['train', '--target-node-task', '--workspace-start-protocol', 'physical_entry_v1']):
        args = parse_args()
    assert args.episode_seconds == 15.
    with patch('sys.argv', ['train', '--target-node-task', '--episode-seconds', '17']):
        assert parse_args().episode_seconds == 17.


def test_saved_evaluation_keeps_start_protocol_and_budget(tmp_path, mask_path):
    model = tmp_path / 'model.zip'
    saved = dict(target_node_task=True, target_workspace=WorkspaceMask(mask_path).manifest(),
                 workspace_scene_family='cable_x075_v1', episode_seconds=15., workspace_entry_seconds=8.)
    for protocol, seconds in [(None, 15.), ('policy_from_initial_v2', 15.), ('policy_from_initial_v2', 23.), ('policy_from_initial_v3', 15.)]:
        cfg = dict(saved, episode_seconds=seconds)
        if protocol is not None:
            cfg['workspace_start_protocol'] = protocol
        (tmp_path / 'training_config.json').write_text(json.dumps(cfg))
        with patch('sys.argv', ['eval', '--model', str(model)]):
            args = eval_args()
        assert args.workspace_start_protocol == (protocol or 'physical_entry_v1')
        assert args.episode_seconds == seconds
    with patch('sys.argv', ['eval', '--model', str(model), '--workspace-start-protocol', 'physical_entry_v1',
                           '--episode-seconds', '15']):
        args = eval_args()
    assert args.workspace_start_protocol == 'physical_entry_v1' and args.episode_seconds == 15.


@pytest.mark.parametrize('protocol,immediate', [
    ('policy_from_initial_v3', True), ('policy_from_initial_v2', False), ('physical_entry_v1', False)])
@pytest.mark.parametrize('scene', ['dev_combined_l1_nominal_x075_v1', 'id_combined_l1_nominal'])
def test_explicit_scene_wait_overrides_initial_protocol(mask_path, protocol, immediate, scene):
    env = make_rl_env(robot='nero', action_mode='task_space_vertical_down',
        scenario_names=[scene], episode_seconds=.04,
        target_workspace=mask_path, workspace_start_protocol=protocol, workspace_entry_seconds=0.,
        rl_config=RLConfig(target_node_task=True))
    try:
        _, info = env.reset(seed=81, options={'goal_node_index': 20})
        assert env.data.time == 0.
        assert env.base_env.config.arm_motion_start_immediate is immediate
        released = immediate and scene == 'id_combined_l1_nominal'
        assert info['arm_motion_delayed'] is (not released)
        assert info['arm_motion_start_time'] == pytest.approx(0. if released else .82)
        requested = env.base_env.ready_ctrl.copy()
        requested[0] += .001
        applied = env.base_env._limit_robot_action(requested)
        if released:
            assert abs(applied[0] - env.base_env.ready_ctrl[0]) > 1e-12
        else:
            np.testing.assert_array_equal(applied[:7], env.base_env.ready_ctrl[:7])
    finally:
        env.close()
