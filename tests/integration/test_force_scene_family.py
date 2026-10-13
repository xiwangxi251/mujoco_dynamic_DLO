"""Explicit force-family selection must never redirect saved old-family runs."""
import json
import sys
import zipfile
from types import SimpleNamespace

import pytest

from panda_cable_grasp.rl import train
from panda_cable_grasp.rl.evaluate import parse_args as evaluation_args
from panda_cable_grasp.rl.ordered_curriculum import GROUPS, MIXES, structure_for_family
from panda_cable_grasp.rl.rollout_curriculum import (
    checkpoint_phase, configure_env, difficulties, structure_for, RolloutMotionCurriculum,
)
from panda_cable_grasp.rl.workspace import WORKSPACE_SCENE_FAMILIES, workspace_stages
from panda_cable_grasp.scenarios import get_scenario

FORCE = 'cable_x075_force_v1'
OLD = 'cable_x075_v1'


@pytest.mark.parametrize('target', [False, True])
def test_experimental_surface_family_is_explicit_and_curriculum_uses_its_names(monkeypatch,target):
    family='cable_x075_surface_v1'
    argv=['train','--workspace-scene-family',family]
    if target:argv+=['--target-node-task','--workspace-curriculum']
    else:argv+=['--ordinary-family-curriculum','--rollout-ability-curriculum','--eval-freq','0','--eval-workers','0']
    monkeypatch.setattr(sys,'argv',argv)
    args=train.parse_args()
    assert tuple(args.training_scenario_names)==WORKSPACE_SCENE_FAMILIES[family]
    if target:
        assert args.target_workspace is not None
    else:
        assert {name for group in args.curriculum_stages for name in group}==set(WORKSPACE_SCENE_FAMILIES[family])


def test_target_training_explicit_force_family_keeps_default_workspace(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train', '--target-node-task', '--workspace-scene-family', FORCE,
                                     '--workspace-curriculum'])
    args = train.parse_args()
    assert tuple(args.training_scenario_names) == WORKSPACE_SCENE_FAMILIES[FORCE]
    assert tuple(args.eval_scenario_names) == WORKSPACE_SCENE_FAMILIES[FORCE]
    assert args.target_workspace is not None
    assert workspace_stages(args.training_scenario_names) == workspace_stages(WORKSPACE_SCENE_FAMILIES[OLD])


@pytest.mark.parametrize('ordered,rollout', [(False, False), (True, False), (True, True)])
def test_ordinary_curriculum_uses_only_explicit_force_family(monkeypatch, ordered, rollout):
    argv = ['train', '--workspace-scene-family', FORCE, '--ordinary-family-curriculum']
    if ordered:
        argv += ['--ordered-ability-curriculum', '--eval-episodes', '20', '--eval-seed-pool-size', '50']
    if rollout:
        argv += ['--rollout-ability-curriculum', '--eval-freq', '0', '--eval-workers', '0']
    monkeypatch.setattr(sys, 'argv', argv)
    args = train.parse_args()
    names = set(WORKSPACE_SCENE_FAMILIES[FORCE])
    assert {n for group in args.curriculum_stages for n in group} == names
    assert {n for mix in args.curriculum_training_mixes for n, _ in mix} == names
    if ordered:
        groups, mixes = structure_for_family(FORCE)
        assert args.curriculum_stages == groups
        assert args.curriculum_training_mixes == mixes


def test_rollout_consolidation_and_first_worker_configuration_use_force_names(tmp_path):
    groups, phases, mixes = structure_for('consolidate', FORCE)
    assert set(groups[-1]) == set(WORKSPACE_SCENE_FAMILIES[FORCE])
    assert len(phases) == 15
    assert [p for _, p in mixes[-1]] == [p for _, p in structure_for('consolidate')[2][-1]]
    calls = {}
    env = SimpleNamespace(set_rollout_curriculum=lambda *values: calls.update(values=values))
    configure_env(env, 5, FORCE)
    phase, names, weights, values = calls['values']
    assert phase == 5 and set(names) <= set(WORKSPACE_SCENE_FAMILIES[FORCE])
    assert values == difficulties(5, groups[:6])
    callback = RolloutMotionCurriculum(output_dir=tmp_path, gate_profile='consolidate',
        scene_family=FORCE, stages=groups, training_mixes=mixes, update_every=1,
        required_evaluations=2, aligned_pinch_rate=.6, secured_rate=.6, success_rate=.6)
    callback.phase_index = 5
    callback._new_windows()
    assert set(callback.windows.queues) <= set(WORKSPACE_SCENE_FAMILIES[FORCE])
    assert callback.protocol['terminal_training_mix'] == mixes[-1]
    assert structure_for_family(OLD) == (GROUPS, MIXES)


@pytest.mark.parametrize('family', [OLD, FORCE])
def test_saved_evaluation_keeps_family_and_controller_versions(tmp_path, monkeypatch, family):
    names = WORKSPACE_SCENE_FAMILIES[family]
    (tmp_path/'training_config.json').write_text(json.dumps(dict(
        workspace_scene_family=family, eval_scenario_names=list(names),
        robot='nero', controller_profile='enhanced6_bounded_dls_v1',
        table_execution_guard_enabled=True, table_finger_collision_filter_enabled=False)))
    monkeypatch.setattr(sys, 'argv', ['eval', '--model', str(tmp_path/'policy.zip')])
    args = evaluation_args()
    assert tuple(args.scenario_names) == names
    rigid = get_scenario(names[1])
    replay = get_scenario(names[5])
    assert rigid.rigid_drive_version == ('force_tape_v1' if family == FORCE else 'feedback_v1')
    assert replay.replay_controller_version == ('force_tape_v5' if family == FORCE else 'legacy_pd_v1')


def test_old_rollout_checkpoint_restores_old_family_and_rejects_force_family(tmp_path):
    path = tmp_path/'policy.zip'
    state = dict(version='ordered_rollout_gate_v1', phase=5, windows={})
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('data', json.dumps(dict(rollout_gate_state=state)))
    (tmp_path/'training_config.json').write_text(json.dumps(dict(workspace_scene_family=OLD)))
    assert checkpoint_phase(path, scene_family=OLD) == 5
    with pytest.raises(ValueError, match='different scene family'):
        checkpoint_phase(path, scene_family=FORCE)


def test_force_rollout_checkpoint_pins_family_without_external_manifest(tmp_path):
    path = tmp_path/'policy.zip'
    state = dict(version='ordered_rollout_gate_v1', phase=5, windows={}, scene_family=FORCE)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('data', json.dumps(dict(rollout_gate_state=state)))
    assert checkpoint_phase(path, scene_family=FORCE) == 5
    with pytest.raises(ValueError, match='different scene family'):
        checkpoint_phase(path, scene_family=OLD)
