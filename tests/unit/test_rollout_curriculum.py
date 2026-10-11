import json
from types import SimpleNamespace
import zipfile

import pytest

from panda_cable_grasp.rl.rollout_curriculum import Windows, checkpoint_phase, difficulties, VERSION, RolloutMotionCurriculum, FAST_PROTOCOL
from panda_cable_grasp.rl.ordered_curriculum import GROUPS, MIXES
from panda_cable_grasp.rl.train import cumulative_linear_schedule, resolve_global_schedule


GOOD = dict(success=1, loaded=1, secured=1, attempt=1)
BAD = dict(success=0, loaded=1, secured=1, attempt=1)


def fill(w, scene, n, row=GOOD):
    for _ in range(n):
        w.add(scene, row)


def test_weak_target_cannot_be_pooled_with_strong_target():
    w = Windows(('strong', 'weak'), ())
    fill(w, 'strong', 100)
    fill(w, 'weak', 100, BAD)
    assert not w.decide()['passed']
    assert w.decide() is None


def test_windows_need_fresh_denominators_and_survive_checkpoint():
    w = Windows(('a', 'b'), ())
    fill(w, 'a', 100)
    fill(w, 'b', 99)
    assert w.decide() is None
    fill(w, 'b', 1)
    assert w.decide()['consecutive'] == 1
    assert w.decide() is None
    w = Windows(('a', 'b'), (), json.loads(json.dumps(w.state())))
    fill(w, 'a', 100)
    fill(w, 'b', 100)
    assert w.decide()['promote']


def test_every_replay_scene_blocks_advance_and_failure_resets_streak():
    w = Windows(('target',), ('old_a', 'old_b'))
    fill(w, 'target', 100)
    fill(w, 'old_a', 50)
    fill(w, 'old_b', 50, BAD)
    assert not w.decide()['passed']
    for n, k in [('target', 100), ('old_a', 50), ('old_b', 50)]:
        fill(w, n, k)
    assert w.decide()['consecutive'] == 1
    fill(w, 'target', 100, BAD)
    fill(w, 'old_a', 50)
    fill(w, 'old_b', 50)
    assert w.decide()['consecutive'] == 0


@pytest.mark.parametrize('metric', ['loaded', 'secured'])
def test_success_without_loaded_or_secured_cannot_pass(metric):
    w = Windows(('target',), ())
    fill(w, 'target', 100, dict(GOOD, **{metric: 0}))
    assert not w.decide()['passed']


def test_fast_gate_boundary_and_separate_weak_scene():
    w = Windows(('a', 'b'), ('replay',), protocol=FAST_PROTOCOL)
    boundary = [dict(success=int(i < 40), loaded=int(i < 50), secured=int(i < 40), attempt=1)
                for i in range(100)]
    for row in boundary:
        w.add('a', row)
        w.add('b', row)
    for i in range(19):
        w.add('replay', dict(GOOD, success=int(i < 6)))
    assert w.decide() is None
    w.add('replay', BAD)
    assert w.decide()['promote']
    assert w.decide() is None
    fill(w, 'a', 100)
    fill(w, 'b', 100, BAD)
    fill(w, 'replay', 20)
    assert not w.decide()['passed']


def test_rule_change_keeps_phase_but_clears_promotion_evidence(tmp_path):
    c = RolloutMotionCurriculum(output_dir=tmp_path, initial_phase=0, gate_profile='fast',
        stages=GROUPS, training_mixes=MIXES)
    previous = Windows(GROUPS[1], GROUPS[0])
    fill(previous, GROUPS[1][0], 100)
    fill(previous, GROUPS[0][0], 50)
    model = SimpleNamespace(rollout_gate_state=dict(version=VERSION, phase=1,
        windows=previous.state(), episode_counter=150),
        get_env=lambda: SimpleNamespace(env_method=lambda *a: None),
        logger=SimpleNamespace(record=lambda *a: None))
    c.model = model
    c.num_timesteps = 1000
    c._on_training_start()
    try:
        assert c.phase_index == 1 and c.episode_counter == 150
        assert model.rollout_gate_migration['windows_reset']
        assert all(len(q) == 0 for q in c.windows.queues.values())
        assert model.rollout_gate_state['version'] == FAST_PROTOCOL['version']
    finally:
        c._on_training_end()


def test_replay_uses_mastered_difficulty_and_targets_use_current_difficulty():
    d = difficulties(5)
    assert d[GROUPS[2][0]] == 1
    assert d[GROUPS[3][0]] == pytest.approx(1 / 3)
    assert d[GROUPS[0][0]] == 0


def test_checkpoint_phase_is_available_before_worker_reset(tmp_path):
    p = tmp_path / 'checkpoint.zip'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('data', json.dumps(dict(ordinary_evaluation_phase=0,
            rollout_gate_state=dict(version=VERSION, phase=5))))
    assert checkpoint_phase(p) == 5
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('data', json.dumps(dict(rollout_gate_state=dict(version='unknown', phase=5))))
    with pytest.raises(ValueError, match='version'):
        checkpoint_phase(p)


def test_callback_excludes_old_phase_episode_and_records_current_episode(tmp_path):
    c = RolloutMotionCurriculum(output_dir=tmp_path, initial_phase=1,
        stages=GROUPS, training_mixes=MIXES)
    model = SimpleNamespace(rollout_gate_state={}, get_env=lambda: SimpleNamespace(env_method=lambda *a: None),
        logger=SimpleNamespace(record=lambda *a: None))
    c.model = model
    c.num_timesteps = 10
    c._on_training_start()
    info = dict(scenario_name=GROUPS[1][0], rollout_episode_phase=0, rollout_episode_difficulty=0,
        episode=dict(success=True, loaded_lift=True, ever_grasped=True, lift_attempt=True))
    try:
        c.locals = dict(dones=[True], infos=[info])
        c._on_step()
        assert len(c.windows.queues[GROUPS[1][0]]) == 0
        info = dict(info, rollout_episode_phase=1)
        c.locals = dict(dones=[True], infos=[info])
        c._on_step()
        assert len(model.rollout_gate_state['windows']['queues'][GROUPS[1][0]]) == 1
        assert model.rollout_gate_state['episode_counter'] == 2
    finally:
        c._on_training_end()


def test_global_lr_keeps_original_slope_for_resumed_and_short_runs():
    f = cumulative_linear_schedule(.0003, .00003, 20_000_000, 20_000_000)
    assert f(1 - 1_399_776 / 20_000_000) == pytest.approx(.000281103024)
    prefix = cumulative_linear_schedule(.0003, .00003, 20_000_000, 1_500_000)
    assert prefix(0) == pytest.approx(.00027975)


def test_saved_global_schedule_restores_without_new_horizon_and_rejects_extension():
    saved = dict(total_timesteps=20_000_000, learning_rate_initial=.0003)
    args = SimpleNamespace(schedule_total_timesteps=0, restart_schedules_on_resume=False, timesteps=18_600_224)
    assert resolve_global_schedule(args, saved, 1_399_776) == saved
    args.timesteps += 1
    with pytest.raises(ValueError, match='exceeds'):
        resolve_global_schedule(args, saved, 1_399_776)
    args.timesteps -= 1
    args.schedule_total_timesteps = 30_000_000
    with pytest.raises(ValueError, match='differs'):
        resolve_global_schedule(args, saved, 1_399_776)
