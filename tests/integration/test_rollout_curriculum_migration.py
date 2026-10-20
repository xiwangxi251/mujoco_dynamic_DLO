import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from stable_baselines3 import PPO

from panda_cable_grasp.rl import train
from panda_cable_grasp.rl.rollout_curriculum import GROUPS, VERSION, FAST_PROTOCOL, difficulties, Windows, CONSOLIDATION_PROTOCOL


def cli(output, extra=()):
    reference = Path(__file__).parents[2] / 'archive/session_height_ordered_20m_20261011/rollout_only_20261011/reference_config.json'
    bank = os.environ.get('PANDA_CABLE_TEST_REPLAY_BANK_DIR')
    bank_args = ['--replay-bank-dir', bank] if bank else []
    return ['train', '--output', str(output), '--robot', 'nero',
        '--action-mode', 'task_space_vertical_down', '--observation-mode', 'state',
        '--workspace-scene-family', 'cable_x075_v1', '--ordinary-family-curriculum',
        '--rollout-ability-curriculum', '--eval-freq', '0', '--eval-workers', '0',
        '--ordinary-reward-profile', 'session_height_v1', '--rl-config-file', str(reference),
        '--workers', '3', '--n-steps', '2', '--batch-size', '2', '--n-epochs', '1',
        '--torch-threads', '1', '--device', 'cpu', '--timesteps', '14',
        '--schedule-total-timesteps', '20000000', '--learning-rate-initial', '.0003',
        '--learning-rate-final', '.00003', '--entropy-coef-initial', '.01',
        '--entropy-coef-final', '.001', '--seed', '20261010', *bank_args, *extra]


def test_reset_uses_actual_phase_and_mastered_replay_difficulty(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', cli(tmp_path))
    args = train.parse_args()
    args.rollout_initial_phase = 1
    env = train.make_worker(0, args)()
    try:
        _, info = env.reset(seed=1)
        assert info['rollout_episode_phase'] == 1
        env.set_rollout_curriculum(5, (GROUPS[2][0],), (1.,), difficulties(5))
        _, _, _, _, info = env.step(np.zeros(5))
        assert info['rollout_episode_phase'] == 1
        _, info = env.reset(seed=1)
        assert info['rollout_episode_phase'] == 5
        assert info['rollout_episode_difficulty'] == 1
        assert info['scenario_name'] == GROUPS[2][0]
    finally:
        env.close()


def test_native_spawned_train_and_resume_without_eval(tmp_path, monkeypatch):
    def forbidden_eval(*args, **kwargs):
        raise AssertionError('Independent evaluation factory must never be constructed')
    monkeypatch.setattr(train, 'make_eval_worker', forbidden_eval)
    # Plot rendering is separate from the training semantics checked here.
    monkeypatch.setattr(train, 'plot_training_curves', lambda *args, **kwargs: None)
    first = tmp_path / 'first'
    monkeypatch.setattr(sys, 'argv', cli(first))
    train.main()
    cfg = json.loads((first / 'training_config.json').read_text())
    assert cfg['actual_eval_workers'] == 0 and not cfg['independent_evaluation_enabled']
    assert cfg['sampler'] == 'async_fixed_quota'
    assert cfg['ordered_curriculum']['version'] == VERSION
    model = PPO.load(first / 'final_model.zip', device='cpu')
    assert model.num_timesteps == 14
    assert model._n_updates == 3
    w = Windows(GROUPS[1], GROUPS[0])
    for _ in range(50):
        w.add(GROUPS[1][0], dict(success=1, loaded=1, secured=1, attempt=1))
    model.rollout_gate_state = dict(version=VERSION, phase=1, windows=w.state(), episode_counter=50)
    model.ordinary_evaluation_phase = 1
    checkpoint = first / 'resume_input.zip'
    model.save(checkpoint)
    second = tmp_path / 'second'
    # Inherit the saved global horizon; do not require the caller to re-enter it.
    argv = cli(second, ['--resume', str(checkpoint), '--timesteps', '12', '--schedule-total-timesteps', '0'])
    monkeypatch.setattr(sys, 'argv', argv)
    train.main()
    loaded = PPO.load(second / 'final_model.zip', device='cpu')
    assert loaded.num_timesteps == 26 and loaded._n_updates == 5
    assert loaded.rollout_gate_state['phase'] == 1
    assert len(loaded.rollout_gate_state['windows']['queues'][GROUPS[1][0]]) == 50
    assert loaded.training_global_schedule['total_timesteps'] == 20_000_000
    assert loaded.policy.optimizer.param_groups[0]['lr'] == pytest.approx(.0003 - .00027 * 26 / 20_000_000)
    assert loaded.ent_coef == pytest.approx(.01 - .009 * 26 / 20_000_000)
    health = json.loads((second / 'training_health.json').read_text())
    assert health['parameters_finite'] and health['rollout_fps'] > 0
    assert health['phase'] == 1
    assert health['async_rollout']['no_pending_steps']
    assert sum(health['async_rollout']['per_worker_steps']) == health['async_rollout']['transitions']

    third = tmp_path / 'fast'
    monkeypatch.setattr(sys, 'argv', cli(third, ['--resume', str(second / 'final_model.zip'),
        '--timesteps', '6', '--schedule-total-timesteps', '0', '--rollout-gate-profile', 'fast',
        '--sampler', 'sync']))
    train.main()
    fast = PPO.load(third / 'final_model.zip', device='cpu')
    assert fast.num_timesteps == 32 and fast.rollout_gate_state['phase'] == 1
    assert fast.rollout_gate_state['version'] == FAST_PROTOCOL['version']
    assert fast.rollout_gate_migration['windows_reset']
    assert len(fast.rollout_gate_state['windows']['queues'][GROUPS[1][0]]) == 0
    cfg = json.loads((third / 'training_config.json').read_text())
    assert cfg['rollout_gate_profile'] == 'fast' and cfg['actual_eval_workers'] == 0
    assert cfg['sampler'] == 'sync'

    fast.rollout_gate_state=dict(version=FAST_PROTOCOL['version'],phase=13,
        windows=dict(consecutive=18),episode_counter=2000)
    qualified=third/'qualified.zip';fast.save(qualified)
    fourth=tmp_path/'consolidate'
    monkeypatch.setattr(sys,'argv',cli(fourth,['--resume',str(qualified),'--timesteps','6',
        '--schedule-total-timesteps','0','--rollout-gate-profile','consolidate']))
    train.main()
    consolidated=PPO.load(fourth/'final_model.zip',device='cpu')
    assert consolidated.num_timesteps==38
    assert consolidated.rollout_gate_state['phase']==14
    assert consolidated.rollout_gate_state['version']==CONSOLIDATION_PROTOCOL['version']
    cfg=json.loads((fourth/'training_config.json').read_text())
    assert cfg['rollout_initial_phase']==14 and len(cfg['ordered_curriculum']['phases'])==15
    assert cfg['actual_eval_workers']==0
    assert cfg['sampler']=='async_fixed_quota'
    assert len(consolidated.rollout_gate_state['windows']['queues'])==7


def test_rollout_cli_rejects_independent_eval(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', cli(tmp_path, ['--eval-freq', '100000']))
    with pytest.raises(SystemExit):
        train.parse_args()
