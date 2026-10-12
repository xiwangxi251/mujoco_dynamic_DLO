"""Idempotent startup chain: failure probe, real warm-start smoke, formal C."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from supervise_speed_training import utc, atomic, birth


def read(p):
    return json.loads(p.read_text())


def alive(pid, expected):
    try:
        values = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return values[19] == expected and values[0] != 'Z'
    except FileNotFoundError:
        return False


def preflight(root):
    p = read(root/'plan.json')
    assert p['starting_timesteps'] == 16997280 and p['planned'] == 20000000
    assert hashlib.sha256(Path(p['checkpoint']).read_bytes()).hexdigest() == p['checkpoint_sha256']
    source = Path(p['source'])
    for name, digest in p['source_hashes'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    for name, digest in p['replay_bank_hashes'].items():
        assert hashlib.sha256((Path(p['configuration']['replay_bank_dir'])/name).read_bytes()).hexdigest() == digest, name
    return p


def launch(root, name, workers, budget, fail=False):
    p = preflight(root)
    out = root/'runs'/name
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out/'launch.lock', os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.close(fd)
    command = [p['python'], '-m', 'panda_cable_grasp.rl.train_speed_curriculum', '--plan', str(root/'plan.json'),
               '--output', str(out), '--workers', str(workers), '--additional', str(budget)]
    if fail:
        command = [p['python'], '-c', "import sys;print('intentional C supervisor probe',file=sys.stderr);sys.exit(7)"]
    m = dict(run_id=name, output=str(out), command=command, cwd=p['source'], environment=p['environment'],
             budget=budget, workers=workers, source_commit=p['source_commit'], configuration_sha256=p['configuration_sha256'])
    atomic(out/'manifest.json', m)
    s = dict(run_id=name, status='starting', experiment_state='starting', planned=budget, completed=0,
        model_global_steps=16997280, original151_pretraining_steps=16997280,
        source_commit=p['source_commit'], configuration_sha256=p['configuration_sha256'],
        starting_checkpoint_sha256=p['checkpoint_sha256'], source_updated_at=utc(), heartbeat_at=utc(), last_progress_at=utc(),
        startup_deadline_at=utc(time.time()+900), first_update_deadline_at=utc(time.time()+1800),
        progress_deadline_at=utc(time.time()+1800), source_interval_seconds=10, heartbeat_timeout_seconds=60,
        sync_interval_seconds=120, sync_timeout_seconds=240, recovery_attempts=0, recovery_limit=2,
        last_recovery_action=None, last_recovery_at=None, workers=workers, error=None, exit_code=None,
        result_paths=dict(output=str(out), final_model=str(out/'final_model.zip')))
    atomic(out/'status.json', s)
    with (out/'supervisor.log').open('ab') as log:
        child = subprocess.Popen([p['python'], str(Path(p['source'])/'tools/rl/supervise_speed_training.py'), str(out/'manifest.json')],
            cwd=p['source'], env=p['environment'], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    return dict(supervisor_pid=child.pid, supervisor_birth=birth(child.pid))


def verify(root, name, workers, reload=False):
    p = preflight(root)
    out = root/'runs'/name
    s, cfg, receipt = read(out/'status.json'), read(out/'training_config.json'), read(out/'input_load_receipt.json')
    from panda_cable_grasp.rl.speed_curriculum import verify_workers, SpeedState, SpeedPPO
    audits = read(out/'worker_configuration_audit.json')
    assert len(audits) == workers and s['health']['workers_verified'] == workers
    verify_workers(audits, SpeedState().record())
    for rank, a in enumerate(audits):
        assert a['env_config']['seed'] == cfg['seed']+rank
        assert a['rl_config'] == cfg['rl_config']
    assert receipt['input_steps'] == 16997280 and receipt['optimizer_epochs'] == 1715
    assert receipt['loaded_optimizer'] and receipt['parameters_finite']
    assert receipt['original_phase'] == 14 and receipt['migrated_phase'] == 0
    assert abs(receipt['learning_rate']-7.0825728e-5)<1e-12 and abs(receipt['entropy']-.0023608576)<1e-12
    assert cfg['penalty_ramp_schedule'] == p['configuration']['penalty_ramp_schedule']
    report = dict(passed=True, workers_verified=workers, input_identity_verified=True,
                  initial_phase_migration_verified=True, configuration_sha256=p['configuration_sha256'], verified_at=utc())
    if reload:
        import torch
        import numpy as np
        model = SpeedPPO.load(out/'final_model.zip', device='cpu')
        assert model.num_timesteps == 16997280+4096 and model._n_updates == 1725
        assert model.c_added_steps == 4096 and model.c_speed_state['level_index'] == 0
        assert model.c_original151_history['phase'] == 14
        assert s['completed'] == 4096 and s['health']['normal_training'] and s['health']['parameters_finite']
        assert s['completed_episodes'] > 0
        tensors = list(model.policy.parameters())
        for value in model.policy.optimizer.state.values():
            tensors.extend(v for v in value.values() if torch.is_tensor(v))
        assert all(bool(torch.isfinite(t).all()) for t in tensors)
        action, _ = model.predict(np.zeros(99, dtype=np.float32), deterministic=True)
        assert action.shape == (5,) and np.isfinite(action).all()
        report.update(reload_finite=True, smoke_added_samples=4096, smoke_excluded_from_formal=True,
                      completed_episodes=s['completed_episodes'], optimizer_epochs_added=10)
    atomic(root/(name+'_verification.json'), report)
    return report


def coordinate(root):
    p = preflight(root)
    fd = os.open(root/'coordinator.lock', os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.close(fd)
    state = dict(run_id='formal', status='starting', experiment_state='preparing', planned=20000000, completed=0,
        source_commit=p['source_commit'], configuration_sha256=p['configuration_sha256'],
        coordinator_pid=os.getpid(), coordinator_birth=birth(os.getpid()), workers_verified=False,
        first_update_verified=False, stage='preflight', error=None, recovery_attempts=0, recovery_limit=2)
    def update(**values):
        state.update(values, source_updated_at=utc(), heartbeat_at=utc())
        atomic(root/'coordinator_status.json', state)
    def stage(name, seconds):
        update(stage=name, last_progress_at=utc(), progress_deadline_at=utc(time.time()+seconds))
    def wait(name, predicate):
        while True:
            s = read(root/'runs'/name/'status.json')
            if s['status'] == 'failed':
                raise RuntimeError(name+': '+str(s.get('error')))
            if predicate(s):
                return s
            if time.time() > dt.datetime.fromisoformat(state['progress_deadline_at']).timestamp():
                raise TimeoutError(state['stage'])
            update(stage_completed=s.get('completed', 0))
            time.sleep(10)
    try:
        stage('failure_probe', 120)
        launch(root, 'failure_probe', 2, 1, True)
        while True:
            s = read(root/'runs/failure_probe/status.json')
            if s['status'] == 'failed':
                assert s['exit_code'] == 7
                break
            if time.time() > dt.datetime.fromisoformat(state['progress_deadline_at']).timestamp():
                raise TimeoutError('probe')
            update()
            time.sleep(3)
        update(failure_probe_verified=True)
        stage('awaiting_smoke', 1800)
        launch(root, 'smoke', 2, 4096)
        wait('smoke', lambda s: s['status'] == 'complete')
        stage('verifying_smoke', 180)
        verify(root, 'smoke', 2, True)
        update(smoke_verified=True)
        stage('awaiting_formal_workers', 900)
        launch(root, 'formal', 32, 20000000)
        wait('formal', lambda s: s.get('health', {}).get('worker_ready'))
        verify(root, 'formal', 32)
        update(workers_verified=True)
        stage('awaiting_first_normal_update', 1800)
        s = wait('formal', lambda s: s.get('health', {}).get('normal_training') and s['health'].get('parameters_finite') and s['completed'] > 0)
        for role in ('child', 'supervisor'):
            assert alive(s[role+'_pid'], s[role+'_birth'])
        update(status='complete', experiment_state='running', stage='startup_complete', first_update_verified=True,
               completed=s['completed'], health=s['health'])
    except BaseException:
        update(status='failed', experiment_state='needs_attention', error=traceback.format_exc())
        raise


def main():
    args = argparse.ArgumentParser()
    args.add_argument('mode', choices=('dispatch', 'coordinate', 'verify'))
    args.add_argument('root', type=Path)
    a = args.parse_args()
    p = preflight(a.root)
    if a.mode == 'verify':
        print(json.dumps(verify(a.root, 'formal', 32)))
    elif a.mode == 'coordinate':
        coordinate(a.root)
    else:
        receipt = a.root/'dispatch_receipt.json'
        if receipt.exists():
            print(json.dumps(dict(reused=True, **read(receipt))))
            return
        fd = os.open(a.root/'dispatch.lock', os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
        with (a.root/'coordinator.log').open('ab') as log:
            child = subprocess.Popen([p['python'], str(Path(__file__).resolve()), 'coordinate', str(a.root)],
                cwd=p['source'], env=p['environment'], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        result = dict(pid=child.pid, birth=birth(child.pid), dispatched_at=utc())
        atomic(receipt, result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
