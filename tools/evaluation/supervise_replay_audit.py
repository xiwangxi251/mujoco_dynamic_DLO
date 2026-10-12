"""Durable supervisor for force-replay acceptance, including worker failures."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).parent))
from audit_replay_tracking import save, sha


def now(): return datetime.now(timezone.utc)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bank',type=Path)
    p.add_argument('--seconds',type=float,default=15.)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--physics-only',action='store_true')
    p.add_argument('--failure-probe',action='store_true')
    args=p.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    status_path=out/'status.json'
    if status_path.exists(): p.error('Existing run status; choose a fresh output')
    if not args.failure_probe and args.bank is None: p.error('--bank is required')
    started=now()
    command=[sys.executable,'-B',str(ROOT/'tools/evaluation/audit_replay_tracking.py'),
        '--bank',str(args.bank), '--output',str(out/'run'), '--seconds',str(args.seconds),
        '--workers',str(args.workers), '--versions','force_tape_v5']
    if args.physics_only: command+=['--physics-only']
    if args.seeds: command+=['--seeds',*map(str,args.seeds)]
    if args.failure_probe: command=[sys.executable,'-B','-c','raise RuntimeError("intentional supervisor failure probe")']
    source_paths=[ROOT/'src/panda_cable_grasp/env/environment.py', ROOT/'src/panda_cable_grasp/env/force_replay.py',
        ROOT/'src/panda_cable_grasp/scenarios/registry.py', ROOT/'tools/evaluation/audit_replay_tracking.py']
    status=dict(run_identity=out.name,stage='force_replay_acceptance',status='starting',
        supervisor_pid=os.getpid(), supervisor_started_at=started.isoformat(),child_pid=None,
        started_at=started.isoformat(),heartbeat_at=started.isoformat(),last_progress_at=started.isoformat(),
        heartbeat_timeout_seconds=90, progress_deadline_at=(started+timedelta(minutes=15)).isoformat(),
        completed=0, planned=len(args.seeds) if args.seeds else 50,
        source_directory=str(ROOT), branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source_paths},
        command=command, error=None,exit_code=None,recovery_attempts=0,recovery_limit=2,
        last_recovery_action=None,last_recovery_at=None,
        status_path=str(status_path), summary_path=str(out/'run/summary.json'),
        report_path=str(out/'acceptance.json'), stdout_path=str(out/'stdout.log'),stderr_path=str(out/'stderr.log'),
        acceptance_limits=dict(max_episode_position_rmse_m=.005,max_episode_abs_speed_bias=.05,
            max_episode_velocity_vector_rmse_m_s=.025,max_episode_position_error_m=.02,
            max_compiler_playback_cable_qpos_difference=1e-7,
            max_compiler_playback_cable_qvel_difference=1e-6),
        scope='No robot intervention, material node 20, .82s onward; no grasp success evaluation')
    save(status_path,status)
    environment=os.environ.copy()
    environment['PYTHONPATH']=str(ROOT/'src')
    with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
        try:
            child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stdout,stderr=stderr)
            status['child_pid']=child.pid
            status['child_started_at']=now().isoformat()
            save(status_path,status)
            progress=out/'run/progress.json'
            while child.poll() is None:
                status['heartbeat_at']=now().isoformat()
                if progress.exists():
                    value=json.loads(progress.read_text(encoding='utf-8'))
                    if value['completed']>status['completed']:
                        status.update(completed=value['completed'],last_progress_at=value['last_progress_at'],
                            status='running',progress_deadline_at=(now()+timedelta(minutes=15)).isoformat())
                save(status_path,status)
                time.sleep(5)
            status['exit_code']=child.returncode
            if child.returncode:
                status.update(status='failed',error=(out/'stderr.log').read_text(errors='replace')[-6000:])
            else:
                summary=json.loads((out/'run/summary.json').read_text(encoding='utf-8'))
                episodes=summary['episodes']
                checks={
                    'position_rmse':max(r['position_rmse_m'] for r in episodes),
                    'speed_bias':max(abs(r['speed_relative_error']) for r in episodes),
                    'velocity_rmse':max(r['velocity_vector_rmse_m_s'] for r in episodes),
                    'position_peak':max(r['max_position_error_m'] for r in episodes),
                    'compiler_qpos':max(r['compiler_playback_max_cable_qpos_difference'] for r in episodes),
                    'compiler_qvel':max(r['compiler_playback_max_cable_qvel_difference'] for r in episodes),
                }
                limits=dict(position_rmse=.005,speed_bias=.05,velocity_rmse=.025,position_peak=.02,
                            compiler_qpos=1e-7,compiler_qvel=1e-6)
                accepted=len(episodes)==status['planned'] and all(checks[k]<=limits[k] for k in checks)
                save(out/'acceptance.json',dict(accepted=accepted,completed=len(episodes),maxima=checks,limits=limits))
                status.update(status='complete' if accepted else 'needs_attention',completed=len(episodes),
                    error=None if accepted else 'Tracking acceptance limits exceeded; inspect acceptance.json')
        except Exception as exc:
            status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        status['heartbeat_at']=now().isoformat()
        status['finished_at']=now().isoformat()
        save(status_path,status)


if __name__=='__main__': main()
