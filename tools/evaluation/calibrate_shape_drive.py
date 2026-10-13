"""Measure original continuous shape-drive response; never changes its dynamics."""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def measure(job):
    import numpy as np
    sys.path.insert(0, str(ROOT/'src'))
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.evaluation.range_motion_metrics import measure_range_motion
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.scenarios.registry import get_scenario
    scene = get_scenario(job['scene'])
    config = env_config_for_scenario(scene, seed=job['seed'], episode_seconds=15., robot='nero')
    config.disturbance_strength = job['strength']
    config.motion_frequency_scale = job['frequency']
    config.rigid_motion_nominal_speed = .25
    config.rigid_frequency_scale = 1.
    if scene.initial_shape_bank or scene.replay_bank:
        config.replay_bank_dir = os.environ.get('RANGE_CALIBRATION_BANK_DIR')
    env = CableGraspEnv(config)
    started = time.perf_counter()
    try:
        env.reset(seed=job['seed'])
        times=[]; positions=[]; velocities=[]; contacts=[]
        def capture():
            times.append(float(env.data.time))
            positions.append(env.data.xpos[env.cable_ids].copy())
            velocities.append(np.array([env.body_linear_velocity(int(b)) for b in env.cable_ids]))
            contacts.append(len(env._finger_contact_pairs()))
        capture()
        # Same force application/ready controls/physics path as env.step, no PPO,
        # no robot collision disabling and no extra shape feedback or table gate.
        horizon = .06 if job.get('smoke') else 15.
        while env.data.time < horizon-1e-9:
            env.data.xfrc_applied[:] = 0.
            env._apply_cable_disturbance()
            env.data.ctrl[:] = env.ready_ctrl
            env._physics_step()
            capture()
        if max(contacts):
            raise RuntimeError('Robot contact invalidates no-intervention calibration')
        result = dict(job, scenario_id=scene.scenario_id,
            config_hash=hashlib.sha256(repr(config).encode()).hexdigest(),
            initial_motion_hash=env.motion_profile_hash,
            measured=None if job.get('smoke') else measure_range_motion(np.asarray(times),np.asarray(positions),
                env.cable_mass,np.asarray(velocities)),
            full_episode_seconds=times[-1], robot_contact_samples=sum(contacts),
            wall_seconds=time.perf_counter()-started)
        dest=Path(job['output'])
        write(dest/f"{job['id']}.json", result)
        np.savez_compressed(dest/f"{job['id']}.npz", times=times,positions=positions,
            velocities=velocities,masses=env.cable_mass)
        return result
    finally:
        env.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['supervise','batch'],default='supervise')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--probe-failure',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--wide',action='store_true')
    p.add_argument('--recover-missing',action='store_true')
    p.add_argument('--fast',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    status_path=a.output/'status.json'
    if a.mode=='supervise':
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(a.output),
             '--mode','batch','--workers',str(a.workers)]
        if a.probe_failure: cmd.append('--probe-failure')
        if a.smoke: cmd.append('--smoke')
        if a.wide: cmd.append('--wide')
        if a.recover_missing: cmd.append('--recover-missing')
        if a.fast: cmd.append('--fast')
        with (a.output/'stdout.log').open('w') as out,(a.output/'stderr.log').open('w') as err:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err)
            while child.poll() is None:
                try: status=json.loads(status_path.read_text())
                except (FileNotFoundError,json.JSONDecodeError): status={}
                progress_path=a.output/'progress.json'
                if progress_path.exists():
                    progress=json.loads(progress_path.read_text())
                    status.update(progress)
                status.update(supervisor_pid=os.getpid(),child_pid=child.pid,heartbeat_at=now())
                if not status.get('status'): status['status']='starting'
                write(status_path,status)
                time.sleep(2)
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(exit_code=child.returncode,finished_at=now(),heartbeat_at=now())
            if child.returncode:
                status.update(status='failed',error=(a.output/'stderr.log').read_text()[-2000:])
            write(status_path,status)
        return
    if a.probe_failure:
        raise RuntimeError('intentional calibration supervisor failure probe')
    jobs=[]
    def add(scene,seed,strength,frequency):
        ident=f'{len(jobs):02d}_{scene}_seed{seed}_s{strength:.3f}_f{frequency:.3f}'
        jobs.append(dict(id=ident,scene=scene,seed=seed,strength=strength,
                         frequency=frequency,output=str(a.output)))
    for seed in (20260804,20260805):
        for strength in ((.75,2.) if a.wide else (1.,1.5)):
            for frequency in ((.5,1.5) if a.wide else (7/9,1.)):
                add('dev_shape_nominal_current_x075_v1',seed,strength,frequency)
    for scene in ('dev_rigid_l1_nominal_x075_v1','dev_rigid_l1_midshape_v1_x075_v1'):
        add(scene,20260804,1.5,1.)
    for seed in (20260804,20260805):
        add('dev_combined_l1_nominal_x075_v1',seed,1.5,1.)
    if a.wide:
        jobs=jobs[:8]
    if a.fast:
        jobs=[]
        for seed in (20260804,20260805):
            for strength in (2.,3.):
                add('dev_shape_nominal_current_x075_v1',seed,strength,3.)
    if a.recover_missing:
        prior=Path(os.environ['RANGE_CALIBRATION_PRIOR'])
        jobs=[job for job in jobs if not (prior/f"{job['id']}.json").exists()]
        if not jobs: raise ValueError('No missing calibration jobs; refusing duplicate run')
    if a.smoke:
        jobs=jobs[:1]
        jobs[0]['smoke']=True
    files=['src/panda_cable_grasp/env/environment.py','src/panda_cable_grasp/scenarios/registry.py',
           'src/panda_cable_grasp/evaluation/range_motion_metrics.py',
           'tools/evaluation/calibrate_shape_drive.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    write(a.output/'source_identity.json',dict(
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes=hashes,supervisor_and_worker_path=str(Path(__file__).resolve())))
    write(a.output/'manifest.json',dict(jobs=jobs,source=str(ROOT),source_hashes=hashes,
        command_speed=.25,shape_semantics='original live drive, airborne force continues, no pose hold',
        window=[.82,7.42],seconds=15,workers=a.workers,started_at=now()))
    state=dict(run_identity=a.output.name,status='running',stage='approved_calibration',
        completed=0,planned=len(jobs),worker_ready=True,source_updated_at=now(),
        child_pid=os.getpid(),child_started_at=now(),
        last_progress_at=now(),heartbeat_timeout_seconds=90,
        progress_deadline_at=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=8)).isoformat(),
        recovery_attempts=0,recovery_limit=2)
    write(status_path,state)
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(measure,j) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            result=future.result(); results.append(result)
            state.update(completed=len(results),last_progress_at=now(),source_updated_at=now())
            write(a.output/'progress.json',state)
    write(a.output/'results.json',results)
    state.update(status='complete',completed=len(results),finished_at=now(),source_updated_at=now())
    write(status_path,state)

if __name__=='__main__':
    main()
