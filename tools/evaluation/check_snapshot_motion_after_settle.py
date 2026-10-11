"""Recompute existing snapshots through 3 s, relative to post-drop baselines."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import mujoco
import numpy as np
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.control import configure_profile
from panda_cable_grasp.scenarios import get_scenario


def save(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-csv', type=Path, required=True)
    parser.add_argument('--bank-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.samples_csv.open(encoding='utf-8-sig', newline='') as f:
        inputs = list(csv.DictReader(f))[:args.limit]
    rows_path = args.output / 'samples.json'
    rows = json.loads(rows_path.read_text(encoding='utf-8')) if args.resume and rows_path.exists() else []
    completed = {(r['seed'], r['snapshot_fraction']) for r in rows}
    if len(completed) != len(rows):
        raise ValueError('Duplicate completed sample identity')
    started = time.monotonic()
    start_at = datetime.now(timezone.utc).isoformat()
    progress_at = start_at
    env = None
    fraction_now = None
    bank_path = args.bank_dir / 'replay_src_shape_nominal_v1.npz'
    identity = dict(source_csv=str(args.samples_csv.resolve()),
                    csv_sha256=hashlib.sha256(args.samples_csv.read_bytes()).hexdigest(),
                    source_bank=str(bank_path.resolve()),
                    bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                    scene='dev_static_midshape_v1_x075_v1', baselines_s=[.2, .3], end_s=3.,
                    sample_hz=50, control_profile='enhanced6_bounded_dls_v1', table_guard=True,
                    arm='ready targets and open jaws for the entire 3 seconds')
    manifest = args.output / 'manifest.json'
    if args.resume and manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != identity:
            raise ValueError('Resume configuration mismatch')
    save(manifest, identity)

    def status(state, **extra):
        now = datetime.now(timezone.utc)
        save(args.output / 'status.json', dict(
            status=state, run=args.output.name, completed=len(rows), planned=len(inputs),
            pid=os.getpid(), start_at=start_at, heartbeat_at=now.isoformat(),
            last_progress_at=progress_at, heartbeat_timeout_seconds=180,
            progress_deadline_at=(datetime.fromisoformat(progress_at)+timedelta(seconds=240)).isoformat(),
            **extra))
    try:
        status('starting')
        for source in inputs:
            seed = int(source['seed']); fraction = float(source['snapshot_fraction'])
            if (seed, fraction) in completed:
                continue
            if fraction_now != fraction:
                if env is not None:
                    env.close()
                overrides = get_scenario(identity['scene']).to_env_overrides()
                overrides.update(initial_shape_time_range=(fraction, fraction), arm_motion_start_delay=.82)
                cfg = EnvConfig(robot='nero', seed=seed, replay_bank_dir=str(args.bank_dir.resolve()),
                                replay_seed_fallback='error', episode_seconds=3.1, **overrides)
                cfg = configure_profile(cfg, 'enhanced6_bounded_dls_v1')
                cfg.table_execution_guard_enabled = True
                env = CableGraspEnv(cfg)
                geom_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f'cableG{i}')
                            for i in range(len(env.cable_ids))]
                if min(geom_ids) < 0:
                    raise ValueError('Missing collision geometries')
                cable_geoms = set(geom_ids); robot_geoms = set(map(int, env.geometric_safety_geom_ids))
                pairs = [(i,j) for i in range(len(geom_ids)) for j in range(i+3,len(geom_ids))]
                dt = float(env.model.opt.timestep * cfg.frame_skip)
                if abs(dt-.02) > 1e-9:
                    raise ValueError('Expected 50Hz control sampling')
                fraction_now = fraction
            env.reset(seed=seed)
            if int(env._initial_shape_step) != int(source['snapshot_step']):
                raise ValueError('Snapshot frame does not match the original experiment')
            distance,i,j = min((float(mujoco.mj_geomDistance(env.model, env.data, geom_ids[i],geom_ids[j],1.,None)),i,j) for i,j in pairs)
            penetration = max(0., -distance)*1000.
            if abs(penetration-float(source['initial_penetration_mm'])) > 1e-6:
                raise ValueError('Initial penetration does not match the original snapshot')
            xyz = [env.data.xpos[env.cable_ids].copy()]; times = [0.]
            contacts_by_step = [False]; initial_hand = env.hand_position.copy(); hand_motion = 0.
            status('running', current_sample=dict(seed=seed,fraction=fraction,time_s=0.))
            for step in range(150):
                _,_,terminated,truncated,_ = env.step(env.ready_ctrl.copy())
                if terminated or truncated:
                    raise ValueError(f'Premature termination at {env.data.time}')
                xyz.append(env.data.xpos[env.cable_ids].copy()); times.append(float(env.data.time))
                hand_motion = max(hand_motion,float(np.linalg.norm(env.hand_position-initial_hand))*1000.)
                contacts_by_step.append(any(float(c.dist)<=0. and
                    ((int(c.geom1) in cable_geoms and int(c.geom2) in robot_geoms) or
                     (int(c.geom2) in cable_geoms and int(c.geom1) in robot_geoms))
                    for c in env.data.contact[:env.data.ncon]))
                if step % 25 == 24:
                    status('running',current_sample=dict(seed=seed,fraction=fraction,time_s=float(env.data.time)))
            points=np.asarray(xyz); frame_times=np.asarray(times)
            if abs(frame_times[-1]-3.) > 1e-8:
                raise ValueError('Did not reach exactly 3 seconds')
            row=dict(seed=seed,snapshot_fraction=fraction,snapshot_step=int(env._initial_shape_step),
                     initial_penetration_mm=penetration,penetration_gt_5mm=bool(penetration>5.),
                     max_hand_displacement_mm=hand_motion)
            for baseline in (.2,.3):
                begin=int(round(baseline/dt))
                if abs(frame_times[begin]-baseline)>1e-8:
                    raise ValueError('Baseline frame time mismatch')
                delta=(points[begin:]-points[begin])*1000.
                norms=np.linalg.norm(delta,axis=2)
                ti,ni=np.unravel_index(np.argmax(norms),norms.shape)
                prefix=f'from_{int(baseline*1000)}ms'
                row.update({prefix+'_max_mm':float(norms[ti,ni]),prefix+'_gt_5mm':bool(norms[ti,ni]>5.),
                            prefix+'_node':int(ni),prefix+'_time_s':float(frame_times[begin+ti]),
                            prefix+'_dx_mm':float(delta[ti,ni,0]),prefix+'_dy_mm':float(delta[ti,ni,1]),
                            prefix+'_dz_mm':float(delta[ti,ni,2]),
                            prefix+'_max_xy_mm':float(np.max(np.linalg.norm(delta[:,:,:2],axis=2))),
                            prefix+'_final_max_mm':float(np.max(norms[-1])),
                            prefix+'_robot_contact_steps':int(sum(contacts_by_step[begin:]))})
            name=f'seed_{seed}_frame_{row["snapshot_step"]}.npz'
            np.savez_compressed(args.output/name,cable_positions=points,frame_times=frame_times)
            rows.append(row); completed.add((seed,fraction)); save(rows_path,rows)
            progress_at=datetime.now(timezone.utc).isoformat()
            status('running')
            print(json.dumps(dict(completed=len(rows),sample=row)),flush=True)
        kept=[r for r in rows if not r['penetration_gt_5mm']]
        by_baseline={}
        for baseline in (.2,.3):
            key=f'from_{int(baseline*1000)}ms'
            by_baseline[str(baseline)]=dict(
                displacement_gt_5mm_all=sum(r[key+'_gt_5mm'] for r in rows),
                displacement_gt_5mm_after_filter=sum(r[key+'_gt_5mm'] for r in kept),
                displacement_gt_5mm_after_filter_percent=100.*sum(r[key+'_gt_5mm'] for r in kept)/len(kept) if kept else None,
                median_displacement_mm_after_filter=float(np.median([r[key+'_max_mm'] for r in kept])) if kept else None,
                robot_contact_samples_after_filter=sum(r[key+'_robot_contact_steps']>0 for r in kept))
        csv_path=args.output/'samples.csv'
        with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        result=dict(samples=len(rows),initial_penetration_gt_5mm=sum(r['penetration_gt_5mm'] for r in rows),
                    retained_after_filter=len(kept),by_baseline=by_baseline,
                    sampling='Original 100 snapshots: 50 trajectories at fixed fractions 0.35 and 0.65; no resampling',
                    metric='Maximum across nodes and 50Hz frames of 3D distance from the same node at baseline, through 3s',
                    elapsed_seconds=time.monotonic()-started,per_sample_csv=str(csv_path.resolve()),
                    limitation='Absolute 3D displacement includes overall motion; it is not a pure bending metric')
        report=args.output/'summary.json';save(report,result)
        status('complete',results=result,report=str(report.resolve()))
    except BaseException:
        status('failed',error=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()


if __name__=='__main__':
    main()
