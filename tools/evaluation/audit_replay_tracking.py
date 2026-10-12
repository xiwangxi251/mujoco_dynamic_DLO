"""Compare replay controllers against the same bank with the arm held ready.

Runs are short, paired tracking diagnostics, not grasp-success evaluations.
Output includes source/bank identities and every measured/reference sample.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import mujoco
import numpy as np
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.control import configure_profile
from panda_cable_grasp.env.replay_bank import ContinuousReplayReference
from panda_cable_grasp.scenarios import get_scenario


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def measure(job):
    seed, version, bank, seconds, output, physics_only = job
    scene = get_scenario('dev_rigid_replay_shape_nominal_x075_v1')
    overrides = scene.to_env_overrides()
    overrides['replay_controller_version'] = version
    cfg = configure_profile(EnvConfig(
        robot='nero', seed=seed, episode_seconds=seconds, replay_bank_dir=bank,
        replay_seed_fallback='error', target_selection='middle', **overrides,
    ), 'enhanced6_bounded_dls_v1')
    env = CableGraspEnv(cfg)
    started = time.monotonic()
    try:
        env.reset(seed=seed)
        model = env.model
        probe = mujoco.MjData(model)
        node = env._replay_tracked_index
        body = int(env.cable_ids[node])
        trajectory = env._replay_bank_active.trajectory(env._replay_entry, node)
        reference = ContinuousReplayReference(trajectory, env._replay_dt, env._replay_valid_steps)
        anchor = env._replay_anchor_tracked.copy()
        jac = np.empty((3, model.nv))
        jrot = np.empty_like(jac)
        cable_geoms = set(np.flatnonzero(np.isin(model.geom_bodyid, env.cable_ids)).tolist())
        robot_geoms = set(env.geometric_safety_geom_ids)
        samples = []
        contacts = 0
        state_kind = mujoco.mjtState.mjSTATE_INTEGRATION
        states = []
        shape0 = None
        max_shape_error = 0.
        def capture():
            nonlocal contacts, shape0, max_shape_error
            state = np.empty(mujoco.mj_stateSize(model, state_kind))
            mujoco.mj_getState(model, env.data, state, state_kind)
            states.append(state)
            probe.qpos[:] = env.data.qpos
            probe.qvel[:] = env.data.qvel
            mujoco.mj_kinematics(model, probe)
            mujoco.mj_comPos(model, probe)
            mujoco.mj_jacBody(model, probe, jac, jrot, body)
            sample = reference.sample(env._replay_elapsed)
            xy = probe.xpos[body, :2].copy()
            velocity = (jac @ probe.qvel)[:2]
            target = anchor + sample.position
            xyz = probe.xpos[env.cable_ids]
            pair = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
            if shape0 is None:
                shape0 = pair
            max_shape_error = max(max_shape_error, float(np.abs(pair-shape0).max()))
            mujoco.mj_collision(model, probe)
            for c in probe.contact:
                if c.dist <= 0 and (
                    (int(c.geom1) in cable_geoms and int(c.geom2) in robot_geoms)
                    or (int(c.geom2) in cable_geoms and int(c.geom1) in robot_geoms)
                ):
                    contacts += 1
            samples.append(np.r_[env.data.time, xy, velocity, target, sample.velocity])
        capture()
        termination = None
        for _ in range(int(np.ceil(seconds/(model.opt.timestep*cfg.frame_skip)))):
            if physics_only:
                # The same disturbance + bounded-gravity physics path. Only
                # arm-policy guards and task scoring are omitted. Require a
                # separate same-state comparison to the full env.step path.
                for _ in range(cfg.frame_skip):
                    env.data.xfrc_applied[:] = 0.
                    env._apply_cable_disturbance()
                    env.data.ctrl[:] = env.ready_ctrl
                    env._physics_step()
                terminated = truncated = False
                info = {}
            else:
                _, _, terminated, truncated, info = env.step(env.ready_ctrl)
            capture()
            if terminated or truncated:
                termination = info.get('termination_reason')
                break
        data = np.array(samples)
        assert contacts == 0, f'Robot contacted the cable: seed={seed}, version={version}'
        assert np.isfinite(data).all()
        assert data[-1, 0] >= seconds-1e-8, (seed, termination, data[-1, 0])
        selected = data[:, 0] >= .82-1e-8
        t = data[selected, 0]
        v = data[selected, 3:5]
        vr = data[selected, 7:9]
        e = data[selected, 1:3]-data[selected, 5:7]
        dt = np.diff(t)
        def average(x):
            return float(np.sum((x[:-1]+x[1:])*.5*dt)/dt.sum())
        speed = average(np.linalg.norm(v, axis=1))
        reference_speed = average(np.linalg.norm(vr, axis=1))
        result = dict(seed=seed, version=version, tracked_node=node, frames=len(data),
            actual_speed_m_s=speed, reference_speed_m_s=reference_speed,
            speed_relative_error=(speed-reference_speed)/max(reference_speed, 1e-12),
            velocity_vector_rmse_m_s=average(np.sum((v-vr)**2, axis=1))**.5,
            position_rmse_m=average(np.sum(e**2, axis=1))**.5,
            max_position_error_m=float(np.linalg.norm(e, axis=1).max()),
            max_shape_error_m=max_shape_error, robot_contacts=contacts,
            measured_seconds=float(data[-1, 0]), window_start_s=.82,
            elapsed_wall_s=time.monotonic()-started, env_config=asdict(cfg))
        result['physics_only'] = physics_only
        dest = Path(output)/version/f'seed_{seed}'
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest/'tracking.npz', samples=data, states=states,
            state_kind=int(state_kind),
            columns=np.array(['time_s','x','y','vx','vy','reference_x','reference_y','reference_vx','reference_vy']))
        result['trajectory_sha256'] = sha(dest/'tracking.npz')
        save(dest/'result.json', result)
        return result
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--seconds', type=float, default=3.)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--physics-only', action='store_true', help='Use after a saved-state equivalence check to the full environment path')
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= .82:
        p.error('--seconds must exceed .82')
    if args.workers < 1:
        p.error('--workers must be positive')
    args.output = args.output.resolve()
    if (args.output/'manifest.json').exists():
        p.error('Output already contains a run; choose a fresh directory')
    args.output.mkdir(parents=True, exist_ok=True)
    bank_file = args.bank.resolve()/'replay_src_shape_nominal_v1.npz'
    with np.load(bank_file) as a:
        bank_seeds = [int(s) for s in a['seeds']]
        seeds = bank_seeds if args.seeds is None else args.seeds
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(bank_seeds):
        p.error('Seeds must be distinct members of the selected bank')
    versions = ['legacy_pd_v1', 'continuous_ff_v2']
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_directory=str(ROOT), branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),
            ROOT/'src/panda_cable_grasp/env/environment.py', ROOT/'src/panda_cable_grasp/env/replay_bank.py',
            ROOT/'src/panda_cable_grasp/scenarios/registry.py']},
        bank_file=str(bank_file), bank_sha256=sha(bank_file), seeds=seeds,
        versions=versions, seconds=args.seconds, planned=2*len(seeds),
        physics_only=args.physics_only,
        protocol='Arm ready, no grasp policy; paired seed/node; common C2 reference; 0.82s onward; physical state refreshed on an independent probe')
    save(args.output/'manifest.json', manifest)
    jobs = [(seed, version, str(args.bank.resolve()), args.seconds, str(args.output), args.physics_only) for seed in seeds for version in versions]
    begun = time.monotonic()
    if args.workers == 1:
        results = [measure(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(measure, jobs))
    summaries = {}
    for version in versions:
        group = [r for r in results if r['version'] == version]
        keys = ['actual_speed_m_s','reference_speed_m_s','velocity_vector_rmse_m_s','position_rmse_m','max_position_error_m']
        summary = {key:float(np.mean([r[key] for r in group])) for key in keys}
        summary['pooled_speed_relative_error'] = summary['actual_speed_m_s']/summary['reference_speed_m_s']-1
        summary['max_episode_position_rmse_m'] = max(r['position_rmse_m'] for r in group)
        summary['episodes'] = len(group)
        summaries[version] = summary
    save(args.output/'summary.json', dict(completed=len(results), elapsed_wall_s=time.monotonic()-begun,
        by_version=summaries, episodes=results))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
