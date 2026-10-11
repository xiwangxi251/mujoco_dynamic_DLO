"""Measure initial penetration and motion during the arm's 0.82 s wait."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import mujoco
import numpy as np
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.control import configure_profile
from panda_cable_grasp.scenarios import get_scenario


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'status.json'
    rows = []
    started = time.monotonic()
    env = None
    def status(state, **extra):
        save(status_path, dict(status=state, completed=len(rows), planned=100,
                               pid=os.getpid(), heartbeat_at=datetime.now(timezone.utc).isoformat(), **extra))
    try:
        bank_path = args.bank_dir / 'replay_src_shape_nominal_v1.npz'
        with np.load(bank_path) as bank:
            seeds = bank['seeds'].astype(int).tolist()
        if len(seeds) != 50 or len(set(seeds)) != 50:
            raise ValueError('This 100-snapshot design requires exactly 50 distinct source trajectories')
        scene = get_scenario('dev_static_midshape_v1_x075_v1')
        status('starting')
        csv_path = args.output / 'samples.csv'
        if args.resume and csv_path.exists():
            with csv_path.open(newline='', encoding='utf-8-sig') as source:
                for row in csv.DictReader(source):
                    for key in row:
                        if key in ('penetration_gt_5mm', 'displacement_gt_5mm'):
                            row[key] = row[key] == 'True'
                        elif key in ('seed', 'snapshot_step', 'max_displacement_node', 'robot_cable_contact_control_steps'):
                            row[key] = int(row[key])
                        elif key != 'worst_pair':
                            row[key] = float(row[key])
                    rows.append(row)
        completed = {(r['seed'], r['snapshot_fraction']) for r in rows}
        if len(completed) != len(rows):
            raise ValueError('Duplicate snapshot records during resume')
        with csv_path.open('a' if rows else 'w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])) if rows else None
            for fraction in (.35, .65):
                overrides = scene.to_env_overrides()
                overrides.update(initial_shape_time_range=(fraction, fraction), arm_motion_start_delay=.82)
                cfg = EnvConfig(robot='nero', seed=seeds[0], replay_bank_dir=str(args.bank_dir.resolve()),
                                replay_seed_fallback='error', episode_seconds=2., **overrides)
                cfg = configure_profile(cfg, 'enhanced6_bounded_dls_v1')
                cfg.table_execution_guard_enabled = True
                env = CableGraspEnv(cfg)
                geom_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f'cableG{i}')
                            for i in range(len(env.cable_ids))]
                if any(g < 0 for g in geom_ids):
                    raise ValueError('Cable collision geometries are missing')
                cable_geoms = set(geom_ids)
                robot_geoms = set(map(int, env.geometric_safety_geom_ids))
                pairs = [(i, j) for i in range(len(geom_ids)) for j in range(i + 3, len(geom_ids))]
                dt = float(env.model.opt.timestep * cfg.frame_skip)
                steps = int(round(.82 / dt))
                if abs(steps * dt - .82) > 1e-9:
                    raise ValueError('Control timestep does not divide the wait duration')
                for seed in seeds:
                    if (seed, fraction) in completed:
                        continue
                    env.reset(seed=seed)
                    if abs(float(env.data.time)) > 1e-12:
                        raise ValueError('Initial state is already advanced')
                    distance, i, j = min((float(mujoco.mj_geomDistance(env.model, env.data, geom_ids[i], geom_ids[j], 1., None)), i, j)
                                         for i, j in pairs)
                    penetration_mm = max(0., -distance) * 1000.
                    initial = env.data.xpos[env.cable_ids].copy()
                    initial_hand = env.hand_position.copy()
                    maximum = 0.; maximum_node = -1; maximum_time = 0.; contact_steps = 0
                    hand_displacement = 0.; samples = [(0., initial.copy())]
                    for step in range(steps):
                        _, _, terminated, truncated, _ = env.step(env.ready_ctrl.copy())
                        if terminated or truncated:
                            raise RuntimeError(f'Sample {seed}/{fraction} ended before the wait completed at {env.data.time}')
                        positions = env.data.xpos[env.cable_ids].copy()
                        delta = np.linalg.norm(positions - initial, axis=1)
                        node = int(np.argmax(delta)); amount = float(delta[node])
                        if amount > maximum:
                            maximum, maximum_node, maximum_time = amount, node, float(env.data.time)
                        hand_displacement = max(hand_displacement, float(np.linalg.norm(env.hand_position - initial_hand)))
                        contacts = [(int(c.geom1), int(c.geom2)) for c in env.data.contact[:env.data.ncon] if float(c.dist) <= 0.]
                        contact_steps += int(any((a in cable_geoms and b in robot_geoms) or (b in cable_geoms and a in robot_geoms) for a, b in contacts))
                        samples.append((float(env.data.time), positions))
                    if abs(float(env.data.time) - .82) > 1e-8:
                        raise ValueError('Wait measurement missed t=0.82 s')
                    tail = next(p for t, p in samples if t >= .62 - 1e-8)
                    tail_displacement = max(float(np.max(np.linalg.norm(p - tail, axis=1))) for t, p in samples if t >= .62 - 1e-8)
                    row = dict(seed=seed, snapshot_fraction=fraction, snapshot_step=int(env._initial_shape_step),
                               initial_penetration_mm=penetration_mm, worst_pair=f'{i},{j}',
                               penetration_gt_5mm=bool(penetration_mm > 5.),
                               max_node_displacement_mm=maximum * 1000., displacement_gt_5mm=bool(maximum > .005),
                               max_displacement_node=maximum_node, max_displacement_time_s=maximum_time,
                               final_max_node_displacement_mm=float(np.max(np.linalg.norm(samples[-1][1] - initial, axis=1))) * 1000.,
                               last_200ms_max_node_displacement_mm=tail_displacement * 1000.,
                               robot_cable_contact_control_steps=contact_steps, max_hand_displacement_mm=hand_displacement * 1000.,
                               measured_until_s=float(env.data.time))
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row); stream.flush(); rows.append(row)
                    status('running')
                env.close(); env = None
        kept = [r for r in rows if not r['penetration_gt_5mm']]
        result = dict(scene=scene.name, samples=len(rows), source_trajectories=len(seeds), snapshot_fractions=[.35, .65],
                      source_bank=str(bank_path.resolve()), bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                      initial_penetration_method='mj_geomDistance on actual capsule collision geometries; ignore index gap <=2',
                      displacement_method='max over nodes and 50Hz samples of 3D displacement relative to t=0, through 0.82s inclusive',
                      arm='ready targets and open jaws throughout; enhanced6_bounded_dls_v1 with table guard',
                      thresholds_mm=dict(penetration=5., displacement=5.),
                      penetration_gt_5mm=sum(r['penetration_gt_5mm'] for r in rows),
                      displacement_gt_5mm_all=sum(r['displacement_gt_5mm'] for r in rows),
                      retained_after_penetration_filter=len(kept),
                      displacement_gt_5mm_after_penetration_filter=sum(r['displacement_gt_5mm'] for r in kept),
                      displacement_le_5mm_after_penetration_filter=sum(not r['displacement_gt_5mm'] for r in kept),
                      robot_contact_samples_all=sum(r['robot_cable_contact_control_steps'] > 0 for r in rows),
                      robot_contact_samples_after_filter=sum(r['robot_cable_contact_control_steps'] > 0 for r in kept),
                      median_displacement_mm_all=float(np.median([r['max_node_displacement_mm'] for r in rows])),
                      median_displacement_mm_after_filter=float(np.median([r['max_node_displacement_mm'] for r in kept])) if kept else None,
                      elapsed_seconds=time.monotonic() - started, per_sample_csv=str(csv_path.resolve()),
                      limitation='100 snapshots from 50 source trajectories, not 100 independent source trajectories; motion evaluated at 50Hz')
        report_path = args.output / 'summary.json'
        save(report_path, result)
        status('complete', results=result, report=str(report_path.resolve()))
        print(json.dumps(result, ensure_ascii=False))
    except BaseException:
        status('failed', error=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
