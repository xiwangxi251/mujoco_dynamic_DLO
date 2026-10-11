"""Audit initial cable collision geometry without executing a control step."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import mujoco
import numpy as np
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.scenarios import get_scenario


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='dev_static_midshape_v1_x075_v1')
    parser.add_argument('--bank-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold-mm', type=float, default=5.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'status.json'
    bank_path = args.bank_dir / 'replay_src_shape_nominal_v1.npz'
    env = None
    rows = []
    try:
        with np.load(bank_path) as bank:
            seeds = bank['seeds'].astype(int).tolist()
        save(status_path, dict(status='starting', completed=0, planned=len(seeds)))
        scene = get_scenario(args.scene)
        cfg = EnvConfig(robot='nero', seed=seeds[0], replay_bank_dir=str(args.bank_dir.resolve()),
                        replay_seed_fallback='error', **scene.to_env_overrides())
        env = CableGraspEnv(cfg)
        geom_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f'cableG{i}')
                    for i in range(len(env.cable_ids))]
        if any(g < 0 for g in geom_ids):
            raise ValueError('Expected cable collision geometry names are absent')
        if any(int(env.model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE) for g in geom_ids):
            raise ValueError('Expected capsule collision geometry')
        for seed in seeds:
            env.reset(seed=seed)
            if abs(float(env.data.time)) > 1e-12:
                raise ValueError('Reset already advanced simulation time')
            distances = [(float(mujoco.mj_geomDistance(env.model, env.data, geom_ids[i], geom_ids[j], 1., None)), i, j)
                         for i in range(len(geom_ids)) for j in range(i + 3, len(geom_ids))]
            minimum, i, j = min(distances)
            depth = max(0., -minimum)
            rows.append(dict(seed=seed, snapshot_step=int(env._initial_shape_step),
                             max_penetration_mm=depth * 1000., worst_segment_pair=[i, j],
                             removed=bool(depth > args.threshold_mm / 1000.)))
            save(status_path, dict(status='running', completed=len(rows), planned=len(seeds)))
        removed = [r['seed'] for r in rows if r['removed']]
        retained = [r['seed'] for r in rows if not r['removed']]
        result = dict(scene=args.scene, geometry='MuJoCo mj_geomDistance on actual capsule collision geometry',
                      scope='Default scene snapshot for each of the 50 evaluation-bank source seeds; initialization only',
                      ignored_segment_index_gap=2, threshold_mm=args.threshold_mm,
                      source_bank=str(bank_path.resolve()), bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                      initial_shape_time_range=list(cfg.initial_shape_time_range),
                      simulation_steps=0, examined=len(rows), removed_count=len(removed), retained_count=len(retained),
                      removed_seeds=removed, retained_seeds=retained, rows=rows)
        result_path = args.output / 'audit.json'
        save(result_path, result)
        save(status_path, dict(status='complete', completed=len(rows), planned=len(seeds),
                              removed=len(removed), retained=len(retained), report=str(result_path.resolve())))
        print(json.dumps(dict(examined=len(rows), removed=len(removed), retained=len(retained),
                              removed_seeds=removed, report=str(result_path.resolve()))))
    except BaseException:
        save(status_path, dict(status='failed', completed=len(rows), error=traceback.format_exc()))
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
