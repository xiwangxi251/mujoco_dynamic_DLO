"""Frozen seven-scene evaluation of the material-preserving geometry rule."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools/evaluation'), str(ROOT/'tools/recording')]
import mujoco
import numpy as np
from shape_selection_pilot import GeometrySelector
from run_node_feedback_family import NodeOnlyEnv
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.control import configure_profile
from panda_cable_grasp.policies.scripted import PolicyConfig
from panda_cable_grasp.scenarios.registry import SCENARIO_FAMILIES, get_scenario
from record_scripted_scene_family import project_camera_manifest
from build_scripted_family_gallery import SCENE_LABELS


def now():
    return datetime.now(timezone.utc)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist() if isinstance(v, np.ndarray) else str(v)), encoding='utf-8')
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def environment_config(manifest, job):
    scene = get_scenario(job['scene'])
    cfg = configure_profile(EnvConfig(robot='nero', seed=job['seed'],
        target_selection='middle', episode_seconds=15, scenario_name=scene.name,
        scenario_id=scene.scenario_id, scenario_split=scene.split.value,
        replay_bank_dir=manifest['bank'], replay_seed_fallback='error',
        dynamicvla_cameras_enabled=True, **scene.to_env_overrides()), 'enhanced6_bounded_dls_v1')
    cfg.table_execution_guard_enabled = True
    return cfg


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        manifest = load(out/'manifest.json')
        assert manifest['utility_sha256'] == sha(__file__), 'Runner changed after freeze'
        assert manifest['bank'] == str(args.bank), 'Bank changed'
        assert manifest['smoke'] == args.smoke
        return manifest
    with np.load(args.bank/'replay_src_shape_nominal_v1.npz') as data:
        seeds = [int(x) for x in data['seeds']]
    assert seeds == list(range(20260804, 20260854)), seeds
    if args.smoke:
        seeds = seeds[:1]
    scenes = list(SCENARIO_FAMILIES['cable_x075_v1'])
    assert len(scenes) == 7
    # Interleave scenes so all seven receive startup coverage.
    jobs = [dict(scene=scene, seed=seed) for seed in seeds for scene in scenes]
    sources = list((ROOT/'src').rglob('*.py')) + [Path(__file__),
        ROOT/'tools/evaluation/shape_selection_pilot.py', ROOT/'tools/evaluation/run_node_feedback_family.py',
        ROOT/'tools/recording/record_scripted_scene_family.py', ROOT/'tools/recording/build_scripted_family_gallery.py']
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    for source in sources:
        target = out/'source'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    policy = asdict(PolicyConfig(preserve_selected_material_target=True,
        follow_target_tangent=True, use_contact_feedback=False))
    manifest = dict(run_id=out.name, created_at=now().isoformat(), source_directory=str(ROOT),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        source_sha256=hashes, utility_sha256=sha(__file__), bank=str(args.bank),
        bank_sha256=sha(args.bank/'replay_src_shape_nominal_v1.npz'),
        bank_manifest=load(args.bank/'replay_src_shape_nominal_v1.manifest.json'),
        jobs=jobs, scenes=scenes, seeds=seeds, planned_episodes=len(jobs), smoke=args.smoke,
        policy_config=policy, controller='enhanced6_bounded_dls_v1', table_guard=True,
        arm_wait_s=.82, episode_seconds=15, arm_velocity_limits_enabled=True, arm_velocity_limit_scale=1.,
        selection='GeometrySelector with preserve_selected_material_target=True; tangent following and node-height feedback',
        sampling='All 50 paired bank seeds; canonical scenario 30-70% snapshot sampling; no screening or outcome selection',
        limitations=['Development-scene results, not held-out generalization.',
            'Complex snapshots are unfiltered; distribution differs from the earlier selected ten-snapshot pilot.',
            'Geometry scoring uses conservative jaw mesh boxes, not exact collision/reachability guarantees.'],
        states_saved=True, videos_rendered=False,
        success_definition='Unmodified low-level environment ever_success; policy truth reads blocked',
        scene_definitions={name: dict(scenario_id=get_scenario(name).scenario_id,
            overrides=get_scenario(name).to_env_overrides()) for name in scenes})
    manifest['configuration_hash'] = digest({k: manifest[k] for k in
        ('source_sha256', 'bank_sha256', 'jobs', 'policy_config', 'scene_definitions', 'sampling')})
    save(out/'manifest.json', manifest)
    return manifest


def worker(args):
    manifest = load(args.output/'manifest.json')
    for relative, expected in manifest['source_sha256'].items():
        assert sha(ROOT/relative) == expected, 'Frozen source changed: '+relative
    job = manifest['jobs'][args.worker]
    dest = args.output/'episodes'/job['scene']/f"seed_{job['seed']}"
    dest.mkdir(parents=True, exist_ok=True)
    cfg = environment_config(manifest, job)
    env = CableGraspEnv(cfg)
    started = time.monotonic()
    try:
        env.reset(seed=job['seed'])
        policy = GeometrySelector(NodeOnlyEnv(env), PolicyConfig(**manifest['policy_config']))
        assert policy.config.preserve_selected_material_target
        state_kind = mujoco.mjtState.mjSTATE_INTEGRATION
        states, times, actions, phases, targets, quaternions, cables = [], [], [], [], [], [], []
        phase_changes = []
        def capture():
            state = np.empty(mujoco.mj_stateSize(env.model, state_kind))
            mujoco.mj_getState(env.model, env.data, state, state_kind)
            states.append(state)
            times.append(float(env.data.time))
            phases.append(policy.phase.name)
            if policy.locked_segment_index is None:
                target = env.data.xpos[policy.selected_target_body_id].copy()
            else:
                i, a = policy.locked_segment_index, policy.locked_segment_alpha
                ids = policy.locked_node_ids or env.cable_ids
                target = (1-a)*env.data.xpos[ids[i]]+a*env.data.xpos[ids[i+1]]
            targets.append(target.copy())
            quaternions.append(policy.desired_quat.copy())
            cables.append(env.data.xpos[env.cable_ids].copy())
        initial_info = env.info()
        save(dest/'ready.json', dict(**job, pid=os.getpid(), ready_at=now().isoformat(),
            run_id=manifest['run_id'], configuration_hash=manifest['configuration_hash'],
            env_config=asdict(cfg), effective_policy_config=asdict(policy.config),
            initial_shape_step=initial_info.get('initial_shape_step'),
            initial_shape_source_seed=initial_info.get('initial_shape_source_seed')))
        capture()
        steps, previous_phase = 0, None
        while not policy.finished:
            if policy.phase.name != previous_phase:
                previous_phase = policy.phase.name
                phase_changes.append(dict(time_s=float(env.data.time), phase=previous_phase,
                    target_index=policy.selected_target_index, segment=policy.locked_segment_index,
                    segment_alpha=policy.locked_segment_alpha, failure=policy.last_attempt_failure))
            action = policy.action()
            assert np.all(np.isfinite(action)), 'Non-finite action'
            _, _, terminated, truncated, _ = env.step(action)
            actions.append(np.asarray(action).copy())
            capture()
            steps += 1
            if terminated or truncated:
                break
        trajectory = dest/'trajectory.npz'
        np.savez_compressed(trajectory, states=states, frame_times=times,
            frame_indices=np.arange(len(states)), native_actions=actions,
            phases=phases, tracked_target_xyz=targets, desired_quats=quaternions,
            cable_xyz=cables, cable_body_ids=env.cable_ids, state_kind=int(state_kind))
        assert len(states) == steps+1 and np.all(np.diff(times)>0)
        save(dest/'state_metadata.json', dict(env_config=asdict(cfg), policy_config=asdict(policy.config),
            frames=len(states), state_kind=int(state_kind), cameras=project_camera_manifest(cfg),
            trajectory_sha256=sha(trajectory), configuration_hash=manifest['configuration_hash']))
        final_info = env.info()
        save(dest/'result.json', dict(**job, run_id=manifest['run_id'],
            configuration_hash=manifest['configuration_hash'], task_success=bool(env.ever_success),
            duration_s=float(env.data.time), steps=steps, frames=len(states),
            retries=policy.retry_count, policy_result=policy.result,
            last_attempt_failure=policy.last_attempt_failure,
            task_failure_type=final_info.get('task_failure_type'),
            termination_reason=final_info.get('termination_reason'),
            initial_shape_step=initial_info.get('initial_shape_step'),
            initial_shape_source_seed=initial_info.get('initial_shape_source_seed'),
            policy_privilege_read_guard='passed', phase_changes=phase_changes,
            selection_events=getattr(policy, 'selection_events', []), final_info=final_info,
            trajectory_sha256=sha(trajectory), wall_seconds=time.monotonic()-started))
    finally:
        env.close()


def completed_rows(args, manifest):
    rows = [load(p) for p in args.output.glob('episodes/*/*/result.json')]
    allowed = {(j['scene'], j['seed']) for j in manifest['jobs']}
    identities = [(r['scene'], r['seed']) for r in rows]
    assert len(identities) == len(set(identities))
    assert set(identities) <= allowed
    assert all(r['configuration_hash'] == manifest['configuration_hash'] and
        r['run_id'] == manifest['run_id'] and r['policy_privilege_read_guard'] == 'passed' for r in rows)
    return rows


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
    return [center-radius, center+radius]


def finalize(args, manifest, rows):
    assert len(rows) == len(manifest['jobs'])
    by_scene = {}
    for scene in manifest['scenes']:
        subset = [r for r in rows if r['scene'] == scene]
        k, n = sum(r['task_success'] for r in subset), len(subset)
        by_scene[scene] = dict(completed=n, successes=k, success_rate=k/n,
            wilson95=wilson(k, n), failures=dict(Counter(r['task_failure_type'] or r['termination_reason']
                for r in subset if not r['task_success'])))
    summary = dict(run_id=manifest['run_id'], configuration_hash=manifest['configuration_hash'],
        completed=len(rows), planned=len(manifest['jobs']), successes=sum(r['task_success'] for r in rows),
        by_scene=by_scene, states_verified=True, state_files=len(rows), smoke=manifest['smoke'])
    save(args.output/'summary.json', summary)
    columns = ['scene', 'seed', 'task_success', 'duration_s', 'steps', 'retries',
        'task_failure_type', 'termination_reason', 'initial_shape_step', 'initial_shape_source_seed', 'wall_seconds']
    with (args.output/'episodes.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r:(r['scene'], r['seed'])))
    lines = ['# 材料点保持规则：七场景评测', '',
        '几何选点、保持所选材料点、切线跟随、节点高度反馈；NERO、增强6有界DLS、碰桌保护、0.82秒等待、15秒上限。',
        '完整种子集按原场景取快照，无筛选；不能与此前筛选后的十样本直接作为同分布配对比较。',
        '保存完整状态、动作、时间戳和项目相机配置，未渲染视频。', '',
        '| 初始形状＋整体运动＋形状变化 | 成功数 | 成功率 | Wilson 95% |', '|---|---:|---:|---:|']
    for scene, item in by_scene.items():
        lo, hi = item['wilson95']
        lines.append(f"| {SCENE_LABELS.get(scene, scene)} | {item['successes']}/{item['completed']} | {item['success_rate']:.1%} | {lo:.1%}–{hi:.1%} |")
    lines += ['', f"合计：{summary['successes']}/{len(rows)}（{summary['successes']/len(rows):.1%}）。",
        '', f"源码提交：`{manifest['commit']}`；分支：`{manifest['branch']}`。",
        f"配置身份：`{manifest['configuration_hash']}`。", 'Smoke 与正式结果分目录，不能合并样本。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def supervise(args):
    manifest = prepare(args)
    begun = progress = now()
    active, handles = {}, []
    rows = completed_rows(args, manifest)
    done = {(r['scene'], r['seed']) for r in rows}
    pending = [i for i, j in enumerate(manifest['jobs']) if (j['scene'], j['seed']) not in done]
    last_count = len(rows)
    deadline = begun+timedelta(hours=2)
    def status(state, **extra):
        save(args.output/'status.json', dict(run_id=manifest['run_id'], status=state,
            experiment_state=state, sync_state='local', stage='evaluation',
            source_directory=str(ROOT), source_commit=manifest['commit'],
            configuration_hash=manifest['configuration_hash'], supervisor_pid=os.getpid(),
            started_at=begun.isoformat(), supervisor_start_identity=begun.isoformat(),
            workers=[dict(job_id=i, pid=p.pid, started_at=t) for i, (p,t) in active.items()],
            heartbeat_at=now().isoformat(), source_updated_at=now().isoformat(), synced_at=now().isoformat(),
            last_progress_at=progress.isoformat(), heartbeat_interval_seconds=3,
            heartbeat_timeout_seconds=120, progress_deadline_at=(progress+timedelta(minutes=15)).isoformat(),
            startup_deadline_at=(begun+timedelta(minutes=2)).isoformat(),
            expected_completion_at=deadline.isoformat(), completed=last_count,
            planned_episodes=len(manifest['jobs']),
            worker_ready=len(list(args.output.glob('episodes/*/*/ready.json'))),
            smoke=manifest['smoke'], summary=str(args.output/'summary.json'), report=str(args.output/'report.md'), **extra))
    status('starting')
    try:
        if args.fail_probe:
            probe = subprocess.run([sys.executable, '-B', '-c', 'raise SystemExit(7)'], capture_output=True)
            raise RuntimeError(f'Intentional supervisor probe: child exit {probe.returncode}')
        while pending or active:
            while pending and len(active)<args.workers:
                index = pending.pop(0)
                log = (args.output/f'worker_{index}.log').open('wb')
                handles.append(log)
                command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__)),
                    '--output', str(args.output), '--bank', str(args.bank), '--worker', str(index)]
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                active[index] = child, now().isoformat()
            for index, (child, _) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                if code:
                    log = (args.output/f'worker_{index}.log').read_text(encoding='utf-8', errors='replace')[-4000:]
                    raise RuntimeError(f'Worker {index} pid {child.pid} exit {code}: {log}')
                job = manifest['jobs'][index]
                assert (args.output/'episodes'/job['scene']/f"seed_{job['seed']}"/'result.json').exists(), 'Worker exited without result'
                del active[index]
            rows = completed_rows(args, manifest)
            if len(rows) != last_count:
                last_count, progress = len(rows), now()
            if now()>progress+timedelta(minutes=15):
                raise TimeoutError('No completed episode for 15 minutes')
            if now()>deadline:
                raise TimeoutError('Two-hour evaluation deadline exceeded')
            status('running' if list(args.output.glob('episodes/*/*/ready.json')) else 'starting')
            if pending or active:
                time.sleep(3)
        summary = finalize(args, manifest, rows)
        progress = now()
        status('complete', exit_code=0, results=summary)
    except BaseException:
        error = traceback.format_exc()
        for child, _ in active.values():
            if child.poll() is None:
                child.terminate()
        for child, _ in active.values():
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        status('failed', exit_code=1, error=error)
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--fail-probe', action='store_true')
    args = parser.parse_args()
    args.output, args.bank = args.output.resolve(), args.bank.resolve()
    if args.worker is not None:
        worker(args)
    elif args.prepare:
        prepare(args)
    else:
        supervise(args)


if __name__ == '__main__':
    main()
