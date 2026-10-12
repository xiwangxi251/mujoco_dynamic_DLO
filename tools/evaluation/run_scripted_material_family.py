"""Frozen geometry-rule evaluation with a paired material/nearest lock ablation."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import math
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
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig
from panda_cable_grasp.scenarios.registry import SCENARIO_FAMILIES, get_scenario
from record_scripted_scene_family import project_camera_manifest
from build_scripted_family_gallery import SCENE_LABELS


class ConfigurableGeometrySelector(GeometrySelector):
    """Reuse geometric scoring while respecting the explicit lock treatment."""
    def __init__(self, env, config):
        DynamicCableGraspPolicy.__init__(self, env, config)


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
        assert manifest['target_lock_mode'] == args.target_lock_mode
        assert manifest['scenes'] == (args.scenes or list(SCENARIO_FAMILIES['cable_x075_v1']))
        assert manifest.get('baseline_run') == (str(args.baseline) if args.baseline else None)
        return manifest
    with np.load(args.bank/'replay_src_shape_nominal_v1.npz') as data:
        seeds = [int(x) for x in data['seeds']]
    assert seeds == list(range(20260804, 20260854)), seeds
    if args.smoke:
        seeds = seeds[:1]
    scenes = args.scenes or list(SCENARIO_FAMILIES['cable_x075_v1'])
    assert scenes and len(set(scenes)) == len(scenes)
    assert set(scenes) <= set(SCENARIO_FAMILIES['cable_x075_v1'])
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
    policy = asdict(PolicyConfig(preserve_selected_material_target=args.target_lock_mode == 'material',
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
        target_lock_mode=args.target_lock_mode, baseline_run=str(args.baseline) if args.baseline else None,
        selection=f'GeometrySelector with preserve_selected_material_target={policy["preserve_selected_material_target"]}; tangent following and node-height feedback',
        sampling='All 50 paired bank seeds; canonical scenario 30-70% snapshot sampling; no screening or outcome selection',
        limitations=['Development-scene results, not held-out generalization.',
            'Complex snapshots are unfiltered; distribution differs from the earlier selected ten-snapshot pilot.',
            'Geometry scoring uses conservative jaw mesh boxes, not exact collision/reachability guarantees.'],
        states_saved=True, videos_rendered=False,
        success_definition='Unmodified low-level environment ever_success; policy truth reads blocked',
        scene_definitions={name: dict(scenario_id=get_scenario(name).scenario_id,
            overrides=get_scenario(name).to_env_overrides()) for name in scenes})
    if args.baseline:
        baseline = load(args.baseline/'manifest.json')
        assert baseline['bank_sha256'] == manifest['bank_sha256']
        assert set(seeds) <= set(baseline['seeds'])
        baseline_hashes = {p.replace('\\', '/'): h for p, h in baseline['source_sha256'].items()}
        control_sources = {p: h for p, h in hashes.items()
            if p != Path(__file__).relative_to(ROOT).as_posix()}
        assert all(baseline_hashes.get(p) == h for p, h in control_sources.items()), 'Baseline control source differs'
        expected_policy = dict(baseline['policy_config'])
        expected_policy['preserve_selected_material_target'] = policy['preserve_selected_material_target']
        assert digest(policy) == digest(expected_policy), 'More than target lock changed'
        assert all(digest(manifest['scene_definitions'][s]) == digest(baseline['scene_definitions'][s]) for s in scenes)
        manifest['baseline_configuration_hash'] = baseline['configuration_hash']
        manifest['control_sources_identical'] = True
    manifest['configuration_hash'] = digest({k: manifest[k] for k in
        ('source_sha256', 'bank_sha256', 'jobs', 'policy_config', 'scene_definitions', 'sampling', 'target_lock_mode', 'baseline_run')})
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
        policy = ConfigurableGeometrySelector(NodeOnlyEnv(env), PolicyConfig(**manifest['policy_config']))
        assert digest(asdict(policy.config)) == digest(manifest['policy_config'])
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
        initial_state = np.empty(mujoco.mj_stateSize(env.model, state_kind))
        mujoco.mj_getState(env.model, env.data, initial_state, state_kind)
        initial_pair_verified = None
        if manifest.get('baseline_run'):
            baseline_dir = Path(manifest['baseline_run'])/'episodes'/job['scene']/f"seed_{job['seed']}"
            baseline_ready = load(baseline_dir/'ready.json')
            assert digest(baseline_ready['env_config']) == digest(asdict(cfg)), 'Paired environment config differs'
            with np.load(baseline_dir/'trajectory.npz', allow_pickle=False) as baseline_data:
                assert np.array_equal(initial_state, baseline_data['states'][0]), 'Paired initial physics differs'
            initial_pair_verified = True
        save(dest/'ready.json', dict(**job, pid=os.getpid(), ready_at=now().isoformat(),
            run_id=manifest['run_id'], configuration_hash=manifest['configuration_hash'],
            env_config=asdict(cfg), effective_policy_config=asdict(policy.config),
            initial_state_sha256=hashlib.sha256(initial_state.tobytes()).hexdigest(),
            initial_pair_verified=initial_pair_verified,
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
            initial_pair_verified=initial_pair_verified,
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
    if manifest.get('baseline_run'):
        assert all(r['initial_pair_verified'] for r in rows)
        comparisons = {}
        for scene in manifest['scenes']:
            pairs = []
            for row in (r for r in rows if r['scene'] == scene):
                baseline_path = Path(manifest['baseline_run'])/'episodes'/scene/f"seed_{row['seed']}"/'result.json'
                baseline = load(baseline_path)
                assert baseline['scene'] == scene and baseline['seed'] == row['seed']
                assert baseline['configuration_hash'] == manifest['baseline_configuration_hash']
                pairs.append(dict(seed=row['seed'], material_success=baseline['task_success'],
                    nearest_success=row['task_success'], initial_pair_verified=True))
            gained = sum(not p['material_success'] and p['nearest_success'] for p in pairs)
            lost = sum(p['material_success'] and not p['nearest_success'] for p in pairs)
            discordant = gained+lost
            probability = min(1., 2*sum(math.comb(discordant, i) for i in range(min(gained, lost)+1))/2**discordant) if discordant else 1.
            comparisons[scene] = dict(paired_episodes=len(pairs),
                material_successes=sum(p['material_success'] for p in pairs),
                nearest_successes=sum(p['nearest_success'] for p in pairs),
                gained=gained, lost=lost, net=gained-lost, delta_percentage_points=100*(gained-lost)/len(pairs),
                mcnemar_exact_two_sided_p=probability, all_initial_states_identical=True, pairs=pairs)
        summary.update(paired_comparison=comparisons, initial_pair_verified_count=len(rows),
            control_sources_identical=manifest['control_sources_identical'],
            baseline_run=manifest['baseline_run'])
        save(args.output/'comparison.json', comparisons)
    save(args.output/'summary.json', summary)
    columns = ['scene', 'seed', 'task_success', 'duration_s', 'steps', 'retries',
        'task_failure_type', 'termination_reason', 'initial_shape_step', 'initial_shape_source_seed', 'wall_seconds']
    with (args.output/'episodes.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r:(r['scene'], r['seed'])))
    lock_label = '保持所选材料点' if manifest['target_lock_mode'] == 'material' else '恢复夹爪最近线段锁定'
    lines = [f'# 规则评测：{lock_label}', '',
        f'几何选点、{lock_label}、切线跟随、节点高度反馈；NERO、增强6有界DLS、碰桌保护、0.82秒等待、15秒上限。',
        '完整种子集按原场景取快照，无筛选；不能与此前筛选后的十样本直接作为同分布配对比较。',
        '保存完整状态、动作、时间戳和项目相机配置，未渲染视频。', '',
        '| 初始形状＋整体运动＋形状变化 | 成功数 | 成功率 | Wilson 95% |', '|---|---:|---:|---:|']
    for scene, item in by_scene.items():
        lo, hi = item['wilson95']
        lines.append(f"| {SCENE_LABELS.get(scene, scene)} | {item['successes']}/{item['completed']} | {item['success_rate']:.1%} | {lo:.1%}–{hi:.1%} |")
    lines += ['', f"合计：{summary['successes']}/{len(rows)}（{summary['successes']/len(rows):.1%}）。",
        '', f"源码提交：`{manifest['commit']}`；分支：`{manifest['branch']}`。",
        f"配置身份：`{manifest['configuration_hash']}`。", 'Smoke 与正式结果分目录，不能合并样本。']
    if 'paired_comparison' in summary:
        lines += ['', '逐回合初始物理状态完全相同；控制源码、环境配置和种子匹配，仅材料点保持开关不同。', '',
            '| 场景 | 固定材料点 | 最近线段 | 差值 | 新增成功 | 丢失成功 | McNemar双侧精确p |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for scene, item in summary['paired_comparison'].items():
            label = '形变' if 'shape_nominal_current' in scene else '组合运动'
            lines.append(f"| {label} | {item['material_successes']}/{item['paired_episodes']} | {item['nearest_successes']}/{item['paired_episodes']} | {item['delta_percentage_points']:+.1f} pp | {item['gained']} | {item['lost']} | {item['mcnemar_exact_two_sided_p']:.4g} |")
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
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--target-lock-mode', choices=('material', 'nearest'), default='material')
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    args.output, args.bank = args.output.resolve(), args.bank.resolve()
    if args.baseline:
        args.baseline = args.baseline.resolve()
    if args.worker is not None:
        worker(args)
    elif args.prepare:
        prepare(args)
    else:
        try:
            supervise(args)
        except BaseException:
            if not (args.output/'status.json').exists():
                save(args.output/'status.json', dict(run_id=args.output.name, status='failed',
                    experiment_state='failed', stage='preparation', supervisor_pid=os.getpid(),
                    heartbeat_at=now().isoformat(), exit_code=1, error=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
