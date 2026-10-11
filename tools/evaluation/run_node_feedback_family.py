"""Fast parallel node-feedback checks; no rendering or shadow simulation."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools/recording')]
from shape_selection_pilot import GeometrySelector, BANK, load, save, sha
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.control import configure_profile
from panda_cable_grasp.policies.scripted import PolicyConfig
from panda_cable_grasp.scenarios.registry import SCENARIO_FAMILIES, get_scenario


def now():
    return datetime.now(timezone.utc)


class NodeOnlyEnv:
    forbidden = {'finger_contacts', 'finger_normal_forces', 'grasp_state',
                 'grasp_confirmed', 'last_grasp_break', 'success_hold',
                 'success_snapshot', 'ever_success', 'success_now', 'info'}

    def __init__(self, env):
        self.base = env

    def __getattr__(self, name):
        if name in self.forbidden:
            raise AssertionError('Forbidden policy input: '+name)
        return getattr(self.base, name)


def prepare(out, smoke=False):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        return
    samples = load(ROOT/'outputs/shape_selection_rotation_20261010/manifest.json')['samples']
    scenes = list(SCENARIO_FAMILIES['cable_x075_v1'])
    jobs = [dict(scene=scene, sample=sample) for scene in scenes for sample in samples]
    if smoke:
        jobs = [jobs[20], jobs[40]]  # shape change and screened complex static
    sources = [Path(__file__), ROOT/'tools/evaluation/shape_selection_pilot.py',
               ROOT/'src/panda_cable_grasp/policies/scripted.py',
               ROOT/'src/panda_cable_grasp/env/environment.py',
               ROOT/'src/panda_cable_grasp/env/control.py',
               ROOT/'src/panda_cable_grasp/env/table_execution_guard.py',
               ROOT/'src/panda_cable_grasp/env/workspace_geometry.py',
               ROOT/'src/panda_cable_grasp/scenarios/registry.py']
    for source in sources:
        target = out/'source'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    manifest = dict(jobs=jobs, scenes=scenes, seeds=[s['seed'] for s in samples],
        planned_episodes=len(jobs), bank=str(BANK),
        bank_sha256=sha(BANK/'replay_src_shape_nominal_v1.npz'),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        policy_config=asdict(PolicyConfig()), controller='enhanced6_bounded_dls_v1',
        table_guard=True, arm_wait_s=.82, episode_seconds=15, smoke=smoke,
        selection='GeometrySelector: ordered node centerline, known cable radius, known robot jaw geometry; no cable collision-body pose',
        complex_snapshots='Previous ten qualified samples; penetration <=5mm and displacement <=15mm from 0.2 to 3s. Same snapshots for complex static and complex initial shape + simple rigid motion.',
        policy_inputs='ordered cable node positions/velocities, robot state, known dimensions/workspace; contact/grasp/success attributes blocked at runtime',
        recording=False)
    save(out/'manifest.json', manifest)


def worker(out, index):
    manifest = load(out/'manifest.json')
    job = manifest['jobs'][index]
    scene, sample = get_scenario(job['scene']), job['sample']
    dest = out/'episodes'/scene.name/f"seed_{sample['seed']}"
    dest.mkdir(parents=True, exist_ok=True)
    overrides = scene.to_env_overrides()
    if 'midshape' in scene.name:
        overrides['initial_shape_time_range'] = (sample['snapshot_fraction'],)*2
    cfg = configure_profile(EnvConfig(robot='nero', seed=sample['seed'],
        target_selection='middle', episode_seconds=15, scenario_name=scene.name,
        scenario_id=scene.scenario_id, scenario_split=scene.split.value,
        replay_bank_dir=str(BANK), replay_seed_fallback='error', **overrides),
        'enhanced6_bounded_dls_v1')
    cfg.table_execution_guard_enabled = True
    env = CableGraspEnv(cfg)
    started = time.monotonic()
    try:
        env.reset(seed=sample['seed'])
        if 'midshape' in scene.name:
            assert env._initial_shape_step == sample['snapshot_step']
        policy = GeometrySelector(NodeOnlyEnv(env), PolicyConfig(**manifest['policy_config']))
        save(dest/'ready.json', dict(pid=os.getpid(), ready_at=now().isoformat(),
            scene=scene.name, seed=sample['seed'], initial_goal=policy.selected_target_index,
            source_sha256=manifest['source_sha256'], env_config=asdict(cfg)))
        phase_changes = []
        old_phase = None
        steps = 0
        while not policy.finished:
            phase = policy.phase.name
            if phase != old_phase:
                phase_changes.append(dict(time_s=float(env.data.time), phase=phase,
                    target_index=policy.selected_target_index, segment=policy.locked_segment_index,
                    segment_alpha=policy.locked_segment_alpha,
                    target_height_m=policy._tracked_segment_height(),
                    lift_reference_height_m=policy._lift_reference_height,
                    failure=policy.last_attempt_failure))
                old_phase = phase
            action = policy.action()
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        save(dest/'result.json', dict(scene=scene.name, seed=sample['seed'],
            task_success=bool(env.ever_success), duration_s=float(env.data.time), steps=steps,
            initial_shape_fraction=sample['snapshot_fraction'] if 'midshape' in scene.name else None,
            policy_result=policy.result, retries=policy.retry_count,
            last_attempt_failure=policy.last_attempt_failure,
            termination_reason=env.info().get('termination_reason'),
            policy_privilege_read_guard='passed', phase_changes=phase_changes,
            selection_events=getattr(policy, 'selection_events', []),
            wall_seconds=time.monotonic()-started))
    finally:
        env.close()


def coordinate(args):
    jobs = load(args.output/'manifest.json')['jobs']
    pending = list(range(len(jobs)))
    active = []
    handles = []
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                index = pending.pop(0)
                log = (args.output/f'worker_{index}.log').open('wb')
                handles.append(log)
                child = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', str(Path(__file__)),
                    '--output', str(args.output), '--worker', str(index)], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                active.append((index, child))
            for index, child in list(active):
                if child.poll() is not None:
                    if child.returncode:
                        raise RuntimeError(f'Worker {index} exit {child.returncode}: '+
                            (args.output/f'worker_{index}.log').read_text(encoding='utf-8', errors='replace')[-3000:])
                    active.remove((index, child))
            time.sleep(.3)
        rows = [load(p) for p in args.output.glob('episodes/*/*/result.json')]
        assert len(rows) == len(jobs)
        by_scene = {scene: dict(completed=sum(r['scene']==scene for r in rows),
            successes=sum(r['scene']==scene and r['task_success'] for r in rows))
            for scene in dict.fromkeys(j['scene'] for j in jobs)}
        result = dict(completed=len(rows), successes=sum(r['task_success'] for r in rows),
            by_scene=by_scene, episodes=rows,
            input_guard='All episodes completed without policy reading contact/grasp/success truth')
        save(args.output/'summary.json', result)
        lines = ['# 去除接触／抓取真值的规则测试', '',
            '几何选点＋切线旋转；增强6有界DLS、碰桌保护、0.82秒等待、15秒上限。',
            '闭爪0.8秒后抬升；以抬升开始的同一材料点高度为基准，1秒后升高不足3厘米重试。侧移／保持期间低于该高度阈值也重试。',
            '策略接触、抓取和成功状态读取由运行时代理禁止；环境物理与公共成功判据保留。',
            '线缆几何由有序节点位置和已知半径构造；夹爪几何与工作区为已知模型。没有仿真前推。',
            '复杂初始形状复用之前筛选合格的10个快照，其它场景用相同10个种子。未新增筛选。',
            '本次只计算成功率，未录视频；10回合属于开发检查。', '',
            '| 场景 | 成功数 |', '|---|---|']
        from build_scripted_family_gallery import SCENE_LABELS
        lines += [f"| {SCENE_LABELS.get(scene, scene)} | {s['successes']}/{s['completed']} |"
                  for scene, s in by_scene.items()]
        (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    finally:
        for _, child in active:
            if child.poll() is None:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for handle in handles:
            handle.close()


def supervise(args):
    prepare(args.output, args.smoke)
    manifest = load(args.output/'manifest.json')
    begun = progress = now()
    child = None
    def status(state, **extra):
        save(args.output/'status.json', dict(status=state, stage='evaluation',
            supervisor_pid=os.getpid(), child_pid=child.pid if child else None,
            started_at=begun.isoformat(), heartbeat_at=now().isoformat(),
            last_progress_at=progress.isoformat(), heartbeat_timeout_seconds=180,
            progress_deadline_at=(progress+timedelta(seconds=600)).isoformat(),
            completed=len(list(args.output.glob('episodes/*/*/result.json'))),
            worker_ready=len(list(args.output.glob('episodes/*/*/ready.json'))),
            planned_episodes=len(manifest['jobs']), report=str(args.output/'report.md'),
            summary=str(args.output/'summary.json'), **extra))
    status('starting')
    try:
        log = (args.output/'coordinator.log').open('wb')
        command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__)),
            '--output', str(args.output), '--workers', str(args.workers), '--coordinate']
        if args.fail_probe:
            command = [sys.executable, '-B', '-c', 'raise RuntimeError("intentional supervisor failure probe")']
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        save(args.output/'launch_receipt.json', dict(command=command, commit=manifest['commit'],
            source_sha256=manifest['source_sha256'], supervisor_pid=os.getpid(),
            child_pid=child.pid, started_at=begun.isoformat()))
        last_completed = 0
        while child.poll() is None:
            count = len(list(args.output.glob('episodes/*/*/result.json')))
            if count != last_completed:
                progress, last_completed = now(), count
            status('running' if list(args.output.glob('episodes/*/*/ready.json')) else 'starting')
            if now() > progress+timedelta(seconds=600):
                raise TimeoutError('No completed episode for 600 seconds')
            time.sleep(3)
        log.close()
        if child.returncode or not (args.output/'summary.json').exists():
            raise RuntimeError(f'Coordinator exit {child.returncode}: '+
                (args.output/'coordinator.log').read_text(encoding='utf-8', errors='replace')[-3500:])
        result = load(args.output/'summary.json')
        progress = now()
        status('complete', results={k: result[k] for k in ('completed', 'successes', 'by_scene', 'input_guard')})
    except BaseException:
        if child is not None and child.poll() is None:
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        status('failed', error=traceback.format_exc())
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--worker', type=int)
    p.add_argument('--coordinate', action='store_true')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--fail-probe', action='store_true')
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.worker is not None:
        worker(args.output, args.worker)
    elif args.coordinate:
        coordinate(args)
    elif args.prepare:
        prepare(args.output, args.smoke)
    else:
        supervise(args)


if __name__ == '__main__':
    main()
