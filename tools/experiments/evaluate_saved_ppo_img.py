"""Pair an image estimator with a frozen state PPO and saved evaluation seeds.

The baseline's source commit, configuration, banks and checkpoint are retained.
Only _sample_cable_state changes. Historical GT results are reused after a
complete GT episode in each scenario reproduces its saved outcome and duration.
The supervisor records failures even when a worker cannot write its status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, birth=stat[19], state=stat[0],
                    command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1])
    except FileNotFoundError:
        return None


def prepare(args):
    out, baseline = args.output, args.baseline
    assert not out.exists(), 'Output already exists; refusing duplicate run'
    manifest = read(baseline / 'manifest.json')
    source = Path(manifest['source'])
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == manifest['source_commit']
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    assert sha(baseline / 'checkpoint.zip') == manifest['checkpoint_sha256']
    summary = read(baseline / 'summary.json')
    assert summary['status'] == 'complete'
    selected = [e for e in summary['episodes'] if e['seed'] in manifest['seeds'][:args.episodes]]
    scenes = manifest['scenarios']
    for scene in scenes:
        es = [e for e in selected if e['scenario_name'] == scene]
        assert len(es) == args.episodes and len({e['seed'] for e in es}) == args.episodes
    out.mkdir(parents=True)
    for name in ['checkpoint.zip', 'training_config.json']:
        shutil.copy2(baseline / name, out / name)
    shutil.copy2(args.img_checkpoint, out / 'img_checkpoint.pt')
    shutil.copy2(__file__, out / 'runner.py')
    archive = out / 'source.tar'
    subprocess.run(['git', '-C', str(source), 'archive', '--format=tar', '-o', str(archive), commit], check=True)
    frozen = out / 'frozen_source'
    frozen.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(frozen)
    archive.unlink()
    write(out / 'baseline_episodes.json', selected)
    config = read(out / 'training_config.json')
    for name, value in manifest['bank_sha256'].items():
        assert sha(Path(manifest['evaluation_bank']) / name) == value
    source_hashes = {str(p.relative_to(frozen)): sha(p) for p in frozen.rglob('*')
                     if p.is_file() and (p.suffix in ('.py', '.xml', '.json', '.npz') or p.name == 'pyproject.toml')}
    new = dict(run_id=out.name, baseline_root=str(baseline), source_commit=commit,
               frozen_source=str(frozen), source_hashes=source_hashes,
               checkpoint_steps=manifest['checkpoint_steps'], checkpoint_sha256=manifest['checkpoint_sha256'],
               img_checkpoint_source=str(args.img_checkpoint), img_checkpoint_sha256=sha(out / 'img_checkpoint.pt'),
               configuration_sha256=sha(out / 'training_config.json'),
               baseline_episodes_sha256=sha(out / 'baseline_episodes.json'),
               bank_sha256=manifest['bank_sha256'], evaluation_bank=manifest['evaluation_bank'],
               script_sha256=sha(out / 'runner.py'), development_commit=args.development_commit,
               scenarios=scenes, seeds=manifest['seeds'][:args.episodes], episodes_per_scene=args.episodes,
               expected_episodes=len(selected), workers=7, deterministic=True,
               observation='img_unet_hist; 14 world positions, frame-difference velocity; 99-D PPO',
               missing_estimate='previous prediction; zeros before first valid estimate; never GT',
               render_resolution=[480, 360], control_hz=50, artificial_latency_seconds=0,
               policy_device='cpu', perception_gpus=[0, 1, 2, 3, 4, 5],
               success_metric='task_success from frozen base environment',
               reward_and_physics='unchanged; GT diagnostic reads do not enter the policy observation',
               recovery_limit=2, planned_wall_budget_seconds=5400,
               invariants=['checkpoint', 'perception checkpoint', 'source', 'configuration', 'banks', 'seeds', 'physics', 'success criterion'])
    write(out / 'manifest.json', new)
    now = time.time()
    write(out / 'status.json', dict(run_id=out.name, status='prepared', experiment_state='prepared',
          sync_state='not_started', source_updated_at=now, heartbeat_at=now,
          last_progress_at=now, heartbeat_timeout_seconds=120, progress_deadline_at=now+600,
          completed_episodes=0, planned_episodes=len(selected), recovery_attempts=0, recovery_limit=2,
          last_recovery_action=None, last_recovery_at=None, error=None, exit_code=None,
          result_path=str(out / 'comparison.json'), manifest_path=str(out / 'manifest.json')))
    print(json.dumps(new), flush=True)


def worker(args):
    out = args.output
    if args.fault:
        raise RuntimeError('Intentional supervisor failure test')
    manifest = read(out / 'manifest.json')
    frozen = Path(manifest['frozen_source'])
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', CUDA_VISIBLE_DEVICES=str(args.worker % 6))
    sys.path.insert(0, str(frozen / 'src'))
    sys.path.insert(0, str(frozen / 'tools/perception'))
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
    from dataclasses import asdict
    from run_closed_loop_img import ImgStack
    from panda_cable_grasp.perception.model_img import UNetSkeleton
    from panda_cable_grasp.perception.closed_loop import PerceptionConfig

    torch.set_num_threads(1)
    cfg = read(out / 'training_config.json')
    for file, value in [('checkpoint.zip', manifest['checkpoint_sha256']), ('img_checkpoint.pt', manifest['img_checkpoint_sha256']),
                        ('training_config.json', manifest['configuration_sha256'])]:
        assert sha(out / file) == value
    assert sha(out / 'runner.py') == manifest['script_sha256']
    scene = manifest['scenarios'][args.worker]
    values = cfg['rl_config']
    assert sha(values['action_workspace_path']) == values['action_workspace_sha256']
    env = make_rl_env(action_mode=cfg['action_mode'], robot=cfg['robot'], seed=manifest['seeds'][0],
            disturbance_strength=cfg['disturbance'], episode_seconds=cfg['episode_seconds'],
            dynamicvla_cameras_enabled=True, scenario_names=(scene,), rl_config=RLConfig(**values),
            geometric_safety_enabled=cfg['geometric_safety_enabled'],
            table_finger_collision_filter_enabled=cfg['table_finger_collision_filter_enabled'],
            controller_profile=cfg['controller_profile'], table_execution_guard_enabled=cfg['table_execution_guard_enabled'],
            replay_bank_dir=cfg['eval_replay_bank_dir'], replay_seed_fallback='error')
    env.base_env.config.contact_read_cache_enabled = cfg['contact_read_cache_enabled']
    env.set_penalty_ramp_progress(manifest['checkpoint_steps'], 48, cfg['penalty_ramp_schedule'])
    env.set_motion_difficulty(0.0 if args.worker < 2 else 1.0)
    assert asdict(env.rl_config) == values
    model = PPO.load(out / 'checkpoint.zip', device='cpu', custom_objects={'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0})
    assert model.num_timesteps == manifest['checkpoint_steps']
    assert model.observation_space.shape == (99,) and model.action_space.shape == (5,)
    original_sample = env._sample_cable_state
    original_reset = env.reset
    baseline = {e['seed']: e for e in read(out / 'baseline_episodes.json') if e['scenario_name'] == scene}
    stage = out / ('smoke' if args.smoke else 'formal')
    seeds = manifest['seeds'][:1] if args.smoke else manifest['seeds']
    # A real, complete GT control checks exact reproduction, not merely reset.
    if args.smoke:
        seed = seeds[0]
        set_random_seed(seed)
        obs, initial = env.reset(seed=seed)
        n, total = 0, 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            n += 1
            total += reward
            if done or truncated:
                break
        expected = baseline[seed]
        assert bool(info['task_success']) == expected['task_success'], (scene, 'GT outcome mismatch')
        assert n == expected['steps'], (scene, n, expected['steps'])
        assert np.isclose(total, expected['episode_return'], atol=1e-5), (scene, total, expected['episode_return'])
        write(stage / f'gt_{args.worker}.json', dict(scene=scene, seed=seed, task_success=bool(info['task_success']),
                steps=n, episode_return=total, exact_saved_baseline_reproduced=True))

    ckpt = torch.load(out / 'img_checkpoint.pt', map_location='cuda', weights_only=False)
    assert ckpt['cin'] == 8, 'Expected img_unet_hist checkpoint with prev-map feedback'
    perception = UNetSkeleton(cin=ckpt['cin'], width=ckpt['width']).to('cuda').eval()
    perception.load_state_dict(ckpt['model'])
    stack = ImgStack(env.base_env, perception, PerceptionConfig(), 'prev')
    state = {}
    def reset(*a, **k):
        stack.reset()
        state.clear()
        state.update(last=(np.zeros((14, 3)), np.zeros((14, 3))), calls=0, missing=0, errs=[], flips=[], latencies=[])
        return original_reset(*a, **k)
    def estimate():
        start = time.monotonic()
        prediction = stack.estimate()
        state['latencies'].append(time.monotonic()-start)
        state['calls'] += 1
        if prediction is None:
            state['missing'] += 1
        else:
            pos, vel = prediction['nodes14'], prediction['vel14']
            assert pos.shape == vel.shape == (14, 3) and np.isfinite(pos).all() and np.isfinite(vel).all()
            state['last'] = pos, vel
            # Diagnostic only: never choose direction, reset state or correct predictions using GT.
            gt, _ = original_sample()
            state['errs'].append(float(np.linalg.norm(pos-gt, axis=1).mean()*1000))
            state['flips'].append(float(np.linalg.norm(pos[::-1]-gt, axis=1).mean()*1000))
        return state['last']
    env.reset = reset
    env._sample_cable_state = estimate
    write(stage / f'ready_{args.worker}.json', dict(at=time.time(), scene=scene, pid=os.getpid(),
            checkpoint_steps=model.num_timesteps, configuration_verified=True, cuda_device=args.worker % 6))
    try:
        for seed in seeds:
            destination = stage / f'scene_{args.worker}' / f'seed_{seed}.json'
            if destination.exists():
                existing = read(destination)
                assert existing['config_hash'] == manifest['configuration_sha256'] and existing['checkpoint_sha256'] == manifest['checkpoint_sha256']
                continue
            set_random_seed(seed)
            obs, initial = env.reset(seed=seed)
            assert obs.shape == (99,) and np.isfinite(obs).all()
            assert env.base_env.config.scenario_name == scene
            for key in ['initial_shape_entry', 'initial_shape_step', 'initial_shape_source_seed', 'replay_entry_seed', 'motion_profile_hash', 'public_success_version']:
                assert initial.get(key) == baseline[seed]['initial_info'].get(key), (scene, seed, key)
            n, total, start = 0, 0.0, time.time()
            while True:
                action, _ = model.predict(obs, deterministic=True)
                assert np.isfinite(action).all()
                obs, reward, done, truncated, info = env.step(action)
                assert np.isfinite(obs).all()
                n += 1
                total += reward
                if n % 25 == 0:
                    write(stage / f'progress_{args.worker}.json', dict(at=time.time(), scene=scene, seed=seed, steps=n))
                if done or truncated or (args.smoke and n >= 3):
                    break
            errors = np.asarray(state['errs'])
            reverse = np.asarray(state['flips'])
            result = dict(scenario_name=scene, scene_index=args.worker, seed=seed, steps=n,
                    task_success=bool(info['task_success']), gt_task_success=baseline[seed]['task_success'],
                    loaded_lift=bool(info.get('loaded_lift', False)), secured=bool(info.get('ever_grasped', False)),
                    episode_return=total, wall_seconds=time.time()-start, checkpoint_steps=model.num_timesteps,
                    checkpoint_sha256=manifest['checkpoint_sha256'], config_hash=manifest['configuration_sha256'],
                    estimate_calls=state['calls'], missing_estimates=state['missing'],
                    strict_ordered_mpne_mm=float(errors.mean()) if len(errors) else None,
                    flip_tolerant_mpne_mm=float(np.minimum(errors, reverse).mean()) if len(errors) else None,
                    estimate_latency_mean_s=float(np.mean(state['latencies'])),
                    termination_reason=info.get('termination_reason'), difficulty=0 if args.worker<2 else 1,
                    height_penalty_scale=env._last_height_penalty_scale, safety_penalty_scale=env._last_safety_penalty_scale,
                    initial_identity={k:initial.get(k) for k in ['scenario_name','episode_seed','initial_shape_entry','initial_shape_step','initial_shape_source_seed','replay_entry_seed','motion_profile_hash']})
            write(destination, result)
            write(stage / f'progress_{args.worker}.json', dict(at=time.time(), scene=scene, seed=seed, completed_episode=True))
    finally:
        stack.renderer.close()
        env.close()


def prepare_remaining(args):
    """Prepare disjoint remaining seeds without changing or launching the first run."""
    out, previous = args.output, args.previous_run
    assert not out.exists(), 'Refusing to overwrite an existing continuation'
    old = read(previous / 'manifest.json')
    assert read(previous / 'smoke_status.json')['status'] == 'smoke_complete'
    assert read(previous / 'fault_status.json')['status'] == 'failed'
    baseline_root = Path(old['baseline_root'])
    baseline_manifest = read(baseline_root / 'manifest.json')
    baseline_summary = read(baseline_root / 'summary.json')
    assert baseline_summary['status'] == 'complete'
    assert old['checkpoint_sha256'] == baseline_manifest['checkpoint_sha256']
    assert old['source_commit'] == baseline_manifest['source_commit']
    seeds = [s for s in baseline_manifest['seeds'] if s not in old['seeds']]
    assert len(seeds) == args.episodes and not set(seeds).intersection(old['seeds'])
    selected = [e for e in baseline_summary['episodes'] if e['seed'] in seeds]
    for scene in old['scenarios']:
        assert len([e for e in selected if e['scenario_name'] == scene]) == args.episodes
    for name, expected in [('checkpoint.zip', old['checkpoint_sha256']),
                           ('img_checkpoint.pt', old['img_checkpoint_sha256']),
                           ('training_config.json', old['configuration_sha256'])]:
        assert sha(previous / name) == expected
    original_source = Path(old['frozen_source'])
    for relative, expected in old['source_hashes'].items():
        assert sha(original_source / relative) == expected
    out.mkdir(parents=True)
    for name in ['checkpoint.zip', 'img_checkpoint.pt', 'training_config.json',
                 'fault_status.json', 'smoke_status.json']:
        shutil.copy2(previous / name, out / name)
    frozen = out / 'frozen_source'
    shutil.copytree(original_source, frozen, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__, out / 'runner.py')
    write(out / 'baseline_episodes.json', selected)
    manifest = dict(old)
    manifest.update(run_id=out.name, frozen_source=str(frozen), seeds=seeds,
                    episodes_per_scene=args.episodes, expected_episodes=len(selected),
                    predecessor_root=str(previous), predecessor_expected_episodes=old['expected_episodes'],
                    script_sha256=sha(out / 'runner.py'), development_commit=args.development_commit,
                    baseline_episodes_sha256=sha(out / 'baseline_episodes.json'),
                    preflight_inherited_from=str(previous),
                    preflight_semantics='Identical frozen worker, model, config, banks and physics; prior seven-scene GT and perception checks reused',
                    planned_wall_budget_seconds=5400)
    write(out / 'manifest.json', manifest)
    now = time.time()
    write(out / 'status.json', dict(run_id=out.name, status='prepared_waiting_predecessor',
          experiment_state='prepared_waiting_predecessor', sync_state='not_started',
          source_updated_at=now, heartbeat_at=now, last_progress_at=now,
          heartbeat_timeout_seconds=120, progress_deadline_at=None,
          completed_episodes=0, planned_episodes=len(selected), recovery_attempts=0,
          recovery_limit=2, last_recovery_action=None, last_recovery_at=None,
          error=None, exit_code=None, result_path=str(out / 'comparison.json'),
          manifest_path=str(out / 'manifest.json'), predecessor_root=str(previous)))
    print(json.dumps(dict(run_id=out.name, status='prepared_waiting_predecessor',
          episodes=len(selected), seeds=seeds, checkpoint_sha256=old['checkpoint_sha256'])), flush=True)


def aggregate(out):
    manifest = read(out / 'manifest.json')
    episodes = [read(p) for p in sorted((out/'formal').glob('scene_*/seed_*.json'))]
    scenes = []
    for scene in manifest['scenarios']:
        es = [e for e in episodes if e['scenario_name'] == scene]
        gt = sum(e['gt_task_success'] for e in es)
        pred = sum(e['task_success'] for e in es)
        scenes.append(dict(scenario_name=scene, completed=len(es), planned=manifest['episodes_per_scene'],
                gt_success=gt, perception_success=pred, delta_success=pred-gt,
                gt_success_rate=gt/len(es) if es else None, perception_success_rate=pred/len(es) if es else None,
                fixed=sum(e['task_success'] and not e['gt_task_success'] for e in es),
                broke=sum(e['gt_task_success'] and not e['task_success'] for e in es)))
    return dict(run_id=manifest['run_id'], completed_episodes=len(episodes), planned_episodes=manifest['expected_episodes'],
                scenes=scenes, episodes=episodes)


def supervise(args):
    out = args.output
    status_path = out / ('fault_status.json' if args.fault else 'smoke_status.json' if args.smoke else 'status.json')
    status = read(out / 'status.json')
    now = time.time()
    status.update(status='starting', experiment_state='starting', supervisor_identity=identity(os.getpid()),
                  source_updated_at=now, heartbeat_at=now, last_progress_at=now, progress_deadline_at=now+600,
                  startup_deadline_at=now+600, wall_deadline_at=now+5400, workers=[])
    write(status_path, status)
    jobs = []
    stage = out / ('smoke' if args.smoke else 'formal')
    stage.mkdir(exist_ok=True)
    try:
        for index in (range(1) if args.fault else range(7)):
            command = [sys.executable, str(out/'runner.py'), '--mode', 'worker', '--output', str(out), '--worker', str(index)]
            if args.smoke: command.append('--smoke')
            if args.fault: command.append('--fault')
            with (stage/f'worker_{index}{"_fault" if args.fault else ""}.log').open('w') as log:
                process = subprocess.Popen(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            jobs.append((index, process))
            status['workers'].append(identity(process.pid))
        while jobs:
            for index, process in list(jobs):
                code = process.poll()
                if code is not None:
                    if code:
                        status['exit_code'] = code
                        log = stage/f'worker_{index}{"_fault" if args.fault else ""}.log'
                        raise RuntimeError(f'worker {index} exited {code}: '+log.read_text()[-2500:])
                    jobs.remove((index, process))
            now = time.time()
            files = list(stage.glob('progress_*.json')) + list(stage.glob('ready_*.json')) + list(stage.glob('gt_*.json'))
            progress = max([status['last_progress_at']]+[p.stat().st_mtime for p in files])
            result = aggregate(out)
            status.update(source_updated_at=now, heartbeat_at=now, last_progress_at=progress,
                    progress_deadline_at=progress+600, completed_episodes=result['completed_episodes'], scenes=result['scenes'])
            if len(list(stage.glob('ready_*.json'))) == 7:
                status.update(status='running', experiment_state='running')
            write(status_path, status)
            if now > status['progress_deadline_at'] or now > status['wall_deadline_at']:
                raise RuntimeError('Evaluation exceeded progress or wall deadline')
            if jobs: time.sleep(5)
        if args.smoke:
            assert len(list(stage.glob('gt_*.json'))) == 7
            assert len(list(stage.glob('scene_*/seed_*.json'))) == 7
        else:
            result = aggregate(out)
            assert result['completed_episodes'] == result['planned_episodes']
            assert all(s['completed']==s['planned'] for s in result['scenes'])
            result['status'] = 'complete'
            write(out/'comparison.json', result)
        status.update(status='smoke_complete' if args.smoke else 'complete', experiment_state='complete',
                      completed_at=time.time(), source_updated_at=time.time(), heartbeat_at=time.time())
    except Exception:
        status.update(status='failed', experiment_state='failed', error=traceback.format_exc(),
                      source_updated_at=time.time(), heartbeat_at=time.time())
        for _, process in jobs:
            if process.poll() is None: process.terminate()
        for _, process in jobs:
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill()
    finally:
        write(status_path, status)
    if status['status']=='failed':
        print(status['error'], flush=True)
        if not args.fault: raise SystemExit(1)


def launch(args):
    out = args.output
    manifest = read(out / 'manifest.json')
    if manifest.get('predecessor_root'):
        previous = Path(manifest['predecessor_root'])
        assert read(previous / 'status.json')['status'] == 'complete', 'Previous batch not complete'
        result = read(previous / 'comparison.json')
        assert result['status'] == 'complete'
        assert result['completed_episodes'] == manifest['predecessor_expected_episodes']
        assert all(s['completed'] == s['planned'] for s in result['scenes'])
        old = read(previous / 'manifest.json')
        assert not set(manifest['seeds']).intersection(old['seeds'])
        for key in ['checkpoint_sha256', 'img_checkpoint_sha256', 'configuration_sha256', 'source_commit', 'bank_sha256']:
            assert manifest[key] == old[key]
    assert read(out/'smoke_status.json')['status'] == 'smoke_complete'
    assert read(out/'fault_status.json')['status'] == 'failed'
    assert 'Intentional supervisor failure test' in read(out/'fault_status.json')['error']
    assert not (out/'launch_receipt.json').exists(), 'Refusing duplicate launch'
    status = read(out/'status.json')
    status.update(status='starting', experiment_state='starting', source_updated_at=time.time(),
                  heartbeat_at=time.time(), startup_deadline_at=time.time()+600)
    write(out/'status.json', status)
    with (out/'supervisor.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(out/'runner.py'), '--mode', 'supervise', '--output', str(out)],
                    cwd=out, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(supervisor_identity=identity(process.pid), launched_at=time.time(), output=str(out))
    write(out/'launch_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare','prepare_remaining','worker','supervise','launch'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--previous-run', type=Path)
    parser.add_argument('--img-checkpoint', type=Path)
    parser.add_argument('--development-commit')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--fault', action='store_true')
    parsed = parser.parse_args()
    globals()[parsed.mode](parsed)
