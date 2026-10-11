"""Fresh default scripted rollouts with native HD workspace/target annotations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.control import configure_profile
from panda_cable_grasp.env.workspace_geometry import DEFAULT_TARGET_WORKSPACE, default_workspace_sector
from panda_cable_grasp.evaluation.benchmark import _base_row
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig
from panda_cable_grasp.scenarios.registry import SCENARIO_FAMILIES, get_scenario
from target_node_annotation import project, clip_segment

WIDTH, HEIGHT, FPS = 1280, 960, 50
STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS
WORKSPACE = default_workspace_sector()
WORKSPACE_BOUNDARY = WORKSPACE.boundary()
LABELS = ('静态', '整体运动 L1', '持续形变', '整体运动＋形变', '复杂静态',
          '普通形状整体回放', '复杂形状整体回放')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value), encoding='utf-8')
    temporary.replace(path)


def project_camera_manifest(config):
    return dict(source='src/panda_cable_grasp/env/environment.py: EnvConfig camera rig',
        global_camera=dict(name=config.dynamicvla_opst_camera_name,
            position_world=list(config.dynamicvla_opst_camera_pos),
            quaternion_world=list(config.dynamicvla_opst_camera_quat),
            fovy=config.dynamicvla_camera_fovy),
        wrist_camera=dict(name=config.dynamicvla_wrist_camera_name,
            position_in_hand=list(config.dynamicvla_wrist_camera_pos),
            quaternion_in_hand=list(config.dynamicvla_wrist_camera_quat),
            fovy=config.dynamicvla_camera_fovy))


def physical_state(env):
    state = np.empty(mujoco.mj_stateSize(env.model, STATE))
    mujoco.mj_getState(env.model, env.data, state, STATE)
    return state


def active_target(env, policy):
    if policy.locked_segment_index is None:
        node = int(policy.selected_target_index)
        return env.data.xpos[policy.selected_target_body_id].copy(), float(node)
    index, alpha = policy.locked_segment_index, policy.locked_segment_alpha
    ids = policy.locked_node_ids or env.cable_ids
    xyz = (1-alpha)*env.data.xpos[ids[index]] + alpha*env.data.xpos[ids[index+1]]
    return xyz.copy(), float(index+alpha)


def connector(scene, a, b):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                      np.zeros(3), np.eye(3).ravel(), np.array([.13, .70, 1., 1.]))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .002, a, b)
    scene.ngeom += 1


def marker(scene, xyz, radius, rgba):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                      np.array([radius, 0., 0.]), xyz, np.eye(3).ravel(), np.asarray(rgba))
    scene.ngeom += 1


def verify_video(path, frames):
    result = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_frames',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,codec_name:frame=key_frame,best_effort_timestamp_time,pict_type',
        '-of', 'json', str(path)]))
    stream, rows = result['streams'][0], result['frames']
    assert (stream['width'], stream['height'], stream['avg_frame_rate'], int(stream['nb_frames'])) == (WIDTH, HEIGHT, '50/1', frames)
    times = np.array([float(r['best_effort_timestamp_time']) for r in rows])
    keys = [i for i, r in enumerate(rows) if r['key_frame']]
    assert len(times) == frames and np.max(np.abs(np.diff(times)-.02), initial=0.) < 1e-6
    assert max(np.diff(keys), default=0) <= 12
    assert not any(r['pict_type'] == 'B' for r in rows)
    return dict(**stream, max_keyframe_spacing_frames=int(max(np.diff(keys), default=0)),
                sha256=digest(path), frame_timestamps_verified=True)


def record(job):
    if sys.platform != 'win32':
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(job['gpu'])
    scene = get_scenario(job['scene'])
    dest = Path(job['output']) / scene.name / f"seed_{job['seed']}"
    if (dest/'metadata.json').exists():
        metadata = json.loads((dest/'metadata.json').read_text(encoding='utf-8'))
        if metadata['protocol_sha256'] != job['protocol_sha256']:
            raise RuntimeError(f'Existing episode protocol differs: {dest}')
        return metadata['episode_row']
    dest.mkdir(parents=True, exist_ok=True)
    config = configure_profile(EnvConfig(robot='nero', seed=job['seed'], target_selection='middle',
        episode_seconds=15., scenario_name=scene.name, scenario_id=scene.scenario_id,
        scenario_split=scene.split.value, replay_bank_dir=job['bank'], replay_seed_fallback='error',
        dynamicvla_cameras_enabled=True,
        **scene.to_env_overrides()), job['controller_profile'])
    config.table_execution_guard_enabled = job['table_guard']
    env = CableGraspEnv(config)
    env.reset(seed=job['seed'])
    policy = DynamicCableGraspPolicy(env, PolicyConfig(**job.get('rule_config', {})))
    initial = env.info()
    env.model.vis.global_.offwidth = max(WIDTH, env.model.vis.global_.offwidth)
    env.model.vis.global_.offheight = max(HEIGHT, env.model.vis.global_.offheight)
    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH, max_geom=10000)
    cameras = {'global': config.dynamicvla_opst_camera_name,
               'wrist': config.dynamicvla_wrist_camera_name}
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'cable_marker')
    if site >= 0:
        env.model.site_rgba[site, 3] = 0
    font_path = ('C:/Windows/Fonts/arial.ttf' if sys.platform == 'win32'
                 else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, 25)
    writers = {view: subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate', str(FPS),
        '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', '-threads', '1',
        '-g', '12', '-keyint_min', '12', '-sc_threshold', '0', '-bf', '0',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dest/f'{view}.mp4')],
        stdin=subprocess.PIPE) for view in ('global', 'wrist')}
    states, times, targets, cable, metrics, actions = [], [], [], [], [], []
    render_difference = 0.
    terminated = truncated = False
    started = time.monotonic()

    def frame(info):
        nonlocal render_difference
        before = physical_state(env)
        point, material_index = active_target(env, policy)
        states.append(before); times.append(float(env.data.time)); targets.append(point)
        cable.append(env.data.xpos[env.cable_ids].copy())
        metrics.append(dict(time=times[-1], active_target_material_index=material_index,
            initial_selected_node=int(policy.selected_target_index), phase=policy.phase.name,
            success=bool(env.ever_success), grasp_confirmed=bool(env.grasp_confirmed),
            actual_grasped_body=info.get('grasped_body_id'), lifted_fraction=info.get('lifted_fraction', 0),
            success_hold=float(env.success_hold)))
        metrics[-1].update(desired_quat=policy.desired_quat.tolist(),
            actual_hand_rotation=env.data.xmat[env.hand_id].reshape(3, 3).tolist(),
            hand_xyz=env.hand_position.tolist(), arm_qpos=env.data.qpos[env.arm_qpos_adr].tolist(),
            arm_qvel=env.data.qvel[env.arm_dof_adr].tolist())
        desired_rotation = np.empty(9)
        mujoco.mju_quat2Mat(desired_rotation, policy.desired_quat)
        desired_closing = desired_rotation.reshape(3, 3) @ env.gripper_closing_axis_local
        actual_closing = env.data.xmat[env.hand_id].reshape(3, 3) @ env.gripper_closing_axis_local
        metrics[-1].update(desired_closing_yaw_deg=float(np.degrees(np.arctan2(desired_closing[1], desired_closing[0]))),
            actual_closing_yaw_deg=float(np.degrees(np.arctan2(actual_closing[1], actual_closing[0]))))
        for view, writer in writers.items():
            renderer.update_scene(env.data,
                camera=cameras[view], scene_option=option)
            for a, b in zip(WORKSPACE_BOUNDARY, np.roll(WORKSPACE_BOUNDARY, -1, axis=0)):
                connector(renderer.scene, a, b)
            marker(renderer.scene, point, .006, [1., .72, .05, 1.])
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            draw.rectangle((12, 12, 1268, 146), fill='#eef3fa')
            draw.text((26, 18), f"RULE | {scene.name} | {view.upper()}", font=font, fill='#16324d')
            draw.text((26, 50), f"seed {job['seed']} | t={times[-1]:.2f}s | {policy.phase.name} | success {bool(env.ever_success)}", font=font, fill='#16324d')
            draw.text((26, 82), f"YELLOW: tracked target u={material_index:.2f} | initial middle node {policy.selected_target_index}", font=font, fill='#16324d')
            draw.text((26, 114), f"Yaw desired/actual {metrics[-1]['desired_closing_yaw_deg']:.1f}/{metrics[-1]['actual_closing_yaw_deg']:.1f} deg | Q7 speed {np.degrees(env.data.qvel[env.arm_dof_adr[-1]]):.1f} deg/s", font=font, fill='#16324d')
            corners = [project(renderer.scene, p, WIDTH, HEIGHT) for p in WORKSPACE_BOUNDARY]
            for a, b in zip(corners, corners[1:] + corners[:1]):
                if a[2] > 0 and b[2] > 0:
                    segment = clip_segment(a[:2], b[:2], WIDTH, HEIGHT)
                    if segment:
                        draw.line(segment, fill='#112337', width=8)
                        draw.line(segment, fill='#42baff', width=4)
            x, y, depth = project(renderer.scene, point, WIDTH, HEIGHT)
            if depth > 0 and 20 <= x <= WIDTH-20 and 165 <= y <= HEIGHT-20:
                draw.ellipse((x-18, y-18, x+18, y+18), outline='#112337', width=8)
                draw.ellipse((x-18, y-18, x+18, y+18), outline='#ffe553', width=4)
                draw.ellipse((x-3, y-3, x+3, y+3), fill='#ffe553')
            writer.stdin.write(np.asarray(img).tobytes())
            if len(states) == 1:
                img.save(dest/f'{view}_poster.png')
        render_difference = max(render_difference, float(np.max(np.abs(physical_state(env)-before))))
        assert render_difference == 0.

    try:
        frame(initial)
        info = initial
        while not policy.finished:
            action = policy.action()
            _, _, terminated, truncated, info = env.step(action)
            actions.append(np.asarray(action).copy())
            frame(info)
            if terminated or truncated:
                break
        final = env.info()
        if env.ever_success and policy.result == 'running':
            policy.result = 'success'
        final.update(base_success=env.ever_success, success=policy.result == 'success',
                     ever_pinched=env.last_grasped_body_id is not None)
        row = _base_row('scripted', job['episode'], job['seed'], scene, initial, final, env.grasp_break_history)
        row.update(scene=scene.name, seed=job['seed'], directory=f"{scene.name}/seed_{job['seed']}",
            goal_node=int(policy.selected_target_index), policy_result=policy.result,
            duration_s=float(env.data.time), frames=len(states), steps=len(actions),
            max_lifted_fraction=float(max(m['lifted_fraction'] for m in metrics)),
            max_success_hold_s=float(max(m['success_hold'] for m in metrics)),
            wall_seconds=time.monotonic()-started)
        for writer in writers.values():
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError(f'Video encoder failed: {dest}')
        np.savez_compressed(dest/'trajectory.npz', states=np.asarray(states), frame_times=times,
            frame_indices=np.arange(len(states)), cable_xyz=np.asarray(cable), tracked_target_xyz=targets,
            native_actions=np.asarray(actions), cable_body_ids=env.cable_ids)
        save_json(dest/'metrics.json', metrics)
        encoding = {view: verify_video(dest/f'{view}.mp4', len(states)) for view in writers}
        save_json(dest/'metadata.json', dict(protocol_sha256=job['protocol_sha256'],
            scene=scene.asdict(), seed=job['seed'], env_config=asdict(env.config),
            policy_config=asdict(policy.config), initial_info=initial, final_info=final, episode_row=row,
            recording=dict(width=WIDTH, height=HEIGHT, fps=FPS, native=True, render_max_state_difference=render_difference,
                workspace=WORKSPACE.manifest(), workspace_display_only=True,
                target_marker='current world position of selected/locked material point; no extra policy calls',
                state_kind=int(STATE), cameras=project_camera_manifest(config)), encoding_verification=encoding,
            trajectory_sha256=digest(dest/'trajectory.npz'), grasp_break_history=env.grasp_break_history))
        return row
    finally:
        for writer in writers.values():
            if writer.poll() is None:
                if not writer.stdin.closed:
                    writer.stdin.close()
                writer.wait()
        renderer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--controller-profile', choices=('legacy', 'enhanced6_bounded_dls_v1'), default='enhanced6_bounded_dls_v1')
    parser.add_argument('--table-guard', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--follow-target-tangent', action=argparse.BooleanOptionalAction,
                        default=PolicyConfig().follow_target_tangent)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260804)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 4, 5])
    parser.add_argument('--scenes', nargs='+', default=list(SCENARIO_FAMILIES['cable_x075_v1']))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), Path(__file__).with_name('target_node_annotation.py'), ROOT/'src/panda_cable_grasp/env/environment.py',
        ROOT/'src/panda_cable_grasp/policies/scripted.py', ROOT/'src/panda_cable_grasp/scenarios/registry.py',
        ROOT/'src/panda_cable_grasp/env/control.py',
        ROOT/'src/panda_cable_grasp/env/workspace_geometry.py', DEFAULT_TARGET_WORKSPACE]
    protocol = dict(family='cable_x075_v1', method='scripted', robot='nero', scenes=args.scenes,
        seeds=list(range(args.seed, args.seed+args.episodes)), episode_seconds=15.,
        controller_profile=args.controller_profile, table_execution_guard_enabled=args.table_guard,
        rule_config=asdict(PolicyConfig(follow_target_tangent=args.follow_target_tangent)), bank=str(args.bank.resolve()),
        bank_sha256=digest(args.bank/'replay_src_shape_nominal_v1.npz'),
        bank_manifest=json.loads((args.bank/'replay_src_shape_nominal_v1.manifest.json').read_text()),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_files},
        workspace_display_only=True, workspace=WORKSPACE.manifest(), target_selection='middle + original nearest-point lock',
        width=WIDTH, height=HEIGHT, fps=FPS,
        cameras=project_camera_manifest(EnvConfig(robot='nero', dynamicvla_cameras_enabled=True)),
        original_scene_registry={name: get_scenario(name).asdict() for name in args.scenes})
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, default=json_value).encode()).hexdigest()
    manifest = args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['protocol_sha256'] != protocol_hash:
        raise RuntimeError('Output protocol changed; use a new directory')
    save_json(manifest, dict(protocol_sha256=protocol_hash, **protocol))
    rows = []
    summary = dict(status='running', expected_episodes=len(args.scenes)*args.episodes, completed_episodes=0,
                   episodes=rows, protocol_sha256=protocol_hash)
    save_json(args.output/'summary.json', summary)
    jobs = []
    for episode, seed in enumerate(protocol['seeds'], 1):
        for name in args.scenes:
            jobs.append(dict(scene=name, seed=seed, episode=episode, output=str(args.output),
                bank=str(args.bank.resolve()), controller_profile=args.controller_profile,
                rule_config=protocol['rule_config'],
                table_guard=args.table_guard, gpu=args.gpus[len(jobs) % len(args.gpus)], protocol_sha256=protocol_hash))
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(record, job) for job in jobs]):
                row = future.result()
                rows.append(row)
                summary['completed_episodes'] = len(rows)
                save_json(args.output/'summary.json', summary)
                print(json.dumps(dict(completed=len(rows), expected=len(jobs), scene=row['scene'],
                    seed=row['seed'], success=row['task_success'], duration=row['duration_s'])), flush=True)
        summary.update(status='complete', by_scene={name: dict(episodes=args.episodes,
            successes=sum(bool(r['task_success']) for r in rows if r['scene'] == name),
            success_rate=sum(bool(r['task_success']) for r in rows if r['scene'] == name)/args.episodes)
            for name in args.scenes})
        save_json(args.output/'summary.json', summary)
    except Exception as error:
        summary.update(status='failed', error=repr(error))
        save_json(args.output/'summary.json', summary)
        raise


if __name__ == '__main__':
    main()
