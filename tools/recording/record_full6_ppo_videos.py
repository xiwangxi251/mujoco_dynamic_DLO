"""Native HD/50Hz recordings of the workspace PPO's actual closed loop."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

RUN = Path(os.environ.get('FULL6_PPO_RUN', Path(__file__).resolve().parent))
sys.path.insert(0, str(RUN))
import train_targetnode_full6_workspace_eval13 as trainer
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def state(base):
    result = np.empty(mujoco.mj_stateSize(base.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(base.model, base.data, result, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return result


class Encoder:
    def __init__(self, path):
        self.proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1280x960',
            '-framerate', '50', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '16', '-threads', '2', '-g', '12', '-keyint_min', '12',
            '-sc_threshold', '0', '-bf', '0', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)

    def write(self, rgb):
        self.proc.stdin.write(rgb.tobytes())

    def close(self):
        self.proc.stdin.close()
        assert self.proc.wait() == 0


def encode_check(path, frames):
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,codec_name,duration',
        '-of', 'json', str(path)]))['streams'][0]
    assert (info['width'], info['height'], info['avg_frame_rate'], info['codec_name']) == (1280, 960, '50/1', 'h264')
    assert int(info['nb_frames']) == frames
    timeline = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=key_frame,best_effort_timestamp_time',
        '-of', 'json', str(path)]))['frames']
    times = np.array([float(f['best_effort_timestamp_time']) for f in timeline])
    assert len(times) == frames and np.max(abs(np.diff(times)-.02)) < 1e-6
    keys = [i for i, frame in enumerate(timeline) if frame['key_frame']]
    spacing = max(np.diff(keys), default=0)
    assert spacing <= 12
    return dict(**info, max_keyframe_spacing_frames=int(spacing), timestamps_verified=True)


def capture(job):
    checkpoint, stage, episode, output = job
    trainer.torch.set_num_threads(1)
    seed = 20260804 + episode
    directory = Path(output)/f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    model = trainer.PPO.load(checkpoint, device='cpu')
    env = trainer.CourseEnv(1000+episode, evaluation=True)
    env.set_stage(stage)
    obs, initial = env.reset(seed=seed)
    base = env.env.base_env
    goal = env.env.goal.index
    physics_grasp_trace = []
    original_grasp_update = base._update_physical_grasp_state
    def traced_grasp_update(gripper_closed):
        original_grasp_update(gripper_closed)
        forces = base.finger_normal_forces()
        physics_grasp_trace.append(dict(time=float(base.data.time),
            applied_single_finger_target_mm=1000*float(base.data.ctrl[base.gripper_actuator_id]),
            actual_gap_mm=1000*base.finger_aperture,
            left_force_n=float(forces[base.left_finger_id]),right_force_n=float(forces[base.right_finger_id]),
            pad_pairs=base._finger_contact_pairs(),
            grasp_node=env.env.goal.actual_index(),grasp_confirmed=bool(base.grasp_confirmed),
            bilateral_near_grasp=bool(base._grasp_has_bilateral_pad_contact()),
            grasp_break_count=len(base.grasp_break_history)))
    if hasattr(base, '_grasp_has_bilateral_pad_contact'):
        base._update_physical_grasp_state = traced_grasp_update
    initial_xyz = base.data.xpos[base.cable_ids[goal]].copy()
    base.model.vis.global_.offwidth = max(1280, base.model.vis.global_.offwidth)
    base.model.vis.global_.offheight = max(960, base.model.vis.global_.offheight)
    renderer = mujoco.Renderer(base.model, height=960, width=1280, max_geom=10000)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (.52, .06, .13)
    camera.distance, camera.azimuth, camera.elevation = 1.45, 135, -32
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(base.config.dynamicvla_wrist_camera_quat))
    rotation = rotation.reshape(3, 3)
    position = np.asarray(base.config.dynamicvla_wrist_camera_pos)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 25)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    writers = {view: Encoder(directory/f'{view}.mp4') for view in ('global', 'wrist')}
    states, observations, actions, rewards, rows = [state(base)], [obs.copy()], [], [], []
    start_time = base.data.time
    frames = 0
    start_wall = time.monotonic()

    def frame(info):
        nonlocal frames
        for view, writer in writers.items():
            renderer.update_scene(base.data, camera=camera, scene_option=option)
            if view == 'wrist':
                hand = base.data.xmat[base.hand_id].reshape(3, 3)
                for c in renderer.scene.camera:
                    c.pos[:] = base.data.xpos[base.hand_id] + hand @ position
                    c.forward[:] = -(hand @ rotation)[:, 2]
                    c.up[:] = (hand @ rotation)[:, 1]
                    extent = c.frustum_near * np.tan(np.radians(base.config.dynamicvla_camera_fovy) / 2)
                    c.frustum_top, c.frustum_bottom = extent, -extent
            target_xyz = base.data.xpos[base.cable_ids[goal]].copy()
            marker = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([.0045, 0., 0.]), target_xyz, np.eye(3).ravel(), np.array([1., .72, .04, .85]))
            renderer.scene.ngeom += 1
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((16, 14, 1264, 145), radius=10, fill='#f2f6fb')
            draw.text((30, 22), f'PPO step {model.num_timesteps:,} | seed {seed} | {view.upper()} | 1x / 50 fps', font=font, fill='#172f49')
            draw.text((30, 59), f'Target node {goal} (+/-{env.env.rl_config.target_node_tolerance}) | grasped {env.env.goal.actual_index()} | t={base.data.time-start_time:.2f}s', font=small, fill='#304d68')
            draw.text((30, 88), f'Secured {bool(info.get("secured_grasp", False))} | lifted cable {100*info.get("lifted_fraction", 0):.0f}% | success hold {base.success_hold:.2f}/{base.config.success_hold_seconds:.2f}s', font=small, fill='#304d68')
            draw.text((30, 116), f'Task success: {bool(info.get("task_success", False))} | yellow marker: fixed material target node', font=small, fill='#304d68')
            writer.write(np.asarray(image))
            if frames == 0:
                image.save(directory/f'{view}_poster.png')
            if frames == 250:
                image.save(directory/f'{view}_at5s.png')
            if info.get('task_success'):
                image.save(directory/f'{view}_success.png')
        frames += 1

    info = initial
    frame(info)
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            actions.append(action.copy())
            rewards.append(reward)
            observations.append(obs.copy())
            states.append(state(base))
            rows.append(dict(time=base.data.time-start_time, task_success=bool(info.get('task_success', False)),
                secured=bool(info.get('secured_grasp', False)), actual_node=env.env.goal.actual_index(),
                lifted_fraction=float(info.get('lifted_fraction', 0)), success_hold=base.success_hold,
                goal_inside=bool(info.get('workspace_goal_current_inside', False)),
                strict_hold=float(info.get('strict_success_hold', 0)),
                applied_single_finger_target_mm=1000*float(base.data.ctrl[base.gripper_actuator_id]),
                actual_gap_mm=1000*base.finger_aperture,
                left_force_n=float(info.get('left_finger_normal_force',0)),
                right_force_n=float(info.get('right_finger_normal_force',0)),
                raw_bilateral=bool(info.get('raw_bilateral_contact',False)),
                grasp_break_count=len(base.grasp_break_history)))
            frame(info)
            if len(actions) % 150 == 0:
                print(f'seed={seed} steps={len(actions)} wall={time.monotonic()-start_wall:.1f}s', flush=True)
            if done or truncated:
                break
    finally:
        for writer in writers.values():
            writer.close()
        renderer.close()
        env.close()
    # Rerun without any renderer to verify recording has not changed motion.
    plain = trainer.CourseEnv(1000+episode, evaluation=True)
    plain.set_stage(stage)
    plain_obs, _ = plain.reset(seed=seed)
    difference = float(np.max(abs(state(plain.env.base_env)-states[0])))
    for i in range(len(actions)):
        replay_action, _ = model.predict(plain_obs, deterministic=True)
        assert np.array_equal(replay_action, actions[i])
        plain_obs, replay_reward, d, t, replay_info = plain.step(replay_action)
        difference = max(difference, float(np.max(abs(state(plain.env.base_env)-states[i+1]))))
        assert np.array_equal(plain_obs, observations[i+1])
        assert replay_reward == rewards[i]
        assert (d or t) == (i == len(actions)-1)
    plain.close()
    assert difference <= 1e-12, difference
    np.savez_compressed(directory/'trajectory.npz', states=np.asarray(states),
        observations=np.asarray(observations), actions=np.asarray(actions), rewards=np.asarray(rewards),
        frame_indices=np.arange(frames), frame_times=np.arange(frames)*.02)
    metadata = dict(seed=seed, episode=episode, actual_seed=initial['actual_seed'], stage=stage,
        stage_name=trainer.STAGES[stage]['name'], goal_node=goal, goal_initial_xyz=initial_xyz.tolist(),
        first_correct=info.get('first_grasp_correct'), task_success=bool(info.get('task_success', False)),
        terminated=bool(done), truncated=bool(truncated), return_=sum(rewards),
        max_lifted_fraction=max(r['lifted_fraction'] for r in rows),
        max_success_hold_s=max(r['success_hold'] for r in rows),
        final_info=info, steps=len(actions), frames=frames, simulation_duration_s=len(actions)*.02,
        recording='fresh deterministic physical rollout; independent native render at every actual 20 ms state',
        checkpoint=str(checkpoint), checkpoint_steps=model.num_timesteps,
        checkpoint_sha256=trainer.sha(checkpoint), controller_profile=trainer.PROFILE,
        execution_module_sha256=trainer.sha(RUN/'full6_workspace_env.py'),
        recording_source_sha256=trainer.sha(Path(__file__)),
        evaluation_bank=str(trainer.BANK), reward_source_sha256=trainer.configuration(20000000,24)['reward_source_sha256'],
        width=1280, height=960, fps=50, playback_speed=1, codec='h264', crf=16,
        global_camera=dict(lookat=camera.lookat.tolist(), distance=camera.distance, azimuth=camera.azimuth, elevation=camera.elevation),
        wrist_camera=dict(local_position=position.tolist(), local_rotation=rotation.tolist(),
                          fovy_degrees=base.config.dynamicvla_camera_fovy),
        no_renderer_rerun_max_state_difference=difference,
        encoding_verification={view: encode_check(directory/f'{view}.mp4', frames) for view in writers},
        directory=directory.name)
    (directory/'metadata.json').write_text(json.dumps(metadata, indent=2, default=lambda v:
        v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)))
    (directory/'metrics.json').write_text(json.dumps(rows, indent=2))
    (directory/'physics_grasp_trace.json').write_text(json.dumps(physics_grasp_trace))
    (directory/'grasp_break_history.json').write_text(json.dumps(base.grasp_break_history,indent=2))
    print(f'DONE seed={seed} goal={goal} success={metadata["task_success"]} state_difference={difference}', flush=True)
    return {k: metadata[k] for k in ('seed','stage','stage_name','goal_node','first_correct','task_success',
        'steps','frames','simulation_duration_s','checkpoint_steps','checkpoint_sha256','directory',
        'max_lifted_fraction','max_success_hold_s','no_renderer_rerun_max_state_difference')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--limit', type=int, help='short smoke recording only')
    args = parser.parse_args()
    if args.limit is not None:
        raise ValueError('Use full episodes for delivered videos')
    checkpoint = args.checkpoint or sorted((RUN/'checkpoints').glob('step_*.zip'))[-1]
    stage = args.stage if args.stage is not None else json.loads((RUN/'curriculum_state.json').read_text())['stage']
    output = RUN/'videos'/checkpoint.stem
    output.mkdir(parents=True, exist_ok=False)
    jobs = [(str(checkpoint), stage, e, str(output)) for e in range(args.episodes)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        rows = list(pool.map(capture, jobs))
    (output/'manifest.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
        selected_episodes='first four consecutive held-out seeds; no success-based selection',
        episodes=rows, passed=True), indent=2))
    print(json.dumps(dict(output=str(output), results=rows)), flush=True)


if __name__ == '__main__':
    main()
