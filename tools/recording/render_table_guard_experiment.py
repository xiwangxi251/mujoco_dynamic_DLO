"""Native rendering of paired safety experiment's saved 20 ms states."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

RUN = Path(os.environ['FULL6_PPO_RUN'])
sys.path.insert(0, str(RUN))
import record_full6_ppo_videos as original
from target_node_annotation import annotate, annotate_tolerance
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(directory):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    rows = json.loads((directory/'metrics.json').read_text())
    original.trainer.torch.set_num_threads(1)
    env = original.trainer.CourseEnv(1000+meta['episode'], evaluation=True)
    env.set_stage(meta.get('stage',0))
    if meta.get('scenario_override'):
        env.env.set_training_scenario_distribution(original.trainer.SCENES,
            tuple(float(s == meta['scenario_override']) for s in original.trainer.SCENES))
    env.reset(seed=meta['seed'])
    base = env.env.base_env
    assert env.env.goal.index == meta['goal_node']
    with np.load(directory/'trajectory.npz') as source:
        states, times = source['states'].copy(), source['frame_times'].copy()
    assert len(states) == meta['frames'] == len(rows)+1
    assert np.max(abs(np.diff(times)-.02)) < 1e-9
    source_sha = original.trainer.sha(directory/'trajectory.npz')
    assert original.trainer.sha(directory/'trajectory.npz') == meta['trajectory_sha256']
    assert __import__('hashlib').sha256(original.state(base).tobytes()).hexdigest() == meta['initial_state_sha256']
    marker = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_SITE, 'cable_marker')
    if marker >= 0:
        base.model.site_rgba[marker, 3] = 0
    base.model.vis.global_.offwidth, base.model.vis.global_.offheight = 1280, 960
    renderer = mujoco.Renderer(base.model, height=960, width=1280, max_geom=10000)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (.52,.06,.13)
    camera.distance, camera.azimuth, camera.elevation = 1.45,135,-32
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(base.config.dynamicvla_wrist_camera_quat))
    rotation = rotation.reshape(3,3)
    position = np.asarray(base.config.dynamicvla_wrist_camera_pos)
    font_path = 'C:/Windows/Fonts/arial.ttf' if os.name == 'nt' else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 21)
    numbers = ImageFont.truetype(font_path, 27)
    writers = {v:original.Encoder(directory/f'{v}.mp4') for v in ('global','wrist')}
    node, tolerance = meta['goal_node'], meta['target_node_tolerance']
    allowed = list(range(max(0,node-tolerance),min(len(base.cable_ids)-1,node+tolerance)+1))
    max_difference = 0.
    try:
        for index, saved in enumerate(states):
            mujoco.mj_setState(base.model, base.data, saved, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(base.model, base.data)
            max_difference = max(max_difference,float(np.max(abs(original.state(base)-saved))))
            row = rows[max(0,index-1)]
            for view, writer in writers.items():
                renderer.update_scene(base.data,camera=camera,scene_option=option)
                if view == 'wrist':
                    hand = base.data.xmat[base.hand_id].reshape(3,3)
                    for c in renderer.scene.camera:
                        c.pos[:] = base.data.xpos[base.hand_id]+hand@position
                        c.forward[:] = -(hand@rotation)[:,2]
                        c.up[:] = (hand@rotation)[:,1]
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rounded_rectangle((16,14,1264,145),radius=10,fill='#f2f6fb')
                method = 'RULE / shared Full6 execution' if meta.get('policy_kind') == 'scripted' else f'PPO {meta["checkpoint_steps"]:,}'
                draw.text((30,22),f'{meta["variant"].upper()} | {method} | seed {meta["seed"]} | {view.upper()} | 1x / 50 fps',font=font,fill='#172f49')
                draw.text((30,59),f't={times[index]:.2f}s | target {node} (+/-{tolerance}, original task) | grasped {row["actual_node"]} | success {row["task_success"]}',font=small,fill='#304d68')
                motion_note = (f' | goal {row["goal_speed_m_s"]:.2f} m/s | WS {"IN" if row["workspace_goal_inside"] else "OUT"}' if meta.get('dynamic_test') else '')
                draw.text((30,88),f'Surface clearance {1000*row["surface_clearance_m"]:.1f} mm | down {row["down_speed_m_s"]:.3f} m/s | contact {row["normal_force_N"]:.1f} N'+motion_note,font=small,fill='#304d68')
                phase = f' | phase {row["phase"]}' if 'phase' in row else ''
                if meta.get('dynamic_test'):
                    phase += ' | '+meta['scenario'].replace('id_','').replace('_nominal_current','').replace('_nominal','')
                draw.text((30,116),f'Guard {row["guard_active"]}{phase} | YELLOW: target | GREEN: success range | saved-state replay',font=small,fill='#304d68')
                annotate_tolerance(draw,renderer.scene,base.data.xpos[np.asarray(base.cable_ids)[allowed]],allowed,numbers,numbers)
                annotate(draw,renderer.scene,base.data.xpos[base.cable_ids[node]],node,numbers,latin_font=numbers)
                writer.write(np.asarray(image))
                if index in (0,100,250,500,750):
                    image.save(directory/f'{view}_{index:04d}.png')
                if index == 0:
                    image.save(directory/f'{view}_poster.png')
            if index%250 == 0:
                print(f'RENDER {meta["variant"]} {meta["seed"]} {index}/{len(states)}',flush=True)
    finally:
        for writer in writers.values():
            writer.close()
        renderer.close()
        env.close()
    assert max_difference == 0
    assert original.trainer.sha(directory/'trajectory.npz') == source_sha
    report = dict(trajectory_sha256=source_sha, rendered_state_max_difference=max_difference,
        checkpoint_sha256=meta['checkpoint_sha256'], policy_kind=meta.get('policy_kind','ppo'),
        renderer_source_sha256=original.trainer.sha(Path(__file__)),fps=50, width=1280,height=960,
        rendering='each saved physical state at 20 ms; no interpolation or policy rerun',
        target_annotation='circle only, no label box or arrow', tolerance=tolerance,
        encoding={v:original.encode_check(directory/f'{v}.mp4',len(states)) for v in writers})
    (directory/'video_verification.json').write_text(json.dumps(report,indent=2))
    return dict(variant=meta['variant'],seed=meta['seed'],**report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--episodes',type=int,default=4)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--start-episode',type=int,default=0)
    parser.add_argument('--episode-indices',type=int,nargs='+')
    args = parser.parse_args()
    episode_indices = args.episode_indices or range(args.start_episode,args.start_episode+args.episodes)
    directories = [args.directory/v/f'seed_{20260804+e}' for e in episode_indices for v in ('baseline','protected')]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        results = list(pool.map(render,[str(p) for p in directories]))
    (args.directory/f'video_manifest_{args.start_episode}.json').write_text(json.dumps(results,indent=2))
