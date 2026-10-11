"""Render the original saved states with explicit target-node annotations."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

RUN = Path(os.environ.get('FULL6_PPO_RUN', Path(__file__).resolve().parent))
sys.path.insert(0, str(RUN))
import record_full6_ppo_videos as original
from target_node_annotation import annotate, annotate_tolerance
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(directory, display_tolerance=None, output_suffix='clean'):
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    metrics = json.loads((directory/'metrics.json').read_text())
    original.trainer.torch.set_num_threads(1)
    env = original.trainer.CourseEnv(1000+metadata['episode'], evaluation=True)
    env.set_stage(metadata['stage'])
    env.reset(seed=metadata['seed'])
    base = env.env.base_env
    assert env.env.goal.index == metadata['goal_node']
    original_tolerance = int(env.env.rl_config.target_node_tolerance)
    tolerance = original_tolerance if display_tolerance is None else display_tolerance
    preview = tolerance != original_tolerance
    goal_node = metadata['goal_node']
    first,last = max(0,goal_node-tolerance), min(len(base.cable_ids)-1,goal_node+tolerance)
    allowed = list(range(first,last+1))
    assert all((abs(node-goal_node) <= tolerance) == (node in allowed)
               for node in range(len(base.cable_ids)))
    with np.load(directory/'trajectory.npz') as source:
        states = source['states'].copy()
        times = source['frame_times'].copy()
    assert len(states) == metadata['frames'] == len(metrics)+1
    assert np.max(abs(np.diff(times)-.02)) < 1e-9
    source_sha = original.trainer.sha(directory/'trajectory.npz')
    # This fixed debug site is unrelated to the chosen material node.
    marker = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_SITE, 'cable_marker')
    if marker >= 0:
        base.model.site_rgba[marker,3] = 0.
    base.model.vis.global_.offwidth = max(1280, base.model.vis.global_.offwidth)
    base.model.vis.global_.offheight = max(960, base.model.vis.global_.offheight)
    renderer = mujoco.Renderer(base.model, height=960, width=1280, max_geom=10000)
    camera = mujoco.MjvCamera()
    for name in ('distance','azimuth','elevation'):
        setattr(camera, name, metadata['global_camera'][name])
    camera.lookat[:] = metadata['global_camera']['lookat']
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    local_position = np.array(metadata['wrist_camera']['local_position'])
    local_rotation = np.array(metadata['wrist_camera']['local_rotation'])
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    chinese = ImageFont.truetype('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',27)
    numbers = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',27)
    writers = {v: original.Encoder(directory/f'{v}_{output_suffix}.mp4') for v in ('global','wrist')}
    tracks = {v: [] for v in writers}
    state_difference = 0.
    try:
        for index, saved in enumerate(states):
            mujoco.mj_setState(base.model, base.data, saved, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(base.model, base.data)
            state_difference = max(state_difference, float(np.max(abs(original.state(base)-saved))))
            xyz = base.data.xpos[base.cable_ids[metadata['goal_node']]].copy()
            allowed_xyz = base.data.xpos[np.asarray(base.cable_ids)[allowed]].copy()
            info = metrics[index-1] if index else dict(actual_node=None, secured=False, lifted_fraction=0., success_hold=0., task_success=False)
            for view, writer in writers.items():
                renderer.update_scene(base.data, camera=camera, scene_option=option)
                if view == 'wrist':
                    hand_rotation = base.data.xmat[base.hand_id].reshape(3,3)
                    rotation = hand_rotation @ local_rotation
                    for c in renderer.scene.camera:
                        c.pos[:] = base.data.xpos[base.hand_id]+hand_rotation @ local_position
                        c.forward[:] = -rotation[:,2]
                        c.up[:] = rotation[:,1]
                        extent = c.frustum_near*np.tan(np.radians(metadata['wrist_camera'].get('fovy_degrees',base.config.dynamicvla_camera_fovy))/2)
                        c.frustum_top, c.frustum_bottom = extent, -extent
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rounded_rectangle((16,14,1264,145),radius=10,fill='#f2f6fb')
                draw.text((30,22),f'PPO step {metadata["checkpoint_steps"]:,} | seed {metadata["seed"]} | {view.upper()} | 1x / 50 fps',font=font,fill='#172f49')
                range_label = 'preview range' if preview else 'allowed'
                draw.text((30,59),f'Target node {goal_node} (+/-{tolerance}) | {range_label} {first}-{last} | grasped {info["actual_node"]} | t={times[index]:.2f}s',font=small,fill='#304d68')
                draw.text((30,88),f'Secured {info["secured"]} | lifted cable {100*info["lifted_fraction"]:.0f}% | success hold {info["success_hold"]:.2f}/{base.config.success_hold_seconds:.2f}s',font=small,fill='#304d68')
                legend = (f'Original +/-{original_tolerance} task success: {info["task_success"]} | GREEN: +/-{tolerance} visual preview'
                    if preview else f'Task success: {info["task_success"]} | YELLOW: specified node | GREEN: allowed material-node range')
                draw.text((30,116),legend,font=small,fill='#304d68')
                tolerance_track = annotate_tolerance(draw,renderer.scene,allowed_xyz,allowed,chinese,numbers)
                track = annotate(draw, renderer.scene, xyz, metadata['goal_node'], chinese, latin_font=numbers)
                tracks[view].append(dict(frame=index, time_s=float(times[index]), xyz=xyz.tolist(),
                    tolerance_nodes=tolerance_track, **track))
                writer.write(np.asarray(image))
                if index in (0,100,250,500,750):
                    image.save(directory/f'{view}_{output_suffix}_{index:04d}.png')
                if index == 0:
                    image.save(directory/f'{view}_{output_suffix}_poster.png')
            if index % 250 == 0:
                print(f'clean annotation seed={metadata["seed"]} nodes={first}-{last} frame={index}/{len(states)}',flush=True)
    finally:
        for writer in writers.values():
            writer.close()
        renderer.close()
        env.close()
    assert state_difference == 0.
    assert original.trainer.sha(directory/'trajectory.npz') == source_sha
    report = dict(seed=metadata['seed'], goal_node=metadata['goal_node'], frames=len(states),
        source_trajectory_sha256=source_sha, source_checkpoint_sha256=metadata['checkpoint_sha256'],
        source_recording_metadata_sha256=original.trainer.sha(directory/'metadata.json'),
        rendering='annotated rendering of the same saved physical states; no new policy rollout',
        target='fixed material node body center projected through the actual per-frame camera',
        target_node_tolerance=tolerance, allowed_nodes=allowed,
        original_target_node_tolerance=original_tolerance, visual_preview_only=preview,
        output_suffix=output_suffix, original_metrics_unchanged=True,
        tolerance_definition='abs(actual_material_node-goal_material_node) <= tolerance; endpoints inclusive',
        target_label_box_enabled=False, target_arrow_enabled=False,
        bottom_tolerance_legend_enabled=False, top_status_hud_enabled=True,
        static_cable_marker_hidden=True, source_trajectory_unchanged=True,
        rendered_state_max_difference=state_difference,
        script_sha256=original.trainer.sha(Path(__file__)),
        annotation_source_sha256=original.trainer.sha(RUN/'target_node_annotation.py'),
        encoding={v:original.encode_check(directory/f'{v}_{output_suffix}.mp4',len(states)) for v in writers},
        visible_frames={v:sum(r['onscreen'] for r in track) for v,track in tracks.items()},
        files={v:f'{directory.name}/{v}_{output_suffix}.mp4' for v in writers})
    (directory/f'{output_suffix}_tracks.json').write_text(json.dumps(tracks,indent=2))
    (directory/f'{output_suffix}_annotation.json').write_text(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--display-tolerance',type=int)
    parser.add_argument('--output-suffix',default='clean')
    args = parser.parse_args()
    if args.display_tolerance is not None and args.display_tolerance < 0:
        parser.error('--display-tolerance must be nonnegative')
    if not args.output_suffix.replace('_','').isalnum():
        parser.error('--output-suffix must contain letters, digits or underscores')
    directories = sorted(args.directory.glob('seed_*'))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        render_episode = partial(render,display_tolerance=args.display_tolerance,output_suffix=args.output_suffix)
        rows = list(pool.map(render_episode,[str(d) for d in directories]))
    (args.directory/f'{args.output_suffix}_annotation_manifest.json').write_text(json.dumps(dict(passed=True,episodes=rows),indent=2))
    print(f'{args.output_suffix.upper()} ANNOTATIONS COMPLETE',flush=True)
