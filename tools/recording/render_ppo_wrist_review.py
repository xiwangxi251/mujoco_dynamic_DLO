"""Restore the archived fixed wrist axis; replay states with command and gap HUD."""
import os
import sys
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

RUN = Path(os.environ['FULL6_PPO_RUN'])
os.environ.setdefault('MUJOCO_GL','glfw')
sys.path.insert(0,str(RUN))
import record_full6_ppo_videos as recording
from target_node_annotation import annotate, annotate_tolerance
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(directory):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    metrics = json.loads((directory/'metrics.json').read_text())
    source_sha = recording.trainer.sha(directory/'trajectory.npz')
    data = np.load(directory/'trajectory.npz')
    states,times,actions = data['states'],data['frame_times'],data['actions']
    env = recording.trainer.CourseEnv(1000+meta['episode'],evaluation=True)
    env.set_stage(meta['stage'])
    env.reset(seed=meta['seed'])
    base = env.env.base_env
    assert env.env.goal.index == meta['goal_node']
    base.model.vis.global_.offwidth = 1280
    base.model.vis.global_.offheight = 960
    marker = mujoco.mj_name2id(base.model,mujoco.mjtObj.mjOBJ_SITE,'cable_marker')
    if marker >= 0:
        base.model.site_rgba[marker,3] = 0
    renderer = mujoco.Renderer(base.model,height=960,width=1280,max_geom=10000)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = meta['global_camera']['lookat']
    for key in ('distance','azimuth','elevation'):
        setattr(camera,key,meta['global_camera'][key])
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    position = np.array(meta['wrist_camera']['local_position'])
    old_rotation = np.array(meta['wrist_camera']['local_rotation'])
    # Keep the camera's recorded fixed optical axis. A TCP look-at exposes
    # the gripper housing and occludes the task view.
    forward = -old_rotation[:,2]
    up = old_rotation[:,1]
    visual = {(int(base.model.geom_bodyid[i]),int(base.model.geom_dataid[i]))
              for i in range(base.model.ngeom)
              if base.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and base.model.geom_group[i] == 1}
    duplicates = {i for i in range(base.model.ngeom)
                  if base.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and base.model.geom_group[i] == 0
                  and (int(base.model.geom_bodyid[i]),int(base.model.geom_dataid[i])) in visual}
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
    numbers = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',27)
    writer = recording.Encoder(directory/'wrist_tips.mp4')
    tolerance = env.env.rl_config.target_node_tolerance
    allowed = list(range(max(0,meta['goal_node']-tolerance),min(len(base.cable_ids),meta['goal_node']+tolerance+1)))
    closed = False
    max_difference = 0
    tracks = []
    try:
        for index,state in enumerate(states):
            if index:
                signal = float(actions[index-1,4])
                if signal <= -.35: closed = True
                elif signal >= .35: closed = False
            else: signal = 0.
            mujoco.mj_setState(base.model,base.data,state,mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(base.model,base.data)
            max_difference = max(max_difference,float(np.max(abs(recording.state(base)-state))))
            renderer.update_scene(base.data,camera=camera,scene_option=option)
            hand = base.data.xmat[base.hand_id].reshape(3,3)
            for cam in renderer.scene.camera:
                cam.pos[:] = base.data.xpos[base.hand_id]+hand@position
                cam.forward[:] = hand@forward
                cam.up[:] = hand@up
                extent = cam.frustum_near*np.tan(np.radians(base.config.dynamicvla_camera_fovy)/2)
                cam.frustum_top,cam.frustum_bottom = extent,-extent
            # Duplicate visual/collision meshes coincide; hide the redundant
            # collision copy in the display scene to prevent z fighting.
            for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in duplicates:
                    geom.rgba[3] = 0.
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            row = metrics[index-1] if index else dict(actual_node=None,secured=False,lifted_fraction=0.,success_hold=0.)
            draw.rounded_rectangle((16,14,1264,145),radius=10,fill='#f2f6fb')
            draw.text((30,22),f'PPO {meta["checkpoint_steps"]:,} | seed {meta["seed"]} | WRIST fixed axis | t={times[index]:.2f}s',font=font,fill='#172f49')
            draw.text((30,59),f'Grip command: {"CLOSE" if closed else "OPEN"} | action={signal:+.2f} | actual gap={base.finger_aperture*1000:.2f} mm',font=small,fill='#304d68')
            draw.text((30,88),f'Target {meta["goal_node"]} (+/-{tolerance}) | grasp state node {row["actual_node"]} | secured {row["secured"]}',font=small,fill='#304d68')
            draw.text((30,116),f'Lifted {100*row["lifted_fraction"]:.0f}% | hold {row["success_hold"]:.2f}/0.80s | FOV 73.74 deg | saved-state replay',font=small,fill='#304d68')
            annotate_tolerance(draw,renderer.scene,base.data.xpos[np.array(base.cable_ids)[allowed]],allowed,numbers,numbers)
            annotate(draw,renderer.scene,base.data.xpos[base.cable_ids[meta['goal_node']]],meta['goal_node'],numbers,latin_font=numbers)
            writer.write(np.asarray(image))
            tracks.append(dict(time_s=float(times[index]),command='CLOSE' if closed else 'OPEN',normalized_grip_action=signal,actual_gap_mm=base.finger_aperture*1000))
            if index in (0,40,45,250): image.save(directory/f'wrist_tips_{index:04d}.png')
            if index == 0: image.save(directory/'wrist_tips_poster.png')
            if index%250 == 0: print(f'WRIST {meta["seed"]} {index}/{len(states)}',flush=True)
    finally:
        writer.close()
        renderer.close()
        env.close()
    assert max_difference == 0 and source_sha == recording.trainer.sha(directory/'trajectory.npz')
    check = recording.encode_check(directory/'wrist_tips.mp4',len(states))
    report = dict(seed=meta['seed'],source_trajectory_sha256=source_sha,source_trajectory_unchanged=True,
        rendered_state_max_difference=max_difference,fovy=73.7397952917,local_position=position.tolist(),
        local_forward=forward.tolist(),local_up=up.tolist(),camera_aim='original fixed optical axis; TCP look-at removed',
        archive_reference='outputs/control_unification_20261002/scripted_pair_hd_20261002/raw/baseline/episodes/scripted/id_static/seed_2026100200/wrist.mp4',
        archive_commit='d70b249fd623f13c3bc1f670bdb3da2e9cd8685a',
        camera_scope='recording display only',duplicate_collision_meshes_hidden=sorted(duplicates),
        encoding=check,script_sha256=recording.trainer.sha(Path(__file__)),
        file=f'{directory.name}/wrist_tips.mp4',poster=f'{directory.name}/wrist_tips_poster.png')
    (directory/'wrist_tips_tracks.json').write_text(json.dumps(tracks,indent=2))
    (directory/'wrist_tips_verification.json').write_text(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    root = RUN/'videos/step_003538944'
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        results = list(pool.map(render,[str(p) for p in sorted(root.glob('seed_*'))]))
    (root/'wrist_tips_manifest.json').write_text(json.dumps(dict(passed=True,episodes=results),indent=2))
    print('WRIST FIXED AXIS COMPLETE',flush=True)
