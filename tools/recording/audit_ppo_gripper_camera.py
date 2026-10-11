"""Read recorded actions/joint states and compare wrist rendering fields of view."""
import os
import sys
import json
from pathlib import Path

RUN = Path(os.environ['FULL6_PPO_RUN'])
os.environ.setdefault('MUJOCO_GL', 'glfw')
sys.path.insert(0, str(RUN))
import train_targetnode_full6_workspace_eval13 as trainer
import mujoco
import numpy as np
from PIL import Image

root = RUN/'videos/step_003538944'
env = trainer.CourseEnv(1000, evaluation=True)
env.reset(seed=20260804)
base = env.env.base_env
camera = mujoco.MjvCamera()
camera.lookat[:] = (.52, .06, .13)
camera.distance, camera.azimuth, camera.elevation = 1.45, 135, -32
option = mujoco.MjvOption()
option.geomgroup[3] = 0
base.model.vis.global_.offwidth = 1280
base.model.vis.global_.offheight = 960
renderer = mujoco.Renderer(base.model, height=960, width=1280, max_geom=10000)
report = dict(method='Saved actions and joint states only; no policy inference or env.step',
    finger_qpos_addresses=base.finger_qpos_adr.tolist(), episodes=[])
for directory in sorted(root.glob('seed_*')):
    source = np.load(directory/'trajectory.npz')
    meta = json.loads((directory/'metadata.json').read_text())
    apertures = []
    for state in source['states']:
        mujoco.mj_setState(base.model,base.data,state,mujoco.mjtState.mjSTATE_FULLPHYSICS)
        apertures.append(base.finger_aperture)
    closed = False
    command_states = []
    transitions = []
    for index, action in enumerate(source['actions']):
        prior = closed
        if action[4] <= -.35:
            closed = True
        elif action[4] >= .35:
            closed = False
        command_states.append(closed)
        if closed != prior:
            transitions.append(dict(time_s=round(index*.02,2), command='CLOSE' if closed else 'OPEN',
                normalized_action=float(action[4]), aperture_before_mm=1000*apertures[index],
                aperture_after_20ms_mm=1000*apertures[index+1]))
    open_periods = []
    for index, event in enumerate(transitions):
        if event['command'] == 'OPEN':
            end = transitions[index+1]['time_s'] if index+1<len(transitions) else 15.
            lo,hi = round(event['time_s']/.02),round(end/.02)
            open_periods.append(dict(start_s=event['time_s'],duration_s=round(end-event['time_s'],2),
                peak_aperture_mm=1000*max(apertures[lo:hi+1])))
    report['episodes'].append(dict(seed=meta['seed'], transitions=transitions,open_periods=open_periods,
        actual_aperture_mm_range=[1000*min(apertures),1000*max(apertures)]))
    print(meta['seed'], 'OPEN transitions',len(open_periods),'sample',open_periods[:4],flush=True)

directory = root/'seed_20260804'
meta = json.loads((directory/'metadata.json').read_text())
source = np.load(directory/'trajectory.npz')
position = np.array(meta['wrist_camera']['local_position'])
rotation = np.array(meta['wrist_camera']['local_rotation'])
projections = []
for index in (0,40,45,250):
    mujoco.mj_setState(base.model,base.data,source['states'][index],mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(base.model,base.data)
    hand = base.data.xmat[base.hand_id].reshape(3,3)
    for mode in ('original','fov_fixed','review'):
        renderer.update_scene(base.data,camera=camera,scene_option=option)
        for cam in renderer.scene.camera:
            cam.pos[:] = base.data.xpos[base.hand_id]+hand@position
            cam.forward[:] = -(hand@rotation)[:,2]
            cam.up[:] = (hand@rotation)[:,1]
            if mode != 'original':
                extent = cam.frustum_near*np.tan(np.radians(base.config.dynamicvla_camera_fovy)/2)
                cam.frustum_top = extent
                cam.frustum_bottom = -extent
            if mode == 'review':
                forward = np.array(base.GRASP_CENTER_LOCAL)-position
                forward /= np.linalg.norm(forward)
                up = rotation[:,1]-forward*np.dot(rotation[:,1],forward)
                up /= np.linalg.norm(up)
                cam.forward[:] = hand@forward
                cam.up[:] = hand@up
        path = root/f'wrist_{mode}_{index:04d}.png'
        Image.fromarray(renderer.render()).save(path)
        cam = renderer.scene.camera[0]
        projections.append(dict(frame=index,mode=mode,fovy_degrees=float(np.degrees(2*np.arctan(cam.frustum_top/cam.frustum_near))),
            frustum_near=cam.frustum_near,frustum_width=cam.frustum_width,file=path.name))
report['wrist_preview'] = projections
(root/'gripper_camera_audit.json').write_text(json.dumps(report,indent=2))
renderer.close()
env.close()
print('AUDIT COMPLETE',flush=True)
