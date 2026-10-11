"""Single-state intervention; uses the original recording's frozen implementation."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/scripted_seven_default_20261009/source'
sys.path.insert(0, str(SOURCE / 'src'))
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig

OUT = ROOT / 'outputs/rule809_deepen_20261010'
EP = ROOT / 'outputs/scripted_seven_default_20261009/dev_combined_l1_nominal_x075_v1/seed_20260809'
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf-8')

def state(env, spec=SPEC):
    value = np.empty(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data, value, spec)
    return value

def probe():
    OUT.mkdir(parents=True, exist_ok=True)
    save('status.json', dict(status='simulating'))
    meta = json.loads((EP / 'metadata.json').read_text())
    original = np.load(EP / 'trajectory.npz')
    config = dict(meta['env_config'], replay_bank_dir=str(ROOT / 'outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks'))
    env = CableGraspEnv(EnvConfig(**config))
    try:
        env.reset(seed=20260809)
        policy = DynamicCableGraspPolicy(env, PolicyConfig(**meta['policy_config']))
        i, prefix_error = 0, 0.
        while True:
            prefix_error = max(prefix_error, float(np.abs(state(env, mujoco.mjtState.mjSTATE_FULLPHYSICS) - original['states'][i]).max()))
            if env.data.time >= 2.98 - 1e-6:
                break
            env.step(policy.action())
            i += 1
        assert prefix_error < 1e-8, prefix_error
        start = env.hand_position.copy()
        nearest, _, segment, alpha = policy._nearest_cable_point(start)
        goal = start.copy()
        goal[2] = nearest[2]
        branch_time = float(env.data.time)
        states, times, metrics, targets, actions = [], [], [], [], []
        stage = 'DEEPEN'
        lift_time, lift_start = None, None
        def frame():
            point, distance, k, a = policy._nearest_cable_point(env.hand_position)
            info = env.info()
            states.append(state(env)); times.append(float(env.data.time)); targets.append(goal.copy())
            metrics.append(dict(time=times[-1], stage=stage, hand_xyz=env.hand_position.copy(), nearest_xyz=point,
                nearest_segment=int(k), centerline_distance_m=float(distance), aperture_m=float(env.finger_aperture),
                grasp_confirmed=bool(env.grasp_confirmed), grasped_body=info.get('grasped_body_id'),
                lifted_fraction=info.get('lifted_fraction'), success_hold_s=float(env.success_hold), success=bool(env.ever_success)))
        frame()
        while env.data.time < branch_time + 7.:
            now = float(env.data.time)
            if stage == 'DEEPEN':
                desired = goal.copy()
                if (env.hand_position[2] <= goal[2] + .002 and now - branch_time >= .2) or now - branch_time >= 1.:
                    stage = 'LIFT'
                    lift_start = env.hand_position.copy(); lift_time = now
            if stage in ('LIFT', 'HOLD'):
                blend = policy._smoothstep((now - lift_time) / policy.config.lift_seconds)
                desired = lift_start + np.array([0., 0., policy.config.lift_distance * blend])
                if now - lift_time >= policy.config.lift_seconds:
                    stage = 'HOLD'
            action = policy._ik_action(desired, policy.HOLD_GRIPPER_CTRL)
            _, _, terminated, truncated, info = env.step(action)
            actions.append(action.copy()); frame()
            if terminated or truncated:
                break
        np.savez_compressed(OUT / 'trajectory.npz', states=np.asarray(states), frame_times=times, targets=targets, native_actions=actions, state_spec=int(SPEC))
        mujoco.mj_saveModel(env.model, str(OUT / 'model.mjb'), None)
        save('metrics.json', metrics)
        lift_index = next((j for j, r in enumerate(metrics) if r['stage'] == 'LIFT'), len(metrics)-1)
        summary = dict(status='simulation_complete', source='scripted_seven_default_20261009', scene=meta['scene']['name'], seed=20260809,
            branch_time_s=branch_time, prefix_max_state_difference=prefix_error, intervention='Keep jaws closed; hold horizontal target; lower grasp center to initial nearest cable center height, for at most 1 s; then lift 0.22 m using original 2.4 s lift schedule, and hold. No reopening, retries, lateral retargeting, or changes to physics/limits/success criterion.',
            branch_hand_xyz=start, downward_target_xyz=goal, start_center_height_gap_m=float(start[2]-nearest[2]),
            reached_hand_xyz_before_lift=metrics[max(0,lift_index-1)]['hand_xyz'], lift_start_time_s=lift_time,
            min_hand_z_during_deepen_m=min(r['hand_xyz'][2] for r in metrics if r['stage']=='DEEPEN'),
            task_success=bool(env.ever_success), end_time_s=float(env.data.time), final_info=env.info(), grasp_break_history=env.grasp_break_history)
        save('summary.json', summary)
        save('metadata.json', dict(source_metadata=meta, actual_env_config=config, protocol=summary['intervention'], branch_time_s=branch_time, prefix_max_state_difference=prefix_error, renderer_camera='project rl/evaluate.py camera'))
        save('status.json', dict(status='simulation_complete', task_success=bool(env.ever_success), summary='summary.json'))
        print(json.dumps({k: v for k,v in summary.items() if k not in ('final_info','grasp_break_history')}, ensure_ascii=False, indent=2, default=lambda x:x.tolist()))
    finally:
        env.close()

def render():
    import subprocess
    from PIL import Image, ImageDraw, ImageFont
    sys.path.insert(0, str(SOURCE / 'tools/recording'))
    from record_scripted_scene_family import connector, marker, RECTANGLE, verify_video
    save('status.json', dict(status='rendering'))
    model = mujoco.MjModel.from_binary_path(str(OUT / 'model.mjb'))
    data = mujoco.MjData(model)
    trajectory = np.load(OUT / 'trajectory.npz')
    metrics = json.loads((OUT / 'metrics.json').read_text())
    metadata = json.loads((OUT / 'metadata.json').read_text())
    config = metadata['actual_env_config']
    model.vis.global_.offwidth = max(1280,model.vis.global_.offwidth)
    model.vis.global_.offheight = max(960,model.vis.global_.offheight)
    renderer = mujoco.Renderer(model, height=960, width=1280, max_geom=10000)
    camera = mujoco.MjvCamera(); camera.lookat[:] = [.55,0.,.30]
    camera.distance, camera.azimuth, camera.elevation = 1.65,135,-25
    option = mujoco.MjvOption(); option.geomgroup[3]=0
    site = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'cable_marker')
    if site >= 0: model.site_rgba[site,3]=0
    hand_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'link7' if config['robot']=='nero' else 'hand')
    # Actual robot hand body comes from the same env implementation.
    if hand_id < 0:
        raise RuntimeError('Cannot identify robot hand')
    rotation=np.empty(9); mujoco.mju_quat2Mat(rotation,np.asarray(config['dynamicvla_wrist_camera_quat'])); rotation=rotation.reshape(3,3)
    wrist=np.asarray(config['dynamicvla_wrist_camera_pos'])
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    writers={view: subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x960','-framerate','50','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','16','-threads','1','-g','12','-keyint_min','12','-sc_threshold','0','-bf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/f'{view}.mp4')],stdin=subprocess.PIPE) for view in ('global','wrist')}
    try:
        for i, recorded in enumerate(trajectory['states']):
            mujoco.mj_setState(model,data,recorded,SPEC); mujoco.mj_forward(model,data)
            for view, writer in writers.items():
                renderer.update_scene(data,camera=camera,scene_option=option)
                if view=='wrist':
                    hand=data.xmat[hand_id].reshape(3,3)
                    for cam in renderer.scene.camera:
                        cam.pos[:]=data.xpos[hand_id]+hand@wrist
                        cam.forward[:]=-(hand@rotation)[:,2]; cam.up[:]=(hand@rotation)[:,1]
                        extent=cam.frustum_near*np.tan(np.radians(config['dynamicvla_camera_fovy'])/2)
                        cam.frustum_top,cam.frustum_bottom=extent,-extent
                for a,b in zip(RECTANGLE,np.roll(RECTANGLE,-1,axis=0)): connector(renderer.scene,a,b)
                marker(renderer.scene,trajectory['targets'][i],.006,[1.,.72,.05,1.])
                img=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(img)
                draw.rectangle((12,12,1268,110),fill='#eef3fa')
                draw.text((26,18),f"809 BRANCH FROM 2.98s | t={trajectory['frame_times'][i]:.2f}s | {metrics[i]['stage']} | {view.upper()}",font=font,fill='#16324d')
                draw.text((26,50),f"Keep closed, deeper then lift | success {metrics[i]['success']} | aperture {metrics[i]['aperture_m']*1000:.1f} mm",font=font,fill='#16324d')
                draw.text((26,82),'BLUE: workspace | YELLOW: initial downward target (fixed XY)',font=font,fill='#16324d')
                writer.stdin.write(np.asarray(img).tobytes())
                if i==0: img.save(OUT/f'{view}_poster.png')
    finally:
        renderer.close()
        for writer in writers.values():
            writer.stdin.close()
            if writer.wait()!=0: raise RuntimeError('Encoder failed')
    verification={view:verify_video(OUT/f'{view}.mp4',len(metrics)) for view in writers}
    save('video_verification.json',verification)
    summary=json.loads((OUT/'summary.json').read_text()); summary['status']='complete'; save('summary.json',summary)
    save('status.json',dict(status='complete',task_success=summary['task_success'],summary='summary.json',global_video='global.mp4',wrist_video='wrist.mp4'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--render',action='store_true'); args=parser.parse_args()
    try:
        render() if args.render else probe()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True); save('status.json',dict(status='failed',error=str(error))); raise
