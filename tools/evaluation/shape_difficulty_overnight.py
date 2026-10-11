"""Staged whole-shape scripted grasp difficulty study (fixed vertical gripper)."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import mujoco
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.control import configure_profile
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig
from panda_cable_grasp.scenarios.registry import get_scenario

OUT = ROOT/'outputs/shape_difficulty_overnight_20261010'
BANK_DIR = ROOT/'outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks'
BANK = BANK_DIR/'replay_src_shape_nominal_v1.npz'
BANK_SHA256 = 'e42b22a1ab3d2b5a9674fd772a2954794dcc87c3d858bf8c9faff91df1b85e73'
FRACTIONS = (.35,.50,.65)
LAYOUTS = ((.73,-.04,-4.0),(.75,0.,0.),(.77,.04,4.0))
MODES = ('middle','middle_geometry','whole_geometry')
MEASUREMENT_VERSION = 'capsule_axes_preunlock_contact_v2'
LEGACY_MEASUREMENT_VERSION = 'node_centers_window_0p84_v1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def save_json(path, data):
    p=Path(path); t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8'); t.replace(p)

def merge_status(path, updates):
    """Serialize coordinator/guardian read-modify-write updates on Windows and POSIX."""
    path=Path(path); lock_path=path.with_suffix(path.suffix+'.lock')
    with lock_path.open('a+b') as lock:
        lock.seek(0,os.SEEK_END)
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try:
            try: current=json.loads(path.read_text(encoding='utf-8-sig'))
            except (FileNotFoundError,json.JSONDecodeError): current={}
            current.update(updates)
            save_json(path,current)
        finally:
            lock.seek(0)
            if os.name=='nt': msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def geometry_features(xy):
    xy=np.asarray(xy,dtype=float); d=np.diff(xy,axis=0); lens=np.linalg.norm(d,axis=1); total=float(lens.sum())
    centered=xy-xy.mean(0); cov=centered.T@centered/max(1,len(xy)); eig=np.linalg.eigvalsh(cov)[::-1]
    bbox=np.ptp(xy,axis=0); radius=float(np.sqrt(np.mean(np.sum(centered**2,axis=1))))
    heading=np.arctan2(d[:,1],d[:,0]); turn=np.arctan2(np.sin(np.diff(heading)),np.cos(np.diff(heading)))
    curvature=np.abs(turn)/np.maximum((lens[:-1]+lens[1:])*.5,1e-8)
    sep=[]
    for i in range(len(xy)):
        for j in range(i+7,len(xy)):
            sep.append(float(np.linalg.norm(xy[i]-xy[j])))
    sep=np.asarray(sep)
    in_work= ((xy[:,0]>=.30)&(xy[:,0]<=.82)&(xy[:,1]>=-.43)&(xy[:,1]<=.43))
    return dict(global_bbox_diag=float(np.linalg.norm(bbox)), global_convex_proxy=float(bbox[0]*bbox[1]),
        global_pca_compactness=float(eig[1]/max(eig[0],1e-9)), global_radius_gyration=radius,
        global_path_chord_ratio=total/max(float(np.linalg.norm(xy[-1]-xy[0])),1e-8),
        global_total_turn=float(np.abs(turn).sum()), local_curvature_median=float(np.median(curvature)),
        local_curvature_p90=float(np.quantile(curvature,.9)), local_curvature_max=float(curvature.max(initial=0)),
        nonadjacent_clearance_min=float(sep.min(initial=.5)), nonadjacent_clearance_p10=float(np.quantile(sep,.1)),
        nonadjacent_close_pairs_40mm=int(np.sum(sep<.04)), nonadjacent_close_pairs_80mm=int(np.sum(sep<.08)),
        task_center_x=float(xy[:,0].mean()),task_center_y=float(xy[:,1].mean()),task_workspace_fraction=float(in_work.mean()),
        task_home_distance_proxy=float(np.min(np.linalg.norm(xy-np.array([.45,0.]),axis=1))))

def segment_distance(p1,q1,p2,q2):
    u=q1-p1; v=q2-p2; w=p1-p2; a=float(u@u); b=float(u@v); c=float(v@v); d=float(u@w); e=float(v@w); den=a*c-b*b
    s=0. if den<1e-12 else float(np.clip((b*e-c*d)/den,0.,1.)); t=float(np.clip((b*s+e)/max(c,1e-12),0.,1.)); s=float(np.clip((b*t-d)/max(a,1e-12),0.,1.))
    return float(np.linalg.norm((p1+s*u)-(p2+t*v)))

def max_nonadjacent_penetration(points,radius):
    worst=0.
    for i in range(len(points)-1):
        for j in range(i+3,len(points)-1):
            worst=max(worst,2.*radius-segment_distance(points[i],points[i+1],points[j],points[j+1]))
    return max(0.,worst)

def episode_identity(source_seed,fraction,layout,mode,scene,angle=60.,measurement_version=MEASUREMENT_VERSION):
    center_x,yaw=LAYOUTS[int(layout)][0],LAYOUTS[int(layout)][2]
    scenario=get_scenario(scene)
    policy_contract=dict(target_selection_mode=mode,follow_target_tangent=False,
        target_min_crossing_angle_deg=float(angle),corridor_clearance=True,intercept_segment_tolerance=4)
    env_contract=dict(robot='nero',episode_seconds=15.,controller='enhanced6_bounded_dls_v1',
        table_execution_guard_enabled=True,arm_motion_start_delay=.82,initial_shape_time_range=(float(fraction),float(fraction)),
        initial_cable_center_x=center_x,initial_cable_yaw_deg=yaw,scene=scenario.asdict(),
        bank_sha256=BANK_SHA256,selector_implementation='whole-shape-selector-study-v1',
        measurement_version=measurement_version)
    config_hash=hashlib.sha256(json.dumps(dict(policy=policy_contract,environment=env_contract),
        sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()
    identity_payload=dict(source_seed=int(source_seed),snapshot_fraction=float(fraction),layout=int(layout),
        center_x=float(center_x),yaw_deg=float(yaw),selector=mode,scene=scene,config_hash=config_hash)
    run_identity=hashlib.sha256(json.dumps(identity_payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return run_identity,config_hash

def capsule_axis_segment(env, geom_id):
    """Return the transformed centerline of a MuJoCo capsule and its radius."""
    if int(env.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        raise ValueError(f'Cable collision geom {geom_id} is not a capsule')
    center=np.asarray(env.data.geom_xpos[geom_id],dtype=float)
    rotation=np.asarray(env.data.geom_xmat[geom_id],dtype=float).reshape(3,3)
    half_length=float(env.model.geom_size[geom_id,1])
    axis=rotation[:,2]
    return center-half_length*axis,center+half_length*axis,float(env.model.geom_size[geom_id,0])

def segment_distance_robust(p1,q1,p2,q2):
    """Minimum distance for finite 3-D segments, including parallel/degenerate cases."""
    p1=np.asarray(p1,dtype=float);q1=np.asarray(q1,dtype=float)
    p2=np.asarray(p2,dtype=float);q2=np.asarray(q2,dtype=float)
    u=q1-p1;v=q2-p2
    a=float(u@u);c=float(v@v);b=float(u@v);w=p1-p2
    d=float(u@w);e=float(v@w);den=a*c-b*b
    candidates=[]
    # Any boundary optimum has at least one endpoint. Project all endpoints
    # onto the opposite finite segment; also include the interior stationary pair.
    for point, start, direction, length2 in ((p1,p2,v,c),(q1,p2,v,c),(p2,p1,u,a),(q2,p1,u,a)):
        t=0. if length2<=1e-24 else float(np.clip((point-start)@direction/length2,0.,1.))
        candidates.append(float(np.linalg.norm(point-(start+t*direction))))
    if den>1e-20*max(a*c,1.):
        s=(b*e-c*d)/den;t=(a*e-b*d)/den
        if 0.<=s<=1. and 0.<=t<=1.:
            candidates.append(float(np.linalg.norm((p1+s*u)-(p2+t*v))))
    return min(candidates)

def max_nonadjacent_capsule_penetration(env,neighbor_gap=2):
    """Measure exact capsule surface overlap from MuJoCo's current geom transforms."""
    capsules=[capsule_axis_segment(env,int(g)) for g in env.cable_geom_ids]
    worst=0.
    for i,(a0,a1,ra) in enumerate(capsules):
        for j in range(i+neighbor_gap+1,len(capsules)):
            b0,b1,rb=capsules[j]
            worst=max(worst,ra+rb-segment_distance_robust(a0,a1,b0,b1))
    return max(0.,float(worst))

def summarize_preunlock_motion(samples,unlock_time=.82):
    """Summarize cable displacement and terminal-window speed before arm unlock.

    samples is an ordered sequence of (time_s, cable_node_positions) pairs.
    The reported terminal windows distinguish early settling from motion that
    is still present immediately before the first unlocked control step.
    """
    if not samples:
        return dict(max_displacement_m=None,max_displacement_node=None,max_displacement_time_s=None,
            window_200ms_max_node_displacement_m=None,window_200ms_max_node_speed_m_s=None,
            window_100ms_max_node_displacement_m=None,window_100ms_max_node_speed_m_s=None,
            last_sample_time_s=None,window_covered=False)
    t0=np.asarray(samples[0][1],dtype=float)
    max_amount=-1.;max_node=0;max_time=float(samples[0][0])
    for t,positions in samples:
        delta=np.linalg.norm(np.asarray(positions,dtype=float)-t0,axis=1)
        node=int(np.argmax(delta));amount=float(delta[node])
        if amount>max_amount:max_amount,max_node,max_time=amount,node,float(t)
    out=dict(max_displacement_m=max(0.,max_amount),max_displacement_node=max_node,
        max_displacement_time_s=max_time,last_sample_time_s=float(samples[-1][0]),
        window_covered=abs(float(samples[-1][0])-unlock_time)<=1e-8)
    for window,label in ((.20,'window_200ms'),(.10,'window_100ms')):
        selected=[(float(t),np.asarray(p,dtype=float)) for t,p in samples if float(t)>=unlock_time-window-1e-9]
        node_disp=np.zeros(len(t0));node_speed=np.zeros(len(t0))
        for (ta,pa),(tb,pb) in zip(selected,selected[1:]):
            dt=tb-ta
            if dt<=1e-12: continue
            node_disp=np.maximum(node_disp,np.linalg.norm(pb-pa,axis=1))
            node_speed=np.maximum(node_speed,np.linalg.norm(pb-pa,axis=1)/dt)
        out[f'{label}_max_node_displacement_m']=float(node_disp.max(initial=0.))
        out[f'{label}_max_node_speed_m_s']=float(node_speed.max(initial=0.))
    return out

def robot_cable_contact_snapshot(env):
    cable=set(map(int,env.cable_geom_ids));robot=set(map(int,env.geometric_safety_geom_ids))
    pairs=set();physical=set()
    for contact in env.data.contact[:env.data.ncon]:
        a,b=int(contact.geom1),int(contact.geom2)
        if (a in cable and b in robot) or (b in cable and a in robot):
            pair=tuple(sorted((a,b)));pairs.add(pair)
            if float(contact.dist)<=0.:physical.add(pair)
    return pairs,physical

def run_job(job):
    """One actual independent static scene execution; no shadow or candidate rollout."""
    measurement_version=job.get('measurement_version',LEGACY_MEASUREMENT_VERSION)
    exact_measurements=measurement_version==MEASUREMENT_VERSION
    scene=get_scenario(job['scene'])
    overrides=scene.to_env_overrides()
    overrides.update(initial_shape_time_range=(job['fraction'],job['fraction']),
        initial_cable_center_x=job['center_x'],initial_cable_yaw_deg=job['yaw_deg'],
        arm_motion_start_delay=.82)
    env_cfg=EnvConfig(robot='nero',seed=job['sim_seed'],target_selection='middle',episode_seconds=15.,
        scenario_name=scene.name,scenario_id=scene.scenario_id,scenario_split=scene.split.value,
        replay_bank_dir=str(BANK_DIR),replay_seed_fallback='error',**overrides)
    env_cfg=configure_profile(env_cfg,'enhanced6_bounded_dls_v1'); env_cfg.table_execution_guard_enabled=True
    env=CableGraspEnv(env_cfg)
    try:
        obs, info=env.reset(seed=job['source_seed'])
        penetration=(max_nonadjacent_capsule_penetration(env) if exact_measurements else
            max_nonadjacent_penetration(env.data.xpos[env.cable_ids].copy(),float(env.cable_radius)))
        if penetration>.005:
            return dict(job_id=job['job_id'],source_seed=job['source_seed'],source_index=job['source_index'],fraction=job['fraction'],
                layout=job['layout'],scene=scene.name,selector=job['mode'],filtered=True,max_nonadjacent_penetration_m=penetration,
                measurement_version=measurement_version,penetration_measurement_version=measurement_version,success=None,passive_motion_m=None,
                preunlock_window_covered=False,relaxation_gt_5mm=True,run_identity=job['run_identity'],
                configuration_hash=job['configuration_hash'],feature_geometry=job['features'])
        pconf=PolicyConfig(target_selection_mode=job['mode'],follow_target_tangent=False,
            target_min_crossing_angle_deg=job['min_crossing_angle_deg'],corridor_clearance=True,
            intercept_segment_tolerance=4)
        policy=DynamicCableGraspPolicy(env,pconf)
        xyz0=env.data.xpos[env.cable_ids].copy(); movement=.0
        movement_node=0; movement_time=0.; preunlock_last_time=None; preunlock_samples=0
        preunlock_contact_samples=0; preunlock_physical_contact_samples=0
        preunlock_contact_pairs=set(); preunlock_physical_pairs=set()
        preunlock_positions=[]
        unlock_time=float(env.config.arm_motion_start_delay)
        def observe_preunlock():
            nonlocal movement,movement_node,movement_time,preunlock_last_time,preunlock_samples
            nonlocal preunlock_contact_samples,preunlock_physical_contact_samples
            if not exact_measurements or float(env.data.time)>unlock_time+1e-9:
                return
            t=float(env.data.time)
            if preunlock_last_time is not None and t<=preunlock_last_time+1e-12:
                return
            displacement=np.linalg.norm(env.data.xpos[env.cable_ids]-xyz0,axis=1)
            node=int(np.argmax(displacement)); amount=float(displacement[node])
            if amount>movement:
                movement,movement_node,movement_time=amount,node,t
            pairs,physical=robot_cable_contact_snapshot(env)
            preunlock_contact_samples+=int(bool(pairs));preunlock_physical_contact_samples+=int(bool(physical))
            preunlock_contact_pairs.update(pairs);preunlock_physical_pairs.update(physical)
            preunlock_positions.append((t,env.data.xpos[env.cable_ids].copy()))
            preunlock_samples+=1;preunlock_last_time=t
        if exact_measurements:
            observe_preunlock()
        saved_states=[]
        if job.get('capture_trace'):
            state=np.empty(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_FULLPHYSICS))
            mujoco.mj_getState(env.model,env.data,state,mujoco.mjtState.mjSTATE_FULLPHYSICS);saved_states.append(state.copy())
        while not policy.finished:
            action=policy.action(); _,_,terminated,truncated,info=env.step(action)
            observe_preunlock()
            if job.get('capture_trace'):
                state=np.empty(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_FULLPHYSICS))
                mujoco.mj_getState(env.model,env.data,state,mujoco.mjtState.mjSTATE_FULLPHYSICS);saved_states.append(state.copy())
            if not exact_measurements and float(env.data.time)<=.84:
                movement=max(movement,float(np.max(np.linalg.norm(env.data.xpos[env.cable_ids]-xyz0,axis=1))))
            if terminated or truncated: break
        final=env.info()
        if env.ever_success and policy.result=='running': policy.result='success'
        preunlock_summary=(summarize_preunlock_motion(preunlock_positions,unlock_time) if exact_measurements else {})
        row=dict(job_id=job['job_id'],source_seed=job['source_seed'],source_index=job['source_index'],
            fraction=job['fraction'],layout=job['layout'],scene=scene.name,selector=job['mode'],
            measurement_version=measurement_version,
            min_crossing_angle_deg=job['min_crossing_angle_deg'],seed=job['sim_seed'],
            success=bool(policy.result=='success'),ever_success=bool(env.ever_success),
            failure_reason=str(policy.last_attempt_failure or final.get('termination_reason') or policy.result),
            selected_index=int(policy.selected_target_index),selected_crossing_deg=float(policy.selected_target_crossing_angle_deg),
            selected_corridor_clearance=float(policy.selected_target_corridor_clearance),
            initial_target_xyz=env.data.xpos[policy.selected_target_body_id].tolist(),
            passive_motion_m=movement,relaxation_gt_5mm=bool(movement>.005),filtered=False,max_nonadjacent_penetration_m=penetration,
            penetration_measurement_version=measurement_version,
            preunlock_window_covered=bool(exact_measurements and preunlock_summary.get('window_covered',False)),
            preunlock_max_motion_m=(preunlock_summary.get('max_displacement_m') if exact_measurements else None),
            preunlock_max_motion_node=(preunlock_summary.get('max_displacement_node') if exact_measurements else None),
            preunlock_max_motion_time_s=(preunlock_summary.get('max_displacement_time_s') if exact_measurements else None),preunlock_sample_count=(preunlock_samples if exact_measurements else None),
            preunlock_window_200ms_max_node_displacement_m=preunlock_summary.get('window_200ms_max_node_displacement_m'),
            preunlock_window_200ms_max_node_speed_m_s=preunlock_summary.get('window_200ms_max_node_speed_m_s'),
            preunlock_window_100ms_max_node_displacement_m=preunlock_summary.get('window_100ms_max_node_displacement_m'),
            preunlock_window_100ms_max_node_speed_m_s=preunlock_summary.get('window_100ms_max_node_speed_m_s'),
            preunlock_last_sample_time_s=preunlock_summary.get('last_sample_time_s'),
            preunlock_robot_cable_contact_sample_count=(preunlock_contact_samples if exact_measurements else None),
            preunlock_robot_cable_physical_contact_sample_count=(preunlock_physical_contact_samples if exact_measurements else None),
            preunlock_robot_cable_contact_pairs=sorted(preunlock_contact_pairs) if exact_measurements else None,
            preunlock_robot_cable_physical_contact_pairs=sorted(preunlock_physical_pairs) if exact_measurements else None,
            retries=int(policy.retry_count),time_s=float(env.data.time),grasp_confirmed=bool(env.grasp_confirmed),
            grasped_body=final.get('grasped_body_id'),
            close_blocked_segment=int(policy.close_gate_blocked_segment),
            close_blocked_distance=int(policy.close_gate_blocked_distance),
            close_blocked_vertical=int(policy.close_gate_blocked_vertical),
            close_blocked_box=int(policy.close_gate_blocked_box),
            max_recover_xy_drift=float(policy.max_recover_xy_drift),
            run_identity=job['run_identity'],configuration_hash=job['configuration_hash'],feature_geometry=job['features'])
        if saved_states:
            trace=OUT/'traces'/f"seed_{job['source_seed']}_frac_{int(job['fraction']*100):02d}_layout_{job['layout']}.npz"
            trace.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(trace,states=np.asarray(saved_states),frame_times=np.arange(len(saved_states),dtype=float)*env.config.control_dt,
                job_json=np.asarray(json.dumps(job)),selected_rule=np.asarray(job['mode']))
            row['trace_path']=str(trace.relative_to(ROOT))
        return row
    finally: env.close()

def wilson(k,n,z=1.96):
    if not n:return [None,None]
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,c-h),min(1,c+h)]

def render_representative(job, trace_path, dest):
    """Render saved full-physics states from the experiment's named cameras."""
    import subprocess
    from PIL import Image
    scene=get_scenario(job['scene']); overrides=scene.to_env_overrides()
    overrides.update(initial_shape_time_range=(job['fraction'],job['fraction']),initial_cable_center_x=job['center_x'],
        initial_cable_yaw_deg=job['yaw_deg'],arm_motion_start_delay=.82)
    cfg=EnvConfig(robot='nero',seed=job['sim_seed'],target_selection='middle',episode_seconds=15.,
        scenario_name=scene.name,scenario_id=scene.scenario_id,scenario_split=scene.split.value,replay_bank_dir=str(BANK_DIR),
        replay_seed_fallback='error',dynamicvla_cameras_enabled=True,**overrides)
    cfg=configure_profile(cfg,'enhanced6_bounded_dls_v1');cfg.table_execution_guard_enabled=True
    env=CableGraspEnv(cfg);env.reset(seed=job['source_seed'])
    width,height,fps=1280,960,50; dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    env.model.vis.global_.offwidth=max(width,env.model.vis.global_.offwidth);env.model.vis.global_.offheight=max(height,env.model.vis.global_.offheight)
    renderer=mujoco.Renderer(env.model,height=height,width=width,max_geom=10000)
    cams={'global':cfg.dynamicvla_opst_camera_name,'wrist':cfg.dynamicvla_wrist_camera_name}
    trace=np.load(trace_path); states=trace['states'];times=trace['frame_times']
    writers={v:subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{width}x{height}','-framerate',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast',
        '-crf','16','-threads','1','-g','12','-keyint_min','12','-sc_threshold','0','-bf','0','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(dest/f'{v}.mp4')],stdin=subprocess.PIPE) for v in cams}
    try:
        for index,state in enumerate(states):
            mujoco.mj_setState(env.model,env.data,state,mujoco.mjtState.mjSTATE_FULLPHYSICS);mujoco.mj_forward(env.model,env.data)
            for view,camera in cams.items():
                renderer.update_scene(env.data,camera=camera);image=Image.fromarray(renderer.render())
                if index==0:image.save(dest/f'{view}_poster.png')
                writers[view].stdin.write(np.asarray(image).tobytes())
        for p in writers.values():p.stdin.close()
        if any(p.wait()!=0 for p in writers.values()):raise RuntimeError('ffmpeg failed for representative trace')
        meta=dict(scene=scene.name,seed=job['source_seed'],fraction=job['fraction'],layout=job['layout'],selector=job['mode'],
            trace=str(Path(trace_path).relative_to(ROOT)),frames=len(states),frame_times_verified=bool(np.allclose(np.diff(times),.02,atol=1e-9)),
            recording=dict(width=width,height=height,fps=fps,state_replay=True,source='saved MuJoCo mjSTATE_FULLPHYSICS states',
                cameras={'source':'src/panda_cable_grasp/env/environment.py: EnvConfig camera rig',
                    'global':dict(name=cfg.dynamicvla_opst_camera_name,position_world=list(cfg.dynamicvla_opst_camera_pos),quaternion_world=list(cfg.dynamicvla_opst_camera_quat),fovy=cfg.dynamicvla_camera_fovy),
                    'wrist':dict(name=cfg.dynamicvla_wrist_camera_name,position_in_hand=list(cfg.dynamicvla_wrist_camera_pos),quaternion_in_hand=list(cfg.dynamicvla_wrist_camera_quat),fovy=cfg.dynamicvla_camera_fovy)}))
        save_json(dest/'metadata.json',meta)
        return meta
    finally:renderer.close();env.close()

def build_gallery(rows, chosen):
    import subprocess
    out=OUT/'videos'; enc=[]; bykey={(r['source_seed'],r['fraction']):r for r in rows if r.get('trace_path')}
    rates={(int(r['source_seed']),float(r['fraction'])):float(r['failure_rate']) for r in chosen}
    candidates=sorted(((rate,key,bykey.get(key)) for key,rate in rates.items() if key in bykey),key=lambda x:x[0])
    if len(candidates)<3:raise RuntimeError(f'Only {len(candidates)} saved holdout traces available; cannot build three representative videos')
    picks=[('easy',candidates[0]),('medium',candidates[len(candidates)//2]),('hard',candidates[-1])]
    gallery=[]
    for label,(rate,key,row) in picks:
        trace=ROOT/row['trace_path'];job=json.loads(np.load(trace)['job_json'].item())
        dest=out/label;meta=render_representative(job,trace,dest)
        meta.update(label=label,empirical_failure_rate=rate,success=bool(row['success']),seed=row['source_seed'])
        save_json(dest/'metadata.json',meta)
        for view in ('global','wrist'):
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames',
                '-show_entries','stream=width,height,avg_frame_rate,nb_frames,codec_name:frame=key_frame,best_effort_timestamp_time,pict_type',
                '-of','json',str(dest/f'{view}.mp4')]))
            stream,frames=probe['streams'][0],probe['frames'];times=np.array([float(x['best_effort_timestamp_time']) for x in frames]);keys=[i for i,x in enumerate(frames) if x['key_frame']]
            checks=dict(width=stream['width'],height=stream['height'],fps=stream['avg_frame_rate'],frames=int(stream['nb_frames']),
                keyframe_max_spacing=max(np.diff(keys),default=0),b_frames=any(x['pict_type']=='B' for x in frames),timestamps_20ms=bool(np.allclose(np.diff(times),.02,atol=1e-6)),
                sha256=digest(dest/f'{view}.mp4'))
            if (checks['width'],checks['height'],checks['fps'],checks['frames'])!=(1280,960,'50/1',len(times)) or checks['keyframe_max_spacing']>12 or checks['b_frames'] or not checks['timestamps_20ms']:
                raise RuntimeError(f'Video encoding verification failed: {dest}/{view}.mp4: {checks}')
            enc.append(dict(label=label,view=view,**checks))
        gallery.append(dict(label=label,scene=row['scene'],seed=row['source_seed'],failure_rate=rate,directory=f'videos/{label}'))
    save_json(OUT/'encoding_verification.json',enc)
    data=json.dumps(gallery,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>完整形状难度代表回合</title><style>body{font:16px system-ui;background:#eef2f7;color:#18334b;margin:24px}main{max-width:1500px;margin:auto}.bar{display:flex;gap:12px;align-items:center}section{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;background:#101820}article{background:white;padding:12px;border-radius:8px}input[type=range]{width:70%}</style><main><h1>完整形状难度代表回合</h1><p>所有画面由已保存的MuJoCo物理状态重渲染。颜色标签按本实验固定规则选择器下的经验失败率分位；仅适用于固定竖直夹爪与当前控制器。</p><div class="bar"><select id="case"></select><button id="play">同步播放</button><button id="pause">暂停</button><button id="restart">重播</button><button id="copy">复制回合时刻</button><span id="ref"></span><input id="seek" type="range" min="0" max="15" step="0.02" value="0"></div><section><article><h2>Global · dynamicvla_opst_camera</h2><video id="g" preload="auto"></video></article><article><h2>Wrist · dynamicvla_wrist_camera</h2><video id="w" preload="auto"></video></article></section><p>时间：<span id="clock">0.00</span>s · <a href="report.md">报告</a> · <a href="encoding_verification.json">编码验证</a></p></main><script id="data" type="application/json">'''+data+'''</script><script>const D=JSON.parse(document.getElementById('data').textContent),g=document.getElementById('g'),w=document.getElementById('w'),sel=document.getElementById('case'),seek=document.getElementById('seek');for(const x of D){let o=document.createElement('option');o.value=D.indexOf(x);o.textContent=`${x.label} · ${x.scene} · seed=${x.seed} · failure=${(100*x.failure_rate).toFixed(0)}%`;sel.append(o)}function load(){let x=D[+sel.value];for(const [v,k] of [[g,'global'],[w,'wrist']]){v.pause();v.removeAttribute('controls');v.src=x.directory+'/'+k+'.mp4';v.load()}seek.value=0}sel.onchange=load;document.getElementById('play').onclick=()=>[g,w].forEach(v=>v.play());document.getElementById('pause').onclick=()=>[g,w].forEach(v=>{v.pause();v.controls=true});document.getElementById('restart').onclick=()=>[g,w].forEach(v=>{v.currentTime=0;v.play()});seek.oninput=()=>[g,w].forEach(v=>{v.pause();v.currentTime=+seek.value});function tick(){let t=Math.max(g.currentTime||0,w.currentTime||0);if(!g.paused&&!w.paused){let target=Math.min(g.currentTime,w.currentTime);for(const v of [g,w])if(!v.seeking&&Math.abs(v.currentTime-target)>.06)v.currentTime=target}if(document.activeElement!==seek)seek.value=t;document.getElementById('clock').textContent=(+seek.value).toFixed(2);document.getElementById('ref').textContent=`shape_difficulty_overnight_20261010 | ${D[+sel.value].scene} | seed=${D[+sel.value].seed} | t=${(+seek.value).toFixed(2)}s`;requestAnimationFrame(tick)}document.getElementById('copy').onclick=()=>navigator.clipboard.writeText(document.getElementById('ref').textContent);load();tick()</script>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    return gallery,enc

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=6); ap.add_argument('--max-episodes',type=int,default=0)
    ap.add_argument('--prequalification-only',action='store_true',help='run only geometry and pre-unlock motion qualification')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    status_path=OUT/'status.json'; started=time.time()
    def status(stage,completed,total,message,**extra):
        now=time.time(); stamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        terminal=stage in {'complete','smoke_complete','prequalification_diagnostic_complete'}
        payload=dict(experiment='shape_difficulty_overnight_20261010',status='complete' if terminal else 'running',stage=stage,
            updated_at=stamp,heartbeat_at=stamp,progress_deadline=time.strftime('%Y-%m-%dT%H:%M:%S%z',time.localtime(now+3600)),
            progress_timeout_minutes=60,completed_episodes=completed,planned_episodes=total,
            progress_fraction=completed/max(1,total),pid=os.getpid(),coordinator_pid=os.getpid(),message=message,report=str(OUT/'report.md'))
        payload.update(extra)
        merge_status(status_path,payload)
    try:
        if digest(BANK)!=BANK_SHA256: raise RuntimeError('Replay bank SHA-256 mismatch')
        z=np.load(BANK); seeds=z['seeds'].astype(int); positions=z['positions_xy']; valid=z['valid_steps'].astype(int)
        audit=json.loads((ROOT/'outputs/static_snapshot_audit_20261010/audit.json').read_text(encoding='utf-8'))
        keep_seeds={int(r['seed']) for r in audit['rows'] if r['kind']=='complex_snapshot' and r['max_penetration_m']<=.005}
        eligible=[i for i,s in enumerate(seeds) if int(s) in keep_seeds]
        rng=np.random.default_rng(20261010); perm=rng.permutation(eligible); dev=set(perm[:8].tolist()); fit=set(perm[8:28].tolist()); hold=set(perm[28:].tolist())
        split={i:('development' if i in dev else 'fit' if i in fit else 'holdout') for i in range(len(seeds))}
        catalog=[]
        for i,s in enumerate(seeds):
            for f in FRACTIONS:
                step=min(max(int(f*valid[i]),0),valid[i]-1); feat=geometry_features(positions[i,step])
                catalog.append(dict(source_index=i,source_seed=int(s),split=split[i],fraction=f,frame=step,**feat))
        with (OUT/'shape_catalog.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(catalog[0]));w.writeheader();w.writerows(catalog)
        save_json(OUT/'protocol.json',dict(bank=str(BANK),bank_sha256=BANK_SHA256,bank_entries=len(seeds),eligible_after_audit_5mm=len(eligible),snapshots_per_source=3,
            measurement_version=MEASUREMENT_VERSION,penetration='transformed MuJoCo capsule axes and geom radii, finite segment distance; ignore index gap <=2',
            passive_motion='max per-node displacement through t=0.82s inclusive before first unlocked command; record sampled robot-cable contacts',
            trajectory_group_split={'development':sorted(int(seeds[i]) for i in dev),'fit':sorted(int(seeds[i]) for i in fit),'holdout':sorted(int(seeds[i]) for i in hold)},
            selectors=list(MODES),tangent=False,orientation='fixed vertical',controller='enhanced6_bounded_dls_v1',table_execution_guard=True,
            episode_seconds=15,settle_delay_s=.82,success='environment existing low-level criterion',
            snapshot_fractions=FRACTIONS,layouts=LAYOUTS,collision_filter_new_experiment='exclude from primary difficulty analysis if reconstructed nonadjacent penetration >5mm; source-bank XY cannot certify 3D crossing',
            prohibited_future_simulation='none; every selector sees only current scene geometry and each evaluation is an independent actual episode'))
        # Dev tuning compares the historical midpoint, existing middle geometry,
        # and full-span selectors. Holdout source trajectories are never used here.
        dev_jobs=[]
        serial=0
        def add(target,i,f,layout_id,mode,scene='dev_static_midshape_v1_x075_v1',angle=60.):
            nonlocal serial
            feature=next(r for r in catalog if r['source_index']==i and r['fraction']==f)
            x,y,yaw=LAYOUTS[layout_id]; serial+=1
            run_id,config_hash=episode_identity(int(seeds[i]),f,layout_id,mode,scene,angle,MEASUREMENT_VERSION)
            target.append(dict(job_id=serial,run_identity=run_id,configuration_hash=config_hash,measurement_version=MEASUREMENT_VERSION,source_seed=int(seeds[i]),source_index=i,fraction=f,layout=layout_id,
                center_x=x,yaw_deg=yaw,mode=mode,min_crossing_angle_deg=angle,scene=scene,
                sim_seed=int(seeds[i]),features={k:v for k,v in feature.items() if k.startswith(('global_','local_','nonadjacent_','task_'))}))
        for i in sorted(dev):
            for f in FRACTIONS:
                for layout in range(3):
                    for mode in MODES: add(dev_jobs,i,f,layout,mode)
        preflight_jobs=[]
        for i in sorted(dev):
            add(preflight_jobs,i,.35,0,'middle_geometry')
        planned=(len(preflight_jobs)+2*len(dev)) if args.prequalification_only else len({j['run_identity'] for j in dev_jobs})+len(fit|hold)*len(FRACTIONS)*3*2
        if args.max_episodes: dev_jobs=dev_jobs[:args.max_episodes]; planned=len(dev_jobs)
        rows=[]
        legacy_episodes_path=OUT/'episodes_full.csv'
        episodes_path=OUT/'episodes_resumable.csv'
        v2_episodes_path=OUT/'episodes_measurement_v2.csv'
        source_paths=[]
        historical_source=episodes_path if episodes_path.exists() else legacy_episodes_path
        if historical_source.exists(): source_paths.append(historical_source)
        if v2_episodes_path.exists(): source_paths.append(v2_episodes_path)
        if source_paths:
            numeric={'job_id':int,'source_seed':int,'source_index':int,'fraction':float,'layout':int,'min_crossing_angle_deg':float,'seed':int,
                'selected_index':int,'selected_crossing_deg':float,'selected_corridor_clearance':float,'passive_motion_m':float,
                'max_nonadjacent_penetration_m':float,'retries':int,'time_s':float,'grasped_body':int,'close_blocked_segment':int,
                'close_blocked_distance':int,'close_blocked_vertical':int,'close_blocked_box':int,'max_recover_xy_drift':float,
                'preunlock_max_motion_m':float,'preunlock_max_motion_node':int,'preunlock_max_motion_time_s':float,'preunlock_sample_count':int,
                'preunlock_robot_cable_contact_sample_count':int,'preunlock_robot_cable_physical_contact_sample_count':int,
                'preunlock_window_200ms_max_node_displacement_m':float,'preunlock_window_200ms_max_node_speed_m_s':float,
                'preunlock_window_100ms_max_node_displacement_m':float,'preunlock_window_100ms_max_node_speed_m_s':float,
                'preunlock_last_sample_time_s':float}
            bools={'success','ever_success','relaxation_gt_5mm','filtered','grasp_confirmed','preunlock_window_covered'}
            for source_path in source_paths:
              with source_path.open(newline='',encoding='utf-8-sig') as f:
                for raw in csv.DictReader(f):
                    row=dict(raw)
                    for k,typ in numeric.items():
                        if row.get(k) not in (None,''): row[k]=typ(row[k])
                        else: row[k]=None
                    for k in bools:
                        row[k]=None if row.get(k,'')=='' else row[k].lower()=='true'
                    row['feature_geometry']=json.loads(row.pop('feature_geometry_json','{}') or '{}')
                    if row.get('success') is not None: row['success']=bool(row['success'])
                    if not row.get('run_identity'):
                        row['run_identity'],row['configuration_hash']=episode_identity(row['source_seed'],row['fraction'],row['layout'],row['selector'],row['scene'],row.get('min_crossing_angle_deg') or 60.,LEGACY_MEASUREMENT_VERSION)
                    row['measurement_version']=row.get('measurement_version') or row.get('penetration_measurement_version') or LEGACY_MEASUREMENT_VERSION
                    if row.get('source_seed') is not None: rows.append(row)
        errors=[]; done=len({r['run_identity'] for r in rows if r.get('measurement_version')==MEASUREMENT_VERSION})
        csv_fields=['job_id','run_identity','configuration_hash','measurement_version','penetration_measurement_version','source_seed','source_index','fraction','layout','scene','selector','min_crossing_angle_deg','seed','success','ever_success','failure_reason','selected_index','selected_crossing_deg','selected_corridor_clearance','initial_target_xyz','passive_motion_m','relaxation_gt_5mm','filtered','max_nonadjacent_penetration_m','preunlock_window_covered','preunlock_max_motion_m','preunlock_max_motion_node','preunlock_max_motion_time_s','preunlock_last_sample_time_s','preunlock_sample_count','preunlock_window_200ms_max_node_displacement_m','preunlock_window_200ms_max_node_speed_m_s','preunlock_window_100ms_max_node_displacement_m','preunlock_window_100ms_max_node_speed_m_s','preunlock_robot_cable_contact_sample_count','preunlock_robot_cable_physical_contact_sample_count','preunlock_robot_cable_contact_pairs','preunlock_robot_cable_physical_contact_pairs','retries','time_s','grasp_confirmed','grasped_body','close_blocked_segment','close_blocked_distance','close_blocked_vertical','close_blocked_box','max_recover_xy_drift','trace_path','feature_geometry_json']
        status('tuning',done,planned,'catalog complete; development selector tuning started or resumed',catalog_shapes=len(catalog),source_sha256=BANK_SHA256,eligible_sources=len(eligible),resumed_completed=done)
        def execute_batch(batch,stage):
            nonlocal done
            completed_ids={r['run_identity'] for r in rows if r.get('measurement_version')==MEASUREMENT_VERSION}
            batch=[job for job in batch if job['run_identity'] not in completed_ids]
            with ProcessPoolExecutor(max_workers=max(1,min(args.workers,6))) as pool:
                future_jobs={pool.submit(run_job,j):j for j in batch}
                for fut in as_completed(future_jobs):
                    completed_row=None
                    try: completed_row=fut.result(); rows.append(completed_row)
                    except Exception as e: errors.append(dict(job=future_jobs[fut],error=repr(e)))
                    if completed_row is not None:
                        is_new=not v2_episodes_path.exists()
                        with v2_episodes_path.open('a',newline='',encoding='utf-8-sig') as f:
                            row=completed_row
                            fields=csv_fields
                            out={k:row.get(k) for k in fields}
                            out['feature_geometry_json']=json.dumps(row.get('feature_geometry',{}),separators=(',',':'))
                            writer=csv.DictWriter(f,fieldnames=fields)
                            if is_new: writer.writeheader()
                            writer.writerow(out);f.flush()
                    done+=1
                    status(stage,done,planned,f'{stage}: completed {done}/{planned} actual episodes',catalog_shapes=len(catalog),errors=len(errors))
        if args.max_episodes:
            execute_batch(dev_jobs,'tuning')
            save_json(OUT/'partial.json',dict(reason='smoke run requested; no frozen evaluation/report/gallery completed',episodes=len(rows)))
            status('smoke_complete',done,planned,'smoke run only; full staged study was not started',errors=len(errors))
            return
        status('prequalification',done,planned,'checking exact capsule overlap and collecting pre-unlock motion/contact diagnostics')
        execute_batch(preflight_jobs,'prequalification')
        def collision_valid_dev_sources():
            return {int(r['source_seed']) for r in rows if r.get('scene')=='dev_static_midshape_v1_x075_v1'
                and r.get('measurement_version')==MEASUREMENT_VERSION and r.get('preunlock_window_covered')
                and r.get('selector')=='middle_geometry' and abs(float(r.get('fraction',-1))-.35)<1e-9
                and int(r.get('layout',-1))==0 and not r.get('filtered')
                and float(r.get('max_nonadjacent_penetration_m',float('inf'))) <= .005}
        collision_valid=collision_valid_dev_sources()
        if len(collision_valid)<math.ceil(len(dev)*.5):
            status('prequalification_extension',done,planned,'fewer than half of development trajectories pass the pre-registered 5mm capsule-overlap screen at 35%; checking two alternate snapshots')
            extension=[]
            for i in sorted(dev):
                for fraction in (.50,.65):
                    if int(seeds[i]) not in collision_valid:
                        add(extension,i,fraction,0,'middle_geometry')
            execute_batch(extension,'prequalification_extension')
            collision_valid |= {int(r['source_seed']) for r in rows if r.get('scene')=='dev_static_midshape_v1_x075_v1'
                and r.get('measurement_version')==MEASUREMENT_VERSION and r.get('preunlock_window_covered')
                and r.get('selector')=='middle_geometry' and r.get('fraction') in (.50,.65)
                and int(r.get('layout',-1))==0 and not r.get('filtered')
                and float(r.get('max_nonadjacent_penetration_m',float('inf'))) <= .005}
        qualification_rows=[r for r in rows if r.get('measurement_version')==MEASUREMENT_VERSION
            and r.get('scene')=='dev_static_midshape_v1_x075_v1' and r.get('selector')=='middle_geometry'
            and int(r.get('layout',-1))==0 and r.get('preunlock_window_covered')]
        save_json(OUT/'static_qualification_v2.json',dict(measurement_version=MEASUREMENT_VERSION,
            development_source_count=len(dev),collision_valid_source_count=len(collision_valid),
            collision_valid_source_seeds=sorted(collision_valid),
            movement_threshold_policy='descriptive only; no unvalidated 5mm passive-drift exclusion',
            collision_screen='exclude from primary candidate pool only if exact initial capsule-surface overlap exceeds the user-authorized 5mm threshold',
            candidate_rows=qualification_rows))
        correction_report=[
            '# 测量修正与有限资格诊断', '',
            '旧 `episodes_resumable.csv` 保留原样，测量身份为 `node_centers_window_0p84_v1`。其中穿透是节点中心线距离减两倍最大半径的近似代理，不是 MuJoCo 实际胶囊表面重叠；被动位移窗口包含 t=0.82s 解锁后一个控制周期。不得将旧穿透值解释为实际物理穿透，也不得把旧漂移量解释为纯被动漂移。',
            '新测量身份 `capsule_axes_preunlock_contact_v2` 按 MuJoCo capsule geom_xpos/geom_xmat/geom_size 计算有限胶囊轴距离和表面重叠，排除索引差≤2的相邻碰撞体；位移仅采样至 t=0.82s（首个解锁动作之前），同时记录机械臂-线缆接触采样数，以及末尾100ms/200ms每节点最大位移和速度。',
            '只有用户明确授权的初始非邻接胶囊表面重叠>5mm会触发几何筛除。被动位移5mm只作为描述标志，不作为排除门槛；尾窗位移/速度用于判断早期松弛后是否已稳定。',
            f'本轮有限资格诊断记录 {len(qualification_rows)} 条有效时间窗观测，{len(collision_valid)} 条开发源轨迹通过初始几何筛查（开发源数 {len(dev)}）。逐样本见 `static_qualification_v2.json`，原始新测量回合见 `episodes_measurement_v2.csv`。诊断模式不会自动启动完整810回合。',
        ]
        (OUT/'measurement_correction_v2.md').write_text('\n\n'.join(correction_report)+'\n',encoding='utf-8')
        if args.prequalification_only:
            status('prequalification_diagnostic_complete',done,planned,
                'limited v2 qualification complete; no 810-episode evaluation was started',
                collision_valid_sources=len(collision_valid),development_sources=len(dev),
                report=str(OUT/'measurement_correction_v2.md'),qualification=str(OUT/'static_qualification_v2.json'))
            return
        if len(collision_valid)<math.ceil(len(dev)*.5):
            status('complete',done,planned,'inconclusive: too few development source trajectories pass the initial capsule-overlap screen',
                collision_valid_sources=len(collision_valid),report=str(OUT/'measurement_correction_v2.md'),
                recommendation='Review the finite v2 qualification diagnostics before committing to full evaluation.')
            return
        execute_batch(dev_jobs,'tuning')
        # Freeze the rule using development trajectories only.
        dev_rows=[r for r in rows if r.get('measurement_version')==MEASUREMENT_VERSION and not r.get('filtered')
            and r.get('preunlock_window_covered') and split[r['source_index']]=='development'
            and r['scene']=='dev_static_midshape_v1_x075_v1']
        means={m:float(np.mean([not r['success'] for r in dev_rows if r['selector']==m])) for m in MODES if any(r['selector']==m for r in dev_rows)}
        chosen=min(means,key=means.get) if means else 'middle'
        baseline=means.get('middle',1.); improvement=baseline-means.get(chosen,baseline)
        if improvement<=0: chosen='middle'
        save_json(OUT/'selector_freeze.json',dict(selected_rule=chosen,development_failure_means=means,improvement_over_middle=improvement,development_source_seeds=sorted(int(seeds[i]) for i in dev),frozen_before_fit_and_holdout=True))
        frozen_jobs=[]
        for i in sorted(fit|hold):
            for f in FRACTIONS:
                for layout in range(3):
                    add(frozen_jobs,i,f,layout,chosen)
                    if i in hold and f==.50 and layout==1:
                        frozen_jobs[-1]['capture_trace']=True
                    add(frozen_jobs,i,f,layout,chosen,scene='dev_static_x075_v1')
        planned=len({j['run_identity'] for j in dev_jobs+frozen_jobs})
        total=planned
        status('frozen_evaluation',done,planned,'selector frozen from development set; fit and holdout evaluation started',selected_rule=chosen,errors=len(errors))
        execute_batch(frozen_jobs,'frozen_evaluation')
        # Additional fixed-selector interpretation: selector comparison remains a
        # development-only calibration, while empirical difficulty is reported by
        # source/time shape across three controlled layouts.
        by_shape={}
        for r in rows:
            if r.get('measurement_version')!=MEASUREMENT_VERSION or r.get('filtered') or not r.get('preunlock_window_covered') or r['scene']!='dev_static_midshape_v1_x075_v1' or r['selector']!=chosen: continue
            key=f"{r['source_seed']}@{r['fraction']:.2f}"
            q=by_shape.setdefault(key,dict(source_seed=r['source_seed'],fraction=r['fraction'],failures=0,n=0,selectors={}))
            q['n']+=1; q['failures']+=int(not r['success']); q['selectors'].setdefault(r['selector'],[]).append(int(not r['success']))
        ratings=[]
        for key,q in by_shape.items():
            q['failure_rate']=q['failures']/q['n'];q['wilson95']=wilson(q['failures'],q['n']);q['difficulty_bin']=('easy' if q['failure_rate']<.34 else 'medium' if q['failure_rate']<.67 else 'hard');ratings.append(q)
        ratings.sort(key=lambda r:(r['failure_rate'],r['source_seed'],r['fraction']))
        with (OUT/'shape_difficulty.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=['source_seed','fraction','n','failures','failure_rate','wilson95','difficulty_bin','selectors_json']);w.writeheader()
            for q in ratings:w.writerow({k:q.get(k) for k in ['source_seed','fraction','n','failures','failure_rate','wilson95','difficulty_bin']}|{'selectors_json':json.dumps(q['selectors'])})
        analysis_rows=[r for r in rows if r.get('measurement_version')==MEASUREMENT_VERSION]
        report=['# 完整形状经验难度试验','',f"阶段：{'partial' if errors or (args.max_episodes and args.max_episodes<900) else 'complete'}；新测量回合 {len(analysis_rows)}/{planned}。",'',
            '## 方法与冻结边界','',f"源库 SHA-256 `{BANK_SHA256}`，50条源轨迹，经历史筛查后41条合格；每条在35%、50%、65%取快照；按源轨迹分组为8条开发、20条拟合、13条留出。",
            '穿透使用实际 MuJoCo capsule geom_xpos/geom_xmat/geom_size 变换后的胶囊轴和半径，有限线段距离显式处理平行、退化及端点情况，忽略索引差≤2的相邻胶囊。被动位移取 reset 至 t=0.82s（含，首个解锁控制动作之前）的最大单节点位移；记录机械臂-线缆接触采样及末尾100ms/200ms每节点最大位移和速度。5mm被动漂移只作描述指标，不作筛除门槛。',
            '所有控制回合均为独立真实MuJoCo执行；未使用expert、shadow rollout或未来仿真选点。控制器固定为增强6有界DLS、固定竖直夹爪、桌面执行保护、0.82秒等待和原环境成功标准。结果只描述规则选点与该固定执行配置。',
            '## 选点调参与经验难度','',f"开发组 selector episode failure means: `{json.dumps(means,ensure_ascii=False)}`。选择 `{chosen}`；相对中点失败率改善 {improvement:.3f}。改善不为正时保留中点。",f"实际形状评级数 {len(ratings)}；评级根据完整回合失败比例分为探索性 easy/medium/hard，Wilson区间见 `shape_difficulty.csv`。同一源轨迹的三个时刻快照可能相关，不能视为独立形状样本。","",
            '## 结果范围与限制','',f"运行错误 {len(errors)}。初始胶囊表面重叠超过5mm的样本排除；高于5mm的解锁前位移不作为剔除条件，单独报告并可作敏感性分析。旧 `episodes_resumable.csv` 以v1 node-center代理结果原样保留，新v2回合见 `episodes_measurement_v2.csv`。当前库只存XY轨迹，不能证明源轨迹中的3D交叉关系。",'由于成功率由单一规则策略、固定朝向与当前控制器测得，难度排序是策略与执行器依赖的经验描述，不表示跨学习策略的普适难度或因果因素。若最终覆盖不足、源轨迹过少或留出效能不稳，评级只作试点结果。','',f"可复现 protocol: `outputs/shape_difficulty_overnight_20261010/protocol.json`。新测量回合: `outputs/shape_difficulty_overnight_20261010/episodes_measurement_v2.csv`。旧回合保留: `outputs/shape_difficulty_overnight_20261010/episodes_resumable.csv`。"]
        (OUT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        if errors: raise RuntimeError(f'{len(errors)} rollout jobs failed; partial rows retained in episodes_measurement_v2.csv')
        status('representative_videos',len(analysis_rows),total,'rollouts complete; rendering named-camera representative traces and verifying encoding',selected_rule=chosen,report=str(OUT/'report.md'))
        gallery,enc=build_gallery(analysis_rows,ratings)
        import socket,subprocess
        port=8784
        for candidate in range(8784,8790):
            with socket.socket() as sock:
                try:sock.bind(('127.0.0.1',candidate));port=candidate;break
                except OSError:continue
        log=(OUT/'gallery_server.log').open('ab')
        flags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
        server=subprocess.Popen([sys.executable,'-B','-X','utf8','-m','http.server',str(port),'--bind','127.0.0.1','--directory',str(OUT)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=flags)
        log.close()
        save_json(OUT/'gallery.json',dict(url=f'http://127.0.0.1:{port}/index.html',server_pid=server.pid,items=gallery))
        status('complete',len(analysis_rows),total,'all stages complete: source-grouped catalog, development selector tuning, frozen evaluation, empirical difficulty analyses, saved-state representative videos, gallery and encoding checks',selected_rule=chosen,dev_failure_means=means,report=str(OUT/'report.md'),gallery_url=f'http://127.0.0.1:{port}/index.html',gallery_server_pid=server.pid,representative_videos=len(gallery),encoding_checks=len(enc),errors=0)
    except Exception as e:
        stamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        merge_status(status_path,dict(experiment='shape_difficulty_overnight_20261010',status='failed',stage='failed',updated_at=stamp,heartbeat_at=stamp,pid=os.getpid(),coordinator_pid=os.getpid(),message=repr(e),traceback=traceback.format_exc(),report=str(OUT/'report.md')))
        raise
if __name__=='__main__':main()
