"""Paired static-snapshot selector pilot; analytic selection, no shadow rollout."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib, json, os, re, subprocess, sys, time, traceback
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools/recording')]
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.control import configure_profile
from panda_cable_grasp.env.workspace_geometry import default_workspace_sector
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig
from panda_cable_grasp.scenarios.registry import get_scenario
from record_scripted_scene_family import (physical_state, project_camera_manifest, verify_video,
    connector, marker, WORKSPACE_BOUNDARY, project, clip_segment)
from build_scripted_family_gallery import PAGE

SCENE = 'dev_static_midshape_v1_x075_v1'
BANK = ROOT/'outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks'
FILTER = ROOT/'outputs/snapshot_motion_3s_100_20261010/parallel/filter_t020_pen5_disp15.json'
AUDIT = FILTER.parent
MODES = ('middle', 'geometry')
STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x)), encoding='utf-8')
    for attempt in range(10):
        try:tmp.replace(path);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.05)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def trajectory(row):
    name = f"seed_{row['seed']}_frame_{row['snapshot_step']}.npz"
    path = AUDIT.parent/name
    if not path.exists():
        paths = list(AUDIT.glob('shard_*/result/'+name)); assert len(paths)==1, paths
        path = paths[0]
    return path

def features(points):
    xy=points[:,:2]; delta=np.diff(xy,axis=0); lens=np.linalg.norm(delta,axis=1)
    headings=np.arctan2(delta[:,1],delta[:,0]); turns=np.abs(np.arctan2(np.sin(np.diff(headings)),np.cos(np.diff(headings))))
    distances=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
    far=np.abs(np.arange(len(xy))[:,None]-np.arange(len(xy))[None,:])>=6
    return dict(extent_m=float(np.linalg.norm(np.ptp(xy,axis=0))),
        radius_gyration_m=float(np.sqrt(np.mean(np.sum((xy-xy.mean(0))**2,axis=1)))),
        path_chord_ratio=float(lens.sum()/max(np.linalg.norm(xy[-1]-xy[0]),.02)),
        total_turn_rad=float(turns.sum()),turn_p90_rad=float(np.quantile(turns,.9)),
        distant_close_pairs=int(np.sum((distances<.06)&far)//2),
        minimum_distant_center_distance_m=float(distances[far].min()),
        workspace_fraction=float(default_workspace_sector().contains_xy(xy).mean()))

def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():
        assert load(output/'manifest.json')['utility_sha256']==sha(__file__)
        return
    rows=load(FILTER)['retained']; catalog=[]
    for row in rows:
        with np.load(trajectory(row),allow_pickle=False) as data:
            positions=data['positions'] if 'positions' in data else data['cable_positions']
            times=data['times'] if 'times' in data else data['frame_times']
            pts=positions[int(np.argmin(np.abs(times-.82)))]
        catalog.append(dict(**row, features_at_082s=features(pts)))
    keys=list(catalog[0]['features_at_082s'])
    x=np.asarray([[r['features_at_082s'][k] for k in keys] for r in catalog])
    low,high=np.quantile(x,[.1,.9],axis=0); scaled=(x-np.median(x,axis=0))/np.maximum(high-low,1e-6)
    selected=[int(np.argmin(np.linalg.norm(scaled,axis=1)))]
    while len(selected)<10:
        eligible=[i for i,r in enumerate(catalog) if r['seed'] not in {catalog[j]['seed'] for j in selected}]
        index=max(eligible,key=lambda i:float(np.min(np.linalg.norm(scaled[i]-scaled[selected],axis=1))))
        selected.append(index)
    samples=[catalog[i] for i in selected]
    code_paths=[Path(__file__),ROOT/'src/panda_cable_grasp/policies/scripted.py',
        ROOT/'src/panda_cable_grasp/env/environment.py',ROOT/'src/panda_cable_grasp/env/control.py',
        ROOT/'src/panda_cable_grasp/env/workspace_geometry.py',ROOT/'tools/recording/record_scripted_scene_family.py',
        ROOT/'tools/recording/build_scripted_family_gallery.py']
    frozen=output/'source'
    for path in code_paths:
        target=frozen/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    save(output/'manifest.json',dict(samples=samples,catalog=catalog,utility_sha256=sha(__file__),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in code_paths},filter_sha256=sha(FILTER),
        bank=str(BANK),bank_sha256=sha(BANK/'replay_src_shape_nominal_v1.npz'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        selection='Median representative then farthest-point coverage of normalized full-shape and task features at t=0.82; ten distinct source trajectories; no outcome-based selection',
        holdout_source_seeds=sorted({r['seed'] for r in rows}-{r['seed'] for r in samples}),
        policy_config=asdict(PolicyConfig(follow_target_tangent=False)),
        controller='enhanced6_bounded_dls_v1',table_guard=True,arm_wait_s=.82,episode_limit_s=15,
        workspace=default_workspace_sector().manifest(),
        selector='Only _choose_initial_target replaced; config target_selection_mode remains middle in both arms. Fixed desired orientation; analytic conservative swept mesh AABB, branch count, tangent angle, local span, reach. Original nearest-segment lock retained.',
        limitations=['15mm screening covers 0.2-3s only in original layout; does not certify immobility for 15s.',
            'Mesh AABBs are conservative geometric proxies, not exact collision or pose reachability certificates.',
            '10-pair development pilot, not an independent whole-shape difficulty rating.'],
        cameras=project_camera_manifest(EnvConfig(robot='nero',dynamicvla_cameras_enabled=True))))

def make_env(row):
    scene=get_scenario(SCENE); overrides=scene.to_env_overrides()
    overrides.update(initial_shape_time_range=(row['snapshot_fraction'],)*2,arm_motion_start_delay=.82)
    cfg=configure_profile(EnvConfig(robot='nero',seed=row['seed'],target_selection='middle',episode_seconds=15,
        scenario_name=SCENE,scenario_id=scene.scenario_id,scenario_split=scene.split.value,
        replay_bank_dir=str(BANK),replay_seed_fallback='error',dynamicvla_cameras_enabled=True,**overrides),
        'enhanced6_bounded_dls_v1')
    cfg.table_execution_guard_enabled=True
    env=CableGraspEnv(cfg);env.reset(seed=row['seed'])
    assert env._initial_shape_step==row['snapshot_step']
    return env

class GeometrySelector(DynamicCableGraspPolicy):
    def __init__(self, env, config=None):
        super().__init__(env, replace(config or PolicyConfig(),
                                     preserve_selected_material_target=True))

    def _choose_initial_target(self, desired_rotation):
        env=self.env; ids=self._policy_node_ids(); points=env.data.xpos[ids].copy()
        r=env.data.xmat[env.hand_id].reshape(3,3); hand=env.data.xpos[env.hand_id]
        boxes=[]
        for gid in range(env.model.ngeom):
            body=int(env.model.geom_bodyid[gid]); ancestors=[]
            while body: ancestors.append(body);body=int(env.model.body_parentid[body])
            if env.hand_id not in ancestors or not (env.model.geom_contype[gid] or env.model.geom_conaffinity[gid]):continue
            gr=env.data.geom_xmat[gid].reshape(3,3); pos=env.data.geom_xpos[gid]
            if env.model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
                mid=int(env.model.geom_dataid[gid]);adr=int(env.model.mesh_vertadr[mid]);num=int(env.model.mesh_vertnum[mid])
                verts=env.model.mesh_vert[adr:adr+num]@gr.T+pos
            elif env.model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX:
                corners=np.asarray([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)])*env.model.geom_size[gid]
                verts=corners@gr.T+pos
            else:raise ValueError(f'Unsupported jaw collision geom type {env.model.geom_type[gid]}')
            local=(verts-hand)@r
            lo,hi=local.min(0),local.max(0)
            # Sweep open fingers through an 80mm vertical descent, in tool coordinates.
            shift=-.08*np.asarray(env.gripper_approach_axis_local)
            boxes.append((np.minimum(lo,lo+shift),np.maximum(hi,hi+shift)))
        assert boxes,'Missing actual jaw collision bodies'
        # Approximate centerline from ordered observed nodes and known radius.
        # No cable collision-body pose/orientation or forward physics is read.
        cloud=[]
        for start,end in zip(points[:-1],points[1:]):
            count=max(2,int(np.ceil(np.linalg.norm(end-start)/.002))+1)
            cloud.append(start+np.linspace(0,1,count)[:,None]*(end-start))
        cloud=np.concatenate(cloud);radius=float(env.cable_radius)
        closing=desired_rotation@env.gripper_closing_axis_local
        lateral=desired_rotation@env.gripper_lateral_axis_local
        approach=desired_rotation@env.gripper_approach_axis_local
        candidates=[]
        for index in range(2,len(ids)-2):
            point=points[index]; local=(cloud-point)@desired_rotation+env.GRASP_CENTER_LOCAL
            clearance=1.
            for lo,hi in boxes:
                q=np.abs(local-(lo+hi)/2)-(hi-lo)/2
                sdf=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
                clearance=min(clearance,float(sdf.min()-radius))
            delta=points-point
            inside=(np.abs(delta@closing)<.036)&(np.abs(delta@lateral)<.035)&(np.abs(delta@approach)<.028)
            branches=int(np.sum(inside & ~np.r_[False,inside[:-1]]))
            tangent=points[index+1]-points[index-1];tangent/=max(np.linalg.norm(tangent),1e-9)
            angle=float(np.degrees(np.arccos(np.clip(abs(tangent@closing),0,1))))
            a=points[index]-points[index-2];b=points[index+2]-points[index]
            bend=float(np.degrees(np.arccos(np.clip(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9),-1,1))))
            support=float(np.linalg.norm(points[index+2]-points[index-2]))
            in_work=bool(default_workspace_sector().contains_xy(point[:2]))
            in_limits=bool(self.config.intercept_x_limits[0]<=point[0]<=self.config.intercept_x_limits[1]
                and self.config.intercept_y_limits[0]<=point[1]<=self.config.intercept_y_limits[1]
                and self.config.intercept_z_limits[0]<=point[2]<=self.config.intercept_z_limits[1])
            reach=float(np.linalg.norm(point-env.hand_position))
            quality=2*min(angle/60,1)-min(bend/90,1)+min(support/.08,1)-reach/.6
            candidates.append(dict(node=index,in_workspace=in_work,in_execution_limits=in_limits,
                swept_mesh_box_clearance_mm=1000*clearance,branches=branches,crossing_angle_deg=angle,
                bend_deg=bend,support_m=support,reach_m=reach,quality=quality))
        eligible=[c for c in candidates if c['in_workspace'] and c['in_execution_limits']]
        pool=eligible or candidates
        chosen=max(pool,key=lambda c:(c['swept_mesh_box_clearance_mm']>=1,c['branches']<=1,
            c['crossing_angle_deg']>=60,c['quality'],min(c['swept_mesh_box_clearance_mm'],15),-c['node']))
        self.selected_target_index=chosen['node']; self.selected_target_body_id=ids[chosen['node']]
        self.selected_target_crossing_angle_deg=chosen['crossing_angle_deg']
        self.selected_target_corridor_clearance=chosen['swept_mesh_box_clearance_mm']/1000
        if not hasattr(self,'selection_events'):self.selection_events=[]
        self.selection_events.append(dict(time_s=float(env.data.time),chosen=chosen,candidates=candidates,
            fallback_no_measured_workspace_candidate=not bool(eligible),mesh_boxes_tool_m=boxes))

def dest_for(output,row,mode):
    return output/f"seed_{row['seed']}_frame_{row['snapshot_step']}"/mode

def simulate(output,index):
    manifest=load(output/'manifest.json');row=manifest['samples'][index//2];mode=MODES[index%2]
    dest=dest_for(output,row,mode);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():return
    env=make_env(row)
    try:
        initial=physical_state(env);initial_hash=hashlib.sha256(initial.tobytes()).hexdigest()
        follow_rotation=bool(manifest.get('rotation_test') and mode=='geometry')
        policy=(GeometrySelector if mode=='geometry' else DynamicCableGraspPolicy)(env,PolicyConfig(follow_target_tangent=follow_rotation))
        if follow_rotation:
            baseline=load(dest_for(output,row,'middle')/'result.json')
            assert baseline['initial_state_sha256']==initial_hash,'Baseline initialization changed'
            assert baseline['initial_goal']==policy.selected_target_index,'Initial selection changed'
        save(dest/'worker_ready.json',dict(pid=os.getpid(),ready_at=datetime.now(timezone.utc).isoformat(),
            seed=row['seed'],fraction=row['snapshot_fraction'],mode=mode,follow_target_tangent=follow_rotation,
            initial_state_sha256=initial_hash,initial_goal=policy.selected_target_index,utility_sha256=sha(__file__)))
        states=[];times=[];cable=[];targets=[];metrics=[];actions=[]
        kinematic_data=mujoco.MjData(env.model)
        def capture():
            states.append(physical_state(env));times.append(float(env.data.time))
            # mj_step caches poses before its final integration. Compute the saved
            # post-integration poses on independent data; leave policy caches alone.
            kinematic_data.qpos[:]=env.data.qpos
            mujoco.mj_kinematics(env.model,kinematic_data)
            cable.append(kinematic_data.xpos[env.cable_ids].copy())
            if policy.locked_segment_index is None:
                point=kinematic_data.xpos[policy.selected_target_body_id].copy();node=float(policy.selected_target_index)
            else:
                j,a=policy.locked_segment_index,policy.locked_segment_alpha;node=float(j+a);node_ids=policy.locked_node_ids or env.cable_ids
                point=(1-a)*kinematic_data.xpos[node_ids[j]]+a*kinematic_data.xpos[node_ids[j+1]]
            targets.append(point.copy());metrics.append(dict(time_s=times[-1],phase=policy.phase.name,node=node,
                selected_node=int(policy.selected_target_index),grasp_confirmed=bool(env.grasp_confirmed),success=bool(env.ever_success),
                hand_xyz=env.hand_position.tolist(),lifted_fraction=float(env.info().get('lifted_fraction',0))))
            desired=np.empty(9);mujoco.mju_quat2Mat(desired,policy.desired_quat)
            dc=desired.reshape(3,3)@env.gripper_closing_axis_local
            ac=kinematic_data.xmat[env.hand_id].reshape(3,3)@env.gripper_closing_axis_local
            metrics[-1].update(desired_closing_yaw_deg=float(np.degrees(np.arctan2(dc[1],dc[0]))),
                actual_closing_yaw_deg=float(np.degrees(np.arctan2(ac[1],ac[0]))),
                follow_target_tangent=follow_rotation)
        capture();start=time.monotonic()
        while not policy.finished:
            action=policy.action();actions.append(np.asarray(action).copy())
            _,_,terminated,truncated,info=env.step(action);capture()
            if terminated or truncated:break
        np.savez_compressed(dest/'trajectory.npz',states=states,times=times,cable=cable,targets=targets,actions=actions)
        result=dict(seed=row['seed'],fraction=row['snapshot_fraction'],snapshot_step=row['snapshot_step'],mode=mode,
            scene=SCENE,directory=str(dest.relative_to(output)).replace('\\','/'),task_success=bool(env.ever_success),
            policy_result=policy.result,duration_s=float(env.data.time),frames=len(states),
            failure_reason='success' if env.ever_success else policy.last_attempt_failure or env.info().get('termination_reason') or policy.result,
            initial_goal=metrics[0]['selected_node'],retry_count=policy.retry_count,initial_state_sha256=initial_hash,
            utility_sha256=sha(__file__),wall_seconds=time.monotonic()-start)
        save(dest/'metrics.json',metrics)
        save(dest/'simulation.json',dict(env_config=asdict(env.config),policy_config=asdict(policy.config),
            selection_events=getattr(policy,'selection_events',[]),result=result,final_info=env.info()))
        save(dest/'result.json',result)
    finally:env.close()

def render(output,index,max_frames=None):
    row=load(output/'manifest.json')['samples'][index//2];mode=MODES[index%2];dest=dest_for(output,row,mode)
    if max_frames is None and (dest/'metadata.json').exists():return
    env=make_env(row);renderer=None;writers={}
    try:
        data=np.load(dest/'trajectory.npz',allow_pickle=False);metrics=load(dest/'metrics.json')
        count=len(data['states']) if max_frames is None else min(max_frames,len(data['states']))
        env.model.vis.global_.offwidth=1280;env.model.vis.global_.offheight=960
        site=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_SITE,'cable_marker')
        if site>=0:env.model.site_rgba[site,3]=0
        renderer=mujoco.Renderer(env.model,height=960,width=1280,max_geom=10000)
        option=mujoco.MjvOption();option.geomgroup[3]=0
        cameras=dict(global_=env.config.dynamicvla_opst_camera_name,wrist=env.config.dynamicvla_wrist_camera_name)
        rotation_test=bool(load(output/'manifest.json').get('rotation_test'))
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
        target_dest=dest if max_frames is None else output/'smoke_video'/mode
        target_dest.mkdir(parents=True,exist_ok=True)
        writers={view:subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
            '-video_size','1280x960','-framerate','50','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','16',
            '-threads','1','-g','12','-keyint_min','12','-sc_threshold','0','-bf','0','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(target_dest/('global.mp4' if view=='global_' else 'wrist.mp4'))],stdin=subprocess.PIPE) for view in cameras}
        position_error=0.;render_error=0.
        for frame in range(count):
            mujoco.mj_setState(env.model,env.data,data['states'][frame],STATE);mujoco.mj_forward(env.model,env.data)
            position_error=max(position_error,float(np.max(np.abs(env.data.xpos[env.cable_ids]-data['cable'][frame]))))
            before=physical_state(env);point=data['targets'][frame];m=metrics[frame]
            for view,camera in cameras.items():
                renderer.update_scene(env.data,camera=camera,scene_option=option)
                for a,b in zip(WORKSPACE_BOUNDARY,np.roll(WORKSPACE_BOUNDARY,-1,axis=0)):connector(renderer.scene,a,b)
                marker(renderer.scene,point,.006,[1,.72,.05,1])
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle((12,12,1268,144),fill='#eef3fa')
                lines=[f"{mode.upper()} | seed {row['seed']} | fraction {row['snapshot_fraction']:.2f}, frame {row['snapshot_step']} | {view}",
                    f"t={m['time_s']:.2f}s | {m['phase']} | target u={m['node']:.2f} | success {m['success']}",
                    'YELLOW: selected / locked target | BLUE: measured workspace | fixed gripper yaw',
                    f"Original screening: penetration {row['initial_penetration_mm']:.2f} mm | 0.2-3s displacement {row['displacement_t020_to_3s_max_mm']:.2f} mm"]
                if rotation_test and mode=='geometry':
                    lines[0]=f"GEOMETRY + WORLD Z ROTATION | seed {row['seed']} | snapshot {row['snapshot_fraction']:.2f} / frame {row['snapshot_step']} | {view}"
                    lines[2]=f"Closing-axis yaw desired/actual: {m['desired_closing_yaw_deg']:.1f}/{m['actual_closing_yaw_deg']:.1f} deg | yellow: target | blue: workspace"
                for j,line in enumerate(lines):draw.text((24,16+31*j),line,font=font,fill='#17314b')
                x,y,z=project(renderer.scene,point,1280,960)
                if z>0 and 20<x<1260 and 156<y<940:draw.ellipse((x-17,y-17,x+17,y+17),outline='#ffe553',width=4)
                writers[view].stdin.write(np.asarray(image).tobytes())
                if frame==0:image.save(target_dest/('global_poster.png' if view=='global_' else 'wrist_poster.png'))
            render_error=max(render_error,float(np.max(np.abs(physical_state(env)-before))))
        for writer in writers.values():writer.stdin.close();assert writer.wait(timeout=60)==0
        writers={}
        assert position_error<1e-8 and render_error==0,(position_error,render_error)
        encoding={v:verify_video(target_dest/(v+'.mp4'),count) for v in ('global','wrist')}
        save(target_dest/'metadata.json',dict(result=load(dest/'result.json'),cameras=project_camera_manifest(env.config),
            recording='Re-rendered saved actual rollout states; no physics rerun',fps=50,width=1280,height=960,
            position_reproduction_max_m=position_error,render_state_difference=render_error,encoding=encoding,
            simulation_metadata=str((dest/'simulation.json').relative_to(output)),utility_sha256=sha(__file__)))
    finally:
        for writer in writers.values():
            if not writer.stdin.closed:writer.stdin.close()
            writer.wait(timeout=60)
        if renderer is not None:renderer.close()
        env.close()

def results(output):
    return [load(p) for p in sorted(output.glob('seed_*/middle/result.json'))]+[load(p) for p in sorted(output.glob('seed_*/geometry/result.json'))]

def build_gallery(output):
    rows=results(output); pairs=[]
    for sample in load(output/'manifest.json')['samples']:
        found=[r for r in rows if r['seed']==sample['seed']]
        if len(found)==2 and all((output/r['directory']/'metadata.json').exists() for r in found):pairs.append(found)
    table='<table><tr><th>快照</th><th>原中点</th><th>几何选点</th></tr>'+''.join(
        '<tr><td>seed '+str(p[0]['seed'])+' / '+str(p[0]['fraction'])+'</td>'+''.join(
            '<td>'+('成功' if r['task_success'] else '失败')+' / 初选节点 '+str(r['initial_goal'])+'</td>' for r in sorted(p,key=lambda r:MODES.index(r['mode'])))+'</tr>' for p in pairs)+'</table>'
    visibility_source=(ROOT.parent/'archive/reward_v2_20261005/reward_v6_gallery_template.html').read_text(encoding='utf-8')
    visibility=next(s for s in re.findall(r'<script>.*?</script>',visibility_source,re.S) if 'function showControls(video,show)' in s)
    page=PAGE.replace('__TABLE__',table).replace('__VISIBILITY__',visibility).replace('__RUN_NAME__',json.dumps(output.name))
    page=page.replace('__SCENES__',f'<option value="{SCENE}">复杂静态 · 筛选后</option>')
    page=page.replace('__SEEDS__',''.join(f'<option value="{p[0]["seed"]}">seed {p[0]["seed"]} · {p[0]["fraction"]:.0%} / 帧{p[0]["snapshot_step"]}</option>' for p in pairs))
    page=page.replace('__PLAYERS__',''.join(f'<article class="player" id="player-{mode}"><h2>{label}<span class="badge" id="badge-{mode}"></span></h2><video id="video-{mode}" controls playsinline preload="auto"></video><p class="meta" id="meta-{mode}"></p></article>' for mode,label in [('middle','原中点选取'),('geometry','新几何选点')]))
    page=page.replace('__DATA__',json.dumps(rows,ensure_ascii=False).replace('</','<\\/'))
    page=page.replace("ids=['global','wrist']","ids=['middle','geometry']")
    page=page.replace('<option value="both">全局＋腕部</option><option value="global">全局</option><option value="wrist">腕部</option>',
        '<option value="global">项目全局</option><option value="wrist">项目腕部</option>')
    page=re.sub(r'<title>.*?</title>','<title>规则选点 · 10快照配对测试</title>',page,count=1)
    page=re.sub(r'<h1>.*?</h1>','<h1>规则选点 · 10快照配对测试</h1>',page,count=1)
    page=re.sub(r'<div class="subtitle">.*?</div>','<div class="subtitle">筛选后10个不同源轨迹快照；固定夹爪朝向，增强6＋有界DLS，碰桌保护，0.82秒等待，15秒上限。仅替换选点。</div>',page,count=1)
    page=re.sub(r'<p id="note" class="note">.*?</p>','<p id="note" class="note">黄圈：当前选中／锁定目标；蓝线：新测扇区工作区。两种规则初始物理状态完全相同，录像重渲染实际保存状态。新规则仍保留原最近点锁定，初选点可能与最终抓取点不同。初始稳定性仅按原0.2—3秒数据筛选。短片结束停末帧。</p>',page,count=1)
    load_js="""function load(){cancel();const view=$('view').value;let duration=0;ids.forEach(id=>{const e=D.find(e=>String(e.seed)===$('goal').value&&e.mode===id);if(!e)return;duration=Math.max(duration,e.duration_s);const v=$('video-'+id);v.pause();v.src=e.directory+'/'+view+'.mp4';v.poster=e.directory+'/'+view+'_poster.png';v.playbackRate=Number($('rate').value);v.load();$('badge-'+id).textContent=e.task_success?'成功':'失败';$('badge-'+id).className='badge'+(e.task_success?' pass':'');$('meta-'+id).innerHTML=`快照 ${e.fraction} / 帧${e.snapshot_step} · 初选节点 ${e.initial_goal} · 时长 ${e.duration_s.toFixed(2)}s<br>${e.failure_reason}<br><span id="frame-clock-${id}">t=0.00s</span> · <a href="${e.directory}/metadata.json">完整记录</a>`});$('seek').value=0;$('seek').max=duration;const u=new URL(location.href);u.searchParams.set('scene',$('scene').value);u.searchParams.set('seed',$('goal').value);history.replaceState(null,'',u)}"""
    page=re.sub(r'function load\(\)\{.*?(?=\nasync function play)',lambda _:load_js,page,count=1,flags=re.S)
    ref_js="function reference(){const e=D.find(e=>String(e.seed)===$('goal').value);const t=scrubbing||pending!==null?Number($('seek').value):Math.max(...vs().map(v=>v.currentTime||0));return `${runName} | ${$('scene').value} | seed=${$('goal').value} | fraction=${e?.fraction} | frame=${e?.snapshot_step} | view=${$('view').value} | comparison=middle,geometry | t=${(Math.round(t/.02)*.02).toFixed(2)}s`}"
    page=re.sub(r'function reference\(\)\{[^\n]+',lambda _:ref_js,page,count=1)
    page=page.replace('<a href="episodes.csv">下载 CSV</a>','')
    if load(output/'manifest.json').get('rotation_test'):
        page=page.replace('规则选点 · 10快照配对测试','几何选点 · 旋转开关配对测试')
        page=page.replace('原中点选取','几何选点：不旋转（复用）').replace('新几何选点','几何选点：切线旋转')
        page=page.replace('<th>原中点</th>','<th>不旋转（复用）</th>').replace('<th>几何选点</th>','<th>切线旋转（新增）</th>')
        page=page.replace('固定夹爪朝向，增强6','仅右侧开启目标切线绕世界Z轴旋转；增强6')
        page=page.replace('仅替换选点。','两组沿用同一选点，左侧复用前次录像，右侧新增真实回合。')
        page=page.replace('comparison=middle,geometry','comparison=no_rotation,tangent_rotation')
    (output/'index.html').write_text(page,encoding='utf-8')

def summary(output):
    rows=results(output);by_mode={mode:dict(completed=sum(r['mode']==mode for r in rows),success=sum(r['mode']==mode and r['task_success'] for r in rows)) for mode in MODES}
    paired=[]
    for row in load(output/'manifest.json')['samples']:
        p=[r for r in rows if r['seed']==row['seed']]
        if len(p)==2:
            assert p[0]['initial_state_sha256']==p[1]['initial_state_sha256'],'Paired states differ'
            pair={r['mode']:r['task_success'] for r in p};paired.append(dict(seed=row['seed'],fraction=row['snapshot_fraction'],**pair))
    result=dict(by_mode=by_mode,paired=paired,rescued=sum(not p['middle'] and p['geometry'] for p in paired),
        degraded=sum(p['middle'] and not p['geometry'] for p in paired),episodes=rows)
    save(output/'summary.json',result)
    lines=['# 规则选点：10个筛选后快照配对试验','',f'原中点：{by_mode["middle"]["success"]}/{by_mode["middle"]["completed"]}；几何选点：{by_mode["geometry"]["success"]}/{by_mode["geometry"]["completed"]}。',
        f'失败转成功：{result["rescued"]}；成功转失败：{result["degraded"]}。','',
        '100个原快照按初始穿透≤5mm及0.2—3秒最大节点位移≤15mm保留74个（47条源轨迹）；本次从其中选10条不同源轨迹各一个快照。按整体几何和任务布局特征覆盖选择，未按抓取结果选择。',
        '仅替换初始选点，固定朝向；原最近点锁定、预测、闭爪、重试、增强6有界DLS、碰桌保护、0.82秒等待、15秒上限与公共物理成功条件相同。',
        '新规则：实际张开夹爪碰撞网格的工具系包围盒及80mm下探扫掠、2mm间隔胶囊轴采样、夹持框材料分支数、切线方向、局部支撑与距离。工作区使用实测环形扇区并同时满足原执行界限。包围盒是保守近似，不是精确碰撞证书。',
        '两组初始物理状态SHA256相同。保存完整执行状态、动作、当前目标、阶段及选点候选；视频按实际状态重渲染，逐帧核对线缆位置。',
        '这是开发试点：不能据此确定整条形状的普适难度，不保证3秒后没有弹性恢复，不改变线缆材料。留出源轨迹名单在manifest中。','',
        '| seed | 比例 | 原中点 | 几何选点 |','|---|---|---|---|']
    lines += [f'| {p["seed"]} | {p["fraction"]} | {int(p["middle"])} | {int(p["geometry"])} |' for p in paired]
    if load(output/'manifest.json').get('rotation_test'):
        lines=['# 几何选点：按目标切线绕世界Z轴旋转的配对试验','',
            f'不旋转（复用此前几何选点回合）：{by_mode["middle"]["success"]}/{by_mode["middle"]["completed"]}；切线旋转（新增）：{by_mode["geometry"]["success"]}/{by_mode["geometry"]["completed"]}。',
            f'失败转成功：{result["rescued"]}；成功转失败：{result["degraded"]}。','',
            '同样10个快照和同样几何选点。只启用follow_target_tangent：根据当前选中或锁定绳段切线更新世界竖直Z轴朝向，使闭合方向与切线垂直；180度等价姿态取最近一支。',
            '初始选点仍按原固定朝向的评分选择，以保持这次只比较旋转开关。原最近点锁定可能改变实际跟踪材料点，旋转规则跟随当前跟踪点。',
            '其他控制、预测、闭爪、重试、碰桌保护、0.82秒等待、15秒时间上限和成功判据一致。两个分支初始物理状态SHA256必须相同。',
            '左侧10回合与视频复用shape_selection_pilot_20261010的geometry分支，没有重新执行；右侧为10个新增真实回合。源码、配置、复用源文件SHA256见manifest和baseline_receipt。',
            '保存期望及实际夹爪闭合轴yaw供检查执行偏差。既有IK旋转时中心漂移仍可能影响结果，本次不修改IK。','',
            '| seed | 比例 | 不旋转 | 切线旋转 |','|---|---|---|---|']
        lines += [f'| {p["seed"]} | {p["fraction"]} | {int(p["middle"])} | {int(p["geometry"])} |' for p in paired]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

def coordinator(args):
    output=args.output; prepare(output); total=args.pairs*2;progress=datetime.now(timezone.utc)
    def status(state,stage,**extra):
        rotation=bool(load(output/'manifest.json').get('rotation_test'))
        save(output/'status.json',dict(status=state,stage=stage,pid=os.getpid(),planned_episodes=total,
            completed_episodes=len(results(output)),completed_videos=len(list(output.glob('seed_*/*/metadata.json'))),
            heartbeat_at=datetime.now(timezone.utc).isoformat(),last_progress_at=progress.isoformat(),
            heartbeat_timeout_seconds=180,progress_deadline_at=(progress+timedelta(seconds=420)).isoformat(),
            report=str(output/'report.md'),gallery=f'http://127.0.0.1:{args.port}/index.html',
            rotation_test=rotation,reused_baseline_episodes=args.pairs if rotation else 0,
            newly_completed_episodes=sum(r['mode']=='geometry' for r in results(output)) if rotation else len(results(output)),
            newly_planned_episodes=args.pairs if rotation else total,
            newly_completed_videos=len(list(output.glob('seed_*/geometry/metadata.json'))) if rotation else len(list(output.glob('seed_*/*/metadata.json'))),**extra))
    def run_stage(stage):
        nonlocal progress
        pending=[i for i in range(total) if not (dest_for(output,load(output/'manifest.json')['samples'][i//2],MODES[i%2])/('result.json' if stage=='simulate' else 'metadata.json')).exists()]
        children=[];handles=[];last=0
        try:
            while pending or children:
                while pending and len(children)<args.workers:
                    i=pending.pop(0);log=(output/f'worker_{stage}_{i}.log').open('ab');handles.append(log)
                    child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(Path(__file__)), '--output',str(output),'--worker',str(i),'--stage',stage],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                    children.append((i,child))
                for i,child in list(children):
                    if child.poll() is not None:
                        if child.returncode:raise RuntimeError(f'{stage} worker {i} exit {child.returncode}: '+(output/f'worker_{stage}_{i}.log').read_text(encoding='utf-8',errors='replace')[-3500:])
                        children.remove((i,child));progress=datetime.now(timezone.utc)
                        summary(output)
                        if stage=='render':build_gallery(output)
                if time.monotonic()-last>=2:status('running',stage,active_workers=[dict(index=i,pid=c.pid) for i,c in children]);last=time.monotonic()
                if datetime.now(timezone.utc)>progress+timedelta(seconds=420):raise RuntimeError(f'{stage} made no completed work for 420s')
                time.sleep(.25)
        finally:
            for _,child in children:
                if child.poll() is None:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            for handle in handles:handle.close()
    status('starting','preparing')
    if args.fail_probe:raise RuntimeError('Intentional selector-pilot supervisor failure probe')
    run_stage('simulate')
    if args.smoke:
        for i in range(total):render(output,i,max_frames=3)
    else:run_stage('render')
    r=summary(output)
    if not args.smoke:build_gallery(output)
    status('complete','smoke_complete' if args.smoke else 'complete',results={k:r[k] for k in ('by_mode','rescued','degraded')})

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pairs',type=int,default=10)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--port',type=int,default=8787)
    p.add_argument('--worker',type=int);p.add_argument('--stage',choices=['simulate','render']);p.add_argument('--prepare',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--fail-probe',action='store_true');args=p.parse_args();args.output=args.output.resolve()
    if args.prepare:prepare(args.output);return
    if args.worker is not None:
        (simulate if args.stage=='simulate' else render)(args.output,args.worker);return
    try:coordinator(args)
    except BaseException:
        save(args.output/'status.json',dict(status='failed',error=traceback.format_exc(),heartbeat_at=datetime.now(timezone.utc).isoformat()))
        raise

if __name__=='__main__':main()
