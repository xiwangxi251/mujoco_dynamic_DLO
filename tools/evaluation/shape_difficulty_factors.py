"""Complete the whole-shape factor study without speculative simulation planning."""
from __future__ import annotations
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/evaluation'),str(ROOT/'tools/recording')]
import mujoco
import numpy as np
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.kinematics import point_jacobian, rotation_to_quat, quat_error
from panda_cable_grasp.policies.scripted import Phase
from shape_selection_pilot import GeometrySelector, load as _load_once, save, sha
import run_shape_difficulty_formal as formal


def load(path):
    # Windows can briefly deny opening a file during atomic replacement.
    # This is a control-plane read retry, never a simulation rerun.
    for attempt in range(20):
        try:
            return _load_once(path)
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)


class StudySelector(GeometrySelector):
    def __init__(self, env, config, job):
        self.study_job=job
        super().__init__(env,config)

    def _choose_initial_target(self, desired_rotation):
        super()._choose_initial_target(desired_rotation)
        mode=self.study_job.get('selector','geometry')
        event=self.selection_events[-1]; candidates=event['candidates']
        pool=[c for c in candidates if c['in_workspace'] and c['in_execution_limits']] or candidates
        if mode=='middle':
            chosen=min(pool,key=lambda c:(abs(c['node']-(len(self._policy_node_ids())-1)/2),c['node']))
        elif mode=='nearest':
            chosen=min(pool,key=lambda c:(c['reach_m'],c['node']))
        else:
            chosen=event['chosen']
        self.selected_target_index=int(chosen['node'])
        self.selected_target_body_id=self._policy_node_ids()[self.selected_target_index]
        self.selected_target_crossing_angle_deg=chosen['crossing_angle_deg']
        self.selected_target_corridor_clearance=chosen['swept_mesh_box_clearance_mm']/1000
        event.update(chosen=chosen,selector=mode)

    def _predicted_segment(self,prediction_horizon=None):
        point=super()._predicted_segment(prediction_horizon)
        return point+np.array(self.study_job.get('target_offset_m',[0.,0.,0.]))

    def _update_target_tangent_orientation(self):
        if self.phase not in {Phase.SETTLE,Phase.APPROACH,Phase.INTERCEPT,Phase.CLOSE}:
            return
        super()._update_target_tangent_orientation()
        angle=np.radians(self.study_job.get('yaw_offset_deg',0.))
        if angle:
            c,s=np.cos(angle),np.sin(angle);rotation=np.empty(9)
            mujoco.mju_quat2Mat(rotation,self.desired_quat)
            self.desired_quat=rotation_to_quat(np.array([[c,-s,0],[s,c,0],[0,0,1]])@rotation.reshape(3,3))


def transform_snapshot(env, layout):
    """Rigidly transform the complete snapshot; no change to its internal shape."""
    xyz=env.data.xpos[env.cable_ids].copy(); center=xyz.mean(0)
    angle=np.radians(layout.get('yaw_deg',0.)); c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    shift=np.array([layout.get('dx_m',0.),layout.get('dy_m',0.),0.])
    address=env.cable_free_qadr
    env.data.qpos[address:address+3]=rotation@(env.data.qpos[address:address+3]-center)+center+shift
    quat=np.empty(4)
    mujoco.mju_mulQuat(quat,np.array([np.cos(angle/2),0,0,np.sin(angle/2)]),env.data.qpos[address+3:address+7])
    env.data.qpos[address+3:address+7]=quat
    mujoco.mj_forward(env.model,env.data)
    expected=(xyz-center)@rotation.T+center+shift
    assert np.max(np.abs(env.data.xpos[env.cable_ids]-expected))<1e-9


def scalar_features(row):
    candidates=row['selection_events'][0]['candidates']; selected=row['selection_events'][0]['chosen']
    result={**row['settled_features']}
    result.pop('extent_m',None)  # axis-aligned bounding-box size depends on placement yaw
    for field in ('bend_deg','swept_mesh_box_clearance_mm','crossing_angle_deg','support_m','branches'):
        values=np.array([c[field] for c in candidates],dtype=float)
        for q in (.1,.5,.9):result[f'candidate_{field}_q{int(100*q)}']=float(np.quantile(values,q))
        result[f'chosen_{field}']=float(selected[field])
    result.update(row['choice_features']);result.pop('good_candidate_fraction',None)
    return result


def local_geometry(points,index,radius):
    index=int(np.clip(index,1,len(points)-2))
    a=points[index]-points[max(index-2,0)];b=points[min(index+2,len(points)-1)]-points[index]
    bend=np.degrees(np.arccos(np.clip(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12),-1,1)))
    starts=points[:-1];vectors=np.diff(points,axis=0)
    alpha=np.clip(np.sum((points[index]-starts)*vectors,axis=1)/np.maximum(np.sum(vectors*vectors,axis=1),1e-12),0,1)
    distances=np.linalg.norm(starts+alpha[:,None]*vectors-points[index],axis=1)
    mask=np.abs(np.arange(len(distances))-index)>4
    return dict(bend_deg=float(bend),neighbor_clearance_mm=float(1000*(distances[mask].min()-2*radius)),
                support_mm=float(1000*np.linalg.norm(a+b)))


_analysis_env=None


def analysis_env(meta):
    global _analysis_env
    if _analysis_env is None:
        _analysis_env=CableGraspEnv(EnvConfig(**meta['env_config']))
        _analysis_env.reset(seed=meta['env_config']['seed'])
        atexit.register(_analysis_env.close)
    return _analysis_env


def pad_tip_projection(env,axis):
    projections=[]
    for gid in env.finger_collision_geom_ids:
        rotation=env.data.geom_xmat[gid].reshape(3,3);center=env.data.geom_xpos[gid]
        if env.model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=int(env.model.geom_dataid[gid]);start=int(env.model.mesh_vertadr[mesh]);count=int(env.model.mesh_vertnum[mesh])
            vertices=env.model.mesh_vert[start:start+count]@rotation.T+center
        else:
            corners=np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)])
            vertices=(corners*env.model.geom_size[gid])@rotation.T+center
        projections.append(float(np.max(vertices@axis)))
    return max(projections)


def contact_kind(env):
    sides={env.left_finger_id:set(),env.right_finger_id:set()}
    ids=list(env.cable_ids)
    for contact in env.data.contact:
        if contact.dist>0:continue
        for pad,other in ((int(contact.geom1),int(contact.geom2)),(int(contact.geom2),int(contact.geom1))):
            if pad in env.pad_geom_ids and int(env.model.geom_bodyid[other]) in ids:
                sides[int(env.model.geom_bodyid[pad])].add(ids.index(int(env.model.geom_bodyid[other])))
    a,b=sides.values()
    if not a and not b:return 'none'
    if not a or not b:return 'one_side'
    return 'same_strand_bilateral' if min(abs(i-j) for i in a for j in b)<=2 else 'different_strands_bilateral'


def diagnose(out,index):
    manifest=load(out/'manifest.json');row=manifest['baseline_rows'][index]
    dest=out/'diagnostics'/f"{row['sample_id']}.json"
    if dest.exists():return load(dest)
    directory=Path(manifest['baseline_run'])/row['directory'];meta=load(directory/'state_metadata.json')
    env=analysis_env(meta)
    with np.load(directory/'reference_model.npz',allow_pickle=False) as ref:env.model.body_quat[:]=ref['body_quat']
    with np.load(directory/'trajectory.npz',allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    # Only replay kinematics/collision; never advance the simulated time.
    phases=data['phases'];times=data['frame_times'];closing=np.flatnonzero((phases=='CLOSE') & np.r_[True,phases[:-1]!='CLOSE'])
    intercept=np.flatnonzero((phases=='INTERCEPT') & np.r_[True,phases[:-1]!='INTERCEPT'])
    selected=data['selected_nodes'];kind=int(data['state_kind']);attempts=[];path=[];error=0.
    frames=sorted(set(list(range(0,len(times),5))+closing.tolist()+[min(i+10,len(times)-1) for i in closing]+[min(i+35,len(times)-1) for i in closing]))
    geometries=[]
    for frame in frames:
        mujoco.mj_setState(env.model,env.data,data['states'][frame],kind)
        mujoco.mj_kinematics(env.model,env.data);mujoco.mj_comPos(env.model,env.data)
        error=max(error,float(np.max(np.abs(env.data.xpos[env.cable_ids]-data['cable_xyz'][frame]))))
        rotation=env.data.xmat[env.hand_id].reshape(3,3);hand=env.hand_position.copy()
        target=data['cable_xyz'][frame,int(selected[frame])]
        actual=rotation@env.gripper_closing_axis_local;axis=rotation@env.gripper_approach_axis_local
        desired_matrix=np.empty(9);mujoco.mju_quat2Mat(desired_matrix,data['desired_quats'][frame])
        desired=desired_matrix.reshape(3,3)@env.gripper_closing_axis_local
        yaw_error=np.degrees(np.arccos(np.clip(abs(actual@desired),0,1)))
        orientation_vector=quat_error(rotation_to_quat(rotation),data['desired_quats'][frame])
        full_orientation_error=float(np.degrees(2*np.arcsin(np.clip(np.linalg.norm(orientation_vector)/2,0,1))))
        residual=(target-hand)@rotation
        points=data['cable_xyz'][frame];j=int(selected[frame]);tangent=points[min(j+1,len(points)-1)]-points[max(j-1,0)]
        tangent/=max(np.linalg.norm(tangent),1e-12)
        crossing=np.degrees(np.arccos(np.clip(abs(actual@tangent),0,1)))
        path.append(dict(time_s=float(times[frame]),phase=str(phases[frame]),hand_xyz=hand.tolist(),
            target_error_mm=float(1000*np.linalg.norm(target-hand)),orientation_error_deg=full_orientation_error))
        if phases[frame] in ('APPROACH','INTERCEPT'):
            mujoco.mj_collision(env.model,env.data)
            path[-1]['finger_contact_kind']=contact_kind(env)
        if frame in closing:
            jacp,jacr=point_jacobian(env.model,env.data,env.hand_id,env.GRASP_CENTER_LOCAL)
            singular=float(np.linalg.svd(np.vstack([jacp,jacr])[:,env.arm_dof_adr],compute_uv=False)[-1])
            limits=env.model.jnt_range[env.arm_joint_ids];q=env.data.qpos[env.arm_qpos_adr]
            margin=float(np.min(np.minimum(q-limits[:,0],limits[:,1]-q)))
            attempts.append(dict(frame=int(frame),time_s=float(times[frame]),selected_node=j,
                **local_geometry(points,j,env.cable_radius),target_error_mm=float(1000*np.linalg.norm(target-hand)),
                closing_offset_mm=float(1000*residual@env.gripper_closing_axis_local),
                lateral_offset_mm=float(1000*residual@env.gripper_lateral_axis_local),
                approach_offset_mm=float(1000*residual@env.gripper_approach_axis_local),
                enclosure_depth_mm=float(1000*(pad_tip_projection(env,axis)-target@axis)),
                orientation_error_deg=full_orientation_error,closing_axis_error_deg=float(yaw_error),crossing_angle_deg=float(crossing),
                minimum_task_singular_value=singular,joint_limit_margin_rad=margin,contacts=[]))
        if any(frame==min(i+offset,len(times)-1) for i in closing for offset in (0,10,35)):
            mujoco.mj_collision(env.model,env.data)
            if attempts:attempts[-1]['contacts'].append(dict(time_s=float(times[frame]),kind=contact_kind(env)))
    assert error<1e-8,(row['sample_id'],error)
    initial=data['cable_xyz'][np.argmin(np.abs(times-.82))]
    distances=np.linalg.norm(initial[:,None,:]-initial[None,:,:],axis=2)
    features=scalar_features(row)
    features.update(shape_diameter_m=float(distances.max()),shape_com_x_m=float(initial[:,0].mean()),
        shape_com_y_m=float(initial[:,1].mean()),table_margin_m=float(min(np.min(initial[:,:2]-env.table_xy_min),np.min(env.table_xy_max-initial[:,:2]))-env.cable_radius))
    initial_chosen=int(row['selection_events'][0]['chosen']['node'])
    features.update({f'initial_local_{k}':v for k,v in local_geometry(initial,initial_chosen,env.cable_radius).items()})
    positions=np.array([p['hand_xyz'] for p in path]);length=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())
    first_attempt=attempts[0] if attempts else None
    # Executed configuration is outcome diagnosis, not an initial-shape predictor.
    result=dict(sample_id=row['sample_id'],seed=row['seed'],task_success=row['task_success'],
        features=features,attempts=attempts,path_length_m=length,
        approach_frames=sum(p['phase'] in ('APPROACH','INTERCEPT') for p in path),
        approach_finger_contact_frames=sum(p.get('finger_contact_kind','none')!='none' for p in path),
        path_max_orientation_error_deg=max((p['orientation_error_deg'] for p in path if p['phase'] in ('APPROACH','INTERCEPT')),default=None),
        reached_close=bool(attempts),first_close=first_attempt,kinematic_reproduction_error_m=error,
        initial_target_xyz=initial[initial_chosen].tolist())
    save(dest,result);return result


def choose_core(rows,count):
    keys=list(scalar_features(rows[0]));x=np.array([[scalar_features(r)[k] for k in keys] for r in rows])
    lo,hi=np.quantile(x,[.1,.9],axis=0);scaled=(x-np.median(x,axis=0))/np.maximum(hi-lo,1e-6)
    picked=[int(np.argmin(np.linalg.norm(scaled,axis=1)))]
    while len(picked)<count:
        used={rows[i]['seed'] for i in picked}
        eligible=[i for i,r in enumerate(rows) if r['seed'] not in used]
        picked.append(max(eligible,key=lambda i:float(np.min(np.linalg.norm(scaled[i]-scaled[picked],axis=1)))))
    return [rows[i] for i in picked]


def prepare(out,base,preflight=False):
    if (out/'manifest.json').exists():raise RuntimeError('Already prepared; resume frozen entrypoint')
    out.mkdir(parents=True,exist_ok=True);baseline=load(base/'summary.json');original=load(base/'manifest.json')
    rows=baseline['episodes'];core=choose_core(rows,64)
    cases=[dict(condition='selector_middle',selector='middle',stage_group='strategy'),
           dict(condition='selector_nearest',selector='nearest',stage_group='strategy')]
    for axis in (0,1):
        for size in (.005,.010):
            for sign in (-1,1):
                offset=[0.,0.,0.];offset[axis]=sign*size
                cases.append(dict(condition=f"offset_{'xy'[axis]}_{int(1000*size)*sign:+d}mm",target_offset_m=offset,stage_group='tolerance'))
    for angle in (-10.,-5.,5.,10.):cases.append(dict(condition=f'yaw_{angle:+g}deg',yaw_offset_deg=angle,stage_group='tolerance'))
    for key in ('dx_m','dy_m'):
        for size in (-.03,.03):cases.append(dict(condition=f'layout_{key}_{size:+g}',layout={key:size},stage_group='layout'))
    for angle in (-15.,15.):cases.append(dict(condition=f'layout_yaw_{angle:+g}',layout=dict(yaw_deg=angle),stage_group='layout'))
    jobs=[]
    for row in core:
        for case in cases:
            job=dict(sample_id=f"{row['sample_id']}__{case['condition']}",baseline_sample_id=row['sample_id'],
                seed=row['seed'],snapshot_fraction=row['snapshot_fraction'],snapshot_step=row['snapshot_step'],bank=row['bank'],
                selector='geometry')
            job.update(case);jobs.append(job)
        # Repeating a deterministic simulation verbatim adds no evidence.
        # Three independent, predeclared calibration draws provide repeated
        # trials of the same shape under the same small-error distribution.
        for repetition in range(3):
            seed=2026101100+int(row['sample_id'].split('_')[1])*3+repetition
            rng=np.random.default_rng(seed)
            jobs.append(dict(sample_id=f"{row['sample_id']}__repeat_{repetition}",baseline_sample_id=row['sample_id'],
                seed=row['seed'],snapshot_fraction=row['snapshot_fraction'],snapshot_step=row['snapshot_step'],bank=row['bank'],
                selector='geometry',condition=f'repeat_{repetition}',stage_group='repetition',calibration_seed=seed,
                target_offset_m=[float(rng.normal(0,.002)),float(rng.normal(0,.002)),0.],
                yaw_offset_deg=float(rng.normal(0,2.))))
    if preflight:
        selected_conditions=['selector_middle','selector_nearest','offset_x_+5mm','yaw_+5deg','layout_dx_m_+0.03','repeat_0']
        jobs=[next(j for j in jobs if j['condition']==condition) for condition in selected_conditions]
        rows=rows[:2];core=core[:1]
    source=out/'source';shutil.copytree(base/'source',source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('shape_difficulty_factors.py','run_shape_difficulty_formal.py'):
        shutil.copy2(ROOT/'tools/evaluation'/name,source/'tools/evaluation'/name)
    hashes={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
    manifest=dict(factor_extension=True,jobs=jobs,baseline_run=str(base.resolve()),baseline_summary_sha256=sha(base/'summary.json'),
        baseline_rows=rows,core=core,core_selection='Outcome-blind farthest feature coverage, one shape per source trajectory',
        policy_config=original['policy_config'],source_hashes=hashes,commit=original['commit'],preflight=preflight,
        baseline_reused=True,planned_new_episodes=len(jobs),bank_sha256=original['bank_sha256'],
        stages=['saved_state_factor_diagnosis','multi_selector_tolerance_layout_episodes','static_pose_ik_probes',
                'matched_shape_pairs_interactions_cross_selector_rating','representative_videos'],
        protocol='All three selectors share material locking, tangent rotation, controller, guards, wait and success. Perturbations are fixed command biases, not physics changes. Three repetitions per shape independently sample XY calibration N(0,2mm) and yaw N(0,2deg); nominal deterministic replicas are not counted as independent evidence. Layout is rigid transform only. Static IK does not step physics. Matching thresholds fixed before outcome comparison.')
    manifest['configuration_hash']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    save(out/'manifest.json',manifest)
    progress(out,'starting',0,len(jobs))
    save(out/'status.json',dict(status='starting',heartbeat_at=now().isoformat(),progress_deadline_at=(now()+timedelta(minutes=10)).isoformat()))


def now():return datetime.now(timezone.utc)


def progress(out,stage,completed,planned,**extra):
    save(out/'progress.json',dict(stage=stage,completed=completed,planned=planned,last_progress_at=now().isoformat(),**extra))


def pooled(out,jobs,function,workers,stage):
    results=[];todo=list(jobs)
    progress(out,stage,0,len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active={}
        while todo or active:
            while todo and len(active)<workers:
                i=todo.pop(0);active[pool.submit(function,out,i)]=i
            done,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future);results.append(future.result())
                progress(out,stage,len(results),len(jobs))
    return results


def static_ik(out,index):
    manifest=load(out/'manifest.json');row=manifest['core'][index]
    path=out/'ik'/f"{row['sample_id']}.json"
    if path.exists():return load(path)
    directory=Path(manifest['baseline_run'])/row['directory'];meta=load(directory/'state_metadata.json')
    env=analysis_env(meta)
    with np.load(directory/'reference_model.npz',allow_pickle=False) as ref:env.model.body_quat[:]=ref['body_quat']
    with np.load(directory/'trajectory.npz',allow_pickle=False) as data:
        frame=int(np.argmin(np.abs(data['frame_times']-.82)));state=data['states'][frame];kind=int(data['state_kind'])
        target=data['tracked_target_xyz'][frame].copy();desired=data['desired_quats'][frame].copy()
        seeds=[state];close=np.flatnonzero(data['phases']=='CLOSE')
        if len(close):seeds.append(data['states'][close[0]])
    base=np.empty(9);mujoco.mju_quat2Mat(base,desired);base=base.reshape(3,3)
    bounds=env.model.jnt_range[env.arm_joint_ids];probes=[]
    for yaw in (0.,-45.,45.,90.):
        a=np.radians(yaw);c,s=np.cos(a),np.sin(a);quat=rotation_to_quat(np.array([[c,-s,0],[s,c,0],[0,0,1]])@base)
        best=None
        for seed in seeds:
            mujoco.mj_setState(env.model,env.data,seed,kind)
            for iteration in range(90):
                mujoco.mj_kinematics(env.model,env.data);mujoco.mj_comPos(env.model,env.data)
                rotation=env.data.xmat[env.hand_id].reshape(3,3)
                translation=target-env.hand_position;orientation=quat_error(rotation_to_quat(rotation),quat)
                position_error=float(np.linalg.norm(translation));angle=float(np.degrees(2*np.arcsin(np.clip(np.linalg.norm(orientation)/2,0,1))))
                score=position_error/.005+angle/5
                if best is None or score<best['score']:
                    best=dict(score=score,position_error_mm=position_error*1000,angle_error_deg=angle,iterations=iteration,
                              found=position_error<=.005 and angle<=5)
                if position_error<=.005 and angle<=5:break
                jp,jr=point_jacobian(env.model,env.data,env.hand_id,env.GRASP_CENTER_LOCAL)
                jac=np.vstack([jp,.25*jr])[:,env.arm_dof_adr];error=np.r_[translation,.25*orientation]
                step=jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*.0025,error)
                step*=min(1.,.12/max(np.linalg.norm(step),1e-12))
                env.data.qpos[env.arm_qpos_adr]=np.clip(env.data.qpos[env.arm_qpos_adr]+step,bounds[:,0]+1e-5,bounds[:,1]-1e-5)
        probes.append(dict(yaw_offset_deg=yaw,**best))
    result=dict(sample_id=row['sample_id'],probes=probes,meaning='Bounded static IK search; failure does not prove unreachable, success does not certify collision-free motion')
    save(path,result);return result


def cross_predictions(x,y,groups):
    unique=np.random.default_rng(2026101004).permutation(np.unique(groups));folds={g:i%5 for i,g in enumerate(unique)}
    assignment=np.array([folds[g] for g in groups]);score=np.full(len(y),np.nan)
    if len(unique)<10 or len(np.unique(y))<2:return score,None
    for f in range(5):
        test=assignment==f
        if test.any():score[test]=formal.fit_predict(x[~test],y[~test],x[test])
    return score,formal.auc(y,score)


def matched_pairs(diagnostics,global_keys,local_keys):
    keys=global_keys+local_keys+['shape_com_x_m','shape_com_y_m','workspace_fraction']
    x=np.array([[r['features'][k] for k in keys] for r in diagnostics]);q=np.quantile(x,[.1,.9],axis=0)
    z=(x-np.median(x,axis=0))/np.maximum(q[1]-q[0],1e-6)
    n=len(global_keys);m=len(local_keys);groups=np.array([r['seed'] for r in diagnostics]);pairs={}
    for name,matched,contrasted in [('global_varies_local_matched',np.r_[np.arange(n,n+m),np.arange(n+m,len(keys))],np.arange(n)),
                                     ('local_varies_global_matched',np.r_[np.arange(n),np.arange(n+m,len(keys))],np.arange(n,n+m))]:
        candidates=[]
        for i in range(len(z)):
            distance=np.sqrt(np.mean((z-z[i])[:,matched]**2,axis=1));contrast=np.sqrt(np.mean((z-z[i])[:,contrasted]**2,axis=1))
            valid=np.flatnonzero((np.arange(len(z))>i)&(groups!=groups[i])&(distance<=.20)&(contrast>=.50))
            for j in valid:candidates.append((float(distance[j]),-float(contrast[j]),i,int(j)))
        chosen=[];used=set();used_sources=set()
        for distance,contrast,i,j in sorted(candidates):
            if i in used or j in used or groups[i] in used_sources or groups[j] in used_sources:continue
            chosen.append(dict(a=diagnostics[i]['sample_id'],b=diagnostics[j]['sample_id'],matched_distance=distance,
                contrasted_distance=-contrast,success_a=diagnostics[i]['task_success'],success_b=diagnostics[j]['task_success']))
            used.update((i,j))
            used_sources.update((groups[i],groups[j]))
            if len(chosen)>=32:break
        by_id={r['sample_id']:r for r in diagnostics};effects=[]
        contrasted_keys=global_keys if name.startswith('global') else local_keys
        for factor in contrasted_keys:
            differences=[]
            for pair in chosen:
                a,b=by_id[pair['a']],by_id[pair['b']]
                delta=b['features'][factor]-a['features'][factor]
                if abs(delta)>1e-9:
                    differences.append(float(np.sign(delta)*(int(not b['task_success'])-int(not a['task_success']))))
            if differences:
                rng=np.random.default_rng(2026101005);values=np.array(differences)
                bootstrap=np.mean(values[rng.integers(0,len(values),(500,len(values)))],axis=1)
                effects.append(dict(factor=factor,pairs=len(values),higher_minus_lower_failure=float(values.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist()))
        pairs[name]=dict(pairs=chosen,count=len(chosen),paired_factor_effects=effects,criterion='Robust-scaled RMS match <=0.20; contrast >=0.50; source trajectories never repeated within a contrast; matching ignores outcome',
            conclusion='Observational matched contrast; residual coupling remains. Not a synthetic isolated geometry intervention.')
    return pairs


def finalize(out,diagnostics,new_results,ik_results):
    manifest=load(out/'manifest.json');core=manifest['core'];all_base=load(Path(manifest['baseline_run'])/'summary.json')['episodes']
    global_keys=['radius_gyration_m','path_chord_ratio','total_turn_rad','turn_p90_rad','distant_close_pairs','minimum_distant_center_distance_m','shape_diameter_m']
    local_keys=['candidate_bend_deg_q50','candidate_bend_deg_q90','candidate_swept_mesh_box_clearance_mm_q10',
                'candidate_support_m_q50','candidate_branches_q90','initial_local_neighbor_clearance_mm']
    # Initial geometry only for prediction; executed pose/path/contact never enter this model.
    task_keys=['shape_com_x_m','shape_com_y_m','table_margin_m','workspace_fraction','eligible_fraction','chosen_crossing_angle_deg']
    keys=global_keys+local_keys+task_keys
    x=np.array([[r['features'][k] for k in keys] for r in diagnostics]);y=np.array([not r['task_success'] for r in diagnostics],float);groups=np.array([r['seed'] for r in diagnostics])
    group_metrics={};predictions={}
    for label,columns in [('global',list(range(len(global_keys)))),('local',list(range(len(global_keys),len(global_keys)+len(local_keys)))),
                           ('geometry',list(range(len(global_keys)+len(local_keys)))),('geometry_and_task',list(range(len(keys))))]:
        score,value=cross_predictions(x[:,columns],y,groups);group_metrics[label]=value;predictions[label]=score
    # Three predeclared interactions, no search for the best-looking product.
    lookup={k:i for i,k in enumerate(keys)}
    pairs_of_factors=[('distant_close_pairs','candidate_swept_mesh_box_clearance_mm_q10'),
                      ('radius_gyration_m','candidate_bend_deg_q90'),('total_turn_rad','workspace_fraction')]
    interaction=np.c_[x,*[x[:,lookup[a]]*x[:,lookup[b]] for a,b in pairs_of_factors]]
    _,interaction_auc=cross_predictions(interaction,y,groups)
    matches=matched_pairs(diagnostics,global_keys,local_keys)
    save(out/'matched_pairs.json',matches)
    condition_summary={}
    for condition in sorted({r['condition'] for r in new_results}):
        rows=[r for r in new_results if r['condition']==condition and not r['filtered']]
        baseline={r['sample_id']:r for r in all_base}
        condition_summary[condition]=dict(completed=len(rows),successes=sum(r['task_success'] for r in rows),
            first_attempt_successes=sum(r['task_success'] and r['retries']==0 for r in rows),
            rescued=sum(r['task_success'] and not baseline[r['baseline_sample_id']]['task_success'] for r in rows),
            degraded=sum(not r['task_success'] and baseline[r['baseline_sample_id']]['task_success'] for r in rows),
            filtered=sum(r['condition']==condition and r['filtered'] for r in new_results))
    lookup_rows={r['sample_id']:r for r in all_base};baseline_core=[lookup_rows[r['sample_id']] for r in core]
    condition_summary['geometry_baseline']=dict(completed=len(core),successes=sum(r['task_success'] for r in baseline_core),reused=True)
    repetition_counts=[]
    for base in baseline_core:
        trials=[r for r in new_results if r['stage_group']=='repetition' and r['baseline_sample_id']==base['sample_id'] and not r['filtered']]
        if trials:repetition_counts.append(dict(sample_id=base['sample_id'],completed=len(trials),successes=sum(r['task_success'] for r in trials)))
    repeat_stability=dict(shapes=len(repetition_counts),all_pass=sum(r['successes']==r['completed'] for r in repetition_counts),
                          all_fail=sum(r['successes']==0 for r in repetition_counts),mixed=sum(0<r['successes']<r['completed'] for r in repetition_counts))
    save(out/'repeat_trials.json',dict(distribution='Independent fixed XY calibration biases N(0,2mm), yaw N(0,2deg); three trials/shape',by_shape=repetition_counts,stability=repeat_stability))
    # Held-out selector transfer: train the baseline shape model outside the 64 test source groups.
    transfer={};diag_lookup={r['sample_id']:r for r in diagnostics};core_seeds={r['seed'] for r in core}
    train=np.array([g not in core_seeds for g in groups])
    if train.sum() and len(np.unique(y[train]))>1:
        for selector in ('middle','nearest'):
            rows=[r for r in new_results if r.get('selector')==selector and not r['filtered'] and r['baseline_sample_id'] in diag_lookup]
            test=np.array([[diag_lookup[r['baseline_sample_id']]['features'][k] for k in keys] for r in rows])
            if len(rows):
                score=formal.fit_predict(x[train],y[train],test);truth=np.array([not r['task_success'] for r in rows],float)
                transfer[selector]=dict(auc=formal.auc(truth,score),samples=len(rows),failures=int(truth.sum()),train_source_groups=int(len(np.unique(groups[train]))))
    summaries=[]
    with (out/'full_factors.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sample_id','seed','success']+list(diagnostics[0]['features']))
        writer.writeheader();writer.writerows(dict(sample_id=r['sample_id'],seed=r['seed'],success=r['task_success'],**r['features']) for r in diagnostics)
    for key in keys:
        values=np.array([r['features'][key] for r in diagnostics]);lo,hi=np.quantile(values,[.25,.75])
        summaries.append(dict(factor=key,low_n=int((values<=lo).sum()),high_n=int((values>=hi).sum()),
            low_failure=float(y[values<=lo].mean()),high_failure=float(y[values>=hi].mean())))
    save(out/'all_factor_associations.json',summaries)
    executed_keys=['bend_deg','neighbor_clearance_mm','enclosure_depth_mm','orientation_error_deg','crossing_angle_deg','target_error_mm','minimum_task_singular_value','joint_limit_margin_rad']
    executed=[]
    for key in executed_keys:
        rows=[r for r in diagnostics if r['first_close'] is not None];values=np.array([r['first_close'][key] for r in rows]);truth=np.array([not r['task_success'] for r in rows],float)
        if len(values):
            lo,hi=np.quantile(values,[.25,.75]);executed.append(dict(factor=key,low_failure=float(truth[values<=lo].mean()),high_failure=float(truth[values>=hi].mean())))
    save(out/'executed_configuration_associations.json',executed)
    contacts={}
    for row in diagnostics:
        if not row['first_close']:label='never_closed'
        else:label=row['first_close']['contacts'][-1]['kind'] if row['first_close']['contacts'] else 'unmeasured'
        contacts.setdefault(label,dict(episodes=0,successes=0));contacts[label]['episodes']+=1;contacts[label]['successes']+=int(row['task_success'])
    rating=[]
    for i,r in enumerate(diagnostics):rating.append(dict(sample_id=r['sample_id'],source_seed=r['seed'],failure_probability=float(predictions['geometry_and_task'][i]) if np.isfinite(predictions['geometry_and_task'][i]) else None))
    save(out/'expanded_difficulty_scores.json',rating)
    summary=dict(diagnosed_episodes=len(diagnostics),new_episodes=len(new_results),core_shapes=len(core),conditions=condition_summary,
        feature_group_auc=group_metrics,interaction_auc=interaction_auc,interactions=pairs_of_factors,
        cross_selector_transfer=transfer,matched_pairs={k:v['count'] for k,v in matches.items()},
        repeated_trial_stability=repeat_stability,
        static_ik={str(yaw):dict(found=sum(next(p for p in r['probes'] if p['yaw_offset_deg']==yaw)['found'] for r in ik_results),total=len(ik_results)) for yaw in (0.,-45.,45.,90.)},
        contact_groups=contacts,report=str(out/'report.md'),full_factors_csv=str(out/'full_factors.csv'),
        boundaries=['Rule selectors only; no PPO/DP generalization','Command-bias grid measures policy robustness, not continuous intrinsic tolerance boundary',
                    'Layout is task condition, not shape complexity','Static IK search failure is not proof of unreachability','Observational matching retains coupled factors'])
    text=['# 完整形状难度：补全实验','',f"已有{len(diagnostics)}回合状态诊断；{len(core)}个不同源轨迹形状，新增{len(new_results)}回合。基线复用，不重跑。",'',
          '所有新增回合保持增强6有界DLS、切线旋转、材料点锁定、碰桌保护、0.82秒等待、15秒预算与成功标准。未做仿真前推选点。',
          '几何选点、中间可达节点、最近可达节点三种规则共用执行器；可达候选仅指工作区和执行坐标限制，不能当作完整位姿可达认证。',
          '位置偏差为世界XY各±5/±10mm；方向偏差为±5/±10度。布局为整条绳子X/Y平移±30mm、绕质心转±15度。每个固定网格条件一次。另对每个形状做3次独立标定偏差重复：XY各N(0,2mm)、yaw N(0,2度)，只描述该误差分布下的稳定性，不把确定性原样重跑算作独立证据。','',
          '## 全部候选因素','', '| 因素 | 最低四分位失败率 | 最高四分位失败率 |','|---|---:|---:|']
    text += [f"| {r['factor']} | {r['low_failure']:.1%} ({r['low_n']}) | {r['high_failure']:.1%} ({r['high_n']}) |" for r in summaries]
    text += ['','## 执行时夹持配置（事后诊断，不用于初始形状评分）','','| 因素 | 低组失败率 | 高组失败率 |','|---|---:|---:|']
    text += [f"| {r['factor']} | {r['low_failure']:.1%} | {r['high_failure']:.1%} |" for r in executed]
    text += ['','包围深度定义为目标中心相对实际指尖最前平面的进入距离，沿实际夹爪接近轴测量，正值表示已进入指尖后方；不是直接的双侧接触认证。',
        '接触类型来自保存状态重建的碰撞接触，仅用于分析，未输入策略。路径来自每0.1秒的实际夹爪中心轨迹；接近阶段指垫接触、最大姿态误差、任务雅可比与限位余量用作执行风险诊断。','',
        '## 配对策略、偏差与布局测试','','| 条件 | 成功/完成 | 失败转成功 | 成功转失败 |','|---|---:|---:|---:|']
    text += [f"| {k} | {v['successes']}/{v['completed']} | {v.get('rescued','—')} | {v.get('degraded','—')} |" for k,v in condition_summary.items()]
    text += ['','## 评级、相互作用与因素对照','',f'按源轨迹隔离的5折AUC：{json.dumps(group_metrics,ensure_ascii=False)}。加入三个预先指定交互项：{interaction_auc}。',
        f'留出64个形状来源后的跨选点规则验证：{json.dumps(transfer,ensure_ascii=False)}。',
        f'固定匹配阈值下的形状对照数量：{summary["matched_pairs"]}；详见matched_pairs.json，未为凑数量放宽阈值。',
        f'固定中心、不同yaw的静态IK求解结果：{summary["static_ik"]}。该探针不推进物理时间，不参与规则规划。',
        f'首次闭爪接触分组：{contacts}。','',
        f'3次小标定误差重复的稳定性：{repeat_stability}。偏差条件还记录首次尝试成功数，避免只看重试后的成功掩盖容错差异。',
        '候选弯曲、净空、方向特征来自原几何选择器的近似评分，不能直接当作真实可抓性或完整位姿可达性认证；实际夹持配置单独分析。',
        '评分仅在所测规则、材料参考形状、工作区条件下成立。跨规则排序若不一致，保留策略依赖性；匹配对照不是严格独立操纵，不能据此宣称唯一决定因素。']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    save(out/'summary.json',dict(**summary,episodes=[r for r in new_results if not r['filtered']]))
    return summary


def coordinate(args):
    out=args.output;manifest=load(out/'manifest.json')
    diagnostics=pooled(out,range(len(manifest['baseline_rows'])),diagnose,args.workers,'saved_state_factor_diagnosis')
    episodes=pooled(out,range(len(manifest['jobs'])),formal.worker,args.workers,'multi_selector_tolerance_layout_episodes')
    ik=pooled(out,range(len(manifest['core'])),static_ik,args.workers,'static_pose_ik_probes')
    progress(out,'analysis',len(episodes),len(manifest['jobs']))
    summary=finalize(out,diagnostics,episodes,ik)
    if not manifest['preflight']:
        from render_saved_node_family import render
        eligible=load(out/'summary.json')['episodes'];chosen=[]
        for group in ('strategy','tolerance','layout'):
            rows=[(i,r) for i,r in enumerate(eligible) if r['stage_group']==group]
            failures=[i for i,r in rows if not r['task_success']]
            if rows:chosen.append(failures[0] if failures else rows[0][0])
        video_rows=[]
        for i in chosen:
            progress(out,'recording',len(video_rows),len(chosen));render(out,i)
            r=eligible[i];video_rows.append(dict(sample_id=r['sample_id'],seed=r['seed'],directory=r['directory'],success=r['task_success']))
        formal.build_page(out,video_rows)
        base=Path(manifest['baseline_run']);relative=out.relative_to(base).as_posix()
        original_examples=load(base/'summary.json')['representative_videos']
        combined=original_examples+[dict(r,directory=relative+'/'+r['directory']) for r in video_rows]
        formal.build_page(base,combined)
        summary.update(gallery='http://127.0.0.1:8792/index.html',representative_videos=video_rows)
        save(out/'summary.json',dict(**summary,episodes=eligible))
    progress(out,'complete',len(episodes),len(manifest['jobs']))


def supervise(args):
    out=args.output;manifest=load(out/'manifest.json');started=now();child=None;child_started=started;retries=0
    def status(state,**extra):
        p=load(out/'progress.json');last=datetime.fromisoformat(p['last_progress_at'])
        deadline=last+timedelta(minutes=15 if p['stage']=='recording' else 10)
        save(out/'status.json',dict(status=state,stage=p['stage'],completed=p['completed'],planned=p['planned'],
            planned_new_episodes=len(manifest['jobs']),configuration_hash=manifest['configuration_hash'],
            heartbeat_at=now().isoformat(),last_progress_at=last.isoformat(),progress_deadline_at=deadline.isoformat(),heartbeat_timeout_seconds=180,
            supervisor_pid=os.getpid(),supervisor_started_at=started.isoformat(),child_pid=child.pid if child else None,
            child_started_at=child_started.isoformat(),recovery_attempts=retries,report=str(out/'report.md'),**extra))
        return deadline
    status('starting')
    try:
        while True:
            child_started=now()
            with (out/'coordinator.log').open('ab') as log:
                command=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--output',str(out),'--coordinate','--workers',str(args.workers)]
                if args.fail_probe:command.append('--fail-probe')
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                save(out/'launch_receipt.json',dict(configuration_hash=manifest['configuration_hash'],command=command,
                    supervisor_pid=os.getpid(),child_pid=child.pid,child_started_at=child_started.isoformat(),source_commit=manifest['commit']))
                while child.poll() is None:
                    if now()>status('running'):
                        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                        raise TimeoutError('Real progress deadline exceeded')
                    time.sleep(3)
            if child.returncode==0:
                assert load(out/'progress.json')['stage']=='complete'
                result=load(out/'summary.json');result.pop('episodes',None);status('complete',results=result);return
            error=(out/'coordinator.log').read_text(encoding='utf8',errors='replace')[-4000:]
            if args.fail_probe or retries>=1:raise RuntimeError(error)
            retries+=1;save(out/'recovery.json',dict(at=now().isoformat(),error=error,attempt=retries,action='Resume matching completed records only'))
            progress(out,'recovering',0,len(manifest['jobs']))
    except BaseException:
        if child and child.poll() is None:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        status('failed',error=traceback.format_exc(),exit_code=child.returncode if child else None);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=ROOT/'outputs/shape_difficulty_formal_1000_20261010')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--coordinate',action='store_true');parser.add_argument('--fail-probe',action='store_true');parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();args.output=args.output.resolve()
    if args.prepare:prepare(args.output,args.base.resolve(),args.preflight)
    elif args.coordinate:
        if args.fail_probe:raise RuntimeError('Intentional extension supervisor failure probe')
        coordinate(args)
    else:supervise(args)


if __name__=='__main__':main()
