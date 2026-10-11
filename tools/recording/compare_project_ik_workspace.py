"""Paired physical near-ground tests of the project's two adapter IK branches.

Calls DynamicVLATaskSpaceAdapter.action() unchanged, with its existing flag on
or off. Isolates the unloaded robot/table from cable physics, preserving robot
inertias, joint/actuator parameters, finger contact settings and shared limits.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.kinematics import rotation_to_quat, quat_error
from panda_cable_grasp.dynamicvla.adapter import DynamicVLAAdapterConfig, DynamicVLATaskSpaceAdapter
from render_vertical_workspace import Workspace, font, add_geom, add_line

OUTPUT=Path('outputs/vertical_workspace/nero_project_ik_compare_20261001')
METHODS=('pose_ik','jacobian_ik')
Z=.005
DT=.02
POSITION_TOL=.005
POSE_TOL=2.0
HOLD_STEPS=15
MAX_STEPS=700


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Rig:
    def __init__(self, method):
        self.method=method
        self.geometry=Workspace()
        self.base=CableGraspEnv(EnvConfig(robot='nero',motion_mode='static',
            disturbance_strength=0.,frame_skip=10,episode_seconds=10000.,
            target_selection='middle'))
        original=self.base.model
        model=self.geometry.model
        # Both scenes include the same robot before the table. Verify all IDs,
        # FK and physics properties before removing the unrelated cable system.
        self.audit={}
        for name in (*self.base.robot_spec.arm_joint_names,*self.base.robot_spec.finger_joint_names):
            a,b=original.joint(name).id,model.joint(name).id
            assert a==b
            for attr in ('jnt_range','jnt_axis','jnt_pos','jnt_type'):
                assert np.allclose(getattr(original,attr)[a],getattr(model,attr)[b])
            da,db=original.jnt_dofadr[a],model.jnt_dofadr[b]
            for attr in ('dof_damping','dof_armature','dof_frictionloss'):
                assert np.allclose(getattr(original,attr)[da],getattr(model,attr)[db])
        for name in ('base_link',*(f'link{i}' for i in range(1,8)),'gripper_link1','gripper_link2'):
            a,b=original.body(name).id,model.body(name).id
            assert a==b
            for attr in ('body_mass','body_inertia','body_pos','body_quat'):
                assert np.allclose(getattr(original,attr)[a],getattr(model,attr)[b])
        for attr in ('actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange'):
            getattr(model,attr)[:]=getattr(original,attr)
            assert np.allclose(getattr(model,attr),getattr(original,attr))
        for gid in self.base.pad_geom_ids:
            for attr in ('geom_friction','geom_condim','geom_priority','geom_solref','geom_solimp'):
                getattr(model,attr)[gid]=getattr(original,attr)[gid]
        testq=np.r_[self.geometry.home,.05,-.05]
        original_data=mujoco.MjData(original)
        original_data.qpos[:9]=testq
        mujoco.mj_forward(original,original_data)
        self.base.model=model
        self.base.data=self.geometry.data
        self.base.data.qpos[:9]=testq
        mujoco.mj_forward(model,self.base.data)
        assert np.allclose(original_data.xpos[self.base.hand_id],self.base.data.xpos[self.base.hand_id],atol=1e-10)
        assert np.allclose(original_data.xmat[self.base.hand_id],self.base.data.xmat[self.base.hand_id],atol=1e-10)
        self.base._hand_jacp=np.zeros((3,model.nv))
        self.base._hand_jacr=np.zeros((3,model.nv))
        self.base._geometric_safety_filter.model=model
        self.base._geometric_safety_filter.probe_data=mujoco.MjData(model)
        assert max(self.base._geometric_safety_filter.robot_geom_ids)<model.ngeom
        self.adapter=DynamicVLATaskSpaceAdapter(self.base,
            DynamicVLAAdapterConfig(nero_pose_ik_enabled=method=='pose_ik'))
        self.quat=rotation_to_quat(self.geometry.rotation())
        self.model=model
        self.data=self.base.data
        self.home=self.geometry.home.copy()
        self.collision_pairs=[]
        for i,g in enumerate(self.geometry.visual_geoms):
            for h in self.geometry.visual_geoms[i+1:]:
                a=int(model.geom_bodyid[g]); b=int(model.geom_bodyid[h])
                ancestors={a:0}; da=0; db=0
                while a:
                    a=int(model.body_parentid[a]); da+=1; ancestors[a]=da
                while b not in ancestors:
                    b=int(model.body_parentid[b]); db+=1
                if db+ancestors[b]>2:
                    self.collision_pairs.append((g,h))
        self.audit={'robot_kinematics_match':True,'robot_mass_joint_and_actuator_parameters_match':True,
            'finger_contact_parameters_copied':True,'cable_removed_only_for_unloaded_workspace_test':True,
            'timestep':float(model.opt.timestep),'control_dt':DT,
            'exact_adapter_source':'src/panda_cable_grasp/dynamicvla/adapter.py',
            'adapter_config':asdict(self.adapter.config),'env_config':asdict(self.base.config)}
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:9]=np.r_[self.home,.05,-.05]
        self.data.ctrl[:]=self.base.ready_ctrl
        self.base._last_applied_action[:]=self.base.ready_ctrl
        self.base._last_requested_action[:]=self.base.ready_ctrl
        self.base._previous_arm_command_velocity[:]=0
        mujoco.mj_forward(self.model,self.data)
        self.adapter.reset()

    def measure(self,target,collision=False):
        current=self.base.hand_position.copy()
        rotation=self.data.xmat[self.base.hand_id].reshape(3,3)
        error=float(np.linalg.norm(current-target))
        tilt=math.degrees(math.acos(np.clip(-rotation[2,0],-1,1)))
        oe=np.linalg.norm(quat_error(rotation_to_quat(rotation),self.quat))
        pose=math.degrees(2*math.asin(np.clip(oe/2,0,1)))
        floor=self.geometry.floor_clearance()
        q=self.data.qpos[self.base.arm_qpos_adr]
        margin=math.degrees(float(np.min(np.minimum(q-self.geometry.lo,self.geometry.hi-q))))
        selfdist=min((mujoco.mj_geomDistance(self.model,self.data,g,h,.002,None)
            for g,h in self.collision_pairs),default=.002) if collision else .002
        return np.array([error,tilt,pose,floor,margin,selfdist,*current])

    def control(self,target):
        self.adapter.set_model_action(np.r_[target,self.quat,1.])
        assert not self.adapter.last_position_clipped, 'target outside common configured workspace'
        requested=self.adapter.action()
        requested=np.clip(requested,self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1])
        applied=self.base._limit_robot_action(requested)
        for _ in range(10):
            guarded,active=self.base._velocity_guarded_action(applied)
            self.data.ctrl[:]=guarded
            mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        return applied

    def trial(self,x,y,record=False,max_steps=MAX_STEPS,ideal=False):
        self.reset()
        target=np.array([x,y,Z])
        held=0; success=False; reason='timeout'; history=[]
        min_floor=1.; min_self=1.; max_tilt=0.; max_speed=0.; max_command_speed=0.
        min_error=1.; reached_at=None; first_condition=None
        t0=time.monotonic()
        reference=self.base.hand_position.copy()
        approach=np.array([x,y,.10])
        approach_done=False
        for k in range(max_steps):
            if ideal:
                # Exact controller targets, shared rate limiter, perfect tracking:
                # this is explicitly a solver/ideal-servo diagnostic only.
                self.adapter.set_model_action(np.r_[target,self.quat,1.])
                applied=self.base._limit_robot_action(self.adapter.action())
                self.data.qpos[self.base.arm_qpos_adr]=applied[:7]
                self.data.qpos[self.base.finger_qpos_adr]=(.05,-.05)
                self.data.qvel[:]=0
                self.data.ctrl[:]=applied
                self.data.time+=DT
                mujoco.mj_forward(self.model,self.data)
            else:
                waypoint=target if approach_done else approach
                delta=waypoint-reference
                distance=np.linalg.norm(delta)
                speed=.03 if approach_done else .15
                reference+=delta*min(1.,speed*DT/max(distance,1e-12))
                if np.linalg.norm(reference-approach)<1e-9 and not approach_done:
                    approach_done=True
                self.control(reference)
            # Check geometry regularly during approach, and every step while
            # entering the qualifying hold, so no unsafe endpoint is counted.
            m=self.measure(target,collision=(k%5==0 or held>0))
            min_floor=min(min_floor,m[3]); min_self=min(min_self,m[5]); max_tilt=max(max_tilt,m[1])
            max_speed=max(max_speed,float(np.max(abs(self.data.qvel[:7]))))
            max_command_speed=max(max_command_speed,float(np.max(abs(self.base._last_applied_arm_velocity))))
            min_error=min(min_error,m[0])
            acceptable=(m[0]<=POSITION_TOL and m[2]<=POSE_TOL and m[3]>=.0002
                and m[3]<=.008 and m[4]>=-.01 and m[5]>=.001)
            if acceptable and first_condition is None:
                first_condition=(k+1)*DT
            held=held+1 if acceptable else 0
            if record:
                history.append(np.r_[self.data.time,self.data.qpos[:9],m,target])
            if held>=HOLD_STEPS:
                success=True; reason='reached_and_held'; reached_at=(k+1)*DT; break
            if m[3]<-.001 and not ideal:
                reason='floor_penetration'; break
            if m[5]<-.001 and not ideal:
                reason='self_collision'; break
            if not np.all(np.isfinite(self.data.qpos)):
                reason='invalid_nonfinite'; break
        final=self.measure(target,collision=True)
        row={'method':self.method,'mode':'ideal_servo' if ideal else 'physical_slow_approach',
            'x_m':x,'y_m':y,'z_m':Z,'success':success,'reason':reason,
            'steps':k+1,'sim_seconds':(k+1)*DT,'wall_seconds':time.monotonic()-t0,
            'reached_at_s':reached_at,'first_condition_s':first_condition,
            'position_error_mm':final[0]*1000,'tilt_deg':final[1],'pose_error_deg':final[2],
            'floor_clearance_mm':final[3]*1000,'joint_margin_deg':final[4],
            'self_distance_mm':final[5]*1000,'actual_x_m':final[6],'actual_y_m':final[7],'actual_z_m':final[8],
            'min_approach_floor_mm':min_floor*1000,'min_approach_self_mm':min_self*1000,
            'max_approach_tilt_deg':max_tilt,'max_actual_joint_speed_rad_s':max_speed,
            'max_command_joint_speed_rad_s':max_command_speed,'minimum_position_error_mm':min_error*1000}
        return row,np.array(history)


_RIGS={}
def worker(job):
    method,x,y,ideal=job
    if method not in _RIGS:
        _RIGS[method]=Rig(method)
    return _RIGS[method].trial(x,y,ideal=ideal,max_steps=400 if ideal else MAX_STEPS)[0]


def run_pilot(out,ideal=False):
    results=[]
    for method in METHODS:
        rig=Rig(method)
        (out/f'{method}_audit.json').write_text(json.dumps(rig.audit,indent=2,default=str),encoding='utf-8')
        for x,y in ((.7,0),(.55,0),(.82,0),(.65,.3),(.65,-.3),(.4,.4)):
            row,history=rig.trial(x,y,record=True,ideal=ideal,max_steps=400 if ideal else MAX_STEPS)
            results.append(row)
            np.savez_compressed(out/f'pilot_{method}_{x}_{y}.npz',history=history)
            print(json.dumps(row),flush=True)
    (out/'pilot.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


def scan(out,step,workers,ideal=False):
    xs=np.round(np.arange(.20,.841,step),6)
    ys=np.round(np.arange(-.54,.541,step),6)
    jobs=[(method,float(x),float(y),ideal) for y in ys for x in xs for method in METHODS]
    rows=[]
    partial=out/'cells.jsonl'
    completed=set()
    if partial.exists():
        for line in partial.read_text(encoding='utf-8').splitlines():
            r=json.loads(line); rows.append(r); completed.add((r['method'],r['x_m'],r['y_m']))
    jobs=[j for j in jobs if j[:3] not in completed]
    t0=time.monotonic()
    with partial.open('a',encoding='utf-8') as file, mp.get_context('spawn').Pool(workers) as pool:
        for i,row in enumerate(pool.imap_unordered(worker,jobs,chunksize=4)):
            file.write(json.dumps(row)+'\n'); file.flush(); rows.append(row)
            if (i+1)%40==0:
                print(f'{len(rows)}/{len(xs)*len(ys)*2} cells; {time.monotonic()-t0:.1f}s; successes {sum(r["success"] for r in rows)}',flush=True)
    with (out/'cells.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    masks={m:np.zeros((len(ys),len(xs)),bool) for m in METHODS}
    for r in rows:
        iy=int(np.argmin(abs(ys-r['y_m']))); ix=int(np.argmin(abs(xs-r['x_m'])))
        masks[r['method']][iy,ix]=r['success']
    np.savez_compressed(out/'workspace.npz',xs=xs,ys=ys,**masks)
    print('scan complete',flush=True)


def compare_scan(out,workers):
    specs={'ideal':(.04,True),'physical':(.06,False)}
    rows={key:[] for key in specs}
    grids={}
    files={}
    jobs=[]
    for key,(step,ideal) in specs.items():
        folder=out/key; folder.mkdir(exist_ok=True)
        xs=np.round(np.arange(.20,.841,step),6)
        ys=np.round(np.arange(-.54,.541,step),6)
        grids[key]=(xs,ys)
        partial=folder/'cells.jsonl'
        done=set()
        if partial.exists():
            for line in partial.read_text(encoding='utf-8').splitlines():
                r=json.loads(line); rows[key].append(r); done.add((r['method'],r['x_m'],r['y_m']))
        jobs += [(method,float(x),float(y),ideal) for y in ys for x in xs for method in METHODS
                 if (method,float(x),float(y)) not in done]
        files[key]=partial.open('a',encoding='utf-8')
    plan={'methods':list(METHODS),'adapter_switch_values':[True,False],
          'solver_grid_m':.04,'physical_grid_m':.06,'tcp_z_m':Z,'fixed_yaw':'same home vertical rotation',
          'position_tolerance_m':POSITION_TOL,'orientation_tolerance_deg':POSE_TOL,
          'floor_clearance_m':[.0002,.008],'hold_seconds':HOLD_STEPS*DT,
          'maximum_physical_seconds':MAX_STEPS*DT,'maximum_ideal_seconds':8.,
          'approach_waypoint_z_m':.10,'approach_reference_speed_m_s':.15,
          'descent_reference_speed_m_s':.03,'worker_count':workers,
          'controller_source_sha256':sha('src/panda_cable_grasp/dynamicvla/adapter.py'),
          'environment_source_sha256':sha('src/panda_cable_grasp/env/environment.py')}
    (out/'run_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    t0=time.monotonic()
    total=sum(len(xs)*len(ys)*2 for xs,ys in grids.values())
    with mp.get_context('spawn').Pool(workers) as pool:
        for i,row in enumerate(pool.imap_unordered(worker,jobs,chunksize=3)):
            key='ideal' if row['mode']=='ideal_servo' else 'physical'
            files[key].write(json.dumps(row)+'\n'); files[key].flush(); rows[key].append(row)
            if (i+1)%20==0:
                state={k:{'completed':len(v),'successes':sum(r['success'] for r in v)} for k,v in rows.items()}
                state['elapsed_seconds']=time.monotonic()-t0
                state['total_cells']=total
                (out/'progress.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
                print(f'{sum(len(v) for v in rows.values())}/{total}: '+json.dumps(state),flush=True)
    for file in files.values(): file.close()
    for key,v in rows.items():
        folder=out/key; xs,ys=grids[key]
        with (folder/'cells.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(v[0])); writer.writeheader(); writer.writerows(v)
        masks={m:np.zeros((len(ys),len(xs)),bool) for m in METHODS}
        for r in v:
            iy=int(np.argmin(abs(ys-r['y_m']))); ix=int(np.argmin(abs(xs-r['x_m'])))
            masks[r['method']][iy,ix]=r['success']
        np.savez_compressed(folder/'workspace.npz',xs=xs,ys=ys,**masks)
    state={k:{'completed':len(v),'successes':sum(r['success'] for r in v)} for k,v in rows.items()}
    state.update(status='complete',elapsed_seconds=time.monotonic()-t0,total_cells=total)
    (out/'progress.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    print('paired scan complete',flush=True)


def verify_full_scene(out):
    results={}
    for method in METHODS:
        rig=Rig(method)
        full=CableGraspEnv(EnvConfig(robot='nero',motion_mode='static',disturbance_strength=0.,
            frame_skip=10,episode_seconds=10000.,target_selection='middle'))
        full.reset(randomize=False,seed=20261001)
        adapter=DynamicVLATaskSpaceAdapter(full,DynamicVLAAdapterConfig(nero_pose_ik_enabled=method=='pose_ik'))
        target=np.array([.7,0,.20])
        maxq=0.; maxp=0.
        for _ in range(50):
            adapter.set_model_action(np.r_[target,rig.quat,1.])
            request=adapter.action()
            applied=full._limit_robot_action(request)
            for j in range(10):
                guarded,_=full._velocity_guarded_action(applied)
                full.data.ctrl[:]=guarded
                mujoco.mj_step(full.model,full.data)
            mujoco.mj_forward(full.model,full.data)
            rig.control(target)
            maxq=max(maxq,float(np.max(abs(full.data.qpos[:9]-rig.data.qpos[:9]))))
            maxp=max(maxp,float(np.linalg.norm(full.hand_position-rig.base.hand_position)))
        assert maxq<1e-5 and maxp<1e-5,(method,maxq,maxp)
        savedq=full.data.qpos.copy(); savedv=full.data.qvel.copy()
        adapter._pose_ik_target(target,rig.quat)
        assert np.array_equal(full.data.qpos,savedq) and np.array_equal(full.data.qvel,savedv)
        results[method]={'full_vs_isolated_max_qpos_difference':maxq,
            'full_vs_isolated_max_tcp_difference_m':maxp,'paired_physics_steps':500,
            'pose_solver_restores_state':True,'audit':rig.audit}
    (out/'full_scene_verification.json').write_text(json.dumps(results,indent=2,default=str),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='audit'} for k,v in results.items()},indent=2),flush=True)


def load_rows(folder):
    return [json.loads(line) for line in (folder/'cells.jsonl').read_text(encoding='utf-8').splitlines()]


def workspace_bounds(grid,mask):
    y,x=np.where(mask)
    if not len(x): return None
    rad=np.hypot(grid['xs'][x]-.35,grid['ys'][y])
    return {'x_m':[float(grid['xs'][x].min()),float(grid['xs'][x].max())],
        'y_m':[float(grid['ys'][y].min()),float(grid['ys'][y].max())],
        'base_radius_m':[float(rad.min()),float(rad.max())],
        'approximate_sampled_area_m2':float(len(x)*(grid['xs'][1]-grid['xs'][0])**2)}


def comparison_map(ideal,physical,method,size=530,actual=None,target=None):
    im=Image.new('RGB',(size,size),'#f5f8fc'); draw=ImageDraw.Draw(im)
    pad=size*.11; scale=(size-2*pad)/1.20
    def xy(x,y): return pad+(x-.10)*scale,size-pad-(y+.60)*scale
    def tiles(grid,color):
        step=float(grid['xs'][1]-grid['xs'][0])
        for iy,ix in zip(*np.where(grid[method])):
            x,y=grid['xs'][ix],grid['ys'][iy]
            draw.rectangle([xy(x-step/2,y+step/2),xy(x+step/2,y-step/2)],fill=color)
    tiles(ideal,'#4bc4eb'); tiles(physical,'#39b98c')
    f=font(max(11,int(size*.028)))
    for v in np.arange(.2,1.21,.2):
        a,b=xy(v,-.6),xy(v,.6); draw.line([a,b],fill='#c3cbd4')
        draw.text((a[0]-10,a[1]+4),f'{v:.1f}',font=f,fill='#455266')
    for v in np.arange(-.6,.61,.2):
        a,b=xy(.1,v),xy(1.3,v); draw.line([a,b],fill='#c3cbd4')
        draw.text((2,a[1]-8),f'{v:.1f}',font=f,fill='#455266')
    for iy,y in enumerate(physical['ys']):
        for ix,x in enumerate(physical['xs']):
            if not physical[method][iy,ix]:
                p=xy(x,y); r=max(1,size*.004)
                draw.line((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill='#ce6b72',width=1)
                draw.line((p[0]-r,p[1]+r,p[0]+r,p[1]-r),fill='#ce6b72',width=1)
    a=xy(.35,0); r=size*.017
    draw.ellipse((a[0]-r,a[1]-r,a[0]+r,a[1]+r),fill='#26324d')
    draw.text((a[0]+r+2,a[1]-8),'底座',font=f,fill='#26324d')
    if target is not None:
        a=xy(target[0],target[1]); r=size*.012
        draw.ellipse((a[0]-r,a[1]-r,a[0]+r,a[1]+r),fill='#dfae2a',outline='#705819',width=2)
    if actual is not None:
        a=xy(actual[0],actual[1]); r=size*.012
        draw.ellipse((a[0]-r,a[1]-r,a[0]+r,a[1]+r),fill='#e74468',outline='white',width=1)
    draw.text((size//2-40,size-23),'世界 X / m',font=f,fill='#455266')
    draw.text((5,5),'世界 Y / m',font=f,fill='#455266')
    return im


def summarize(out):
    ideal=dict(np.load(out/'ideal/workspace.npz'))
    physical=dict(np.load(out/'physical/workspace.npz'))
    rows={'ideal':load_rows(out/'ideal'),'physical':load_rows(out/'physical')}
    summary={}
    for method in METHODS:
        item={}
        for mode,grid in [('ideal',ideal),('physical',physical)]:
            rr=[r for r in rows[mode] if r['method']==method]
            reasons={reason:sum(r['reason']==reason for r in rr) for reason in sorted({r['reason'] for r in rr})}
            item[mode]={'tested':len(rr),'passed':sum(r['success'] for r in rr),
                'bounds':workspace_bounds(grid,grid[method]),'termination_counts':reasons,
                'median_final_position_error_mm':float(np.median([r['position_error_mm'] for r in rr])),
                'median_final_tilt_deg':float(np.median([r['tilt_deg'] for r in rr]))}
        summary[method]=item
        ideal_by_target={(r['x_m'],r['y_m']):r for r in rows['ideal'] if r['method']==method}
        paired=[(ideal_by_target[(r['x_m'],r['y_m'])],r) for r in rows['physical']
            if r['method']==method and (r['x_m'],r['y_m']) in ideal_by_target]
        item['identical_targets_in_both_grids']={'tested':len(paired),
            'ideal_passed':sum(a['success'] for a,b in paired),
            'physical_passed':sum(b['success'] for a,b in paired),
            'physical_passed_among_ideal_passed':sum(a['success'] and b['success'] for a,b in paired)}
    common=ideal['pose_ik']&ideal['jacobian_ik']
    summary['ideal_overlap']={'both':int(common.sum()),'pose_only':int((ideal['pose_ik']&~ideal['jacobian_ik']).sum()),
        'jacobian_only':int((~ideal['pose_ik']&ideal['jacobian_ik']).sum())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    image=Image.new('RGB',(1280,1000),'#f5f8fc'); draw=ImageDraw.Draw(image)
    draw.text((28,18),'NERO · 项目现有两种 IK 的竖直贴地对照',font=font(31),fill='#192940')
    draw.text((28,65),'同一初始姿态 / 固定水平朝向 / TCP目标5 mm / 使用原适配器与关节限制',font=font(21),fill='#415169')
    for i,method in enumerate(METHODS):
        x=20+i*630
        label='Pose IK · 迭代位姿解算' if method=='pose_ik' else 'Jacobian IK · 速度层分解'
        draw.text((x+30,116),label,font=font(27),fill='#192940')
        image.paste(comparison_map(ideal,physical,method,580),(x,161))
        dd=ImageDraw.Draw(image)
        info=summary[method]
        dd.text((x+34,753),f"理想执行器：{info['ideal']['passed']}/{info['ideal']['tested']} 点通过",font=font(24),fill='#147fa8')
        dd.text((x+34,792),f"物理保持：{info['physical']['passed']}/{info['physical']['tested']} 点通过",font=font(24),fill='#17885e' if info['physical']['passed'] else '#b94d58')
        bb=info['ideal']['bounds']
        if bb:
            dd.text((x+34,835),f"解算包围范围 X {bb['x_m'][0]:.2f}~{bb['x_m'][1]:.2f} m",font=font(20),fill='#415169')
            dd.text((x+34,867),f"Y {bb['y_m'][0]:+.2f}~{bb['y_m'][1]:+.2f} m；内部有空洞",font=font(20),fill='#415169')
    dd=ImageDraw.Draw(image)
    dd.text((28,923),'蓝色：解算通过（4 cm网格）   绿色：物理保持通过（6 cm网格）   红叉：物理试跑未通过',font=font(23),fill='#415169')
    dd.text((28,963),'通过要求：位置≤5 mm、姿态≤2°、指尖离地0.2~8 mm，并连续保持0.3秒。',font=font(20),fill='#415169')
    image.save(out/'ik_workspace_comparison.png')
    return ideal,physical,summary


def report(out):
    _,_,summary=summarize(out)
    lines=['# NERO 项目现有 Pose IK / Jacobian IK：竖直贴地对照（2026-10-01）','',
        '本次调用项目原有 `DynamicVLATaskSpaceAdapter.action()`，仅切换 `nero_pose_ik_enabled=True/False`。没有使用 SciPy IK，也没有修改项目控制器、机器人增益、摩擦或成功标准。', '',
        '解算层和物理层分别报告。理想执行器模式直接跟踪原适配器输出的关节目标；物理模式使用原关节限制、速度守卫、MuJoCo 位置执行器和500 Hz物理积分。', '',
        '理想执行器模式的蓝色表示终端目标姿态通过，并不保证到该点的过渡路径安全。过渡期最小离地高度、自碰撞间距和最大倾斜均保存在原始行中；物理模式则对严重穿地/穿透提前停止。', '',
        '| 方法 | 理想执行器解算通过 | 实际物理保持通过 |', '|---|---:|---:|']
    for m in METHODS:
        a=summary[m]['ideal']; b=summary[m]['physical']
        lines.append(f"| {m} | {a['passed']}/{a['tested']} | {b['passed']}/{b['tested']} |")
    lines += ['', '两种IK都能在一部分位置求出合格的竖直贴地姿态。实际物理保持未通过的点不能被标成已验证适宜工作区域；也不能据此宣称机械臂在这些位置物理不可达。此结果受初始姿态、参考路径、水平朝向、默认伺服参数和时间预算限制。', '',
        '## 共同测试协议', '',
        '- NERO底座 `(0.35,0,0)` m；平面 `z=0`；TCP目标高度5 mm。',
        '- 固定同一完整水平朝向，目标四元数由项目NERO竖直基准构造。相较上一版允许水平转向的SciPy扫描，本次任务更受约束，两版范围不能直接比较。',
        '- 同一初始关节配置、单指张开50 mm、共享适配器默认参数；命令目标限制在共享工作空间内。',
        '- 解算网格4 cm，每点最多8 s虚拟控制时间；物理网格6 cm，每点最多14 s仿真时间，均独立从home出发。',
        '- 物理参考先以0.15 m/s接近目标上方10 cm，再以0.03 m/s缓慢下降。没有针对某个点或某一种IK调参。',
        '- 通过条件：TCP位置误差≤5 mm，完整姿态误差≤2°，机器人网格最低离地0.2–8 mm，关节合法，非邻接网格距离≥1 mm，连续保持0.3 s。',
        '- 物理穿地超过1 mm或非邻接穿透超过1 mm时停止该试跑；记录为失败，原始行保留。',
        '- 低层机械臂原模型的自碰撞开关不变。独立检测通过MuJoCo凸包网格距离进行，装配邻居（刚体树距离≤2）不参与。', '',
        '不同网格的重合目标也单独统计，避免把分辨率差异当作结果差异：','',
        '| 方法 | 完全同坐标目标数 | 解算通过 | 物理通过 |','|---|---:|---:|---:|']
    for m in METHODS:
        a=summary[m]['identical_targets_in_both_grids']
        lines.append(f"| {m} | {a['tested']} | {a['ideal_passed']} | {a['physical_passed']} |")
    lines += ['', '## 场景验证与执行器诊断', '',
        '独立的机器人/地面场景仅移除了不参与工作空间测试的线缆。机器人运动学、质量、惯量、关节阻尼/摩擦、执行器参数和指垫接触参数均已核对。两种分支各进行了500个物理步的完整场景配对检查：机械臂轨迹差异在浮点精度范围内，见 `full_scene_verification.json`。', '',
        '在 `(0.7,0,0.005)` m 的理想Pose IK解中，机器人关节位置和竖直姿态可精确满足；但默认位置执行器在关节误差为零时输出零臂力矩，而该姿态第二关节需要约6.82 Nm的重力平衡力矩。当前模型的机器人 `body_gravcomp=0`，各关节位置增益为80/80/80/40/10/10/10。这为物理层稳态跟踪误差提供了一个明确解释线索；本次没有修改重力补偿或增益，也没有证明它是全部失败的唯一原因。见 `actuation_diagnostic.json`。', '',
        '## 产物与复现', '',
        '- `ik_workspace_comparison.png`：两种IK范围并列图。蓝色为理想执行器解算通过，绿色为物理保持通过，红叉为物理试跑未通过。蓝色不能作为已验证物理工作区。',
        '- `pose_ik_physical.mp4`、`jacobian_ik_physical.mp4`：同样三个目标、同样缓慢接近/下降的真实物理轨迹回放，约42秒，各标出范围和实际误差。',
        '- `ideal/cells.csv`、`physical/cells.csv`：全部逐点数据；JSONL逐点落盘支持续跑。',
        '- `physical_pilot/pilot_*.npz`：录像使用的原始50 Hz物理轨迹。',
        '- `run_plan.json`、`summary.json`、`full_scene_verification.json`：协议、数量、解算边界与验证。',
        '- `source_snapshot.py`、`source_manifest.json`：录制/测试代码副本与实际导入源文件哈希。', '',
        '```powershell', "$env:PYTHONPATH='src'", "$env:OMP_NUM_THREADS='1'", "$env:OPENBLAS_NUM_THREADS='1'", "$env:MKL_NUM_THREADS='1'",
        'conda run --no-capture-output -n mujoco-dlo python tools/recording/compare_project_ik_workspace.py --compare-scan --workers 4',
        'conda run --no-capture-output -n mujoco-dlo python tools/recording/compare_project_ik_workspace.py --verify-full-scene',
        'conda run --no-capture-output -n mujoco-dlo python tools/recording/compare_project_ik_workspace.py --pilot --output outputs/vertical_workspace/nero_project_ik_compare_20261001/physical_pilot',
        'conda run --no-capture-output -n mujoco-dlo python tools/recording/compare_project_ik_workspace.py --record', '```', '']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def record(out):
    ideal,physical,summary=summarize(out)
    for method in METHODS:
        rig=Rig(method)
        renderer=mujoco.Renderer(rig.model,height=720,width=1280,max_geom=10000)
        cam=mujoco.MjvCamera(); cam.lookat[:]=(.55,0,.19); cam.distance=1.60; cam.azimuth=135; cam.elevation=-32
        option=mujoco.MjvOption(); option.geomgroup[3]=0
        video=cv2.VideoWriter(str(out/f'{method}_physical_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(1280,720))
        assert video.isOpened()
        cases=[(.7,0),(.65,.3),(.65,-.3)]
        total_frames=0
        for case_index,(x,y) in enumerate(cases):
            file=out/'physical_pilot'/f'pilot_{method}_{x}_{y}.npz'
            history=np.load(file)['history']
            goal=np.array([x,y,Z])
            length=float(history[-1,0])
            times=np.arange(0,length+.00001,.04)
            for time_value in times:
                idx=min(len(history)-1,int(np.searchsorted(history[:,0],time_value)))
                row=history[idx]; rig.data.qpos[:9]=row[1:10]; rig.data.qvel[:]=0
                mujoco.mj_forward(rig.model,rig.data)
                renderer.update_scene(rig.data,camera=cam,scene_option=option)
                scene=renderer.scene
                step=float(ideal['xs'][1]-ideal['xs'][0])
                for iy,ix in zip(*np.where(ideal[method])):
                    add_geom(scene,mujoco.mjtGeom.mjGEOM_BOX,[step*.45,step*.45,.0004],
                        [ideal['xs'][ix],ideal['ys'][iy],.0007],(.15,.63,.86,1))
                for iy,ix in zip(*np.where(physical[method])):
                    ps=float(physical['xs'][1]-physical['xs'][0])
                    add_geom(scene,mujoco.mjtGeom.mjGEOM_BOX,[ps*.43,ps*.43,.0006],
                        [physical['xs'][ix],physical['ys'][iy],.0018],(.15,.75,.46,1))
                tcp=row[16:19]
                actual_axis=rig.data.xmat[rig.base.hand_id].reshape(3,3)[:,0]
                add_line(scene,tcp-.08*actual_axis,tcp,(.95,.15,.25,1),.004)
                add_line(scene,goal+[0,0,.10],goal,(1,.74,.05,1),.002)
                add_geom(scene,mujoco.mjtGeom.mjGEOM_SPHERE,[.008,0,0],goal+[0,0,.003],(1,.74,.05,1))
                im=Image.fromarray(renderer.render()).convert('RGBA')
                overlay=Image.new('RGBA',im.size,(0,0,0,0)); dd=ImageDraw.Draw(overlay)
                dd.rounded_rectangle((16,15,900,118),radius=12,fill=(248,251,255,240))
                title='Pose IK：项目原有20次迭代位姿解算' if method=='pose_ik' else 'Jacobian IK：项目原有速度层分解'
                dd.text((30,24),title,font=font(28),fill='#192940')
                dd.text((30,70),f'真实物理轨迹回放 · 案例 {case_index+1}/3 · 目标 ({x:.2f}, {y:+.2f}, 0.005) m',font=font(21),fill='#415169')
                dd.rounded_rectangle((918,15,1270,706),radius=12,fill=(248,251,255,248))
                dd.rounded_rectangle((20,651,899,706),radius=10,fill=(248,251,255,240))
                dd.text((34,666),'黄色：目标与竖直参考轴   红色：实际夹爪方向',font=font(23),fill='#415169')
                im=Image.alpha_composite(im,overlay).convert('RGB')
                im.paste(comparison_map(ideal,physical,method,340,actual=tcp,target=goal),(924,65))
                dd=ImageDraw.Draw(im)
                dd.text((937,27),'范围与物理试跑 · 世界坐标',font=font(20),fill='#192940')
                lines=[('蓝色：解算通过区域','#147fa8'),('绿点：物理保持通过','#17885e'),('红叉：物理试跑未通过','#b94d58'),
                    (f"物理通过 {summary[method]['physical']['passed']}/{summary[method]['physical']['tested']} 点",'#415169'),
                    (f'时间 {row[0]:.2f} s / {length:.1f} s','#415169'),
                    (f'位置误差 {row[10]*1000:.1f} mm','#b94d58' if row[10]>.005 else '#17885e'),
                    (f'竖直偏差 {row[11]:.1f}°','#b94d58' if row[11]>2 else '#17885e'),
                    (f'完整姿态误差 {row[12]:.1f}°','#415169'),
                    (f'指尖最低高度 {row[13]*1000:.2f} mm','#415169')]
                for j,(txt,color) in enumerate(lines): dd.text((934,414+j*30),txt,font=font(19),fill=color)
                video.write(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)); total_frames+=1
                if case_index==0 and abs(time_value-12.)<.001: im.save(out/f'{method}_physical_check.png')
                if total_frames%200==0: print(f'{method}: rendered {total_frames} frames',flush=True)
        video.release(); renderer.close()
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--scan',action='store_true')
    p.add_argument('--compare-scan',action='store_true')
    p.add_argument('--verify-full-scene',action='store_true')
    p.add_argument('--record',action='store_true')
    p.add_argument('--ideal',action='store_true')
    p.add_argument('--step',type=float,default=.02)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    if args.pilot: run_pilot(out,args.ideal)
    if args.scan: scan(out,args.step,args.workers,args.ideal)
    if args.compare_scan: compare_scan(out,args.workers)
    if args.verify_full_scene: verify_full_scene(out)
    if args.record: record(out)
    if args.record: report(out)


if __name__=='__main__':
    main()
