"""Measure actual unloaded near-floor motion with the frozen improved rule IK.

Uses the benchmark's DiagnosticPolicy / DiagnosticEnv unchanged. Only the
requested pose/path is supplied by this workspace task. No ideal servo replay
or SciPy solve is used to decide reachability.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/evaluation'))
import compare_scripted_tracking_fix as frozen
import compare_project_ik_workspace as old
from render_vertical_workspace import font,add_geom,add_line
from panda_cable_grasp.policies.scripted import PolicyConfig,Phase

OUT=ROOT/'outputs/vertical_workspace/nero_improved_jacobian_20261001'
VARIANT='tracking_fix_orientation4'
DT=.02
Z=.005
SOURCES=('src/panda_cable_grasp/policies/scripted.py','src/panda_cable_grasp/env/environment.py',
    'src/panda_cable_grasp/env/kinematics.py','assets/mujoco/nero/nero.xml',
    'tools/evaluation/compare_scripted_tracking_fix.py','tools/recording/render_vertical_workspace.py',
    'tools/recording/compare_project_ik_workspace.py','tools/recording/record_improved_workspace.py')


def env_factory(config):
    frozen.OUT=OUT/'control_audit'
    env=frozen.DiagnosticEnv(config,VARIANT)
    # Construct while the original cable is present. Workspace motion never
    # invokes target selection or grasp state-machine updates.
    env.workspace_policy=frozen.DiagnosticPolicy(env,PolicyConfig())
    return env


class ImprovedRig(old.Rig):
    def __init__(self):
        old.CableGraspEnv=env_factory
        super().__init__('improved_scripted_jacobian')
        self.policy=self.base.workspace_policy
        self.policy.desired_quat=self.quat.copy()
        self.policy.desired_approach_axis=self.geometry.rotation()@self.base.gripper_approach_axis_local
        self.policy.phase=Phase.INTERCEPT
        self.audit.update(controller='DiagnosticPolicy._ik_action (project scripted Jacobian)',
            variant=VARIANT,policy_config=asdict(self.policy.config),
            workspace_target='fingers +local-X downward; fixed yaw, as prior workspace task',
            gravity_feedforward='frozen DiagnosticEnv._physics_step at 500 Hz',
            arm_kp_scale=4.,arm_kv_scale=2.,orientation_error_scale=4.,
            adapter_not_used_for_control=True)
        self.reset()

    def reset(self):
        super().reset()
        if hasattr(self,'policy'):
            self.policy.phase=Phase.INTERCEPT
        self.base.max_ff=np.zeros(7);self.base.max_total_torque=np.zeros(7)
        self.base.force_saturations=0
        self.peak_velocity=np.zeros(7);self.velocity_steps=0;self.over_steps=0

    def control(self,target):
        self.policy.phase=Phase.INTERCEPT
        request=self.policy._ik_action(np.asarray(target),self.base.gripper_open_ctrl)
        applied=self.base._limit_robot_action(np.clip(request,self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1]))
        for _ in range(10):
            guarded,_=self.base._velocity_guarded_action(applied)
            self.data.ctrl[:]=guarded
            self.base._physics_step()
            velocity=np.abs(self.data.qvel[self.base.arm_dof_adr])
            self.peak_velocity=np.maximum(self.peak_velocity,velocity)
            self.velocity_steps+=1
            self.over_steps+=int(np.any(velocity>self.base._arm_velocity_limits+1e-12))
        mujoco.mj_forward(self.model,self.data)
        return applied

    def trial(self,x,y,record=False):
        self.reset();target=np.array([x,y,Z]);reference=self.base.hand_position.copy()
        approach=np.array([x,y,.10]);descend=False;held=0;history=[]
        min_floor=1.;min_self=1.;max_tilt=0.;max_pose=0.;qualified=False;reason='timeout'
        start=time.monotonic()
        for k in range(800):
            delta=(target if descend else approach)-reference
            reference+=delta*min(1.,(.03 if descend else .15)*DT/max(np.linalg.norm(delta),1e-12))
            if not descend and np.linalg.norm(reference-approach)<1e-9:descend=True
            self.control(reference)
            m=self.measure(target,collision=(k%5==0 or held>0))
            min_floor=min(min_floor,m[3]);min_self=min(min_self,m[5]);max_tilt=max(max_tilt,m[1]);max_pose=max(max_pose,m[2])
            finger_floor=min(float(self.geometry.geom_vertices(g)[:,2].min()) for g in self.geometry.finger_geoms)
            ok=m[0]<=.005 and m[2]<=2. and .0002<=finger_floor<=.008 and m[3]>=.0002 and m[4]>=0 and m[5]>=.001
            held=held+1 if ok else 0
            if record:history.append(np.r_[self.data.time,self.data.qpos[:9],m,target,reference,finger_floor])
            if held>=100:
                qualified=True;reason='endpoint_held_2s';break
            if m[3]<-.001:reason='floor_penetration';break
            if m[5]<-.001:reason='self_collision';break
            if not np.all(np.isfinite(self.data.qpos)):reason='invalid_nonfinite';break
        final=self.measure(target,collision=True)
        ratio=float(np.max(self.peak_velocity/self.base._arm_velocity_limits))
        # Endpoint reachability and path suitability are distinct observations.
        # A 5-degree full-pose envelope is checked over the complete approach.
        path_vertical=bool(max_pose<=5.)
        motion_pass=bool(qualified and path_vertical and ratio<=1.01)
        row={'x_m':float(x),'y_m':float(y),'z_m':Z,'endpoint_pass':qualified,'motion_pass':motion_pass,
            'reason':reason,'steps':k+1,'sim_seconds':self.data.time,'wall_seconds':time.monotonic()-start,
            'position_error_mm':final[0]*1000,'tilt_deg':final[1],'pose_error_deg':final[2],
            'robot_floor_mm':final[3]*1000,'finger_floor_mm':finger_floor*1000,
            'joint_margin_deg':final[4],'self_distance_mm':final[5]*1000,
            'min_path_floor_mm':min_floor*1000,'min_path_self_mm':min_self*1000,
            'max_path_tilt_deg':max_tilt,'max_path_pose_deg':max_pose,'path_vertical_5deg':path_vertical,
            'peak_velocity_to_limit':ratio,'velocity_exceedance_substep_ratio':self.over_steps/max(1,self.velocity_steps),
            'peak_joint_velocity_rad_s':self.peak_velocity.tolist(),
            'max_combined_torque_nm':self.base.max_total_torque.tolist(),
            'torque_saturation_steps':self.base.force_saturations,'actual_tcp_m':final[6:9].tolist()}
        return row,np.asarray(history)


_RIG=None
def worker(job):
    global _RIG
    if _RIG is None:_RIG=ImprovedRig()
    return _RIG.trial(*job)[0]


def manifest(step):
    rig=ImprovedRig()
    plan={'robot':'NERO','controller':VARIANT,'grid_step_m':step,'world_base_m':[.35,0,0],
        'grid_x_m':[-.25,.95],'grid_y_m':[-.60,.60],'tcp_z_m':Z,'control_dt_s':DT,
        'position_tolerance_mm':5.,'terminal_full_pose_tolerance_deg':2.,'hold_s':2.,
        'finger_floor_mm':[.2,8.],'maximum_trial_s':16.,'fixed_yaw':True,
        'approach_speed_m_s':.15,'descent_speed_m_s':.03,'approach_z_m':.10,
        'motion_path_pose_limit_deg':5.,'motion_peak_velocity_ratio_limit':1.01,
        'velocity_ratio_tolerance_note':'1% numerical allowance; exact exceedance ratio also recorded',
        'scope':'unloaded actual-physics workspace; requested targets, not full grasp episodes; no real hardware claim',
        'audit':rig.audit,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}}
    (OUT/'manifest.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    mujoco.mj_saveModel(rig.model,str(OUT/'isolated_model.mjb'),None)
    print(json.dumps({'home_tcp':rig.base.hand_position.tolist(),'home_rotation':rig.data.xmat[rig.base.hand_id].reshape(3,3).tolist(),'desired_rotation':rig.geometry.rotation().tolist()},indent=2),flush=True)


def pilot():
    rig=ImprovedRig();rows=[]
    for x,y in ((.7,0),(.65,.3),(.65,-.3),(.4,.4),(.82,0),(.55,0)):
        row,h=rig.trial(x,y,True);rows.append(row)
        np.savez_compressed(OUT/f'pilot_{x}_{y}.npz',history=h)
        print(json.dumps(row),flush=True)
    (OUT/'pilot.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


def scan(step,workers):
    xs=np.round(np.arange(-.25,.951,step),6);ys=np.round(np.arange(-.6,.601,step),6)
    file=OUT/'cells.jsonl';rows=[json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
    done={(r['x_m'],r['y_m']) for r in rows};jobs=[(float(x),float(y)) for y in ys for x in xs if (float(x),float(y)) not in done]
    started=time.monotonic()
    with file.open('a',encoding='utf-8') as f,mp.get_context('spawn').Pool(workers) as pool:
        for r in pool.imap_unordered(worker,jobs,chunksize=1):
            rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
            state={'completed':len(rows),'total':len(xs)*len(ys),'endpoint_pass':sum(r['endpoint_pass'] for r in rows),
                'motion_pass':sum(r['motion_pass'] for r in rows),'elapsed_wall_s':time.monotonic()-started}
            (OUT/'progress.json').write_text(json.dumps(state,indent=2))
            if len(rows)%25==0:print(json.dumps(state),flush=True)
    (OUT/'cells.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (OUT/'cells.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    data={'xs':xs,'ys':ys}
    for key in ('endpoint_pass','motion_pass'):
        mask=np.zeros((len(ys),len(xs)),bool)
        for r in rows:mask[np.argmin(abs(ys-r['y_m'])),np.argmin(abs(xs-r['x_m']))]=r[key]
        data[key]=mask
    data['interior']=data['motion_pass']&(ndimage.distance_transform_edt(np.pad(data['motion_pass'],1))[1:-1,1:-1]*step>=.06)
    np.savez_compressed(OUT/'workspace.npz',**data)
    summarize(data,rows,step)


def bounds(d,mask,step):
    iy,ix=np.where(mask)
    if not len(ix):return None
    radius=np.hypot(d['xs'][ix]-.35,d['ys'][iy])
    return {'cells':len(ix),'area_m2':len(ix)*step**2,'x_m':[float(d['xs'][ix].min()),float(d['xs'][ix].max())],
        'y_m':[float(d['ys'][iy].min()),float(d['ys'][iy].max())],'base_radius_m':[float(radius.min()),float(radius.max())]}


def summarize(d,rows,step):
    result={key:bounds(d,d[key],step) for key in ('endpoint_pass','motion_pass','interior')}
    result.update(tested=len(rows),grid_step_m=step,counts={reason:sum(r['reason']==reason for r in rows) for reason in sorted({r['reason'] for r in rows})},
        endpoint_pass_but_path_pose_failed=sum(r['endpoint_pass'] and not r['path_vertical_5deg'] for r in rows),
        endpoint_pass_but_actual_speed_failed=sum(r['endpoint_pass'] and r['peak_velocity_to_limit']>1.01 for r in rows),
        bounding_boxes_contain_holes=True,scan_boundary_hit=bool(np.any(d['endpoint_pass'][[0,-1]]) or np.any(d['endpoint_pass'][:,[0,-1]])))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--scan',action='store_true')
    p.add_argument('--step',type=float,default=.04);p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'manifest.json').exists():manifest(args.step)
    if args.pilot:pilot()
    if args.scan:scan(args.step,args.workers)


if __name__=='__main__':main()
