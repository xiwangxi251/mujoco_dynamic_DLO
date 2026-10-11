"""Scan native full-6D DLS with the previously probed three interventions."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import mujoco
import numpy as np
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[2]
WORKTREE=ROOT.parent/'worktrees/scripted-tracking-20261001'
sys.path.insert(0,str(WORKTREE/'src'))
sys.path.insert(0,str(WORKTREE/'tools/recording'))
import record_standard_jik_workspace as native

OUT=ROOT/'outputs/vertical_workspace/scripted_136_retest_20261004/full6_gravity_servo_gain10'
_RIG=None

class EnhancedRig(native.Rig):
    def __init__(self):
        super().__init__(cartesian_linear_velocity_limit=.90)
        self.model.actuator_gainprm[:7,0]*=4
        self.model.actuator_biasprm[:7,1]*=4
        self.model.actuator_biasprm[:7,2]*=2
        original_error=self.policy._orientation_error
        def error10():
            error,angle=original_error()
            return error*(10./6.),angle
        self.policy._orientation_error=error10
        self.gravity_probe=mujoco.MjData(self.model)
        self.audit.update(orientation_gain=10.,arm_kp_scale=4.,arm_kv_scale=2.,
            gravity_feedforward_enabled=True,
            gravity_feedforward='qfrc_bias at qvel=0; 500 Hz; bounded together with actuator PD torque')
        self.reset()

    def reset(self):
        super().reset()
        self.max_ff=np.zeros(7)
        self.max_torque=np.zeros(7)
        self.saturation_steps=0
        self.guard_steps=0

    def control(self,target,phase):
        self.policy.phase=phase
        request=self.policy._ik_action(np.asarray(target,dtype=float),self.env.gripper_open_ctrl)
        applied=self.env._limit_robot_action(np.clip(request,self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1]))
        for _ in range(10):
            guarded,active=self.env._velocity_guarded_action(applied)
            self.guard_steps+=int(active)
            self.data.ctrl[:]=guarded
            self.data.qfrc_applied[:]=0
            pd=(self.model.actuator_gainprm[:7,0]*guarded[:7]
                +self.model.actuator_biasprm[:7,0]
                +self.model.actuator_biasprm[:7,1]*self.data.qpos[self.env.arm_qpos_adr]
                +self.model.actuator_biasprm[:7,2]*self.data.qvel[self.env.arm_dof_adr])
            bounds=self.model.actuator_forcerange[:7]
            pd=np.clip(pd,bounds[:,0],bounds[:,1])
            self.gravity_probe.qpos[:]=self.data.qpos
            self.gravity_probe.qvel[:]=0
            mujoco.mj_forward(self.model,self.gravity_probe)
            requested_ff=self.gravity_probe.qfrc_bias[self.env.arm_dof_adr].copy()
            total=np.clip(pd+requested_ff,bounds[:,0],bounds[:,1])
            self.saturation_steps+=int(np.any(abs(total-pd-requested_ff)>1e-9))
            ff=total-pd
            self.data.qfrc_applied[self.env.arm_dof_adr]=ff
            self.max_ff=np.maximum(self.max_ff,abs(ff))
            self.max_torque=np.maximum(self.max_torque,abs(total))
            mujoco.mj_step(self.model,self.data)
            velocity=abs(self.data.qvel[self.env.arm_dof_adr])
            self.peak_velocity=np.maximum(self.peak_velocity,velocity)
            self.velocity_steps+=1
            self.over_steps+=int(np.any(velocity>self.env._arm_velocity_limits+1e-12))
        mujoco.mj_forward(self.model,self.data)

    def trial(self,x,y):
        row=super().trial(x,y)
        row.update(max_gravity_ff_nm=self.max_ff.tolist(),max_combined_torque_nm=self.max_torque.tolist(),
            torque_saturation_substeps=self.saturation_steps,velocity_guard_substeps=self.guard_steps)
        return row

def worker(job):
    global _RIG
    if _RIG is None:_RIG=EnhancedRig()
    return _RIG.trial(*job)

def pilot():
    rig=EnhancedRig()
    results=[]
    for x,y in ((.7,0.),(.65,.3)):
        states=[]
        original=rig.control
        def record(target,phase):
            original(target,phase)
            states.append(rig.data.qpos[:9].copy())
        rig.control=record
        row=rig.trial(x,y)
        rig.control=original
        file=OUT.parent/'pose_diagnosis'/f'full_all_three_{x}_{y}.npz'
        with np.load(file) as saved:reference=saved['qpos']
        difference=float(np.max(abs(np.asarray(states)-reference[:len(states)])))
        assert difference<1e-12,difference
        np.savez_compressed(OUT/f'pilot_{x}_{y}.npz',qpos=np.asarray(states))
        results.append(dict(goal=[x,y],result=row,previous_ablation_prefix_max_difference=difference))
        print(json.dumps(results[-1]),flush=True)
    (OUT/'pilot_verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

def summarize(rows,xs,ys,step):
    rows.sort(key=lambda r:(r['y_m'],r['x_m']))
    assert len(rows)==len(xs)*len(ys)==len({(r['x_m'],r['y_m']) for r in rows})
    data=dict(xs=xs,ys=ys)
    for key in ('endpoint_pass','motion_pass'):
        mask=np.zeros((len(ys),len(xs)),bool)
        for r in rows:mask[np.argmin(abs(ys-r['y_m'])),np.argmin(abs(xs-r['x_m']))]=r[key]
        data[key]=mask
    data['interior']=data['motion_pass']&(ndimage.distance_transform_edt(np.pad(data['motion_pass'],1))[1:-1,1:-1]*step>=.06)
    summary=dict(tested=len(rows),grid_step_m=step,tcp_z_m=.005,
        counts={reason:sum(r['reason']==reason for r in rows) for reason in sorted({r['reason'] for r in rows})})
    for key in ('endpoint_pass','motion_pass','interior'):
        mask=data[key];iy,ix=np.where(mask);selected=[r for r in rows if mask[np.argmin(abs(ys-r['y_m'])),np.argmin(abs(xs-r['x_m']))]]
        radius=np.hypot(xs[ix]-.35,ys[iy])
        summary[key]=dict(cells=len(ix),sampled_area_m2=len(ix)*step**2,
            x_m=[float(xs[ix].min()),float(xs[ix].max())] if len(ix) else None,
            y_m=[float(ys[iy].min()),float(ys[iy].max())] if len(ix) else None,
            base_radius_m=[float(radius.min()),float(radius.max())] if len(ix) else None,
            median_position_error_mm=float(np.median([r['position_error_mm'] for r in selected])) if selected else None,
            median_pose_error_deg=float(np.median([r['pose_error_deg'] for r in selected])) if selected else None)
    labels,count=ndimage.label(data['motion_pass'])
    summary['motion_connected_components_4_neighbors']=count
    summary['motion_component_sizes']=sorted([int((labels==i).sum()) for i in range(1,count+1)],reverse=True)
    summary['boundary_reached']=bool(data['endpoint_pass'][[0,-1],:].any() or data['endpoint_pass'][:,[0,-1]].any())
    summary['bounding_box_contains_failed_cells']=True
    np.savez_compressed(OUT/'workspace.npz',**data)
    (OUT/'cells.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (OUT/'cells.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--pilot-only',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    step=.04;xs=np.round(np.arange(-.25,.951,step),6);ys=np.round(np.arange(-.6,.601,step),6)
    rig=EnhancedRig()
    sources=[Path(__file__),Path(native.__file__),WORKTREE/'src/panda_cable_grasp/policies/scripted.py',WORKTREE/'src/panda_cable_grasp/env/environment.py',WORKTREE/'assets/mujoco/nero/nero.xml']
    manifest=dict(controller=rig.audit,grid_x_m=xs.tolist(),grid_y_m=ys.tolist(),grid_step_m=step,
        tcp_z_m=.005,world_base_m=[.35,0,0],control_dt_s=.02,physics_dt_s=.002,
        approach_speed_m_s=.15,approach_height_m=.10,descent_speed_m_s=.03,maximum_trial_s=16.,
        position_tolerance_mm=5.,pose_tolerance_deg=2.,hold_s=2.,motion_path_pose_limit_deg=5.,
        motion_peak_velocity_ratio_limit=1.01,finger_floor_mm=[.2,8.],robot_floor_min_mm=.2,
        stop_on_robot_floor_below_mm=-1.,self_distance_min_mm=1.,same_protocol_as='scripted_136_retest_20261004 versions 1/3/6',
        scope='unloaded physical closed-loop XY scan at fixed height and yaw; grid area estimate',
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    mujoco.mj_saveModel(rig.model,str(OUT/'isolated_model.mjb'),None)
    pilot()
    if args.pilot_only:return
    file=OUT/'cells.jsonl';rows=[json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
    done={(r['x_m'],r['y_m']) for r in rows};jobs=[(float(x),float(y)) for y in ys for x in xs if (float(x),float(y)) not in done]
    start=time.monotonic()
    with file.open('a',encoding='utf-8') as stream,mp.get_context('spawn').Pool(args.workers) as pool:
        for row in pool.imap_unordered(worker,jobs,chunksize=1):
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            progress=dict(completed=len(rows),total=len(xs)*len(ys),endpoint_pass=sum(r['endpoint_pass'] for r in rows),motion_pass=sum(r['motion_pass'] for r in rows),elapsed_s=time.monotonic()-start)
            (OUT/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
            if len(rows)%50==0:print(json.dumps(progress),flush=True)
    summarize(rows,xs,ys,step)

if __name__=='__main__':main()
