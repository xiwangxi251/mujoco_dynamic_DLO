"""Enhanced full6: identical unlimited reference inputs, compare DLS/box DLS."""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import scan_full6_enhanced_workspace as base
import numpy as np
import mujoco
from scipy.optimize import lsq_linear

OUT=base.OUT/'bounded_dls_comparison'
_RIG=None
_VARIANT=None
GOALS=((.55,-.28,'negative_y'),(.55,.28,'positive_y'),
       (.67,-.28,'negative_y_outer'),(.67,.28,'positive_y_outer'),
       (.71,0.,'center'),(.47,.16,'edge_failure'))


class ComparisonRig(base.EnhancedRig):
    def __init__(self,variant='bounded'):
        self.variant=variant
        super().__init__()
        self.audit.update(experiment_variant=variant,
            reference_speed_ramp_enabled=False,actual_arrival_gate_enabled=False,
            reference_protocol='upper waypoint at first 20 ms update; final target at all subsequent updates',
            joint_position_bounds_in_solver=(variant=='bounded'),
            nullspace_preference='same exact full-task ready pull, inside regularizer for bounded solve')
        if variant=='bounded':
            self.audit['controller']='full6 DLS with joint position and velocity box constraints'
            self.policy._ik_action=self.bounded_action

    def reset(self):
        super().reset()
        self.solve_times=[]
        self.solve_active_steps=0
        self.solve_failures=0
        self.command_clip_steps=0
        self.max_command_clip=0.
        self.max_solver_bound_violation=0.

    def task_data(self,target):
        p=self.policy
        J=p._task_jacobian()
        linear=p._limit_vector_norm(6.*(target-self.env.hand_position),
                                  p.config.nero_standard_jacobian_linear_velocity_limit)
        error,_=p._orientation_error()
        angular=p._limit_vector_norm(2.5*error,1.40)
        q=self.data.qpos[self.env.arm_qpos_adr].copy()
        bias=(np.eye(7)-np.linalg.pinv(J,rcond=1e-5)@J)@(.8*(self.env.ready_qpos[:7]-q))
        limits=p.config.policy_joint_velocity_fraction*np.asarray(self.env.config.arm_joint_velocity_limits)
        ranges=self.model.jnt_range[self.env.arm_joint_ids]
        return J,np.r_[linear,angular],q,bias,limits,ranges

    def bounded_action(self,target,gripper):
        start=time.perf_counter()
        J,V,q,bias,limits,ranges=self.task_data(np.asarray(target))
        h=self.policy.config.ik_target_horizon
        lower=np.maximum(-limits,(ranges[:,0]-q)/h)
        upper=np.minimum(limits,(ranges[:,1]-q)/h)
        if np.any(lower>=upper):
            self.solve_failures+=1
            raise RuntimeError('joint position/velocity bounds infeasible at current actual state')
        damping=self.policy.config.nero_standard_jacobian_damping
        A=np.vstack((J,damping*np.eye(7)))
        b=np.r_[V,damping*bias]
        solve=lsq_linear(A,b,bounds=(lower,upper),method='bvls',tol=1e-10,max_iter=100)
        if not solve.success or not np.all(np.isfinite(solve.x)):
            self.solve_failures+=1
            raise RuntimeError(f'bounded DLS failed: {solve.message}')
        velocity=solve.x
        self.max_solver_bound_violation=max(self.max_solver_bound_violation,
            float(np.max(np.maximum(lower-velocity,velocity-upper))))
        self.solve_active_steps+=int(np.any(solve.active_mask!=0))
        unbounded=q+h*velocity
        clipped=np.clip(unbounded,ranges[:,0],ranges[:,1])
        clip=float(np.max(abs(unbounded-clipped)))
        self.max_command_clip=max(self.max_command_clip,clip)
        self.command_clip_steps+=int(clip>1e-9)
        action=np.empty(8);action[:7]=clipped;action[self.env.gripper_actuator_id]=gripper
        self.policy.last_desired=np.asarray(target).copy()
        self.solve_times.append(time.perf_counter()-start)
        return action

    def control(self,target,phase):
        if self.variant=='baseline':
            self.policy.phase=phase
            J,V,q,bias,limits,ranges=self.task_data(np.asarray(target))
            damping=self.policy.config.nero_standard_jacobian_damping
            velocity=J.T@np.linalg.solve(J@J.T+damping**2*np.eye(6),V)+bias
            velocity*=min(1.,float(np.min(limits/np.maximum(abs(velocity),1e-12))))
            raw=q+self.policy.config.ik_target_horizon*velocity
            clip=float(np.max(abs(raw-np.clip(raw,ranges[:,0],ranges[:,1]))))
            self.command_clip_steps+=int(clip>1e-9)
            self.max_command_clip=max(self.max_command_clip,clip)
        super().control(target,phase)

    def trial(self,x,y):
        self.reset()
        target=np.array([x,y,base.native.Z],dtype=float)
        approach=np.array([x,y,.10])
        held=0;min_floor=min_self=1.;max_tilt=max_pose=0.;reason='timeout'
        started=time.monotonic()
        for step in range(800):
            reference=approach if step==0 else target
            self.control(reference,base.native.Phase.INTERCEPT)
            m=self.measure(target,collision=(step%5==0 or held>0))
            min_floor=min(min_floor,m[3]);min_self=min(min_self,m[5])
            max_tilt=max(max_tilt,m[1]);max_pose=max(max_pose,m[2])
            finger=min(float(self.geometry.geom_vertices(g)[:,2].min()) for g in self.geometry.finger_geoms)
            acceptable=(m[0]<=.005 and m[2]<=2. and .0002<=finger<=.008
                and m[3]>=.0002 and m[4]>=0. and m[5]>=.001)
            held=held+1 if acceptable else 0
            if held>=100:reason='endpoint_held_2s';break
            if m[3]<-.001:reason='floor_penetration';break
            if m[5]<-.001:reason='self_collision';break
            if not np.all(np.isfinite(self.data.qpos)):reason='invalid_nonfinite';break
        m=self.measure(target,collision=True)
        ratio=float(np.max(self.peak_velocity/self.env._arm_velocity_limits))
        endpoint=reason=='endpoint_held_2s'
        return dict(x_m=float(x),y_m=float(y),z_m=base.native.Z,
            endpoint_pass=endpoint,motion_pass=bool(endpoint and max_pose<=5. and ratio<=1.01),
            reason=reason,steps=step+1,sim_seconds=float(self.data.time),wall_seconds=time.monotonic()-started,
            position_error_mm=m[0]*1000,tilt_deg=m[1],pose_error_deg=m[2],
            robot_floor_mm=m[3]*1000,finger_floor_mm=finger*1000,joint_margin_deg=m[4],self_distance_mm=m[5]*1000,
            min_path_floor_mm=min_floor*1000,min_path_self_mm=min_self*1000,
            max_path_tilt_deg=max_tilt,max_path_pose_deg=max_pose,peak_velocity_to_limit=ratio,
            velocity_exceedance_substep_ratio=self.over_steps/max(1,self.velocity_steps),
            peak_joint_velocity_rad_s=self.peak_velocity.tolist(),table_proximity_limited_steps=int(self.env._table_proximity_limited_steps),
            max_gravity_ff_nm=self.max_ff.tolist(),max_combined_torque_nm=self.max_torque.tolist(),
            torque_saturation_substeps=self.saturation_steps,velocity_guard_substeps=self.guard_steps,
            joint_target_clip_steps=self.command_clip_steps,max_joint_target_clip_rad=self.max_command_clip,
            solver_bound_active_steps=self.solve_active_steps,solver_failures=self.solve_failures,
            max_solver_bound_violation_rad_s=self.max_solver_bound_violation,
            solver_median_ms=float(np.median(self.solve_times)*1000) if self.solve_times else None,
            solver_p99_ms=float(np.quantile(self.solve_times,.99)*1000) if self.solve_times else None,
            solver_max_ms=float(max(self.solve_times)*1000) if self.solve_times else None)


def source_hashes():
    old=json.loads((base.OUT/'manifest.json').read_text())
    files=[Path(__file__)]+[Path(p) for p in old['source_sha256']]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    old=ComparisonRig('baseline');new=ComparisonRig('bounded')
    assert old.env.config==new.env.config
    assert old.policy.config==new.policy.config
    for name in ('jnt_range','actuator_gainprm','actuator_biasprm','actuator_forcerange'):
        assert np.array_equal(getattr(old.model,name),getattr(new.model,name)),name
    for rig in (old,new):
        rig.data.qpos[:7]+=np.array([.03,-.02,.02,.04,-.03,.01,-.02])
        mujoco.mj_forward(rig.model,rig.data)
    target=old.env.hand_position.copy()+np.array([.001,-.002,.001])
    J,V,q,bias,limits,ranges=new.task_data(target)
    raw=J.T@np.linalg.solve(J@J.T+.04**2*np.eye(6),V)+bias
    assert np.all(abs(raw)<limits) and np.all(q+.11*raw>ranges[:,0]) and np.all(q+.11*raw<ranges[:,1])
    expected=old.policy._ik_action(target,old.env.gripper_open_ctrl)
    actual=new.policy._ik_action(target,new.env.gripper_open_ctrl)
    inactive_difference=float(np.max(abs(expected-actual)))
    assert inactive_difference<1e-10,inactive_difference
    inputs=[];control=old.control
    def record(target,phase):
        if len(inputs)<3:inputs.append(np.asarray(target).tolist())
        return control(target,phase)
    old.control=record
    old.trial(.55,-.28)
    assert inputs==[[.55,-.28,.10],[.55,-.28,.005],[.55,-.28,.005]]
    checks=dict(same_environment_configuration=True,same_policy_parameters=True,same_servo_and_force_bounds=True,
        unconstrained_action_max_difference=inactive_difference,actual_arrival_gate_enabled=False,
        reference_speed_ramp_enabled=False,first_three_references=inputs,source_sha256=source_hashes())
    (OUT/'implementation_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


def pilot():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for variant in ('baseline','bounded'):
        rig=ComparisonRig(variant)
        for x,y,name in GOALS:
            states=[];references=[];control=rig.control
            def record(target,phase):
                result=control(target,phase)
                states.append(rig.data.qpos[:9].copy());references.append(np.asarray(target).copy())
                return result
            rig.control=record
            row=rig.trial(x,y);rig.control=control
            row.update(variant=variant,goal_name=name);rows.append(row)
            np.savez_compressed(OUT/f'pilot_{variant}_{name}.npz',qpos=states,reference=references)
            print(json.dumps(row),flush=True)
    (OUT/'pilot_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


def worker(job):
    global _RIG,_VARIANT
    variant,x,y=job
    if _RIG is None or _VARIANT!=variant:_RIG=ComparisonRig(variant);_VARIANT=variant
    row=_RIG.trial(x,y);row['variant']=variant
    return row


def scan(workers):
    original=json.loads((base.OUT/'manifest.json').read_text())
    xs=np.asarray(original['grid_x_m']);ys=np.asarray(original['grid_y_m'])
    jobs=[];rows={};files={}
    for variant in ('baseline','bounded'):
        dest=OUT/variant;dest.mkdir(parents=True,exist_ok=True)
        rig=ComparisonRig(variant)
        manifest=dict(original)
        manifest.update(controller=rig.audit,reference_speed_ramp_enabled=False,actual_arrival_gate_enabled=False,
            approach_speed_m_s=None,descent_speed_m_s=None,
            waypoint_transition='reference jumps to upper waypoint first frame; final target from second frame, no actual-state gate',
            same_protocol_as='baseline/bounded comparison with identical reference schedule',source_sha256=source_hashes())
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        mujoco.mj_saveModel(rig.model,str(dest/'isolated_model.mjb'),None)
        path=dest/'cells.jsonl';rows[variant]=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        done={(r['x_m'],r['y_m']) for r in rows[variant]}
        jobs.extend((variant,float(x),float(y)) for y in ys for x in xs if (float(x),float(y)) not in done)
        files[variant]=path.open('a',encoding='utf-8')
    # Keep workers on the same rig across chunks while both arms are evaluated.
    jobs.sort(key=lambda job:(job[2],job[0],job[1]))
    start=time.monotonic()
    try:
        with mp.get_context('spawn').Pool(workers) as pool:
            for row in pool.imap_unordered(worker,jobs,chunksize=8):
                variant=row['variant'];rows[variant].append(row)
                files[variant].write(json.dumps(row)+'\n');files[variant].flush()
                done=sum(len(r) for r in rows.values())
                if done%100==0:print(json.dumps(dict(done=done,total=1922,by_variant={v:len(r) for v,r in rows.items()},wall_s=round(time.monotonic()-start,1))),flush=True)
    finally:
        for f in files.values():f.close()
    for variant in rows:
        base.OUT=OUT/variant
        base.summarize(rows[variant],xs,ys,.04)


def videos(variant):
    from record_retest_136_videos import capture
    def factory(directory):
        rig=ComparisonRig(variant)
        return rig,rig.env,base.native
    for x,y,name in GOALS:
        capture(6,x,y,name,rig_factory=factory,output_root=OUT/'videos'/variant,
            scan_directory=OUT/variant,
            label='增强6 · '+('带约束DLS' if variant=='bounded' else '原DLS'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('audit','pilot','scan','videos'),default='pilot')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--variant',choices=('baseline','bounded'),default='bounded')
    args=p.parse_args()
    if args.mode=='audit':audit()
    elif args.mode=='pilot':pilot()
    elif args.mode=='scan':scan(args.workers)
    else:videos(args.variant)
