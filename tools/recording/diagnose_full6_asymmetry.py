"""Separate asymmetric controller performance from sampled geometric feasibility."""
import json
import hashlib
import math
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scan_full6_enhanced_workspace import EnhancedRig, OUT, native

DEST=OUT/'asymmetry'

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    rows=json.loads((OUT/'cells.json').read_text())
    indexed={(r['x_m'],r['y_m']):r for r in rows}
    stats={}
    for name,predicate in (('positive',lambda r:r['y_m']>0),('negative',lambda r:r['y_m']<0)):
        selected=[r for r in rows if predicate(r)]
        timeouts=[r for r in selected if r['reason']=='timeout']
        stats[name]=dict(tested=len(selected),endpoint=sum(r['endpoint_pass'] for r in selected),
            motion=sum(r['motion_pass'] for r in selected),floor=sum(r['reason']=='floor_penetration' for r in selected),
            timeout=len(timeouts),timeout_final_conditions_overlapping=dict(
                position_over_5mm=sum(r['position_error_mm']>5 for r in timeouts),
                pose_over_2deg=sum(r['pose_error_deg']>2 for r in timeouts),
                joint_outside_nominal_range=sum(r['joint_margin_deg']<0 for r in timeouts),
                position_and_pose_ok_but_joint_outside=sum(r['position_error_mm']<=5 and r['pose_error_deg']<=2 and r['joint_margin_deg']<0 for r in timeouts)))
    mirrored=[(indexed[x,y],indexed[x,-y]) for x,y in indexed if y>0]
    pairing=dict(total_pairs=len(mirrored),positive_endpoint_only=sum(a['endpoint_pass'] and not b['endpoint_pass'] for a,b in mirrored),
        negative_endpoint_only=sum(b['endpoint_pass'] and not a['endpoint_pass'] for a,b in mirrored),
        positive_motion_only=sum(a['motion_pass'] and not b['motion_pass'] for a,b in mirrored),
        negative_motion_only=sum(b['motion_pass'] and not a['motion_pass'] for a,b in mirrored))
    print(json.dumps(dict(stats=stats,pairing=pairing)),flush=True)
    samples=[]
    for x,y in ((.67,.28),(.67,-.28),(.55,.28),(.55,-.28),(.63,-.24)):
        rig=EnhancedRig();env=rig.env;model,data=rig.model,rig.data
        trace=[];original=rig.control
        def record(reference,phase):
            q=data.qpos[:7].copy()
            rig.policy.phase=phase
            jac=rig.policy._task_jacobian()
            linear=rig.policy._limit_vector_norm(6*(reference-env.hand_position),.90)
            error,_=rig.policy._orientation_error()
            angular=rig.policy._limit_vector_norm(2.5*error,1.4)
            qv=jac.T@np.linalg.solve(jac@jac.T+.04**2*np.eye(6),np.r_[linear,angular])
            qv+=(np.eye(7)-np.linalg.pinv(jac,rcond=1e-5)@jac)@(.8*(env.ready_qpos[:7]-q))
            scale=min(1.,float(np.min(env._arm_velocity_limits/np.maximum(abs(qv),1e-12))))
            unbounded=q+.11*qv*scale
            lo,hi=model.jnt_range[env.arm_joint_ids].T
            clipped=np.clip(unbounded,lo,hi)
            action=rig.policy._ik_action(reference,env.gripper_open_ctrl)
            assert np.max(abs(action[:7]-clipped))<1e-10
            clipped_joints=(np.flatnonzero(abs(unbounded-clipped)>1e-10)+1).tolist()
            before=float(data.time)
            original(reference,phase)
            m=rig.measure(np.array([x,y,.005]))
            margin=np.minimum(data.qpos[:7]-lo,hi-data.qpos[:7])
            trace.append(dict(time_s=float(data.time),command_time_s=before,pose_deg=float(m[2]),position_mm=float(m[0]*1000),
                joint_margin_deg=float(np.degrees(margin.min())),nearest_joint=int(margin.argmin()+1),
                joint6_rad=float(data.qpos[5]),clipped_joints=clipped_joints,
                cartesian_sigma_min=float(np.linalg.svd(jac,compute_uv=False)[-1]),floor_mm=float(m[3]*1000)))
        rig.control=record
        row=rig.trial(x,y)
        saved=indexed[x,y]
        assert row['reason']==saved['reason'] and row['steps']==saved['steps']
        for key in ('position_error_mm','pose_error_deg','max_path_pose_deg','joint_margin_deg'):
            assert abs(row[key]-saved[key])<1e-9
        q=data.qpos[:7].copy();lo,hi=model.jnt_range[env.arm_joint_ids].T
        margin=np.minimum(q-lo,hi-q);worst=int(margin.argmin())
        sample=dict(goal_m=[x,y,.005],scan_result=row,q_final_rad=q.tolist(),nearest_joint=env.robot_spec.arm_joint_names[worst],
            joint_ranges_rad=model.jnt_range[env.arm_joint_ids].tolist(),
            first_command_clipped=next((r for r in trace if r['clipped_joints']),None),
            first_pose_over_5deg=next((r for r in trace if r['pose_deg']>5),None),
            maximum_pose_frame=max(trace,key=lambda r:r['pose_deg']),repeat_matches_scan=True)
        (DEST/f'trace_{x}_{y}.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
        samples.append(sample)
        print(json.dumps(sample),flush=True)
    geometry=[]
    prior_path=OUT.parent.parent/'nero_improved_jacobian_20261001/asymmetry_diagnostic.json'
    prior=json.loads(prior_path.read_text()) if prior_path.exists() else None
    rng=np.random.default_rng(20261004)
    for x,y in ((.67,.28),(.67,-.28)):
        rig=EnhancedRig();env=rig.env;model,data=rig.model,rig.data
        target=np.array([x,y,.005]);lo,hi=model.jnt_range[env.arm_joint_ids].T
        starts=[rig.home.copy()]
        if prior:
            for old in prior['mirrored_targets']:
                if np.allclose(old['target_m'],target) and old['fixed_full_pose_geometric_solution']:
                    starts.append(np.asarray(old['fixed_full_pose_geometric_solution']['q_rad']))
        starts.extend(rng.uniform(lo+.03,hi-.03,size=(12,7)))
        def residual(q):
            data.qpos[:7]=q;data.qpos[env.finger_qpos_adr]=env.robot_spec.ready_finger_qpos
            mujoco.mj_forward(model,data)
            error=native.quat_error(native.rotation_to_quat(data.xmat[env.hand_id].reshape(3,3)),rig.quat)
            return np.r_[env.hand_position-target,.18*error]
        solutions=[]
        for i,start in enumerate(starts):
            fit=least_squares(residual,np.clip(start,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=200,
                ftol=1e-9,gtol=1e-9,xtol=1e-9)
            residual(fit.x);m=rig.measure(target,collision=True)
            finger=min(rig.geometry.geom_vertices(g)[:,2].min() for g in rig.geometry.finger_geoms)
            if m[0]<=.001 and m[2]<=.6 and m[3]>=.0002 and m[4]>=0 and m[5]>=.001 and .0002<=finger<=.008:
                solutions.append(dict(q_rad=fit.x.tolist(),position_error_mm=float(m[0]*1000),pose_error_deg=float(m[2]),
                    floor_mm=float(m[3]*1000),joint_margin_deg=float(m[4]),joint6_rad=float(fit.x[5]),start_index=i))
                if len(solutions)>=3:break
        result=dict(goal_m=target.tolist(),legal_fixed_pose_found=bool(solutions),solutions=solutions,
                    scope='bounded multistart geometric diagnostic; no path or physical hold validation')
        geometry.append(result);print(json.dumps(result),flush=True)
    output=dict(statistics=stats,mirrored_pair_outcomes=pairing,samples=samples,geometric_probes=geometry,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scan_unchanged=True)
    (DEST/'diagnostic.json').write_text(json.dumps(output,indent=2),encoding='utf-8')

if __name__=='__main__':main()
