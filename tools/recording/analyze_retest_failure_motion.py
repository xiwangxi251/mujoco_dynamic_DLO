"""Inspect saved physical states; distinguish feedback loss, limits and contact."""
import argparse
import json
import mujoco
import numpy as np
from record_retest_136_videos import create_rig, OUT

def main():
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,required=True);a=p.parse_args()
    directory=OUT/f'version_{a.version}'/'finger_floor'
    rig, env, _=create_rig(a.version,directory/'motion_audit')
    model,data=rig.model,rig.data
    archive=np.load(directory/'states.npz');states=archive['states'];refs=archive['reference']
    metadata=json.loads((directory/'metadata.json').read_text())
    lo,hi=model.jnt_range[env.arm_joint_ids].T
    frames=[]
    for index,state in enumerate(states):
        data.time=state[0];data.qpos[:9]=state[1:10];data.qvel[:9]=state[10:19];data.ctrl[:]=state[19:27]
        mujoco.mj_forward(model,data)
        m=rig.measure(np.array(metadata['goal_m']),collision=False)
        j=rig.policy._task_jacobian();jp,jr=j[:3],j[3:]
        null=np.eye(7)-np.linalg.pinv(jp,rcond=1e-5)@jp
        sv=np.linalg.svd(jr@null,compute_uv=False)
        lam=.04+.12*np.clip((.10-sv[-1])/.10,0,1)**2
        q=data.qpos[:7];margin=np.minimum(q-lo,hi-q)
        finger=min(rig.geometry.geom_vertices(g)[:,2].min() for g in rig.geometry.finger_geoms)
        record=dict(time_s=float(data.time),pose_deg=float(m[2]),position_mm=float(m[0]*1000),
            floor_mm=float(m[3]*1000),finger_mm=float(finger*1000),
            joint_margin_deg=float(np.degrees(margin.min())),nearest_joint=int(margin.argmin()+1),
            orientation_nullspace_sigma_min=float(sv[-1]),orientation_damping=float(lam),
            minimum_orientation_mode_transmission=float(sv[-1]**2/(sv[-1]**2+lam**2)),
            contact_count=int(data.ncon),command_error_norm_rad=float(np.linalg.norm(data.ctrl[:7]-q)))
        record.update(actual_linear_m_s=(jp@data.qvel[env.arm_dof_adr]).tolist(),
                      actual_angular_rad_s=(jr@data.qvel[env.arm_dof_adr]).tolist())
        if a.version in (1,3) and index<len(states)-1:
            reference=refs[index+1]
            linear=rig.policy._limit_vector_norm(6*(reference-env.hand_position),.95)
            error,_=rig.policy._orientation_error()
            angular=rig.policy._limit_vector_norm(2.5*error,1.4)
            pv=jp.T@np.linalg.solve(jp@jp.T+.04**2*np.eye(3),linear)
            jo=jr@null
            ov=jo.T@np.linalg.solve(jo@jo.T+lam**2*np.eye(3),angular-jr@pv)
            qv=pv+ov+(np.eye(7)-np.linalg.pinv(j,rcond=1e-5)@j)@(.8*(env.ready_qpos[:7]-q))
            scale=min(1.,float(np.min(np.asarray(env.config.arm_joint_velocity_limits)/np.maximum(abs(qv),1e-12))))
            qv*=scale
            unbounded=q+.11*qv;bounded=np.clip(unbounded,lo,hi)
            policy_action=rig.policy._ik_action(reference,env.gripper_open_ctrl)
            assert np.max(abs(policy_action[:7]-bounded))<1e-10
            record.update(command_clipped_joints=(np.flatnonzero(abs(unbounded-bounded)>1e-10)+1).tolist(),
                global_velocity_scale=scale,requested_linear_m_s=linear.tolist(),
                requested_angular_rad_s=angular.tolist(),
                achieved_linear_before_joint_clipping_m_s=(jp@qv).tolist(),
                achieved_angular_before_joint_clipping_rad_s=(jr@qv).tolist(),
                achieved_linear_after_joint_clipping_m_s=(jp@((bounded-q)/.11)).tolist(),
                achieved_angular_after_joint_clipping_rad_s=(jr@((bounded-q)/.11)).tolist())
        frames.append(record)
    def first(predicate):
        return next((r for r in frames if predicate(r)),None)
    result=dict(version=a.version,goal=metadata['goal_m'],end=metadata['result'],
        events=dict(pose_30deg=first(lambda r:r['pose_deg']>30),pose_90deg=first(lambda r:r['pose_deg']>90),
            pose_150deg=first(lambda r:r['pose_deg']>150),joint_margin_below_half_deg=first(lambda r:r['joint_margin_deg']<.5),
            joint_command_clipped=first(lambda r:bool(r.get('command_clipped_joints'))),
            table_contact=first(lambda r:r['contact_count']>0),below_minus_1mm=first(lambda r:r['floor_mm'] < -1)),
        samples=[frames[min(range(len(frames)),key=lambda i:abs(frames[i]['time_s']-t))] for t in (0,.5,1,2,4,6,8,10,12,14,15.74,16)])
    (directory/'motion_diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(directory/'motion_diagnosis_trace.npz',
        time_s=np.array([r['time_s'] for r in frames]),pose_deg=np.array([r['pose_deg'] for r in frames]),
        nullspace_sigma=np.array([r['orientation_nullspace_sigma_min'] for r in frames]),
        joint_margin_deg=np.array([r['joint_margin_deg'] for r in frames]))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
