"""Check isolation and exact frozen benchmark controller provenance."""
import hashlib
import json
import numpy as np
import mujoco
import record_improved_workspace as w

rig=w.ImprovedRig()
manifest=json.loads((w.ROOT/'outputs/benchmarks/nero_scripted_tracking_fix_20261001/main/manifest.json').read_text())
for p,sha in manifest['source_sha256'].items():
    assert hashlib.sha256((w.ROOT/p).read_bytes()).hexdigest()==sha,p
full=w.env_factory(w.old.EnvConfig(robot='nero',motion_mode='static',disturbance_strength=0.,
    frame_skip=10,episode_seconds=10000.,target_selection='middle'))
full.reset(randomize=False,seed=20261001)
full.data.qpos[:9]=rig.data.qpos[:9];full.data.qvel[:]=0
full.data.ctrl[:]=rig.data.ctrl
full._last_applied_action[:]=rig.base._last_applied_action
full._last_requested_action[:]=rig.base._last_requested_action
full._previous_arm_command_velocity[:]=0
mujoco.mj_forward(full.model,full.data)
policy=full.workspace_policy;policy.desired_quat=rig.quat.copy();policy.phase=w.Phase.INTERCEPT
goal=np.array([.7,0,.20]);maxq=0.;maxp=0.;maxaction=0.
for _ in range(50):
    request=policy._ik_action(goal,full.gripper_open_ctrl)
    rig_request=rig.policy._ik_action(goal,rig.base.gripper_open_ctrl)
    maxaction=max(maxaction,float(np.max(np.abs(request-rig_request))))
    applied=full._limit_robot_action(np.clip(request,full.model.actuator_ctrlrange[:,0],full.model.actuator_ctrlrange[:,1]))
    for _ in range(10):
        guarded,_=full._velocity_guarded_action(applied);full.data.ctrl[:]=guarded;full._physics_step()
    mujoco.mj_forward(full.model,full.data)
    rig.control(goal)
    maxq=max(maxq,float(np.max(np.abs(full.data.qpos[:9]-rig.data.qpos[:9]))))
    maxp=max(maxp,float(np.linalg.norm(full.hand_position-rig.base.hand_position)))
assert maxq<1e-5 and maxp<1e-5 and maxaction<1e-5,(maxq,maxp,maxaction)
result={'frozen_grasp_benchmark_control_source_hashes_match':True,
    'complete_vs_isolated_physics_substeps':500,'max_joint_difference_rad':maxq,
    'max_tcp_difference_m':maxp,'max_requested_action_difference':maxaction,
    'full_scene_gravity_feedforward_and_improved_ik_pairing_passed':True,
    'verification_target_m':goal.tolist(),'workspace_control_phase':'INTERCEPT',
    'source_sha256':hashlib.sha256(__file__.encode()).hexdigest()}
result['source_sha256']=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
(w.OUT/'full_scene_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
