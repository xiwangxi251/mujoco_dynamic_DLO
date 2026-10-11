"""Check raw PPO decoder equivalence and the shared actuator-only intervention."""
import argparse
from functools import partial
from pathlib import Path
import mujoco
import numpy as np
from learned_tracking_common import LearnedTrackingEnv,dump
from panda_cable_grasp.rl import environment as rl

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    original=rl.CableGraspEnv
    kwargs=dict(action_mode='task_space',robot='nero',seed=2026100900,episode_seconds=15.,frame_skip=10,
        scenario_names=('id_static',),rl_config=rl.RLConfig(singularity_avoidance_enabled=False),geometric_safety_enabled=False)
    raw=rl.make_rl_env(**kwargs)
    rl.CableGraspEnv=partial(LearnedTrackingEnv,variant='baseline',output=args.output)
    baseline=rl.make_rl_env(**kwargs)
    a,ia=raw.reset(seed=2026100900);b,ib=baseline.reset(seed=2026100900)
    assert np.array_equal(a,b) and raw.observation_space.shape==(99,)
    rng=np.random.default_rng(2026100900);deltas=[]
    for _ in range(30):
        action=rng.uniform(-.3,.3,5);action[4]=1.
        ao=raw.step(action);bo=baseline.step(action)
        deltas.append(float(np.max(np.abs(raw.data.qpos-baseline.data.qpos))))
        assert np.array_equal(ao[0],bo[0]) and ao[1:4]==bo[1:4]
        assert ao[4]['task_success']==bo[4]['task_success']
    assert max(deltas)==0.
    rl.CableGraspEnv=partial(LearnedTrackingEnv,variant='tracking_fix',output=args.output)
    improved=rl.make_rl_env(**kwargs);improved.reset(seed=2026100900)
    bm,im=baseline.model,improved.model
    assert np.array_equal(bm.actuator_gainprm[7:],im.actuator_gainprm[7:])
    assert np.array_equal(bm.actuator_biasprm[7:],im.actuator_biasprm[7:])
    assert np.allclose(im.actuator_gainprm[:7,0],4*bm.actuator_gainprm[:7,0])
    assert np.allclose(im.actuator_biasprm[:7,1],4*bm.actuator_biasprm[:7,1])
    assert np.allclose(im.actuator_biasprm[:7,2],2*bm.actuator_biasprm[:7,2])
    for attr in ('body_mass','body_inertia','dof_damping','dof_frictionloss','jnt_range','geom_friction','geom_size','actuator_forcerange'):
        assert np.array_equal(getattr(bm,attr),getattr(im,attr)),attr
    base=improved.base_env;probe=mujoco.MjData(im)
    gravity_deltas=[]
    for _ in range(20):
        probe.qpos[:]=base.data.qpos;probe.qpos[:7]+=rng.uniform(-.1,.1,7);probe.qvel[:]=0
        mujoco.mj_forward(im,probe)
        base.gravity_data.qpos[:]=probe.qpos[:9];base.gravity_data.qvel[:]=0
        mujoco.mj_forward(base.gravity_model,base.gravity_data)
        gravity_deltas.append(float(np.max(np.abs(probe.qfrc_bias[:7]-base.gravity_data.qfrc_bias[:7]))))
    assert max(gravity_deltas)<1e-10
    result={'ppo_baseline_decoder_and_physics_exactly_match_original':True,'matched_control_steps':30,
        'max_qpos_difference':max(deltas),'observation_dimension':99,'finger_gains_unchanged':True,
        'physical_parameters_and_original_torque_limits_unchanged':True,
        'gain_scales':{'kp':4,'kv':2},'gravity_probe_poses':20,
        'max_full_scene_vs_robot_only_gravity_difference_nm':max(gravity_deltas)}
    dump(args.output/'verification.json',result);print(result,flush=True)
    for env in (raw,baseline,improved):env.close()

if __name__=='__main__':main()
