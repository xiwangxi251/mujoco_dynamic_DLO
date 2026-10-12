"""Small paired no-contact check of current-family L1 force playback."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
from panda_cable_grasp.env.environment import CableGraspEnv,EnvConfig
from panda_cable_grasp.env.control import configure_profile
from panda_cable_grasp.scenarios import get_scenario
from audit_replay_tracking import save,sha

SCENES=('dev_rigid_l1_nominal_x075_v1','dev_rigid_l1_midshape_v1_x075_v1','dev_combined_l1_nominal_x075_v1')


def measure(job):
    name,seed,bank,out=job
    config=configure_profile(EnvConfig(robot='nero',seed=seed,episode_seconds=2.,
        replay_bank_dir=bank,target_selection='middle',**get_scenario(name).to_env_overrides()),'enhanced6_bounded_dls_v1')
    rows={}
    compiler_error={}
    for version in ('feedback_v1','force_tape_v1'):
        env=CableGraspEnv(replace(config,rigid_drive_version=version))
        try:
            env.reset(seed=seed)
            probe=mujoco.MjData(env.model)
            jac=np.zeros((3,env.model.nv)); jr=np.zeros_like(jac)
            body=int(env.target_body_id)
            samples=[]
            for tick in range(1000):
                env.data.xfrc_applied[:]=0.
                env._apply_cable_disturbance()
                env.data.ctrl[:]=env.ready_ctrl
                env._physics_step()
                if (tick+1)%10: continue
                probe.qpos[:]=env.data.qpos; probe.qvel[:]=env.data.qvel
                mujoco.mj_forward(env.model,probe)
                for contact in probe.contact:
                    g1,g2=int(contact.geom1),int(contact.geom2)
                    if contact.dist<=0 and (
                        g1 in env.geometric_safety_geom_ids and int(env.model.geom_bodyid[g2]) in env.cable_set
                        or g2 in env.geometric_safety_geom_ids and int(env.model.geom_bodyid[g1]) in env.cable_set
                    ): raise AssertionError('Robot contacted cable')
                mujoco.mj_jacBody(env.model,probe,jac,jr,body)
                xyz=probe.xpos[env.cable_ids].copy()
                com=np.average(xyz,axis=0,weights=env.cable_mass)
                reference=env._rigid_reference_xy-env._rigid_reference_com_xy
                centered=xyz[:,:2]-com[:2]
                cov=(reference*np.sqrt(env.cable_mass[:,None])).T@(centered*np.sqrt(env.cable_mass[:,None]))
                left,_,right=np.linalg.svd(cov)
                rotation=left@right
                if np.linalg.det(rotation)<0:
                    left[:,-1]*=-1; rotation=left@right
                yaw=math.atan2(rotation[0,1],rotation[0,0])
                samples.append(np.r_[env.data.time,probe.xpos[body,:2],(jac@probe.qvel)[:2],com[:2],yaw])
            rows[version]=np.array(samples)
            if version=='force_tape_v1':
                tape=env._replay_force_tape
                compiler_error={
                    'qpos':float(np.abs(env.data.qpos[env.cable_free_qadr:]-tape.final_qpos[env.cable_free_qadr:]).max()),
                    'qvel':float(np.abs(env.data.qvel[env.cable_free_dadr:]-tape.final_qvel[env.cable_free_dadr:]).max()),
                    'force_sha256':tape.sha256,
                }
        finally: env.close()
    old,new=rows['feedback_v1'],rows['force_tape_v1']
    mask=new[:,0]>=.82-1e-8
    a,b=old[mask],new[mask]
    p=np.linalg.norm(a[:,1:3]-b[:,1:3],axis=1)
    v=np.linalg.norm(a[:,3:5]-b[:,3:5],axis=1)
    result=dict(scene=name,seed=seed,seconds=2.,window_start=.82,
        position_rmse_m=float(np.sqrt(np.mean(p*p))),position_max_m=float(p.max()),
        velocity_vector_rmse_m_s=float(np.sqrt(np.mean(v*v))),
        speed_relative_error=float(np.linalg.norm(b[:,3:5],axis=1).mean()/np.linalg.norm(a[:,3:5],axis=1).mean()-1),
        com_position_rmse_m=float(np.sqrt(np.mean(np.sum((a[:,5:7]-b[:,5:7])**2,axis=1)))),
        compiler_difference=compiler_error,robot_contacts=0)
    com_old=np.gradient(old[:,5:7],old[:,0],axis=0)[mask]
    com_new=np.gradient(new[:,5:7],new[:,0],axis=0)[mask]
    yaw_error=np.arctan2(np.sin(b[:,7]-a[:,7]),np.cos(b[:,7]-a[:,7]))
    result['overall_velocity_rmse_m_s']=float(np.sqrt(np.mean(np.sum((com_old-com_new)**2,axis=1))))
    result['overall_speed_relative_error']=float(np.linalg.norm(com_new,axis=1).mean()/np.linalg.norm(com_old,axis=1).mean()-1)
    result['overall_yaw_rmse_rad']=float(np.sqrt(np.mean(yaw_error*yaw_error)))
    dest=Path(out)/name/f'seed_{seed}'; dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'paired.npz',**rows)
    save(dest/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bank',required=True)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if (args.output/'summary.json').exists(): p.error('Existing result')
    jobs=[(name,seed,args.bank,str(args.output)) for name in SCENES for seed in (20260804,20260805)]
    started=time.monotonic()
    save(args.output/'manifest.json',dict(source_directory=str(ROOT),
        source_sha256={n:sha(ROOT/n) for n in ('src/panda_cable_grasp/env/environment.py',
            'src/panda_cable_grasp/env/force_replay.py','src/panda_cable_grasp/scenarios/registry.py')},
        seeds=[20260804,20260805],scenes=SCENES,seconds=2.,planned_pairs=6,
        limits={'position_rmse_m':.005,'velocity_vector_rmse_m_s':.025,'absolute_speed_bias':.05},
        protocol='NERO enhanced6, material middle; no robot contact; compared to original feedback motion; low-level physics'))
    with ProcessPoolExecutor(max_workers=2) as pool: results=list(pool.map(measure,jobs))
    accepted=all(r['com_position_rmse_m']<.005 and r['overall_velocity_rmse_m_s']<.025
        and abs(r['overall_speed_relative_error'])<.05 and r['overall_yaw_rmse_rad']<.025
        and r['compiler_difference']['qpos']<1e-7
        and r['compiler_difference']['qvel']<1e-6 for r in results)
    save(args.output/'summary.json',dict(completed_pairs=len(results),accepted=accepted,
        elapsed_wall_s=time.monotonic()-started,episodes=results))
    print(json.dumps({'accepted':accepted,'completed_pairs':len(results),'episodes':results},indent=2))


if __name__=='__main__': main()
