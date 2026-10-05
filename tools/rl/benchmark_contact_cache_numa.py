"""Isolated cache equivalence and 24-worker affinity timing; no policy updates."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['MUJOCO_GL']='egl'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from panda_cable_grasp.rl.train import parse_args, make_configured_env, make_worker
from panda_cable_grasp.rl.affinity import prepare_worker_affinity


class TimedEnvironment(gym.Wrapper):
    def step(self, action):
        start=time.perf_counter()
        obs,reward,done,truncated,info=self.env.step(action)
        info['_perf_step_seconds']=time.perf_counter()-start
        return obs,reward,done,truncated,info


def timed_factory(rank,args):
    factory=make_worker(rank,args)
    return lambda:TimedEnvironment(factory())


def serial_gate(args,policy,output,steps):
    reference=[];actions=[];times={};counts={};summaries={}
    for enabled in (False,True):
        local=copy.copy(args);local.contact_read_cache=enabled
        env=make_configured_env(local,seed=local.seed,scenario_names=local.training_scenario_names)
        obs,info=env.reset(seed=20260804);episode=0;start=time.perf_counter()
        base=env.base_env;compute=base._compute_pad_contact_samples;read_count=0
        def counted():
            nonlocal read_count
            read_count+=1
            return compute()
        base._compute_pad_contact_samples=counted
        ever_pinched=ever_secured=False
        try:
            for index in range(steps):
                if not enabled:
                    action,_=policy.predict(obs,deterministic=True);actions.append(action.copy())
                else:action=actions[index]
                obs,reward,done,truncated,info=env.step(action)
                ever_pinched |= bool(info.get('ever_pinched'))
                ever_secured |= bool(info.get('ever_grasped'))
                vector=np.concatenate((obs,[reward,done,truncated],base.data.qpos,base.data.qvel,
                    base.data.ctrl,base.data.qfrc_constraint,base.data.qfrc_actuator))
                scoring={k:v for k,v in info.items() if k.startswith('reward_') or k in (
                    'task_success','secured_grasp','success','strict_success','goal_node_index',
                    'goal_node_error','first_grasp_correct','grasp_confirmed','termination_reason',
                    'active_open_after_secured_count','physical_slip_after_secured_count')}
                score=json.dumps(scoring,sort_keys=True,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))
                if not enabled:reference.append((vector.copy(),score))
                else:
                    expected,expected_score=reference[index]
                    if not np.array_equal(expected,vector) or score!=expected_score:
                        raise AssertionError(f'Cache changed physical state/observation/scoring at step {index}')
                if done or truncated:
                    episode+=1;obs,info=env.reset(seed=20260804+episode)
            times[str(enabled)]=time.perf_counter()-start
            counts[str(enabled)]=read_count
            summaries[str(enabled)]=dict(completed_episodes=episode,ever_pinched=ever_pinched,ever_secured=ever_secured)
        finally:env.close()
        print('serial_variant',enabled,times[str(enabled)],read_count,flush=True)
    digest=hashlib.sha256(b''.join(v.tobytes()+s.encode() for v,s in reference)).hexdigest()
    result=dict(steps=steps,physical_and_observation_max_difference=0,reward_and_grasp_identical=True,
                resets_included=True,trace_sha256=digest,seconds=times,raw_contact_reads=counts,episodes=summaries)
    np.savez_compressed(output/'equivalence_trace.npz',vectors=np.asarray([v for v,_ in reference]),actions=np.asarray(actions))
    return result


def parallel_trial(args,policy,enabled,affinity,steps,allowed):
    os.sched_setaffinity(0,allowed)
    local=copy.copy(args);local.contact_read_cache=enabled;local.worker_affinity=affinity
    local.worker_affinity_plan=prepare_worker_affinity(local)
    initialized=time.perf_counter()
    vec=SubprocVecEnv([timed_factory(rank,local) for rank in range(local.workers)],start_method='spawn')
    try:
        vec.seed(local.seed);obs=vec.reset()
        for _ in range(16):
            action,_=policy.predict(obs,deterministic=True);obs,_,_,_=vec.step(action)
        initialization=time.perf_counter()-initialized
        digest=hashlib.sha256();worker_mean=[];worker_max=[];ipc=[];inference=0.
        start=time.perf_counter()
        for _ in range(steps):
            t=time.perf_counter();action,_=policy.predict(obs,deterministic=True);inference+=time.perf_counter()-t
            t=time.perf_counter();obs,rewards,dones,infos=vec.step(action);duration=time.perf_counter()-t
            worker=[info['_perf_step_seconds'] for info in infos]
            worker_mean.append(float(np.mean(worker)));worker_max.append(max(worker))
            ipc.append(max(0.,duration-max(worker)))
            digest.update(obs.tobytes());digest.update(rewards.tobytes());digest.update(dones.tobytes())
        duration=time.perf_counter()-start
        return dict(cache=enabled,affinity=affinity,workers=local.workers,vector_steps=steps,
            transitions=steps*local.workers,seconds=duration,transitions_per_second=steps*local.workers/duration,
            initialization_seconds=initialization,inference_seconds=inference,
            mean_worker_step_seconds=float(np.mean(worker_mean)),
            mean_slowest_worker_seconds=float(np.mean(worker_max)),
            mean_parent_wait_minus_slowest_seconds=float(np.mean(ipc)),
            trace_sha256=digest.hexdigest(),affinity_plan=local.worker_affinity_plan)
    finally:
        vec.close();os.sched_setaffinity(0,allowed)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--equivalence-steps',type=int,default=750)
    parser.add_argument('--timing-steps',type=int,default=128)
    opts=parser.parse_args();opts.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((opts.source_run/'launch_manifest.json').read_text())
    with patch('sys.argv',['bench',*manifest['command'][4:]]):args=parse_args()
    checkpoints=sorted((opts.source_run/'checkpoints').glob('*.zip'),key=lambda p:p.stat().st_mtime)
    checkpoint=checkpoints[-1] if checkpoints else Path(manifest['source_checkpoint'])
    torch.set_num_threads(1);policy=PPO.load(checkpoint,device='cpu')
    result=dict(source_run=str(opts.source_run),checkpoint=str(checkpoint),workers=args.workers,
                solver_or_physics_changes=False,learner_updates=False)
    result['equivalence']=serial_gate(args,policy,opts.output,opts.equivalence_steps)
    (opts.output/'equivalence.json').write_text(json.dumps(result,indent=2))
    print('EQUIVALENCE_PASS',json.dumps(result['equivalence']),flush=True)
    allowed=set(os.sched_getaffinity(0));trials=[]
    for enabled,affinity in [(False,'none'),(True,'none'),(True,'numa')]:
        row=parallel_trial(args,policy,enabled,affinity,opts.timing_steps,allowed);trials.append(row)
        result['parallel_trials']=trials
        (opts.output/'summary.json').write_text(json.dumps(result,indent=2))
        print('PARALLEL_RESULT',json.dumps(row),flush=True)
    assert len({row['trace_sha256'] for row in trials})==1,'Cache or affinity changed parallel observations/rewards'
    result['parallel_trace_identical']=True
    (opts.output/'summary.json').write_text(json.dumps(result,indent=2))
    print('BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':main()
