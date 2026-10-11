"""Paired evaluation of a frozen 99-dimensional state PPO with actuator fixes."""
import argparse
from functools import partial
import json
import multiprocessing as mp
from pathlib import Path
import platform
import time
import numpy as np
from learned_tracking_common import LearnedTrackingEnv,SCENES,VARIANTS,sha,dump,summarize_pairs

MODEL=None

def worker(job):
    global MODEL
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from panda_cable_grasp.rl import environment as rl
    from panda_cable_grasp.evaluation.benchmark import base_row
    from panda_cable_grasp.scenarios.registry import get_scenario
    model_path,output,scene,seed=job
    if MODEL is None:MODEL=PPO.load(model_path,device='cpu')
    assert MODEL.observation_space.shape==(99,) and MODEL.action_space.shape==(5,)
    results=[]
    for variant in (VARIANTS if seed%2==0 else VARIANTS[::-1]):
        started=time.monotonic()
        rl.CableGraspEnv=partial(LearnedTrackingEnv,variant=variant,output=output)
        env=rl.make_rl_env(action_mode='task_space',robot='nero',seed=seed,disturbance_strength=1.5,
            episode_seconds=15.,frame_skip=10,scenario_names=(scene,),
            rl_config=rl.RLConfig(singularity_avoidance_enabled=False),geometric_safety_enabled=False)
        env.set_motion_difficulty(1.)
        observation,initial=env.reset(seed=seed)
        trace=[];terminated=truncated=False;reward_sum=0.;steps=0;info={}
        while not (terminated or truncated):
            action,_=MODEL.predict(observation,deterministic=True)
            observation,reward,terminated,truncated,info=env.step(action)
            reward_sum+=float(reward);steps+=1
            if steps%5==0:trace.append(np.r_[env.data.time,env.data.qpos,env.data.ctrl])
        base=env.base_env
        row=base_row('state_ppo',1,seed,get_scenario(scene),initial,info,base.grasp_break_history)
        row.update(base.audit());row.update(steps=steps,sim_time=float(env.data.time),episode_return=reward_sum,
            wall_seconds=time.monotonic()-started,policy_internal_success=bool(info.get('strict_success',False)),
            execution_host=platform.node(),checkpoint_sha256=sha(model_path),action_mode='task_space',
            observation_dimension=99,action_dimension=5)
        trace_path=Path(output)/'traces'/f'{scene}_{seed}_{variant}.npz';trace_path.parent.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(trace_path,states=np.asarray(trace),nq=env.model.nq,sample_hz=10.)
        row['diagnostic_trace']=str(trace_path.relative_to(output));results.append(row)
        env.close()
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026101000);parser.add_argument('--trials',type=int,default=50)
    parser.add_argument('--workers',type=int,default=12);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    jobs=[(str(args.model.resolve()),str(args.output.resolve()),scene,seed) for scene in SCENES for seed in range(args.seed,args.seed+args.trials)]
    manifest={'method':'state_ppo','checkpoint':str(args.model.resolve()),'checkpoint_sha256':sha(args.model),
        'seeds':list(range(args.seed,args.seed+args.trials)),'scenarios':list(SCENES),'variants':list(VARIANTS),
        'planned_episodes':len(jobs)*2,'observation_dimension':99,'deterministic_policy':True,
        'action_mode':'task_space','frame_skip':10,'episode_seconds':15.,'singularity_avoidance_enabled':False,
        'workspace_restriction_added':False,'angular_feedback_x4_applied':False}
    if (args.output/'manifest.json').exists():assert json.loads((args.output/'manifest.json').read_text())==manifest
    dump(args.output/'manifest.json',manifest)
    file=args.output/'episodes.jsonl'
    rows=[json.loads(l) for l in file.read_text().splitlines()] if file.exists() else []
    done={(r['scenario_name'],r['requested_seed']) for r in rows}
    pending=[j for j in jobs if (j[2],j[3]) not in done]
    started=time.monotonic()
    with file.open('a',encoding='utf-8') as f,mp.get_context('spawn').Pool(args.workers) as pool:
        for pair in pool.imap_unordered(worker,pending,chunksize=1):
            summarize_pairs(pair)
            for row in pair:rows.append(row);f.write(json.dumps(row)+'\n')
            f.flush();dump(args.output/'progress.json',{'completed_episodes':len(rows),'total_episodes':len(jobs)*2,
                'successes':{v:sum(r['task_success'] for r in rows if r['variant']==v) for v in VARIANTS},
                'elapsed_s':time.monotonic()-started})
            print(f"{len(rows)}/{len(jobs)*2} {pair[0]['scenario_name']} seed={pair[0]['requested_seed']} "+str({r['variant']:r['task_success'] for r in pair}),flush=True)
    assert len(rows)==len(jobs)*2
    dump(args.output/'episodes.json',rows);dump(args.output/'summary.json',summarize_pairs(rows))
    print(json.dumps(summarize_pairs(rows)['overall'],indent=2),flush=True)

if __name__=='__main__':main()
