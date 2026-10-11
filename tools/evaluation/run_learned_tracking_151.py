"""Run both frozen learned policies on 151; do not alter its primary checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from learned_tracking_common import ROOT,SCENES,VARIANTS,sha,dump,summarize_pairs

PPO=Path('/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/train/ppo_nero_state_heightband6_3m_20260928_010749/final_model.zip')
DVLA=Path('/data1/hxai/mujoco/DynamicVLA/runs/opt2910_single_continue_20260926_2308/checkpoints/opt2910_single_continue/snapshots/final_epoch01')
OFFICIAL=Path('/data1/hxai/mujoco/DynamicVLA/scripts/inference.py')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('pilot','main'),required=True)
    parser.add_argument('--trials',type=int,required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--dvla-only',action='store_true',help='PPO is paired locally; do not duplicate those runs')
    args=parser.parse_args()
    bundle=json.loads((ROOT/'learned_bundle_manifest.json').read_text())
    for p,h in bundle['sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(PPO)=='4ba154c80334cd35efef32e44661326af327c8752aa5ed0f0722b162020e4d61'
    assert sha(DVLA/'model.safetensors')=='c56ef12d45599e1d69ac2a7d08f914dd1a7d2962ea6fe9075fa59c30fd8ae87b'
    output=ROOT/'results'/args.stage;output.mkdir(parents=True,exist_ok=False)
    official_copy=ROOT/'official_inference_frozen.py'
    if not official_copy.exists():official_copy.write_bytes(OFFICIAL.read_bytes())
    identity={'source_bundle_sha256':sha(ROOT/'learned_bundle_manifest.json'),'ppo_checkpoint_sha256':sha(PPO),
        'dvla_checkpoint_sha256':sha(DVLA/'model.safetensors'),'dvla_config_sha256':sha(DVLA/'config.json'),
        'official_inference_source_sha256':sha(official_copy),'paired_seeds':list(range(args.seed,args.seed+args.trials)),
        'scenarios':list(SCENES),'variants':list(VARIANTS),'stage':args.stage,
        'episodes_per_method':args.trials*len(SCENES)*2,'workspace_restriction_added':False,
        'angular_feedback_x4_applied':False,'checkpoint_retraining':False,'no_video':True}
    dump(output/'manifest.json',identity)
    common_env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1','MUJOCO_GL':'egl','CUBLAS_WORKSPACE_CONFIG':':4096:8',
        'HF_HOME':'/data1/hxai/hf_cache','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1'}
    ppo_log=(output/'ppo.log').open('w')
    ppo_cmd=[sys.executable,str(ROOT/'tools/evaluation/compare_state_ppo_tracking_fix.py'),'--model',str(PPO),
        '--output',str(output/'ppo'),'--seed',str(args.seed),'--trials',str(args.trials),'--workers','12']
    ppo=None if args.dvla_only else subprocess.Popen(ppo_cmd,cwd=ROOT,env=common_env,stdout=ppo_log,stderr=subprocess.STDOUT)
    waiting=[(scene,variant) for i,scene in enumerate(SCENES) for variant in (VARIANTS if i%2==0 else VARIANTS[::-1])]
    slots=list(range(6));active=[];finished=[];index=0
    try:
        while waiting or active:
            while waiting and slots:
                scene,variant=waiting.pop(0);gpu=slots.pop(0);index+=1
                case=output/'dvla'/scene/variant;case.mkdir(parents=True)
                img_port=18400+index*10;act_port=img_port+2
                server_cmd=[sys.executable,str(ROOT/'tools/evaluation/learned_dynamicvla_server.py'),
                    '--variant',variant,'--audit-root',str(case),'--robot','nero','--scenario',scene,
                    '--trials',str(args.trials),'--seed',str(args.seed),'--episode-seconds','15','--headless',
                    '--wrist-camera','old','--sync','--execute-steps','8','--client-timeout','600','--ack-timeout','120',
                    '--img-port',str(img_port),'--act-port',str(act_port),'--video-dir',str(case),'--run-name','run']
                client_cmd=[sys.executable,str(ROOT/'tools/evaluation/learned_dynamicvla_client.py'),
                    '--inference-source',str(official_copy),'--weights',str(DVLA),'--seed',str(args.seed),
                    '--img-port',str(img_port),'--act-port',str(act_port),'--output',str(case/'client')]
                server_log=(case/'server.log').open('w');client_log=(case/'client.log').open('w')
                process_env={**common_env,'CUDA_VISIBLE_DEVICES':str(gpu),'MUJOCO_EGL_DEVICE_ID':'0'}
                server=subprocess.Popen(server_cmd,cwd=ROOT,env=process_env,stdout=server_log,stderr=subprocess.STDOUT)
                client=subprocess.Popen(client_cmd,cwd=ROOT,env=process_env,stdout=client_log,stderr=subprocess.STDOUT)
                active.append(dict(scene=scene,variant=variant,gpu=gpu,server=server,client=client,logs=[server_log,client_log],case=case))
                dump(case/'launch.json',{'server':server_cmd,'client':client_cmd,'gpu':gpu,'server_pid':server.pid,'client_pid':client.pid})
            for task in active[:]:
                a,b=task['server'].poll(),task['client'].poll()
                if a not in (None,0) or b not in (None,0):raise RuntimeError(f"DVLA process failed: {task['case']} server={a} client={b}")
                if a==0 and b==0:
                    for log in task['logs']:log.close()
                    lines=(task['case']/'episodes_audit.jsonl').read_text().splitlines()
                    assert len(lines)==args.trials
                    seeds=[int(l) for l in (task['case']/'client/episode_rng_seeds.jsonl').read_text().splitlines()]
                    assert seeds==identity['paired_seeds'],seeds
                    active.remove(task);slots.append(task['gpu']);finished.append(str(task['case']))
            if ppo is not None and ppo.poll() not in (None,0):raise RuntimeError('PPO process failed; see ppo.log')
            progress={'finished_dvla_jobs':len(finished),'total_dvla_jobs':8,
                'active_dvla':[{k:t[k] for k in ('scene','variant','gpu')} for t in active],
                'ppo_running':ppo is not None and ppo.poll() is None,'ppo_paired_locally':args.dvla_only}
            for task in active:
                path=task['case']/'progress.json'
                if path.exists():progress.setdefault('dvla_completed_episodes',{})[f"{task['scene']}:{task['variant']}"]=json.loads(path.read_text())
            dump(output/'progress.json',progress)
            time.sleep(5)
        if ppo is not None:assert ppo.wait()==0
        dvla_rows=[]
        for scene in SCENES:
            for variant in VARIANTS:
                dvla_rows.extend(json.loads(l) for l in (output/'dvla'/scene/variant/'episodes_audit.jsonl').read_text().splitlines())
        dump(output/'dvla/episodes.json',dvla_rows)
        dump(output/'dvla/summary.json',summarize_pairs(dvla_rows))
        summary={'dynamicvla':summarize_pairs(dvla_rows),'identity':identity}
        if ppo is not None:summary['state_ppo']=json.loads((output/'ppo/summary.json').read_text())
        dump(output/'summary.json',summary);(output/'COMPLETE').write_text('All paired audits passed.\n')
        print(json.dumps({k:v['overall'] for k,v in summary.items() if k!='identity'},indent=2),flush=True)
    finally:
        # Only children created by this isolated run are touched.
        for task in active:
            for child in (task['server'],task['client']):
                if child.poll() is None:child.terminate()
        if ppo is not None and ppo.poll() is None:ppo.terminate()
        ppo_log.close()

if __name__=='__main__':main()
