"""Supplement the frozen Jacobian experiment with 300 actual Pose IK episodes.

Only the new Pose + gravity feedforward + kp x4 / kv x2 group is simulated.
Existing Jacobian groups are reused only after source, scene, seed, physics,
policy-config and actuator-config checks. No Jacobian feedback x4 is silently
converted into a Pose orientation weight. Pose weight remains 1.0.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import mujoco
import numpy as np

import compare_scripted_tracking_fix as previous
from pose_scripted_entry import PoseIKScriptedPolicy
from panda_cable_grasp.dynamicvla.adapter import DynamicVLAAdapterConfig

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'outputs/benchmarks/nero_scripted_tracking_fix_20261001'
OUT=ROOT/'outputs/benchmarks/nero_pose_scripted_tracking_fix_20261001'
POSE='pose_ik_tracking_fix'
REFERENCES=('baseline','tracking_fix','tracking_fix_orientation4')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(job):
    previous.OUT=OUT
    previous.DiagnosticPolicy=PoseIKScriptedPolicy
    row=previous.worker(job)
    policy=previous._POLICY
    assert isinstance(policy,PoseIKScriptedPolicy) and policy.pose_ik_calls>0
    row.update(ik_solver='DynamicVLATaskSpaceAdapter._pose_ik_target',
        pose_ik_calls=policy.pose_ik_calls,
        pose_adapter_config=asdict(policy.pose_adapter.config),
        angular_feedback_x4_applied=False,
        command_semantics='absolute stage position and original desired quaternion; common environment motion limits')
    return row


def summarize(folder,rows,seeds,scenarios,stage):
    old=json.loads((OLD/stage/'episodes.json').read_text(encoding='utf-8'))
    byold={(r['scenario_name'],r['requested_seed'],r['variant']):r for r in old}
    bynew={(r['scenario_name'],r['requested_seed']):r for r in rows}
    assert len(rows)==len(bynew)==len(seeds)*len(scenarios)
    for sc in scenarios:
        for seed in seeds:
            pose=bynew[sc.name,seed]
            jac=byold[sc.name,seed,'tracking_fix']
            for key in ('scene_fingerprint','base_scene_fingerprint','scenario_id',
                'motion_profile_hash','physical_model_hash','actuator_hash',
                'env_config','scripted_config','actual_episode_seed'):
                if key in ('env_config','scripted_config'):
                    # Fresh dataclasses retain tuples; stored JSON has lists.
                    assert json.dumps(pose[key],sort_keys=True)==json.dumps(jac[key],sort_keys=True),(sc.name,seed,key)
                else:
                    assert pose[key]==jac[key],(sc.name,seed,key)
            assert json.dumps(pose['pose_adapter_config'],sort_keys=True)==json.dumps(asdict(DynamicVLAAdapterConfig()),sort_keys=True)
            assert pose['pose_ik_calls']>0 and not pose['angular_feedback_x4_applied']
            if pose['task_success']:
                assert pose['success_snapshot']['success_hold']>=.8-1e-12
    summary={}
    for name in [sc.name for sc in scenarios]+['overall']:
        keys=[(sc.name,s) for sc in scenarios if name in (sc.name,'overall') for s in seeds]
        poses=[bynew[k] for k in keys]
        groups={v:[byold[n,s,v] for n,s in keys] for v in REFERENCES}
        comparisons={v:previous.paired_stats([r['task_success'] for r in g],
            [r['task_success'] for r in poses]) for v,g in groups.items()}
        if name=='overall':
            for v,c in comparisons.items():
                block=np.asarray([np.mean([int(bynew[sc.name,s]['task_success'])-
                    int(byold[sc.name,s,v]['task_success']) for sc in scenarios]) for s in seeds])
                rng=np.random.default_rng(1012026)
                boot=block[rng.integers(0,len(seeds),size=(20000,len(seeds)))].mean(axis=1)
                c['paired_bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
                c['bootstrap_unit']='paired seed block including all six scenes'
                c['mcnemar_exact_p']=None
                if len(seeds)<2:
                    c['paired_bootstrap95']=None
                    c['uncertainty_note']='one pilot seed block; insufficient for pooled uncertainty estimation'
        summary[name]={'variants':{v:previous.bench._aggregate(g) for v,g in groups.items()},
            'pose_vs_reference':comparisons}
        summary[name]['variants'][POSE]=previous.bench._aggregate(poses)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    check={'new_pose_rows':len(rows),'reused_jacobian_rows':len(old),
        'paired_source_scene_seed_physics_and_execution_checks_passed':True,
        'same_original_success_criterion':True,'all_new_rows_have_actual_pose_solver_calls':True,
        'pose_orientation_weight':1.,'jacobian_angular_feedback_x4_not_applied_to_pose':True}
    (folder/'verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps(summary['overall'],indent=2),flush=True)


def run(stage,workers):
    scenarios=previous.list_suite_scenarios('core')
    seeds=[2026100000] if stage=='pilot' else list(range(2026100100,2026100150))
    folder=OUT/stage; folder.mkdir(parents=True,exist_ok=True)
    previous.OUT=OUT
    old_manifest=json.loads((OLD/'main/manifest.json').read_text(encoding='utf-8'))
    for p,h in old_manifest['source_sha256'].items():
        assert sha(ROOT/p)==h,p
    source_paths=list(old_manifest['source_sha256'])+[
        'src/panda_cable_grasp/dynamicvla/adapter.py','src/panda_cable_grasp/env/kinematics.py',
        'tools/evaluation/compare_scripted_tracking_fix.py',
        'tools/evaluation/pose_scripted_entry.py','tools/evaluation/compare_scripted_pose_fix.py',
        'tools/recording/render_vertical_workspace.py']
    hashes={p:sha(ROOT/p) for p in source_paths}
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,
        'robot':'nero','seeds':seeds,'scenarios':[asdict(sc) for sc in scenarios],
        'new_variant':POSE,'new_episodes':len(seeds)*len(scenarios),
        'solver':'project DynamicVLATaskSpaceAdapter._pose_ik_target',
        'pose_adapter_config':asdict(DynamicVLAAdapterConfig()),
        'execution':'static arm gravity feedforward at 500Hz; kp x4 kv x2; original total torque limits',
        'pose_orientation_weight':1.,'jacobian_angular_feedback_x4_applied':False,
        'command_semantics':'absolute stage position and original desired quaternion; unlike old velocity-level IK',
        'reused_reference_manifest_sha256':sha(OLD/stage/'manifest.json'),
        'reused_reference_episodes_sha256':sha(OLD/stage/'episodes.json'),
        'source_sha256':hashes,'workers':workers,'mujoco':mujoco.__version__,
        'success':'original task_success, confirmed physical grasp and 0.80s continuous hold; max15s',
        'scope':'engineering supplement; not a replacement for frozen paper results; no claim of improvement over untested stock Pose IK'}
    file=folder/'manifest.json'
    if file.exists():
        saved=json.loads(file.read_text(encoding='utf-8'))
        for key in ('seeds','source_sha256','pose_adapter_config','reused_reference_episodes_sha256'):
            assert saved[key]==manifest[key],key
    else:
        file.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        (folder/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    streamfile=folder/'episodes.jsonl'
    rows=[json.loads(line) for line in streamfile.read_text(encoding='utf-8').splitlines()] if streamfile.exists() else []
    done={(r['scenario_name'],r['requested_seed']) for r in rows}
    jobs=[]
    for job in previous.jobs(stage,seeds,scenarios):
        if job['variant']=='baseline' and (job['scenario'].name,job['seed']) not in done:
            job.update(variant=POSE,output_dir=str(OUT)); jobs.append(job)
    total=len(seeds)*len(scenarios); start=time.monotonic()
    with streamfile.open('a',encoding='utf-8') as stream,mp.get_context('spawn').Pool(workers) as pool:
        for row in pool.imap_unordered(worker,jobs,chunksize=1):
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            (folder/'progress.json').write_text(json.dumps({'stage':stage,'completed':len(rows),
                'total':total,'task_successes':sum(r['task_success'] for r in rows),
                'elapsed_wall_s':time.monotonic()-start},indent=2),encoding='utf-8')
            print(f"{stage} {len(rows)}/{total} {row['scenario_name']} seed={row['requested_seed']}: "
                f"success={row['task_success']} calls={row['pose_ik_calls']} wall={row['wall_seconds']:.1f}s",flush=True)
    for p,h in hashes.items():
        assert sha(ROOT/p)==h,p
    (folder/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    previous.bench._write_csv(folder/'episodes.csv',rows)
    summarize(folder,rows,seeds,scenarios,stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('pilot','main'),required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    run(args.stage,args.workers)
