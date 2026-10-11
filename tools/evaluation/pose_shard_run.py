"""Run explicit Pose episode shards or cross-host pilot replication."""
import argparse
import hashlib
import json
import multiprocessing as mp
import platform
from pathlib import Path
import time

import numpy as np
import compare_scripted_pose_fix as experiment

ROOT=Path(__file__).resolve().parents[2]
OUT=experiment.OUT
ORIGINAL_POLICY=experiment.previous.DiagnosticPolicy


def worker(item):
    sc=experiment.previous.get_scenario(item['scenario'])
    job=next(j for j in experiment.previous.jobs(item['stage'],[item['seed']],[sc]) if j['variant']=='baseline')
    job['variant']=item['variant']
    if item['variant']==experiment.POSE:
        row=experiment.worker(job)
    else:
        experiment.previous.OUT=OUT
        experiment.previous.DiagnosticPolicy=ORIGINAL_POLICY
        row=experiment.previous.worker(job)
    row.update(execution_host=platform.node(),execution_platform=platform.platform(),
        shard_stage=item['stage'])
    return row


def check_bundle():
    path=ROOT/'pose_bundle_manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        for p,h in manifest['sha256'].items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    main=json.loads((OUT/'main/manifest.json').read_text())
    for p,h in main['source_sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p


def run(items,stage,workers):
    folder=OUT/stage;folder.mkdir(parents=True,exist_ok=True)
    file=folder/'episodes.jsonl'
    rows=[json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
    done={(r['scenario_name'],r['requested_seed'],r['variant']) for r in rows}
    jobs=[dict(item,stage=stage) for item in items
        if (item['scenario'],item['seed'],item['variant']) not in done]
    started=time.monotonic()
    with file.open('a') as stream,mp.get_context('spawn').Pool(workers) as pool:
        for row in pool.imap_unordered(worker,jobs,chunksize=1):
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            (folder/'progress.json').write_text(json.dumps({'stage':stage,'completed':len(rows),
                'total':len(items),'successes':sum(r['task_success'] for r in rows),
                'elapsed_wall_s':time.monotonic()-started,'hostname':platform.node()},indent=2))
            print(f"{stage} {len(rows)}/{len(items)} {row['scenario_name']} {row['requested_seed']} {row['variant']} success={row['task_success']}",flush=True)
    (folder/'episodes.json').write_text(json.dumps(rows,indent=2))
    return rows


def validate(rows):
    old=json.loads((experiment.OLD/'pilot/episodes.json').read_text())
    pose=json.loads((OUT/'pilot/episodes.json').read_text())
    refs={(r['scenario_name'],r['requested_seed'],r['variant']):r for r in old+pose}
    mismatch=[]
    for r in rows:
        ref=refs[r['scenario_name'],r['requested_seed'],r['variant']]
        for key in ('scene_fingerprint','base_scene_fingerprint','scenario_id','motion_profile_hash',
            'physical_model_hash','actuator_hash','task_success','task_failure_type','steps',
            'ever_confirmed_grasp','physical_slip_after_confirmed_count','policy_result'):
            if r[key]!=ref[key]:
                mismatch.append({'scene':r['scenario_name'],'variant':r['variant'],'field':key,
                    'reference':ref[key],'server':r[key]})
        for key,tol in [('sim_time',1e-8),('success_hold',1e-8),('max_z',1e-6),('lifted_fraction',1e-8)]:
            if abs(r[key]-ref[key])>tol:
                mismatch.append({'scene':r['scenario_name'],'variant':r['variant'],'field':key,
                    'reference':ref[key],'server':r[key]})
        delta=float(np.max(np.abs(np.asarray(json.loads(r['max_abs_actual_arm_velocity']))-
            np.asarray(json.loads(ref['max_abs_actual_arm_velocity'])))))
        if delta>1e-5:
            mismatch.append({'scene':r['scenario_name'],'variant':r['variant'],
                'field':'actual_joint_velocity','max_delta':delta})
    result={'hostname':platform.node(),'replicated_episodes':len(rows),
        'all_cross_host_checks_passed':not mismatch,'mismatches':mismatch}
    (OUT/'validation/verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--jobs',type=Path)
    parser.add_argument('--stage',default='shards/server')
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args()
    check_bundle()
    if args.validate:
        jobs=[{'scenario':sc.name,'seed':2026100000,'variant':v}
            for sc in experiment.previous.list_suite_scenarios('core')
            for v in ('tracking_fix',experiment.POSE)]
        validate(run(jobs,'validation',min(args.workers,12)))
    else:
        assert args.jobs is not None
        jobs=json.loads(args.jobs.read_text())
        run(jobs,args.stage,args.workers)
