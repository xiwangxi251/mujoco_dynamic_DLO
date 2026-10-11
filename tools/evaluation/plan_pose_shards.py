"""Freeze remaining episode allocation after the original local pool is stopped."""
import hashlib
import json
from pathlib import Path
import platform
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/benchmarks/nero_pose_scripted_tracking_fix_20261001'
file=OUT/'main/episodes.jsonl'
raw=file.read_bytes()
(OUT/'main/pre_handoff_episodes.jsonl').write_bytes(raw)
lines=raw.decode('utf-8').splitlines();rows=[];partial=False
for i,line in enumerate(lines):
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:
        assert i==len(lines)-1
        partial=True
assert len({(r['scenario_name'],r['requested_seed']) for r in rows})==len(rows)
(OUT/'main/pre_handoff_episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
manifest=json.loads((OUT/'main/manifest.json').read_text(encoding='utf-8'))
done={(r['scenario_name'],r['requested_seed']) for r in rows}
local=[];server=[];rng=np.random.default_rng(20261001)
for sc in manifest['scenarios']:
    pending=[s for s in manifest['seeds'] if (sc['name'],s) not in done]
    rng.shuffle(pending)
    n_server=int(np.ceil(len(pending)*2/3))
    for i,seed in enumerate(pending):
        item={'scenario':sc['name'],'seed':seed,'variant':'pose_ik_tracking_fix'}
        if i<n_server:
            server.extend([dict(item,variant=v) for v in ('pose_ik_tracking_fix','tracking_fix','tracking_fix_orientation4')])
        else:local.append(item)
assert len(rows)+len(local)+len(server)//3==300
for name,jobs in [('local',local),('server',server)]:
    path=OUT/'shards'/name;path.mkdir(parents=True,exist_ok=True)
    (path/'jobs.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
plan={'local_hostname':platform.node(),'already_completed_local_pose':len(rows),
    'remaining_local_pose':len(local),'remaining_server_pose':len(server)//3,
    'server_matching_reference_episodes':2*len(server)//3,
    'allocation_seed':20261001,'allocation':'stratified by scene; random two thirds of pending cases on server',
    'partial_final_record_excluded_and_case_rescheduled':partial,
    'partial_raw_sha256':hashlib.sha256(raw).hexdigest(),
    'local_jobs_sha256':hashlib.sha256((OUT/'shards/local/jobs.json').read_bytes()).hexdigest(),
    'server_jobs_sha256':hashlib.sha256((OUT/'shards/server/jobs.json').read_bytes()).hexdigest(),
    'shard_runner_sha256':hashlib.sha256((ROOT/'tools/evaluation/pose_shard_run.py').read_bytes()).hexdigest(),
    'same_host_pairing_required':['tracking_fix','tracking_fix_orientation4'],
    'reason_for_new_server_references':'cross-host pilot scores match but physical trajectories differ'}
(OUT/'execution_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps(plan,indent=2))
