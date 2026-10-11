"""Merge all 300 Pose cases with platform-matched corrected Jacobian references."""
import hashlib
import json
from pathlib import Path
import numpy as np
import compare_scripted_pose_fix as e

OUT=e.OUT;OLD=e.OLD


def main():
    plan=json.loads((OUT/'execution_plan.json').read_text(encoding='utf-8'))
    partial=json.loads((OUT/'main/pre_handoff_episodes.json').read_text(encoding='utf-8'))
    local=json.loads((OUT/'shards/local/episodes.json').read_text(encoding='utf-8'))
    server=json.loads((OUT/'server_export/shards/server/episodes.json').read_text(encoding='utf-8'))
    for name,path in [('local_jobs_sha256',OUT/'shards/local/jobs.json'),
        ('server_jobs_sha256',OUT/'server_export/jobs_server.json')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==plan[name]
    assert hashlib.sha256((OUT/'main/pre_handoff_episodes.jsonl').read_bytes()).hexdigest()==plan['partial_raw_sha256']
    assert partial==[json.loads(line) for line in (OUT/'main/pre_handoff_episodes.jsonl').read_text().splitlines()]
    assert hashlib.sha256((e.ROOT/'tools/evaluation/pose_shard_run.py').read_bytes()).hexdigest()==plan['shard_runner_sha256']
    assert (OUT/'pose_bundle_manifest.json').read_bytes()==(OUT/'server_export/pose_bundle_manifest.json').read_bytes()
    runtime=json.loads((OUT/'server_export/server_runtime.json').read_text())
    assert runtime['all_bundled_hashes_unchanged'] and runtime['hostname']=='node151'
    for actual,path in [(local,OUT/'shards/local/jobs.json'),(server,OUT/'server_export/jobs_server.json')]:
        expected=json.loads(path.read_text())
        actual_keys={(r['scenario_name'],r['requested_seed'],r['variant']) for r in actual}
        expected_keys={(r['scenario'],r['seed'],r['variant']) for r in expected}
        assert len(actual_keys)==len(actual)==len(expected_keys) and actual_keys==expected_keys
    assert len(partial)==plan['already_completed_local_pose']
    assert len(local)==plan['remaining_local_pose']
    assert len(server)==3*plan['remaining_server_pose']
    for r in partial:r['execution_host']=plan['local_hostname'];r['shard_stage']='pre_handoff_local'
    for r in server:
        r['compiled_model']=str(Path('server_export')/r['compiled_model'])
        r['diagnostic_trace']=str(Path('server_export')/r['diagnostic_trace'])
    poses=partial+local+[r for r in server if r['variant']==e.POSE]
    overrides=[r for r in server if r['variant']!=e.POSE]
    assert len(overrides)==plan['server_matching_reference_episodes']
    assert len(poses)==300
    bypose={(r['scenario_name'],r['requested_seed']):r for r in poses}
    assert len(bypose)==300
    old=json.loads((OLD/'main/episodes.json').read_text(encoding='utf-8'))
    refs={(r['scenario_name'],r['requested_seed'],r['variant']):dict(r,execution_host=plan['local_hostname']) for r in old}
    for r in overrides:refs[r['scenario_name'],r['requested_seed'],r['variant']]=r
    for (name,seed),pose in bypose.items():
        assert pose['pose_ik_calls']>0 and not pose['angular_feedback_x4_applied']
        for v in ('tracking_fix','tracking_fix_orientation4'):
            ref=refs[name,seed,v]
            assert pose['execution_host']==ref['execution_host'],(name,seed,v)
            for field in ('scene_fingerprint','base_scene_fingerprint','scenario_id','motion_profile_hash',
                'physical_model_hash','actuator_hash','env_config','scripted_config','actual_episode_seed'):
                assert pose[field]==ref[field],(name,seed,field)
    scenarios=e.previous.list_suite_scenarios('core');seeds=list(range(2026100100,2026100150))
    summary={}
    for name in [sc.name for sc in scenarios]+['overall']:
        keys=[(sc.name,s) for sc in scenarios if name in (sc.name,'overall') for s in seeds]
        ps=[bypose[k] for k in keys]
        groups={v:[refs[n,s,v] for n,s in keys] for v in e.REFERENCES}
        comparisons={}
        for v in ('tracking_fix','tracking_fix_orientation4'):
            c=e.previous.paired_stats([r['task_success'] for r in groups[v]],[r['task_success'] for r in ps])
            c['platform_matched_per_case']=True
            if name=='overall':
                block=np.asarray([np.mean([int(bypose[sc.name,s]['task_success'])-
                    int(refs[sc.name,s,v]['task_success']) for sc in scenarios]) for s in seeds])
                rng=np.random.default_rng(1012026)
                boot=block[rng.integers(0,len(seeds),size=(20000,len(seeds)))].mean(axis=1)
                c['paired_bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
                c['bootstrap_unit']='paired seed block including all six scenes'
                c['mcnemar_exact_p']=None
            comparisons[v]=c
        summary[name]={'variants':{v:e.previous.bench._aggregate(g) for v,g in groups.items()},
            'pose_vs_reference':comparisons,'stock_jacobian_background_only':True}
        summary[name]['variants'][e.POSE]=e.previous.bench._aggregate(ps)
    folder=OUT/'main'
    (folder/'episodes.json').write_text(json.dumps(poses,indent=2),encoding='utf-8')
    (folder/'merged_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in poses),encoding='utf-8')
    e.previous.bench._write_csv(folder/'episodes.csv',poses)
    (folder/'matched_reference_overrides.json').write_text(json.dumps(overrides,indent=2),encoding='utf-8')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    checks={'new_pose_rows':300,'new_same_host_reference_rows':len(overrides),
        'reused_original_reference_rows':900-len(overrides),'host_matched_corrected_reference_groups':True,
        'paired_source_scene_seed_physics_and_execution_checks_passed':True,
        'same_original_success_criterion':True,'all_new_rows_have_actual_pose_solver_calls':True,
        'pose_orientation_weight':1.,'jacobian_angular_feedback_x4_not_applied_to_pose':True,
        'stock_jacobian_background_only':True,'execution_plan':plan,
        'merge_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    (folder/'progress.json').write_text(json.dumps({'completed':300,'total':300,
        'task_successes':sum(r['task_success'] for r in poses),'state':'merged and platform-paired'},indent=2),encoding='utf-8')
    print(json.dumps(summary['overall'],indent=2))


if __name__=='__main__':main()
