"""Dynamic rule diagnostic: exact paired execution, scene strata and gallery."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from report_table_guard_experiment import main as common_report

SCENES = ('id_rigid_l1_nominal','id_shape_nominal_current','id_combined_l1_nominal')
NAMES = ('整体运动','形变','运动＋形变')


def read_rows(directory, variant):
    return [json.loads(p.read_text()) for p in sorted((directory/variant).glob('seed_*/metadata.json'))]


def main(directory, reference, plan):
    baseline, protected = read_rows(directory,'baseline'), read_rows(directory,'protected')
    assert len(baseline) == len(protected) == 12
    old = json.loads((reference/'baseline/seed_20260804/metadata.json').read_text())
    old_guard = json.loads((reference/'protected/seed_20260804/metadata.json').read_text())
    shared = ('controller_profile','execution_source_sha256','frozen_reward_sha256','frozen_target_sha256',
        'frozen_rule_sha256','rule_config','target_node_tolerance','guard_source_sha256')
    paired = shared+('seed','actual_seed','initial_state_sha256','goal_node','goal_initial_xyz','scenario','stage','motion_difficulty')
    for a,b in zip(baseline,protected):
        for key in paired:
            assert a[key] == b[key], (a['seed'],key)
        for row in (a,b):
            assert row['dynamic_test'] and row['motion_difficulty'] == 1
            assert row['scenario'] == SCENES[row['episode']%3]
            assert row['pilot_limit'] is None and (row['terminated'] or row['truncated'])
            for key in shared:
                assert row[key] == old[key], (row['seed'],key)
        assert b['guard']['config'] == old_guard['guard']['config']
    assert baseline[0]['passive_observer_max_state_difference'] == 0
    selected = set()
    for scenario in SCENES:
        pairs = [(a,b) for a,b in zip(baseline,protected) if a['scenario'] == scenario]
        assert len(pairs) == 4
        selected.add(pairs[0][0]['episode'])
        # Show each scene's first pair, plus representative success/failure if
        # the first pair lacks it. This is diagnostic selection, not the tally.
        for success in (True,False):
            candidates = [(a,b) for a,b in pairs if a['task_success'] == success or b['task_success'] == success]
            if candidates:
                selected.add(candidates[0][0]['episode'])
    showcase = dict(episodes=sorted(selected),selection='First pair of each scene plus first pair representing success/failure when present; all 12 pairs enter statistics.')
    (directory/'showcase_episodes.json').write_text(json.dumps(showcase,indent=2))
    geometry = json.loads((reference/'paired_geometry_reference.json').read_text()) if (reference/'paired_geometry_reference.json').exists() else json.loads((reference/'measurement_verification.json').read_text())
    measurement = dict(passive_observer_full_episode_exact_match=True,baseline_max_state_difference=0,
        frames=baseline[0]['frames'],policy='scripted',
        geometry_jacobian_max_error=geometry['geometry_jacobian_max_error'],
        geometry_verification_reference=str(reference/'measurement_verification.json'),
        note='First dynamic baseline repeated without Observer; identical states, actions, observations and bookkeeping rewards. Geometry check reused from same frozen model.')
    (directory/'measurement_verification.json').write_text(json.dumps(measurement,indent=2))
    if plan:
        print(' '.join(map(str,showcase['episodes'])))
        return
    common_report(directory)
    results = []
    for scene in SCENES:
        for variant, rows in (('baseline',baseline),('protected',protected)):
            subset = [r for r in rows if r['scenario'] == scene]
            successes = [r for r in subset if r['task_success']]
            results.append(dict(scenario=scene,variant=variant,episodes=len(subset),
                success=len(successes),first_correct=sum(r['first_correct'] is True for r in subset),
                first_wrong=sum(r['first_correct'] is False for r in subset),
                no_first_grasp=sum(r['first_correct'] is None for r in subset),
                contact_episodes=sum(r['table_contact_episode'] for r in subset),
                mean_success_time_s=float(np.mean([r['duration_s'] for r in successes])) if successes else None,
                mean_workspace_inside_fraction=float(np.mean([r['workspace_inside_fraction'] for r in subset])),
                goal_pregrasp_max_speed_m_s=max(r['goal_pregrasp_max_speed_m_s'] for r in subset),
                intervention_ratio=sum(r['guard']['intervention_steps'] for r in subset)/sum(r['guard']['guard_steps'] for r in subset) if variant=='protected' else 0.,
                failure_seeds=[r['seed'] for r in subset if not r['task_success']]))
    audit = dict(shared_with_static_rule_execution_verified=True,shared_fields=list(shared),paired_fields=list(paired),
        guard_config_identical=True,scene_results=results,video_selection=showcase,
        scope='4 paired episodes per full nominal dynamic scene; privileged rule state; no dynamic PPO run and no hardware claim.')
    (directory/'dynamic_scene_summary.json').write_text(json.dumps(audit,indent=2))
    summary = json.loads((directory/'summary.json').read_text())
    summary['dynamic_scene_audit'] = audit
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    table = '<h2>动态场景结果</h2><table style="width:100%;line-height:2"><tr><th>场景</th><th>执行层</th><th>成功</th><th>首次抓对</th><th>撞桌</th><th>成功用时</th><th>保护介入</th><th>抓前工作区内</th></tr>'
    for r in results:
        seconds = f'{r["mean_success_time_s"]:.2f} s' if r['mean_success_time_s'] is not None else '—'
        table += f'<tr><td>{NAMES[SCENES.index(r["scenario"])]}</td><td>{"保护" if r["variant"]=="protected" else "原执行层"}</td><td>{r["success"]}/4</td><td>{r["first_correct"]}/4</td><td>{r["contact_episodes"]}/4</td><td>{seconds}</td><td>{r["intervention_ratio"]:.1%}</td><td>{r["mean_workspace_inside_fraction"]:.1%}</td></tr>'
    table += '</table><p>每类仅 4 组，全部保留。nominal 强度，15 秒回合；奖励和 ±4 成功范围未修改。视频选择用于诊断，不代表总体成功率。</p>'
    html = (directory/'index.html').read_text(encoding='utf8').replace('<h2>12 组配对结果</h2>',table+'<h2>12 组配对结果</h2>')
    (directory/'index.html').write_text(html,encoding='utf8')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--plan',action='store_true')
    args = parser.parse_args()
    main(args.directory,args.reference,args.plan)
