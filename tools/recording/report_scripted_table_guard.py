"""Rule/PPO shared-execution audit, rule comparison gallery and result table."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from report_table_guard_experiment import main as common_report


def main(directory,ppo_directory):
    directory,ppo_directory = Path(directory),Path(ppo_directory)
    first = json.loads((directory/'baseline/seed_20260804/metadata.json').read_text())
    assert first['passive_observer_max_state_difference'] == 0
    old = json.loads((ppo_directory/'measurement_verification.json').read_text())
    verification = dict(passive_observer_full_episode_exact_match=True,
        baseline_max_state_difference=0.,frames=first['frames'],policy='scripted',
        source_metadata_sha256=__import__('hashlib').sha256((directory/'baseline/seed_20260804/metadata.json').read_bytes()).hexdigest(),
        geometry_jacobian_max_error=old['geometry_jacobian_max_error'],
        geometry_verification_reference=str(ppo_directory/'measurement_verification.json'),
        note='Exact full rule episode rerun without Observer; geometry check reuses identical previous environment and model.')
    (directory/'measurement_verification.json').write_text(json.dumps(verification,indent=2))
    common_report(directory)
    rule = json.loads((directory/'paired_manifest.json').read_text())['results']
    ppo = json.loads((ppo_directory/'paired_manifest.json').read_text())['results']
    ppo_by_key = {(r['seed'],r['variant']):r for r in ppo}
    fixed = ('initial_state_sha256','actual_seed','goal_node','goal_initial_xyz','execution_source_sha256',
        'frozen_reward_sha256','frozen_target_sha256','training_config_sha256','target_node_tolerance','guard_source_sha256')
    for r in rule:
        previous = ppo_by_key[r['seed'],r['variant']]
        for key in fixed:
            assert r[key] == previous[key], (r['seed'],key)
        if r['variant'] == 'protected':
            assert r['guard']['config'] == previous['guard']['config']
    comparisons = []
    for name,group in [('ppo',ppo),('scripted',rule)]:
        for variant in ('baseline','protected'):
            subset = [r for r in group if r['variant'] == variant]
            successes = [r for r in subset if r['task_success']]
            comparisons.append(dict(method=name,variant=variant,episodes=len(subset),
                contact_episodes=sum(r['table_contact_episode'] for r in subset),
                first_correct=sum(r['first_correct'] is True for r in subset),
                success=len(successes),mean_success_time_s=float(np.mean([r['duration_s'] for r in successes])) if successes else None,
                intervention_ratio=sum(r['guard']['intervention_steps'] for r in subset)/sum(r['guard']['guard_steps'] for r in subset) if variant=='protected' else 0.,
                min_surface_clearance_m=min(r['min_surface_clearance_m'] for r in subset)))
    rule_by_key = {(r['seed'],r['variant']):r for r in rule}
    common = [(rule_by_key[s,'baseline'],rule_by_key[s,'protected']) for s in range(20260804,20260820)
        if rule_by_key[s,'baseline']['task_success'] and rule_by_key[s,'protected']['task_success']]
    paired_times = dict(episodes=len(common),
        baseline_mean_s=float(np.mean([a['duration_s'] for a,b in common])),
        protected_mean_s=float(np.mean([b['duration_s'] for a,b in common])),
        mean_difference_s=float(np.mean([b['duration_s']-a['duration_s'] for a,b in common])),
        median_difference_s=float(np.median([b['duration_s']-a['duration_s'] for a,b in common])))
    comparison = dict(shared_execution_verified=True,verified_fields=list(fixed),protection_config_identical=True,
        scene='static_workspace_all_nodes',episodes_per_condition=16,success_tolerance=4,
        results=comparisons,rule_common_success_paired_times=paired_times,reward_comparison_valid=False,
        scope='Same 16 initial states and assigned material targets; PPO frozen checkpoint vs fixed rule FSM, exploratory static diagnostic.')
    (directory/'ppo_rule_comparison.json').write_text(json.dumps(comparison,indent=2))
    report = json.loads((directory/'summary.json').read_text())
    report['previous_ppo_shared_execution_comparison'] = comparison
    (directory/'summary.json').write_text(json.dumps(report,indent=2))
    # Shared-success bars clarify that task success, rather than how much
    # cable happened to be lifted before the stop time, is the comparable end.
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    labels = ['PPO original','PPO guarded','Rule original','Rule guarded']
    colors = ['#db704e','#168d8a','#db704e','#168d8a']
    for ax,key,title in zip(axes,['contact_episodes','success'],['Episodes with table contact','Task success episodes']):
        values = [r[key] for r in comparisons]
        bars = ax.bar(labels,values,color=colors)
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+.2,f'{value}/16',ha='center')
        ax.set(title=title,ylim=(0,18),yticks=[0,4,8,12,16])
        ax.tick_params(axis='x',labelrotation=15)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('Shared execution / identical initial states, goals and tolerance | static diagnostic')
    fig.savefig(directory/'ppo_rule_comparison.png',dpi=180)
    plt.close(fig)
    table = '<h2>与上一轮 PPO 的同条件结果</h2><table style="border-collapse:collapse;width:100%;line-height:2"><tr><th>方法</th><th>执行层</th><th>撞桌</th><th>首次抓对</th><th>完整成功</th><th>成功用时均值</th></tr>'
    for r in comparisons:
        seconds = f'{r["mean_success_time_s"]:.2f} s' if r['mean_success_time_s'] is not None else '—'
        table += f'<tr><td>{"规则方法" if r["method"]=="scripted" else "PPO"}</td><td>{"近桌面保护" if r["variant"]=="protected" else "原执行层"}</td><td>{r["contact_episodes"]}/16</td><td>{r["first_correct"]}/16</td><td>{r["success"]}/16</td><td>{seconds}</td></tr>'
    table += '</table><p>固定静态小样本诊断；成功范围沿用 ±4。规则方法仅用状态机决策，不用 PPO 权重；累计奖励不作跨方法比较。</p><img src="ppo_rule_comparison.png" style="width:100%;background:white;border-radius:12px">'
    html = (directory/'index.html').read_text(encoding='utf8')
    html = html.replace('<h2>16 组配对结果</h2>',table+'<h2>16 组规则配对结果</h2>')
    (directory/'index.html').write_text(html,encoding='utf8')
    print(json.dumps(comparison,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--ppo-directory',required=True,type=Path)
    args = parser.parse_args()
    main(args.directory,args.ppo_directory)
