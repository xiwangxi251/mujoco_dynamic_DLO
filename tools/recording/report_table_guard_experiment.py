"""Audit, plots and synchronized gallery for the paired execution experiment."""
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_retest_136_gallery import HTML


def main(directory):
    directory = Path(directory)
    variants = ('baseline','protected')
    rows = []
    groups = {}
    for variant in variants:
        groups[variant] = []
        for path in sorted((directory/variant).glob('seed_*/metadata.json')):
            meta = json.loads(path.read_text())
            meta['directory'] = path.parent.relative_to(directory).as_posix()
            groups[variant].append(meta)
            rows.append(meta)
    n = len(groups['baseline'])
    assert n > 0 and n == len(groups['protected'])
    is_rule = rows[0].get('policy_kind') == 'scripted'
    is_dynamic = bool(rows[0].get('dynamic_test'))
    method_title = 'Rule policy' if is_rule else f'PPO step {rows[0]["checkpoint_steps"]:,}'
    fixed = ('initial_state_sha256','actual_seed','goal_node','goal_initial_xyz','checkpoint_sha256',
             'execution_source_sha256','frozen_reward_sha256','frozen_target_sha256','training_config_sha256','target_node_tolerance','guard_source_sha256')
    if is_rule:
        fixed += ('frozen_rule_sha256','rule_config','reference_ppo_checkpoint_sha256')
    for a,b in zip(groups['baseline'],groups['protected']):
        assert a['seed'] == b['seed']
        assert a['reward_changes'] is b['reward_changes'] is False
        for key in fixed:
            assert a[key] == b[key], (a['seed'],key)
    assert [r['seed'] for r in groups['baseline']] == list(range(20260804,20260804+n))
    # All showcased clips must have completed independent encoding and
    # saved-trajectory integrity checks before publishing the gallery.
    showcase = list(range(min(4,n)))
    if (directory/'showcase_episodes.json').exists():
        showcase = json.loads((directory/'showcase_episodes.json').read_text())['episodes']
    elif is_rule:
        showcase = sorted(set(showcase) | {r['episode'] for r in rows if not r['task_success']})
    video_checks = []
    for e in showcase:
        for variant in variants:
            video_directory = directory/variant/f'seed_{20260804+e}'
            check = json.loads((video_directory/'video_verification.json').read_text())
            assert check['rendered_state_max_difference'] == 0
            for view in ('global','wrist'):
                assert (video_directory/f'{view}.mp4').is_file()
                assert check['encoding'][view]['timestamps_verified']
                assert check['encoding'][view]['max_keyframe_spacing_frames'] <= 12
            video_checks.append(dict(variant=variant,seed=20260804+e,**check))
    measurement = json.loads((directory/'measurement_verification.json').read_text())
    assert measurement['passive_observer_full_episode_exact_match']
    (directory/'video_manifest.json').write_text(json.dumps(video_checks,indent=2))
    metrics = ('peak_first_impact_down_speed_m_s','peak_contact_down_speed_m_s',
               'peak_table_normal_force_N','table_normal_impulse_Ns','table_contact_s',
               'min_surface_clearance_m','max_lifted_fraction','return_')
    if is_rule:
        metrics = tuple(k for k in metrics if k != 'return_')+('duration_s',)
    summaries = {}
    for variant, group in groups.items():
        summaries[variant] = dict(episodes=len(group),
            table_contact_episodes=sum(r['table_contact_episode'] for r in group),
            first_correct=sum(r['first_correct'] is True for r in group),
            first_wrong=sum(r['first_correct'] is False for r in group),
            no_first_grasp=sum(r['first_correct'] is None for r in group),
            success=sum(r['task_success'] for r in group),
            table_contact_events=sum(r['table_contact_events'] for r in group),
            mean_success_time_s=float(np.mean([r['duration_s'] for r in group if r['task_success']])) if any(r['task_success'] for r in group) else None,
            **{key:dict(mean=float(np.mean([r[key] for r in group])),
                         median=float(np.median([r[key] for r in group])),
                         min=float(min(r[key] for r in group)), max=float(max(r[key] for r in group))) for key in metrics})
        if variant == 'protected':
            total = sum(r['guard']['guard_steps'] for r in group)
            summaries[variant].update(intervention_ratio=sum(r['guard']['intervention_steps'] for r in group)/total,
                feedback_ratio=sum(r['guard']['velocity_feedback_steps'] for r in group)/total,
                projection_residual_steps=sum(r['guard']['projection_residual_steps'] for r in group),
                guard_config=group[0]['guard']['config'])
    rng = np.random.default_rng(20261004)
    bootstrap = {}
    for key in metrics:
        difference = np.array([b[key]-a[key] for a,b in zip(groups['baseline'],groups['protected'])])
        draws = difference[rng.integers(0,len(difference),size=(10000,len(difference)))].mean(axis=1)
        bootstrap[key] = dict(mean_difference_protected_minus_baseline=float(difference.mean()),
            paired_bootstrap_95_percent_interval=np.quantile(draws,[.025,.975]).tolist())
    report = dict(checkpoint=rows[0]['checkpoint'],checkpoint_steps=rows[0]['checkpoint_steps'],
        policy_kind='scripted' if is_rule else 'ppo', reward_comparison_valid=not is_rule,
        checkpoint_sha256=rows[0]['checkpoint_sha256'],stage=rows[0]['stage'],episodes_per_variant=n,
        seeds=[r['seed'] for r in groups['baseline']],summaries=summaries,paired_differences=bootstrap,
        pair_verification=dict(passed=True,identical_fields=list(fixed)),
        success_tolerance=rows[0]['target_node_tolerance'],reward_modified=False,
        uncertainty='small paired dynamic diagnostic; not the frozen paper benchmark or hardware validation' if is_dynamic else 'small static-stage paired diagnostic; not the frozen paper benchmark or hardware validation',
        collision_definition='robot-table contact with contact.dist<0 and normal_force>1e-6 N at 500 Hz',
        impact_speed_definition='pre-physics normal velocity of lowest vertex of contacting finger body; zero for no-contact episodes',
        force_definition='sum of robot-table contact normal forces, simulated values; not measured hardware forces',
        guard_scope='5 deduplicated link7/flange/palm/finger meshes; lower arm geometry not protected',
        guard_type='500 Hz command-only servo velocity projection with distance/braking envelope and actual velocity feedback',
        limitations=['Only position commands change; actuator gains/torque limits, contact geometry, physics, observations and reward definitions remain unchanged.',
            'No joint acceleration/jerk filter in this prototype.',
            'Projection residuals are reported; no formal invariant-safety guarantee.',
            ('Rule state machine targets the assigned material node; no learned policy loaded; static diagnostic only.' if is_rule else
             'Existing PPO is evaluated without retraining; other policy families not yet tested.')])
    if is_dynamic:
        report['limitations'][-1] = 'Rule state machine with assigned node and full nominal dynamic motion; privileged state diagnostic.'
    (directory/'summary.json').write_text(json.dumps(report,indent=2))
    fields = ('variant','seed','actual_seed','scenario','goal_node','first_grasp_node','first_correct','task_success',
        'table_contact_episode','table_contact_events','table_contact_s',*metrics,'steps','checkpoint_sha256','initial_state_sha256')
    with (directory/'episodes.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (directory/'paired_manifest.json').write_text(json.dumps(dict(results=rows,paired_verification_passed=True),indent=2))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    plot_keys = ['peak_first_impact_down_speed_m_s','peak_table_normal_force_N','min_surface_clearance_m','duration_s' if is_rule else 'max_lifted_fraction']
    plot_labels = ['Peak impact descent speed (m/s)','Peak table normal force (N)','Minimum surface clearance (mm)','Episode end time (s)' if is_rule else 'Maximum lifted cable fraction (%)']
    for ax,key,label,scale in zip(axes.flat,plot_keys,plot_labels,[1,1,1000,1 if is_rule else 100]):
        for variant,color in zip(variants,['#db704e','#168d8a']):
            ax.plot(np.arange(1,n+1),[r[key]*scale for r in groups[variant]],'o-',color=color,label=variant,linewidth=1.3,markersize=4)
        ax.set(xlabel='Paired held-out episode',ylabel=label,xticks=sorted(set([1,n]+list(range(4,n+1,4)))))
        ax.grid(alpha=.18)
        if key == 'min_surface_clearance_m':
            ax.axhline(0,color='#333',linewidth=.8)
            ax.axhline(1,color='#168d8a',linewidth=.8,linestyle=':',label='1 mm nominal margin')
        ax.legend(fontsize=9)
    fig.suptitle(f'{method_title} | unchanged success | {n} paired {"dynamic" if is_dynamic else "static"} episodes')
    fig.savefig(directory/'paired_results.png',dpi=180)
    plt.close(fig)
    # Actual time histories of the first predeclared seed, without picking a
    # favorable episode after seeing its outcome.
    fig, axes = plt.subplots(3,1,figsize=(12,8),layout='constrained',sharex=True)
    for variant,color in zip(variants,['#db704e','#168d8a']):
        data = json.loads((directory/variant/'seed_20260804/metrics.json').read_text())
        for ax,key,scale in zip(axes,['surface_clearance_m','down_speed_m_s','normal_force_N'],[1000,1,1]):
            ax.plot([r['time'] for r in data],[r[key]*scale for r in data],color=color,label=variant,linewidth=1)
    for ax,label in zip(axes,['Surface clearance (mm)','Downward surface speed (m/s)','Table normal force (N)']):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].axhline(0,color='black',linewidth=.7)
    axes[-1].set_xlabel('Actual simulation time (s)')
    fig.suptitle('First paired seed 20260804 | 50 Hz display, contacts monitored at 500 Hz')
    fig.savefig(directory/'first_seed_trace.png',dpi=180)
    plt.close(fig)
    # Retain the tested coalesced 80 ms seek controller from the existing gallery.
    s = HTML.replace('repeat(3,minmax(0,1fr))','repeat(2,minmax(0,1fr))')
    s = s.replace('版本 1、3、6 · 工作范围实际录像',('规则方法' if is_rule else 'PPO')+' · 原执行层 / 近桌面保护')
    s = s.replace('相同 home、目标、末端朝向和测试轨迹。','同一检查点、初始状态、目标节点和种子。奖励与成功判定保持当前版本（±4）。')
    if is_rule:
        s = s.replace('同一检查点、初始状态、目标节点和种子。奖励与成功判定保持当前版本（±4）。',
            '规则状态机使用与上一轮 PPO 相同的初始状态、指定目标节点和种子，公共执行层参数一致。成功判定沿用 ±4。')
    options = ''.join(f'<option value="{20260804+i}">{"补充失败诊断 · " if i>=4 else ""}第{i+1}组 · seed {20260804+i}</option>' for i in showcase)
    if is_dynamic:
        names = {'id_rigid_l1_nominal':'整体运动','id_shape_nominal_current':'形变','id_combined_l1_nominal':'运动＋形变'}
        by_episode = {r['episode']:r for r in groups['baseline']}
        options = ''.join(f'<option value="{20260804+i}">{names[by_episode[i]["scenario"]]} · seed {20260804+i}</option>' for i in showcase)
        s = s.replace('规则方法 · 原执行层 / 近桌面保护','规则方法动态场景 · 原执行层 / 近桌面保护')
        s = s.replace('规则状态机使用与上一轮 PPO 相同的初始状态、指定目标节点和种子，公共执行层参数一致。成功判定沿用 ±4。',
            '整体运动、形变、运动与形变叠加，nominal 强度；保护前后场景、初始状态和指定节点一致。沿用上一轮执行参数，成功判定 ±4。')
    s = re.sub(r'<label>目标 <select id="goal">.*?</select></label>',f'<label>种子 <select id="goal">{options}</select></label>',s)
    s = s.replace('三路同步进度','两路同步进度')
    video_selection = f'前4个连续种子，另补充{len(showcase)-4}组失败诊断对照' if len(showcase)>4 else '前4个连续种子'
    if is_dynamic:
        video_selection = '每类首组，并补充不同成功/失败结果的诊断对照'
    note = (f'{n} 组配对测试；视频展示{video_selection}，全局/腕部共 {4*len(showcase)} 段。'
        '黄色圆圈表示目标节点，绿色显示当前未修改的 ±4 成功范围。'
        '视频是每 20 ms 保存的真实物理状态重放。接触力为仿真值。拖动进度会暂停并在定位后恢复原播放状态。')
    s = re.sub(r'<div class="note" id="note">.*?</div>',f'<div class="note" id="note">{note}</div>',s)
    s = re.sub(r'<footer>.*?</footer>',
        f'<h2>{n} 组配对结果</h2><img src="paired_results.png" style="width:100%;background:white;border-radius:12px">'
        '<h2>第一组时间序列</h2><img src="first_seed_trace.png" style="width:100%;background:white;border-radius:12px">'
        '<footer><a href="summary.json">汇总指标</a><a href="episodes.csv">逐回合 CSV</a>'
        '<a href="paired_manifest.json">初始状态 / 来源核查</a><a href="measurement_verification.json">观测无干扰核查</a></footer>',s)
    s = s.replace('ids=[1,3,6]',"ids=['baseline','protected']")
    load = r'''function load(){cancel();const seed=Number($('goal').value),view=$('view').value;ids.forEach(id=>{const e=D.find(e=>e.variant===id&&e.seed===seed),v=$('video-'+id);v.pause();v.src=e.directory+'/'+view+'.mp4';v.poster=e.directory+'/'+view+'_poster.png';v.playbackRate=Number($('rate').value);v.load();$('badge-'+id).textContent=e.table_contact_episode?'发生桌面接触':'无桌面接触';$('badge-'+id).className='badge'+(e.table_contact_episode?'':' pass');$('meta-'+id).innerHTML=`目标 ${e.goal_node} · 首抓 ${e.first_grasp_node} · 容差内 ${e.first_correct} · 完整成功 ${e.task_success}<br>撞击下降峰值 ${e.peak_first_impact_down_speed_m_s.toFixed(3)} m/s · 接触力峰值 ${e.peak_table_normal_force_N.toFixed(1)} N<br>最小离桌 ${(e.min_surface_clearance_m*1000).toFixed(2)} mm · 干预 ${e.guard?(100*e.guard.intervention_ratio).toFixed(1)+'%':'0%'}<br><span id="frame-clock-${id}">当前播放帧：t=0.00 s</span> · <a href="${e.directory}/metadata.json">运行记录</a>`});$('seek').value=0}
async function play'''
    s = re.sub(r'function load\(\).*?\nasync function play',lambda _:load,s,flags=re.S)
    tick = r'''function tick(){const t=Math.max(...vs().map(v=>v.currentTime||0));ids.forEach(id=>{$('frame-clock-'+id).textContent=`当前播放帧：t=${($('video-'+id).currentTime||0).toFixed(2)} s`});if(!scrubbing&&pending===null&&!vs().some(v=>v.seeking)){$('seek').value=t;$('clock').textContent=`${t.toFixed(2)} / ${Number($('seek').max).toFixed(2)} s`;}requestAnimationFrame(tick)}window.addEventListener('pointerup',()=>{if(scrubbing)end()});load();tick();
</script>'''
    s = re.sub(r'function tick\(\).*?</script>',lambda _:tick,s,flags=re.S)
    players = ''.join(f'<article class="player"><div class="head"><strong>{name}</strong><span id="badge-{v}" class="badge"></span></div><video id="video-{v}" controls playsinline preload="auto" aria-label="{name}"></video><div class="meta" id="meta-{v}"></div></article>' for v,name in [('baseline','原执行层'),('protected','近桌面保护')])
    # Rule diagnostics can contain undefined angles. JSON.parse requires strict
    # JSON; use null in the browser payload while retaining the raw run logs.
    def browser_json(value):
        if isinstance(value, dict):
            return {k: browser_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [browser_json(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    s = s.replace('__PLAYERS__',players).replace('__DATA__',json.dumps(browser_json(rows),ensure_ascii=False,allow_nan=False))
    (directory/'index.html').write_text(s,encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    main(parser.parse_args().directory)
