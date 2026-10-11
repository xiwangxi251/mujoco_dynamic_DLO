"""Audit and report the completed frozen paired scripted-control ablation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/benchmarks/nero_scripted_tracking_fix_20261001'
VARIANTS=('baseline','tracking_fix','tracking_fix_orientation4')
LABELS={'baseline':'Jacobian 原设置','tracking_fix':'Jacobian＋执行层修正','tracking_fix_orientation4':'Jacobian＋执行层修正＋姿态反馈×4'}
SCENES={'id_static':'静态','id_shape_nominal_current':'形变',
        'id_rigid_l1_nominal':'整体运动 L1','id_rigid_l2_nominal':'整体运动 L2',
        'id_combined_l1_nominal':'组合运动 L1','id_combined_l2_nominal':'组合运动 L2'}


def plot_success_rates(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    names=list(SCENES)+['overall']
    labels=list(SCENES.values())+['合计']
    colors=('#738194','#1c9678','#497bc3')
    x=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(13,5.2),layout='constrained')
    for i,v in enumerate(VARIANTS):
        rates=[100*summary[name]['variants'][v]['task_success_rate'] for name in names]
        bars=ax.bar(x+(i-1)*.25,rates,.23,label=LABELS[v],color=colors[i])
        ax.bar_label(bars,labels=[f'{r:.0f}%' if name!='overall' else f'{r:.1f}%' for name,r in zip(names,rates)],fontsize=9,padding=3)
    ax.set(xticks=x,xticklabels=labels,ylim=(0,109),ylabel='原环境任务成功率 / %',
        title='NERO 规则方法（Jacobian IK）：执行层修正的配对抓取测试')
    ax.set_yticks(np.arange(0,101,20))
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    ax.set_title('NERO 规则方法（Jacobian IK）：执行层修正的配对抓取测试',pad=40)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False)
    fig.supxlabel('6 个场景 × 每组 50 回合；相同 seed 配对；共 900 回合。差值置信区间见报告。',fontsize=10)
    fig.savefig(out/'success_rates.png',dpi=180)
    fig.savefig(out/'success_rates.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    out=args.output
    folder=out/'main'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    rows=json.loads((folder/'episodes.json').read_text(encoding='utf-8'))
    verification=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
    assert len(rows)==900==verification['unique_rows']
    assert hashlib.sha256((folder/'source_snapshot.py').read_bytes()).hexdigest()==manifest['script_sha256']
    assert hashlib.sha256((ROOT/'tools/evaluation/compare_scripted_tracking_fix.py').read_bytes()).hexdigest()==manifest['script_sha256']
    for path,expected in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    assert all((out/r['compiled_model']).exists() for r in rows)
    assert all((out/r['diagnostic_trace']).exists() for r in rows)
    import mujoco
    model_hashes={}
    for model_path in {r['compiled_model'] for r in rows}:
        model=mujoco.MjModel.from_binary_path(str(out/model_path))
        h=hashlib.sha256()
        for attr in ('actuator_gainprm','actuator_biasprm'):
            h.update(attr.encode()); h.update(np.asarray(getattr(model,attr)).tobytes())
        h.update(model.opt.gravity.tobytes()); h.update(str(model.opt.timestep).encode())
        model_hashes[model_path]=h.hexdigest()
    assert all(model_hashes[r['compiled_model']]==r['actuator_hash'] for r in rows)
    for r in rows:
        with np.load(out/r['diagnostic_trace']) as trace:
            states=trace['states']; nq=int(trace['nq'])
            assert states.ndim==2 and states.shape[1]==nq+17 and states.shape[0]>0
            assert np.all(np.isfinite(states)),r['diagnostic_trace']
            assert np.all(np.diff(states[:,0])>0),r['diagnostic_trace']
            assert states[-1,0]<=r['sim_time']+1e-8
    assert all(r['success_snapshot'] is not None and r['success_snapshot']['success_hold']>=.8-1e-12
               for r in rows if r['task_success'])
    bykey={(r['scenario_name'],r['requested_seed'],r['variant']):r for r in rows}
    for name,seed,_ in bykey:
        group=[bykey[name,seed,v] for v in VARIANTS]
        assert all(r['env_config']==group[0]['env_config'] for r in group)
    assert all(np.all(np.isfinite(np.asarray(r['gravity_ff_max_nm']))) for r in rows)
    metrics={}
    for v in VARIANTS:
        rr=[r for r in rows if r['variant']==v]
        limits=np.asarray(rr[0]['env_config']['arm_joint_velocity_limits'],dtype=float)
        peak=np.max([np.asarray(json.loads(r['max_abs_actual_arm_velocity']),dtype=float)/limits for r in rr])
        phases={}
        for phase in sorted({p for r in rr for p in r['phase_tracking']}):
            pp=[r['phase_tracking'][phase] for r in rr if phase in r['phase_tracking']]
            n=sum(p['steps'] for p in pp)
            phases[phase]={'control_steps':n,
                'mean_tilt_deg':sum(p['tilt_sum_deg'] for p in pp)/n,
                'max_tilt_deg':max(p['max_tilt_deg'] for p in pp),
                'mean_joint_target_error_norm_rad':sum(p['joint_error_sum_rad'] for p in pp)/n}
        metrics[v]={'phase_tracking':phases,
            'ever_bilateral_candidate':sum(r['ever_bilateral_candidate'] for r in rr),
            'ever_confirmed_grasp':sum(r['ever_confirmed_grasp'] for r in rr),
            'physical_slip_after_confirmed_episodes':sum(r['physical_slip_after_confirmed_count']>0 for r in rr),
            'force_diagnostics_instrumented':v!='baseline',
            'total_torque_saturation_substeps':sum(r['total_torque_saturation_steps'] for r in rr) if v!='baseline' else None,
            'max_arm_gravity_ff_nm':np.max([r['gravity_ff_max_nm'] for r in rr],axis=0).tolist(),
            'max_total_arm_torque_nm':np.max([r['total_torque_max_nm'] for r in rr],axis=0).tolist() if v!='baseline' else None,
            'mean_actual_joint_velocity_exceedance_ratio':float(np.mean([r['actual_joint_velocity_exceedance_ratio'] for r in rr])),
            'episodes_with_actual_joint_velocity_exceedance':sum(r['actual_joint_velocity_exceedance_ratio']>0 for r in rr),
            'max_peak_joint_velocity_to_limit_ratio':float(peak),
            'mean_motion_limit_active_ratio':float(np.mean([r['motion_limit_active_ratio'] for r in rr]))}
    (out/'tracking_and_grasp_diagnostics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    source=out/'remote_launch.json'
    runtime=json.loads(source.read_text(encoding='utf-8')) if source.exists() else {'hostname':'local','workers':manifest['workers']}
    primary=summary['overall']['comparisons']['tracking_fix']
    seeds=manifest['seeds']
    extra_delta=np.array([np.mean([
        int(bykey[name,seed,'tracking_fix_orientation4']['task_success'])-
        int(bykey[name,seed,'tracking_fix']['task_success']) for name in SCENES]) for seed in seeds])
    rng=np.random.default_rng(1012026)
    extra_boot=extra_delta[rng.integers(0,len(seeds),size=(20000,len(seeds)))].mean(axis=1)
    extra={'reference':'tracking_fix','comparison':'tracking_fix_orientation4',
        'delta_rate':float(extra_delta.mean()),'paired_bootstrap95':np.quantile(extra_boot,[.025,.975]).tolist(),
        'bootstrap_unit':'paired seed block including all six scenes',
        'improved':sum(not bykey[name,seed,'tracking_fix']['task_success'] and bykey[name,seed,'tracking_fix_orientation4']['task_success'] for name in SCENES for seed in seeds),
        'worsened':sum(bykey[name,seed,'tracking_fix']['task_success'] and not bykey[name,seed,'tracking_fix_orientation4']['task_success'] for name in SCENES for seed in seeds)}
    (out/'orientation_increment_comparison.json').write_text(json.dumps(extra,indent=2),encoding='utf-8')
    plot_success_rates(out,summary)
    lines=['# NERO 规则方法（Jacobian IK）：重力前馈与伺服修正的配对抓取测试','',
        '**入口纠正说明：这900回合全部使用规则策略原有的速度层分层Jacobian IK，没有调用项目迭代Pose IK。它们只评估该旧IK路径的执行层修正，不能作为Pose IK组合的抓取成功率证据。实验选择没有遵循此前已明确的Pose IK方向，是本次设计失误；保留原数据用于审计与复用对照，不将其重新标成Pose IK结果。**','',
        '2026-10-01。完整工程消融，使用当前本地冻结源码；没有替换历史论文或旧规则方法结果。', '',
        '## 主要结果','',
        '| 场景 | 原默认 | 执行层修正 | 再加强姿态反馈 |','|---|---:|---:|---:|']
    for name in SCENES:
        cell=summary[name]['variants']
        values=[f"{cell[v]['task_successes']}/{cell[v]['episodes']}（{100*cell[v]['task_success_rate']:.1f}%）" for v in VARIANTS]
        lines.append('| '+SCENES[name]+' | '+' | '.join(values)+' |')
    cell=summary['overall']['variants']
    lines.append('| **合计** | '+' | '.join(f"**{cell[v]['task_successes']}/{cell[v]['episodes']}（{100*cell[v]['task_success_rate']:.1f}%）**" for v in VARIANTS)+' |')
    lines+=['',f"主要比较为执行层修正相对原默认：成功率差值 {primary['delta_rate']*100:+.2f} 个百分点；按相同seed的六场景整体重采样得到95%配对区间 [{primary['paired_bootstrap95'][0]*100:+.2f}, {primary['paired_bootstrap95'][1]*100:+.2f}] 个百分点。新增成功 {primary['improved']} 回合，退化 {primary['worsened']} 回合。",'',
        f"加强姿态反馈相对仅执行层修正：差值 {100*extra['delta_rate']:+.2f} 个百分点，95%配对区组区间 [{100*extra['paired_bootstrap95'][0]:+.2f}, {100*extra['paired_bootstrap95'][1]:+.2f}]；新增成功 {extra['improved']} 回合，退化 {extra['worsened']} 回合。该比较检验额外姿态反馈收益，不把第三组相对原默认的全部收益归因于姿态反馈。",'',
        '本次结果支持执行层修正提高六场景平均任务成功率，整体运动场景的提升最明显。但形变场景仅执行层修正从40%下降到30%，因此不能声称所有场景都改善。额外姿态反馈的总体增量区间跨过0，目前还不能明确确认这项额外收益。', '',
        f"执行层修正后，形成确认抓取的回合由{metrics['baseline']['ever_confirmed_grasp']}增加到{metrics['tracking_fix']['ever_confirmed_grasp']}，但滑脱失败由{cell['baseline']['task_physical_slip_failures']}增加到{cell['tracking_fix']['task_physical_slip_failures']}；回合平均超速物理子步比例由{100*metrics['baseline']['mean_actual_joint_velocity_exceedance_ratio']:.3f}%增加到{100*metrics['tracking_fix']['mean_actual_joint_velocity_exceedance_ratio']:.3f}%。这证明原任务分数改善，并不证明当前增益同时满足实际速度约束。后续应保留重力补偿的候选方案，继续调节伺服增益与抬升轨迹。本次修改仅在隔离测试包装中实施，生产控制文件没有替换。",'',
        '## 样本、配置与成功判定','',
        '- core全部6个场景，每配置每场景50回合；三配置共900回合。seeds为2026100100–2026100149，在运行前写入manifest。18回合先导使用另一个seed，不进入主表。',
        '- 原默认：当前规则策略与原执行层。',
        '- 执行层修正：不改规则策略；仅臂关节kp×4、kv×2，并在500 Hz物理步加入姿态相关的静态重力前馈；PD＋前馈总力矩按原限幅裁剪。',
        '- 再加强姿态反馈：在执行层修正基础上，把规则IK中的四元数姿态误差反馈×4（等效角速度增益2.5→10），随后仍使用原1.4 rad/s角速度限制；返回给阶段门控的原转角不变。该项单独报告，未混入主要执行层比较。',
        '- 规则的目标选择、预测时域、阶段门控、重试规则、指爪增益、线缆材料、摩擦、接触参数、重力、质量和速度限制不变；没有换成Pose IK，也没有关闭摩擦。',
        '- 相同scene_fingerprint、base_scene_fingerprint、scenario_id、motion_profile_hash及物理参数哈希逐seed核验；只有明确记录的执行器增益和控制前馈不同。',
        '- 仍以原环境task_success/ever_success计数：真实双侧确认夹持、夹爪闭合、目标段高度与整线离桌比例、中心距离等条件连续满足0.80秒；每回合最多15秒，原整体运动越界条件保持。策略内部成功和末端对准不代替任务成功。',
        f"- 执行平台：{runtime.get('hostname')}；MuJoCo {manifest['mujoco']}，Python {manifest['python']}，{runtime.get('workers',manifest['workers'])}个CPU worker。主表全部回合来自同一平台。",'',
        '## 配对变化与不确定性','',
        '| 场景 | 修改 | 新增成功 | 退化 | 差值/百分点 | 95%配对bootstrap区间 | McNemar原始p |',
        '|---|---|---:|---:|---:|---|---:|']
    for name in SCENES:
        for v in VARIANTS[1:]:
            c=summary[name]['comparisons'][v]
            low,high=c['paired_bootstrap95']
            lines.append(f"| {SCENES[name]} | {LABELS[v]} | {c['improved']} | {c['worsened']} | {100*c['delta_rate']:+.1f} | [{100*low:+.1f}, {100*high:+.1f}] | {c['mcnemar_exact_p']:.5f} |")
    lines+=['', '场景内采用exact McNemar和配对seed bootstrap；p值为未经多重比较校正的诊断值。总体bootstrap以seed为区组，把同seed的六场景一起重采样；不把它们当成300个完全独立样本。总体不报告合并McNemar p值。50回合/场景属于本次工程验证，不能替代协议中正式新论文实验至少100回合/场景的要求。', '',
        '## 失败类型与执行诊断','',
        '| 公共结果类别 | 原默认 | 执行层修正 | 再加强姿态反馈 |','|---|---:|---:|---:|']
    for outcome in summary['overall']['variants']['baseline']['outcome_counts']:
        lines.append('| '+outcome+' | '+' | '.join(str(summary['overall']['variants'][v]['outcome_counts'][outcome]) for v in VARIANTS)+' |')
    lines+=['','各阶段的倾角/关节误差是实际执行轨迹诊断，阶段时长随控制改变，不能将它们当成相同轨迹窗口上的单独因果比较；详见tracking_and_grasp_diagnostics.json。','',
        '| 执行诊断 | 原默认 | 执行层修正 | 再加强姿态反馈 |','|---|---:|---:|---:|']
    for key,label in [('ever_bilateral_candidate','形成双侧候选回合'),('ever_confirmed_grasp','形成确认抓取回合'),
        ('physical_slip_after_confirmed_episodes','确认后曾物理滑脱回合'),('total_torque_saturation_substeps','PD＋前馈联合力矩限幅子步')]:
        lines.append('| '+label+' | '+' | '.join(str(metrics[v][key]) if metrics[v][key] is not None else '未额外插桩' for v in VARIANTS)+' |')
    lines.append('| 回合平均瞬时关节超速子步占比 | '+' | '.join(f"{100*metrics[v]['mean_actual_joint_velocity_exceedance_ratio']:.3f}%" for v in VARIANTS)+' |')
    lines.append('| 最坏峰值关节速度/该关节限制 | '+' | '.join(f"{metrics[v]['max_peak_joint_velocity_to_limit_ratio']:.3f}×" for v in VARIANTS)+' |')
    lines+=['', '确认后曾滑脱计数包括后来重试成功的回合，不等于互斥的主失败类别。当前速度守卫在关节达到速度阈值后停止同向驱动，物理步仍可能产生瞬时超限；它不是实际速度的硬裁剪。原始逐回合actual_joint_velocity_exceedance_ratio仍保留，并按配置汇总到诊断JSON；不能把伺服修正自动视为真机安全参数。', '',
        '## 验证与原始产物','',
        '- independent_verification.json：诊断包装与项目原benchmark的基线逐项一致；20个姿态的机器人重力探针与完整场景重力矩一致，位置伺服力矩计算与MuJoCo输出一致。',
        '- main/manifest.json、main/source_snapshot.py：预先固定的seed、控制项和实际运行源码。',
        '- main/episodes.json、main/episodes.csv、main/episodes.jsonl：全部900次逐回合数据、公共成功判定和失败分类。',
        '- main/summary.json、main/verification.json：逐场景统计、配对变化、完整性与场景一致性核验。',
        '- main/traces/*.npz：10 Hz诊断轨迹，含完整qpos、控制量、末端与目标位置、倾角和阶段；models/*.mjb为实际编译模型。',
        '- tracking_and_grasp_diagnostics.json：阶段跟踪、双侧候选、确认抓取、滑脱和执行器限幅诊断。',
        '- success_rates.png / success_rates.svg：成功率汇总图；orientation_increment_comparison.json：姿态反馈的增量配对比较。', '',
        '```powershell', "$env:PYTHONPATH='src'", "$env:OMP_NUM_THREADS='1'", "$env:OPENBLAS_NUM_THREADS='1'", "$env:MKL_NUM_THREADS='1'",
        'conda run --no-capture-output -n mujoco-dlo python tools/evaluation/compare_scripted_tracking_fix.py --stage main --episodes 50 --workers 8',
        'conda run --no-capture-output -n mujoco-dlo python tools/evaluation/report_scripted_tracking_fix.py', '```','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    verification.update(production_source_hashes_unchanged=True,all_compiled_models_and_traces_present=True,
        all_successes_have_original_080s_snapshot=True,environment_configs_equal_within_pairs=True,
        all_compiled_model_actuator_hashes_match=True,all_traces_decodable_finite_and_time_ordered=True)
    (folder/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(summary['overall'],indent=2))


if __name__=='__main__':
    main()
