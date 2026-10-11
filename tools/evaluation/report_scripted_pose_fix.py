"""Report the completed Pose IK supplement and its frozen paired references."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'outputs/benchmarks/nero_scripted_tracking_fix_20261001'
OUT=ROOT/'outputs/benchmarks/nero_pose_scripted_tracking_fix_20261001'
POSE='pose_ik_tracking_fix'
VARIANTS=('baseline','tracking_fix','tracking_fix_orientation4',POSE)
LABELS=('Jacobian 原设置','Jacobian＋执行层修正','Jacobian＋执行层修正＋反馈×4','Pose＋执行层修正')
SCENES={'id_static':'静态','id_shape_nominal_current':'形变',
    'id_rigid_l1_nominal':'整体运动 L1','id_rigid_l2_nominal':'整体运动 L2',
    'id_combined_l1_nominal':'组合运动 L1','id_combined_l2_nominal':'组合运动 L2'}


def main():
    folder=OUT/'main'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    rows=json.loads((folder/'episodes.json').read_text(encoding='utf-8'))
    old=json.loads((OLD/'main/episodes.json').read_text(encoding='utf-8'))
    override_path=folder/'matched_reference_overrides.json'
    overrides=json.loads(override_path.read_text(encoding='utf-8')) if override_path.exists() else []
    if overrides:
        byref={(r['scenario_name'],r['requested_seed'],r['variant']):r for r in old}
        for r in overrides:byref[r['scenario_name'],r['requested_seed'],r['variant']]=r
        old=list(byref.values())
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    verification=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
    assert len(rows)==300 and len(old)==900
    assert len({(r['scenario_name'],r['requested_seed']) for r in rows})==300
    for p,h in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    assert hashlib.sha256((OLD/'main/episodes.json').read_bytes()).hexdigest()==manifest['reused_reference_episodes_sha256']
    assert hashlib.sha256((OLD/'main/manifest.json').read_bytes()).hexdigest()==manifest['reused_reference_manifest_sha256']
    assert (folder/'source_snapshot.py').read_bytes()==(ROOT/'tools/evaluation/compare_scripted_pose_fix.py').read_bytes()
    from compare_scripted_tracking_fix import digest_arrays
    model_hashes={};physics_hashes={};torque_bounds={}
    for p in {r['compiled_model'] for r in rows+overrides}:
        model=mujoco.MjModel.from_binary_path(str(OUT/p))
        h=hashlib.sha256()
        for attr in ('actuator_gainprm','actuator_biasprm'):
            h.update(attr.encode());h.update(np.asarray(getattr(model,attr)).tobytes())
        h.update(model.opt.gravity.tobytes());h.update(str(model.opt.timestep).encode())
        model_hashes[p]=h.hexdigest()
        physics_hashes[p]=digest_arrays(model,(
            'body_mass','body_inertia','body_pos','body_quat','body_gravcomp',
            'dof_damping','dof_armature','dof_frictionloss','jnt_range','jnt_axis','jnt_pos',
            'geom_size','geom_friction','geom_solref','geom_solimp','geom_contype','geom_conaffinity',
            'actuator_ctrlrange','actuator_forcerange','jnt_actfrcrange','eq_data'))
        torque_bounds[p]=np.max(np.abs(model.actuator_forcerange[:7]),axis=1)
    for r in rows+overrides:
        assert model_hashes[r['compiled_model']]==r['actuator_hash']
        assert physics_hashes[r['compiled_model']]==r['physical_model_hash']
        assert np.all(np.asarray(r['total_torque_max_nm'])<=torque_bounds[r['compiled_model']]+1e-9)
        if r['variant']==POSE:
            assert r['pose_ik_calls']>0 and r['ik_solver']=='DynamicVLATaskSpaceAdapter._pose_ik_target'
            assert not r['angular_feedback_x4_applied']
        if r['task_success']:
            assert r['success_snapshot']['success_hold']>=.8-1e-12
        with np.load(OUT/r['diagnostic_trace']) as trace:
            states=trace['states']
            assert states.ndim==2 and states.shape[0]>0 and states.shape[1]==int(trace['nq'])+17
            assert np.all(np.isfinite(states)) and np.all(np.diff(states[:,0])>0)
    allrows=old+rows
    diagnostics={}
    for v in VARIANTS:
        rr=[r for r in allrows if r['variant']==v]
        limits=np.asarray(rr[0]['env_config']['arm_joint_velocity_limits'])
        diagnostics[v]={'confirmed_grasp_episodes':sum(r['ever_confirmed_grasp'] for r in rr),
            'physical_slip_failure_episodes':sum(r['task_failure_type']=='physical_slip_after_confirmed_grasp' for r in rr),
            'episodes_with_any_confirmed_slip':sum(r['physical_slip_after_confirmed_count']>0 for r in rr),
            'mean_episode_actual_velocity_exceedance_ratio':float(np.mean([r['actual_joint_velocity_exceedance_ratio'] for r in rr])),
            'worst_peak_actual_velocity_to_limit':float(np.max([np.asarray(json.loads(r['max_abs_actual_arm_velocity']))/limits for r in rr])),
            'mean_pose_solver_calls_per_episode':float(np.mean([r.get('pose_ik_calls',0) for r in rr]))}
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    lines=['# NERO 规则方法：Pose IK 补测与 Jacobian 对照','',
        f"新增300回合Pose IK真实物理抓取，并新增{len(overrides)}回合151同平台Jacobian修正版对照；其余对照复用原本机冻结结果。所有主要比较均逐case匹配执行平台、初始场景、运动、seed、物理模型和对应执行器增益。", '',
        '**本次Pose配置：项目原有20次迭代Pose IK＋500 Hz重力前馈＋臂kp×4、kv×2；原力矩限制、物理参数和任务成功判定保留。Pose姿态权重为1.0，没有把Jacobian姿态反馈×4改名成Pose权重×4。**','',
        '| 场景 | Jacobian 原设置 | Jacobian＋执行层修正 | Jacobian＋修正＋反馈×4 | Pose＋执行层修正 |',
        '|---|---:|---:|---:|---:|']
    for name,label in list(SCENES.items())+[('overall','**合计**')]:
        g=summary[name]['variants']
        lines.append('| '+label+' | '+' | '.join(f"{g[v]['task_successes']}/{g[v]['episodes']}（{100*g[v]['task_success_rate']:.1f}%）" for v in VARIANTS)+' |')
    lines+=['','## 同执行层的配对比较','',
        '| Pose 相对参考配置 | 新增成功 | 退化 | 成功率差/百分点 | 95%配对区组区间 |',
        '|---|---:|---:|---:|---|']
    for v,label in zip(VARIANTS[1:3],LABELS[1:3]):
        c=summary['overall']['pose_vs_reference'][v]
        lo,hi=c['paired_bootstrap95']
        lines.append(f"| {label} | {c['improved']} | {c['worsened']} | {100*c['delta_rate']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines+=['',
        '本次Pose配置的总体成功率为64.0%，没有超过同执行层Jacobian的66.0%；二者差值为−2.00个百分点，配对区间跨0，不能断言二者有稳定差异。相对Jacobian＋修正＋反馈×4的70.0%，Pose低6.00个百分点，95%配对区间为[−12.00, −0.33]。这组完整闭环数据不支持把当前Pose接入配置作为提高成功率的默认方案；不能据此概括所有Pose IK实现。', '',
        '分场景看，Pose在形变场景达到50.0%，但整体运动L1/L2只有82.0%/88.0%，对应Jacobian＋修正为98.0%/100.0%。这些是观测到的回归，尚未通过额外消融确定具体原因。', '',
        '总体区间以相同seed的六场景为一个区组重采样20,000次。每场景50回合，属于工程验证；不替换正式论文冻结结果，也不替代新论文实验的100回合/场景要求。', '',
        'Windows与151的12回合先导成败一致，但部分轨迹数值不一致，因此151上的每个Pose回合同平台补跑两种Jacobian修正版。表中的修正版对照使用这些配对数据，可能与原冻结的66.0%/70.3%不同。原本机900回合文件及其结果保持不变；原Jacobian未修正50.0%仅作背景，不用于跨平台的配对因果结论。', '',
        'Pose直接求解状态机给出的绝对位置和原目标四元数；旧Jacobian先生成限幅任务空间速度再形成短时关节目标。共同环境的关节/夹爪运动限制保留。比较的是这两条项目控制路径的完整闭环效果，并非剥离命令生成方式的纯数学求解器比较。', '',
        '本次没有新增“原执行层下的Pose IK”组，因此不能根据它单独估计执行层修正对Pose IK的因果收益。原Jacobian未修正组仅作背景对照；同执行层的Jacobian＋修正组是主要参考。', '',
        '## 抓取与速度诊断','',
        '| 指标 | Jacobian 原设置 | Jacobian＋修正 | Jacobian＋修正＋反馈×4 | Pose＋修正 |',
        '|---|---:|---:|---:|---:|']
    for key,label in [('confirmed_grasp_episodes','形成确认抓取回合'),
        ('physical_slip_failure_episodes','终局物理滑脱失败'),
        ('episodes_with_any_confirmed_slip','确认后曾滑脱回合')]:
        lines.append('| '+label+' | '+' | '.join(str(diagnostics[v][key]) for v in VARIANTS)+' |')
    lines.append('| 回合平均超速物理子步占比 | '+' | '.join(f"{100*diagnostics[v]['mean_episode_actual_velocity_exceedance_ratio']:.3f}%" for v in VARIANTS)+' |')
    lines.append('| 最坏峰值速度/关节限制 | '+' | '.join(f"{diagnostics[v]['worst_peak_actual_velocity_to_limit']:.3f}×" for v in VARIANTS)+' |')
    lines+=['', '曾滑脱包括后来重试成功的回合，不能与终局滑脱失败混同。成功率仍使用原task_success：真实确认夹持、抬升等条件持续0.80秒；实际超速单独报告，没有隐藏或修改它的定义。', '',
        '## 原始产物','',
        '- main/manifest.json / source_snapshot.py：冻结配置、300个新回合的种子和原对照文件哈希。',
        '- main/episodes.json / episodes.csv / merged_episodes.jsonl：合并后的300个新增Pose回合，逐条记录真实Pose解算器调用次数。',
        '- main/pre_handoff_episodes.jsonl、shards/local、server_export/shards/server：各批原始回合记录；main/episodes.jsonl保留切分前的163回合日志。',
        '- main/summary.json / verification.json：四配置的配对统计及一致性核验。',
        '- main/traces、shards/local/traces、server_export/shards/server/traces及对应models目录：各批物理轨迹及编译模型，已逐项检查可读取性。',
        '- diagnostics.json / success_rates.png：抓取与速度诊断及分场景结果图。',
        '- 原本机对照保留于相邻nero_scripted_tracking_fix_20261001目录，文件哈希校验通过。',
        '- execution_plan.json / server_export：分片清单、151平台复现记录、同平台对照及运行版本；main/matched_reference_overrides.json记录新增对照，不覆盖原冻结结果。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    verification.update(all_new_traces_readable_finite=True,all_saved_actuator_hashes_match=True,
        all_saved_physical_model_hashes_match=True,all_recorded_combined_torques_within_original_limits=True,
        all_source_and_reused_artifact_hashes_match=True)
    (folder/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    plot(summary)
    print(json.dumps(summary['overall'],indent=2))


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    names=list(SCENES)+['overall'];x=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(14,5.7),layout='constrained')
    colors=('#7c8799','#19957b','#4678bd','#c37b30')
    for i,(v,label,color) in enumerate(zip(VARIANTS,LABELS,colors)):
        rates=[100*summary[n]['variants'][v]['task_success_rate'] for n in names]
        bars=ax.bar(x+(i-1.5)*.21,rates,.195,label=label,color=color)
        ax.bar_label(bars,labels=[f'{r:.0f}%' if n!='overall' else f'{r:.1f}%' for n,r in zip(names,rates)],fontsize=8,padding=3)
    ax.set(xticks=x,xticklabels=list(SCENES.values())+['合计'],ylim=(0,110),ylabel='原任务成功率 / %')
    ax.set_yticks(np.arange(0,101,20));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_title('NERO 规则方法：Pose IK 补测与冻结 Jacobian 对照',pad=54)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
    fig.supxlabel('每配置每场景50回合；两种修正版逐case同平台配对；原Jacobian设置仅作背景。',fontsize=10)
    fig.savefig(OUT/'success_rates.png',dpi=180);fig.savefig(OUT/'success_rates.svg')
    plt.close(fig)


if __name__=='__main__':
    main()
