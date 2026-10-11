"""Report completed paired learned-policy runs, without replacing paper results."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from learned_tracking_common import SCENES,VARIANTS,dump,summarize_pairs

def seed_block_interval(rows):
    by={(r['scenario_name'],r['requested_seed'],r['variant']):r for r in rows}
    seeds=sorted({r['requested_seed'] for r in rows})
    differences=np.array([[int(by[s,seed,'tracking_fix']['task_success'])-int(by[s,seed,'baseline']['task_success']) for s in SCENES] for seed in seeds])
    rng=np.random.default_rng(20261001)
    # Reuse of one seed across all four scenes is retained as one bootstrap block.
    samples=differences[rng.integers(0,len(seeds),(10000,len(seeds)))].mean(axis=(1,2))
    return (np.quantile(samples,[.025,.975])*100).tolist()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ppo',type=Path,required=True);parser.add_argument('--dvla',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    data={};rows_by={}
    methods=[('State PPO',args.ppo)]+([('DVLA',args.dvla)] if args.dvla else [])
    for method,path in methods:
        rows=json.loads((path/'episodes.json').read_text())
        assert len(rows)==400 and len({(r['scenario_name'],r['requested_seed'],r['variant']) for r in rows})==400
        assert all(sum(r['scenario_name']==s and r['variant']==v for r in rows)==50 for s in SCENES for v in VARIANTS)
        result=summarize_pairs(rows)
        result['overall']['seed_block_bootstrap95_percentage_points']=seed_block_interval(rows)
        result['overall']['primary_interval_method']='bootstrap paired seed blocks, retaining all four scenes per seed'
        result['failures']={v:dict(Counter(r['task_failure_type'] for r in rows if r['variant']==v and not r['task_success'])) for v in VARIANTS}
        result['joint_tracking_error_mean_rad']={v:float(np.mean([r['joint_tracking_error_mean_rad'] for r in rows if r['variant']==v])) for v in VARIANTS}
        result['execution_hosts']=sorted({r.get('execution_host','node151') for r in rows})
        result['actual_joint_speed_exceedance_cases']={}
        constrained=[]
        for v in VARIANTS:
            count=0
            for r in rows:
                if r['variant']!=v:continue
                vel=r.get('max_abs_actual_arm_velocity',[]);limit=r.get('arm_joint_velocity_limits',[])
                if isinstance(vel,str):vel=json.loads(vel)
                if isinstance(limit,str):limit=json.loads(limit)
                assert len(vel)==7 and len(limit)==7,'Actual joint speed audit is required'
                exceeded=bool(np.any(np.array(vel)>np.array(limit)+1e-8))
                count+=int(exceeded)
                constrained.append({**r,'task_success':bool(r['task_success'] and not exceeded)})
            result['actual_joint_speed_exceedance_cases'][v]=count
        result['task_success_and_no_actual_joint_speed_exceedance']=summarize_pairs(constrained)
        result['task_success_and_no_actual_joint_speed_exceedance']['overall']['seed_block_bootstrap95_percentage_points']=seed_block_interval(constrained)
        data[method]=result;rows_by[method]=rows
    dump(args.output/'comparison.json',data)
    lines=['# DVLA 与状态 PPO：执行层修正配对实验','',
        f'每个方法固定同一学习权重，4 个核心场景 × 50 个配对种子。原设置与改进设置各 200 回合。本报告已完成 {len(data)*400} 回合。',
        '改进仅为机器人重力前馈（500 Hz）、臂 kp×4 / kv×2；原学习动作解码和 IK 保留，没有加姿态反馈×4，没有缩小工作区，也没有重新训练。',
        '使用环境公共 task_success。新工程实验不替换论文冻结结果。','',
        *(['DVLA 尚未运行，不计为失败，也没有成功率结论。',''] if not args.dvla else []),
        '| 方法 | 原设置 | 改进设置 | 差值（百分点） | 配对95%区间 | 获益 / 退步 |',
        '|---|---:|---:|---:|---|---:|']
    for method,result in data.items():
        o=result['overall'];lo,hi=o['seed_block_bootstrap95_percentage_points']
        lines.append(f"| {method} | {o['baseline_successes']}/200（{o['baseline_rate']:.1%}） | {o['corrected_successes']}/200（{o['corrected_rate']:.1%}） | {o['difference_percentage_points']:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {o['gained_cases']} / {o['lost_cases']} |")
    lines+=['','| 方法 / 场景 | 原设置 | 改进设置 |','|---|---:|---:|']
    for method,result in data.items():
        for scene,o in result['scenarios'].items():lines.append(f"| {method} / {scene} | {o['baseline_successes']}/50 | {o['corrected_successes']}/50 |")
    lines+=['','实际速度审计（原 configured joint velocity limits + 1e-8 数值容差，500 Hz 物理子步峰值）：',
        '| 方法 | 原设置超速回合 | 改进设置超速回合 | 原设置成功且不超速 | 改进设置成功且不超速 |',
        '|---|---:|---:|---:|---:|']
    for method,result in data.items():
        count=result['actual_joint_speed_exceedance_cases'];c=result['task_success_and_no_actual_joint_speed_exceedance']['overall']
        lines.append(f"| {method} | {count['baseline']}/200 | {count['tracking_fix']}/200 | {c['baseline_successes']}/200 | {c['corrected_successes']}/200 |")
    lines+=['','“成功且不超速”是额外约束诊断，没有替换公共 task_success，也没有删掉超速回合。若改进组超速更多，不能只凭公共成功率上升就宣布该配置满足真实机械臂的运动限制。']
    lines+=['','结论按配对差值区间判断；区间跨零时，不宣称有稳定提升。完整配对审计、失败分类、跟踪误差和实际关节超速事件见 comparison.json。',
        '总体主区间按 seed 区组 bootstrap：每次重采样保留同一个 seed 的四个场景，避免把它们误当成四个独立随机种子。原 case-level 配对区间也保留在 JSON 中。',
        '本实验只评估现有学习策略直接接入新执行层的表现，不能回答重新训练后的最优收益，也不能作为真机迁移验证。','']
    (args.output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,5),layout='constrained')
    x=np.arange(len(data)*2);width=.32
    for offset,v,label,color in ((-width/2,'baseline','Original execution','#708ba6'),(width/2,'corrected','Gravity FF + kp x4 / kv x2','#228b78')):
        values=[value for m in data for value in (data[m]['overall'][v+'_rate']*100,
            data[m]['task_success_and_no_actual_joint_speed_exceedance']['overall'][v+'_rate']*100)]
        bars=ax.bar(x+offset,values,width,label=label,color=color)
        ax.bar_label(bars,labels=[f'{value:.1f}%' for value in values],padding=3)
    ax.set_xticks(x,[label for m in data for label in (m+'\nPublic task success',m+'\nTask success + no joint overspeed')]);ax.set_ylabel('Rate (%)');ax.set_ylim(0,105)
    ax.set_title('Frozen learned policies: 200 paired cases each');ax.legend(loc='upper left')
    fig.savefig(args.output/'success_rates.png',dpi=180);plt.close(fig)
    print(json.dumps({m:r['overall'] for m,r in data.items()},indent=2))

if __name__=='__main__':main()
