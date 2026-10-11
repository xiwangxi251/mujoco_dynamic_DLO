"""Verify the enhanced-full6 solver experiment; maps, traces, synced videos."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from experiment_bounded_full6 import OUT,GOALS


def read(variant,name):return json.loads((OUT/variant/name).read_text())


def report():
    variants=('baseline','bounded')
    records={v:read(v,'cells.json') for v in variants}
    summaries={v:read(v,'summary.json') for v in variants}
    manifests={v:read(v,'manifest.json') for v in variants}
    arrays={}
    for variant in variants:
        with np.load(OUT/variant/'workspace.npz') as saved:arrays[variant]={k:saved[k].copy() for k in saved.files}
    assert manifests['baseline']['reference_speed_ramp_enabled'] is False
    assert manifests['bounded']['actual_arrival_gate_enabled'] is False
    for key in ('grid_x_m','grid_y_m','tcp_z_m','control_dt_s','physics_dt_s','position_tolerance_mm',
                'pose_tolerance_deg','hold_s','motion_path_pose_limit_deg','motion_peak_velocity_ratio_limit',
                'finger_floor_mm','robot_floor_min_mm','maximum_trial_s','waypoint_transition'):
        assert manifests['baseline'][key]==manifests['bounded'][key],key
    for key in ('DLS_lambda','position_gain','orientation_gain','uniform_cartesian_linear_cap_m_s',
                'ik_target_horizon_s','arm_acceleration_limit_enabled','gravity_feedforward_enabled',
                'env_cartesian_cap_enabled','arm_kp_scale','arm_kv_scale'):
        assert manifests['baseline']['controller'][key]==manifests['bounded']['controller'][key],key
    expected={(float(x),float(y)) for x in arrays['baseline']['xs'] for y in arrays['baseline']['ys']}
    checks={}
    for variant in variants:
        data=arrays[variant];rows=records[variant];summary=summaries[variant]
        assert len(rows)==len({(r['x_m'],r['y_m']) for r in rows})==961
        assert {(r['x_m'],r['y_m']) for r in rows}==expected
        for path,digest in manifests[variant]['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        for key in ('endpoint_pass','motion_pass','interior'):
            assert int(data[key].sum())==summary[key]['cells']
            assert abs(summary[key]['sampled_area_m2']-data[key].sum()*.0016)<1e-12
        for r in rows:
            assert r['motion_pass']==(r['endpoint_pass'] and r['max_path_pose_deg']<=5 and r['peak_velocity_to_limit']<=1.01)
            assert r['endpoint_pass']==(r['reason']=='endpoint_held_2s')
            assert r['steps']<=800 and abs(r['sim_seconds']-.02*r['steps'])<1e-8
            assert np.isfinite([r['position_error_mm'],r['pose_error_deg'],r['peak_velocity_to_limit']]).all()
            if r['endpoint_pass']:
                assert r['position_error_mm']<=5 and r['pose_error_deg']<=2
                assert .2<=r['finger_floor_mm']<=8 and r['robot_floor_mm']>=.2
                assert r['joint_margin_deg']>=0 and r['self_distance_mm']>=1
            if variant=='bounded':
                assert r['joint_target_clip_steps']==0 and r['max_joint_target_clip_rad']<1e-9
                assert r['solver_failures']==0 and r['max_solver_bound_violation_rad_s']<1e-9
            iy=np.argmin(abs(data['ys']-r['y_m']));ix=np.argmin(abs(data['xs']-r['x_m']))
            for key in ('endpoint_pass','motion_pass'):assert bool(data[key][iy,ix])==r[key]
        assert np.array_equal(data['interior'],data['motion_pass']&(ndimage.distance_transform_edt(np.pad(data['motion_pass'],1))[1:-1,1:-1]*.04>=.06))
        endpoint=[r for r in rows if r['endpoint_pass']]
        checks[variant]=dict(unique_cells=961,source_hashes_match=True,success_criteria_verified=True,
            joint_target_clip_steps=sum(r['joint_target_clip_steps'] for r in rows),
            trials_with_target_clipping=sum(r['joint_target_clip_steps']>0 for r in rows),
            solver_failures=sum(r['solver_failures'] for r in rows),
            actual_speed_over_1_percent_trials=sum(r['peak_velocity_to_limit']>1.01 for r in rows),
            final_actual_joint_position_outside_trials=sum(r['joint_margin_deg']<0 for r in rows),
            final_position_pose_ok_but_joint_outside=sum(r['position_error_mm']<=5 and r['pose_error_deg']<=2 and r['joint_margin_deg']<0 for r in rows),
            endpoint_path_pose_over_5deg=sum(r['max_path_pose_deg']>5 for r in endpoint),
            endpoint_actual_speed_over_1_percent=sum(r['peak_velocity_to_limit']>1.01 for r in endpoint),
            trials_with_solver_bounds_active=sum(r['solver_bound_active_steps']>0 for r in rows),
            solver_median_of_trial_medians_ms=float(np.median([r['solver_median_ms'] for r in rows])) if variant=='bounded' else None,
            solver_max_observed_ms=max((r['solver_max_ms'] or 0) for r in rows))
    old=arrays['baseline'];new=arrays['bounded']
    transition={key:dict(gained=int((new[key]&~old[key]).sum()),lost=int((old[key]&~new[key]).sum()),
        kept=int((old[key]&new[key]).sum())) for key in ('endpoint_pass','motion_pass')}
    pilots=json.loads((OUT/'pilot_results.json').read_text())
    selected=[]
    for x,y,name in GOALS:
        paired={v:next(r for r in records[v] if abs(r['x_m']-x)<1e-8 and abs(r['y_m']-y)<1e-8) for v in variants}
        for variant,row in paired.items():
            pilot=next(r for r in pilots if r['variant']==variant and r['goal_name']==name)
            assert row['reason']==pilot['reason'] and row['steps']==pilot['steps']
            for key in ('position_error_mm','pose_error_deg','max_path_pose_deg'):assert abs(row[key]-pilot[key])<1e-9
        selected.append(dict(name=name,goal_m=[x,y,.005],**paired))
    result=dict(summaries=summaries,verification=checks,transitions=transition,selected_goals=selected,
        same_reference_protocol=True,original_limits_and_servo_preserved=True,
        previous_actual_arrival_gate_removed=True)
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    xs,ys=old['xs'],old['ys'];extent=(xs[0]-.02,xs[-1]+.02,ys[0]-.02,ys[-1]+.02)
    fig,axes=plt.subplots(1,3,figsize=(18,6.5),layout='constrained')
    colors=['#e8edf2','#87bce8','#138569']
    for ax,variant,label in ((axes[0],'baseline','Enhanced 6 / original DLS'),(axes[1],'bounded','Enhanced 6 / bounded DLS')):
        d=arrays[variant];s=summaries[variant]
        ax.imshow(d['endpoint_pass'].astype(int)+d['motion_pass'].astype(int),origin='lower',extent=extent,
            cmap=ListedColormap(colors),norm=BoundaryNorm([-.5,.5,1.5,2.5],3),interpolation='nearest')
        ax.set_title(label)
        ax.text(.02,.02,f"Endpoint {s['endpoint_pass']['cells']}/961 | Motion {s['motion_pass']['cells']}/961\nEndpoint area ~{s['endpoint_pass']['sampled_area_m2']:.4f} m²\nMotion area ~{s['motion_pass']['sampled_area_m2']:.4f} m²",
            transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.94,edgecolor='none'))
    diff=new['endpoint_pass'].astype(int)-old['endpoint_pass'].astype(int)
    axes[2].imshow(diff,origin='lower',extent=extent,cmap=ListedColormap(['#d95660','#e8edf2','#4a9fc9']),
        norm=BoundaryNorm([-1.5,-.5,.5,1.5],3),interpolation='nearest')
    axes[2].set_title('Endpoint pass change: blue gained / red lost')
    axes[2].text(.02,.02,f"Gained {transition['endpoint_pass']['gained']} | Lost {transition['endpoint_pass']['lost']}",
        transform=axes[2].transAxes,bbox=dict(facecolor='white',alpha=.94,edgecolor='none'))
    for ax in axes:
        ax.set_aspect('equal');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.grid(alpha=.15)
        ax.plot(.35,0,'o',color='#223b53',markersize=4)
    fig.legend([Patch(color=c) for c in colors],['Not passed','Endpoint only','Motion passed'],
        loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Enhanced full6 only | same direct reference sequence | no actual-arrival gate | TCP z=5 mm / 4 cm grid')
    fig.savefig(OUT/'workspace_comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
    # Full physical histories of the two negative-Y examples, collected by
    # the pilot before scanning. No archived trajectory from the gated test.
    import mujoco
    from experiment_bounded_full6 import ComparisonRig,base
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for row_id,(x,y,name) in enumerate((GOALS[0],GOALS[2])):
        for variant,color in (('baseline','#cb575e'),('bounded','#168b71')):
            with np.load(OUT/f'pilot_{variant}_{name}.npz') as saved:qpos=saved['qpos']
            rig=ComparisonRig(variant);position=[];pose=[]
            for q in qpos:
                rig.data.qpos[:9]=q;mujoco.mj_forward(rig.model,rig.data)
                position.append(np.linalg.norm(rig.env.hand_position-np.array([x,y,.005]))*1000)
                error=base.native.quat_error(base.native.rotation_to_quat(rig.data.xmat[rig.env.hand_id].reshape(3,3)),rig.quat)
                pose.append(np.degrees(2*np.arcsin(np.clip(.5*np.linalg.norm(error),0,1))))
            times=.02*np.arange(1,len(qpos)+1)
            for ax,values in zip(axes[row_id],(position,pose,qpos[:,5])):
                ax.plot(times,values,label=variant,color=color,linewidth=1.8)
            assert abs(position[-1]-next(p['position_error_mm'] for p in pilots if p['variant']==variant and p['goal_name']==name))<1e-8
        axes[row_id,0].set_ylabel(f'Goal ({x:.2f}, {y:+.2f})\nTCP error (mm)')
        axes[row_id,0].axhline(5,color='#617487',linestyle=':',linewidth=1)
        axes[row_id,1].axhline(5,color='#617487',linestyle=':',linewidth=1)
        axes[row_id,2].axhline(-.73,color='#617487',linestyle=':',linewidth=1,label='q6 lower bound')
        for ax in axes[row_id]:ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax,title in zip(axes[0],('Position error to final target','Full-pose error (deg)','Joint 6 position (rad)')):ax.set_title(title)
    fig.suptitle('Enhanced6: original DLS vs bounded DLS | identical reference schedule')
    fig.savefig(OUT/'negative_y_traces.png',dpi=170,bbox_inches='tight');plt.close(fig)
    text='''# 增强版本6：原DLS与带约束DLS对照

只比较增强6：kp×4、kv×2、500 Hz有界重力前馈、姿态反馈增益10。完整6D任务同步求解，阻尼0.04、前瞻0.11 s，保留原有末端和关节限速及实际速度保护。

两组都删除测试额外0.15/0.03 m/s参考速度斜坡及误加的实际到达5 mm门限。参考序列：第一帧上方航点(x,y,0.10)，第二帧起最终目标(x,y,0.005)。上方航点仅占20 ms，不声称实际机械臂先到上方再下降。工作范围和视频都使用这同一个协议。历史含实际到达门限的159/21结果不作为本次基线。

带约束DLS求解：min ||J*v-V||²+lambda²||v-v_null||²；位置约束q_min<=q+0.11*v<=q_max；速度约束使用原有关节速度边界。v_null是原有完整任务精确零空间内ready回拉。所有偏好一起在约束内求解，位置clip只修正浮点误差。

每组961点，4 cm网格、固定朝向、同一home、最大16 s。终点阈值位置5 mm、姿态2°且离桌/关节/自碰撞条件合格连续2 s；全程通过再要求路径最大姿态≤5°、实际关节峰值速度≤原限值1.01倍。指令约束不保证实际物理关节速度和位置完全合规。

| 指标 | 原DLS | 带约束DLS |
|---|---:|---:|
'''
    for key,label in (('endpoint_pass','终点通过点数'),('motion_pass','全程通过点数')):
        text+=f"| {label} | {summaries['baseline'][key]['cells']} | {summaries['bounded'][key]['cells']} |\n"
        text+=f"| {label}对应网格面积m² | {summaries['baseline'][key]['sampled_area_m2']:.4f} | {summaries['bounded'][key]['sampled_area_m2']:.4f} |\n"
    text+='\n外包围坐标内存在失败点，面积为网格估计。求解时间是在并行扫描/本机环境下测量，不是硬实时保证。原始单元记录、配置哈希、实施核对、逐点增减以及物理速度违规均保存在本目录。\n'
    text+='\n## 本次结果及限制\n\n'
    text+=f"终点新增{transition['endpoint_pass']['gained']}点、失去{transition['endpoint_pass']['lost']}点。负Y终点通过："+str(sum(r['endpoint_pass'] and r['y_m']<0 for r in records['baseline']))+' → '+str(sum(r['endpoint_pass'] and r['y_m']<0 for r in records['bounded']))+'点。\n\n'
    for reason,label in (('floor_penetration','穿地停止'),('timeout','超时')):
        text+=f"{label}：{summaries['baseline']['counts'].get(reason,0)} → {summaries['bounded']['counts'].get(reason,0)} 次。\n\n"
    text+='两组全部终点通过的轨迹都曾超过5°全程姿态阈值，所以本次全程通过均为0；去掉测试额外参考限速后输入直接跳到目标，与历史缓慢参考轨迹的范围不能直接对比。\n\n'
    text+=f"实际速度超原限值1%以上的试验：{checks['baseline']['actual_speed_over_1_percent_trials']} → {checks['bounded']['actual_speed_over_1_percent_trials']}；终点通过样本中：{checks['baseline']['endpoint_actual_speed_over_1_percent']} → {checks['bounded']['endpoint_actual_speed_over_1_percent']}。关节指令合规不代表实际速度、实际关节位置合规。\n\n"
    text+='外侧负Y点(0.67,-0.28)：原DLS在16秒超时、最终位置误差12.67mm；带约束DLS在2.70秒完成终点保持、最终误差0.654mm，路径姿态峰值仍为6.39°。丢失点(0.67,-0.24)的新解终点位置/姿态已合格，但实际关节略越界，因此仍判超时。\n\n'
    text+='这是隔离的增强6实验求解器，尚未替换生产策略默认实现；求解时纳入限位后消除了目标截断，但仍需处理实际伺服动态和路径姿态偏差。视频均为重新运行物理仿真的原生1280×960/50fps录像，逐帧关节状态与未渲染重跑差异为0。\n'
    (OUT/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(verification=checks,transitions=transition),indent=2),flush=True)


def gallery():
    from build_retest_136_gallery import HTML
    dest=OUT/'videos';episodes=[];checks=[]
    for variant,ident in (('baseline',1),('bounded',2)):
        for x,y,name in GOALS:
            directory=dest/variant/'version_6'/name
            metadata=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
            metadata.update(directory=str(directory.relative_to(dest)).replace('\\','/'),comparison_id=ident)
            assert metadata['version']==6 and metadata['previous_saved_joint_state_max_difference']==0
            episodes.append(metadata)
            for view in ('global','wrist'):
                file=directory/f'{view}.mp4'
                info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames',
                    '-show_entries','stream=width,height,avg_frame_rate,nb_frames,codec_name:frame=key_frame,best_effort_timestamp_time,pict_type',
                    '-of','json',str(file)],text=True))
                s=info['streams'][0];frames=info['frames']
                assert (s['width'],s['height'],s['avg_frame_rate'],s['codec_name'])==(1280,960,'50/1','h264')
                assert len(frames)==int(s['nb_frames'])==metadata['frames']
                assert max(abs(float(f['best_effort_timestamp_time'])-i/50) for i,f in enumerate(frames))<1e-6
                keys=[i for i,f in enumerate(frames) if f['key_frame']]
                gop=max([b-a for a,b in zip(keys,keys[1:])]+[len(frames)-keys[-1]])
                assert gop<=12 and all(f['pict_type']!='B' for f in frames)
                checks.append(dict(file=str(file.relative_to(dest)).replace('\\','/'),frames=len(frames),fps=50,
                    resolution=[1280,960],max_gop=gop,trajectory_difference=0))
    html=HTML.replace('版本 1、3、6 · 工作范围实际录像','增强6 · 原DLS / 带约束DLS')
    html=html.replace('相同 home、目标、末端朝向和测试轨迹。',
        '只比较增强6，固定同一home、朝向、kp×4/kv×2、重力前馈及姿态增益10。保留原有限速。两组参考序列相同：上方航点发送一帧，随后直接发送终点，无额外参考限速和实际到达门限。')
    options=''.join(f'<option value="{name}">{label}：({x:.2f}, {y:+.2f}, 0.005) m</option>'
        for (x,y,name),label in zip(GOALS,('负Y','正Y','外侧负Y','外侧正Y','中心','边缘失败点')))
    html=re.sub(r'<select id="goal">.*?</select>',f'<select id="goal">{options}</select>',html,count=1)
    html=html.replace('ids=[1,3,6]','ids=[1,2]').replace('e.version===id','e.comparison_id===id')
    html=html.replace('repeat(3,minmax(0,1fr))','repeat(2,minmax(0,1fr))')
    html=html.replace('三路同步进度','两种求解器同步进度').replace('暂停三路','暂停两路')
    # Chromium may keep the poster over a paused video after its first seek.
    # Once a real decoded frame exists, show that frame instead of the poster.
    html=html.replace("vs().forEach(v=>{v.addEventListener('seeked'", "vs().forEach(v=>{v.addEventListener('loadeddata',()=>v.removeAttribute('poster'));v.addEventListener('seeked'")
    html=html.replace('°${floor}',
        '°<br>全程姿态峰值 ${e.result.max_path_pose_deg.toFixed(2)}° · 实际速度峰值/限值 ${e.result.peak_velocity_to_limit.toFixed(3)}${floor}<br>JIK目标截断 ${e.result.joint_target_clip_steps} 步')
    note='每张卡片为结束统计，当前帧指标见录像HUD。两组增强6仅求解器不同；参考目标从第二帧起直接给终点，无实际到达门限。终点通过需位置≤5 mm、姿态≤2°及离桌/关节/自碰撞合规保持2秒；全程通过还需姿态峰值≤5°、实际关节速度峰值≤限值1.01倍。'
    html=re.sub(r"\$\('note'\)\.textContent=goal\.includes\('floor'\).*?;\$\('seek'\)\.value=0",f"$('note').textContent={json.dumps(note,ensure_ascii=False)};$('seek').value=0",html,count=1)
    html=re.sub(r'<div class="note" id="note">.*?</div>',f'<div class="note" id="note">{note}</div>',html,count=1)
    html=re.sub(r'<footer>.*?</footer>',
        '<footer><a href="../workspace_comparison.png" target="_blank">新旧求解器范围图</a><a href="../negative_y_traces.png" target="_blank">负Y轨迹曲线</a><a href="../comparison.json" target="_blank">完整结果</a><a href="../README.md" target="_blank">测试协议</a><a href="video_verification.json" target="_blank">视频核查</a></footer>',html,count=1)
    players=''.join(f'<article class="player"><div class="head"><strong>{label}</strong><span id="badge-{ident}" class="badge"></span></div><video id="video-{ident}" controls playsinline preload="auto" aria-label="{label}录像"></video><div class="meta" id="meta-{ident}"></div></article>' for ident,label in ((1,'增强6 · 原DLS'),(2,'增强6 · 带约束DLS')))
    (dest/'index.html').write_text(html.replace('__PLAYERS__',players).replace('__DATA__',json.dumps(episodes,ensure_ascii=False)),encoding='utf-8')
    (dest/'gallery_manifest.json').write_text(json.dumps(episodes,indent=2,ensure_ascii=False),encoding='utf-8')
    (dest/'video_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(dict(clips=len(checks),all_checks_passed=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gallery',action='store_true');args=p.parse_args()
    gallery() if args.gallery else report()
