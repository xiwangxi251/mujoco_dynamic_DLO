"""Verify and visualize direct-waypoint full6 scans and their comparison videos."""
import hashlib
import json
import re
from pathlib import Path
import argparse
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from matplotlib.patches import Patch
from scan_full6_no_extra_motion_limit import OUT


def report():
    manifest=json.loads((OUT/'manifest.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text())
    rows=json.loads((OUT/'cells.json').read_text())
    previous=json.loads((OUT.parent/'cells.json').read_text())
    with np.load(OUT/'workspace.npz') as saved:
        data={k:saved[k].copy() for k in saved.files}
    xs,ys=data['xs'],data['ys']
    indexed={(r['x_m'],r['y_m']):r for r in rows}
    assert len(rows)==len(indexed)==961
    assert set(indexed)=={(float(x),float(y)) for x in xs for y in ys}
    for path,digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    assert manifest['reference_speed_ramp_enabled'] is False
    assert manifest['approach_speed_m_s'] is None and manifest['descent_speed_m_s'] is None
    assert manifest['original_motion_limits_audit']['original_motion_limits_identical'] is True
    assert manifest['controller']['uniform_cartesian_linear_cap_m_s']==.9
    for r in rows:
        assert abs(r['sim_seconds']-.02*r['steps'])<1e-8 and r['steps']<=800
        assert r['motion_pass']==(r['endpoint_pass'] and r['max_path_pose_deg']<=5 and r['peak_velocity_to_limit']<=1.01)
        assert r['endpoint_pass']==(r['reason']=='endpoint_held_2s')
        if r['endpoint_pass']:
            assert r['position_error_mm']<=5 and r['pose_error_deg']<=2
            assert .2<=r['finger_floor_mm']<=8 and r['robot_floor_mm']>=.2
            assert r['joint_margin_deg']>=0 and r['self_distance_mm']>=1
            assert r['descent_started']
        assert np.isfinite([r['position_error_mm'],r['pose_error_deg'],r['peak_velocity_to_limit']]).all()
        iy=np.argmin(abs(ys-r['y_m']));ix=np.argmin(abs(xs-r['x_m']))
        for key in ('endpoint_pass','motion_pass'):assert bool(data[key][iy,ix])==r[key]
    for key in ('endpoint_pass','motion_pass','interior'):
        assert int(data[key].sum())==summary[key]['cells']
        assert abs(summary[key]['sampled_area_m2']-data[key].sum()*.0016)<1e-12
    assert np.array_equal(data['interior'],data['motion_pass'] & (ndimage.distance_transform_edt(np.pad(data['motion_pass'],1))[1:-1,1:-1]*.04>=.06))
    verification=dict(unique_cells=961,source_hashes_match=True,criteria_and_masks_consistent=True,
        original_motion_limits_identical=True,reference_speed_ramp_enabled=False,
        old_reference_speed_m_s=[.15,.03],native_cartesian_linear_cap_m_s=.9,
        actual_velocity_guard_substeps=sum(r['velocity_guard_substeps'] for r in rows))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    def grid(records,key):
        result=np.zeros((len(ys),len(xs)))
        for r in records:result[np.argmin(abs(ys-r['y_m'])),np.argmin(abs(xs-r['x_m']))]=r[key]
        return result
    extent=(xs[0]-.02,xs[-1]+.02,ys[0]-.02,ys[-1]+.02)
    fig,axes=plt.subplots(2,2,figsize=(12.5,12),layout='constrained')
    colors=['#e8edf2','#87bce8','#138569']
    for ax,records,label in ((axes[0,0],previous,'Previous: reference ramp 0.15 / 0.03 m/s'),
                              (axes[0,1],rows,'Updated: direct waypoints / native limits retained')):
        ax.imshow(grid(records,'endpoint_pass')+grid(records,'motion_pass'),origin='lower',extent=extent,
            cmap=ListedColormap(colors),norm=BoundaryNorm([-.5,.5,1.5,2.5],3),interpolation='nearest')
        endpoint=sum(r['endpoint_pass'] for r in records);motion=sum(r['motion_pass'] for r in records)
        ax.set_title(label,fontsize=11)
        ax.text(.02,.02,f'Endpoint {endpoint}/961 | Motion {motion}/961\nMotion area ~{motion*.0016:.4f} m²',
            transform=ax.transAxes,bbox=dict(facecolor='white',edgecolor='none',alpha=.94))
    im=axes[1,0].imshow(np.maximum(grid(rows,'position_error_mm'),.1),origin='lower',extent=extent,
        cmap='magma',norm=LogNorm(vmin=1,vmax=1000),interpolation='nearest')
    axes[1,0].set_title('Updated: TCP position error at trial end')
    fig.colorbar(im,ax=axes[1,0],shrink=.75,label='mm (log scale)')
    im=axes[1,1].imshow(grid(rows,'max_path_pose_deg'),origin='lower',extent=extent,
        cmap='viridis',vmin=0,vmax=180,interpolation='nearest')
    axes[1,1].set_title('Updated: maximum pose error over recorded path')
    fig.colorbar(im,ax=axes[1,1],shrink=.75,label='degrees')
    for ax in axes.ravel():
        ax.set_aspect('equal');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
        ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.grid(alpha=.15)
        ax.plot(.35,0,'o',color='#223b53',markersize=4)
    fig.legend([Patch(color=c) for c in colors],['Not passed','Endpoint only','Motion passed'],
        loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Enhanced full6 JIK | TCP z=5 mm | 4 cm grid | original motion limits unchanged')
    fig.savefig(OUT/'workspace_map.png',dpi=170,bbox_inches='tight');plt.close(fig)
    goals=[indexed[(x,y)] for x,y in ((.67,.28),(.67,-.28),(.55,.28),(.55,-.28))]
    comparison=dict(old_summary=json.loads((OUT.parent/'summary.json').read_text()),
        new_summary=summary,selected_goals=goals,
        y_sides={side:dict(tested=sum(r['y_m']*sign>0 for r in rows),
            endpoint=sum(r['endpoint_pass'] for r in rows if r['y_m']*sign>0),
            motion=sum(r['motion_pass'] for r in rows if r['y_m']*sign>0))
            for side,sign in (('positive',1),('negative',-1))})
    (OUT/'comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    text='''# 取消测试额外运动限速后的工作范围

删除测试程序的参考位置速度斜坡（进场 0.15 m/s、下降 0.03 m/s），直接把当前航点送给原控制器。保留两个航点：先到目标上方 z=0.10 m；实际 TCP 距上方航点≤5 mm 后，切换至最终 z=0.005 m。没有参考轨迹速度上限。

保留原有完整 6D DLS 的 0.90 m/s 末端线速度上限、关节速度限制、500 Hz 实际速度保护、0.11 s 前瞻和关节位置限幅。保持增强伺服 kp×4、kv×2，重力前馈和姿态增益 10。额外加速度限制和近桌面速度限制仍关闭。

测试仍为相同 home、固定朝向、961 个 4 cm 网格点，最多 16 s。终点位置≤5 mm、姿态≤2°并满足离桌/关节/自碰撞条件连续保持 2 s；全程通过再要求最大姿态误差≤5°、实际关节速度不超过限值的 1.01 倍。

| 范围 | 通过点数 | 网格面积估计 m² | X 样本边界 m | Y 样本边界 m |
|---|---:|---:|---|---|
'''
    for key,label in (('endpoint_pass','终点通过'),('motion_pass','全程运动通过')):
        s=summary[key];text+=f"| {label} | {s['cells']} | {s['sampled_area_m2']:.4f} | {s['x_m']} | {s['y_m']} |\n"
    text+='\n图左上是历史慢速参考轨迹，右上是取消斜坡后的新扫描。外包围坐标内有失败点，面积是网格估计。新数据单独保存，旧图和旧视频不替换。\n'
    (OUT/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(verification=verification,summary=summary,selected_goals=goals),indent=2),flush=True)


def gallery():
    import build_full6_asymmetry_gallery as gallery_base
    gallery_base.OUT=OUT/'videos'
    # Run the existing video/frame/provenance checks, then replace explanatory
    # text so it is driven by the new recordings, never historical conclusions.
    gallery_base.main()
    path=OUT/'videos/index.html'
    html=path.read_text(encoding='utf-8')
    html=html.replace('增强版本 6 · 正负 Y 对照录像','增强版本 6 · 取消测试额外限速')
    html=html.replace('相同 home、固定朝向；目标 Y 取正负镜像。',
        '取消测试程序 0.15/0.03 m/s 参考轨迹速度斜坡；原有末端 0.90 m/s、关节速度限制和实际速度保护保持。先到上方航点，再下降；目标 Y 取正负镜像。')
    html=html.replace('限位与超时：(0.67, ±0.28, 0.005) m','目标组 A：(0.67, ±0.28, 0.005) m')
    html=html.replace('终点与全程的区别：(0.55, ±0.28, 0.005) m','目标组 B：(0.55, ±0.28, 0.005) m')
    html=html.replace('${floor}<br><span id="frame-clock-',
        '${floor}<br>结束时航点阶段：${e.result.descent_started?\'已进入下降\':\'停在上方航点\'}<br><span id="frame-clock-')
    html=html.replace('先以 0.15 m/s 接近目标上方 10 cm，再以 0.03 m/s 下降，检查终点保持 2 秒。版本 3 成功后结束，短录像停在末帧，版本 1、6 继续到 16 秒。',
        '直接发送上方航点，实际到达后发送下降航点；原有运动限速保持，检查终点保持 2 秒。短录像停在末帧。')
    html=re.sub(r"\(goal==='limit_pair'\?.*?\)\+' 终点通过：",
        "'本页为取消测试额外限速后的新物理录像，默认 1×。 终点通过：",html,count=1)
    html=re.sub(r'<footer>.*?</footer>',
        '<footer><a href="../workspace_map.png" target="_blank">新工作范围图</a><a href="../comparison.json" target="_blank">新旧结果</a><a href="../manifest.json" target="_blank">配置与原有限速核查</a><a href="video_verification.json" target="_blank">录像核查</a></footer>',html,count=1)
    path.write_text(html,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gallery',action='store_true');a=p.parse_args()
    gallery() if a.gallery else report()
