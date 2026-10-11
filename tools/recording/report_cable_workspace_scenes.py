"""Derived report only; leave the recording source and physics frozen."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vertical_workspace/scripted_136_retest_20261004/cable_workspace_scenes'
rows=json.loads((OUT/'gallery_manifest.json').read_text(encoding='utf-8'))
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc',size=13)
all_points=[np.load(OUT/r['scene']/'trajectory.npz')['cable_xyz'] for r in rows]
lo=np.min([p[...,:2].min(axis=(0,1)) for p in all_points],axis=0)
hi=np.max([p[...,:2].max(axis=(0,1)) for p in all_points],axis=0)
lo=np.floor((lo-.06)*10)/10;hi=np.ceil((hi+.06)*10)/10
fig,axes=plt.subplots(2,3,figsize=(16,11),sharex=True,sharey=True)
for row,p,ax in zip(rows,all_points,axes.flat):
    target=row['target_node']
    ax.scatter(p[::5,:,0].ravel(),p[::5,:,1].ravel(),s=2,alpha=.13,color='#1686bb')
    ax.plot(p[:,target,0],p[:,target,1],color='#e85039',lw=1.4)
    ax.add_patch(Rectangle((.6,-.22),.24,.52,fill=False,edgecolor='#c49a00',lw=2.5))
    ax.scatter(.35,0,color='#333',marker='s',s=25)
    ax.set_title(row['label'],fontproperties=font)
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.grid(alpha=.2)
    ax.tick_params(labelbottom=True,labelleft=True)
    ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
fig.suptitle('线缆当前摆位：蓝色为整缆节点运动轨迹，红色为本回合抽到的节点14，黄色为候选工作区\n机械臂固定；seed 20260804；XY投影；坐标范围包含全部保存的节点位置',fontproperties=FontProperties(fname='C:/Windows/Fonts/msyh.ttc',size=16))
fig.tight_layout();fig.savefig(OUT/'motion_overview.png',dpi=160);plt.close(fig)

checks=[]
for row in rows:
    d=OUT/row['scene'];verification=json.loads((d/'video_verification.json').read_text())
    assert row['robot_qpos_max_change']==row['robot_link_position_max_change_m']==0
    assert row['robot_cable_contact_substep_pairs']==0
    assert verification['rendered_fullphysics_state_max_difference']==0
    assert hashlib.sha256((d/'trajectory.npz').read_bytes()).hexdigest()==row['trajectory_sha256']
    for path,digest in row['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    for view in ('top','global','wrist'):
        url=f'http://127.0.0.1:8766/cable_workspace_scenes/{row["scene"]}/{view}.mp4'
        req=urllib.request.Request(url,headers={'Range':'bytes=0-15'})
        with urllib.request.urlopen(req) as response:
            data=response.read();assert response.status==206 and len(data)==16
            assert response.headers['Content-Range'].startswith('bytes 0-15/')
            checks.append(dict(scene=row['scene'],view=view,http_status=response.status,content_range=response.headers['Content-Range']))
report=dict(scenes=6,clips=18,all_robot_poses_fixed=True,all_rendered_states_exact=True,
    all_robot_cable_contact_pairs_zero=True,http_byte_range_checks=checks,
    overview_full_saved_xy_bounds_m=dict(minimum=lo.tolist(),maximum=hi.tolist()),
    derived_report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(OUT/'README.md').write_text('''# 线缆六场景与候选工作区

查看 `index.html`。六个正式 core 场景，同一随机种子 20260804。
俯视、全局、腕部均为原生 1280×960 / 50fps，H.264 CRF16，GOP≤12，无B帧。

- 黄色矩形：X=0.60～0.84，Y=−0.22～0.30m，渲染在桌上6mm高度，仅视觉标注。
- 蓝色物体：原线缆。红点：本次环境默认随机抽取的目标节点14；不代表所有策略的目标选取。
- 青色线：该节点截至当前帧的XY轨迹，投影到桌上8mm高度，仅视觉标注。
- 整缆全部保存节点轨迹的总览见 `motion_overview.png`；图中黑色方点为底座XY位置。

未平移线缆、未修改驱动、材料、桌面或碰撞配置。仅制作可视化固定夹具：
每个2ms物理步后恢复机械臂7关节与夹爪2关节的ready位置，清零对应速度与warmstart。
线缆自由度不修改，不运行任何抓取策略，不计算任务成功率。
机械臂位置变化为0，六个回合均没有机器人与线缆接触；来源、参数见 manifest.json 与各场景 metadata.json。
先保存全物理状态，再逐帧渲染；渲染状态差值为0，不插值。静态/形变15秒；整体运动沿用质心越过原Y=0.70出口线停止。

框内比例仅为XY节点中心落入矩形的帧统计，不代表完整姿态可达、指定节点能抓到或任务成功。
这是单个种子的当前摆位展示，不能视作全部随机回合的活动范围。
''',encoding='utf-8')
print(json.dumps(dict(scenes=6,clips=18,all_checks_passed=True)),flush=True)
