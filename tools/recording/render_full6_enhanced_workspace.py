"""Show passed regions and continuous tracking errors without failure-color dominance."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm,LogNorm
from matplotlib.patches import Patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'outputs/vertical_workspace/scripted_136_retest_20261004'
OUT=PARENT/'full6_gravity_servo_gain10'

def values(rows,xs,ys,key):
    grid=np.zeros((len(ys),len(xs)),float)
    for r in rows:grid[np.argmin(abs(ys-r['y_m'])),np.argmin(abs(xs-r['x_m']))]=r[key]
    return grid

def runs(mask,xs):
    indices=np.flatnonzero(mask)
    if not len(indices):return []
    groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    return [[float(xs[g[0]]),float(xs[g[-1]])] for g in groups]

def main():
    rows=json.loads((OUT/'cells.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text())
    with np.load(OUT/'workspace.npz') as saved:
        data={k:saved[k].copy() for k in saved.files}
    xs,ys=data['xs'],data['ys'];extent=(xs[0]-.02,xs[-1]+.02,ys[0]-.02,ys[-1]+.02)
    previous=json.loads((PARENT/'version_3/cells.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12.5,12),layout='constrained')
    colors=['#e8edf2','#87bce8','#138569']
    cmap=ListedColormap(colors);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    for ax,records,label in ((axes[0,0],previous,'3: position-first / enhanced servo + gravity / gain 10'),
                             (axes[0,1],rows,'6 + gravity + enhanced servo + angular gain 10')):
        cat=values(records,xs,ys,'endpoint_pass')+values(records,xs,ys,'motion_pass')
        ax.imshow(cat,origin='lower',extent=extent,cmap=cmap,norm=norm,interpolation='nearest')
        endpoint=sum(r['endpoint_pass'] for r in records);motion=sum(r['motion_pass'] for r in records)
        ax.set_title(label,fontsize=11)
        ax.text(.02,.02,f'Endpoint {endpoint}/961 | Motion {motion}/961\nMotion area ~{motion*.0016:.4f} m²',
            transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.94))
    position=values(rows,xs,ys,'position_error_mm')
    pose=values(rows,xs,ys,'max_path_pose_deg')
    image=axes[1,0].imshow(np.maximum(position,.1),origin='lower',extent=extent,cmap='magma',
        norm=LogNorm(vmin=1,vmax=1000),interpolation='nearest')
    axes[1,0].set_title('TCP position error at trial end (mm)')
    fig.colorbar(image,ax=axes[1,0],shrink=.75,label='mm (log scale)')
    image=axes[1,1].imshow(pose,origin='lower',extent=extent,cmap='viridis',vmin=0,vmax=180,interpolation='nearest')
    axes[1,1].set_title('Maximum full-pose error over the recorded path')
    fig.colorbar(image,ax=axes[1,1],shrink=.75,label='degrees')
    for ax in axes.ravel():
        ax.set_aspect('equal');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
        ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.grid(alpha=.15)
        ax.plot([.35],[0],'o',color='#223b53',markersize=4)
    fig.legend([Patch(color=c) for c in colors],['Not passed','Endpoint only','Motion passed'],
        loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('NERO near-table workspace | TCP z=5 mm | 4 cm grid | same home and trajectory',fontsize=14)
    fig.savefig(OUT/'workspace_map.png',dpi=170,bbox_inches='tight');plt.close(fig)
    sections=[dict(y_m=float(y),motion_x_intervals_m=runs(data['motion_pass'][i],xs),
                   endpoint_x_intervals_m=runs(data['endpoint_pass'][i],xs)) for i,y in enumerate(ys)]
    (OUT/'cross_sections.json').write_text(json.dumps(sections,indent=2),encoding='utf-8')
    base_summary=json.loads((PARENT/'version_3/summary.json').read_text())
    old_area=base_summary['motion_pass']['area_m2'];new_area=summary['motion_pass']['sampled_area_m2']
    markdown=f'''# 版本 6 + 重力前馈 + 增强伺服 + 姿态增益 10：近桌面工作范围

组合为完整 6D DLS（lambda=0.04），臂 kp×4、kv×2，500 Hz 有界重力前馈，姿态反馈增益 10。保留版本 6 的限速、0.11 s 前瞻、关节范围和碰撞配置。

961 点实际物理扫描，4 cm 网格，目标 TCP z=5 mm。每点从相同 home 开始，固定姿态/偏航，先以 0.15 m/s 进场至目标上方 10 cm，再以 0.03 m/s 下降；最多 16 s。

终点通过：位置误差≤5 mm、完整姿态误差≤2°，夹指离桌 0.2–8 mm、全机器人离桌≥0.2 mm、关节位置合法、非相邻网格自碰撞检查通过，连续保持 2 s。运动通过再要求全程姿态误差≤5°、实际关节速度不超过限速（1%数值容差）。任一被检查部位 z<-1 mm 会提前终止。

| 范围 | 通过网格数 | 网格面积估计 (m²) | X 样本边界 (m) | Y 样本边界 (m) |
|---|---:|---:|---|---|
'''
    for key,label in (('endpoint_pass','终点通过'),('motion_pass','运动通过'),('interior','运动区域内部 ≥6 cm')):
        s=summary[key]
        markdown+=f"| {label} | {s['cells']} | {s['sampled_area_m2']:.4f} | {s['x_m']} | {s['y_m']} |\n"
    markdown+=f'''
运动通过面积相对版本 3（{old_area:.4f} m²）变化 {(new_area/old_area-1)*100:+.1f}%。

坐标边界只是所有通过样本的外包围框，框内存在失败点。面积按通过点数×0.04²估计；不是连续几何精确面积，也不是任意高度/姿态的三维可达体积。逐 Y 截面的连续通过 X 样本区间见 cross_sections.json。

主图仅区分运动通过、仅终点通过和未通过。两个诊断图显示各自测试结束时的位置误差及截至结束时的全程最大姿态误差；穿地提前停止的点没有 16 s 后续数据。

pilot_verification.json 核查两个历史 all_three 消融轨迹的前缀，关节最大差值为 0；manifest.json 保存完整配置与源文件哈希。cells.json/csv 为原始扫描记录，summary.json 为数值摘要。
'''
    (OUT/'README.md').write_text(markdown,encoding='utf-8')
    print(json.dumps(dict(summary=summary,comparison_to_version3_motion_area_percent=(new_area/old_area-1)*100,
                         center_y_zero=next(s for s in sections if abs(s['y_m'])<1e-8))),flush=True)

if __name__=='__main__':main()
