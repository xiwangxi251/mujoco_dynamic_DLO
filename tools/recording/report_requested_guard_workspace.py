"""Suggested box intersection and largest sampled rectangle, separate from scan."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch,Rectangle
import probe_requested_guard_workspace as run

s=json.loads((run.OUT/'summary.json').read_text())
rows=json.loads((run.OUT/'acceptance_cells.json').read_text())['moderate']
key={(r['x_m'],r['y_m']):r for r in rows}
mask=np.array([[3 if key[(float(x),float(y))]['motion'] else 2 if key[(float(x),float(y))]['endpoint'] else 0 for x in run.XS] for y in run.YS])
fig,ax=plt.subplots(figsize=(7.4,9.8));fig.subplots_adjust(top=.85,bottom=.16,left=.15,right=.94)
colors=['#e8edf2','#ef8878','#8abfec','#239879']
ax.imshow(mask,origin='lower',extent=(.6,1.,-.5,.5),cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,4),4),interpolation='nearest')
rect=s['moderate']['largest_motion_rectangle'];x0,x1=rect['x_cell_edges_m'];y0,y1=rect['y_cell_edges_m']
ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#172d52',linewidth=2.5,linestyle='--'))
ax.set_title('Candidate: endpoint 10 mm / 3 deg; path <=8 deg\nMotion intersection 0.1856 m² / 46.4%\nLargest sampled rectangle 0.1248 m²',fontsize=12)
ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_xticks([.6,.7,.8,.9,1.]);ax.grid(alpha=.15)
fig.suptitle('Guarded bounded enhanced6 in the requested workspace',fontsize=14)
fig.legend([Patch(color=colors[i]) for i in (0,2,3)]+[Patch(facecolor='none',edgecolor='#172d52',linestyle='--')],
    ['Endpoint failed','Endpoint only','Full motion passed','Largest fully passed sampled rectangle'],loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False)
fig.savefig(run.OUT/'recommended_workspace.png',dpi=170,bbox_inches='tight');plt.close(fig)
(run.OUT/'README.md').write_text('''# 期望方框的验收门槛分析

当前带碰桌保护的有界增强6。X=[0.6,1.0]、Y=[-0.5,0.5]、Z=0.005m，方框面积0.4m²。
250个4cm方格中心覆盖方框，另测四个角点(不进入面积计数)。网格与上一轮偏移不同，点数不能直接相减。
每点同一home，首个20ms给Z=0.10m上方参考，其后直接给Z=0.005m终点参考；无实际到达切换门限或额外参考限速。
每点完整16秒物理仿真并保存20ms历史与qpos，再逐档检查最早连续100帧(2秒)同时满足全部验收条件的窗口。
各档用相同物理轨迹，控制器/guard默认参数/模型/原有限速/伺服/重力前馈均未改。与原始trial前缀qpos最大差0。

固定条件：手指离桌0.2–8mm，被检查机器人几何离地>=0.2mm，实际关节越界容差0.15°，原自碰撞谓词无碰撞，16秒含2秒保持。
自碰撞记录沿用原脚本的二值距离代理，并非新计算的精确1mm距离。每帧均做同一碰撞检查。
全程通过额外要求到最早终点验收时刻的最大姿态偏差不超过对应档门槛、实际峰值关节速度/原上限<=1.01。

原档终点5mm/2°，路径5°：终点135/250(0.216m²)，全程50/250(0.080m²)。
仅路径放宽8°：终点不变，全程115/250(0.184m²)。
候选工程工作区门槛终点10mm/3°，路径8°：终点141/250(0.2256m²)，全程116/250(0.1856m²)。
更松15mm/5°，路径10°：终点147/250(0.2352m²)，全程仍116/250。
候选档最大完整通过采样矩形：X=[0.60,0.84]、Y=[-0.22,0.30]，面积0.1248m²。
只算终点时候选档最大矩形X=[0.60,0.88]、Y=[-0.30,0.30]，面积0.168m²。

这些面积是格点采样估计，矩形边界为方格边缘，仅保证内部格点已通过；不能声称所有连续坐标或所有运动起点都通过。
10mm/3°/8°为工作区筛选候选，不更改实际抓取成功定义。需保留精确终点时，可用5mm/2°/8°，其全程交集仅比候选少1个点。
右侧两个角点16秒末的位置误差约234mm。放宽终点到50mm/3°只能覆盖70.4%，100mm/3°为83.6%(仅终点)。
因此当前控制器/姿态/高度/起点/16s协议下，不能靠适度放宽要求覆盖整个期望方框。这不是几何不可达的证明。
旧范围扫描与新门槛均保留，原验收定义未覆盖。

来源与复现：manifest.json、cells.json、traces/、acceptance_cells.json、summary.json、verification.json。
运行 tools/recording/probe_requested_guard_workspace.py --workers 6；完成后运行 verify_requested_guard_workspace.py 与 report_requested_guard_workspace.py。
''',encoding='utf-8')
print(json.dumps(dict(rectangle=rect,report=str(run.OUT/'recommended_workspace.png'))))
