"""Validate a continuous floor-level path and render the measured workspace."""
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage

import record_improved_workspace as w
from render_vertical_workspace import font,add_geom,add_line

OUT=w.OUT


def load():
    with np.load(OUT/'workspace.npz') as file:d={k:file[k] for k in file.files}
    return d,json.loads((OUT/'summary.json').read_text()),json.loads((OUT/'manifest.json').read_text())


def map_image(d,size=650,actual=None,path=None,target=None):
    image=Image.new('RGB',(size,size),'#f5f8fc');dd=ImageDraw.Draw(image)
    pad=.12*size;scale=.76*size/1.36
    def xy(x,y):return pad+(x+.33)*scale,size-pad-(y+.68)*scale
    step=float(d['xs'][1]-d['xs'][0])
    for key,color in (('endpoint_pass','#6cb2e7'),('motion_pass','#30b58a'),('interior','#16876b')):
        for iy,ix in zip(*np.where(d[key])):
            x,y=d['xs'][ix],d['ys'][iy]
            dd.rectangle([xy(x-step/2,y+step/2),xy(x+step/2,y-step/2)],fill=color)
    f=font(max(12,int(size*.028)))
    for value in np.arange(-.2,1.01,.2):
        a,b=xy(value,-.68),xy(value,.68);dd.line([a,b],fill='#bbc7d3',width=1)
        dd.text((a[0]-12,a[1]+5),f'{value:.1f}',font=f,fill='#455266')
    for value in np.arange(-.6,.61,.2):
        a,b=xy(-.33,value),xy(1.03,value);dd.line([a,b],fill='#bbc7d3',width=1)
        dd.text((2,a[1]-9),f'{value:+.1f}',font=f,fill='#455266')
    contours,_=cv2.findContours(d['motion_pass'].astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        pts=[xy(d['xs'][ix],d['ys'][iy]) for ix,iy in contour[:,0]]
        if len(pts)>1:dd.line(pts+[pts[0]],fill='#174d45',width=max(1,size//300))
    a=xy(.35,0);r=size*.016
    dd.ellipse((a[0]-r,a[1]-r,a[0]+r,a[1]+r),fill='#253651')
    dd.text((a[0]+r+2,a[1]-9),'底座',font=f,fill='#253651')
    if path is not None:
        dd.line([xy(p[0],p[1]) for p in path],fill='#e0a921',width=max(2,size//160))
    for p,color in ((target,'#e0a921'),(actual,'#ed416b')):
        if p is not None:
            x,y=xy(p[0],p[1]);r=size*.009;dd.ellipse((x-r,y-r,x+r,y+r),fill=color,outline='white',width=1)
    dd.text((size*.43,size-25),'世界 X / m',font=f,fill='#455266')
    dd.text((5,4),'世界 Y / m',font=f,fill='#455266')
    return image


def shortest(mask,start,goal):
    queue=deque([start]);prev={start:None}
    while queue:
        node=queue.popleft()
        if node==goal:break
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            neighbor=node[0]+dy,node[1]+dx
            if 0<=neighbor[0]<mask.shape[0] and 0<=neighbor[1]<mask.shape[1] and mask[neighbor] and neighbor not in prev:
                prev[neighbor]=node;queue.append(neighbor)
    assert goal in prev
    result=[];node=goal
    while node is not None:result.append(node);node=prev[node]
    return result[::-1]


def demo(d):
    # A connected interior region prevents interpolation across map holes.
    labels,n=ndimage.label(d['interior'])
    if not n:labels,n=ndimage.label(d['motion_pass'])
    counts=np.bincount(labels.ravel());counts[0]=0;label=int(np.argmax(counts));mask=labels==label
    ys,xs=np.where(mask);assert len(xs)>0
    start_index=int(np.argmin((d['xs'][xs]-.7)**2+d['ys'][ys]**2));start=(int(ys[start_index]),int(xs[start_index]))
    # Traverse representative extents using verified neighboring sample cells.
    candidates=[(int(ys[i]),int(xs[i])) for i in (np.argmax(xs),np.argmax(ys),np.argmin(xs),np.argmin(ys))]
    route=[start]
    for goal in candidates+[start]:route+=shortest(mask,route[-1],goal)[1:]
    # The path may cover a subset of the measured region; it is not a boundary.
    coords=np.array([[d['xs'][ix],d['ys'][iy],w.Z] for iy,ix in route])
    rig=w.ImprovedRig();initial,h=rig.trial(coords[0,0],coords[0,1],True)
    assert initial['motion_pass'],initial
    history=list(h);reference=coords[0].copy();stage=[0]*len(history);valid=True;failure=None
    for index,goal in enumerate(coords[1:]):
        for _ in range(300):
            delta=goal-reference;distance=np.linalg.norm(delta)
            reference+=delta*min(1.,.02*w.DT/max(distance,1e-12))
            rig.control(reference);m=rig.measure(reference,collision=True)
            finger=min(float(rig.geometry.geom_vertices(g)[:,2].min()) for g in rig.geometry.finger_geoms)
            history.append(np.r_[rig.data.time,rig.data.qpos[:9],m,reference,reference,finger]);stage.append(1)
            if not (m[0]<=.005 and m[2]<=2. and .0002<=finger<=.008 and m[3]>=.0002 and m[4]>=0 and m[5]>=.001):
                valid=False;failure={'waypoint_index':index,'reference':reference.tolist(),'metrics':m.tolist(),'finger_floor_m':finger};break
            if np.linalg.norm(reference-goal)<1e-9:break
        if not valid:break
    # End with a stable physical hold rather than freezing the final frame.
    if valid:
        for _ in range(100):
            rig.control(reference);m=rig.measure(reference,collision=True)
            finger=min(float(rig.geometry.geom_vertices(g)[:,2].min()) for g in rig.geometry.finger_geoms)
            history.append(np.r_[rig.data.time,rig.data.qpos[:9],m,reference,reference,finger]);stage.append(2)
            valid=valid and bool(m[0]<=.005 and m[2]<=2. and .0002<=finger<=.008 and m[3]>=.0002 and m[4]>=0 and m[5]>=.001)
    history=np.asarray(history);stage=np.asarray(stage);ground=history[stage>0]
    meta={'continuous_ground_motion_pass':bool(valid),'failure':failure,'waypoint_count':len(coords),
        'reference_path_length_m':float(np.sum(np.linalg.norm(np.diff(coords[:,:2],axis=0),axis=1))),
        'ground_reference_speed_m_s':.02,'sim_seconds':float(history[-1,0]),
        'ground_max_position_error_mm':float(np.max(ground[:,10])*1000) if len(ground) else None,
        'ground_max_tilt_deg':float(np.max(ground[:,11])) if len(ground) else None,
        'ground_max_full_pose_error_deg':float(np.max(ground[:,12])) if len(ground) else None,
        'ground_finger_clearance_mm':[float(np.min(ground[:,25])*1000),float(np.max(ground[:,25])*1000)] if len(ground) else None,
        'peak_velocity_to_limit':float(np.max(rig.peak_velocity/rig.base._arm_velocity_limits)),
        'actual_velocity_exceedance_substeps':rig.over_steps,
        'max_combined_torque_nm':rig.base.max_total_torque.tolist(),
        'path_role':'one measured continuous demonstration inside workspace; not its boundary'}
    valid=valid and meta['peak_velocity_to_limit']<=1.01;meta['continuous_ground_motion_pass']=bool(valid)
    np.savez_compressed(OUT/'continuous_demo.npz',history=history,stage=stage,waypoints=coords)
    (OUT/'continuous_demo.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)
    assert valid,'Continuous path did not pass; retain its raw diagnostic and choose a shorter interior demonstration.'
    return history,stage,coords


def verify(plan):
    for p,sha in plan['source_sha256'].items():assert hashlib.sha256((w.ROOT/p).read_bytes()).hexdigest()==sha,p
    assert (OUT/'source_snapshot.py').read_bytes()==Path(w.__file__).read_bytes()
    rows=json.loads((OUT/'cells.json').read_text());assert len(rows)==961
    assert len({(r['x_m'],r['y_m']) for r in rows})==len(rows)
    for r in rows:
        assert np.all(np.isfinite([r['position_error_mm'],r['pose_error_deg'],r['finger_floor_mm'],r['peak_velocity_to_limit']]))
        if r['motion_pass']:assert r['endpoint_pass'] and r['max_path_pose_deg']<=5 and r['peak_velocity_to_limit']<=1.01
    rig=w.ImprovedRig();bounds=np.max(np.abs(rig.model.actuator_forcerange[:7]),axis=1)
    assert all(np.all(np.asarray(r['max_combined_torque_nm'])<=bounds+1e-9) for r in rows)
    verification={'rows':len(rows),'unique':len(rows),'all_controller_and_physics_source_hashes_match':True,
        'all_recorded_combined_torques_within_original_limits':True,'finite_endpoint_metrics':True,
        'motion_pass_includes_path_pose_and_speed_checks':True,
        'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')


def report(d,summary,plan,meta):
    im=Image.new('RGB',(1280,1050),'#f5f8fc');dd=ImageDraw.Draw(im)
    dd.text((25,20),'NERO · 改进控制器的固定朝向试跑通过区',font=font(30),fill='#182b43')
    dd.text((25,70),'Jacobian＋重力前馈＋kp×4/kv×2＋姿态反馈×4；实际物理扫描',font=font(23),fill='#415169')
    im.paste(map_image(d,840,path=meta.get('actual_path')),(0,130));dd=ImageDraw.Draw(im)
    details=[('绿色：运动过程也通过','#188569'),('深绿：距采样边界≥6 cm','#166d5a'),('蓝色：仅终点保持通过','#398fc0'),('空白：当前试跑未通过','#55667a')]
    for j,(txt,col) in enumerate(details):dd.text((875,150+j*42),txt,font=font(22),fill=col)
    text=['固定水平朝向；夹爪竖直向下','TCP目标离地5 mm','实际指尖离地0.2–8 mm','终点姿态≤2°，保持2秒','全过程姿态≤5°；检查超速','4 cm网格；色块内部有空洞']
    for j,txt in enumerate(text):dd.text((875,355+j*35),txt,font=font(19),fill='#415169')
    for key,y,label in [('motion_pass',600,'运动通过范围'),('interior',765,'内部余量范围')]:
        bb=summary[key]
        if bb:
            for j,txt in enumerate([label,f"X {bb['x_m'][0]:+.2f}～{bb['x_m'][1]:+.2f} m",f"Y {bb['y_m'][0]:+.2f}～{bb['y_m'][1]:+.2f} m",f"网格面积约 {bb['area_m2']:.3f} m²"]):
                dd.text((875,y+j*32),txt,font=font(21),fill='#415169')
    dd.text((28,1000),'当前配置、初始姿态和路径下的空载仿真结果；包围框不是全部可用区域。',font=font(22),fill='#415169')
    im.save(OUT/'workspace_map.png')
    lines=['# NERO 改进控制器：固定朝向、同一起点的试跑通过区域','',
        '**范围口径更正：本图是当前控制器在固定水平朝向、同一home和指定路径下的通过区域，不是机械臂的完整几何可达集合。先前以“工作区”统称容易误解。旧SciPy圆环测的是水平转向自由、多初值求解的几何可达集合；目标高度、网格、开口和通过条件也不同，不能直接将两张图视为改进前后机械臂可达面积的对比。**', '',
        '坐标网格关于底座对称，原始测量结果本身不对称。补查(0.67,±0.28,0.005)m：两点都有满足相同固定姿态的几何解；负侧点的原控制试跑在joint6限位附近，位置误差12.387 mm，而正侧点通过。负侧点初始化到一个几何解后的局部保持仍未通过，不能把原因单独归结为初始姿态，也不能把这些几何解当作已验证的运动轨迹。详见asymmetry_diagnostic.json和workspace_scope_comparison.png。', '',
        '直接调用抓取实验冻结的规则 Jacobian IK（DiagnosticPolicy._ik_action）和执行层（DiagnosticEnv._physics_step），重力前馈500 Hz、kp×4/kv×2、姿态误差反馈×4；未调用适配器或SciPy解算器来判断范围。', '',
        '本任务固定指尖向下、固定水平朝向，TCP目标z=5 mm，张开夹爪，移除无关线缆。机器人质量、惯量、关节参数、指垫接触、原力矩范围和环境限制保留。竖直姿态基准沿用项目NERO标定；本任务提供固定测试轨迹，不执行抓取状态机，因此不是抓取成功率测试。', '',
        '每点从同一home独立出发：0.15 m/s移到目标上方10 cm，0.03 m/s下降，每点预算16秒。终点位置误差≤5 mm、完整姿态误差≤2°、指尖离地0.2–8 mm、机器人不穿地、关节合法、非邻接凸包距离≥1 mm，持续保持2秒。', '',
        '“运动通过”另要求整个进场路径完整姿态误差≤5°、实际关节速度峰值≤原限制的1.01倍（1%数值余量），同时保留精确超速子步占比；不是把命令限速误当成实际限速。', '',
        '| 区域 | 通过点 | 世界X/m | 世界Y/m | 采样面积/m² |','|---|---:|---|---|---:|']
    for key,label in [('endpoint_pass','终点贴地保持'),('motion_pass','进场与保持均通过'),('interior','距运动区域边界≥6 cm')]:
        bb=summary[key]
        if bb:lines.append(f"| {label} | {bb['cells']} | {bb['x_m'][0]:+.2f}～{bb['x_m'][1]:+.2f} | {bb['y_m'][0]:+.2f}～{bb['y_m'][1]:+.2f} | {bb['area_m2']:.3f} |")
    lines+=['',f"共扫描{summary['tested']}点，4 cm分辨率。底座世界坐标(0.35,0,0)，地面z=0。包围框内有空洞；面积按通过点数×网格面积估算，边界精度受网格限制。",
        '未通过表示当前初始姿态、固定水平朝向、参考路径和时间预算下失败，不证明机械臂在数学或物理上绝对不可达。内部余量区域仅保留平面边界余量，不能称为真机安全认证区域。', '',
        '连续贴地录像是实际物理轨迹的2倍速回放：先从home下降，再在通过区域内部沿相邻网格移动，最后继续控制保持2秒。黄色线是这一条已验证路径，不是工作区边界。','',
        '连续横移4 cm/s和3 cm/s的先行验证分别因位置误差5.061 mm、5.002 mm超过5 mm门限而停止，原始轨迹保留在continuous_demo_004_failed.*和continuous_demo_003_failed.*。最终视频验证速度为2 cm/s；这些试验只调整演示参考速度，没有修改扫描结果、控制器、物理参数或通过门限。', '',
        '```json',json.dumps({k:v for k,v in meta.items() if k!='actual_path'},ensure_ascii=False,indent=2),'```','',
        '产物：workspace_map.png、improved_vertical_workspace.mp4、cells.json/csv/jsonl、workspace.npz、continuous_demo.npz/json、manifest.json、source_snapshot.py、isolated_model.mjb、verification.json。',
        '代码未修改生产策略、机器人资产或原抓取数据。本结果为空载仿真工程测量，未验证带载或真机性能。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def movie(d,summary,history,stage,coords):
    rig=w.ImprovedRig();renderer=mujoco.Renderer(rig.model,height=720,width=1280,max_geom=10000)
    cam=mujoco.MjvCamera();cam.lookat[:]=(.35,.08,.16);cam.distance=1.95;cam.azimuth=125;cam.elevation=-38
    option=mujoco.MjvOption();option.geomgroup[3]=0
    raw=OUT/'improved_vertical_workspace_raw.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),25,(1280,720));assert writer.isOpened()
    step=float(d['xs'][1]-d['xs'][0]);duration=history[-1,0]
    bbox=summary['motion_pass'];playback_speed=2.;frames=int(np.ceil(duration/playback_speed*25));ground_path=history[stage>0,16:19]
    for frame,t in enumerate(np.arange(frames)/25*playback_speed):
        idx=min(len(history)-1,np.searchsorted(history[:,0],t));r=history[idx]
        rig.data.qpos[:9]=r[1:10];rig.data.qvel[:]=0;mujoco.mj_forward(rig.model,rig.data)
        renderer.update_scene(rig.data,camera=cam,scene_option=option);scene=renderer.scene
        for iy,ix in zip(*np.where(d['endpoint_pass'])):
            color=(.08,.57,.43,1) if d['interior'][iy,ix] else ((.13,.74,.50,1) if d['motion_pass'][iy,ix] else (.34,.67,.87,1))
            add_geom(scene,mujoco.mjtGeom.mjGEOM_BOX,[step*.47,step*.47,.0004],[d['xs'][ix],d['ys'][iy],.001],color)
        contours,_=cv2.findContours(d['motion_pass'].astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts=contour[:,0]
            for a,b in zip(pts,np.roll(pts,-1,axis=0)):
                add_line(scene,[d['xs'][a[0]],d['ys'][a[1]],.003],[d['xs'][b[0]],d['ys'][b[1]],.003],(.05,.28,.23,1),.0015)
        for value in np.arange(-.6,.61,.2):add_line(scene,[-.25,value,.002],[.95,value,.002],(.7,.79,.88,1),.0006)
        for value in np.arange(-.2,1.,.2):add_line(scene,[value,-.6,.002],[value,.6,.002],(.7,.79,.88,1),.0006)
        for a,b in zip(ground_path[::15],ground_path[15::15]):add_line(scene,a+[0,0,.006],b+[0,0,.006],(1,.71,.05,1),.002)
        tcp=r[16:19];add_geom(scene,mujoco.mjtGeom.mjGEOM_SPHERE,[.006,0,0],tcp+[0,0,.004],(.95,.2,.4,1))
        axis=rig.data.xmat[rig.base.hand_id].reshape(3,3)[:,0];add_line(scene,tcp-.07*axis,tcp,(.95,.2,.4,1),.003)
        im=Image.fromarray(renderer.render()).convert('RGBA');overlay=Image.new('RGBA',im.size);dd=ImageDraw.Draw(overlay)
        dd.rounded_rectangle((15,15,904,113),radius=12,fill=(248,251,255,240))
        dd.text((29,22),'NERO · 当前控制器的固定朝向试跑通过区',font=font(27),fill='#182b43')
        dd.text((29,66),'Jacobian＋重力前馈＋伺服修正＋姿态反馈×4',font=font(22),fill='#415169')
        dd.rounded_rectangle((919,15,1271,705),radius=12,fill=(248,251,255,249))
        dd.rounded_rectangle((18,655,904,705),radius=10,fill=(248,251,255,240))
        dd.text((30,666),'固定朝向、同一起点的试跑结果 · 2×回放 · 不代表完整几何可达范围',font=font(20),fill='#415169')
        im=Image.alpha_composite(im,overlay).convert('RGB');im.paste(map_image(d,340,actual=tcp,path=ground_path[::10]),(925,62));dd=ImageDraw.Draw(im)
        dd.text((935,25),'俯视范围 · 世界坐标 / m',font=font(20),fill='#182b43')
        status=('接近与下降','连续贴地移动','贴地保持')[int(stage[idx])]
        lines=[('绿：运动通过；蓝：仅终点通过','#188569'),
            (f"X {bbox['x_m'][0]:+.2f}～{bbox['x_m'][1]:+.2f} m",'#415169'),
            (f"Y {bbox['y_m'][0]:+.2f}～{bbox['y_m'][1]:+.2f} m",'#415169'),
            (f'{status}  {r[0]:.1f}/{duration:.1f} s','#415169'),
            (f'位置误差 {r[10]*1000:.2f} mm','#415169'),
            (f'完整姿态误差 {r[12]:.2f}°','#415169'),
            (f'实际指尖离地 {r[25]*1000:.2f} mm','#415169'),
            (f'TCP ({tcp[0]:+.3f}, {tcp[1]:+.3f})','#415169'),
            ('固定朝向；空载仿真测量','#55667a')]
        for j,(txt,col) in enumerate(lines):dd.text((934,416+j*30),txt,font=font(18),fill=col)
        writer.write(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR))
        if frame in (0,frames//2,frames-1):im.save(OUT/f'checkframe_{frame:04d}.png')
        if frame%250==0:print(f'render {frame}/{frames}',flush=True)
    writer.release();renderer.close()
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(raw),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'improved_vertical_workspace.mp4')],check=True)
    cap=cv2.VideoCapture(str(OUT/'improved_vertical_workspace.mp4'));ok,first=cap.read()
    assert ok and first.shape[:2]==(720,1280)
    actual_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert actual_frames==frames;cap.release()
    (OUT/'video_verification.json').write_text(json.dumps({'frames':frames,'fps':25,'duration_s':frames/25,'simulation_duration_s':float(duration),'playback_speed':playback_speed,'resolution':[1280,720],'first_frame_decodes':True,'h264_yuv420p':True,'actual_renderer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'fixed-yaw from-home controller pass region; not full kinematic workspace'},indent=2))


def main():
    d,summary,plan=load();verify(plan)
    if (OUT/'continuous_demo.npz').exists():
        h=np.load(OUT/'continuous_demo.npz');history,stage,coords=h['history'],h['stage'],h['waypoints']
        assert json.loads((OUT/'continuous_demo.json').read_text())['continuous_ground_motion_pass']
    else:history,stage,coords=demo(d)
    meta=json.loads((OUT/'continuous_demo.json').read_text());meta['actual_path']=history[stage>0,16:19].tolist()
    report(d,summary,plan,meta);movie(d,summary,history,stage,coords)


if __name__=='__main__':main()
