"""Native corner recordings; rerun the frozen 16 s range probe and check qpos."""
import argparse
import hashlib
import json
import multiprocessing as mp
import re
import subprocess
import time
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
import probe_requested_guard_workspace as run
from record_retest_136_videos import RGBEncoder

PARENT_BOX=run.OUT
run.OUT=PARENT_BOX/'rectangle_060_084'
OUT=run.OUT/'corner_videos'
GOALS=((.6,-.22,'near_negative'),(.6,.3,'near_positive'),(.84,-.22,'far_negative'),(.84,.3,'far_positive'))


def record(goal):
    x,y,name=goal;dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((run.OUT/'manifest.json').read_text());run.guard.assert_sources(manifest['source_sha256'])
    row=next(r for r in json.loads((run.OUT/'cells.json').read_text()) if (r['x_m'],r['y_m'])==(x,y))
    saved=np.load(run.OUT/row['trace']);expected=saved['qpos'];expected_history=saved['history']
    rig=run.guard.GuardRig(True);rig.reset();env=rig.env;model=rig.model;data=rig.data
    model.vis.global_.offwidth=max(1280,model.vis.global_.offwidth);model.vis.global_.offheight=max(960,model.vis.global_.offheight)
    renderer=mujoco.Renderer(model,height=960,width=1280,max_geom=10000)
    camera=mujoco.MjvCamera();camera.lookat[:]=(.61,0,.14);camera.distance=1.95;camera.azimuth=135;camera.elevation=-35
    option=mujoco.MjvOption();option.geomgroup[3]=0
    local_rotation=np.empty(9);mujoco.mju_quat2Mat(local_rotation,np.asarray(env.config.dynamicvla_wrist_camera_quat));local_rotation=local_rotation.reshape(3,3)
    local_position=np.asarray(env.config.dynamicvla_wrist_camera_pos)
    writers={v:RGBEncoder(dest/f'{v}.mp4',1280,960,50) for v in ('global','wrist')}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    target=np.array([x,y,.005]);approach=np.array([x,y,.10]);states=[];actual=[];history=[]
    corners=np.array([[.6,-.22,.006],[.84,-.22,.006],[.84,.3,.006],[.6,.3,.006]])
    started=time.monotonic()

    def frame():
        current=env.hand_position.copy();m=rig.measure(target,collision=True)
        finger=min(float(rig.geometry.geom_vertices(g)[:,2].min()) for g in rig.geometry.finger_geoms)
        ratio=float(np.max(rig.peak_velocity/env._arm_velocity_limits))
        history.append([float(data.time),m[0]*1000,m[2],finger*1000,m[3]*1000,m[4]-.15,m[5]*1000,ratio,m[1]])
        states.append(np.r_[data.time,data.qpos[:9],data.qvel[:9],data.ctrl]);actual.append(current)
        for view,writer in writers.items():
            renderer.update_scene(data,camera=camera,scene_option=option)
            if view=='wrist':
                rotation=data.xmat[env.hand_id].reshape(3,3);position=data.xpos[env.hand_id]+rotation@local_position
                world_rotation=rotation@local_rotation
                for c in renderer.scene.camera:c.pos[:]=position;c.forward[:]=-world_rotation[:,2];c.up[:]=world_rotation[:,1]
            for a,b in zip(corners,np.roll(corners,-1,axis=0)):
                g=renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.95,.73,.13,.8]))
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.0015,a,b);renderer.scene.ngeom+=1
            g=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.009,0,0]),target,np.eye(3).ravel(),np.array([1,.73,.02,1]));renderer.scene.ngeom+=1
            im=Image.fromarray(renderer.render());d=ImageDraw.Draw(im)
            d.rounded_rectangle((16,14,1264,135),radius=10,fill='#f6f9fc')
            d.text((30,20),f"带约束增强6＋碰桌保护 · {'全局' if view=='global' else '腕部'} · 实际物理重跑 · 1×",font=font,fill='#20334b')
            d.text((30,60),f'目标 ({x:.2f}, {y:+.2f}, 0.005) m | t={data.time:.2f} s | 位置误差 {m[0]*1000:.1f} mm | 姿态 {m[2]:.2f}°',font=small,fill='#334d66')
            d.text((30,96),f'夹指离桌 {finger*1000:+.3f} mm | 黄色框：候选矩形；黄点：当前角点目标',font=small,fill='#334d66')
            writer.write(np.asarray(im))
            if len(states)==1:im.save(dest/f'{view}_poster.png')
        if len(states)%200==0:print(json.dumps(dict(goal=name,frames=len(states),wall_s=round(time.monotonic()-started,1))),flush=True)

    try:
        frame()
        for step in range(row['steps']):
            rig.control(approach if step==0 else target,run.guard.exp.base.native.Phase.INTERCEPT)
            frame()
    finally:
        for writer in writers.values():writer.release()
        renderer.close()
    states=np.asarray(states);history=np.asarray(history)
    difference=float(np.max(abs(states[1:,1:10]-expected[:,:9])))
    measurement_difference=float(np.max(abs(history[1:]-expected_history)))
    assert difference==0 and measurement_difference<1e-8,(difference,measurement_difference)
    assert rig.table_guard.steps==row['steps']*10
    run.guard.assert_sources(manifest['source_sha256'])
    np.savez_compressed(dest/'states.npz',states=states,actual_tcp=actual,history=history,frame_index=np.arange(len(states)),frame_time=states[:,0])
    accepted={name:run.acceptance(history[1:],pos,pose,path) for name,pos,pose,path in run.TIERS}
    meta=dict(goal_name=name,goal_m=target.tolist(),directory=name,result=row,acceptance=accepted,
        width=1280,height=960,fps=50,frames=len(states),simulation_duration_s=float(data.time),playback_speed=1,
        capture='fresh full-16-second physical rerun, native render each 20 ms state; not stopped on endpoint acceptance',
        previous_saved_joint_state_max_difference=difference,measurement_max_difference=measurement_difference,
        trajectory_source=str(run.OUT/row['trace']),trajectory_source_sha256=run.guard.digest(run.OUT/row['trace']),
        scan_manifest_sha256=run.guard.digest(run.OUT/'manifest.json'),recording_source_sha256=run.guard.digest(__file__),
        source_sha256=manifest['source_sha256'],controller_audit=rig.audit,guard=rig.table_guard.summary(),
        codec='h264',crf=16,gop=12,b_frames=0)
    (dest/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(dict(goal=name,frames=len(states),qpos_difference=difference,measurement_difference=measurement_difference)),flush=True)
    return meta


def gallery():
    from build_retest_136_gallery import HTML
    episodes=[];checks=[]
    for x,y,name in GOALS:
        m=json.loads((OUT/name/'metadata.json').read_text());m['version']=-1 if y<0 else 1;m['pair_name']='near' if x==.6 else 'far'
        a=m['acceptance']['moderate'];f=m['result']['final']
        m['result']={**m['result'],**dict(endpoint_pass=a['endpoint'],motion_pass=a['motion'],reason='timeout',
            position_error_mm=f['position_mm'],pose_error_deg=f['pose_deg'],finger_floor_mm=f['finger_mm'])}
        episodes.append(m)
        for view in ('global','wrist'):
            path=OUT/name/f'{view}.mp4'
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames',
                '-show_entries','stream=width,height,avg_frame_rate,nb_frames,codec_name:frame=key_frame,best_effort_timestamp_time,pict_type','-of','json',str(path)],text=True))
            stream=info['streams'][0];frames=info['frames'];keys=[i for i,f in enumerate(frames) if f['key_frame']]
            gap=max([b-a for a,b in zip(keys,keys[1:])]+[len(frames)-keys[-1]])
            assert (stream['width'],stream['height'],stream['avg_frame_rate'],stream['codec_name'])==(1280,960,'50/1','h264')
            assert len(frames)==m['frames']==801 and gap<=12 and all(f['pict_type']!='B' for f in frames)
            assert max(abs(float(f['best_effort_timestamp_time'])-i/50) for i,f in enumerate(frames))<1e-6
            checks.append(dict(file=str(path.relative_to(OUT)),frames=801,width=1280,height=960,fps=50,gop=gap,qpos_difference=m['previous_saved_joint_state_max_difference'],uniform_20ms=True))
    html=HTML.replace('版本 1、3、6 · 工作范围实际录像','候选矩形 · 四个角点 · 增强6＋碰桌保护')
    html=html.replace('相同 home、目标、末端朝向和测试轨迹。','同一home与固定末端朝向，沿用范围补测的完整16秒协议。黄色框为X=0.60～0.84、Y=−0.22～0.30，黄点为当前目标。')
    html=re.sub(r'<select id="goal">.*?</select>','<select id="goal"><option value="near">X=0.60，Y=−0.22与+0.30</option><option value="far">X=0.84，Y=−0.22与+0.30</option></select>',html,count=1)
    html=html.replace('ids=[1,3,6]','ids=[-1,1]').replace('e.goal_name===goal','e.pair_name===goal').replace("e.goal_name===$('goal').value","e.pair_name===$('goal').value")
    html=html.replace('repeat(3,minmax(0,1fr))','repeat(2,minmax(0,1fr))').replace('三路同步进度','两角点同步进度').replace('暂停三路','暂停两路')
    note="'黄色矩形：X=0.60～0.84m、Y=−0.22～0.30m。四个边界角点单独物理测试；原最大矩形统计检查的是4cm格子中心，并非这些边界点。卡片按候选门槛：终点位置10mm/姿态3°、全程偏姿8°、速度比≤1.01、原安全条件及连续2秒保持。完整16秒录像，终点通过仍继续录制；卡片是结束统计，视频HUD是当前帧。原控制器、保护参数、限速、首帧上方参考后直接终点的协议均保持。拖动暂停两路，定位后恢复。'"
    html=re.sub(r"\$\('note'\)\.textContent=goal\.includes\('floor'\).*?;\$\('seek'\)\.value=0",f"$('note').textContent={note};$('seek').value=0",html,count=1)
    html=html.replace("vs().forEach(v=>{v.addEventListener('seeked'","vs().forEach(v=>{v.addEventListener('loadeddata',()=>v.removeAttribute('poster'));v.addEventListener('seeked'")
    html=re.sub(r'<footer>.*?</footer>','<footer><a href="../../recommended_workspace.png" target="_blank">期望工作区与矩形</a><a href="../../acceptance_comparison.png" target="_blank">门槛对比图</a><a href="video_verification.json" target="_blank">视频核验</a><a href="gallery_manifest.json" target="_blank">来源与角点验收</a></footer>',html,count=1)
    players=''.join(f'<article class="player"><div class="head"><strong>{label}</strong><span id="badge-{i}" class="badge"></span></div><video id="video-{i}" controls playsinline preload="auto" aria-label="{label}录像"></video><div class="meta" id="meta-{i}"></div></article>' for i,label in ((-1,'负Y角点：Y=−0.22'),(1,'正Y角点：Y=+0.30')))
    (OUT/'index.html').write_text(html.replace('__PLAYERS__',players).replace('__DATA__',json.dumps(episodes,ensure_ascii=False)),encoding='utf-8')
    (OUT/'gallery_manifest.json').write_text(json.dumps(episodes,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'video_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(dict(clips=len(checks),verified=True)),flush=True)



def prepare():
    run.OUT.mkdir(parents=True,exist_ok=True)
    (run.OUT/'traces').mkdir(exist_ok=True)
    frozen=json.loads((PARENT_BOX/'manifest.json').read_text())
    hashed={**frozen['source_sha256'],str(run.guard.Path(__file__).resolve()):run.guard.digest(__file__)}
    run.guard.assert_sources(hashed)
    manifest=dict(frozen,source_sha256=hashed,box_x_m=[.6,.84],box_y_m=[-.22,.3],
        grid_cell_centers_x_m=None,grid_cell_centers_y_m=None,total_box_area_m2=.1248,
        additional_corners=[(x,y) for x,y,name in GOALS],scope='four exact boundary corner points; not a new area scan')
    (run.OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    run.init(hashed)
    rows=[]
    for x,y,name in GOALS:
        r=run.worker((x,y));rows.append(r)
        h=np.load(run.OUT/r['trace'])['history']
        print(json.dumps(dict(goal=name,result=r['final'],moderate=run.acceptance(h,10,3,8))),flush=True)
    (run.OUT/'cells.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('record','gallery'),default='record');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='gallery':gallery()
    else:
        prepare()
        with mp.get_context('spawn').Pool(a.workers) as pool:list(pool.imap_unordered(record,GOALS,chunksize=1))
        gallery()
