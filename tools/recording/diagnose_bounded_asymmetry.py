"""Fresh mirror-pair diagnosis and native recordings for bounded enhanced6."""
import argparse
import json
import hashlib
import re
import subprocess
from collections import Counter
import numpy as np
import experiment_bounded_full6 as exp

OUT=exp.OUT/'asymmetry'
GOALS=((.55,-.36,'residual_negative'),(.55,.36,'residual_positive'),
       (.67,-.24,'joint_negative'),(.67,.24,'joint_positive'))


def diagnose():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((exp.OUT/'bounded/manifest.json').read_text())
    for path,digest in manifest['source_sha256'].items():
        assert hashlib.sha256(exp.Path(path).read_bytes()).hexdigest()==digest,path
    cells=json.loads((exp.OUT/'bounded/cells.json').read_text())
    stats={}
    for side,test in (('negative',lambda r:r['y_m']<0),('positive',lambda r:r['y_m']>0),('zero',lambda r:r['y_m']==0)):
        rows=[r for r in cells if test(r)];timeouts=[r for r in rows if r['reason']=='timeout']
        stats[side]=dict(total=len(rows),counts=dict(Counter(r['reason'] for r in rows)),
            timeout_final_violations=dict(position=sum(r['position_error_mm']>5 for r in timeouts),
            pose=sum(r['pose_error_deg']>2 for r in timeouts),joint=sum(r['joint_margin_deg']<0 for r in timeouts),
            fingers=sum(not .2<=r['finger_floor_mm']<=8 for r in timeouts)))
    results=[]
    for x,y,name in GOALS:
        rig=exp.ComparisonRig('bounded');frames=[];commands=[]
        original_action=rig.policy._ik_action
        def action(target,gripper):
            J,V,q,bias,limits,ranges=rig.task_data(target)
            cmd=original_action(target,gripper)
            velocity=(cmd[:7]-q)/.11
            lower=np.maximum(-limits,(ranges[:,0]-q)/.11)
            upper=np.minimum(limits,(ranges[:,1]-q)/.11)
            raw=J.T@np.linalg.solve(J@J.T+.04**2*np.eye(6),V)+bias
            commands.append(dict(q_before=q.tolist(),requested=cmd[:7].tolist(),velocity=velocity.tolist(),
                bound_active=(np.isclose(velocity,lower,atol=1e-7)|np.isclose(velocity,upper,atol=1e-7)).tolist(),
                task_velocity=V.tolist(),predicted_task_velocity=(J@velocity).tolist(),raw_velocity=raw.tolist()))
            return cmd
        rig.policy._ik_action=action
        original_control=rig.control
        def control(target,phase):
            original_control(target,phase)
            m=rig.measure(np.array([x,y,.005]),collision=False)
            q=rig.data.qpos[rig.env.arm_qpos_adr].copy()
            ranges=rig.model.jnt_range[rig.env.arm_joint_ids]
            margins=np.minimum(q-ranges[:,0],ranges[:,1]-q)
            frames.append(dict(t=float(rig.data.time),position_mm=m[0]*1000,pose_deg=m[2],
                q=q.tolist(),qvel=rig.data.qvel[rig.env.arm_dof_adr].tolist(),
                joint_margins_deg=np.degrees(margins).tolist(),applied=rig.data.ctrl[:7].tolist(),
                fingers_mm=min(float(rig.geometry.geom_vertices(g)[:,2].min()) for g in rig.geometry.finger_geoms)*1000))
        rig.control=control
        row=rig.trial(x,y)
        saved=next(r for r in cells if abs(r['x_m']-x)<1e-8 and abs(r['y_m']-y)<1e-8)
        assert row['reason']==saved['reason'] and row['steps']==saved['steps']
        for key in ('position_error_mm','pose_error_deg','joint_margin_deg'):
            assert abs(row[key]-saved[key])<1e-8
        last=frames[-1];tail=[f for f in frames if f['t']>=max(0,last['t']-4)]
        contacts=[]
        for contact in rig.data.contact:
            if contact.dist<=0:
                contacts.append(dict(geoms=[rig.model.geom(int(g)).name for g in contact.geom],distance_mm=float(contact.dist)*1000))
        summary=dict(name=name,goal_m=[x,y,.005],result=row,final=last,final_command=commands[-1],
            final_contacts=contacts,final_4s=dict(position_min_mm=min(f['position_mm'] for f in tail),
                position_max_mm=max(f['position_mm'] for f in tail),pose_min_deg=min(f['pose_deg'] for f in tail),
                pose_max_deg=max(f['pose_deg'] for f in tail),max_abs_joint_speed=max(max(abs(v) for v in f['qvel']) for f in tail)),
            active_joint_steps=np.asarray([c['bound_active'] for c in commands]).sum(axis=0).tolist())
        (OUT/f'{name}_history.json').write_text(json.dumps(dict(frames=frames,commands=commands)),encoding='utf-8')
        results.append(summary)
        print(json.dumps(summary),flush=True)
    (OUT/'diagnostic.json').write_text(json.dumps(dict(stats=stats,selected=results,source_hashes_verified=True),indent=2),encoding='utf-8')


def videos():
    from record_retest_136_videos import capture
    for x,y,name in GOALS:
        def factory(directory):
            rig=exp.ComparisonRig('bounded');return rig,rig.env,exp.base.native
        capture(6,x,y,name,rig_factory=factory,output_root=OUT/'videos',
            scan_directory=exp.OUT/'bounded',label='增强6 · 带约束DLS')


def gallery():
    from build_retest_136_gallery import HTML
    dest=OUT/'videos';episodes=[];checks=[]
    for x,y,name in GOALS:
        directory=dest/'version_6'/name
        m=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        assert m['version']==6 and m['previous_saved_joint_state_max_difference']==0
        assert m['scan_manifest_sha256']==hashlib.sha256((exp.OUT/'bounded/manifest.json').read_bytes()).hexdigest()
        m.update(directory=str(directory.relative_to(dest)).replace('\\','/'),side=1 if y>0 else -1,pair_name=name.split('_')[0])
        episodes.append(m)
        for view in ('global','wrist'):
            file=directory/f'{view}.mp4'
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames',
                '-show_entries','stream=width,height,avg_frame_rate,nb_frames,codec_name:frame=key_frame,best_effort_timestamp_time,pict_type','-of','json',str(file)],text=True))
            s=info['streams'][0];frames=info['frames']
            assert (s['width'],s['height'],s['avg_frame_rate'],s['codec_name'])==(1280,960,'50/1','h264')
            assert len(frames)==int(s['nb_frames'])==m['frames']
            assert max(abs(float(f['best_effort_timestamp_time'])-i/50) for i,f in enumerate(frames))<1e-6
            keys=[i for i,f in enumerate(frames) if f['key_frame']]
            gop=max([b-a for a,b in zip(keys,keys[1:])]+[len(frames)-keys[-1]])
            assert gop<=12 and all(f['pict_type']!='B' for f in frames)
            checks.append(dict(file=str(file.relative_to(dest)).replace('\\','/'),frames=len(frames),fps=50,
                width=1280,height=960,max_gop=gop,trajectory_difference=0))
    html=HTML.replace('版本 1、3、6 · 工作范围实际录像','增强6 · 带约束DLS · 正负Y失败对照')
    html=html.replace('相同 home、目标、末端朝向和测试轨迹。',
        '两侧均为带约束增强6，使用当前范围图的相同控制器与测试协议：同一home、固定完整朝向、原有限速、kp×4/kv×2、重力前馈、姿态增益10。上方参考只发送一帧，随后直接发送终点。每个目标最多16秒，需在时限内完成连续2秒的终点保持。')
    html=re.sub(r'<select id="goal">.*?</select>',
        '<select id="goal"><option value="residual">残余位置误差：(0.55, ±0.36, 0.005) m</option><option value="joint">实际关节越界：(0.67, ±0.24, 0.005) m</option></select>',html,count=1)
    html=html.replace('ids=[1,3,6]','ids=[1,-1]').replace('e.version===id','e.side===id')
    html=html.replace('e.goal_name===goal','e.pair_name===goal').replace("e.goal_name===$('goal').value","e.pair_name===$('goal').value")
    html=html.replace('repeat(3,minmax(0,1fr))','repeat(2,minmax(0,1fr))')
    html=html.replace('三路同步进度','正负Y同步进度').replace('暂停三路','暂停两路')
    html=html.replace("vs().forEach(v=>{v.addEventListener('seeked'", "vs().forEach(v=>{v.addEventListener('loadeddata',()=>v.removeAttribute('poster'));v.addEventListener('seeked'")
    html=html.replace('°${floor}', '°<br>实际关节限位裕量 ${e.result.joint_margin_deg.toFixed(3)}°（负值＝越界）${floor}')
    note="(goal==='residual'?'负Y在关节3/6约束与夹指桌面接触状态下停住，最终位置误差22.57mm；其镜像正Y终点通过。':'负Y末端位置1.27mm、姿态0.35°已合格，但实际关节6低于下限约0.124°，终点保持无法累计；其镜像正Y终点通过。')+' 超时指16秒仿真时限内未完成连续2秒全部条件合格的保持。单独满足末端位置/姿态不等于终点通过。卡片是结束统计，视频HUD是当前帧。短片停末帧。'"
    html=re.sub(r"\$\('note'\)\.textContent=goal\.includes\('floor'\).*?;\$\('seek'\)\.value=0",f"$('note').textContent={note};$('seek').value=0",html,count=1)
    html=re.sub(r'<footer>.*?</footer>',
        '<footer><a href="../../workspace_comparison.png" target="_blank">当前范围图</a><a href="../diagnostic.json" target="_blank">本次逐点诊断</a><a href="../../README.md" target="_blank">测试协议</a><a href="video_verification.json" target="_blank">录像核查</a></footer>',html,count=1)
    players=''.join(f'<article class="player"><div class="head"><strong>{label}</strong><span id="badge-{side}" class="badge"></span></div><video id="video-{side}" controls playsinline preload="auto" aria-label="{label}录像"></video><div class="meta" id="meta-{side}"></div></article>' for side,label in ((1,'增强6 · 正Y'),(-1,'增强6 · 负Y')))
    (dest/'index.html').write_text(html.replace('__PLAYERS__',players).replace('__DATA__',json.dumps(episodes,ensure_ascii=False)),encoding='utf-8')
    (dest/'gallery_manifest.json').write_text(json.dumps(episodes,indent=2,ensure_ascii=False),encoding='utf-8')
    (dest/'video_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(dict(clips=len(checks),all_checks_passed=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--videos',action='store_true');parser.add_argument('--gallery',action='store_true');args=parser.parse_args()
    gallery() if args.gallery else videos() if args.videos else diagnose()
