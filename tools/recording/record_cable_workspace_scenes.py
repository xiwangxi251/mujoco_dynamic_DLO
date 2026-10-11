"""Cable-only motion visualization with an explicitly fixed NERO ready pose.

Separate visualization: no policy, no grasp evaluation, no cable repositioning.
The robot coordinates are restored after each 2 ms physics step by request.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from panda_cable_grasp.env.environment import CableGraspEnv,EnvConfig,XML_PATH
from panda_cable_grasp.scenarios.registry import CORE_SCENARIO_NAMES,get_scenario
sys.path.insert(0,str(ROOT.parent/'worktrees/scripted-tracking-20261001/tools/recording'))
from human_episode_recording import RGBEncoder

OUT=ROOT/'outputs/vertical_workspace/scripted_136_retest_20261004/cable_workspace_scenes'
SEED=20260804
NAMES=('静态','局部形变','整体运动 L1','整体运动 L2','运动＋形变 L1','运动＋形变 L2')
RECT=np.array([[.60,-.22,.006],[.84,-.22,.006],[.84,.30,.006],[.60,.30,.006]])
VIEWS=('top','global','wrist')
STATE=mujoco.mjtState.mjSTATE_FULLPHYSICS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def make_env(scene):
    registered=get_scenario(scene)
    config=EnvConfig(robot='nero',seed=SEED,scenario_name=scene,
        scenario_id=registered.scenario_id,scenario_split=registered.split.value,
        **registered.to_env_overrides())
    env=CableGraspEnv(config);env.reset(seed=SEED)
    return env

def state(env):
    value=np.empty(mujoco.mj_stateSize(env.model,STATE))
    mujoco.mj_getState(env.model,env.data,value,STATE)
    return value

def inside(points):
    return (points[...,0]>=.6)&(points[...,0]<=.84)&(points[...,1]>=-.22)&(points[...,1]<=.30)

def simulate(scene):
    dest=OUT/scene;dest.mkdir(parents=True,exist_ok=True)
    previous=np.load(dest/'trajectory.npz')['states'].copy() if (dest/'trajectory.npz').exists() else None
    env=make_env(scene);m,d=env.model,env.data
    qadr=np.r_[env.arm_qpos_adr,[m.jnt_qposadr[m.joint(n).id] for n in env.robot_spec.finger_joint_names]]
    dadr=np.r_[env.arm_dof_adr,[m.jnt_dofadr[m.joint(n).id] for n in env.robot_spec.finger_joint_names]]
    fixed=d.qpos[qadr].copy();arm_bodies=np.array([m.body(f'link{i}').id for i in range(1,8)])
    initial_body=d.xpos[arm_bodies].copy()
    values=[state(env)];points=[d.xpos[env.cable_ids].copy()];times=[float(d.time)]
    max_pose=max_body=0.;robot_cable_contacts=0;reason='15s_limit'
    cable_bodies=set(env.cable_ids);robot_bodies=set(range(1,env.hand_id+1))
    robot_bodies.update(m.body(n).id for n in (env.robot_spec.left_finger_body_name,env.robot_spec.right_finger_body_name))
    for step in range(750):
        for sub in range(10):
            d.xfrc_applied[:]=0.;env._apply_cable_disturbance()
            d.ctrl[:]=env.ready_ctrl
            mujoco.mj_step(m,d)
            # Requested visualization fixture; cable DOFs and forces unchanged.
            d.qpos[qadr]=fixed;d.qvel[dadr]=0.;d.qacc_warmstart[dadr]=0.
            mujoco.mj_forward(m,d)
            max_pose=max(max_pose,float(np.max(abs(d.qpos[qadr]-fixed))))
            max_body=max(max_body,float(np.max(abs(d.xpos[arm_bodies]-initial_body))))
            for c in d.contact:
                a,b=(int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2))
                if (a in cable_bodies and b in robot_bodies) or (b in cable_bodies and a in robot_bodies):robot_cable_contacts+=1
        assert np.all(np.isfinite(d.qpos))
        values.append(state(env));points.append(d.xpos[env.cable_ids].copy());times.append(float(d.time))
        if (step+1)%250==0:print(json.dumps(dict(simulating=scene,frame=step+1,time_s=round(float(d.time),2))),flush=True)
        if env.rigid_motion_finished:
            reason='original_rigid_exit_line';break
    points=np.asarray(points);times=np.asarray(times);values=np.asarray(values)
    target=env.cable_ids.index(env.target_body_id)
    mask=inside(points);target_mask=mask[:,target]
    previous_difference=float(np.max(abs(values-previous))) if previous is not None else None
    if previous is not None:assert previous_difference==0
    np.savez_compressed(dest/'trajectory.npz',states=values,cable_xyz=points,frame_times=times,
        frame_index=np.arange(len(times)),robot_qpos=fixed,robot_qpos_adr=qadr,robot_dof_adr=dadr,
        cable_body_ids=env.cable_ids,target_node=target)
    metadata=dict(scene=scene,label=NAMES[CORE_SCENARIO_NAMES.index(scene)],seed=SEED,
        directory=scene,frames=len(times),duration_s=float(times[-1]),termination=reason,
        config=asdict(env.config),registered_scenario=get_scenario(scene).asdict(),
        target_node=target,initial_target_xyz=points[0,target].tolist(),
        robot_fixed_pose='original NERO ready; open fingers; explicit projection every 2 ms',
        robot_qpos_max_change=max_pose,robot_link_position_max_change_m=max_body,
        previous_simulation_state_max_difference=previous_difference,
        robot_cable_contact_substep_pairs=robot_cable_contacts,
        whole_cable_xy_min=points[...,:2].min(axis=(0,1)).tolist(),
        whole_cable_xy_max=points[...,:2].max(axis=(0,1)).tolist(),
        target_xy_min=points[:,target,:2].min(axis=0).tolist(),target_xy_max=points[:,target,:2].max(axis=0).tolist(),
        target_inside_recorded_frame_fraction=float(target_mask.mean()),
        any_node_inside_recorded_frame_fraction=float(mask.any(axis=1).mean()),
        mean_nodes_inside_fraction=float(mask.mean()),
        overlap_definition='XY node-center inclusion only; not reachability or task success',
        trajectory_sha256=sha(dest/'trajectory.npz'),source_sha256=source_hashes(),
        width=1280,height=960,fps=50,views=VIEWS,native_render=True,
        interpolation=False,playback_speed=1,physics_timestep_s=float(m.opt.timestep))
    assert max_pose==0 and max_body==0
    (dest/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    env.close();print(json.dumps(dict(prepared=scene,frames=len(times),duration=times[-1],arm_motion=0)),flush=True)
    return metadata

def source_hashes():
    files=[Path(__file__),ROOT/'src/panda_cable_grasp/env/environment.py',ROOT/'src/panda_cable_grasp/env/objects.py',
        ROOT/'src/panda_cable_grasp/scenarios/registry.py',XML_PATH,ROOT/'assets/mujoco/nero/nero.xml',
        ROOT.parent/'worktrees/scripted-tracking-20261001/tools/recording/human_episode_recording.py']
    return {str(p):sha(p) for p in files}

def connector(renderer,a,b,radius,rgba):
    g=renderer.scene.geoms[renderer.scene.ngeom]
    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.asarray(rgba))
    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,a,b);renderer.scene.ngeom+=1

def render(scene):
    dest=OUT/scene;meta=json.loads((dest/'metadata.json').read_text(encoding='utf-8'))
    assert source_hashes()==meta['source_sha256'] and sha(dest/'trajectory.npz')==meta['trajectory_sha256']
    env=make_env(scene);m,d=env.model,env.data
    saved=np.load(dest/'trajectory.npz');states=saved['states'];points=saved['cable_xyz'];times=saved['frame_times'];target=int(saved['target_node'])
    assert np.array_equal(state(env),states[0])
    marker=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'cable_marker')
    if marker>=0:m.site_rgba[marker,3]=0
    m.vis.global_.offwidth=max(m.vis.global_.offwidth,1280);m.vis.global_.offheight=max(m.vis.global_.offheight,960)
    renderer=mujoco.Renderer(m,width=1280,height=960,max_geom=10000)
    global_camera=mujoco.MjvCamera();global_camera.lookat[:]=(.61,0,.12)
    global_camera.distance=2.10;global_camera.azimuth=135;global_camera.elevation=-38
    top_camera=mujoco.MjvCamera();top_camera.lookat[:]=(.65,0,0)
    top_camera.distance=2.10;top_camera.azimuth=90;top_camera.elevation=-90
    option=mujoco.MjvOption();option.geomgroup[3]=0
    local_rotation=np.empty(9);mujoco.mju_quat2Mat(local_rotation,np.asarray(env.config.dynamicvla_wrist_camera_quat));local_rotation=local_rotation.reshape(3,3)
    local_pos=np.asarray(env.config.dynamicvla_wrist_camera_pos)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    writers={v:RGBEncoder(dest/f'{v}.mp4',1280,960,50) for v in VIEWS}
    max_diff=0.;start=time.monotonic()
    try:
        for index,stored in enumerate(states):
            mujoco.mj_setState(m,d,stored,STATE);mujoco.mj_forward(m,d)
            max_diff=max(max_diff,float(np.max(abs(state(env)-stored))))
            assert np.max(abs(d.xpos[env.cable_ids]-points[index]))<1e-12
            node_mask=inside(points[index]);p=points[index,target]
            for view,writer in writers.items():
                renderer.update_scene(d,camera=top_camera if view=='top' else global_camera,scene_option=option)
                if view=='wrist':
                    hand=d.xmat[env.hand_id].reshape(3,3);rotation=hand@local_rotation
                    for c in renderer.scene.camera:
                        c.pos[:]=d.xpos[env.hand_id]+hand@local_pos;c.forward[:]=-rotation[:,2];c.up[:]=rotation[:,1]
                for a,b in zip(RECT,np.roll(RECT,-1,axis=0)):connector(renderer,a,b,.002,(1,.76,.03,1))
                # Material target trajectory, accumulated only through this frame.
                trail=points[:index+1:5,target].copy();trail[:,2]=.008
                for a,b in zip(trail[:-1],trail[1:]):
                    if np.linalg.norm(a-b)>1e-7:connector(renderer,a,b,.0008,(.0,.77,.9,.6))
                g=renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.010,0,0]),p+np.array([0,0,.008]),np.eye(3).ravel(),np.array([1,.16,.09,1]));renderer.scene.ngeom+=1
                im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                draw.rounded_rectangle((16,14,1264,135),radius=10,fill='#f6f9fc')
                label={'top':'俯视','global':'全局','wrist':'腕部'}[view]
                draw.text((30,20),f"{meta['label']} · {label} · 机械臂固定 · seed {SEED} · 1×",font=font,fill='#20334b')
                draw.text((30,60),f't={times[index]:.2f}s | 目标节点 {target} XY=({p[0]:.3f}, {p[1]:+.3f})m | '+('框内' if node_mask[target] else '框外')+f' | 框内节点 {node_mask.sum()}/{len(node_mask)}',font=small,fill='#334d66')
                draw.text((30,96),'黄框：X 0.60～0.84 / Y −0.22～0.30 m；红点：原目标节点；青线：其已运动轨迹',font=small,fill='#334d66')
                writer.write(np.asarray(im))
                if index==0:im.save(dest/f'{view}_poster.png')
                if view=='top' and index in (100,250,400):im.save(dest/f'top_{index:04d}.png')
            if index%250==0:print(json.dumps(dict(render=scene,frame=index,total=len(states),wall_s=round(time.monotonic()-start,1))),flush=True)
    finally:
        for writer in writers.values():writer.release()
        renderer.close();env.close()
    assert max_diff==0 and source_hashes()==meta['source_sha256']
    checks=[]
    for view in VIEWS:
        result=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames',
            '-show_entries','stream=width,height,avg_frame_rate,nb_frames,codec_name:frame=key_frame,best_effort_timestamp_time,pict_type','-of','json',str(dest/f'{view}.mp4')],text=True))
        stream=result['streams'][0];frames=result['frames'];keys=[i for i,f in enumerate(frames) if f['key_frame']]
        gap=max([b-a for a,b in zip(keys,keys[1:])]+[len(frames)-keys[-1]])
        assert (stream['width'],stream['height'],stream['avg_frame_rate'],stream['codec_name'])==(1280,960,'50/1','h264')
        assert len(frames)==len(states) and gap<=12 and all(f['pict_type']!='B' for f in frames)
        assert max(abs(float(f['best_effort_timestamp_time'])-i/50) for i,f in enumerate(frames))<1e-6
        checks.append(dict(view=view,frames=len(frames),maximum_gop_frames=gap,uniform_20ms=True))
    report=dict(rendered_fullphysics_state_max_difference=max_diff,robot_qpos_max_change=meta['robot_qpos_max_change'],
        trajectory_sha256=meta['trajectory_sha256'],width=1280,height=960,fps=50,encoding=checks,
        capture='native rendering of newly simulated saved 20 ms states; no interpolation')
    (dest/'video_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(render_done=scene,frames=len(states),views=3,state_difference=max_diff)),flush=True)
    return report

def overview():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,3,figsize=(16,10),sharex=True,sharey=True)
    for scene,ax in zip(CORE_SCENARIO_NAMES,axes.flat):
        saved=np.load(OUT/scene/'trajectory.npz');p=saved['cable_xyz'];target=int(saved['target_node'])
        ax.scatter(p[::5,:,0].ravel(),p[::5,:,1].ravel(),s=2,alpha=.10,color='#1686bb')
        ax.plot(p[:,target,0],p[:,target,1],color='#e85039',lw=1.4)
        ax.add_patch(Rectangle((.6,-.22),.24,.52,fill=False,edgecolor='#c49a00',lw=2.5))
        ax.scatter(.35,0,color='#333',marker='s',s=25)
        ax.set_title(scene.replace('id_',''));ax.set_xlim(-.10,1.30);ax.set_ylim(-.85,.85);ax.set_aspect('equal');ax.grid(alpha=.2)
        ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
    fig.suptitle('Current cable placement | blue: all node paths; red: target path; yellow: candidate rectangle\nFixed robot, seed 20260804; XY projection only',fontsize=16)
    fig.tight_layout();fig.savefig(OUT/'motion_overview.png',dpi=160);plt.close(fig)

def gallery():
    rows=[json.loads((OUT/s/'metadata.json').read_text(encoding='utf-8')) for s in CORE_SCENARIO_NAMES]
    for s in CORE_SCENARIO_NAMES:assert (OUT/s/'video_verification.json').exists()
    (OUT/'gallery_manifest.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    html=GALLERY.replace('__DATA__',json.dumps(rows,ensure_ascii=False))
    cards=''.join(f'<article class="player" id="card-{i}"><h2>{r["label"]}</h2><video id="video-{i}" controls playsinline preload="auto" aria-label="{r["label"]}录像"></video><p id="meta-{i}"></p></article>' for i,r in enumerate(rows))
    (OUT/'index.html').write_text(html.replace('__PLAYERS__',cards),encoding='utf-8')
    overview()

GALLERY=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>线缆六场景与工作区</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0b1320;color:#e9eef5;font-family:Segoe UI,Microsoft YaHei,sans-serif}main{max-width:1900px;margin:auto;padding:22px}h1{font-size:26px}h2{font-size:17px;margin:12px}p,.note{color:#bac8da;font-size:14px;line-height:1.7}button,select{padding:10px 14px;border:1px solid #435875;border-radius:8px;color:#eef4ff;background:#20314a;cursor:pointer;font:inherit}.toolbar,.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}.players{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.player{background:#152237;border:1px solid #30435e;border-radius:12px;overflow:hidden}.player p{margin:12px}video{display:block;width:100%;aspect-ratio:4/3;background:#070d16}.timeline{display:flex;align-items:center;gap:12px;flex:1;min-width:270px}input{width:100%}output{white-space:nowrap;font-size:14px}.note{border:1px solid #30435e;padding:14px;border-radius:8px}a{color:#a9ccff}.focus{position:fixed;inset:0;overflow:auto;background:#080f1b;z-index:100;padding:18px}.focus .players{grid-template-columns:repeat(2,minmax(0,1fr))}.focus .player p{display:none}[hidden]{display:none!important}@media(max-width:1000px){.players{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.players,.focus .players{grid-template-columns:1fr}}label{display:flex;align-items:center;gap:8px}</style>
<main><h1>线缆六场景与候选工作区</h1><p>机械臂固定在原 ready 姿态、夹爪张开；线缆沿用当前场景的位置、驱动和材料。黄色框：X=0.60～0.84、Y=−0.22～0.30 m。每个场景同一 seed，原生 1280×960 / 50 fps / 1×播放。</p>
<div class="toolbar"><label>场景 <select id="group"><option value="all">全部六场景</option><option value="shape">静态 / 形变</option><option value="rigid">整体运动 L1 / L2</option><option value="combined">叠加运动 L1 / L2</option></select></label><label>视角 <select id="view"><option value="top">俯视（便于比范围）</option><option value="global">全局</option><option value="wrist">腕部</option></select></label></div>
<section id="viewing"><div class="players">__PLAYERS__</div><div class="controls"><button id="play">同步播放</button><button id="pause">暂停</button><button id="restart">从头播放</button><button id="expand">放大对比</button><label>速度 <select id="rate"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><div class="timeline"><input id="seek" type="range" min="0" max="15" value="0" step="0.02" aria-label="场景同步进度"><output id="clock">0.00 / 15.00 s</output></div></div></section>
<div class="note" id="note">红点为该回合原本选中的目标节点，青线是它截至当前帧的运动轨迹。框内节点比例只描述 XY 重叠，不代表抓取成功。静态和形变最多15秒；整体运动场景沿用越过原出口线即结束，短视频停在末帧。这里展示一个种子的活动范围，不是全部随机回合的包络。</div><p><a href="motion_overview.png" target="_blank">六场景完整运动范围俯视图</a> · <a href="gallery_manifest.json" target="_blank">场景与来源记录</a></p></main>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),groups={all:[0,1,2,3,4,5],shape:[0,1],rigid:[2,3],combined:[4,5]};let ids=groups.all;const vs=()=>ids.map(i=>$('video-'+i));
let scrubbing=false,resumeAfter=false,pending=null,timer=0,lastDispatch=0;
function pause(){vs().forEach(v=>v.pause())}
function load(){D.forEach((e,i)=>$('video-'+i).pause());clearTimeout(timer);timer=0;pending=null;scrubbing=false;resumeAfter=false;ids=groups[$('group').value];const view=$('view').value;D.forEach((e,i)=>{$('card-'+i).hidden=!ids.includes(i);const v=$('video-'+i);if(ids.includes(i)){v.src=e.directory+'/'+view+'.mp4';v.poster=e.directory+'/'+view+'_poster.png';v.playbackRate=Number($('rate').value);v.load()}$('meta-'+i).innerHTML=`seed ${e.seed} · ${e.duration_s.toFixed(2)}秒 · 原目标节点 ${e.target_node}<br>目标在框内帧比例 ${(100*e.target_inside_recorded_frame_fraction).toFixed(1)}% · 平均框内节点 ${(100*e.mean_nodes_inside_fraction).toFixed(1)}%<br><span id="time-${i}">当前 t=0.00s</span> · <a href="${e.directory}/metadata.json" target="_blank">场景记录</a>`});$('seek').max=Math.max(...ids.map(i=>D[i].duration_s+.02));$('seek').value=0;document.querySelector('.players').style.gridTemplateColumns=ids.length===2?'repeat(2,minmax(0,1fr))':'';const u=new URL(location.href);u.searchParams.set('group',$('group').value);u.searchParams.set('view',view);history.replaceState(null,'',u)}
async function play(){await Promise.allSettled(vs().filter(v=>v.currentTime<v.duration-.021).map(v=>v.play()))}
function finish(){if(pending!==null||vs().some(v=>v.seeking))return;if(!scrubbing&&resumeAfter){resumeAfter=false;play()}}
function pump(){if(timer||pending===null||vs().some(v=>v.seeking||v.readyState<1)){finish();return}const delay=Math.max(0,80-(performance.now()-lastDispatch));if(delay){timer=setTimeout(()=>{timer=0;pump()},delay);return}const t=pending;pending=null;lastDispatch=performance.now();$('seek').dataset.appliedTarget=String(t);$('seek').dataset.dispatches=String(Number($('seek').dataset.dispatches||0)+1);vs().forEach(v=>{const target=Math.min(t,Math.max(0,v.duration-.02));if(Math.abs(v.currentTime-target)>.009)v.currentTime=target});finish()}
function begin(){if(scrubbing)return;scrubbing=true;resumeAfter=vs().some(v=>!v.paused&&!v.ended);pause()}
function request(){begin();pending=Number($('seek').value);$('seek').dataset.target=String(pending);pump()}
function end(){scrubbing=false;pump();finish()}
$('group').addEventListener('change',load);$('view').addEventListener('change',load);$('play').addEventListener('click',play);$('pause').addEventListener('click',()=>{resumeAfter=false;pause()});$('restart').addEventListener('click',()=>{pause();resumeAfter=true;pending=0;scrubbing=false;pump()});$('rate').addEventListener('change',()=>vs().forEach(v=>v.playbackRate=Number($('rate').value)));$('seek').addEventListener('pointerdown',begin);$('seek').addEventListener('input',request);for(const event of ['change','pointerup','pointercancel','blur'])$('seek').addEventListener(event,end);
D.forEach((e,i)=>{const v=$('video-'+i);v.addEventListener('seeked',()=>{pump();finish()});v.addEventListener('loadedmetadata',pump);v.addEventListener('loadeddata',()=>v.removeAttribute('poster'));v.addEventListener('error',()=>{$('note').textContent='视频加载失败，请刷新。'})});
$('expand').addEventListener('click',()=>{$('viewing').classList.toggle('focus');$('expand').textContent=$('viewing').classList.contains('focus')?'退出大画面':'放大对比'});document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('viewing').classList.remove('focus');$('expand').textContent='放大对比'}});
function tick(){const t=Math.max(...vs().map(v=>v.currentTime||0));ids.forEach(i=>$('time-'+i).textContent=`当前 t=${Math.min($('video-'+i).currentTime||0,D[i].duration_s).toFixed(2)}s`);if(!scrubbing&&pending===null&&!vs().some(v=>v.seeking))$('seek').value=t;$('clock').textContent=`${Number($('seek').value).toFixed(2)} / ${Number($('seek').max).toFixed(2)} s`;requestAnimationFrame(tick)}const params=new URLSearchParams(location.search);if(groups[params.get('group')])$('group').value=params.get('group');if(['top','global','wrist'].includes(params.get('view')))$('view').value=params.get('view');load();tick();
</script></html>'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('prepare','render','gallery','all'),default='all');p.add_argument('--workers',type=int,default=2);p.add_argument('--scenes',nargs='+',default=list(CORE_SCENARIO_NAMES));a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.mode in ('prepare','all'):
        manifest=dict(scenes=list(CORE_SCENARIO_NAMES),seed=SEED,workspace_x_m=[.6,.84],workspace_y_m=[-.22,.3],
            scope='current placement visualization, no grasp policy or success evaluation',
            robot_freeze='9 robot qpos restored, corresponding qvel/warmstart set zero every 2ms; cable DOFs untouched',
            source_sha256=source_hashes(),nominal_cable_parameters_unchanged=True,positions_unchanged=True)
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:list(pool.map(simulate,a.scenes))
    if a.mode in ('render','all'):
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:list(pool.map(render,a.scenes))
    if a.mode in ('gallery','all'):gallery()

if __name__=='__main__':main()
