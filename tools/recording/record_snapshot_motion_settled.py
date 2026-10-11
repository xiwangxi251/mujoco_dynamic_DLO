"""Record 3-second static waits relative to the single 0.2-second baseline."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from record_snapshot_motion import (
    mujoco,np,Image,ImageDraw,ImageFont,CableGraspEnv,EnvConfig,configure_profile,
    get_scenario,project_camera_manifest,physical_state,verify_video,digest,project,clip_segment,
    save,gallery as original_gallery)

SOURCE=ROOT/'outputs/snapshot_motion_3s_100_20261010/parallel/samples.json'
BANK=ROOT/'outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks'


def gallery(output,episodes,selection='first'):
    original_gallery(output,episodes)
    p=output/'index.html';page=p.read_text(encoding='utf-8')
    page=page.replace('Math.min(.82,','Math.min(3.0,').replace("$('seek').max=.82","$('seek').max=3.0")
    title='位移超过15mm的8个快照 · 从0.2秒起算' if selection=='gt15' else '静置3秒位移 · 从0.2秒起算'
    description='初始穿透≤5mm、0.2–3秒最大节点位移>15mm的8个快照（35%与65%源帧）' if selection=='gt15' else '同样前10个快照（50条轨迹库、35%时刻）'
    page=page.replace('初始快照位移 · 10个样本',title)
    page=re.sub(r'<div class="subtitle">.*?</div>',
        f'<div class="subtitle">{description}；机械臂保持不动、夹爪张开，录像0—3秒。仅以0.2秒各节点位置为基准，此前不计量。项目全局＋腕部相机，1280×960、50fps。</div>',page,count=1)
    table='<table><tr><th>样本</th><th>快照比例／源帧</th><th>初始穿透 mm</th><th>0.2–3秒最大位移 mm</th></tr>'+''.join(
        f'<tr><td>seed={e["seed"]}</td><td>{e["fraction"]:.0%} / {e["snapshot_step"]}</td><td>{e["penetration_mm"]:.2f}</td><td>{e["displacement_mm"]:.2f}</td></tr>' for e in episodes)+'</table>'
    page=re.sub(r'<table>.*?</table>',lambda _:table,page,count=1,flags=re.S)
    for i,e in enumerate(episodes):
        page=page.replace(f'样本 {i+1} · seed {e["seed"]}',f'样本 {i+1} · seed {e["seed"]} · {e["fraction"]:.0%} / 帧{e["snapshot_step"]}')
    page=page.replace('等待期间最大位移 ${e.displacement_mm.toFixed(2)} mm',
        '0.2–3秒最大位移 ${e.displacement_mm.toFixed(2)} mm')
    page=re.sub(r'<p id="note" class="note">.*?</p>',
        '<p id="note" class="note">仅以0.2秒为基准，显示当前最大节点位移和截至当前时刻的历史最大值。0.2秒之前为WAIT。青色圆圈：当前最大位移节点在0.2秒的位置；黄色：同一节点当前位置，连线表示位移。位移为三维距离，同时显示XYZ分量。新录像逐帧核对3秒重算时保存的节点位置。支持0.25倍速。</p>',page,count=1)
    p.write_text(page,encoding='utf-8')


def record(row,output):
    seed=int(row['seed']);fraction=float(row['snapshot_fraction']);frame_index=int(row['snapshot_step'])
    dest=output/f'seed_{seed}_frame_{frame_index}'
    if (dest/'metadata.json').exists():
        m=json.loads((dest/'metadata.json').read_text(encoding='utf-8'))
        assert m['source_hash']==digest(Path(__file__))
        return m['episode']
    dest.mkdir(parents=True,exist_ok=True)
    trajectory=ROOT/'outputs/snapshot_motion_3s_100_20261010'/f'seed_{seed}_frame_{frame_index}.npz'
    if not trajectory.exists():
        candidates=list(SOURCE.parent.glob(f'shard_*/result/{trajectory.name}'))
        assert len(candidates)==1,f'Expected one saved trajectory for {seed}/{frame_index}, found {len(candidates)}'
        trajectory=candidates[0]
    with np.load(trajectory) as data: expected=data['cable_positions'].copy();expected_times=data['frame_times'].copy()
    scene=get_scenario('dev_static_midshape_v1_x075_v1');overrides=scene.to_env_overrides()
    overrides.update(initial_shape_time_range=(fraction,fraction),arm_motion_start_delay=.82)
    cfg=configure_profile(EnvConfig(robot='nero',seed=seed,replay_bank_dir=str(BANK),replay_seed_fallback='error',
        episode_seconds=3.1,dynamicvla_cameras_enabled=True,**overrides),'enhanced6_bounded_dls_v1')
    cfg.table_execution_guard_enabled=True
    env=CableGraspEnv(cfg);renderer=None;writers={}
    try:
        env.reset(seed=seed);assert int(env._initial_shape_step)==frame_index
        env.model.vis.global_.offwidth=1280;env.model.vis.global_.offheight=960
        renderer=mujoco.Renderer(env.model,height=960,width=1280)
        cameras={'global':cfg.dynamicvla_opst_camera_name,'wrist':cfg.dynamicvla_wrist_camera_name}
        option=mujoco.MjvOption();option.geomgroup[3]=0
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
        for view in cameras:
            writers[view]=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
                '-video_size','1280x960','-framerate','50','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','16',
                '-threads','1','-g','12','-keyint_min','12','-sc_threshold','0','-bf','0','-pix_fmt','yuv420p','-movflags','+faststart',
                str(dest/f'{view}.mp4')],stdin=subprocess.PIPE)
        baselines={};hist={.2:0.};states=[];times=[];points=[];metrics=[];render_difference=0.;position_error=0.
        for frame in range(151):
            if frame:
                _,_,terminated,truncated,_=env.step(env.ready_ctrl.copy());assert not terminated and not truncated
            before=physical_state(env);xyz=env.data.xpos[env.cable_ids].copy();t=float(env.data.time)
            assert abs(t-expected_times[frame])<1e-8
            position_error=max(position_error,float(np.max(np.abs(xyz-expected[frame])))*1000.)
            assert position_error<1e-5,f'Node positions differ from source: {position_error} mm'
            if frame==10:baselines[.2]=xyz.copy()
            frame_metrics={'time_s':t};labels={};selected=None
            for b in (.2,):
                if b not in baselines:
                    labels[b]=f'BASE t={b:.2f}s | WAIT: measurement starts at {b:.2f}s';continue
                delta=(xyz-baselines[b])*1000.;norms=np.linalg.norm(delta,axis=1)
                node=int(np.argmax(norms));current=float(norms[node]);hist[b]=max(hist[b],current)
                frame_metrics[str(b)]=dict(current_max_node=node,current_max_mm=current,historical_max_mm=hist[b],delta_xyz_mm=delta[node].tolist())
                labels[b]=f'BASE t={b:.2f}s | current max {current:.2f} mm (node {node}) | historical max {hist[b]:.2f} mm'
                selected=(node,delta[node])
            states.append(before);times.append(t);points.append(xyz);metrics.append(frame_metrics)
            for view,camera in cameras.items():
                renderer.update_scene(env.data,camera=camera,scene_option=option)
                img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img)
                if selected is not None:
                    node,d=selected
                    a=project(renderer.scene,baselines[.2][node],1280,960);b=project(renderer.scene,xyz[node],1280,960)
                    if a[2]>0 and b[2]>0:
                        clipped=clip_segment(a[:2],b[:2])
                        if clipped:draw.line(clipped,fill='#ffe653',width=4)
                    for p,color in ((a,'#3df1fa'),(b,'#ffe653')):
                        x,y,depth=p
                        if depth>0 and 15<x<1265 and 180<y<945:draw.ellipse((x-12,y-12,x+12,y+12),outline=color,width=4)
                    detail=f'node {node}: dX={d[0]:+.2f}, dY={d[1]:+.2f}, dZ={d[2]:+.2f} mm'
                else:detail='Initial programmed drop is excluded; no displacement marker before t=0.20s'
                draw.rectangle((12,12,1268,140),fill='#eef3fa')
                lines=[f'STATIC 0-3s | seed {seed} | snapshot {fraction:.2f}, frame {frame_index} | t={t:.2f}s | {view.upper()}',
                    labels[.2],detail,
                    f'CYAN: node at 0.20s | YELLOW: now | initial penetration {row["initial_penetration_mm"]:.2f} mm']
                for k,line in enumerate(lines):draw.text((24,15+29*k),line,font=font,fill='#17314b')
                writers[view].stdin.write(np.asarray(img).tobytes())
                if frame==0:img.save(dest/f'{view}_poster.png')
            render_difference=max(render_difference,float(np.max(np.abs(physical_state(env)-before))));assert render_difference==0.
        assert abs(times[-1]-3.)<1e-8
        reproduction_error=abs(hist[.2]-row['from_200ms_max_mm']);assert reproduction_error<1e-5
        for writer in writers.values():writer.stdin.close();assert writer.wait(timeout=60)==0
        writers={};encoding={v:verify_video(dest/f'{v}.mp4',151) for v in cameras}
        np.savez_compressed(dest/'trajectory.npz',states=np.asarray(states),frame_times=np.asarray(times),cable_positions=np.asarray(points))
        episode=dict(scene=scene.name,seed=seed,fraction=fraction,snapshot_step=frame_index,directory=dest.name,
            penetration_mm=row['initial_penetration_mm'],displacement_mm=hist[.2],baseline_s=.2,
            reproduction_error_mm=reproduction_error,duration_s=3.)
        save(dest/'metadata.json',dict(episode=episode,cameras=project_camera_manifest(cfg),source_hash=digest(Path(__file__)),
            frame_metrics=metrics,encoding=encoding,render_physics_difference=render_difference,
            max_node_position_reproduction_error_mm=position_error,recording_kind='fresh execution matching saved 3-second node trajectories',
            displacement_definition='Single baseline at 0.2s; maximum 3D node distance at 50Hz through 3s'))
        return episode
    finally:
        for writer in writers.values():
            if writer.stdin and not writer.stdin.closed:writer.stdin.close()
            writer.wait(timeout=60)
        if renderer is not None:renderer.close()
        env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=10)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--worker',type=int);p.add_argument('--fail-probe',action='store_true')
    p.add_argument('--selection',choices=['first','gt15'],default='first');p.add_argument('--port',type=int,default=8785);a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads(SOURCE.read_text(encoding='utf-8'))
    if a.selection=='gt15':rows=[r for r in rows if r['initial_penetration_mm']<=5. and r['from_200ms_max_mm']>15.]
    rows=rows[:a.limit]
    if a.worker is not None:
        record(rows[a.worker],a.output);return
    now=datetime.now(timezone.utc).isoformat();progress=now;episodes=[];children=[];handles=[]
    def status(state,**extra):
        save(a.output/'status.json',dict(status=state,completed=len(episodes),planned=len(rows),pid=os.getpid(),
            heartbeat_at=datetime.now(timezone.utc).isoformat(),last_progress_at=progress,heartbeat_timeout_seconds=180,
            progress_deadline_at=(datetime.fromisoformat(progress)+timedelta(seconds=300)).isoformat(),**extra))
    def refresh():
        return [json.loads(m.read_text(encoding='utf-8'))['episode'] for r in rows
            if (m:=a.output/f'seed_{r["seed"]}_frame_{r["snapshot_step"]}'/'metadata.json').exists()]
    try:
        status('starting')
        if a.fail_probe:raise RuntimeError('Intentional recording supervisor failure probe')
        save(a.output/'manifest.json',dict(source=str(SOURCE),source_sha256=digest(SOURCE),baseline_s=.2,end_s=3.,
            selection='initial penetration <=5mm and maximum displacement >15mm' if a.selection=='gt15' else 'Same first ten snapshot-motion samples',
            selected_samples=[dict(seed=r['seed'],fraction=r['snapshot_fraction'],frame=r['snapshot_step']) for r in rows],
            cameras=project_camera_manifest(EnvConfig(robot='nero',dynamicvla_cameras_enabled=True)),
            width=1280,height=960,fps=50,controller='enhanced6_bounded_dls_v1',table_guard=True,
            action='ready targets and open jaws throughout',source_hash=digest(Path(__file__)),workers=a.workers))
        queue=list(range(len(rows)));start=time.monotonic()
        save(a.output/'launch_receipt.json',dict(pid=os.getpid(),started_at=now,configuration=dict(workers=a.workers,limit=a.limit),source_hash=digest(Path(__file__))))
        while queue or children:
            while queue and len(children)<a.workers:
                index=queue.pop(0);log=(a.output/f'worker_{index}.log').open('wb');handles.append(log)
                child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(Path(__file__)),
                    '--output',str(a.output),'--limit',str(a.limit),'--worker',str(index),'--selection',a.selection],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                children.append((child,index))
            for child,index in children[:]:
                code=child.poll()
                if code is None:continue
                if code:raise RuntimeError((a.output/f'worker_{index}.log').read_text(encoding='utf-8',errors='replace')[-4000:])
                children.remove((child,index))
            updated=refresh()
            if len(updated)!=len(episodes):
                episodes=updated;progress=datetime.now(timezone.utc).isoformat()
                save(a.output/'summary.json',dict(episodes=episodes));gallery(a.output,episodes,a.selection)
            status('running')
            if time.monotonic()-start>900:raise TimeoutError('Recording exceeded 15 minutes')
            if queue or children:time.sleep(2)
        assert len(episodes)==len(rows)
        status('complete',gallery=f'http://127.0.0.1:{a.port}/index.html',report=str(a.output/'summary.json'))
    except BaseException:
        for child,_ in children:
            if child.poll() is None:child.terminate()
        status('failed',error=traceback.format_exc());raise
    finally:
        for log in handles:log.close()

if __name__=='__main__':main()
