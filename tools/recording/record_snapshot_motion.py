"""Record exact reproductions of the first ten snapshot-motion audit samples."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.control import configure_profile
from panda_cable_grasp.scenarios import get_scenario
from record_scripted_scene_family import project_camera_manifest, physical_state, verify_video, digest
from target_node_annotation import project, clip_segment
from build_scripted_family_gallery import PAGE


def save(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def gallery(output, rows):
    visibility_source = (ROOT.parent / 'archive/reward_v2_20261005/reward_v6_gallery_template.html').read_text(encoding='utf-8')
    visibility = next(s for s in re.findall(r'<script>.*?</script>', visibility_source, re.S) if 'function showControls(video,show)' in s)
    table = '<table><tr><th>样本</th><th>初始穿透 mm</th><th>等待期间最大位移 mm</th></tr>' + ''.join(
        f"<tr><td>seed={r['seed']} / frame={r['snapshot_step']}</td><td>{r['penetration_mm']:.2f}</td><td>{r['displacement_mm']:.2f}</td></tr>" for r in rows) + '</table>'
    page = PAGE.replace('__TABLE__', table).replace('__VISIBILITY__', visibility)
    page = page.replace('__RUN_NAME__', json.dumps(output.name))
    page = page.replace('__SCENES__', '<option value="dev_static_midshape_v1_x075_v1">复杂静态快照 · 等待过程</option>')
    page = page.replace('__SEEDS__', ''.join(f'<option value="{r["seed"]}">样本 {i+1} · seed {r["seed"]}</option>' for i, r in enumerate(rows)))
    page = page.replace('__PLAYERS__', ''.join(f'<article class="player" id="player-{view}"><h2>{label}<span class="badge" id="badge-{view}"></span></h2><video id="video-{view}" controls playsinline preload="auto"></video><p class="meta" id="meta-{view}"></p></article>' for view, label in [('global','项目全局相机'),('wrist','项目腕部相机')]))
    page = page.replace('__DATA__', json.dumps(rows, ensure_ascii=False).replace('</', '<\\/'))
    page = re.sub(r'<title>.*?</title>', '<title>初始快照位移 · 10个样本</title>', page, count=1)
    page = re.sub(r'<h1>.*?</h1>', '<h1>初始快照位移 · 10个样本</h1>', page, count=1)
    page = re.sub(r'<div class="subtitle">.*?</div>', '<div class="subtitle">原100样本中按测试顺序取前10个；50条评测库，快照35%；机械臂保持就绪目标，夹爪张开，记录0—0.82秒；原生1280×960、50fps。</div>', page, count=1)
    note = '这是按原快照和配置重新执行的录像。青色：同一节点的初始位置；黄色：当前位移最大的节点；连线显示位移。HUD同时显示当前最大位移、历史最大位移及该节点XYZ位移分量。位移按三维位置计算，包含整体移动。可用0.25倍速查看。'
    page = re.sub(r'<p id="note" class="note">.*?</p>', '<p id="note" class="note">'+note+'</p>', page, count=1)
    page = re.sub(r'<footer>.*?</footer>', '<footer><a href="summary.json">录像记录</a><a href="manifest.json">相机与配置</a></footer>', page, flags=re.S)
    reference = "function reference(){const e=D.find(e=>e.scene===$('scene').value&&String(e.seed)===$('goal').value);const t=Math.min(.82,scrubbing||pending!==null?Number($('seek').value):Math.max(...vs().map(v=>v.currentTime||0)));return `${runName} | ${e.scene} | seed=${e.seed} | fraction=${e.fraction} | frame=${e.snapshot_step} | t=${(Math.round(t/.02)*.02).toFixed(2)}s`}"
    page = re.sub(r'function reference\(\)\{[^\n]+', lambda _: reference, page, count=1)
    load = """function load(){cancel();const e=D.find(e=>e.scene===$('scene').value&&String(e.seed)===$('goal').value),view=$('view').value;ids.forEach(id=>{const v=$('video-'+id);v.pause();v.src=e.directory+'/'+id+'.mp4';v.poster=e.directory+'/'+id+'_poster.png';v.playbackRate=Number($('rate').value);v.load();$('player-'+id).hidden=view!=='both'&&view!==id;$('badge-'+id).textContent=e.penetration_mm>5?'初始穿透筛除':'初始穿透保留';$('meta-'+id).innerHTML=`快照 ${e.fraction} · 源帧 ${e.snapshot_step} · 初始穿透 ${e.penetration_mm.toFixed(2)} mm<br>等待期间最大位移 ${e.displacement_mm.toFixed(2)} mm · 复现差值 ${e.reproduction_error_mm.toExponential(2)} mm<br><span id="frame-clock-${id}">t=0.00 s</span> · <a href="${e.directory}/metadata.json">完整记录</a>`});document.querySelector('.players').style.gridTemplateColumns=view==='both'?'':'1fr';$('seek').value=0;$('seek').max=.82;const u=new URL(location.href);u.searchParams.set('scene',e.scene);u.searchParams.set('seed',e.seed);history.replaceState(null,'',u)}"""
    page = re.sub(r'function load\(\)\{.*?(?=\nasync function play)', lambda _: load, page, count=1, flags=re.S)
    page = page.replace("($('video-'+id).currentTime||0).toFixed(2)", "Math.min(.82,$('video-'+id).currentTime||0).toFixed(2)")
    (output/'index.html').write_text(page, encoding='utf-8')


def record(row, output, bank_dir):
    seed = int(row['seed']); fraction = float(row['snapshot_fraction'])
    dest = output / f'seed_{seed}_frame_{row["snapshot_step"]}'
    if (dest/'metadata.json').exists():
        return json.loads((dest/'metadata.json').read_text(encoding='utf-8'))['episode']
    dest.mkdir(parents=True, exist_ok=True)
    scene = get_scenario('dev_static_midshape_v1_x075_v1')
    overrides = scene.to_env_overrides()
    overrides.update(initial_shape_time_range=(fraction,fraction), arm_motion_start_delay=.82)
    config = configure_profile(EnvConfig(robot='nero', seed=seed, replay_bank_dir=str(bank_dir), replay_seed_fallback='error',
        episode_seconds=2., dynamicvla_cameras_enabled=True, **overrides), 'enhanced6_bounded_dls_v1')
    config.table_execution_guard_enabled = True
    env = CableGraspEnv(config); renderer = None; writers = {}
    try:
        env.reset(seed=seed)
        assert env._initial_shape_step == int(row['snapshot_step'])
        initial = env.data.xpos[env.cable_ids].copy()
        env.model.vis.global_.offwidth = 1280; env.model.vis.global_.offheight = 960
        renderer = mujoco.Renderer(env.model, height=960, width=1280)
        cameras = {'global':config.dynamicvla_opst_camera_name, 'wrist':config.dynamicvla_wrist_camera_name}
        option = mujoco.MjvOption(); option.geomgroup[3] = 0
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 25)
        writers = {view:subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
            '-video_size','1280x960','-framerate','50','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','16',
            '-threads','1','-g','12','-keyint_min','12','-sc_threshold','0','-bf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/f'{view}.mp4')], stdin=subprocess.PIPE) for view in cameras}
        states=[]; times=[]; positions=[]; metrics=[]; history=0.; render_difference=0.
        for frame in range(42):
            if frame:
                _,_,terminated,truncated,_ = env.step(env.ready_ctrl.copy())
                assert not terminated and not truncated
            before = physical_state(env)
            xyz = env.data.xpos[env.cable_ids].copy(); delta = xyz-initial
            norms = np.linalg.norm(delta,axis=1); node=int(np.argmax(norms)); current=float(norms[node])*1000.
            history=max(history,current); t=float(env.data.time)
            states.append(before); times.append(t); positions.append(xyz)
            metrics.append(dict(time_s=t,current_max_node=node,current_max_mm=current,historical_max_mm=history,delta_xyz_mm=(delta[node]*1000.).tolist()))
            for view,camera in cameras.items():
                renderer.update_scene(env.data,camera=camera,scene_option=option)
                img=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(img)
                a=project(renderer.scene,initial[node],1280,960); b=project(renderer.scene,xyz[node],1280,960)
                if a[2]>0 and b[2]>0:
                    clipped=clip_segment(a[:2],b[:2])
                    if clipped: draw.line(clipped,fill='#ffe653',width=4)
                for p,color in ((a,'#3df1fa'),(b,'#ffe653')):
                    x,y,depth=p
                    if depth>0 and 15<x<1265 and 165<y<945:
                        draw.ellipse((x-12,y-12,x+12,y+12),outline=color,width=4)
                draw.rectangle((12,12,1268,151),fill='#eef3fa')
                dx,dy,dz=delta[node]*1000.
                labels=[f"STATIC WAIT | seed {seed} | snapshot {fraction:.2f}, frame {row['snapshot_step']} | {view.upper()}",
                    f"t={t:.2f}s | current max {current:.2f} mm (node {node}) | historical max {history:.2f} mm",
                    f"node {node}: dX={dx:+.2f}, dY={dy:+.2f}, dZ={dz:+.2f} mm | initial penetration {float(row['initial_penetration_mm']):.2f} mm",
                    'CYAN: initial position | YELLOW: current position | ready targets, open jaws']
                for k,label in enumerate(labels):draw.text((24,17+32*k),label,font=font,fill='#17314b')
                writers[view].stdin.write(np.asarray(img).tobytes())
                if frame==0:img.save(dest/f'{view}_poster.png')
            render_difference=max(render_difference,float(np.max(np.abs(physical_state(env)-before))))
            assert render_difference==0.
        assert abs(times[-1]-.82)<1e-8
        reproduction_error=abs(history-float(row['max_node_displacement_mm']))
        assert reproduction_error<1e-5, f'Motion differs from source measurement by {reproduction_error} mm'
        for writer in writers.values():
            writer.stdin.close()
            assert writer.wait(timeout=60)==0
        writers={}
        encoding={view:verify_video(dest/f'{view}.mp4',42) for view in cameras}
        np.savez_compressed(dest/'trajectory.npz',states=np.asarray(states),frame_times=np.asarray(times),cable_positions=np.asarray(positions))
        episode=dict(scene=scene.name,seed=seed,fraction=fraction,snapshot_step=int(row['snapshot_step']),directory=dest.name,
            penetration_mm=float(row['initial_penetration_mm']),displacement_mm=history,reproduction_error_mm=reproduction_error,duration_s=.82)
        save(dest/'metadata.json',dict(episode=episode,cameras=project_camera_manifest(config),frame_metrics=metrics,
            encoding=encoding,render_physics_difference=render_difference,recording_kind='fresh reproduction of original measured sample',
            displacement_definition='max 3D per-node displacement relative to t=0 at 50Hz; historical maximum over t<=0.82s'))
        return episode
    finally:
        for writer in writers.values():
            if writer.stdin:writer.stdin.close()
            writer.wait(timeout=60)
        if renderer is not None:renderer.close()
        env.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--limit',type=int,default=10)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    source=ROOT/'outputs/static_snapshot_motion_100_20261010/samples.csv'
    summary=json.loads((source.parent/'summary.json').read_text(encoding='utf-8'))
    bank_dir=Path(summary['source_bank']).parent
    with source.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))[:args.limit]
    episodes=[]
    def status(state,**extra):
        now=datetime.now(timezone.utc).isoformat()
        save(args.output/'status.json',dict(status=state,completed=len(episodes),planned=args.limit,pid=os.getpid(),heartbeat_at=now,last_progress_at=now,**extra))
    try:
        status('starting')
        save(args.output/'manifest.json',dict(source_csv=str(source),source_csv_sha256=digest(source),bank=str(bank_dir),
            source_bank_sha256=summary['bank_sha256'],selection='first ten samples in source CSV order, including penetration-filtered samples',
            cameras=project_camera_manifest(EnvConfig(robot='nero',dynamicvla_cameras_enabled=True)),controller='enhanced6_bounded_dls_v1',
            table_guard=True,arm_motion_start_delay=.82,action='ready targets with open jaws throughout',width=1280,height=960,fps=50,
            source_files={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'src/panda_cable_grasp/env/environment.py',ROOT/'src/panda_cable_grasp/env/control.py')}))
        for row in rows:
            episodes.append(record(row,args.output,bank_dir))
            save(args.output/'summary.json',dict(episodes=episodes))
            gallery(args.output,episodes)
            status('running')
        status('complete',gallery='http://127.0.0.1:8784/index.html',report=str((args.output/'summary.json').resolve()),episodes=episodes)
        print(json.dumps(dict(completed=len(episodes)),ensure_ascii=False))
    except BaseException:
        status('failed',error=traceback.format_exc()); raise


if __name__=='__main__':main()
