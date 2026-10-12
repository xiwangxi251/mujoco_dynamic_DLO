"""Render recorded node-feedback states, with one progressively filled gallery."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools/evaluation'), str(ROOT/'tools/recording')]
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from run_node_feedback_family import save, load, sha
from record_scripted_scene_family import connector, marker, project, verify_video, project_camera_manifest, WORKSPACE_BOUNDARY
from build_node_video_gallery import build_gallery

COMPLEX = 'dev_static_midshape_v1_x075_v1'


def now():
    return datetime.now(timezone.utc)


def rows(out):
    return load(out/'summary.json')['episodes']


def dest_for(out, row):
    if row.get('directory'):
        return out/row['directory']
    return out/'episodes'/row['scene']/f"seed_{row['seed']}"


def completed(out):
    return list(out.glob('episodes/*/*/video_metadata.json'))


def render(out, index, smoke=False):
    row = rows(out)[index]
    dest = dest_for(out,row)
    video_dest = out/row['video_directory'] if row.get('video_directory') else dest
    if not smoke and (video_dest/'video_metadata.json').exists():
        return
    meta = load(dest/'state_metadata.json')
    cfg = EnvConfig(**meta['env_config'])
    # State-only runs may omit cameras. Enable the project's named rig for
    # independent rendering; cameras add no generalized state or physics.
    recording_camera_enabled=not cfg.dynamicvla_cameras_enabled
    cfg.dynamicvla_cameras_enabled=True
    env = CableGraspEnv(cfg)
    renderer = None
    writers = {}
    started = time.monotonic()
    video_dest = out/'render_smoke' if smoke else video_dest
    video_dest.mkdir(parents=True,exist_ok=True)
    try:
        env.reset(seed=cfg.seed)
        if meta.get('reference_model_file'):
            reference_path=dest/meta['reference_model_file']
            assert sha(reference_path)==meta['reference_model_sha256']
            with np.load(reference_path,allow_pickle=False) as reference:
                env.model.body_quat[:]=reference['body_quat']
            env.data=mujoco.MjData(env.model)
            env._invalidate_contact_cache()
        with np.load(dest/'trajectory.npz',allow_pickle=False) as archive:
            data = {key:archive[key] for key in archive.files}
        assert sha(dest/'trajectory.npz') == meta['trajectory_sha256']
        frames = min(3,len(data['states'])) if smoke else len(data['states'])
        state_kind = int(data['state_kind'])
        env.model.vis.global_.offwidth=1280
        env.model.vis.global_.offheight=960
        renderer = mujoco.Renderer(env.model,height=960,width=1280,max_geom=10000)
        option=mujoco.MjvOption()
        option.geomgroup[3]=0
        cameras={'global':cfg.dynamicvla_opst_camera_name,'wrist':cfg.dynamicvla_wrist_camera_name}
        for camera in cameras.values():
            assert mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_CAMERA,camera)>=0,camera
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
        for view in cameras:
            writers[view] = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
                '-f','rawvideo','-pixel_format','rgb24','-video_size','1280x960','-framerate','50',
                '-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','16','-threads','1',
                '-g','12','-keyint_min','12','-sc_threshold','0','-bf','0','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(video_dest/f'{view}.mp4')],stdin=subprocess.PIPE)
        save(video_dest/'render_ready.json',dict(pid=os.getpid(),ready_at=now().isoformat(),scene=row['scene'],seed=row['seed'],frames=frames))
        error = state_error = auxiliary_error = 0.
        reference_kinematics=mujoco.MjData(env.model)
        for frame in range(frames):
            mujoco.mj_setState(env.model,env.data,data['states'][frame],state_kind)
            # The evaluated recorder saved xpos before the final integration's
            # derived poses were refreshed. Its full integration state is the
            # authoritative timestamped physical state. Compute an independent
            # pose reference from that saved qpos, and report auxiliary lag.
            reference_kinematics.qpos[:]=env.data.qpos
            mujoco.mj_kinematics(env.model,reference_kinematics)
            mujoco.mj_forward(env.model,env.data)
            error = max(error,float(np.max(np.abs(env.data.xpos[env.cable_ids]-reference_kinematics.xpos[env.cable_ids]))))
            auxiliary_error=max(auxiliary_error,float(np.max(np.abs(env.data.xpos[env.cable_ids]-data['cable_xyz'][frame]))))
            before=np.empty(mujoco.mj_stateSize(env.model,state_kind))
            mujoco.mj_getState(env.model,env.data,before,state_kind)
            point=data['tracked_target_xyz'][frame]
            for view,camera in cameras.items():
                renderer.update_scene(env.data,camera=camera,scene_option=option)
                for a,b in zip(WORKSPACE_BOUNDARY,np.roll(WORKSPACE_BOUNDARY,-1,axis=0)):
                    connector(renderer.scene,a,b)
                marker(renderer.scene,point,.006,[1,.72,.05,1])
                image=Image.fromarray(renderer.render())
                draw=ImageDraw.Draw(image)
                draw.rectangle((12,12,1268,144),fill='#eef3fa')
                lines=[f"{row.get('arm_label', 'RULE NODE FEEDBACK')} | {row['scene']} | {view}",
                    f"seed {row['seed']} | t={data['frame_times'][frame]:.2f}s | {data['phases'][frame]} | success {row['task_success']}",
                    'YELLOW: recorded tracked target | BLUE: measured workspace | tangent rotation ON',
                    row.get('source_run', 'Saved simulation states')]
                for j,line in enumerate(lines):
                    draw.text((24,16+31*j),line,font=font,fill='#17314b')
                x,y,z=project(renderer.scene,point,1280,960)
                if z>0 and 20<x<1260 and 156<y<940:
                    draw.ellipse((x-18,y-18,x+18,y+18),outline='#ffe553',width=4)
                writers[view].stdin.write(np.asarray(image).tobytes())
                if frame==0:
                    image.save(video_dest/f'{view}_poster.png')
            after=np.empty_like(before)
            mujoco.mj_getState(env.model,env.data,after,state_kind)
            state_error=max(state_error,float(np.max(np.abs(after-before))))
        for writer in writers.values():
            writer.stdin.close()
            assert writer.wait(timeout=60)==0
        writers={}
        assert error<1e-8 and state_error==0,(error,state_error)
        encoding={view:verify_video(video_dest/f'{view}.mp4',frames) for view in cameras}
        save(video_dest/'video_metadata.json',dict(scene=row['scene'],seed=row['seed'],frames=frames,
            fps=50,width=1280,height=960,cameras=project_camera_manifest(cfg),
            recording_camera_enabled_for_replay=recording_camera_enabled,
            reference_model_sha256=meta.get('reference_model_sha256'),
            source_trajectory_sha256=meta['trajectory_sha256'],render_script_sha256=sha(__file__),
            source_directory=str(dest),source_run=row.get('source_run'),
            frame_times=data['frame_times'][:frames].tolist(),
            reproduction_max_position_error_m=error,render_state_difference=state_error,
            position_reference='Independent kinematics from the authoritative saved qpos',
            stored_auxiliary_xyz_max_discrepancy_m=auxiliary_error,
            encoding_verification=encoding,wall_seconds=time.monotonic()-started,
            capture='Saved actual rollout states; no policy rerun or physics stepping'))
    finally:
        for writer in writers.values():
            if writer.poll() is None:
                writer.kill()
                writer.wait()
        if renderer is not None:
            renderer.close()
        env.close()


def coordinate(args):
    all_rows=rows(args.output)
    priority=[i for i,r in enumerate(all_rows) if r['scene']==COMPLEX]
    remaining=[i for i,r in enumerate(all_rows) if r['scene']!=COMPLEX]
    for indices in (priority,remaining):
        pending=[i for i in indices if not (dest_for(args.output,all_rows[i])/'video_metadata.json').exists()]
        children=[]
        handles=[]
        try:
            while pending or children:
                while pending and len(children)<args.workers:
                    i=pending.pop(0)
                    log=(args.output/f'render_worker_{i}.log').open('wb')
                    handles.append(log)
                    child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(Path(__file__)),
                        '--output',str(args.output),'--worker',str(i)],cwd=ROOT,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                    children.append((i,child))
                for i,child in list(children):
                    if child.poll() is not None:
                        if child.returncode:
                            raise RuntimeError(f'Render worker {i} exit {child.returncode}: '+
                                (args.output/f'render_worker_{i}.log').read_text(encoding='utf-8',errors='replace')[-3000:])
                        children.remove((i,child))
                        build_gallery(args.output)
                time.sleep(.5)
        finally:
            for _,child in children:
                if child.poll() is None:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            for handle in handles:
                handle.close()
    build_gallery(args.output)
    assert len(completed(args.output))==len(all_rows)
    report=args.output/'video_report.md'
    report.write_text('# 保存状态录像\n\n70/70 回合的项目全局与腕部视频已完成。'
        '\n\n1280×960、50 fps；直接还原保存状态，无策略重跑或物理推进。'
        '\n\n每回合 video_metadata.json 保存相机配置、源状态哈希、编码检查与还原误差。'
        '\n\n播放入口：index.html。\n',encoding='utf-8')


def supervise(args):
    out=args.output
    build_gallery(out)
    for code in (Path(__file__),Path(__file__).with_name('build_node_video_gallery.py')):
        target=out/'source'/code.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(code.read_bytes())
    started=progress=now()
    child=None
    last_count=len(completed(out))
    def status(state,**extra):
        paths=completed(out)
        by_scene={scene:sum(p.parent.parent.name==scene for p in paths)
            for scene in load(out/'manifest.json')['scenes']}
        save(out/'video_status.json',dict(status=state,stage='recording',supervisor_pid=os.getpid(),
            child_pid=child.pid if child else None,started_at=started.isoformat(),heartbeat_at=now().isoformat(),
            last_progress_at=progress.isoformat(),progress_deadline_at=(progress+timedelta(seconds=900)).isoformat(),
            heartbeat_timeout_seconds=180,completed_episodes=len(paths),planned_episodes=70,
            by_scene=by_scene,complex_static_completed=by_scene[COMPLEX],
            report=str(out/'video_report.md'),
            gallery=f'http://127.0.0.1:{args.port}/index.html',**extra))
    status('starting')
    try:
        log=(out/'render_coordinator.log').open('wb')
        command=[sys.executable,'-B','-X','utf8',str(Path(__file__)),'--output',str(out),
            '--coordinate','--workers',str(args.workers),'--port',str(args.port)]
        if args.fail_probe:
            command=[sys.executable,'-B','-c','raise RuntimeError("intentional render supervisor failure probe")']
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
        save(out/'video_launch_receipt.json',dict(command=command,supervisor_pid=os.getpid(),
            child_pid=child.pid,started_at=started.isoformat(),render_source_sha256=sha(__file__)))
        while child.poll() is None:
            count=len(completed(out))
            if count!=last_count:
                progress,last_count=now(),count
            ready=[p for p in out.glob('episodes/*/*/render_ready.json')
                if datetime.fromisoformat(load(p)['ready_at'])>=started]
            status('running' if ready else 'starting')
            if now()>progress+timedelta(seconds=900):
                raise TimeoutError('No video completed for 900 seconds')
            time.sleep(3)
        log.close()
        if child.returncode:
            raise RuntimeError((out/'render_coordinator.log').read_text(encoding='utf-8',errors='replace')[-3500:])
        progress=now()
        status('complete')
    except BaseException:
        if child is not None and child.poll() is None:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        status('failed',error=traceback.format_exc())
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--port',type=int,default=8790)
    p.add_argument('--worker',type=int)
    p.add_argument('--coordinate',action='store_true')
    p.add_argument('--gallery-only',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--fail-probe',action='store_true')
    args=p.parse_args()
    args.output=args.output.resolve()
    if args.gallery_only:
        build_gallery(args.output)
    elif args.worker is not None:
        render(args.output,args.worker,args.smoke)
    elif args.coordinate:
        coordinate(args)
    else:
        supervise(args)


if __name__=='__main__':
    main()
