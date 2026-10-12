"""Record a physical lap of the measured sector using the saved scan configuration."""
from __future__ import annotations
import argparse
from dataclasses import asdict, fields
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(args):
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
    from panda_cable_grasp.env.kinematics import rotation_to_quat
    from panda_cable_grasp.env.table_execution_guard import TableGeometry
    from target_node_annotation import project, clip_segment
    from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig
    from record_scripted_scene_family import verify_video, project_camera_manifest

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.intentional_failure:
        raise RuntimeError('Intentional supervisor failure-reporting check')
    if (out / 'global.mp4').exists():
        raise FileExistsError('Refusing to overwrite an existing recording')
    original = json.loads(args.reference.read_text(encoding='utf-8'))
    scan = json.loads(args.scan.read_text(encoding='utf-8'))
    sector = scan['sampled_sector']
    assert sector['inner_radius_m'] == .28 and sector['outer_radius_m'] == .56
    assert sector['angle_min_deg'] == -30 and sector['angle_max_deg'] == 70
    known = {f.name for f in fields(EnvConfig)}
    assert set(original['config']) <= known
    config = EnvConfig(**original['config'])
    # Cameras and episode budget are recording-only changes to the saved setup.
    config.dynamicvla_cameras_enabled = True
    config.episode_seconds = 10000.
    env = CableGraspEnv(config)
    env.reset(seed=config.seed)
    policy = DynamicCableGraspPolicy(env, PolicyConfig())
    closing = env.data.xmat[env.hand_id].reshape(3, 3) @ env.gripper_closing_axis_local
    closing[2] = 0
    closing /= np.linalg.norm(closing)
    approach = np.array([0., 0., -1.])
    rotation = np.column_stack((approach, np.cross(closing, approach), closing))
    policy.desired_quat = rotation_to_quat(rotation)
    policy.desired_approach_axis = approach.copy()
    geometry = TableGeometry(env)
    z = geometry.top + .005
    center = np.asarray(scan['center_xy_m'])
    lo, hi = np.radians([sector['angle_min_deg'], sector['angle_max_deg']])
    ri, ro = sector['inner_radius_m'], sector['outer_radius_m']
    speed = .12
    def point(r, angle):
        return np.r_[center + r*np.array([np.cos(angle), np.sin(angle)]), z]
    segments = []
    def line(name, a, b, seconds, lap=False):
        segments.append((name, max(1, int(np.ceil(seconds/.02))), lambda t: a+(b-a)*t, lap))
    def arc(name, r, start, end):
        segments.append((name, int(np.ceil(r*abs(end-start)/speed/.02)),
                         lambda t: point(r, start+(end-start)*t), True))
    p0, p1, p2, p3 = point(ri, lo), point(ro, lo), point(ro, hi), point(ri, hi)
    hover = p0.copy(); hover[2] = geometry.top + .105
    initial = env.hand_position.copy()
    line('APPROACH', initial, hover, 3.)
    line('APPROACH HOLD', hover, hover, 1.)
    line('DESCEND', hover, p0, 2.)
    line('START HOLD', p0, p0, 2.)
    line('RADIAL -30 OUT', p0, p1, (ro-ri)/speed, True)
    line('CORNER 1 HOLD', p1, p1, 2., True)
    arc('OUTER ARC -30 TO 70', ro, lo, hi)
    line('CORNER 2 HOLD', p2, p2, 2., True)
    line('RADIAL 70 IN', p2, p3, (ro-ri)/speed, True)
    line('CORNER 3 HOLD', p3, p3, 2., True)
    arc('INNER ARC 70 TO -30', ri, hi, lo)
    line('CORNER 4 HOLD', p0, p0, 2., True)
    planned = sum(n for _, n, _, _ in segments)
    if args.smoke:
        segments = [('SMOKE', 4, lambda t: initial, False)]
        planned = 4
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    size = mujoco.mj_stateSize(env.model, spec)
    states, rows, commands = [], [], []
    def progress(stage, completed, total):
        save(out/'worker_progress.json', dict(stage=stage, completed=completed, planned=total,
             updated_at=utc(), worker_ready=True))
    progress('simulating', 0, planned)
    for name, count, target_fn, lap in segments:
        for k in range(count):
            target = target_fn((k+1)/count)
            action = policy._ik_action(target, env.gripper_closed_ctrl)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                raise RuntimeError(f'Unexpected environment termination: {env.last_termination_reason}')
            state = np.empty(size)
            mujoco.mj_getState(env.model, env.data, state, spec)
            states.append(state); commands.append(action.copy())
            rows.append(dict(time=float(env.data.time), phase=name, lap=lap,
                desired_xyz=target.tolist(), actual_xyz=env.hand_position.tolist(),
                position_error_mm=float(np.linalg.norm(env.hand_position-target)*1000),
                pose_error_deg=float(np.degrees(policy._orientation_error()[1])),
                gripper_clearance_mm=float(min(p[1] for p in geometry.points())*1000)))
            if len(rows) % 25 == 0:
                progress('simulating', len(rows), planned)
        print(json.dumps(rows[-1]), flush=True)
    save(out/'metrics.json', rows)
    np.savez_compressed(out/'trajectory.npz', states=states, native_commands=commands,
                        times=[r['time'] for r in rows], desired=[r['desired_xyz'] for r in rows],
                        actual=[r['actual_xyz'] for r in rows])
    lap_rows = [r for r in rows if r['lap']]
    moving = [r for r in lap_rows if 'HOLD' not in r['phase']]
    result = dict(status='rendering', run_name=out.name, seed=config.seed, scenario='near_table_sector_boundary_lap',
        sector=sector, center_xy_m=scan['center_xy_m'], config=asdict(config),
        source_directory=str(ROOT), source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
            ROOT/'src/panda_cable_grasp/env/environment.py',ROOT/'src/panda_cable_grasp/env/control.py',
            ROOT/'src/panda_cable_grasp/policies/scripted.py',ROOT/'src/panda_cable_grasp/env/table_execution_guard.py']},
        reference_record=str(args.reference), reference_sha256=digest(args.reference),
        scan_record=str(args.scan), scan_sha256=digest(args.scan),
        camera_manifest=project_camera_manifest(config), frames=len(rows), simulation_duration_s=len(rows)/50,
        target_tcp_height_m=.005, reference_speed_m_s=speed, desired_rotation=rotation.tolist(),
        reference_lap_completed=not args.smoke,
        interpretation='Fresh continuous boundary demonstration; not a new workspace scan or proof of continuous reachability.',
        protocol='Saved inner-arc physical configuration; 3s approach, 1s hover, 2s descent, 2s hold; radial -30 outward, outer arc to70, radial70 inward, inner arc to-30 at .12m/s, 2s at each corner. Fixed downward closed jaws; cable outside region. Original limits and guard.',
        moving_mean_error_mm=float(np.mean([r['position_error_mm'] for r in moving])) if moving else None,
        moving_max_error_mm=max((r['position_error_mm'] for r in moving),default=None),
        moving_fraction_position_lt15mm=float(np.mean([r['position_error_mm']<15 for r in moving])) if moving else None,
        moving_fraction_pose_lt5deg=float(np.mean([r['pose_error_deg']<5 for r in moving])) if moving else None,
        corner_holds=[r for i,r in enumerate(rows) if r['phase'].startswith('CORNER') and
                      (i==len(rows)-1 or rows[i+1]['phase']!=r['phase'])])
    save(out/'summary.json', result)
    progress('rendering', 0, len(rows))
    env.model.vis.global_.offwidth = 1280; env.model.vis.global_.offheight = 960
    renderer = mujoco.Renderer(env.model, width=1280, height=960, max_geom=10000)
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    cameras = dict(global_=config.dynamicvla_opst_camera_name, wrist=config.dynamicvla_wrist_camera_name)
    cameras['global'] = cameras.pop('global_')
    boundary = [point(ro,a) for a in np.linspace(lo,hi,100)] + [point(ri,a) for a in np.linspace(hi,lo,100)]
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 23)
    writers = {v:subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo',
        '-pixel_format','rgb24','-video_size','1280x960','-framerate','50','-i','-', '-an',
        '-c:v','libx264','-preset','veryfast','-crf','16','-threads','1','-g','12','-keyint_min','12',
        '-sc_threshold','0','-bf','0','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{v}.mp4')],
        stdin=subprocess.PIPE) for v in cameras}
    max_difference = 0.
    trail = []
    try:
        for index,(state,row) in enumerate(zip(states,rows)):
            mujoco.mj_setState(env.model,env.data,state,spec); mujoco.mj_forward(env.model,env.data)
            before=np.empty(size); mujoco.mj_getState(env.model,env.data,before,spec)
            if row['lap'] and index%5==0:
                p=np.array(row['actual_xyz']); p[2]=z; trail.append(p)
            for view,writer in writers.items():
                renderer.update_scene(env.data,camera=cameras[view],scene_option=option)
                img=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(img)
                def world_line(a,b,color,width):
                    pa,pb=project(renderer.scene,a,1280,960),project(renderer.scene,b,1280,960)
                    if min(pa[2],pb[2])>0:
                        clipped=clip_segment(pa[:2],pb[:2])
                        if clipped: draw.line(clipped,fill=color,width=width)
                for a,b in zip(boundary,boundary[1:]+boundary[:1]): world_line(a,b,'#ffae45',4)
                for a,b in zip(trail,trail[1:]): world_line(a,b,'#36edb6',3)
                p=np.array(row['desired_xyz'])
                for axis in np.eye(3)[:2]: world_line(p-.008*axis,p+.008*axis,'#ff5353',4)
                draw.rectangle((12,12,1268,150),fill='#eef4fb')
                lines=[f'NEW WORKSPACE BOUNDARY | NERO | {view.upper()} | t={index/50:.2f}s',
                    f'r=0.28..0.56 m | -30..70 deg | {row["phase"]}',
                    f'ORANGE reference | GREEN actual TCP | RED target | speed 0.12 m/s',
                    f'error {row["position_error_mm"]:.1f} mm | pose {row["pose_error_deg"]:.1f} deg | clearance {row["gripper_clearance_mm"]:.1f} mm']
                for j,label in enumerate(lines): draw.text((26,18+31*j),label,font=font,fill='#17324c')
                if index==0 or (row['lap'] and not rows[index-1]['lap']): img.save(out/f'{view}_poster.png')
                writer.stdin.write(np.asarray(img,dtype=np.uint8).tobytes())
            after=np.empty(size); mujoco.mj_getState(env.model,env.data,after,spec)
            max_difference=max(max_difference,float(np.max(abs(before-after))))
            if (index+1)%25==0: progress('rendering',index+1,len(rows))
    finally:
        for writer in writers.values():
            writer.stdin.close()
        codes=[writer.wait() for writer in writers.values()]
        renderer.close(); env.close()
    assert codes==[0,0] and max_difference==0
    progress('verifying',len(rows),len(rows))
    result.update(video_formats={v:verify_video(out/f'{v}.mp4',len(rows)) for v in cameras},
                  render_state_difference=max_difference, status='complete',local_page_ready=False)
    save(out/'metadata.json',result)
    build_gallery(args, result)
    save(out/'summary.json',result)
    progress('complete',len(rows),len(rows))


def build_gallery(args, result):
    page=args.template.read_text(encoding='utf-8')
    page=page.replace('near_table_inner_arc028_20261009',args.output.name)
    page=page.replace("negative_y_polar_reach_r'+e.radius_m+'_arc-30to70", "near_table_sector_boundary_lap")
    page=re.sub(r'<title>.*?</title>','<title>新工作区完整边界绕行 · −30°～70°</title>',page)
    page=re.sub(r'<h1>.*?</h1>','<h1>新工作区完整边界绕行 · −30°～70°</h1>',page)
    page=re.sub(r'<select id="goal">.*?</select>',f'<select id="goal"><option value="{result["seed"]}">完整边界一周</option></select>',page)
    page=page.replace('内弧绕行演示','完整边界绕行演示').replace('内弧−30°～70°','完整周界−30°～70°')
    note='圆心(0.35,0)m；半径0.28～0.56m，角度−30°～70°，+X为0°、向+Y为正。按−30°径向外移→外弧到70°→70°径向内移→内弧回到−30°绕行一周，各角点停留2秒。TCP目标离桌5mm，固定朝下闭爪；增强6＋有界DLS、原限速和碰桌保护开启、旧过滤器关闭。橙线为参考边界，绿线为实际TCP轨迹，红十字为当前目标。使用环境命名全局与腕部相机，原生1280×960/50fps。连续绕行演示不等于重新测定工作区。'
    if result['moving_mean_error_mm'] is not None:
        note+=f' 运动段平均误差{result["moving_mean_error_mm"]:.1f}mm、最大{result["moving_max_error_mm"]:.1f}mm；位置＜15mm帧占{100*result["moving_fraction_position_lt15mm"]:.1f}%，姿态＜5°帧占{100*result["moving_fraction_pose_lt5deg"]:.1f}%。'
    page=re.sub(r'<div class="note" id="note">.*?</div>',f'<div class="note" id="note">{html.escape(note)}</div>',page,flags=re.S)
    data=[dict(seed=result['seed'],directory='.',simulation_duration_s=result['simulation_duration_s'])]
    page=re.sub(r'(<script id="data" type="application/json">).*?(</script>)',lambda m:m[1]+json.dumps(data)+m[2],page,flags=re.S)
    # The retained synchronization and showControls scripts include scrubbing and copy-reference support.
    (args.output/'index.html').write_text(page,encoding='utf-8')
    result.update(local_page_ready=True,video_url=f'http://127.0.0.1:8766/{args.output.name}/index.html')


def supervise(args):
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    status=dict(status='starting',run_name=out.name,stage='starting',completed=0,planned=None,
                supervisor_pid=os.getpid(),supervisor_started_at=utc(),heartbeat_at=utc(),
                last_progress_at=utc(),heartbeat_timeout_seconds=30,progress_timeout_seconds=180,
                progress_deadline_at=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=180)).isoformat(),
                output=str(out),summary_path=str(out/'summary.json'),error=None,exit_code=None)
    save(out/'status.json',status)
    command=[sys.executable,str(Path(__file__)),*sys.argv[1:]]
    command.remove('--supervise')
    with (out/'stdout.log').open('ab',buffering=0) as stdout,(out/'stderr.log').open('ab',buffering=0) as stderr:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,env=os.environ.copy())
        status.update(child_pid=child.pid,child_started_at=utc())
        while True:
            status['heartbeat_at']=utc()
            progress=out/'worker_progress.json'
            if progress.exists():
                record=json.loads(progress.read_text(encoding='utf-8'))
                if (record['stage'],record['completed'])!=(status['stage'],status['completed']):
                    status.update(last_progress_at=record['updated_at'],progress_deadline_at=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=180)).isoformat())
                status.update(stage=record['stage'],completed=record['completed'],planned=record['planned'],status='running')
            code=child.poll()
            if code is not None:
                summary=out/'summary.json'
                valid=code==0 and summary.exists() and json.loads(summary.read_text(encoding='utf-8')).get('local_page_ready')
                status.update(status='complete' if valid else 'failed',exit_code=code,
                    error=None if valid else (out/'stderr.log').read_text(errors='replace')[-4000:])
                save(out/'status.json',status)
                return
            save(out/'status.json',status)
            time.sleep(3)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','reference','scan','template'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--supervise',action='store_true')
    parser.add_argument('--intentional-failure',action='store_true',help=argparse.SUPPRESS)
    arguments=parser.parse_args()
    (supervise if arguments.supervise else record)(arguments)
