"""Render all 450 recorded rule-test episodes and a progressive comparison page."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/evaluation'),str(ROOT/'tools/recording')]
from run_node_feedback_family import save, load, sha
from render_saved_node_family import render
from build_rule_comparison_gallery import build_gallery


def now():
    return datetime.now(timezone.utc)


def prepare(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        manifest=load(out/'manifest.json')
        assert manifest['smoke']==args.smoke
        assert all(sha(ROOT/p)==h for p,h in manifest['render_source_sha256'].items()), 'Frozen render source changed'
        return manifest
    runs={'material':args.results/'scripted_material50_20261012','nearest':args.results/'scripted_nearest50_20261012'}
    entries=[]
    source_runs={}
    for arm,run in runs.items():
        manifest=load(run/'manifest.json')
        status=load(run/'status.json')
        assert status['status']=='complete'
        for relative,expected in manifest['source_sha256'].items():
            if relative.replace('\\','/').startswith('src/'):
                assert sha(ROOT/relative)==expected, 'Recording physics source differs: '+relative
        source_runs[arm]=dict(directory=str(run),commit=manifest['commit'],
            configuration_hash=manifest['configuration_hash'],manifest_sha256=sha(run/'manifest.json'),
            summary_sha256=sha(run/'summary.json'))
        for job in manifest['jobs']:
            episode=run/'episodes'/job['scene']/f"seed_{job['seed']}"
            result=load(episode/'result.json')
            meta=load(episode/'state_metadata.json')
            assert result['configuration_hash']==manifest['configuration_hash']
            assert sha(episode/'trajectory.npz')==meta['trajectory_sha256']==result['trajectory_sha256']
            entries.append(dict(scene=job['scene'],seed=job['seed'],arm=arm,
                source_run=run.name,directory=str(episode),
                video_directory=f"videos/{arm}/{job['scene']}/seed_{job['seed']}",
                arm_label='FIXED MATERIAL POINT' if arm=='material' else 'NEAREST SEGMENT',
                task_success=result['task_success'],duration_s=result['duration_s'],frames=meta['frames'],
                source_trajectory_sha256=meta['trajectory_sha256']))
    assert len(entries)==450
    # Prioritize a complete same-seed comparison, then all shape/combined pairs.
    entries.sort(key=lambda r:(
        0 if r['scene']=='dev_combined_l1_nominal_x075_v1' and r['seed']==20260805 else
        1 if r['scene'] in ('dev_shape_nominal_current_x075_v1','dev_combined_l1_nominal_x075_v1') else 2,
        r['seed'],r['scene'],r['arm']))
    if args.smoke:
        entries=entries[:2]+[
            next(r for r in entries if r['scene']=='dev_shape_nominal_current_x075_v1' and r['seed']==20260805),
            next(r for r in entries if r['scene']=='dev_static_midshape_v1_x075_v1' and r['seed']==20260807)]
    sources=list((ROOT/'src').rglob('*.py'))+[Path(__file__),
        ROOT/'tools/recording/render_saved_node_family.py',ROOT/'tools/recording/build_rule_comparison_gallery.py',
        ROOT/'tools/recording/build_scripted_family_gallery.py',ROOT/'tools/recording/record_scripted_scene_family.py',
        ROOT/'tools/evaluation/run_node_feedback_family.py']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    for source in sources:
        target=out/'source'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
    visibility_template=next(p/'archive/reward_v2_20261005/reward_v6_gallery_template.html'
        for p in ROOT.parents if (p/'archive/reward_v2_20261005/reward_v6_gallery_template.html').exists())
    (out/'source/visibility_template.html').write_bytes(visibility_template.read_bytes())
    manifest=dict(run_id=out.name,created_at=now().isoformat(),source_directory=str(ROOT),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        source_git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        source_runs=source_runs,render_source_sha256=hashes,visibility_template_sha256=sha(visibility_template),planned_episodes=len(entries),
        planned_videos=2*len(entries),planned_frames=sum(r['frames'] for r in entries),
        width=1280,height=960,fps=50,smoke=args.smoke,seeds=list(range(20260804,20260854)),
        scenes=list(dict.fromkeys(r['scene'] for r in entries)),
        rendering='Saved actual states; no policy execution, no integration, no resampling',
        authorization_chain='Render, verify encoding and state restoration, browser-check comparison page, notify completion')
    manifest['configuration_hash']=hashlib.sha256(json.dumps(dict(entries=entries,sources=hashes),sort_keys=True).encode()).hexdigest()
    save(out/'manifest.json',manifest)
    save(out/'summary.json',dict(episodes=entries,source_runs=source_runs))
    build_gallery(out)
    return manifest


def metadata(out,row):
    path=out/row['video_directory']/'video_metadata.json'
    if not path.exists():
        return None
    meta=load(path)
    assert meta['source_trajectory_sha256']==row['source_trajectory_sha256']
    assert (meta['width'],meta['height'],meta['fps'],meta['frames'])==(1280,960,50,row['frames'])
    assert meta['reproduction_max_position_error_m']<1e-8 and meta['render_state_difference']==0
    for view in ('global','wrist'):
        verification=meta['encoding_verification'][view]
        assert verification['frame_timestamps_verified'] and verification['max_keyframe_spacing_frames']<=12
        assert (out/row['video_directory']/f'{view}.mp4').exists()
    return meta


def supervise(args):
    manifest=prepare(args)
    out=args.output
    entries=load(out/'summary.json')['episodes']
    begun=progress=now()
    pending=[i for i,r in enumerate(entries) if metadata(out,r) is None]
    active,handles={},[]
    completed=len(entries)-len(pending)
    def status(state,**extra):
        save(out/'status.json',dict(run_id=manifest['run_id'],configuration_hash=manifest['configuration_hash'],
            status=state,experiment_state=state,sync_state='local',stage='rendering',
            source_commit=manifest['source_commit'],supervisor_pid=os.getpid(),started_at=begun.isoformat(),
            supervisor_start_identity=begun.isoformat(),
            workers=[dict(job_id=i,pid=p.pid,started_at=t) for i,(p,t) in active.items()],
            heartbeat_at=now().isoformat(),source_updated_at=now().isoformat(),synced_at=now().isoformat(),
            last_progress_at=progress.isoformat(),heartbeat_interval_seconds=3,heartbeat_timeout_seconds=120,
            progress_deadline_at=(progress+timedelta(minutes=15)).isoformat(),
            startup_deadline_at=(begun+timedelta(minutes=2)).isoformat(),
            expected_completion_at=(begun+timedelta(hours=2)).isoformat(),
            completed_episodes=completed,planned_episodes=len(entries),completed_videos=2*completed,
            planned_videos=2*len(entries),worker_ready=len(list((out/'videos').glob('*/*/*/render_ready.json'))),
            smoke=manifest['smoke'],gallery=f'http://127.0.0.1:{args.port}/{out.name}/index.html',
            report=str(out/'video_report.md'),browser_verification_required=True,**extra))
    status('starting')
    try:
        if args.fail_probe:
            probe=subprocess.run([sys.executable,'-B','-c','raise SystemExit(7)'])
            raise RuntimeError(f'Intentional render supervisor child exit {probe.returncode}')
        while pending or active:
            while pending and len(active)<args.workers:
                index=pending.pop(0)
                log=(out/f'render_worker_{index}.log').open('wb')
                handles.append(log)
                command=[sys.executable,'-B','-X','utf8',str(Path(__file__)),'--output',str(out),
                    '--results',str(args.results),'--worker',str(index)]
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
                active[index]=child,now().isoformat()
            changed=False
            for index,(child,_) in list(active.items()):
                code=child.poll()
                if code is None:
                    continue
                if code:
                    error=(out/f'render_worker_{index}.log').read_text(encoding='utf-8',errors='replace')[-4000:]
                    raise RuntimeError(f'Render worker {index} pid {child.pid} exit {code}: {error}')
                assert metadata(out,entries[index]) is not None, 'Worker exited without verified videos'
                del active[index]
                completed+=1
                progress=now()
                changed=True
            if changed:
                build_gallery(out)
            if now()>progress+timedelta(minutes=15):
                raise TimeoutError('No completed rendered episode for 15 minutes')
            if now()>begun+timedelta(hours=2):
                raise TimeoutError('Two-hour rendering deadline exceeded')
            ready=list((out/'videos').glob('*/*/*/render_ready.json'))
            status('running' if ready else 'starting')
            if pending or active:
                time.sleep(3)
        checks=[metadata(out,r) for r in entries]
        assert all(checks) and len(checks)==len(entries)
        build_gallery(out)
        total_frames=sum(m['frames'] for m in checks)
        results=dict(completed_episodes=completed,planned_episodes=len(entries),
            completed_videos=completed*2,planned_videos=len(entries)*2,
            encoded_frames=total_frames*2,all_encoding_verified=True,all_source_states_verified=True,
            max_position_error_m=max(m['reproduction_max_position_error_m'] for m in checks),
            stored_auxiliary_xyz_max_discrepancy_m=max(m['stored_auxiliary_xyz_max_discrepancy_m'] for m in checks),
            max_render_state_difference=max(m['render_state_difference'] for m in checks),
            by_arm={arm:sum(r['arm']==arm for r in entries) for arm in ('material','nearest')})
        save(out/'video_summary.json',results)
        (out/'video_report.md').write_text(f"# 规则测试保存状态录像\n\n{completed}/{len(entries)}回合，{completed*2}段全局／腕部视频。\n\n原生1280×960、50 fps，H.264 CRF16，GOP≤12，无B帧。\n\n实际保存状态重渲染；无策略重跑、无物理积分。两批测试成功率保持原记录。\n\n全部状态还原和编码校验通过；位置最大还原误差{results['max_position_error_m']:.3g}m，渲染前后状态差{results['max_render_state_difference']:.3g}。\n\n浏览器播放验证单独记录于browser_verification.json。\n",encoding='utf-8')
        progress=now()
        status('complete',exit_code=0,results=results)
    except BaseException:
        error=traceback.format_exc()
        for child,_ in active.values():
            if child.poll() is None:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        status('failed',exit_code=1,error=error)
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--port',type=int,default=8796)
    parser.add_argument('--worker',type=int)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--fail-probe',action='store_true')
    args=parser.parse_args()
    args.output,args.results=args.output.resolve(),args.results.resolve()
    if args.worker is not None:
        manifest=load(args.output/'manifest.json')
        assert all(sha(ROOT/p)==h for p,h in manifest['render_source_sha256'].items())
        render(args.output,args.worker)
    elif args.prepare:
        prepare(args)
    else:
        try:
            supervise(args)
        except BaseException:
            if not (args.output/'status.json').exists():
                save(args.output/'status.json',dict(run_id=args.output.name,status='failed',stage='preparation',
                    heartbeat_at=now().isoformat(),exit_code=1,error=traceback.format_exc()))
            raise


if __name__=='__main__':
    main()
