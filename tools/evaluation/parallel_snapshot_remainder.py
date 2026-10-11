"""Supervise isolated snapshot shards and merge existing local results."""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
def save(path,value):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=8);p.add_argument('--fail-probe',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    out=(a.output or ROOT/'outputs'/('pilot' if a.pilot else 'parallel')).resolve();out.mkdir(parents=True,exist_ok=True)
    start=datetime.now(timezone.utc).isoformat(); progress=start;count=0;children=[]
    def status(state,**more):
        save(out/'status.json',dict(status=state,run='snapshot_motion_3s_100_20261010',
            completed=count,planned=total,pid=os.getpid(),start_at=start,
            heartbeat_at=datetime.now(timezone.utc).isoformat(),last_progress_at=progress,
            progress_deadline_at=(datetime.fromisoformat(progress)+timedelta(seconds=240)).isoformat(),
            heartbeat_timeout_seconds=180,execution_host=socket.gethostname(),workers=a.workers,**more))
    with (ROOT/'inputs/samples.csv').open(encoding='utf-8-sig',newline='') as f: original=list(csv.DictReader(f))
    previous=json.loads((ROOT/'inputs/completed.json').read_text(encoding='utf-8'))
    done={(r['seed'],r['snapshot_fraction']) for r in previous}
    jobs=original[:1] if a.pilot else [r for r in original if (int(r['seed']),float(r['snapshot_fraction'])) not in done]
    if a.pilot: previous=[]
    count=len(previous);total=count+len(jobs)
    try:
        status('starting')
        if a.fail_probe: raise RuntimeError('Intentional parallel supervisor failure probe')
        shards=[jobs[i::min(a.workers,len(jobs))] for i in range(min(a.workers,len(jobs)))]
        logs=[]
        for i,items in enumerate(shards):
            folder=out/f'shard_{i}';folder.mkdir(exist_ok=True)
            with (folder/'input.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
            log=(folder/'worker.log').open('wb');logs.append(log)
            (folder/'result').mkdir(exist_ok=True)
            save(folder/'result/status.json',dict(status='starting',completed=0))
            cmd=[sys.executable,'-B','-X','utf8',str(ROOT/'tools/evaluation/check_snapshot_motion_after_settle.py'),
                 '--samples-csv',str(folder/'input.csv'),'--bank-dir',str(ROOT/'inputs/bank'),
                 '--output',str(folder/'result'),'--limit',str(len(items))]
            child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
            children.append((child,folder,len(items)))
        save(out/'launch_receipt.json',dict(supervisor_pid=os.getpid(),started_at=start,
             workers=[dict(pid=c.pid,folder=str(f),planned=n) for c,f,n in children],
             local_completed=len(previous),remaining=len(jobs)))
        began=time.monotonic()
        while True:
            next_count=len(previous);finished=0
            for child,folder,n in children:
                sp=folder/'result/status.json';s=json.loads(sp.read_text(encoding='utf-8')) if sp.exists() else {}
                if s.get('status')=='failed' or child.poll() not in (None,0):
                    raise RuntimeError(s.get('error') or (folder/'worker.log').read_text(errors='replace')[-2000:])
                next_count+=s.get('completed',0)
                if child.poll()==0:
                    if s.get('status')!='complete':raise RuntimeError('Worker exited without complete result')
                    finished+=1
            if next_count!=count:count=next_count;progress=datetime.now(timezone.utc).isoformat()
            status('running')
            if finished==len(children):break
            if time.monotonic()-began>900:raise TimeoutError('Parallel run exceeded 15 minutes')
            time.sleep(3)
        rows=previous[:]
        for _,folder,_ in children:
            rows+=json.loads((folder/'result/samples.json').read_text(encoding='utf-8'))
        keyed={(r['seed'],r['snapshot_fraction']):r for r in rows}
        if len(keyed)!=len(rows) or len(rows)!=total:raise ValueError('Duplicate or missing results')
        order={(int(r['seed']),float(r['snapshot_fraction'])):i for i,r in enumerate(original)}
        rows.sort(key=lambda r:order[r['seed'],r['snapshot_fraction']]);save(out/'samples.json',rows)
        with (out/'samples.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        kept=[r for r in rows if not r['penetration_gt_5mm']];baselines={}
        for b in (.2,.3):
            k=f'from_{int(b*1000)}ms';gt=sum(r[k+'_gt_5mm'] for r in kept)
            values=sorted(r[k+'_max_mm'] for r in kept);n=len(values)
            baselines[str(b)]=dict(displacement_gt_5mm_all=sum(r[k+'_gt_5mm'] for r in rows),
                displacement_gt_5mm_after_filter=gt,displacement_gt_5mm_after_filter_percent=100.*gt/n if n else None,
                median_displacement_mm_after_filter=(values[n//2]+values[(n-1)//2])/2 if n else None,
                robot_contact_samples_after_filter=sum(r[k+'_robot_contact_steps']>0 for r in kept))
        result=dict(samples=len(rows),initial_penetration_gt_5mm=len(rows)-len(kept),retained_after_filter=len(kept),
             by_baseline=baselines,per_sample_csv=str(out/'samples.csv'),
             sampling='Original 100 snapshots: 50 trajectories at fixed fractions 0.35 and 0.65; no resampling',
             metric='Maximum node 3D distance relative to each baseline through 3 seconds, sampled at 50Hz',
             local_completed=len(previous),parallel_completed=len(jobs),execution_host=socket.gethostname(),workers=a.workers,
             limitation='Absolute displacement includes overall motion; not a pure bending metric')
        save(out/'summary.json',result)
        if not a.pilot:
            with tarfile.open(out/'results.tar.gz','w:gz') as archive:
                for name in ('samples.csv','samples.json','summary.json','launch_receipt.json'):
                    archive.add(out/name,arcname=name)
                for _,folder,_ in children:
                    for file in (folder/'result').glob('*.npz'):archive.add(file,arcname=file.name)
        progress=datetime.now(timezone.utc).isoformat();status('complete',results=result,report=str(out/'summary.json'))
    except BaseException:
        for child,_,_ in children:
            if child.poll() is None:child.terminate()
        status('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
