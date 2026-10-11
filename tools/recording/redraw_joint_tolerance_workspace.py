"""Separate 0.15-degree actual-joint acceptance experiment; physics unchanged."""
import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
import experiment_bounded_full6 as exp

OUT=exp.OUT/'joint_tolerance_015'
TOL=.15
RIG=None
VARIANT=None


def original():
    data={v:json.loads((exp.OUT/v/'cells.json').read_text()) for v in ('baseline','bounded')}
    for v in data:
        manifest=json.loads((exp.OUT/v/'manifest.json').read_text())
        for path,digest in manifest['source_sha256'].items():
            assert hashlib.sha256(exp.Path(path).read_bytes()).hexdigest()==digest,path
    return data


def final_ok(r):
    return (r['position_error_mm']<=5 and r['pose_error_deg']<=2 and .2<=r['finger_floor_mm']<=8
        and r['robot_floor_mm']>=.2 and r['joint_margin_deg']>=-TOL and r['self_distance_mm']>=1)


def draw(data,path,preview=False):
    xs=np.array(sorted({r['x_m'] for r in data['baseline']}));ys=np.array(sorted({r['y_m'] for r in data['baseline']}))
    masks={};counts={};fig,axes=plt.subplots(1,3,figsize=(18,7.6))
    fig.subplots_adjust(top=.82,bottom=.14,wspace=.21,left=.045,right=.99)
    extent=(xs[0]-.02,xs[-1]+.02,ys[0]-.02,ys[-1]+.02)
    for ax,v,label in ((axes[0],'baseline','Enhanced6 / original DLS'),(axes[1],'bounded','Enhanced6 / bounded DLS')):
        mask=np.zeros((len(ys),len(xs)),dtype=bool)
        for r in data[v]:mask[np.argmin(abs(ys-r['y_m'])),np.argmin(abs(xs-r['x_m']))]=final_ok(r) if preview else r['endpoint_pass']
        masks[v]=mask;counts[v]=dict(cells=int(mask.sum()),area_m2=float(mask.sum()*.0016),
            negative_y=int(mask[ys<0].sum()),positive_y=int(mask[ys>0].sum()))
        ax.imshow(mask,origin='lower',extent=extent,cmap=ListedColormap(['#e8edf2','#87bce8']),norm=BoundaryNorm([-.5,.5,1.5],2),interpolation='nearest')
        ax.set_title(label)
        kind='Final-state eligible (hold unverified)' if preview else 'Endpoint held >=2 s within 16 s'
        ax.text(.02,.02,f"{kind}\n{mask.sum()}/961 | Area ~{mask.sum()*.0016:.4f} m²\nY<0: {mask[ys<0].sum()} | Y>0: {mask[ys>0].sum()}",transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.94,edgecolor='none'))
    diff=masks['bounded'].astype(int)-masks['baseline'].astype(int)
    axes[2].imshow(diff,origin='lower',extent=extent,cmap=ListedColormap(['#d95660','#e8edf2','#4a9fc9']),norm=BoundaryNorm([-1.5,-.5,.5,1.5],3),interpolation='nearest')
    axes[2].set_title('New solver change: blue gained / red lost')
    counts['transition']=dict(gained=int((diff>0).sum()),lost=int((diff<0).sum()))
    for ax in axes:
        ax.set_aspect('equal');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.grid(alpha=.15);ax.plot(.35,0,'o',color='#223b53',markersize=4)
    fig.suptitle(('PREVIEW: final state only; 2 s hold not verified' if preview else 'Verified 2 s endpoint hold; 16 s simulation limit')+'\nActual joint tolerance 0.15 deg | position 5 mm / pose 2 deg | same physics, floor and self-collision criteria')
    fig.legend([Patch(color='#e8edf2'),Patch(color='#87bce8')],['Not eligible at final state' if preview else 'Endpoint not passed','Final-state eligible' if preview else 'Endpoint passed'],loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False)
    fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig)
    return counts


def worker(job):
    global RIG,VARIANT
    v,x,y=job
    if RIG is None or VARIANT!=v:
        RIG=None
        gc.collect()
        RIG=exp.ComparisonRig(v);VARIANT=v
        measure=RIG.measure
        def relaxed(target,collision=False):
            result=list(measure(target,collision=collision));result[4]+=TOL;return tuple(result)
        RIG.measure=relaxed
    row=RIG.trial(x,y);row['joint_margin_deg']-=TOL
    row.update(variant=v,joint_tolerance_deg=TOL,provenance='fresh physical rerun; only actual-joint acceptance tolerance changed')
    if row['endpoint_pass']:assert final_ok(row)
    return row


def main(mode,workers):
    OUT.mkdir(parents=True,exist_ok=True);data=original()
    if mode=='preview':
        counts=draw(data,OUT/'preview_final_state.png',True)
        (OUT/'preview_summary.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts));return
    results={v:[] for v in data};done_keys=set()
    progress=OUT/'rerun_cells.jsonl'
    if progress.exists():
        for line in progress.read_text().splitlines():
            r=json.loads(line);key=(r['variant'],r['x_m'],r['y_m']);assert key not in done_keys
            done_keys.add(key);results[r['variant']].append(r)
    jobs=[]
    for v,rows in data.items():
        for r in rows:
            if r['endpoint_pass']:
                copy=dict(r,variant=v,joint_tolerance_deg=TOL,provenance='reused strict 0-degree pass; its >=2-second hold proves relaxed criterion')
                results[v].append(copy)
            elif (v,r['x_m'],r['y_m']) not in done_keys:
                jobs.append((v,r['x_m'],r['y_m']))
    index={(v,r['x_m'],r['y_m']):r for v,rows in data.items() for r in rows}
    jobs.sort(key=lambda j:(j[0],not final_ok(index[j]),j[2],j[1]))
    started=time.monotonic();total=len(jobs);count=0
    with progress.open('a',encoding='utf-8') as log,mp.get_context('spawn').Pool(workers) as pool:
        for r in pool.imap_unordered(worker,jobs,chunksize=1):
            results[r['variant']].append(r);log.write(json.dumps(r)+'\n');log.flush();count+=1
            if count%100==0 or count==12 or count==total:
                print(json.dumps(dict(done=count,total=total,wall_s=round(time.monotonic()-started,1),confirmed_passes={v:sum(t['endpoint_pass'] for t in rows) for v,rows in results.items()})),flush=True)
            if count==12:
                # Fast confirmed subset; unrerun failures are explicitly unverified.
                known={(v,t['x_m'],t['y_m']):t for v,rows in results.items() for t in rows}
                provisional={v:[known.get((v,r['x_m'],r['y_m']),r) for r in rows] for v,rows in data.items()}
                counts=draw(provisional,OUT/'provisional_confirmed.png')
                # This intermediate figure is not the full new-standard scan.
                (OUT/'provisional_warning.txt').write_text('Only newly eligible final-state candidates and strict passes are verified. Other original failures remain unverified under the relaxed hold criterion. Use workspace_comparison.png when the full scan finishes.')
    for v in results:
        assert len(results[v])==len({(r['x_m'],r['y_m']) for r in results[v]})==961
        results[v].sort(key=lambda r:(r['y_m'],r['x_m']))
        (OUT/f'{v}_cells.json').write_text(json.dumps(results[v],indent=2),encoding='utf-8')
    counts=draw(results,OUT/'workspace_comparison.png')
    counts['criteria']=dict(actual_joint_tolerance_deg=TOL,hold_s=2,maximum_simulation_s=16,position_mm=5,pose_deg=2,physics_unchanged=True)
    counts['rerun_failure_points']=1476
    counts['counts']={v:{key:sum(r['reason']==key for r in rows) for key in sorted({r['reason'] for r in rows})} for v,rows in results.items()}
    counts['added_over_strict']={v:sum(r['endpoint_pass'] for r in results[v])-sum(r['endpoint_pass'] for r in data[v]) for v in data}
    (OUT/'summary.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts,indent=2),flush=True)
    (OUT/'README.md').write_text('# 实际关节限位容差0.15°的独立重评估\n\n仅改变实际关节的终点验收容差为0.15°。JIK位置/速度边界、模型物理限位、伺服、重力前馈、原有限速均保持。位置5mm、完整姿态2°、离桌/自碰撞条件、连续2秒保持、16秒总时限保持。原零容差已通过的446个点直接保留；其余1476点均从同一home重新运行物理仿真，避免仅用最终状态误判连续保持。preview_final_state.png仅为结束状态预览；workspace_comparison.png为完整保持时间核验结果。\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('preview','scan'),required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args();main(a.mode,a.workers)
