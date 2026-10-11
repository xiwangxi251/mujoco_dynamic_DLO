"""Frozen-controller trajectories for acceptance analysis in the requested box."""
import argparse
import hashlib
import json
import multiprocessing as mp
import time
import numpy as np
import scan_bounded_table_guard_workspace as guard

OUT=guard.OUT/'requested_box'
RIG=None
SOURCE=None
XS=np.round(np.arange(.62,.999,.04),6)
YS=np.round(np.arange(-.48,.481,.04),6)
CORNERS=((.6,-.5),(1.,-.5),(.6,.5),(1.,.5))
TIERS=[('strict',5,2,5),('path8_only',5,2,8),('moderate',10,3,8),
       ('moderate_path10',10,3,10),('looser_diagnostic',15,5,10)]


def sources():
    frozen=json.loads((guard.OUT/'manifest.json').read_text())['source_sha256']
    guard.assert_sources(frozen)
    return {**frozen,str(guard.Path(__file__).resolve()):guard.digest(__file__)}


def init(hashed):
    global SOURCE
    SOURCE=hashed


def worker(job):
    global RIG
    if RIG is None:RIG=guard.GuardRig(True)
    guard.assert_sources(SOURCE)
    x,y=job;rig=RIG;rig.reset();target=np.array([x,y,.005]);approach=np.array([x,y,.10])
    history=[];qpos=[];started=time.monotonic();termination='16s_completed'
    for step in range(800):
        rig.control(approach if step==0 else target,guard.exp.base.native.Phase.INTERCEPT)
        m=rig.measure(target,collision=True)
        finger=min(float(rig.geometry.geom_vertices(g)[:,2].min()) for g in rig.geometry.finger_geoms)
        ratio=float(np.max(rig.peak_velocity/rig.env._arm_velocity_limits))
        history.append([float(rig.data.time),m[0]*1000,m[2],finger*1000,m[3]*1000,m[4]-.15,m[5]*1000,ratio,m[1]])
        qpos.append(rig.data.qpos.copy())
        if m[3]<-.001:termination='floor_penetration';break
        if m[5]<-.001:termination='self_collision';break
        if not np.all(np.isfinite(rig.data.qpos)):termination='invalid_nonfinite';break
    history=np.asarray(history)
    name=f'{x:.2f}_{y:+.2f}'
    np.savez_compressed(OUT/'traces'/f'{name}.npz',history=history,qpos=qpos,
        columns=np.array(['time_s','position_mm','pose_deg','finger_mm','robot_floor_mm','joint_margin_deg','self_distance_mm','peak_velocity_ratio','tilt_deg']))
    assert rig.solve_failures==0 and rig.command_clip_steps==0
    assert rig.table_guard.steps==len(history)*10
    return dict(x_m=x,y_m=y,termination=termination,steps=len(history),wall_seconds=time.monotonic()-started,
        final=dict(zip(['time_s','position_mm','pose_deg','finger_mm','robot_floor_mm','joint_margin_deg','self_distance_mm','peak_velocity_ratio','tilt_deg'],history[-1].tolist())),
        guard=rig.table_guard.summary(),trace=f'traces/{name}.npz')


def acceptance(history,pos,pose,path):
    safe=(history[:,3]>=.2)&(history[:,3]<=8)&(history[:,4]>=.2)&(history[:,5]>=-.15)&(history[:,6]>=1)
    acceptable=safe&(history[:,1]<=pos)&(history[:,2]<=pose)
    held=0;first=None
    for i,a in enumerate(acceptable):
        held=held+1 if a else 0
        if held>=100:first=i;break
    motionfirst=None
    prefixpose=np.maximum.accumulate(history[:,2])
    if first is not None:
        # Path maxima only increase; later hold cannot fix an earlier path violation.
        if prefixpose[first]<=path and history[first,7]<=1.01:motionfirst=first
    return dict(endpoint=first is not None,motion=motionfirst is not None,
        endpoint_time_s=float(history[first,0]) if first is not None else None,
        path_pose_at_endpoint_deg=float(prefixpose[first]) if first is not None else None,
        velocity_ratio_at_endpoint=float(history[first,7]) if first is not None else None)


def scan(workers):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'traces').mkdir(exist_ok=True)
    hashed=sources();mpath=OUT/'manifest.json'
    if mpath.exists():assert json.loads(mpath.read_text())['source_sha256']==hashed
    else:
        mpath.write_text(json.dumps(dict(source_sha256=hashed,box_x_m=[.6,1.],box_y_m=[-.5,.5],
            grid_cell_centers_x_m=XS.tolist(),grid_cell_centers_y_m=YS.tolist(),additional_corners=CORNERS,
            grid_step_m=.04,total_box_area_m2=.4,maximum_trial_s=16,hold_s=2,actual_joint_tolerance_deg=.15,
            controller_unchanged=True,guard_config_unchanged=True,full_16s_no_endpoint_early_stop=True,
            collision_check_every_control_frame=True,tiers=TIERS),indent=2))
    log=OUT/'cells.jsonl';rows=[json.loads(t) for t in log.read_text().splitlines()] if log.exists() else []
    done={(r['x_m'],r['y_m']) for r in rows}
    jobs=[(float(x),float(y)) for y in YS for x in XS]+list(CORNERS)
    jobs=[j for j in jobs if j not in done]
    start=time.monotonic()
    with log.open('a',encoding='utf-8') as stream,mp.get_context('spawn').Pool(workers,initializer=init,initargs=(hashed,)) as pool:
        for row in pool.imap_unordered(worker,jobs,chunksize=1):
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            progress=dict(completed=len(rows),total=254,elapsed_s=round(time.monotonic()-start,1))
            (OUT/'progress.json').write_text(json.dumps(progress))
            if len(rows)%25==0:print(json.dumps(progress),flush=True)
    assert len(rows)==len({(r['x_m'],r['y_m']) for r in rows})==254
    guard.assert_sources(hashed)
    (OUT/'cells.json').write_text(json.dumps(rows,indent=2))
    analyze()


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    rows=json.loads((OUT/'cells.json').read_text());bykey={(r['x_m'],r['y_m']):r for r in rows}
    history={k:np.load(OUT/r['trace'])['history'] for k,r in bykey.items()}
    all_results={};summary={};masks={}
    for name,pos,pose,path in TIERS:
        results={k:acceptance(h,pos,pose,path) for k,h in history.items()}
        all_results[name]=[dict(x_m=k[0],y_m=k[1],**r) for k,r in results.items()]
        masks[name]=np.array([[3 if results[(float(x),float(y))]['motion'] else 2 if results[(float(x),float(y))]['endpoint'] else 0 for x in XS] for y in YS])
        endpoint=masks[name]>=2;motion=masks[name]==3
        summary[name]=dict(position_mm=pos,pose_deg=pose,path_pose_deg=path,endpoint_cells=int(endpoint.sum()),motion_cells=int(motion.sum()),
            endpoint_area_m2=round(float(endpoint.sum()*.0016),4),motion_area_m2=round(float(motion.sum()*.0016),4),
            endpoint_box_coverage_percent=round(float(endpoint.mean()*100),1),motion_box_coverage_percent=round(float(motion.mean()*100),1),
            corners=[dict(x_m=x,y_m=y,**results[(x,y)]) for x,y in CORNERS])
    # Threshold frontier: safety unchanged, genuine continuous hold evaluated on all histories.
    frontier=[]
    for pos in (5,10,15,20,30,50,100,200,300):
        for pose in (2,3,5,10,15,30):
            passed=sum(acceptance(history[(float(x),float(y))],pos,pose,180)['endpoint'] for y in YS for x in XS)
            frontier.append(dict(position_mm=pos,pose_deg=pose,endpoint_cells=passed,coverage_percent=passed/250*100))
    # Best possible with arbitrary pose/position tolerance; retain all safety/hold criteria.
    safety_only=sum(acceptance(history[(float(x),float(y))],1e9,180,180)['endpoint'] for y in YS for x in XS)
    summary['safety_only_upper_bound_cells']=safety_only
    summary['safety_only_upper_bound_area_m2']=round(safety_only*.0016,4)
    summary['frontier']=frontier
    summary['box_area_m2']=.4;summary['grid_cells']=250;summary['additional_corner_points']=4
    # Largest axis-aligned rectangle made exclusively of successful sampled cells.
    def rectangle(mask):
        best=(0,0,0,0,0)
        for bottom in range(len(YS)):
            eligible=np.ones(len(XS),bool)
            for top in range(bottom,len(YS)):
                eligible &= mask[top]
                start=0
                for right in range(len(XS)+1):
                    if right<len(XS) and eligible[right]:continue
                    count=(right-start)*(top-bottom+1)
                    if count>best[0]:best=(count,start,right-1,bottom,top)
                    start=right+1
        count,left,right,bottom,top=best
        return dict(cells=count,area_m2=round(count*.0016,4),
            x_cell_edges_m=[round(float(XS[left]-.02),3),round(float(XS[right]+.02),3)] if count else None,
            y_cell_edges_m=[round(float(YS[bottom]-.02),3),round(float(YS[top]+.02),3)] if count else None,
            interpretation='Every 4 cm cell center in this rectangle passed; no continuous reachability guarantee.')
    for name in masks:
        summary[name]['largest_endpoint_rectangle']=rectangle(masks[name]>=2)
        summary[name]['largest_motion_rectangle']=rectangle(masks[name]==3)
    colors=['#e8edf2','#ef8878','#8abfec','#239879']
    fig,axes=plt.subplots(1,4,figsize=(15,8));fig.subplots_adjust(top=.83,bottom=.14,left=.05,right=.98,wspace=.38)
    for ax,name in zip(axes,('strict','path8_only','moderate','looser_diagnostic')):
        s=summary[name];ax.imshow(masks[name],origin='lower',extent=(.6,1.,-.5,.5),cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,4),4),interpolation='nearest')
        ax.set_title(f"{s['position_mm']} mm / {s['pose_deg']} deg\nPath <= {s['path_pose_deg']} deg",fontsize=11)
        ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_xticks([.6,.8,1.]);ax.grid(alpha=.15)
        ax.text(.02,.02,f"Endpoint {s['endpoint_cells']}/250\n{s['endpoint_area_m2']:.4f} m² ({s['endpoint_box_coverage_percent']}%)\nMotion {s['motion_cells']}/250\n{s['motion_area_m2']:.4f} m² ({s['motion_box_coverage_percent']}%)",transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.95,edgecolor='none'))
    fig.suptitle('Requested box: X 0.6-1.0 m / Y -0.5 to 0.5 m / Z 5 mm\nSame guarded bounded enhanced6; original velocity limits, safety criteria, 16 s and 2 s hold retained')
    fig.legend([Patch(color=colors[i]) for i in (0,2,3)],['Endpoint failed','Endpoint only','Full motion passed'],loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,frameon=False)
    fig.savefig(OUT/'acceptance_comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'acceptance_cells.json').write_text(json.dumps(all_results,indent=2))
    worst=sorted([dict(x_m=k[0],y_m=k[1],**r['final']) for k,r in bykey.items()],key=lambda r:r['position_mm'],reverse=True)[:10]
    (OUT/'worst_final_states.json').write_text(json.dumps(worst,indent=2))
    guard.assert_sources(json.loads((OUT/'manifest.json').read_text())['source_sha256'])
    print(json.dumps({k:v for k,v in summary.items() if k!='frontier'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('scan','analyze'),default='scan');p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    scan(a.workers) if a.mode=='scan' else analyze()
