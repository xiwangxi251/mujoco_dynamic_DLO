"""Independent enhanced6 bounded-DLS workspace experiment with the user's guard."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import gc
import experiment_bounded_full6 as exp
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/rl'))
import table_execution_guard as tg

OUT=exp.OUT/'table_guard_015'
TOL=.15
RIG=None
FROZEN=None

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sources():
    previous=json.loads((exp.OUT/'bounded/manifest.json').read_text())['source_sha256']
    for p,h in previous.items():assert digest(p)==h,p
    return {**previous,str(Path(__file__).resolve()):digest(__file__),str(Path(tg.__file__).resolve()):digest(tg.__file__)}

def assert_sources(expected):
    for p,h in expected.items():assert digest(p)==h,f'Source changed during experiment: {p}'

def relax(rig):
    measure=rig.measure
    def accepted(target,collision=False):
        values=list(measure(target,collision=collision));values[4]+=TOL
        return tuple(values)
    rig.measure=accepted

class GuardRig(exp.ComparisonRig):
    def __init__(self,enabled=True):
        self.table_guard_enabled=enabled
        super().__init__('bounded')
        self.table_geometry=tg.TableGeometry(self.env)
        self.table_guard=tg.TableExecutionGuard(self.env,self.table_geometry)
        self.original_velocity_guard=self.env._velocity_guarded_action
        def execution_hook(action):
            requested,active=self.original_velocity_guard(action)
            points=self.table_geometry.points()
            self.observe_points(points)
            if self.table_guard_enabled:
                target=self.table_guard.project(requested,points)
                self.max_guard_target_delta_rad=max(self.max_guard_target_delta_rad,float(np.max(abs(target[:7]-requested[:7]))))
                return target,active or self.table_guard.last_active
            return requested,active
        self.env._velocity_guarded_action=execution_hook
        relax(self)
        self.reset()

    def reset(self):
        super().reset()
        self.min_monitored_clearance_m=np.inf
        self.max_guard_target_delta_rad=0.
        if hasattr(self,'table_guard'):
            g=self.table_guard
            g.steps=g.interventions=g.feedback_steps=g.residual_steps=0
            g.last_active=False;g.last_min_clearance=np.inf

    def observe_points(self,points):
        self.min_monitored_clearance_m=min(self.min_monitored_clearance_m,min((p[1] for p in points),default=np.inf))

    def control(self,target,phase):
        super().control(target,phase)
        self.observe_points(self.table_geometry.points())

    def trial(self,x,y):
        row=super().trial(x,y);row['joint_margin_deg']-=TOL
        row.update(table_guard_enabled=self.table_guard_enabled,joint_tolerance_deg=TOL,
            table_guard=self.table_guard.summary(),
            min_monitored_table_clearance_mm=float(self.min_monitored_clearance_m*1000) if np.isfinite(self.min_monitored_clearance_m) else None,
            max_guard_target_delta_rad=self.max_guard_target_delta_rad,
            provenance='fresh physical rerun; user TableExecutionGuard after original execution velocity guard at 500 Hz')
        assert row['table_guard']['guard_steps']==row['steps']*10 if self.table_guard_enabled else row['table_guard']['guard_steps']==0
        return row

def capture(rig,x,y):
    states=[];references=[];original=rig.control
    def record(target,phase):
        original(target,phase);states.append(rig.data.qpos.copy());references.append(np.asarray(target).copy())
    rig.control=record
    row=rig.trial(x,y);rig.control=original
    return row,np.asarray(states),np.asarray(references)

def pilot():
    OUT.mkdir(parents=True,exist_ok=True)
    hashed=sources();goals=[(.71,0),(.67,-.24),(.55,-.36),(.55,.36),(.47,.16)]
    direct=exp.ComparisonRig('bounded');relax(direct)
    expected,qpos,refs=capture(direct,*goals[0]);expected['joint_margin_deg']-=TOL
    gains={n:getattr(direct.model,n).copy() for n in ('actuator_gainprm','actuator_biasprm','actuator_forcerange','jnt_range')}
    del direct;gc.collect()
    off=GuardRig(False)
    actual,off_states,off_refs=capture(off,*goals[0])
    difference=float(np.max(abs(qpos-off_states)))
    assert difference==0 and np.array_equal(refs,off_refs)
    compared=['reason','steps','endpoint_pass','motion_pass','position_error_mm','pose_error_deg','joint_margin_deg','min_path_floor_mm','peak_velocity_to_limit']
    for key in compared:assert expected[key]==actual[key],key
    del off;gc.collect()
    protected=GuardRig(True)
    for n,values in gains.items():assert np.array_equal(getattr(protected.model,n),values),n
    assert asdict(protected.table_guard.config)==asdict(tg.GuardConfig())
    assert refs[:3].tolist()==[[.71,0.,.10],[.71,0.,.005],[.71,0.,.005]]
    audits=[]
    for x,y in goals:
        row,states,reference=capture(protected,x,y)
        assert row['table_guard']['intervention_steps']>0
        np.savez_compressed(OUT/f'pilot_protected_{x}_{y}.npz',qpos=states,reference=reference)
        audits.append(row);print(json.dumps(row),flush=True)
    result=dict(off_max_qpos_difference=difference,off_result_matches_direct=True,
        model_gains_limits_force_ranges_unchanged=True,guard_config_unmodified=True,
        execution_hook='after original _velocity_guarded_action; every 2 ms physics substep',
        references_first_three=refs[:3].tolist(),counter_reset_each_trial=True,
        source_sha256=hashed,protected_pilots=audits)
    assert_sources(hashed)
    (OUT/'pilot_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

def worker(job):
    global RIG
    if RIG is None:RIG=GuardRig(True)
    # Never combine results from different user guard implementations.
    assert digest(tg.__file__)==FROZEN[str(Path(tg.__file__).resolve())]
    return RIG.trial(*job)

def initialize(hashes):
    global FROZEN
    FROZEN=hashes

def grid():
    manifest=json.loads((exp.OUT/'bounded/manifest.json').read_text())
    return np.asarray(manifest['grid_x_m']),np.asarray(manifest['grid_y_m'])

def scan(workers):
    OUT.mkdir(parents=True,exist_ok=True)
    hashed=sources();xs,ys=grid()
    manifest_path=OUT/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text());assert manifest['source_sha256']==hashed
    else:
        audit=json.loads((OUT/'pilot_audit.json').read_text());assert audit['source_sha256']==hashed
        manifest=dict(source_sha256=hashed,guard_config=asdict(tg.GuardConfig()),controller='bounded enhanced6',
            grid_x_m=xs.tolist(),grid_y_m=ys.tolist(),grid_step_m=.04,actual_joint_acceptance_tolerance_deg=TOL,
            control_dt_s=.02,physics_dt_s=.002,maximum_trial_s=16.,hold_s=2.,position_mm=5.,pose_deg=2.,
            finger_floor_mm=[.2,8.],robot_floor_min_mm=.2,self_distance_min_mm=1.,
            reference_protocol='first 20 ms frame upper z=.10; subsequent frames final z=.005; no actual-arrival gate or reference speed ramp',
            execution_hook='user TableExecutionGuard.project after existing execution velocity guard at every 2 ms substep',
            baseline='joint_tolerance_015/bounded_cells.json',
            coverage='link7, flange, palm, both fingers; lower-arm links not protected')
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    path=OUT/'protected_cells.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done={(r['x_m'],r['y_m']) for r in rows};assert len(done)==len(rows)
    jobs=[(float(x),float(y)) for y in ys for x in xs if (float(x),float(y)) not in done]
    started=time.monotonic()
    with path.open('a',encoding='utf-8') as stream,mp.get_context('spawn').Pool(workers,initializer=initialize,initargs=(hashed,)) as pool:
        for row in pool.imap_unordered(worker,jobs,chunksize=1):
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            progress=dict(completed=len(rows),total=961,endpoint_pass=sum(r['endpoint_pass'] for r in rows),
                motion_pass=sum(r['motion_pass'] for r in rows),elapsed_s=round(time.monotonic()-started,1),
                reasons=dict(Counter(r['reason'] for r in rows)))
            (OUT/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
            if len(rows)%25==0:print(json.dumps(progress),flush=True)
    assert_sources(hashed);assert len(rows)==961
    rows.sort(key=lambda r:(r['y_m'],r['x_m']))
    (OUT/'protected_cells.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    report()

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    xs,ys=grid();rows=dict(off=json.loads((exp.OUT/'joint_tolerance_015/bounded_cells.json').read_text()),
        on=json.loads((OUT/'protected_cells.json').read_text()))
    expected={(float(x),float(y)) for y in ys for x in xs}
    verification={};summary={};masks={};npdata=dict(xs=xs,ys=ys)
    for variant,data in rows.items():
        assert len(data)==961 and {(r['x_m'],r['y_m']) for r in data}==expected
        for r in data:
            assert r['steps']<=800 and abs(r['sim_seconds']-.02*r['steps'])<1e-9
            if r['endpoint_pass']:
                assert r['reason']=='endpoint_held_2s' and r['steps']>=100
                assert r['position_error_mm']<=5 and r['pose_error_deg']<=2
                assert .2<=r['finger_floor_mm']<=8 and r['robot_floor_mm']>=.2
                assert r['joint_margin_deg']>=-.15 and r['self_distance_mm']>=1
            if variant=='on':
                assert r['table_guard_enabled'] and r['table_guard']['guard_steps']==10*r['steps']
                assert r['table_guard']['config']==asdict(tg.GuardConfig())
            assert r['solver_failures']==0 and r['joint_target_clip_steps']==0
        mask=np.zeros((len(ys),len(xs)),bool)
        for r in data:mask[np.argmin(abs(ys-r['y_m'])),np.argmin(abs(xs-r['x_m']))]=r['endpoint_pass']
        masks[variant]=mask;npdata[f'{variant}_endpoint_pass']=mask
        passrows=[r for r in data if r['endpoint_pass']]
        summary[variant]=dict(endpoint_cells=len(passrows),sampled_area_m2=len(passrows)*.0016,
            motion_cells=sum(r['motion_pass'] for r in data),failure_reasons=dict(Counter(r['reason'] for r in data)),
            x_m=[min(r['x_m'] for r in passrows),max(r['x_m'] for r in passrows)] if passrows else None,
            y_m=[min(r['y_m'] for r in passrows),max(r['y_m'] for r in passrows)] if passrows else None,
            negative_y_cells=sum(r['y_m']<0 for r in passrows),positive_y_cells=sum(r['y_m']>0 for r in passrows),
            zero_y_cells=sum(r['y_m']==0 for r in passrows))
        verification[variant]=dict(unique_grid_points=961,all_passes_final_criteria_verified=True,steps_and_sim_time_verified=True,
            solver_failures=0,pre_guard_joint_target_clip_steps=0)
    protected=rows['on'];clearances=[r['min_monitored_table_clearance_mm'] for r in protected if r['min_monitored_table_clearance_mm'] is not None]
    summary['guard']=dict(config=asdict(tg.GuardConfig()),coverage='link7, flange, palm, both fingers; lower-arm links not protected',
        intervention_trials=sum(r['table_guard']['intervention_steps']>0 for r in protected),
        intervention_substeps=sum(r['table_guard']['intervention_steps'] for r in protected),
        feedback_substeps=sum(r['table_guard']['velocity_feedback_steps'] for r in protected),
        projection_residual_trials=sum(r['table_guard']['projection_residual_steps']>0 for r in protected),
        projection_residual_substeps=sum(r['table_guard']['projection_residual_steps'] for r in protected),
        total_guard_substeps=sum(r['table_guard']['guard_steps'] for r in protected),
        minimum_monitored_table_clearance_mm=min(clearances),
        monitored_penetration_trials=sum(r['min_monitored_table_clearance_mm'] is not None and r['min_monitored_table_clearance_mm']<0 for r in protected))
    diff=masks['on'].astype(int)-masks['off'].astype(int)
    summary['transition']=dict(gained=int((diff>0).sum()),lost=int((diff<0).sum()),net=int(diff.sum()))
    summary['criteria']=dict(position_mm=5,pose_deg=2,actual_joint_tolerance_deg=.15,continuous_hold_s=2,maximum_trial_s=16)
    summary['interpretation']='Cell area is a sampled workspace estimate, not bounding-box area or proof of global reachability.'
    np.savez_compressed(OUT/'workspace.npz',**npdata)
    fig,axes=plt.subplots(1,3,figsize=(18,7.6));fig.subplots_adjust(top=.82,bottom=.14,wspace=.21,left=.045,right=.99)
    extent=(xs[0]-.02,xs[-1]+.02,ys[0]-.02,ys[-1]+.02)
    for ax,variant,title in ((axes[0],'off','Bounded enhanced6 / guard OFF'),(axes[1],'on','Bounded enhanced6 / guard ON')):
        ax.imshow(masks[variant],origin='lower',extent=extent,cmap=ListedColormap(['#e8edf2','#87bce8']),norm=BoundaryNorm([-.5,.5,1.5],2),interpolation='nearest')
        ax.set_title(title)
        s=summary[variant]
        ax.text(.02,.02,f"Endpoint: {s['endpoint_cells']}/961 | sampled area {s['sampled_area_m2']:.4f} m²\nY<0: {s['negative_y_cells']} | Y>0: {s['positive_y_cells']}\nMotion: {s['motion_cells']}/961",transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.94,edgecolor='none'))
    axes[2].imshow(diff,origin='lower',extent=extent,cmap=ListedColormap(['#d95660','#e8edf2','#4a9fc9']),norm=BoundaryNorm([-1.5,-.5,.5,1.5],3),interpolation='nearest')
    axes[2].set_title(f"Guard effect: gained {summary['transition']['gained']} / lost {summary['transition']['lost']}")
    for ax in axes:
        ax.set_aspect('equal');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.grid(alpha=.15);ax.plot(.35,0,'o',color='#223b53',markersize=4)
    fig.suptitle('User table execution guard ON/OFF | same bounded enhanced6, home and 4 cm grid\nEndpoint held 2 s within 16 s | position 5 mm / full pose 2 deg / actual joint tolerance 0.15 deg')
    fig.legend([Patch(color=c) for c in ('#e8edf2','#87bce8','#4a9fc9','#d95660')],['Endpoint not passed','Endpoint passed','Gained with guard','Lost with guard'],loc='lower center',bbox_to_anchor=(.5,.045),ncol=4,frameon=False)
    fig.savefig(OUT/'workspace_comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
    hashed=json.loads((OUT/'manifest.json').read_text())['source_sha256'];assert_sources(hashed)
    verification['source_hashes_unchanged']=True
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text('# 碰桌保护独立范围实验\n\n当前带约束增强6，执行层在原有速度保护之后、每2ms物理子步调用用户的TableExecutionGuard.project，所有GuardConfig参数保留默认值。保护关闭基线来自完整0.15°容差重评估，保护开启的961点均从同一home重新运行。两者同一4cm网格、Z=5mm、第一帧上方目标Z=10cm然后直接最终目标、16秒总时限含连续2秒保持。\n\n位置≤5mm、完整姿态≤2°、手指离桌0.2–8mm、被检查机器人几何离地≥0.2mm、自碰撞距离≥1mm、实际关节越界容差0.15°。全程运动另要求最大姿态误差≤5°与峰值速度比≤1.01。模型限位/原有限速/增强伺服/重力前馈未变。\n\n保护覆盖link7、flange、palm和两手指；不保护下方臂段。因此机器人任意部位的floor失败不能等同于被保护夹爪穿桌。projection_residual_steps记录投影后目标仍未满足全部下降约束的子步，不能宣称硬安全保证。范围面积为通过格点数×0.0016m²，包围盒内部可能有失败点。\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('pilot','scan','report'),required=True);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.mode=='pilot':pilot()
    elif args.mode=='scan':scan(args.workers)
    else:report()
