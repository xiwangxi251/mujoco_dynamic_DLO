"""Paired NERO scripted grasp diagnostics; production files remain unchanged.

Uses the project's benchmark episode runner, task success and failure taxonomy.
Interventions are isolated in subclasses: joint gravity feedforward, kp x4/kv x2,
and (in a separate arm) orientation feedback x4. No grasp geometry, cable
physics, speed limits, target selection or state-machine changes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time

import mujoco
import numpy as np
from scipy.stats import binomtest

from panda_cable_grasp.env.environment import CableGraspEnv
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig
from panda_cable_grasp.evaluation import benchmark as bench
from panda_cable_grasp.scenarios.registry import get_scenario, list_suite_scenarios

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/recording'))
from render_vertical_workspace import make_model

OUT=ROOT/'outputs/benchmarks/nero_scripted_tracking_fix_20261001'
VARIANTS=('baseline','tracking_fix','tracking_fix_orientation4')
_GRAVITY_MODEL=None
_ENV=None
_POLICY=None
_JOB=None


def digest_arrays(model, attrs):
    h=hashlib.sha256()
    for attr in attrs:
        a=np.asarray(getattr(model,attr)); h.update(attr.encode()); h.update(a.tobytes())
    h.update(model.opt.gravity.tobytes()); h.update(str(model.opt.timestep).encode())
    return h.hexdigest()


class DiagnosticEnv(CableGraspEnv):
    def __init__(self, config, variant):
        global _ENV, _GRAVITY_MODEL
        super().__init__(config)
        self.variant=variant
        self.force_saturations=0
        self.max_ff=np.zeros(7)
        self.max_total_torque=np.zeros(7)
        self.phase_metrics={}
        self.trace=[]
        self.control_steps=0
        self.physics_hash=digest_arrays(self.model,(
            'body_mass','body_inertia','body_pos','body_quat','body_gravcomp',
            'dof_damping','dof_armature','dof_frictionloss','jnt_range','jnt_axis','jnt_pos',
            'geom_size','geom_friction','geom_solref','geom_solimp','geom_contype','geom_conaffinity',
            'actuator_ctrlrange','actuator_forcerange','jnt_actfrcrange','eq_data'))
        assert not config.geometric_safety_enabled, 'the diagnostic torque hook requires normal physics stepping'
        if variant != 'baseline':
            self.model.actuator_gainprm[:7,0] *= 4
            self.model.actuator_biasprm[:7,1] *= 4
            self.model.actuator_biasprm[:7,2] *= 2
            assert np.allclose(self.model.actuator_gear[:7,0],1)
            assert np.all(self.model.actuator_dyntype[:7]==0)
            if _GRAVITY_MODEL is None:
                _GRAVITY_MODEL=make_model()
            self.gravity_model=_GRAVITY_MODEL
            self.gravity_data=mujoco.MjData(self.gravity_model)
            assert np.allclose(self.gravity_model.opt.gravity,self.model.opt.gravity)
            for name in ('base_link',*(f'link{i}' for i in range(1,8)),'gripper_link1','gripper_link2'):
                a,b=self.model.body(name).id,self.gravity_model.body(name).id
                assert a==b
                for attr in ('body_mass','body_inertia','body_pos','body_quat'):
                    assert np.allclose(getattr(self.model,attr)[a],getattr(self.gravity_model,attr)[b])
            assert np.allclose(self.model.jnt_qposadr[:9],self.gravity_model.jnt_qposadr)
        self.actuator_hash=digest_arrays(self.model,('actuator_gainprm','actuator_biasprm'))
        model_file=OUT/'models'/f'{config.scenario_name}_{variant}_{self.physics_hash[:12]}.mjb'
        model_file.parent.mkdir(parents=True,exist_ok=True)
        if not model_file.exists():
            temporary=model_file.with_suffix(f'.{os.getpid()}.tmp')
            mujoco.mj_saveModel(self.model,str(temporary),None)
            os.replace(temporary,model_file)
        self.model_file=str(model_file.relative_to(OUT))
        _ENV=self

    def _physics_step(self):
        if self.variant != 'baseline':
            self.gravity_data.qpos[:]=self.data.qpos[:9]
            self.gravity_data.qvel[:]=0
            mujoco.mj_forward(self.gravity_model,self.gravity_data)
            ff=self.gravity_data.qfrc_bias[:7].copy()
            q=self.data.qpos[self.arm_qpos_adr]
            v=self.data.qvel[self.arm_dof_adr]
            pd=(self.model.actuator_gainprm[:7,0]*self.data.ctrl[:7]
                +self.model.actuator_biasprm[:7,0]
                +self.model.actuator_biasprm[:7,1]*q
                +self.model.actuator_biasprm[:7,2]*v)
            bounds=self.model.actuator_forcerange[:7]
            pd=np.clip(pd,bounds[:,0],bounds[:,1])
            total=np.clip(pd+ff,bounds[:,0],bounds[:,1])
            self.force_saturations+=int(np.any(abs(total-pd-ff)>1e-9))
            self.data.qfrc_applied[self.arm_dof_adr]=total-pd
            self.max_ff=np.maximum(self.max_ff,abs(total-pd))
            self.max_total_torque=np.maximum(self.max_total_torque,abs(total))
        return super()._physics_step()

    def step(self, action):
        result=super().step(action)
        self.control_steps+=1
        policy=_POLICY
        if policy is not None:
            phase=policy.phase.name
            tilt=float(np.degrees(policy._tilt_error()))
            qerror=float(np.linalg.norm(self.data.ctrl[:7]-self.data.qpos[self.arm_qpos_adr]))
            metric=self.phase_metrics.setdefault(phase,{'steps':0,'tilt_sum_deg':0.,'max_tilt_deg':0.,'joint_error_sum_rad':0.})
            metric['steps']+=1; metric['tilt_sum_deg']+=tilt
            metric['max_tilt_deg']=max(metric['max_tilt_deg'],tilt)
            metric['joint_error_sum_rad']+=qerror
            if self.control_steps%5==0:
                self.trace.append(np.r_[self.data.time,self.data.qpos,self.data.ctrl,
                    self.hand_position,self.target_position(),tilt,policy.phase.value])
        return result


class DiagnosticPolicy(DynamicCableGraspPolicy):
    def __init__(self, env, config):
        global _POLICY
        super().__init__(env,config)
        _POLICY=self

    def _orientation_error(self):
        error,angle=super()._orientation_error()
        if self.env.variant=='tracking_fix_orientation4':
            error=4*error
        return error,angle


def worker(job):
    global _ENV, _POLICY, _JOB
    _ENV=None; _POLICY=None; _JOB=job
    variant=job['variant']
    bench.CableGraspEnv=partial(DiagnosticEnv,variant=variant)
    bench.DynamicCableGraspPolicy=DiagnosticPolicy
    started=time.monotonic()
    row=bench._run_policy_episode(job)
    env=_ENV
    assert env is not None
    row.update(variant=variant,wall_seconds=time.monotonic()-started,
        compiled_model=env.model_file,physical_model_hash=env.physics_hash,
        actuator_hash=env.actuator_hash,gravity_ff_max_nm=env.max_ff.tolist(),
        total_torque_max_nm=env.max_total_torque.tolist(),
        total_torque_saturation_steps=env.force_saturations,
        phase_tracking=env.phase_metrics,env_config=asdict(env.config),
        scripted_config=asdict(_POLICY.config),success_snapshot=env.success_snapshot,
        grasp_break_history=env.grasp_break_history)
    trace_dir=OUT/job['stage']/'traces'; trace_dir.mkdir(parents=True,exist_ok=True)
    file=trace_dir/f"{job['scenario'].name}_{job['seed']}_{variant}.npz"
    np.savez_compressed(file,states=np.asarray(env.trace),nq=env.model.nq,
        sample_hz=10.,columns_note='time,qpos[nq],ctrl[8],tcp[3],target[3],tilt_deg,phase_id')
    row['diagnostic_trace']=str(file.relative_to(OUT))
    return row


def jobs(stage,seeds,scenarios):
    return [dict(method='scripted',variant=v,stage=stage,episode=i+1,seed=s,
        scenario=scenario,disturbance=1.5,episode_seconds=15.,robot='nero',
        recording=False,compiled_model='provided_by_diagnostic_env',output_dir=str(OUT),
        strict_vertical_gripper=False,scripted_target_selection='middle',
        arm_actuator_gain_scale=1.,scripted_options={})
        for i,s in enumerate(seeds) for scenario in scenarios for v in VARIANTS]


def run(stage,seeds,scenarios,workers):
    folder=OUT/stage; folder.mkdir(parents=True,exist_ok=True)
    file=folder/'episodes.jsonl'
    rows=[]
    if file.exists():
        rows=[json.loads(s) for s in file.read_text(encoding='utf-8').splitlines()]
    done={(r['scenario_name'],r['requested_seed'],r['variant']) for r in rows}
    pending=[j for j in jobs(stage,seeds,scenarios) if (j['scenario'].name,j['seed'],j['variant']) not in done]
    total=len(seeds)*len(scenarios)*len(VARIANTS)
    started=time.monotonic()
    with file.open('a',encoding='utf-8') as stream,mp.get_context('spawn').Pool(workers) as pool:
        for row in pool.imap_unordered(worker,pending,chunksize=1):
            stream.write(json.dumps(row,allow_nan=True)+'\n'); stream.flush(); rows.append(row)
            progress={'stage':stage,'completed':len(rows),'total':total,'elapsed_wall_s':time.monotonic()-started,
                'successes':{v:sum(r['task_success'] for r in rows if r['variant']==v) for v in VARIANTS}}
            (folder/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
            print(f"{stage} {len(rows)}/{total} {row['scenario_name']} seed={row['requested_seed']} {row['variant']}: "
                f"success={row['task_success']} outcome={row['task_failure_type']} wall={row['wall_seconds']:.1f}s",flush=True)
    (folder/'episodes.json').write_text(json.dumps(rows,indent=2,allow_nan=True),encoding='utf-8')
    bench._write_csv(folder/'episodes.csv',rows)
    summarize(folder,rows,seeds,scenarios)
    return rows


def paired_stats(a,b):
    assert len(a)==len(b)
    aa=np.array(a,dtype=int); bb=np.array(b,dtype=int)
    improved=int(np.sum((aa==0)&(bb==1))); worsened=int(np.sum((aa==1)&(bb==0)))
    difference=bb-aa
    rng=np.random.default_rng(1012026)
    boot=difference[rng.integers(0,len(a),size=(20000,len(a)))].mean(axis=1)
    return {'n':len(a),'both_success':int(np.sum((aa==1)&(bb==1))),
        'both_failure':int(np.sum((aa==0)&(bb==0))),'improved':improved,'worsened':worsened,
        'delta_rate':float(difference.mean()),'paired_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
        'mcnemar_exact_p':float(binomtest(improved,improved+worsened,.5).pvalue) if improved+worsened else 1.}


def summarize(folder,rows,seeds,scenarios):
    bykey={(r['scenario_name'],r['requested_seed'],r['variant']):r for r in rows}
    assert len(bykey)==len(rows)==len(seeds)*len(scenarios)*len(VARIANTS)
    for scenario in scenarios:
        for seed in seeds:
            triplet=[bykey[scenario.name,seed,v] for v in VARIANTS]
            for field in ('scene_fingerprint','base_scene_fingerprint','scenario_id','motion_profile_hash','physical_model_hash'):
                assert len({r[field] for r in triplet})==1,(scenario.name,seed,field)
            assert triplet[1]['actuator_hash']==triplet[2]['actuator_hash']
            assert triplet[0]['actuator_hash']!=triplet[1]['actuator_hash']
            assert all(r['actual_episode_seed']==seed for r in triplet)
            assert all(r['scripted_config']==triplet[0]['scripted_config'] for r in triplet)
    summary={}
    for scenario in scenarios:
        name=scenario.name
        group={v:[bykey[name,s,v] for s in seeds] for v in VARIANTS}
        summary[name]={'variants':{v:bench._aggregate(g) for v,g in group.items()},
            'comparisons':{v:paired_stats([r['task_success'] for r in group['baseline']],
                [r['task_success'] for r in group[v]]) for v in VARIANTS[1:]}}
    groups={v:[r for r in rows if r['variant']==v] for v in VARIANTS}
    summary['overall']={'variants':{v:bench._aggregate(g) for v,g in groups.items()},
        'comparisons':{v:paired_stats([bykey[sc.name,s,'baseline']['task_success'] for sc in scenarios for s in seeds],
            [bykey[sc.name,s,v]['task_success'] for sc in scenarios for s in seeds]) for v in VARIANTS[1:]}}
    # The six scenes share each seed. Resample the seed block rather than
    # pretending all scene/seed outcomes are independent in the pooled CI.
    for v in VARIANTS[1:]:
        block_delta=np.array([np.mean([
            int(bykey[sc.name,s,v]['task_success'])-int(bykey[sc.name,s,'baseline']['task_success'])
            for sc in scenarios]) for s in seeds])
        rng=np.random.default_rng(1012026)
        boot=block_delta[rng.integers(0,len(seeds),size=(20000,len(seeds)))].mean(axis=1)
        c=summary['overall']['comparisons'][v]
        c['paired_bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
        c['bootstrap_unit']='paired seed block including all six scenes'
        c['n_seed_blocks']=len(seeds)
        c['mcnemar_exact_p']=None
        c['mcnemar_note']='reported per scene only; pooled scenes share seeds'
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (folder/'verification.json').write_text(json.dumps({'unique_rows':len(rows),
        'paired_scene_and_motion_fingerprints_match':True,'physical_parameters_match_except_declared_actuator_gains':True,
        'same_success_criterion':True,'same_policy_config':True,'paired_seed_list':list(seeds)},indent=2),encoding='utf-8')
    print(json.dumps(summary['overall'],indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('pilot','main'),default='pilot')
    parser.add_argument('--episodes',type=int,default=50)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    scenarios=list_suite_scenarios('core')
    if args.stage=='pilot':
        seeds=[2026100000]
    else:
        seeds=list(range(2026100100,2026100100+args.episodes))
    manifest=OUT/args.stage/'manifest.json'; manifest.parent.mkdir(parents=True,exist_ok=True)
    source_paths=['src/panda_cable_grasp/policies/scripted.py','src/panda_cable_grasp/env/environment.py',
        'src/panda_cable_grasp/evaluation/benchmark.py','src/panda_cable_grasp/scenarios/registry.py',
        'src/panda_cable_grasp/evaluation/failure_taxonomy.py','assets/mujoco/nero/nero.xml']
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'robot':'nero','stage':args.stage,
        'scope':'paired engineering ablation; historical paper results not replaced',
        'seeds':seeds,'episodes_per_scenario':len(seeds),'scenarios':[asdict(s) for s in scenarios],
        'variants':{'baseline':'current unchanged scripted and execution',
            'tracking_fix':'same scripted; static arm gravity feedforward at 500Hz; arm kp x4 kv x2; sum torque clamped to original limits',
            'tracking_fix_orientation4':'tracking_fix plus quaternion orientation-error feedback x4 before original angular speed limit; stage angles unchanged'},
        'episode_seconds':15.,'control_hz':50.,'physics_hz':500.,'success':'original environment ever_success/task_success, continuous 0.80s',
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'policy_default':asdict(PolicyConfig()),'workers':args.workers,
        'python':platform.python_version(),'mujoco':mujoco.__version__}
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        for key in ('seeds','variants','source_sha256','script_sha256'):
            assert old[key]==plan[key],f'resumption manifest mismatch: {key}'
    else:
        manifest.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    run(args.stage,seeds,scenarios,args.workers)


if __name__=='__main__':
    main()
