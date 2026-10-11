"""Reuse the frozen actuator correction without changing learned action decoders."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import sys
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/recording'))
import compare_scripted_tracking_fix as frozen

VARIANTS=('baseline','tracking_fix')
SCENES=('id_static','id_rigid_l1_nominal','id_shape_nominal_current','id_combined_l1_nominal')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

class LearnedTrackingEnv(frozen.DiagnosticEnv):
    def __init__(self,config,variant,output):
        frozen.OUT=Path(output)/'control_audit'
        frozen._POLICY=None
        self.experiment_variant=variant
        super().__init__(config,variant)
        self.original_limits=self.model.actuator_forcerange[:7].copy()
        self.initial_state_hash=None
        self.max_joint_tracking_error=np.zeros(7)
        self.sum_joint_tracking_error=np.zeros(7)
        self.tracking_samples=0

    def reset(self,*args,**kwargs):
        result=super().reset(*args,**kwargs)
        if hasattr(self,'max_ff'):
            self.max_ff=np.zeros(7);self.max_total_torque=np.zeros(7);self.force_saturations=0
        self.max_joint_tracking_error=np.zeros(7);self.sum_joint_tracking_error=np.zeros(7);self.tracking_samples=0
        self.initial_state_hash=hashlib.sha256(self.data.qpos.tobytes()+self.data.qvel.tobytes()).hexdigest()
        self.initial_info=result[1].copy()
        return result

    def step(self,action):
        out=super().step(action)
        error=np.abs(self.data.ctrl[:7]-self.data.qpos[self.arm_qpos_adr])
        self.max_joint_tracking_error=np.maximum(self.max_joint_tracking_error,error)
        self.sum_joint_tracking_error+=error;self.tracking_samples+=1
        return out

    def audit(self):
        assert np.allclose(self.model.actuator_forcerange[:7],self.original_limits)
        assert np.all(self.max_total_torque<=np.max(np.abs(self.original_limits),axis=1)+1e-8)
        return {'variant':self.experiment_variant,'physical_model_hash':self.physics_hash,
            'actuator_hash':self.actuator_hash,'initial_state_sha256':self.initial_state_hash,
            'gravity_ff_max_nm':self.max_ff.tolist(),'total_torque_max_nm':self.max_total_torque.tolist(),
            'total_torque_saturation_steps':self.force_saturations,
            'joint_tracking_error_max_rad':self.max_joint_tracking_error.tolist(),
            'joint_tracking_error_mean_rad':(self.sum_joint_tracking_error/max(1,self.tracking_samples)).tolist(),
            'arm_kp_scale':1 if self.experiment_variant=='baseline' else 4,
            'arm_kv_scale':1 if self.experiment_variant=='baseline' else 2,
            'angular_feedback_x4_applied':False,'env_config':asdict(self.config),
            'success_snapshot':self.success_snapshot,'compiled_model':self.model_file}

def check_pairs(rows):
    by={}
    for r in rows:
        key=(r['scenario_name'],int(r['requested_seed']))
        assert r['actual_episode_seed']==r['requested_seed'],key
        assert r['variant'] not in by.setdefault(key,{})
        by[key][r['variant']]=r
    for key,pair in by.items():
        assert set(pair)==set(VARIANTS),(key,set(pair))
        for field in ('scene_fingerprint','base_scene_fingerprint','physical_model_hash','initial_state_sha256','env_config'):
            assert pair['baseline'][field]==pair['tracking_fix'][field],(key,field)
    return by

def summarize_pairs(rows):
    from scipy.stats import binomtest
    pairs=check_pairs(rows)
    def group(keys):
        a=np.array([pairs[k]['baseline']['task_success'] for k in keys],dtype=int)
        b=np.array([pairs[k]['tracking_fix']['task_success'] for k in keys],dtype=int)
        gained=int(np.sum((a==0)&(b==1)));lost=int(np.sum((a==1)&(b==0)))
        rng=np.random.default_rng(20261001)
        delta=b-a
        boots=delta[rng.integers(0,len(keys),(10000,len(keys)))].mean(axis=1)
        def wilson(n,N):
            z=1.959963984540054;p=n/N;den=1+z*z/N
            center=(p+z*z/(2*N))/den;half=z*np.sqrt(p*(1-p)/N+z*z/(4*N*N))/den
            return [float(center-half),float(center+half)]
        return {'paired_cases':len(keys),'baseline_successes':int(a.sum()),'corrected_successes':int(b.sum()),
            'baseline_rate':float(a.mean()),'corrected_rate':float(b.mean()),
            'baseline_wilson95':wilson(a.sum(),len(a)),'corrected_wilson95':wilson(b.sum(),len(b)),
            'difference_percentage_points':float(delta.mean()*100),
            'paired_bootstrap95_percentage_points':(np.quantile(boots,[.025,.975])*100).tolist(),
            'gained_cases':gained,'lost_cases':lost,'both_success':int(np.sum((a==1)&(b==1))),
            'both_failure':int(np.sum((a==0)&(b==0))),
            'mcnemar_exact_p':float(binomtest(gained,gained+lost,.5).pvalue) if gained+lost else 1.}
    result={'overall':group(list(pairs)),'scenarios':{s:group([k for k in pairs if k[0]==s]) for s in SCENES if any(k[0]==s for k in pairs)},
        'intervention':'same learned checkpoint and action decoder; arm gravity feedforward at 500 Hz and kp x4/kv x2; no angular feedback x4',
        'pair_checks_passed':True}
    return result
