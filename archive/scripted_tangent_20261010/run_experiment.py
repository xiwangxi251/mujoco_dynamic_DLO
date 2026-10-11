"""Frozen-source recording coordinator for target-tangent and single joint tests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/scripted_tangent_20261010'
SOURCE = OUT/'source'
BANK = ROOT/'outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks'
SCENES = ['dev_shape_nominal_current_x075_v1','dev_static_midshape_v1_x075_v1']

def save(name,value):
    path=OUT/name; temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); temporary.replace(path)

def freeze():
    if SOURCE.exists():
        provenance=json.loads((SOURCE/'source_provenance.json').read_text(encoding='utf-8'))
        for relative, expected in provenance['source_sha256'].items():
            if hashlib.sha256((SOURCE/relative).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Frozen source changed: '+relative)
        # Resume the verified frozen source even when live development continues.
        return
    for folder in ('src','assets'):
        shutil.copytree(ROOT/folder,SOURCE/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    tools=SOURCE/'tools/recording'; tools.mkdir(parents=True)
    for name in ('record_scripted_scene_family.py','target_node_annotation.py'):
        shutil.copy2(ROOT/'tools/recording'/name,tools/name)
    (SOURCE/'source_provenance.json').write_text(json.dumps(dict(source_root=str(ROOT),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*.py')}
        ),indent=2),encoding='utf-8')
    (SOURCE/'working_changes.patch').write_bytes(subprocess.check_output(['git','diff','--','src','tests'],cwd=ROOT))

def rotation(recording):
    Original=recording.DynamicCableGraspPolicy
    OriginalEnv=recording.CableGraspEnv
    substeps=[]
    class InstrumentedEnv(OriginalEnv):
        def _physics_step(self):
            result=super()._physics_step()
            substeps.append([float(self.data.time), *self.data.qpos[self.arm_qpos_adr], *self.data.qvel[self.arm_dof_adr]])
            return result
    class Joint7Probe(Original):
        def action(self):
            now=float(self.env.data.time)
            command=self.env.ready_ctrl.copy()
            if now >= 1.:
                self.phase=SimpleNamespace(name='JOINT7_MINUS_90')
                command[self.env.arm_actuator_ids[-1]]=self.env.ready_qpos[self.env.arm_qpos_adr[-1]]-np.pi/2
            if now >= 4.:
                self.finished=True; self.result='rotation_probe_complete'
            return command
    recording.DynamicCableGraspPolicy=Joint7Probe
    recording.CableGraspEnv=InstrumentedEnv
    try:
        recording.record(dict(scene='dev_static_x075_v1',seed=20260804,episode=1,
            output=str(OUT/'joint7'),bank=str(BANK),controller_profile='enhanced6_bounded_dls_v1',
            table_guard=True,gpu=0,protocol_sha256='single_joint7_minus90_v1',rule_config={}))
    finally:
        recording.DynamicCableGraspPolicy=Original
        recording.CableGraspEnv=OriginalEnv
    dest=OUT/'joint7/dev_static_x075_v1/seed_20260804'
    metrics=json.loads((dest/'metrics.json').read_text(encoding='utf-8'))
    raw=np.asarray(substeps)
    np.savetxt(OUT/'joint7_500hz.csv',raw,delimiter=',',header='time,'+','.join(f'q{i}' for i in range(1,8))+','+','.join(f'v{i}' for i in range(1,8)),comments='')
    q=raw[:,1:8]; v=raw[:,8:15]; t=raw[:,0]
    start=int(np.abs(t-1).argmin()); goal=q[0,-1]-np.pi/2
    mask=t>=1.; error=np.abs(q[:,-1]-goal); tol=np.radians(2.)
    first=np.flatnonzero(mask & (error<=tol))
    stable=next((i for i in range(start,len(t)) if np.all(error[i:]<=tol)),None)
    statistics=dict(requested_rotation_deg=-90., measurement_hz=500, start_angle_deg=float(np.degrees(q[0,-1])),
        target_angle_deg=float(np.degrees(goal)),final_angle_deg=float(np.degrees(q[-1,-1])),
        peak_actual_speed_deg_s=float(np.max(np.abs(np.degrees(v[mask,-1])))),
        first_within_2deg_seconds=None if not len(first) else float(t[first[0]]-1.),
        stable_within_2deg_seconds=None if stable is None else float(t[stable]-1.),
        other_joints_max_actual_deviation_deg=np.degrees(np.max(np.abs(q-q[0]),axis=0)[:-1]).tolist(),
        control='Only joint7 target changes; enhanced6 servo and physical execution guards stay on.',directory=str(dest.relative_to(OUT)))
    save('joint7_speed.json',statistics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(8,5),sharex=True)
    axes[0].plot(t,np.degrees(q[:,-1]),label='actual joint 7')
    axes[0].plot(t,np.where(t<1,np.degrees(q[0,-1]),np.degrees(goal)),'--',label='command')
    axes[0].set_ylabel('Angle (deg)'); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(t,np.degrees(v[:,-1])); axes[1].set_ylabel('Speed (deg/s)'); axes[1].set_xlabel('Episode time (s)'); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT/'joint7_speed.png',dpi=160); plt.close(fig)
    return statistics

def finish_page():
    # Reuse the established gallery UI, including scrubbing and native-controls behavior.
    template=(ROOT/'outputs/scripted_seven_default_20261009/index.html').read_text(encoding='utf-8')
    (OUT/'gallery_template.html').write_text(template,encoding='utf-8')
    import build_gallery
    build_gallery.build(OUT)

def main():
    OUT.mkdir(parents=True,exist_ok=True); save('status.json',dict(status='starting'))
    try:
        freeze()
        sys.path.insert(0,str(SOURCE/'tools/recording'))
        sys.path.insert(0,str(SOURCE/'src'))
        import record_scripted_scene_family as recording
        save('status.json',dict(status='rotation_recording'))
        stats=(json.loads((OUT/'joint7_speed.json').read_text(encoding='utf-8'))
               if (OUT/'joint7_speed.json').is_file() else rotation(recording))
        (OUT/'scenes').mkdir(exist_ok=True)
        (OUT/'scenes/summary.json').write_text(json.dumps(dict(status='running',episodes=[],by_scene={})),encoding='utf-8')
        finish_page()
        save('status.json',dict(status='scene_recording',joint7_speed=stats))
        arguments=[sys.executable,'-B',str(SOURCE/'tools/recording/record_scripted_scene_family.py'),
            '--output',str(OUT/'scenes'),'--bank',str(BANK),'--workers','3','--episodes','10','--seed','20260804',
            '--follow-target-tangent','--scenes',*SCENES]
        subprocess.run(arguments,cwd=ROOT,check=True)
        summary=json.loads((OUT/'scenes/summary.json').read_text(encoding='utf-8'))
        finish_page()
        save('status.json',dict(status='complete',by_scene=summary['by_scene'],joint7_speed=stats,
            gallery='http://127.0.0.1:8783/index.html',completed_episodes=summary['completed_episodes']))
    except Exception as error:
        save('status.json',dict(status='failed',error=f'{type(error).__name__}: {error}')); raise

if __name__=='__main__':
    main()
