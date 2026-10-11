"""One bounded worker-ready confirmation after the launcher has exited."""
import json
from pathlib import Path
import subprocess

ROOT='/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/train/ppo_session_height_ordered_scratch_20m_20261011/rollout_only_20261011'
SCRIPT='''import importlib.util,json,time
from pathlib import Path
p=Path(ROOT)
spec=importlib.util.spec_from_file_location('resume_launcher',p/'run_rollout.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
deadline=time.time()+180
while time.time()<deadline:
    m=r.read(p/'launch_manifest.json');h=r.read(p/'worker_health.json')
    if m and not r.identity(m['pid']):
        print((p/'train.log').read_text(errors='replace')[-5000:])
        raise SystemExit('Trainer exited before worker-ready')
    if h.get('worker_ready') and h.get('parameters_finite') and time.time()-h['updated_at']<300:
        r.confirm()
        break
    time.sleep(2)
else: raise SystemExit('Bounded startup confirmation expired; do not relaunch')
'''
if __name__=='__main__':
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60022','hxai@10.108.17.151',
        '/data1/hxai/miniconda3/envs/dynamicvla/bin/python -'],input='ROOT='+repr(ROOT)+'\n'+SCRIPT,
        text=True,capture_output=True,timeout=210)
    print(p.stdout)
    if p.returncode:
        print(p.stderr);raise SystemExit(p.returncode)
    receipt=json.loads(p.stdout)
    local=Path(__file__).resolve().parent
    local.mkdir(exist_ok=True)
    (local/'startup_receipt.json').write_text(json.dumps(receipt,indent=2))
