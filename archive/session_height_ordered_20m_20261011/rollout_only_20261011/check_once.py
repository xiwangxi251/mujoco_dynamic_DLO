"""One compact authoritative read, with one bounded stale/missing fallback."""
import json
from pathlib import Path
import subprocess
import time

LOCAL = Path(__file__).resolve().parent
ROOT = '/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/train/ppo_session_height_ordered_scratch_20m_20261011/rollout_only_20261011'
SCRIPT = '''import json,time
from pathlib import Path
p=Path(ROOT)
try:s=json.loads((p/'status.json').read_text())
except FileNotFoundError:s=dict(status='missing',run=ROOT,reason='Source status missing',alert_key='missing')
terminal=s.get('status')=='complete'
if not terminal and s.get('status')!='missing' and time.time()-s.get('source_updated_at',0)>300:
    s.update(status='monitor_stale',reason='Source status older than 300 seconds',alert_key='monitor_stale')
if s.get('status') in ('monitor_stale','missing','monitor_failed'):
    m=json.loads((p/'launch_manifest.json').read_text()) if (p/'launch_manifest.json').exists() else {}
    def identity(pid):
        try:
            q=Path('/proc')/str(pid);st=(q/'stat').read_text().rsplit(')',1)[1].split()
            return dict(birth=st[19],state=st[0],command=(q/'cmdline').read_bytes().decode().strip('\\0').split('\\0'))
        except FileNotFoundError:return None
    s['fallback']=dict(manifest=m,training_identity=identity(m.get('pid')),supervisor_identity=identity(m.get('supervisor_pid')),
        health=json.loads((p/'worker_health.json').read_text()) if (p/'worker_health.json').exists() else {})
if s.get('status') in ('running','starting') and time.time()>s.get('progress_deadline_at',0):
    s.update(status='needs_attention',reason='Real progress deadline exceeded',alert_key='progress_deadline')
print(json.dumps(s))
'''
try:
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60022','hxai@10.108.17.151',
        '/data1/hxai/miniconda3/envs/dynamicvla/bin/python -'], input='ROOT='+repr(ROOT)+'\n'+SCRIPT,
        text=True,capture_output=True,timeout=45,check=True)
    out = json.loads(result.stdout)
    out.update(sync_state='complete',synced_at=time.time())
except Exception as error:
    out = dict(status='sync_failed',run=ROOT,reason=str(error),alert_key='sync_failed',synced_at=time.time())
    if (LOCAL/'status.json').exists(): out['last_known'] = json.loads((LOCAL/'status.json').read_text())
p = LOCAL/'status.tmp'
p.write_text(json.dumps(out,indent=2),encoding='utf-8')
p.replace(LOCAL/'status.json')
print(json.dumps(out))
