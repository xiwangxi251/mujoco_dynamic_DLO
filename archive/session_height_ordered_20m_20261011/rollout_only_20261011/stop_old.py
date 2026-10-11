"""Stop only the authorized old trainer tree after the replacement passes."""
import json
import os
from pathlib import Path
import signal
import time
from prepare import OLD,ROOT,snapshot

def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(birth=s[19],parent=int(s[1]),state=s[0],command=(p/'cmdline').read_bytes().decode().strip('\0').split('\0'))
    except (FileNotFoundError,ProcessLookupError):return None

def sig(pid,birth,value):
    s=identity(pid)
    if s and s['birth']==birth and s['state']!='Z':
        try:os.kill(pid,value)
        except ProcessLookupError:pass

def main():
    receipt=ROOT/'stop_previous_receipt.json'
    if receipt.exists():print(receipt.read_text());return
    assert json.loads((ROOT/'resume_preflight.json').read_text())['status']=='passed'
    m=json.loads((OLD/'launch_manifest.json').read_text())
    assert m['pid']==8621 and m['birth']=='86102' and m['supervisor_pid']==8620 and m['supervisor_birth']=='86096'
    a=identity(m['pid']);s=identity(m['supervisor_pid'])
    assert a and a['birth']==m['birth'] and a['command']==m['command']
    assert s and s['birth']==m['supervisor_birth']
    processes={m['pid']:m['birth'],m['supervisor_pid']:m['supervisor_birth']}
    sig(m['supervisor_pid'],m['supervisor_birth'],signal.SIGSTOP)
    sig(m['pid'],m['birth'],signal.SIGSTOP)
    for _ in range(4):
        allp={int(p.name):identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
        children={p:v['birth'] for p,v in allp.items() if v and v['parent'] in processes and p not in processes}
        processes.update(children)
        for p,b in children.items():sig(p,b,signal.SIGSTOP)
    try:inputs=snapshot()
    except Exception:
        for p,b in processes.items():sig(p,b,signal.SIGCONT)
        raise
    for p,b in processes.items():sig(p,b,signal.SIGTERM);sig(p,b,signal.SIGCONT)
    deadline=time.monotonic()+8
    while time.monotonic()<deadline and any((v:=identity(p)) and v['birth']==b and v['state']!='Z' for p,b in processes.items()):time.sleep(.2)
    for p,b in processes.items():sig(p,b,signal.SIGKILL)
    remaining=[p for p,b in processes.items() if (v:=identity(p)) and v['birth']==b and v['state']!='Z']
    assert not remaining
    out=dict(status='stopped_for_rollout_gates',stopped_at=time.time(),process_tree=processes,remaining_live=remaining,
        checkpoint=inputs['checkpoint'],reason=inputs['reason'])
    receipt.write_text(json.dumps(out,indent=2))
    (OLD/'status.json').write_text(json.dumps(dict(status='stopped_by_user',source_updated_at=time.time(),run=str(OLD),next_run=str(ROOT),reason=inputs['reason'])))
    print(json.dumps(out))

if __name__=='__main__':main()
