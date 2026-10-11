"""Prepare inputs before stopping the live identity-checked training job."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile
import torch

BASE=Path('/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/train/ppo_session_height_ordered_scratch_20m_20261011')
OLD=BASE/'resume_after_reboot_20261011'
ROOT=BASE/'rollout_only_20261011'

def finite(x):
    if torch.is_tensor(x):return bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
    return True

def snapshot():
    candidates=sorted((OLD/'checkpoints').glob('*.zip'),key=lambda p:int(p.stem.split('_')[-2]),reverse=True)
    for p in candidates:
        try:
            with zipfile.ZipFile(p) as z:
                assert z.testzip() is None
                d=json.loads(z.read('data'))
                for name in ('policy.pth','policy.optimizer.pth'):
                    with z.open(name) as f:assert finite(torch.load(f,map_location='cpu',weights_only=False))
            break
        except (OSError,AssertionError,zipfile.BadZipFile):continue
    else:raise RuntimeError('No valid finite checkpoint')
    dest=ROOT/'inputs'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest==hashlib.sha256(dest.read_bytes()).hexdigest()
    record=dict(path=str(dest),original_path=str(p),sha256=digest,timesteps=d['num_timesteps'],
        optimizer_epochs=d['_n_updates'],entropy=d['ent_coef'],phase=d.get('ordinary_evaluation_phase',0))
    phase=record['phase']
    out=dict(checkpoint=record,restored_phase=phase,qualifying_evaluations=0,
        remaining_timesteps=20000000-record['timesteps'],budget=20000000,
        commit='87b59957db9be2f3d6daf287d4af1b4d49a284b3',prepared_at=time.time(),
        schedule='Original cumulative 20M LR/entropy and penalty ramps preserved',
        reason='User authorized removal of independent evaluations and per-scene training rollout promotion',
        original_manifest=json.loads((OLD/'launch_manifest.json').read_text()))
    (ROOT/'resume_plan.json').write_text(json.dumps(out,indent=2))
    # Keep historical curves through checkpoint; promotion windows start fresh.
    with (OLD/'training_metrics.csv').open() as f:
        r=csv.DictReader(f);fields=r.fieldnames;rows=[x for x in r if int(x['timesteps'])<=record['timesteps']]
    with (ROOT/'training_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return out

if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'launch_manifest.json').exists()
    for name in ('plan.json','reference_config.json','preflight.json','eval_bank_receipt.json'):
        shutil.copy2(BASE/name,ROOT/name)
    print(json.dumps(snapshot()))
