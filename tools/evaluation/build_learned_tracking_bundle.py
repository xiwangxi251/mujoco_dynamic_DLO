"""Create a credential-free source snapshot for the authorized 151 experiment."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/benchmarks/nero_learned_tracking_fix_20261001'
OUT.mkdir(parents=True,exist_ok=True)
old=json.loads((ROOT/'outputs/benchmarks/nero_scripted_tracking_fix_20261001/bundle_manifest.json').read_text())
files=[p for p in old['sha256'] if p.startswith('assets/')]
files += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'src').rglob('*.py')]
files += ['pyproject.toml','tools/recording/render_vertical_workspace.py','tools/evaluation/compare_scripted_tracking_fix.py',
    'tools/evaluation/learned_tracking_common.py','tools/evaluation/compare_state_ppo_tracking_fix.py',
    'tools/evaluation/learned_dynamicvla_server.py','tools/evaluation/learned_dynamicvla_client.py',
    'tools/evaluation/run_learned_tracking_151.py','tools/evaluation/build_learned_tracking_bundle.py',
    'tools/evaluation/verify_learned_tracking_fix.py','tools/evaluation/report_learned_tracking_fix.py']
files += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'configs').rglob('*') if p.is_file()]
files=list(dict.fromkeys(files))
manifest={'purpose':'paired state-PPO and DVLA evaluation of frozen gravity feedforward + kp x4/kv x2',
    'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
    'main_scenes':['id_static','id_rigid_l1_nominal','id_shape_nominal_current','id_combined_l1_nominal'],
    'main_seeds':list(range(2026101000,2026101050)),'pilot_seeds':[2026100900],
    'episodes_per_method_per_arm':200,'main_episodes_total':800,
    'learned_checkpoints_frozen':True,'angular_feedback_x4_applied':False,'workspace_restricted':False,
    'success_criterion':'unchanged environment public task_success','scope':'engineering paired evaluation; does not replace frozen manuscript results'}
path=OUT/'learned_bundle_manifest.json';path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
archive=OUT/'learned_source_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in files:tar.add(ROOT/p,arcname=p,recursive=False)
    tar.add(path,arcname='learned_bundle_manifest.json',recursive=False)
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(files)}))
