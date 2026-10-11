"""Build the user-authorized, credential-free frozen 151 experiment snapshot."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'outputs/benchmarks/nero_scripted_tracking_fix_20261001'
OUT=ROOT/'outputs/benchmarks/nero_pose_scripted_tracking_fix_20261001'
catalog=json.loads((OLD/'bundle_manifest.json').read_text(encoding='utf-8'))
files=[p for p in catalog['sha256'] if p.startswith(('src/','assets/'))]
files+=['tools/recording/render_vertical_workspace.py',
    'tools/evaluation/compare_scripted_tracking_fix.py','tools/evaluation/pose_scripted_entry.py',
    'tools/evaluation/compare_scripted_pose_fix.py','tools/evaluation/pose_shard_run.py',
    'tools/evaluation/report_scripted_pose_fix.py']
for stage in ('main','pilot'):
    for filename in ('manifest.json','episodes.json'):
        files.append(str((OLD/stage/filename).relative_to(ROOT)).replace('\\','/'))
for p in ('pilot/episodes.json','pilot/manifest.json','main/manifest.json','main/source_snapshot.py'):
    files.append(str((OUT/p).relative_to(ROOT)).replace('\\','/'))
files=list(dict.fromkeys(files))
metadata={'purpose':'user-authorized partial Pose IK experiment on project 151',
    'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
manifest=OUT/'pose_bundle_manifest.json'
manifest.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
archive=OUT/'pose_source_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in files:
        tar.add(ROOT/p,arcname=p,recursive=False)
    tar.add(manifest,arcname='pose_bundle_manifest.json',recursive=False)
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(files)}))
