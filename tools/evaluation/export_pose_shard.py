"""Export only the completed server shard, replication evidence and models."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import mujoco
import numpy
import scipy

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/benchmarks/nero_pose_scripted_tracking_fix_20261001'
jobs=json.loads((ROOT/'jobs_server.json').read_text())
rows=json.loads((OUT/'shards/server/episodes.json').read_text())
assert len(rows)==len(jobs)
manifest=json.loads((ROOT/'pose_bundle_manifest.json').read_text())
for p,h in manifest['sha256'].items():
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
metadata={'hostname':platform.node(),'platform':platform.platform(),'python':sys.version,
    'mujoco':mujoco.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,
    'workers':24,'cpu_count':os.cpu_count(),'server_rows':len(rows),
    'jobs_sha256':hashlib.sha256((ROOT/'jobs_server.json').read_bytes()).hexdigest(),
    'all_bundled_hashes_unchanged':True}
(ROOT/'server_runtime.json').write_text(json.dumps(metadata,indent=2))
archive=ROOT/'server_results.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in ('shards/server','models','validation'):
        tar.add(OUT/p,arcname='server_export/'+p)
    for p in ('jobs_server.json','pose_bundle_manifest.json','server_runtime.json'):
        tar.add(ROOT/p,arcname='server_export/'+p)
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'metadata':metadata},indent=2))
