"""Safely extract the completed, authenticated server result archive."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/benchmarks/nero_pose_scripted_tracking_fix_20261001'
archive=OUT/'server_results.tar.gz'
assert archive.stat().st_size==539135145
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers()
    for member in members:
        assert not member.issym() and not member.islnk()
        target=(OUT/member.name).resolve()
        assert target.is_relative_to((OUT/'server_export').resolve()),member.name
        assert member.isfile() or member.isdir(),member.name
    tar.extractall(OUT,members=members,filter='data')
meta={'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
    'member_count':len(members),'all_paths_confined_to_server_export':True,
    'archive_gzip_integrity_read_completed':True}
(OUT/'server_import_verification.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
