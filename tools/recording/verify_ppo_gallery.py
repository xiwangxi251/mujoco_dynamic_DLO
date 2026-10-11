"""Verify native PPO camera clips, local byte ranges and source provenance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen

directory, url = Path(sys.argv[1]), sys.argv[2].rstrip('/')
manifest = json.loads((directory/'manifest.json').read_text())
annotation = json.loads((directory/'clean_annotation_manifest.json').read_text())
assert manifest['passed'] and annotation['passed']
rows = manifest['episodes']
expected_sha = sys.argv[3]
review_path = directory/'wrist_review_manifest.json'
if (directory/'wrist_tips_manifest.json').exists():
    review_path = directory/'wrist_tips_manifest.json'
review = json.loads(review_path.read_text()) if review_path.exists() else None
review_by_seed = {r['seed']:r for r in review['episodes']} if review else {}
clips, states = [], []
for row, annotated in zip(rows, annotation['episodes']):
    assert row['seed'] == annotated['seed']
    assert row['checkpoint_sha256'] == expected_sha
    episode = directory/row['directory']
    metadata = json.loads((episode/'metadata.json').read_text())
    assert metadata['no_renderer_rerun_max_state_difference'] <= 1e-12
    assert annotated['rendered_state_max_difference'] == 0 and annotated['source_trajectory_unchanged']
    states.append(dict(seed=row['seed'],sha256=hashlib.sha256((episode/'trajectory.npz').read_bytes()).hexdigest()))
    for view in ('global','wrist'):
        relative = annotated['files'][view]
        if view == 'wrist' and row['seed'] in review_by_seed:
            entry = review_by_seed[row['seed']]
            assert entry['source_trajectory_unchanged'] and entry['rendered_state_max_difference'] == 0
            relative = entry['file']
        path = directory/relative
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_streams','-show_frames','-show_entries',
            'stream=width,height,avg_frame_rate,nb_frames,codec_name:frame=pict_type,key_frame,best_effort_timestamp_time',
            '-of','json',str(path)]))
        stream, frames = info['streams'][0], info['frames']
        assert (stream['width'],stream['height'],stream['avg_frame_rate'],stream['codec_name']) == (1280,960,'50/1','h264')
        assert len(frames) == row['frames'] == int(stream['nb_frames'])
        assert all(abs(float(f['best_effort_timestamp_time'])-i*.02)<1e-6 for i,f in enumerate(frames))
        assert all(f['pict_type'] != 'B' for f in frames)
        keys = [i for i,f in enumerate(frames) if f['key_frame']]
        assert max([b-a for a,b in zip(keys,keys[1:])]+[len(frames)-keys[-1]]) <= 12
        with urlopen(Request(url+'/'+relative,headers={'Range':'bytes=2048-3071'})) as response:
            assert response.status == 206
            assert response.headers['Content-Range'] == f'bytes 2048-3071/{path.stat().st_size}'
            payload = response.read()
        with path.open('rb') as source:
            source.seek(2048)
            assert payload == source.read(1024)
        clips.append(dict(file=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            frames=len(frames),native_resolution=[1280,960],fps=50,uniform_20ms=True,b_frames=0,max_gop_frames=12,http_range_status=206))
report = dict(passed=True,checkpoint_sha256=expected_sha,source_trajectories=states,clips=clips)
(directory/'local_gallery_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(passed=True,episodes=len(rows),clips=len(clips),checkpoint_steps=rows[0]['checkpoint_steps'])))
