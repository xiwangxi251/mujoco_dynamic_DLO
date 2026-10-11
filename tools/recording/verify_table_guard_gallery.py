"""Verify downloaded artifacts, encoding and local HTTP range responses."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen

directory = Path(sys.argv[1])
root_url = sys.argv[2]
rows = json.loads((directory/'paired_manifest.json').read_text())['results']
showcase = {(r['seed'],r['variant']) for r in json.loads((directory/'video_manifest.json').read_text())}
clips = []
for row in rows:
    episode = directory/row['directory']
    assert hashlib.sha256((episode/'trajectory.npz').read_bytes()).hexdigest() == row['trajectory_sha256']
    if (row['seed'],row['variant']) not in showcase:
        continue
    for view in ('global','wrist'):
        relative = f'{row["directory"]}/{view}.mp4'
        path = directory/relative
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_streams','-show_frames','-show_entries',
            'stream=width,height,avg_frame_rate,nb_frames,codec_name:frame=pict_type,key_frame,best_effort_timestamp_time',
            '-of','json',str(path)]))
        stream, frames = info['streams'][0],info['frames']
        assert (stream['width'],stream['height'],stream['avg_frame_rate'],stream['codec_name']) == (1280,960,'50/1','h264')
        assert len(frames) == int(stream['nb_frames']) == row['frames']
        assert all(abs(float(f['best_effort_timestamp_time'])-i*.02)<1e-6 for i,f in enumerate(frames))
        assert all(f['pict_type'] != 'B' for f in frames)
        keys = [i for i,f in enumerate(frames) if f['key_frame']]
        assert max([b-a for a,b in zip(keys,keys[1:])]+[len(frames)-keys[-1]]) <= 12
        start, end = 2048,3071
        with urlopen(Request(root_url+'/'+relative,headers={'Range':f'bytes={start}-{end}'})) as response:
            status, content_range, body = response.status,response.headers.get('Content-Range'),response.read()
        assert status == 206
        assert content_range == f'bytes {start}-{end}/{path.stat().st_size}'
        with path.open('rb') as source:
            source.seek(start)
            assert body == source.read(end-start+1)
        clips.append(dict(file=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            frames=len(frames),native_resolution=[1280,960],fps=50,b_frames=0,uniform_20ms=True,
            http_range_status=status,range_bytes_match=True))
assert len(clips) == 2*len(showcase)
report = dict(passed=True,downloaded_trajectory_sha_verified=len(rows),clips=clips)
(directory/'local_gallery_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(passed=True,trajectory_checks=len(rows),clips=len(clips),http_range_status=206)))
