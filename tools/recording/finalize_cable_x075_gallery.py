"""Add sequential review controls after all 120 recordings are complete."""
import json
from pathlib import Path
import numpy as np
import record_cable_x075_20episodes as batch


def main():
    root=batch.OUT
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_episodes']==120
    html=(root/'index.html').read_text(encoding='utf-8')
    html=html.replace('<label>回合 <select id="episode">', '<button id="previous-episode">上一回合</button><button id="next-episode">下一回合</button><label>回合 <select id="episode">')
    options=''.join(f'<option value="single{i}">{label}（单独查看）</option>' for i,label in enumerate(batch.LABELS))
    html=html.replace('<option value="all">全部六场景</option>', '<option value="all">全部六场景</option>'+options)
    html=html.replace('combined:[4,5]};', 'combined:[4,5],single0:[0],single1:[1],single2:[2],single3:[3],single4:[4],single5:[5]};')
    html=html.replace("ids.length===2?'repeat(2,minmax(0,1fr))':''", "ids.length===1?'minmax(0,1100px)':ids.length===2?'repeat(2,minmax(0,1fr))':''")
    html=html.replace("$('episode').addEventListener('change',", "function changeEpisode(delta){const n=Math.max(1,Math.min(20,Number($('episode').value)+delta));$('episode').value=String(n);D=ALL.filter(e=>e.episode===n);load();}$('previous-episode').addEventListener('click',()=>changeEpisode(-1));$('next-episode').addEventListener('click',()=>changeEpisode(1));$('episode').addEventListener('change',")
    (root/'index.html').write_text(html,encoding='utf-8')
    data=json.loads((root/'gallery_manifest.json').read_text(encoding='utf-8'))
    physics_checks=[]
    for row in data:
        with np.load(root/row['directory']/'trajectory.npz') as saved:
            finite=bool(np.all(np.isfinite(saved['states'])))
            time_error=float(np.max(abs(saved['frame_times']-np.arange(row['frames'])*.02)))
            physics_checks.append(dict(scene=row['scene'],episode=row['episode'],
                all_fullphysics_states_finite=finite,maximum_time_error_s=time_error,
                uniform_physics_time=bool(time_error<1e-8)))
    for scene in batch.SCENES:
        chosen=[r for r in data if r['scene']==scene]
        assert len(chosen)==20 and len({r['trajectory_sha256'] for r in chosen})==20
    checks={
        'episodes':120, 'distinct_recordings_per_scene':20, 'clips':360,
        'all_robot_qpos_max_change':max(r['robot_qpos_max_change'] for r in data),
        'all_robot_link_position_max_change_m':max(r['robot_link_position_max_change_m'] for r in data),
        'robot_cable_contact_substep_pairs':sum(r['robot_cable_contact_substep_pairs'] for r in data),
        'all_fullphysics_states_finite':all(c['all_fullphysics_states_finite'] for c in physics_checks),
        'all_physics_timestamps_uniform':all(c['uniform_physics_time'] for c in physics_checks),
        'physics_checks':physics_checks,
        'per_scene':[dict(scene=s,
            initial_center_x_min_m=min(r['initial_center_xyz'][0] for r in data if r['scene']==s),
            initial_center_x_max_m=max(r['initial_center_xyz'][0] for r in data if r['scene']==s),
            initial_center_y_min_m=min(r['initial_center_xyz'][1] for r in data if r['scene']==s),
            initial_center_y_max_m=max(r['initial_center_xyz'][1] for r in data if r['scene']==s),
            target_node_indices=sorted({r['target_node'] for r in data if r['scene']==s})) for s in batch.SCENES],
    }
    batch.write_json(root/'inspection_summary.json',checks)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    manifest['worker_plan']={'first_six_reference':4,'ascending_rounds_2_to_20':6,'ahead_rounds_20_to_16':4,
        'ahead_recordings_reused_after_provenance_check':True}
    manifest['driver_sha256']={str(p):batch.base.sha(p) for p in (
        Path(__file__),Path(__file__).with_name('continue_cable_x075_batch.py'),
        Path(__file__).with_name('record_cable_x075_tail.py'))}
    batch.write_json(root/'manifest.json',manifest)
    (root/'README.md').write_text('''# X=0.75 cable scene family: 120 recordings

- Explicit family `cable_x075_v1`, six DEV variants, twenty fresh episodes per scene.
- Paired seeds 20260804..20260823; replay bank must match the seed (no fallback).
- Initial X basis 0.750 m with unchanged ±0.100 m random offsets. Bank snapshots
  retain their original drift and displacement, translated to the new basis.
- Robot fixed at the original ready pose, open fingers. No grasp policy and no
  PPO workspace-conditioned reset. Red target is the original environment draw.
- Yellow rectangle: X [0.60,0.84], Y [-0.22,0.30] metres. XY inclusion is geometric
  overlap only, not task success or robot reachability.
- Three views each: top/global/wrist, independently rendered at native 1280x960,
  50 fps. Real saved 20 ms physics states; no frame interpolation/upscaling.
- H264 CRF16, maximum GOP12, B frames off, faststart. Each encoding is checked
  for resolution, timestamps, actual frame count and keyframe spacing.
- Normal maximum episode 15 s; L1 episodes stop after the original exit line.
  Shorter videos hold their final frame during synchronized comparisons.
- Choose a round and scene in index.html; use previous/next round to review
  all twenty. Default view is top; global and wrist are selectable.
- Original scenario definitions, explicit PPO six-scene list, and earlier
  galleries are preserved. This is a cable placement visualization experiment.

Artifacts: manifest.json (source hashes/config), gallery_manifest.json (all
episodes), summary.json, inspection_summary.json, verification.json (360 HTTP
206 checks). Each episode contains its trajectory, metadata, three videos,
posters and encoding/state verification. Browser QA is recorded separately.
''',encoding='utf-8')
    print(json.dumps({k:v for k,v in checks.items() if k!='physics_checks'},indent=2))


if __name__=='__main__':main()
