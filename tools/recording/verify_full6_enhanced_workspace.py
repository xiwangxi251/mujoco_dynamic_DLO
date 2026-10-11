"""Validate full scan completeness, physical success criteria and provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vertical_workspace/scripted_136_retest_20261004/full6_gravity_servo_gain10'

def main():
    rows=json.loads((OUT/'cells.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text())
    indexed={(r['x_m'],r['y_m']):r for r in rows}
    expected={(float(x),float(y)) for x in manifest['grid_x_m'] for y in manifest['grid_y_m']}
    assert len(rows)==len(indexed)==961 and set(indexed)==expected
    for path,digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    with np.load(OUT/'workspace.npz') as saved:
        data={key:saved[key].copy() for key in saved.files}
    for key in ('endpoint_pass','motion_pass','interior'):
        assert data[key].shape==(31,31)
        assert int(data[key].sum())==summary[key]['cells']
        assert abs(summary[key]['sampled_area_m2']-summary[key]['cells']*.0016)<1e-12
    assert np.all(data['motion_pass']<=data['endpoint_pass'])
    assert np.all(data['interior']<=data['motion_pass'])
    for row in rows:
        assert np.isfinite([row['position_error_mm'],row['pose_error_deg'],row['robot_floor_mm'],row['finger_floor_mm'],row['peak_velocity_to_limit']]).all()
        assert row['steps']<=800
        assert abs(row['sim_seconds']-.02*row['steps'])<1e-8
        assert row['endpoint_pass']==(row['reason']=='endpoint_held_2s')
        assert row['motion_pass']==(row['endpoint_pass'] and row['max_path_pose_deg']<=5 and row['peak_velocity_to_limit']<=1.01)
        if row['endpoint_pass']:
            assert row['position_error_mm']<=5 and row['pose_error_deg']<=2
            assert .2<=row['finger_floor_mm']<=8 and row['robot_floor_mm']>=.2
            assert row['joint_margin_deg']>=0 and row['self_distance_mm']>=1
        if row['reason']=='floor_penetration':assert row['robot_floor_mm'] < -1
        iy=np.argmin(abs(data['ys']-row['y_m']));ix=np.argmin(abs(data['xs']-row['x_m']))
        for key in ('endpoint_pass','motion_pass'):assert bool(data[key][iy,ix])==row[key]
    pilot=json.loads((OUT/'pilot_verification.json').read_text())
    assert len(pilot)==2 and all(r['previous_ablation_prefix_max_difference']==0 for r in pilot)
    report=dict(unique_cells=961,shared_grid_with_previous_scan=True,finite_metrics=True,
        success_criteria_and_masks_consistent=True,source_hashes_match=True,
        previous_ablation_prefixes_exact=True,
        control_dt_s=.02,physics_dt_s=.002,arm_servo_kp_scale=4,arm_servo_kv_scale=2,angular_gain=10,
        bounded_gravity_feedforward_hz=500,
        torque_saturation_substeps=sum(r['torque_saturation_substeps'] for r in rows),
        velocity_guard_substeps=sum(r['velocity_guard_substeps'] for r in rows),
        selected_goals=[indexed[(x,y)] for x,y in ((.71,0.),(.67,.28),(.47,.16))])
    previous=json.loads((OUT.parent/'version_6/cells.json').read_text())
    assert {(r['x_m'],r['y_m']) for r in previous}==expected
    labels,count=ndimage.label(data['motion_pass'])
    components=[]
    for label in range(1,count+1):
        iy,ix=np.where(labels==label)
        components.append(dict(cells=len(ix),area_m2=len(ix)*.0016,
            x_m=[float(data['xs'][ix].min()),float(data['xs'][ix].max())],
            y_m=[float(data['ys'][iy].min()),float(data['ys'][iy].max())]))
    report['motion_components']=sorted(components,key=lambda s:s['cells'],reverse=True)
    assert [s['cells'] for s in report['motion_components']]==summary['motion_component_sizes']
    iy,ix=np.where(data['motion_pass'])
    assert not data['motion_pass'][iy.min():iy.max()+1,ix.min():ix.max()+1].all()
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
