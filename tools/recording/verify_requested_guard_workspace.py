"""Audit threshold holds and unchanged dynamics in requested-box trajectories."""
import json
import numpy as np
import probe_requested_guard_workspace as run

rows=json.loads((run.OUT/'cells.json').read_text())
manifest=json.loads((run.OUT/'manifest.json').read_text())
run.guard.assert_sources(manifest['source_sha256'])
accepted=json.loads((run.OUT/'acceptance_cells.json').read_text())
tiers={name:(pos,pose,path) for name,pos,pose,path in run.TIERS}
bykey={(r['x_m'],r['y_m']):r for r in rows}
assert len(bykey)==254
for name,results in accepted.items():
    pos,pose,path=tiers[name]
    for r in results:
        h=np.load(run.OUT/bykey[(r['x_m'],r['y_m'])]['trace'])['history']
        assert np.max(abs(h[:,0]-.02*np.arange(1,len(h)+1)))<1e-8
        assert np.all(np.diff(h[:,7])>=-1e-12)
        actual=run.acceptance(h,pos,pose,path)
        assert all(r[k]==v for k,v in actual.items())
        if r['endpoint']:
            i=round(r['endpoint_time_s']/.02)-1
            block=h[i-99:i+1]
            assert len(block)==100
            assert np.all(block[:,1]<=pos) and np.all(block[:,2]<=pose)
            assert np.all((block[:,3]>=.2)&(block[:,3]<=8))
            assert np.all(block[:,4]>=.2) and np.all(block[:,5]>=-.15) and np.all(block[:,6]>=1)
            if r['motion']:assert h[:i+1,2].max()<=path and h[i,7]<=1.01

# Physical reference uses the frozen original stop condition and measurements.
rig=run.guard.GuardRig(True)
key=(.7,0.)
saved=np.load(run.OUT/bykey[key]['trace'])
record=[];control=rig.control
def collect(target,phase):
    control(target,phase);record.append(rig.data.qpos.copy())
rig.control=collect
row=rig.trial(*key)
delta=float(np.max(abs(np.asarray(record)-saved['qpos'][:len(record)])))
assert delta==0
strict=next(r for r in accepted['strict'] if (r['x_m'],r['y_m'])==key)
assert strict['endpoint']==row['endpoint_pass'] and strict['motion']==row['motion_pass']
assert abs(strict['endpoint_time_s']-row['sim_seconds'])<1e-8
report=dict(trajectory_points=254,area_grid_cells=250,extra_corners=4,
    original_controller_prefix_max_qpos_difference=delta,reference_goal=list(key),
    strict_stop_time_matches_original=True,all_continuous_100_frame_holds_verified=True,
    path_and_velocity_checked_up_to_actual_acceptance_time=True,sources_unchanged=True,
    collision_check='same collision predicate, evaluated every frame; reported self_distance is the existing binary proxy, not a new distance estimate')
(run.OUT/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
