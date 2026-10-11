"""Audit relaxed-joint endpoint acceptance without changing physical limits."""
import hashlib
import inspect
import json
import redraw_joint_tolerance_workspace as run

old=run.original()
new={v:json.loads((run.OUT/f'{v}_cells.json').read_text()) for v in old}
report={}
for variant,rows in new.items():
    reference={(r['x_m'],r['y_m']):r for r in old[variant]}
    assert len(rows)==len({(r['x_m'],r['y_m']) for r in rows})==961
    assert {(r['x_m'],r['y_m']) for r in rows}==set(reference)
    reused=rerun=unchanged_failure=0;added=[]
    for row in rows:
        before=reference[(row['x_m'],row['y_m'])]
        assert row['joint_tolerance_deg']==.15
        assert row['steps']<=800 and abs(row['sim_seconds']-.02*row['steps'])<1e-8
        assert row['endpoint_pass']==(row['reason']=='endpoint_held_2s')
        assert row['motion_pass']==(row['endpoint_pass'] and row['max_path_pose_deg']<=5 and row['peak_velocity_to_limit']<=1.01)
        if row['endpoint_pass']:assert run.final_ok(row)
        if before['endpoint_pass']:
            reused+=1;assert row['endpoint_pass']
            assert row['provenance'].startswith('reused strict')
            for key,value in before.items():assert row[key]==value,key
        else:
            rerun+=1;assert row['provenance'].startswith('fresh physical')
            if row['endpoint_pass']:added.append(dict(goal=[row['x_m'],row['y_m'],row['z_m']],held_by_s=row['sim_seconds'],raw_joint_margin_deg=row['joint_margin_deg']))
            else:
                assert row['reason']==before['reason'] and row['steps']==before['steps']
                for key in ('position_error_mm','pose_error_deg','joint_margin_deg','robot_floor_mm','finger_floor_mm','self_distance_mm','max_path_pose_deg','peak_velocity_to_limit'):
                    assert abs(row[key]-before[key])<1e-8,(variant,row['x_m'],row['y_m'],key)
                unchanged_failure+=1
        if variant=='bounded':assert row['joint_target_clip_steps']==0 and row['solver_failures']==0
    report[variant]=dict(unique_cells=961,reused_strict_passes=reused,rerun_failures=rerun,
        remaining_failures_match_original_physics=unchanged_failure,added_passes=added,
        endpoint_pass=sum(r['endpoint_pass'] for r in rows),motion_pass=sum(r['motion_pass'] for r in rows))
manifest=json.loads((run.OUT/'manifest.json').read_text())
manifest['criterion_worker_sha256']=hashlib.sha256(inspect.getsource(run.worker).encode()).hexdigest()
manifest['execution_note']='Resumed saved complete rows after memory exhaustion; grouped variants and collected unused rigs to reduce memory. Controller/physics and 0.15-degree criterion unchanged.'
(run.OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
(run.OUT/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
