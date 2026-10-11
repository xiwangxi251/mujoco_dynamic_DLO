"""Audit visible mesh clearance separately from enabled table contacts."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from record_retest_136_videos import create_rig, ROOT

OUT = ROOT / 'outputs/vertical_workspace/scripted_136_retest_20261004/floor_diagnosis'

def floor_details(rig, env):
    model, data = rig.model, rig.data
    parts = []
    for g in rig.geometry.visual_geoms:
        body = model.body(int(model.geom_bodyid[g])).name
        if body == 'base_link':
            continue
        vertices = rig.geometry.geom_vertices(g)
        point = vertices[np.argmin(vertices[:, 2])]
        parts.append(dict(geom_id=int(g), body=body,
            mesh=model.mesh(int(model.geom_dataid[g])).name,
            floor_mm=float(point[2] * 1000), lowest_point_m=point.tolist()))
    parts.sort(key=lambda p: p['floor_mm'])
    table = env.table_geom_id
    collisions = []
    for g in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[g])).name
        if body not in {'link'+str(i) for i in range(1, 8)} | {'gripper_link1', 'gripper_link2'}:
            continue
        allowed = bool((model.geom_contype[g] & model.geom_conaffinity[table]) or
                       (model.geom_contype[table] & model.geom_conaffinity[g]))
        distance = float(mujoco.mj_geomDistance(model, data, g, table, 10, np.zeros(6))) if allowed else None
        collisions.append(dict(geom_id=g, body=body, group=int(model.geom_group[g]),
            mesh=model.mesh(int(model.geom_dataid[g])).name if model.geom_dataid[g] >= 0 else None,
            contype=int(model.geom_contype[g]), conaffinity=int(model.geom_conaffinity[g]),
            table_collision_enabled=allowed, table_distance_mm=None if distance is None else distance*1000))
    contacts = []
    for c in data.contact[:data.ncon]:
        if table in c.geom:
            g = int(c.geom[1] if c.geom[0] == table else c.geom[0])
            contacts.append(dict(geom_id=g, body=model.body(int(model.geom_bodyid[g])).name, distance_mm=float(c.dist*1000)))
    return dict(visual_mesh_parts=parts, collision_geoms=collisions, table_contacts=contacts)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version', type=int, required=True)
    args=parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    v=args.version
    rig, env, _=create_rig(v, OUT/f'version_{v}')
    rows=json.loads((OUT.parent/f'version_{v}/cells.json').read_text())
    failures=[r for r in rows if r['reason']=='floor_penetration']
    stats=dict(floor_failures=len(failures), finger_mesh_below_zero=sum(r['finger_floor_mm']<0 for r in failures),
               finger_mesh_below_minus_1mm=sum(r['finger_floor_mm'] < -1 for r in failures))
    samples=[]
    for x,y,name in ((.55,0.,'body_floor'),(.47,.16,'finger_floor'),(.51,-.04,'finger_floor_v1')):
        result=rig.trial(x,y)
        row=result if v==6 else result[0]
        archived=next(r for r in rows if abs(r['x_m']-x)<1e-8 and abs(r['y_m']-y)<1e-8)
        assert row['steps']==archived['steps'] and row['reason']==archived['reason']
        for key in ('robot_floor_mm','finger_floor_mm','pose_error_deg','position_error_mm'):
            assert abs(row[key]-archived[key])<1e-8,(key,row[key],archived[key])
        detail=floor_details(rig,env)
        samples.append(dict(name=name,result=row,details=detail,scan_result_matches=True))
        print(json.dumps(dict(version=v,goal=name,reason=row['reason'],time=row['sim_seconds'],
            lowest=detail['visual_mesh_parts'][:4],contacts=detail['table_contacts'])),flush=True)
    (OUT/f'version_{v}.json').write_text(json.dumps(dict(version=v,stats=stats,samples=samples),indent=2),encoding='utf-8')

if __name__=='__main__':main()
