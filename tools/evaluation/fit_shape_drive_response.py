"""Report measured coverage and propose joint inputs, without claiming validation."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import Delaunay

def solve(outputs,drivers,target):
    outputs=np.asarray(outputs);drivers=np.asarray(drivers)
    origin=outputs.min(axis=0);scale=np.ptp(outputs,axis=0)
    mesh=Delaunay((outputs-origin)/scale)
    point=(np.asarray(target)-origin)/scale
    index=int(mesh.find_simplex(point,tol=1e-9))
    if index<0:return dict(covered=False)
    t=mesh.transform[index];first=t[:-1]@(point-t[-1]);weights=np.r_[first,1-first.sum()]
    return dict(covered=True,proposed_driver_values=(weights@drivers[mesh.simplices[index]]).tolist(),
                note='Interpolation proposal; physical verification required, not acceptance')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[]
    for folder in a.input:
        for path in folder.glob('*seed*.json'):
            row=json.loads(path.read_text())
            if row.get('scene')=='dev_shape_nominal_current_x075_v1':rows.append(row)
    report=dict(status='diagnostic_only_not_physics_validated',rows=rows,seeds={},mean_map={},
                target_A_mm=[48.2760507645,72.4140761468],target_V_m_s=[.1638248665,.2106319712])
    targets=list(itertools.product(report['target_A_mm'],report['target_V_m_s']))
    for seed in sorted({r['seed'] for r in rows}):
        subset=[r for r in rows if r['seed']==seed]
        outputs=[[r['measured']['shape_amplitude_rms_mm'],r['measured']['shape_speed_m_s']] for r in subset]
        drivers=[[r['strength'],r['frequency']] for r in subset]
        report['seeds'][str(seed)]=[dict(target=list(t),**solve(outputs,drivers,t)) for t in targets]
    groups={}
    for row in rows:groups.setdefault((row['strength'],row['frequency']),[]).append(row)
    means=[];inputs=[]
    for (s,f),group in sorted(groups.items()):
        if len(group)!=2:continue
        inputs.append([s,f]);means.append(np.mean([[r['measured']['shape_amplitude_rms_mm'],
            r['measured']['shape_speed_m_s']] for r in group],axis=0).tolist())
    report['mean_map']=dict(outputs=means,drivers=inputs,
        corners=[dict(target=list(t),**solve(means,inputs,t)) for t in targets])
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(seeds=report['seeds'],mean_corners=report['mean_map']['corners']),indent=2))

if __name__=='__main__':main()
