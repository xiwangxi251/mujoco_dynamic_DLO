"""Package the user-selected measured sector without treating its bounding box as valid."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from panda_cable_grasp.env.workspace_geometry import AnnularSector

def build(summary_path,output):
    summary_path=Path(summary_path).resolve(); output=Path(output).resolve()
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    selected=summary['sampled_sector']
    sector=AnnularSector(tuple(summary['center_xy_m']),selected['inner_radius_m'],
        selected['outer_radius_m'],selected['angle_min_deg'],selected['angle_max_deg'])
    samples=json.loads((summary_path.parent/'polar_cells.json').read_text(encoding='utf-8'))
    selected_samples=[r for r in samples if sector.contains_xy([r['x_m'],r['y_m']])]
    if len(selected_samples)!=selected['sampled_points'] or not all(r['endpoint_pass'] for r in selected_samples):
        raise ValueError('Selected sector does not match its passing sample record')
    boundary=sector.boundary()
    bounds=np.array([boundary[:,0].min(),boundary[:,0].max(),boundary[:,1].min(),boundary[:,1].max()])
    step=.04
    xs=np.arange(np.floor(bounds[0]/step)*step, np.ceil(bounds[1]/step)*step+step/2,step)
    ys=np.arange(np.floor(bounds[2]/step)*step, np.ceil(bounds[3]/step)*step+step/2,step)
    x,y=np.meshgrid(xs,ys); mask=sector.contains_xy(np.stack((x,y),axis=-1))
    provenance=dict(source_summary=summary_path.relative_to(ROOT).as_posix(),
        source_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        source_cells_sha256=hashlib.sha256((summary_path.parent/'polar_cells.json').read_bytes()).hexdigest(),
        source_commit=summary['source_commit'],criteria=summary['criteria'],
        tested_points=summary['tested_points'],endpoint_passes=summary['endpoint_passes'],
        selected_passing_samples=len(selected_samples),area_m2=selected['area_m2'],
        interpretation=summary['interpretation'],membership='analytic annular sector; grid is auxiliary',
        z_range_m=[-.005,.10],z_range_note='Inherited diagnostic height window; scan verifies near-table poses, not the full height interval.')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,xs=xs,ys=ys,interior=mask,valid=mask,
        annular_sector_xy=np.array([*sector.center,sector.inner_radius,sector.outer_radius,sector.angle_min_deg,sector.angle_max_deg]),
        provenance_json=np.array(json.dumps(provenance,ensure_ascii=False)))
    print(json.dumps(dict(output=str(output),**sector.manifest(),accepted_grid_centers=int(mask.sum()),selected_passing_samples=len(selected_samples)),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--summary',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); build(args.summary,args.output)
