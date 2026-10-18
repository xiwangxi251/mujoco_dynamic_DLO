"""Restore the scan's selected regular sectors; preserve raw point identities."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from panda_cable_grasp.rl.action_workspace import ActionWorkspace


def build(summary_path, output):
    summary_path, output = Path(summary_path), Path(output)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    candidate_path = summary_path.with_name('workspace_candidate.json')
    candidate = json.loads(candidate_path.read_text(encoding='utf-8'))
    rows_path = summary_path.with_name('cells.json')
    rows = json.loads(rows_path.read_text(encoding='utf-8'))
    layers = [dict(height_m=row['height_m'], **{
        key: row['sampled_sector'][key] for key in
        ('inner_radius_m', 'outer_radius_m', 'angle_min_deg', 'angle_max_deg')})
        for row in summary['layers'] if row['sampled_sector'] is not None]
    layers.sort(key=lambda row: row['height_m'])
    bands = []
    for a, b in zip(layers, layers[1:]):
        band = [max(a['inner_radius_m'], b['inner_radius_m']),
                min(a['outer_radius_m'], b['outer_radius_m']),
                max(a['angle_min_deg'], b['angle_min_deg']),
                min(a['angle_max_deg'], b['angle_max_deg']), a['height_m'], b['height_m']]
        if band[0] >= band[1] or band[2] >= band[3]:
            raise ValueError('Adjacent layer sectors have no common transfer region')
        bands.append(band)
    metadata = dict(version='measured_layer_sectors_v1', center_xy_m=candidate['center_xy_m'],
        height_reference='TCP relative to table top', source_commit=summary['source_commit'],
        source_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        source_candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        source_cells_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        source_tests=len(rows), source_passes=sum(row['endpoint_pass'] for row in rows),
        selection='original per-height sampled_sector; adjacent sectors intersect between measured heights',
        interpolation_limitations='finite fixed-down closed-jaw unloaded samples; interiors and other poses not certified',
        layer_sectors=layers, cells_r_theta_z=bands,
        excluded_heights_m=[row['height_m'] for row in summary['layers'] if row['sampled_sector'] is None],
        unselected_refinement_heights_m=sorted(set(row['height_m'] for row in rows) -
            set(row['height_m'] for row in summary['layers'])))
    encoded = json.dumps(metadata, ensure_ascii=False, indent=2) + '\n'
    temp = output.with_suffix('.candidate.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp.write_bytes(encoded.encode('utf-8'))
    workspace = ActionWorkspace(temp)
    accepted = [row for row in rows if workspace.contains([row['x_m'], row['y_m'], row['height_m']])]
    if any(not row['endpoint_pass'] for row in accepted):
        raise ValueError('Restored sectors include a measured failure; candidate retained for diagnosis')
    temp.replace(output)
    print(json.dumps(dict(output=str(output), selected_layers=len(layers), bands=len(bands),
        measured_accepted=len(accepted), measured_failures_accepted=0,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.summary, args.output)
