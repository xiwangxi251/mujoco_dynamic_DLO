"""Display endpoint versus motion acceptance for the frozen guard experiment."""
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import experiment_bounded_full6 as exp

OUT = exp.OUT / 'table_guard_015'


def main():
    data = {
        'off': json.loads((exp.OUT / 'joint_tolerance_015/bounded_cells.json').read_text()),
        'on': json.loads((OUT / 'protected_cells.json').read_text()),
    }
    xs = sorted({r['x_m'] for r in data['off']})
    ys = sorted({r['y_m'] for r in data['off']})
    details = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
    fig.subplots_adjust(left=.07, right=.98, bottom=.14, top=.84, wspace=.23)
    colors = ['#e8edf2', '#ef8878', '#8abfec', '#239879']
    for ax, variant in zip(axes, ('off', 'on')):
        rows = data[variant]
        assert len(rows) == len({(r['x_m'], r['y_m']) for r in rows}) == 961
        grid = np.zeros((len(ys), len(xs)), int)
        for r in rows:
            value = 3 if r['motion_pass'] else 2 if r['endpoint_pass'] else 1 if r['reason'] == 'floor_penetration' else 0
            grid[ys.index(r['y_m']), xs.index(r['x_m'])] = value
        ax.imshow(grid, origin='lower', extent=(xs[0]-.02, xs[-1]+.02, ys[0]-.02, ys[-1]+.02),
                  cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(-.5,4),4), interpolation='nearest')
        ax.set_title('Bounded enhanced6 / guard ' + variant.upper())
        ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)'); ax.grid(alpha=.15)
        ax.plot(.35, 0, 'o', color='#223b53', markersize=4)
        timeout = [r for r in rows if r['reason'] == 'timeout']
        violations = {
            'position_over_5mm': lambda r: r['position_error_mm'] > 5,
            'pose_over_2deg': lambda r: r['pose_error_deg'] > 2,
            'actual_joint_outside_015deg': lambda r: r['joint_margin_deg'] < -.15,
            'finger_outside_02_to_8mm': lambda r: not .2 <= r['finger_floor_mm'] <= 8,
            'robot_floor_below_02mm': lambda r: r['robot_floor_mm'] < .2,
            'self_distance_below_1mm': lambda r: r['self_distance_mm'] < 1,
        }
        details[variant] = dict(reasons=dict(Counter(r['reason'] for r in rows)),
            motion_pass=sum(r['motion_pass'] for r in rows),
            endpoint_only=sum(r['endpoint_pass'] and not r['motion_pass'] for r in rows),
            timeout_final_violations={k: sum(fn(r) for r in timeout) for k, fn in violations.items()},
            timeout_final_all_criteria_satisfied=sum(not any(fn(r) for fn in violations.values()) for r in timeout),
            timeout_violation_counts_overlap=True)
        d=details[variant]
        ax.text(.02,.02,f"Motion passed: {d['motion_pass']}/961\nEndpoint only: {d['endpoint_only']}/961\nFloor stop: {d['reasons'].get('floor_penetration',0)} | Timeout: {len(timeout)}",
            transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.95,edgecolor='none'))
    fig.suptitle('Table guard: endpoint and full-motion acceptance\nSame 16 s limit / continuous 2 s hold / actual joint tolerance 0.15 deg')
    fig.legend([Patch(color=c) for c in colors], ['Timeout', 'Floor penetration stop', 'Endpoint only', 'Motion passed'],
        loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,frameon=False)
    fig.savefig(OUT/'workspace_status.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    details['motion_criteria'] = 'Endpoint held 2 s, maximum path pose error <=5 deg, peak actual joint speed / original joint limit <=1.01.'
    details['timeout_interpretation'] = 'No continuous 2 s acceptance within 16 s. Final violation counts overlap and are diagnostics, not independent root-cause counts.'
    (OUT/'status_diagnosis.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(json.dumps(details,indent=2))


if __name__ == '__main__': main()
