"""Compare affected reward terms on four saved trajectories without integration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
from panda_cable_grasp.rl.lift_credit import HighWaterLiftCredit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rejudged = json.loads((args.source/'gripper_rejudgement_data.json').read_text())
    env = make_rl_env(robot='nero', action_mode='task_space_vertical_down',
                      scenario_names=['id_static'], rl_config=RLConfig(target_node_task=True))
    top = env.data.geom_xpos[env.base_env.table_geom_id, 2]+env.model.geom_size[env.base_env.table_geom_id, 2]
    records = []
    try:
        for ep in rejudged['episodes']:
            seed, goal = ep['seed'], ep['goal_node']
            source = args.source/f'seed_{seed}'/'trajectory.npz'
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with np.load(source, allow_pickle=False) as archive:
                states, times = archive['states'], archive['frame_times']
                recorded_rewards = archive['rewards']
            assert mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS) == states.shape[1]
            assert len(ep['new_confirmed']) == len(states)
            np.testing.assert_allclose(times, ep['time_s'], atol=1e-10)
            previous_below = None
            previous_fraction = 0.
            rows = []
            credit = None
            first_paid = False
            for i, state in enumerate(states):
                mujoco.mj_setState(env.model, env.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                mujoco.mj_forward(env.model, env.data)
                heights = np.clip(env.data.xpos[env.base_env.cable_ids, 2]-top, 0., .18)
                fraction = float(np.clip(ep['lifted_fraction'][i], 0., .3))
                if i == 0:
                    credit = HighWaterLiftCredit(heights, fraction)
                    previous_fraction = fraction
                    continue
                held = ep['new_node'][i] if ep['new_confirmed'][i] else None
                error = abs(held-goal) if held is not None else None
                # Both sides use the SAME accepted physical_holding_v2 sampled
                # judgement. Only reward tolerance and progress bookkeeping differ.
                old_match = error is not None and error <= 4
                below = .18-float(heights[held]) if held is not None else None
                old_lift = 40*(previous_below-below) if old_match and previous_below is not None else 0.
                previous_below = below if old_match else None
                old_cable = 3*(fraction-previous_fraction) if old_match else 0.
                previous_fraction = fraction
                scale = 1. if error is not None and error <= 2 else {3:.5, 4:.25}.get(error, 0.)
                lift, cable = credit.update(heights, fraction, held, scale)
                old_position = new_position = 0.
                if held is not None and not first_paid:
                    old_position = max(0., 2.-.25*error)
                    new_position = 2.-.25*error if error <= 4 else 0.
                    first_paid = True
                rows.append(dict(time_s=float(times[i]), node=held, node_error=error,
                                 old_lift=old_lift, new_lift=40*lift,
                                 old_cable=old_cable, new_cable=3*cable,
                                 old_position=old_position, new_position=new_position,
                                 original_recorded_total=float(recorded_rewards[i-1])))
            csv_path = args.output/f'seed_{seed}.csv'
            with csv_path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader(); writer.writerows(rows)
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
            t = [row['time_s'] for row in rows]
            for ax, term, title in zip(axes[:2], ('lift', 'cable'), ('Node lift credit', 'Cable lifted-fraction credit')):
                for version, color in (('old', '#d87c35'), ('new', '#168f75')):
                    values = [row[f'{version}_{term}'] for row in rows]
                    ax.plot(t, np.cumsum(values), label=f'{version}: cumulative reward', color=color)
                ax.set_title(title); ax.set_ylabel('Reward'); ax.legend(); ax.grid(alpha=.2)
            axes[2].plot(t, np.cumsum([row['original_recorded_total'] for row in rows]), color='#496eb6',
                         label='Original recorded TOTAL (reference only)')
            axes[2].set_title('Original total reward; new total is not reconstructed from partial terms')
            axes[2].legend(); axes[2].grid(alpha=.2); axes[2].set_xlabel('Original simulation time (s)')
            fig.suptitle(f'Saved trajectory seed={seed}, goal={goal} | no physics integration')
            png = args.output/f'seed_{seed}.png'; fig.savefig(png, dpi=150); plt.close(fig)
            record = dict(seed=seed, goal=goal, source=str(source.resolve()), sha256=digest,
                          rows=len(rows), csv=csv_path.name, png=png.name)
            for term in ('lift', 'cable', 'position'):
                for version in ('old', 'new'):
                    values = np.array([row[f'{version}_{term}'] for row in rows])
                    record[f'{version}_{term}_net'] = float(values.sum())
                    record[f'{version}_{term}_positive'] = float(np.maximum(values, 0.).sum())
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
            records.append(record)
    finally:
        env.close()
    summary = dict(integration_steps=0, episodes=records,
                   compared_terms=['node lift', 'cable lifted fraction', 'first position'],
                   judgement='same previously accepted physical_holding_v2 50 Hz reconstruction on both sides',
                   limitation='Saved 20 ms samples; no 500 Hz contact truth. New full returns/success rates are not inferred.')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    html = '<html><meta charset="utf-8"><title>Reward v2 comparison</title><body style="background:#edf1f4;font-family:sans-serif"><h1>四段原轨迹：奖励修正前后</h1><p>相同50 Hz采样夹持判定，仅对照两项抬升信用与首次位置奖励；未运行策略或积分物理。底部蓝线为原记录总奖励，不是新总奖励。</p>'
    for row in records:
        html += f'<h2>seed {row["seed"]}，目标节点 {row["goal"]}</h2><img style="width:min(1200px,100%)" src="{row["png"]}"><p>抬升正奖励 {row["old_lift_positive"]:.3f} → {row["new_lift_positive"]:.3f}；比例正奖励 {row["old_cable_positive"]:.3f} → {row["new_cable_positive"]:.3f}</p>'
    (args.output/'index.html').write_text(html+'</body></html>', encoding='utf-8')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
