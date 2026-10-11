"""Paired latest-checkpoint evaluation of a command-only table guard."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

RUN = Path(os.environ['FULL6_PPO_RUN'])
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_targetnode_full6_workspace_eval13 as trainer
from table_execution_guard import TableGeometry, TableExecutionGuard
import mujoco
import numpy as np


def state(base):
    out = np.empty(mujoco.mj_stateSize(base.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(base.model, base.data, out, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return out


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda v:
        v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)))


class Observer:
    def __init__(self, base, enabled):
        self.base = base
        self.geometry = TableGeometry(base)
        self.guard = TableExecutionGuard(base, self.geometry) if enabled else None
        self.points = self.geometry.points()
        self.minimum = min(r[1] for r in self.points)
        self.steps = self.contact_steps = self.events = 0
        self.in_contact = False
        self.peak_force = self.impulse = self.peak_impact_speed = self.peak_contact_speed = 0.
        self.contact_speeds = []
        self.physics = base._physics_step
        self.velocity_guard = base._velocity_guarded_action
        self.dt = float(base.model.opt.timestep)
        self.row_force = self.row_speed = 0.
        base._velocity_guarded_action = self.control
        base._physics_step = self.step

    def control(self, action):
        target, active = self.velocity_guard(action)
        self.points = self.geometry.points()
        self.minimum = min(self.minimum, min(r[1] for r in self.points))
        if self.guard:
            target = self.guard.project(target, self.points)
            active = active or self.guard.last_active
        return target, active

    def step(self):
        result = self.physics()
        b = self.base
        force = 0.
        contact = False
        contact_geoms = set()
        for index, c in enumerate(b.data.contact):
            if self.geometry.table not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.geometry.table else c.geom1
            # Robot meshes are before the first cable body. Exclude cable.
            body = int(b.model.geom_bodyid[other])
            if body == 0 or body >= min(b.cable_ids):
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(b.model, b.data, index, f)
            if c.dist < 0 and f[0] > 1e-6:
                contact = True
                contact_geoms.add(body)
                force += float(f[0])
        # For a flat table, first-impact normal speed is measured at the
        # lowest surface vertex immediately before that actual physics step.
        speed = max((max(0., -r[3]) for r in self.points if int(b.model.geom_bodyid[r[0]]) in contact_geoms), default=0.)
        if contact:
            self.contact_steps += 1
            self.contact_speeds.append(speed)
            if not self.in_contact:
                self.events += 1
                self.peak_impact_speed = max(self.peak_impact_speed, speed)
            self.peak_contact_speed = max(self.peak_contact_speed, speed)
        self.in_contact = contact
        self.peak_force = max(self.peak_force, force)
        self.impulse += force*self.dt
        self.row_force = max(self.row_force, force)
        self.row_speed = max(self.row_speed, speed)
        self.steps += 1
        return result

    def summary(self):
        return dict(physics_steps=self.steps, table_contact_events=self.events,
                    table_contact_steps=self.contact_steps, table_contact_s=self.contact_steps*self.dt,
                    table_contact_episode=bool(self.contact_steps),
                    min_surface_clearance_m=self.minimum,
                    peak_first_impact_down_speed_m_s=self.peak_impact_speed,
                    peak_contact_down_speed_m_s=self.peak_contact_speed,
                    peak_table_normal_force_N=self.peak_force, table_normal_impulse_Ns=self.impulse,
                    guard=self.guard.summary() if self.guard else None)


def evaluate(job):
    checkpoint, episode, variant, output, limit = job
    trainer.torch.set_num_threads(1)
    started = time.monotonic()
    seed = 20260804+episode
    directory = Path(output)/variant/f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    policy = trainer.PPO.load(checkpoint, device='cpu')
    env = trainer.CourseEnv(1000+episode, evaluation=True)
    env.set_stage(0)
    obs, initial = env.reset(seed=seed)
    base = env.env.base_env
    initial_state = state(base)
    initial_sha = hashlib.sha256(initial_state.tobytes()).hexdigest()
    observer = Observer(base, variant == 'protected')
    trajectory, observations, actions, rewards, rows = [initial_state], [obs.copy()], [], [], []
    done = truncated = False
    info = initial
    while not (done or truncated):
        action, _ = policy.predict(obs, deterministic=True)
        observer.row_force = observer.row_speed = 0.
        obs, reward, done, truncated, info = env.step(action)
        trajectory.append(state(base))
        observations.append(obs.copy())
        actions.append(action.copy())
        rewards.append(float(reward))
        rows.append(dict(time=len(actions)*.02, task_success=bool(info.get('task_success', False)),
            secured=bool(info.get('secured_grasp', False)), actual_node=env.env.goal.actual_index(),
            lifted_fraction=float(info.get('lifted_fraction', 0)), success_hold=float(base.success_hold),
            surface_clearance_m=min(p[1] for p in observer.points),
            down_speed_m_s=max(max(0.,-p[3]) for p in observer.points),
            normal_force_N=observer.row_force, contact_down_speed_m_s=observer.row_speed,
            guard_active=bool(observer.guard and observer.guard.last_active)))
        if limit and len(actions) >= limit:
            break
    summary = dict(seed=seed, episode=episode, actual_seed=initial['actual_seed'], variant=variant,
        stage=0, stage_name=trainer.STAGES[0]['name'], goal_node=env.env.goal.index,
        goal_initial_xyz=initial['workspace_initial_goal_xyz'], initial_state_sha256=initial_sha,
        first_correct=info.get('first_grasp_correct'), first_grasp_node=info.get('first_grasp_index'),
        task_success=bool(info.get('task_success', False)), terminated=bool(done), truncated=bool(truncated),
        return_=sum(rewards), max_lifted_fraction=max(r['lifted_fraction'] for r in rows),
        steps=len(actions), frames=len(trajectory), duration_s=len(actions)*.02,
        checkpoint=str(checkpoint), checkpoint_steps=policy.num_timesteps,
        checkpoint_sha256=trainer.sha(checkpoint), execution_source_sha256=trainer.sha(RUN/'full6_workspace_env.py'),
        frozen_reward_sha256=trainer.sha(RUN/'frozen_code/src/panda_cable_grasp/rl/environment.py'),
        frozen_target_sha256=trainer.sha(RUN/'frozen_code/src/panda_cable_grasp/rl/target_node.py'),
        training_config_sha256=trainer.sha(RUN/'training_config.json'),
        target_node_tolerance=int(env.env.rl_config.target_node_tolerance), reward_changes=False,
        guard_source_sha256=trainer.sha(Path(__file__).parent/'table_execution_guard.py'),
        runner_source_sha256=trainer.sha(Path(__file__)), pilot_limit=limit,
        wall_s=time.monotonic()-started, **observer.summary())
    np.savez_compressed(directory/'trajectory.npz', states=np.asarray(trajectory),
        observations=np.asarray(observations), actions=np.asarray(actions), rewards=np.asarray(rewards),
        frame_times=np.arange(len(trajectory))*.02)
    summary['trajectory_sha256'] = trainer.sha(directory/'trajectory.npz')
    save(directory/'metadata.json', summary)
    save(directory/'metrics.json', rows)
    env.close()
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--start-episode', type=int, default=0)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    jobs = [(args.checkpoint, e, v, args.output, args.limit)
            for e in range(args.start_episode,args.start_episode+args.episodes) for v in ('baseline','protected')]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        results = list(pool.map(evaluate, jobs))
    pairs = []
    for e in range(args.episodes):
        a, b = results[2*e:2*e+2]
        for key in ('initial_state_sha256','actual_seed','goal_node','goal_initial_xyz',
                    'checkpoint_sha256','frozen_reward_sha256','frozen_target_sha256','target_node_tolerance'):
            assert a[key] == b[key], (e, key)
        pairs.append(dict(seed=a['seed'], initial_state_identical=True, goal_identical=True))
    save(Path(args.output)/'manifest.json', dict(episodes_per_variant=args.episodes, workers=args.workers,
        stage=0, selection='first consecutive held-out seeds, no outcome-based selection',
        results=results, pairs=pairs, paired_verification_passed=True))
