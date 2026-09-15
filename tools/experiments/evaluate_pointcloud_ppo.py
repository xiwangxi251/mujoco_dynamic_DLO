"""Parallel deterministic evaluation for RGB-D point-cloud PPO checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
from panda_cable_grasp.rl.pointcloud import (
    DLOPointCloudObservation,
    PointCloudObservationConfig,
)


SCENARIOS = (
    "id_static",
    "id_shape_nominal_current",
    "id_rigid_l1_nominal",
    "id_combined_l1_nominal",
)


@dataclass(frozen=True)
class EvalEnvFactory:
    scenario: str
    seed: int
    pointcloud: PointCloudObservationConfig
    table_finger_collision_filter: bool

    def __call__(self):
        env = make_rl_env(
            action_mode="task_space_vertical_down",
            robot="nero",
            seed=self.seed,
            disturbance_strength=1.5,
            episode_seconds=15.0,
            dynamicvla_cameras_enabled=True,
            scenario_names=(self.scenario,),
            rl_config=RLConfig(singularity_avoidance_enabled=False),
            geometric_safety_enabled=False,
            table_finger_collision_filter_enabled=self.table_finger_collision_filter,
        )
        wrapped = DLOPointCloudObservation(env, self.pointcloud)
        wrapped.set_training_scenarios((self.scenario,))
        wrapped.set_motion_difficulty(1.0)
        return wrapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers-per-scenario", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20270915)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-table-finger-collision-filter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cloud_config = PointCloudObservationConfig(
        point_count=384,
        width=480,
        height=360,
        camera_update_steps=5,
        sensor_delay_steps=3,
        voxel_size_m=0.002,
    )
    worker_scenarios = tuple(
        scenario
        for scenario in SCENARIOS
        for _ in range(args.workers_per_scenario)
    )
    factories = [
        EvalEnvFactory(
            scenario,
            args.seed + rank,
            cloud_config,
            not args.disable_table_finger_collision_filter,
        )
        for rank, scenario in enumerate(worker_scenarios)
    ]
    env = SubprocVecEnv(factories, start_method="spawn")
    model = PPO.load(args.model, device=args.device)
    observations = env.reset()
    records: dict[str, list[dict[str, float | int | bool]]] = {
        scenario: [] for scenario in SCENARIOS
    }
    completed = 0
    target = args.episodes * len(SCENARIOS)
    try:
        while completed < target:
            actions, _ = model.predict(observations, deterministic=True)
            observations, _, dones, infos = env.step(actions)
            for rank, done in enumerate(dones):
                if not done:
                    continue
                scenario = worker_scenarios[rank]
                if len(records[scenario]) >= args.episodes:
                    continue
                info = infos[rank]
                record = {
                    "episode": len(records[scenario]) + 1,
                    "success": bool(info.get("success", False)),
                    "strict_success": bool(info.get("strict_success", False)),
                    "pinch": bool(info.get("ever_pinched", False)),
                    "aligned_pinch": bool(info.get("ever_aligned_pinch", False)),
                    "lift_attempt": bool(info.get("lift_attempt", False)),
                    "loaded_lift": bool(info.get("loaded_lift", False)),
                    "grasp": bool(info.get("ever_grasped", False)),
                    "length": int(info.get("episode_steps", 0)),
                    "return": float(info.get("episode_return", np.nan)),
                }
                records[scenario].append(record)
                completed += 1
                if completed % 10 == 0:
                    print(f"progress={completed}/{target}", flush=True)
    finally:
        env.close()

    rows = []
    metric_names = (
        "success",
        "strict_success",
        "pinch",
        "aligned_pinch",
        "lift_attempt",
        "loaded_lift",
        "grasp",
    )
    for scenario in SCENARIOS:
        episodes = records[scenario]
        row: dict[str, str | int | float] = {
            "scenario": scenario,
            "episodes": len(episodes),
        }
        for name in metric_names:
            row[f"{name}_rate"] = float(np.mean([item[name] for item in episodes]))
        row["mean_length"] = float(np.mean([item["length"] for item in episodes]))
        finite_returns = [item["return"] for item in episodes if np.isfinite(item["return"])]
        row["mean_return"] = float(np.mean(finite_returns)) if finite_returns else float("nan")
        rows.append(row)

    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "episodes.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "model": str(args.model.resolve()),
                "episodes_per_scenario": args.episodes,
                "workers_per_scenario": args.workers_per_scenario,
                "seed": args.seed,
                "motion_difficulty": 1.0,
                "table_finger_collision_filter_enabled": (
                    not args.disable_table_finger_collision_filter
                ),
                "pointcloud": asdict(cloud_config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in rows:
        print(
            f"{row['scenario']} success={row['success_rate']:.1%} "
            f"strict={row['strict_success_rate']:.1%} "
            f"pinch={row['pinch_rate']:.1%} grasp={row['grasp_rate']:.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
