"""Paired ablation for the scripted policy's closed-gripper command.

The two conditions share the exact scenario and episode seed.  Besides strict
task outcomes, the runner measures aperture, pad normal force, and contact
penetration while a grasp is physically confirmed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from cable_grasp_env import CableGraspEnv
from dynamic_grasp_policy import DynamicCableGraspPolicy
from experiment_scenarios import get_scenario, list_scenario_names
from failure_taxonomy import classify_task_outcome, scene_fingerprint
from motion_diagnostics import env_config_for_scenario
from project_paths import output_path


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stat(values: list[float], operation: str) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    if operation == "median":
        return float(np.median(array))
    if operation == "min":
        return float(np.min(array))
    if operation == "p95":
        return float(np.quantile(array, 0.95))
    raise ValueError(operation)


def _pad_penetration_mm(env: CableGraspEnv) -> float | None:
    depths: list[float] = []
    for contact in env.data.contact[:env.data.ncon]:
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(env.model.geom_bodyid[geom1])
        body2 = int(env.model.geom_bodyid[geom2])
        if (
            (geom1 in env.pad_geom_ids and body2 in env.cable_set)
            or (geom2 in env.pad_geom_ids and body1 in env.cable_set)
        ):
            depths.append(max(0.0, -1000.0 * float(contact.dist)))
    return max(depths) if depths else None


def _run_episode(job: tuple[str, int, float, float, int]) -> dict[str, Any]:
    scenario_name, seed, control, episode_seconds, sample_every = job
    scenario = get_scenario(scenario_name)
    env = CableGraspEnv(env_config_for_scenario(
        scenario, seed=seed, episode_seconds=episode_seconds,
    ))
    DynamicCableGraspPolicy.HOLD_GRIPPER_CTRL = float(control)
    policy = DynamicCableGraspPolicy(env)
    _, initial_info = env.reset(seed=seed)
    policy.reset()

    apertures: list[float] = []
    min_side_forces: list[float] = []
    max_side_forces: list[float] = []
    penetrations: list[float] = []
    step_index = 0
    while not policy.finished and env.data.time < env.config.episode_seconds:
        action = policy.action()
        _, _, _, truncated, _ = env.step(action)
        step_index += 1
        if env.grasp_confirmed and step_index % sample_every == 0:
            aperture = 1000.0 * float(np.sum(
                env.data.qpos[env.finger_qpos_adr]
            ))
            forces = env.finger_normal_forces()
            left = float(forces[env.left_finger_id])
            right = float(forces[env.right_finger_id])
            apertures.append(aperture)
            min_side_forces.append(min(left, right))
            max_side_forces.append(max(left, right))
            penetration = _pad_penetration_mm(env)
            if penetration is not None:
                penetrations.append(penetration)
        if truncated:
            policy.result = "failed_timeout"
            policy.finished = True

    final_info = env.info()
    task_success = bool(env.ever_success)
    outcome = classify_task_outcome(
        task_success=task_success,
        ever_bilateral_candidate=bool(final_info["ever_bilateral_candidate"]),
        ever_confirmed_grasp=bool(final_info["ever_confirmed_grasp"]),
        break_events=env.grasp_break_history,
    )
    return {
        "scenario": scenario_name,
        "seed": seed,
        "hold_gripper_ctrl": float(control),
        "scene_fingerprint": scene_fingerprint(initial_info),
        "task_success": task_success,
        "policy_result": policy.result,
        "outcome": outcome,
        "ever_bilateral_candidate": bool(
            final_info["ever_bilateral_candidate"]
        ),
        "ever_confirmed_grasp": bool(final_info["ever_confirmed_grasp"]),
        "physical_slip_after_confirmed_count": int(
            final_info["physical_slip_after_confirmed_count"]
        ),
        "confirmed_metric_samples": len(apertures),
        "aperture_median_mm": _stat(apertures, "median"),
        "aperture_min_mm": _stat(apertures, "min"),
        "min_side_force_median_n": _stat(min_side_forces, "median"),
        "min_side_force_p95_n": _stat(min_side_forces, "p95"),
        "max_side_force_median_n": _stat(max_side_forces, "median"),
        "max_side_force_p95_n": _stat(max_side_forces, "p95"),
        "penetration_median_mm": _stat(penetrations, "median"),
        "penetration_p95_mm": _stat(penetrations, "p95"),
        "sim_time": float(env.data.time),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _exact_mcnemar(control_20_wins: int, control_0_wins: int) -> float | None:
    discordant = control_20_wins + control_0_wins
    if discordant == 0:
        return None
    lower = min(control_20_wins, control_0_wins)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1))
    return min(1.0, 2.0 * tail / (2.0 ** discordant))


def _paired_mean_ci(
    differences: list[float], *, seed: int, samples: int = 20_000,
) -> dict[str, Any]:
    if not differences:
        return {"pairs": 0, "mean_difference": None, "bootstrap95": None}
    values = np.asarray(differences, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    bootstrap = values[indices].mean(axis=1)
    return {
        "pairs": len(values),
        "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "bootstrap95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def _summarize(rows: list[dict[str, Any]], bootstrap_seed: int) -> dict[str, Any]:
    by_control: dict[str, Any] = {}
    for control in (0.0, 20.0):
        selected = [row for row in rows if row["hold_gripper_ctrl"] == control]
        confirmed = [row for row in selected if row["ever_confirmed_grasp"]]
        by_control[str(int(control))] = {
            "episodes": len(selected),
            "task_successes": sum(row["task_success"] for row in selected),
            "task_success_rate": float(np.mean([
                row["task_success"] for row in selected
            ])),
            "confirmed_grasps": len(confirmed),
            "confirmed_grasp_rate": float(np.mean([
                row["ever_confirmed_grasp"] for row in selected
            ])),
            "episodes_with_confirmed_physical_slip": sum(
                row["physical_slip_after_confirmed_count"] > 0
                for row in selected
            ),
            "outcomes": dict(Counter(row["outcome"] for row in selected)),
        }

    by_seed: dict[int, dict[float, dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[
            float(row["hold_gripper_ctrl"])
        ] = row
    pairs: list[dict[str, Any]] = []
    metric_names = (
        "aperture_median_mm",
        "max_side_force_p95_n",
        "penetration_p95_mm",
    )
    differences: dict[str, list[float]] = {name: [] for name in metric_names}
    wins_20 = wins_0 = 0
    for seed, pair in sorted(by_seed.items()):
        if set(pair) != {0.0, 20.0}:
            raise RuntimeError(f"incomplete pair for seed {seed}: {sorted(pair)}")
        zero, twenty = pair[0.0], pair[20.0]
        if zero["scene_fingerprint"] != twenty["scene_fingerprint"]:
            raise RuntimeError(f"scene fingerprint mismatch for seed {seed}")
        if twenty["task_success"] and not zero["task_success"]:
            wins_20 += 1
        if zero["task_success"] and not twenty["task_success"]:
            wins_0 += 1
        pair_row: dict[str, Any] = {
            "seed": seed,
            "ctrl_0_success": zero["task_success"],
            "ctrl_20_success": twenty["task_success"],
            "ctrl_0_outcome": zero["outcome"],
            "ctrl_20_outcome": twenty["outcome"],
        }
        for name in metric_names:
            left, right = zero[name], twenty[name]
            difference = None if left is None or right is None else right - left
            pair_row[f"{name}_difference_20_minus_0"] = difference
            if difference is not None:
                differences[name].append(float(difference))
        pairs.append(pair_row)

    return {
        "by_control": by_control,
        "paired_success": {
            "ctrl_20_wins": wins_20,
            "ctrl_0_wins": wins_0,
            "ties": len(pairs) - wins_20 - wins_0,
            "risk_difference_20_minus_0": (
                by_control["20"]["task_success_rate"]
                - by_control["0"]["task_success_rate"]
            ),
            "exact_mcnemar_p": _exact_mcnemar(wins_20, wins_0),
        },
        "paired_metrics_20_minus_0": {
            name: _paired_mean_ci(values, seed=bootstrap_seed + index)
            for index, (name, values) in enumerate(differences.items())
        },
        "pairs": pairs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=list_scenario_names(),
        default="id_shape_nominal_current",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20280901)
    parser.add_argument("--episode-seconds", type=float, default=28.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=output_path("gripper_ctrl_ablation")
    )
    args = parser.parse_args()
    if args.seeds < 1 or args.workers < 1 or args.sample_every < 1:
        parser.error("--seeds, --workers, and --sample-every must be positive")
    if args.episode_seconds <= 0.0:
        parser.error("--episode-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    seeds = [args.seed + index for index in range(args.seeds)]
    jobs = [
        (args.scenario, seed, control, args.episode_seconds, args.sample_every)
        for seed in seeds
        for control in (0.0, 20.0)
    ]
    output = args.output / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_episode, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda item: (item["seed"], item["hold_gripper_ctrl"]))
            _write_csv(output / "episodes.csv", rows)
            print(
                f"completed={len(rows)}/{len(jobs)} seed={row['seed']} "
                f"ctrl={row['hold_gripper_ctrl']:g} success={row['task_success']} "
                f"outcome={row['outcome']}",
                flush=True,
            )

    summary = _summarize(rows, bootstrap_seed=args.seed + 100_000)
    _write_csv(output / "pairs.csv", summary.pop("pairs"))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "arguments": vars(args) | {"output": str(args.output.resolve())},
        "scenario": get_scenario(args.scenario).asdict(),
        "seeds": seeds,
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in {
                "ablation": Path(__file__),
                "environment": ROOT / "cable_grasp_env.py",
                "policy": ROOT / "dynamic_grasp_policy.py",
                "scenario_registry": ROOT / "experiment_scenarios.py",
            }.items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"output={output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
