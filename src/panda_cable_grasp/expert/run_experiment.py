"""Run small no-video experiments for the privileged formula expert."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.benchmark import base_row, summarize
from ..evaluation.defaults import DEFAULT_EVALUATION_SEED
from ..env.environment import CableGraspEnv
from ..scenarios.registry import get_scenario, list_scenario_names
from ..evaluation.motion_diagnostics import env_config_for_scenario
from ..paths import output_path

from .formula_intercept_policy import FormulaInterceptConfig, FormulaInterceptExpert


DEFAULT_SCENARIOS = (
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
)


def run_episode(
    scenario_name: str,
    episode: int,
    seed: int,
    episode_seconds: float,
    expert_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scenario = get_scenario(scenario_name)
    env = CableGraspEnv(env_config_for_scenario(
        scenario,
        seed=seed,
        episode_seconds=episode_seconds,
    ))
    try:
        _, initial_info = env.reset(seed=seed)
        policy = FormulaInterceptExpert(
            env,
            FormulaInterceptConfig(**(expert_config or {})),
        )
        min_target_distance = math.inf
        termination_reason: str | None = None
        steps = 0
        previous_phase = policy.phase
        phase_transitions: list[str] = []
        while not policy.finished and env.data.time < env.config.episode_seconds:
            action = policy.action()
            _, _, _, truncated, step_info = env.step(action)
            steps += 1
            if policy.phase is not previous_phase:
                phase_transitions.append(
                    f"{env.data.time:.3f}:{previous_phase.name}->{policy.phase.name}"
                )
                previous_phase = policy.phase
            min_target_distance = min(
                min_target_distance,
                float(np.linalg.norm(
                    env.hand_position - (
                        policy._locked_segment_position(0.0)
                        if policy.locked_segment_index is not None
                        else policy._selected_segment(0.0)
                    )
                )),
            )
            if truncated:
                termination_reason = step_info.get("termination_reason")
                policy.result = (
                    "failed_motion_boundary"
                    if termination_reason == "rigid_motion_boundary_crossed"
                    else "failed_timeout"
                )
                policy.finished = True
        info = env.info()
        info["ever_pinched"] = env.last_grasped_body_id is not None
        info["base_success"] = env.ever_success
        info["success"] = policy.result == "success"
        row = base_row(
            "privileged_formula_expert", episode, seed, scenario,
            initial_info, info, env.grasp_break_history,
        )
        row.update({
            "steps": steps,
            "sim_time": float(env.data.time),
            "episode_return": np.nan,
            "min_target_distance": min_target_distance,
            "policy_result": policy.result,
            "terminated": env.ever_success,
            "truncated": termination_reason is not None,
            "final_phase": policy.phase.name,
            "phase_transitions": "|".join(phase_transitions),
            **policy.expert_info(),
        })
        return row
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios", nargs="+", choices=list_scenario_names(),
        default=list(DEFAULT_SCENARIOS),
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=1)
    defaults = FormulaInterceptConfig()
    parser.add_argument(
        "--assumed-reach-speed", type=float,
        default=defaults.assumed_reach_speed,
    )
    parser.add_argument(
        "--search-shape-all-segments", action=argparse.BooleanOptionalAction,
        default=defaults.search_shape_all_segments,
    )
    parser.add_argument(
        "--record-intercept-failures", action=argparse.BooleanOptionalAction,
        default=defaults.record_intercept_failures,
    )
    parser.add_argument(
        "--shape-use-scripted-fallback", action=argparse.BooleanOptionalAction,
        default=defaults.shape_use_scripted_fallback,
    )
    parser.add_argument(
        "--dynamic-portfolio-enabled", action=argparse.BooleanOptionalAction,
        default=defaults.dynamic_portfolio_enabled,
    )
    parser.add_argument(
        "--close-capture-distance", type=float,
        default=defaults.close_capture_distance,
    )
    parser.add_argument(
        "--combined-close-capture-distance", type=float,
        default=defaults.combined_close_capture_distance,
    )
    parser.add_argument(
        "--close-prediction-horizon", type=float,
        default=defaults.close_prediction_horizon,
    )
    parser.add_argument(
        "--output", type=Path,
        default=output_path("benchmarks", "privileged_formula_expert"),
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.workers <= 0 or args.episode_seconds <= 0.0:
        parser.error("episodes, workers and episode-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    run_dir = args.output / (
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    jobs = [
        (scenario, episode, args.seed + episode - 1)
        for scenario in args.scenarios
        for episode in range(1, args.episodes + 1)
    ]
    expert_config = asdict(FormulaInterceptConfig(
        assumed_reach_speed=args.assumed_reach_speed,
        search_shape_all_segments=args.search_shape_all_segments,
        record_intercept_failures=args.record_intercept_failures,
        shape_use_scripted_fallback=args.shape_use_scripted_fallback,
        dynamic_portfolio_enabled=args.dynamic_portfolio_enabled,
        close_capture_distance=args.close_capture_distance,
        combined_close_capture_distance=args.combined_close_capture_distance,
        close_prediction_horizon=args.close_prediction_horizon,
    ))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_episode, scenario, episode, seed, args.episode_seconds,
                expert_config,
            ): (scenario, episode)
            for scenario, episode, seed in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"scenario={row['scenario_name']} episode={row['episode']} "
                f"success={row['task_success']} result={row['policy_result']} "
                f"time={row['sim_time']:.3f}s",
                flush=True,
            )
    rows.sort(key=lambda row: (row["scenario_name"], row["episode"]))
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (run_dir / "episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "policy": "FormulaInterceptExpert",
        "privileged": True,
        "scenarios": args.scenarios,
        "episodes_per_scenario": args.episodes,
        "seed": args.seed,
        "episode_seconds": args.episode_seconds,
        "workers": args.workers,
        "expert_config": expert_config,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"output={run_dir.resolve()}")


if __name__ == "__main__":
    main()
