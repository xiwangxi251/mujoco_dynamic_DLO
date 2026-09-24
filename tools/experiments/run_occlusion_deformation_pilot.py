"""Paired 2x2 occlusion x deformation pilot for point-cloud PPO policies.

Cells (one episode per seed per cell, all four cells share the seed list):

  deform scenario   x normal render
  deform scenario   x gripper-hidden render
  replay scenario   x normal render
  replay scenario   x gripper-hidden render

Every episode calls ``env.reset(seed=S)`` explicitly, so a replay cell
replays the bank entry recorded under the same seed as the paired deform
episode: identical initial curve, target index, and tracked-node kinematics.
The only intended difference across render cells is the observation stream;
across scenario cells it is whether non-tracked nodes deform or co-move.

Primary endpoint: the render-mode penalty on task success,
``success(normal) - success(hidden)``, compared between the deform pair and
the replay pair (paired bootstrap over seeds).  A substantially larger
penalty in the deform pair supports the occlusion x deformation interaction
hypothesis; equal penalties reject it.
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from panda_cable_grasp.scenarios.registry import get_scenario  # noqa: E402


CELL_KEYS = (
    "deform_normal",
    "deform_hidden",
    "replay_normal",
    "replay_hidden",
)
RENDER_MODE = {
    "deform_normal": "normal",
    "deform_hidden": "gripper_hidden",
    "replay_normal": "normal",
    "replay_hidden": "gripper_hidden",
}


@dataclass(frozen=True)
class RunConfig:
    deform_scenario: str
    replay_scenario: str
    model_path: str
    device: str
    episode_seconds: float
    robot: str
    table_finger_collision_filter: bool
    intervention: str = "gripper_hidden"
    target_mask_radius_m: float = 0.0
    depth_noise_std_at_1m: float = 0.0
    pixel_dropout_p: float = 0.0
    frame_drop_p: float = 0.0


@dataclass(frozen=True)
class EpisodeTask:
    cell: str
    seed: int


def _git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True,
                text=True, timeout=15, check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    """Convert numpy scalars/containers in info dicts into plain JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


_CTX: dict[str, object] = {}


def _worker_env(config: RunConfig, cell: str):
    """One wrapped env per cell per worker process (model compiles once)."""
    from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
    from panda_cable_grasp.rl.pointcloud import (
        DLOPointCloudObservation,
        PointCloudObservationConfig,
    )

    envs: dict[str, object] = _CTX.setdefault("envs", {})
    if cell not in envs:
        scenario = (
            config.deform_scenario
            if cell.startswith("deform")
            else config.replay_scenario
        )
        env = make_rl_env(
            action_mode="task_space_vertical_down",
            robot=config.robot,
            seed=0,
            disturbance_strength=1.5,
            episode_seconds=config.episode_seconds,
            dynamicvla_cameras_enabled=True,
            scenario_names=(scenario,),
            rl_config=RLConfig(singularity_avoidance_enabled=False),
            geometric_safety_enabled=False,
            table_finger_collision_filter_enabled=(
                config.table_finger_collision_filter
            ),
        )
        # intervention=gripper_hidden 时 *_hidden 格关闭夹爪渲染；
        # intervention=target_mask 时 *_hidden 格改为删除目标段
        # target_mask_radius_m 半径内的观测点，渲染保持正常；
        # intervention=depth_noise 时 *_hidden 格注入深度噪声/像素缺失/
        # 丢帧，渲染保持正常。
        render_mode = (
            RENDER_MODE[cell]
            if config.intervention == "gripper_hidden"
            else "normal"
        )
        mask_radius = (
            config.target_mask_radius_m
            if config.intervention == "target_mask" and cell.endswith("hidden")
            else 0.0
        )
        noise_on = (
            config.intervention == "depth_noise" and cell.endswith("hidden")
        )
        wrapped = DLOPointCloudObservation(
            env,
            PointCloudObservationConfig(
                point_count=384,
                width=480,
                height=360,
                camera_update_steps=5,
                sensor_delay_steps=3,
                voxel_size_m=0.002,
                render_mode=render_mode,
                target_mask_radius_m=mask_radius,
                depth_noise_std_at_1m=(
                    config.depth_noise_std_at_1m if noise_on else 0.0
                ),
                pixel_dropout_p=(
                    config.pixel_dropout_p if noise_on else 0.0
                ),
                frame_drop_p=config.frame_drop_p if noise_on else 0.0,
            ),
        )
        # 配对评估必须命中库内 seed；未命中即报错而不是静默换一条运动。
        env.unwrapped.base_env.config.replay_seed_fallback = "error"
        wrapped.set_training_scenarios((scenario,))
        wrapped.set_motion_difficulty(1.0)
        envs[cell] = wrapped
    return envs[cell]


def _worker_model(config: RunConfig):
    if "model" not in _CTX:
        from stable_baselines3 import PPO

        _CTX["model"] = PPO.load(config.model_path, device=config.device)
    return _CTX["model"]


def _run_episode(job: tuple[RunConfig, EpisodeTask]) -> dict:
    config, task = job
    env = _worker_env(config, task.cell)
    model = _worker_model(config)
    observation, _ = env.reset(seed=task.seed)
    done = False
    info: dict = {}
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)
    base_env = env.unwrapped.base_env
    record = {
        "cell": task.cell,
        "scenario": (
            config.deform_scenario
            if task.cell.startswith("deform")
            else config.replay_scenario
        ),
        "render_mode": (
            RENDER_MODE[task.cell]
            if config.intervention == "gripper_hidden"
            else "normal"
        ),
        "intervention": config.intervention,
        "target_mask_radius_m": (
            config.target_mask_radius_m
            if config.intervention == "target_mask"
            and task.cell.endswith("hidden")
            else 0.0
        ),
        "depth_noise_std_at_1m": (
            config.depth_noise_std_at_1m
            if config.intervention == "depth_noise"
            and task.cell.endswith("hidden")
            else 0.0
        ),
        "pixel_dropout_p": (
            config.pixel_dropout_p
            if config.intervention == "depth_noise"
            and task.cell.endswith("hidden")
            else 0.0
        ),
        "frame_drop_p": (
            config.frame_drop_p
            if config.intervention == "depth_noise"
            and task.cell.endswith("hidden")
            else 0.0
        ),
        "seed": task.seed,
        "episode_seed": base_env.episode_seed,
        "success": bool(info.get("success", False)),
        "strict_success": bool(info.get("strict_success", False)),
        "episode_steps": int(info.get("episode_steps", 0)),
        "episode_return": float(info.get("episode_return", np.nan)),
        "termination_reason": info.get("termination_reason"),
        "target_index": int(base_env.cable_index[base_env.target_body_id]),
        "motion_profile_hash": base_env.motion_profile_hash,
        "replay_entry_index": info.get("replay_entry_index"),
        "replay_entry_seed": info.get("replay_entry_seed"),
        "replay_tracked_index": info.get("replay_tracked_index"),
        "ever_pinched": bool(info.get("ever_pinched", False)),
        "ever_aligned_pinch": bool(info.get("ever_aligned_pinch", False)),
        "ever_grasped": bool(info.get("ever_grasped", False)),
        "info": _jsonable(info),
    }
    return record


def _paired_summary(records: list[dict], seeds: list[int]) -> dict:
    """Per-cell rates plus paired bootstrap for the interaction endpoint."""
    by_cell_seed: dict[tuple[str, int], dict] = {
        (record["cell"], record["seed"]): record for record in records
    }
    paired_seeds = [
        seed for seed in seeds
        if all((cell, seed) in by_cell_seed for cell in CELL_KEYS)
    ]
    cells = {}
    for cell in CELL_KEYS:
        outcomes = [
            float(by_cell_seed[(cell, seed)]["success"])
            for seed in paired_seeds
        ]
        strict = [
            float(by_cell_seed[(cell, seed)]["strict_success"])
            for seed in paired_seeds
        ]
        cells[cell] = {
            "episodes": len(outcomes),
            "success_rate": float(np.mean(outcomes)) if outcomes else np.nan,
            "strict_success_rate": (
                float(np.mean(strict)) if strict else np.nan
            ),
        }

    def per_seed(key: str) -> np.ndarray:
        return np.asarray(
            [float(by_cell_seed[(key, seed)]["success"]) for seed in paired_seeds]
        )

    deform_delta = per_seed("deform_normal") - per_seed("deform_hidden")
    replay_delta = per_seed("replay_normal") - per_seed("replay_hidden")
    interaction = deform_delta - replay_delta

    def bootstrap_ci(values: np.ndarray, draws: int = 20000) -> list[float]:
        if not len(values):
            return [float("nan"), float("nan")]
        rng = np.random.default_rng(0)
        means = rng.choice(values, size=(draws, len(values))).mean(axis=1)
        return [float(np.percentile(means, 2.5)),
                float(np.percentile(means, 97.5))]

    def discordants(delta: np.ndarray) -> dict[str, int]:
        return {
            "normal_only_success": int(np.sum(delta > 0)),
            "hidden_only_success": int(np.sum(delta < 0)),
        }

    return {
        "paired_seeds": paired_seeds,
        "paired_episode_count": len(paired_seeds),
        "cells": cells,
        "render_penalty": {
            "deform": {
                "mean": float(np.mean(deform_delta)) if len(deform_delta) else np.nan,
                "ci95": bootstrap_ci(deform_delta),
                **discordants(deform_delta),
            },
            "replay": {
                "mean": float(np.mean(replay_delta)) if len(replay_delta) else np.nan,
                "ci95": bootstrap_ci(replay_delta),
                **discordants(replay_delta),
            },
        },
        "interaction": {
            "mean": float(np.mean(interaction)) if len(interaction) else np.nan,
            "ci95": bootstrap_ci(interaction),
        },
    }


def parse_seed_list(text: str) -> list[int]:
    text = text.strip()
    path = Path(text)
    if path.is_file():
        return [
            int(line.strip())
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if ".." in part:
            start, stop = part.split("..", 1)
            seeds.extend(range(int(start), int(stop) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--deform-scenario", required=True)
    parser.add_argument("--replay-scenario", required=True)
    parser.add_argument(
        "--seeds", required=True,
        help="comma list, start..stop range, or file with one seed per line",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--intervention",
        choices=("gripper_hidden", "target_mask", "depth_noise"),
        default="gripper_hidden",
        help="counterfactual applied in the *_hidden cells",
    )
    parser.add_argument(
        "--target-mask-radius", type=float, default=0.06,
        help="meters; only used with --intervention target_mask",
    )
    parser.add_argument(
        "--depth-noise-std", type=float, default=0.0,
        help="sigma = k*z^2 coefficient (m at z=1m); "
        "only used with --intervention depth_noise",
    )
    parser.add_argument(
        "--pixel-dropout", type=float, default=0.0,
        help="per-pixel missing-return probability; depth_noise only",
    )
    parser.add_argument(
        "--frame-drop", type=float, default=0.0,
        help="whole-frame dropout probability; depth_noise only",
    )
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument("--robot", default="nero")
    parser.add_argument(
        "--disable-table-finger-collision-filter", action="store_true"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.jsonl"
    seeds = parse_seed_list(args.seeds)
    deform = get_scenario(args.deform_scenario)
    replay = get_scenario(args.replay_scenario)

    done: set[tuple[str, int]] = set()
    records: list[dict] = []
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            done.add((record["cell"], int(record["seed"])))
            records.append(record)

    run_config = RunConfig(
        deform_scenario=deform.name,
        replay_scenario=replay.name,
        model_path=str(args.model.resolve()),
        device=args.device,
        episode_seconds=args.episode_seconds,
        robot=args.robot,
        table_finger_collision_filter=(
            not args.disable_table_finger_collision_filter
        ),
        intervention=args.intervention,
        target_mask_radius_m=args.target_mask_radius,
        depth_noise_std_at_1m=args.depth_noise_std,
        pixel_dropout_p=args.pixel_dropout,
        frame_drop_p=args.frame_drop,
    )
    tasks = [
        (run_config, EpisodeTask(cell=cell, seed=seed))
        for seed in seeds
        for cell in CELL_KEYS
        if (cell, seed) not in done
    ]
    total = len(tasks)
    print(
        f"episodes to run: {total} "
        f"(seeds={len(seeds)} x cells={len(CELL_KEYS)}, "
        f"resuming {len(done)})",
        flush=True,
    )

    started = time.time()
    completed = 0
    with results_path.open("a", encoding="utf-8") as stream:
        if args.workers <= 1:
            for task in tasks:
                record = _run_episode(task)
                records.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"progress={completed}/{total}", flush=True)
        else:
            context = mp.get_context("spawn")
            with context.Pool(args.workers) as pool:
                for record in pool.imap_unordered(_run_episode, tasks):
                    records.append(record)
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f"progress={completed}/{total}", flush=True)

    summary = _paired_summary(records, seeds)
    manifest = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": getpass.getuser(),
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "deform_scenario": {
            "name": deform.name,
            "scenario_id": deform.scenario_id,
            "scenario_hash": deform.scenario_hash,
        },
        "replay_scenario": {
            "name": replay.name,
            "scenario_id": replay.scenario_id,
            "scenario_hash": replay.scenario_hash,
        },
        "seeds": seeds,
        "intervention": args.intervention,
        "target_mask_radius_m": args.target_mask_radius,
        "depth_noise_std_at_1m": args.depth_noise_std,
        "pixel_dropout_p": args.pixel_dropout,
        "frame_drop_p": args.frame_drop,
        "cells": {cell: RENDER_MODE[cell] for cell in CELL_KEYS},
        "episode_seconds": args.episode_seconds,
        "robot": args.robot,
        "table_finger_collision_filter": (
            run_config.table_finger_collision_filter
        ),
        "device": args.device,
        "git": _git_state(),
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for cell in CELL_KEYS:
        row = summary["cells"][cell]
        print(
            f"{cell}: n={row['episodes']} "
            f"success={row['success_rate']:.1%} "
            f"strict={row['strict_success_rate']:.1%}",
            flush=True,
        )
    penalty = summary["render_penalty"]
    print(
        f"render penalty deform={penalty['deform']['mean']:+.3f} "
        f"ci95={penalty['deform']['ci95']} | "
        f"replay={penalty['replay']['mean']:+.3f} "
        f"ci95={penalty['replay']['ci95']}",
        flush=True,
    )
    print(
        f"interaction={summary['interaction']['mean']:+.3f} "
        f"ci95={summary['interaction']['ci95']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
