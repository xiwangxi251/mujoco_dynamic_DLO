"""Record free-run material-node trajectories for ``rigid_replay_v1`` banks.

For each requested seed the source scenario is reset identically to a paired
evaluation episode, then stepped with the robot held at its ready pose using
the environment's own disturbance field (no policy, no success bookkeeping).
Every cable node's planar position is recorded at control rate; the replay
profile later replays one chosen node's motion as the rigid transform of the
whole shape-frozen cable.

Paired usage: generate entries for exactly the evaluation seed list, so a
``rigid_replay`` episode with seed S replays the motion that the source
episode with seed S exhibited.  Extra unpaired entries (``--extra-entries``)
supply the motion distribution for RL training resets that carry no seed.

Rigid/combined sources stop recording when the cable COM crosses the
scenario's exit line (the point where a real episode would terminate); later
rows are padded with the last recorded pose so the replayed cable simply
holds at the boundary instead of diving off the table.
"""

from __future__ import annotations

import argparse
import datetime
import getpass
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

from panda_cable_grasp.env.environment import (  # noqa: E402
    CableGraspEnv,
    EnvConfig,
)
from panda_cable_grasp.env.replay_bank import BANK_FORMAT_VERSION  # noqa: E402
from panda_cable_grasp.scenarios.registry import get_scenario  # noqa: E402


@dataclass(frozen=True)
class EntryRequest:
    index: int
    seed: int


@dataclass(frozen=True)
class WorkerJob:
    scenario_name: str
    episode_seconds: float
    robot: str
    frame_skip: int | None
    requests: tuple[EntryRequest, ...]


def _git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True,
                text=True, timeout=15, check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - manifest must not fail the build
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


_WORKER_ENV: "CableGraspEnv | None" = None


def _worker_env(
    scenario_name: str,
    episode_seconds: float,
    robot: str,
    frame_skip: int | None = None,
):
    """One lazily-constructed env per worker process (model compiles once)."""
    global _WORKER_ENV
    if _WORKER_ENV is None:
        scenario = get_scenario(scenario_name)
        overrides = scenario.to_env_overrides()
        if frame_skip is not None:
            overrides["frame_skip"] = frame_skip
        _WORKER_ENV = CableGraspEnv(
            EnvConfig(
                robot=robot,
                seed=0,
                episode_seconds=episode_seconds,
                dynamicvla_cameras_enabled=False,
                **overrides,
            )
        )
    return _WORKER_ENV


def _record_entry(
    env: "CableGraspEnv", seed: int, episode_seconds: float
) -> dict[str, np.ndarray | int | float]:
    """Replay one free-run episode and return the recorded node track."""
    import mujoco

    env.reset(randomize=True, seed=seed)

    cable_ids = np.asarray(env.cable_ids, dtype=np.int64)
    node_count = len(cable_ids)
    frame_skip = max(1, env.config.frame_skip)
    control_dt = float(env.model.opt.timestep * frame_skip)
    horizon = int(round(episode_seconds / control_dt)) + 1
    positions = np.empty((horizon, node_count, 2), dtype=np.float32)
    positions[0] = env.data.xpos[cable_ids, :2]
    placed_com = np.average(
        env.data.xpos[cable_ids, :2], axis=0, weights=env.cable_mass
    )
    target_index = int(env.cable_index[env.target_body_id])

    # Rigid/combined sources stop at the episode-termination boundary, then
    # hold the final pose; shape sources always record the full horizon.
    exit_y = (
        float(env.config.rigid_motion_exit_y)
        if env.config.motion_mode in {"rigid", "combined"}
        else None
    )

    ready_ctrl = env.ready_ctrl.copy()
    valid = horizon
    for row in range(1, horizon):
        for _ in range(frame_skip):
            env.data.xfrc_applied[:] = 0.0
            env._apply_cable_disturbance()
            env.data.ctrl[:] = ready_ctrl
            mujoco.mj_step(env.model, env.data)
        positions[row] = env.data.xpos[cable_ids, :2]
        if exit_y is not None:
            com_y = float(np.average(
                env.data.xpos[cable_ids, 1], weights=env.cable_mass
            ))
            if com_y >= exit_y:
                valid = row + 1
                break
    if valid < horizon:
        positions[valid:] = positions[valid - 1]
    return {
        "positions_xy": positions,
        "valid_steps": valid,
        "placed_com_xy": placed_com,
        "target_index": target_index,
        "cable_ids": cable_ids,
        "control_dt": control_dt,
        "frame_skip": frame_skip,
        "physics_timestep": float(env.model.opt.timestep),
    }


def _worker(job: WorkerJob) -> dict[int, dict[str, np.ndarray | int | float]]:
    env = _worker_env(
        job.scenario_name, job.episode_seconds, job.robot, job.frame_skip
    )
    results: dict[int, dict[str, np.ndarray | int | float]] = {}
    for request in job.requests:
        results[request.index] = _record_entry(
            env, request.seed, job.episode_seconds
        )
    return results


def parse_seed_list(text: str) -> list[int]:
    """Accept ``a,b,c``, ``start..stop`` ranges, or a file with one per line."""
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
    parser.add_argument("--source-scenario", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds", default="",
        help="comma list, start..stop range, or file with one seed per line",
    )
    parser.add_argument(
        "--extra-entries", type=int, default=0,
        help="anonymous entries with generated seeds for unseeded training "
        "resets",
    )
    parser.add_argument(
        "--extra-seed-base", type=int, default=88_000_000,
        help="first seed of the anonymous entry block",
    )
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument("--robot", default="nero")
    parser.add_argument(
        "--frame-skip", type=int, default=None,
        help="override env frame_skip (default: EnvConfig default 10 = 50Hz; "
             "use 20 for the 25Hz learned-policy protocol)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--chunk-entries", type=int, default=32,
        help="entries recorded per worker task",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = get_scenario(args.source_scenario)
    paired_seeds = parse_seed_list(args.seeds) if args.seeds else []
    extra_seeds = [
        args.extra_seed_base + index for index in range(args.extra_entries)
    ]
    all_seeds = paired_seeds + extra_seeds
    if not all_seeds:
        raise SystemExit("no seeds requested")
    if len(set(all_seeds)) != len(all_seeds):
        raise SystemExit("seed list contains duplicates")

    requests = tuple(
        EntryRequest(index=index, seed=seed)
        for index, seed in enumerate(all_seeds)
    )
    jobs = [
        WorkerJob(
            scenario_name=scenario.name,
            episode_seconds=args.episode_seconds,
            robot=args.robot,
            frame_skip=args.frame_skip,
            requests=requests[start:start + args.chunk_entries],
        )
        for start in range(0, len(requests), args.chunk_entries)
    ]

    started = time.time()
    results: dict[int, dict[str, np.ndarray | int | float]] = {}
    if args.workers <= 1:
        for job in jobs:
            results.update(_worker(job))
            print(f"entries={len(results)}/{len(requests)}", flush=True)
    else:
        context = mp.get_context("spawn")
        with context.Pool(args.workers) as pool:
            for shard in pool.imap_unordered(_worker, jobs):
                results.update(shard)
                print(f"entries={len(results)}/{len(requests)}", flush=True)

    first = results[0]
    positions = np.stack(
        [results[index]["positions_xy"] for index in range(len(all_seeds))]
    )
    node_count = positions.shape[2]
    horizon = positions.shape[1]
    placed = np.stack(
        [results[index]["placed_com_xy"] for index in range(len(all_seeds))]
    )
    valid = np.asarray(
        [results[index]["valid_steps"] for index in range(len(all_seeds))],
        dtype=np.int64,
    )
    targets = np.asarray(
        [results[index]["target_index"] for index in range(len(all_seeds))],
        dtype=np.int64,
    )
    cable_ids = np.asarray(first["cable_ids"], dtype=np.int64)
    control_dt = float(first["control_dt"])

    overrides = scenario.sample_for_episode(0).to_env_overrides()
    meta = {
        "format": "replay_bank",
        "format_version": BANK_FORMAT_VERSION,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": getpass.getuser(),
        "source_scenario": scenario.name,
        "source_scenario_id": scenario.scenario_id,
        "source_scenario_hash": scenario.scenario_hash,
        "source_motion_mode": scenario.motion_type.value,
        "source_env_overrides": overrides,
        "source_start_y_placement": (
            scenario.motion_type.value in {"rigid", "combined"}
        ),
        "paired_entry_count": len(paired_seeds),
        "anonymous_entry_count": len(extra_seeds),
        "control_dt": control_dt,
        "physics_timestep": float(first["physics_timestep"]),
        "frame_skip": int(first["frame_skip"]),
        "episode_seconds": args.episode_seconds,
        "horizon_steps": horizon,
        "node_count": node_count,
        "robot": args.robot,
        "git": _git_state(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        format_version=np.asarray(BANK_FORMAT_VERSION, dtype=np.int32),
        meta_json=np.asarray(json.dumps(meta)),
        seeds=np.asarray(all_seeds, dtype=np.int64),
        positions_xy=positions.astype(np.float32),
        valid_steps=valid,
        placed_com_xy=placed.astype(np.float64),
        target_index=targets,
        cable_ids=cable_ids,
    )
    manifest = dict(meta)
    manifest.update(
        {
            "output": str(args.output.resolve()),
            "bytes": args.output.stat().st_size,
            "elapsed_seconds": time.time() - started,
        }
    )
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {args.output} entries={len(all_seeds)} "
        f"({len(paired_seeds)} paired + {len(extra_seeds)} anonymous) "
        f"horizon={horizon} nodes={node_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
