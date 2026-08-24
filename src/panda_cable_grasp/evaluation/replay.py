"""Validate or replay a recorded FULLPHYSICS evaluation episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ..runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import mujoco
import numpy as np


def load_episode(episode_dir: Path) -> tuple[mujoco.MjModel, np.ndarray, np.ndarray, int]:
    episode_dir = Path(episode_dir).resolve()
    metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / metadata["files"]["trajectory"]) as trajectory:
        states = np.asarray(trajectory["states"], dtype=np.float64)
        times = np.asarray(trajectory["state_times"], dtype=np.float64)
        state_spec = int(trajectory["state_spec"])
    model_reference = Path(metadata["result"]["compiled_model"])
    run_dir = episode_dir.parents[3]
    model_path = model_reference if model_reference.is_absolute() else run_dir / model_reference
    model = mujoco.MjModel.from_binary_path(str(model_path))
    expected = mujoco.mj_stateSize(model, state_spec)
    if states.ndim != 2 or states.shape[1] != expected:
        raise ValueError(
            f"state/model mismatch: states={states.shape}, expected width={expected}"
        )
    if times.shape != (states.shape[0],) or np.any(np.diff(times) < 0.0):
        raise ValueError("state_times must be monotonic and match states")
    return model, states, times, state_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or replay an evaluation episode")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    return args


def main() -> None:
    args = parse_args()
    model, states, times, state_spec = load_episode(args.episode_dir)
    data = mujoco.MjData(model)
    for state in states:
        mujoco.mj_setState(model, data, state, state_spec)
        mujoco.mj_forward(model, data)
    print(
        f"replay_check=OK states={len(states)} sim_time={times[-1]:.6f}s",
        flush=True,
    )
    if args.check_only:
        return

    from mujoco import viewer

    with viewer.launch_passive(model, data) as handle:
        for index, state in enumerate(states):
            if not handle.is_running():
                break
            mujoco.mj_setState(model, data, state, state_spec)
            mujoco.mj_forward(model, data)
            handle.sync()
            if index + 1 < len(states):
                time.sleep(max(0.0, (times[index + 1] - times[index]) / args.speed))


if __name__ == "__main__":
    main()
