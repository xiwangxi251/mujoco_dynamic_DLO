"""Evaluate the privileged state-PPO with learned perception instead of GT.

The state PPO consumes a 99-dim observation: 14-node cable positions +
velocities relative to the TCP frame (84 dims, privileged) + robot
proprioception (15 dims).  This script patches the RL env's
``_sample_cable_state`` so those 84 dims come from the img_unet_hist (C)
perception stack driven by live camera point clouds — everything the
policy sees is non-privileged.  Rewards, grasp mechanics and success
judgement still use the simulator state.

Usage (one process per scenario/GPU):
    python tools/experiments/evaluate_ppo_perception.py \
        --model outputs/rl/train/_ckpt/state1553/final_model.zip \
        --img-checkpoint perception_runs/runs/img_unet_hist/checkpoint_best.pt \
        --scenarios id_static --episodes 50 --workers-per-scenario 10 \
        --gpu 0 --output outputs/rl/eval/ppo_perception/id_static
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "tools" / "perception"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from panda_cable_grasp.rl.environment import RLConfig, make_rl_env

SCENARIOS = (
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
    "id_rigid_replay_shape_nominal",
)


def _bind_perception(env, model, histmode: str, est_log: Path):
    """Patch env so policy-visible cable state comes from perception."""
    import torch
    from panda_cable_grasp.perception.closed_loop import PerceptionConfig
    from run_closed_loop_img import ImgStack

    stack = ImgStack(env.base_env, model, PerceptionConfig(), histmode)
    state = {"last": (np.zeros((14, 3)), np.zeros((14, 3))),
             "errs": [], "ep_idx": 0}

    def est_state() -> tuple[np.ndarray, np.ndarray]:
        r = stack.estimate()
        if r is not None:
            state["last"] = (r["nodes14"], r["vel14"])
            gt = env.base_env.data.xpos[env.base_env.cable_ids]
            from panda_cable_grasp.perception.dataset import (
                resample_polyline_weighted,
            )
            gt14, _, _ = resample_polyline_weighted(gt, None, 14)
            state["errs"].append(float(np.sqrt(
                ((r["nodes14"] - gt14) ** 2).sum(-1).mean())) * 1000)
        return state["last"]

    orig_reset = env.reset
    orig_step = env.step

    def reset(*a, **k):
        stack.reset()
        state["last"] = (np.zeros((14, 3)), np.zeros((14, 3)))
        state["errs"] = []
        return orig_reset(*a, **k)

    def step(action):
        out = orig_step(action)
        terminated, truncated = out[2], out[3]
        if terminated or truncated:
            errs = state["errs"]
            with est_log.open("a") as fh:
                fh.write(json.dumps({
                    "ep": state["ep_idx"],
                    "mpne_mean": float(np.mean(errs)) if errs else None,
                    "mpne_p90": float(np.percentile(errs, 90))
                    if errs else None,
                    "n_est": len(errs),
                    "success": bool(out[4].get("success", False)),
                }) + "\n")
            state["ep_idx"] += 1
        return out

    env._sample_cable_state = est_state
    env.reset = reset
    env.step = step


@dataclass(frozen=True)
class EvalEnvFactory:
    scenario: str
    seed: int
    gpu: int
    img_checkpoint: str
    est_log: str
    table_finger_collision_filter: bool

    def __call__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        os.environ.setdefault("MUJOCO_GL", "egl")
        import torch
        from panda_cable_grasp.perception.model_img import UNetSkeleton

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
            table_finger_collision_filter_enabled=(
                self.table_finger_collision_filter
            ),
        )
        ckpt = torch.load(
            self.img_checkpoint, map_location="cuda", weights_only=False
        )
        model = UNetSkeleton(
            cin=ckpt["cin"], width=ckpt["width"]
        ).to("cuda").eval()
        model.load_state_dict(ckpt["model"])
        histmode = {5: "none", 8: "prev", 11: "prev+occ"}[ckpt["cin"]]
        _bind_perception(env, model, histmode, Path(self.est_log))
        if hasattr(env, "set_training_scenarios"):
            env.set_training_scenarios((self.scenario,))
        if hasattr(env, "set_motion_difficulty"):
            env.set_motion_difficulty(1.0)
        return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--img-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers-per-scenario", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20270915)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--disable-table-finger-collision-filter",
                        action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scenarios = (
        tuple(s.strip() for s in args.scenarios.split(",") if s.strip())
        if args.scenarios else SCENARIOS
    )
    est_log = args.output / "est_mpne.jsonl"
    worker_scenarios = tuple(
        s for s in scenarios for _ in range(args.workers_per_scenario)
    )
    factories = [
        EvalEnvFactory(
            scenario,
            args.seed + rank,
            args.gpu,
            str(args.img_checkpoint.resolve()),
            str(est_log),
            not args.disable_table_finger_collision_filter,
        )
        for rank, scenario in enumerate(worker_scenarios)
    ]
    env = SubprocVecEnv(factories, start_method="spawn")
    model = PPO.load(args.model, device="cpu")
    observations = env.reset()
    records: dict[str, list[dict]] = {s: [] for s in scenarios}
    completed = 0
    target = args.episodes * len(scenarios)
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
                records[scenario].append({
                    "episode": len(records[scenario]) + 1,
                    "episode_seed": info.get("episode_seed"),
                    "success": bool(info.get("success", False)),
                    "strict_success": bool(
                        info.get("strict_success", False)
                    ),
                    "pinch": bool(info.get("ever_pinched", False)),
                    "aligned_pinch": bool(
                        info.get("ever_aligned_pinch", False)
                    ),
                    "lift_attempt": bool(info.get("lift_attempt", False)),
                    "loaded_lift": bool(info.get("loaded_lift", False)),
                    "grasp": bool(info.get("ever_grasped", False)),
                    "length": int(info.get("episode_steps", 0)),
                    "return": float(info.get("episode_return", np.nan)),
                })
                completed += 1
                if completed % 10 == 0:
                    print(f"progress={completed}/{target}", flush=True)
    finally:
        env.close()

    rows = []
    for scenario in scenarios:
        eps = records[scenario]
        row = {"scenario": scenario, "episodes": len(eps)}
        for name in ("success", "strict_success", "pinch",
                     "aligned_pinch", "lift_attempt", "loaded_lift",
                     "grasp"):
            row[f"{name}_rate"] = float(
                np.mean([e[name] for e in eps])) if eps else float("nan")
        row["mean_length"] = float(
            np.mean([e["length"] for e in eps])) if eps else float("nan")
        rows.append(row)
        print(json.dumps(row), flush=True)

    with (args.output / "summary.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "episodes.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps({
        "model": str(args.model.resolve()),
        "img_checkpoint": str(args.img_checkpoint.resolve()),
        "episodes_per_scenario": args.episodes,
        "workers_per_scenario": args.workers_per_scenario,
        "seed": args.seed,
        "perception": "img_unet_hist C model; _sample_cable_state patched",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
