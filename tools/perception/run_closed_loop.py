"""Closed-loop grasp evaluation with the learned DLO state estimator.

Runs scripted-policy episodes where cable state comes from
PerceptionStack (live rendering + OccDyn-DLO) instead of ground truth.
A GT-driven DynamicGraspPolicy arm on the same seeds provides the
reference success rate.

Example:
    python tools/perception/run_closed_loop.py \
        --checkpoint runs/full/checkpoint_best.pt \
        --scenarios id_static id_combined_l1_nominal --episodes 8 \
        --device cuda:3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from panda_cable_grasp.runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import numpy as np
import torch

from panda_cable_grasp.env.environment import CableGraspEnv
from panda_cable_grasp.evaluation.motion_diagnostics import (
    env_config_for_scenario,
)
from panda_cable_grasp.perception.closed_loop import (
    PerceptionConfig,
    PerceptionStack,
    PerceptualGraspPolicy,
)
from panda_cable_grasp.perception.dataset import resample_polyline_weighted
from panda_cable_grasp.perception.model import (
    DLOStateEstimator,
    EstimatorConfig,
)
try:
    from panda_cable_grasp.policies.scripted import DynamicGraspPolicy
except ImportError:
    from panda_cable_grasp.policies.scripted import (
        DynamicCableGraspPolicy as DynamicGraspPolicy,
    )
from panda_cable_grasp.policies.scripted import PolicyConfig
from panda_cable_grasp.scenarios.registry import get_scenario


def make_env(scenario_name: str, seed: int, episode_seconds: float):
    scenario = get_scenario(scenario_name)
    cfg = env_config_for_scenario(
        scenario, seed=seed, episode_seconds=episode_seconds,
        robot="nero",
    )
    cfg.dynamicvla_cameras_enabled = True
    return CableGraspEnv(cfg)


def policy_config_for(scenario_name: str) -> PolicyConfig:
    # Match the dataset-collection tuning (config.json of
    # nero_scripted_dynamic20_4x1000_20260906): scripted_config applies to
    # every scenario; combined_scripted_config overrides for the combined
    # split. Using defaults otherwise makes even GT control fail.
    kw: dict = {"lift_distance": 0.3}
    if "combined" in scenario_name:
        kw.update(
            prediction_horizon=0.2,
            approach_prediction_horizon=0.2,
            approach_position_tolerance=0.1,
        )
    return PolicyConfig(**kw)


def run_episode(
    scenario_name: str, seed: int, episode_seconds: float,
    model, pcfg: PerceptionConfig, device: torch.device,
    control: str,
) -> dict:
    env = make_env(scenario_name, seed, episode_seconds)
    stack = None
    pol_cfg = policy_config_for(scenario_name)
    if control == "estimator":
        stack = PerceptionStack(env, model, pcfg)
        policy = PerceptualGraspPolicy(env, stack, pol_cfg)
    else:
        policy = DynamicGraspPolicy(env, pol_cfg)
    env.reset(seed=seed)
    policy.reset()

    errs, steps, t0 = [], 0, time.time()
    est_ms = []
    while not policy.finished and env.data.time < env.config.episode_seconds:
        if stack is not None:
            policy.new_control_step()
            ts = time.time()
            est = policy._est_nodes()
            est_ms.append((time.time() - ts) * 1000)
            if est is not None:
                gt = env.data.xpos[env.cable_ids]
                gt14, _, _ = resample_polyline_weighted(
                    gt, None, pcfg.node_count
                )
                errs.append(
                    float(np.sqrt(
                        ((est["nodes14"] - gt14) ** 2).sum(-1).mean()
                    )) * 1000
                )
        action = policy.action()
        _, _, _, truncated, _ = env.step(action)
        steps += 1
        if truncated:
            policy.result = "failed_timeout"
            policy.finished = True
    out = {
        "scenario": scenario_name,
        "seed": seed,
        "control": control,
        "success": bool(env.ever_success),
        "result": policy.result,
        "steps": steps,
        "wall_s": round(time.time() - t0, 1),
    }
    if errs:
        out["est_mpne_mm"] = float(np.mean(errs))
        out["est_p90_mm"] = float(np.percentile(errs, 90))
    if est_ms:
        out["est_ms_mean"] = float(np.mean(est_ms))
        out["est_ms_p95"] = float(np.percentile(est_ms, 95))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--scenarios", nargs="+", default=["id_static"])
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=20300000)
    p.add_argument("--episode-seconds", type=float, default=15.0)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--controls", nargs="+",
                   default=["estimator", "gt"],
                   choices=["estimator", "gt"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    model, pcfg = None, PerceptionConfig()
    if "estimator" in args.controls:
        payload = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
        raw_cfg = {
            k: v for k, v in payload["config"].items()
            if k in EstimatorConfig.__dataclass_fields__
        }
        raw_cfg.setdefault("use_future_hand", False)
        raw_cfg.setdefault("use_history", False)
        cfg = EstimatorConfig(**raw_cfg)
        model = DLOStateEstimator(cfg).to(device).eval()
        model.load_state_dict(payload["model"])
        pcfg.center = np.asarray(payload["center"], dtype=np.float64)
        pcfg.scale = float(payload["scale"])
        pcfg.history = args.history

    results = []
    for control in args.controls:
        for scenario_name in args.scenarios:
            for i in range(args.episodes):
                seed = args.seed_base + i
                try:
                    r = run_episode(
                        scenario_name, seed, args.episode_seconds,
                        model, pcfg, device, control,
                    )
                except Exception as exc:
                    r = {"scenario": scenario_name, "seed": seed,
                         "control": control, "success": False,
                         "result": f"exception:{exc}"}
                print(json.dumps(r), flush=True)
                results.append(r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    # summary
    for control in args.controls:
        for s in args.scenarios:
            sub = [r for r in results
                   if r["control"] == control and r["scenario"] == s]
            if sub:
                rate = np.mean([r["success"] for r in sub])
                print(f"{control:9s} {s:28s} success={rate:.2f} "
                      f"({sum(r['success'] for r in sub)}/{len(sub)})")


if __name__ == "__main__":
    main()
