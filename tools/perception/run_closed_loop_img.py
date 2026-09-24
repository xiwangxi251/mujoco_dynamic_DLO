"""Closed-loop grasp evaluation with the IMAGE DLO model (img_unet_hist / C).

Same protocol as run_closed_loop.py but the estimator is the rasterised
top-down UNet (mask + s + z maps) instead of the point-cloud estimator.
Perception consumes only live camera point clouds + gripper pose —
no privileged cable state. The scripted DynamicGraspPolicy arm on the
same seeds is the GT reference.

Example:
    python tools/perception/run_closed_loop_img.py \
        --checkpoint perception_runs/runs/img_unet_hist/checkpoint_best.pt \
        --scenarios id_static id_combined_l1_nominal --episodes 5 \
        --device cuda:0 --out perception_runs/loop_img_c.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from panda_cable_grasp.runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import numpy as np
import torch

from panda_cable_grasp.env.environment import CableGraspEnv
from panda_cable_grasp.evaluation.motion_diagnostics import (
    env_config_for_scenario,
)
from panda_cable_grasp.perception.closed_loop import (
    HSV_LOWER,
    HSV_UPPER,
    PerceptionConfig,
    PerceptionStack,
    PerceptualGraspPolicy,
    optical_to_world,
)
from panda_cable_grasp.perception.dataset import (
    rasterize_torch,
    resample_polyline_weighted,
)
from panda_cable_grasp.perception.model_img import UNetSkeleton
try:
    from panda_cable_grasp.policies.scripted import DynamicGraspPolicy
except ImportError:
    from panda_cable_grasp.policies.scripted import (
        DynamicCableGraspPolicy as DynamicGraspPolicy,
    )
from panda_cable_grasp.policies.scripted import PolicyConfig
from panda_cable_grasp.scenarios.registry import get_scenario

from test_skel_extract import extract_skel

W, H = 192, 256


class ImgStack(PerceptionStack):
    """PerceptionStack whose estimator is the image UNet.

    Reuses _observe() (live camera clouds, robot masks, hand pose) but
    rasterises the current cloud and runs the mask/s/z model with
    prev-map self feedback, then skeleton-extracts 14 world nodes.
    """

    def __init__(self, env, model: UNetSkeleton, cfg: PerceptionConfig,
                 histmode: str, node_count: int = 14) -> None:
        super().__init__(env, model, cfg)
        self.histmode = histmode
        self.node_count = node_count
        self.prev_map = torch.zeros(1, 3, H, W)
        self.prev_chain: np.ndarray | None = None
        self.prev_time: float | None = None
        self.streak = 0

    def reset(self) -> None:
        super().reset()
        dev = next(self.model.parameters()).device
        self.prev_map = torch.zeros(1, 3, H, W, device=dev)
        self.prev_chain = None
        self.prev_time = None
        self.streak = 0

    def _observe(self) -> None:
        """Trimmed observation for the image model.

        The rasteriser only needs an unordered world-frame cloud, so we
        skip the segmentation pass (robot mask is unused), voxel
        downsampling and FPS ordering that the point-cloud estimator
        requires.  2 cameras x (RGB + depth) = 4 renders instead of 6.
        """
        from panda_cable_grasp.rl.pointcloud import (
            backproject_mask,
            segment_hsv,
        )

        data = self.env.data
        for ci, cam in enumerate(self.cameras):
            self.renderer.update_scene(data, camera=cam)
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(data, camera=cam)
            depth = np.asarray(self.renderer.render(), dtype=np.float32)
            self.renderer.disable_depth_rendering()
            pts = backproject_mask(
                depth, segment_hsv(rgb, HSV_LOWER, HSV_UPPER),
                self.intrinsics[ci],
            )
            world = optical_to_world(
                pts.astype(np.float64),
                data.cam_xpos[self.cam_ids[ci]].astype(np.float64),
                data.cam_xmat[self.cam_ids[ci]].reshape(3, 3)
                .astype(np.float64),
            )
            (self.hist_o if ci == 0 else self.hist_w).append(
                world.astype(np.float32)
            )

    @torch.no_grad()
    def estimate(
        self, future_hand_pos: np.ndarray | None = None
    ) -> dict[str, np.ndarray] | None:
        self._observe()
        dev = next(self.model.parameters()).device
        pts_np = np.concatenate(
            [self.hist_o[-1], self.hist_w[-1]], axis=0
        ).astype(np.float32)                                   # (768,3)
        hand_body = __import__("mujoco").mj_name2id(
            self.env.model, __import__("mujoco").mjtObj.mjOBJ_BODY,
            "link7",
        )
        hp = self.env.data.xpos[hand_body]
        hq = self.env.data.xquat[hand_body]
        hand = np.concatenate([hp, hq]).astype(np.float32)     # (7,)
        pts = torch.from_numpy(pts_np)[None].to(dev)
        img, _ = rasterize_torch(
            pts, torch.from_numpy(hand)[None].to(dev),
            torch.zeros(1, 2, 3, device=dev),
        )
        inp = img
        if "prev" in self.histmode:
            inp = torch.cat([img, self.prev_map.to(dev)], dim=1)
        ot = self.model(inp)
        om = ot[0].cpu().numpy()
        if "prev" in self.histmode:
            bm = (torch.sigmoid(ot[0, 0]) > 0.5).float()
            self.prev_map = torch.stack(
                [bm, ot[0, 1] * bm, ot[0, 2] * bm])[None].detach()
        pm = 1 / (1 + np.exp(-om[0])) > 0.5
        p = extract_skel(pm, om[2], smap=om[1])
        if p is None:
            return None
        if self.prev_chain is not None:
            ef = np.linalg.norm(p - self.prev_chain, axis=1).mean()
            eb = np.linalg.norm(p[::-1] - self.prev_chain, axis=1).mean()
            if eb < ef:
                self.streak += 1
                if self.streak < 25:
                    p = p[::-1]
            else:
                self.streak = 0
        t = float(self.env.data.time)
        vel = np.zeros_like(p)
        if self.prev_chain is not None and self.prev_time is not None:
            dt = t - self.prev_time
            if dt > 1e-4:
                vel = (p - self.prev_chain) / dt
        self.prev_chain = p.copy()
        self.prev_time = t
        pos40, vel40, _ = resample_polyline_weighted(p, vel, 40)
        return {
            "nodes14": p.astype(np.float64),
            "vel14": vel.astype(np.float64),
            "pos40": pos40,
            "vel40": vel40,
        }


def make_env(scenario_name: str, seed: int, episode_seconds: float,
             control_hz: float = 50.0):
    scenario = get_scenario(scenario_name)
    cfg = env_config_for_scenario(
        scenario, seed=seed, episode_seconds=episode_seconds,
        robot="nero",
    )
    cfg.dynamicvla_cameras_enabled = True
    env = CableGraspEnv(cfg)
    env.config.frame_skip = max(
        1, round(1.0 / (control_hz * env.model.opt.timestep))
    )
    return env


def policy_config_for(scenario_name: str) -> PolicyConfig:
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
    model, histmode: str, device: torch.device, control: str,
    control_hz: float = 50.0,
) -> dict:
    env = make_env(scenario_name, seed, episode_seconds, control_hz)
    stack = None
    pol_cfg = policy_config_for(scenario_name)
    if control == "estimator":
        stack = ImgStack(env, model, PerceptionConfig(), histmode)
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
                gt14, _, _ = resample_polyline_weighted(gt, None, 14)
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
        "control_hz": float(
            1.0 / (env.model.opt.timestep * env.config.frame_skip)
        ),
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
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="explicit held-out seeds; overrides --seed-base")
    p.add_argument("--seed-base", type=int, default=20300000)
    p.add_argument("--episode-seconds", type=float, default=15.0)
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--controls", nargs="+",
                   default=["estimator", "gt"],
                   choices=["estimator", "gt"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    model, histmode = None, "none"
    if "estimator" in args.controls:
        ckpt = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
        model = UNetSkeleton(
            cin=ckpt["cin"], width=ckpt["width"]
        ).to(device).eval()
        model.load_state_dict(ckpt["model"])
        histmode = {5: "none", 8: "prev", 11: "prev+occ"}[ckpt["cin"]]

    results = []
    for control in args.controls:
        for scenario_name in args.scenarios:
            seeds = (args.seeds if args.seeds is not None
                     else [args.seed_base + i
                           for i in range(args.episodes)])
            for seed in seeds:
                try:
                    r = run_episode(
                        scenario_name, seed, args.episode_seconds,
                        model, histmode, device, control,
                        args.control_hz,
                    )
                except Exception as exc:
                    r = {"scenario": scenario_name, "seed": seed,
                         "control": control, "success": False,
                         "result": f"exception:{exc}"}
                print(json.dumps(r), flush=True)
                results.append(r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    for control in args.controls:
        for s in args.scenarios:
            sub = [r for r in results
                   if r["control"] == control and r["scenario"] == s]
            if sub:
                rate = np.mean([r["success"] for r in sub])
                mpne = np.mean([r.get("est_mpne_mm", np.nan)
                                for r in sub])
                print(f"{control:9s} {s:28s} success={rate:.2f} "
                      f"n={len(sub)} mpne={mpne:.1f}mm")


if __name__ == "__main__":
    main()
