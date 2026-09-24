"""Render a closed-loop episode to mp4: GT cable vs estimated cable.

Layout per video frame (all panels 480x360, RGB space until final convert):
  [ opst camera | wrist camera | top-down world XY view ]

Overlays (cable renders blue -> blue is never used for annotations):
  green chain    = ground-truth cable nodes
  red chain      = OccDyn-DLO current estimate
  orange chain   = OccDyn-DLO future prediction (top-down panel only)
  yellow cross   = policy's filtered intercept target
  magenta disc   = hand XY (top-down panel only)
  gray dots      = observed segmentation point cloud (top-down panel only)

Example:
    python tools/perception/viz_closed_loop.py \
        --checkpoint runs/full/checkpoint_best.pt \
        --scenario id_static --seed 20300001 --out viz.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from panda_cable_grasp.runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import cv2
import mujoco
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
from panda_cable_grasp.policies.scripted import PolicyConfig
from panda_cable_grasp.scenarios.registry import get_scenario

RENDER_H, RENDER_W = 360, 480
_CAM_OPTICAL = np.diag([1.0, -1.0, -1.0])

# RGB colours (MuJoCo renderer output is RGB; converted once at write time).
C_GT = (0, 230, 0)        # green
C_EST = (255, 45, 45)     # red
C_FUT = (255, 165, 0)     # orange
C_TGT = (255, 225, 0)     # yellow
C_HAND = (255, 0, 255)    # magenta
C_CLOUD = (205, 205, 205)  # light gray
C_GRID = (70, 70, 70)

# Top-down view: fixed workspace bounds so motion is visible frame-to-frame.
XY_CENTER = np.array([0.55, 0.0])
XY_SPAN = 1.15  # metres covered along the shorter panel axis


def policy_config_for(scenario_name: str) -> PolicyConfig:
    kw: dict = {"lift_distance": 0.3}
    if "combined" in scenario_name:
        kw.update(
            prediction_horizon=0.2,
            approach_prediction_horizon=0.2,
            approach_position_tolerance=0.1,
        )
    return PolicyConfig(**kw)


def project_points(
    pts_w: np.ndarray,
    intrinsics: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """world (N,3) -> pixel (u,v) + in-front flag."""
    w2o = _CAM_OPTICAL @ cam_mat.T
    rel = (pts_w - cam_pos) @ w2o.T
    z = np.maximum(rel[:, 2], 1e-9)
    u = intrinsics[0, 0] * rel[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * rel[:, 1] / z + intrinsics[1, 2]
    ok = (rel[:, 2] > 0.01) & (u >= 0) & (u < RENDER_W) & (v >= 0) & (
        v < RENDER_H
    )
    return np.stack([u, v], axis=1), ok


def draw_chain(
    img: np.ndarray, uv: np.ndarray, ok: np.ndarray, color
) -> None:
    pts = np.round(uv).astype(int)
    for i in range(len(pts) - 1):
        if ok[i] and ok[i + 1]:
            cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, 3)
    for i, p in enumerate(pts):
        if ok[i]:
            cv2.circle(img, tuple(p), 4, color, -1)


def draw_cross(img: np.ndarray, uv, ok, color, size: int = 10) -> None:
    if not ok:
        return
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.line(img, (u - size, v), (u + size, v), color, 2)
    cv2.line(img, (u, v - size), (u, v + size), color, 2)


def draw_topdown(
    clouds: list[np.ndarray],
    gt_w: np.ndarray,
    est_w: np.ndarray,
    fut_w: np.ndarray | None,
    target_w: np.ndarray,
    hand_w: np.ndarray,
) -> np.ndarray:
    """Isometric top-down world-XY panel (RENDER_W x RENDER_H)."""
    panel = np.full((RENDER_H, RENDER_W, 3), 30, dtype=np.uint8)
    margin = 24.0
    ppm = min(
        (RENDER_W - 2 * margin) / XY_SPAN,
        (RENDER_H - 2 * margin) / XY_SPAN,
    )

    def px(pts: np.ndarray) -> np.ndarray:
        xy = np.asarray(pts, dtype=np.float64)[:, :2] - XY_CENTER
        out = np.empty((len(xy), 2))
        out[:, 0] = RENDER_W / 2 + xy[:, 0] * ppm
        out[:, 1] = RENDER_H / 2 - xy[:, 1] * ppm
        return np.round(out).astype(np.int32)

    for gx in np.arange(-0.2, 1.4, 0.2):
        x = px(np.array([[gx, 0.0, 0.0]]))[0, 0]
        cv2.line(panel, (x, 0), (x, RENDER_H - 1), C_GRID, 1)
    for gy in np.arange(-0.8, 0.81, 0.2):
        y = px(np.array([[0.0, gy, 0.0]]))[0, 1]
        cv2.line(panel, (0, y), (RENDER_W - 1, y), C_GRID, 1)

    for cloud in clouds:
        for p in px(cloud):
            if 0 <= p[0] < RENDER_W and 0 <= p[1] < RENDER_H:
                cv2.circle(panel, tuple(p), 2, C_CLOUD, -1)

    cv2.polylines(panel, [px(gt_w)], False, C_GT, 1, cv2.LINE_AA)
    if fut_w is not None:
        fp = px(fut_w)
        for i in range(len(fp) - 1):
            cv2.line(panel, tuple(fp[i]), tuple(fp[i + 1]), C_FUT, 1)
    ep = px(est_w)
    cv2.polylines(panel, [ep], False, C_EST, 2, cv2.LINE_AA)
    for p in ep:
        cv2.circle(panel, tuple(p), 4, C_EST, -1)

    tp = px(target_w[None])[0]
    cv2.line(panel, (tp[0] - 9, tp[1]), (tp[0] + 9, tp[1]), C_TGT, 2)
    cv2.line(panel, (tp[0], tp[1] - 9), (tp[0], tp[1] + 9), C_TGT, 2)
    hp = px(hand_w[None])[0]
    cv2.circle(panel, tuple(hp), 7, C_HAND, 2)

    cv2.putText(panel, "top-down XY", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(
        panel, "green=GT red=est orange=future", (10, RENDER_H - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return panel


class TrackedView:
    """Auto-fit top-down XY view that follows the cable.

    Mirrors the `_world_panel` logic in the dlo_state_estimation
    TrackDLO visualisation: bounds come from the observed cloud (2-98
    percentile), GT and estimate, with a minimum span and light temporal
    smoothing so the view tracks the cable without jitter.
    """

    def __init__(self, w: int = 720, h: int = 540) -> None:
        self.w, self.h = w, h
        self._bounds: np.ndarray | None = None

    def _fit(self, *arrays: np.ndarray) -> np.ndarray:
        cores = []
        for arr in arrays:
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim != 2 or len(a) == 0:
                continue
            a = a[np.isfinite(a[:, :2]).all(1), :2]
            if len(a) > 50:  # robust bounds for the dense cloud
                a = np.percentile(a, [2, 98], axis=0)
            if len(a):
                cores.append(a)
        if not cores:
            return self._bounds
        xy = np.concatenate(cores, 0)
        low, high = xy.min(0), xy.max(0)
        span = np.maximum(high - low, [0.15, 0.15])
        ctr = 0.5 * (low + high)
        span = span + 2 * np.maximum(0.04, 0.15 * span)
        ratio = self.w / self.h
        if span[0] / span[1] < ratio:
            span[0] = span[1] * ratio
        else:
            span[1] = span[0] / ratio
        bounds = np.array([ctr[0] - span[0] / 2, ctr[0] + span[0] / 2,
                           ctr[1] - span[1] / 2, ctr[1] + span[1] / 2])
        if self._bounds is None:
            self._bounds = bounds
        else:
            self._bounds = 0.75 * self._bounds + 0.25 * bounds
        return self._bounds

    def px(self, pts: np.ndarray) -> np.ndarray:
        b = self._bounds
        xy = np.asarray(pts, dtype=np.float64)[:, :2]
        out = np.empty((len(xy), 2))
        out[:, 0] = (xy[:, 0] - b[0]) / (b[1] - b[0]) * (self.w - 1)
        out[:, 1] = (1 - (xy[:, 1] - b[2]) / (b[3] - b[2])) * (self.h - 1)
        return np.round(out).astype(np.int32)

    def fit(self, *arrays: np.ndarray) -> None:
        self._fit(*arrays)

    def draw(
        self,
        clouds: list[np.ndarray],
        gt_w: np.ndarray,
        est_w: np.ndarray,
        fut_w: np.ndarray | None,
        cloud_only: bool = False,
    ) -> np.ndarray:
        panel = np.full((self.h, self.w, 3), 36, dtype=np.uint8)
        if self._bounds is None:
            return panel
        for f in np.linspace(0, 1, 6):
            x = int(f * (self.w - 1))
            y = int(f * (self.h - 1))
            cv2.line(panel, (x, 0), (x, self.h - 1), C_GRID, 1)
            cv2.line(panel, (0, y), (self.w - 1, y), C_GRID, 1)
        for cloud in clouds:
            for p in self.px(cloud):
                if 0 <= p[0] < self.w and 0 <= p[1] < self.h:
                    cv2.circle(panel, tuple(p), 2, C_CLOUD, -1)
        if not cloud_only:
            cv2.polylines(panel, [self.px(gt_w)], False, C_GT, 2,
                          cv2.LINE_AA)
            if fut_w is not None:
                fp = self.px(fut_w)
                for i in range(len(fp) - 1):
                    cv2.line(panel, tuple(fp[i]), tuple(fp[i + 1]),
                             C_FUT, 1)
            ep = self.px(est_w)
            cv2.polylines(panel, [ep], False, C_EST, 2, cv2.LINE_AA)
            for p in ep:
                cv2.circle(panel, tuple(p), 5, C_EST, -1)
        m_per_px = (self._bounds[1] - self._bounds[0]) / self.w
        bar_px = max(8, int(0.10 / m_per_px))
        cv2.line(panel, (20, self.h - 26), (20 + bar_px, self.h - 26),
                 (230, 230, 230), 2)
        cv2.putText(panel, "10cm", (20, self.h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return panel


def _hand_pos(env: CableGraspEnv) -> np.ndarray:
    return np.asarray(env.hand_position, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenario", default="id_static")
    p.add_argument("--seed", type=int, default=20300001)
    p.add_argument("--episode-seconds", type=float, default=15.0)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--stride", type=int, default=1,
                   help="record every Nth control step")
    p.add_argument("--tracked-out", type=Path, default=None,
                   help="second video: auto-fit tracked top-down + "
                        "cloud-only panel, no policy overlays")
    args = p.parse_args()

    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    raw_cfg = {
        k: v for k, v in payload["config"].items()
        if k in EstimatorConfig.__dataclass_fields__
    }
    raw_cfg.setdefault("use_future_hand", False)
    raw_cfg.setdefault("use_history", False)
    model = DLOStateEstimator(EstimatorConfig(**raw_cfg)).to(device).eval()
    model.load_state_dict(payload["model"])
    pcfg = PerceptionConfig(
        center=np.asarray(payload["center"], dtype=np.float64),
        scale=float(payload["scale"]),
        history=args.history,
    )

    scenario = get_scenario(args.scenario)
    cfg = env_config_for_scenario(
        scenario, seed=args.seed,
        episode_seconds=args.episode_seconds, robot="nero",
    )
    cfg.dynamicvla_cameras_enabled = True
    env = CableGraspEnv(cfg)

    stack = PerceptionStack(env, model, pcfg)
    policy = PerceptualGraspPolicy(env, stack, policy_config_for(args.scenario))
    env.reset(seed=args.seed)
    policy.reset()

    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps, (RENDER_W * 3, RENDER_H),
    )
    tracker = TrackedView() if args.tracked_out else None
    tracked_writer = (
        cv2.VideoWriter(
            str(args.tracked_out), cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps, (tracker.w * 2, tracker.h),
        )
        if tracker else None
    )
    cams = stack.cameras
    cam_ids = stack.cam_ids
    intr = stack.intrinsics

    step = 0
    errs: list[float] = []
    while (
        not policy.finished
        and env.data.time < env.config.episode_seconds
    ):
        policy.new_control_step()
        est = policy._est_nodes()
        gt = env.data.xpos[env.cable_ids]
        gt14, _, _ = resample_polyline_weighted(gt, None, pcfg.node_count)
        err = float(np.sqrt(((est["nodes14"] - gt14) ** 2).sum(-1)
                            .mean())) * 1000
        errs.append(err)

        action = policy.action()
        _, _, _, truncated, _ = env.step(action)
        if truncated:
            policy.result = "failed_timeout"
            policy.finished = True

        if step % args.stride:
            step += 1
            continue
        step += 1

        frames = []
        for ci, cam in enumerate(cams):
            stack.renderer.update_scene(env.data, camera=cam)
            rgb = stack.renderer.render().copy()
            cam_pos = env.data.cam_xpos[cam_ids[ci]]
            cam_mat = env.data.cam_xmat[cam_ids[ci]].reshape(3, 3)
            uv_g, ok_g = project_points(gt14, intr[ci], cam_pos, cam_mat)
            uv_e, ok_e = project_points(
                est["nodes14"], intr[ci], cam_pos, cam_mat
            )
            draw_chain(rgb, uv_g, ok_g, C_GT)
            draw_chain(rgb, uv_e, ok_e, C_EST)
            uv_t, ok_t = project_points(
                policy.filtered_target[None], intr[ci], cam_pos, cam_mat
            )
            draw_cross(rgb, uv_t[0], ok_t[0], C_TGT)
            cv2.putText(
                rgb, f"{cam}  t={env.data.time:4.1f}s  "
                     f"{policy.phase}  err={err:5.1f}mm",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1,
            )
            frames.append(rgb)
        frames.append(
            draw_topdown(
                [stack.hist_o[-1], stack.hist_w[-1]],
                gt, est["nodes14"], est.get("future14"),
                policy.filtered_target, _hand_pos(env),
            )
        )
        writer.write(cv2.cvtColor(np.hstack(frames), cv2.COLOR_RGB2BGR))

        if tracker is not None:
            clouds = [stack.hist_o[-1], stack.hist_w[-1]]
            cloud_all = np.concatenate(clouds)
            tracker.fit(cloud_all, gt, est["nodes14"])
            left = tracker.draw(
                clouds, gt, est["nodes14"], est.get("future14")
            )
            right = tracker.draw(clouds, gt, est["nodes14"], None,
                                 cloud_only=True)
            cv2.putText(left, "tracked XY: cloud + GT + est + future",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(left, f"t={env.data.time:4.1f}s  "
                        f"err={err:5.1f}mm", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(right, "observed cloud only (same view)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tracked_writer.write(
                cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_RGB2BGR)
            )

    writer.release()
    if tracked_writer is not None:
        tracked_writer.release()
    print(
        f"result={policy.result} success={env.ever_success} "
        f"steps={step} est_mpne={np.mean(errs):.1f}mm -> {args.out}"
    )


if __name__ == "__main__":
    main()
