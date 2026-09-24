"""Sequential episode-level evaluation (self-feedback) — PRIMARY metric.

Replays whole held-out test episodes in order. For models with
``use_prev_pos`` the model's OWN previous output is fed back as the
prior (deployment-equivalent); for other variants frames are simply
independent.

Reports per-scenario:
  * strict ordered MPNE (mm), mean/median over all frames
  * flip rate: fraction of frames where reversed-pred matches GT better

Usage:
  python tools/perception/eval_sequential.py \
      --packed-dir <packed> --checkpoint <ckpt> \
      --scenarios id_static id_shape_nominal_current \
      --seed-files test_seeds_id_static.txt test_seeds_id_shape.txt \
      --eps-per-scenario 10 --device cuda:3
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from panda_cable_grasp.perception.model import (  # noqa: E402
    DLOStateEstimator,
    EstimatorConfig,
)

KEYS = ("points", "pos14", "masks", "hand", "hand_fut")


def load_episode(packed_dir: Path, scenario: str, seed: int):
    """Return list-of-dicts for one episode, or None."""
    for posf in sorted(
        glob.glob(str(packed_dir / f"{scenario}__part*.pos14.npy"))
    ):
        stem = posf[: -len(".pos14.npy")]
        seeds = np.load(stem + ".seeds.npy", mmap_mode="r")
        sel = np.where(seeds[:] == seed)[0]
        if not len(sel):
            continue
        frame_i = np.load(stem + ".frame_i.npy", mmap_mode="r")
        order = sel[np.argsort(frame_i[sel])]
        arrs = {
            k: np.load(stem + "." + k + ".npy", mmap_mode="r") for k in KEYS
        }
        return [
            {k: np.asarray(v[i]) for k, v in arrs.items()} for i in order
        ]
    return None


def run_episode(model, cfg, frames, center, scale, dev, max_frames=None,
                repair=False, repair_thresh=0.025):
    """Self-feedback sequential pass; returns (pred, gt) world-frame."""
    preds, gts = [], []
    prev_norm = None
    for i, f in enumerate(frames):
        if max_frames and i >= max_frames:
            break
        pts = f["points"].astype(np.float32)
        hand = f["hand"].astype(np.float32)
        b = {
            "points_opst": torch.from_numpy(
                (pts[0] - center) / scale
            )[None, None],
            "points_wrist": torch.from_numpy(
                (pts[1] - center) / scale
            )[None, None],
            "hand_pos": torch.from_numpy(
                (hand[:3] - center) / scale
            )[None],
            "hand_quat": torch.from_numpy(hand[3:])[None],
        }
        if cfg.predict_future or cfg.use_future_hand:
            hf = f["hand_fut"].astype(np.float32)
            b["hand_pos_future"] = torch.from_numpy(
                (hf[:3] - center) / scale
            )[None]
            b["hand_quat_future"] = torch.from_numpy(hf[3:])[None]
        if cfg.use_mask:
            b["robot_mask_opst"] = torch.from_numpy(
                f["masks"][0].astype(np.float32)
            )[None]
            b["robot_mask_wrist"] = torch.from_numpy(
                f["masks"][1].astype(np.float32)
            )[None]
        if cfg.use_prev_pos:
            if prev_norm is None:
                b["prev_pos"] = torch.zeros(1, cfg.node_count, 3)
                b["has_prev"] = torch.zeros(1)
            else:
                b["prev_pos"] = torch.from_numpy(prev_norm)[None]
                b["has_prev"] = torch.ones(1)
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.no_grad():
            out = model(b)
        pred = out["pos"][0].cpu().numpy() * scale + center
        if cfg.use_prev_pos:
            prev_norm = ((pred - center) / scale).astype(np.float32)
        if repair:
            from panda_cable_grasp.perception.repair import (
                repair_shortcuts,
            )
            cloud = pts.reshape(-1, 3).astype(np.float64)
            cloud = cloud[np.abs(cloud).sum(1) > 1e-6]
            pred, _ = repair_shortcuts(
                pred.astype(np.float64), cloud,
                support_thresh=repair_thresh,
            )
        preds.append(pred.astype(np.float64))
        gts.append(f["pos14"].astype(np.float64))
    return np.asarray(preds), np.asarray(gts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenarios", nargs="*", required=True)
    p.add_argument("--seed-files", nargs="*", type=Path, required=True)
    p.add_argument("--eps-per-scenario", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repair", action="store_true",
                   help="apply repair_shortcuts post-processing to every "
                        "prediction (chord re-routing through cloud)")
    p.add_argument("--repair-thresh", type=float, default=0.025)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False)
    raw_cfg = {
        k: v for k, v in payload["config"].items()
        if k in EstimatorConfig.__dataclass_fields__
    }
    cfg = EstimatorConfig(**raw_cfg)
    dev = torch.device(args.device)
    model = DLOStateEstimator(cfg).to(dev).eval()
    model.load_state_dict(payload["model"])
    center = np.asarray(payload["center"], np.float32)
    scale = float(payload["scale"])

    report = {}
    for sc, sf in zip(args.scenarios, args.seed_files):
        seeds = [
            int(s) for s in sf.read_text().split()
            if s.strip()
        ][: args.eps_per_scenario]
        errs, flips, ep_means = [], [], []
        xy_errs, z_errs = [], []
        for seed in seeds:
            frames = load_episode(args.packed_dir, sc, seed)
            if frames is None:
                print(f"  ! seed {seed} not found in {sc}", flush=True)
                continue
            pred, gt = run_episode(
                model, cfg, frames, center, scale, dev, args.max_frames,
                repair=args.repair, repair_thresh=args.repair_thresh,
            )
            e = np.linalg.norm(pred - gt, axis=2).mean(1) * 1000
            er = np.linalg.norm(pred[:, ::-1] - gt, axis=2).mean(1) * 1000
            exy = np.linalg.norm(
                (pred - gt)[..., :2], axis=2).mean(1) * 1000
            ez = np.abs(pred[..., 2] - gt[..., 2]).mean(1) * 1000
            errs.append(e)
            xy_errs.append(exy)
            z_errs.append(ez)
            flips.append(er < e)
            ep_means.append(float(e.mean()))
            print(
                f"  {sc} seed={seed} frames={len(e)} "
                f"mpne={e.mean():.1f} xy={exy.mean():.1f} "
                f"z={ez.mean():.1f} flips={(er < e).sum()}",
                flush=True,
            )
        e = np.concatenate(errs)
        exy = np.concatenate(xy_errs)
        ez = np.concatenate(z_errs)
        fl = np.concatenate(flips)
        report[sc] = {
            "n_eps": len(errs),
            "n_frames": int(len(e)),
            "mpne_mm": float(e.mean()),
            "mpne_xy_mm": float(exy.mean()),
            "mae_z_mm": float(ez.mean()),
            "mpne_med_mm": float(np.median(e)),
            "ep_means": ep_means,
            "ep_mean_of_means": float(np.mean(ep_means)),
            "flip_frames": int(fl.sum()),
            "flip_rate": float(fl.mean()),
        }
        print(
            f"== {sc}: {len(errs)} eps, {len(e)} frames, "
            f"MPNE={e.mean():.1f} XY={exy.mean():.1f} Z={ez.mean():.1f} "
            f"med={np.median(e):.1f} flip_rate={fl.mean() * 100:.1f}%",
            flush=True,
        )
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
