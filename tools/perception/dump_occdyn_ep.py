"""Dump per-frame OccDyn-DLO predictions on one held-out episode.

Reads frames of a single packed episode (contiguous, in order), runs a
trained checkpoint, and writes one npz for render_tracked_npz.py:

    cloud: (T, 768, 3) merged dual-camera cloud (world)
    gt:    (T, 14, 3)
    pred:  (T, 14, 3)
    ok:    (T,) bool

Usage:
    python tools/perception/dump_occdyn_ep.py \
        --packed-dir <packed> --checkpoint <ckpt> \
        --scenario id_combined_l1_nominal --ep-rank 0 --out ep.npz
"""

from __future__ import annotations

import argparse
import glob
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


def episode_frames(packed_dir: Path, scenario: str, seed: int):
    """Yield (points(2,P,3), pos14, mask, hand, hand_fut) per frame of
    the episode whose seed matches, in order."""
    for posf in sorted(glob.glob(str(packed_dir / f"{scenario}__part*.pos14.npy"))):
        stem = posf[: -len(".pos14.npy")]
        seeds = np.load(stem + ".seeds.npy", mmap_mode="r")
        sel = np.where(seeds[:] == seed)[0]
        if not len(sel):
            continue
        frame_i = np.load(stem + ".frame_i.npy", mmap_mode="r")
        order = sel[np.argsort(frame_i[sel])]
        arrs = {
            k: np.load(stem + "." + k + ".npy", mmap_mode="r")
            for k in ("points", "pos14", "masks", "hand", "hand_fut")
        }
        print(f"episode seed={seed} frames={len(order)}", flush=True)
        for i in order:
            yield {k: np.asarray(v[i]) for k, v in arrs.items()}
        return
    raise SystemExit(f"seed {seed} not found in {scenario}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--seed", type=int, required=True,
                   help="episode seed to dump (use held-out test seeds)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repair", action="store_true",
                   help="post-process each prediction with "
                        "repair_shortcuts (chord re-routing through "
                        "orphaned cloud points)")
    p.add_argument("--repair-thresh", type=float, default=0.025,
                   help="metres; segment samples / orphan cutoff")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = {
        k: v
        for k, v in payload["config"].items()
        if k in EstimatorConfig.__dataclass_fields__
    }
    for flag in ("use_future_hand", "use_history", "use_voting", "local_k"):
        raw_cfg.setdefault(flag, 0 if flag == "local_k" else False)
    cfg = EstimatorConfig(**raw_cfg)
    dev = torch.device(args.device)
    model = DLOStateEstimator(cfg).to(dev).eval()
    model.load_state_dict(payload["model"])
    center = np.asarray(payload["center"], np.float32)
    scale = float(payload["scale"])

    clouds, gts, preds = [], [], []
    prev_norm = None
    for i, f in enumerate(
        episode_frames(args.packed_dir, args.scenario, args.seed)
    ):
        if args.max_frames and i >= args.max_frames:
            break
        pts = f["points"].astype(np.float32)          # (2, P, 3)
        hand = f["hand"].astype(np.float32)           # (7,)
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
            # self-feedback tracking prior: own previous estimate
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
        cloud = pts.reshape(-1, 3)
        cloud = cloud[np.abs(cloud).sum(1) > 1e-6]
        if args.repair:
            from panda_cable_grasp.perception.repair import (
                repair_shortcuts,
            )
            pred, _ = repair_shortcuts(
                pred.astype(np.float64), cloud.astype(np.float64),
                support_thresh=args.repair_thresh,
            )
        clouds.append(cloud)
        gts.append(f["pos14"].astype(np.float64))
        preds.append(pred.astype(np.float64))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cloud=np.asarray(clouds, dtype=object),
        gt=np.asarray(gts),
        pred=np.asarray(preds),
        ok=np.ones(len(gts), bool),
        allow_pickle=True,
    )
    print(f"saved {len(gts)} frames -> {args.out}")


if __name__ == "__main__":
    main()
