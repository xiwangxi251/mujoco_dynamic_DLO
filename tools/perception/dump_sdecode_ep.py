"""Dump an episode decoding the chain from predicted arc coordinates.

For models with `arc_head`, two decodings of the same forward pass:
  * pos_direct — the trained node head output
  * pos_sarg   — per node j, the cloud point whose predicted s is
                 closest to j/(M-1) (hard assignment; keeps every node
                 literally on the cloud and fixes endpoint identity via
                 the learned s direction)

Writes the same npz layout as dump_occdyn_ep.py with pred = pos_sarg so
render_tracked_npz.py / cloud_adherence.py work unchanged, plus saves
pos_direct for comparison.

Usage mirrors dump_occdyn_ep.py.
"""

from __future__ import annotations

import argparse
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
from dump_occdyn_ep import episode_frames  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--decode", choices=["sarg", "direct"], default="sarg")
    p.add_argument("--out", type=Path, required=True)
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

    clouds, gts, preds, directs = [], [], [], []
    for i, f in enumerate(
        episode_frames(args.packed_dir, args.scenario, args.seed)
    ):
        if args.max_frames and i >= args.max_frames:
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
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.no_grad():
            out = model(b)
        direct = out["pos"][0].cpu().numpy() * scale + center
        cloud = pts.reshape(-1, 3)
        cloud = cloud[np.abs(cloud).sum(1) > 1e-6]
        if args.decode == "sarg" and "arc_s" in out:
            s = out["arc_s"][0].cpu().numpy()          # (Ntot,)
            tgt = np.linspace(0.0, 1.0, cfg.node_count)
            idx = np.abs(s[None, :] - tgt[:, None]).argmin(1)
            pred = cloud[idx].astype(np.float64)
        else:
            pred = direct
        clouds.append(cloud)
        gts.append(f["pos14"].astype(np.float64))
        preds.append(pred.astype(np.float64))
        directs.append(direct.astype(np.float64))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cloud=np.asarray(clouds, dtype=object),
        gt=np.asarray(gts),
        pred=np.asarray(preds),
        pos_direct=np.asarray(directs),
        ok=np.ones(len(gts), bool),
        allow_pickle=True,
    )
    e_dir = np.linalg.norm(
        np.asarray(directs) - np.asarray(gts), axis=2
    ).mean() * 1000
    e_s = np.linalg.norm(
        np.asarray(preds) - np.asarray(gts), axis=2
    ).mean() * 1000
    print(f"saved {len(gts)} frames -> {args.out}")
    print(f"direct={e_dir:.1f}mm  sdecode={e_s:.1f}mm")


if __name__ == "__main__":
    main()
