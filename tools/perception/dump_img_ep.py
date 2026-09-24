"""Dump per-frame predictions of the 2.5-D UNet skeleton model.

Same npz layout as dump_occdyn_ep.py (cloud/gt/pred/ok) so
render_tracked_npz.py and cloud_adherence.py work unchanged.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from panda_cable_grasp.perception.dataset import (  # noqa: E402
    PackedDLOImageDataset,
    rasterize_torch,
)
from panda_cable_grasp.perception.model_img import UNetSkeleton  # noqa: E402
from dump_occdyn_ep import episode_frames  # noqa: E402
from train_img import extract_nodes  # noqa: E402


def rasterize(pts: np.ndarray, hand: np.ndarray,
              ds: PackedDLOImageDataset) -> torch.Tensor:
    """(N,3) world cloud + (7,) hand -> (1,5,H,W) tensor (CPU)."""
    pts_t = torch.from_numpy(np.asarray(pts, np.float32))[None]
    hand_t = torch.from_numpy(np.asarray(hand, np.float32))[None]
    dummy = torch.zeros(1, 2, 3)  # tgt output unused for inference
    img, _ = rasterize_torch(
        pts_t, hand_t, dummy,
        ds.X0, ds.X1, ds.Y0, ds.Y1, ds.W, ds.H,
    )
    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--maps-out", type=Path, default=None,
                   help="optionally save predicted maps (mask,s,z) as npz")
    args = p.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False)
    dev = torch.device(args.device)
    model = UNetSkeleton(cin=5, width=payload["width"]).to(dev).eval()
    model.load_state_dict(payload["model"])
    node_count = payload["node_count"]
    ds = PackedDLOImageDataset.__new__(PackedDLOImageDataset)  # constants only

    clouds, gts, preds, maps = [], [], [], []
    for i, f in enumerate(
        episode_frames(args.packed_dir, args.scenario, args.seed)
    ):
        if args.max_frames and i >= args.max_frames:
            break
        pts = f["points"].astype(np.float32)
        hand = f["hand"].astype(np.float32)
        img = rasterize(pts.reshape(-1, 3), hand, ds).to(dev)
        with torch.no_grad():
            out = model(img)
        pred = extract_nodes(out, node_count, ds)[0].cpu().numpy()
        cloud = pts.reshape(-1, 3)
        clouds.append(cloud[np.abs(cloud).sum(1) > 1e-6])
        gts.append(f["pos14"].astype(np.float64))
        preds.append(pred.astype(np.float64))
        if args.maps_out:
            maps.append(out[0].cpu().numpy())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cloud=np.asarray(clouds, dtype=object),
        gt=np.asarray(gts),
        pred=np.asarray(preds),
        ok=np.ones(len(gts), bool),
        allow_pickle=True,
    )
    if args.maps_out:
        np.savez_compressed(args.maps_out, maps=np.asarray(maps))
    e = np.linalg.norm(np.asarray(preds) - np.asarray(gts), axis=2).mean() * 1000
    print(f"saved {len(gts)} frames -> {args.out}  mpne={e:.1f}mm")


if __name__ == "__main__":
    main()
