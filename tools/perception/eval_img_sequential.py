"""Sequential episode-level eval for the 2.5-D UNet skeleton model.

Same protocol as eval_sequential.py: full held-out episodes in order,
strict ordered MPNE + flip rate. The model is per-frame (no temporal
feedback) but episodes are replayed in order for protocol parity.
"""

from __future__ import annotations

import argparse
import json
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
)
from panda_cable_grasp.perception.model_img import UNetSkeleton  # noqa: E402
from eval_sequential import load_episode  # noqa: E402
from train_img import extract_nodes  # noqa: E402
from dump_img_ep import rasterize  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenarios", nargs="*", required=True)
    p.add_argument("--seed-files", nargs="*", type=Path, required=True)
    p.add_argument("--eps-per-scenario", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False)
    dev = torch.device(args.device)
    model = UNetSkeleton(cin=5, width=payload["width"]).to(dev).eval()
    model.load_state_dict(payload["model"])
    node_count = payload["node_count"]
    ds = PackedDLOImageDataset.__new__(PackedDLOImageDataset)

    report = {}
    for sc, sf in zip(args.scenarios, args.seed_files):
        seeds = [
            int(s) for s in sf.read_text().split() if s.strip()
        ][: args.eps_per_scenario]
        errs, flips, ep_means = [], [], []
        xy_errs, z_errs = [], []
        for seed in seeds:
            frames = load_episode(args.packed_dir, sc, seed)
            if frames is None:
                print(f"  ! seed {seed} not found in {sc}", flush=True)
                continue
            preds, gts = [], []
            for f in frames:
                img = rasterize(
                    f["points"].astype(np.float32).reshape(-1, 3),
                    f["hand"].astype(np.float32), ds,
                ).to(dev)
                with torch.no_grad():
                    out = model(img)
                pred = extract_nodes(out, node_count, ds)[0].cpu().numpy()
                preds.append(pred.astype(np.float64))
                gts.append(np.asarray(f["pos14"], np.float64))
            pred = np.asarray(preds)
            gt = np.asarray(gts)
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
            print(f"  {sc} seed={seed} frames={len(e)} "
                  f"mpne={e.mean():.1f} xy={exy.mean():.1f} "
                  f"z={ez.mean():.1f} flips={(er < e).sum()}",
                  flush=True)
        e = np.concatenate(errs)
        exy = np.concatenate(xy_errs)
        ez = np.concatenate(z_errs)
        fl = np.concatenate(flips)
        report[sc] = {
            "n_eps": len(errs), "n_frames": int(len(e)),
            "mpne_mm": float(e.mean()),
            "mpne_xy_mm": float(exy.mean()),
            "mae_z_mm": float(ez.mean()),
            "mpne_med_mm": float(np.median(e)),
            "ep_means": ep_means,
            "flip_frames": int(fl.sum()),
            "flip_rate": float(fl.mean()),
        }
        print(f"== {sc}: {len(errs)} eps, MPNE={e.mean():.1f} "
              f"XY={exy.mean():.1f} Z={ez.mean():.1f} "
              f"flip={fl.mean() * 100:.1f}%", flush=True)
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
