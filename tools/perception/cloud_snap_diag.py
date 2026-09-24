"""Diagnose how well the estimator exploits the observed point cloud.

For each test frame computes (world space):
  * est -> cloud distance per node (does the chain hug visible geometry?)
  * gt  -> cloud distance per node (cloud coverage sanity check)
  * ordered est->gt error per node
  * "missed-snap": nodes whose GT sits within R_mm of the cloud (i.e. the
    observation visibly contains the cable there) but the estimate is
    still E_mm+ away from GT.

Runs on the packed mmap dataset; model checkpoint required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import torch

from panda_cable_grasp.perception.dataset import PackedDLODataset
from panda_cable_grasp.perception.model import (
    DLOStateEstimator,
    EstimatorConfig,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--scenario", default=None,
                   help="restrict to one scenario (default: all)")
    p.add_argument("--test-seeds", type=Path, nargs="+", required=True)
    p.add_argument("--max-frames", type=int, default=4000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
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
    cfg = EstimatorConfig(**raw_cfg)
    model = DLOStateEstimator(cfg).to(device).eval()
    model.load_state_dict(payload["model"])
    center = np.asarray(payload["center"], dtype=np.float64)
    scale = float(payload["scale"])

    exclude: set[str] = set()
    for sf in args.test_seeds:
        exclude |= set(sf.read_text().split())
    ds = PackedDLODataset(
        args.packed_dir,
        history=max(1, 4 if cfg.use_history else 1),
        center=center,
        scale=scale,
        seed_filter=exclude,
        exclude_seed_files=True,
        scenarios=[args.scenario] if args.scenario else None,
    )
    print(f"frames={len(ds)}")
    idx = np.linspace(0, len(ds) - 1, min(args.max_frames, len(ds))).astype(
        np.int64
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx.tolist()),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    est2cloud, gt2cloud, ord_err, npts = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            out = model(batch)
            est = out["pos"].cpu().numpy() * scale + center  # (B,14,3)
            gt = batch["node_pos"].cpu().numpy() * scale + center
            co = batch["points_opst"][:, -1].cpu().numpy() * scale + center
            cw = batch["points_wrist"][:, -1].cpu().numpy() * scale + center
            for b in range(est.shape[0]):
                cloud = np.concatenate([co[b], cw[b]], axis=0)
                cloud = cloud[np.isfinite(cloud).all(1)]
                cloud = cloud[np.abs(cloud).sum(1) > 1e-6]
                if len(cloud) < 8:
                    continue
                d_est = np.sqrt(
                    ((est[b][:, None, :] - cloud[None]) ** 2).sum(-1)
                ).min(1)
                d_gt = np.sqrt(
                    ((gt[b][:, None, :] - cloud[None]) ** 2).sum(-1)
                ).min(1)
                est2cloud.append(d_est)
                gt2cloud.append(d_gt)
                ord_err.append(np.sqrt(((est[b] - gt[b]) ** 2).sum(-1)))
                npts.append(len(cloud))

    est2cloud = np.concatenate(est2cloud) * 1000
    gt2cloud = np.concatenate(gt2cloud) * 1000
    ord_err = np.concatenate(ord_err) * 1000
    print(f"used_frames={len(npts)}  cloud_pts/frame={np.mean(npts):.0f}")
    print(f"est->cloud  mean={est2cloud.mean():.1f}mm  "
          f"med={np.median(est2cloud):.1f}mm  "
          f"p90={np.percentile(est2cloud, 90):.1f}mm")
    print(f"gt ->cloud  mean={gt2cloud.mean():.1f}mm  "
          f"med={np.median(gt2cloud):.1f}mm  "
          f"p90={np.percentile(gt2cloud, 90):.1f}mm")
    print(f"est->gt ord mean={ord_err.mean():.1f}mm  "
          f"med={np.median(ord_err):.1f}mm")
    for r in (10, 20, 30):
        near = gt2cloud < r
        if near.sum():
            print(
                f"GT node has cloud<{r}mm: frac={near.mean():.2f}  "
                f"est_ord_err={ord_err[near].mean():.1f}mm  "
                f"est->cloud={est2cloud[near].mean():.1f}mm"
            )


if __name__ == "__main__":
    main()
