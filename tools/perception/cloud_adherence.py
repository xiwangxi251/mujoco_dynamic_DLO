"""Cloud-adherence + strand-assignment diagnostics on a dump npz.

Loads a dump produced by dump_occdyn_ep.py (cloud/gt/pred per frame) and
reports, per frame and aggregated:

  ordered_mm   strict ordered MPNE (comparable metric)
  rev_mm       reversed-order error (large fwd-rev gap => endpoint flip)
  pred2cloud   mean dist from each pred NODE to nearest cloud point
               (are nodes on the cloud?)
  seg2cloud    mean dist from samples along pred SEGMENTS to nearest
               cloud point (does the chain float through empty space?)
  cloud2pred   mean dist from each cloud point to nearest pred segment
               (does the chain cover the visible cable?)
  uncov_frac   fraction of cloud points > thresh from the chain
  node_far     fraction of pred nodes > thresh from cloud

Usage:
  python tools/perception/cloud_adherence.py --dump ep.npz [--thresh 0.03]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _seg_samples(a: np.ndarray, b: np.ndarray, k: int = 12) -> np.ndarray:
    t = np.linspace(0.0, 1.0, k)
    return a[:, None] + t[None, :, None] * (b - a)[:, None]  # (S,K,3)


def _pt2seg(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,3) points to (S,) segments -> (N,S) distances."""
    ab = b - a                                   # (S,3)
    ap = pts[:, None] - a[None]                  # (N,S,3)
    t = np.clip(
        (ap * ab[None]).sum(-1)
        / np.maximum((ab * ab).sum(-1), 1e-12)[None],
        0.0, 1.0,
    )
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None] - proj, axis=-1)


def frame_metrics(pred: np.ndarray, gt: np.ndarray, cloud: np.ndarray,
                  thresh: float) -> dict:
    fwd = np.linalg.norm(pred - gt, axis=1).mean()
    rev = np.linalg.norm(pred[::-1] - gt, axis=1).mean()
    a, b = pred[:-1], pred[1:]
    d_node = np.linalg.norm(
        pred[:, None] - cloud[None], axis=-1
    ).min(1)
    samp = _seg_samples(a, b).reshape(-1, 3)
    d_seg = np.linalg.norm(
        samp[:, None] - cloud[None], axis=-1
    ).min(1)
    d_c2p = _pt2seg(cloud, a, b).min(1)
    return {
        "ordered_mm": fwd * 1000,
        "rev_mm": rev * 1000,
        "pred2cloud_mm": float(d_node.mean()) * 1000,
        "seg2cloud_mm": float(d_seg.mean()) * 1000,
        "cloud2pred_mm": float(d_c2p.mean()) * 1000,
        "uncov_frac": float((d_c2p > thresh).mean()),
        "node_far_frac": float((d_node > thresh).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--thresh", type=float, default=0.03)
    ap.add_argument("--worst", type=int, default=8,
                    help="print the N worst frames by ordered err")
    args = ap.parse_args()
    z = np.load(args.dump, allow_pickle=True)
    cloud, gt, pred = z["cloud"], z["gt"], z["pred"]
    rows = []
    for i in range(len(pred)):
        c = cloud[i] if cloud.dtype == object else cloud[i]
        c = c[np.abs(c).sum(1) > 1e-6]
        rows.append(frame_metrics(
            np.asarray(pred[i], np.float64),
            np.asarray(gt[i], np.float64),
            np.asarray(c, np.float64), args.thresh,
        ))
    keys = rows[0].keys()
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    hi = [r for r in rows if r["ordered_mm"] > 50]
    hi_agg = (
        {k: float(np.mean([r[k] for r in hi])) for k in keys}
        if hi else {}
    )
    print(json.dumps({
        "n_frames": len(rows),
        "n_hi_err": len(hi),
        "all": {k: round(v, 2) for k, v in agg.items()},
        "hi_err": {k: round(v, 2) for k, v in hi_agg.items()},
    }, indent=1))
    order = np.argsort([-r["ordered_mm"] for r in rows])
    print("worst frames:")
    for i in order[: args.worst]:
        r = rows[i]
        print(f"  f{i:3d} ord={r['ordered_mm']:6.1f} rev={r['rev_mm']:6.1f} "
              f"p2c={r['pred2cloud_mm']:5.1f} s2c={r['seg2cloud_mm']:5.1f} "
              f"c2p={r['cloud2pred_mm']:5.1f} uncov={r['uncov_frac']:.2f}")


if __name__ == "__main__":
    main()
