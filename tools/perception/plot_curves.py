"""Plot error-vs-occlusion and per-scenario comparisons from eval dumps.

Reads the npz produced by evaluate_estimator.py --dump-npz and writes
figures next to it. Multiple dumps may be passed for variant comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def curve_by_bucket(err: np.ndarray, occ: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(occ, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 5:
            continue
        xs.append(0.5 * (edges[b] + edges[b + 1]))
        ys.append(float(np.sqrt((err[m] ** 2).mean())))
    return np.asarray(xs), np.asarray(ys)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dumps", nargs="+", type=Path,
                   help="test_preds.npz files, labelled by parent dir name")
    p.add_argument("--scale-mm", type=float, default=500.0,
                   help="dataset-units-per-mm factor (scale*1000)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for dump in args.dumps:
        z = np.load(dump, allow_pickle=True)
        name = dump.parent.name
        err = np.sqrt(((z["pred_pos"] - z["pos"]) ** 2).sum(-1))  # (B,M)
        occ = z["occ_rate"]
        # frame-level mean error vs frame occlusion rate
        ferr = err.mean(axis=1)
        xs, ys = curve_by_bucket(ferr, occ)
        axes[0].plot(xs, ys * args.scale_mm, "o-", label=name)
        # occluded-node error vs frame occlusion rate
        vis = z["vis_any"]
        occ_err = np.where(vis, np.nan, err)
        ferr_occ = np.nanmean(occ_err, axis=1)
        xs, ys = curve_by_bucket(ferr_occ[~np.isnan(ferr_occ)],
                                 occ[~np.isnan(ferr_occ)])
        axes[1].plot(xs, ys * args.scale_mm, "o-", label=name)
        # per-scenario MPNE bars
        scen = z["scenario"].astype(str)
        labels = sorted(set(scen.tolist()))
        vals = [
            float(np.sqrt((err[scen == s] ** 2).mean())) * args.scale_mm
            for s in labels
        ]
        x = np.arange(len(labels))
        axes[2].plot(x, vals, "o-", label=name)

    axes[0].set(xlabel="frame occluded fraction", ylabel="MPNE (mm)",
                title="state error vs occlusion")
    axes[1].set(xlabel="frame occluded fraction", ylabel="occluded-node MPNE (mm)",
                title="hidden-node error vs occlusion")
    axes[2].set(xlabel="scenario", ylabel="MPNE (mm)", title="per scenario")
    for ax, dump in zip(axes[:2], [None, None]):
        ax.grid(alpha=0.3)
    labels = sorted(set(np.load(args.dumps[0], allow_pickle=True)
                        ["scenario"].astype(str).tolist()))
    axes[2].set_xticks(np.arange(len(labels)), labels, rotation=20)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
