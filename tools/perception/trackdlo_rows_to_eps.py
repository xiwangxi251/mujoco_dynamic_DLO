"""Split a run_trackdlo_eval rows.npz into per-episode npz files for
aggregate_seq_baseline.py.

Output: <out_dir>/<episode_name>.npz with keys pred (nodes_cam),
gt (truth_cam), ok (tracking_ok).
"""
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    d = np.load(args.rows, allow_pickle=True)
    eps = d["episode"].astype(str)
    ok = d["tracking_ok"].astype(bool)
    pred = d["nodes_cam"]
    gt = d["truth_cam"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ep in sorted(set(eps)):
        idx = np.where(eps == ep)[0]
        np.savez_compressed(
            args.out_dir / f"{ep}.npz",
            pred=pred[idx],
            gt=gt[idx],
            ok=ok[idx],
        )
        print(ep, len(idx), "frames")


if __name__ == "__main__":
    main()
