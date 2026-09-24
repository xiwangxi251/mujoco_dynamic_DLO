"""Dump per-frame UniStateDLO-repro predictions for video rendering.

Mirrors the eval in unistatedlo_nero_repro/tools/cross_coarse_eval.py:
per-frame normalisation -> CrossCoarse -> un-normalise -> smooth_chain
-> 0.75/0.25 temporal EMA. Writes one npz:

    cloud: (T, N, 3)  cached segmented cloud (world)
    gt:    (T, 50, 3) GT node chain (world)
    pred:  (T, 50, 3) UniStateDLO coarse prediction (world)
    ok:    (T,) bool

Scenario -> checkpoint mapping follows the reported reproduction
numbers: static/rigid use runs_static_rigid/cross_static_rigid.pt,
shape/combined use runs_shape_combined/cross_shape_combined.pt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_REPRO = Path("/data1/hxai/unistatedlo_nero_repro")
sys.path.insert(0, str(_REPRO / "tools"))
sys.path.insert(0, str(_REPRO))

from cross_coarse_visualize import CrossCoarse, smooth_chain  # noqa: E402

_GROUP = {
    "id_static": ("nero_static_rigid.yaml", "runs_static_rigid/cross_static_rigid.pt"),
    "id_rigid_l1_nominal": (
        "nero_static_rigid.yaml",
        "runs_static_rigid/cross_static_rigid.pt",
    ),
    "id_shape_nominal_current": (
        "nero_shape_combined.yaml",
        "runs_shape_combined/cross_shape_combined.pt",
    ),
    "id_combined_l1_nominal": (
        "nero_shape_combined.yaml",
        "runs_shape_combined/cross_shape_combined.pt",
    ),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ep-rank", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    cfg_name, ckpt_rel = _GROUP[args.scenario]
    cfg = yaml.safe_load((_REPRO / "configs" / cfg_name).read_text())
    d = cfg["dataset"]
    device = torch.device(args.device)
    model = CrossCoarse(int(d["nodes"]), int(cfg["model"]["feature_dim"]))
    model.load_state_dict(
        torch.load(_REPRO / ckpt_rel, map_location=device)["model"]
    )
    model.to(device).eval()

    manifest = json.loads(
        (Path(d["cache_root"]) / "manifest.json").read_text()
    )
    records = [
        r
        for r in manifest["records"]
        if r["scenario"] == args.scenario and r["split"] == "test"
    ]
    record = records[args.ep_rank]
    with np.load(record["cache"], allow_pickle=False) as data:
        points = data["points"].astype(np.float32)
        truth = data["nodes"].astype(np.float32)
    if args.max_frames:
        points = points[: args.max_frames]
        truth = truth[: args.max_frames]

    centers = points.mean(axis=1)
    scales = np.maximum(
        np.ptp(points, axis=1).max(axis=1), 0.05
    ).astype(np.float32)
    norm = (points - centers[:, None, :]) / scales[:, None, None]
    preds = []
    with torch.no_grad():
        for s in range(0, len(norm), 128):
            out = model(torch.from_numpy(norm[s : s + 128]).to(device))
            preds.append(out.cpu().numpy())
    preds = np.concatenate(preds) * scales[:, None, None] + centers[:, None, :]

    smoothed = []
    prev = None
    for pr in preds:
        q = smooth_chain(pr)
        if prev is not None:
            q = 0.75 * q + 0.25 * prev
        prev = q
        smoothed.append(q)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cloud=points,
        gt=truth,
        pred=np.asarray(smoothed, np.float32),
        ok=np.ones(len(smoothed), bool),
    )
    print(
        f"{args.scenario} seed={record['seed']} "
        f"frames={len(smoothed)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
