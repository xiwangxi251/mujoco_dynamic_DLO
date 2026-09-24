"""Pack per-episode npz files into per-scenario flat arrays (mmap-friendly).

Eliminates per-sample npz decompression and transforms: stores
world-frame point clouds, resampled node targets, visibility, masks and
hand data already in model-ready form.

Output per scenario (uncompressed npz, memmap-loadable):
  points   (N, 2, P, 3)   float16 world-frame clouds
  masks    (N, 2, 90,120) uint8 robot occluder masks
  pos14    (N, M, 3)      float32
  vel14    (N, M, 3)      float32
  fut14    (N, M, 3)      float32 (position at t+future_steps, clamped)
  vis14    (N, 2, M)      uint8
  hand     (N, 7)         float32 pos+quat at t
  hand_fut (N, 7)         float32 pos+quat at t+H
  lead     (N,)           float32 seconds to future frame
  ep_id    (N,)           int32 episode index (per-scenario)
  frame_i  (N,)           int32 frame index inside episode
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from panda_cable_grasp.perception.dataset import (
    optical_to_world,
    resample_polyline_weighted,
)


def pack_scenario(
    scenario_dir: Path, out_prefix: Path, node_count: int,
    future_steps: int, min_future_frac: float = 0.5,
    eps_per_chunk: int = 64,
) -> dict:
    """Stream episodes into chunked uncompressed npz files.

    Accumulating all rows in RAM exhausts the 125GB shared host, so every
    ``eps_per_chunk`` episodes are flushed to ``<scenario>__partNN.npz``.
    """
    files = sorted(scenario_dir.glob("seed_*.npz"))
    rows = {k: [] for k in (
        "points", "masks", "pos", "vel", "fut", "vis", "hand", "hand_fut",
        "lead", "ep_id", "frame_i", "seeds_row",
    )}
    kept = skipped = chunk = 0
    seeds: list[int] = []
    chunk_ep0 = 0

    def flush(ep_i: int) -> None:
        nonlocal chunk
        if not rows["lead"]:
            return
        out = {
            "points": np.asarray(rows["points"], np.float16),
            "masks": np.asarray(rows["masks"], np.uint8),
            "pos14": np.asarray(rows["pos"], np.float32),
            "vel14": np.asarray(rows["vel"], np.float32),
            "fut14": np.asarray(rows["fut"], np.float32),
            "vis14": np.asarray(rows["vis"], np.uint8),
            "hand": np.asarray(rows["hand"], np.float32),
            "hand_fut": np.asarray(rows["hand_fut"], np.float32),
            "lead": np.asarray(rows["lead"], np.float32),
            "ep_id": np.asarray(rows["ep_id"], np.int32),
            "frame_i": np.asarray(rows["frame_i"], np.int32),
            "seeds": np.asarray(rows["seeds_row"], np.int64),
        }
        base = f"{out_prefix}__part{chunk:02d}"
        for k, arr in out.items():
            np.save(f"{base}.{k}.npy", arr)  # raw .npy -> true os mmap
        for k in rows:
            rows[k] = []
        chunk += 1

    for ep_i, path in enumerate(files):
        if ep_i - chunk_ep0 >= eps_per_chunk:
            flush(ep_i)
            chunk_ep0 = ep_i
        with np.load(path, allow_pickle=True) as z:
            seed = int(z["seed"])
            seeds.append(seed)
            n = len(z["time"])
            P = z["points"].shape[2]
            for f in range(n):
                if (n - 1 - f) < future_steps * min_future_frac:
                    skipped += 1
                    continue
                fu = min(f + future_steps, n - 1)
                pts_f = np.empty((2, P, 3), np.float32)
                for c in range(2):
                    pts_f[c] = optical_to_world(
                        z["points"][f, c].astype(np.float64),
                        z["cam_pos"][f, c].astype(np.float64),
                        z["cam_mat"][f, c].astype(np.float64),
                    )
                p, v, frac = resample_polyline_weighted(
                    z["node_pos"][f], z["node_vel"][f], node_count
                )
                lo = np.floor(frac).astype(int)
                hi = np.clip(lo + 1, 0, len(frac) - 1)
                vs = z["node_vis"][f]
                kept += 1
                rows["points"].append(pts_f)
                rows["masks"].append(np.asarray(z["robot_mask"][f]).copy())
                rows["pos"].append(p.astype(np.float32))
                rows["vel"].append(v.astype(np.float32))
                rows["fut"].append(
                    resample_polyline_weighted(
                        z["node_pos"][fu], None, node_count
                    )[0].astype(np.float32)
                )
                rows["vis"].append(vs[:, lo] & vs[:, hi])
                rows["hand"].append(
                    np.concatenate(
                        [z["hand_pos"][f], z["hand_quat"][f]]
                    ).astype(np.float32)
                )
                rows["hand_fut"].append(
                    np.concatenate(
                        [z["hand_pos"][fu], z["hand_quat"][fu]]
                    ).astype(np.float32)
                )
                rows["lead"].append(
                    float(z["time"][fu] - z["time"][f])
                )
                rows["ep_id"].append(ep_i)
                rows["frame_i"].append(f)
                rows["seeds_row"].append(seed)
    flush(len(files))
    return {"scenario": scenario_dir.name, "frames": kept,
            "skipped": skipped, "episodes": len(files),
            "chunks": chunk}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--node-count", type=int, default=14)
    p.add_argument("--future-steps", type=int, default=8)
    p.add_argument("--scenarios", nargs="*", default=None)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for scen_dir in sorted(Path(args.data_root).iterdir()):
        if not scen_dir.is_dir():
            continue
        if args.scenarios and scen_dir.name not in args.scenarios:
            continue
        t0 = time.time()
        existing = list(
            args.out_dir.glob(f"{scen_dir.name}__part*.points.npy")
        )
        if existing:
            print(f"skip {scen_dir.name} ({len(existing)} chunks exist)")
            continue
        info = pack_scenario(
            scen_dir, args.out_dir / scen_dir.name,
            args.node_count, args.future_steps,
        )
        info["sec"] = round(time.time() - t0, 1)
        print(info, flush=True)


if __name__ == "__main__":
    main()
