"""Overlay pred mask (red) vs GT mask (green) per frame -> video."""
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import cv2

from panda_cable_grasp.perception.dataset import (
    PackedDLOImageDataset, rasterize_torch)
from dump_occdyn_ep import episode_frames

packed, scenario, seed, maps_npz, out = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
max_f = int(sys.argv[6]) if len(sys.argv) > 6 else 10**9

ds = PackedDLOImageDataset.__new__(PackedDLOImageDataset)
maps = np.load(maps_npz)["maps"]
X0, X1, Y0, Y1 = ds.X0, ds.X1, ds.Y0, ds.Y1
W, H = ds.W, ds.H
S = 3


def world2px(pts):
    px = (pts[:, 0] - X0) / (X1 - X0) * W
    py = (pts[:, 1] - Y0) / (Y1 - Y0) * H
    return np.stack([px, H - py], 1).astype(np.int32)


def draw_chain(canvas, pts, color):
    p = world2px(pts)
    for i in range(len(p) - 1):
        cv2.line(canvas, tuple(p[i]), tuple(p[i + 1]), color, 2)
    for q in p:
        cv2.circle(canvas, tuple(q), 3, color, -1)


d = np.load(sys.argv[4].replace("maps", "ep"), allow_pickle=True) \
    if False else None
frames = episode_frames(Path(packed), scenario, seed)
wr = None
i = 0
for f in frames:
    if i >= min(len(maps), max_f):
        break
    pts = torch.from_numpy(f["points"].astype(np.float32).reshape(-1, 3))[None]
    hand = torch.from_numpy(f["hand"].astype(np.float32))[None]
    chain = torch.from_numpy(f["pos14"].astype(np.float32))[None]
    _, tgt = rasterize_torch(pts, hand, chain,
                             ds.X0, ds.X1, ds.Y0, ds.Y1, ds.W, ds.H)
    gm = tgt[0, 0].numpy()                       # GT mask
    pm = 1 / (1 + np.exp(-maps[i][0]))           # pred prob
    pmb = (pm > 0.5).astype(np.float32)

    # RGB overlay: G=GT, R=pred -> overlap yellow
    ov = np.zeros((H, W, 3), np.uint8)
    ov[..., 1] = (gm * 255).astype(np.uint8)     # green = GT
    ov[..., 2] = (pm * 255).astype(np.uint8)     # red = pred prob
    # also faint blue = input occupancy for context
    img, _ = rasterize_torch(pts, hand, chain,
                             ds.X0, ds.X1, ds.Y0, ds.Y1, ds.W, ds.H)
    occ = img[0, 0].numpy()
    ov[..., 0] = np.clip(occ * 160, 0, 255).astype(np.uint8)
    ov = cv2.resize(ov, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
    draw_chain(ov, f["pos14"].astype(np.float64), (80, 255, 80))
    cv2.putText(ov, "G=GT mask  R=pred mask  B=cloud occ", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(ov, f"f{i}  inter={int((pmb*gm).sum())} "
                f"union={int((pmb+gm>0).sum())}",
                (10, H * S - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (220, 220, 220), 1)
    if wr is None:
        wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                             (W * S, H * S))
    wr.write(ov)
    i += 1
wr.release()
print(f"wrote {out} {i} frames")
