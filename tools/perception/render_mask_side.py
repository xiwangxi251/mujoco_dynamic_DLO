"""Side-by-side GT mask vs pred mask video."""
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


def draw_chain(canvas, pts, color, r=4):
    p = world2px(pts)
    for i in range(len(p) - 1):
        cv2.line(canvas, tuple(p[i]), tuple(p[i + 1]), color, 2)
    for q in p:
        cv2.circle(canvas, tuple(q), r, color, -1)


def base_img(occ):
    """faint blue occupancy background."""
    b = np.zeros((H, W, 3), np.uint8)
    b[..., 0] = np.clip(occ * 200, 0, 255).astype(np.uint8)
    return cv2.resize(b, (W * S, H * S), interpolation=cv2.INTER_NEAREST)


wr = None
i = 0
for f in episode_frames(Path(packed), scenario, seed):
    if i >= min(len(maps), max_f):
        break
    raw = f["points"].astype(np.float32).reshape(-1, 3)
    raw = raw[np.abs(raw).sum(1) > 1e-6]
    pts = torch.from_numpy(f["points"].astype(np.float32).reshape(-1, 3))[None]
    hand = torch.from_numpy(f["hand"].astype(np.float32))[None]
    chain = torch.from_numpy(f["pos14"].astype(np.float32))[None]
    img, tgt = rasterize_torch(pts, hand, chain,
                             ds.X0, ds.X1, ds.Y0, ds.Y1, ds.W, ds.H)
    # raster rows are iy ~ +y (row 0 = y_min at TOP when drawn);
    # flip vertically so all panels use y-up like the world plots
    occ = np.flipud(img[0, 0].numpy())
    gm = np.flipud(tgt[0, 0].numpy())
    pm = np.flipud(1 / (1 + np.exp(-maps[i][0])))

    # left: raw cloud XY scatter + GT chain
    left = np.zeros((H * S, W * S, 3), np.uint8)
    pc = world2px(raw)
    ok = (pc[:, 0] >= 0) & (pc[:, 0] < W) & (pc[:, 1] >= 0) & (pc[:, 1] < H)
    left_pix = np.zeros((H, W), np.uint8)
    left_pix[pc[ok, 1], pc[ok, 0]] = 255
    left[..., 2] = cv2.resize(left_pix, (W * S, H * S),
                              interpolation=cv2.INTER_NEAREST)
    draw_chain(left, f["pos14"].astype(np.float64), (255, 255, 0))
    cv2.putText(left, "cloud XY + GT chain", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    mid = base_img(occ)
    gm3 = cv2.resize(
        cv2.cvtColor((gm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        (W * S, H * S), interpolation=cv2.INTER_NEAREST)
    mid[..., 1] = np.maximum(mid[..., 1], gm3[..., 1])
    draw_chain(mid, f["pos14"].astype(np.float64), (255, 255, 0))
    cv2.putText(mid, "GT mask + chain", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    right = base_img(occ)
    pm3 = cv2.resize(
        cv2.cvtColor((np.clip(pm, 0, 1) * 255).astype(np.uint8),
                     cv2.COLOR_GRAY2BGR),
        (W * S, H * S), interpolation=cv2.INTER_NEAREST)
    right[..., 2] = np.maximum(right[..., 2], pm3[..., 2])
    draw_chain(right, f["pos14"].astype(np.float64), (255, 255, 0))
    cv2.putText(right, "PRED mask", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    iou = ((pm > 0.5) & (gm > 0.5)).sum() / max(((pm > 0.5) | (gm > 0.5)).sum(), 1)
    for imgc in (left, mid, right):
        cv2.putText(imgc, f"f{i}  IoU={iou:.2f}", (10, H * S - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    frame = np.concatenate([left, mid, right], axis=1)
    if wr is None:
        wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                             frame.shape[:2][::-1])
    wr.write(frame)
    i += 1
wr.release()
print(f"wrote {out} {i} frames")
