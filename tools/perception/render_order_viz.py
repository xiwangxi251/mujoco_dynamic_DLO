"""Visualise predicted ORDERING: pred s-map + indexed chains GT vs pred."""
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

packed, scenario, seed, maps_npz, npz, out = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5],
    sys.argv[6])
max_f = int(sys.argv[7]) if len(sys.argv) > 7 else 10**9

ds = PackedDLOImageDataset.__new__(PackedDLOImageDataset)
maps = np.load(maps_npz)["maps"]
d = np.load(npz, allow_pickle=True)
X0, X1, Y0, Y1 = ds.X0, ds.X1, ds.Y0, ds.Y1
W, H = ds.W, ds.H
S = 3
M = 14


def w2p(pts):
    px = (pts[:, 0] - X0) / (X1 - X0) * W
    py = (pts[:, 1] - Y0) / (Y1 - Y0) * H
    return np.stack([px, H - py], 1).astype(np.int32)


def idx_color(j):
    """node index -> rainbow BGR (0=red .. 13=violet)."""
    hsv = np.uint8([[[int(j / (M - 1) * 160), 255, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_indexed(canvas, pts):
    p = w2p(pts) * S  # canvas is upscaled Sx
    for i in range(len(p) - 1):
        cv2.line(canvas, tuple(p[i]), tuple(p[i + 1]),
                 idx_color(i), 2)
    for i, q in enumerate(p):
        cv2.circle(canvas, tuple(q), 4, idx_color(i), -1)
        cv2.putText(canvas, str(i), (q[0] + 6, q[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def crop_all(imgs, box):
    """imgs: list of (H*S, W*S) canvases; box in raster px (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = [int(round(v * S)) for v in box]
    h, w = imgs[0].shape[:2]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w, x1); y1 = min(h, y1)
    return [im[y0:y1, x0:x1] for im in imgs]


wr = None
i = 0
for f in episode_frames(Path(packed), scenario, seed):
    if i >= min(len(maps), len(d["gt"]), max_f):
        break
    raw = f["points"].astype(np.float32).reshape(-1, 3)
    raw = raw[np.abs(raw).sum(1) > 1e-6]
    pts = torch.from_numpy(f["points"].astype(np.float32).reshape(-1, 3))[None]
    hand = torch.from_numpy(f["hand"].astype(np.float32))[None]
    chain = torch.from_numpy(f["pos14"].astype(np.float32))[None]
    img, _ = rasterize_torch(pts, hand, chain,
                             ds.X0, ds.X1, ds.Y0, ds.Y1, ds.W, ds.H)
    occ = np.flipud(img[0, 0].numpy())

    # left: pred s map (rainbow where mask) on occ background
    pm = np.flipud(1 / (1 + np.exp(-maps[i][0])))
    smap = np.flipud(maps[i][1])
    left = np.zeros((H, W, 3), np.uint8)
    left[..., 0] = np.clip(occ * 160, 0, 255).astype(np.uint8)
    hue = np.zeros((H, W, 3), np.uint8)
    hue[..., 0] = np.clip(smap * 160, 0, 160).astype(np.uint8)  # hue
    hue[..., 1] = 255
    hue[..., 2] = (pm > 0.5).astype(np.uint8) * 255
    left = np.maximum(left, cv2.cvtColor(hue, cv2.COLOR_HSV2BGR))
    left = cv2.resize(left, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
    cv2.putText(left, "PRED s map", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # mid: GT chain indexed, on cloud
    mid = np.zeros((H, W, 3), np.uint8)
    mid[..., 0] = np.clip(occ * 160, 0, 255).astype(np.uint8)
    mid = cv2.resize(mid, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
    draw_indexed(mid, np.asarray(f["pos14"], np.float64))
    cv2.putText(mid, "GT chain idx 0(red)->13(violet)", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # right: pred chain indexed, on pred mask (visible) + faint occ
    right = np.zeros((H, W, 3), np.uint8)
    right[..., 0] = np.clip(occ * 120, 0, 255).astype(np.uint8)
    mvis = (pm > 0.5)
    right[..., 1] = mvis.astype(np.uint8) * 200
    right[..., 2] = mvis.astype(np.uint8) * 60
    right = cv2.resize(right, (W * S, H * S),
                       interpolation=cv2.INTER_NEAREST)
    pred_i = np.asarray(d["pred"][i], np.float64)
    pred_ok = np.isfinite(pred_i).all()
    if pred_ok:
        draw_indexed(right, pred_i)
    else:
        cv2.putText(right, "PRED: no path", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)
    cv2.putText(right, "PRED chain idx", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if pred_ok:
        e = np.linalg.norm(pred_i - np.asarray(f["pos14"]),
                           axis=1).mean() * 1000
        er = np.linalg.norm(pred_i[::-1] - np.asarray(f["pos14"]),
                            axis=1).mean() * 1000
    else:
        e = er = np.nan
    tag = " FLIP" if np.isfinite(er) and er < e else ""

    # crop to cable bbox (cloud ∪ GT ∪ pred) + 15% pad, all panels same box
    parts = [raw[:, :2], np.asarray(f["pos14"])[:, :2]]
    if pred_ok:
        parts.append(pred_i[:, :2])
    allw = np.concatenate(parts)
    allw = allw[np.isfinite(allw).all(1)]
    pc = w2p(np.concatenate([allw, np.zeros((len(allw), 1))], 1))
    x0, x1 = pc[:, 0].min() - 20, pc[:, 0].max() + 20
    y0, y1 = pc[:, 1].min() - 20, pc[:, 1].max() + 20
    pad = int(0.15 * max(x1 - x0, y1 - y0))
    box = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    left, mid, right = crop_all([left, mid, right], box)
    PO = 640
    left = cv2.resize(left, (PO, PO), interpolation=cv2.INTER_NEAREST)
    mid = cv2.resize(mid, (PO, PO), interpolation=cv2.INTER_NEAREST)
    right = cv2.resize(right, (PO, PO), interpolation=cv2.INTER_NEAREST)

    for im, tt in zip((left, mid, right),
                      ("pred s", "GT chain idx", "PRED chain idx")):
        cv2.putText(im, tt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(im, f"f{i} err={e:.0f}mm{tag}", (10, PO - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (80, 80, 255) if tag else (220, 220, 220), 2)
    frame = np.concatenate([left, mid, right], axis=1)
    if wr is None:
        wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                             frame.shape[:2][::-1])
    wr.write(frame)
    i += 1
wr.release()
print(f"wrote {out} {i} frames")
