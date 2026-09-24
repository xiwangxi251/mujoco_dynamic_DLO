"""Compare occ vs pred skeleton+bridge: bridges drawn RED, skeleton cyan,
GT chain green underneath."""
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label

from panda_cable_grasp.perception.dataset import (
    PackedDLOImageDataset, rasterize_torch)
from dump_occdyn_ep import episode_frames
from test_skel_extract import bridge_components

packed, scenario, seed, maps_npz, out = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])

ds = PackedDLOImageDataset.__new__(PackedDLOImageDataset)
maps = np.load(maps_npz)["maps"]
X0, X1, Y0, Y1, W, H = ds.X0, ds.X1, ds.Y0, ds.Y1, ds.W, ds.H
S = 3


def w2p(pts):
    px = (pts[:, 0] - X0) / (X1 - X0) * W
    py = (pts[:, 1] - Y0) / (Y1 - Y0) * H
    return np.stack([px, H - py], 1).astype(np.int32)


def skel_bridge(pm):
    sk0 = skeletonize(pm).astype(np.uint8)
    sk1 = bridge_components(sk0.copy())
    bridges = (sk1 > 0) & (sk0 == 0)      # pixels added by bridging
    return sk0, sk1, bridges


wr = None
i = 0
for f in episode_frames(Path(packed), scenario, seed):
    if i >= len(maps):
        break
    raw = f["points"].astype(np.float32).reshape(-1, 3)
    raw = raw[np.abs(raw).sum(1) > 1e-6]
    pts = torch.from_numpy(f["points"].astype(np.float32).reshape(-1, 3))[None]
    hand = torch.from_numpy(f["hand"].astype(np.float32))[None]
    chain = torch.from_numpy(f["pos14"].astype(np.float32))[None]
    img, _ = rasterize_torch(pts, hand, chain, X0, X1, Y0, Y1, W, H)
    occ = np.flipud(img[0, 0].numpy())
    gtpx = w2p(np.asarray(f["pos14"])) * S

    panels = []
    for tag, m in (("occ(raw cloud)", occ > 0),
                   ("pred(model)", np.flipud(
                       1 / (1 + np.exp(-maps[i][0]))) > 0.5)):
        sk0, sk1, br = skel_bridge(m)
        nc = label(sk0, connectivity=2).max()
        cv = np.zeros((H, W, 3), np.uint8)
        cv[..., 1] = m.astype(np.uint8) * 110      # mask faint green
        cv[sk1 > 0] = (255, 255, 0)                # skeleton cyan
        cv[br] = (0, 0, 255)                       # bridges red
        cv = cv2.resize(cv, (W * S, H * S),
                        interpolation=cv2.INTER_NEAREST)
        cv2.polylines(cv, [gtpx], False, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append((cv, f"{tag} comp={nc}"))

    # zoom to content bbox
    allw = np.concatenate([raw[:, :2], np.asarray(f["pos14"])[:, :2]])
    pc = w2p(np.concatenate([allw, np.zeros((len(allw), 1))], 1))
    x0, x1 = pc[:, 0].min() - 15, pc[:, 0].max() + 15
    y0, y1 = pc[:, 1].min() - 15, pc[:, 1].max() + 15
    pad = int(0.2 * max(x1 - x0, y1 - y0))
    bx = (max(0, (x0 - pad) * S), max(0, (y0 - pad) * S),
          min(W * S, (x1 + pad) * S), min(H * S, (y1 + pad) * S))
    PO = 720
    out_panels = []
    for cv, tt in panels:
        c = cv[bx[1]:bx[3], bx[0]:bx[2]]
        c = cv2.resize(c, (PO, PO), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, tt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(c, f"f{i}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)
        cv2.putText(c, "cyan=skel red=bridge green=GT", (10, PO - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        out_panels.append(c)
    frame = np.concatenate(out_panels, axis=1)
    if wr is None:
        wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                             frame.shape[:2][::-1])
    wr.write(frame)
    i += 1
wr.release()
print(f"wrote {out} {i} frames")
