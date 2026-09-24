"""Clip of the ring-closure extractor failure: mask+skeleton+GT+pred
nodes for a frame window around f290 on shape 20283218."""
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import cv2
from skimage.morphology import skeletonize

X0, X1, Y0, Y1, W, H = -0.20, 1.15, -0.95, 0.85, 192, 256
maps = np.load(sys.argv[1])["maps"]          # (N,3,H,W)
dump = np.load(sys.argv[2])
gt, pred = dump["gt"], dump["pred"]
f0, f1, out = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
S = 3


def w2p(pts):
    return np.stack([(pts[:, 0] - X0) / (X1 - X0) * W,
                     (pts[:, 1] - Y0) / (Y1 - Y0) * H], 1)


vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                     (W * S, H * S))
for f in range(f0, f1):
    om = maps[f]
    pm = 1 / (1 + np.exp(-om[0])) > 0.5
    sk = skeletonize(pm)
    canvas = np.zeros((H * S, W * S, 3), np.uint8)
    canvas[pm.repeat(S, 0).repeat(S, 1)] = (0, 90, 0)
    canvas[sk.repeat(S, 0).repeat(S, 1)] = (0, 255, 255)
    g = (w2p(gt[f]) * S).astype(int)
    for i in range(13):
        cv2.line(canvas, tuple(g[i]), tuple(g[i + 1]), (0, 255, 0), 2)
    q = (w2p(pred[f]) * S).astype(int)
    for i in range(13):
        cv2.line(canvas, tuple(q[i]), tuple(q[i + 1]), (0, 0, 255), 2)
    for i, q_ in enumerate(q):
        cv2.circle(canvas, tuple(q_), 4, (255, 255, 255), -1)
        cv2.putText(canvas, str(i), tuple(q_ + 4), 0, 0.4,
                    (255, 255, 0), 1)
    canvas = np.ascontiguousarray(np.flipud(canvas))
    cv2.putText(canvas, f"f{f}", (10, 30), 0, 0.8, (0, 255, 255), 2)
    vw.write(canvas)
vw.release()
print("wrote", out)
