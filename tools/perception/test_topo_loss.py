"""Unit test: clDice + s-order losses behave as intended.

Synthetic GT: open arc. Compare loss for:
  A identical        B 2px gap        C spurious bridge (cycle)
  D 3px shift        E extra phantom strand
Expect: B,C,E >> A,D under clDice while Dice barely moves.
s-order: correct order ~0, reversed >> 0.
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import cv2
from train_img import cldice_loss, s_order_loss
from panda_cable_grasp.perception.dataset import PackedDLOImageDataset

W, H = 192, 256
ds = PackedDLOImageDataset.__new__(PackedDLOImageDataset)


def arc_mask(gap=None, bridge=None, shift=0, extra=False):
    m = np.zeros((H, W), np.uint8)
    pts = []
    for t in np.linspace(0, np.pi, 120):
        x = int(60 + 60 * np.cos(t)) + shift
        y = int(140 - 60 * np.sin(t))
        pts.append((x, y))
    cv2.polylines(m, [np.array(pts)], False, 1, 1)
    if gap is not None:
        m[gap:gap + 2, 60] = 0
    if bridge is not None:
        cv2.line(m, (bridge[0], bridge[1]), (bridge[2], bridge[3]), 1, 1)
    if extra:
        pts2 = [(x + 12, y - 8) for x, y in pts[10:70]]
        cv2.polylines(m, [np.array(pts2)], False, 1, 1)
    return torch.from_numpy(m[None, None].astype(np.float32))


def dice(pm, gt):
    return 1 - 2 * (pm * gt).sum() / (pm.sum() + gt.sum() + 1)


gt = arc_mask()
# B: real 2px break at the arc apex (60,80); C: chord across the arc
# legs making a closed loop; F: 4px break
m_b = gt.clone(); m_b[0, 0, 78:82, 57:64] = 0
m_f = gt.clone(); m_f[0, 0, 76:84, 56:66] = 0
m_c = gt.clone()
cv2.line(m_c[0, 0].numpy(), (15, 128), (105, 128), 1, 1)  # closes a loop
cases = {
    "A identical": gt.clone(),
    "B 2px break@apex": m_b,
    "F 4px break@apex": m_f,
    "C chord->loop": m_c,
    "D shift+3px": arc_mask(shift=3),
    "E phantom strand": arc_mask(extra=True),
}
for k, pm in cases.items():
    print(f"{k:18s} dice={float(dice(pm, gt)):.4f} "
          f"cldice={float(cldice_loss(pm, gt)):.4f}")

# s-order: GT chain along the arc, s 0->1 left to right
M = 14
ts = np.linspace(0, np.pi, M)
node = np.stack([60 + 60 * np.cos(ts), 140 - 60 * np.sin(ts)], 1)
node_w = np.concatenate(
    [node[:, :1] / W * (ds.X1 - ds.X0) + ds.X0,
     node[:, 1:2] / H * (ds.Y1 - ds.Y0) + ds.Y0,
     np.zeros((M, 1))], 1)[None].astype(np.float32)
s_good = torch.zeros(1, H, W)
dense = np.stack([60 + 60 * np.cos(np.linspace(0, np.pi, 300)),
                  140 - 60 * np.sin(np.linspace(0, np.pi, 300))], 1)
for k, (x, y) in enumerate(dense):
    s_good[0, int(y), int(x)] = k / (len(dense) - 1)
s_good = torch.nn.functional.max_pool2d(
    s_good[None], 3, 1, 1)[0]          # thicken 1px for stable sampling
s_flip = torch.where(s_good > 0, 1 - s_good, s_good)
lo = s_order_loss(s_good, torch.from_numpy(node_w), ds)
lf = s_order_loss(s_flip, torch.from_numpy(node_w), ds)
print(f"s-order: correct={float(lo):.4f}  flipped={float(lf):.4f}")
