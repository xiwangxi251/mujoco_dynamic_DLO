"""Render predicted img maps (mask/s/z) + extracted chain to a video."""
import sys
import numpy as np
import cv2

maps_npz, npz, out = sys.argv[1], sys.argv[2], sys.argv[3]
maps = np.load(maps_npz)["maps"]
d = np.load(npz, allow_pickle=True)
X0, X1, Y0, Y1 = -0.20, 1.15, -0.95, 0.85
W, H = 192, 256

S = 3  # upscale factor
panel = (H * S, W * S)


def to_img(a, cmap=None, norm=None):
    a = a.astype(np.float32)
    if norm is None:
        a = (a - a.min()) / max(a.max() - a.min(), 1e-6)
    else:
        a = np.clip(a / norm, 0, 1)
    u8 = (a * 255).astype(np.uint8)
    if cmap is None:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(u8, cmap)


def world2px(pts):
    px = (pts[:, 0] - X0) / (X1 - X0) * W
    py = (pts[:, 1] - Y0) / (Y1 - Y0) * H
    return np.stack([px, H - py], 1).astype(np.int32)  # flip y for image


def draw_chain(canvas, pts, color):
    p = world2px(pts)
    for i in range(len(p) - 1):
        cv2.line(canvas, tuple(p[i]), tuple(p[i + 1]), color, 2)
    for q in p:
        cv2.circle(canvas, tuple(q), 3, color, -1)


fps = 15
vw = panel[1] * 4
vh = panel[0]
wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
n = len(maps)
for i in range(n):
    m = maps[i][:, ::-1]  # raster row ~ +y; flip to y-up display
    pm = 1 / (1 + np.exp(-m[0]))
    p0 = cv2.resize(to_img(pm, None, 1.0), (panel[1], panel[0]),
                    interpolation=cv2.INTER_NEAREST)
    smap = np.where(pm > 0.5, m[1], np.nan)
    smapn = np.nan_to_num(smap)
    p1 = cv2.resize(to_img(smapn, cv2.COLORMAP_HSV, 1.0), (panel[1], panel[0]),
                    interpolation=cv2.INTER_NEAREST)
    p2 = cv2.resize(to_img(m[2], cv2.COLORMAP_VIRIDIS), (panel[1], panel[0]),
                    interpolation=cv2.INTER_NEAREST)
    # chain overlay on mask
    p3 = cv2.resize(to_img(pm, None, 1.0), (panel[1], panel[0]),
                    interpolation=cv2.INTER_NEAREST)
    draw_chain(p3, d["gt"][i], (60, 200, 60))
    draw_chain(p3, d["pred"][i], (60, 60, 230))
    for k, (img, t) in enumerate(
        zip([p0, p1, p2, p3], ["pred mask", "pred s", "pred z",
                               "chain: green=GT red=pred"])):
        cv2.putText(img, t, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(img, f"f{i}", (10, vh - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
    wr.write(np.concatenate([p0, p1, p2, p3], axis=1))
wr.release()
print(f"wrote {out} {n} frames")
