"""Greedy s-gradient tracing extractor vs band centroid on dumped maps."""
import sys
import numpy as np

npz, maps_npz = sys.argv[1], sys.argv[2]
d = np.load(npz, allow_pickle=True)
maps = np.load(maps_npz)["maps"]
gt = d["gt"]
X0, X1, Y0, Y1 = -0.20, 1.15, -0.95, 0.85
W, H = 192, 256
M = 14


def band_extract(m):
    pm = 1 / (1 + np.exp(-m[0])) > 0.5
    s = m[1][pm]; z = m[2][pm]
    ys, xs = np.nonzero(pm)
    xw = (xs + 0.5) / W * (X1 - X0) + X0
    yw = (ys + 0.5) / H * (Y1 - Y0) + Y0
    out = np.zeros((M, 3))
    for j in range(M):
        t = j / (M - 1)
        band = np.abs(s - t) < 0.5 / (M - 1)
        if band.sum() == 0:
            k = np.argmin(np.abs(s - t))
            out[j] = xw[k], yw[k], z[k]
        else:
            out[j] = xw[band].mean(), yw[band].mean(), z[band].mean()
    return out


def trace_extract(m, step_px=3):
    """Greedy walk: start at min-s pixel; repeatedly move to the unvisited
    mask pixel within `step_px` whose s is closest to s+ds, preferring
    continuing direction."""
    pm = 1 / (1 + np.exp(-m[0])) > 0.5
    ys, xs = np.nonzero(pm)
    s = m[1][pm]; z = m[2][pm]
    xw = (xs + 0.5) / W * (X1 - X0) + X0
    yw = (ys + 0.5) / H * (Y1 - Y0) + Y0
    n = len(s)
    if n < 2:
        return np.repeat(np.stack([xw[:1].mean(), yw[:1].mean(),
                                   z[:1].mean()])[None], M, 0)
    px = np.stack([xw, yw], 1)
    visited = np.zeros(n, bool)
    cur = int(np.argmin(s))
    path = [cur]
    visited[cur] = True
    # typical ds per pixel-step ~ total s range / typical path length
    ds = 1.0 / max(n * 0.7, 2)
    for _ in range(n - 1):
        d2 = ((px - px[cur]) ** 2).sum(1)
        near = (d2 < (step_px * (X1 - X0) / W) ** 2) & ~visited
        if near.sum() == 0:
            # jump to nearest unvisited pixel
            d2[visited] = np.inf
            cur = int(np.argmin(d2))
        else:
            tgt_s = s[cur] + ds
            cand = np.flatnonzero(near)
            cur = int(cand[np.argmin(np.abs(s[cand] - tgt_s))])
        visited[cur] = True
        path.append(cur)
    path = np.asarray(path)
    pts = np.stack([xw[path], yw[path], z[path]], 1)
    step = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    cum /= max(cum[-1], 1e-9)
    out = np.zeros((M, 3))
    for j in range(M):
        t = j / (M - 1)
        i = min(max(np.searchsorted(cum, t), 1), len(pts) - 1)
        f = (t - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
        out[j] = pts[i - 1] + f * (pts[i] - pts[i - 1])
    return out


e_b, e_w = [], []
for i in range(len(gt)):
    pb = band_extract(maps[i])
    pw = trace_extract(maps[i])
    e_b.append(np.linalg.norm(pb - gt[i], axis=1).mean() * 1000)
    e_w.append(np.linalg.norm(pw - gt[i], axis=1).mean() * 1000)
e_b = np.asarray(e_b); e_w = np.asarray(e_w)
print(f"band {e_b.mean():.1f}   trace {e_w.mean():.1f}")
worst = np.argsort(e_b)[-8:]
for i in worst:
    print(f"  f{i}: band {e_b[i]:.0f}  trace {e_w[i]:.0f}")
print("worse-by-trace frames:", int((e_w - e_b > 5).sum()))
