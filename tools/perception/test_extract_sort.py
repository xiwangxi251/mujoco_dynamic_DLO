"""Compare s-band centroid vs sort-by-s walk extraction on dumped maps."""
import sys
import numpy as np

npz, maps_npz = sys.argv[1], sys.argv[2]
d = np.load(npz, allow_pickle=True)
maps = np.load(maps_npz)["maps"]
gt = d["gt"]
X0, X1, Y0, Y1 = -0.20, 1.15, -0.95, 0.85
W, H = 192, 256
M = 14


def extract_band(m):
    pm = 1 / (1 + np.exp(-m[0])) > 0.5
    if pm.sum() < 2:
        pm = 1 / (1 + np.exp(-m[0])) > 0.1
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


def extract_sort(m):
    pm = 1 / (1 + np.exp(-m[0])) > 0.5
    if pm.sum() < 2:
        pm = 1 / (1 + np.exp(-m[0])) > 0.1
    s = m[1][pm]; z = m[2][pm]
    ys, xs = np.nonzero(pm)
    xw = (xs + 0.5) / W * (X1 - X0) + X0
    yw = (ys + 0.5) / H * (Y1 - Y0) + Y0
    o = np.argsort(s)
    xw, yw, z, s = xw[o], yw[o], z[o], s[o]
    # walk the s-sorted sequence, resample evenly by chord length
    pts = np.stack([xw, yw], 1)
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    if cum[-1] < 1e-9:
        return np.tile(pts[0], (M, 1)).astype(object).astype(float) \
            if False else np.repeat(pts[:1], M, 0)
    # median-filter xy along s order to kill single-pixel strand jumps
    k = 5
    xs_s = np.convolve(xw, np.ones(k) / k, "same")
    ys_s = np.convolve(yw, np.ones(k) / k, "same")
    pts_s = np.stack([xs_s, ys_s], 1)
    step_s = np.linalg.norm(np.diff(pts_s, axis=0), axis=1)
    cum_s = np.concatenate([[0.0], np.cumsum(step_s)])
    cum_s = cum_s / max(cum_s[-1], 1e-9)
    out = np.zeros((M, 3))
    for j in range(M):
        t = j / (M - 1)
        i = np.searchsorted(cum_s, t)
        i = min(max(i, 1), len(pts_s) - 1)
        f = (t - cum_s[i - 1]) / max(cum_s[i] - cum_s[i - 1], 1e-9)
        out[j, 0] = xs_s[i - 1] + f * (xs_s[i] - xs_s[i - 1])
        out[j, 1] = ys_s[i - 1] + f * (ys_s[i] - ys_s[i - 1])
        out[j, 2] = z[i - 1] + f * (z[i] - z[i - 1])
    return out


e_band, e_sort = [], []
for i in range(len(gt)):
    pb = extract_band(maps[i])
    ps = extract_sort(maps[i])
    e_band.append(np.linalg.norm(pb - gt[i], axis=1).mean() * 1000)
    e_sort.append(np.linalg.norm(ps - gt[i], axis=1).mean() * 1000)
e_band = np.asarray(e_band); e_sort = np.asarray(e_sort)
print(f"band: {e_band.mean():.1f} mm   sort-walk: {e_sort.mean():.1f} mm")
worst = np.argsort(e_band)[-6:]
for i in worst:
    print(f"  f{i}: band {e_band[i]:.0f}  sort {e_sort[i]:.0f}")
