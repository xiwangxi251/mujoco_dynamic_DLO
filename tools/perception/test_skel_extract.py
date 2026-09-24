"""Skeleton-walk extractor: pred mask -> skeleton -> longest geodesic path
-> 14 nodes by arc length. Zero-training baseline vs s-binning."""
import sys
from collections import deque

import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label

X0, X1, Y0, Y1 = -0.20, 1.15, -0.95, 0.85
W, H, M = 192, 256, 14

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def bfs_farthest(adj, start):
    dist, par = {start: 0}, {start: -1}
    q = deque([start])
    far = start
    while q:
        u = q.popleft()
        if dist[u] > dist[far]:
            far = u
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                par[v] = u
                q.append(v)
    path, u = [], far
    while u != -1:
        path.append(u)
        u = par[u]
    return far, path[::-1], dist


def endpoint_s(smap, pm, x, y, r=4):
    """mean predicted s in an r-radius patch around (x,y), masked."""
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    v = smap[y0:y1, x0:x1][pm[y0:y1, x0:x1]]
    return v.mean() if len(v) else np.nan


def bridge_components(sk, max_gap=18):
    """Connect nearest pixels of separate skeleton components with a
    straight 1px line when gap <= max_gap px (cable is one object;
    gaps = occlusion the model failed to inpaint)."""
    for _ in range(4):  # at most a few breaks
        lab = label(sk, connectivity=2)
        n = lab.max()
        if n <= 1:
            break
        best = None
        for a in range(1, n):
            ya, xa = np.where(lab == a)
            for b in range(a + 1, n + 1):
                yb, xb = np.where(lab == b)
                # coarse: pairwise min dist
                dd = np.hypot(ya[:, None] - yb[None, :],
                              xa[:, None] - xb[None, :])
                k = dd.argmin()
                if best is None or dd.flat[k] < best[0]:
                    ia, ib = np.unravel_index(k, dd.shape)
                    best = (dd.flat[k], ya[ia], xa[ia], yb[ib], xb[ib])
        if best is None or best[0] > max_gap:
            break
        _, y0, x0, y1, x1 = best
        cv2.line(sk, (x0, y0), (x1, y1), 1, 1)
    return sk


def extract_skel(pm, zmap, smap=None):
    sk = skeletonize(pm).astype(np.uint8)
    sk = bridge_components(sk)
    ys, xs = np.where(sk)
    if len(ys) < M:
        return None
    idx = {(y, x): i for i, (y, x) in enumerate(zip(ys, xs))}
    adj = [[] for _ in range(len(ys))]
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in NB:
            j = idx.get((y + dy, x + dx))
            if j is not None and j > i:
                adj[i].append(j)
                adj[j].append(i)
    # largest connected component
    seen, comp = set(), []
    for s0 in range(len(ys)):
        if s0 in seen:
            continue
        _, _, dist = bfs_farthest(adj, s0)
        nodes = list(dist)
        if len(nodes) > len(comp):
            comp = nodes
        seen.update(nodes)
    sub = {v: k for k, v in enumerate(comp)}
    adj2 = [[sub[v] for v in adj[u] if v in sub] for u in comp]
    # graph diameter = main chain
    a, _, _ = bfs_farthest(adj2, 0)
    b, path, _ = bfs_farthest(adj2, a)
    orig = np.array(comp)          # path indices -> original pixel idx
    coords = np.array([[xs[orig[p]], ys[orig[p]]] for p in path],
                      np.float64)  # (x,y) px
    # orient: endpoint with lower predicted s -> node 0
    if smap is not None:
        sa = endpoint_s(smap, pm, *coords[0].astype(int))
        sb = endpoint_s(smap, pm, *coords[-1].astype(int))
        if not np.isnan(sa) and not np.isnan(sb) and sb < sa:
            coords = coords[::-1]
    # resample M points by arc length
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    out = np.zeros((M, 3))
    for j in range(M):
        t = j / (M - 1) * L
        k = min(int(np.searchsorted(cum, t, "right")) - 1, len(seg) - 1)
        f = (t - cum[k]) / max(seg[k], 1e-9)
        xy = coords[k] * (1 - f) + coords[k + 1] * f
        px, py = int(round(xy[0])), int(round(xy[1]))
        out[j, 0] = (xy[0] + 0.5) / W * (X1 - X0) + X0
        out[j, 1] = (xy[1] + 0.5) / H * (Y1 - Y0) + Y0
        out[j, 2] = zmap[min(max(py, 0), H - 1), min(max(px, 0), W - 1)]
    return out


def main():
    maps = np.load(sys.argv[1])["maps"]      # (N,3,H,W) pred maps
    npz = np.load(sys.argv[2])               # dump with gt
    gt = npz["gt"]
    errs_s, errs_tol, flips = [], [], 0
    preds = []
    prev = None
    for f in range(len(gt)):
        pm = 1 / (1 + np.exp(-maps[f][0])) > 0.5
        p = extract_skel(pm, maps[f][2], smap=maps[f][1])
        if p is None:
            p = np.full((M, 3), np.nan)
        elif prev is not None:
            # temporal anchor: full-chain match vs previous 14 nodes
            ef = np.linalg.norm(p - prev, axis=1).mean()
            eb = np.linalg.norm(p[::-1] - prev, axis=1).mean()
            if eb < ef:
                p = p[::-1]
        if not np.isnan(p[0, 0]):
            prev = p.copy()
        preds.append(p)
        e = np.linalg.norm(p - gt[f], axis=1).mean() * 1000
        er = np.linalg.norm(p[::-1] - gt[f], axis=1).mean() * 1000
        errs_s.append(min(e, er) if np.isnan(e) else e)
        errs_tol.append(er)
        flips += er < e

    preds = np.array(preds)
    np.savez("/tmp/img_ep10_skel.npz", pred=preds)
    valid = ~np.isnan(errs_s)
    print(f"skeleton-walk: n={valid.sum()} "
          f"strict={np.nanmean(errs_s):.1f}mm flips={flips}/{len(gt)}")


if __name__ == "__main__":
    main()
