"""Oracle-endpoint diagnostic: if endpoints were known perfectly, does the
pred-mask skeleton still yield wrong paths?

Per frame: skeletonize pred mask -> bridge comps -> largest component ->
snap GT node0/node13 onto skeleton -> LONGEST simple path between them
(bounded DFS) -> resample 14 nodes -> strict MPNE vs GT.

Outputs npz with oracle pred (keys gt/pred) so render_order_viz.py works.
"""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
import cv2
from test_skel_extract import bridge_components, bfs_farthest, NB


def bridge_components_ep(sk, max_gap=18):
    """Endpoint-only bridging: a cable break happens at a strand TIP, so
    only connect degree-1 endpoints of the smaller component to the
    nearest pixel of the other component."""
    for _ in range(4):
        lab = label(sk, connectivity=2)
        n = lab.max()
        if n <= 1:
            break
        best = None
        for a in range(1, n):
            ya, xa = np.where(lab == a)
            for b in range(a + 1, n + 1):
                yb, xb = np.where(lab == b)
                # smaller comp must attach via its endpoints
                if len(ya) <= len(yb):
                    ep_ys, ep_xs, t_ys, t_xs = ya, xa, yb, xb
                else:
                    ep_ys, ep_xs, t_ys, t_xs = yb, xb, ya, xa
                eps = [(y, x) for y, x in zip(ep_ys, ep_xs)
                       if sum(sk[y + dy, x + dx] for dy, dx in NB) == 1]
                if not eps:
                    continue  # ring comp: no tip to break from
                ey = np.array([e[0] for e in eps])
                ex = np.array([e[1] for e in eps])
                dd = np.hypot(ey[:, None] - t_ys[None, :],
                              ex[:, None] - t_xs[None, :])
                k = dd.argmin()
                if best is None or dd.flat[k] < best[0]:
                    ia, ib = np.unravel_index(k, dd.shape)
                    best = (dd.flat[k], ey[ia], ex[ia],
                            t_ys[ib], t_xs[ib])
        if best is None or best[0] > max_gap:
            break
        _, y0, x0, y1, x1 = best
        cv2.line(sk, (int(x0), int(y0)), (int(x1), int(y1)), 1, 1)
    return sk

X0, X1, Y0, Y1 = -0.20, 1.15, -0.95, 0.85
W, H, M = 192, 256, 14
DFS_CAP = 400_000


def build_graph(sk):
    ys, xs = np.where(sk)
    idx = {(y, x): i for i, (y, x) in enumerate(zip(ys, xs))}
    adj = [[] for _ in range(len(ys))]
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in NB:
            j = idx.get((y + dy, x + dx))
            if j is not None and j > i:
                w = float(np.hypot(dy, dx))
                adj[i].append((j, w))
                adj[j].append((i, w))
    # largest connected component
    seen, comp = set(), []
    for s0 in range(len(ys)):
        if s0 in seen:
            continue
        d = {s0}
        q = deque([s0])
        while q:
            u = q.popleft()
            for v, _ in adj[u]:
                if v not in d:
                    d.add(v)
                    q.append(v)
        if len(d) > len(comp):
            comp = list(d)
        seen |= d
    return xs, ys, adj, set(comp)


def w2px(pt):
    return ((pt[0] - X0) / (X1 - X0) * W,
            (pt[1] - Y0) / (Y1 - Y0) * H)


def snap(px, py, xs, ys, comp):
    best, bd = -1, 1e18
    for i in comp:
        d = (xs[i] - px) ** 2 + (ys[i] - py) ** 2
        if d < bd:
            bd, best = d, i
    return best, bd ** 0.5


def longest_path(adj, comp, a, b):
    """Longest simple path a->b on subgraph comp; bounded DFS.
    Returns (path, weight, expansions, capped)."""
    nodes = list(comp)
    sub = {v: k for k, v in enumerate(nodes)}
    ad = [[] for _ in nodes]
    for u in nodes:
        for v, w in adj[u]:
            if v in sub:
                ad[sub[u]].append((sub[v], w))
    sa, sb = sub[a], sub[b]
    best, best_w = None, -1.0
    visited = np.zeros(len(nodes), bool)
    stack = [(sa, iter(ad[sa]))]
    visited[sa] = True
    path = [sa]
    wt = [0.0]
    exp = 0
    capped = False
    while stack:
        u, it = stack[-1]
        nxt = None
        for v, w in it:
            exp += 1
            if not visited[v]:
                nxt = (v, w)
                break
        if exp > DFS_CAP:
            capped = True
            break
        if nxt is None:
            stack.pop()
            visited[u] = False
            path.pop()
            wt.pop()
            continue
        v, w = nxt
        visited[v] = True
        path.append(v)
        wt.append(wt[-1] + w)
        stack.append((v, iter(ad[v])))
        if v == sb and wt[-1] > best_w:
            best_w = wt[-1]
            best = list(path)
    if best is None:
        return None, 0.0, exp, capped
    return [nodes[p] for p in best], best_w, exp, capped


def resample(coords, zmap):
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    out = np.zeros((M, 3))
    for j in range(M):
        t = j / (M - 1) * L
        k = min(int(np.searchsorted(cum, t, "right")) - 1,
                len(seg) - 1)
        f = (t - cum[k]) / max(seg[k], 1e-9)
        xy = coords[k] * (1 - f) + coords[k + 1] * f
        px, py = int(round(xy[0])), int(round(xy[1]))
        out[j, 0] = (xy[0] + 0.5) / W * (X1 - X0) + X0
        out[j, 1] = (xy[1] + 0.5) / H * (Y1 - Y0) + Y0
        out[j, 2] = zmap[min(max(py, 0), H - 1), min(max(px, 0), W - 1)]
    return out, L / W * (X1 - X0)


def main():
    maps = np.load(sys.argv[1])["maps"]
    d = np.load(sys.argv[2])
    gt = d["gt"]
    out_npz = sys.argv[3]
    import os
    bridge = bridge_components_ep if os.environ.get(
        "BRIDGE") == "ep" else bridge_components
    print("bridge:", bridge.__name__)
    preds = np.full((len(gt), M, 3), np.nan)
    stats = []
    for f in range(len(gt)):
        pm = 1 / (1 + np.exp(-np.clip(maps[f][0], -30, 30))) > 0.5
        sk = bridge(skeletonize(pm).astype(np.uint8))
        xs, ys, adj, comp = build_graph(sk)
        e0, d0 = snap(*w2px(gt[f, 0]), xs, ys, comp)
        e1, d1 = snap(*w2px(gt[f, -1]), xs, ys, comp)
        path, plen, exp, capped = longest_path(adj, comp, e0, e1)
        if path is None or len(path) < M:
            stats.append((f, -1, -1, d0, d1, exp, capped, "nopath"))
            continue
        coords = np.array([[xs[p], ys[p]] for p in path], np.float64)
        p, Lm = resample(coords, maps[f][2])
        preds[f] = p
        stats.append((f, Lm, plen, d0, d1, exp, capped, "ok"))
        if f % 50 == 0:
            print(f, "L=%.2f" % Lm, "snap=%.1f/%.1fpx" % (d0, d1),
                  "exp=%d" % exp, "cap" if capped else "", flush=True)
    np.savez(out_npz, pred=preds, gt=gt)
    errs = np.full(len(gt), np.nan)
    for f in range(len(gt)):
        if not np.isnan(preds[f, 0, 0]):
            errs[f] = np.linalg.norm(preds[f] - gt[f], axis=1).mean() * 1000
    ok = ~np.isnan(errs)
    print(f"oracle-endpoint: n={ok.sum()}/{len(gt)} "
          f"strict={np.nanmean(errs):.1f}mm "
          f"median={np.nanmedian(errs):.1f}mm")
    bad = np.where(ok & (errs > 40))[0]
    print("frames err>40mm:", bad.tolist()[:60])
    nop = [s for s in stats if s[-1] == "nopath"]
    cap = [s for s in stats if s[-2] is True or s[6]]
    print("nopath:", len(nop), "capped:", len(cap))


if __name__ == "__main__":
    main()
