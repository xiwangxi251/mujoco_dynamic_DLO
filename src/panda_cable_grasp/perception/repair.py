"""Inference-time chord-shortcut repair for predicted DLO chains.

Failure mode addressed: the regressor occasionally connects two nodes
with a straight chord through the interior of a loop, leaving a visible
cloud arc uncovered. Neither node positions nor cloud coverage alone
flag it — the segment *between* nodes floats in empty space.

Repair: detect unsupported segments (samples along the segment far from
every cloud point), collect the orphaned cloud points (far from the
whole chain), and re-route the node span through them via shortest path
on a radius-limited kNN graph — the geometric analogue of tracing the
cable through the cloud. Endpoint nodes are never moved, so endpoint
identity cannot be introduced by the repair.

Pure numpy; cost is a few hundred microseconds per frame.
"""

from __future__ import annotations

import heapq

import numpy as np


def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,3) points to segment a-b -> (N,) distances."""
    ab = b - a
    t = np.clip(
        ((p - a) * ab).sum(-1) / max(float((ab * ab).sum()), 1e-12),
        0.0, 1.0,
    )
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _resample_path(path: np.ndarray, n: int) -> np.ndarray:
    """Uniform-by-arclength resample of a polyline to n points."""
    seg = path[1:] - path[:-1]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
    total = max(cum[-1], 1e-12)
    tgt = np.linspace(0.0, total, n)
    out = np.empty((n, 3))
    j = 0
    for i, s in enumerate(tgt):
        while j < len(cum) - 2 and cum[j + 1] < s:
            j += 1
        t = 0.0 if cum[j + 1] - cum[j] < 1e-12 else (
            (s - cum[j]) / (cum[j + 1] - cum[j])
        )
        out[i] = path[j] + t * (path[j + 1] - path[j])
    return out


def _dijkstra(
    verts: np.ndarray, edges: list[list[tuple[int, float]]],
    src: int, dst: int,
) -> list[int] | None:
    dist = np.full(len(verts), np.inf)
    prev = np.full(len(verts), -1, dtype=np.int64)
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist[u]:
            continue
        for v, w in edges[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[dst]):
        return None
    path = [dst]
    while path[-1] != src:
        path.append(int(prev[path[-1]]))
    return path[::-1]


def repair_shortcuts(
    pred: np.ndarray,
    cloud: np.ndarray,
    support_thresh: float = 0.025,
    min_support: float = 0.5,
    knn_k: int = 10,
    radius_mult: float = 3.0,
    radius_max: float = 0.06,
    corridor: float = 0.10,
    samples_per_seg: int = 12,
) -> tuple[np.ndarray, dict]:
    """Re-route unsupported chain segments through orphaned cloud points.

    pred  (M,3) metres, cloud (N,3) metres. Returns (fixed, info) where
    info reports how many segments were replaced.
    """
    pred = np.asarray(pred, dtype=np.float64)
    cloud = np.asarray(cloud, dtype=np.float64)
    M = len(pred)
    info = {"bad_segs": 0, "runs_fixed": 0, "runs_failed": 0,
            "runs_skipped": 0}
    if M < 3 or len(cloud) < 8:
        return pred, info

    # 1) per-segment cloud support: fraction of samples near any point
    a, b = pred[:-1], pred[1:]
    ts = np.linspace(0.0, 1.0, samples_per_seg)
    samp = a[:, None] + ts[None, :, None] * (b - a)[:, None]  # (S,K,3)
    d = np.linalg.norm(
        samp[:, :, None, :] - cloud[None, None, :, :], axis=-1
    ).min(-1)                                                    # (S,K)
    support = (d < support_thresh).mean(1)                       # (S,)
    bad = support < min_support
    info["bad_segs"] = int(bad.sum())
    if not bad.any():
        return pred, info

    # 2) orphan cloud points: far from every chain segment
    dseg = np.stack(
        [_point_seg_dist(cloud, a[i], b[i]) for i in range(M - 1)], axis=1
    )
    orphan = dseg.min(1) > support_thresh
    orph = cloud[orphan]
    if len(orph) < 3:
        return pred, info

    # 3) radius-limited kNN graph over orphans; radius adapts to local
    #    point spacing so edges cannot bridge across to a wrong strand
    nn = np.linalg.norm(
        orph[:, None, :] - orph[None, :, :], axis=-1
    )
    np.fill_diagonal(nn, np.inf)
    med_nn = float(np.median(nn.min(1)))
    radius = min(radius_mult * max(med_nn, 1e-4), radius_max)

    # maximal runs of bad segments: segs lo..hi -> replace nodes lo+1..hi
    fixed = pred.copy()
    s = 0
    S = M - 1
    while s < S:
        if not bad[s]:
            s += 1
            continue
        e = s
        while e + 1 < S and bad[e + 1]:
            e += 1
        # run covers segments s..e; anchors = node s and node e+1.
        # Corridor gate: only orphans lying near the bad segments can be
        # the missing arc. If none exist, the low support means the
        # segment passes through an *occluded* region — rerouting it to
        # a distant strand would destroy a correct prediction.
        cor = np.zeros(len(orph), dtype=bool)
        for i in range(s, e + 1):
            cor |= _point_seg_dist(orph, fixed[i], fixed[i + 1]) < corridor
        near_idx = np.where(cor)[0]
        if len(near_idx) < 3:
            info["runs_skipped"] += 1
            s = e + 1
            continue
        n_rep = e - s  # interior nodes s+1..e  => count e-s
        sub = orph[near_idx]
        src, dst = len(sub), len(sub) + 1
        verts = np.vstack([sub, fixed[s], fixed[e + 1]])
        edges: list[list[tuple[int, float]]] = [
            [] for _ in range(len(verts))
        ]
        dv = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=-1)
        order = np.argsort(dv, axis=1)
        # orphan<->orphan edges respect the tight radius (no bridging to
        # a neighbouring strand); anchors attach to their k nearest
        # orphans unconditionally — they are expected to sit off-cloud
        for u in range(len(sub)):
            for v in order[u, : knn_k + 1]:
                if u == v or v >= len(sub) or dv[u, v] > radius:
                    continue
                edges[u].append((int(v), float(dv[u, v])))
        for u in (src, dst):
            for v in order[u, : knn_k]:
                if v >= len(sub):
                    continue
                edges[u].append((int(v), float(dv[u, v])))
                edges[v].append((int(u), float(dv[u, v])))
        path_idx = _dijkstra(verts, edges, src, dst)
        if path_idx is None:
            info["runs_failed"] += 1
        else:
            path = verts[np.asarray(path_idx)]
            if len(path) >= 2:
                fixed[s + 1:e + 1] = _resample_path(path, n_rep + 2)[1:-1]
                info["runs_fixed"] += 1
        s = e + 1
    return fixed, info
