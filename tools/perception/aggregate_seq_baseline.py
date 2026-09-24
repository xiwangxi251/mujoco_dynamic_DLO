"""Aggregate sequential-episode dumps into strict-ordered MPNE + flip rate.

Reads dump npz files produced by any per-frame dumper under a tree like

    <root>/<scenario>/<seed-or-ep>.npz          (one npz per episode)
    <root>/<scenario>/<seed>/frame_*.npz       (MP2CDLO per-frame style)

Every chain (gt and pred) is arc-length resampled to 14 nodes so the
strict ordered error is comparable across methods regardless of native
node count. A frame counts as "flipped" when the reversed resampled
prediction matches GT better.

Usage:
    python aggregate_seq_baseline.py --root /tmp/seqbase/trackdlo --label trackdlo
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def resample(points: np.ndarray, n: int = 14) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(points[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, points[:, c]) for c in range(3)], 1)


def eval_episode(pred, gt, ok):
    """Per-frame strict fwd/rev errors plus a per-episode direction-oracle MPNE:
    pick whichever FIXED direction (fwd or rev) is better over the whole
    episode — models like TrackDLO choose an arbitrary direction once at
    init, so this is the honest middle ground vs per-frame flipping."""
    errs, revs, flips = [], [], []
    for p, g, o in zip(pred, gt, ok):
        p, g = np.asarray(p), np.asarray(g)
        if not o or p.ndim != 2 or g.ndim != 2 or len(p) < 2 or len(g) < 2:
            continue
        p14, g14 = resample(p), resample(g)
        e = np.linalg.norm(p14 - g14, axis=1).mean()
        er = np.linalg.norm(p14[::-1] - g14, axis=1).mean()
        errs.append(e * 1000)
        revs.append(er * 1000)
        flips.append(er < e)
    errs, revs = np.asarray(errs), np.asarray(revs)
    epdir = min(errs.sum(), revs.sum()) / max(len(errs), 1)
    return errs, np.asarray(flips), epdir


def load_ep_file(path: str):
    d = np.load(path, allow_pickle=True)
    gt = np.asarray(d["gt"])
    key = "pred" if "pred" in d else "kp"
    pred = np.asarray(d[key])
    ok = np.asarray(d["ok"]) if "ok" in d else np.ones(len(pred), bool)
    if pred.ndim == 2:
        # single-frame file (MP2CDLO per-frame dump): wrap to (1, N, 3)
        pred, gt = pred[None], gt[None]
        ok = np.asarray([bool(np.asarray(ok).all())])
    return pred, gt, ok


def iter_episodes(root: str):
    """Yield (scenario, [npz_paths]) — per-file or per-frame-dir episodes."""
    for sc in sorted(os.listdir(root)):
        scd = os.path.join(root, sc)
        if not os.path.isdir(scd):
            continue
        for item in sorted(os.listdir(scd)):
            p = os.path.join(scd, item)
            if item.endswith(".npz"):
                yield sc, [p]
            elif os.path.isdir(p):
                yield sc, sorted(glob.glob(os.path.join(p, "*.npz")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    report = {}
    by_sc: dict[str, list] = {}
    for sc, files in iter_episodes(args.root):
        by_sc.setdefault(sc, []).append(files)
    for sc, ep_file_lists in by_sc.items():
        errs, flips, epdirs = [], [], []
        for fs in ep_file_lists:
            if not fs:
                continue
            pred, gt, ok = [], [], []
            for f in fs:
                pr, g, o = load_ep_file(f)
                pred.append(np.asarray(pr))
                gt.append(np.asarray(g))
                ok.append(np.asarray(o))
            pred = np.concatenate(pred)
            gt = np.concatenate(gt)
            ok = np.concatenate(ok)
            e, fl, epdir = eval_episode(pred, gt, ok)
            errs.append(e)
            flips.append(fl)
            epdirs.append(epdir)
        if not errs:
            continue
        e = np.concatenate(errs)
        fl = np.concatenate(flips)
        report[sc] = dict(
            n_eps=len(errs), n_frames=int(len(e)),
            mpne_mm=float(e.mean()), mpne_med_mm=float(np.median(e)),
            epdir_mpne_mm=float(np.mean(epdirs)),
            flip_rate=float(fl.mean()), flip_frames=int(fl.sum()),
        )
        print(
            f"{args.label:14s} {sc:28s} eps={len(errs):2d} frames={len(e):5d} "
            f"MPNE={e.mean():6.1f} med={np.median(e):6.1f} "
            f"epdir={np.mean(epdirs):6.1f} flip={fl.mean() * 100:4.1f}%",
            flush=True,
        )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
