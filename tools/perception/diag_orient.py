"""Orientation-policy diagnostic: replays dumped pred maps and compares
direction rules per frame.

Evidence sources per frame (skeleton path is unordered geometry):
  s-map:  endpoint with lower mean predicted s -> node 0
  temporal: direction whose full 14-node chain is closer to prev frame

Policies evaluated:
  s       - s-map only
  t       - temporal only (current production rule; f0 uses s)
  fuse    - s-map unless temporal confident AND s-map ambiguous
  tfixed  - temporal, but s-map disagreement with margin > m for k
            consecutive frames forces re-anchor (flip-lock breaker)
  oracle  - whichever direction is closer to GT (upper bound)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from test_skel_extract import extract_skel, endpoint_s

X0, X1, Y0, Y1 = -0.20, 1.15, -0.95, 0.85
W, H, M = 192, 256, 14


def to_px(xy):
    return (int(round((xy[0] - X0) / (X1 - X0) * W)),
            int(round((xy[1] - Y0) / (Y1 - Y0) * H)))


def frame_evidence(pm, smap, zmap):
    """Returns (p, sa, sb); p is s-oriented. Temporal margins are
    computed inside run_policy against that policy's OWN prev —
    prev depends on past decisions, so ef/eb cannot be precomputed."""
    p = extract_skel(pm, zmap, smap=smap)
    if p is None:
        return None
    sa = endpoint_s(smap, pm, *to_px(p[0, :2]))
    sb = endpoint_s(smap, pm, *to_px(p[-1, :2]))
    return p, sa, sb


def run_policy(recs, gt, policy, m=0.3, k=3, w=0.5):
    """recs: list of (p, sa, sb, ef, eb). Returns (mpne, flips).
    p arrives s-oriented (p[0] = lower-s end); sa<s b by construction,
    so s-confidence margin = sb - sa >= 0; temporal votes flip iff
    eb < ef with margin ef - eb (metres)."""
    prev = None
    disagree_streak = hi = 0
    errs, flips = [], 0
    for i, r in enumerate(recs):
        if r is None:
            errs.append(np.nan)
            continue
        p, sa, sb = r
        s_ok = not (np.isnan(sa) or np.isnan(sb))
        ms = (sb - sa) if s_ok else 0.0             # s confidence
        if prev is not None:
            ef = np.linalg.norm(p - prev, axis=1).mean()
            eb = np.linalg.norm(p[::-1] - prev, axis=1).mean()
        else:
            ef = eb = np.nan
        t_wants_flip = prev is not None and eb < ef
        t_margin = (ef - eb) if t_wants_flip else 0.0
        if policy == "s" or prev is None:
            pass                                    # p already s-oriented
        elif policy == "t":
            if t_wants_flip:
                p = p[::-1]
        elif policy == "sveto":
            # temporal may flip only when s-map is NOT confident
            if t_wants_flip and ms < m:
                p = p[::-1]
        elif policy == "vote":
            # common currency: expected error if wrong ≈ margin*scale;
            # s margin in [0,1] * w metres/unit
            if t_wants_flip and t_margin > ms * w:
                p = p[::-1]
        elif policy == "streak":
            # temporal governs; a SUSTAINED s-vs-t conflict means the
            # temporal lock is wrong -> defect to s-map.  s-map errors
            # are intermittent so they never trigger the streak.
            if t_wants_flip:
                disagree_streak += 1
            else:
                disagree_streak = 0
            if t_wants_flip and disagree_streak < k:
                p = p[::-1]
        elif policy == "streakconf":
            # same, but only confident s-map opposition counts toward
            # the streak (ms > m)
            if t_wants_flip and ms > m:
                disagree_streak += 1
            elif not t_wants_flip:
                disagree_streak = 0
            if t_wants_flip and disagree_streak < k:
                p = p[::-1]
        elif policy == "twotier":
            # hi-conflict (ms > m) defects fast (k frames); any
            # sustained conflict defects slowly (30 frames)
            if t_wants_flip:
                disagree_streak += 1
                hi = hi + 1 if ms > m else 0
            else:
                disagree_streak = hi = 0
            if t_wants_flip and hi < k and disagree_streak < 30:
                p = p[::-1]
        elif policy == "tfixed":
            # temporal governs, but persistent confident s-disagreement
            # breaks the lock: flip back and stay broken k frames
            if disagree_streak > 0:
                disagree_streak -= 1                # s-map controls now
            elif t_wants_flip:
                if ms > m:
                    disagree_streak = k             # detect lock, veto
                else:
                    p = p[::-1]
        e = np.linalg.norm(p - gt[i], axis=1).mean() * 1000
        er = np.linalg.norm(p[::-1] - gt[i], axis=1).mean() * 1000
        errs.append(e)
        flips += er < e
        prev = p.copy()
    return np.nanmean(errs), flips


def main():
    maps = np.load(sys.argv[1])["maps"]
    gt = np.load(sys.argv[2])["gt"]
    # pass 1: gather per-frame evidence (skeleton geometry fixed)
    recs = []
    for f in range(len(gt)):
        om = maps[f]
        pm = 1 / (1 + np.exp(-om[0])) > 0.5
        recs.append(frame_evidence(pm, om[1], om[2]))
    # agreement stats vs GT (oracle prev: best-case temporal evidence)
    s_right = t_right = agree = n = 0
    prev = None
    for i, r in enumerate(recs):
        if r is None:
            continue
        p, sa, sb = r
        e = np.linalg.norm(p - gt[i], axis=1).mean()
        er = np.linalg.norm(p[::-1] - gt[i], axis=1).mean()
        s_right += e <= er
        if prev is not None:
            ef = np.linalg.norm(p - prev, axis=1).mean()
            eb = np.linalg.norm(p[::-1] - prev, axis=1).mean()
            t_flip = eb < ef
            t_right += (er < e) == t_flip
            agree += not t_flip
        n += 1
        prev = (p[::-1] if er < e else p).copy()   # oracle prev for stats
    print(f"n={n}  s-map correct={s_right / n * 100:.0f}%  "
          f"temporal correct={t_right / n * 100:.0f}%  "
          f"agree={agree / max(n - 1, 1) * 100:.0f}%")
    for pol, kw in (
            ("s", {}), ("t", {}),
            ("sveto", {"m": 0.5}), ("sveto", {"m": 0.8}),
            ("vote", {"w": 0.3}),
            ("streak", {"k": 10}), ("streak", {"k": 25}),
            ("streak", {"k": 40}),
            ("streakconf", {"m": 0.7, "k": 12}),
            ("twotier", {"m": 0.85, "k": 5}),
            ("twotier", {"m": 0.85, "k": 3}),
            ("twotier", {"m": 0.9, "k": 5}),
            ("tfixed", {"m": 0.5, "k": 3})):
        mp, fl = run_policy(recs, gt, pol, **kw)
        tag = f"{pol}{kw}" if kw else pol
        print(f"  {tag:22s}: strict={mp:5.1f}mm  flips={fl}/{n}")


if __name__ == "__main__":
    main()
