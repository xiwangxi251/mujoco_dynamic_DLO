"""Disagreement-run analysis: for each frame, does temporal want to flip
the s-oriented chain? If s-map is RIGHT, a long run of 't wants flip'
means temporal is locked-wrong; scattered runs mean s-map flickers."""
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


maps = np.load(sys.argv[1])["maps"]
gt = np.load(sys.argv[2])["gt"]
prev = None
runs, cur = [], 0
margins = []          # (ms_s, ef-eb, correct?) per conflict frame
for f in range(len(gt)):
    om = maps[f]
    pm = 1 / (1 + np.exp(-om[0])) > 0.5
    p = extract_skel(pm, om[2], smap=om[1])
    if p is None:
        continue
    sa = endpoint_s(om[1], pm, *to_px(p[0, :2]))
    sb = endpoint_s(om[1], pm, *to_px(p[-1, :2]))
    if prev is not None:
        ef = np.linalg.norm(p - prev, axis=1).mean()
        eb = np.linalg.norm(p[::-1] - prev, axis=1).mean()
        if eb < ef:                      # temporal wants to flip s-orient
            cur += 1
            e = np.linalg.norm(p - gt[f], axis=1).mean()
            er = np.linalg.norm(p[::-1] - gt[f], axis=1).mean()
            margins.append((sb - sa, ef - eb, er < e))
        else:
            if cur:
                runs.append(cur)
            cur = 0
        prev = (p[::-1] if eb < ef else p).copy()
    else:
        prev = p.copy()
if cur:
    runs.append(cur)
print(f"t-wants-flip runs: {sorted(runs, reverse=True)[:15]}")
mc = np.array(margins) if margins else np.zeros((0, 3))
if len(mc):
    ok = mc[:, 2].astype(bool)
    print(f"conflict frames={len(mc)}  temporal-right={ok.sum()}  "
          f"s-right={(~ok).sum()}")
    print(f"  when temporal right: ms={mc[ok, 0].mean():.2f} "
          f"tm={mc[ok, 1].mean() * 1000:.0f}mm")
    if (~ok).sum():
        print(f"  when s-map right:    ms={mc[~ok, 0].mean():.2f} "
              f"tm={mc[~ok, 1].mean() * 1000:.0f}mm")
