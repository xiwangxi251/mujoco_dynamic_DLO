"""Visualise dumped img-model predictions: mask, s-map, chain vs cloud."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz, maps_npz, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
frames = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0]

d = np.load(npz, allow_pickle=True)
maps = np.load(maps_npz)["maps"]
X0, X1, Y0, Y1 = -0.20, 1.15, -0.95, 0.85
for f in frames:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    m = maps[f]
    ax = axes[0]
    ax.imshow(1 / (1 + np.exp(-m[0])), cmap="gray", origin="lower",
              extent=[X0, X1, Y0, Y1])
    ax.set_title(f"f{f} pred mask")
    ax = axes[1]
    sm = np.ma.masked_where(1 / (1 + np.exp(-m[0])) < 0.5, m[1])
    ax.imshow(sm, cmap="hsv", origin="lower",
              extent=[X0, X1, Y0, Y1], vmin=0, vmax=1)
    ax.set_title("pred s (masked)")
    ax = axes[2]
    cl = d["cloud"][f]
    ax.scatter(cl[:, 0], cl[:, 1], s=1, c="lightgray")
    ax.plot(d["gt"][f][:, 0], d["gt"][f][:, 1], "g-o", ms=4, label="gt")
    ax.plot(d["pred"][f][:, 0], d["pred"][f][:, 1], "r-x", ms=4,
            label="pred")
    ax.legend()
    ax.set_title("chain vs cloud")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_f{f}.png", dpi=110)
    err = np.linalg.norm(d["pred"][f] - d["gt"][f], axis=1).mean() * 1000
    print(f, round(float(err), 1))
