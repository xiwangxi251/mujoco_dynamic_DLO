"""Render the image-model I/O: 5 input channels, GT maps, predictions."""
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from panda_cable_grasp.perception.dataset import (
    PackedDLOImageDataset, rasterize_torch)
from dump_occdyn_ep import episode_frames

packed = sys.argv[1]
scenario, seed = sys.argv[2], int(sys.argv[3])
want = [int(x) for x in sys.argv[4].split(",")]
maps_npz = sys.argv[5] if len(sys.argv) > 5 else None
out_prefix = sys.argv[6] if len(sys.argv) > 6 else "/tmp/img_io"

ds = PackedDLOImageDataset.__new__(PackedDLOImageDataset)
pred_maps = np.load(maps_npz)["maps"] if maps_npz else None

for i, f in enumerate(episode_frames(Path(packed), scenario, seed)):
    if i not in want:
        if i > max(want):
            break
        continue
    pts = torch.from_numpy(f["points"].astype(np.float32).reshape(-1, 3))[None]
    hand = torch.from_numpy(f["hand"].astype(np.float32))[None]
    chain = torch.from_numpy(f["pos14"].astype(np.float32))[None]
    img, tgt = rasterize_torch(pts, hand, chain,
                               ds.X0, ds.X1, ds.Y0, ds.Y1, ds.W, ds.H)
    img = img[0].numpy(); tg = tgt[0].numpy()

    ncol = 3 if pred_maps is None else 4
    fig, axes = plt.subplots(3, ncol, figsize=(5 * ncol, 14))
    titles = ["input ch0 occ", "ch1 z_mean", "ch2 z_min", "ch3 z_max",
              "ch4 hand", "GT mask", "GT s", "GT z"]
    chan = [img[0], img[1], img[2], img[3], img[4], tg[0], tg[1], tg[2]]
    for ax, t, c in zip(axes.flat, titles, chan):
        ax.imshow(c, cmap="viridis", origin="lower")
        ax.set_title(f"f{i} {t}")
        ax.axis("off")
    if pred_maps is not None:
        pm = pred_maps[i]
        extra = [1 / (1 + np.exp(-pm[0])), pm[1], pm[2]]
        for ax, t, c in zip(axes.flat[len(chan):],
                            ["PRED mask", "PRED s", "PRED z"], extra):
            ax.imshow(c, cmap="viridis", origin="lower")
            ax.set_title(f"f{i} {t}")
            ax.axis("off")
    for ax in axes.flat[len(chan) + (3 if pred_maps is not None else 0):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_f{i}.png", dpi=100)
    print(f"saved {out_prefix}_f{i}.png")
