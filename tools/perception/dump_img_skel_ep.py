"""Dump img-UNet predictions using skeleton+bridge+temporal-anchor
extractor, same npz layout as dump_occdyn_ep for render_tracked_npz."""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from panda_cable_grasp.perception.dataset import (
    PackedDLOImageDataset, rasterize_torch)
from panda_cable_grasp.perception.model_img import UNetSkeleton
from dump_occdyn_ep import episode_frames
from train_img import occ_raster
from test_skel_extract import extract_skel

packed, ckpt, scen, seed, out = (
    Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]),
    sys.argv[5])
maps_out = sys.argv[6] if len(sys.argv) > 6 else None
dev = "cuda:0"

payload = torch.load(ckpt, map_location="cpu", weights_only=False)
cin = payload.get("cin", 5)
histmode = payload.get(
    "histmode", {5: "none", 8: "prev", 11: "prev+occ"}[cin])
model = UNetSkeleton(cin=cin, width=payload["width"]).to(dev).eval()
model.load_state_dict(payload["model"])
M = payload["node_count"]
ds = PackedDLOImageDataset.__new__(PackedDLOImageDataset)
X0, X1, Y0, Y1, W, H = ds.X0, ds.X1, ds.Y0, ds.Y1, ds.W, ds.H

clouds, gts, preds, maps = [], [], [], []
prev = None
streak = 0                                       # sustained s-vs-t conflict
prev_map = torch.zeros(1, 3, H, W, device=dev)  # self-feedback history
hbuf = deque(maxlen=16)
for f in episode_frames(packed, scen, seed):
    pts = torch.from_numpy(f["points"].astype(np.float32).reshape(-1, 3))[None]
    hand = torch.from_numpy(f["hand"].astype(np.float32))[None]
    img, _ = rasterize_torch(pts, hand, torch.zeros(1, 2, 3),
                             X0, X1, Y0, Y1, W, H)
    img = img.to(dev)
    chs = [img]
    if "prev" in histmode:
        chs.append(prev_map)
    if "occ" in histmode:
        for lag in PackedDLOImageDataset.HIST_LAGS:
            if len(hbuf) >= lag:
                pk = torch.from_numpy(
                    hbuf[-lag].astype(np.float32)
                    .reshape(-1, 3))[None].to(dev)
                chs.append(occ_raster(pk, X0, X1, Y0, Y1, W, H)[:, None])
            else:
                chs.append(torch.zeros(1, 1, H, W, device=dev))
    if len(chs) > 1:
        img = torch.cat(chs, 1)
    with torch.no_grad():
        ot = model(img)
    hbuf.append(f["points"].astype(np.float32))
    om = ot[0].cpu().numpy()
    if "prev" in histmode:
        bm = (torch.sigmoid(ot[0, 0]) > 0.5).float()
        prev_map = torch.stack(
            [bm, ot[0, 1] * bm, ot[0, 2] * bm])[None]
    pm = 1 / (1 + np.exp(-om[0])) > 0.5
    p = extract_skel(pm, om[2], smap=om[1])
    if p is None:
        p = np.full((M, 3), np.nan)
    elif prev is not None:
        ef = np.linalg.norm(p - prev, axis=1).mean()
        eb = np.linalg.norm(p[::-1] - prev, axis=1).mean()
        if eb < ef:
            streak += 1
            if streak < 25:
                p = p[::-1]
        else:
            streak = 0
    if not np.isnan(p[0, 0]):
        prev = p.copy()
    cloud = f["points"].astype(np.float32).reshape(-1, 3)
    clouds.append(cloud[np.abs(cloud).sum(1) > 1e-6])
    gts.append(np.asarray(f["pos14"], np.float64))
    preds.append(p)
    maps.append(om)

np.savez(out, cloud=np.asarray(clouds, dtype=object),
         gt=np.asarray(gts), pred=np.asarray(preds),
         ok=np.ones(len(gts), bool), allow_pickle=True)
if maps_out:
    np.savez(maps_out, maps=np.asarray(maps))
e = np.linalg.norm(np.asarray(preds) - np.asarray(gts), axis=2).mean(1) * 1000
er = np.linalg.norm(np.asarray(preds)[:, ::-1] - np.asarray(gts),
                    axis=2).mean(1) * 1000
print(f"wrote {out} n={len(gts)} strict={np.nanmean(e):.1f}mm "
      f"flips={(er < e).sum()}")
