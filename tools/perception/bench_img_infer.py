"""Deployment-latency bench: per-frame rasterize -> UNet -> skeleton
extract timing on a real episode stream, batch=1 (as deployed)."""
import sys
import time
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

packed, ckpt = Path(sys.argv[1]), sys.argv[2]
scen, seed = sys.argv[3], int(sys.argv[4])
dev = "cuda:0"

payload = torch.load(ckpt, map_location="cpu", weights_only=False)
cin = payload.get("cin", 5)
histmode = payload.get(
    "histmode", {5: "none", 8: "prev", 11: "prev+occ"}[cin])
model = UNetSkeleton(cin=cin, width=payload["width"]).to(dev).eval()
model.load_state_dict(payload["model"])
nparam = sum(p.numel() for p in model.parameters())
ds = PackedDLOImageDataset.__new__(PackedDLOImageDataset)
X0, X1, Y0, Y1, W, H = ds.X0, ds.X1, ds.Y0, ds.Y1, ds.W, ds.H

prev_map = torch.zeros(1, 3, H, W, device=dev)
hbuf = deque(maxlen=16)
t_ras = t_fwd = t_ext = 0.0
n = 0
for f in episode_frames(packed, scen, seed):
    t0 = time.perf_counter()
    pts = torch.from_numpy(
        f["points"].astype(np.float32).reshape(-1, 3))[None]
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
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        ot = model(img)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    om = ot[0].cpu().numpy()
    pm = 1 / (1 + np.exp(-om[0])) > 0.5
    p = extract_skel(pm, om[2], smap=om[1])
    t3 = time.perf_counter()
    hbuf.append(f["points"].astype(np.float32))
    if "prev" in histmode:
        bm = (torch.sigmoid(ot[0, 0]) > 0.5).float()
        prev_map = torch.stack(
            [bm, ot[0, 1] * bm, ot[0, 2] * bm])[None]
    t_ras += t1 - t0
    t_fwd += t2 - t1
    t_ext += t3 - t2
    n += 1
print(f"histmode={histmode} params={nparam / 1e6:.2f}M  n={n}")
print(f"rasterize+hist: {t_ras / n * 1000:.1f} ms | "
      f"forward: {t_fwd / n * 1000:.1f} ms | "
      f"extract: {t_ext / n * 1000:.1f} ms | "
      f"TOTAL: {(t_ras + t_fwd + t_ext) / n * 1000:.1f} ms/frame")
