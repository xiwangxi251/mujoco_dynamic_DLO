"""Ablation: pred mask vs raw occupancy (≈perfect colour segmentation)
through the SAME skeleton-walk + bridge + temporal-anchor extractor."""
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from panda_cable_grasp.perception.dataset import (
    PackedDLOImageDataset, rasterize_torch)
from dump_occdyn_ep import episode_frames
from test_skel_extract import extract_skel, M

packed, scenario, seed, maps_npz, npz = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])

ds = PackedDLOImageDataset.__new__(PackedDLOImageDataset)
maps = np.load(maps_npz)["maps"]
d = np.load(npz)
X0, X1, Y0, Y1, W, H = ds.X0, ds.X1, ds.Y0, ds.Y1, ds.W, ds.H

res = {"pred": [], "occ": []}
prev = {"pred": None, "occ": None}
i = 0
for f in episode_frames(Path(packed), scenario, seed):
    if i >= len(maps):
        break
    pts = torch.from_numpy(f["points"].astype(np.float32).reshape(-1, 3))[None]
    hand = torch.from_numpy(f["hand"].astype(np.float32))[None]
    chain = torch.from_numpy(f["pos14"].astype(np.float32))[None]
    img, _ = rasterize_torch(pts, hand, chain, X0, X1, Y0, Y1, W, H)
    occ_m = img[0, 0].numpy() > 0          # colour-seg equivalent
    zm = img[0, 1].numpy()                 # z_mean channel for occ z
    pm = 1 / (1 + np.exp(-maps[i][0])) > 0.5

    for tag, msk, zmap, sm in (
            ("pred", pm, maps[i][2], maps[i][1]),
            ("occ", occ_m, zm, maps[i][1])):
        p = extract_skel(msk, zmap, smap=sm)
        if p is None:
            p = np.full((M, 3), np.nan)
        elif prev[tag] is not None:
            ef = np.linalg.norm(p - prev[tag], axis=1).mean()
            eb = np.linalg.norm(p[::-1] - prev[tag], axis=1).mean()
            if eb < ef:
                p = p[::-1]
        if not np.isnan(p[0, 0]):
            prev[tag] = p.copy()
        e = np.linalg.norm(p - f["pos14"], axis=1).mean() * 1000
        er = np.linalg.norm(p[::-1] - f["pos14"], axis=1).mean() * 1000
        res[tag].append((e, er))
        if tag == "pred":
            globals().setdefault("_ppred", []).append(p)
        else:
            globals().setdefault("_pocc", []).append(p)
    i += 1

for tag in ("pred", "occ"):
    a = np.array(res[tag])
    ok = ~np.isnan(a[:, 0])
    print(f"{tag}: strict={np.nanmean(a[:,0]):.1f}mm  "
          f"flips={(a[:,1]<a[:,0]).sum()}/{ok.sum()}  "
          f"flip-tol={np.nanmean(np.minimum(a[:,0],a[:,1])):.1f}mm")
np.savez("/tmp/img_ep10_skel_occ.npz",
         pred=np.array(_pocc), gt=d["gt"][:len(_pocc)])
np.savez("/tmp/img_ep10_skel_pred.npz",
         pred=np.array(_ppred), gt=d["gt"][:len(_ppred)])
