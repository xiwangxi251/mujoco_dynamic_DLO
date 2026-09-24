"""Eval with occlusion-split mask metrics: mask IoU separately on
pixels that were VISIBLE (occ>0) vs OCCLUDED (GT+ but occ==0).
Tests the hypothesis: does temporal input help exactly where the
current frame underdetermines geometry?"""
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

packed, ckpt = Path(sys.argv[1]), sys.argv[2]
n_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
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

SCEN = ["id_static", "id_rigid_l1_nominal",
        "id_shape_nominal_current", "id_combined_l1_nominal"]
seeds_root = Path("/data1/hxai/mujoco/perception_runs")

for scen in SCEN:
    seeds = [int(l) for l in open(
        seeds_root / f"test_seeds_{scen}.txt")][:n_eps]
    errs, flips = [], 0
    # occlusion-split mask metrics
    inter_vis = union_vis = inter_occ = gt_occ = pred_occ = 0
    for seed in seeds:
        prev = None
        streak = 0
        prev_map = torch.zeros(1, 3, H, W, device=dev)
        hbuf = deque(maxlen=16)
        for f in episode_frames(packed, scen, seed):
            pts = torch.from_numpy(
                f["points"].astype(np.float32).reshape(-1, 3))[None]
            hand = torch.from_numpy(f["hand"].astype(np.float32))[None]
            img, tgt = rasterize_torch(pts, hand,
                                       torch.from_numpy(
                                           np.asarray(
                                               f["pos14"], np.float32)
                                       )[None],
                                       X0, X1, Y0, Y1, W, H)
            occ = img[:, 0] > 0
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
                        chs.append(occ_raster(
                            pk, X0, X1, Y0, Y1, W, H)[:, None])
                    else:
                        chs.append(torch.zeros(1, 1, H, W, device=dev))
            if len(chs) > 1:
                img = torch.cat(chs, 1)
            with torch.no_grad():
                ot = model(img)
            hbuf.append(f["points"].astype(np.float32))
            if "prev" in histmode:
                bm = (torch.sigmoid(ot[0, 0]) > 0.5).float()
                prev_map = torch.stack(
                    [bm, ot[0, 1] * bm, ot[0, 2] * bm])[None]
            om = ot[0].cpu().numpy()
            gt = np.asarray(f["pos14"], np.float64)

            pm = torch.sigmoid(ot[0, 0]) > 0.5
            gm = tgt[0, 0] > 0.5
            oc = occ[0].cpu()
            # visible region: GT line pixels that ARE observed
            vis = gm & oc.cpu()
            occd = gm & ~oc.cpu()          # GT line where cloud blind
            inter_vis += int((pm.cpu() & vis).sum())
            union_vis += int(vis.sum())
            inter_occ += int((pm.cpu() & occd).sum())
            gt_occ += int(occd.sum())
            pred_occ += int((pm.cpu() & ~oc.cpu()).sum())

            p = extract_skel(
                pm.cpu().numpy(), om[2], smap=om[1])
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
            e = np.linalg.norm(p - gt, axis=1).mean() * 1000
            er = np.linalg.norm(p[::-1] - gt, axis=1).mean() * 1000
            errs.append(e)
            flips += er < e
        print(f"  {scen} seed={seed} done", flush=True)
    errs = np.array(errs)
    occ_frac = gt_occ / max(union_vis + gt_occ, 1)
    print(f"{scen}: n={len(errs)} strict={np.nanmean(errs):.1f}mm "
          f"flips={flips} | occ_px={occ_frac * 100:.0f}% of GT | "
          f"recall_vis={inter_vis / max(union_vis, 1):.3f} "
          f"recall_occ={inter_occ / max(gt_occ, 1):.3f} "
          f"occ_fp={pred_occ / max(gt_occ, 1):.2f}", flush=True)
