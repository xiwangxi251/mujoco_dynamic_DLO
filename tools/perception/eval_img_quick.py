"""Quick S_sc eval: N held-out episodes per scenario.
For each frame: UNet -> maps -> {s-binning, skeleton+bridge+temporal} nodes.
Reports strict ordered MPNE + flip rate per extractor."""
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
from train_img import extract_nodes, occ_raster
from test_skel_extract import extract_skel

packed = Path(sys.argv[1])
ckpt = sys.argv[2]
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
    sf = seeds_root / f"test_seeds_{scen}.txt"
    seeds = [int(l) for l in open(sf)][:n_eps]
    agg = {"bin": [], "skel": []}
    flips = {"bin": 0, "skel": 0}
    nfr = 0
    for seed in seeds:
        prev = None
        streak = 0                       # sustained s-vs-temporal conflict
        prev_map = torch.zeros(1, 3, H, W, device=dev)
        hbuf = deque(maxlen=16)          # past raw clouds for occ history
        for f in episode_frames(packed, scen, seed):
            pts = torch.from_numpy(
                f["points"].astype(np.float32).reshape(-1, 3))[None]
            hand = torch.from_numpy(f["hand"].astype(np.float32))[None]
            dum = torch.zeros(1, 2, 3)
            img, _ = rasterize_torch(pts, hand, dum, X0, X1, Y0, Y1, W, H)
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
                out = model(img)
            hbuf.append(f["points"].astype(np.float32))
            if "prev" in histmode:
                bm = (torch.sigmoid(out[0, 0]) > 0.5).float()
                prev_map = torch.stack(
                    [bm, out[0, 1] * bm, out[0, 2] * bm])[None]
            gt = np.asarray(f["pos14"], np.float64)
            om = out[0].cpu().numpy()

            # extractor A: s-binning (original)
            pb = extract_nodes(out, M, ds)[0].cpu().numpy()
            # extractor B: skeleton + bridge + temporal anchor
            pm = 1 / (1 + np.exp(-om[0])) > 0.5
            ps = extract_skel(pm, om[2], smap=om[1])
            if ps is None:
                ps = np.full((M, 3), np.nan)
            elif prev is not None:
                ef = np.linalg.norm(ps - prev, axis=1).mean()
                eb = np.linalg.norm(ps[::-1] - prev, axis=1).mean()
                if eb < ef:
                    streak += 1          # temporal wants flip vs s-orient
                    if streak < 25:
                        ps = ps[::-1]
                else:
                    streak = 0
            if not np.isnan(ps[0, 0]):
                prev = ps.copy()

            for tag, p in (("bin", pb), ("skel", ps)):
                e = np.linalg.norm(p - gt, axis=1).mean() * 1000
                er = np.linalg.norm(p[::-1] - gt, axis=1).mean() * 1000
                agg[tag].append(e)
                flips[tag] += er < e
            nfr += 1
        print(f"  {scen} seed={seed} done", flush=True)
    eb = np.nanmean(agg["bin"]); es = np.nanmean(agg["skel"])
    print(f"{scen}: n={nfr}  "
          f"s-bin={eb:.1f}mm(flips {flips['bin']})  "
          f"skel+temp={es:.1f}mm(flips {flips['skel']})", flush=True)
