"""Train the 2.5-D top-down UNet skeleton model.

Same packed dataset + same seed-split protocol as train_estimator.py so
results are directly comparable. The chain is extracted at eval time by
sorting predicted-mask pixels by their predicted arc coordinate s and
resampling to node_count nodes (x,y from pixel centres, z from the
predicted height channel).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from panda_cable_grasp.perception.dataset import (  # noqa: E402
    PackedDLOImageDataset,
    rasterize_torch,
)
from panda_cable_grasp.perception.model_img import UNetSkeleton  # noqa: E402


def move(batch: dict, dev: torch.device) -> dict:
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def extract_nodes(
    out: torch.Tensor, node_count: int, ds: PackedDLOImageDataset,
) -> torch.Tensor:
    """(B,3,H,W) maps -> (B,M,3) metres.

    Per node j: pixels with |s_pred - j/(M-1)| inside the inner half-bin
    among mask>0.5 pixels; node xy = centroid of pixel centres in world,
    z = mean predicted z over those pixels. Falls back to the closest-s
    pixel when a bin is empty.
    """
    B, _, H, W = out.shape
    m = torch.sigmoid(out[:, 0]) > 0.5
    s = out[:, 1]
    zp = out[:, 2]
    res = torch.zeros(B, node_count, 3, device=out.device)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=out.device),
        torch.arange(W, device=out.device),
        indexing="ij",
    )
    for b in range(B):
        sel = m[b]
        if sel.sum() < 2:
            sel = torch.sigmoid(out[b, 0]) > 0.1
        sp = s[b][sel]
        zp_pix = zp[b][sel]
        xw = (xs[sel].float() + 0.5) / W * (ds.X1 - ds.X0) + ds.X0
        yw = (ys[sel].float() + 0.5) / H * (ds.Y1 - ds.Y0) + ds.Y0
        for j in range(node_count):
            t = j / (node_count - 1)
            band = (sp - t).abs() < (0.5 / (node_count - 1))
            if band.sum() == 0:
                k = (sp - t).abs().argmin()
                res[b, j] = torch.stack([xw[k], yw[k], zp_pix[k]])
            else:
                res[b, j, 0] = xw[band].mean()
                res[b, j, 1] = yw[band].mean()
                res[b, j, 2] = zp_pix[band].mean()
    return res


def occ_raster(pts, X0=-0.20, X1=1.15, Y0=-0.95, Y1=0.85,
               W=192, H=256):
    """Occupancy-only raster of a raw cloud -> (B,H,W) log-count map."""
    B = pts.shape[0]
    valid = pts.abs().sum(-1) > 1e-6
    ix = ((pts[..., 0] - X0) / (X1 - X0) * W).long().clamp(0, W - 1)
    iy = ((pts[..., 1] - Y0) / (Y1 - Y0) * H).long().clamp(0, H - 1)
    cnt = torch.zeros(B, H * W, device=pts.device, dtype=torch.float32)
    cnt.scatter_add_(1, iy * W + ix, valid.float())
    return (torch.log1p(cnt) / 3.0).view(B, H, W)


def hist_occ_channels(hist_pts, ds):
    """(B,K,P,3) raw clouds -> (B,K,H,W) occupancy rasters."""
    outs = []
    for k in range(hist_pts.shape[1]):
        outs.append(occ_raster(hist_pts[:, k], ds.X0, ds.X1,
                               ds.Y0, ds.Y1, ds.W, ds.H))
    return torch.stack(outs, 1)


def corrupt_prev(
    pm: torch.Tensor, has_prev: torch.Tensor,
    flip_p: float = 0.15, shift: int = 3, drop_p: float = 0.05,
) -> torch.Tensor:
    """Training-time corruption of prev (mask,s,z) maps: simulates
    self-feedback error so the model cannot blindly copy history.
    flip_p reverses arc direction (s -> 1-s) — teaches the model to
    DETECT and REPAIR a flipped prior instead of inheriting it."""
    B = pm.shape[0]
    out = pm * has_prev.view(B, 1, 1, 1)
    dy = int(torch.randint(-shift, shift + 1, (1,)))
    dx = int(torch.randint(-shift, shift + 1, (1,)))
    out = torch.roll(out, shifts=(dy, dx), dims=(2, 3))
    fl = (torch.rand(B, device=pm.device) < flip_p)
    out[fl, 1] = 1.0 - out[fl, 1]
    if drop_p > 0:
        keep = (torch.rand(B, 1, pm.shape[2], pm.shape[3],
                           device=pm.device) > drop_p).float()
        out = out * keep
    return out


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-img, 3, 1, 1)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, 3, 1, 1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable soft skeletonization (clDice, Shit et al. CVPR'21)."""
    img = img.clamp(0, 1)
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(pm: torch.Tensor, gt: torch.Tensor,
                iters: int = 10) -> torch.Tensor:
    """Topology-preserving skeleton loss: penalises breaks that disconnect
    the cable and bridges that fuse separate strands — the errors plain
    Dice/BCE ignore."""
    sp = soft_skeletonize(pm, iters)
    sg = soft_skeletonize(gt, iters)
    tprec = ((sp * gt).sum() + 1.0) / (sp.sum() + 1.0)
    tsens = ((sg * pm).sum() + 1.0) / (sg.sum() + 1.0)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


def s_order_loss(smap: torch.Tensor, node_pos: torch.Tensor,
                 ds, margin: float = 0.02) -> torch.Tensor:
    """All-pairs ordering loss on predicted s at GT node locations:
    for every i<j, require s(j) > s(i) + margin.
    - samples via 5x5 max-pool so node/pixel misalignment doesn't
      blend in zero background;
    - all-pairs (not just adjacent) gives a monotone ramp of gradients
      on a degenerate constant field -> no flat saddle."""
    B, M = node_pos.shape[0], node_pos.shape[1]
    smax = F.max_pool2d(smap[:, None], 5, stride=1, padding=2)
    xy = node_pos[..., :2]
    gx = (xy[..., 0] - ds.X0) / (ds.X1 - ds.X0) * 2 - 1
    gy = (xy[..., 1] - ds.Y0) / (ds.Y1 - ds.Y0) * 2 - 1
    grid = torch.stack([gx, gy], -1)[:, None]       # (B,1,M,2)
    sv = F.grid_sample(smax, grid, align_corners=True)[:, 0, 0]  # (B,M)
    d = sv[:, :, None] - sv[:, None, :]             # (B,i,j) = s_i - s_j
    iu = torch.triu(torch.ones(M, M, device=smap.device,
                             dtype=torch.bool), diagonal=1)
    viol = F.relu(d[:, iu] + margin)                # i<j: s_i - s_j + m
    return viol.mean()


def compute_loss(out: torch.Tensor, tgt: torch.Tensor,
                 node_pos: torch.Tensor = None, ds=None,
                 w_cldice: float = 0.0, w_sorder: float = 0.0) -> tuple:
    """out (B,3,H,W): logit, s, z. tgt (B,3,H,W): mask, s, z."""
    gt_mask = tgt[:, 0]
    # dilate GT mask ~3 px so s/z supervision covers a tolerance band
    region = F.max_pool2d(gt_mask[:, None], 7, stride=1, padding=3)[:, 0]
    region = region > 0.5
    bce = F.binary_cross_entropy_with_logits(
        out[:, 0], gt_mask, pos_weight=torch.tensor(60.0, device=out.device)
    )
    pm = torch.sigmoid(out[:, 0])
    inter = (pm * gt_mask).sum()
    dice = 1.0 - 2.0 * inter / (pm.sum() + gt_mask.sum() + 1.0)
    s_l1 = ((out[:, 1] - tgt[:, 1]).abs() * region).sum() / region.sum().clamp(
        min=1
    )
    z_l1 = ((out[:, 2] - tgt[:, 2]).abs() * region).sum() / region.sum().clamp(
        min=1
    )
    total = bce + 2.0 * dice + 4.0 * s_l1 + 4.0 * z_l1
    logs = {"bce": float(bce), "dice": float(dice),
            "s_l1": float(s_l1), "z_l1": float(z_l1)}
    if w_cldice > 0:
        cl = cldice_loss(pm, gt_mask)
        total = total + w_cldice * cl
        logs["cldice"] = float(cl)
    if w_sorder > 0 and node_pos is not None:
        so = s_order_loss(out[:, 1], node_pos, ds)
        total = total + w_sorder * so
        logs["s_order"] = float(so)
    return total, logs


@torch.no_grad()
def evaluate(model, loader, device, ds, node_count, max_batches: int = 16,
             histmode: str = "none"):
    model.eval()
    n = pos_sq = 0
    iou_n = iou_d = s_l1 = 0.0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = move(batch, device)
        img, tgt = rasterize_torch(
            batch["pts"], batch["hand"], batch["node_pos"]
        )
        chs = [img]
        if "prev" in histmode:
            _, pmap = rasterize_torch(
                batch["pts"], batch["hand"], batch["prev_chain"]
            )
            pmap = pmap * batch["has_prev"].view(-1, 1, 1, 1)
            chs.append(pmap)
        if "occ" in histmode:
            chs.append(hist_occ_channels(batch["hist_pts"], ds))
        img = torch.cat(chs, 1)
        img = img.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda"):
            out = model(img).float()
        pm = torch.sigmoid(out[:, 0]) > 0.5
        gm = tgt[:, 0] > 0.5
        iou_n += float((pm & gm).sum())
        iou_d += float((pm | gm).sum())
        s_l1 += float(
            ((out[:, 1] - tgt[:, 1]).abs() * gm).sum()
            / gm.sum().clamp(min=1)
        )
        pred = extract_nodes(out, node_count, ds)
        pos_sq += float(
            (pred - batch["node_pos"]).pow(2).sum(-1).sum()
        )
        n += pred.shape[0] * pred.shape[1]
    return {
        "mpne_mm": 1000.0 * float(np.sqrt(pos_sq / max(n, 1))),
        "s_l1_inmask": s_l1 / max(len(loader), 1),
        "mask_iou": iou_n / max(iou_d, 1.0),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--exclude-seeds-files", nargs="*", type=Path,
                   default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--node-count", type=int, default=14)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history", action="store_true",
                   help="feed corrupted prev-frame (mask,s,z) maps as "
                        "3 extra input channels")
    p.add_argument("--histmode", default=None,
                   choices=["none", "prev", "occ", "prev+occ"],
                   help="history channels: prev=self-fed pred maps, "
                        "occ=raw occupancy at lags 1/4/16")
    p.add_argument("--cldice", type=float, default=0.0,
                   help="weight for skeleton-connectivity (clDice) loss")
    p.add_argument("--sorder", type=float, default=0.0,
                   help="weight for pairwise s-ordering loss at GT nodes")
    args = p.parse_args()
    histmode = args.histmode or ("prev" if args.history else "none")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    exclude = set()
    for sf in args.exclude_seeds_files or []:
        exclude |= {
            s.strip() for s in Path(sf).read_text().splitlines()
            if s.strip()
        }
    rng = np.random.default_rng(args.seed)
    all_seeds: set[int] = set()
    for sf in Path(args.packed_dir).glob("*__part*.seeds.npy"):
        all_seeds |= set(np.load(sf, mmap_mode="r").tolist())
    nontest = sorted(s for s in all_seeds if str(s) not in exclude)
    n_val = max(1, int(len(nontest) * 0.15))
    perm = rng.permutation(len(nontest))
    val_seeds = {str(nontest[i]) for i in perm[:n_val]}
    train_seeds = {str(nontest[i]) for i in perm[n_val:]}
    train_ds = PackedDLOImageDataset(
        args.packed_dir, seed_filter=train_seeds, preload=True
    )
    val_ds = PackedDLOImageDataset(
        args.packed_dir, seed_filter=val_seeds, preload=True
    )
    print(f"img train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_ld = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_ld = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=max(2, args.workers // 2), pin_memory=True,
    )

    cin = 5 + 3 * ("prev" in histmode) + 3 * ("occ" in histmode)
    model = UNetSkeleton(cin=cin, width=args.width).to(device)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    log_f = open(args.out / "train_log.jsonl", "a", buffering=1)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        agg: dict[str, float] = {}
        nb = 0
        for batch in train_ld:
            batch = move(batch, device)
            img, tgt = rasterize_torch(
                batch["pts"], batch["hand"], batch["node_pos"]
            )
            chs = [img]
            if "prev" in histmode:
                _, pmap = rasterize_torch(
                    batch["pts"], batch["hand"], batch["prev_chain"]
                )
                pmap = corrupt_prev(pmap, batch["has_prev"])
                chs.append(pmap)
            if "occ" in histmode:
                chs.append(hist_occ_channels(batch["hist_pts"], train_ds))
            img = torch.cat(chs, 1)
            img = img.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda"):
                out = model(img)
                loss, logs = compute_loss(
                    out.float(), tgt, batch["node_pos"], train_ds,
                    w_cldice=args.cldice, w_sorder=args.sorder)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
            if nb % 500 == 0:
                el = time.time() - t0
                print(f"  ep{epoch} batch {nb} "
                      f"({el/max(nb,1)*1000:.0f}ms/b)", flush=True)
        sched.step()
        metrics = evaluate(model, val_ld, device, val_ds,
                           args.node_count, histmode=histmode)
        metrics.update({k: v / max(nb, 1) for k, v in agg.items()})
        metrics["epoch"] = epoch
        metrics["sec"] = round(time.time() - t0, 1)
        print(json.dumps(metrics), flush=True)
        log_f.write(json.dumps(metrics) + "\n")
        if metrics["mpne_mm"] < best:
            best = metrics["mpne_mm"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "width": args.width,
                    "cin": cin,
                    "histmode": histmode,
                    "node_count": args.node_count,
                    "window": [train_ds.X0, train_ds.X1,
                               train_ds.Y0, train_ds.Y1,
                               train_ds.W, train_ds.H],
                    "metrics": metrics,
                },
                args.out / "checkpoint_best.pt",
            )
    log_f.close()
    print(f"best_val_mpne_mm={best:.2f}")


if __name__ == "__main__":
    main()
