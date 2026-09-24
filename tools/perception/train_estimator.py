"""Train the occlusion-conditioned DLO state estimator.

Example:
    python tools/perception/train_estimator.py \
        --data-root /data1/hxai/mujoco/perception_runs/dataset_v1 \
        --variant full --epochs 60 --batch 256 --device cuda:3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import dataclasses

import numpy as np
import torch
from torch.utils.data import DataLoader

from panda_cable_grasp.perception.dataset import (
    DLOFrameDataset,
    PackedDLODataset,
    split_episodes,
)
from panda_cable_grasp.perception.model import (
    DLOStateEstimator,
    EstimatorConfig,
    gaussian_nll,
)

VARIANTS = {
    # ours
    "full": {},
    # ablations of ours
    "no_mask": {"use_mask": False},
    "no_wrist": {"use_wrist": False},
    "no_hist": {"use_history": False},
    "no_fhand": {"use_future_hand": False},
    "no_heads": {
        "predict_velocity": False, "predict_future": False,
        "predict_logvar": False,
    },
    # baseline-style regressors (single frame, no conditioning)
    "reg_only": {
        "use_mask": False, "use_wrist": False, "use_decoder_attn": False,
        "use_history": False,
        "predict_velocity": False, "predict_future": False,
        "predict_logvar": False,
    },
    "reg_attn": {
        "use_mask": False, "use_wrist": False, "use_history": False,
        "predict_velocity": False, "predict_future": False,
        "predict_logvar": False,
    },
    "dual_attn": {
        "use_mask": False, "use_history": False,
        "predict_velocity": False, "predict_future": False,
        "predict_logvar": False,
    },
    # dual_attn + local-geometric encoder (kNN edge features) — tests
    # whether PointNet++-style local context helps topology decisions
    "dual_local": {
        "use_mask": False, "use_history": False,
        "predict_velocity": False, "predict_future": False,
        "predict_logvar": False,
        "local_k": 16,
    },
    # dual_local + segment-level cloud-coverage loss — targets the
    # shortcut-chord failure (chain cuts through a loop's interior,
    # leaving whole visible strands uncovered)
    "dual_local_cov": {
        "use_mask": False, "use_history": False,
        "predict_velocity": False, "predict_future": False,
        "predict_logvar": False,
        "local_k": 16,
        "cloud_chamfer_w": 0.5,
        "cloud_e2c_w": 0.0,  # pure coverage; the node->cloud pull can
        # drag nodes onto the wrong strand at self-crossings
    },
    # dual_local + dense arc-length parameterisation: per-point s in
    # [0,1] supervision + soft-bin chain extraction — connectivity is
    # explicit and densely supervised (targets shortcut chords)
    "arc_dense": {
        "use_mask": False, "use_history": False,
        "predict_velocity": False, "predict_future": False,
        "predict_logvar": False,
        "local_k": 16,
        "arc_head": True, "arc_w": 1.0,
    },
    # arc head as dense auxiliary task only — direct node regression
    # stays the output; tests whether per-point s supervision alone
    # improves strand assignment without the extraction bottleneck
    "arc_aux": {
        "use_mask": False, "use_history": False,
        "predict_velocity": False, "predict_future": False,
        "predict_logvar": False,
        "local_k": 16,
        "arc_head": True, "arc_w": 1.0, "arc_extract": False,
    },
    # topology-regularised full model: inextensible segment-length loss +
    # cloud-coverage chamfer to punish branch swaps on visible strands
    "full_topo": {
        "seg_len_w": 1.0,
        "cloud_chamfer_w": 0.5,
    },
    # dual_local + tracking prior: previous estimate feeds per-point
    # distance channel + per-query spatial anchor, plus per-point hand
    # distance and an endpoint-identity aux head — targets the
    # direction-flip failure observed on shape/combined episodes
    "dual_local_track": {
        "use_mask": False, "use_history": False,
        "predict_velocity": False, "predict_future": False,
        "predict_logvar": False,
        "local_k": 16,
        "use_prev_pos": True, "hand_point_feat": True, "end_head_w": 0.1,
    },
    # EPN3D-style baseline proxy: dual-camera single frame + per-point
    # voting branch gated against the attention-decoder regression
    "vote_attn": {
        "use_mask": False, "use_history": False,
        "predict_velocity": False, "predict_future": False,
        "predict_logvar": False,
        "use_voting": True,
    },
}


def move(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def compute_loss(
    out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
    cfg: EstimatorConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = torch.zeros((), device=out["pos"].device)
    logs: dict[str, float] = {}
    if cfg.predict_logvar:
        nll = gaussian_nll(out["pos"], batch["node_pos"], out["logvar"])
        loss = loss + nll
        logs["nll"] = float(nll)
    l1 = (out["pos"] - batch["node_pos"]).abs().mean()
    loss = loss + l1
    logs["pos_l1"] = float(l1)
    if cfg.end_head_w > 0.0 and "end_logit" in out:
        # aux: which physical end is nearer the hand — gives endpoint
        # queries a supervised reason to encode hand-relative identity
        d0 = (batch["node_pos"][:, 0] - batch["hand_pos"]).norm(dim=-1)
        d1 = (batch["node_pos"][:, -1] - batch["hand_pos"]).norm(dim=-1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            out["end_logit"], (d0 < d1).float()
        )
        loss = loss + cfg.end_head_w * bce
        logs["end_bce"] = float(bce)
    if "arc_s" in out and "arc_s_gt" in batch:
        # dense arc-coordinate supervision: every valid cloud point learns
        # its position along the cable — the signal that sparse 14-node
        # regression cannot provide
        cloud = torch.cat(
            [batch["points_opst"][:, -1], batch["points_wrist"][:, -1]],
            dim=1,
        )
        valid = cloud[..., :3].abs().sum(-1) > 1e-6
        sgt = batch["arc_s_gt"].reshape(valid.shape)
        sl1 = ((out["arc_s"] - sgt).abs() * valid).sum() / valid.sum().clamp(
            min=1
        )
        loss = loss + cfg.arc_w * sl1
        logs["arc_l1"] = float(sl1)
    if "pos_vote" in out:
        lv = (out["pos_vote"] - batch["node_pos"]).abs().mean()
        loss = loss + 0.5 * lv
        logs["vote_l1"] = float(lv)
    if cfg.predict_velocity:
        lv = (out["vel"] - batch["node_vel"]).abs().mean()
        loss = loss + 0.5 * lv
        logs["vel_l1"] = float(lv)
    if cfg.predict_future:
        lf = (out["future_pos"] - batch["node_pos_future"]).abs().mean()
        loss = loss + lf
        logs["future_l1"] = float(lf)
    if cfg.seg_len_w > 0.0:
        # inextensible-DLO prior: adjacent-node spacing must match GT arc
        # lengths; wrong-branch/shortcut paths violate it locally
        for key, pred in (("pos", out["pos"]), ("future", out.get("future_pos"))):
            if pred is None:
                continue
            tgt = (
                batch["node_pos"] if key == "pos"
                else batch["node_pos_future"]
            )
            ls = (
                (pred[:, 1:] - pred[:, :-1]).norm(dim=-1)
                - (tgt[:, 1:] - tgt[:, :-1]).norm(dim=-1)
            ).abs().mean()
            loss = loss + cfg.seg_len_w * ls
            logs[f"seglen_{key}"] = float(ls)
    if cfg.cloud_chamfer_w > 0.0:
        # coverage term: every observed cloud point must be near the chain
        # *segments* (not just nodes) — catches branch swaps / shortcut
        # chords through a loop's interior that node-wise distance misses.
        # est->cloud direction is applied only to nodes labelled visible,
        # so occluded nodes are not dragged toward unrelated strands.
        cloud = torch.cat(
            [batch["points_opst"][:, -1], batch["points_wrist"][:, -1]],
            dim=1,
        )
        valid = cloud.abs().sum(-1) > 1e-6
        has_cloud = valid.any(-1)
        # point-to-segment distance: project each cloud point onto every
        # predicted segment, clamp to the segment, take the min
        a, b = out["pos"][:, :-1], out["pos"][:, 1:]        # (B,S,3)
        ab = b - a
        ap = cloud[:, :, None] - a[:, None]                 # (B,N,S,3)
        t = (ap * ab[:, None]).sum(-1) / (
            (ab * ab).sum(-1)[:, None].clamp(min=1e-9)
        )
        proj = a[:, None] + t.clamp(0.0, 1.0)[..., None] * ab[:, None]
        c2e = (cloud[:, :, None] - proj).norm(dim=-1).min(-1).values
        # hinge: only cloud points >~50 mm off the chain contribute —
        # makes the term a pure gross-shortcut fixer with zero gradient
        # on well-covered regions instead of a co-objective
        c2e = (c2e - 0.1).clamp(min=0.0)
        c2e = (c2e * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        cham = (c2e * has_cloud).sum() / has_cloud.sum().clamp(min=1)
        d2 = torch.cdist(out["pos"], cloud).masked_fill(
            ~valid[:, None, :], 1e9
        )
        vis = batch.get("node_vis")
        if cfg.cloud_e2c_w > 0.0 and vis is not None and float(vis.sum()) > 0:
            vis_any = vis.float().amax(1)  # visible in either camera
            e2c = d2.min(-1).values.clamp(max=1.0)
            w = vis_any * has_cloud[:, None]
            cham = cham + cfg.cloud_e2c_w * (e2c * w).sum() / w.sum().clamp(min=1)
        loss = loss + cfg.cloud_chamfer_w * cham
        logs["chamfer"] = float(cham)
    return loss, logs


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
    scale: float,
) -> dict[str, float]:
    model.eval()
    tot = {"n": 0, "pos_sq": 0.0, "vis_sq": 0.0, "occ_sq": 0.0,
           "n_vis": 0, "n_occ": 0, "vel_sq": 0.0, "fut_sq": 0.0}
    for batch in loader:
        batch = move(batch, device)
        out = model(batch)
        err = out["pos"] - batch["node_pos"]
        sq = err.pow(2).sum(-1)  # (B,M)
        vis = batch["node_vis"].amax(dim=1) > 0.5  # (B,M) visible anywhere
        tot["n"] += sq.numel()
        tot["pos_sq"] += float(sq.sum())
        tot["n_vis"] += int(vis.sum())
        tot["n_occ"] += int((~vis).sum())
        tot["vis_sq"] += float(sq[vis].sum())
        tot["occ_sq"] += float(sq[~vis].sum())
        if "vel" in out:
            tot["vel_sq"] += float(
                (out["vel"] - batch["node_vel"]).pow(2).sum(-1).sum()
            )
        if "future_pos" in out:
            tot["fut_sq"] += float(
                (out["future_pos"] - batch["node_pos_future"])
                .pow(2).sum(-1).sum()
            )
    m = scale * 1000.0  # back to mm
    return {
        "mpne_mm": m * float(np.sqrt(tot["pos_sq"] / max(tot["n"], 1))),
        "vis_mm": m * float(np.sqrt(tot["vis_sq"] / max(tot["n_vis"], 1))),
        "occ_mm": m * float(np.sqrt(tot["occ_sq"] / max(tot["n_occ"], 1))),
        "occ_frac": tot["n_occ"] / max(tot["n"], 1),
        "vel_rmse_ms": scale * float(np.sqrt(tot["vel_sq"] / max(tot["n"], 1)))
        if tot["vel_sq"] else 0.0,
        "future_mm": m * float(np.sqrt(tot["fut_sq"] / max(tot["n"], 1)))
        if tot["fut_sq"] else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--packed-dir", type=Path, default=None,
                   help="use pack_dataset.py output instead of raw npz")
    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--exclude-seeds-files", nargs="*", type=Path,
                   default=None,
                   help="episodes whose seed appears in these files are "
                        "excluded from train/val (shared test split)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--prev-noise", type=float, default=0.0,
                   help="Gaussian std (normalized units) added to the "
                        "prev_pos prior on TRAIN samples — simulates "
                        "self-feedback error")
    p.add_argument("--prev-flip", type=float, default=0.0,
                   help="probability of reversing prev_pos on TRAIN samples")
    p.add_argument("--cov-warmup", type=int, default=0,
                   help="epochs before the cloud-coverage term switches "
                        "on — lets pos_l1 establish node assignment first")
    p.add_argument("--xsamp", type=float, default=0.0,
                   help="extra sampling weight for self-crossing frames "
                        "(non-adjacent GT nodes closer than one arc "
                        "step); 0 disables oversampling")
    p.add_argument("--future-steps", type=int, default=8)
    p.add_argument("--node-count", type=int, default=14)
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    exclude = set()
    for sf in args.exclude_seeds_files or []:
        exclude |= {
            s.strip() for s in Path(sf).read_text().splitlines()
            if s.strip()
        }
    if args.packed_dir is not None:
        # collect seeds from per-chunk seeds.npy; drop test seeds
        rng = np.random.default_rng(args.seed)
        all_seeds: set[int] = set()
        for sf in Path(args.packed_dir).glob("*__part*.seeds.npy"):
            all_seeds |= set(np.load(sf, mmap_mode="r").tolist())
        nontest = sorted(s for s in all_seeds if str(s) not in exclude)
        n_val = max(1, int(len(nontest) * 0.15))
        perm = rng.permutation(len(nontest))
        val_seeds = {str(nontest[i]) for i in perm[:n_val]}
        train_seeds = {str(nontest[i]) for i in perm[n_val:]}
        train_ds = PackedDLODataset(
            args.packed_dir, history=args.history, seed_filter=train_seeds,
            prev_noise_std=args.prev_noise, prev_flip_p=args.prev_flip,
        )
        val_ds = PackedDLODataset(
            args.packed_dir, history=args.history, seed_filter=val_seeds,
            center=train_ds.center, scale=train_ds.scale,
        )
        print(f"packed train={len(train_ds)} val={len(val_ds)} "
              f"({len(train_seeds)}/{len(val_seeds)} eps)")
    else:
        train_files, val_files = split_episodes(
            args.data_root, scenarios=args.scenarios, exclude_seeds=exclude
        )
        if args.max_episodes:
            train_files = train_files[: args.max_episodes]
        print(f"train eps={len(train_files)} val eps={len(val_files)}")

        train_ds = DLOFrameDataset(
            train_files, future_steps=args.future_steps,
            history=args.history, node_count=args.node_count,
        )
        val_ds = DLOFrameDataset(
            val_files, future_steps=args.future_steps,
            history=args.history, node_count=args.node_count,
            center=train_ds.center, scale=train_ds.scale,
        )
    print(f"train frames={len(train_ds)} val frames={len(val_ds)} "
          f"center={train_ds.center} scale={train_ds.scale}")

    train_sampler = None
    if args.xsamp > 0.0 and isinstance(train_ds, PackedDLODataset):
        # crossing-frame oversampling: a frame is "hard" when two
        # non-adjacent GT nodes sit closer than the mean arc step —
        # i.e. a self-crossing or strand approach, exactly the frames
        # where strand assignment fails
        w = np.ones(len(train_ds), dtype=np.float64)
        ii, jj = np.indices((train_ds.arrays[0]["pos14"].shape[1],) * 2)
        adj = np.abs(ii - jj) < 3
        for idx, (ai, r) in enumerate(train_ds.row_of):
            p = np.asarray(train_ds.arrays[ai]["pos14"][r], np.float32)
            mean_seg = np.linalg.norm(p[1:] - p[:-1], axis=1).mean()
            d = np.linalg.norm(p[:, None] - p[None], axis=-1)
            d[adj] = np.inf
            if d.min() < mean_seg:
                w[idx] += args.xsamp
        train_sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(w), num_samples=len(w), replacement=True
        )
        print(f"xsamp: {(w > 1).mean() * 100:.1f}% frames boosted "
              f"x{1.0 + args.xsamp:.0f}")
    train_ld = DataLoader(
        train_ds, batch_size=args.batch,
        shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_ld = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=max(2, args.workers // 2), pin_memory=True,
    )

    cfg = EstimatorConfig(
        node_count=args.node_count, **VARIANTS[args.variant]
    )
    model = DLOStateEstimator(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs
    )

    args.out.mkdir(parents=True, exist_ok=True)
    log_f = open(args.out / "train_log.jsonl", "a", buffering=1)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        agg: dict[str, float] = {}
        nb = 0
        cfg_ep = dataclasses.replace(
            cfg,
            cloud_chamfer_w=(
                cfg.cloud_chamfer_w if epoch >= args.cov_warmup else 0.0
            ),
        )
        for batch in train_ld:
            batch = move(batch, device)
            out = model(batch)
            loss, logs = compute_loss(out, batch, cfg_ep)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        sched.step()
        metrics = evaluate(model, val_ld, device, train_ds.scale)
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
                    "config": vars(cfg),
                    "center": train_ds.center,
                    "scale": train_ds.scale,
                    "metrics": metrics,
                },
                args.out / "checkpoint_best.pt",
            )
    log_f.close()
    print(f"best_val_mpne_mm={best:.2f}")


if __name__ == "__main__":
    main()
