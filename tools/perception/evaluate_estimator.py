"""Evaluate a trained DLO state estimator vs simple baselines.

Baselines (all operate on the same stored episodes):
  * persistence:    predict node_pos[t] = GT node_pos[t-1] (oracle dynamics
                    reference for how much the cable moves per frame)
  * const_vel:      GT pos[t-1] + vel[t-1]*dt (oracle kinematic reference)
  * visible_hold:   occluded nodes keep last GT-visible position, visible
                    nodes get GT (upper bound for track-and-coast methods)

Metrics: per-node L2 (mm), visible/occluded split (occluded = invisible in
both cameras), velocity RMSE, future-Δt error, per-scenario breakdown.
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

import numpy as np
import torch
from torch.utils.data import DataLoader

from panda_cable_grasp.perception.dataset import DLOFrameDataset
from panda_cable_grasp.perception.model import DLOStateEstimator, EstimatorConfig


def move(batch, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def node_rmse_mm(pred: torch.Tensor, gt: torch.Tensor, scale: float) -> float:
    return float(
        torch.sqrt((pred - gt).pow(2).sum(-1).mean()) * scale * 1000.0
    )


@torch.no_grad()
def run_eval(model, loader, device, scale: float, m: int):
    """Collect predictions + targets across the loader."""
    if model is not None:
        model.eval()
    rows = []
    for batch in loader:
        batch_d = move(batch, device)
        out = model(batch_d) if model is not None else None
        vis_any = batch_d["node_vis"].amax(dim=1) > 0.5
        rec = {
            "pos": batch_d["node_pos"].cpu().numpy(),
            "vis_any": vis_any.cpu().numpy(),
        }
        if out is not None:
            rec["pred_pos"] = out["pos"].cpu().numpy()
            if "vel" in out:
                rec["pred_vel"] = out["vel"].cpu().numpy()
                rec["gt_vel"] = batch_d["node_vel"].cpu().numpy()
            if "future_pos" in out:
                rec["pred_future"] = out["future_pos"].cpu().numpy()
                rec["gt_future"] = batch_d["node_pos_future"].cpu().numpy()
            if "logvar" in out:
                rec["pred_std"] = np.exp(
                    0.5 * out["logvar"].cpu().numpy()
                )
        # task-relevant node: node closest to the hand at each frame
        near = (batch_d["node_pos"] - batch_d["hand_pos"].unsqueeze(1)
                ).norm(dim=-1).argmin(dim=1)  # (B,)
        rec["near_node"] = near.cpu().numpy()
        rows.append(rec)
    return rows


def split_err(pred: np.ndarray, gt: np.ndarray, vis: np.ndarray):
    """Return (all, visible, occluded) RMSE in dataset units."""
    sq = ((pred - gt) ** 2).sum(-1)
    v = vis.astype(bool)
    return (
        float(np.sqrt(sq.mean())),
        float(np.sqrt(sq[v].mean())) if v.any() else float("nan"),
        float(np.sqrt(sq[~v].mean())) if (~v).any() else float("nan"),
    )


@torch.no_grad()
def measure_latency(model, loader, device, iters: int = 50) -> dict:
    batch = move(next(iter(loader)), device)
    single = {k: v[:1] for k, v in batch.items() if torch.is_tensor(v)}
    for bs, tag in ((1, single), (batch[list(batch)[0]].shape[0], batch)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(tag)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ms = (time.perf_counter() - t0) / iters * 1000
        if tag is single:
            out1 = ms
        else:
            outb = ms
    return {"batch1_ms": round(out1, 2), "batch_ms": round(outb, 2)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--packed-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--seeds-files", nargs="*", type=Path, default=None,
                   help="restrict to seeds listed in these files")
    p.add_argument("--limit-episodes", type=int, default=40)
    p.add_argument("--node-count", type=int, default=14)
    p.add_argument("--future-steps", type=int, default=8)
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dump-npz", type=Path, default=None,
                   help="save per-frame predictions/GT for curve plots")
    args = p.parse_args()

    wanted: set[str] | None = None
    if args.seeds_files:
        wanted = set()
        for sf in args.seeds_files:
            wanted |= {
                s.strip() for s in Path(sf).read_text().splitlines()
                if s.strip()
            }
    if args.packed_dir is not None:
        from panda_cable_grasp.perception.dataset import PackedDLODataset

        ds = PackedDLODataset(
            args.packed_dir, history=args.history, seed_filter=wanted,
            scenarios=args.scenarios,
        )
        files = None
    else:
        files = sorted(Path(args.data_root).rglob("seed_*.npz"))
        if args.scenarios:
            keep = set(args.scenarios)
            files = [f for f in files if f.parent.name in keep]
        if wanted:
            files = [
                f for f in files if f.stem.removeprefix("seed_") in wanted
            ]
        files = files[: args.limit_episodes]
        ds = DLOFrameDataset(
            files, future_steps=args.future_steps, history=args.history,
            node_count=args.node_count,
        )
    scale = ds.scale
    device = torch.device(args.device)

    model = None
    if args.checkpoint is not None:
        payload = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
        raw_cfg = {
            k: v for k, v in payload["config"].items()
            if k in EstimatorConfig.__dataclass_fields__
        }
        # checkpoints saved before a flag existed -> default it off
        for flag in ("use_future_hand", "use_history"):
            raw_cfg.setdefault(flag, False)
        cfg = EstimatorConfig(**raw_cfg)
        model = DLOStateEstimator(cfg).to(device)
        model.load_state_dict(payload["model"])
        print("loaded checkpoint; train-time metrics:", payload.get("metrics"))

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=4)
    rows = run_eval(model, loader, device, scale, args.node_count)

    # ------- model metrics -------
    report: dict = {"n_frames": len(ds), "scenarios": args.scenarios}
    if model is not None:
        pred = np.concatenate([r["pred_pos"] for r in rows])
        gt = np.concatenate([r["pos"] for r in rows])
        vis = np.concatenate([r["vis_any"] for r in rows])
        a, v, o = split_err(pred, gt, vis)
        report["model"] = {
            "mpne_mm": a * scale * 1000,
            "vis_mm": v * scale * 1000,
            "occ_mm": o * scale * 1000,
        }
        if "pred_vel" in rows[0]:
            pv = np.concatenate([r["pred_vel"] for r in rows])
            gv = np.concatenate([r["gt_vel"] for r in rows])
            report["model"]["vel_rmse_ms"] = float(
                np.sqrt(((pv - gv) ** 2).sum(-1).mean())
            ) * scale
        if "pred_future" in rows[0]:
            pf = np.concatenate([r["pred_future"] for r in rows])
            gf = np.concatenate([r["gt_future"] for r in rows])
            report["model"]["future_mpne_mm"] = float(
                np.sqrt(((pf - gf) ** 2).sum(-1).mean())
            ) * scale * 1000
        report["model"]["latency_ms"] = measure_latency(
            model, loader, device
        )
        # interception-node errors (node nearest the hand per frame)
        near = np.concatenate([r["near_node"] for r in rows])
        pn = np.take_along_axis(pred, near[:, None, None], 1).squeeze(1)
        gn = np.take_along_axis(gt, near[:, None, None], 1).squeeze(1)
        report["model"]["near_node_mm"] = float(
            np.sqrt(((pn - gn) ** 2).sum(-1).mean())
        ) * scale * 1000
        if "pred_future" in rows[0]:
            pnf = np.take_along_axis(pf, near[:, None, None], 1).squeeze(1)
            gnf = np.take_along_axis(gf, near[:, None, None], 1).squeeze(1)
            report["model"]["near_node_future_mm"] = float(
                np.sqrt(((pnf - gnf) ** 2).sum(-1).mean())
            ) * scale * 1000

    # ------- baselines on raw episodes (need frame t-1; recompute) -------
    base_stats: dict[str, list] = {
        "persistence": [], "const_vel": [], "visible_hold": [],
        "persistence_vis": [], "persistence_occ": [],
        "const_vel_occ": [], "visible_hold_occ": [],
        "fut_stay": [], "fut_const_vel": [], "fut_stay_occ": [],
        "fut_const_vel_occ": [], "fut_stay_near": [], "fut_cv_near": [],
    }
    # iterate raw episodes directly for temporal baselines
    from panda_cable_grasp.perception.dataset import resample_polyline_weighted

    def add_baseline_row(pos_t, pos_p, vel_p, dt, pos_f, vel_t, lead,
                         vis_any, hand):
        pers = pos_p
        cv = pos_p + vel_p * dt
        hold = np.where(vis_any[:, None], pos_t, pos_p)
        base_stats["persistence"].append(((pers - pos_t) ** 2).sum(-1))
        base_stats["const_vel"].append(((cv - pos_t) ** 2).sum(-1))
        base_stats["visible_hold"].append(((hold - pos_t) ** 2).sum(-1))
        base_stats["persistence_vis"].append(
            ((pers - pos_t) ** 2).sum(-1)[vis_any]
        )
        base_stats["persistence_occ"].append(
            ((pers - pos_t) ** 2).sum(-1)[~vis_any]
        )
        base_stats["const_vel_occ"].append(
            ((cv - pos_t) ** 2).sum(-1)[~vis_any]
        )
        base_stats["visible_hold_occ"].append(
            ((hold - pos_t) ** 2).sum(-1)[~vis_any]
        )
        if pos_f is not None:
            stay = pos_t
            cvf = pos_t + vel_t * lead
            base_stats["fut_stay"].append(((stay - pos_f) ** 2).sum(-1))
            base_stats["fut_const_vel"].append(((cvf - pos_f) ** 2).sum(-1))
            base_stats["fut_stay_occ"].append(
                ((stay - pos_f) ** 2).sum(-1)[~vis_any]
            )
            base_stats["fut_const_vel_occ"].append(
                ((cvf - pos_f) ** 2).sum(-1)[~vis_any]
            )
            near = int(np.argmin(np.linalg.norm(pos_t - hand, axis=1)))
            base_stats["fut_stay_near"].append(
                np.array([(stay - pos_f)[near] ** 2]).sum(-1)
            )
            base_stats["fut_cv_near"].append(
                np.array([(cvf - pos_f)[near] ** 2]).sum(-1)
            )

    if args.packed_dir is not None:
        # baselines from packed arrays: group consecutive rows by episode
        base_scale = 1.0  # packed pos14 are raw metres
        for z in ds.arrays:
            lead = np.asarray(z["lead"])
            ep = np.asarray(z["ep_id"])
            frm = np.asarray(z["frame_i"])
            pos = np.asarray(z["pos14"])
            vel = np.asarray(z["vel14"])
            fut = np.asarray(z["fut14"])
            vis = np.asarray(z["vis14"]).astype(bool)
            hand = np.asarray(z["hand"])[:, :3]
            keep_rows = np.ones(len(lead), bool)
            if wanted:
                keep_rows &= np.isin(
                    np.asarray(z["seeds"]),
                    np.asarray([int(s) for s in wanted]),
                )
            for r in np.flatnonzero(keep_rows):
                if r == 0 or ep[r] != ep[r - 1] or frm[r] != frm[r - 1] + 1:
                    continue  # first kept frame of the episode
                vis_any = vis[r].any(0)
                fut_ok = lead[r] > 0
                add_baseline_row(
                    pos[r], pos[r - 1], vel[r - 1],
                    0.04,  # stride-2 sampling at 50 Hz states
                    fut[r] if fut_ok else None, vel[r], float(lead[r]),
                    vis_any, hand[r],
                )
    else:
        base_scale = 1.0  # raw ep node_pos are metres
        for f in files:
            with np.load(f, allow_pickle=True) as z:
                ep = {k: z[k] for k in z.files}
            n = len(ep["time"])
            for t in range(1, n):
                pos_t, _, frac = resample_polyline_weighted(
                    ep["node_pos"][t], None, args.node_count
                )
                pos_p, vel_p, _ = resample_polyline_weighted(
                    ep["node_pos"][t - 1], ep["node_vel"][t - 1],
                    args.node_count,
                )
                dt = ep["time"][t] - ep["time"][t - 1]
                vis_src = ep["node_vis"][t]
                lo = np.floor(frac).astype(int)
                hi = np.clip(lo + 1, 0, len(frac) - 1)
                vis = vis_src[:, lo] & vis_src[:, hi]
                vis_any = vis.any(0)
                pos_f = vel_t = lead = None
                fu = min(t + args.future_steps, n - 1)
                if fu - t >= max(1, args.future_steps * 0.5):
                    pos_f, _, _ = resample_polyline_weighted(
                        ep["node_pos"][fu], None, args.node_count
                    )
                    lead = ep["time"][fu] - ep["time"][t]
                    _, vel_t, _ = resample_polyline_weighted(
                        ep["node_pos"][t], ep["node_vel"][t],
                        args.node_count,
                    )
                add_baseline_row(
                    pos_t, pos_p, vel_p, dt, pos_f, vel_t, lead,
                    vis_any, ep["hand_pos"][t],
                )
    for k, arr in base_stats.items():
        if arr:
            cat = np.concatenate(arr)
            # baseline errors are in world metres -> x1000 for mm
            report[k] = float(np.sqrt(cat.mean())) * base_scale * 1000

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    if args.dump_npz:
        args.dump_npz.parent.mkdir(parents=True, exist_ok=True)
        merged = {
            k: np.concatenate([r[k] for r in rows])
            for k in rows[0]
        }
        if args.packed_dir is not None:
            scen = []
            occ = []
            for ai, r in ds.row_of:
                z = ds.arrays[ai]
                scen.append(z["_scenario"])
                occ.append(1.0 - float(np.asarray(z["vis14"][r]).mean()))
            merged["scenario"] = np.asarray(scen)
            merged["occ_rate"] = np.asarray(occ, dtype=np.float32)
        else:
            scenarios = np.asarray(
                [str(ds._ep(e)["scenario"]) for e, _ in ds.index]
            )
            occ_rate = np.asarray(
                [
                    1.0 - float(ds._ep(e)["node_vis"][f].mean())
                    for e, f in ds.index
                ],
                dtype=np.float32,
            )
            merged["scenario"] = scenarios
            merged["occ_rate"] = occ_rate
        np.savez_compressed(args.dump_npz, **merged)


if __name__ == "__main__":
    main()
