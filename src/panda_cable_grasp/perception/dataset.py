"""Torch dataset over re-rendered DLO perception episodes.

Each episode npz (produced by tools/perception/build_dataset.py) stores
per-frame partial point clouds from the opposite + wrist cameras, robot
occluder masks, ordered cable node positions/velocities and per-node
visibility flags.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def optical_to_world(
    points: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray
) -> np.ndarray:
    """points: (N,3) optical frame (x right, y down, z forward)."""
    rot = cam_mat @ np.diag([1.0, -1.0, -1.0])
    return points @ rot.T + cam_pos


def resample_polyline_weighted(
    points: np.ndarray, values: np.ndarray | None, count: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Resample a polyline to ``count`` arc-length-uniform nodes.

    Returns (positions, interpolated_values, bracketing_index_fraction).
    ``bracketing_index_fraction`` maps each output node to a fractional index
    into the input chain, useful for resampling flags like visibility.
    """
    points = np.asarray(points, dtype=np.float64)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    if cumulative[-1] < 1e-9:
        cumulative[-1] = 1.0
    s = np.linspace(0.0, cumulative[-1], count)
    pos = np.column_stack(
        [np.interp(s, cumulative, points[:, ax]) for ax in range(3)]
    )
    frac = np.interp(s, cumulative, np.arange(len(points)))
    vals = None
    if values is not None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            vals = np.interp(s, cumulative, values)
        else:
            vals = np.column_stack(
                [
                    np.interp(s, cumulative, values[:, ax])
                    for ax in range(values.shape[1])
                ]
            )
    return pos, vals, frac


class DLOFrameDataset(Dataset):
    """Per-frame samples; future targets are shifted inside each episode."""

    def __init__(
        self,
        episode_files: list[Path],
        future_steps: int = 8,
        history: int = 1,
        node_count: int = 14,
        center: np.ndarray | None = None,
        scale: float = 0.5,
        min_future_frac: float = 0.5,
    ) -> None:
        self.future_steps = int(future_steps)
        self.history = max(1, int(history))
        self.node_count = int(node_count)
        self.scale = float(scale)
        self.min_future_frac = float(min_future_frac)
        self.files = list(episode_files)
        # lazy per-worker episode cache (npz decompresses to ~15-20MB each;
        # eager loading of >1k episodes under forked DataLoader workers OOMs)
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        self._cache_order: list[int] = []
        self.cache_size = 24
        self.index: list[tuple[int, int]] = []
        frame_counts = []
        for ep_i, path in enumerate(self.files):
            with np.load(path, allow_pickle=True) as z:
                n = len(z["time"])
                node_pos = z["node_pos"]
            frame_counts.append((n, node_pos.mean(axis=(0, 1))))
            for f in range(self.history - 1, n):
                fu = f + self.future_steps
                # keep frames whose future target still has >= min fraction
                # of the requested lead time (drop only the tail)
                if fu < n or (n - 1 - f) >= self.future_steps * min_future_frac:
                    self.index.append((ep_i, f))
        if center is None:
            tot = sum(n for n, _ in frame_counts)
            center = (
                sum(c * n for n, c in frame_counts) / max(tot, 1)
            )
        self.center = np.asarray(center, dtype=np.float32)

    def _ep(self, ep_i: int) -> dict[str, np.ndarray]:
        ep = self._cache.get(ep_i)
        if ep is None:
            with np.load(self.files[ep_i], allow_pickle=True) as z:
                ep = {k: z[k] for k in z.files}
            self._cache[ep_i] = ep
            self._cache_order.append(ep_i)
            if len(self._cache_order) > self.cache_size:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
        return ep

    def __len__(self) -> int:
        return len(self.index)

    def _frame_cloud(self, ep, f: int) -> tuple[np.ndarray, np.ndarray]:
        clouds = []
        for ci in range(2):
            pts = ep["points"][f, ci]
            world = optical_to_world(
                pts.astype(np.float64),
                ep["cam_pos"][f, ci].astype(np.float64),
                ep["cam_mat"][f, ci].astype(np.float64),
            )
            clouds.append(world.astype(np.float32))
        return clouds[0], clouds[1]

    def _targets(
        self, ep, f: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos, vel, frac = resample_polyline_weighted(
            ep["node_pos"][f], ep["node_vel"][f], self.node_count
        )
        lo = np.floor(frac).astype(int)
        hi = np.clip(lo + 1, 0, len(frac) - 1)
        vis_src = ep["node_vis"][f]  # (2,40)
        # a resampled node counts as visible from a camera only when both
        # bracketing source nodes are visible there
        vis = vis_src[:, lo] & vis_src[:, hi]
        return (
            pos.astype(np.float32),
            vel.astype(np.float32),
            vis,
        )

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ep_i, f = self.index[i]
        ep = self._ep(ep_i)
        n = len(ep["time"])
        fu = min(f + self.future_steps, n - 1)
        lead = float(ep["time"][fu] - ep["time"][f])

        hist_o, hist_w = [], []
        for k in range(self.history):
            # index 0 = oldest, index -1 = current frame
            fo, fw = self._frame_cloud(ep, f - (self.history - 1 - k))
            hist_o.append(fo)
            hist_w.append(fw)
        hist_o = np.stack(hist_o)
        hist_w = np.stack(hist_w)

        pos, vel, vis = self._targets(ep, f)
        fpos, _, _ = self._targets(ep, fu)

        return {
            "points_opst": torch.from_numpy(
                (hist_o - self.center) / self.scale
            ),  # (K,384,3)
            "points_wrist": torch.from_numpy(
                (hist_w - self.center) / self.scale
            ),
            "robot_mask_opst": torch.from_numpy(
                ep["robot_mask"][f, 0].astype(np.float32)
            ),
            "robot_mask_wrist": torch.from_numpy(
                ep["robot_mask"][f, 1].astype(np.float32)
            ),
            "hand_pos": torch.from_numpy(
                (ep["hand_pos"][f] - self.center) / self.scale
            ),
            "hand_quat": torch.from_numpy(ep["hand_quat"][f].astype(np.float32)),
            "node_pos": torch.from_numpy(
                (pos - self.center) / self.scale
            ),
            "node_vel": torch.from_numpy(vel / self.scale),
            "node_pos_future": torch.from_numpy(
                (fpos - self.center) / self.scale
            ),
            "hand_pos_future": torch.from_numpy(
                (ep["hand_pos"][fu] - self.center) / self.scale
            ),
            "hand_quat_future": torch.from_numpy(
                ep["hand_quat"][fu].astype(np.float32)
            ),
            "lead_time": torch.tensor(lead, dtype=torch.float32),
            "node_vis": torch.from_numpy(vis.astype(np.float32)),  # (2,M)
        }


PACKED_KEYS = (
    "points", "masks", "pos14", "vel14", "fut14", "vis14",
    "hand", "hand_fut", "lead", "ep_id", "frame_i", "seeds",
)


class PackedDLODataset(Dataset):
    """Fast mmap-backed dataset over pack_dataset.py chunk output.

    Each chunk is a set of raw ``<scenario>__partNN.<key>.npy`` files that
    mmap directly (no per-sample decompression). History is gathered by
    stepping back ``frame_i`` within the same episode (consecutive rows
    are consecutive frames of one episode).
    """

    def __init__(
        self,
        packed_dir: str | Path,
        history: int = 1,
        center: np.ndarray | None = None,
        scale: float = 0.5,
        seed_filter: set[str] | None = None,
        exclude_seed_files: bool = False,
        scenarios: list[str] | None = None,
        prev_noise_std: float = 0.0,
        prev_flip_p: float = 0.0,
    ) -> None:
        self.history = max(1, int(history))
        self.scale = float(scale)
        # training-time corruption of the prev-estimate prior: simulates
        # self-feedback error so the model cannot learn to just copy it
        self.prev_noise_std = float(prev_noise_std)
        self.prev_flip_p = float(prev_flip_p)
        self._rng = np.random.default_rng(0)
        self.arrays: list[dict[str, np.ndarray]] = []
        self.row_of: list[tuple[int, int]] = []  # (chunk_i, row)
        filt = (
            np.asarray([int(s) for s in seed_filter], dtype=np.int64)
            if seed_filter is not None
            else None
        )
        packed_dir = Path(packed_dir)
        anchors = sorted(packed_dir.glob("*__part*.points.npy"))
        ai = 0
        for path in anchors:
            scen = path.name.split("__part")[0]
            if scenarios is not None and scen not in set(scenarios):
                continue
            base = str(path)[: -len(".points.npy")]
            z = {
                k: np.load(f"{base}.{k}.npy", mmap_mode="r")
                for k in PACKED_KEYS
            }
            z["_scenario"] = scen
            keep = np.ones(len(z["lead"]), dtype=bool)
            if filt is not None:
                seeds = np.asarray(z["seeds"])  # per-row seed
                keep &= (
                    np.isin(seeds, filt)
                    if not exclude_seed_files
                    else ~np.isin(seeds, filt)
                )
            self.arrays.append(z)
            for r in np.flatnonzero(keep):
                self.row_of.append((ai, int(r)))
            ai += 1
        if center is None:
            c = np.zeros(3)
            tot = 0
            for z in self.arrays:
                c += z["pos14"][:].mean(axis=(0, 1)) * len(z["lead"])
                tot += len(z["lead"])
            center = c / max(tot, 1)
        self.center = np.asarray(center, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.row_of)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ai, r = self.row_of[i]
        z = self.arrays[ai]
        f = int(z["frame_i"][r])
        # index 0 = oldest, index -1 = current; pad by repeating frame 0
        ks = [r - min(self.history - 1 - k, f) for k in range(self.history)]
        pts = np.stack([z["points"][rr].astype(np.float32) for rr in ks])
        # previous frame's GT nodes as tracking prior (zeros when absent)
        prev = np.zeros_like(z["pos14"][r], dtype=np.float32)
        has_prev = 0.0
        if (
            r > 0
            and z["ep_id"][r - 1] == z["ep_id"][r]
            and int(z["frame_i"][r - 1]) == f - 1
        ):
            prev = (z["pos14"][r - 1] - self.center) / self.scale
            has_prev = 1.0
            if self.prev_flip_p > 0 and self._rng.random() < self.prev_flip_p:
                prev = prev[::-1].copy()
            if self.prev_noise_std > 0:
                prev = prev + self._rng.normal(
                    0.0, self.prev_noise_std, prev.shape
                ).astype(np.float32)
        return {
            "points_opst": torch.from_numpy(
                (pts[:, 0] - self.center) / self.scale
            ),
            "points_wrist": torch.from_numpy(
                (pts[:, 1] - self.center) / self.scale
            ),
            "robot_mask_opst": torch.from_numpy(
                z["masks"][r, 0].astype(np.float32)
            ),
            "robot_mask_wrist": torch.from_numpy(
                z["masks"][r, 1].astype(np.float32)
            ),
            "hand_pos": torch.from_numpy(
                (z["hand"][r, :3] - self.center) / self.scale
            ),
            "hand_quat": torch.from_numpy(z["hand"][r, 3:].copy()),
            "hand_pos_future": torch.from_numpy(
                (z["hand_fut"][r, :3] - self.center) / self.scale
            ),
            "hand_quat_future": torch.from_numpy(
                z["hand_fut"][r, 3:].copy()
            ),
            "node_pos": torch.from_numpy(
                (z["pos14"][r] - self.center) / self.scale
            ),
            "node_vel": torch.from_numpy(z["vel14"][r] / self.scale),
            "node_pos_future": torch.from_numpy(
                (z["fut14"][r] - self.center) / self.scale
            ),
            "lead_time": torch.tensor(float(z["lead"][r])),
            "node_vis": torch.from_numpy(z["vis14"][r].astype(np.float32)),
            "prev_pos": torch.from_numpy(prev.astype(np.float32)),
            "has_prev": torch.tensor(has_prev),
            # GT arc coordinate of each cloud point: project onto the GT
            # polyline and take the normalised arc length s in [0,1]
            "arc_s_gt": torch.from_numpy(
                _arc_s_of_points(z["points"][r], z["pos14"][r])
            ),
        }


def _arc_s_of_points(pts: np.ndarray, chain: np.ndarray) -> np.ndarray:
    """pts (2,N,3) world frame, chain (M,3) -> (2,N) float32 s in [0,1].

    Each cloud point is projected onto the closest GT segment; its arc
    coordinate is the cumulative arc length at the projection, divided by
    total chain length. Invalid (zero-padded) points get s=0.
    """
    pts = np.asarray(pts, dtype=np.float64)
    chain = np.asarray(chain, dtype=np.float64)
    seg = chain[1:] - chain[:-1]                        # (S,3)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = max(cum[-1], 1e-9)
    flat = pts.reshape(-1, 3)
    ap = flat[:, None, :] - chain[None, :-1, :]         # (P,S,3)
    t = np.clip(
        (ap * seg[None]).sum(-1) / np.maximum(seg_len ** 2, 1e-12)[None],
        0.0, 1.0,
    )
    proj = chain[None, :-1, :] + t[..., None] * seg[None]
    d = np.linalg.norm(flat[:, None, :] - proj, axis=-1)
    j = d.argmin(1)
    s = (cum[j] + t[np.arange(len(flat)), j] * seg_len[j]) / total
    s = s.reshape(pts.shape[:2]).astype(np.float32)
    s[np.abs(pts).sum(-1) <= 1e-6] = 0.0
    return s


def rasterize_torch(
    pts: torch.Tensor,
    hand: torch.Tensor,
    chain: torch.Tensor,
    X0: float = -0.20, X1: float = 1.15,
    Y0: float = -0.95, Y1: float = 0.85,
    W: int = 192, H: int = 256,
    samples_per_seg: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU batch rasteriser for the 2.5-D top-down image model.

    pts   (B,P,3) world-frame cloud (zero-padded rows are dropped)
    hand  (B,7)   gripper pose (only xy used)
    chain (B,M,3) GT nodes in metres

    Returns img (B,5,H,W): occupancy, z_mean, z_min, z_max, hand marker;
    tgt (B,3,H,W): GT skeleton mask, arc coord s, height z.  The GT
    polyline is drawn over the FULL cable including occluded parts, so
    mask/s/z supervision is dense and forces the net to inpaint hidden
    skeleton regions.
    """
    B, P, _ = pts.shape
    dev = pts.device
    f32 = torch.float32
    valid = pts.abs().sum(-1) > 1e-6                       # (B,P)
    ix = ((pts[..., 0] - X0) / (X1 - X0) * W).long().clamp(0, W - 1)
    iy = ((pts[..., 1] - Y0) / (Y1 - Y0) * H).long().clamp(0, H - 1)
    flat = (iy * W + ix)                                 # (B,P)
    vf = valid.to(f32)
    pz = pts[..., 2]
    cnt = torch.zeros(B, H * W, device=dev, dtype=f32)
    cnt.scatter_add_(1, flat, vf)
    zsum = torch.zeros(B, H * W, device=dev, dtype=f32)
    zsum.scatter_add_(1, flat, torch.where(valid, pz, torch.zeros_like(pz)))
    zmin = torch.full((B, H * W), float("inf"), device=dev, dtype=f32)
    zmin.scatter_reduce_(
        1, flat, torch.where(valid, pz, torch.full_like(pz, float("inf"))),
        reduce="amin",
    )
    zmax = torch.full((B, H * W), -float("inf"), device=dev, dtype=f32)
    zmax.scatter_reduce_(
        1, flat, torch.where(valid, pz, torch.full_like(pz, -float("inf"))),
        reduce="amax",
    )
    occ = cnt > 0
    img = torch.zeros(B, 5, H, W, device=dev, dtype=f32)
    img[:, 0] = (torch.log1p(cnt) / 3.0).view(B, H, W)
    img[:, 1] = torch.where(occ, zsum / cnt.clamp(min=1),
                            torch.zeros_like(cnt)).view(B, H, W)
    img[:, 2] = torch.where(occ, zmin, torch.zeros_like(cnt)).view(B, H, W)
    img[:, 3] = torch.where(occ, zmax, torch.zeros_like(cnt)).view(B, H, W)
    # hand marker: gaussian blob at gripper xy
    ys, xs = torch.meshgrid(
        torch.arange(H, device=dev, dtype=f32),
        torch.arange(W, device=dev, dtype=f32),
        indexing="ij",
    )
    hx = (hand[:, 0] - X0) / (X1 - X0) * W
    hy = (hand[:, 1] - Y0) / (Y1 - Y0) * H
    img[:, 4] = torch.exp(
        -((ys[None] - hy[:, None, None]) ** 2
          + (xs[None] - hx[:, None, None]) ** 2) / (2 * 6.0 ** 2)
    )

    # GT dense polyline: interpolate each of the M-1 segments
    M = chain.shape[1]
    seg = chain[:, 1:] - chain[:, :-1]                   # (B,M-1,3)
    seglen = seg.norm(dim=-1)                            # (B,M-1)
    cum = torch.cat(
        [torch.zeros(B, 1, device=dev, dtype=f32),
         seglen.cumsum(1)], dim=1
    )                                                    # (B,M)
    total = cum[:, -1:].clamp(min=1e-9)                  # (B,1)
    t = torch.linspace(0.0, 1.0, samples_per_seg, device=dev, dtype=f32)
    px = (chain[:, :-1, None] + t[None, None, :, None]
          * seg[:, :, None]).reshape(B, -1, 3)           # (B,(M-1)S,3)
    ps = ((cum[:, :-1, None] + t[None, None] * seglen[..., None])
          / total[..., None]).reshape(B, -1)             # (B,(M-1)S)
    jx = ((px[..., 0] - X0) / (X1 - X0) * W).long().clamp(0, W - 1)
    jy = ((px[..., 1] - Y0) / (Y1 - Y0) * H).long().clamp(0, H - 1)
    fj = jy * W + jx                                     # (B,(M-1)S)
    m_flat = torch.zeros(B, H * W, device=dev, dtype=f32)
    m_flat.scatter_(1, fj, torch.ones_like(fj, dtype=f32))
    s_flat = torch.zeros(B, H * W, device=dev, dtype=f32)
    s_flat.scatter_(1, fj, ps)
    z_flat = torch.zeros(B, H * W, device=dev, dtype=f32)
    z_flat.scatter_(1, fj, px[..., 2])
    tgt = torch.stack([m_flat, s_flat, z_flat], 1).view(B, 3, H, W)
    return img, tgt


class PackedDLOImageDataset(PackedDLODataset):
    """Raw-field dataset for the 2.5-D top-down image model.

    Per item returns only compact raw arrays (pts/hand/node_pos); the
    (5,H,W) raster and (3,H,W) GT maps are built on GPU in-batch by
    rasterize_torch — this keeps worker->main IPC at ~10 KB/item instead
    of ~1.6 MB/item (the rasterised tensors), which was the bottleneck.
    """

    # world window covered by the raster (1st-99th pct of GT chains)
    X0, X1 = -0.20, 1.15
    Y0, Y1 = -0.95, 0.85
    W, H = 192, 256  # x-cols, y-rows  (~7 mm/px)

    def __init__(self, *args, preload: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if preload:
            # mmap random access over ~7 GB is disk-bound; pull the three
            # arrays this dataset actually reads into RAM (shared COW by
            # forked dataloader workers).
            for z in self.arrays:
                for k in ("points", "pos14", "hand"):
                    z[k] = np.array(z[k])  # np.asarray keeps memmap!

    HIST_LAGS = (1, 4, 16)   # downsampled raw-observation history

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ai, r = self.row_of[i]
        z = self.arrays[ai]
        pts = np.asarray(z["points"][r], np.float32).reshape(-1, 3)
        hand = np.asarray(z["hand"][r], np.float32)
        chain = np.asarray(z["pos14"][r], np.float32)
        # previous frame's GT chain (same episode only) for history input
        prev_chain = np.zeros_like(chain)
        has_prev = np.float32(0.0)
        if (
            r > 0
            and z["ep_id"][r - 1] == z["ep_id"][r]
            and int(z["frame_i"][r - 1]) == int(z["frame_i"][r]) - 1
        ):
            prev_chain = np.asarray(z["pos14"][r - 1], np.float32)
            has_prev = np.float32(1.0)
        # raw point clouds at lags 1/4/16 — REAL observations, not model
        # output: zero exposure bias, carries pre-occlusion evidence
        hist = np.zeros((len(self.HIST_LAGS),) + pts.shape, np.float32)
        for k, lag in enumerate(self.HIST_LAGS):
            j = r - lag
            if (
                j >= 0
                and z["ep_id"][j] == z["ep_id"][r]
                and int(z["frame_i"][j]) == int(z["frame_i"][r]) - lag
            ):
                hist[k] = np.asarray(
                    z["points"][j], np.float32).reshape(-1, 3)
        return {
            "pts": torch.from_numpy(pts),
            "hand": torch.from_numpy(hand),
            "node_pos": torch.from_numpy(chain),
            "prev_chain": torch.from_numpy(prev_chain),
            "has_prev": torch.tensor(has_prev),
            "hist_pts": torch.from_numpy(hist),
        }


def split_episodes(
    root: Path, val_fraction: float = 0.15, seed: int = 0,
    scenarios: list[str] | None = None,
    exclude_seeds: set[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    files = sorted(Path(root).rglob("seed_*.npz"))
    if scenarios:
        keep = set(scenarios)
        files = [f for f in files if f.parent.name in keep]
    if exclude_seeds:
        files = [
            f for f in files
            if f.stem.removeprefix("seed_") not in exclude_seeds
        ]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_fraction))
    val_idx = set(perm[:n_val].tolist())
    train = [f for i, f in enumerate(files) if i not in val_idx]
    val = [f for i, f in enumerate(files) if i in val_idx]
    return train, val
