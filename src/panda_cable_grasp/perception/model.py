"""Occlusion-conditioned DLO state estimator.

Inputs (per frame):
  * partial cable point clouds from opposite + wrist cameras (world frame,
    normalized)
  * robot-occluder silhouette masks per camera (known-geometry prior)
  * hand pose (position + quaternion)

Outputs per ordered node (M):
  * position (3), velocity (3), log-variance (1), future position (3) at a
    fixed lead time.

Variants are controlled by flags so the same class implements the
regression-only baseline (``use_mask=False, use_wrist=False,
use_decoder_attn=False``) and the full model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class PointEncoder(nn.Module):
    """Lightweight PointNet-style per-point encoder."""

    def __init__(self, out_dim: int = 256, in_dim: int = 3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, out_dim, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, N, 3) -> (B, N, C) per-point features."""
        return self.mlp(points.transpose(1, 2)).transpose(1, 2)


class LocalPointEncoder(nn.Module):
    """PointNet++/DGCNN-style encoder: per-point MLP + kNN edge features.

    Each point aggregates its k nearest neighbours in xyz via an edge
    MLP on [x_i, x_j - x_i], giving tokens local-geometric context that
    a per-point MLP cannot see (e.g. which strand continues where at a
    self-crossing).
    """

    def __init__(self, out_dim: int = 256, k: int = 16, in_dim: int = 3) -> None:
        super().__init__()
        self.k = k
        self.point_mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_dim // 2, 1),
            nn.ReLU(inplace=True),
        )
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(2 * in_dim, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim // 2, 1),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, 1),
            nn.ReLU(inplace=True),
        )

    def _knn_idx(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3) -> (B, N, k) neighbour indices, chunked to
        # keep the pairwise distance matrix bounded for large B*N.
        idxs = []
        for p in xyz.split(512):
            d = torch.cdist(p, p)
            idxs.append(d.topk(self.k + 1, largest=False).indices[..., 1:])
        return torch.cat(idxs, dim=0)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, C_in); neighbourhoods are computed on xyz only,
        # extra channels (hand distance, prev-chain distance) ride along
        # as per-point / per-edge features.
        B, N, C = points.shape
        pf = self.point_mlp(points.transpose(1, 2))          # (B, C/2, N)
        knn = self._knn_idx(points[..., :3])                 # (B, N, k)
        nb = torch.gather(
            points, 1,
            knn.reshape(B, -1, 1).expand(-1, -1, C),
        ).view(B, N, self.k, C)                              # (B, N, k, C)
        edge = torch.cat(
            [points.unsqueeze(2).expand(-1, -1, self.k, -1), nb - points.unsqueeze(2)],
            dim=-1,
        )                                                    # (B, N, k, 2C)
        ef = self.edge_mlp(
            edge.permute(0, 3, 1, 2)
        ).amax(dim=-1)                                       # (B, C/2, N)
        return self.fuse(torch.cat([pf, ef], dim=1)).transpose(1, 2)


class MaskEncoder(nn.Module):
    """Tiny CNN over the downsampled occluder mask (90x120)."""

    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.net(mask.unsqueeze(1))


class NodeDecoder(nn.Module):
    """M learned node queries cross-attending to point features."""

    def __init__(
        self, node_count: int, feat_dim: int = 256, cond_dim: int = 128
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(node_count, feat_dim) * 0.02)
        self.cond_proj = nn.Linear(cond_dim, feat_dim)
        self.attn = nn.MultiheadAttention(
            feat_dim, num_heads=4, batch_first=True
        )
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(
        self, point_feats: torch.Tensor, cond: torch.Tensor,
        query_extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """point_feats: (B, Ntot, C); cond: (B, cond_dim)."""
        B = point_feats.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        q = q + self.cond_proj(cond).unsqueeze(1)
        if query_extra is not None:
            q = q + query_extra
        attended, _ = self.attn(q, point_feats, point_feats)
        q = self.norm1(q + attended)
        return self.norm2(q + self.ffn(q))


@dataclass
class EstimatorConfig:
    node_count: int = 14
    point_count: int = 384
    feat_dim: int = 256
    cond_dim: int = 192
    use_wrist: bool = True
    use_mask: bool = True
    use_decoder_attn: bool = True
    use_history: bool = True
    use_future_hand: bool = True
    predict_velocity: bool = True
    predict_future: bool = True
    predict_logvar: bool = True
    seg_len_w: float = 0.0
    cloud_chamfer_w: float = 0.0
    cloud_e2c_w: float = 1.0  # weight of the visible-node -> cloud pull
    # inside the coverage term; 0 disables it (it can drag a node onto
    # the wrong strand at self-crossings)
    use_voting: bool = False
    local_k: int = 0  # >0 -> use LocalPointEncoder with k neighbours
    use_prev_pos: bool = False  # tracking prior: previous 14-node estimate
    hand_point_feat: bool = False  # per-point |x - hand| channel
    end_head_w: float = 0.0  # >0 -> aux head: which end is nearer the hand
    # dense arc-length parameterisation: every cloud point predicts its
    # arc coordinate s in [0,1] along the cable; the chain is extracted
    # by soft-binning points along s (connectivity becomes an explicit,
    # densely-supervised quantity instead of implicit in the decoder)
    arc_head: bool = False
    arc_w: float = 1.0        # weight of the per-point s supervision
    # True: output chain is extracted by soft-binning predicted s
    # (connectivity explicit). False: direct node head stays the output
    # and the arc head is a dense auxiliary task only.
    arc_extract: bool = True


class DLOStateEstimator(nn.Module):
    def __init__(self, config: EstimatorConfig | None = None) -> None:
        super().__init__()
        self.config = config or EstimatorConfig()
        cfg = self.config
        in_dim = 3 + int(cfg.hand_point_feat) + int(cfg.use_prev_pos)
        enc = (
            (lambda: LocalPointEncoder(cfg.feat_dim, k=cfg.local_k, in_dim=in_dim))
            if cfg.local_k > 0
            else (lambda: PointEncoder(cfg.feat_dim, in_dim=in_dim))
        )
        self.opst_encoder = enc()
        self.wrist_encoder = enc() if cfg.use_wrist else None
        self.frame_embed = (
            nn.Embedding(8, cfg.feat_dim) if cfg.use_history else None
        )
        self.mask_encoder = (
            MaskEncoder(64) if cfg.use_mask else None
        )
        self.hand_mlp = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU(inplace=True), nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        cond_in = cfg.feat_dim  # fused cloud feature
        self.cloud_fuse = nn.Sequential(
            nn.Linear(cfg.feat_dim * (2 if cfg.use_wrist else 1), cfg.feat_dim),
            nn.ReLU(inplace=True),
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(
                cond_in + 64 + (64 if cfg.use_mask else 0),
                cfg.cond_dim,
            ),
            nn.ReLU(inplace=True),
        )
        self.decoder = NodeDecoder(cfg.node_count, cfg.feat_dim, cfg.cond_dim)
        # tracking prior: embed the previous frame's estimate per node so
        # query i attends near its own previous position (endpoints keep
        # their physical identity across frames instead of flipping).
        self.prev_embed = (
            nn.Sequential(
                nn.Linear(4, 64), nn.ReLU(inplace=True),
                nn.Linear(64, cfg.feat_dim),
            )
            if cfg.use_prev_pos
            else None
        )
        # aux head forcing endpoint queries to encode hand-relative identity
        self.end_head = (
            nn.Linear(cfg.feat_dim, 1) if cfg.end_head_w > 0 else None
        )
        head_dim = cfg.feat_dim if cfg.use_decoder_attn else cfg.cond_dim
        out_per_node = 3
        if cfg.predict_velocity:
            out_per_node += 3
        if cfg.predict_logvar:
            out_per_node += 1
        self.head = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, cfg.node_count * out_per_node),
        )
        self.vote_head = None
        self.vote_mix = None
        self.arc_head_net = (
            nn.Sequential(
                nn.Linear(cfg.feat_dim, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )
            if cfg.arc_head
            else None
        )
        if cfg.use_voting:
            # EPN3D-style point-to-node voting: every visible point votes an
            # offset to each node plus a confidence; node position is the
            # confidence-weighted mean of point+offset, blended with the
            # decoder-regressed position by a per-node gate.
            self.vote_head = nn.Sequential(
                nn.Linear(cfg.feat_dim, 128), nn.ReLU(inplace=True),
                nn.Linear(128, cfg.node_count * 4),
            )
            self.vote_mix = nn.Linear(cfg.feat_dim, 1)
        self.future_head = None
        if cfg.predict_future:
            fut_in = head_dim
            self.future_hand_mlp = (
                nn.Sequential(
                    nn.Linear(10, 64), nn.ReLU(inplace=True),
                    nn.Linear(64, 64), nn.ReLU(inplace=True),
                )
                if cfg.use_future_hand
                else None
            )
            if cfg.use_future_hand:
                fut_in += 64
            self.future_head = nn.Sequential(
                nn.Linear(fut_in, head_dim),
                nn.ReLU(inplace=True),
                nn.Linear(head_dim, 3),
            )

    def _encode_camera(
        self, encoder: PointEncoder, frames: torch.Tensor
    ) -> torch.Tensor:
        """frames: (B,K,N,Cin) -> tokens (B,K*N,C) with history embeddings."""
        B, K, N, Cin = frames.shape
        feats = encoder(frames.reshape(B * K, N, Cin))
        if self.frame_embed is not None:
            # per-row frame age: rows are ordered (batch, oldest..newest);
            # the newest frame always gets embedding index 0
            age = (K - 1 - torch.arange(K, device=frames.device)).repeat(B)
            feats = feats + self.frame_embed(age).unsqueeze(1)
        return feats.reshape(B, K * N, -1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cfg = self.config
        pts_o = batch["points_opst"]
        pts_w = batch["points_wrist"]
        if not cfg.use_history:
            pts_o = pts_o[:, -1:]
            pts_w = pts_w[:, -1:]
        if cfg.hand_point_feat or cfg.use_prev_pos:
            # extra per-point channels appended on the CURRENT frame only:
            # |x - hand| and distance to the previous chain estimate.
            def _aug(pts: torch.Tensor) -> torch.Tensor:
                chans = [pts]
                if cfg.hand_point_feat:
                    chans.append(
                        (pts - batch["hand_pos"][:, None, None, :])
                        .norm(dim=-1, keepdim=True)
                    )
                if cfg.use_prev_pos:
                    prev = batch["prev_pos"]              # (B, M, 3)
                    flat = pts[..., :3].reshape(pts.shape[0], -1, 3)
                    d = torch.cdist(flat, prev).min(dim=-1).values
                    chans.append(
                        d.reshape(*pts.shape[:-1], 1)
                    )                                     # (B, K, N, 1)
                return torch.cat(chans, dim=-1)

            cur_o = _aug(pts_o[:, -1:])
            cur_w = _aug(pts_w[:, -1:])
            pad_o = pts_o[:, :-1]
            pad_w = pts_w[:, :-1]
            zpad = (
                lambda x, n: torch.cat(
                    [x, x.new_zeros(*x.shape[:-1], n)], dim=-1
                )
            )
            n_extra = cur_o.shape[-1] - 3
            pts_o = torch.cat([zpad(pad_o, n_extra), cur_o], dim=1)
            pts_w = torch.cat([zpad(pad_w, n_extra), cur_w], dim=1)
        tok_o = self._encode_camera(self.opst_encoder, pts_o)
        glob_o = tok_o[:, -cfg.point_count:].amax(dim=1)  # newest frame
        if cfg.use_wrist:
            tok_w = self._encode_camera(self.wrist_encoder, pts_w)
            glob_w = tok_w[:, -cfg.point_count:].amax(dim=1)
            cloud = self.cloud_fuse(torch.cat([glob_o, glob_w], dim=-1))
            all_feats = torch.cat([tok_o, tok_w], dim=1)
        else:
            cloud = self.cloud_fuse(glob_o)
            all_feats = tok_o
        cond_parts = [cloud, self.hand_mlp(
            torch.cat([batch["hand_pos"], batch["hand_quat"]], dim=-1)
        )]
        if cfg.use_mask:
            mo = self.mask_encoder(batch["robot_mask_opst"])
            mw = self.mask_encoder(batch["robot_mask_wrist"])
            cond_parts.append(mo + mw)
        cond = self.cond_fuse(torch.cat(cond_parts, dim=-1))

        q_extra = None
        if self.prev_embed is not None:
            hp = batch["has_prev"].view(-1, 1, 1).expand(
                -1, cfg.node_count, -1
            )
            q_extra = self.prev_embed(
                torch.cat([batch["prev_pos"], hp], dim=-1)
            )                                             # (B, M, C)
        if cfg.use_decoder_attn:
            node_feats = self.decoder(all_feats, cond, q_extra)  # (B,M,C)
        else:
            node_feats = cond.unsqueeze(1).expand(-1, cfg.node_count, -1)
        B, M = cond.shape[0], cfg.node_count
        out = self.head(node_feats).view(B, M, -1)
        result = {"pos": out[..., :3]}
        if self.end_head is not None:
            result["end_logit"] = self.end_head(
                node_feats[:, 0] - node_feats[:, -1]
            ).squeeze(-1)                                 # (B,)
        if cfg.use_voting:
            toks = [tok_o[:, -cfg.point_count:]]
            xyzs = [pts_o[:, -1]]
            if cfg.use_wrist:
                toks.append(tok_w[:, -cfg.point_count:])
                xyzs.append(pts_w[:, -1])
            tok = torch.cat(toks, dim=1)        # (B, Ntot, C)
            xyz = torch.cat(xyzs, dim=1)        # (B, Ntot, 3)
            vo = self.vote_head(tok).view(B, tok.shape[1], M, 4)
            offs, conf = vo[..., :3], vo[..., 3]
            pad = xyz.abs().sum(-1) <= 1e-6     # zero-padded slots
            conf = conf.masked_fill(pad[:, :, None], -1e4)
            w = conf.softmax(dim=1)             # over points
            pos_vote = (
                w.unsqueeze(-1) * (xyz[:, :, None, :] + offs)
            ).sum(1)
            g = torch.sigmoid(self.vote_mix(node_feats))  # (B,M,1)
            result["pos_vote"] = pos_vote
            result["pos"] = g * result["pos"] + (1.0 - g) * pos_vote
        if self.arc_head_net is not None:
            toks = [tok_o[:, -cfg.point_count:]]
            xyzs = [pts_o[:, -1, :, :3]]
            if cfg.use_wrist:
                toks.append(tok_w[:, -cfg.point_count:])
                xyzs.append(pts_w[:, -1, :, :3])
            tok = torch.cat(toks, dim=1)          # (B, Ntot, C)
            xyz = torch.cat(xyzs, dim=1)          # (B, Ntot, 3)
            arc_s = torch.sigmoid(
                self.arc_head_net(tok)
            ).squeeze(-1)                         # (B, Ntot)
            result["arc_s"] = arc_s
            if not cfg.arc_extract:
                idx = 3
                if cfg.predict_velocity:
                    result["vel"] = out[..., idx:idx + 3]
                    idx += 3
                if cfg.predict_logvar:
                    result["logvar"] = out[..., idx:idx + 3]
                if cfg.predict_future:
                    result["future_pos"] = self.future_head(node_feats)
                if self.end_head is not None:
                    result["end_logit"] = self.end_head(
                        node_feats
                    ).squeeze(-1)
                return result
            # soft-bin extraction: node j = weighted centroid of points
            # with predicted s in bin j (bin width 1/M, edges softened)
            M = cfg.node_count
            centers = torch.linspace(0.0, 1.0, M, device=xyz.device)
            half, tau = 0.5 / M, 0.03
            w = torch.sigmoid(
                (arc_s[:, :, None] - (centers - half)[None, None]) / tau
            ) * torch.sigmoid(
                ((centers + half)[None, None] - arc_s[:, :, None]) / tau
            )                                     # (B, Ntot, M)
            pad = xyz.abs().sum(-1) <= 1e-6
            w = w.masked_fill(pad[:, :, None], 0.0)
            num = (w[:, :, :, None] * xyz[:, :, None, :]).sum(1)
            den = w.sum(1)[..., None].clamp(min=1e-4)
            result["pos"] = num / den             # (B, M, 3)
        idx = 3
        if cfg.predict_velocity:
            result["vel"] = out[..., idx:idx + 3]
            idx += 3
        if cfg.predict_logvar:
            result["logvar"] = out[..., idx:idx + 1]
            idx += 1
        if cfg.predict_future:
            fin = node_feats
            if cfg.use_future_hand:
                fh = self.future_hand_mlp(torch.cat([
                    batch["hand_pos_future"] - batch["hand_pos"],
                    batch["hand_pos_future"],
                    batch["hand_quat_future"],
                ], dim=-1))
                fin = torch.cat(
                    [fin, fh.unsqueeze(1).expand(-1, M, -1)], dim=-1
                )
            result["future_pos"] = self.future_head(fin).view(B, M, 3)
        return result


def gaussian_nll(
    pred: torch.Tensor, target: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """logvar broadcast per node; returns mean NLL."""
    var = logvar.exp()
    return 0.5 * (
        ((pred - target) ** 2) / var + logvar
    ).mean()
