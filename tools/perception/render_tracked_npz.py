"""Render a tracked top-down XY video from per-frame dump files.

Two input layouts are supported:

1. ``--dump episode.npz`` — a single npz written by dump_trackdlo_ep.py
   with ``cloud`` (object array of (Ni,3)), ``gt`` (T,M,3),
   ``pred`` (T,K,3) and ``ok`` (T,).
2. ``--dump dir/`` — a directory of per-frame ``*.npz`` each holding
   ``cloud`` (N,3), ``gt`` (M,3) and ``kp`` (K,3), as written by
   MP2CDLO ``eval_packed.py --dump-dir``.

The video mirrors the OccDyn tracked-view layout: left panel shows
cloud + thin green GT + estimate, right panel shows the observed cloud
only; both share identical auto-fit, EMA-smoothed bounds with a 10 cm
scale bar. No policy overlays.
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

C_BG = (36, 36, 36)
C_GRID = (48, 48, 48)
C_CLOUD = (150, 150, 150)
C_GT = (60, 200, 60)
C_EST = (60, 60, 230)
C_TXT = (220, 220, 220)


class TrackedView:
    """Auto-fit XY view; bounds from cloud percentiles + GT + estimate."""

    def __init__(self, w: int = 720, h: int = 540) -> None:
        self.w, self.h = w, h
        self._bounds = None

    def fit(self, *arrays: np.ndarray) -> None:
        cores = []
        for arr in arrays:
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim != 2 or len(a) == 0:
                continue
            a = a[np.isfinite(a[:, :2]).all(1), :2]
            if len(a) > 50:
                a = np.percentile(a, [2, 98], axis=0)
            if len(a):
                cores.append(a)
        if not cores:
            return
        xy = np.concatenate(cores, 0)
        low, high = xy.min(0), xy.max(0)
        span = np.maximum(high - low, [0.15, 0.15])
        ctr = 0.5 * (low + high)
        span = span + 2 * np.maximum(0.04, 0.15 * span)
        ratio = self.w / self.h
        if span[0] / span[1] < ratio:
            span[0] = span[1] * ratio
        else:
            span[1] = span[0] / ratio
        bounds = np.array(
            [
                ctr[0] - span[0] / 2,
                ctr[0] + span[0] / 2,
                ctr[1] - span[1] / 2,
                ctr[1] + span[1] / 2,
            ]
        )
        self._bounds = (
            bounds if self._bounds is None else 0.75 * self._bounds + 0.25 * bounds
        )

    def px(self, pts: np.ndarray) -> np.ndarray:
        b = self._bounds
        xy = np.asarray(pts, dtype=np.float64)[:, :2]
        out = np.empty((len(xy), 2))
        out[:, 0] = (xy[:, 0] - b[0]) / (b[1] - b[0]) * (self.w - 1)
        out[:, 1] = (1 - (xy[:, 1] - b[2]) / (b[3] - b[2])) * (self.h - 1)
        return np.round(out).astype(np.int32)

    def draw(
        self,
        cloud: np.ndarray,
        gt: np.ndarray | None,
        pred: np.ndarray | None,
        cloud_only: bool = False,
        node_r: int = 0,
    ) -> np.ndarray:
        panel = np.full((self.h, self.w, 3), C_BG, dtype=np.uint8)
        if self._bounds is None:
            return panel
        for f in np.linspace(0, 1, 6):
            x = int(f * (self.w - 1))
            y = int(f * (self.h - 1))
            cv2.line(panel, (x, 0), (x, self.h - 1), C_GRID, 1)
            cv2.line(panel, (0, y), (self.w - 1, y), C_GRID, 1)
        if len(cloud):
            for p in self.px(cloud):
                if 0 <= p[0] < self.w and 0 <= p[1] < self.h:
                    cv2.circle(panel, tuple(p), 2, C_CLOUD, -1)
        if not cloud_only:
            if gt is not None and len(gt):
                cv2.polylines(panel, [self.px(gt)], False, C_GT, 2, cv2.LINE_AA)
            if pred is not None and len(pred):
                pp = self.px(pred)
                cv2.polylines(panel, [pp], False, C_EST, 2, cv2.LINE_AA)
                if node_r:
                    for p in pp:
                        cv2.circle(panel, tuple(p), node_r, C_EST, -1)
        m_per_px = (self._bounds[1] - self._bounds[0]) / self.w
        bar_px = max(8, int(0.10 / m_per_px))
        cv2.line(
            panel, (20, self.h - 26), (20 + bar_px, self.h - 26), (230, 230, 230), 2
        )
        cv2.putText(
            panel, "10cm", (20, self.h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (200, 200, 200), 1,
        )
        return panel


def resample_arc(points: np.ndarray, num: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(points[:1], num, axis=0)
    t = np.linspace(0, s[-1], num)
    return np.stack([np.interp(t, s, points[:, c]) for c in range(3)], axis=1)


def ordered_err_mm(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Ordered node error; resamples pred to len(gt), allows flip."""
    if pred is None or gt is None or len(pred) < 2 or len(gt) < 2:
        return None
    p = resample_arc(pred, len(gt)) if len(pred) != len(gt) else pred
    e1 = np.sqrt(((p - gt) ** 2).sum(-1)).mean()
    e2 = np.sqrt(((p[::-1] - gt) ** 2).sum(-1)).mean()
    return min(e1, e2) * 1000


def load_frames(dump: str):
    """Yield (cloud, gt, pred, ok) per frame."""
    if os.path.isdir(dump):
        files = sorted(glob.glob(os.path.join(dump, "*.npz")))
        for fp in files:
            z = np.load(fp, allow_pickle=True)
            pred = z["pred"] if "pred" in z else z["kp"]
            yield (
                np.asarray(z["cloud"], np.float64),
                np.asarray(z["gt"], np.float64),
                np.asarray(pred, np.float64),
                bool(z["ok"]) if "ok" in z else True,
            )
    else:
        z = np.load(dump, allow_pickle=True)
        n = len(z["gt"])
        oks = z["ok"] if "ok" in z else np.ones(n, bool)
        for i in range(n):
            yield (
                np.asarray(z["cloud"][i], np.float64),
                np.asarray(z["gt"][i], np.float64),
                np.asarray(z["pred"][i], np.float64),
                bool(oks[i]),
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="npz file or dir of npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--node-r", type=int, default=4)
    args = ap.parse_args()

    view = TrackedView()
    vw = None
    n = 0
    errs = []
    for cloud, gt, pred, ok in load_frames(args.dump):
        if args.max_frames and n >= args.max_frames:
            break
        view.fit(cloud, gt, pred)
        left = view.draw(cloud, gt, pred if ok else pred,
                         node_r=args.node_r)
        right = view.draw(cloud, None, None, cloud_only=True)
        frame = np.concatenate([left, right], axis=1)
        err = ordered_err_mm(pred, gt)
        if err is not None:
            errs.append(err)
        label = args.title
        status = "" if ok else "  [TRACK LOST]"
        cv2.putText(
            frame, f"{label}  f{n}" + (f"  err={err:.0f}mm" if err else "")
            + status,
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TXT, 1,
        )
        if vw is None:
            vw = cv2.VideoWriter(
                args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                (frame.shape[1], frame.shape[0]),
            )
        vw.write(frame)
        n += 1
    if vw is not None:
        vw.release()
    if errs:
        print(
            f"wrote {args.out}  frames={n}  "
            f"ordered_err mean={np.mean(errs):.1f}mm "
            f"med={np.median(errs):.1f}mm"
        )
    else:
        print(f"wrote {args.out}  frames={n}")


if __name__ == "__main__":
    main()
