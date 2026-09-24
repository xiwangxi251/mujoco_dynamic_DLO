"""Evaluate the standalone TrackDLO baseline on our perception episodes.

Replays each raw episode, re-renders the fixed opposite camera (RGB+D),
feeds it to TrackDLOTracker, and scores the returned camera-frame node
chain against MuJoCo ground truth using trackdlo_standalone.metrics.

Usage:
    python tools/perception/run_trackdlo_eval.py \
        --raw-root <raw>/scripted --render-root <rendered_root> \
        --scenario id_static --seeds-file test_seeds_id_static.txt \
        --episodes 20 --stride 2 --out-dir <out>/trackdlo/id_static
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

from panda_cable_grasp.runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import mujoco
import numpy as np

_TRACKDLO = Path(
    "/data1/hxai/mujoco/visual_dlo_eval_20260906/trackdlo_standalone/src"
)
if str(_TRACKDLO) not in sys.path:
    sys.path.insert(0, str(_TRACKDLO))

from trackdlo_standalone import TrackDLOConfig, TrackDLOTracker  # noqa: E402
from trackdlo_standalone.metrics import frame_metrics  # noqa: E402

CAMERA = "dynamicvla_opst_camera"
HSV_LOWER = (112, 180, 80)
HSV_UPPER = (130, 255, 255)


def name_ids(model: mujoco.MjModel, prefix: str) -> np.ndarray:
    ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if name.startswith(prefix):
            ids.append(i)
    return np.asarray(ids, dtype=np.int64)


def camera_matrix(w: int, h: int, fovy_deg: float) -> np.ndarray:
    f = 0.5 * h / np.tan(np.deg2rad(fovy_deg) * 0.5)
    return np.array(
        [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def world_to_camera(
    pts_world: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray
) -> np.ndarray:
    world_to_optical = np.diag([1.0, -1.0, -1.0]) @ cam_mat.T
    return (pts_world - cam_pos) @ world_to_optical.T


def run_episode(
    episode_dir: Path,
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    cable_bodies: np.ndarray,
    cam_id: int,
    intrinsics: np.ndarray,
    stride: int,
    max_frames: int | None,
    config: TrackDLOConfig,
) -> list[dict]:
    with np.load(episode_dir / "trajectory.npz", allow_pickle=True) as traj:
        states = np.asarray(traj["states"], dtype=np.float64)
        spec = int(traj["state_spec"])
    data = mujoco.MjData(model)
    idx = np.arange(0, len(states), stride)
    if max_frames:
        idx = idx[:max_frames]
    tracker = TrackDLOTracker(intrinsics, config)
    rows = []
    for row, si in enumerate(idx):
        mujoco.mj_setState(model, data, states[si], spec)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=CAMERA)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=CAMERA)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        result = tracker.update(rgb, depth)
        truth_cam = world_to_camera(
            data.xpos[cable_bodies].astype(np.float64),
            data.cam_xpos[cam_id],
            data.cam_xmat[cam_id].reshape(3, 3),
        )
        m = frame_metrics(result.nodes_camera, truth_cam)
        m.update(
            frame=row,
            tracking_ok=result.tracking_ok,
            reinitialized=result.reinitialized,
            tracking_ms=result.tracking_ms,
            total_ms=result.total_ms,
            visible_nodes=len(result.visible_nodes),
            failure_reason=result.failure_reason or "",
        )
        # store chains for post-hoc strict-ordered / flip aggregation
        # (camera frame is fine: MPNE is rigid-transform invariant)
        nc = result.nodes_camera
        m["nodes_cam"] = (
            np.asarray(nc, dtype=np.float64)
            if nc is not None and np.asarray(nc).shape == (config.num_nodes, 3)
            else np.full((config.num_nodes, 3), np.nan)
        )
        m["truth_cam"] = truth_cam
        rows.append(m)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--render-root", type=Path, required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--seeds-file", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--num-nodes", type=int, default=40)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--cfg", nargs="*", default=None,
                   help="TrackDLOConfig overrides, e.g. --cfg beta=0.6")
    args = p.parse_args()

    overrides = {}
    for kv in args.cfg or []:
        k, _, v = kv.partition("=")
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v

    scenario_dir = args.raw_root / args.scenario
    episodes = sorted(
        d for d in scenario_dir.iterdir() if (d / "trajectory.npz").exists()
    )
    if args.seeds_file is not None:
        wanted = {
            s.strip() for s in args.seeds_file.read_text().splitlines()
            if s.strip()
        }
        episodes = [
            d for d in episodes
            if d.name.removeprefix("seed_") in wanted or d.name in wanted
        ]
    episodes = episodes[args.shard :: args.num_shards][: args.episodes]

    model_path = (
        args.render_root / args.scenario / "models" / f"{args.scenario}.mjb"
    )
    model = mujoco.MjModel.from_binary_path(str(model_path))
    cable_bodies = name_ids(model, "cable")
    cam_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA
    )
    intrinsics = camera_matrix(
        args.width, args.height, float(model.cam_fovy[cam_id])
    )
    renderer = mujoco.Renderer(
        model, height=args.height, width=args.width
    )
    config = TrackDLOConfig(
        num_nodes=args.num_nodes,
        hsv_lower=HSV_LOWER,
        hsv_upper=HSV_UPPER,
        **overrides,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    t0 = time.time()
    for i, ep in enumerate(episodes):
        try:
            rows = run_episode(
                ep, model, renderer, cable_bodies, cam_id, intrinsics,
                args.stride, args.max_frames, config,
            )
            for r in rows:
                r["episode"] = ep.name
            all_rows.extend(rows)
            ep_ok = sum(r["tracking_ok"] for r in rows)
            err = np.mean([r["ordered_error_m"] for r in rows if r["tracking_ok"]])
            print(
                f"[{i + 1}/{len(episodes)}] {ep.name}: "
                f"ok={ep_ok}/{len(rows)} ordered_err={err * 1000:.1f}mm",
                flush=True,
            )
        except Exception as exc:
            print(f"FAIL {ep.name}: {exc}", flush=True)
    renderer.close()

    ok = [r for r in all_rows if r["tracking_ok"]]
    reasons: dict[str, int] = {}
    for r in all_rows:
        if not r["tracking_ok"]:
            reasons[r["failure_reason"] or "unknown"] = (
                reasons.get(r["failure_reason"] or "unknown", 0) + 1
            )
    summary = {
        "scenario": args.scenario,
        "episodes": len(episodes),
        "frames": len(all_rows),
        "tracking_ok": len(ok),
        "tracking_ok_frac": len(ok) / max(len(all_rows), 1),
        "failure_reasons": reasons,
        "wall_s": time.time() - t0,
        "mean_ordered_err_m": float(
            np.mean([r["ordered_error_m"] for r in ok])
        ) if ok else float("nan"),
        "p90_ordered_err_m": float(
            np.percentile([r["ordered_error_m"] for r in ok], 90)
        ) if ok else float("nan"),
        "mean_frame_err_m": float(
            np.mean([r["frame_error_m"] for r in ok])
        ) if ok else float("nan"),
        "mean_tracking_ms": float(
            np.mean([r["tracking_ms"] for r in all_rows])
        ) if all_rows else float("nan"),
        "p95_tracking_ms": float(
            np.percentile([r["tracking_ms"] for r in all_rows], 95)
        ) if all_rows else float("nan"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        args.out_dir / "rows.npz",
        **{
            k: np.asarray([r[k] for r in all_rows])
            for k in all_rows[0]
            if k != "episode"
        },
        episode=np.asarray([r["episode"] for r in all_rows]),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
