"""Dump per-frame TrackDLO predictions for video rendering.

Replays one raw episode, runs TrackDLOTracker on the re-rendered
RGB-D stream, and writes a single npz with world-frame arrays:

    cloud: (T, N, 3)   observed segmented points (world)
    gt:    (T, M, 3)   MuJoCo cable body positions (world)
    pred:  (T, K, 3)   TrackDLO node chain (world)
    ok:    (T,) bool   tracking_ok flag

Usage:
    python tools/perception/dump_trackdlo_ep.py \
        --trajectory <ep>/trajectory.npz --model <...>/id_static.mjb \
        --out dump.npz --stride 1 --max-frames 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from panda_cable_grasp.runtime import configure_mujoco_runtime  # noqa: E402

configure_mujoco_runtime()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

_TRACKDLO = Path("/data1/hxai/mujoco/visual_dlo_eval_20260906/trackdlo_standalone/src")
if str(_TRACKDLO) not in sys.path:
    sys.path.insert(0, str(_TRACKDLO))

from trackdlo_standalone import TrackDLOConfig, TrackDLOTracker  # noqa: E402

CAMERA = "dynamicvla_opst_camera"
HSV_LOWER = (112, 180, 80)
HSV_UPPER = (130, 255, 255)


def name_ids(model: mujoco.MjModel, prefix: str) -> np.ndarray:
    return np.asarray(
        [
            i
            for i in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(prefix)
        ],
        dtype=np.int64,
    )


def camera_matrix(w: int, h: int, fovy_deg: float) -> np.ndarray:
    f = 0.5 * h / np.tan(np.deg2rad(fovy_deg) * 0.5)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


def camera_to_world(pts_cam: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    # world_to_camera: p_cam = (p_w - pos) @ (R_opt @ cam_mat.T).T
    # inverse: p_w = pos + cam_mat @ R_opt @ p_cam ; R_opt = diag(1,-1,-1)
    pts = np.asarray(pts_cam, dtype=np.float64)
    fixed = pts * np.array([1.0, -1.0, -1.0])
    return cam_pos + fixed @ cam_mat.T


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--num-nodes", type=int, default=40)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    args = p.parse_args()

    with np.load(args.trajectory, allow_pickle=True) as traj:
        states = np.asarray(traj["states"], dtype=np.float64)
        spec = int(traj["state_spec"])

    model = mujoco.MjModel.from_binary_path(str(args.model))
    data = mujoco.MjData(model)
    cable_bodies = name_ids(model, "cable")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    intrinsics = camera_matrix(args.width, args.height, float(model.cam_fovy[cam_id]))
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    tracker = TrackDLOTracker(
        intrinsics,
        TrackDLOConfig(
            num_nodes=args.num_nodes,
            hsv_lower=HSV_LOWER,
            hsv_upper=HSV_UPPER,
        ),
    )

    idx = np.arange(0, len(states), args.stride)
    if args.max_frames:
        idx = idx[: args.max_frames]

    clouds, gts, preds, oks = [], [], [], []
    for si in idx:
        mujoco.mj_setState(model, data, states[si], spec)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=CAMERA)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=CAMERA)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        res = tracker.update(rgb, depth)
        cam_pos = data.cam_xpos[cam_id].copy()
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
        gt_w = data.xpos[cable_bodies].astype(np.float64)

        pred_w = camera_to_world(res.nodes_camera, cam_pos, cam_mat)
        cloud_cam = getattr(res, "observed_points_camera", None)
        cloud_w = (
            camera_to_world(cloud_cam, cam_pos, cam_mat)
            if cloud_cam is not None and len(cloud_cam)
            else np.zeros((0, 3))
        )
        clouds.append(cloud_w.astype(np.float32))
        gts.append(gt_w.astype(np.float32))
        preds.append(pred_w.astype(np.float32))
        oks.append(bool(res.tracking_ok))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cloud=np.asarray(clouds, dtype=object),
        gt=np.asarray(gts),
        pred=np.asarray(preds),
        ok=np.asarray(oks),
        allow_pickle=True,
    )
    print(f"saved {len(gts)} frames -> {args.out}  ok={sum(oks)}/{len(oks)}")


if __name__ == "__main__":
    main()
