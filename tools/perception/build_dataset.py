"""Re-render recorded FULLPHYSICS episodes into a DLO perception dataset.

For every sampled state we store, per camera (opposite + wrist):
  * the deployed HSV-segmented partial point cloud (camera optical frame)
  * a downsampled robot-occluder silhouette mask (segmentation rendering)
  * a downsampled visible-cable mask (segmentation rendering)
and labels:
  * ordered cable node positions / linear velocities (world frame)
  * per-node visibility flags per camera
  * hand (link7-equivalent) pose, camera poses, intrinsics

Usage:
    python tools/perception/build_dataset.py \
        --raw-root <raw_dataset>/scripted \
        --render-root <rendered_dataset_root> \
        --scenario id_static --episodes 200 --stride 2 \
        --shard 0 --num-shards 1 \
        --out-dir <out>/<scenario>
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

import cv2
import mujoco
import numpy as np

from panda_cable_grasp.rl.pointcloud import (
    backproject_mask,
    camera_matrix,
    segment_hsv,
    voxel_downsample,
    fixed_farthest_point_sample,
)

CAMERAS = ("dynamicvla_opst_camera", "dynamicvla_wrist_camera")
HSV_LOWER = (112, 180, 80)
HSV_UPPER = (130, 255, 255)
MASK_W, MASK_H = 120, 90


def name_ids(model: mujoco.MjModel, obj: int, prefix: str) -> np.ndarray:
    ids = []
    for i in range(
        model.nbody if obj == mujoco.mjtObj.mjOBJ_BODY else model.ngeom
    ):
        name = mujoco.mj_id2name(model, obj, i) or ""
        if name.startswith(prefix):
            ids.append(i)
    return np.asarray(ids, dtype=np.int64)


def body_subtree(model: mujoco.MjModel, root_body: int) -> set[int]:
    out = set()
    for b in range(model.nbody):
        p = b
        while p != 0 and p != root_body:
            p = int(model.body_parentid[p])
        if p == root_body:
            out.add(b)
    return out


def render_cam(
    renderer: mujoco.Renderer, data: mujoco.MjData, camera: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    return rgb, depth, seg


def node_visibility(
    node_pos: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    margin: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Project nodes into the depth image; visible iff depth matches."""
    h, w = depth.shape
    world_to_optical = np.diag([1.0, -1.0, -1.0]) @ cam_mat.T
    rel = (node_pos - cam_pos) @ world_to_optical.T
    z = np.maximum(rel[:, 2], 1e-9)
    u = intrinsics[0, 0] * rel[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * rel[:, 1] / z + intrinsics[1, 2]
    inside = (rel[:, 2] > 0.01) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    vis = np.zeros(len(node_pos), dtype=bool)
    uv = np.full((len(node_pos), 2), -1.0, dtype=np.float32)
    if inside.any():
        idx = np.flatnonzero(inside)
        ui = np.clip(np.round(u[idx]).astype(int), 0, w - 1)
        vi = np.clip(np.round(v[idx]).astype(int), 0, h - 1)
        vis[idx] = np.abs(depth[vi, ui] - rel[idx, 2]) < margin
        uv[idx] = np.stack([u[idx], v[idx]], axis=1)
    return vis, uv


def downsample_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8), (MASK_W, MASK_H), interpolation=cv2.INTER_NEAREST
    )


def process_episode(
    episode_dir: Path,
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    cable_bodies: np.ndarray,
    cable_geoms: set[int],
    robot_geoms: set[int],
    hand_body: int,
    stride: int,
    point_count: int,
    voxel_size: float,
    out_path: Path,
) -> int:
    with np.load(episode_dir / "trajectory.npz", allow_pickle=True) as traj:
        states = np.asarray(traj["states"], dtype=np.float64)
        times = np.asarray(traj["state_times"], dtype=np.float64)
        spec = int(traj["state_spec"])
    meta = json.loads((episode_dir / "episode.json").read_text())
    success = bool(meta.get("result", {}).get("success", False))

    data = mujoco.MjData(model)
    n_cam = len(CAMERAS)
    cam_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMERAS
    ]
    intrinsics = [
        camera_matrix(renderer.width, renderer.height, float(model.cam_fovy[c]))
        for c in cam_ids
    ]

    idx = np.arange(0, len(states), stride)
    n = len(idx)
    M = len(cable_bodies)
    buf = {
        "time": np.empty(n, np.float64),
        "node_pos": np.empty((n, M, 3), np.float32),
        "node_vel": np.empty((n, M, 3), np.float32),
        "hand_pos": np.empty((n, 3), np.float32),
        "hand_quat": np.empty((n, 4), np.float32),
        "points": np.empty((n, n_cam, point_count, 3), np.float32),
        "points_frac": np.empty((n, n_cam), np.float32),
        "cam_pos": np.empty((n, n_cam, 3), np.float32),
        "cam_mat": np.empty((n, n_cam, 3, 3), np.float32),
        "node_vis": np.empty((n, n_cam, M), bool),
        "robot_mask": np.empty((n, n_cam, MASK_H, MASK_W), np.uint8),
        "cable_mask": np.empty((n, n_cam, MASK_H, MASK_W), np.uint8),
    }
    for row, si in enumerate(idx):
        mujoco.mj_setState(model, data, states[si], spec)
        mujoco.mj_forward(model, data)
        buf["time"][row] = times[si]
        buf["node_pos"][row] = data.xpos[cable_bodies]
        buf["node_vel"][row] = data.cvel[cable_bodies, 3:6]
        buf["hand_pos"][row] = data.xpos[hand_body]
        buf["hand_quat"][row] = data.xquat[hand_body]
        for ci, cam in enumerate(CAMERAS):
            rgb, depth, seg = render_cam(renderer, data, cam)
            geom_seg = seg[:, :, 0].astype(np.int64)
            mask = segment_hsv(rgb, HSV_LOWER, HSV_UPPER)
            pts = backproject_mask(depth, mask, intrinsics[ci])
            pts = voxel_downsample(pts, voxel_size)
            ordered, frac = fixed_farthest_point_sample(pts, point_count)
            buf["points"][row, ci] = ordered
            buf["points_frac"][row, ci] = frac
            buf["cam_pos"][row, ci] = data.cam_xpos[cam_ids[ci]]
            buf["cam_mat"][row, ci] = data.cam_xmat[cam_ids[ci]].reshape(3, 3)
            vis, _ = node_visibility(
                buf["node_pos"][row].astype(np.float64),
                depth,
                intrinsics[ci],
                buf["cam_pos"][row, ci].astype(np.float64),
                buf["cam_mat"][row, ci].astype(np.float64),
            )
            buf["node_vis"][row, ci] = vis
            robot = np.isin(geom_seg, list(robot_geoms))
            cable = np.isin(geom_seg, list(cable_geoms))
            buf["robot_mask"][row, ci] = downsample_mask(robot)
            buf["cable_mask"][row, ci] = downsample_mask(cable)
    np.savez_compressed(
        out_path,
        **buf,
        intrinsics=np.stack(intrinsics),
        camera_names=np.asarray(CAMERAS),
        scenario=meta.get("result", {}).get("scenario", ""),
        seed=int(meta.get("result", {}).get("seed", -1)),
        success=success,
        episode=episode_dir.name,
    )
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--render-root", type=Path, required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seeds-file", type=Path, default=None,
                   help="text file with one seed per line; overrides "
                        "--episodes selection")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--point-count", type=int, default=384)
    p.add_argument("--voxel-size", type=float, default=0.002)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--hand-body", default="link7")
    args = p.parse_args()

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

    cable_bodies = name_ids(model, mujoco.mjtObj.mjOBJ_BODY, "cable")
    cable_bset = set(int(b) for b in cable_bodies)
    cable_geoms = {
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in cable_bset
    }
    robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    robot_bodies = body_subtree(model, robot_root)
    robot_geoms = {
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in robot_bodies
    }
    hand_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, args.hand_body
    )
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames = 0
    for i, ep in enumerate(episodes):
        out_path = args.out_dir / f"{ep.name}.npz"
        if out_path.exists():
            continue
        try:
            frames += process_episode(
                ep, model, renderer, cable_bodies, cable_geoms, robot_geoms,
                hand_body, args.stride, args.point_count, args.voxel_size,
                out_path,
            )
        except Exception as exc:  # keep batch going, log failures
            print(f"[shard {args.shard}] FAIL {ep.name}: {exc}", flush=True)
        if (i + 1) % 20 == 0:
            rate = frames / max(time.time() - t0, 1e-9)
            print(
                f"[shard {args.shard}] {i + 1}/{len(episodes)} eps, "
                f"{frames} frames, {rate:.1f} fps",
                flush=True,
            )
    renderer.close()
    print(f"[shard {args.shard}] done episodes={len(episodes)} frames={frames}")


if __name__ == "__main__":
    main()
