"""Smoke test: replay FULLPHYSICS states, render cameras, extract DLO nodes.

Usage (server):
    python tools/perception/probe_replay.py \
        --episode-dir <raw trajectory dir containing trajectory.npz> \
        --model <scenario>.mjb --states 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from panda_cable_grasp.runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import mujoco
import numpy as np

from panda_cable_grasp.rl.pointcloud import (
    backproject_mask,
    camera_matrix,
    segment_hsv,
)


def cable_body_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.startswith("cable"):
            ids.append(body_id)
    return np.asarray(ids, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with np.load(args.episode_dir / "trajectory.npz", allow_pickle=True) as traj:
        states = np.asarray(traj["states"], dtype=np.float64)
        times = np.asarray(traj["state_times"], dtype=np.float64)
        state_spec = int(traj["state_spec"])

    model = mujoco.MjModel.from_binary_path(str(args.model))
    data = mujoco.MjData(model)

    expected = mujoco.mj_stateSize(model, state_spec)
    assert states.shape[1] == expected, (states.shape, expected)

    cam_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(model.ncam)
    ]
    print("cameras:", cam_names)
    print("nbody:", model.nbody, "ngeom:", model.ngeom)

    cable_ids = cable_body_ids(model)
    print("cable bodies:", len(cable_ids))
    cable_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b))
        for b in cable_ids
    ]
    print("first/last cable body:", cable_names[0], cable_names[-1])

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    print("gripper_base body id:", hand_id)

    renderer = mujoco.Renderer(model, height=360, width=480)
    camera_name = "dynamicvla_opst_camera"
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    intrinsics = camera_matrix(480, 360, float(model.cam_fovy[cam_id]))

    for index in range(min(args.states, len(states))):
        mujoco.mj_setState(model, data, states[index], state_spec)
        mujoco.mj_forward(model, data)

        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_name)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        mask = segment_hsv(rgb, (112, 180, 80), (130, 255, 255))
        points = backproject_mask(depth, mask, intrinsics)

        node_pos = data.xpos[cable_ids].copy()
        node_linvel = data.cvel[cable_ids, 3:6].copy()
        node_rotvel = data.cvel[cable_ids, :3]
        com_z = node_pos[:, 2].mean()
        speed = np.linalg.norm(node_linvel, axis=1)

        # project nodes to image, compare with rendered depth -> visibility
        cam_pos = data.cam_xpos[cam_id]
        cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
        world_to_optical = np.diag([1.0, -1.0, -1.0]) @ cam_rot.T
        rel = (node_pos - cam_pos) @ world_to_optical.T
        in_front = rel[:, 2] > 0.01
        u = intrinsics[0, 0] * rel[:, 0] / np.maximum(rel[:, 2], 1e-9) + intrinsics[0, 2]
        v = intrinsics[1, 1] * rel[:, 1] / np.maximum(rel[:, 2], 1e-9) + intrinsics[1, 2]
        inside = in_front & (u >= 0) & (u < 480) & (v >= 0) & (v < 360)
        vis = np.zeros(len(cable_ids), dtype=bool)
        ui = np.clip(np.round(u[inside]).astype(int), 0, 479)
        vi = np.clip(np.round(v[inside]).astype(int), 0, 359)
        dep = depth[vi, ui]
        vis[np.flatnonzero(inside)] = np.abs(dep - rel[inside, 2]) < 0.02

        print(
            f"state {index} t={times[index]:.3f} "
            f"segpx={int(mask.sum())} pts={len(points)} "
            f"nodes={len(cable_ids)} z_com={com_z:.3f} "
            f"vel|min/med/max|={speed.min():.3f}/{np.median(speed):.3f}/{speed.max():.3f} "
            f"visible={vis.sum()}/{len(vis)}"
        )
        if args.out is not None and index == min(args.states, len(states)) - 1:
            import cv2

            overlay = rgb.copy()
            overlay[mask > 0] = (0, 0, 255)
            cv2.imwrite(str(args.out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            np.savez(
                args.out.with_suffix(".npz"),
                node_pos=node_pos,
                node_linvel=node_linvel,
                depth=depth,
                rgb=rgb,
                mask=mask,
                vis=vis,
            )
    renderer.close()
    print("probe=OK")


if __name__ == "__main__":
    main()
