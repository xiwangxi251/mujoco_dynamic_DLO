"""Render paired point-cloud observations over a recorded FULLPHYSICS trajectory.

Replays saved states from a trajectory.npz and, for each frame, renders the
policy camera twice — normal gripper visibility and the gripper-hidden
counterfactual — then regenerates the 384-point policy cloud for each. The
physics is identical across the pair, so the only difference is what the
policy would have observed.

Output: side-by-side mp4 (normal | hidden) with the policy points overlaid on
the camera image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from panda_cable_grasp.perception.rendering import GripperVisibilityToggle  # noqa: E402
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env  # noqa: E402
from panda_cable_grasp.rl.pointcloud import (  # noqa: E402
    backproject_mask,
    camera_matrix,
    fixed_farthest_point_sample,
    segment_hsv,
    voxel_downsample,
)

HSV_LOWER = (112, 180, 80)
HSV_UPPER = (130, 255, 255)
VOXEL_M = 0.002
SCALE_M = 0.50
N_POINTS = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def capture_pair(base, renderer, toggle, cam_name, cam_id, intrinsics):
    """Render rgb + policy cloud for normal and gripper-hidden visibility."""

    def grab(hidden: bool):
        def _render():
            renderer.disable_depth_rendering()
            renderer.update_scene(base.data, camera=cam_name)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(base.data, camera=cam_name)
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            return rgb, depth

        if hidden:
            with toggle.hidden():
                rgb, depth = _render()
        else:
            rgb, depth = _render()

        mask = segment_hsv(rgb, HSV_LOWER, HSV_UPPER)
        masked_depth = np.where(mask > 0, depth, np.inf).astype(np.float32)
        local_depth = cv2.erode(masked_depth, np.ones((5, 5), np.uint8))
        mask = np.where(
            (mask > 0) & (depth <= local_depth + 0.020), 255, 0
        ).astype(np.uint8)
        cam_points = backproject_mask(depth, mask, intrinsics)
        cam_points = voxel_downsample(cam_points, VOXEL_M)

        cam_rot = np.asarray(base.data.cam_xmat[cam_id]).reshape(3, 3)
        optical_to_world = cam_rot @ np.diag([1.0, -1.0, -1.0])
        world = cam_points @ optical_to_world.T + base.data.cam_xpos[cam_id]
        hand_pos = base.hand_position.copy()
        hand_rot = np.asarray(base.data.xmat[base.hand_id]).reshape(3, 3)
        tcp = (world - hand_pos) @ hand_rot
        policy_pts, _ = fixed_farthest_point_sample(tcp / SCALE_M, N_POINTS)
        # back to camera frame for overlay: tcp -> world -> camera-optical
        policy_world = (policy_pts * SCALE_M) @ hand_rot.T + hand_pos
        cam_optical = (policy_world - base.data.cam_xpos[cam_id]) @ optical_to_world
        return rgb, cam_optical, int(len(cam_points))

    return grab(False), grab(True)


def label_bar(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, 24), (0, 0, 0), -1)
    cv2.putText(
        img, text, (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return img


def rgb_panel(rgb, label):
    return label_bar(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy(), label)


def cloud_panel(pixels, label, count, width, height):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    for px, py, pz in pixels:
        depth01 = float(np.clip((pz - 0.3) / 0.9, 0.0, 1.0))
        color = (
            int(255 * depth01),
            80,
            int(255 * (1.0 - depth01)),
        )
        cv2.circle(panel, (int(px), int(py)), 3, color, -1)
    return label_bar(panel, f"{label}  n=384 raw={count}")


def main() -> None:
    args = parse_args()
    archive = np.load(args.trajectory)
    states = archive["states"]
    frame_idx = archive["frame_state_indices"]
    if args.max_frames > 0:
        frame_idx = frame_idx[: args.max_frames]

    env = make_rl_env(
        action_mode="task_space_vertical_down",
        robot="nero",
        seed=0,
        disturbance_strength=1.5,
        episode_seconds=args.episode_seconds,
        dynamicvla_cameras_enabled=True,
        scenario_names=(args.scenario,),
        rl_config=RLConfig(singularity_avoidance_enabled=False),
        geometric_safety_enabled=False,
        table_finger_collision_filter_enabled=True,
    )
    env.base_env.config.replay_seed_fallback = "error"
    env.reset(seed=args.seed)

    base = env.base_env
    cam_name = base.config.dynamicvla_opst_camera_name
    cam_id = int(base.dynamicvla_opst_camera_id)
    width, height = 480, 360
    intrinsics = camera_matrix(
        width, height, float(base.model.cam_fovy[cam_id])
    )
    renderer = mujoco.Renderer(base.model, height=height, width=width)
    toggle = GripperVisibilityToggle(base.model)

    args.output.mkdir(parents=True, exist_ok=True)
    video_path = args.output / "pointcloud_normal_vs_hidden.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (width * 2, height * 2),
    )
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    try:
        for step, si in enumerate(frame_idx):
            mujoco.mj_setState(base.model, base.data, states[int(si)], spec)
            mujoco.mj_forward(base.model, base.data)
            (rgb_n, pts_n, raw_n), (rgb_h, pts_h, raw_h) = capture_pair(
                base, renderer, toggle, cam_name, cam_id, intrinsics
            )
            def px(points):
                z = points[:, 2]
                valid = z > 1e-6
                out = np.zeros((len(points), 3))
                out[:, 0] = intrinsics[0, 0] * points[:, 0] / np.where(valid, z, 1) + intrinsics[0, 2]
                out[:, 1] = intrinsics[1, 1] * points[:, 1] / np.where(valid, z, 1) + intrinsics[1, 2]
                out[:, 2] = z
                return out[valid]
            top = np.concatenate(
                [
                    rgb_panel(rgb_n, "normal camera"),
                    rgb_panel(rgb_h, "gripper_hidden camera"),
                ],
                axis=1,
            )
            bottom = np.concatenate(
                [
                    cloud_panel(px(pts_n), "normal cloud", raw_n, width, height),
                    cloud_panel(px(pts_h), "hidden cloud", raw_h, width, height),
                ],
                axis=1,
            )
            writer.write(np.concatenate([top, bottom], axis=0))
            if step % 50 == 0:
                print(f"frame {step}/{len(frame_idx)}", flush=True)
    finally:
        writer.release()
        renderer.close()
        env.close()
    print(f"wrote {video_path}", flush=True)


if __name__ == "__main__":
    main()
