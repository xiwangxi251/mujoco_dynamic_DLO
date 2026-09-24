"""Closed-loop evaluation: run the scripted grasp policy with estimated
cable state instead of ground truth.

The estimator consumes live camera renders (opst + wrist), the robot
occluder mask rendered from the known kinematic configuration, and the
hand pose — the same inputs as training. The policy is a
``DynamicGraspPolicy`` subclass that substitutes estimated node
positions/velocities for the privileged ``env.data`` reads used for
cable tracking. Everything else (contact/grasp-state machine, IK)
is untouched.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np
import torch

from panda_cable_grasp.policies.scripted import PolicyConfig

try:  # class was renamed across branches
    from panda_cable_grasp.policies.scripted import (
        DynamicGraspPolicy as _BasePolicy,
    )
except ImportError:
    from panda_cable_grasp.policies.scripted import (
        DynamicCableGraspPolicy as _BasePolicy,
    )

from .dataset import optical_to_world, resample_polyline_weighted
from .model import DLOStateEstimator

HSV_LOWER = (112, 180, 80)
HSV_UPPER = (130, 255, 255)
MASK_W, MASK_H = 120, 90


@dataclass
class PerceptionConfig:
    point_count: int = 384
    voxel_size: float = 0.002
    history: int = 4
    node_count: int = 14
    center: np.ndarray | None = None
    scale: float = 0.5
    device: str = "cuda"
    # horizon of the model's future_pos head in seconds
    # (dataset: 8 packed frames x 0.04 s); 0 disables the learned-future
    # intercept path in PerceptualGraspPolicy._predicted_segment
    future_horizon: float = 0.32


class PerceptionStack:
    """Renders both cameras + robot masks, runs the estimator."""

    def __init__(
        self,
        env,
        model: DLOStateEstimator,
        cfg: PerceptionConfig,
        cameras: tuple[str, str] = (
            "dynamicvla_opst_camera", "dynamicvla_wrist_camera",
        ),
    ) -> None:
        self.env = env
        self.model = model.eval()
        self.cfg = cfg
        self.cameras = cameras
        self.renderer = mujoco.Renderer(
            env.model, height=360, width=480,
        )
        from panda_cable_grasp.rl.pointcloud import camera_matrix

        self.cam_ids = [
            mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_CAMERA, c
            )
            for c in cameras
        ]
        self.intrinsics = [
            camera_matrix(480, 360, float(env.model.cam_fovy[c]))
            for c in self.cam_ids
        ]
        # robot subtree geoms for the occluder silhouette
        robot_root = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        robot_bodies = set()
        for b in range(env.model.nbody):
            p = b
            while p != 0 and p != robot_root:
                p = int(env.model.body_parentid[p])
            if p == robot_root:
                robot_bodies.add(b)
        self.robot_geoms = {
            g for g in range(env.model.ngeom)
            if int(env.model.geom_bodyid[g]) in robot_bodies
        }
        self.hist_o: deque = deque(maxlen=cfg.history)
        self.hist_w: deque = deque(maxlen=cfg.history)
        self.hist_ro: deque = deque(maxlen=1)
        self.hist_rw: deque = deque(maxlen=1)

    def reset(self) -> None:
        self.hist_o.clear()
        self.hist_w.clear()
        self.hist_ro.clear()
        self.hist_rw.clear()

    def _observe(self) -> None:
        from panda_cable_grasp.rl.pointcloud import (
            backproject_mask,
            fixed_farthest_point_sample,
            segment_hsv,
            voxel_downsample,
        )

        env, cfg = self.env, self.cfg
        data = env.data
        clouds, masks = [], []
        for ci, cam in enumerate(self.cameras):
            self.renderer.update_scene(data, camera=cam)
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(data, camera=cam)
            depth = np.asarray(self.renderer.render(), dtype=np.float32)
            self.renderer.disable_depth_rendering()
            self.renderer.enable_segmentation_rendering()
            self.renderer.update_scene(data, camera=cam)
            seg = np.asarray(self.renderer.render())[:, :, 0].astype(np.int64)
            self.renderer.disable_segmentation_rendering()
            pts = backproject_mask(
                depth, segment_hsv(rgb, HSV_LOWER, HSV_UPPER),
                self.intrinsics[ci],
            )
            pts = voxel_downsample(pts, cfg.voxel_size)
            ordered, _ = fixed_farthest_point_sample(
                pts, cfg.point_count
            )
            world = optical_to_world(
                ordered.astype(np.float64),
                data.cam_xpos[self.cam_ids[ci]].astype(np.float64),
                data.cam_xmat[self.cam_ids[ci]].reshape(3, 3)
                .astype(np.float64),
            )
            clouds.append(world.astype(np.float32))
            robot = np.isin(seg, list(self.robot_geoms))
            masks.append(
                cv2.resize(
                    robot.astype(np.uint8), (MASK_W, MASK_H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.float32)
            )
        self.hist_o.append(clouds[0])
        self.hist_w.append(clouds[1])
        self.hist_ro.append(masks[0])
        self.hist_rw.append(masks[1])

    @torch.no_grad()
    def estimate(
        self, future_hand_pos: np.ndarray | None = None
    ) -> dict[str, np.ndarray] | None:
        """Returns normalized-space and world-space node predictions.

        ``future_hand_pos`` is the controller's intended hand position at
        the lead time (e.g. the filtered intercept target); falls back to
        the current hand pose when unavailable.
        """
        cfg = self.cfg
        self._observe()
        if len(self.hist_o) < 1:
            return None
        # dataset convention: index 0 = oldest, index -1 = current;
        # deques already hold oldest..newest — pad missing old frames by
        # repeating the oldest available
        pad = cfg.history - len(self.hist_o)
        ho = [self.hist_o[0]] * pad + list(self.hist_o)
        hw = [self.hist_w[0]] * pad + list(self.hist_w)
        center = cfg.center
        hand_body = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "link7"
        )
        hand_pos = self.env.data.xpos[hand_body]
        hand_quat = self.env.data.xquat[hand_body]
        if future_hand_pos is None:
            future_hand_pos = hand_pos
        batch = {
            "points_opst": torch.as_tensor(
                (np.stack(ho) - center) / cfg.scale
            ).float()[None],
            "points_wrist": torch.as_tensor(
                (np.stack(hw) - center) / cfg.scale
            ).float()[None],
            "robot_mask_opst": torch.as_tensor(
                self.hist_ro[-1]
            ).float()[None],
            "robot_mask_wrist": torch.as_tensor(
                self.hist_rw[-1]
            ).float()[None],
            "hand_pos": torch.as_tensor(
                (hand_pos - center) / cfg.scale
            ).float()[None],
            "hand_quat": torch.as_tensor(hand_quat).float()[None],
            "hand_pos_future": torch.as_tensor(
                (future_hand_pos - center) / cfg.scale
            ).float()[None],
            "hand_quat_future": torch.as_tensor(
                hand_quat
            ).float()[None],
        }
        dev = next(self.model.parameters()).device
        batch = {k: v.to(dev) for k, v in batch.items()}
        out = self.model(batch)
        nodes = out["pos"][0].cpu().numpy() * cfg.scale + center
        vel = (
            out["vel"][0].cpu().numpy() * cfg.scale
            if "vel" in out
            else np.zeros_like(nodes)
        )
        res = {
            "nodes14": nodes.astype(np.float64),
            "vel14": vel.astype(np.float64),
        }
        if "logvar" in out:
            res["std14"] = np.exp(
                0.5 * out["logvar"][0].cpu().numpy()
            ) * cfg.scale
        if "future_pos" in out:
            res["future14"] = (
                out["future_pos"][0].cpu().numpy() * cfg.scale + center
            ).astype(np.float64)
            res["future_h"] = cfg.future_horizon
        # interpolate 14-node prediction back to the 40 MuJoCo bodies
        pos40, vel40, _ = resample_polyline_weighted(
            res["nodes14"], res["vel14"], 40
        )
        res["pos40"], res["vel40"] = pos40, vel40
        return res


class PerceptualGraspPolicy(_BasePolicy):
    """Scripted policy driven by estimated cable state.

    Overrides the three privileged cable-state reads:
    ``_nearest_cable_point`` (segment distances), ``_predicted_segment``
    (target extrapolation) and ``_lock_segment_near`` (contact-side
    re-targeting). Env-level quantities that exist on a real robot
    (hand pose, grasp/contact state) are still read from the env.
    """

    def __init__(
        self,
        env,
        estimator: PerceptionStack,
        config: PolicyConfig | None = None,
    ) -> None:
        # base __init__ calls self.reset(), which needs the estimator
        self.est = estimator
        self._last_est: dict[str, np.ndarray] | None = None
        self._est_log: list[dict] = []
        super().__init__(env, config)

    def reset(self) -> None:
        super().reset()
        self.est.reset()
        self._last_est = None
        self._est_log = []
        # base reset() seeds filtered_target from privileged state; replace
        # it with the first perception-based estimate
        est = self.est.estimate()
        if est is not None:
            mid = len(est["pos40"]) // 2
            self.filtered_target = est["pos40"][mid].copy()
            self._last_est = est

    def _est_nodes(self) -> dict[str, np.ndarray] | None:
        if self._last_est is None:
            # filtered_target is the policy's intended intercept point —
            # a realistic proxy for the hand's position at the lead time
            hint = getattr(self, "filtered_target", None)
            self._last_est = self.est.estimate(future_hand_pos=hint)
        return self._last_est

    def new_control_step(self) -> None:
        """Caller must invoke once per env step before action()."""
        self._last_est = None

    # --- privileged reads overridden ------------------------------------

    def _initial_target(self) -> np.ndarray:
        est = self._est_nodes()
        if est is None:
            return self.env.target_position()
        mid = len(est["pos40"]) // 2
        return est["pos40"][mid].copy()

    def _cable_positions(self) -> np.ndarray:
        est = self._est_nodes()
        if est is None:
            return self.env.data.xpos[self.env.cable_ids]
        return est["pos40"]

    def _cable_velocity_at(self, index: int, alpha: float) -> np.ndarray:
        est = self._est_nodes()
        if est is None:
            body0 = self.env.cable_ids[index]
            body1 = self.env.cable_ids[index + 1]
            return (
                (1.0 - alpha) * self.env.body_linear_velocity(body0)
                + alpha * self.env.body_linear_velocity(body1)
            )
        return (1.0 - alpha) * est["vel40"][index] + alpha * est[
            "vel40"
        ][index + 1]

    def _nearest_cable_point(
        self, point: np.ndarray
    ) -> tuple[np.ndarray, float, int, float]:
        positions = self._cable_positions()
        starts = positions[:-1]
        vectors = positions[1:] - starts
        lengths_squared = np.sum(vectors * vectors, axis=1)
        alpha = np.sum((point - starts) * vectors, axis=1) / np.maximum(
            lengths_squared, 1e-12
        )
        alpha = np.clip(alpha, 0.0, 1.0)
        projected = starts + alpha[:, None] * vectors
        distances = np.linalg.norm(projected - point, axis=1)
        index = int(np.argmin(distances))
        return (
            projected[index].copy(), float(distances[index]),
            index, float(alpha[index]),
        )

    def _lock_segment_near(self, point: np.ndarray) -> np.ndarray:
        nearest, _, index, alpha = self._nearest_cable_point(point)
        self.locked_segment_index = index
        self.locked_segment_alpha = alpha
        self.filtered_target = nearest.copy()
        return nearest

    def _predicted_segment(
        self, prediction_horizon: float | None = None
    ) -> np.ndarray:
        est = self._est_nodes()
        if self.locked_segment_index is None:
            if est is None:
                position = self.env.target_position()
                velocity = self.env.target_velocity()
            else:
                mid = len(est["pos40"]) // 2
                position = est["pos40"][mid]
                velocity = est["vel40"][mid]
        else:
            index = self.locked_segment_index
            alpha = self.locked_segment_alpha
            if est is None:
                body0 = self.env.cable_ids[index]
                body1 = self.env.cable_ids[index + 1]
                position = (
                    (1.0 - alpha) * self.env.data.xpos[body0]
                    + alpha * self.env.data.xpos[body1]
                )
            else:
                position = (
                    (1.0 - alpha) * est["pos40"][index]
                    + alpha * est["pos40"][index + 1]
                )
            velocity = self._cable_velocity_at(index, alpha)
        velocity = np.clip(velocity, -0.8, 0.8)
        if prediction_horizon is None:
            from panda_cable_grasp.policies.scripted import Phase

            if self.phase in {Phase.SETTLE, Phase.APPROACH}:
                prediction_horizon = self.config.approach_prediction_horizon
            elif self.phase is Phase.CLOSE:
                prediction_horizon = self.config.close_prediction_horizon
            else:
                prediction_horizon = self.config.prediction_horizon
        predicted = position + prediction_horizon * velocity
        # Prefer the learned future head when available: it predicts the
        # node chain at H_fut ≈ 0.32 s end-to-end, so the displacement
        # captures dynamics that linear velocity extrapolation misses
        # (pendulum swing, contact transients). Scale it to the requested
        # horizon; falls back to v·t when the estimate lacks the head.
        if (
            est is not None
            and "future14" in est
            and est.get("future_h", 0.0) > 0.0
        ):
            fut = est["future14"]  # (14,3) world, at t + future_h
            cur = est["nodes14"]
            if self.locked_segment_index is None:
                t14 = 0.5 * (len(cur) - 1)
            else:
                t14 = (
                    (self.locked_segment_index + self.locked_segment_alpha)
                    / (len(est["pos40"]) - 1) * (len(cur) - 1)
                )
            i14 = int(np.clip(np.floor(t14), 0, len(cur) - 2))
            a14 = t14 - i14
            seg_fut = (1.0 - a14) * fut[i14] + a14 * fut[i14 + 1]
            seg_cur = (1.0 - a14) * cur[i14] + a14 * cur[i14 + 1]
            learned_disp = seg_fut - seg_cur
            scale = prediction_horizon / max(est["future_h"], 1e-6)
            predicted = position + learned_disp * scale
        predicted[0] = np.clip(predicted[0], *self.config.intercept_x_limits)
        predicted[1] = np.clip(predicted[1], *self.config.intercept_y_limits)
        predicted[2] = np.clip(predicted[2], *self.config.intercept_z_limits)
        self.filtered_target += self.config.target_filter_alpha * (
            predicted - self.filtered_target
        )
        return self.filtered_target.copy()
