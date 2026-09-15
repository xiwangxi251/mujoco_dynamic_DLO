"""RGB-D point-cloud observations and a compact PointNet++ PPO encoder."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import cv2
import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


@dataclass(frozen=True)
class PointCloudObservationConfig:
    """Camera front-end settings used by the point-cloud policy."""

    point_count: int = 384
    width: int = 480
    height: int = 360
    camera_update_steps: int = 5
    sensor_delay_steps: int = 3
    voxel_size_m: float = 0.002
    position_scale_m: float = 0.50
    hsv_lower: tuple[int, int, int] = (112, 180, 80)
    hsv_upper: tuple[int, int, int] = (130, 255, 255)

    def __post_init__(self) -> None:
        if self.point_count < 32:
            raise ValueError("point_count must be at least 32")
        if self.width < 32 or self.height < 32:
            raise ValueError("camera resolution is too small")
        if self.camera_update_steps < 1 or self.sensor_delay_steps < 0:
            raise ValueError("camera cadence and delay must be non-negative")
        if self.voxel_size_m <= 0.0 or self.position_scale_m <= 0.0:
            raise ValueError("point-cloud scales must be positive")


def camera_matrix(width: int, height: int, fovy_degrees: float) -> np.ndarray:
    focal = 0.5 * height / np.tan(np.deg2rad(fovy_degrees) * 0.5)
    return np.asarray(
        [
            [focal, 0.0, (width - 1) * 0.5],
            [0.0, focal, (height - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def segment_hsv(
    rgb: np.ndarray,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
) -> np.ndarray:
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv, np.asarray(lower, dtype=np.uint8), np.asarray(upper, dtype=np.uint8)
    )
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # RGB antialiasing can label a cable-edge pixel whose depth sample belongs
    # to the table behind it.  Removing one boundary pixel prevents those
    # depth discontinuities from becoming extreme points in FPS.
    return cv2.erode(opened, kernel, iterations=1)


def backproject_mask(
    depth_m: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0.0)
    rows, cols = np.nonzero(valid)
    z = np.asarray(depth_m[rows, cols], dtype=np.float64)
    x = (cols - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.column_stack((x, y, z))


def voxel_downsample(points: np.ndarray, leaf_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    keys = np.floor(points / float(leaf_size)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )


def fixed_farthest_point_sample(
    points: np.ndarray, count: int
) -> tuple[np.ndarray, float]:
    """Return an FPS-ordered fixed-size cloud and its non-duplicated fraction."""

    points = np.asarray(points, dtype=np.float64)
    finite = points[np.isfinite(points).all(axis=1)]
    if not len(finite):
        return np.zeros((count, 3), dtype=np.float32), 0.0
    unique_count = min(len(finite), int(count))
    selected = np.empty(unique_count, dtype=np.int64)
    center = np.mean(finite, axis=0)
    selected[0] = int(np.argmax(np.sum((finite - center) ** 2, axis=1)))
    minimum_distance = np.sum((finite - finite[selected[0]]) ** 2, axis=1)
    minimum_distance[selected[0]] = -1.0
    for index in range(1, unique_count):
        selected[index] = int(np.argmax(minimum_distance))
        distance = np.sum((finite - finite[selected[index]]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected[: index + 1]] = -1.0
    ordered = finite[selected]
    if unique_count < count:
        ordered = ordered[np.arange(count) % unique_count]
    return ordered.astype(np.float32), float(unique_count / count)


class DLOPointCloudObservation(gym.Wrapper):
    """Replace privileged DLO state with delayed RGB-D point observations."""

    def __init__(
        self,
        env: gym.Env,
        config: PointCloudObservationConfig | None = None,
    ) -> None:
        super().__init__(env)
        self.config = config or PointCloudObservationConfig()
        base = self.unwrapped.base_env
        if not base.config.dynamicvla_cameras_enabled:
            raise ValueError("point-cloud observation requires DynamicVLA cameras")
        self._base = base
        self._camera_id = int(base.dynamicvla_opst_camera_id)
        self._camera_name = base.config.dynamicvla_opst_camera_name
        self._intrinsics = camera_matrix(
            self.config.width,
            self.config.height,
            float(base.model.cam_fovy[self._camera_id]),
        )
        self._renderer = mujoco.Renderer(
            base.model, height=self.config.height, width=self.config.width
        )
        self._step_index = 0
        self._pending: deque[tuple[int, np.ndarray, float]] = deque()
        self._delivered_points = np.zeros(
            (self.config.point_count, 3), dtype=np.float32
        )
        self._delivered_fraction = 0.0
        self.last_capture_ms = 0.0
        self.last_raw_point_count = 0
        self.last_voxel_point_count = 0
        self.observation_space = gym.spaces.Dict(
            {
                "points": gym.spaces.Box(
                    -10.0,
                    10.0,
                    shape=(self.config.point_count, 3),
                    dtype=np.float32,
                ),
                "proprio": gym.spaces.Box(
                    -10.0, 10.0, shape=(16,), dtype=np.float32
                ),
            }
        )

    def _capture(self) -> tuple[np.ndarray, float]:
        started = perf_counter()
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(self._base.data, camera=self._camera_name)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self._base.data, camera=self._camera_name)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()

        mask = segment_hsv(rgb, self.config.hsv_lower, self.config.hsv_upper)
        # Reject pixels whose depth is substantially behind another masked
        # sample in the same local cable neighbourhood.  These are raster
        # boundary mismatches, not measurements of the blue cable surface.
        masked_depth = np.where(mask > 0, depth, np.inf).astype(np.float32)
        local_depth = cv2.erode(masked_depth, np.ones((5, 5), np.uint8))
        mask = np.where(
            (mask > 0) & (depth <= local_depth + 0.020), 255, 0
        ).astype(np.uint8)
        camera_points = backproject_mask(depth, mask, self._intrinsics)
        self.last_raw_point_count = int(len(camera_points))
        camera_points = voxel_downsample(camera_points, self.config.voxel_size_m)
        self.last_voxel_point_count = int(len(camera_points))

        camera_rotation = np.asarray(
            self._base.data.cam_xmat[self._camera_id], dtype=np.float64
        ).reshape(3, 3)
        optical_to_world = camera_rotation @ np.diag([1.0, -1.0, -1.0])
        world_points = (
            camera_points @ optical_to_world.T
            + self._base.data.cam_xpos[self._camera_id]
        )
        hand_position = self._base.hand_position.copy()
        hand_rotation = np.asarray(
            self._base.data.xmat[self._base.hand_id], dtype=np.float64
        ).reshape(3, 3)
        tcp_points = (world_points - hand_position) @ hand_rotation
        points, fraction = fixed_farthest_point_sample(
            tcp_points / self.config.position_scale_m,
            self.config.point_count,
        )
        self.last_capture_ms = 1000.0 * (perf_counter() - started)
        return np.clip(points, -10.0, 10.0), fraction

    def _queue_capture(self, *, immediate: bool = False) -> None:
        points, fraction = self._capture()
        available_step = (
            self._step_index
            if immediate
            else self._step_index + self.config.sensor_delay_steps
        )
        self._pending.append((available_step, points, fraction))

    def _deliver_available(self) -> None:
        while self._pending and self._pending[0][0] <= self._step_index:
            _, self._delivered_points, self._delivered_fraction = self._pending.popleft()

    def _replace_observation(self, state_observation: np.ndarray) -> dict[str, np.ndarray]:
        proprio = np.concatenate(
            (
                np.asarray(state_observation, dtype=np.float32)[-15:],
                np.asarray([self._delivered_fraction], dtype=np.float32),
            )
        )
        return {
            "points": self._delivered_points.copy(),
            "proprio": proprio.astype(np.float32, copy=False),
        }

    def reset(self, **kwargs):
        state_observation, info = self.env.reset(**kwargs)
        self._step_index = 0
        self._pending.clear()
        self._queue_capture(immediate=True)
        self._deliver_available()
        info = dict(info)
        info.update(self.pointcloud_info())
        return self._replace_observation(state_observation), info

    def step(self, action):
        state_observation, reward, terminated, truncated, info = self.env.step(action)
        self._step_index += 1
        if self._step_index % self.config.camera_update_steps == 0:
            self._queue_capture()
        self._deliver_available()
        info = dict(info)
        info.update(self.pointcloud_info())
        return (
            self._replace_observation(state_observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def pointcloud_info(self) -> dict[str, float | int]:
        return {
            "pointcloud_capture_ms": self.last_capture_ms,
            "pointcloud_raw_count": self.last_raw_point_count,
            "pointcloud_voxel_count": self.last_voxel_point_count,
            "pointcloud_unique_fraction": self._delivered_fraction,
        }

    def set_training_scenario_distribution(
        self, scenario_names, probabilities
    ) -> None:
        self.env.set_training_scenario_distribution(scenario_names, probabilities)

    def set_training_scenarios(self, scenario_names) -> None:
        self.env.set_training_scenarios(scenario_names)

    def set_motion_difficulty(self, difficulty: float) -> None:
        self.env.set_motion_difficulty(difficulty)

    def close(self) -> None:
        self._renderer.close()
        self.env.close()


class _SetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int,
        nsample: int,
        input_channels: int,
        channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.npoint = int(npoint)
        self.nsample = int(nsample)
        layers: list[nn.Module] = []
        previous = input_channels + 3
        for channel in channels:
            layers.extend(
                (nn.Conv2d(previous, channel, 1), nn.ReLU(inplace=True))
            )
            previous = channel
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centroids = xyz[:, : self.npoint, :]
        distances = torch.cdist(centroids, xyz)
        indices = distances.topk(
            min(self.nsample, xyz.shape[1]), dim=-1, largest=False
        ).indices
        batch = torch.arange(xyz.shape[0], device=xyz.device)[:, None, None]
        grouped_xyz = xyz[batch, indices]
        local_xyz = grouped_xyz - centroids[:, :, None, :]
        if features is None:
            grouped = local_xyz
        else:
            grouped_features = features[batch, indices]
            grouped = torch.cat((local_xyz, grouped_features), dim=-1)
        encoded = self.mlp(grouped.permute(0, 3, 1, 2)).amax(dim=-1)
        return centroids, encoded.transpose(1, 2)


class PointNet2FeaturesExtractor(BaseFeaturesExtractor):
    """Compact two-stage PointNet++ SSG encoder for 384-point observations."""

    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=320)
        self.sa1 = _SetAbstraction(96, 16, 0, (32, 32, 64))
        self.sa2 = _SetAbstraction(24, 16, 64, (64, 96, 128))
        self.global_mlp = nn.Sequential(
            nn.Conv1d(131, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(inplace=True),
        )
        self.proprio_mlp = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True)
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        xyz = observations["points"]
        xyz1, features1 = self.sa1(xyz, None)
        xyz2, features2 = self.sa2(xyz1, features1)
        global_input = torch.cat((xyz2, features2), dim=-1).transpose(1, 2)
        cloud_feature = self.global_mlp(global_input).amax(dim=-1)
        proprio_feature = self.proprio_mlp(observations["proprio"])
        return torch.cat((cloud_feature, proprio_feature), dim=-1)
