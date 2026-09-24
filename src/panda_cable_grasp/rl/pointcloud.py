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

from ..perception.rendering import GripperVisibilityToggle


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
    # normal：正常渲染；gripper_hidden：夹爪视觉组整体关闭的对抗事实渲染；
    # mixed：每回合在两者间随机（训练增广，使两种观测都在分布内）。
    render_mode: str = "normal"
    # >0 时从观测云中删除距目标段中心该半径（米）内的所有点，
    # 模拟「目标接触区域被前景遮挡」的部分可观测干预。0 = 关闭。
    target_mask_radius_m: float = 0.0
    # 深度相机噪声模型（深度域注入，反投影之前生效）：
    #   depth_noise_std_at_1m：轴向高斯噪声系数 k，sigma = k * z^2（米），
    #     对应结构光/双目相机的视差量化特性（Nguyen 2012 / Khoshelham 模型）
    #   noise_episode_bias：sigma 的多大比例作为回合内固定偏置（模拟
    #     传感器的固定模式噪声；剩余部分逐帧独立重采）
    #   pixel_dropout_p：每个像素独立缺失返回的概率（holes）
    #   frame_drop_p：每次计划采集整帧丢弃的概率（云保持上一帧不变）
    depth_noise_std_at_1m: float = 0.0
    noise_episode_bias: float = 0.5
    pixel_dropout_p: float = 0.0
    frame_drop_p: float = 0.0

    def __post_init__(self) -> None:
        if self.point_count < 32:
            raise ValueError("point_count must be at least 32")
        if self.width < 32 or self.height < 32:
            raise ValueError("camera resolution is too small")
        if self.camera_update_steps < 1 or self.sensor_delay_steps < 0:
            raise ValueError("camera cadence and delay must be non-negative")
        if self.voxel_size_m <= 0.0 or self.position_scale_m <= 0.0:
            raise ValueError("point-cloud scales must be positive")
        if self.render_mode not in {"normal", "gripper_hidden", "mixed"}:
            raise ValueError(
                "render_mode must be 'normal', 'gripper_hidden' or 'mixed'"
            )
        if not 0.0 <= self.target_mask_radius_m < 0.5:
            raise ValueError("target_mask_radius_m must be in [0, 0.5)")
        if not 0.0 <= self.depth_noise_std_at_1m < 1.0:
            raise ValueError("depth_noise_std_at_1m must be in [0, 1)")
        if not 0.0 <= self.noise_episode_bias <= 1.0:
            raise ValueError("noise_episode_bias must be in [0, 1]")
        if not 0.0 <= self.pixel_dropout_p < 1.0:
            raise ValueError("pixel_dropout_p must be in [0, 1)")
        if not 0.0 <= self.frame_drop_p < 1.0:
            raise ValueError("frame_drop_p must be in [0, 1)")


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
        self._gripper_visibility = GripperVisibilityToggle(base.model)
        # 独立的渲染模式 Generator：mixed 增广抽签不消耗环境 rng，
        # 不破坏配对 seed 下的物理抽签序列。
        self._render_rng = np.random.default_rng(
            (0 if base.config.seed is None else int(base.config.seed))
            ^ 0x6A11C
        )
        self._active_render_mode = self.config.render_mode
        self._step_index = 0
        self._pending: deque[tuple[int, np.ndarray, float]] = deque()
        self._delivered_points = np.zeros(
            (self.config.point_count, 3), dtype=np.float32
        )
        self._delivered_fraction = 0.0
        self.last_capture_ms = 0.0
        self.last_raw_point_count = 0
        self.last_voxel_point_count = 0
        self.last_masked_count = 0
        # 深度噪声专用 Generator：按回合 episode_seed 播种——同 seed 的
        # 配对格子（deform/replay）吃到逐位相同的噪声序列，噪声本身
        # 不构成混杂变量；与 _render_rng 相互独立。
        self._noise_rng = np.random.default_rng(0)
        self._noise_bias: np.ndarray | None = None
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

        def _grab() -> tuple[np.ndarray, np.ndarray]:
            self._renderer.disable_depth_rendering()
            self._renderer.update_scene(
                self._base.data, camera=self._camera_name
            )
            rgb = self._renderer.render().copy()
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(
                self._base.data, camera=self._camera_name
            )
            depth = self._renderer.render().copy()
            self._renderer.disable_depth_rendering()
            return rgb, depth

        if self._active_render_mode == "gripper_hidden":
            with self._gripper_visibility.hidden():
                rgb, depth = _grab()
        else:
            rgb, depth = _grab()

        depth = self._corrupt_depth(depth)
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
        if self.config.target_mask_radius_m > 0.0 and len(world_points):
            target = np.asarray(self._base.target_position(), dtype=np.float64)
            keep = (
                np.linalg.norm(world_points - target, axis=1)
                > self.config.target_mask_radius_m
            )
            world_points = world_points[keep]
        self.last_masked_count = int(len(world_points))
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

    def _corrupt_depth(self, depth: np.ndarray) -> np.ndarray:
        """Inject depth-sensor noise in the depth domain, before backprojection.

        模型组成（均只在有限深度像素上生效）：
        - 轴向高斯噪声 sigma = k*z^2（结构光/双目视差量化的标准模型）；
        - 回合内固定的逐像素偏置（noise_episode_bias 比例），模拟传感器
          固定模式噪声——纯逐帧 iid 噪声会被策略时间平均掉，偏乐观；
        - 逐像素独立缺失返回（holes → inf，与真实无效深度同路径）。
        """
        cfg = self.config
        if cfg.depth_noise_std_at_1m <= 0.0 and cfg.pixel_dropout_p <= 0.0:
            return depth
        out = depth.astype(np.float64, copy=True)
        finite = np.isfinite(out)
        if cfg.depth_noise_std_at_1m > 0.0 and finite.any():
            sigma = cfg.depth_noise_std_at_1m * np.square(out[finite])
            if self._noise_bias is None:
                self._noise_bias = np.zeros_like(out)
                self._noise_bias[finite] = self._noise_rng.normal(
                    0.0, cfg.noise_episode_bias * sigma
                )
            jitter = self._noise_rng.normal(
                0.0, (1.0 - cfg.noise_episode_bias) * sigma
            )
            noisy = out[finite] + self._noise_bias[finite] + jitter
            out[finite] = np.clip(noisy, 1e-4, None)
        if cfg.pixel_dropout_p > 0.0:
            drop = self._noise_rng.random(out.shape) < cfg.pixel_dropout_p
            out[drop] = np.inf
        return out.astype(np.float32)

    def _queue_capture(self, *, immediate: bool = False) -> None:
        # 整帧丢帧：传感器本次采集失败，交付云保持上一帧不变。
        if (
            not immediate
            and self.config.frame_drop_p > 0.0
            and self._noise_rng.random() < self.config.frame_drop_p
        ):
            return
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
        if self.config.render_mode == "mixed":
            self._active_render_mode = (
                "gripper_hidden"
                if int(self._render_rng.integers(0, 2)) == 1
                else "normal"
            )
        else:
            self._active_render_mode = self.config.render_mode
        # 每回合重播噪声种子：episode_seed 决定本回合的噪声流，
        # 配对评估里 deform/replay 两格用同一 seed → 相同噪声轨迹。
        episode_seed = getattr(self._base, "episode_seed", None)
        self._noise_rng = np.random.default_rng(
            (0 if episode_seed is None else int(episode_seed)) ^ 0x9E3779
        )
        self._noise_bias = None
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
            "pointcloud_masked_count": self.last_masked_count,
            "pointcloud_gripper_hidden": int(
                self._active_render_mode == "gripper_hidden"
            ),
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
