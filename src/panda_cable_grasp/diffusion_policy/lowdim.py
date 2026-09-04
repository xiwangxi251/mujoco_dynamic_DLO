"""Low-dimensional Diffusion Policy for the Panda cable task.

This module is deliberately separate from the visual policy.  It consumes the
privileged expert ``.npz`` trajectories for DLO state, while reusing the
already converted LeRobot parquet actions when available.  Consequently the
training path never decodes video and never performs per-sample FK.

The observation is the absolute end-effector pose and gripper aperture,
followed by 16 uniformly sampled DLO keypoints relative to the gripper and the
relative target position.  ``include_velocity`` adds the corresponding DLO
and target velocities.  The history length is the experimental variable H.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..dynamicvla.finetune.convert_dataset import (
    JointTargetForwardKinematics,
    _select_frame_indices,
    _wxyz_to_euler_xyz,
)
from ..env.kinematics import rotation_to_quat
from .config import (
    ACTION_HIGH,
    ACTION_LOW,
    denormalize_array,
    normalize_array,
)
from .model import ConditionalUnet1D, DDPMSchedule
from scipy.spatial.transform import Rotation


def wxyz_to_euler_xyz(quaternions: np.ndarray) -> np.ndarray:
    """Use the DynamicVLA conversion convention for observations/actions."""

    return _wxyz_to_euler_xyz(quaternions)


def euler_xyz_to_wxyz(euler_angles: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles to scalar-first quaternions."""

    values = np.asarray(euler_angles, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError(f"Euler array must end in 3 values, got {values.shape}")
    return Rotation.from_euler("xyz", values).as_quat(
        scalar_first=True
    ).astype(np.float32)


@dataclass(frozen=True)
class LowDimPolicyConfig:
    """Architecture and optimization settings for the low-dimensional DP."""

    observation_horizon: int = 2
    prediction_horizon: int = 16
    action_horizon: int = 8
    num_keypoints: int = 16
    include_velocity: bool = False
    diffusion_step_embed_dim: int = 256
    unet_down_dims: tuple[int, ...] = (256, 512, 1024)
    unet_kernel_size: int = 5
    unet_groups: int = 8
    diffusion_steps: int = 100
    inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    batch_size: int = 100
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    seed: int = 20260804

    def __post_init__(self) -> None:
        for name in (
            "observation_horizon", "prediction_horizon", "action_horizon",
            "num_keypoints", "diffusion_step_embed_dim", "unet_kernel_size",
            "unet_groups", "diffusion_steps", "inference_steps", "batch_size",
            "warmup_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_horizon > self.prediction_horizon:
            raise ValueError("action_horizon cannot exceed prediction_horizon")
        if not self.unet_down_dims or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.unet_down_dims
        ):
            raise ValueError("unet_down_dims must contain positive integers")
        if any(value % self.unet_groups != 0 for value in self.unet_down_dims):
            raise ValueError("all unet_down_dims must be divisible by unet_groups")
        if self.inference_steps > self.diffusion_steps:
            raise ValueError("inference_steps cannot exceed diffusion_steps")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.grad_clip_norm < 0.0:
            raise ValueError("grad_clip_norm must be non-negative")

    @property
    def state_dim(self) -> int:
        base = 3 + 3 + 1 + self.num_keypoints * 3 + 3
        return base + (self.num_keypoints * 3 + 3 if self.include_velocity else 0)

    def asdict(self) -> dict:
        value = asdict(self)
        value["unet_down_dims"] = list(self.unet_down_dims)
        return value


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _keypoint_indices(node_count: int, count: int) -> np.ndarray:
    if node_count < 1:
        raise ValueError("DLO must contain at least one node")
    if count < 1:
        raise ValueError("num_keypoints must be positive")
    return np.rint(np.linspace(0, node_count - 1, count)).astype(np.int64)


def make_lowdim_feature(
    hand_position: np.ndarray,
    hand_quaternion: np.ndarray,
    cable_positions: np.ndarray,
    target_position: np.ndarray,
    gripper_aperture: float,
    *,
    keypoint_indices: np.ndarray,
    cable_velocities: np.ndarray | None = None,
    target_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Build one low-dimensional observation in the train/eval shared order."""

    hand_position = np.asarray(hand_position, dtype=np.float32).reshape(3)
    hand_quaternion = np.asarray(hand_quaternion, dtype=np.float32).reshape(4)
    cable_positions = np.asarray(cable_positions, dtype=np.float32)
    target_position = np.asarray(target_position, dtype=np.float32).reshape(3)
    if cable_positions.ndim != 2 or cable_positions.shape[-1] != 3:
        raise ValueError(f"cable_positions must have shape (N,3), got {cable_positions.shape}")
    indices = np.asarray(keypoint_indices, dtype=np.int64)
    if np.any(indices >= len(cable_positions)):
        raise ValueError("keypoint index exceeds cable node count")
    hand_euler = wxyz_to_euler_xyz(hand_quaternion)
    cable_relative = cable_positions[indices] - hand_position[None, :]
    target_relative = target_position - hand_position
    parts = [
        hand_position,
        hand_euler,
        np.asarray([gripper_aperture], dtype=np.float32),
        cable_relative.reshape(-1),
        target_relative,
    ]
    if cable_velocities is not None or target_velocity is not None:
        if cable_velocities is None or target_velocity is None:
            raise ValueError("cable and target velocities must be provided together")
        cable_velocities = np.asarray(cable_velocities, dtype=np.float32)
        target_velocity = np.asarray(target_velocity, dtype=np.float32).reshape(3)
        if cable_velocities.shape != cable_positions.shape:
            raise ValueError("cable velocity and position shapes must match")
        parts.extend([cable_velocities[indices].reshape(-1), target_velocity])
    return np.concatenate(parts).astype(np.float32, copy=False)


@lru_cache(maxsize=32)
def _model_gripper_qpos_addresses(model_path: str) -> tuple[int, ...]:
    import mujoco

    model = mujoco.MjModel.from_binary_path(model_path)
    addresses: list[int] = []
    for name in ("finger_joint1", "finger_joint2"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"gripper joint {name} not found in {model_path}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return tuple(addresses)


@lru_cache(maxsize=32)
def _forward_kinematics(model_path: str) -> JointTargetForwardKinematics:
    return JointTargetForwardKinematics(Path(model_path))


@dataclass
class LowDimEpisode:
    trajectory: Path
    scenario: str
    seed: int
    observation: np.ndarray
    action: np.ndarray


class _LeRobotActionCache:
    """Read preconverted 7D actions from the existing local LeRobot dataset."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        manifest_path = self.root / "cable_conversion_manifest.json"
        info_path = self.root / "meta" / "info.json"
        if not manifest_path.is_file() or not info_path.is_file():
            raise FileNotFoundError(
                f"LeRobot action cache is missing manifest/info files: {self.root}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        info = json.loads(info_path.read_text(encoding="utf-8"))
        self.chunk_size = int(info.get("chunks_size", 1000))
        self.source_kind = self.manifest.get("joint_target_source")
        self.entries = {
            str(Path(item["source"]).expanduser().resolve()): item
            for item in self.manifest.get("episodes", [])
        }
        if not self.entries:
            raise ValueError(f"LeRobot action cache has no episode manifest: {self.root}")

    def action_for(self, trajectory: Path, expected_length: int) -> np.ndarray | None:
        entry = self.entries.get(str(trajectory.expanduser().resolve()))
        if entry is None:
            return None
        episode_index = int(entry["episode_index"])
        parquet = (
            self.root / "data" / f"chunk-{episode_index // self.chunk_size:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        if not parquet.is_file():
            raise FileNotFoundError(parquet)
        import pandas as pd

        frame = pd.read_parquet(parquet, columns=["action"])
        action = np.asarray(frame["action"].tolist(), dtype=np.float32)
        if action.shape != (expected_length, 7):
            raise ValueError(
                f"cached action/frame mismatch for {trajectory}: "
                f"{action.shape} vs {(expected_length, 7)}"
            )
        return action


def _discover_trajectories(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".npz":
            paths.add(path)
        elif path.is_dir():
            paths.update(path.rglob("episode_*.npz"))
        else:
            raise FileNotFoundError(path)
    if not paths:
        raise ValueError("no episode_*.npz trajectories found")
    return sorted(path.resolve() for path in paths)


class LowDimEpisodeDataset(Dataset):
    """In-memory frame-window dataset with episode-safe history/future padding."""

    def __init__(
        self,
        inputs: Iterable[Path],
        *,
        config: LowDimPolicyConfig,
        action_source: str = "requested",
        action_cache: Path | None = None,
        episode_indices: Iterable[int] | None = None,
        gripper_threshold: float = 127.5,
    ) -> None:
        if action_source not in {"requested", "applied"}:
            raise ValueError("action_source must be requested or applied")
        self.config = config
        self.action_source = action_source
        self.gripper_threshold = float(gripper_threshold)
        self._action_cache = (
            _LeRobotActionCache(Path(action_cache)) if action_cache is not None else None
        )
        if self._action_cache is not None and self._action_cache.source_kind not in {
            None, action_source
        }:
            raise ValueError(
                f"action cache stores {self._action_cache.source_kind!r}, "
                f"but requested {action_source!r}"
            )
        self._all_episodes = [
            self._load_episode(path) for path in _discover_trajectories(inputs)
        ]
        self._total_episode_count = len(self._all_episodes)
        if self._total_episode_count < 1:
            raise ValueError("dataset contains no episodes")
        selected = (
            list(range(self._total_episode_count))
            if episode_indices is None else list(episode_indices)
        )
        self._set_subset(selected)
        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None

    def _load_episode(self, path: Path) -> LowDimEpisode:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "times", "hand_position", "hand_quaternion", "cable_positions",
                "target_positions", "states", "model_file", "scenario_name", "seed",
            }
            if self.config.include_velocity:
                required.update({"cable_velocities", "target_velocities"})
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"{path} is missing low-dimensional fields {missing}")
            times = np.asarray(data["times"], dtype=np.float64)
            selected = _select_frame_indices(times, 25)
            hand_position = np.asarray(data["hand_position"], dtype=np.float32)[selected]
            hand_quaternion = np.asarray(data["hand_quaternion"], dtype=np.float32)[selected]
            cable_positions = np.asarray(data["cable_positions"], dtype=np.float32)[selected]
            target_positions = np.asarray(data["target_positions"], dtype=np.float32)[selected]
            states = np.asarray(data["states"], dtype=np.float32)[selected]
            model_path = (path.parent / _scalar(data["model_file"])).resolve()
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            addresses = _model_gripper_qpos_addresses(str(model_path))
            # mj_getState() stores time before qpos, so convert model qpos
            # addresses to indices in the serialized state vector explicitly.
            qpos_indices = np.asarray(addresses, dtype=np.int64) + 1
            if states.shape[1] <= int(np.max(qpos_indices)):
                raise ValueError(f"states do not contain gripper qpos in {path}: {states.shape}")
            gripper = states[:, qpos_indices].sum(axis=1)
            keypoints = _keypoint_indices(cable_positions.shape[1], self.config.num_keypoints)
            cable_velocities = None
            target_velocities = None
            if self.config.include_velocity:
                cable_velocities = np.asarray(data["cable_velocities"], dtype=np.float32)[selected]
                target_velocities = np.asarray(data["target_velocities"], dtype=np.float32)[selected]
            feature_rows = [
                make_lowdim_feature(
                    hand_position[index], hand_quaternion[index], cable_positions[index],
                    target_positions[index], gripper[index], keypoint_indices=keypoints,
                    cable_velocities=None if cable_velocities is None else cable_velocities[index],
                    target_velocity=None if target_velocities is None else target_velocities[index],
                )
                for index in range(len(selected))
            ]
            observation = np.stack(feature_rows).astype(np.float32)
            command = np.asarray(data[f"{self.action_source}_actions"], dtype=np.float64)
            if command.shape != (len(times), 8):
                raise ValueError(
                    f"{self.action_source}_actions must have shape (T,8), got {command.shape}"
                )
            action = None if self._action_cache is None else self._action_cache.action_for(path, len(selected))
            if action is None:
                action_position, action_quaternion = _forward_kinematics(str(model_path)).poses(
                    command[selected, :7]
                )
                action = np.concatenate(
                    (
                        action_position,
                        wxyz_to_euler_xyz(action_quaternion),
                        np.where(command[selected, 7] > self.gripper_threshold, 1.0, -1.0)[:, None],
                    ),
                    axis=-1,
                ).astype(np.float32)
            scenario = _scalar(data["scenario_name"])
            seed = int(data["seed"])
        if len(observation) < 1 or len(action) != len(observation):
            raise ValueError(f"empty or misaligned low-dimensional episode: {path}")
        return LowDimEpisode(
            path, scenario, seed, observation, np.asarray(action, dtype=np.float32)
        )

    def _set_subset(self, selected: list[int]) -> None:
        if not selected:
            raise ValueError("dataset split contains no episodes")
        if any(index < 0 or index >= self._total_episode_count for index in selected):
            raise IndexError("episode split index is out of range")
        self.episode_indices = selected
        self._sample_offsets = np.cumsum(
            [0] + [len(self._all_episodes[index].observation) for index in selected],
            dtype=np.int64,
        )

    def subset(self, episode_indices: Iterable[int]) -> "LowDimEpisodeDataset":
        dataset = self.__class__.__new__(self.__class__)
        dataset.config = self.config
        dataset.action_source = self.action_source
        dataset.gripper_threshold = self.gripper_threshold
        dataset._action_cache = self._action_cache
        dataset._all_episodes = self._all_episodes
        dataset._total_episode_count = self._total_episode_count
        dataset._set_subset(list(episode_indices))
        dataset.obs_mean = self.obs_mean
        dataset.obs_std = self.obs_std
        return dataset

    @property
    def total_episode_count(self) -> int:
        return self._total_episode_count

    @property
    def episode_count(self) -> int:
        return len(self.episode_indices)

    @property
    def state_dim(self) -> int:
        return int(self._all_episodes[0].observation.shape[-1])

    @property
    def action_low(self) -> np.ndarray:
        return np.asarray(ACTION_LOW, dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.asarray(ACTION_HIGH, dtype=np.float32)

    def fit_observation_stats(
        self, episode_indices: Iterable[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = self.episode_indices if episode_indices is None else list(episode_indices)
        total = 0
        sums = np.zeros(self.state_dim, dtype=np.float64)
        squares = np.zeros(self.state_dim, dtype=np.float64)
        for index in selected:
            values = self._all_episodes[index].observation.astype(np.float64)
            total += len(values)
            sums += values.sum(axis=0)
            squares += np.square(values).sum(axis=0)
        if total < 1:
            raise ValueError("cannot fit observation stats on an empty split")
        mean = sums / total
        variance = np.maximum(squares / total - np.square(mean), 1.0e-8)
        std = np.maximum(np.sqrt(variance), 1.0e-3)
        return mean.astype(np.float32), std.astype(np.float32)

    def set_observation_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.shape != (self.state_dim,) or std.shape != (self.state_dim,):
            raise ValueError("observation statistics have the wrong shape")
        self.obs_mean = mean
        self.obs_std = np.maximum(std, 1.0e-3)

    def __len__(self) -> int:
        return int(self._sample_offsets[-1])

    @staticmethod
    def _history_indices(index: int, horizon: int) -> list[int]:
        return [max(0, index - horizon + 1 + offset) for offset in range(horizon)]

    @staticmethod
    def _future_indices(index: int, count: int, length: int) -> list[int]:
        return [min(length - 1, index + offset) for offset in range(count)]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        if self.obs_mean is None or self.obs_std is None:
            raise RuntimeError("call set_observation_stats before reading samples")
        if item < 0:
            item += len(self)
        episode_position = int(np.searchsorted(self._sample_offsets, item, side="right") - 1)
        frame_index = int(item - self._sample_offsets[episode_position])
        episode = self._all_episodes[self.episode_indices[episode_position]]
        history = self._history_indices(frame_index, self.config.observation_horizon)
        future = self._future_indices(
            frame_index, self.config.prediction_horizon, len(episode.action)
        )
        observation = (episode.observation[history] - self.obs_mean) / self.obs_std
        action = normalize_array(episode.action[future], ACTION_LOW, ACTION_HIGH)
        return {
            "obs": torch.from_numpy(np.asarray(observation, dtype=np.float32)),
            "action": torch.from_numpy(np.asarray(action, dtype=np.float32)),
        }


class LowDimDiffusionPolicy(nn.Module):
    """Conditional 1-D action diffusion model for low-dimensional observations."""

    action_dim = 7

    def __init__(self, config: LowDimPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        global_condition_dim = config.observation_horizon * self.state_dim
        self.denoiser = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_condition_dim=global_condition_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=tuple(config.unet_down_dims),
            kernel_size=config.unet_kernel_size,
            groups=config.unet_groups,
        )
        self.schedule = DDPMSchedule(config)

    def forward_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3 or action.ndim != 3:
            raise ValueError("expected obs (B,H,D) and action (B,T,7)")
        timestep = torch.randint(
            0, self.config.diffusion_steps, (action.shape[0],), device=action.device
        )
        noise = torch.randn_like(action)
        noisy_action = self.schedule.add_noise(action, noise, timestep)
        prediction = self.denoiser(
            noisy_action, timestep, obs.reshape(obs.shape[0], -1)
        )
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(
        self, obs: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return self.schedule.sample(
            self.denoiser,
            obs.reshape(obs.shape[0], -1),
            (obs.shape[0], self.config.prediction_horizon, self.action_dim),
            self.config.inference_steps,
            generator,
        )


class LowDimPolicyRunner:
    """Online receding-horizon runner sharing the DynamicVLA task-space adapter."""

    def __init__(
        self, env, checkpoint: Path, *, device: str = "cpu", deterministic: bool = True
    ) -> None:
        from ..dynamicvla.adapter import DynamicVLATaskSpaceAdapter

        self.env = env
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        try:
            payload = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            payload = torch.load(self.checkpoint_path, map_location=self.device)
        if not isinstance(payload, dict) or payload.get("format") != "panda_cable_lowdim_diffusion_policy_v1":
            raise ValueError("invalid low-dimensional Diffusion Policy checkpoint")
        values = dict(payload["config"])
        values["unet_down_dims"] = tuple(values["unet_down_dims"])
        self.config = LowDimPolicyConfig(**values)
        self.model = LowDimDiffusionPolicy(self.config).to(self.device)
        self.model.load_state_dict(payload["ema_model"] or payload["model"])
        self.model.eval()
        self.obs_mean = np.asarray(payload["obs_mean"], dtype=np.float32)
        self.obs_std = np.asarray(payload["obs_std"], dtype=np.float32)
        self.action_low = np.asarray(payload.get("action_low", ACTION_LOW), dtype=np.float32)
        self.action_high = np.asarray(payload.get("action_high", ACTION_HIGH), dtype=np.float32)
        self.keypoints = _keypoint_indices(len(env.cable_ids), self.config.num_keypoints)
        self.adapter = DynamicVLATaskSpaceAdapter(env)
        self.deterministic = bool(deterministic)
        self._history: deque[np.ndarray] = deque(maxlen=self.config.observation_horizon)
        self._action_queue: deque[np.ndarray] = deque()
        self._generator: torch.Generator | None = None
        self._inference_count = 0
        self._last_model_action = np.full(8, np.nan, dtype=np.float32)
        self.result = "running"
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.adapter.reset()
        self._history.clear()
        self._action_queue.clear()
        self._inference_count = 0
        self._last_model_action[:] = np.nan
        self.result = "running"
        if seed is None or not self.deterministic:
            self._generator = None
        else:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(int(seed))

    def _observation(self) -> np.ndarray:
        rotation = self.env.data.xmat[self.env.hand_id].reshape(3, 3)
        quaternion = rotation_to_quat(rotation)
        cable_positions = self.env.data.xpos[self.env.cable_ids].copy()
        target_position = self.env.target_position()
        cable_velocity = None
        target_velocity = None
        if self.config.include_velocity:
            cable_velocity = np.stack(
                [
                    self.env.body_linear_velocity(body_id)
                    for body_id in self.env.cable_ids
                ]
            )
            target_velocity = self.env.target_velocity()
        return make_lowdim_feature(
            self.env.hand_position,
            quaternion,
            cable_positions,
            target_position,
            float(np.sum(self.env.data.qpos[self.env.finger_qpos_adr])),
            keypoint_indices=self.keypoints,
            cable_velocities=cable_velocity,
            target_velocity=target_velocity,
        )

    def _sample_action_chunk(self) -> None:
        observation = np.asarray(self._history, dtype=np.float32)
        value = torch.from_numpy(observation)[None].to(self.device)
        normalized = self.model.sample(value, generator=self._generator)[0].cpu().numpy()
        model_chunk = denormalize_array(normalized, self.action_low, self.action_high)
        task_space_chunk = np.concatenate(
            (
                model_chunk[:, :3],
                euler_xyz_to_wxyz(model_chunk[:, 3:6]),
                model_chunk[:, 6:7],
            ),
            axis=-1,
        )
        self._action_queue.extend(
            np.asarray(task_space_chunk[: self.config.action_horizon], dtype=np.float32)
        )
        self._inference_count += 1

    def action(self) -> np.ndarray:
        observation = (self._observation() - self.obs_mean) / self.obs_std
        self._history.append(observation.astype(np.float32))
        while len(self._history) < self.config.observation_horizon:
            self._history.appendleft(self._history[0].copy())
        if not self._action_queue:
            self._sample_action_chunk()
        task_space_action = self._action_queue.popleft()
        self._last_model_action[:] = task_space_action
        self.adapter.set_model_action(task_space_action)
        return self.adapter.action()

    def policy_info(self) -> dict:
        diagnostics = self.adapter.diagnostics()
        return {
            "diffusion_inference_count": self._inference_count,
            "diffusion_action_queue": len(self._action_queue),
            "diffusion_model_action": self._last_model_action.copy(),
            "diffusion_position_clipped": bool(diagnostics["position_clipped"]),
            "diffusion_quaternion_repaired": bool(diagnostics["quaternion_repaired"]),
        }


__all__ = [
    "LowDimDiffusionPolicy", "LowDimEpisodeDataset", "LowDimPolicyConfig",
    "LowDimPolicyRunner", "make_lowdim_feature",
]
