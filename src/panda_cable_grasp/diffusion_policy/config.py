"""Configuration and fixed normalization for the DynamicVLA-compatible DP."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DiffusionPolicyConfig:
    """Training and inference parameters for the visual action diffusion model.

    The defaults mirror the image-based Diffusion Policy configuration used by
    DynaMimicGen: 84x84 RGB inputs with a 76x76 crop, a short observation
    history, a 16-step prediction chunk, a ResNet18 + SpatialSoftmax visual
    encoder, and a three-level 1-D conditional U-Net action head.
    """

    observation_horizon: int = 2
    prediction_horizon: int = 16
    action_horizon: int = 8
    image_height: int = 84
    image_width: int = 84
    crop_height: int = 76
    crop_width: int = 76
    image_feature_dim: int = 64
    spatial_num_keypoints: int = 32
    diffusion_step_embed_dim: int = 256
    unet_down_dims: tuple[int, ...] = (512, 1024, 2048)
    unet_kernel_size: int = 5
    unet_groups: int = 8
    diffusion_steps: int = 100
    inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    ema_enabled: bool = True
    ema_power: float = 0.75
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    batch_size: int = 16
    epochs: int = 2000
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    cache_episodes: int = 2
    seed: int = 20260804

    def __post_init__(self) -> None:
        positive_integer_names = (
            "observation_horizon", "prediction_horizon", "action_horizon",
            "image_height", "image_width", "crop_height", "crop_width",
            "image_feature_dim", "spatial_num_keypoints", "diffusion_step_embed_dim",
            "unet_kernel_size", "unet_groups", "diffusion_steps",
            "inference_steps", "batch_size", "epochs", "warmup_steps", "cache_episodes",
        )
        for name in positive_integer_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_horizon > self.prediction_horizon:
            raise ValueError("action_horizon cannot exceed prediction_horizon")
        if self.crop_height > self.image_height or self.crop_width > self.image_width:
            raise ValueError("crop dimensions cannot exceed image dimensions")
        if not self.unet_down_dims:
            raise ValueError("unet_down_dims must not be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.unet_down_dims
        ):
            raise ValueError("unet_down_dims must contain positive integers")
        if any(value % self.unet_groups != 0 for value in self.unet_down_dims):
            raise ValueError("all unet_down_dims must be divisible by unet_groups")
        if self.inference_steps > self.diffusion_steps:
            raise ValueError("inference_steps cannot exceed diffusion_steps")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int):
            raise ValueError("num_workers must be an integer")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not isinstance(self.beta_schedule, str) or not self.beta_schedule:
            raise ValueError("beta_schedule must be a non-empty string")
        for name in (
            "learning_rate", "weight_decay", "grad_clip_norm", "ema_power",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


# DynamicVLA's task-space action is [xyz, quaternion(wxyz), gripper].  The
# position bounds intentionally match DynamicVLAAdapterConfig.  Quaternion and
# gripper values are already naturally represented in [-1, 1].
STATE_LOW = (
    0.20, -0.55, 0.005, -1.0, -1.0, -1.0, -1.0,
)
STATE_HIGH = (
    0.85, 0.55, 0.70, 1.0, 1.0, 1.0, 1.0,
)
ACTION_LOW = (
    0.20, -0.55, 0.005, -1.0, -1.0, -1.0, -1.0, -1.0,
)
ACTION_HIGH = (
    0.85, 0.55, 0.70, 1.0, 1.0, 1.0, 1.0, 1.0,
)


def normalize_array(value, low, high):
    """Map a NumPy-like array to the diffusion model's approximately [-1, 1]."""

    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    lower = np.asarray(low, dtype=np.float32)
    upper = np.asarray(high, dtype=np.float32)
    return 2.0 * (array - lower) / (upper - lower) - 1.0


def denormalize_array(value, low, high):
    """Map a model array back to the physical DynamicVLA task space."""

    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    lower = np.asarray(low, dtype=np.float32)
    upper = np.asarray(high, dtype=np.float32)
    return lower + 0.5 * (array + 1.0) * (upper - lower)


def split_episode_indices(
    episode_count: int, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Make a deterministic episode-level split, avoiding frame leakage."""

    import numpy as np

    if episode_count < 1:
        raise ValueError("episode_count must be positive")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    order = np.arange(episode_count, dtype=np.int64)
    np.random.default_rng(seed).shuffle(order)
    if episode_count == 1 or validation_fraction == 0.0:
        return order.tolist(), []
    validation_count = max(1, int(round(episode_count * validation_fraction)))
    validation_count = min(validation_count, episode_count - 1)
    return order[validation_count:].tolist(), order[:validation_count].tolist()
