"""DynaMimicGen-style visual Diffusion Policy for cable grasping.

The implementation follows the image-based robomimic / DynaMimicGen design:
independent ResNet18 + SpatialSoftmax encoders for each camera, raw low-
dimensional state concatenation, and a FiLM-conditioned 1-D temporal U-Net
that predicts diffusion noise for an action sequence.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as vision_models

from .config import DiffusionPolicyConfig


def _replace_batchnorm_with_groupnorm(
    module: nn.Module, features_per_group: int = 16
) -> None:
    """Replace ResNet BatchNorm with GroupNorm as in the reference setup."""

    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            groups = max(1, child.num_features // features_per_group)
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            _replace_batchnorm_with_groupnorm(child, features_per_group)


class SpatialSoftmax(nn.Module):
    """Learn keypoint heatmaps and return expected normalized coordinates."""

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        num_keypoints: int,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.keypoint_projection = nn.Conv2d(
            input_channels, num_keypoints, kernel_size=1
        )
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, input_width),
            np.linspace(-1.0, 1.0, input_height),
            indexing="xy",
        )
        self.register_buffer(
            "pos_x", torch.from_numpy(pos_x.reshape(1, input_height * input_width)).float()
        )
        self.register_buffer(
            "pos_y", torch.from_numpy(pos_y.reshape(1, input_height * input_width)).float()
        )
        self.temperature = float(temperature)
        self.num_keypoints = num_keypoints

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.keypoint_projection(feature)
        batch = feature.shape[0]
        attention = F.softmax(
            feature.reshape(batch, self.num_keypoints, -1) / self.temperature,
            dim=-1,
        )
        expected_x = torch.sum(self.pos_x * attention, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=-1, keepdim=True)
        return torch.cat((expected_x, expected_y), dim=-1)


class ResNet18SpatialSoftmaxEncoder(nn.Module):
    """Reference image encoder: ResNet18, spatial softmax, linear projection."""

    def __init__(self, config: DiffusionPolicyConfig) -> None:
        super().__init__()
        network = vision_models.resnet18(weights=None)
        _replace_batchnorm_with_groupnorm(network)
        self.backbone = nn.Sequential(*list(network.children())[:-2])
        feature_height = math.ceil(config.crop_height / 32)
        feature_width = math.ceil(config.crop_width / 32)
        self.spatial_softmax = SpatialSoftmax(
            input_channels=512,
            input_height=feature_height,
            input_width=feature_width,
            num_keypoints=config.spatial_num_keypoints,
        )
        self.projection = nn.Linear(
            config.spatial_num_keypoints * 2, config.image_feature_dim
        )
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.crop_height = config.crop_height
        self.crop_width = config.crop_width

    def _crop(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != (self.image_height, self.image_width):
            image = F.interpolate(
                image,
                size=(self.image_height, self.image_width),
                mode="bilinear",
                align_corners=False,
            )
        if (self.crop_height, self.crop_width) == (
            self.image_height,
            self.image_width,
        ):
            return image
        if self.training:
            max_y = self.image_height - self.crop_height
            max_x = self.image_width - self.crop_width
            top = torch.randint(0, max_y + 1, (image.shape[0],), device=image.device)
            left = torch.randint(0, max_x + 1, (image.shape[0],), device=image.device)
        else:
            top = torch.full(
                (image.shape[0],),
                (self.image_height - self.crop_height) // 2,
                device=image.device,
                dtype=torch.long,
            )
            left = torch.full(
                (image.shape[0],),
                (self.image_width - self.crop_width) // 2,
                device=image.device,
                dtype=torch.long,
            )
        return torch.stack(
            [
                image[index, :, row : row + self.crop_height, column : column + self.crop_width]
                for index, (row, column) in enumerate(zip(top.tolist(), left.tolist()))
            ]
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self._crop(image)
        feature = self.backbone(image)
        keypoints = self.spatial_softmax(feature)
        return self.projection(keypoints.flatten(start_dim=1))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().reshape(-1, 1)
        half = self.dimension // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=value.device, dtype=value.dtype)
        )
        embedding = value * frequencies.reshape(1, -1)
        result = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        if result.shape[-1] < self.dimension:
            result = F.pad(result, (0, self.dimension - result.shape[-1]))
        return result


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish, matching the reference U-Net block."""

    def __init__(
        self, input_channels: int, output_channels: int, kernel_size: int, groups: int
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(groups, output_channels),
            nn.Mish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ConditionalResidualBlock1D(nn.Module):
    """Residual temporal block with FiLM scale and bias conditioning."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        condition_dim: int,
        kernel_size: int,
        groups: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(input_channels, output_channels, kernel_size, groups),
                Conv1dBlock(output_channels, output_channels, kernel_size, groups),
            ]
        )
        self.condition_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dim, output_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.output_channels = output_channels

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        result = self.blocks[0](value)
        modulation = self.condition_encoder(condition).reshape(
            condition.shape[0], 2, self.output_channels, 1
        )
        result = modulation[:, 0] * result + modulation[:, 1]
        result = self.blocks[1](result)
        return result + self.residual_conv(value)


class Downsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class Upsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class ConditionalUnet1D(nn.Module):
    """Three-level FiLM-conditioned temporal U-Net from the reference DP."""

    def __init__(
        self,
        input_dim: int,
        global_condition_dim: int,
        diffusion_step_embed_dim: int,
        down_dims: tuple[int, ...],
        kernel_size: int,
        groups: int,
    ) -> None:
        super().__init__()
        all_dims = [input_dim, *down_dims]
        start_dim = down_dims[0]
        condition_dim = diffusion_step_embed_dim + global_condition_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, condition_dim, kernel_size, groups
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, condition_dim, kernel_size, groups
                ),
            ]
        )
        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, condition_dim, kernel_size, groups
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, condition_dim, kernel_size, groups
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.up_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = index >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            condition_dim,
                            kernel_size,
                            groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, condition_dim, kernel_size, groups
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = sample.moveaxis(-1, -2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=value.device)
        elif timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(device=value.device).expand(value.shape[0])
        condition = self.diffusion_step_encoder(timestep)
        if global_condition is not None:
            condition = torch.cat((condition, global_condition), dim=-1)

        skips: list[torch.Tensor] = []
        for residual, residual2, downsample in self.down_modules:
            value = residual(value, condition)
            value = residual2(value, condition)
            skips.append(value)
            value = downsample(value)
        for residual, residual2, upsample in self.up_modules:
            value = torch.cat((value, skips.pop()), dim=1)
            value = residual(value, condition)
            value = residual2(value, condition)
            value = upsample(value)
        return self.final_conv(value).moveaxis(-1, -2)


class DDPMSchedule:
    """Diffusers DDPM schedule used for both training and sampling."""

    def __init__(self, config: DiffusionPolicyConfig) -> None:
        try:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        except ImportError as error:
            raise RuntimeError(
                "Diffusion Policy requires diffusers; install with "
                "`python -m pip install -e \".[diffusion]\"`"
            ) from error
        kwargs = {
            "num_train_timesteps": config.diffusion_steps,
            "beta_schedule": config.beta_schedule,
            "clip_sample": config.clip_sample,
            "prediction_type": "epsilon",
        }
        self.training = DDPMScheduler(**kwargs)
        self.inference = DDPMScheduler(**kwargs)

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.training.add_noise(clean, noise, timestep)

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        condition: torch.Tensor,
        shape: tuple[int, int, int],
        inference_steps: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        value = torch.randn(shape, device=condition.device, generator=generator)
        try:
            self.inference.set_timesteps(inference_steps, device=condition.device)
        except TypeError:
            self.inference.set_timesteps(inference_steps)
        for timestep in self.inference.timesteps:
            noise_prediction = denoiser(value, timestep, condition)
            kwargs = {
                "model_output": noise_prediction,
                "timestep": timestep,
                "sample": value,
            }
            if generator is not None:
                kwargs["generator"] = generator
            value = self.inference.step(**kwargs).prev_sample
        return value


class DiffusionPolicy(nn.Module):
    """DynaMimicGen-style two-camera conditional action diffusion policy."""

    action_dim = 8
    state_dim = 7

    def __init__(self, config: DiffusionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.opst_encoder = ResNet18SpatialSoftmaxEncoder(config)
        self.wrist_encoder = ResNet18SpatialSoftmaxEncoder(config)
        per_observation = 2 * config.image_feature_dim + self.state_dim
        global_condition_dim = config.observation_horizon * per_observation
        self.denoiser = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_condition_dim=global_condition_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.unet_down_dims,
            kernel_size=config.unet_kernel_size,
            groups=config.unet_groups,
        )
        self.schedule = DDPMSchedule(config)

    def encode_condition(
        self, opst_cam: torch.Tensor, wrist_cam: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        if opst_cam.ndim != 5 or wrist_cam.ndim != 5 or state.ndim != 3:
            raise ValueError("expected camera tensors (B,K,C,H,W) and state (B,K,7)")
        batch, horizon = opst_cam.shape[:2]
        if wrist_cam.shape[:2] != (batch, horizon) or state.shape[:2] != (batch, horizon):
            raise ValueError("observation history lengths must match")
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state dimension {self.state_dim}, got {state.shape[-1]}")
        opst = self.opst_encoder(opst_cam.reshape(batch * horizon, *opst_cam.shape[2:]))
        wrist = self.wrist_encoder(wrist_cam.reshape(batch * horizon, *wrist_cam.shape[2:]))
        condition = torch.cat((opst, wrist, state.reshape(batch * horizon, -1)), dim=-1)
        return condition.reshape(batch, -1)

    def forward_loss(
        self,
        opst_cam: torch.Tensor,
        wrist_cam: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.encode_condition(opst_cam, wrist_cam, state)
        timestep = torch.randint(
            0,
            self.config.diffusion_steps,
            (action.shape[0],),
            device=action.device,
        )
        noise = torch.randn_like(action)
        noisy_action = self.schedule.add_noise(action, noise, timestep)
        prediction = self.denoiser(noisy_action, timestep, condition)
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(
        self,
        opst_cam: torch.Tensor,
        wrist_cam: torch.Tensor,
        state: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        condition = self.encode_condition(opst_cam, wrist_cam, state)
        return self.schedule.sample(
            self.denoiser,
            condition,
            (
                opst_cam.shape[0],
                self.config.prediction_horizon,
                self.action_dim,
            ),
            self.config.inference_steps,
            generator,
        )
