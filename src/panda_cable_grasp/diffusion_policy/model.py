"""Conditional action-diffusion model used by the cable-grasping method."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import DiffusionPolicyConfig


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, value: Tensor) -> Tensor:
        value = value.float().reshape(-1, 1)
        half = self.dimension // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=value.device, dtype=value.dtype)
        )
        embedding = value * frequencies.reshape(1, -1)
        result = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        if result.shape[-1] < self.dimension:
            result = torch.nn.functional.pad(result, (0, self.dimension - result.shape[-1]))
        return result


class ImageEncoder(nn.Module):
    """Small image encoder suitable for two 84x84 camera streams."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.projection(self.backbone(image))


class ResidualTemporalBlock(nn.Module):
    """FiLM-conditioned temporal convolution block.

    This is the core of the 1-D conditional U-Net style denoiser used in
    Diffusion Policy: actions remain a time sequence, while the visual/state
    context and diffusion timestep modulate every temporal feature channel.
    """

    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.film = nn.Linear(condition_dim, channels * 2)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        residual = value
        value = self.conv1(torch.nn.functional.silu(self.norm1(value)))
        scale, bias = self.film(condition).chunk(2, dim=-1)
        value = self.norm2(value)
        value = value * (1.0 + scale.unsqueeze(-1)) + bias.unsqueeze(-1)
        value = self.conv2(torch.nn.functional.silu(value))
        return residual + value


class ConditionalTemporalDenoiser(nn.Module):
    """Predict diffusion noise for an action chunk."""

    def __init__(
        self,
        action_dim: int,
        condition_dim: int,
        hidden_dim: int,
        blocks: int,
    ) -> None:
        super().__init__()
        time_dim = 128
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.condition = nn.Sequential(
            nn.Linear(condition_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_projection = nn.Conv1d(action_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(hidden_dim, hidden_dim) for _ in range(blocks)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, action_dim, kernel_size=1),
        )

    def forward(self, noisy_action: Tensor, timestep: Tensor, condition: Tensor) -> Tensor:
        time = self.time_embedding(timestep)
        condition = self.condition(torch.cat((condition, time), dim=-1))
        value = self.input_projection(noisy_action.transpose(1, 2))
        for block in self.blocks:
            value = block(value, condition)
        return self.output(value).transpose(1, 2)


class DDPMSchedule(nn.Module):
    """DDPM training schedule with deterministic DDIM-style inference."""

    def __init__(self, config: DiffusionPolicyConfig) -> None:
        super().__init__()
        betas = torch.linspace(
            config.beta_start, config.beta_end, config.diffusion_steps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def add_noise(self, clean: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
        alpha_bar = self.alpha_bars[timestep].reshape(-1, 1, 1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        denoiser,
        condition: Tensor,
        shape: tuple[int, int, int],
        inference_steps: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        value = torch.randn(shape, device=condition.device, generator=generator)
        schedule = torch.linspace(
            len(self.betas) - 1, 0, inference_steps, device=condition.device,
        ).round().long().unique()
        # ``linspace`` is descending; unique keeps the denoising order on
        # current PyTorch versions, but sort explicitly for portability.
        schedule = torch.sort(schedule, descending=True).values
        for index, timestep in enumerate(schedule):
            timestep_value = int(timestep.item())
            timestep_batch = torch.full(
                (shape[0],), timestep_value, device=condition.device, dtype=torch.long,
            )
            noise_prediction = denoiser(value, timestep_batch, condition)
            alpha_bar = self.alpha_bars[timestep_value]
            clean = (value - (1.0 - alpha_bar).sqrt() * noise_prediction) / alpha_bar.sqrt()
            clean = clean.clamp(-1.5, 1.5)
            if index == len(schedule) - 1:
                value = clean
                continue
            previous_timestep = int(schedule[index + 1].item())
            previous_alpha_bar = self.alpha_bars[previous_timestep]
            value = (
                previous_alpha_bar.sqrt() * clean
                + (1.0 - previous_alpha_bar).sqrt() * noise_prediction
            )
        return value


class DiffusionPolicy(nn.Module):
    """Two-camera + end-effector-state conditional action diffusion policy."""

    action_dim = 8
    state_dim = 7

    def __init__(self, config: DiffusionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config.image_feature_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, config.state_feature_dim),
            nn.LayerNorm(config.state_feature_dim),
            nn.SiLU(),
            nn.Linear(config.state_feature_dim, config.state_feature_dim),
            nn.SiLU(),
        )
        per_observation = 2 * config.image_feature_dim + config.state_feature_dim
        condition_dim = config.observation_horizon * per_observation
        self.denoiser = ConditionalTemporalDenoiser(
            self.action_dim,
            condition_dim,
            config.denoiser_dim,
            config.denoiser_blocks,
        )
        self.schedule = DDPMSchedule(config)

    def encode_condition(
        self, opst_cam: Tensor, wrist_cam: Tensor, state: Tensor
    ) -> Tensor:
        if opst_cam.ndim != 5 or wrist_cam.ndim != 5 or state.ndim != 3:
            raise ValueError("expected camera tensors (B,K,C,H,W) and state (B,K,7)")
        batch, horizon = opst_cam.shape[:2]
        if wrist_cam.shape[:2] != (batch, horizon) or state.shape[:2] != (batch, horizon):
            raise ValueError("observation history lengths must match")
        opst = self.image_encoder(opst_cam.reshape(batch * horizon, *opst_cam.shape[2:]))
        wrist = self.image_encoder(wrist_cam.reshape(batch * horizon, *wrist_cam.shape[2:]))
        state_feature = self.state_encoder(state.reshape(batch * horizon, -1))
        condition = torch.cat((opst, wrist, state_feature), dim=-1)
        return condition.reshape(batch, -1)

    def forward_loss(
        self,
        opst_cam: Tensor,
        wrist_cam: Tensor,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        condition = self.encode_condition(opst_cam, wrist_cam, state)
        timestep = torch.randint(
            0, self.config.diffusion_steps, (action.shape[0],),
            device=action.device,
        )
        noise = torch.randn_like(action)
        noisy_action = self.schedule.add_noise(action, noise, timestep)
        prediction = self.denoiser(noisy_action, timestep, condition)
        return torch.nn.functional.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(
        self,
        opst_cam: Tensor,
        wrist_cam: Tensor,
        state: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        condition = self.encode_condition(opst_cam, wrist_cam, state)
        return self.schedule.ddim_sample(
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
