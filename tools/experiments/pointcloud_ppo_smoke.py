"""Local smoke experiment for the RGB-D 384-point PPO observation path."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from panda_cable_grasp.rl.environment import make_rl_env
from panda_cable_grasp.rl.pointcloud import (
    DLOPointCloudObservation,
    PointCloudObservationConfig,
    PointNet2FeaturesExtractor,
)


def build_env(config: PointCloudObservationConfig) -> DLOPointCloudObservation:
    base = make_rl_env(
        action_mode="task_space_vertical_down",
        robot="nero",
        seed=20260914,
        disturbance_strength=1.5,
        episode_seconds=15.0,
        dynamicvla_cameras_enabled=True,
        scenario_names=("id_static",),
        geometric_safety_enabled=False,
        table_finger_collision_filter_enabled=False,
    )
    return DLOPointCloudObservation(base, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rl/train/pointcloud_ppo_local_smoke"),
    )
    args = parser.parse_args()
    config = PointCloudObservationConfig()
    env = build_env(config)
    try:
        check_env(env, warn=True)
        observation, info = env.reset(seed=20260914)
        print(
            "cloud",
            observation["points"].shape,
            "proprio",
            observation["proprio"].shape,
            "raw",
            info["pointcloud_raw_count"],
            "voxel",
            info["pointcloud_voxel_count"],
            "capture_ms",
            f"{info['pointcloud_capture_ms']:.3f}",
            flush=True,
        )

        extractor = PointNet2FeaturesExtractor(env.observation_space)
        points = torch.from_numpy(observation["points"])[None]
        proprio = torch.from_numpy(observation["proprio"])[None]
        for _ in range(10):
            extractor({"points": points, "proprio": proprio})
        started = perf_counter()
        for _ in range(100):
            extractor({"points": points, "proprio": proprio})
        latency_ms = 1000.0 * (perf_counter() - started) / 100.0
        parameters = sum(parameter.numel() for parameter in extractor.parameters())
        print(
            "extractor_parameters",
            parameters,
            "fp32_mb",
            f"{parameters * 4 / 1e6:.3f}",
            "cpu_forward_ms",
            f"{latency_ms:.3f}",
            flush=True,
        )

        args.output.mkdir(parents=True, exist_ok=True)
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=64,
            n_epochs=1,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={
                "features_extractor_class": PointNet2FeaturesExtractor,
                "activation_fn": torch.nn.ReLU,
                "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
            },
            verbose=1,
            seed=20260914,
            device=args.device,
        )
        model.learn(total_timesteps=args.timesteps)
        model.save(args.output / "model")
        print("saved", args.output / "model.zip", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
