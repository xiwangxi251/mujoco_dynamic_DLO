"""Train the low-dimensional Panda cable Diffusion Policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
import random
import shutil
import time


class ExponentialMovingAverage:
    def __init__(self, model, power: float = 0.75) -> None:
        self.power = float(power)
        self.step_count = 0
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    def update(self, model) -> None:
        self.step_count += 1
        decay = 1.0 - (self.step_count + 1) ** (-self.power)
        for name, value in model.state_dict().items():
            if value.is_floating_point() or value.is_complex():
                self.shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                self.shadow[name].copy_(value.detach())

    def state_dict(self) -> dict:
        return {name: value.clone() for name, value in self.shadow.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a low-dimensional Diffusion Policy on cable expert NPZ data"
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", type=Path,
        help="initialize model and EMA weights from a previous checkpoint",
    )
    parser.add_argument(
        "--action-cache", type=Path,
        help="LeRobot action cache for raw privileged NPZ input; not needed for NERO success-prefix input",
    )
    parser.add_argument("--action-source", choices=("requested", "applied"), default="requested")
    parser.add_argument(
        "--feature-cache", type=Path, nargs="+",
        help="Optional cache of replayed NERO privileged features (*.npz, keyed by episode_index)",
    )
    parser.add_argument("--rotation-format", choices=("euler", "rotvec"), default="euler")
    parser.add_argument("--variant", choices=("p0", "h2", "h7"), default="h2")
    parser.add_argument(
        "--state-input", choices=("dlo58", "ee6"), default="dlo58",
        help="NERO success-prefix input: replayed DLO state (default) or legacy 6D EE pose",
    )
    parser.add_argument("--num-keypoints", type=int, default=16)
    parser.add_argument("--observation-horizon", type=int)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--max-validation-batches", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.variant == "p0":
        args.include_velocity = True
        default_horizon = 2
    elif args.variant == "h2":
        args.include_velocity = False
        default_horizon = 2
    else:
        args.include_velocity = False
        default_horizon = 7
    if args.observation_horizon is None:
        args.observation_horizon = default_horizon
    for name in (
        "num_keypoints", "observation_horizon", "prediction_horizon", "action_horizon",
        "inference_steps", "max_steps", "batch_size", "warmup_steps", "eval_every",
        "max_validation_batches",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.limit_episodes is not None and args.limit_episodes <= 0:
        parser.error("--limit-episodes must be positive")
    return args


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset, args, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def _evaluate(model, loader, device, max_batches: int) -> float | None:
    if loader is None:
        return None
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            obs = batch["obs"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            losses.append(float(model.forward_loss(obs, action).item()))
    return sum(losses) / len(losses) if losses else None


def train(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch

    from .config import ACTION_HIGH, ACTION_LOW, split_episode_indices
    from .lowdim import LowDimDiffusionPolicy, LowDimEpisodeDataset, LowDimPolicyConfig, _discover_trajectories

    _set_seed(args.seed)
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output}; use --overwrite explicitly")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    config = LowDimPolicyConfig(
        observation_horizon=args.observation_horizon,
        prediction_horizon=args.prediction_horizon,
        action_horizon=args.action_horizon,
        num_keypoints=args.num_keypoints,
        include_velocity=args.include_velocity,
        inference_steps=args.inference_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        rotation_format=args.rotation_format,
    )
    roots = [p.expanduser().resolve() for p in args.inputs]
    is_nero_success_prefix = all(
        (r / "cable_conversion_manifest.json").is_file()
        and (r / "meta" / "info.json").is_file()
        for r in roots
    )
    if is_nero_success_prefix:
        if args.state_input == "ee6":
            config = replace(config, observation_dim=6)
        caches = list(args.feature_cache or [])
        if len(caches) == 1 and len(roots) > 1:
            caches = caches * len(roots)
        if caches and len(caches) != len(roots):
            raise ValueError("--feature-cache count must be 1 or match inputs")
        if not caches:
            caches = [None] * len(roots)
        parts = [
            LowDimEpisodeDataset.from_nero_success_prefix(
                r,
                config=config,
                action_source=args.action_source,
                limit_episodes=args.limit_episodes,
                feature_cache=c,
            )
            for r, c in zip(roots, caches)
        ]
        all_dataset = parts[0]
        if len(parts) > 1:
            merged = [e for d in parts for e in d._all_episodes]
            all_dataset._all_episodes = merged
            all_dataset._total_episode_count = len(merged)
            all_dataset._set_subset(list(range(len(merged))))
        all_paths = roots
    else:
        if args.action_cache is None:
            raise ValueError("--action-cache is required for raw NPZ low-dimensional input")
        all_paths = _discover_trajectories(args.inputs)
        if args.limit_episodes is not None:
            if args.limit_episodes > len(all_paths):
                raise ValueError(f"requested {args.limit_episodes} episodes but found {len(all_paths)}")
            all_paths = all_paths[: args.limit_episodes]
        all_dataset = LowDimEpisodeDataset(
            all_paths,
            config=config,
            action_source=args.action_source,
            action_cache=args.action_cache,
        )
    started = time.monotonic()
    train_indices, validation_indices = split_episode_indices(
        all_dataset.total_episode_count, args.validation_fraction, args.seed
    )
    train_dataset = all_dataset.subset(train_indices)
    validation_dataset = all_dataset.subset(validation_indices) if validation_indices else None
    obs_mean, obs_std = all_dataset.fit_observation_stats(train_indices)
    obs_pose_norm_low, obs_pose_norm_high = all_dataset.fit_observation_pose_quantiles(
        train_indices
    )
    action_norm_low, action_norm_high = all_dataset.fit_action_quantiles(train_indices)
    train_dataset.set_observation_stats(obs_mean, obs_std)
    train_dataset.set_observation_pose_norm_bounds(obs_pose_norm_low, obs_pose_norm_high)
    train_dataset.set_action_norm_bounds(action_norm_low, action_norm_high)
    if validation_dataset is not None:
        validation_dataset.set_observation_stats(obs_mean, obs_std)
        validation_dataset.set_observation_pose_norm_bounds(
            obs_pose_norm_low, obs_pose_norm_high
        )
        validation_dataset.set_action_norm_bounds(action_norm_low, action_norm_high)
    train_loader = _loader(train_dataset, args, shuffle=True)
    validation_loader = None if validation_dataset is None else _loader(validation_dataset, args, shuffle=False)
    print(
        f"dataset_ready episodes={all_dataset.total_episode_count} train={train_dataset.episode_count} "
        f"validation={0 if validation_dataset is None else validation_dataset.episode_count} "
        f"train_frames={len(train_dataset)} validation_frames={None if validation_dataset is None else len(validation_dataset)} "
        f"state_dim={config.state_dim} action_dim=7 variant={args.variant} include_velocity={config.include_velocity} "
        f"action_source={args.action_source} action_target=pi05_chunk_delta_xyz_euler_gripper "
        f"state_input={args.state_input} "
        f"normalization=pi05_q01_q99_clip rotation_format={args.rotation_format} "
        f"preprocess_seconds={time.monotonic() - started:.1f}",
        flush=True,
    )

    device = torch.device(args.device)
    model = LowDimDiffusionPolicy(config).to(device)
    resume_payload = None
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        if resume_payload.get("format") != "panda_cable_lowdim_diffusion_policy_v1":
            raise ValueError(f"unsupported resume checkpoint: {args.resume}")
        model.load_state_dict(resume_payload["model"])
    ema = ExponentialMovingAverage(model)
    if resume_payload is not None and resume_payload.get("ema_model") is not None:
        ema.shadow = {
            name: value.detach().clone().to(device)
            for name, value in resume_payload["ema_model"].items()
        }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def schedule(step: int) -> float:
        if step < config.warmup_steps:
            return max(1.0e-8, (step + 1) / max(1, config.warmup_steps))
        progress = (step - config.warmup_steps) / max(1, args.max_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    metrics_path = output / "training_metrics.csv"
    best_validation = float("inf")
    global_step = 0
    epoch = 0
    last_validation = None
    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=("epoch", "step", "train_loss", "validation_loss", "learning_rate"),
        )
        writer.writeheader()
        while global_step < args.max_steps:
            epoch += 1
            model.train()
            epoch_losses = []
            for batch_index, batch in enumerate(train_loader, start=1):
                if global_step >= args.max_steps:
                    break
                obs = batch["obs"].to(device, non_blocking=True)
                action = batch["action"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = model.forward_loss(obs, action)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
                ema.update(model)
                scheduler.step()
                global_step += 1
                epoch_losses.append(float(loss.item()))
                if batch_index == 1 or batch_index % 100 == 0 or global_step == args.max_steps:
                    print(
                        f"progress variant={args.variant} epoch={epoch} batch={batch_index}/{len(train_loader)} "
                        f"percent={100.0 * batch_index / len(train_loader):.1f} step={global_step}/{args.max_steps} "
                        f"loss={loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.3e}",
                        flush=True,
                    )
                if global_step % args.eval_every == 0 or global_step == args.max_steps:
                    last_validation = _evaluate(model, validation_loader, device, args.max_validation_batches)
                    train_loss = sum(epoch_losses) / len(epoch_losses)
                    writer.writerow({
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": train_loss,
                        "validation_loss": last_validation,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    })
                    metrics_file.flush()
                    checkpoint = {
                        "format": "panda_cable_lowdim_diffusion_policy_v1",
                        "model": model.state_dict(),
                        "ema_model": ema.state_dict(),
                        "config": config.asdict(),
                        "obs_mean": obs_mean.tolist(),
                        "obs_std": obs_std.tolist(),
                        "obs_pose_norm_low": obs_pose_norm_low.tolist(),
                        "obs_pose_norm_high": obs_pose_norm_high.tolist(),
                        "action_norm_low": action_norm_low.tolist(),
                        "action_norm_high": action_norm_high.tolist(),
                        # Keep the old field names readable by older tooling.
                        "action_low": action_norm_low.tolist(),
                        "action_high": action_norm_high.tolist(),
                        "action_source": args.action_source,
                        "action_target": "pi05_chunk_delta_xyz_euler_gripper",
                        "observation_normalization": "pi05_q01_q99_pose_plus_train_mean_std_privileged",
                        "action_normalization": "pi05_q01_q99_clip",
                        "variant": args.variant,
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_parameter_count": sum(p.numel() for p in model.parameters()),
                    }
                    torch.save(checkpoint, output / "checkpoint_latest.pt")
                    score = last_validation if last_validation is not None else train_loss
                    if score < best_validation:
                        best_validation = score
                        torch.save(checkpoint, output / "checkpoint_best.pt")
                    print(
                        f"checkpoint step={global_step} epoch={epoch} train_loss={train_loss:.6f} "
                        f"validation_loss={last_validation if last_validation is not None else 'n/a'}",
                        flush=True,
                    )
            if global_step >= args.max_steps:
                break

    manifest = {
        "format": "panda_cable_lowdim_diffusion_policy_v1",
        "method": "diffusion_policy_lowdim",
        "variant": args.variant,
        "architecture": {
            "observation": (
                "ee_xyz_euler (6D DynamicVLA state)"
                if config.observation_dim == 6
                else "ee_xyz_euler_gripper + 16 DLO keypoints relative to EE + relative target"
            ),
            "include_velocity": config.include_velocity,
            "state_dim": config.state_dim,
            "action_dim": 7,
            "action_head": "conditional_unet_1d",
            "unet_down_dims": list(config.unet_down_dims),
            "ema": True,
        },
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "config": config.asdict(),
        "data": {
            "inputs": [str(path) for path in all_paths],
            "action_cache": (
                None
                if args.action_cache is None
                else str(args.action_cache.expanduser().resolve())
            ),
            "train_episode_indices": train_indices,
            "validation_episode_indices": validation_indices,
            "train_frame_count": len(train_dataset),
            "validation_frame_count": None if validation_dataset is None else len(validation_dataset),
            "action_source": args.action_source,
            "action_target": "pi05_chunk_delta_xyz_euler_gripper",
            "observation_normalization": "pi05_q01_q99_pose_plus_train_mean_std_privileged",
            "observation_pose_q01": obs_pose_norm_low.tolist(),
            "observation_pose_q99": obs_pose_norm_high.tolist(),
            "action_normalization": "pi05_q01_q99_clip",
            "action_q01": action_norm_low.tolist(),
            "action_q99": action_norm_high.tolist(),
            "frame_rate_hz": 25,
        },
        "checkpoints": ["checkpoint_latest.pt", "checkpoint_best.pt"],
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"lowdim_output={output} final_step={global_step} final_epoch={epoch}", flush=True)
    return output


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
