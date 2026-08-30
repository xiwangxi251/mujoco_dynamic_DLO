"""Train the cable-grasping visual Diffusion Policy from expert episodes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DynamicVLA-compatible visual Diffusion Policy"
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="expert run/scenario directories or episode_*.npz trajectories",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-source", choices=("requested", "applied"), default="applied")
    parser.add_argument("--gripper-threshold", type=float, default=127.5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--observation-horizon", type=int)
    parser.add_argument("--prediction-horizon", type=int)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-validation-batches", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    for name in (
        "limit_episodes", "epochs", "batch_size", "observation_horizon",
        "prediction_horizon", "action_horizon", "inference_steps", "num_workers",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.max_validation_batches <= 0:
        parser.error("--max-validation-batches must be positive")
    return args


def _config_from_args(args: argparse.Namespace):
    from .config import DiffusionPolicyConfig

    values = asdict(DiffusionPolicyConfig(seed=args.seed))
    for key in (
        "epochs", "batch_size", "learning_rate", "observation_horizon",
        "prediction_horizon", "action_horizon", "inference_steps", "num_workers",
    ):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    return DiffusionPolicyConfig(**values)


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset, *, config, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def _evaluate(model, loader, device, max_batches: int) -> float | None:
    if loader is None:
        return None
    import torch

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            # Diffusion training samples a fresh timestep each call; the
            # validation number is therefore an unbiased noisy estimate.
            losses.append(float(model.forward_loss(**batch).item()))
    return sum(losses) / len(losses) if losses else None


def train(args: argparse.Namespace) -> Path:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for Diffusion Policy; install the optional "
            "extra with `python -m pip install -e \".[diffusion]\"`"
        ) from error

    from .config import DiffusionPolicyConfig
    from .dataset import DiffusionEpisodeDataset, split_episode_indices
    from .model import DiffusionPolicy

    config = _config_from_args(args)
    _set_seed(args.seed)
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output}; use --overwrite explicitly")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    all_dataset = DiffusionEpisodeDataset(
        args.inputs,
        config=config,
        action_source=args.action_source,
        gripper_threshold=args.gripper_threshold,
    )
    episode_indices = list(range(all_dataset.episode_count))
    if args.limit_episodes is not None:
        if args.limit_episodes > len(episode_indices):
            raise ValueError(
                f"requested {args.limit_episodes} episodes but only "
                f"{len(episode_indices)} were found"
            )
        episode_indices = episode_indices[: args.limit_episodes]
    train_indices, validation_indices = split_episode_indices(
        len(episode_indices), args.validation_fraction, args.seed
    )
    train_episode_indices = [episode_indices[index] for index in train_indices]
    validation_episode_indices = [episode_indices[index] for index in validation_indices]
    train_dataset = DiffusionEpisodeDataset(
        args.inputs,
        config=config,
        action_source=args.action_source,
        gripper_threshold=args.gripper_threshold,
        episode_indices=train_episode_indices,
    )
    validation_dataset = None
    if validation_episode_indices:
        validation_dataset = DiffusionEpisodeDataset(
            args.inputs,
            config=config,
            action_source=args.action_source,
            gripper_threshold=args.gripper_threshold,
            episode_indices=validation_episode_indices,
        )
    train_loader = _loader(train_dataset, config=config, shuffle=True)
    validation_loader = (
        None if validation_dataset is None
        else _loader(validation_dataset, config=config, shuffle=False)
    )

    device = torch.device(args.device)
    model = DiffusionPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = max(1, config.epochs * max(1, len(train_loader)))

    def schedule(step: int) -> float:
        if step < config.warmup_steps:
            return max(1e-8, (step + 1) / max(1, config.warmup_steps))
        progress = (step - config.warmup_steps) / max(
            1, total_steps - config.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    metrics_path = output / "training_metrics.csv"
    best_validation = float("inf")
    global_step = 0
    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=("epoch", "step", "train_loss", "validation_loss", "learning_rate"),
        )
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_losses: list[float] = []
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = model.forward_loss(**batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                epoch_losses.append(float(loss.item()))
                global_step += 1
            train_loss = sum(epoch_losses) / len(epoch_losses)
            validation_loss = _evaluate(
                model, validation_loader, device, args.max_validation_batches
            )
            writer.writerow({
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
            metrics_file.flush()
            checkpoint = {
                "format": "panda_cable_diffusion_policy_v1",
                "model": model.state_dict(),
                "config": asdict(config),
                "state_low": train_dataset.state_low.tolist(),
                "state_high": train_dataset.state_high.tolist(),
                "action_low": train_dataset.action_low.tolist(),
                "action_high": train_dataset.action_high.tolist(),
                "action_source": args.action_source,
                "gripper_threshold": args.gripper_threshold,
                "epoch": epoch,
                "global_step": global_step,
            }
            torch.save(checkpoint, output / "checkpoint_latest.pt")
            score = validation_loss if validation_loss is not None else train_loss
            if score < best_validation:
                best_validation = score
                torch.save(checkpoint, output / "checkpoint_best.pt")
            print(
                f"epoch={epoch}/{config.epochs} train_loss={train_loss:.6f} "
                f"validation_loss={validation_loss if validation_loss is not None else 'n/a'}",
                flush=True,
            )

    manifest: dict[str, Any] = {
        "format": "panda_cable_diffusion_policy_v1",
        "method": "diffusion_policy",
        "config": asdict(config),
        "data": {
            "inputs": [str(Path(item).expanduser().resolve()) for item in args.inputs],
            "train_episode_count": len(train_episode_indices),
            "validation_episode_count": len(validation_episode_indices),
            "train_frame_count": len(train_dataset),
            "validation_frame_count": (
                None if validation_dataset is None else len(validation_dataset)
            ),
            "action_source": args.action_source,
            "gripper_threshold": args.gripper_threshold,
            "state_format": "absolute_xyz_quaternion_wxyz",
            "action_format": "absolute_xyz_quaternion_wxyz_gripper_minus1_plus1",
            "camera_keys": [
                "observation.images.opst_cam",
                "observation.images.wrist_cam",
            ],
        },
        "checkpoints": ["checkpoint_latest.pt", "checkpoint_best.pt"],
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"diffusion_policy_output={output}", flush=True)
    return output


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
