"""Create a resolved DynamicVLA config and launch official fine-tuning."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from ...paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamicvla-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--experiment", default="panda-cable-finetune")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpus", default="0", help="CUDA device list, e.g. 0 or 0,1,2,3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "Fine-tuning setup requires PyYAML; install the dynamicvla extra "
            "with `pip install -e '.[dynamicvla]'`."
        ) from error
    root = args.dynamicvla_root.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint
        else root / "ckt" / "dynamic-vla-DOM"
    )
    template = PROJECT_ROOT / "configs" / "dynamicvla" / "cable_finetune.yaml"
    for required in (root / "run.py", dataset / "meta" / "info.json", checkpoint / "config.json", template):
        if not required.exists():
            raise FileNotFoundError(required)
    if min(args.epochs, args.batch_size, args.workers, args.checkpoint_every) <= 0:
        raise ValueError("epochs, batch-size, workers and checkpoint-every must be positive")

    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["CONST"]["EXP_NAME"] = args.experiment
    config["CONST"]["N_WORKERS"] = args.workers
    # DynamicVLA's custom loader accepts an absolute path as the repository ID;
    # pathlib then keeps it instead of prefixing HF_LEROBOT_HOME.
    config["DATASET"]["NAME"] = dataset.as_posix()
    config["TRAIN"]["N_EPOCHS"] = args.epochs
    config["TRAIN"]["BATCH_SIZE"] = args.batch_size
    config["TRAIN"]["OPTIMIZER"]["LR"] = args.learning_rate
    config["TRAIN"]["CKPT_SAVE_FREQ"]["EPOCH"] = args.checkpoint_every
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "runs" / "panda_cable_finetune"
    )
    config["DIR"]["OUTPUT"] = output_dir.as_posix()

    resolved_dir = output_dir / "resolved_configs"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = resolved_dir / f"{args.experiment}.yaml"
    resolved_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one CUDA device")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(gpu_ids)),
        "run.py",
        "-c",
        str(resolved_config),
        "-p",
        str(checkpoint),
        "-e",
        args.experiment,
    ]
    print("command=" + subprocess.list2cmdline(command), flush=True)
    print(f"resolved_config={resolved_config}", flush=True)
    if args.dry_run:
        return
    if importlib.util.find_spec("torchcodec") is None:
        raise RuntimeError(
            "DynamicVLA's training dataset loader imports torchcodec directly; "
            "install a torch-compatible build with `python -m pip install torchcodec`"
        )
    if os.name == "nt":
        raise RuntimeError(
            "Official DynamicVLA training uses NCCL, os.sched_setaffinity and "
            "libcudart.so; launch training on the Linux server. Use --dry-run on Windows."
        )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
