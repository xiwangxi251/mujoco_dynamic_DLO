"""Validate a DynamicVLA checkout/checkpoint without loading model weights."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REQUIRED_FEATURES = {
    "observation.state": (6,),
    "observation.images.opst_cam": (3, 360, 480),
    "observation.images.wrist_cam": (3, 360, 480),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamicvla-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete-weights", action="store_true",
        help="accept config-only Hugging Face downloads still in progress",
    )
    args = parser.parse_args()

    root = args.dynamicvla_root.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    required_source = [
        root / "scripts" / "inference.py",
        root / "policies" / "dynamicvla" / "modeling_dynamicvla.py",
        root / "utils" / "helpers.py",
        root / "requirements.txt",
    ]
    missing_source = [str(path) for path in required_source if not path.is_file()]
    if missing_source:
        raise FileNotFoundError(f"Incomplete DynamicVLA checkout: {missing_source}")

    config_path = weights / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("type") != "dynamicvla":
        raise ValueError(f"Unexpected checkpoint type: {config.get('type')!r}")
    if config.get("n_obs_steps") != 2:
        raise ValueError("The adapter currently expects the DOM checkpoint's 2 observations")
    for name, shape in REQUIRED_FEATURES.items():
        actual = tuple(config.get("input_features", {}).get(name, {}).get("shape", ()))
        if actual != shape:
            raise ValueError(f"Feature {name} has shape {actual}, expected {shape}")
    action_shape = tuple(config.get("output_features", {}).get("action", {}).get("shape", ()))
    if action_shape != (7,):
        raise ValueError(f"Expected Euler task-space action shape (7,), got {action_shape}")
    if not config.get("use_delta_action"):
        raise ValueError("Expected the official DOM delta-action checkpoint")

    weight_files = list(weights.glob("*.safetensors")) + list(weights.glob("*.bin"))
    if not weight_files and not args.allow_incomplete_weights:
        raise FileNotFoundError(
            "Model weights are not complete yet (no .safetensors/.bin file found)"
        )

    packages = [
        "torch", "torchvision", "transformers", "lerobot", "zmq", "scipy",
        "safetensors", "easydict",
    ]
    missing_packages = [name for name in packages if importlib.util.find_spec(name) is None]
    print(f"python={sys.version.split()[0]}")
    print(f"dynamicvla_root={root}")
    print(f"weights={weights}")
    print(f"weight_files={len(weight_files)}")
    print(f"missing_packages={','.join(missing_packages) if missing_packages else 'none'}")
    if missing_packages:
        return 2
    print("dynamicvla_install_check=OK")
    print("model_loaded=false")
    print("model_experiment_run=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

