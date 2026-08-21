"""Load converted data through DynamicVLA's own dataset implementation."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


REQUIRED_FEATURES = [
    "observation.images.opst_cam",
    "observation.images.wrist_cam",
    "observation.state",
    "action",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamicvla-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.dynamicvla_root.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    if not (root / "utils" / "datasets.py").exists():
        raise FileNotFoundError(root / "utils" / "datasets.py")
    if not (dataset_path / "meta" / "info.json").exists():
        raise FileNotFoundError(dataset_path / "meta" / "info.json")
    if importlib.util.find_spec("torchcodec") is None:
        raise RuntimeError(
            "torchcodec is required by DynamicVLA/utils/datasets.py; install a "
            "build compatible with the installed PyTorch"
        )

    sys.path.insert(0, str(root))
    import utils.datasets

    datasets = {}
    for split in ("train", "test"):
        datasets[split] = utils.datasets.LeRobotDataset(
            str(dataset_path),
            split=split,
            pin_memory=False,
            delta_action=True,
            required_features=REQUIRED_FEATURES,
            delta_timestamps={
                "observation": [-2, 0],
                "action": list(range(20)),
            },
        )
        if len(datasets[split]) == 0:
            raise RuntimeError(f"DynamicVLA {split} split is empty")
        sample = datasets[split][0]
        shapes = {
            key: tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
            for key, value in sample.items()
        }
        print(f"split={split} samples={len(datasets[split])} shapes={shapes}")
        print(f"split={split} task={sample['task']!r}")

    expected = {
        "observation.state": (2, 6),
        "action": (20, 7),
        "observation.images.opst_cam": (2, 3, 360, 480),
        "observation.images.wrist_cam": (2, 3, 360, 480),
    }
    for split, dataset in datasets.items():
        sample = dataset[0]
        for key, shape in expected.items():
            if tuple(sample[key].shape) != shape:
                raise RuntimeError(
                    f"{split} {key} shape {tuple(sample[key].shape)} != {shape}"
                )
    print("dataset_check=passed")


if __name__ == "__main__":
    main()

