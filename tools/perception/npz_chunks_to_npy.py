"""Convert packed <scenario>__partNN.npz chunks into raw .npy files.

The .npy layout lets PackedDLODataset mmap every array directly.
Reads uncompressed npz (np.savez output), so this is a fast copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--delete-npz", action="store_true")
    args = p.parse_args()
    for npz in sorted(args.packed_dir.glob("*__part*.npz")):
        base = npz.with_suffix("")  # strip .npz
        done = base.parent / f"{base.name}.seeds.npy"
        if done.exists():
            print(f"skip {npz.name}")
            continue
        with np.load(npz) as z:
            for k in z.files:
                np.save(f"{base}.{k}.npy", z[k])
        print(f"converted {npz.name}")
        if args.delete_npz:
            npz.unlink()


if __name__ == "__main__":
    main()
