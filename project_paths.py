"""Portable project and output path resolution."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT_ENV_VAR = "PANDA_CABLE_OUTPUT_ROOT"


def output_path(*parts: str) -> Path:
    """Return an output path rooted in the repo or configured server storage."""

    configured = os.environ.get(OUTPUT_ROOT_ENV_VAR)
    root = Path(configured).expanduser() if configured else PROJECT_ROOT
    return root.joinpath(*parts)
