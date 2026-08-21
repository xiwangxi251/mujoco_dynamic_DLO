"""Make the repository's ``src`` package importable for legacy entry points."""

from __future__ import annotations

from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parent / "src"


def bootstrap() -> None:
    source = str(SRC_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)

