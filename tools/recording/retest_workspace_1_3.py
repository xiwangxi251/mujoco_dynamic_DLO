"""Rerun the common physical workspace scan for scripted versions 1 and 3.

The worker process imports this module under multiprocessing spawn, so the
version selection is deliberately read at module import time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import record_improved_workspace as benchmark


VERSION = os.environ.get("WORKSPACE_RETEST_VERSION")
if VERSION not in {"1", "3"}:
    raise RuntimeError("Set WORKSPACE_RETEST_VERSION to 1 or 3")

benchmark.OUT = (
    Path(__file__).resolve().parents[2]
    / "outputs/vertical_workspace/scripted_136_retest_20261004"
    / f"version_{VERSION}"
)
benchmark.VARIANT = "baseline" if VERSION == "1" else "tracking_fix_orientation4"


_OriginalRig = benchmark.ImprovedRig


class VersionRig(_OriginalRig):
    def __init__(self):
        super().__init__()
        baseline = VERSION == "1"
        self.audit.update(
            version_number=int(VERSION),
            gravity_feedforward=("disabled" if baseline else "bounded at 500 Hz"),
            arm_kp_scale=(1.0 if baseline else 4.0),
            arm_kv_scale=(1.0 if baseline else 2.0),
            orientation_error_scale=(1.0 if baseline else 4.0),
        )


benchmark.ImprovedRig = VersionRig


if __name__ == "__main__":
    benchmark.main()
    manifest = benchmark.OUT / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["version_number"] = int(VERSION)
        data["runner"] = str(Path(__file__).resolve())
        manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
