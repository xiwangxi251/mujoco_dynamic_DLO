"""Capture actual closed-loop traces from the native version-6 workspace rig."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKTREE = ROOT.parent / "worktrees/scripted-tracking-20261001"
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(WORKTREE / "tools/recording"))
import record_standard_jik_workspace as standard

OUT = ROOT / "outputs/vertical_workspace/scripted_136_retest_20261004/version_6"


def capture(x: float, y: float) -> dict:
    rig = standard.Rig(cartesian_linear_velocity_limit=0.90)
    original_control = rig.control
    target = np.array([x, y, standard.Z])
    trace = []

    def recorded_control(reference, phase):
        original_control(reference, phase)
        measurement = rig.measure(target)
        trace.append(np.r_[
            rig.data.time, rig.env.hand_position, reference,
            measurement[0], measurement[2], measurement[3], rig.data.qpos[:9],
        ])

    rig.control = recorded_control
    row = rig.trial(x, y)
    path = OUT / f"pilot_{x}_{y}.npz"
    np.savez_compressed(path, history=np.asarray(trace))
    return {"goal_xy_m": [x, y], "result": row, "trace_file": path.name}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    records = [capture(*xy) for xy in ((0.7, 0.0), (0.65, 0.3))]
    (OUT / "pilot_traces.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2), flush=True)
