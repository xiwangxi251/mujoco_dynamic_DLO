"""Check completeness, qualification logic, and repeatability of the rerun."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vertical_workspace/scripted_136_retest_20261004"


def load_rows(directory):
    rows = json.loads((directory / "cells.json").read_text(encoding="utf-8"))
    indexed = {(row["x_m"], row["y_m"]): row for row in rows}
    assert len(rows) == len(indexed) == 961
    return indexed


def main():
    report = {}
    reference_grid = None
    for version in (1, 3, 6):
        directory = OUT / f"version_{version}"
        rows = load_rows(directory)
        assert reference_grid is None or rows.keys() == reference_grid
        reference_grid = rows.keys()
        with np.load(directory / "workspace.npz") as workspace:
            endpoint_mask = workspace["endpoint_pass"]
            motion_mask = workspace["motion_pass"]
            interior_mask = workspace["interior"]
            assert endpoint_mask.shape == motion_mask.shape == interior_mask.shape == (31, 31)
            assert int(endpoint_mask.sum()) == sum(r["endpoint_pass"] for r in rows.values())
            assert int(motion_mask.sum()) == sum(r["motion_pass"] for r in rows.values())
            assert np.all(motion_mask <= endpoint_mask)
            assert np.all(interior_mask <= motion_mask)
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            for name, mask in (("endpoint_pass", endpoint_mask),
                               ("motion_pass", motion_mask), ("interior", interior_mask)):
                if summary[name] is None:
                    assert not mask.any()
                else:
                    assert summary[name]["cells"] == int(mask.sum())
        for row in rows.values():
            assert np.isfinite([row["position_error_mm"], row["pose_error_deg"],
                                row["finger_floor_mm"], row["peak_velocity_to_limit"]]).all()
            assert not row["motion_pass"] or (row["endpoint_pass"] and
                       row["max_path_pose_deg"] <= 5.0 and
                       row["peak_velocity_to_limit"] <= 1.01)
        if version in (1, 3):
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["version_number"] == version
            for relative, digest in manifest["source_sha256"].items():
                assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
        report[str(version)] = {
            "cells": len(rows), "endpoint": sum(r["endpoint_pass"] for r in rows.values()),
            "motion": sum(r["motion_pass"] for r in rows.values()),
            "interior": int(interior_mask.sum()),
            "reasons": {name: sum(r["reason"] == name for r in rows.values())
                        for name in sorted({r["reason"] for r in rows.values()})},
        }
    prior = {
        3: ROOT / "outputs/vertical_workspace/nero_improved_jacobian_20261001",
        6: ROOT / "outputs/vertical_workspace/nero_standard_jik_uniform090_20261003",
    }
    for version, directory in prior.items():
        earlier = load_rows(directory)
        current = load_rows(OUT / f"version_{version}")
        matched = sum((current[key]["endpoint_pass"], current[key]["motion_pass"],
                       current[key]["reason"]) ==
                      (earlier[key]["endpoint_pass"], earlier[key]["motion_pass"],
                       earlier[key]["reason"]) for key in current)
        report[str(version)]["previous_run_outcome_matches"] = matched
    report["validation"] = "961 unique shared goals; finite diagnostics; masks and summaries agree; source hashes match for versions 1 and 3"
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
