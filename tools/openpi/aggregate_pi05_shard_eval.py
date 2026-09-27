#!/usr/bin/env python
"""Aggregate per-scenario pi05 shard eval outputs into a run-root summary.

Reads each shard's episodes.csv + manifest.json under
``<run_root>/shards/<scenario>/eval/`` (layout produced by
``eval_pi05_5scenes_s29999_paired50.sh``), writes ``episodes_all.csv`` and
``aggregate_summary.json`` into the run root, and prints a per-scenario table.

Exits non-zero when the run is incomplete (missing shards or episodes), so it
is safe to run for progress checks mid-eval as well as at the end.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(value: object) -> float | None:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _wilson95(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959964
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return centre - margin, centre + margin


def _summarize_rows(rows: list[dict]) -> dict:
    total = len(rows)
    successes = sum(_truthy(r.get("task_success")) for r in rows)
    infer = [
        v
        for r in rows
        if (v := _float(r.get("policy_inference_mean_ms"))) is not None
    ]
    results: dict[str, int] = {}
    for r in rows:
        key = str(r.get("policy_result") or "unknown")
        results[key] = results.get(key, 0) + 1
    low, high = _wilson95(successes, total)
    return {
        "episodes": total,
        "task_successes": successes,
        "task_success_rate": successes / total if total else None,
        "task_success_wilson95": [low, high],
        "policy_result_counts": results,
        "policy_inference_mean_ms": (sum(infer) / len(infer)) if infer else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--shard-glob", default="shards/*/eval")
    args = parser.parse_args()

    run_root = args.run_root
    launch_manifest_path = run_root / "launch_manifest.json"
    launch_manifest = (
        json.loads(launch_manifest_path.read_text(encoding="utf-8"))
        if launch_manifest_path.exists()
        else {}
    )
    expected_scenarios = launch_manifest.get("scenarios")
    expected_seeds = launch_manifest.get("seeds")

    shard_dirs = sorted(
        p for p in run_root.glob(args.shard_glob) if p.is_dir()
    )
    if not shard_dirs:
        print(f"no shard dirs under {run_root}/{args.shard_glob}", file=sys.stderr)
        return 2

    all_rows: list[dict] = []
    fieldnames: list[str] = []
    shard_reports = []
    complete = True
    checkpoints = set()

    for shard in shard_dirs:
        scenario = shard.parent.name
        report = {"scenario": scenario, "shard_dir": str(shard)}
        manifest_path = shard / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report["checkpoint"] = manifest.get("policy", {}).get("checkpoint")
            report["config"] = manifest.get("policy", {}).get("config")
            report["git_commit"] = manifest.get("git_commit")
            checkpoints.add(str(report["checkpoint"]))
            report["policy_contract_exact"] = all(
                manifest.get("policy", {}).get(key) == launch_manifest.get("policy", {}).get(key)
                for key in ("checkpoint", "config", "instruction", "execute_steps")
            )
            complete &= report["policy_contract_exact"]
        else:
            complete = False
        csv_path = shard / "episodes.csv"
        if not csv_path.exists():
            # Count finished episodes from per-episode metadata instead.
            done = sorted(shard.glob("episodes/pi05/*/seed_*/episode.json"))
            report["status"] = "running"
            report["episodes_finished"] = len(done)
            complete = False
            shard_reports.append(report)
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            fieldnames = list(
                dict.fromkeys([*fieldnames, *(rows[0].keys() if rows else [])])
            )
        seeds = [int(r["actual_episode_seed"]) for r in rows]
        report["status"] = "csv_written"
        report["episodes_finished"] = len(rows)
        report["unique_seeds"] = len(set(seeds))
        complete &= len(seeds) == len(set(seeds))
        complete &= all(r.get("scenario_name") == scenario for r in rows)
        report["seed_min"] = min(seeds) if seeds else None
        report["seed_max"] = max(seeds) if seeds else None
        if expected_seeds is not None:
            report["seed_window_exact"] = sorted(set(seeds)) == sorted(
                set(expected_seeds)
            )
            complete &= bool(report["seed_window_exact"])
            complete &= len(rows) == len(expected_seeds)
        report["summary"] = _summarize_rows(rows)
        shard_reports.append(report)
        for r in rows:
            r["_shard_scenario"] = scenario
        all_rows.extend(rows)

    by_scenario = {
        r["scenario"]: r.get("summary")
        for r in shard_reports
        if r.get("summary")
    }
    if expected_scenarios is not None:
        complete &= {r["scenario"] for r in shard_reports} == set(expected_scenarios)
        complete &= len(shard_reports) == len(expected_scenarios)
    complete &= checkpoints == {launch_manifest.get("policy", {}).get("checkpoint")}
    if expected_scenarios is not None and expected_seeds is not None:
        complete &= len(all_rows) == len(expected_scenarios) * len(expected_seeds)
    overall = _summarize_rows(all_rows)
    episodes_path = run_root / "episodes_all.csv"
    if all_rows:
        fieldnames = list(dict.fromkeys([*fieldnames, "_shard_scenario"]))
        with episodes_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    summary = {
        "run_root": str(run_root),
        "complete": complete and bool(all_rows),
        "expected_scenarios": expected_scenarios,
        "total_episodes": len(all_rows),
        "overall_micro": overall,
        "by_scenario": by_scenario,
        "checkpoints": sorted(checkpoints),
        "shards": shard_reports,
    }
    summary_path = run_root / "aggregate_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"run_root={run_root}")
    for r in shard_reports:
        s = r.get("summary") or {}
        rate = s.get("task_success_rate")
        print(
            f"  {r['scenario']}: {r['status']} "
            f"episodes={r.get('episodes_finished')} "
            f"successes={s.get('task_successes', '-')} "
            f"rate={f'{rate:.1%}' if rate is not None else '-'}"
        )
    rate = overall["task_success_rate"]
    print(
        f"overall: episodes={overall['episodes']} "
        f"successes={overall['task_successes']} "
        f"rate={f'{rate:.1%}' if rate is not None else '-'}"
    )
    print(f"episodes_all={episodes_path if all_rows else '(none yet)'}")
    print(f"summary={summary_path}")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
