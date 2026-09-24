from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

from panda_cable_grasp.expert.collect_dataset import (
    COLLECTOR_VERSION,
    _append_jsonl,
    _balanced_quotas,
    _clean_incomplete_artifacts,
    _completed_episode_metadata,
    _make_jobs,
    _make_run_jobs,
    _prepare_run,
    _read_attempt_rows,
    _save_json,
)


def _args(**overrides):
    values = {
        "scenarios": ["id_static"],
        "successes_per_scenario": 5,
        "max_attempts_per_scenario": 20,
        "seed": 100,
        "episode_seconds": 15.0,
        "instruction": "Grasp and lift the cable.",
        "workers": 1,
        "envs_per_scenario": 2,
        "progress_interval": 10.0,
        "run_name": "test_run",
        "resume": False,
    }
    values.update(overrides)
    return Namespace(**values)


class CollectionSchedulerTests(unittest.TestCase):
    def test_balanced_quotas_are_exact(self) -> None:
        self.assertEqual(_balanced_quotas(10, 3), [4, 3, 3])
        self.assertEqual(_balanced_quotas(2, 4), [1, 1, 0, 0])

    def test_jobs_split_successes_attempts_and_seeds_without_overlap(self) -> None:
        jobs = _make_jobs(
            "id_static",
            first_attempt=11,
            attempts=23,
            successes=7,
            envs_per_scenario=3,
        )
        self.assertEqual(sum(job.success_target for job in jobs), 7)
        self.assertEqual(sum(job.attempt_budget for job in jobs), 23)
        allocated = [
            attempt
            for job in jobs
            for attempt in range(
                job.first_attempt, job.first_attempt + job.attempt_budget
            )
        ]
        self.assertEqual(allocated, list(range(11, 34)))
        self.assertEqual(len({job.worker_id for job in jobs}), 3)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            _make_jobs(
                "id_static",
                first_attempt=1,
                attempts=2,
                successes=3,
                envs_per_scenario=2,
            )

    def test_resume_requires_matching_collection_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            created_at = _prepare_run(_args(), run_dir)
            config = json.loads(
                (run_dir / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["collector_version"], COLLECTOR_VERSION)

            resumed = _prepare_run(
                _args(resume=True, workers=4, envs_per_scenario=3),
                run_dir,
            )
            self.assertEqual(resumed, created_at)
            with self.assertRaisesRegex(ValueError, "do not match"):
                _prepare_run(_args(resume=True, seed=101), run_dir)

    def test_remaining_attempts_smaller_than_success_deficit_are_still_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            scenario_dir = run_dir / "id_static"
            scenario_dir.mkdir()
            journal = scenario_dir / "attempts_worker_previous.jsonl"
            for attempt in range(1, 19):
                _append_jsonl(journal, {
                    "attempt": attempt,
                    "dataset_saved": False,
                })

            jobs = _make_run_jobs(_args(), run_dir)

            self.assertEqual(sum(job.attempt_budget for job in jobs), 2)
            self.assertEqual(sum(job.success_target for job in jobs), 2)
            self.assertEqual(jobs[0].first_attempt, 19)

    def test_metadata_is_commit_marker_and_recovers_missing_jsonl_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scenario_dir = Path(temporary)
            stem = "episode_seed0000000100"
            for suffix in (".npz", "_opst.mp4", "_wrist.mp4"):
                (scenario_dir / f"{stem}{suffix}").touch()
            row = {"attempt": 1, "dataset_saved": True}
            _save_json(scenario_dir / f"{stem}.json", {
                "artifacts": {
                    "trajectory": f"{stem}.npz",
                    "opst_video": f"{stem}_opst.mp4",
                    "wrist_video": f"{stem}_wrist.mp4",
                },
                "row": row,
            })
            self.assertEqual(len(_completed_episode_metadata(scenario_dir)), 1)
            self.assertEqual(_read_attempt_rows(scenario_dir), [row])

            _append_jsonl(
                scenario_dir / "attempts_worker_000001_00.jsonl",
                {"attempt": 2, "dataset_saved": False},
            )
            self.assertEqual(
                [item["attempt"] for item in _read_attempt_rows(scenario_dir)],
                [1, 2],
            )

    def test_resume_cleanup_preserves_commit_and_removes_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            scenario_dir = run_dir / "id_static"
            scenario_dir.mkdir()
            committed = "episode_seed0000000001"
            orphan = "episode_seed0000000002"
            for stem in (committed, orphan):
                (scenario_dir / f"{stem}.npz").write_bytes(b"npz")
                (scenario_dir / f"{stem}_opst.mp4").write_bytes(b"opst")
                (scenario_dir / f"{stem}_wrist.mp4").write_bytes(b"wrist")
            _save_json(scenario_dir / f"{committed}.json", {
                "artifacts": {
                    "trajectory": f"{committed}.npz",
                    "opst_video": f"{committed}_opst.mp4",
                    "wrist_video": f"{committed}_wrist.mp4",
                }
            })
            (scenario_dir / f"{orphan}.json").write_text(
                "{broken", encoding="utf-8"
            )

            _clean_incomplete_artifacts(run_dir)

            self.assertTrue((scenario_dir / f"{committed}.json").is_file())
            self.assertTrue((scenario_dir / f"{committed}.npz").is_file())
            self.assertFalse((scenario_dir / f"{orphan}.json").exists())
            self.assertFalse((scenario_dir / f"{orphan}.npz").exists())
            self.assertFalse((scenario_dir / f"{orphan}_opst.mp4").exists())
            self.assertFalse((scenario_dir / f"{orphan}_wrist.mp4").exists())


if __name__ == "__main__":
    unittest.main()
