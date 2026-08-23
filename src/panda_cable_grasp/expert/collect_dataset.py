"""Collect successful image/action trajectories with the privileged expert."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import cv2
import mujoco
import numpy as np

from ..evaluation.benchmark import base_row, git_text, summarize, write_csv
from ..env.environment import CableGraspEnv
from ..env.kinematics import rotation_to_quat
from ..scenarios.registry import get_scenario, list_scenario_names
from ..evaluation.motion_diagnostics import env_config_for_scenario
from ..paths import output_path

from .formula_intercept_policy import FormulaInterceptExpert
from .run_experiment import DEFAULT_SCENARIOS


SCHEMA_VERSION = 2
COLLECTOR_VERSION = 3
POLICY_NAME = "privileged_shadow_expert"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _save_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON file so interruption cannot truncate it."""
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    """Durably append one worker-owned attempt record."""
    payload = json.dumps(
        value, ensure_ascii=False, default=_json_default, separators=(",", ":")
    )
    with path.open("a", encoding="utf-8") as target:
        target.write(payload + "\n")
        target.flush()
        os.fsync(target.fileno())


@dataclass(frozen=True)
class CollectionJob:
    scenario_name: str
    worker_id: str
    first_attempt: int
    attempt_budget: int
    success_target: int


def _balanced_quotas(total: int, workers: int) -> list[int]:
    """Split an integer exactly and deterministically across workers."""
    if total < 0 or workers <= 0:
        raise ValueError("total must be non-negative and workers positive")
    base, remainder = divmod(total, workers)
    return [base + int(index < remainder) for index in range(workers)]


def _make_jobs(
    scenario_name: str,
    *,
    first_attempt: int,
    attempts: int,
    successes: int,
    envs_per_scenario: int,
) -> list[CollectionJob]:
    if successes > attempts:
        raise ValueError("success quota cannot exceed attempt budget")
    worker_count = min(envs_per_scenario, successes, attempts)
    if worker_count <= 0:
        return []
    success_quotas = _balanced_quotas(successes, worker_count)
    attempt_quotas = success_quotas.copy()
    extras = attempts - sum(attempt_quotas)
    for index, quota in enumerate(_balanced_quotas(extras, worker_count)):
        attempt_quotas[index] += quota

    jobs: list[CollectionJob] = []
    next_attempt = first_attempt
    for index, (attempt_budget, success_target) in enumerate(
        zip(attempt_quotas, success_quotas, strict=True)
    ):
        jobs.append(CollectionJob(
            scenario_name=scenario_name,
            worker_id=f"{first_attempt:06d}_{index:02d}",
            first_attempt=next_attempt,
            attempt_budget=attempt_budget,
            success_target=success_target,
        ))
        next_attempt += attempt_budget
    return jobs


class EpisodeBuffer:
    """Aligned control-rate observations, actions, and teacher labels."""

    def __init__(self, env: CableGraspEnv) -> None:
        self.env = env
        self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.state_size = mujoco.mj_stateSize(env.model, self.state_spec)
        self.times: list[float] = []
        self.states: list[np.ndarray] = []
        self.requested_actions: list[np.ndarray] = []
        self.applied_actions: list[np.ndarray] = []
        self.arm_qpos: list[np.ndarray] = []
        self.arm_qvel: list[np.ndarray] = []
        self.hand_position: list[np.ndarray] = []
        self.hand_quaternion: list[np.ndarray] = []
        self.cable_positions: list[np.ndarray] = []
        self.cable_velocities: list[np.ndarray] = []
        self.target_body_ids: list[int] = []
        self.target_positions: list[np.ndarray] = []
        self.target_velocities: list[np.ndarray] = []
        self.phase_names: list[str] = []
        self.expert_segment_indices: list[int] = []
        self.expert_horizons: list[float] = []
        self.expert_scores: list[float] = []

    def record_before_action(
        self, policy: FormulaInterceptExpert, requested_action: np.ndarray,
    ) -> None:
        state = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(
            self.env.model, self.env.data, state, self.state_spec
        )
        rotation = self.env.data.xmat[self.env.hand_id].reshape(3, 3)
        self.times.append(float(self.env.data.time))
        self.states.append(state)
        self.requested_actions.append(np.asarray(requested_action).copy())
        self.arm_qpos.append(self.env.data.qpos[self.env.arm_qpos_adr].copy())
        self.arm_qvel.append(self.env.data.qvel[self.env.arm_dof_adr].copy())
        self.hand_position.append(self.env.hand_position.copy())
        self.hand_quaternion.append(rotation_to_quat(rotation))
        self.cable_positions.append(
            self.env.data.xpos[self.env.cable_ids].copy()
        )
        self.cable_velocities.append(np.asarray([
            self.env.body_linear_velocity(body_id)
            for body_id in self.env.cable_ids
        ]))
        self.target_body_ids.append(int(self.env.target_body_id))
        self.target_positions.append(self.env.target_position())
        self.target_velocities.append(self.env.target_velocity())
        self.phase_names.append(policy.phase.name)
        self.expert_segment_indices.append(int(policy.expert_segment_index))
        self.expert_horizons.append(float(policy.expert_horizon))
        self.expert_scores.append(float(policy.expert_score))

    def record_applied_action(self) -> None:
        self.applied_actions.append(self.env._last_applied_action.copy())

    def save(
        self,
        path: Path,
        *,
        seed: int,
        scenario_name: str,
        instruction: str,
        opst_video_file: str,
        wrist_video_file: str,
        model_file: str,
        result: str,
    ) -> None:
        if not self.states or len(self.states) != len(self.applied_actions):
            raise RuntimeError("episode buffer is empty or action alignment failed")
        np.savez_compressed(
            path,
            schema_version=np.int64(SCHEMA_VERSION),
            state_spec=np.int64(int(self.state_spec)),
            times=np.asarray(self.times),
            states=np.stack(self.states),
            requested_actions=np.stack(self.requested_actions),
            applied_actions=np.stack(self.applied_actions),
            arm_qpos=np.stack(self.arm_qpos),
            arm_qvel=np.stack(self.arm_qvel),
            hand_position=np.stack(self.hand_position),
            hand_quaternion=np.stack(self.hand_quaternion),
            cable_positions=np.stack(self.cable_positions),
            cable_velocities=np.stack(self.cable_velocities),
            target_body_ids=np.asarray(self.target_body_ids, dtype=np.int64),
            target_positions=np.stack(self.target_positions),
            target_velocities=np.stack(self.target_velocities),
            phase_names=np.asarray(self.phase_names),
            expert_segment_indices=np.asarray(
                self.expert_segment_indices, dtype=np.int64
            ),
            expert_prediction_horizons=np.asarray(self.expert_horizons),
            expert_candidate_scores=np.asarray(self.expert_scores),
            image_frame_indices=np.arange(len(self.states), dtype=np.int64),
            seed=np.int64(seed),
            scenario_name=np.asarray(scenario_name),
            instruction=np.asarray(instruction),
            video_file=np.asarray(opst_video_file),
            opst_video_file=np.asarray(opst_video_file),
            wrist_video_file=np.asarray(wrist_video_file),
            camera_names=np.asarray(["opst_cam", "wrist_cam"]),
            model_file=np.asarray(model_file),
            result=np.asarray(result),
        )


def _collect_attempt(
    env: CableGraspEnv,
    policy: FormulaInterceptExpert,
    *,
    seed: int,
    attempt: int,
    instruction: str,
    scenario_dir: Path,
) -> tuple[dict[str, Any], EpisodeBuffer, dict[str, Path], dict[str, Any]]:
    _, initial_info = env.reset(seed=seed)
    policy.reset()
    control_dt = float(env.model.opt.timestep * env.config.frame_skip)
    video_size = (
        env.config.dynamicvla_camera_width,
        env.config.dynamicvla_camera_height,
    )
    temporary_videos = {
        camera_name: scenario_dir
        / f"_attempt_{attempt:04d}_seed{seed}_{camera_name}.mp4"
        for camera_name in ("opst_cam", "wrist_cam")
    }
    writers: dict[str, cv2.VideoWriter] = {}
    for camera_name, path in temporary_videos.items():
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1.0 / control_dt,
            video_size,
        )
        if not writer.isOpened():
            for opened_writer in writers.values():
                opened_writer.release()
            raise RuntimeError(f"cannot create video: {path}")
        writers[camera_name] = writer

    buffer = EpisodeBuffer(env)
    termination_reason: str | None = None
    min_target_distance = float("inf")
    try:
        while not policy.finished and env.data.time < env.config.episode_seconds:
            action = policy.action()
            frames = env.dynamicvla_camera_rgb()
            for camera_name, frame in frames.items():
                writers[camera_name].write(
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                )
            buffer.record_before_action(policy, action)
            _, _, _, truncated, step_info = env.step(action)
            buffer.record_applied_action()
            min_target_distance = min(
                min_target_distance,
                float(np.linalg.norm(
                    env.hand_position - policy._selected_segment(0.0)
                )),
            )
            if truncated:
                termination_reason = step_info.get("termination_reason")
                policy.result = (
                    "failed_motion_boundary"
                    if termination_reason == "rigid_motion_boundary_crossed"
                    else "failed_timeout"
                )
                policy.finished = True
    finally:
        for writer in writers.values():
            writer.release()

    info = env.info()
    info["ever_pinched"] = env.last_grasped_body_id is not None
    info["base_success"] = env.ever_success
    info["success"] = policy.result == "success"
    scenario = get_scenario(env.config.scenario_name)
    row = base_row(
        POLICY_NAME, attempt, seed, scenario, initial_info, info,
        env.grasp_break_history,
    )
    row.update({
        "attempt": attempt,
        "saved_episode": "",
        "dataset_saved": False,
        "steps": len(buffer.states),
        "sim_time": float(env.data.time),
        "min_target_distance": min_target_distance,
        "policy_result": policy.result,
        "termination_reason": termination_reason or "",
        "strict_success": policy.result == "success",
        **policy.expert_info(),
    })
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "attempt": attempt,
        "scenario": scenario.asdict(),
        "instruction": instruction,
        "initial_info": initial_info,
        "final_info": info,
        "policy_result": policy.result,
        "termination_reason": termination_reason,
        "motion_profile_hash": env.motion_profile_hash,
        "expert": policy.expert_info(),
        "camera_rig": {
            "name": "dynamicvla_opposite_and_wrist",
            "width": env.config.dynamicvla_camera_width,
            "height": env.config.dynamicvla_camera_height,
            "fovy": env.config.dynamicvla_camera_fovy,
            "opst_pos_in_base": env.config.dynamicvla_opst_camera_pos,
            "opst_quat_in_base": env.config.dynamicvla_opst_camera_quat,
            "wrist_pos_in_hand": env.config.dynamicvla_wrist_camera_pos,
            "wrist_quat_in_hand": env.config.dynamicvla_wrist_camera_quat,
        },
    }
    return row, buffer, temporary_videos, metadata


def _save_successful_episode(
    *,
    scenario_dir: Path,
    scenario_name: str,
    seed: int,
    model_path: Path,
    instruction: str,
    row: dict[str, Any],
    buffer: EpisodeBuffer,
    temporary_videos: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    stem = f"episode_seed{seed:010d}"
    opst_video_path = scenario_dir / f"{stem}_opst.mp4"
    wrist_video_path = scenario_dir / f"{stem}_wrist.mp4"
    data_path = scenario_dir / f"{stem}.npz"
    metadata_path = scenario_dir / f"{stem}.json"
    destinations = (opst_video_path, wrist_video_path, data_path, metadata_path)
    if any(path.exists() for path in destinations):
        raise FileExistsError(f"episode artifacts already exist for seed {seed}")

    temporary_data = scenario_dir / f".{stem}.{os.getpid()}.npz"
    try:
        buffer.save(
            temporary_data,
            seed=seed,
            scenario_name=scenario_name,
            instruction=instruction,
            opst_video_file=opst_video_path.name,
            wrist_video_file=wrist_video_path.name,
            model_file=model_path.name,
            result=row["policy_result"],
        )
        temporary_videos["opst_cam"].replace(opst_video_path)
        temporary_videos["wrist_cam"].replace(wrist_video_path)
        temporary_data.replace(data_path)
        row.update({
            "saved_episode": seed,
            "dataset_saved": True,
            "video_path": str(opst_video_path.resolve()),
            "opst_video_path": str(opst_video_path.resolve()),
            "wrist_video_path": str(wrist_video_path.resolve()),
            "trajectory_path": str(data_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
        })
        metadata["artifacts"] = {
            "video": opst_video_path.name,
            "opst_video": opst_video_path.name,
            "wrist_video": wrist_video_path.name,
            "trajectory": data_path.name,
            "model": model_path.name,
        }
        metadata["row"] = row
        # Metadata is the commit marker and is written only after every artifact.
        _save_json(metadata_path, metadata)
    except BaseException:
        temporary_data.unlink(missing_ok=True)
        for path in destinations:
            path.unlink(missing_ok=True)
        raise


def _completed_episode_metadata(scenario_dir: Path) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for metadata_path in sorted(scenario_dir.glob("episode_*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            artifacts = metadata["artifacts"]
            required = (
                artifacts["opst_video"],
                artifacts["wrist_video"],
                artifacts["trajectory"],
            )
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if all((scenario_dir / name).is_file() for name in required):
            completed.append(metadata)
    return completed


def _read_attempt_rows(scenario_dir: Path) -> list[dict[str, Any]]:
    by_attempt: dict[int, dict[str, Any]] = {}
    for path in sorted(scenario_dir.glob("attempts_worker_*.jsonl")):
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    row = json.loads(line)
                    attempt = int(row["attempt"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    # A final partial line is possible after power loss.
                    continue
                by_attempt[attempt] = row
    for metadata in _completed_episode_metadata(scenario_dir):
        row = metadata.get("row")
        if isinstance(row, dict) and "attempt" in row:
            by_attempt.setdefault(int(row["attempt"]), row)
    return [by_attempt[key] for key in sorted(by_attempt)]


def _clean_incomplete_artifacts(run_dir: Path) -> None:
    for path in run_dir.rglob("_attempt_*.mp4"):
        path.unlink(missing_ok=True)
    for path in run_dir.rglob(".*.tmp"):
        path.unlink(missing_ok=True)
    for path in run_dir.rglob(".episode_*.npz"):
        path.unlink(missing_ok=True)
    for path in run_dir.rglob(".scenario.*.mjb"):
        path.unlink(missing_ok=True)

    # Metadata is the commit marker. Remove final-named files that were moved
    # into place before a crash but never received a valid marker.
    for scenario_dir in (path for path in run_dir.iterdir() if path.is_dir()):
        committed_artifacts: set[str] = set()
        valid_metadata: set[str] = set()
        for metadata_path in scenario_dir.glob("episode_*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                artifacts = metadata["artifacts"]
                required = {
                    str(artifacts["opst_video"]),
                    str(artifacts["wrist_video"]),
                    str(artifacts["trajectory"]),
                }
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if all((scenario_dir / name).is_file() for name in required):
                committed_artifacts.update(required)
                valid_metadata.add(metadata_path.name)

        for pattern in (
            "episode_*.npz",
            "episode_*_opst.mp4",
            "episode_*_wrist.mp4",
        ):
            for path in scenario_dir.glob(pattern):
                if path.name not in committed_artifacts:
                    path.unlink(missing_ok=True)
        for metadata_path in scenario_dir.glob("episode_*.json"):
            if metadata_path.name not in valid_metadata:
                metadata_path.unlink(missing_ok=True)


def _ensure_model(env: CableGraspEnv, model_path: Path) -> None:
    if model_path.is_file():
        return
    temporary = model_path.with_name(f".scenario.{os.getpid()}.mjb")
    mujoco.mj_saveModel(env.model, str(temporary), None)
    temporary.replace(model_path)


def _collect_scenario_worker(
    args: argparse.Namespace,
    run_dir: Path,
    job: CollectionJob,
) -> dict[str, Any]:
    scenario_name = job.scenario_name
    scenario_dir = run_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario = get_scenario(scenario_name)
    env_config = env_config_for_scenario(
        scenario,
        seed=args.seed,
        episode_seconds=args.episode_seconds,
    )
    env = CableGraspEnv(replace(
        env_config,
        dynamicvla_cameras_enabled=True,
    ))
    policy = FormulaInterceptExpert(env)
    model_path = scenario_dir / "scenario.mjb"
    _ensure_model(env, model_path)
    attempts_path = scenario_dir / f"attempts_worker_{job.worker_id}.jsonl"
    progress_path = scenario_dir / f"progress_worker_{job.worker_id}.json"
    successes = 0
    attempts = 0
    try:
        while successes < job.success_target and attempts < job.attempt_budget:
            attempt = job.first_attempt + attempts
            attempts += 1
            seed = args.seed + attempt - 1
            row, buffer, temporary_videos, metadata = _collect_attempt(
                env,
                policy,
                seed=seed,
                attempt=attempt,
                instruction=args.instruction,
                scenario_dir=scenario_dir,
            )
            row["worker_id"] = job.worker_id
            if row["strict_success"]:
                successes += 1
                _save_successful_episode(
                    scenario_dir=scenario_dir,
                    scenario_name=scenario_name,
                    seed=seed,
                    instruction=args.instruction,
                    model_path=model_path,
                    row=row,
                    buffer=buffer,
                    temporary_videos=temporary_videos,
                    metadata=metadata,
                )
            else:
                for temporary_video in temporary_videos.values():
                    temporary_video.unlink(missing_ok=True)
                row.update({
                    "video_path": "",
                    "opst_video_path": "",
                    "wrist_video_path": "",
                    "trajectory_path": "",
                    "metadata_path": "",
                })
            _append_jsonl(attempts_path, row)
            _save_json(progress_path, {
                "scenario": scenario_name,
                "worker_id": job.worker_id,
                "attempts": attempts,
                "successes": successes,
                "success_target": job.success_target,
                "updated_at": datetime.now().astimezone().isoformat(),
            })
            print(
                f"scenario={scenario_name} worker={job.worker_id} "
                f"attempt={attempt} seed={seed} "
                f"result={row['policy_result']} saved={row['dataset_saved']} "
                f"worker_successes={successes}/{job.success_target}",
                flush=True,
            )
    finally:
        env.close()
    return {
        "scenario": scenario_name,
        "worker_id": job.worker_id,
        "attempts": attempts,
        "successes": successes,
        "complete": successes == job.success_target,
        "environment_config": asdict(env.config),
        "expert_config": asdict(policy.config),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios", nargs="+", choices=list_scenario_names(),
        default=list(DEFAULT_SCENARIOS),
    )
    parser.add_argument("--successes-per-scenario", type=int, default=3)
    parser.add_argument("--max-attempts-per-scenario", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of scenarios to process concurrently. The process limit is "
            "workers * envs-per-scenario."
        ),
    )
    parser.add_argument(
        "--envs-per-scenario",
        type=int,
        default=1,
        help="Independent MuJoCo environment processes for each active scenario.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Seconds between aggregate throughput and ETA reports.",
    )
    parser.add_argument("--instruction", default="Grasp and lift the cable.")
    parser.add_argument(
        "--output", type=Path,
        default=output_path("datasets", "privileged_expert"),
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a collector-v3 run; requires --run-name.",
    )
    args = parser.parse_args()
    if args.successes_per_scenario <= 0:
        parser.error("successes-per-scenario must be positive")
    if args.max_attempts_per_scenario < args.successes_per_scenario:
        parser.error("max-attempts-per-scenario must cover requested successes")
    if args.episode_seconds <= 0.0:
        parser.error("episode-seconds must be positive")
    if args.workers <= 0:
        parser.error("workers must be positive")
    if args.envs_per_scenario <= 0:
        parser.error("envs-per-scenario must be positive")
    if args.progress_interval <= 0.0:
        parser.error("progress-interval must be positive")
    if args.resume and not args.run_name:
        parser.error("--resume requires --run-name")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("scenarios must not contain duplicates")
    return args


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "collector_version": COLLECTOR_VERSION,
        "scenarios": list(args.scenarios),
        "successes_per_scenario": args.successes_per_scenario,
        "max_attempts_per_scenario": args.max_attempts_per_scenario,
        "seed": args.seed,
        "episode_seconds": args.episode_seconds,
        "instruction": args.instruction,
    }


def _prepare_run(args: argparse.Namespace, run_dir: Path) -> str:
    config_path = run_dir / "run_config.json"
    expected = _run_config(args)
    if args.resume:
        if not run_dir.is_dir() or not config_path.is_file():
            raise FileNotFoundError(
                f"collector-v3 run cannot be resumed: {run_dir}"
            )
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        actual = {key: existing.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                "resume arguments do not match run_config.json:\n"
                f"existing={actual}\nrequested={expected}"
            )
        _clean_incomplete_artifacts(run_dir)
        return str(existing["created_at"])

    run_dir.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now().astimezone().isoformat()
    _save_json(config_path, {**expected, "created_at": created_at})
    return created_at


def _snapshot(
    args: argparse.Namespace, run_dir: Path,
) -> tuple[int, int, dict[str, tuple[int, int]]]:
    attempts_total = 0
    successes_total = 0
    per_scenario: dict[str, tuple[int, int]] = {}
    for scenario_name in args.scenarios:
        scenario_dir = run_dir / scenario_name
        attempts = len(_read_attempt_rows(scenario_dir))
        successes = len(_completed_episode_metadata(scenario_dir))
        attempts_total += attempts
        successes_total += successes
        per_scenario[scenario_name] = (attempts, successes)
    return attempts_total, successes_total, per_scenario


def _make_run_jobs(
    args: argparse.Namespace, run_dir: Path,
) -> list[CollectionJob]:
    jobs: list[CollectionJob] = []
    for scenario_name in args.scenarios:
        scenario_dir = run_dir / scenario_name
        rows = _read_attempt_rows(scenario_dir)
        completed = _completed_episode_metadata(scenario_dir)
        remaining_successes = args.successes_per_scenario - len(completed)
        remaining_attempts = args.max_attempts_per_scenario - len(rows)
        if remaining_successes <= 0 or remaining_attempts <= 0:
            continue
        first_attempt = max(
            (int(row["attempt"]) for row in rows),
            default=0,
        ) + 1
        jobs.extend(_make_jobs(
            scenario_name,
            first_attempt=first_attempt,
            attempts=remaining_attempts,
            successes=min(remaining_successes, remaining_attempts),
            envs_per_scenario=args.envs_per_scenario,
        ))
    return jobs


def _format_seconds(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--:--:--"
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _report_progress(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    started: float,
    baseline_attempts: int,
    baseline_successes: int,
) -> None:
    attempts, successes, per_scenario = _snapshot(args, run_dir)
    elapsed = max(time.monotonic() - started, 1e-9)
    new_attempts = attempts - baseline_attempts
    new_successes = successes - baseline_successes
    attempts_per_hour = new_attempts * 3600.0 / elapsed
    successes_per_hour = new_successes * 3600.0 / elapsed
    target = len(args.scenarios) * args.successes_per_scenario
    remaining = max(0, target - successes)
    eta = (
        remaining / successes_per_hour * 3600.0
        if successes_per_hour > 0.0
        else float("inf")
    )
    details = " ".join(
        f"{name}={values[1]}/{args.successes_per_scenario}"
        for name, values in per_scenario.items()
    )
    print(
        "[TOTAL] "
        f"elapsed={_format_seconds(elapsed)} attempts={attempts} "
        f"successes={successes}/{target} "
        f"attempts/h={attempts_per_hour:.1f} "
        f"successes/h={successes_per_hour:.1f} "
        f"eta={_format_seconds(eta)} {details}",
        flush=True,
    )


def _collect_all_scenarios(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    baseline_attempts: int,
    baseline_successes: int,
) -> list[dict[str, Any]]:
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    wave = 0
    while True:
        jobs = _make_run_jobs(args, run_dir)
        if not jobs:
            break
        wave += 1
        jobs_by_scenario = {
            scenario_name: [
                job for job in jobs if job.scenario_name == scenario_name
            ]
            for scenario_name in args.scenarios
        }
        active_names = [
            name for name in args.scenarios if jobs_by_scenario[name]
        ]
        for offset in range(0, len(active_names), args.workers):
            scenario_batch = active_names[offset : offset + args.workers]
            batch_jobs = [
                job
                for scenario_name in scenario_batch
                for job in jobs_by_scenario[scenario_name]
            ]
            process_count = min(
                len(batch_jobs), len(scenario_batch) * args.envs_per_scenario
            )
            collection_mode = "parallel" if process_count > 1 else "serial"
            print(
                f"collection_mode={collection_mode} "
                f"wave={wave} processes={process_count} "
                f"active_scenarios={','.join(scenario_batch)} "
                f"envs_per_scenario={args.envs_per_scenario}",
                flush=True,
            )
            with ProcessPoolExecutor(
                max_workers=process_count,
                mp_context=context,
            ) as executor:
                pending = {
                    executor.submit(
                        _collect_scenario_worker, args, run_dir, job
                    ): job
                    for job in batch_jobs
                }
                while pending:
                    done, _ = wait(
                        pending,
                        timeout=args.progress_interval,
                        return_when=FIRST_COMPLETED,
                    )
                    _report_progress(
                        args,
                        run_dir,
                        started=started,
                        baseline_attempts=baseline_attempts,
                        baseline_successes=baseline_successes,
                    )
                    for future in done:
                        job = pending.pop(future)
                        result = future.result()
                        results.append(result)
                        print(
                            f"scenario={job.scenario_name} "
                            f"worker={job.worker_id} collection_finished "
                            f"complete={result['complete']}",
                            flush=True,
                        )
    return results


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_csv(temporary, rows)
    temporary.replace(path)


def _finalize_scenarios(
    args: argparse.Namespace,
    run_dir: Path,
    worker_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results_by_scenario = {
        result["scenario"]: result for result in worker_results
    }
    all_rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for scenario_name in args.scenarios:
        scenario_dir = run_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        rows = _read_attempt_rows(scenario_dir)
        completed = _completed_episode_metadata(scenario_dir)
        all_rows.extend(rows)
        episodes_path = scenario_dir / "episodes.csv"
        _write_csv_atomic(episodes_path, rows)
        model_path = scenario_dir / "scenario.mjb"
        manifest_path = scenario_dir / "manifest.json"
        previous = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        worker_result = results_by_scenario.get(scenario_name, {})
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "collector_version": COLLECTOR_VERSION,
            "scenario": get_scenario(scenario_name).asdict(),
            "requested_successes": args.successes_per_scenario,
            "collected_successes": len(completed),
            "attempts": len(rows),
            "max_attempts": args.max_attempts_per_scenario,
            "complete": len(completed) >= args.successes_per_scenario,
            "model": {
                "path": str(model_path.resolve()),
                "sha256": _sha256(model_path) if model_path.is_file() else None,
            },
            "episodes_csv": str(episodes_path.resolve()),
            "environment_config": worker_result.get(
                "environment_config", previous.get("environment_config")
            ),
            "expert_config": worker_result.get(
                "expert_config", previous.get("expert_config")
            ),
        }
        _save_json(manifest_path, manifest)
        manifests[scenario_name] = manifest
    _write_csv_atomic(run_dir / "episodes.csv", all_rows)
    return all_rows, manifests


def main() -> None:
    args = parse_args()
    run_name = args.run_name or (
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    )
    run_dir = args.output / run_name
    created_at = _prepare_run(args, run_dir)
    baseline_attempts, baseline_successes, _ = _snapshot(args, run_dir)
    worker_results: list[dict[str, Any]] = []
    failure: BaseException | None = None
    try:
        worker_results = _collect_all_scenarios(
            args,
            run_dir,
            baseline_attempts=baseline_attempts,
            baseline_successes=baseline_successes,
        )
    except BaseException as error:
        failure = error

    all_rows, scenario_manifests = _finalize_scenarios(
        args, run_dir, worker_results
    )
    episodes_path = run_dir / "episodes.csv"
    git_status = git_text("status", "--porcelain=v1")
    source_dir = Path(__file__).resolve().parent
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "created_at": created_at,
        "updated_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "policy": POLICY_NAME,
        "privileged_teacher": True,
        "student_observation_note": (
            "Privileged cable states and motion parameters are teacher labels; "
            "do not expose them to the deployed student policy."
        ),
        "run_dir": str(run_dir.resolve()),
        "scenarios": args.scenarios,
        "successes_per_scenario": args.successes_per_scenario,
        "max_attempts_per_scenario": args.max_attempts_per_scenario,
        "seed": args.seed,
        "episode_seconds": args.episode_seconds,
        "workers": min(args.workers, len(args.scenarios)),
        "envs_per_scenario": args.envs_per_scenario,
        "parallel_backend": "spawn_process_per_scenario_shard",
        "instruction": args.instruction,
        "camera": "dynamicvla_opposite_and_wrist",
        "action_label": "environment_limited_joint_position_command",
        "scenario_manifests": scenario_manifests,
        "episodes_csv": str(episodes_path.resolve()),
        "summary_all_attempts": summarize(all_rows),
        "source_files": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in (
                source_dir / "collect_dataset.py",
                source_dir / "formula_intercept_policy.py",
                source_dir / "shadow_rollout.py",
                source_dir.parent / "env" / "environment.py",
                source_dir.parent / "policies" / "scripted.py",
            )
        },
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    complete = all(
        item["complete"] for item in scenario_manifests.values()
    )
    manifest["complete"] = complete
    manifest["status"] = (
        "failed" if failure is not None else "complete" if complete else "incomplete"
    )
    _save_json(run_dir / "manifest.json", manifest)
    print(
        f"complete={complete} status={manifest['status']} "
        f"output={run_dir.resolve()}",
        flush=True,
    )
    if failure is not None:
        raise failure
    if not complete:
        raise RuntimeError(
            "collection stopped before every scenario reached its success target; "
            "rerun with --resume and the same --run-name"
        )


if __name__ == "__main__":
    main()
