"""Collect successful image/action trajectories with the privileged expert."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys
from typing import Any

import cv2
import mujoco
import numpy as np

from benchmark import _base_row, _git_text, _summary, _write_csv
from cable_grasp_env import CableGraspEnv, rotation_to_quat
from experiment_scenarios import get_scenario, list_scenario_names
from motion_diagnostics import env_config_for_scenario
from project_paths import output_path

from .formula_intercept_policy import FormulaInterceptExpert
from .run_experiment import DEFAULT_SCENARIOS


SCHEMA_VERSION = 1
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
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


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
        video_file: str,
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
            seed=np.int64(seed),
            scenario_name=np.asarray(scenario_name),
            instruction=np.asarray(instruction),
            video_file=np.asarray(video_file),
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
) -> tuple[dict[str, Any], EpisodeBuffer, Path, dict[str, Any]]:
    _, initial_info = env.reset(seed=seed)
    policy.reset()
    control_dt = float(env.model.opt.timestep * env.config.frame_skip)
    video_size = (
        env.config.global_camera_width,
        env.config.global_camera_height,
    )
    temporary_video = scenario_dir / f"_attempt_{attempt:04d}_seed{seed}.mp4"
    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0 / control_dt,
        video_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {temporary_video}")

    buffer = EpisodeBuffer(env)
    termination_reason: str | None = None
    min_target_distance = float("inf")
    try:
        while not policy.finished and env.data.time < env.config.episode_seconds:
            action = policy.action()
            frame = env.camera_rgb()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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
        writer.release()

    info = env.info()
    info["ever_pinched"] = env.last_grasped_body_id is not None
    info["base_success"] = env.ever_success
    info["success"] = policy.result == "success"
    scenario = get_scenario(env.config.scenario_name)
    row = _base_row(
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
    }
    return row, buffer, temporary_video, metadata


def _collect_scenario(
    args: argparse.Namespace,
    run_dir: Path,
    scenario_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenario_dir = run_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=False)
    scenario = get_scenario(scenario_name)
    env = CableGraspEnv(env_config_for_scenario(
        scenario,
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        camera_observation_enabled=True,
    ))
    policy = FormulaInterceptExpert(env)
    model_path = scenario_dir / "scenario.mjb"
    mujoco.mj_saveModel(env.model, str(model_path), None)
    rows: list[dict[str, Any]] = []
    successes = 0
    attempt = 0
    try:
        while (
            successes < args.successes_per_scenario
            and attempt < args.max_attempts_per_scenario
        ):
            attempt += 1
            seed = args.seed + attempt - 1
            row, buffer, temporary_video, metadata = _collect_attempt(
                env,
                policy,
                seed=seed,
                attempt=attempt,
                instruction=args.instruction,
                scenario_dir=scenario_dir,
            )
            if row["strict_success"]:
                successes += 1
                stem = f"episode_{successes:06d}"
                video_path = scenario_dir / f"{stem}_global.mp4"
                data_path = scenario_dir / f"{stem}.npz"
                metadata_path = scenario_dir / f"{stem}.json"
                temporary_video.replace(video_path)
                buffer.save(
                    data_path,
                    seed=seed,
                    scenario_name=scenario_name,
                    instruction=args.instruction,
                    video_file=video_path.name,
                    model_file=model_path.name,
                    result=row["policy_result"],
                )
                metadata["artifacts"] = {
                    "video": video_path.name,
                    "trajectory": data_path.name,
                    "model": model_path.name,
                }
                _save_json(metadata_path, metadata)
                row.update({
                    "saved_episode": successes,
                    "dataset_saved": True,
                    "video_path": str(video_path.resolve()),
                    "trajectory_path": str(data_path.resolve()),
                    "metadata_path": str(metadata_path.resolve()),
                })
            else:
                temporary_video.unlink(missing_ok=True)
                row.update({
                    "video_path": "",
                    "trajectory_path": "",
                    "metadata_path": "",
                })
            rows.append(row)
            print(
                f"scenario={scenario_name} attempt={attempt} seed={seed} "
                f"result={row['policy_result']} saved={row['dataset_saved']} "
                f"successes={successes}/{args.successes_per_scenario}",
                flush=True,
            )
    finally:
        env.close()

    episodes_path = scenario_dir / "episodes.csv"
    _write_csv(episodes_path, rows)
    scenario_manifest = {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario.asdict(),
        "requested_successes": args.successes_per_scenario,
        "collected_successes": successes,
        "attempts": attempt,
        "complete": successes == args.successes_per_scenario,
        "model": {
            "path": str(model_path.resolve()),
            "sha256": _sha256(model_path),
        },
        "episodes_csv": str(episodes_path.resolve()),
        "environment_config": asdict(env.config),
        "expert_config": asdict(policy.config),
    }
    _save_json(scenario_dir / "manifest.json", scenario_manifest)
    return rows, scenario_manifest


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
            "Number of independent scenario worker processes. Each worker "
            "owns its MuJoCo environment and renderer."
        ),
    )
    parser.add_argument("--instruction", default="Grasp and lift the cable.")
    parser.add_argument(
        "--output", type=Path,
        default=output_path("benchmark_runs", "privileged_expert_dataset"),
    )
    parser.add_argument("--run-name")
    args = parser.parse_args()
    if args.successes_per_scenario <= 0:
        parser.error("successes-per-scenario must be positive")
    if args.max_attempts_per_scenario < args.successes_per_scenario:
        parser.error("max-attempts-per-scenario must cover requested successes")
    if args.episode_seconds <= 0.0:
        parser.error("episode-seconds must be positive")
    if args.workers <= 0:
        parser.error("workers must be positive")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("scenarios must not contain duplicates")
    return args


def _collect_all_scenarios(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect scenarios serially or in isolated spawned worker processes."""
    worker_count = min(args.workers, len(args.scenarios))
    results: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    if worker_count == 1:
        for scenario_name in args.scenarios:
            results[scenario_name] = _collect_scenario(
                args, run_dir, scenario_name
            )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    _collect_scenario, args, run_dir, scenario_name
                ): scenario_name
                for scenario_name in args.scenarios
            }
            for future in as_completed(futures):
                scenario_name = futures[future]
                results[scenario_name] = future.result()
                print(f"scenario={scenario_name} collection_finished", flush=True)

    all_rows: list[dict[str, Any]] = []
    scenario_manifests: dict[str, Any] = {}
    for scenario_name in args.scenarios:
        rows, scenario_manifest = results[scenario_name]
        all_rows.extend(rows)
        scenario_manifests[scenario_name] = scenario_manifest
    return all_rows, scenario_manifests


def main() -> None:
    args = parse_args()
    run_name = args.run_name or (
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    )
    run_dir = args.output / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    worker_count = min(args.workers, len(args.scenarios))
    print(
        f"collection_mode={'parallel' if worker_count > 1 else 'serial'} "
        f"workers={worker_count} scenarios={len(args.scenarios)}",
        flush=True,
    )
    all_rows, scenario_manifests = _collect_all_scenarios(args, run_dir)

    episodes_path = run_dir / "episodes.csv"
    _write_csv(episodes_path, all_rows)
    git_status = _git_text("status", "--porcelain=v1")
    source_dir = Path(__file__).resolve().parent
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
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
        "workers": worker_count,
        "parallel_backend": (
            "spawn_process_per_scenario" if worker_count > 1 else "serial"
        ),
        "instruction": args.instruction,
        "camera": "fixed_global",
        "action_label": "environment_limited_joint_position_command",
        "scenario_manifests": scenario_manifests,
        "episodes_csv": str(episodes_path.resolve()),
        "summary_all_attempts": _summary(all_rows),
        "source_files": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in (
                source_dir / "collect_dataset.py",
                source_dir / "formula_intercept_policy.py",
                source_dir / "shadow_rollout.py",
                source_dir.parent / "cable_grasp_env.py",
                source_dir.parent / "dynamic_grasp_policy.py",
            )
        },
        "git_commit": _git_text("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    _save_json(run_dir / "manifest.json", manifest)
    complete = all(
        item["complete"] for item in scenario_manifests.values()
    )
    print(f"complete={complete} output={run_dir.resolve()}", flush=True)
    if not complete:
        raise RuntimeError(
            "collection stopped before every scenario reached its success target"
        )


if __name__ == "__main__":
    main()
