"""Staged NERO geometry calibration sweep.

This diagnostic runner changes only the NERO robot geometry convention used by
the environment.  It keeps the scenario, seeds, policy settings, and task
success definition fixed, and writes a checkpoint after every episode.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import time
from typing import Any

import mujoco
import numpy as np

from panda_cable_grasp.env.environment import (
    CableGraspEnv,
    DEFAULT_ARM_JOINT_VELOCITY_LIMITS,
    ROBOT_SPECS,
)
from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
from panda_cable_grasp.evaluation.recording import EpisodeRecorder
from panda_cable_grasp.expert.formula_intercept_policy import (
    FormulaInterceptConfig,
    FormulaInterceptExpert,
)
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig
from panda_cable_grasp.scenarios.registry import get_scenario


SCENARIOS = ("id_static", "id_rigid_l1_nominal")
ALL_SCENARIOS = (
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
)
VARIANTS = ("scripted", "expert")
BASE_NOMINAL = 0.20
TCP_X_NOMINAL = 0.1733
TCP_Y_NOMINAL = 0.0
TCP_Z_NOMINAL = -0.0235


def outcome(env: CableGraspEnv) -> str:
    if env.ever_success:
        return "success"
    if not env.ever_bilateral_candidate:
        return "never_bilateral_candidate"
    if not env.ever_confirmed_grasp:
        return "bilateral_not_confirmed"
    if any(
        event.get("causal_class") == "physical_slip"
        and event.get("bilateral_confirmed", False)
        for event in env.grasp_break_history
    ):
        return "physical_slip_after_confirmed_grasp"
    if any(
        event.get("causal_class") == "active_open"
        and event.get("bilateral_confirmed", False)
        for event in env.grasp_break_history
    ):
        return "active_open_after_confirmed_grasp"
    return "confirmed_grasp_but_no_task_success"


def apply_calibration(
    base_x: float,
    tcp_dx: float,
    tcp_dy: float,
    tcp_dz: float,
    ready_qpos: list[float] | None = None,
) -> None:
    """Apply one NERO geometry hypothesis in the current worker process."""

    nominal = ROBOT_SPECS["nero"]
    updates: dict[str, Any] = {
        "base_offset": (float(base_x), 0.0, 0.0),
        "grasp_center_local": (
            TCP_X_NOMINAL + float(tcp_dx),
            TCP_Y_NOMINAL + float(tcp_dy),
            TCP_Z_NOMINAL + float(tcp_dz),
        ),
    }
    if ready_qpos is not None:
        if len(ready_qpos) != 7:
            raise ValueError("NERO ready qpos must contain 7 joint values")
        updates["ready_arm_qpos"] = tuple(float(value) for value in ready_qpos)
    ROBOT_SPECS["nero"] = replace(nominal, **updates)


def apply_nero_servo_override(
    env: CableGraspEnv,
    wrist_kp: float | None,
    wrist_kv: float | None,
    arm_kp_scale: float | None = None,
    arm_kv_scale: float | None = None,
) -> None:
    """Optionally override NERO arm servo gains for one experiment."""

    if (
        wrist_kp is None
        and wrist_kv is None
        and arm_kp_scale is None
        and arm_kv_scale is None
    ):
        return
    if (wrist_kp is None) != (wrist_kv is None):
        raise ValueError("nero wrist kp and kv must be supplied together")
    if wrist_kp is not None and (wrist_kp <= 0.0 or wrist_kv is None or wrist_kv < 0.0):
        raise ValueError("nero wrist kp must be positive and kv non-negative")
    if (arm_kp_scale is None) != (arm_kv_scale is None):
        raise ValueError("nero arm kp and kv scales must be supplied together")
    if arm_kp_scale is not None and (
        arm_kp_scale <= 0.0 or arm_kv_scale is None or arm_kv_scale < 0.0
    ):
        raise ValueError("nero arm kp scale must be positive and kv scale non-negative")
    if arm_kp_scale is not None:
        for actuator_name in tuple(f"joint{i}" for i in range(1, 8)):
            actuator_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id < 0:
                raise RuntimeError(f"NERO actuator not found: {actuator_name}")
            kp = float(env.model.actuator_gainprm[actuator_id, 0])
            kv = float(-env.model.actuator_biasprm[actuator_id, 2])
            kp *= float(arm_kp_scale)
            kv *= float(arm_kv_scale)
            env.model.actuator_gainprm[actuator_id, 0] = kp
            env.model.actuator_biasprm[actuator_id, 1] = -kp
            env.model.actuator_biasprm[actuator_id, 2] = -kv
    if wrist_kp is not None:
        for actuator_name in ("joint5", "joint6", "joint7"):
            actuator_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id < 0:
                raise RuntimeError(f"NERO actuator not found: {actuator_name}")
            env.model.actuator_gainprm[actuator_id, 0] = float(wrist_kp)
            env.model.actuator_biasprm[actuator_id, 1] = -float(wrist_kp)
            env.model.actuator_biasprm[actuator_id, 2] = -float(wrist_kv)


def make_policy(
    variant: str,
    env: CableGraspEnv,
    expert_config: FormulaInterceptConfig,
    scripted_config: dict[str, Any],
) -> Any:
    if variant == "scripted":
        return DynamicCableGraspPolicy(env, PolicyConfig(**scripted_config))
    if variant == "expert":
        return FormulaInterceptExpert(env, expert_config)
    raise ValueError(f"unknown variant: {variant}")


class TrajectoryRecorder:
    """Low-overhead FULLPHYSICS recorder without camera rendering or video."""

    def __init__(self, env: CableGraspEnv, episode_dir: Path) -> None:
        self.env = env
        self.base_env = getattr(env, "base_env", env)
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.state_size = mujoco.mj_stateSize(self.base_env.model, self.state_spec)
        self.trajectory = self.episode_dir / "trajectory.npz"
        self.metadata = self.episode_dir / "episode.json"
        self._states: list[np.ndarray] = []
        self._state_times: list[float] = []
        self._policy_actions: list[np.ndarray] = []
        self._requested_actions: list[np.ndarray] = []
        self._applied_actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._reward_components: list[dict[str, float]] = []
        self._closed = False

    def _capture_state(self) -> None:
        state = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(
            self.base_env.model, self.base_env.data, state, self.state_spec
        )
        self._states.append(state)
        self._state_times.append(float(self.base_env.data.time))

    def capture_initial(self) -> None:
        if self._states:
            raise RuntimeError("initial state has already been captured")
        self._capture_state()

    def record_step(
        self,
        policy_action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        if not self._states:
            raise RuntimeError("capture_initial() must be called before record_step()")
        self._capture_state()
        self._policy_actions.append(np.asarray(policy_action, dtype=np.float64).copy())
        requested = info.get(
            "requested_action", getattr(self.base_env, "_last_requested_action", [])
        )
        applied = info.get(
            "applied_action", getattr(self.base_env, "_last_applied_action", [])
        )
        self._requested_actions.append(np.asarray(requested, dtype=np.float64).copy())
        self._applied_actions.append(np.asarray(applied, dtype=np.float64).copy())
        self._rewards.append(float(reward))
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))
        self._reward_components.append({
            str(key): float(value)
            for key, value in info.items()
            if str(key).startswith("reward_") and np.isscalar(value)
        })

    @staticmethod
    def _stack(values: list[np.ndarray]) -> np.ndarray:
        if not values:
            return np.empty((0, 0), dtype=np.float64)
        shapes = {value.shape for value in values}
        if len(shapes) != 1:
            raise RuntimeError(f"inconsistent action shapes in episode: {shapes}")
        return np.stack(values)

    def finish(self, metadata: dict[str, Any]) -> tuple[Path, Path]:
        if self._closed:
            raise RuntimeError("trajectory recorder is already closed")
        reward_names = sorted({
            key for components in self._reward_components for key in components
        })
        reward_values = np.asarray([
            [components.get(name, np.nan) for name in reward_names]
            for components in self._reward_components
        ], dtype=np.float64)
        np.savez_compressed(
            self.trajectory,
            schema_version=np.asarray(1, dtype=np.int64),
            state_spec=np.asarray(int(self.state_spec), dtype=np.int64),
            states=np.stack(self._states),
            state_times=np.asarray(self._state_times, dtype=np.float64),
            policy_actions=self._stack(self._policy_actions),
            requested_actions=self._stack(self._requested_actions),
            applied_actions=self._stack(self._applied_actions),
            rewards=np.asarray(self._rewards, dtype=np.float64),
            terminated=np.asarray(self._terminated, dtype=np.bool_),
            truncated=np.asarray(self._truncated, dtype=np.bool_),
            reward_component_names=np.asarray(reward_names),
            reward_components=reward_values,
            frame_state_indices=np.empty((0,), dtype=np.int64),
            frame_times=np.empty((0,), dtype=np.float64),
            video_fps=np.asarray(0.0, dtype=np.float64),
        )
        document = {
            "recording_schema_version": 1,
            "state_format": "mujoco_mjSTATE_FULLPHYSICS",
            "state_count": len(self._states),
            "control_step_count": len(self._policy_actions),
            "video_frame_count": 0,
            "video_fps": 0.0,
            "files": {"trajectory": self.trajectory.name},
            "result": metadata,
        }
        self.metadata.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.close()
        return self.trajectory, self.metadata

    def discard(self) -> None:
        """Release an unsuccessful attempt without writing a trajectory file."""
        self.close()

    def close(self) -> None:
        self._closed = True


def run_episode(
    label: str,
    base_x: float,
    tcp_dx: float,
    tcp_dy: float,
    tcp_dz: float,
    scenario_name: str,
    variant: str,
    seed: int,
    episode_seconds: float,
    expert_config: FormulaInterceptConfig,
    scripted_config: dict[str, Any],
    nero_use_panda_velocity_limits: bool,
    nero_ready_qpos: list[float] | None,
    nero_wrist_kp: float | None,
    nero_wrist_kv: float | None,
    nero_arm_kp_scale: float | None,
    nero_arm_kv_scale: float | None,
    nero_gripper_force_scale: float | None,
    recording: bool,
    video_fps: float,
    output_root: str,
    success_only: bool,
    trajectory_only: bool = False,
) -> dict[str, Any]:
    if recording and trajectory_only:
        raise ValueError("recording and trajectory_only cannot both be enabled")
    apply_calibration(base_x, tcp_dx, tcp_dy, tcp_dz, nero_ready_qpos)
    scenario = get_scenario(scenario_name)
    config = env_config_for_scenario(
        scenario,
        seed=seed,
        episode_seconds=episode_seconds,
        robot="nero",
    )
    config.target_selection = "middle"
    config.dynamicvla_cameras_enabled = bool(recording)
    if nero_gripper_force_scale is not None:
        if nero_gripper_force_scale <= 0.0:
            raise ValueError("nero gripper force scale must be positive")
        config.gripper_force_scale = float(nero_gripper_force_scale)
    if nero_use_panda_velocity_limits:
        # Keep this as a per-run override so the NERO default profile remains
        # unchanged.  EnvConfig.__post_init__ intentionally replaces the
        # default Panda tuple with NERO limits, so this assignment must happen
        # after env_config_for_scenario() has constructed the config.
        config.arm_joint_velocity_limits = DEFAULT_ARM_JOINT_VELOCITY_LIMITS
    env = CableGraspEnv(config)
    recorder: EpisodeRecorder | TrajectoryRecorder | None = None
    episode_dir: Path | None = None
    try:
        apply_nero_servo_override(
            env,
            nero_wrist_kp,
            nero_wrist_kv,
            nero_arm_kp_scale,
            nero_arm_kv_scale,
        )
        env.reset(seed=seed)
        policy = make_policy(variant, env, expert_config, scripted_config)
        if recording or trajectory_only:
            episode_dir = (
                Path(output_root)
                / label
                / variant
                / scenario_name
                / f"seed_{seed}"
            )
            recorder = (
                EpisodeRecorder(env, episode_dir, video_fps=video_fps)
                if recording
                else TrajectoryRecorder(env, episode_dir)
            )
            recorder.capture_initial()
        steps = 0
        termination_reason: str | None = None
        tracking_error_sum = 0.0
        tracking_error_count = 0
        max_tracking_error = 0.0
        while not policy.finished and env.data.time < episode_seconds:
            action = policy.action()
            desired_position = getattr(policy, "last_desired", env.hand_position).copy()
            _, reward, terminated, truncated, info = env.step(action)
            if recorder is not None:
                recorder.record_step(action, reward, terminated, truncated, info)
            tracking_error = float(np.linalg.norm(desired_position - env.hand_position))
            tracking_error_sum += tracking_error
            tracking_error_count += 1
            max_tracking_error = max(max_tracking_error, tracking_error)
            steps += 1
            if truncated:
                termination_reason = info.get("termination_reason")
                if not env.ever_success:
                    policy.result = (
                        "failed_motion_boundary"
                        if termination_reason == "rigid_motion_boundary_crossed"
                        else "failed_timeout"
                    )
                policy.finished = True
        expert_info = policy.expert_info() if hasattr(policy, "expert_info") else {}
        row = {
            "calibration": label,
            "base_x": float(base_x),
            "tcp_dx": float(tcp_dx),
            "tcp_dy": float(tcp_dy),
            "tcp_dz": float(tcp_dz),
            "scenario": scenario_name,
            "variant": variant,
            "seed": seed,
            "success": bool(env.ever_success),
            "outcome": outcome(env),
            "ever_bilateral_candidate": bool(env.ever_bilateral_candidate),
            "ever_confirmed_grasp": bool(env.ever_confirmed_grasp),
            "steps": steps,
            "sim_time": float(env.data.time),
            "termination_reason": termination_reason,
            "policy_result": policy.result,
            "final_phase": policy.phase.name,
            "retry_count": int(getattr(policy, "retry_count", 0)),
            "attempt_failure_count": int(getattr(policy, "attempt_failure_count", 0)),
            "mean_tcp_tracking_error": (
                tracking_error_sum / tracking_error_count
                if tracking_error_count else 0.0
            ),
            "max_tcp_tracking_error": max_tracking_error,
            "max_tilt_error": float(getattr(policy, "max_tilt_error", 0.0)),
            **expert_info,
        }
        if recorder is not None:
            if trajectory_only and success_only and not bool(row["success"]):
                # Trajectory samples stay in memory until the outcome is known;
                # failed attempts therefore do not incur NPZ write/delete I/O.
                recorder.discard()
                row.update({
                    "global_video": None,
                    "wrist_video": None,
                    "trajectory": None,
                    "video_metadata": None,
                })
                if episode_dir is not None and episode_dir.exists():
                    shutil.rmtree(episode_dir)
            elif trajectory_only:
                trajectory, metadata = recorder.finish(row)
                row.update({
                    "global_video": None,
                    "wrist_video": None,
                    "trajectory": str(trajectory.resolve()),
                    "video_metadata": str(metadata.resolve()),
                })
            else:
                artifacts = recorder.finish(row)
                row.update({
                    "global_video": str(artifacts.global_video.resolve()),
                    "wrist_video": str(artifacts.wrist_video.resolve()),
                    "trajectory": str(artifacts.trajectory.resolve()),
                    "video_metadata": str(artifacts.metadata.resolve()),
                })
                if success_only and not bool(row["success"]):
                    # Keep the attempt outcome in episodes.csv, but do not make
                    # failed attempts part of the trajectory/video dataset.
                    for field in (
                        "global_video",
                        "wrist_video",
                        "trajectory",
                        "video_metadata",
                    ):
                        row[field] = None
                    if episode_dir is not None and episode_dir.exists():
                        shutil.rmtree(episode_dir)
        return row
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    bool_fields = {"success", "ever_bilateral_candidate", "ever_confirmed_grasp"}
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for field in bool_fields:
                if field in row:
                    row[field] = str(row[field]).lower() == "true"
            for field in ("seed",):
                if field in row:
                    row[field] = int(row[field])
            rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    for row in rows:
        outcomes[str(row["outcome"])] = outcomes.get(str(row["outcome"]), 0) + 1
    successes = sum(bool(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "bilateral_candidates": sum(
            bool(row["ever_bilateral_candidate"]) for row in rows
        ),
        "confirmed_grasps": sum(bool(row["ever_confirmed_grasp"]) for row in rows),
        "mean_tcp_tracking_error": (
            sum(float(row["mean_tcp_tracking_error"]) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "max_tcp_tracking_error": max(
            (float(row["max_tcp_tracking_error"]) for row in rows), default=0.0
        ),
        "outcomes": outcomes,
    }


def run_cell(job: dict[str, Any]) -> dict[str, Any]:
    root = Path(job["output_root"])
    cell_dir = root / job["calibration"] / job["variant"] / job["scenario"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cell_dir / "episodes.partial.csv"
    rows = load_rows(checkpoint)
    completed_seeds = {int(row["seed"]) for row in rows}
    expert_config = FormulaInterceptConfig(**job["expert_config"])
    scripted_config = dict(job["scripted_config"])
    target_successes = job.get("successes_per_scenario")
    max_attempts = job.get("max_attempts_per_scenario")
    if target_successes is None:
        attempt_limit = int(job["episodes"])
        target_successes = None
    else:
        attempt_limit = int(max_attempts)
        target_successes = int(target_successes)
    successful_rows = sum(bool(row.get("success", False)) for row in rows)
    for offset in range(attempt_limit):
        if target_successes is not None and successful_rows >= target_successes:
            break
        seed = int(job["seed_start"]) + offset
        if seed in completed_seeds:
            continue
        row = run_episode(
            job["calibration"],
            float(job["base_x"]),
            float(job["tcp_dx"]),
            float(job["tcp_dy"]),
            float(job["tcp_dz"]),
            job["scenario"],
            job["variant"],
            seed,
            float(job["episode_seconds"]),
            expert_config,
            scripted_config,
            bool(job["nero_use_panda_velocity_limits"]),
            job.get("nero_ready_qpos"),
            job.get("nero_wrist_kp"),
            job.get("nero_wrist_kv"),
            job.get("nero_arm_kp_scale"),
            job.get("nero_arm_kv_scale"),
            job.get("nero_gripper_force_scale"),
            bool(job.get("recording", False)),
            float(job.get("video_fps", 25.0)),
            str(job["output_root"]),
            bool(job.get("success_only", False)),
            bool(job.get("trajectory_only", False)),
        )
        rows.append(row)
        if bool(row.get("success", False)):
            successful_rows += 1
        append_checkpoint(checkpoint, row)
    rows.sort(key=lambda row: int(row["seed"]))
    write_rows(cell_dir / "episodes.csv", rows)
    summary = {
        "calibration": job["calibration"],
        "base_x": job["base_x"],
        "tcp_dx": job["tcp_dx"],
        "tcp_dz": job["tcp_dz"],
        "nero_ready_qpos": job.get("nero_ready_qpos"),
        "nero_wrist_kp": job.get("nero_wrist_kp"),
        "nero_wrist_kv": job.get("nero_wrist_kv"),
        "nero_gripper_force_scale": job.get("nero_gripper_force_scale"),
        "scenario": job["scenario"],
        "variant": job["variant"],
        "robot": "nero",
        "target_selection": "middle",
        "successes_per_scenario": job.get("successes_per_scenario"),
        "max_attempts_per_scenario": job.get("max_attempts_per_scenario"),
        "success_only": bool(job.get("success_only", False)),
        "expert_config": job["expert_config"],
        "summary": summarize(rows),
        "checkpoint": str(checkpoint.resolve()),
        "episodes_file": str((cell_dir / "episodes.csv").resolve()),
    }
    (cell_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--successes-per-scenario", type=int, default=None,
        help="stop each scenario cell after this many successful episodes",
    )
    parser.add_argument(
        "--max-attempts-per-scenario", type=int, default=None,
        help="attempt budget for success-quota collection",
    )
    parser.add_argument("--seed", type=int, default=20280804)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--variants", nargs="+", choices=("scripted", "expert"),
        default=["scripted", "expert"],
    )
    parser.add_argument("--scripted-config-json", default="{}")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=ALL_SCENARIOS,
        default=list(SCENARIOS),
    )
    parser.add_argument(
        "--scripted-yaw-offset-values",
        nargs="+",
        type=float,
        default=[0.0],
        help="NERO Scripted horizontal grasp-frame yaw offsets in degrees",
    )
    parser.add_argument(
        "--nero-use-panda-velocity-limits",
        action="store_true",
        help="Override NERO arm joint velocity limits with Panda's limits for this run",
    )
    parser.add_argument(
        "--nero-ready-qpos",
        nargs=7,
        type=float,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Override NERO's seven ready arm joint positions for this run",
    )
    parser.add_argument(
        "--nero-wrist-kp", type=float, default=None,
        help="Override NERO joint5-7 position-servo kp for this run",
    )
    parser.add_argument(
        "--nero-wrist-kv", type=float, default=None,
        help="Override NERO joint5-7 position-servo kv for this run",
    )
    parser.add_argument(
        "--nero-arm-kp-scale", type=float, default=None,
        help="Scale NERO joint1-7 position-servo kp for this run",
    )
    parser.add_argument(
        "--nero-arm-kv-scale", type=float, default=None,
        help="Scale NERO joint1-7 position-servo kv for this run",
    )
    parser.add_argument(
        "--nero-gripper-force-scale", type=float, default=None,
        help="Override NERO gripper force scale for this run",
    )
    parser.add_argument(
        "--record-video", action="store_true",
        help="record global/wrist MP4 videos and FULLPHYSICS trajectories",
    )
    parser.add_argument(
        "--success-only", action="store_true",
        help="when recording, remove failed-attempt artifacts from the dataset",
    )
    parser.add_argument(
        "--trajectory-only", action="store_true",
        help="save successful FULLPHYSICS trajectories without rendering videos",
    )
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--base-values", nargs="+", type=float, default=[0.15, 0.20, 0.25])
    parser.add_argument("--tcp-dx-values", nargs="+", type=float, default=[0.0])
    parser.add_argument(
        "--tcp-dy-values", nargs="+", type=float, default=[0.0],
        help="NERO grasp-center offset along link7 local y in metres",
    )
    parser.add_argument("--tcp-dz-values", nargs="+", type=float, default=[0.0])
    args = parser.parse_args()
    if (
        args.episodes <= 0
        or args.episode_seconds <= 0
        or args.workers < 1
        or args.video_fps <= 0
    ):
        parser.error("episodes, duration, workers, and video-fps must be positive")
    if args.record_video and args.trajectory_only:
        parser.error("--record-video and --trajectory-only are mutually exclusive")
    if args.successes_per_scenario is not None:
        if args.successes_per_scenario <= 0:
            parser.error("successes-per-scenario must be positive")
        if args.max_attempts_per_scenario is None:
            parser.error(
                "--max-attempts-per-scenario is required with "
                "--successes-per-scenario"
            )
        if args.max_attempts_per_scenario < args.successes_per_scenario:
            parser.error(
                "max-attempts-per-scenario must be at least successes-per-scenario"
            )
    elif args.max_attempts_per_scenario is not None:
        parser.error(
            "--max-attempts-per-scenario requires --successes-per-scenario"
        )
    if (args.nero_wrist_kp is None) != (args.nero_wrist_kv is None):
        parser.error("--nero-wrist-kp and --nero-wrist-kv must be supplied together")
    if (args.nero_arm_kp_scale is None) != (args.nero_arm_kv_scale is None):
        parser.error(
            "--nero-arm-kp-scale and --nero-arm-kv-scale must be supplied together"
        )
    if args.output.exists() and not args.resume:
        parser.error(f"output already exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    expert_config = asdict(
        FormulaInterceptConfig(
            shadow_rollout_enabled=False,
            dynamic_portfolio_enabled=False,
        )
    )
    try:
        scripted_config = json.loads(args.scripted_config_json)
    except json.JSONDecodeError as error:
        parser.error(f"invalid --scripted-config-json: {error}")
    if not isinstance(scripted_config, dict):
        parser.error("--scripted-config-json must decode to an object")
    jobs: list[dict[str, Any]] = []
    for base_x in args.base_values:
        for tcp_dx in args.tcp_dx_values:
            for tcp_dy in args.tcp_dy_values:
                for tcp_dz in args.tcp_dz_values:
                    for yaw_offset in args.scripted_yaw_offset_values:
                        label = (
                            f"base_{base_x:.3f}_tcpdx_{tcp_dx:+.3f}"
                            f"_tcpdy_{tcp_dy:+.3f}_tcpdz_{tcp_dz:+.3f}"
                            f"_yaw_{yaw_offset:+.1f}"
                            .replace("+", "p").replace("-", "m").replace(".", "p")
                        )
                        job_scripted_config = dict(scripted_config)
                        job_scripted_config["nero_grasp_yaw_offset_deg"] = yaw_offset
                        for scenario in args.scenarios:
                            for variant in args.variants:
                                jobs.append({
                                    "calibration": label,
                                    "base_x": base_x,
                                    "tcp_dx": tcp_dx,
                                    "tcp_dy": tcp_dy,
                                    "tcp_dz": tcp_dz,
                                    "scenario": scenario,
                                    "variant": variant,
                                    "episodes": args.episodes,
                                    "seed_start": args.seed,
                                    "episode_seconds": args.episode_seconds,
                                    "expert_config": expert_config,
                                    "scripted_config": job_scripted_config,
                                    "nero_use_panda_velocity_limits": (
                                        args.nero_use_panda_velocity_limits
                                    ),
                                    "nero_ready_qpos": args.nero_ready_qpos,
                                    "nero_wrist_kp": args.nero_wrist_kp,
                                    "nero_wrist_kv": args.nero_wrist_kv,
                                    "nero_arm_kp_scale": args.nero_arm_kp_scale,
                                    "nero_arm_kv_scale": args.nero_arm_kv_scale,
                                    "nero_gripper_force_scale": args.nero_gripper_force_scale,
                                    "recording": args.record_video,
                                    "video_fps": args.video_fps,
                                    "successes_per_scenario": (
                                        args.successes_per_scenario
                                    ),
                                    "max_attempts_per_scenario": (
                                        args.max_attempts_per_scenario
                                    ),
                                     "success_only": args.success_only,
                                     "trajectory_only": args.trajectory_only,
                                     "output_root": str(args.output.resolve()),
                                })
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_cell, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            rows.extend(result["rows"])
            summary = result["summary"]
            cell_summary = summary["summary"]
            print(
                f"cell_completed={index}/{len(jobs)} "
                f"{summary['calibration']} {summary['scenario']} {summary['variant']} "
                f"success={cell_summary['successes']}/{cell_summary['episodes']} "
                f"candidate={cell_summary['bilateral_candidates']}/{cell_summary['episodes']}",
                flush=True,
            )
    rows.sort(key=lambda row: (
        str(row["calibration"]), str(row["scenario"]),
        str(row["variant"]), int(row["seed"]),
    ))
    write_rows(args.output / "episodes.csv", rows)
    aggregate: dict[str, Any] = {
        "config": {
            "robot": "nero",
            "scenarios": list(args.scenarios),
            "variants": list(args.variants),
            "episodes_per_cell": args.episodes,
            "successes_per_scenario": args.successes_per_scenario,
            "max_attempts_per_scenario": args.max_attempts_per_scenario,
            "seed_start": args.seed,
            "episode_seconds": args.episode_seconds,
            "workers": args.workers,
            "expert_config": expert_config,
            "scripted_config": scripted_config,
            "nero_use_panda_velocity_limits": args.nero_use_panda_velocity_limits,
            "nero_ready_qpos": args.nero_ready_qpos,
            "nero_wrist_kp": args.nero_wrist_kp,
            "nero_wrist_kv": args.nero_wrist_kv,
            "nero_arm_kp_scale": args.nero_arm_kp_scale,
            "nero_arm_kv_scale": args.nero_arm_kv_scale,
            "nero_gripper_force_scale": args.nero_gripper_force_scale,
            "recording": args.record_video,
            "success_only": args.success_only,
            "trajectory_only": args.trajectory_only,
            "video_fps": args.video_fps,
            "scripted_yaw_offset_values": args.scripted_yaw_offset_values,
            "base_values": args.base_values,
            "tcp_dx_values": args.tcp_dx_values,
            "tcp_dy_values": args.tcp_dy_values,
            "tcp_dz_values": args.tcp_dz_values,
        },
        "by_calibration": {},
        "elapsed_seconds": time.perf_counter() - started,
    }
    for calibration in sorted({str(row["calibration"]) for row in rows}):
        calibration_rows = [row for row in rows if row["calibration"] == calibration]
        aggregate["by_calibration"][calibration] = {
            scenario: {
                variant: summarize([
                    row for row in calibration_rows
                    if row["scenario"] == scenario and row["variant"] == variant
                ])
                for variant in args.variants
            }
            for scenario in args.scenarios
        }
    (args.output / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
