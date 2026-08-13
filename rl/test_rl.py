"""评估已训练PPO策略：支持批量headless统计和MuJoCo实时GUI。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from cable_grasp_env import XML_PATH
from failure_taxonomy import (
    TASK_OUTCOME_TYPES,
    break_causal_class,
    classify_task_outcome,
    confirmed_break_times,
    scene_fingerprint,
)
from .rl_cable_env import RLCableGraspEnv


FAILURE_TYPES = (
    "success",
    "never_pinch",
    "pinch_not_secured",
    "active_open_after_secured",
    "physical_slip_after_secured",
    "open_during_contact_loss_after_secured",
    "secured_but_no_strict_success",
)

EPISODE_FIELDS = (
    "episode",
    "seed",
    "actual_episode_seed",
    "scene_fingerprint",
    "success",
    "failure_type",
    "policy_internal_success",
    "policy_failure_type",
    "task_success",
    "task_failure_type",
    "terminated",
    "truncated",
    "episode_return",
    "steps",
    "sim_time_seconds",
    "wall_time_seconds",
    "initial_cable_dx_m",
    "initial_cable_dy_m",
    "disturbance_phase",
    "disturbance_spatial_phase",
    "target_body_id",
    "rl_target_body_id",
    "grasped_body_id",
    "ever_pinch",
    "ever_secured",
    "ever_bilateral_candidate",
    "ever_confirmed_grasp",
    "first_pinch_time_s",
    "first_secured_time_s",
    "active_open_after_secured",
    "first_active_open_time_s",
    "first_active_open_command_time_s",
    "active_open_command_steps_after_secured",
    "active_open_break_count_after_secured",
    "physical_slip_after_secured",
    "first_physical_slip_time_s",
    "physical_slip_break_count_after_secured",
    "open_during_contact_loss_after_secured",
    "first_open_during_contact_loss_time_s",
    "open_during_contact_loss_break_count_after_secured",
    "last_grasp_break_reason",
    "last_grasp_break_causal_class",
    "active_open_after_confirmed_count",
    "physical_slip_after_confirmed_count",
    "open_during_contact_loss_after_confirmed_count",
    "first_active_open_after_confirmed_time_s",
    "first_physical_slip_after_confirmed_time_s",
    "first_open_during_contact_loss_after_confirmed_time_s",
    "target_distance_final_m",
    "target_distance_min_m",
    "finger_aperture_final_m",
    "grasp_lift_delta_final_m",
    "grasp_lift_delta_peak_m",
    "strict_success_hold_final_s",
    "strict_success_hold_peak_s",
    "lifted_fraction_final",
    "lifted_fraction_peak",
    "max_z_final_m",
    "max_z_peak_m",
    "action_rms",
    "action_rate_rms",
    "action_saturation_fraction",
    "policy_inference_mean_ms",
    "policy_inference_p95_ms",
)


def _finite_or_none(value: object) -> float | int | str | None:
    """Return a CSV/JSON-friendly scalar, leaving missing metrics blank."""
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return str(value)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
    }


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _git_metadata(project_root: Path) -> dict[str, object]:
    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
    except OSError:
        return {"commit": None, "dirty": None}
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _resolve_model_file(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_file():
        return expanded
    zip_candidate = Path(f"{expanded}.zip")
    return zip_candidate if zip_candidate.is_file() else expanded


def _json_arguments(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _compiled_model_sha256(model: object) -> str:
    """Hash the fully compiled MuJoCo model without retaining a no-video artifact."""

    import mujoco

    with tempfile.TemporaryDirectory(prefix="cable_eval_model_") as directory:
        path = Path(directory) / "model.mjb"
        mujoco.mj_saveModel(model, str(path), None)
        digest = _sha256(path)
    if digest is None:
        raise RuntimeError("failed to hash compiled MuJoCo model")
    return digest


@dataclass
class EpisodeDiagnostics:
    """Accumulate method-independent outcome and action diagnostics for one episode."""

    ever_pinch: bool = False
    ever_secured: bool = False
    first_pinch_time: float | None = None
    first_secured_time: float | None = None
    first_active_open_time: float | None = None
    first_active_open_command_time: float | None = None
    active_open_command_steps: int = 0
    active_open_break_count: int = 0
    first_physical_slip_time: float | None = None
    physical_slip_break_count: int = 0
    first_open_during_contact_loss_time: float | None = None
    open_during_contact_loss_break_count: int = 0
    last_grasp_break_reason: str | None = None
    last_grasp_break_causal_class: str | None = None
    min_target_distance: float = math.inf
    peak_grasp_lift_delta: float = 0.0
    peak_strict_success_hold: float = 0.0
    peak_lifted_fraction: float = 0.0
    peak_max_z: float = -math.inf
    action_square_sum: float = 0.0
    action_count: int = 0
    action_rate_square_sum: float = 0.0
    action_rate_count: int = 0
    saturated_action_elements: int = 0
    action_elements: int = 0
    previous_action: np.ndarray | None = None
    inference_seconds: list[float] = field(default_factory=list)
    ever_physical_slip_event: bool = False
    _last_break_key: tuple[float | None, str | None] | None = None

    def observe_info(
        self,
        info: dict,
        sim_time: float,
        env: RLCableGraspEnv,
    ) -> None:
        pinch_now = bool(info.get("pinch_confirmed", False))
        ever_pinch_now = pinch_now or bool(info.get("ever_pinched", False))
        if ever_pinch_now and not self.ever_pinch:
            self.first_pinch_time = sim_time
        self.ever_pinch = self.ever_pinch or ever_pinch_now

        secured_now = bool(info.get("secured_grasp", False))
        ever_secured_now = secured_now or bool(info.get("ever_grasped", False))
        if ever_secured_now and not self.ever_secured:
            self.first_secured_time = sim_time
        self.ever_secured = self.ever_secured or ever_secured_now

        target_distance = _finite_or_none(info.get("target_distance"))
        if isinstance(target_distance, (int, float)):
            self.min_target_distance = min(self.min_target_distance, float(target_distance))
        self.peak_grasp_lift_delta = max(
            self.peak_grasp_lift_delta,
            float(_finite_or_none(info.get("grasp_lift_delta")) or 0.0),
        )
        self.peak_strict_success_hold = max(
            self.peak_strict_success_hold,
            float(_finite_or_none(info.get("strict_success_hold")) or 0.0),
        )
        self.peak_lifted_fraction = max(
            self.peak_lifted_fraction,
            float(_finite_or_none(info.get("lifted_fraction")) or 0.0),
        )
        max_z = _finite_or_none(info.get("max_z"))
        if isinstance(max_z, (int, float)):
            self.peak_max_z = max(self.peak_max_z, float(max_z))

        self.active_open_break_count = int(
            info.get("active_open_after_secured_count", self.active_open_break_count)
        )
        self.physical_slip_break_count = int(
            info.get(
                "physical_slip_after_secured_count",
                self.physical_slip_break_count,
            )
        )
        self.open_during_contact_loss_break_count = int(
            info.get(
                "open_during_contact_loss_after_secured_count",
                self.open_during_contact_loss_break_count,
            )
        )
        self.ever_physical_slip_event = bool(
            self.ever_physical_slip_event
            or info.get("physical_slip_after_secured_event", False)
        )

        event = getattr(env.base_env, "last_grasp_break", None)
        if event is None:
            return
        break_time = _finite_or_none(event.get("time"))
        reason_value = event.get("reason")
        reason = None if reason_value is None else str(reason_value)
        key = (
            float(break_time) if isinstance(break_time, (int, float)) else None,
            reason,
        )
        if key == self._last_break_key:
            return
        self._last_break_key = key
        self.last_grasp_break_reason = reason
        causal_class = break_causal_class(event)
        self.last_grasp_break_causal_class = causal_class
        session_event = bool(
            info.get("active_open_after_secured_event", False)
            or info.get("physical_slip_after_secured_event", False)
            or info.get("open_during_contact_loss_after_secured_event", False)
        )
        if not session_event:
            return
        event_time = key[0] if key[0] is not None else sim_time
        if causal_class == "active_open":
            if self.first_active_open_time is None:
                self.first_active_open_time = event_time
        elif causal_class == "physical_slip":
            if self.first_physical_slip_time is None:
                self.first_physical_slip_time = event_time
        elif causal_class == "open_during_contact_loss":
            if self.first_open_during_contact_loss_time is None:
                self.first_open_during_contact_loss_time = event_time

    def observe_action(
        self,
        action: np.ndarray,
        sim_time: float,
        inference_seconds: float,
        open_threshold: float,
    ) -> None:
        action = np.asarray(action, dtype=np.float64)
        self.action_square_sum += float(np.sum(np.square(action)))
        self.action_count += int(action.size)
        self.saturated_action_elements += int(np.count_nonzero(np.abs(action) >= 0.99))
        self.action_elements += int(action.size)
        if self.previous_action is not None:
            difference = action - self.previous_action
            self.action_rate_square_sum += float(np.sum(np.square(difference)))
            self.action_rate_count += int(difference.size)
        self.previous_action = action.copy()
        self.inference_seconds.append(float(inference_seconds))

        if self.ever_secured and float(action[7]) >= open_threshold:
            self.active_open_command_steps += 1
            if self.first_active_open_command_time is None:
                self.first_active_open_command_time = sim_time

    @property
    def active_open_after_secured(self) -> bool:
        return self.first_active_open_time is not None

    @property
    def physical_slip_after_secured(self) -> bool:
        return self.first_physical_slip_time is not None

    @property
    def open_during_contact_loss_after_secured(self) -> bool:
        return self.first_open_during_contact_loss_time is not None


def classify_failure(success: bool, diagnostics: EpisodeDiagnostics) -> str:
    """Assign one mutually exclusive outcome using the earliest post-secured cause."""
    if success:
        return "success"
    if not diagnostics.ever_pinch:
        return "never_pinch"
    if not diagnostics.ever_secured:
        return "pinch_not_secured"

    candidates = {
        "active_open_after_secured": diagnostics.first_active_open_time,
        "physical_slip_after_secured": diagnostics.first_physical_slip_time,
        "open_during_contact_loss_after_secured": (
            diagnostics.first_open_during_contact_loss_time
        ),
    }
    observed = {name: time for name, time in candidates.items() if time is not None}
    if observed:
        return min(observed, key=lambda name: (float(observed[name]), name))
    if diagnostics.ever_secured:
        return "secured_but_no_strict_success"
    raise RuntimeError("inconsistent failure diagnostics")


def _manifest(
    args: argparse.Namespace,
    model: PPO,
    env: RLCableGraspEnv,
    output_dir: Path,
) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    rl_source = inspect.getsourcefile(type(env))
    base_source = inspect.getsourcefile(type(env.base_env))
    taxonomy_source = Path(__file__).resolve().parents[1] / "failure_taxonomy.py"
    action_seconds = float(
        env.model.opt.timestep * max(1, env.base_env.config.frame_skip)
    )
    return {
        "schema_version": 2,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(output_dir.resolve()),
        "command": [sys.executable, *sys.argv],
        "arguments": _json_arguments(args),
        "episode_seed_rule": {
            "first_seed": args.seed,
            "count": args.episodes,
            "formula": "seed + episode_index (zero based)",
        },
        "evaluator_source": _file_record(Path(__file__)),
        "failure_taxonomy_source": _file_record(taxonomy_source),
        "checkpoint": _file_record(args.model),
        "environment": {
            "source_xml": _file_record(XML_PATH),
            "compiled_model_sha256": _compiled_model_sha256(env.model),
            "base_environment_source": (
                _file_record(Path(base_source)) if base_source else None
            ),
            "rl_environment_source": _file_record(Path(rl_source)) if rl_source else None,
            "env_config": asdict(env.base_env.config),
            "rl_config": asdict(env.rl_config),
            "action_seconds": action_seconds,
            "observation_names": list(env.OBSERVATION_NAMES),
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
        },
        "policy": {
            "algorithm": type(model).__name__,
            "policy_class": type(model.policy).__name__,
            "deterministic": not args.stochastic,
            "saved_num_timesteps": int(model.num_timesteps),
            "sampling_seed": args.seed,
        },
        "recording": {
            "enabled": not args.no_video,
            "fps": args.video_fps if not args.no_video else None,
            "width": args.video_width if not args.no_video else None,
            "height": args.video_height if not args.no_video else None,
        },
        "policy_outcome_types": list(FAILURE_TYPES),
        "task_outcome_types": list(TASK_OUTCOME_TYPES),
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _distribution_version("numpy"),
            "stable_baselines3": _distribution_version("stable-baselines3"),
            "gymnasium": _distribution_version("gymnasium"),
            "mujoco": _distribution_version("mujoco"),
            "torch": _distribution_version("torch"),
            "opencv_python": _distribution_version("opencv-python"),
        },
        "git": _git_metadata(project_root),
    }


def print_episode(episode: int, episode_return: float, steps: int, info: dict) -> None:
    print(
        f"episode={episode} success={bool(info['success'])} return={episode_return:.3f} "
        f"steps={steps} sim_time={steps * 0.02:.3f}s "
        f"target_distance={info['target_distance']:.3f}m "
        f"grasped_body={info['grasped_body_id']} "
        f"pinch={info['pinch_confirmed']} secured={info['secured_grasp']} "
        f"ever_secured={info['ever_grasped']} "
        f"aperture={1000.0 * info['finger_aperture']:.1f}mm "
        f"lift_delta={1000.0 * info['grasp_lift_delta']:.1f}mm "
        f"strict_hold={info['strict_success_hold']:.2f}s "
        f"lifted_fraction={info['lifted_fraction']:.2f} max_z={info['max_z']:.3f}m",
        flush=True,
    )


def _episode_row(
    *,
    episode: int,
    episode_seed: int,
    success: bool,
    failure_type: str,
    task_success: bool,
    task_failure_type: str,
    terminated: bool,
    truncated: bool,
    episode_return: float,
    steps: int,
    sim_time: float,
    wall_time: float,
    initial_info: dict,
    final_info: dict,
    diagnostics: EpisodeDiagnostics,
    break_events: list[dict],
) -> dict[str, object]:
    actual_episode_seed = initial_info.get("episode_seed")
    if actual_episode_seed != episode_seed:
        raise RuntimeError(
            "episode seed mismatch: "
            f"requested={episode_seed}, actual={actual_episode_seed}"
        )
    inference_ms = 1000.0 * np.asarray(diagnostics.inference_seconds, dtype=np.float64)
    action_rms = (
        math.sqrt(diagnostics.action_square_sum / diagnostics.action_count)
        if diagnostics.action_count
        else None
    )
    action_rate_rms = (
        math.sqrt(
            diagnostics.action_rate_square_sum / diagnostics.action_rate_count
        )
        if diagnostics.action_rate_count
        else None
    )
    confirmed_times = confirmed_break_times(break_events)
    return {
        "episode": episode,
        "seed": episode_seed,
        "actual_episode_seed": actual_episode_seed,
        "scene_fingerprint": scene_fingerprint(initial_info),
        "success": success,
        "failure_type": failure_type,
        "policy_internal_success": success,
        "policy_failure_type": failure_type,
        "task_success": task_success,
        "task_failure_type": task_failure_type,
        "terminated": terminated,
        "truncated": truncated,
        "episode_return": float(episode_return),
        "steps": steps,
        "sim_time_seconds": sim_time,
        "wall_time_seconds": wall_time,
        "initial_cable_dx_m": _finite_or_none(initial_info.get("initial_cable_dx")),
        "initial_cable_dy_m": _finite_or_none(initial_info.get("initial_cable_dy")),
        "disturbance_phase": _finite_or_none(initial_info.get("disturbance_phase")),
        "disturbance_spatial_phase": _finite_or_none(
            initial_info.get("disturbance_spatial_phase")
        ),
        "target_body_id": _finite_or_none(initial_info.get("target_body_id")),
        "rl_target_body_id": _finite_or_none(final_info.get("rl_target_body_id")),
        "grasped_body_id": _finite_or_none(final_info.get("grasped_body_id")),
        "ever_pinch": diagnostics.ever_pinch,
        "ever_secured": diagnostics.ever_secured,
        "ever_bilateral_candidate": bool(
            final_info.get("ever_bilateral_candidate", False)
        ),
        "ever_confirmed_grasp": bool(
            final_info.get("ever_confirmed_grasp", False)
        ),
        "first_pinch_time_s": diagnostics.first_pinch_time,
        "first_secured_time_s": diagnostics.first_secured_time,
        "active_open_after_secured": diagnostics.active_open_after_secured,
        "first_active_open_time_s": diagnostics.first_active_open_time,
        "first_active_open_command_time_s": (
            diagnostics.first_active_open_command_time
        ),
        "active_open_command_steps_after_secured": diagnostics.active_open_command_steps,
        "active_open_break_count_after_secured": diagnostics.active_open_break_count,
        "physical_slip_after_secured": diagnostics.physical_slip_after_secured,
        "first_physical_slip_time_s": diagnostics.first_physical_slip_time,
        "physical_slip_break_count_after_secured": diagnostics.physical_slip_break_count,
        "open_during_contact_loss_after_secured": (
            diagnostics.open_during_contact_loss_after_secured
        ),
        "first_open_during_contact_loss_time_s": (
            diagnostics.first_open_during_contact_loss_time
        ),
        "open_during_contact_loss_break_count_after_secured": (
            diagnostics.open_during_contact_loss_break_count
        ),
        "last_grasp_break_reason": diagnostics.last_grasp_break_reason,
        "last_grasp_break_causal_class": diagnostics.last_grasp_break_causal_class,
        "active_open_after_confirmed_count": int(
            final_info.get("active_open_after_confirmed_count", 0)
        ),
        "physical_slip_after_confirmed_count": int(
            final_info.get("physical_slip_after_confirmed_count", 0)
        ),
        "open_during_contact_loss_after_confirmed_count": int(
            final_info.get("open_during_contact_loss_after_confirmed_count", 0)
        ),
        "first_active_open_after_confirmed_time_s": confirmed_times.get(
            "active_open"
        ),
        "first_physical_slip_after_confirmed_time_s": confirmed_times.get(
            "physical_slip"
        ),
        "first_open_during_contact_loss_after_confirmed_time_s": (
            confirmed_times.get("open_during_contact_loss")
        ),
        "target_distance_final_m": _finite_or_none(final_info.get("target_distance")),
        "target_distance_min_m": _finite_or_none(diagnostics.min_target_distance),
        "finger_aperture_final_m": _finite_or_none(final_info.get("finger_aperture")),
        "grasp_lift_delta_final_m": _finite_or_none(
            final_info.get("grasp_lift_delta")
        ),
        "grasp_lift_delta_peak_m": diagnostics.peak_grasp_lift_delta,
        "strict_success_hold_final_s": _finite_or_none(
            final_info.get("strict_success_hold")
        ),
        "strict_success_hold_peak_s": diagnostics.peak_strict_success_hold,
        "lifted_fraction_final": _finite_or_none(final_info.get("lifted_fraction")),
        "lifted_fraction_peak": diagnostics.peak_lifted_fraction,
        "max_z_final_m": _finite_or_none(final_info.get("max_z")),
        "max_z_peak_m": _finite_or_none(diagnostics.peak_max_z),
        "action_rms": action_rms,
        "action_rate_rms": action_rate_rms,
        "action_saturation_fraction": (
            diagnostics.saturated_action_elements / diagnostics.action_elements
            if diagnostics.action_elements
            else None
        ),
        "policy_inference_mean_ms": (
            float(np.mean(inference_ms)) if inference_ms.size else None
        ),
        "policy_inference_p95_ms": (
            float(np.percentile(inference_ms, 95.0)) if inference_ms.size else None
        ),
    }


def run_headless(args: argparse.Namespace, model: PPO) -> None:
    outcomes = Counter({outcome: 0 for outcome in FAILURE_TYPES})
    task_outcomes = Counter({outcome: 0 for outcome in TASK_OUTCOME_TYPES})
    returns: list[float] = []
    lengths: list[int] = []
    success_mismatches = 0
    episodes_with_physical_slip = 0
    successful_episodes_with_physical_slip = 0
    task_success_without_confirmed_grasp = 0
    env = RLCableGraspEnv(
        seed=args.seed,
        disturbance_strength=args.disturbance,
        episode_seconds=args.episode_seconds,
    )

    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    output_dir = args.video_dir / run_name
    suffix = 1
    while output_dir.exists():
        output_dir = args.video_dir / f"{run_name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True)
    csv_path = output_dir / "episodes.csv"
    manifest_path = output_dir / "run_manifest.json"
    manifest = _manifest(args, model, env, output_dir)
    _write_json(manifest_path, manifest)

    renderer = None
    camera = None
    state_spec = None
    state_size = None
    mjb_path: Path | None = None
    completed_episodes = 0
    print(
        f"headless_{'results' if args.no_video else 'recordings'}={output_dir.resolve()}",
        flush=True,
    )

    try:
        if not args.no_video:
            import cv2
            import mujoco

            env.model.vis.global_.offwidth = max(
                int(env.model.vis.global_.offwidth), args.video_width
            )
            env.model.vis.global_.offheight = max(
                int(env.model.vis.global_.offheight), args.video_height
            )
            renderer = mujoco.Renderer(
                env.model, height=args.video_height, width=args.video_width
            )
            mjb_path = output_dir / "model.mjb"
            mujoco.mj_saveModel(env.model, str(mjb_path), None)
            state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
            state_size = mujoco.mj_stateSize(env.model, state_spec)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [0.55, 0.0, 0.30]
            camera.distance = 1.65
            camera.azimuth = 135
            camera.elevation = -25

        with csv_path.open("w", encoding="utf-8", newline="") as csv_stream:
            csv_writer = csv.DictWriter(csv_stream, fieldnames=EPISODE_FIELDS)
            csv_writer.writeheader()
            csv_stream.flush()

            for episode in range(1, args.episodes + 1):
                episode_seed = args.seed + episode - 1
                observation, info = env.reset(seed=episode_seed)
                initial_info = dict(info)
                diagnostics = EpisodeDiagnostics()
                diagnostics.observe_info(info, float(env.data.time), env)
                episode_return = 0.0
                steps = 0
                episode_wall_start = time.perf_counter()
                terminated = False
                truncated = False

                video_path = output_dir / f"episode_{episode:03d}.mp4"
                states_path = output_dir / f"episode_{episode:03d}_states.npz"
                recorded_states: list[np.ndarray] = []
                frame_times: list[float] = []
                recorded_actions: list[np.ndarray] = []
                recorded_rewards: list[float] = []
                action_times: list[float] = []
                next_frame_time = 0.0
                video_writer = None

                if not args.no_video:
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.video_fps,
                        (args.video_width, args.video_height),
                    )
                    if not video_writer.isOpened():
                        video_writer.release()
                        raise RuntimeError(f"无法创建视频文件: {video_path}")

                def write_frame() -> None:
                    if args.no_video:
                        return
                    assert state_size is not None
                    assert state_spec is not None
                    assert renderer is not None
                    assert camera is not None
                    assert video_writer is not None
                    state = np.empty(state_size, dtype=np.float64)
                    mujoco.mj_getState(env.model, env.data, state, state_spec)
                    recorded_states.append(state)
                    frame_times.append(float(env.data.time))
                    renderer.update_scene(env.data, camera=camera)
                    rgb = renderer.render()
                    video_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                if not args.no_video:
                    write_frame()
                    next_frame_time += 1.0 / args.video_fps

                try:
                    while True:
                        predict_start = time.perf_counter()
                        action, _ = model.predict(
                            observation, deterministic=not args.stochastic
                        )
                        inference_seconds = time.perf_counter() - predict_start
                        action = np.asarray(action, dtype=np.float32).copy()
                        diagnostics.observe_action(
                            action,
                            float(env.data.time),
                            inference_seconds,
                            env.rl_config.gripper_open_threshold,
                        )
                        observation, reward, terminated, truncated, info = env.step(action)
                        diagnostics.observe_info(info, float(env.data.time), env)
                        if not args.no_video:
                            recorded_actions.append(action)
                            recorded_rewards.append(float(reward))
                            action_times.append(float(env.data.time))
                        episode_return += float(reward)
                        steps += 1
                        if (
                            not args.no_video
                            and env.data.time + 1e-9 >= next_frame_time
                        ):
                            write_frame()
                            next_frame_time += 1.0 / args.video_fps
                        if terminated or truncated:
                            break
                    if (
                        not args.no_video
                        and (not frame_times or env.data.time - frame_times[-1] > 1e-9)
                    ):
                        write_frame()
                finally:
                    if video_writer is not None:
                        video_writer.release()

                success = bool(info.get("success", terminated))
                failure_type = classify_failure(success, diagnostics)
                task_success = bool(info.get("base_success", False))
                task_failure_type = classify_task_outcome(
                    task_success=task_success,
                    ever_bilateral_candidate=bool(
                        info.get("ever_bilateral_candidate", False)
                    ),
                    ever_confirmed_grasp=bool(
                        info.get("ever_confirmed_grasp", False)
                    ),
                    break_events=env.base_env.grasp_break_history,
                )
                outcomes[failure_type] += 1
                task_outcomes[task_failure_type] += 1
                success_mismatches += int(success != task_success)
                episodes_with_physical_slip += int(
                    diagnostics.ever_physical_slip_event
                )
                successful_episodes_with_physical_slip += int(
                    success and diagnostics.ever_physical_slip_event
                )
                task_success_without_confirmed_grasp += int(
                    task_success
                    and not bool(info.get("ever_confirmed_grasp", False))
                )
                returns.append(episode_return)
                lengths.append(steps)
                completed_episodes += 1

                row = _episode_row(
                    episode=episode,
                    episode_seed=episode_seed,
                    success=success,
                    failure_type=failure_type,
                    task_success=task_success,
                    task_failure_type=task_failure_type,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    episode_return=episode_return,
                    steps=steps,
                    sim_time=float(env.data.time),
                    wall_time=time.perf_counter() - episode_wall_start,
                    initial_info=initial_info,
                    final_info=info,
                    diagnostics=diagnostics,
                    break_events=env.base_env.grasp_break_history,
                )
                csv_writer.writerow(row)
                csv_stream.flush()

                if not args.no_video:
                    assert state_spec is not None
                    assert mjb_path is not None
                    result = "success" if success else "truncated"
                    action_array = (
                        np.stack(recorded_actions)
                        if recorded_actions
                        else np.empty((0, env.action_space.shape[0]), dtype=np.float32)
                    )
                    np.savez_compressed(
                        states_path,
                        states=np.stack(recorded_states),
                        state_spec=np.int64(int(state_spec)),
                        frame_times=np.asarray(frame_times, dtype=np.float64),
                        fps=np.float64(args.video_fps),
                        width=np.int64(args.video_width),
                        height=np.int64(args.video_height),
                        model_file=np.asarray(mjb_path.name),
                        source_xml=np.asarray(str(XML_PATH.resolve())),
                        policy_file=np.asarray(str(args.model.resolve())),
                        mujoco_version=np.asarray(mujoco.__version__),
                        episode=np.int64(episode),
                        seed=np.int64(episode_seed),
                        result=np.asarray(result),
                        episode_return=np.float64(episode_return),
                        actions=action_array,
                        rewards=np.asarray(recorded_rewards, dtype=np.float64),
                        action_times=np.asarray(action_times, dtype=np.float64),
                        stochastic=np.bool_(args.stochastic),
                    )

                print_episode(episode, episode_return, steps, info)
                print(
                    f"  policy_outcome={failure_type} "
                    f"task_outcome={task_failure_type}",
                    flush=True,
                )
                if not args.no_video:
                    print(f"  video={video_path.resolve()}", flush=True)
                    print(f"  states={states_path.resolve()}", flush=True)

        successes = outcomes["success"]
        task_successes = task_outcomes["success"]
        failure_counts = {
            outcome: outcomes[outcome]
            for outcome in FAILURE_TYPES
            if outcome != "success"
        }
        total_failures = args.episodes - successes
        physical_slip_failures = outcomes["physical_slip_after_secured"]
        ambiguous_contact_loss_failures = outcomes[
            "open_during_contact_loss_after_secured"
        ]
        physical_slip_failure_fraction = (
            physical_slip_failures / total_failures
            if total_failures > 0
            else None
        )
        contact_loss_related_failure_fraction = (
            (physical_slip_failures + ambiguous_contact_loss_failures)
            / total_failures
            if total_failures > 0
            else None
        )
        task_failure_counts = {
            outcome: task_outcomes[outcome]
            for outcome in TASK_OUTCOME_TYPES
            if outcome != "success"
        }
        task_failures = args.episodes - task_successes
        task_physical_slip_failures = task_outcomes[
            "physical_slip_after_confirmed_grasp"
        ]
        task_ambiguous_contact_loss_failures = task_outcomes[
            "open_during_contact_loss_after_confirmed_grasp"
        ]
        task_contact_loss_fraction = (
            (
                task_physical_slip_failures
                + task_ambiguous_contact_loss_failures
            ) / task_failures
            if task_failures > 0
            else None
        )
        task_physical_slip_fraction = (
            task_physical_slip_failures / task_failures
            if task_failures > 0
            else None
        )
        summary = {
            "episodes": args.episodes,
            "policy_successes": successes,
            "policy_success_rate": successes / args.episodes,
            "successes": successes,
            "success_rate": successes / args.episodes,
            "task_successes": task_successes,
            "task_success_rate": task_successes / args.episodes,
            "task_policy_success_mismatches": success_mismatches,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_steps": float(np.mean(lengths)),
            "policy_outcome_counts": dict(outcomes),
            "policy_failure_counts": failure_counts,
            "task_outcome_counts": dict(task_outcomes),
            "task_failure_counts": task_failure_counts,
            "policy_physical_slip_failures": physical_slip_failures,
            "task_physical_slip_failures": task_physical_slip_failures,
            "policy_failures": total_failures,
            "task_failures": task_failures,
            "episodes_with_secured_physical_slip": episodes_with_physical_slip,
            "successful_episodes_with_secured_physical_slip": (
                successful_episodes_with_physical_slip
            ),
            "task_success_without_confirmed_grasp": (
                task_success_without_confirmed_grasp
            ),
            "physical_slip_fraction_of_policy_failures": (
                physical_slip_failure_fraction
            ),
            "contact_loss_related_fraction_of_policy_failures": (
                contact_loss_related_failure_fraction
            ),
            "contact_loss_related_fraction_of_task_failures": (
                task_contact_loss_fraction
            ),
            "physical_slip_fraction_of_task_failures": (
                task_physical_slip_fraction
            ),
            "physical_slip_dominates_policy_failures": (
                bool(physical_slip_failure_fraction > 0.5)
                if physical_slip_failure_fraction is not None
                else None
            ),
            "physical_slip_dominates_task_failures": (
                bool(task_physical_slip_fraction > 0.5)
                if task_physical_slip_fraction is not None
                else None
            ),
            "contact_loss_related_dominates_policy_failures": (
                bool(contact_loss_related_failure_fraction > 0.5)
                if contact_loss_related_failure_fraction is not None
                else None
            ),
            "contact_loss_related_dominates_task_failures": (
                bool(task_contact_loss_fraction > 0.5)
                if task_contact_loss_fraction is not None
                else None
            ),
        }
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = summary
        _write_json(manifest_path, manifest)

        print(
            f"episodes={args.episodes} policy_successes={successes} "
            f"policy_success_rate={successes / args.episodes:.1%} "
            f"task_successes={task_successes} "
            f"task_success_rate={task_successes / args.episodes:.1%} "
            f"mean_return={np.mean(returns):.3f} mean_steps={np.mean(lengths):.1f}",
            flush=True,
        )
        print(
            "policy_failure_counts="
            + ",".join(f"{name}:{count}" for name, count in failure_counts.items()),
            flush=True,
        )
        print(
            "contact_loss_related_fraction_of_policy_failures="
            + (
                "n/a"
                if contact_loss_related_failure_fraction is None
                else f"{contact_loss_related_failure_fraction:.1%}"
            ),
            flush=True,
        )
        print(f"episodes_csv={csv_path.resolve()}", flush=True)
        print(f"run_manifest={manifest_path.resolve()}", flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_episodes"] = completed_episodes
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        _write_json(manifest_path, manifest)
        raise
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def run_viewer(args: argparse.Namespace, model: PPO) -> None:
    from mujoco import viewer

    env = RLCableGraspEnv(
        seed=args.seed,
        disturbance_strength=args.disturbance,
        episode_seconds=args.episode_seconds,
    )
    observation, info = env.reset(seed=args.seed)
    episode = 1
    episode_return = 0.0
    steps = 0
    completed = 0
    wall_anchor = time.perf_counter()
    sim_anchor = float(env.data.time)
    reset_at: float | None = None

    print("RL viewer: Space由MuJoCo查看器暂停；关闭窗口结束测试", flush=True)
    with viewer.launch_passive(
        env.model, env.data, show_left_ui=False, show_right_ui=False
    ) as handle:
        handle.cam.lookat[:] = [0.55, 0.0, 0.30]
        handle.cam.distance = 1.65
        handle.cam.azimuth = 135
        handle.cam.elevation = -25
        handle.sync()

        while handle.is_running():
            frame_start = time.perf_counter()
            now = frame_start

            if reset_at is not None and now >= reset_at:
                completed += 1
                if completed >= args.episodes:
                    reset_at = None
                else:
                    episode += 1
                    observation, info = env.reset(seed=args.seed + episode - 1)
                    episode_return = 0.0
                    steps = 0
                    wall_anchor = now
                    sim_anchor = float(env.data.time)
                    reset_at = None

            if reset_at is None and completed < args.episodes:
                target_sim_time = sim_anchor + args.speed * (now - wall_anchor)
                advances = 0
                while env.data.time < target_sim_time and advances < 8:
                    action, _ = model.predict(
                        observation, deterministic=not args.stochastic
                    )
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                    steps += 1
                    advances += 1
                    if terminated or truncated:
                        print_episode(episode, episode_return, steps, info)
                        reset_at = now + 1.0
                        break
                if advances >= 8:
                    wall_anchor = now
                    sim_anchor = float(env.data.time)

            handle.sync()
            remaining = 1.0 / 60.0 - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained cable-grasp PPO policy")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20270804)
    parser.add_argument("--disturbance", type=float, default=1.5)
    parser.add_argument("--episode-seconds", type=float, default=28.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video-dir", type=Path, default=Path("rl_test_videos"),
                        help="headless结果根目录；每次测试建立独立子目录")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="headless批量评估时跳过renderer、MP4和state录制，仅写CSV和manifest",
    )
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    args = parser.parse_args()
    args.speed = float(np.clip(args.speed, 0.25, 8.0))
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.video_fps <= 0.0:
        parser.error("--video-fps must be greater than zero")
    if args.video_width <= 0 or args.video_height <= 0:
        parser.error("--video-width and --video-height must be greater than zero")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    arguments.model = _resolve_model_file(arguments.model)
    set_random_seed(arguments.seed)
    policy = PPO.load(arguments.model, device=arguments.device)
    if arguments.headless:
        run_headless(arguments, policy)
    else:
        run_viewer(arguments, policy)
