"""在冻结场景和相同 seed 上统一评估 scripted、expert 与 PPO。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from ..runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import mujoco
import numpy as np

from ..env.environment import (
    CableGraspEnv,
    EnvConfig,
    PANDA_XML_PATH,
    ROBOT_SPECS,
    XML_PATH,
    resolve_menagerie_panda_dir,
)
from ..policies.scripted import DynamicCableGraspPolicy, PolicyConfig
from ..scenarios.registry import (
    SCENARIO_SUITE_NAMES,
    ScenarioConfig,
    get_scenario,
    list_scenario_names,
    list_suite_scenarios,
)
from .failure_taxonomy import (
    TASK_OUTCOME_TYPES,
    base_scene_fingerprint,
    classify_task_outcome,
    confirmed_break_times,
    scene_fingerprint,
)
from .motion_diagnostics import env_config_for_scenario
from .defaults import DEFAULT_EVALUATION_SEED, DEFAULT_VIDEO_FPS
from .recording import EpisodeRecorder, create_unique_run_dir
from ..paths import output_path


ROOT = Path(__file__).resolve().parents[3]
_PPO_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _legacy_config(seed: int, disturbance: float, seconds: float) -> EnvConfig:
    return EnvConfig(
        seed=seed,
        disturbance_strength=disturbance,
        episode_seconds=seconds,
    )


def _scenario_config(
    scenario: ScenarioConfig | None,
    *,
    seed: int,
    disturbance: float,
    seconds: float,
    robot: str = "panda",
) -> EnvConfig:
    if scenario is None:
        return replace(
            _legacy_config(seed, disturbance, seconds), robot=robot,
        )
    # OOD range scenarios are realized independently per paired episode.  The
    # scenario identity remains stable while concrete physical values are
    # sampled deterministically from the episode seed.
    scenario = scenario.sample_for_episode(seed)
    return replace(env_config_for_scenario(
        scenario,
        seed=seed,
        episode_seconds=seconds,
    ), robot=robot)


def _recordable_config(config: EnvConfig, enabled: bool) -> EnvConfig:
    """Compile the standard camera rig whenever common artifacts are requested."""

    if config.dynamicvla_cameras_enabled == enabled:
        return config
    return replace(config, dynamicvla_cameras_enabled=enabled)


def _scripted_policy_config(options: Any) -> PolicyConfig:
    """Build the scripted policy config, keeping horizon overrides optional."""

    def option(name: str, default: Any = None) -> Any:
        if isinstance(options, dict):
            return options.get(name, default)
        return getattr(options, name, default)

    kwargs: dict[str, Any] = {
        "strict_vertical_gripper": bool(option("strict_vertical_gripper", False)),
    }
    for name in ("prediction_horizon", "approach_prediction_horizon"):
        value = option(name)
        if value is not None:
            kwargs[name] = float(value)
    return PolicyConfig(**kwargs)


def _apply_diagnostic_arm_gain_scale(env: CableGraspEnv, scale: float) -> None:
    """Temporarily scale NERO arm actuator gains for causal diagnostics.

    This is deliberately an evaluation-only intervention.  The XML remains the
    source of the default gains, and the default scale is exactly 1.0.
    """

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("diagnostic arm gain scale must be finite and positive")
    if scale == 1.0:
        return
    arm_actuators = slice(0, 7)
    env.model.actuator_gainprm[arm_actuators, 0] *= scale
    env.model.actuator_biasprm[arm_actuators, 1] *= scale
    env.model.actuator_biasprm[arm_actuators, 2] *= scale


def _config_for_method(config: EnvConfig, method: str) -> EnvConfig:
    """Use one target node for every method in a paired benchmark.

    The scripted baseline historically forced the middle node while expert and
    learned methods inherited the environment's random target.  That made the
    final paired-scene fingerprint check compare different targets under the
    same seed.  Keep the baseline's deterministic middle-node convention and
    apply it consistently to all methods.
    """

    del method
    return replace(config, target_selection="middle")


def _base_row(
    method: str,
    episode: int,
    seed: int,
    scenario: ScenarioConfig | None,
    initial_info: dict[str, Any],
    info: dict[str, Any],
    break_history: list[dict[str, Any]],
) -> dict[str, Any]:
    actual_seed = initial_info.get("episode_seed")
    if actual_seed != seed:
        raise RuntimeError(
            f"episode seed mismatch for {method}: requested={seed}, actual={actual_seed}"
        )
    task_success = bool(info.get("base_success", info.get("success", False)))
    ever_candidate = bool(info.get("ever_bilateral_candidate", False))
    ever_confirmed = bool(info.get("ever_confirmed_grasp", False))
    first_breaks = confirmed_break_times(break_history)
    scenario_name = initial_info.get("scenario_name", "legacy_shape_current")
    scenario_id = initial_info.get("scenario_id") or "legacy-shape-current"
    row = {
        "method": method,
        "episode": episode,
        "scenario_episode_id": f"{scenario_id}:seed-{seed}",
        "scenario_name": scenario_name,
        "scenario_id": scenario_id,
        "scenario_split": initial_info.get("scenario_split", "legacy"),
        "motion_type": initial_info.get("motion_mode", "shape"),
        "motion_profile_version": initial_info.get(
            "motion_profile_version", "legacy_v1"
        ),
        "motion_regularity": initial_info.get(
            "motion_regularity", "quasiperiodic"
        ),
        "amplitude_level": (
            None if scenario is None else scenario.amplitude_level.value
        ),
        "frequency_level": (
            None if scenario is None else scenario.frequency_level.value
        ),
        "disturbance_strength": float(
            initial_info.get("disturbance_strength", 1.5)
        ),
        "motion_frequency_scale": float(
            initial_info.get("motion_frequency_scale", 1.0)
        ),
        "shape_motion_scale": float(initial_info.get("shape_motion_scale", 1.0)),
        "motion_profile_hash": initial_info.get("motion_profile_hash"),
        "rigid_motion_duration": initial_info.get("rigid_motion_duration"),
        "rigid_motion_exit_y": initial_info.get("rigid_motion_exit_y"),
        "rigid_motion_control": initial_info.get("rigid_motion_control"),
        "rigid_path_position_gain": initial_info.get("rigid_path_position_gain"),
        "rigid_velocity_gain": initial_info.get("rigid_velocity_gain"),
        "rigid_translation_max_acceleration": initial_info.get(
            "rigid_translation_max_acceleration"
        ),
        "rigid_motion_com_y": info.get("rigid_motion_com_y"),
        "rigid_motion_nominal_finished": bool(
            info.get("rigid_motion_nominal_finished", False)
        ),
        "rigid_motion_finished": bool(info.get("rigid_motion_finished", False)),
        "rigid_motion_suspended": bool(
            info.get("rigid_motion_suspended", False)
        ),
        "rigid_motion_released": bool(info.get("rigid_motion_released", False)),
        "termination_reason": info.get("termination_reason"),
        "cable_length_scale": float(initial_info.get("cable_length_scale", 1.0)),
        "cable_density_scale": float(initial_info.get("cable_density_scale", 1.0)),
        "cable_stiffness_scale": float(
            initial_info.get("cable_stiffness_scale", 1.0)
        ),
        "cable_damping_scale": float(
            initial_info.get("cable_damping_scale", 1.0)
        ),
        "cable_friction_scale": float(
            initial_info.get("cable_friction_scale", 1.0)
        ),
        "robot_motion_limit_profile": initial_info.get(
            "robot_motion_limit_profile", "unspecified"
        ),
        "arm_joint_velocity_limits": json.dumps(
            np.asarray(initial_info.get(
                "arm_joint_velocity_limits", [],
            )).tolist()
        ),
        "arm_acceleration_limit_enabled": bool(initial_info.get(
            "arm_acceleration_limit_enabled", False
        )),
        "arm_joint_acceleration_limits": json.dumps(
            np.asarray(initial_info.get(
                "arm_joint_acceleration_limits", [],
            )).tolist()
        ),
        "hand_cartesian_velocity_limit_enabled": bool(initial_info.get(
            "hand_cartesian_velocity_limit_enabled", False
        )),
        "hand_linear_velocity_limit": float(initial_info.get(
            "hand_linear_velocity_limit", np.nan
        )),
        "hand_angular_velocity_limit": float(initial_info.get(
            "hand_angular_velocity_limit", np.nan
        )),
        "gripper_finger_velocity_limit": float(initial_info.get(
            "gripper_finger_velocity_limit", np.nan
        )),
        "low_level_velocity_guard_fraction": float(initial_info.get(
            "low_level_velocity_guard_fraction", np.nan
        )),
        "cable_length_ood": bool(scenario and scenario.cable_length_ood),
        "cable_material_profile": (
            "nominal" if scenario is None else scenario.cable_material_profile
        ),
        "cable_material_ood": bool(scenario and scenario.cable_material_ood),
        "ood_factor": None if scenario is None else scenario.ood_factor,
        "ood_level": None if scenario is None else scenario.ood_level,
        "ood_factor_level": (
            None
            if scenario is None or scenario.ood_factor is None
            else f"{scenario.ood_factor}_{scenario.ood_level}"
        ),
        "requested_seed": seed,
        "actual_episode_seed": actual_seed,
        "base_scene_fingerprint": base_scene_fingerprint(initial_info),
        "scene_fingerprint": scene_fingerprint(initial_info),
        "task_success": task_success,
        "policy_internal_success": bool(
            info.get("policy_internal_success", info.get("success", False))
        ),
        "ever_pinched": bool(
            info.get("ever_pinched", info.get("grasped_body_id") is not None)
        ),
        "ever_secured": bool(info.get("ever_grasped", False)),
        "ever_bilateral_candidate": ever_candidate,
        "ever_confirmed_grasp": ever_confirmed,
        "target_body_id": int(initial_info["target_body_id"]),
        "grasped_body_id": info.get("grasped_body_id"),
        "initial_cable_dx": float(initial_info.get("initial_cable_dx", np.nan)),
        "initial_cable_dy": float(initial_info.get("initial_cable_dy", np.nan)),
        "disturbance_phase": float(initial_info.get("disturbance_phase", np.nan)),
        "disturbance_spatial_phase": float(
            initial_info.get("disturbance_spatial_phase", np.nan)
        ),
        "active_open_break_count": int(info.get("active_open_break_count", 0)),
        "physical_slip_break_count": int(info.get("physical_slip_break_count", 0)),
        "active_open_after_secured_count": int(
            info.get("active_open_after_secured_count", 0)
        ),
        "physical_slip_after_secured_count": int(
            info.get("physical_slip_after_secured_count", 0)
        ),
        "active_open_after_confirmed_count": int(
            info.get("active_open_after_confirmed_count", 0)
        ),
        "physical_slip_after_confirmed_count": int(
            info.get("physical_slip_after_confirmed_count", 0)
        ),
        "open_during_contact_loss_after_confirmed_count": int(
            info.get("open_during_contact_loss_after_confirmed_count", 0)
        ),
        "first_active_open_after_confirmed_time": first_breaks.get("active_open"),
        "first_physical_slip_after_confirmed_time": first_breaks.get("physical_slip"),
        "first_open_during_contact_loss_after_confirmed_time": first_breaks.get(
            "open_during_contact_loss"
        ),
        "last_grasp_break_reason": info.get("last_grasp_break_reason"),
        "last_grasp_break_causal_class": info.get(
            "last_grasp_break_causal_class"
        ),
        "lifted_fraction": float(info.get("lifted_fraction", 0.0)),
        "max_z": float(info.get("max_z", 0.0)),
        "success_hold": float(
            info.get("strict_success_hold", info.get("success_hold", 0.0))
        ),
        "motion_limit_active_ratio": float(info.get(
            "motion_limit_active_ratio", 0.0
        )),
        "acceleration_limit_ratio": float(info.get(
            "acceleration_limit_ratio", 0.0
        )),
        "joint_velocity_limit_ratio": float(info.get(
            "joint_velocity_limit_ratio", 0.0
        )),
        "cartesian_velocity_limit_ratio": float(info.get(
            "cartesian_velocity_limit_ratio", 0.0
        )),
        "gripper_velocity_limit_ratio": float(info.get(
            "gripper_velocity_limit_ratio", 0.0
        )),
        "low_level_velocity_guard_ratio": float(info.get(
            "low_level_velocity_guard_ratio", 0.0
        )),
        "actual_joint_velocity_exceedance_ratio": float(info.get(
            "actual_joint_velocity_exceedance_ratio", 0.0
        )),
        "max_abs_actual_arm_velocity": json.dumps(
            np.asarray(info.get("max_abs_actual_arm_velocity", [])).tolist()
        ),
        "max_abs_pre_limit_arm_velocity": json.dumps(
            np.asarray(info.get("max_abs_pre_limit_arm_velocity", [])).tolist()
        ),
        "max_actual_hand_linear_speed": float(info.get(
            "max_actual_hand_linear_speed", 0.0
        )),
        "max_actual_hand_angular_speed": float(info.get(
            "max_actual_hand_angular_speed", 0.0
        )),
        "max_pre_limit_hand_linear_speed": float(info.get(
            "max_pre_limit_hand_linear_speed", 0.0
        )),
        "max_pre_limit_hand_angular_speed": float(info.get(
            "max_pre_limit_hand_angular_speed", 0.0
        )),
        "physics_velocity_limiter_ratio": float(info.get(
            "physics_velocity_limiter_ratio", 0.0
        )),
        "physics_velocity_fence_ratio": float(info.get(
            "physics_velocity_fence_ratio", 0.0
        )),
        "physics_velocity_fence_dof_steps": int(info.get(
            "physics_velocity_fence_dof_steps", 0
        )),
    }
    row["task_failure_type"] = classify_task_outcome(
        task_success=task_success,
        ever_bilateral_candidate=ever_candidate,
        ever_confirmed_grasp=ever_confirmed,
        break_events=break_history,
    )
    return row


def _run_scripted(
    seeds: list[int],
    scenario: ScenarioConfig | None,
    disturbance: float,
    episode_seconds: float,
    robot: str = "panda",
) -> list[dict[str, Any]]:
    config = _scenario_config(
        scenario, seed=seeds[0], disturbance=disturbance, seconds=episode_seconds,
        robot=robot,
    )
    config = _config_for_method(config, "scripted")
    env = CableGraspEnv(config)
    policy = DynamicCableGraspPolicy(env)
    rows: list[dict[str, Any]] = []
    for episode, seed in enumerate(seeds, start=1):
        _, initial_info = env.reset(seed=seed)
        policy.reset()
        min_target_distance = float("inf")
        termination_reason: str | None = None
        while not policy.finished and env.data.time < env.config.episode_seconds:
            action = policy.action()
            _, _, _, truncated, step_info = env.step(action)
            min_target_distance = min(
                min_target_distance,
                float(np.linalg.norm(env.target_position() - env.hand_position)),
            )
            if truncated:
                termination_reason = step_info.get("termination_reason")
                policy.result = (
                    "failed_motion_boundary"
                    if termination_reason == "rigid_motion_boundary_crossed"
                    else "failed_timeout"
                )
                policy.finished = True
        info = env.info()
        info["ever_pinched"] = env.last_grasped_body_id is not None
        info["base_success"] = env.ever_success
        info["success"] = policy.result == "success"
        row = _base_row(
            "scripted", episode, seed, scenario, initial_info, info,
            env.grasp_break_history,
        )
        row.update({
            "steps": int(round(env.data.time / (
                env.model.opt.timestep * max(1, env.config.frame_skip)
            ))),
            "sim_time": float(env.data.time),
            "episode_return": np.nan,
            "min_target_distance": min_target_distance,
            "policy_result": policy.result,
            "terminated": env.ever_success,
            "truncated": termination_reason is not None,
        })
        rows.append(row)
    return rows


def _run_scripted_episode(
    episode: int,
    seed: int,
    scenario: ScenarioConfig | None,
    disturbance: float,
    episode_seconds: float,
    robot: str = "panda",
) -> dict[str, Any]:
    """Run one isolated scripted episode for parallel matrix evaluation."""

    row = _run_scripted(
        [seed], scenario, disturbance, episode_seconds, robot,
    )[0]
    row["episode"] = episode
    return row


def _run_ppo(
    seeds: list[int],
    scenario: ScenarioConfig | None,
    disturbance: float,
    episode_seconds: float,
    model_path: Path,
    device: str,
    robot: str = "panda",
) -> list[dict[str, Any]]:
    from stable_baselines3 import PPO
    from ..rl.environment import RLCableGraspEnv

    config = _scenario_config(
        scenario, seed=seeds[0], disturbance=disturbance, seconds=episode_seconds,
        robot=robot,
    )
    env = RLCableGraspEnv(env_config=config)
    model = PPO.load(model_path, device=device)
    rows: list[dict[str, Any]] = []
    try:
        for episode, seed in enumerate(seeds, start=1):
            observation, info = env.reset(seed=seed)
            initial_info = dict(info)
            episode_return = 0.0
            steps = 0
            min_target_distance = float(info["target_distance"])
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                steps += 1
                min_target_distance = min(
                    min_target_distance, float(info["target_distance"])
                )
                if terminated or truncated:
                    break
            row = _base_row(
                "ppo", episode, seed, scenario, initial_info, info,
                env.base_env.grasp_break_history,
            )
            row.update({
                "steps": steps,
                "sim_time": float(env.data.time),
                "episode_return": episode_return,
                "min_target_distance": min_target_distance,
                "policy_result": "success" if terminated else "truncated",
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            rows.append(row)
    finally:
        env.close()
    return rows


def _cached_ppo_model(model_path: Path, device: str) -> Any:
    """Load a PPO checkpoint once per worker process."""

    key = (str(model_path.resolve()), device)
    model = _PPO_MODEL_CACHE.get(key)
    if model is None:
        from stable_baselines3 import PPO

        model = PPO.load(model_path, device=device)
        _PPO_MODEL_CACHE[key] = model
    return model


def _episode_directory(
    output_dir: Path,
    method: str,
    scenario_name: str,
    seed: int,
) -> Path:
    return output_dir / "episodes" / method / scenario_name / f"seed_{seed}"


def _run_policy_episode(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one isolated episode; safe as a ProcessPool worker target."""

    method = str(job["method"])
    episode = int(job["episode"])
    seed = int(job["seed"])
    scenario = job["scenario"]
    recording = bool(job["recording"])
    config = _scenario_config(
        scenario,
        seed=seed,
        disturbance=float(job["disturbance"]),
        seconds=float(job["episode_seconds"]),
        robot=str(job.get("robot", "panda")),
    )
    config = _config_for_method(config, method)
    config = _recordable_config(config, recording)

    if method == "diffusion_policy":
        # Diffusion Policy consumes the same camera rig as DynamicVLA even when
        # benchmark recordings are disabled.
        from ..diffusion_policy.runner import DiffusionPolicyRunner

        env = CableGraspEnv(replace(config, dynamicvla_cameras_enabled=True))
        base_env = env
        policy = DiffusionPolicyRunner(
            env,
            Path(job["diffusion_policy_model"]),
            device=str(job["device"]),
            deterministic=True,
        )
    elif method == "ppo":
        from ..rl.environment import RLCableGraspEnv

        env: Any = RLCableGraspEnv(env_config=config)
        base_env = env.base_env
        policy = _cached_ppo_model(Path(job["ppo_model"]), str(job["device"]))
    else:
        env = CableGraspEnv(config)
        base_env = env
        if method == "scripted":
            _apply_diagnostic_arm_gain_scale(
                env,
                float(job.get("arm_actuator_gain_scale", 1.0)),
            )
            policy = DynamicCableGraspPolicy(
                env,
                _scripted_policy_config(job),
            )
        elif method == "expert":
            from ..expert.formula_intercept_policy import (
                FormulaInterceptConfig,
                FormulaInterceptExpert,
            )

            policy = FormulaInterceptExpert(
                env,
                FormulaInterceptConfig(
                    strict_vertical_gripper=bool(job["strict_vertical_gripper"])
                ),
            )
        else:  # Defensive: argparse normally prevents this path.
            raise ValueError(f"unsupported evaluation method: {method}")

    recorder: EpisodeRecorder | None = None
    try:
        observation, initial_info = env.reset(seed=seed)
        if method != "ppo":
            if method == "diffusion_policy":
                policy.reset(seed=seed)
            else:
                policy.reset()
        if recording:
            recorder = EpisodeRecorder(
                env,
                _episode_directory(
                    Path(job["output_dir"]), method,
                    str(initial_info.get("scenario_name", config.scenario_name)), seed,
                ),
                video_fps=float(job["video_fps"]),
            )
            recorder.capture_initial()

        episode_return = 0.0
        steps = 0
        min_target_distance = float(initial_info.get("target_distance", np.inf))
        terminated = False
        truncated = False
        info = dict(initial_info)
        while True:
            if method == "ppo":
                action, _ = policy.predict(observation, deterministic=True)
            else:
                if policy.finished:
                    break
                action = policy.action()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            steps += 1
            target_distance = info.get("target_distance")
            if target_distance is None:
                target_distance = np.linalg.norm(
                    base_env.target_position() - base_env.hand_position
                )
            min_target_distance = min(min_target_distance, float(target_distance))
            if recorder is not None:
                recorder.record_step(action, reward, terminated, truncated, info)
            if terminated or truncated:
                break

        if method != "ppo":
            info = base_env.info()
            if base_env.ever_success and policy.result == "running":
                policy.result = "success"
            info["ever_pinched"] = base_env.last_grasped_body_id is not None
            info["base_success"] = base_env.ever_success
            info["success"] = policy.result == "success"
            terminated = bool(base_env.ever_success)
            if truncated and policy.result == "running":
                reason = info.get("termination_reason")
                policy.result = (
                    "failed_motion_boundary"
                    if reason == "rigid_motion_boundary_crossed"
                    else "failed_timeout"
                )
            policy_result = policy.result
        else:
            policy_result = "success" if terminated else "truncated"

        row = _base_row(
            method, episode, seed, scenario, initial_info, info,
            base_env.grasp_break_history,
        )
        row.update({
            "steps": steps,
            "sim_time": float(base_env.data.time),
            "episode_return": episode_return,
            "min_target_distance": min_target_distance,
            "policy_result": policy_result,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "compiled_model": str(job["compiled_model"]),
        })
        if method != "ppo":
            row.update(policy.policy_info())
        if recorder is not None:
            artifacts = recorder.finish(row)
            row.update(artifacts.relative_to(Path(job["output_dir"])))
        else:
            row.update({
                "episode_dir": None,
                "trajectory": None,
                "metadata": None,
                "global_video": None,
                "wrist_video": None,
            })
        return row
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _wilson_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(selected)
    successes = sum(bool(row["task_success"]) for row in selected)
    failures = total - successes
    outcomes = {outcome: 0 for outcome in TASK_OUTCOME_TYPES}
    for row in selected:
        outcomes[str(row["task_failure_type"])] += 1
    slip_failures = outcomes["physical_slip_after_confirmed_grasp"]
    success_low, success_high = _wilson_interval(successes, total, 1.959964)
    _, slip_upper = _wilson_interval(slip_failures, failures, 1.644854)
    return {
        "episodes": total,
        "task_successes": successes,
        "task_success_rate": successes / total if total else None,
        "task_success_wilson95_low": None if total == 0 else success_low,
        "task_success_wilson95_high": None if total == 0 else success_high,
        "policy_internal_successes": sum(
            bool(row["policy_internal_success"]) for row in selected
        ),
        "task_policy_success_mismatches": sum(
            bool(row["task_success"]) != bool(row["policy_internal_success"])
            for row in selected
        ),
        "outcome_counts": outcomes,
        "task_failures": failures,
        "task_physical_slip_failures": slip_failures,
        "physical_slip_fraction_of_task_failures": (
            slip_failures / failures if failures else None
        ),
        "physical_slip_fraction_one_sided95_upper": (
            None if failures == 0 else slip_upper
        ),
        "physical_slip_non_dominance_supported": (
            None if failures == 0 else slip_upper < 0.5
        ),
        "episodes_with_confirmed_physical_slip": sum(
            int(row.get("physical_slip_after_confirmed_count", 0)) > 0
            for row in selected
        ),
    }


def _group(
    selected: list[dict[str, Any]], key: str,
) -> dict[str, dict[str, Any]]:
    values = sorted({str(row[key]) for row in selected})
    return {
        value: _aggregate([row for row in selected if str(row[key]) == value])
        for value in values
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        by_scenario = _group(selected, "scenario_name")
        ood_rows = [row for row in selected if row.get("ood_factor")]
        scenario_rates = [
            float(summary["task_success_rate"])
            for summary in by_scenario.values()
            if summary["task_success_rate"] is not None
        ]
        result[method] = {
            "overall_micro": _aggregate(selected),
            "overall_macro_cell_equal_task_success_rate": (
                float(np.mean(scenario_rates)) if scenario_rates else None
            ),
            "by_scenario": by_scenario,
            "by_motion_type": _group(selected, "motion_type"),
            "by_split": _group(selected, "scenario_split"),
            # ``by_ood_factor`` deliberately pools low/high samples, e.g.
            # amplitude total success rate across both amplitude ranges.
            "by_ood_factor": (
                _group(ood_rows, "ood_factor") if ood_rows else {}
            ),
            "by_ood_factor_level": (
                _group(ood_rows, "ood_factor_level") if ood_rows else {}
            ),
        }
    return result


def _select_scenarios(args: argparse.Namespace) -> list[ScenarioConfig | None]:
    if args.scenario is not None:
        return [get_scenario(args.scenario)]
    if args.scenarios is not None:
        return [get_scenario(name) for name in args.scenarios]
    if args.suite is not None:
        return list(list_suite_scenarios(args.suite))
    return [None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired scenario-matrix benchmark for cable grasping"
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=("scripted", "expert", "ppo", "diffusion_policy"),
        default=("scripted",)
    )
    parser.add_argument("--ppo-model", type=Path)
    parser.add_argument("--diffusion-policy-model", type=Path)
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="paired repeats per scenario",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--disturbance", type=float, default=1.5)
    parser.add_argument(
        "--robot", choices=tuple(sorted(ROBOT_SPECS)), default="panda",
        help="robot model used by the MuJoCo environment",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scenario", choices=list_scenario_names())
    selection.add_argument(
        "--scenarios", nargs="+", choices=list_scenario_names(),
        help="explicit list of scenarios to evaluate",
    )
    selection.add_argument("--suite", choices=SCENARIO_SUITE_NAMES)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument(
        "--prediction-horizon", type=float, default=None,
        help="optional scripted-policy prediction horizon in seconds",
    )
    parser.add_argument(
        "--approach-prediction-horizon", type=float, default=None,
        help="optional scripted-policy APPROACH prediction horizon in seconds",
    )
    parser.add_argument(
        "--arm-actuator-gain-scale", type=float, default=1.0,
        help=(
            "diagnostic-only multiplicative scale for the first seven arm "
            "actuator gains; default 1.0 preserves the XML parameters"
        ),
    )
    parser.add_argument(
        "--scenario-workers", type=int, default=1,
        help="maximum number of scenario cells active at once",
    )
    parser.add_argument(
        "--envs-per-scenario", type=int, default=1,
        help="maximum concurrent episode environments per active scenario",
    )
    parser.add_argument(
        "--workers", type=int,
        help="optional global worker cap (defaults to their product)",
    )
    parser.add_argument(
        "--recording", action=argparse.BooleanOptionalAction, default=True,
        help="write FULLPHYSICS trajectories and global/wrist videos",
    )
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--strict-vertical-gripper",
        action="store_true",
        help=(
            "make the vertical grasp orientation the primary IK task and "
            "solve translation in its nullspace"
        ),
    )
    parser.add_argument("--output", type=Path, default=output_path("benchmarks"))
    args = parser.parse_args()
    args.methods = list(dict.fromkeys(args.methods))
    arm_actuator_gain_scale = getattr(args, "arm_actuator_gain_scale", 1.0)
    positive = (
        args.episodes >= 1
        and args.episode_seconds > 0.0
        and args.scenario_workers >= 1
        and args.envs_per_scenario >= 1
        and (args.workers is None or args.workers >= 1)
        and args.video_fps > 0.0
        and math.isfinite(arm_actuator_gain_scale)
        and arm_actuator_gain_scale > 0.0
    )
    if not positive:
        parser.error("episode, duration, FPS, and worker counts must be positive")
    if args.disturbance < 0.0 or not math.isfinite(args.disturbance):
        parser.error("--disturbance must be finite and non-negative")
    for name in ("prediction_horizon", "approach_prediction_horizon"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0.0):
            parser.error(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    if "ppo" in args.methods and args.ppo_model is None:
        parser.error("--ppo-model is required when evaluating PPO")
    if "diffusion_policy" in args.methods and args.diffusion_policy_model is None:
        parser.error(
            "--diffusion-policy-model is required when evaluating diffusion_policy"
        )
    if args.ppo_model is not None and not args.ppo_model.is_file():
        parser.error(f"PPO model not found: {args.ppo_model}")
    if (
        args.diffusion_policy_model is not None
        and not args.diffusion_policy_model.is_file()
    ):
        parser.error(
            f"Diffusion Policy model not found: {args.diffusion_policy_model}"
        )
    return args


def run_benchmark(args: argparse.Namespace) -> Path:
    """Run a benchmark from a validated argparse-compatible namespace."""

    scenarios = _select_scenarios(args)
    seeds = [args.seed + index for index in range(args.episodes)]
    strict_vertical_gripper = bool(
        getattr(args, "strict_vertical_gripper", False)
    )
    diffusion_policy_model = getattr(args, "diffusion_policy_model", None)
    requested_run_name = getattr(args, "run_name", None)
    if requested_run_name is None:
        output_dir = create_unique_run_dir(args.output)
    else:
        output_dir = Path(args.output) / requested_run_name
        output_dir.mkdir(parents=True, exist_ok=False)

    # Save the exact compiled model used by recorded FULLPHYSICS states before
    # launching workers. Camera bodies are part of the model when recording.
    models_dir = output_dir / "models"
    models_dir.mkdir()
    compiled_models: dict[str, Any] = {}
    model_paths: dict[str, str] = {}
    for scenario in scenarios:
        config = _recordable_config(_scenario_config(
            scenario, seed=args.seed, disturbance=args.disturbance,
            seconds=args.episode_seconds,
            robot=getattr(args, "robot", "panda"),
        ), args.recording or "diffusion_policy" in args.methods)
        model_env = CableGraspEnv(config)
        scenario_name = config.scenario_name
        model_path = models_dir / f"{scenario_name}.mjb"
        mujoco.mj_saveModel(model_env.model, str(model_path), None)
        model_env.close()
        model_paths[scenario_name] = str(model_path.relative_to(output_dir))
        compiled_models[scenario_name] = {
            "path": str(model_path.resolve()),
            "sha256": _sha256(model_path),
        }

    rows: list[dict[str, Any]] = []
    total_jobs = len(scenarios) * len(seeds) * len(args.methods)
    completed = 0
    for batch_start in range(0, len(scenarios), args.scenario_workers):
        scenario_batch = scenarios[
            batch_start:batch_start + args.scenario_workers
        ]
        per_scenario: list[list[dict[str, Any]]] = []
        for scenario in scenario_batch:
            config = _scenario_config(
                scenario, seed=args.seed, disturbance=args.disturbance,
                seconds=args.episode_seconds,
                robot=getattr(args, "robot", "panda"),
            )
            model_path = model_paths[config.scenario_name]
            per_scenario.append([
                {
                    "method": method,
                    "episode": episode,
                    "seed": seed,
                    "scenario": scenario,
                    "disturbance": args.disturbance,
                    "episode_seconds": args.episode_seconds,
                    "ppo_model": args.ppo_model,
                    "diffusion_policy_model": diffusion_policy_model,
                    "device": args.device,
                    "strict_vertical_gripper": strict_vertical_gripper,
                    "prediction_horizon": getattr(
                        args, "prediction_horizon", None,
                    ),
                    "approach_prediction_horizon": getattr(
                        args, "approach_prediction_horizon", None,
                    ),
                    "arm_actuator_gain_scale": getattr(
                        args, "arm_actuator_gain_scale", 1.0,
                    ),
                    "recording": args.recording,
                    "video_fps": args.video_fps,
                    "output_dir": output_dir,
                    "compiled_model": model_path,
                    "robot": getattr(args, "robot", "panda"),
                }
                for episode, seed in enumerate(seeds, start=1)
                for method in args.methods
            ])
        # Round-robin ordering prevents one scenario from monopolizing all
        # process slots and realizes the per-scenario concurrency limit.
        jobs = [
            scenario_jobs[index]
            for index in range(max(map(len, per_scenario)))
            for scenario_jobs in per_scenario
            if index < len(scenario_jobs)
        ]
        worker_limit = len(scenario_batch) * args.envs_per_scenario
        if args.workers is not None:
            worker_limit = min(worker_limit, args.workers)
        worker_limit = min(worker_limit, len(jobs))
        if worker_limit == 1:
            for job in jobs:
                rows.append(_run_policy_episode(job))
                completed += 1
                print(f"completed_episodes={completed}/{total_jobs}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=worker_limit) as executor:
                pending = {
                    executor.submit(_run_policy_episode, job): job for job in jobs
                }
                for future in as_completed(pending):
                    rows.append(future.result())
                    completed += 1
                    print(
                        f"completed_episodes={completed}/{total_jobs}", flush=True,
                    )
    rows.sort(key=lambda row: (
        str(row["scenario_name"]), int(row["episode"]), str(row["method"])
    ))

    fingerprints: dict[str, set[str]] = {}
    methods: dict[str, list[str]] = {}
    for row in rows:
        key = str(row["scenario_episode_id"])
        fingerprints.setdefault(key, set()).add(str(row["scene_fingerprint"]))
        methods.setdefault(key, []).append(str(row["method"]))
    mismatched = [key for key, values in fingerprints.items() if len(values) != 1]
    if mismatched:
        raise RuntimeError(f"paired scene fingerprint mismatch: {mismatched}")
    incomplete = [
        key for key, values in methods.items()
        if sorted(values) != sorted(args.methods)
    ]
    if incomplete:
        raise RuntimeError(f"missing or duplicate method rows: {incomplete}")
    paired_verification: bool | None = True if len(args.methods) > 1 else None
    for row in rows:
        row["paired_scene_match"] = paired_verification

    summary = _summary(rows)
    _write_csv(output_dir / "episodes.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    git_status = _git_text("status", "--porcelain=v1")
    manifest = {
        "schema_version": 4,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "arguments": {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "methods": args.methods,
        "suite": args.suite,
        "scenarios": [
            (
                {"name": "legacy_shape_current", "legacy": True}
                if scenario is None else scenario.asdict()
            )
            for scenario in scenarios
        ],
        "seeds_per_scenario": seeds,
        "episode_seconds": args.episode_seconds,
        "recording": {
            "enabled": args.recording,
            "schema_version": 1,
            "state_format": "mujoco_mjSTATE_FULLPHYSICS",
            "state_sampling": "initial_and_after_every_control_step",
            "video_fps": args.video_fps,
            "cameras": ["opst_cam", "wrist_cam"],
        },
        "parallelism": {
            "scenario_workers": args.scenario_workers,
            "envs_per_scenario": args.envs_per_scenario,
            "global_worker_cap": args.workers,
            "backend": "process",
        },
        "ppo_model": None if args.ppo_model is None else str(args.ppo_model.resolve()),
        "ppo_model_sha256": _sha256(args.ppo_model),
        "diffusion_policy_model": (
            None if diffusion_policy_model is None
            else str(diffusion_policy_model.resolve())
        ),
        "diffusion_policy_model_sha256": _sha256(diffusion_policy_model),
        "robot": getattr(args, "robot", "panda"),
        "source_xml": str(XML_PATH.resolve()),
        "source_xml_sha256": _sha256(XML_PATH),
        "robot_xml": str(
            ROBOT_SPECS[getattr(args, "robot", "panda")].xml_path.resolve()
        ),
        "robot_xml_sha256": _sha256(
            ROBOT_SPECS[getattr(args, "robot", "panda")].xml_path
        ),
        "panda_xml": str(PANDA_XML_PATH.resolve()),
        "panda_xml_sha256": _sha256(PANDA_XML_PATH),
        "menagerie_panda_assets": (
            str(resolve_menagerie_panda_dir())
            if getattr(args, "robot", "panda") == "panda" else None
        ),
        "compiled_models": compiled_models,
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in {
                "benchmark": Path(__file__),
                "recording": ROOT / "src" / "panda_cable_grasp" / "evaluation" / "recording.py",
                "base_environment": ROOT / "src" / "panda_cable_grasp" / "env" / "environment.py",
                "scenario_registry": ROOT / "src" / "panda_cable_grasp" / "scenarios" / "registry.py",
                "motion_diagnostics": ROOT / "src" / "panda_cable_grasp" / "evaluation" / "motion_diagnostics.py",
                "scripted_policy": ROOT / "src" / "panda_cable_grasp" / "policies" / "scripted.py",
                "failure_taxonomy": ROOT / "src" / "panda_cable_grasp" / "evaluation" / "failure_taxonomy.py",
                "rl_environment": ROOT / "src" / "panda_cable_grasp" / "rl" / "environment.py",
                "diffusion_policy": ROOT / "src" / "panda_cable_grasp" / "diffusion_policy" / "runner.py",
            }.items()
        },
        "configs": {
            "scripted_policy": asdict(_scripted_policy_config(args)),
            "diagnostic_arm_actuator_gain_scale": getattr(
                args, "arm_actuator_gain_scale", 1.0,
            ),
        },
        "task_outcome_types": list(TASK_OUTCOME_TYPES),
        "paired_scene_fingerprints_verified": paired_verification,
        "git_commit": _git_text("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "git_status_sha256": (
            None if git_status is None
            else hashlib.sha256(git_status.encode("utf-8")).hexdigest()
        ),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "stable_baselines3": _distribution_version("stable-baselines3"),
        "gymnasium": _distribution_version("gymnasium"),
        "torch": _distribution_version("torch"),
        "summary": summary,
    }
    if "ppo" in args.methods:
        from ..rl.environment import RLConfig
        manifest["configs"]["rl"] = asdict(RLConfig())
    if "expert" in args.methods:
        from ..expert.formula_intercept_policy import FormulaInterceptConfig
        manifest["configs"]["expert"] = asdict(FormulaInterceptConfig(
            strict_vertical_gripper=strict_vertical_gripper
        ))
    if "diffusion_policy" in args.methods:
        from ..diffusion_policy.config import DiffusionPolicyConfig
        manifest["configs"]["diffusion_policy"] = asdict(DiffusionPolicyConfig())
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"benchmark_output={output_dir.resolve()}", flush=True)
    return output_dir


def main() -> None:
    run_benchmark(parse_args())


# Public experiment helpers shared by benchmark and expert collectors.  The
# underscored names remain internal aliases for compatibility with old runs.
git_text = _git_text
distribution_version = _distribution_version
sha256_file = _sha256
base_row = _base_row
write_csv = _write_csv
summarize = _summary


if __name__ == "__main__":
    main()
