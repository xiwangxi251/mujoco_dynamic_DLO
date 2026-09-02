"""Evaluate a served OpenPI pi0.5 policy on the Panda cable benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import mujoco

from ..dynamicvla.adapter import DynamicVLAAdapterConfig, DynamicVLATaskSpaceAdapter
from ..env.environment import (
    CableGraspEnv,
    EnvConfig,
    PANDA_XML_PATH,
    ROBOT_SPECS,
    XML_PATH,
    resolve_menagerie_panda_dir,
)
from ..evaluation.benchmark import base_row, git_text, sha256_file, summarize, write_csv
from ..evaluation.defaults import DEFAULT_EVALUATION_SEED, DEFAULT_VIDEO_FPS
from ..evaluation.recording import EpisodeRecorder
from ..paths import output_path
from ..scenarios.registry import ScenarioConfig, get_scenario, list_scenario_names


CORE_SCENARIOS = (
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
)
DEFAULT_INSTRUCTION = "Grasp and lift the blue cable."


def _env_config(
    scenario: ScenarioConfig,
    *,
    seed: int,
    episode_seconds: float,
    robot: str,
) -> EnvConfig:
    return EnvConfig(
        robot=robot,
        seed=seed,
        episode_seconds=episode_seconds,
        scenario_name=scenario.name,
        scenario_id=scenario.scenario_id,
        scenario_split=scenario.split.value,
        frame_skip=20,
        dynamicvla_cameras_enabled=True,
        target_selection="middle",
        **scenario.to_env_overrides(),
    )


def _observation(env: CableGraspEnv, instruction: str) -> dict:
    """Build the six-state/two-camera schema used by LeRobotCableDataConfig."""

    images = env.dynamicvla_camera_rgb()
    rotation = env.data.xmat[env.hand_id].reshape(3, 3)
    euler_xyz = Rotation.from_matrix(rotation).as_euler("xyz", degrees=False)
    state = np.concatenate((env.hand_position, euler_xyz)).astype(np.float32)
    return {
        "observation/image": images["opst_cam"],
        "observation/wrist_image": images["wrist_cam"],
        "observation/state": state,
        "prompt": instruction,
    }


def _model_action_to_wire(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert pi0.5's absolute xyz/euler/gripper action to xyz/quat/gripper."""

    value = np.asarray(action, dtype=np.float64)
    if value.shape == (1, 7):
        value = value[0]
    if value.shape != (7,):
        raise ValueError(f"Expected pi0.5 action shape (7,), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("pi0.5 action contains non-finite values")
    quaternion = Rotation.from_euler("xyz", value[3:6], degrees=False).as_quat(
        scalar_first=True
    )
    wire = np.concatenate((value[:3], quaternion, value[6:7]))
    return value, wire


def _run_episode(
    *,
    policy,
    scenario: ScenarioConfig,
    episode: int,
    seed: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict:
    env = CableGraspEnv(_env_config(
        scenario,
        seed=seed,
        episode_seconds=args.episode_seconds,
        robot=args.robot,
    ))
    adapter = DynamicVLATaskSpaceAdapter(env, DynamicVLAAdapterConfig())
    model_path = run_dir / "models" / f"{scenario.name}.mjb"
    try:
        _, initial_info = env.reset(seed=seed)
        adapter.reset()
        episode_index = env.trial_index
        episode_dir = (
            run_dir / "episodes" / "pi05" / scenario.name / f"seed_{seed}"
        )
        recorder = EpisodeRecorder(env, episode_dir, video_fps=args.video_fps)
        recorder.capture_initial()
        model_actions = 0
        policy_result = "running"
        termination_reason: str | None = None
        min_target_distance = float("inf")
        inference_ms: list[float] = []
        position_clipped: list[bool] = []
        quaternion_repaired: list[bool] = []
        previous_model_action: np.ndarray | None = None
        step_count = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            start = time.perf_counter()
            response = policy.infer(_observation(env, args.instruction))
            elapsed_ms = 1000.0 * (time.perf_counter() - start)
            inference_ms.append(elapsed_ms)
            model_action, wire_action = _model_action_to_wire(
                np.asarray(response["actions"])[0]
            )
            adapter.set_model_action(wire_action)
            actuator_action = adapter.action()
            _, reward, terminated, truncated, step_info = env.step(actuator_action)
            diagnostics = adapter.diagnostics()
            model_actions += 1
            step_count += 1
            position_clipped.append(bool(diagnostics["position_clipped"]))
            quaternion_repaired.append(bool(diagnostics["quaternion_repaired"]))
            recorder.record_step(
                model_action,
                reward,
                terminated,
                truncated,
                step_info,
                extras={
                    "pose_command": diagnostics["pose_command"],
                    "wire_action": wire_action,
                    "inference_ms": np.asarray(elapsed_ms, dtype=np.float64),
                },
            )
            min_target_distance = min(
                min_target_distance,
                float(np.linalg.norm(env.target_position() - env.hand_position)),
            )
            if previous_model_action is not None:
                # The metric is kept in the manifest through the recorded action
                # stream; retaining this variable documents the intended pairing.
                _ = float(np.linalg.norm(model_action - previous_model_action))
            previous_model_action = model_action
            termination_reason = step_info.get("termination_reason")
            if step_count > int(args.max_steps):
                truncated = True
                termination_reason = "pi05_eval_step_guard"

        info = env.info()
        info["base_success"] = bool(env.ever_success)
        info["success"] = bool(env.ever_success)
        if env.ever_success:
            policy_result = "success"
        elif model_actions == 0:
            policy_result = "failed_no_model_action"
        elif termination_reason == "rigid_motion_boundary_crossed":
            policy_result = "failed_motion_boundary"
        else:
            policy_result = "failed_timeout"
        row = base_row(
            "pi05_lora",
            episode_index,
            seed,
            scenario,
            initial_info,
            info,
            env.grasp_break_history,
        )
        row.update({
            "steps": step_count,
            "sim_time": float(env.data.time),
            "episode_return": np.nan,
            "min_target_distance": min_target_distance,
            "policy_result": policy_result,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "model_action_messages": model_actions,
            "model_action_received": model_actions > 0,
            "position_clip_frames": int(np.count_nonzero(position_clipped)),
            "quaternion_repair_frames": int(np.count_nonzero(quaternion_repaired)),
            "policy_inference_mean_ms": float(np.mean(inference_ms)) if inference_ms else np.nan,
            "policy_inference_p95_ms": float(np.quantile(inference_ms, 0.95)) if inference_ms else np.nan,
        })
        artifacts = recorder.finish(row)
        row.update(artifacts.relative_to(run_dir))
        print(
            f"scenario={scenario.name} seed={seed} result={policy_result} "
            f"steps={step_count} infer_ms={row['policy_inference_mean_ms']:.1f}",
            flush=True,
        )
        return row
    except BaseException:
        # EpisodeRecorder.close() is idempotent and prevents a half-open video
        # writer if the websocket or model returns an invalid action.
        if "recorder" in locals():
            recorder.close()
        raise
    finally:
        env.close()


def run(args: argparse.Namespace) -> Path:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    scenarios = [get_scenario(name) for name in args.scenarios]
    run_dir = Path(args.output) / (args.run_name or datetime.now().strftime("pi05_core_pilot_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "models").mkdir()
    for scenario in scenarios:
        env = CableGraspEnv(_env_config(
            scenario,
            seed=args.seed,
            episode_seconds=args.episode_seconds,
            robot=args.robot,
        ))
        try:
            mujoco.mj_saveModel(env.model, str(run_dir / "models" / f"{scenario.name}.mjb"), None)
        finally:
            env.close()

    print(f"connecting_policy=ws://{args.policy_host}:{args.policy_port}", flush=True)
    policy = WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    metadata = policy.get_server_metadata()
    print(f"policy_metadata={metadata}", flush=True)
    rows: list[dict] = []
    for scenario in scenarios:
        for episode in range(1, args.trials + 1):
            rows.append(_run_episode(
                policy=policy,
                scenario=scenario,
                episode=episode,
                seed=args.seed + episode - 1,
                args=args,
                run_dir=run_dir,
            ))

    episodes_path = run_dir / "episodes.csv"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    write_csv(episodes_path, rows)
    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 4,
        "created_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "method": "pi05_lora",
        "policy": {
            "config": args.policy_config,
            "checkpoint": args.checkpoint,
            "host": args.policy_host,
            "port": args.policy_port,
            "instruction": args.instruction,
            "metadata": metadata,
            "action_format": "absolute_xyz_euler_xyz_gripper_to_xyz_quaternion_wxyz_gripper",
            "delta_action_training_dims": 6,
        },
        "scenarios": [scenario.asdict() for scenario in scenarios],
        "seeds": [args.seed + index for index in range(args.trials)],
        "trials_per_scenario": args.trials,
        "recording": {
            "enabled": True,
            "schema_version": 1,
            "video_fps": args.video_fps,
            "cameras": ["opst_cam", "wrist_cam"],
        },
        "artifacts": {
            "episodes_csv": str(episodes_path.resolve()),
            "summary": str(summary_path.resolve()),
            "model_files": {
                scenario.name: {
                    "path": str((run_dir / "models" / f"{scenario.name}.mjb").resolve()),
                    "sha256": sha256_file(run_dir / "models" / f"{scenario.name}.mjb"),
                }
                for scenario in scenarios
            },
        },
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in {
                "runner": Path(__file__),
                "adapter": Path(__file__).resolve().parents[1] / "dynamicvla" / "adapter.py",
                "environment": Path(__file__).resolve().parents[1] / "env" / "environment.py",
                "scenario_registry": Path(__file__).resolve().parents[1] / "scenarios" / "registry.py",
            }.items()
        },
        "robot": args.robot,
        "source_xml": {"path": str(XML_PATH.resolve()), "sha256": sha256_file(XML_PATH)},
        "robot_xml": {"path": str(args.robot_xml.resolve()), "sha256": sha256_file(args.robot_xml)},
        "panda_xml": {"path": str(PANDA_XML_PATH.resolve()), "sha256": sha256_file(PANDA_XML_PATH)},
        "menagerie_panda_assets": str(resolve_menagerie_panda_dir()) if args.robot == "panda" else None,
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_dirty": bool(git_text("status", "--porcelain=v1")),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "summary": summary,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    total = len(rows)
    successes = sum(bool(row["task_success"]) for row in rows)
    print(f"episodes={episodes_path.resolve()}", flush=True)
    print(f"summary={summary_path.resolve()}", flush=True)
    print(f"completed={total} successes={successes} rate={successes / total:.1%}", flush=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a served pi0.5 policy on Panda cable scenarios")
    parser.add_argument("--scenarios", nargs="+", choices=list_scenario_names(), default=list(CORE_SCENARIOS))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--robot", choices=tuple(sorted(ROBOT_SPECS)), default="panda")
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--policy-host", default="10.1.114.130")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--policy-config", default="pi05_cable_lora_pilot")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, default=output_path("pi05", "evaluation"))
    parser.add_argument("--run-name")
    args = parser.parse_args()
    args.robot_xml = ROBOT_SPECS[args.robot].xml_path
    if args.trials < 1 or args.episode_seconds <= 0.0 or args.video_fps <= 0.0 or args.max_steps < 1:
        parser.error("trials, episode-seconds, video-fps and max-steps must be positive")
    if not args.instruction.strip():
        parser.error("instruction must be non-empty")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
