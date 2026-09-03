"""Zero-shot DynamicVLA evaluation server for the dynamic cable task.

Run this file in the lightweight MuJoCo environment.  Run DynamicVLA's
official ``scripts/inference.py`` in its separate PyTorch environment; the two
processes exchange the official observation/action schema over ZeroMQ.
"""

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

from ..evaluation.benchmark import (
    base_row,
    git_text,
    sha256_file,
    summarize,
    write_csv,
)
from ..evaluation.defaults import DEFAULT_EVALUATION_SEED, DEFAULT_VIDEO_FPS
from ..evaluation.recording import EpisodeRecorder
from ..env.environment import (
    CableGraspEnv,
    EnvConfig,
    PANDA_XML_PATH,
    ROBOT_SPECS,
    XML_PATH,
    resolve_menagerie_panda_dir,
)
from ..dynamicvla.adapter import (
    DynamicVLAAdapterConfig,
    DynamicVLATaskSpaceAdapter,
    make_dynamicvla_observation,
)
from ..scenarios.registry import get_scenario, list_scenario_names
from ..paths import output_path


def env_config_from_args(args: argparse.Namespace) -> EnvConfig:
    scenario = get_scenario(args.scenario)
    return EnvConfig(
        robot=getattr(args, "robot", "panda"),
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        scenario_name=scenario.name,
        scenario_id=scenario.scenario_id,
        scenario_split=scenario.split.value,
        frame_skip=20,  # 25 Hz, matching DOM collection/evaluation.
        dynamicvla_cameras_enabled=True,
        **scenario.to_env_overrides(),
    )


def _recv_latest(socket) -> dict | None:
    import zmq

    message = None
    while True:
        try:
            message = socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return message


def _wait_for_message(socket, key: str, timeout_seconds: float) -> dict | None:
    import zmq

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    deadline = None if timeout_seconds <= 0.0 else time.monotonic() + timeout_seconds
    while deadline is None or time.monotonic() < deadline:
        wait_ms = 1000
        if deadline is not None:
            wait_ms = max(1, min(wait_ms, int(1000 * (deadline - time.monotonic()))))
        if poller.poll(wait_ms):
            matched = None
            while True:
                try:
                    message = socket.recv_pyobj(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                if key in message:
                    matched = message
            if matched is not None:
                return matched
    return None


def _wait_for_sync_action(socket, sync_index: int, timeout_seconds: float) -> dict | None:
    """Wait for exactly the action belonging to one published observation."""
    import zmq

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while time.monotonic() < deadline:
        wait_ms = max(1, min(1000, int(1000 * (deadline - time.monotonic()))))
        if not poller.poll(wait_ms):
            continue
        while True:
            try:
                message = socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if not isinstance(message, dict) or "action" not in message:
                continue
            if message.get("sync_index") != sync_index:
                print(
                    "ignored_stale_sync_action="
                    f"expected_{sync_index}_received_{message.get('sync_index')}",
                    flush=True,
                )
                continue
            return message
    return None


def _send_until_ack(
    obs_socket,
    act_socket,
    payload: dict,
    timeout_seconds: float,
    retry_seconds: float = 0.5,
) -> dict | None:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        obs_socket.send_pyobj(payload)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None
        ack = _wait_for_message(
            act_socket, "ack", min(retry_seconds, remaining)
        )
        if ack is not None:
            return ack


def check_bridge(args: argparse.Namespace) -> None:
    """Compile cameras and validate action/observation shapes without a model run."""

    env = CableGraspEnv(env_config_from_args(args))
    try:
        env.reset(seed=args.seed)
        adapter = DynamicVLATaskSpaceAdapter(env)
        images = env.dynamicvla_camera_rgb()
        current_quat = np.zeros(4)
        import mujoco

        mujoco.mju_mat2Quat(
            current_quat, env.data.xmat[env.hand_id].reshape(-1)
        )
        probe = np.concatenate([
            env.hand_position,
            current_quat,
            np.array([1.0]),
        ])
        adapter.set_model_action(probe)
        actuator_action = adapter.action()
        observation = make_dynamicvla_observation(
            env, args.instruction, index=0
        )
        expected = (
            env.config.dynamicvla_camera_height,
            env.config.dynamicvla_camera_width,
            3,
        )
        assert images["opst_cam"].shape == expected
        assert images["wrist_cam"].shape == expected
        assert observation["observation.state"]["end_effector"]["pos"].shape == (1, 3)
        assert observation["observation.state"]["end_effector"]["quat"].shape == (1, 4)
        assert actuator_action.shape == (8,)
        assert np.all(np.isfinite(actuator_action))
        print("dynamicvla_bridge_check=OK")
        print(f"opst_camera={images['opst_cam'].shape} std={images['opst_cam'].std():.2f}")
        print(f"wrist_camera={images['wrist_cam'].shape} std={images['wrist_cam'].std():.2f}")
        print("model_experiment_run=false")
    finally:
        env.close()


def run_server(args: argparse.Namespace) -> None:
    import mujoco
    import zmq

    scenario = get_scenario(args.scenario)
    sync_mode = bool(args.sync)
    env = CableGraspEnv(env_config_from_args(args))
    adapter = DynamicVLATaskSpaceAdapter(env)
    context = zmq.Context()
    obs_socket = context.socket(zmq.PUB)
    act_socket = context.socket(zmq.PULL)
    obs_socket.setsockopt(zmq.SNDHWM, 1)
    act_socket.setsockopt(zmq.RCVHWM, 4)
    obs_socket.bind(f"tcp://{args.host}:{args.img_port}")
    act_socket.bind(f"tcp://{args.host}:{args.act_port}")

    print(
        f"dynamicvla_server=tcp://{args.host}:{args.img_port} "
        f"actions=tcp://{args.host}:{args.act_port}",
        flush=True,
    )
    print("waiting_for_dynamicvla_client=true", flush=True)
    handshake = _wait_for_message(act_socket, "vla", args.client_timeout)
    if handshake is None:
        env.close()
        obs_socket.close(linger=0)
        act_socket.close(linger=0)
        context.term()
        raise TimeoutError("Timed out waiting for the DynamicVLA inference client")
    vla_name = str(handshake["vla"])
    vla_epoch = int(handshake.get("epoch", 0))
    if sync_mode:
        if handshake.get("protocol") != "dynamicvla_sync_v1":
            env.close()
            obs_socket.close(linger=0)
            act_socket.close(linger=0)
            context.term()
            raise RuntimeError(
                "Synchronous server requires a DynamicVLA client started with --sync"
            )
        client_execute_steps = handshake.get("execute_steps")
        if int(client_execute_steps) != args.execute_steps:
            env.close()
            obs_socket.close(linger=0)
            act_socket.close(linger=0)
            context.term()
            raise RuntimeError(
                "Client/server execute_steps mismatch: "
                f"server={args.execute_steps}, client={client_execute_steps}"
            )
    print(f"dynamicvla_client={vla_name} epoch={vla_epoch}", flush=True)
    if sync_mode:
        print(
            "dynamicvla_sync=true "
            f"execute_steps={args.execute_steps} "
            f"client_chunk_size={handshake.get('chunk_size', 'unknown')}",
            flush=True,
        )
    # The PUSH handshake is sent only after the official SUB socket connects;
    # a short propagation window removes PUB/SUB's initial slow-joiner race.
    time.sleep(0.25)

    run_name = args.run_name or (
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    )
    run_dir = args.video_dir / run_name
    if args.run_name is None:
        suffix = 1
        while run_dir.exists():
            run_dir = args.video_dir / f"{run_name}_{suffix:02d}"
            suffix += 1
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir.resolve()}")
    run_dir.mkdir(parents=True)
    scenario_dir = run_dir
    models_dir = run_dir / "models"
    models_dir.mkdir()
    model_path = models_dir / f"{scenario.name}.mjb"
    mujoco.mj_saveModel(env.model, str(model_path), None)
    control_dt = float(env.model.opt.timestep * env.config.frame_skip)
    successes = 0
    rows: list[dict] = []
    try:
        for trial_offset in range(args.trials):
            episode_seed = args.seed + trial_offset
            _, initial_info = env.reset(seed=episode_seed)
            adapter.reset()
            _recv_latest(act_socket)
            trial_index = env.trial_index
            print(
                f"trial={trial_index} started seed={episode_seed} "
                f"instruction={args.instruction!r}",
                flush=True,
            )

            recorder = EpisodeRecorder(
                env,
                run_dir / "episodes" / "dynamicvla_zero_shot"
                / scenario.name / f"seed_{episode_seed}",
                video_fps=args.video_fps,
            )
            recorder.capture_initial()
            position_clipped: list[bool] = []
            quaternion_repaired: list[bool] = []
            ee_path: list[np.ndarray] = []
            min_target_distance = float("inf")
            model_action_messages = 0
            termination_reason: str | None = None
            previous_wall_step = control_dt
            observation_index = 0
            success = False
            truncated = False
            try:
                while not success and not truncated:
                    tick = time.perf_counter()
                    received_now = False
                    if sync_mode:
                        observation = make_dynamicvla_observation(
                            env,
                            args.instruction if observation_index == 0 else None,
                            index=observation_index,
                            dt_scale=1.0,
                        )
                        observation["sync_mode"] = True
                        observation["sync_index"] = observation_index
                        observation["execute_steps"] = args.execute_steps
                        obs_socket.send_pyobj(observation)
                        message = _wait_for_sync_action(
                            act_socket, observation_index, args.action_timeout
                        )
                        if message is None:
                            termination_reason = "sync_action_timeout"
                            truncated = True
                            print(
                                "sync_action_timeout="
                                f"index_{observation_index}_after_{args.action_timeout:.1f}s",
                                flush=True,
                            )
                            break
                        try:
                            adapter.set_model_action(message["action"])
                            model_action_messages += 1
                            received_now = True
                        except ValueError as error:
                            termination_reason = "sync_invalid_model_action"
                            truncated = True
                            print(
                                f"invalid_sync_model_action={error}", flush=True
                            )
                            break
                    else:
                        message = _recv_latest(act_socket)
                        received_now = bool(message is not None and "action" in message)
                        if received_now:
                            try:
                                adapter.set_model_action(message["action"])
                                model_action_messages += 1
                            except ValueError as error:
                                print(f"ignored_invalid_model_action={error}", flush=True)
                                received_now = False

                        observation = make_dynamicvla_observation(
                            env,
                            args.instruction if observation_index == 0 else None,
                            index=observation_index,
                            dt_scale=max(1.0, previous_wall_step / control_dt),
                        )
                        obs_socket.send_pyobj(observation)
                    diagnostics = adapter.diagnostics()
                    position_clipped.append(diagnostics["position_clipped"])
                    quaternion_repaired.append(diagnostics["quaternion_repaired"])
                    ee_path.append(env.hand_position.copy())

                    action = adapter.action()
                    _, reward, success, truncated, step_info = env.step(action)
                    recorder.record_step(
                        diagnostics["raw_action"], reward, success, truncated,
                        step_info,
                        extras={
                            "pose_commands": diagnostics["pose_command"],
                            "joint_actions": action,
                            "received_action_mask": received_now,
                            "position_clipped": diagnostics["position_clipped"],
                            "quaternion_repaired": diagnostics["quaternion_repaired"],
                        },
                    )
                    min_target_distance = min(
                        min_target_distance,
                        float(np.linalg.norm(env.target_position() - env.hand_position)),
                    )
                    termination_reason = step_info.get("termination_reason")
                    observation_index += 1

                    if not sync_mode:
                        elapsed = time.perf_counter() - tick
                        remaining = control_dt - elapsed
                        if remaining > 0.0:
                            time.sleep(remaining)
                        previous_wall_step = time.perf_counter() - tick
            except BaseException:
                recorder.close()
                raise

            if success:
                result = "success"
            elif termination_reason == "sync_action_timeout":
                result = "failed_sync_action_timeout"
            elif termination_reason == "sync_invalid_model_action":
                result = "failed_sync_invalid_action"
            elif model_action_messages == 0:
                result = "failed_no_model_action"
            elif termination_reason == "rigid_motion_boundary_crossed":
                result = "failed_motion_boundary"
            else:
                result = "failed_timeout"

            info = env.info()
            info["ever_pinched"] = env.last_grasped_body_id is not None
            info["base_success"] = env.ever_success
            row = base_row(
                "dynamicvla_zero_shot", trial_index, episode_seed, scenario,
                initial_info, info, env.grasp_break_history,
            )
            row.update({
                "steps": observation_index,
                "sim_time": float(env.data.time),
                "episode_return": np.nan,
                "min_target_distance": min_target_distance,
                "policy_result": result,
                "terminated": success,
                "truncated": truncated,
                "model_action_messages": model_action_messages,
                "model_action_received": model_action_messages > 0,
                "sync_mode": sync_mode,
                "execute_steps": args.execute_steps if sync_mode else None,
                "position_clip_frames": int(np.count_nonzero(position_clipped)),
                "quaternion_repair_frames": int(np.count_nonzero(quaternion_repaired)),
                "instruction": args.instruction,
                "dynamicvla_name": vla_name,
                "dynamicvla_epoch": vla_epoch,
                "compiled_model": str(model_path.relative_to(run_dir)),
            })
            artifacts = recorder.finish(row)
            row.update(artifacts.relative_to(run_dir))
            rows.append(row)
            successes += int(success)
            print(
                f"trial={trial_index} result={result} sim_time={env.data.time:.3f}s "
                f"model_actions={model_action_messages}",
                flush=True,
            )
            print(f"  episode_dir={artifacts.episode_dir.resolve()}", flush=True)

            episode_result = {
                "env_name": scenario.name,
                "eps_name": f"trial_{trial_index:03d}",
                "success": bool(success),
                "ee_path": np.asarray(ee_path),
            }
            # The patched official client deduplicates this terminal payload by
            # (env_name, eps_name), so retrying is reliable and side-effect free.
            time.sleep(0.25)
            ack = _send_until_ack(
                obs_socket,
                act_socket,
                episode_result,
                args.ack_timeout,
            )
            if ack is None:
                print(
                    f"warning=dynamicvla_ack_timeout trial={trial_index}",
                    flush=True,
                )

        obs_socket.send_pyobj({
            "vla": vla_name,
            "success_rates": {scenario.name: successes / args.trials},
        })

        episodes_path = scenario_dir / "episodes.csv"
        manifest_path = scenario_dir / "manifest.json"
        summary_path = scenario_dir / "summary.json"
        write_csv(episodes_path, rows)
        summary = summarize(rows)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        git_status = git_text("status", "--porcelain=v1")
        manifest = {
            "schema_version": 4,
            "created_at": datetime.now().astimezone().isoformat(),
            "command": [sys.executable, *sys.argv],
            "method": "dynamicvla_zero_shot",
            "dynamicvla": {
                "client_name": vla_name,
                "epoch": vla_epoch,
                "instruction": args.instruction,
                "protocol": (
                    "dynamicvla_sync_v1" if sync_mode else "official_zmq_schema"
                ),
                "synchronous": sync_mode,
                "execute_steps": args.execute_steps if sync_mode else None,
                "predicted_chunk_size": handshake.get("chunk_size"),
                "rotation": "euler_in_model_quaternion_on_wire_wxyz",
                "delta_action": handshake.get("delta_action", True),
                "control_hz": 1.0 / control_dt,
            },
            "output": {
                "run_dir": str(run_dir.resolve()),
                "scenario_dir": str(scenario_dir.resolve()),
            },
            "scenario": scenario.asdict(),
            "seeds": [args.seed + index for index in range(args.trials)],
            "recording": {
                "enabled": True,
                "schema_version": 1,
                "state_format": "mujoco_mjSTATE_FULLPHYSICS",
                "state_sampling": "initial_and_after_every_control_step",
                "video_fps": args.video_fps,
                "cameras": ["opst_cam", "wrist_cam"],
            },
            "camera": {
                "width": env.config.dynamicvla_camera_width,
                "height": env.config.dynamicvla_camera_height,
                "fovy": env.config.dynamicvla_camera_fovy,
                "opst_name": env.config.dynamicvla_opst_camera_name,
                "opst_pos_in_base": env.config.dynamicvla_opst_camera_pos,
                "opst_quat_wxyz_in_base": env.config.dynamicvla_opst_camera_quat,
                "wrist_name": env.config.dynamicvla_wrist_camera_name,
                "wrist_pos_in_hand": env.config.dynamicvla_wrist_camera_pos,
                "wrist_quat_wxyz_in_hand": env.config.dynamicvla_wrist_camera_quat,
            },
            "artifacts": {
                "model": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)},
                "episodes_csv": str(episodes_path.resolve()),
                "summary": str(summary_path.resolve()),
                "global_videos": [row["global_video"] for row in rows],
                "wrist_videos": [row["wrist_video"] for row in rows],
                "trajectories": [row["trajectory"] for row in rows],
            },
            "source_files": {
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in {
                    "runner": Path(__file__),
                    "adapter": Path(__file__).resolve().parents[1]
                    / "dynamicvla" / "adapter.py",
                    "environment": Path(__file__).resolve().parents[1]
                    / "env" / "environment.py",
                    "scenario_registry": Path(__file__).resolve().parents[1]
                    / "scenarios" / "registry.py",
                }.items()
            },
            "configs": {
                "environment": asdict(env.config),
                "adapter": asdict(DynamicVLAAdapterConfig()),
            },
            "robot": env.robot,
            "source_xml": {"path": str(XML_PATH.resolve()), "sha256": sha256_file(XML_PATH)},
            "robot_xml": {
                "path": str(env.robot_spec.xml_path.resolve()),
                "sha256": sha256_file(env.robot_spec.xml_path),
            },
            "panda_xml": {"path": str(PANDA_XML_PATH.resolve()), "sha256": sha256_file(PANDA_XML_PATH)},
            "menagerie_panda_assets": (
                str(resolve_menagerie_panda_dir())
                if env.robot == "panda" else None
            ),
            "git_commit": git_text("rev-parse", "HEAD"),
            "git_dirty": bool(git_status),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "summary": summary,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"episodes={episodes_path.resolve()}", flush=True)
        print(f"manifest={manifest_path.resolve()}", flush=True)
        print(
            f"completed={args.trials} successes={successes} "
            f"rate={successes / args.trials:.1%}",
            flush=True,
        )
    finally:
        env.close()
        obs_socket.close(linger=0)
        act_socket.close(linger=0)
        context.term()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot DynamicVLA evaluation on the MuJoCo cable task"
    )
    parser.add_argument("--scenario", choices=list_scenario_names(), required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument(
        "--robot", choices=tuple(sorted(ROBOT_SPECS)), default="panda",
        help="robot model used by the MuJoCo environment",
    )
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument(
        "--headless", action="store_true",
        help="accepted for CLI compatibility; this server is always offscreen",
    )
    parser.add_argument(
        "--instruction", default="Pick up the blue cable.",
        help="language instruction passed to DynamicVLA",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--img-port", type=int, default=3186)
    parser.add_argument("--act-port", type=int, default=3188)
    parser.add_argument(
        "--sync",
        "--synchronous",
        action="store_true",
        help="Wait for the matching action before every simulation step",
    )
    parser.add_argument(
        "--execute-steps",
        "--execute_steps",
        type=int,
        default=20,
        help="Actions consumed from each predicted chunk before the next prediction",
    )
    parser.add_argument(
        "--action-timeout",
        "--action_timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for the action matching one observation in sync mode",
    )
    parser.add_argument(
        "--client-timeout", type=float, default=0.0,
        help="seconds to wait for the model client; 0 waits indefinitely",
    )
    parser.add_argument("--ack-timeout", type=float, default=60.0)
    parser.add_argument("--video-dir", type=Path, default=output_path("dynamicvla", "evaluation"))
    parser.add_argument("--run-name")
    parser.add_argument(
        "--check-only", action="store_true",
        help="validate cameras/schema/IK without connecting to or running a model",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.episode_seconds <= 0.0:
        parser.error("--episode-seconds must be positive")
    if args.video_fps <= 0.0:
        parser.error("--video-fps must be positive")
    if args.execute_steps < 1:
        parser.error("--execute-steps must be positive")
    if args.action_timeout <= 0.0:
        parser.error("--action-timeout must be positive")
    if not args.instruction.strip():
        parser.error("--instruction must be non-empty")
    return args


def main() -> None:
    parsed_args = parse_args()
    if parsed_args.check_only:
        check_bridge(parsed_args)
    else:
        run_server(parsed_args)


if __name__ == "__main__":
    main()
