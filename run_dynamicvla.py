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

from benchmark import (
    _base_row,
    _git_text,
    _sha256,
    _summary,
    _write_csv,
)
from cable_grasp_env import (
    CableGraspEnv,
    EnvConfig,
    PANDA_XML_PATH,
    XML_PATH,
    resolve_menagerie_panda_dir,
)
from dynamicvla_adapter import (
    DynamicVLAAdapterConfig,
    DynamicVLATaskSpaceAdapter,
    make_dynamicvla_observation,
)
from experiment_scenarios import get_scenario, list_scenario_names
from project_paths import output_path


def env_config_from_args(args: argparse.Namespace) -> EnvConfig:
    scenario = get_scenario(args.scenario)
    return EnvConfig(
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        scenario_name=scenario.name,
        scenario_id=scenario.scenario_id,
        scenario_split=scenario.split.value,
        frame_skip=20,  # 25 Hz, matching DOM collection/evaluation.
        camera_observation_enabled=False,
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
    import cv2
    import mujoco
    import zmq

    scenario = get_scenario(args.scenario)
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
    print(f"dynamicvla_client={vla_name} epoch={vla_epoch}", flush=True)
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
    scenario_dir = run_dir / f"dynamicvla_{scenario.name}"
    if scenario_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {scenario_dir.resolve()}")
    scenario_dir.mkdir(parents=True)

    model_path = scenario_dir / f"{scenario.name}.mjb"
    mujoco.mj_saveModel(env.model, str(model_path), None)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state_size = mujoco.mj_stateSize(env.model, state_spec)
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

            opst_video_path = scenario_dir / f"trial_{trial_index:03d}.mp4"
            wrist_video_path = scenario_dir / f"trial_{trial_index:03d}_wrist.mp4"
            video_size = (
                env.config.dynamicvla_camera_width,
                env.config.dynamicvla_camera_height,
            )
            writers = [
                cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / control_dt,
                    video_size,
                )
                for path in (opst_video_path, wrist_video_path)
            ]
            if not all(writer.isOpened() for writer in writers):
                for writer in writers:
                    writer.release()
                raise RuntimeError("Unable to create DynamicVLA input videos")

            states: list[np.ndarray] = []
            frame_times: list[float] = []
            raw_model_actions: list[np.ndarray] = []
            pose_commands: list[np.ndarray] = []
            joint_actions: list[np.ndarray] = []
            received_action_masks: list[bool] = []
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
                    images = {
                        "opst_cam": observation["observation.images.opst_cam"][0],
                        "wrist_cam": observation["observation.images.wrist_cam"][0],
                    }
                    writers[0].write(cv2.cvtColor(images["opst_cam"], cv2.COLOR_RGB2BGR))
                    writers[1].write(cv2.cvtColor(images["wrist_cam"], cv2.COLOR_RGB2BGR))

                    state = np.empty(state_size, dtype=np.float64)
                    mujoco.mj_getState(env.model, env.data, state, state_spec)
                    diagnostics = adapter.diagnostics()
                    states.append(state)
                    frame_times.append(float(env.data.time))
                    raw_model_actions.append(diagnostics["raw_action"])
                    pose_commands.append(diagnostics["pose_command"])
                    received_action_masks.append(received_now)
                    position_clipped.append(diagnostics["position_clipped"])
                    quaternion_repaired.append(diagnostics["quaternion_repaired"])
                    ee_path.append(env.hand_position.copy())

                    action = adapter.action()
                    joint_actions.append(action.copy())
                    _, _, success, truncated, step_info = env.step(action)
                    min_target_distance = min(
                        min_target_distance,
                        float(np.linalg.norm(env.target_position() - env.hand_position)),
                    )
                    termination_reason = step_info.get("termination_reason")
                    observation_index += 1

                    elapsed = time.perf_counter() - tick
                    remaining = control_dt - elapsed
                    if remaining > 0.0:
                        time.sleep(remaining)
                    previous_wall_step = time.perf_counter() - tick
            finally:
                for writer in writers:
                    writer.release()

            if success:
                result = "success"
            elif model_action_messages == 0:
                result = "failed_no_model_action"
            elif termination_reason == "rigid_motion_boundary_crossed":
                result = "failed_motion_boundary"
            else:
                result = "failed_timeout"

            states_path = scenario_dir / f"trial_{trial_index:03d}_states.npz"
            np.savez_compressed(
                states_path,
                states=np.stack(states),
                state_spec=np.int64(int(state_spec)),
                frame_times=np.asarray(frame_times),
                fps=np.float64(1.0 / control_dt),
                raw_model_actions=np.stack(raw_model_actions),
                pose_commands=np.stack(pose_commands),
                joint_actions=np.stack(joint_actions),
                received_action_mask=np.asarray(received_action_masks),
                position_clipped=np.asarray(position_clipped),
                quaternion_repaired=np.asarray(quaternion_repaired),
                instruction=np.asarray(args.instruction),
                dynamicvla_name=np.asarray(vla_name),
                dynamicvla_epoch=np.int64(vla_epoch),
                model_file=np.asarray(model_path.name),
                opst_video_file=np.asarray(opst_video_path.name),
                wrist_video_file=np.asarray(wrist_video_path.name),
                seed=np.int64(episode_seed),
                scenario_name=np.asarray(scenario.name),
                result=np.asarray(result),
                termination_reason=np.asarray(termination_reason or ""),
            )

            info = env.info()
            info["ever_pinched"] = env.last_grasped_body_id is not None
            info["base_success"] = env.ever_success
            row = _base_row(
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
                "position_clip_frames": int(np.count_nonzero(position_clipped)),
                "quaternion_repair_frames": int(np.count_nonzero(quaternion_repaired)),
                "video_path": str(opst_video_path.resolve()),
                "wrist_video_path": str(wrist_video_path.resolve()),
                "states_path": str(states_path.resolve()),
                "model_path": str(model_path.resolve()),
            })
            rows.append(row)
            successes += int(success)
            print(
                f"trial={trial_index} result={result} sim_time={env.data.time:.3f}s "
                f"model_actions={model_action_messages}",
                flush=True,
            )
            print(f"  video={opst_video_path.resolve()}", flush=True)
            print(f"  wrist_video={wrist_video_path.resolve()}", flush=True)
            print(f"  states={states_path.resolve()}", flush=True)

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
        _write_csv(episodes_path, rows)
        git_status = _git_text("status", "--porcelain=v1")
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "command": [sys.executable, *sys.argv],
            "method": "dynamicvla_zero_shot",
            "dynamicvla": {
                "client_name": vla_name,
                "epoch": vla_epoch,
                "instruction": args.instruction,
                "protocol": "official_zmq_schema",
                "rotation": "euler_in_model_quaternion_on_wire_wxyz",
                "delta_action": True,
                "control_hz": 1.0 / control_dt,
            },
            "output": {
                "run_dir": str(run_dir.resolve()),
                "scenario_dir": str(scenario_dir.resolve()),
            },
            "scenario": scenario.asdict(),
            "seeds": [args.seed + index for index in range(args.trials)],
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
                "model": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
                "episodes_csv": str(episodes_path.resolve()),
                "videos": [row["video_path"] for row in rows],
                "wrist_videos": [row["wrist_video_path"] for row in rows],
                "states": [row["states_path"] for row in rows],
            },
            "source_files": {
                name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for name, path in {
                    "runner": Path(__file__),
                    "adapter": Path(__file__).resolve().parent / "dynamicvla_adapter.py",
                    "environment": Path(__file__).resolve().parent / "cable_grasp_env.py",
                    "scenario_registry": Path(__file__).resolve().parent / "experiment_scenarios.py",
                }.items()
            },
            "configs": {
                "environment": asdict(env.config),
                "adapter": asdict(DynamicVLAAdapterConfig()),
            },
            "source_xml": {"path": str(XML_PATH.resolve()), "sha256": _sha256(XML_PATH)},
            "panda_xml": {"path": str(PANDA_XML_PATH.resolve()), "sha256": _sha256(PANDA_XML_PATH)},
            "menagerie_panda_assets": str(resolve_menagerie_panda_dir()),
            "git_commit": _git_text("rev-parse", "HEAD"),
            "git_dirty": bool(git_status),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "summary": _summary(rows),
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
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
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
        "--client-timeout", type=float, default=0.0,
        help="seconds to wait for the model client; 0 waits indefinitely",
    )
    parser.add_argument("--ack-timeout", type=float, default=60.0)
    parser.add_argument("--video-dir", type=Path, default=output_path("headless_videos"))
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
    if not args.instruction.strip():
        parser.error("--instruction must be non-empty")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.check_only:
        check_bridge(parsed_args)
    else:
        run_server(parsed_args)
