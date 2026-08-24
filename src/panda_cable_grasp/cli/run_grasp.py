"""Scripted-policy CLI: unified headless benchmark or interactive viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from ..env.environment import CableGraspEnv, EnvConfig, XML_PATH
from ..evaluation.benchmark import run_benchmark
from ..evaluation.defaults import DEFAULT_EVALUATION_SEED, DEFAULT_VIDEO_FPS
from ..paths import output_path
from ..policies.scripted import DynamicCableGraspPolicy
from ..scenarios.registry import (
    SCENARIO_SUITE_NAMES,
    get_scenario,
    list_scenario_names,
)


def env_config_from_args(args: argparse.Namespace) -> EnvConfig:
    if args.scenario is None:
        return EnvConfig(
            seed=args.seed,
            episode_seconds=args.episode_seconds,
            disturbance_strength=args.disturbance,
        )
    scenario = get_scenario(args.scenario)
    return EnvConfig(
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        scenario_name=scenario.name,
        scenario_id=scenario.scenario_id,
        scenario_split=scenario.split.value,
        **scenario.to_env_overrides(),
    )


def print_trial_start(env: CableGraspEnv) -> None:
    index = env.cable_ids.index(env.target_body_id)
    print(
        f"trial={env.trial_index} started "
        f"target_segment={index}/{len(env.cable_ids) - 1} "
        f"target_body={env.target_body_id}",
        flush=True,
    )


def run_headless(args: argparse.Namespace) -> None:
    """Delegate historical scripted commands to the canonical evaluator."""

    run_benchmark(argparse.Namespace(
        methods=["scripted"],
        ppo_model=None,
        episodes=args.trials,
        seed=args.seed,
        disturbance=args.disturbance,
        scenario=args.scenario,
        scenarios=args.scenarios,
        suite=args.suite,
        episode_seconds=args.episode_seconds,
        scenario_workers=args.scenario_workers,
        envs_per_scenario=args.envs_per_scenario,
        workers=args.workers,
        recording=True,
        video_fps=args.video_fps,
        device="cpu",
        output=args.video_dir,
        run_name=args.run_name,
    ))


def run_viewer(args: argparse.Namespace) -> None:
    """Run one scripted environment in the passive MuJoCo viewer."""

    from mujoco import viewer

    env = CableGraspEnv(env_config_from_args(args))
    env.reset(seed=args.seed)
    policy = DynamicCableGraspPolicy(env)
    controls = {"reset": False, "paused": False, "speed": float(args.speed)}

    def key_callback(keycode: int) -> None:
        if keycode in (259, ord("R"), ord("r"), ord("N"), ord("n")):
            controls["reset"] = True
        elif keycode == ord(" "):
            controls["paused"] = not controls["paused"]
        elif keycode in (ord("="), ord("]")):
            controls["speed"] = min(8.0, controls["speed"] * 2.0)
        elif keycode in (ord("-"), ord("[")):
            controls["speed"] = max(0.25, controls["speed"] / 2.0)

    print(f"model={XML_PATH}", flush=True)
    print(
        "GUI controls: = or ] speed up, - or [ slow down, Space pause, "
        "Backspace/R reset, N new random trial",
        flush=True,
    )
    print_trial_start(env)
    completed_trials = 0
    trial_reported = False
    reset_after: float | None = None
    try:
        with viewer.launch_passive(
            env.model,
            env.data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as handle:
            handle.cam.lookat[:] = [0.55, 0.0, 0.30]
            handle.cam.distance = 2.10
            handle.cam.azimuth = 135
            handle.cam.elevation = -25
            start_wall = time.perf_counter()
            wall_anchor = start_wall
            sim_anchor = float(env.data.time)
            while handle.is_running():
                frame_start = time.perf_counter()
                now = frame_start
                if args.duration > 0.0 and now - start_wall >= args.duration:
                    break
                if controls["reset"]:
                    env.reset()
                    policy.reset()
                    controls["reset"] = False
                    trial_reported = False
                    reset_after = None
                    wall_anchor = now
                    sim_anchor = float(env.data.time)
                    print_trial_start(env)
                if controls["paused"]:
                    wall_anchor = now
                    sim_anchor = float(env.data.time)
                else:
                    target_time = sim_anchor + controls["speed"] * (now - wall_anchor)
                    steps = 0
                    while env.data.time < target_time and steps < 8 and not policy.finished:
                        _, _, _, truncated, info = env.step(policy.action())
                        steps += 1
                        if truncated and not policy.finished:
                            policy.result = (
                                "failed_motion_boundary"
                                if info.get("termination_reason")
                                == "rigid_motion_boundary_crossed"
                                else "failed_timeout"
                            )
                            policy.finished = True
                    if steps == 8:
                        wall_anchor = now
                        sim_anchor = float(env.data.time)
                    if policy.finished and not trial_reported:
                        completed_trials += 1
                        print(policy.summary(), flush=True)
                        trial_reported = True
                        reset_after = now + 1.0
                if reset_after is not None and now >= reset_after:
                    if args.trials <= 0 or completed_trials < args.trials:
                        controls["reset"] = True
                    reset_after = None
                handle.sync()
                remaining = 1.0 / 60.0 - (time.perf_counter() - frame_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reactive Panda grasping of a continuously deforming cable"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--disturbance", type=float, default=1.5)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scenario", choices=list_scenario_names())
    selection.add_argument("--scenarios", nargs="+", choices=list_scenario_names())
    selection.add_argument("--suite", choices=SCENARIO_SUITE_NAMES)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument(
        "--video-dir", type=Path, default=output_path("scripted"),
        help="headless benchmark output root",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--scenario-workers", type=int, default=1)
    parser.add_argument("--envs-per-scenario", type=int, default=1)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    args.speed = min(8.0, max(0.25, args.speed))
    if args.headless and args.trials <= 0:
        parser.error("--headless requires --trials greater than zero")
    if args.video_fps <= 0.0 or args.episode_seconds <= 0.0:
        parser.error("--video-fps and --episode-seconds must be positive")
    if (
        args.scenario_workers <= 0
        or args.envs_per_scenario <= 0
        or (args.workers is not None and args.workers <= 0)
    ):
        parser.error("parallel worker counts must be positive")
    if not args.headless and (args.scenarios is not None or args.suite is not None):
        parser.error("GUI mode supports one --scenario; use --headless for matrices")
    if args.run_name is not None:
        candidate = Path(args.run_name)
        if (
            not args.run_name.strip()
            or candidate.name != args.run_name
            or args.run_name in {".", ".."}
        ):
            parser.error("--run-name must be one non-empty directory name")
    return args


def main() -> None:
    args = parse_args()
    if args.headless:
        run_headless(args)
    else:
        run_viewer(args)


if __name__ == "__main__":
    main()
