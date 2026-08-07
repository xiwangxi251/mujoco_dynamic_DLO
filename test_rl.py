"""评估已训练PPO策略：支持批量headless统计和MuJoCo实时GUI。"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO

from rl_cable_env import RLCableGraspEnv


def print_episode(episode: int, episode_return: float, steps: int, info: dict) -> None:
    print(
        f"episode={episode} success={bool(info['success'])} return={episode_return:.3f} "
        f"steps={steps} sim_time={steps * 0.02:.3f}s "
        f"target_distance={info['target_distance']:.3f}m "
        f"grasped_body={info['grasped_body_id']} bilateral={info['bilateral_grasp']} "
        f"aperture={1000.0 * info['finger_aperture']:.1f}mm "
        f"lifted_fraction={info['lifted_fraction']:.2f} max_z={info['max_z']:.3f}m",
        flush=True,
    )


def run_headless(args: argparse.Namespace, model: PPO) -> None:
    successes = 0
    returns: list[float] = []
    lengths: list[int] = []
    env = RLCableGraspEnv(
        seed=args.seed,
        disturbance_strength=args.disturbance,
        episode_seconds=args.episode_seconds,
    )
    for episode in range(1, args.episodes + 1):
        observation, info = env.reset(seed=args.seed + episode - 1)
        episode_return = 0.0
        steps = 0
        while True:
            action, _ = model.predict(observation, deterministic=not args.stochastic)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            steps += 1
            if terminated or truncated:
                break
        successes += int(terminated)
        returns.append(episode_return)
        lengths.append(steps)
        print_episode(episode, episode_return, steps, info)
    env.close()
    print(
        f"episodes={args.episodes} successes={successes} "
        f"success_rate={successes / args.episodes:.1%} "
        f"mean_return={np.mean(returns):.3f} mean_steps={np.mean(lengths):.1f}",
        flush=True,
    )


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
    args = parser.parse_args()
    args.speed = float(np.clip(args.speed, 0.25, 8.0))
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    policy = PPO.load(arguments.model, device=arguments.device)
    if arguments.headless:
        run_headless(arguments, policy)
    else:
        run_viewer(arguments, policy)

