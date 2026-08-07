"""使用Stable-Baselines3 PPO训练动态线缆抓取策略。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from rl_cable_env import RLCableGraspEnv
from rl_training_metrics import TrainingMetricsCallback, plot_training_curves


def make_worker(rank: int, args: argparse.Namespace):
    """返回可由Windows spawn进程安全构造的独立环境工厂。"""
    def initialize():
        env = RLCableGraspEnv(
            seed=args.seed + rank,
            disturbance_strength=args.disturbance,
            episode_seconds=args.episode_seconds,
        )
        return env
    return initialize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for dynamic cable grasping")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--disturbance", type=float, default=1.5)
    parser.add_argument("--episode-seconds", type=float, default=28.0)
    parser.add_argument("--output", type=Path, default=Path("runs") / "ppo_cable")
    parser.add_argument("--checkpoint-steps", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=1024,
                        help="PPO rollout steps collected by each worker per update")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.n_steps < 2 or args.batch_size < 2:
        parser.error("--n-steps and --batch-size must be at least 2")
    if args.batch_size > args.n_steps * args.workers:
        parser.error("--batch-size cannot exceed --n-steps * --workers")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    # 启动大批量训练前先检查一次Gymnasium API、shape和数据类型。
    check_candidate = RLCableGraspEnv(
        seed=args.seed,
        disturbance_strength=args.disturbance,
        episode_seconds=args.episode_seconds,
    )
    check_env(check_candidate, warn=True)
    check_candidate.close()

    factories = [make_worker(rank, args) for rank in range(args.workers)]
    if args.workers == 1:
        vector_env = DummyVecEnv(factories)
    else:
        vector_env = SubprocVecEnv(factories, start_method="spawn")
    vector_env = VecMonitor(
        vector_env,
        filename=str(args.output / "monitor.csv"),
        info_keywords=("success", "ever_grasped", "lifted_fraction"),
    )

    if args.resume is None:
        model = PPO(
            "MlpPolicy",
            vector_env,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]}},
            tensorboard_log=str(args.output / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
    else:
        model = PPO.load(args.resume, env=vector_env, device=args.device)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_steps // args.workers),
        save_path=str(args.output / "checkpoints"),
        name_prefix="ppo_cable",
    )
    metrics_path = args.output / "training_metrics.csv"
    metrics_callback = TrainingMetricsCallback(metrics_path, window=100)
    callbacks = CallbackList([checkpoint_callback, metrics_callback])
    configuration = {
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "workers": args.workers,
        "seed": args.seed,
        "disturbance": args.disturbance,
        "episode_seconds": args.episode_seconds,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "observation_names": list(RLCableGraspEnv.OBSERVATION_NAMES),
        "action": "7 normalized joint-target deltas + 1 normalized gripper target",
    }
    (args.output / "training_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=args.resume is None,
        )
        model.save(args.output / "final_model")
        plot_training_curves(metrics_path, args.output, window=100)
    finally:
        vector_env.close()

    print(f"model_saved={args.output / 'final_model.zip'}", flush=True)


if __name__ == "__main__":
    # Windows多进程必须把入口放在此保护块内，避免子进程重复启动训练器。
    main()
