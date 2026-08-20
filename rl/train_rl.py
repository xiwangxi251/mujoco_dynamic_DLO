"""使用Stable-Baselines3 PPO训练动态线缆抓取策略。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from experiment_scenarios import list_scenario_names, list_suite_names
from project_paths import output_path
from .rl_cable_env import RLCableGraspEnv, RLConfig
from .rl_training_metrics import TrainingMetricsCallback, plot_training_curves


def linear_schedule(
    initial_value: float,
    final_value: float,
    *,
    start_progress_remaining: float = 1.0,
):
    """Return a clipped linear schedule for one ``learn`` invocation.

    Stable-Baselines3 measures progress against the model's cumulative timestep
    count when ``reset_num_timesteps=False``.  Normalising by the progress at
    the start of this invocation makes resume runs decay from the requested
    initial value to the requested final value over the newly requested steps,
    rather than jumping part-way through the schedule.
    """
    initial_value = float(initial_value)
    final_value = float(final_value)
    start_progress_remaining = float(start_progress_remaining)
    if start_progress_remaining <= 0.0:
        raise ValueError("start_progress_remaining must be positive")

    def schedule(progress_remaining: float) -> float:
        invocation_progress_remaining = np.clip(
            float(progress_remaining) / start_progress_remaining,
            0.0,
            1.0,
        )
        return float(
            final_value
            + (initial_value - final_value) * invocation_progress_remaining
        )

    return schedule


def make_worker(rank: int, args: argparse.Namespace):
    """返回可由Windows spawn进程安全构造的独立环境工厂。"""
    def initialize():
        initial_disturbance = (
            args.curriculum_start
            if args.curriculum_steps > 0
            else args.disturbance
        )
        env = RLCableGraspEnv(
            seed=args.seed + rank,
            disturbance_strength=initial_disturbance,
            episode_seconds=args.episode_seconds,
            scenario_names=args.training_scenario_names,
        )
        return env
    return initialize


class DisturbanceCurriculumCallback(BaseCallback):
    """逐步增加训练扰动；验证环境始终使用完整目标扰动。"""

    def __init__(self, start: float, end: float, steps: int, update_every: int = 10_000):
        super().__init__(verbose=0)
        self.start = float(start)
        self.end = float(end)
        self.steps = max(1, int(steps))
        self.update_every = max(1, int(update_every))
        self._last_update = -self.update_every

    def _apply(self) -> None:
        fraction = min(1.0, self.num_timesteps / self.steps)
        strength = self.start + fraction * (self.end - self.start)
        self.training_env.env_method("set_disturbance_strength", strength)
        self.logger.record("curriculum/disturbance_strength", strength)
        self._last_update = self.num_timesteps

    def _on_training_start(self) -> None:
        self._apply()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self.update_every:
            self._apply()
        return True


class EntropyCoefficientScheduleCallback(BaseCallback):
    """Linearly update PPO's entropy coefficient at rollout boundaries."""

    def __init__(self, initial: float, final: float, total_timesteps: int):
        super().__init__(verbose=0)
        self.initial = float(initial)
        self.final = float(final)
        self.total_timesteps = max(1, int(total_timesteps))
        self._start_timesteps = 0

    def _apply(self) -> None:
        elapsed = max(0, self.num_timesteps - self._start_timesteps)
        fraction = min(1.0, elapsed / self.total_timesteps)
        coefficient = self.initial + fraction * (self.final - self.initial)
        self.model.ent_coef = float(coefficient)
        self.logger.record("train/entropy_coefficient", coefficient)

    def _on_training_start(self) -> None:
        self._start_timesteps = self.num_timesteps
        self._apply()

    def _on_rollout_end(self) -> None:
        # PPO reads ent_coef immediately after the rollout, during train().
        self._apply()

    def _on_step(self) -> bool:
        return True


class StrictSuccessEvalCallback(BaseCallback):
    """在固定新种子上按严格成功率选最佳模型，而不是按训练回报选。"""

    def __init__(
        self,
        env: RLCableGraspEnv,
        output_dir: Path,
        *,
        eval_freq: int,
        episodes: int,
        seed: int,
    ):
        super().__init__(verbose=0)
        self.env = env
        self.output_dir = Path(output_dir)
        self.eval_freq = max(1, int(eval_freq))
        self.episodes = max(1, int(episodes))
        self.seed = int(seed)
        self.best_success_rate = -1.0
        self.best_mean_return = -np.inf
        self._last_eval = 0
        self.csv_path = self.output_dir / "strict_eval.csv"

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_eval = self.num_timesteps
        if self.csv_path.exists():
            with self.csv_path.open("r", encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    rate = float(row["success_rate"])
                    mean_return = float(row["mean_return"])
                    if (
                        rate > self.best_success_rate
                        or (
                            rate == self.best_success_rate
                            and mean_return > self.best_mean_return
                        )
                    ):
                        self.best_success_rate = rate
                        self.best_mean_return = mean_return
        else:
            with self.csv_path.open("w", encoding="utf-8", newline="") as target:
                csv.writer(target).writerow([
                    "timesteps", "success_rate", "pinch_rate", "grasp_rate",
                    "mean_return", "mean_length",
                ])

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        successes = 0
        pinches = 0
        grasps = 0
        returns: list[float] = []
        lengths: list[int] = []
        for episode in range(self.episodes):
            observation, info = self.env.reset(seed=self.seed + episode)
            episode_return = 0.0
            length = 0
            while True:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_return += reward
                length += 1
                if terminated or truncated:
                    break
            successes += int(bool(info["success"]))
            pinches += int(bool(info["ever_pinched"]))
            grasps += int(bool(info["ever_grasped"]))
            returns.append(episode_return)
            lengths.append(length)

        success_rate = successes / self.episodes
        pinch_rate = pinches / self.episodes
        grasp_rate = grasps / self.episodes
        mean_return = float(np.mean(returns))
        mean_length = float(np.mean(lengths))
        with self.csv_path.open("a", encoding="utf-8", newline="") as target:
            csv.writer(target).writerow([
                self.num_timesteps, success_rate, pinch_rate, grasp_rate,
                mean_return, mean_length,
            ])
        self.logger.record("eval/strict_success_rate", success_rate)
        self.logger.record("eval/pinch_rate", pinch_rate)
        self.logger.record("eval/secured_grasp_rate", grasp_rate)
        self.logger.record("eval/mean_return", mean_return)

        better = (
            success_rate > self.best_success_rate
            or (
                success_rate == self.best_success_rate
                and mean_return > self.best_mean_return
            )
        )
        if better:
            self.best_success_rate = success_rate
            self.best_mean_return = mean_return
            best_dir = self.output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(best_dir / "best_model")
        print(
            f"strict_eval timesteps={self.num_timesteps} "
            f"success={success_rate:.1%} pinch={pinch_rate:.1%} "
            f"grasp={grasp_rate:.1%} mean_return={mean_return:.3f}",
            flush=True,
        )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for dynamic cable grasping")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--disturbance", type=float, default=1.5)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument(
        "--training-distribution", choices=("legacy", "id"), default="legacy",
        help=(
            "legacy preserves the old shape-only task; id samples the frozen "
            "static/rigid/shape/combined ID scenario registry"
        ),
    )
    parser.add_argument(
        "--eval-distribution", choices=("legacy", "core", "id"), default=None,
        help="strict-eval distribution; defaults to the matching training distribution",
    )
    parser.add_argument(
        "--output", type=Path, default=output_path("rl", "runs", "ppo_cable_v3")
    )
    parser.add_argument("--checkpoint-steps", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=1024,
                        help="PPO rollout steps collected by each worker per update")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--learning-rate-initial", "--learning-rate",
        dest="learning_rate_initial", type=float, default=3e-4,
        help="learning rate at the start of this training invocation",
    )
    parser.add_argument(
        "--learning-rate-final", type=float, default=3e-5,
        help="learning rate at the end of this training invocation",
    )
    parser.add_argument(
        "--entropy-coef-initial", "--ent-coef",
        dest="entropy_coef_initial", type=float, default=0.01,
        help="entropy coefficient at the start of this training invocation",
    )
    parser.add_argument(
        "--entropy-coef-final", type=float, default=0.001,
        help="entropy coefficient at the end of this training invocation",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=5,
        help="optimization epochs per PPO rollout (lower is more conservative)",
    )
    parser.add_argument(
        "--target-kl", type=float, default=0.015,
        help="stop a PPO update early above this approximate KL; 0 disables",
    )
    parser.add_argument("--curriculum-start", type=float, default=0.30)
    parser.add_argument("--curriculum-steps", type=int, default=1_000_000,
                        help="0 disables disturbance curriculum")
    parser.add_argument("--curriculum-update-steps", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=100_000,
                        help="environment timesteps between strict evaluations; 0 disables")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=20270804)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--restart-schedules-on-resume", action="store_true",
        help=(
            "restart learning-rate and entropy schedules from their configured "
            "initial values; by default resume continues from checkpoint values"
        ),
    )
    args = parser.parse_args()
    args.training_scenario_names = (
        None
        if args.training_distribution == "legacy"
        else list(list_scenario_names("id"))
    )
    resolved_eval_distribution = args.eval_distribution or args.training_distribution
    if resolved_eval_distribution == "legacy":
        args.eval_scenario_names = None
    elif resolved_eval_distribution == "core":
        args.eval_scenario_names = list(list_suite_names("core"))
    else:
        args.eval_scenario_names = list(list_scenario_names("id"))
    args.eval_distribution = resolved_eval_distribution
    if args.timesteps < 1:
        parser.error("--timesteps must be positive")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.n_steps < 2 or args.batch_size < 2:
        parser.error("--n-steps and --batch-size must be at least 2")
    if args.batch_size > args.n_steps * args.workers:
        parser.error("--batch-size cannot exceed --n-steps * --workers")
    finite_values = (
        args.learning_rate_initial,
        args.learning_rate_final,
        args.entropy_coef_initial,
        args.entropy_coef_final,
        args.target_kl,
        args.disturbance,
        args.curriculum_start,
        args.episode_seconds,
    )
    if not all(np.isfinite(value) for value in finite_values):
        parser.error("floating-point training parameters must be finite")
    if args.learning_rate_initial <= 0.0 or args.learning_rate_final < 0.0:
        parser.error("learning-rate initial must be positive and final non-negative")
    if args.learning_rate_final > args.learning_rate_initial:
        parser.error("--learning-rate-final cannot exceed --learning-rate-initial")
    if args.entropy_coef_initial < 0.0 or args.entropy_coef_final < 0.0:
        parser.error("entropy coefficients must be non-negative")
    if args.entropy_coef_final > args.entropy_coef_initial:
        parser.error("--entropy-coef-final cannot exceed --entropy-coef-initial")
    if args.n_epochs < 1:
        parser.error("--n-epochs must be positive")
    if args.target_kl < 0.0:
        parser.error("--target-kl must be non-negative")
    if args.disturbance < 0.0 or args.curriculum_start < 0.0:
        parser.error("disturbance strengths must be non-negative")
    if args.episode_seconds <= 0.0:
        parser.error("--episode-seconds must be positive")
    if args.checkpoint_steps < 1:
        parser.error("--checkpoint-steps must be positive")
    if args.curriculum_steps < 0 or args.curriculum_update_steps < 1:
        parser.error("curriculum steps must be non-negative and update steps positive")
    if args.eval_freq < 0 or args.eval_episodes < 1:
        parser.error("--eval-freq must be non-negative and --eval-episodes positive")
    if args.resume is not None:
        resume_with_zip = Path(f"{args.resume}.zip")
        if not args.resume.is_file() and not resume_with_zip.is_file():
            parser.error(f"--resume checkpoint does not exist: {args.resume}")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    target_kl = args.target_kl if args.target_kl > 0.0 else None

    # 启动大批量训练前先检查一次Gymnasium API、shape和数据类型。
    check_candidate = RLCableGraspEnv(
        seed=args.seed,
        disturbance_strength=args.disturbance,
        episode_seconds=args.episode_seconds,
        scenario_names=args.training_scenario_names,
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
        info_keywords=(
            "success", "ever_pinched", "ever_grasped", "lifted_fraction",
        ),
    )

    if args.resume is None:
        starting_timesteps = 0
        learning_rate_initial = args.learning_rate_initial
        learning_rate_final = args.learning_rate_final
        entropy_coef_initial = args.entropy_coef_initial
        entropy_coef_final = args.entropy_coef_final
        resume_schedule_mode = "not_applicable"
        learning_rate = linear_schedule(
            learning_rate_initial,
            learning_rate_final,
        )
        model = PPO(
            "MlpPolicy",
            vector_env,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=entropy_coef_initial,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=target_kl,
            policy_kwargs={"net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]}},
            tensorboard_log=str(args.output / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
    else:
        # Explicit overrides make resumed training use this invocation's CLI
        # settings instead of silently inheriting unstable legacy defaults.
        model = PPO.load(
            args.resume,
            env=vector_env,
            device=args.device,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            target_kl=target_kl,
            tensorboard_log=str(args.output / "tensorboard"),
        )
        starting_timesteps = int(model.num_timesteps)
        checkpoint_learning_rate = float(model.policy.optimizer.param_groups[0]["lr"])
        checkpoint_entropy_coef = float(model.ent_coef)
        if not np.isfinite(checkpoint_learning_rate) or not np.isfinite(checkpoint_entropy_coef):
            vector_env.close()
            raise ValueError("resume checkpoint contains non-finite PPO schedule values")
        if args.restart_schedules_on_resume:
            learning_rate_initial = args.learning_rate_initial
            entropy_coef_initial = args.entropy_coef_initial
            resume_schedule_mode = "restart_from_configured_initial"
        else:
            learning_rate_initial = checkpoint_learning_rate
            entropy_coef_initial = checkpoint_entropy_coef
            resume_schedule_mode = "continue_from_checkpoint"

        # A continuation must never increase exploration/update size merely
        # because a checkpoint already went below the configured final value.
        learning_rate_final = min(args.learning_rate_final, learning_rate_initial)
        entropy_coef_final = min(args.entropy_coef_final, entropy_coef_initial)
        model.ent_coef = entropy_coef_initial
        cumulative_target = starting_timesteps + args.timesteps
        start_progress_remaining = args.timesteps / cumulative_target
        learning_rate = linear_schedule(
            learning_rate_initial,
            learning_rate_final,
            start_progress_remaining=start_progress_remaining,
        )
        # load() has already rebuilt the rollout buffer using the CLI overrides;
        # only the learning-rate callable needs replacing after num_timesteps is known.
        model.learning_rate = learning_rate
        model.lr_schedule = learning_rate

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_steps // args.workers),
        save_path=str(args.output / "checkpoints"),
        name_prefix="ppo_cable",
    )
    metrics_path = args.output / "training_metrics.csv"
    metrics_callback = TrainingMetricsCallback(metrics_path, window=100)
    callback_items: list[BaseCallback] = [
        checkpoint_callback,
        metrics_callback,
        EntropyCoefficientScheduleCallback(
            entropy_coef_initial,
            entropy_coef_final,
            args.timesteps,
        ),
    ]
    if args.curriculum_steps > 0:
        callback_items.append(DisturbanceCurriculumCallback(
            args.curriculum_start,
            args.disturbance,
            args.curriculum_steps,
            args.curriculum_update_steps,
        ))
    eval_env: RLCableGraspEnv | None = None
    if args.eval_freq > 0:
        eval_env = RLCableGraspEnv(
            seed=args.eval_seed,
            disturbance_strength=args.disturbance,
            episode_seconds=args.episode_seconds,
            scenario_names=args.eval_scenario_names,
        )
        callback_items.append(StrictSuccessEvalCallback(
            eval_env,
            args.output / "evaluation",
            eval_freq=args.eval_freq,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
        ))
    callbacks = CallbackList(callback_items)
    configuration = {
        "algorithm": "PPO",
        "grasp_model": "physical_friction_v1",
        "output": str(args.output.resolve()),
        "timesteps": args.timesteps,
        "timesteps_semantics": "additional timesteps in this learn invocation",
        "starting_timesteps": starting_timesteps,
        "cumulative_target_timesteps": starting_timesteps + args.timesteps,
        "resume_checkpoint": str(args.resume.resolve()) if args.resume is not None else None,
        "reset_num_timesteps": args.resume is None,
        "workers": args.workers,
        "seed": args.seed,
        "disturbance": args.disturbance,
        "training_distribution": args.training_distribution,
        "training_scenario_names": args.training_scenario_names,
        "eval_distribution": args.eval_distribution,
        "eval_scenario_names": args.eval_scenario_names,
        "episode_seconds": args.episode_seconds,
        "checkpoint_steps": args.checkpoint_steps,
        "device_requested": args.device,
        "device_resolved": str(model.device),
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "learning_rate_initial": learning_rate_initial,
        "learning_rate_final": learning_rate_final,
        "learning_rate_initial_requested": args.learning_rate_initial,
        "learning_rate_final_requested": args.learning_rate_final,
        "learning_rate_schedule": "linear over this learn invocation",
        "entropy_coef_initial": entropy_coef_initial,
        "entropy_coef_final": entropy_coef_final,
        "entropy_coef_initial_requested": args.entropy_coef_initial,
        "entropy_coef_final_requested": args.entropy_coef_final,
        "entropy_coef_schedule": "linear over this learn invocation; updated per rollout",
        "resume_schedule_mode": resume_schedule_mode,
        "restart_schedules_on_resume_requested": args.restart_schedules_on_resume,
        "n_epochs": args.n_epochs,
        "target_kl": target_kl,
        "target_kl_requested": args.target_kl,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
        "curriculum_start": args.curriculum_start,
        "curriculum_steps": args.curriculum_steps,
        "curriculum_update_steps": args.curriculum_update_steps,
        "curriculum_final_disturbance": args.disturbance,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "rl_grasp_definition": (
            "bilateral confirmed pinch + 30 mm lift from pinch height + "
            "0.10 s continuous physical retention"
        ),
        "rl_success_definition": (
            "secured grasp + body z > 0.14 m + lifted fraction >= 0.18 + "
            "distance <= 0.055 m at 50 Hz for 0.80 s, intersected with the "
            "base environment's current confirmed-grasp and 500 Hz continuous "
            "0.80 s geometric task-success window"
        ),
        "observation_names": list(RLCableGraspEnv.OBSERVATION_NAMES),
        "rl_config": asdict(RLConfig()),
        "reward": (
            "one-shot contact/pinch/secured bonuses + capped episode high-water lift "
            "credit + strict terminal reward + distinct active-open/physical-slip "
            "loss causes with separate one-shot gates for contact loss and active "
            "opening + capped strict-hold progress + post-grasp arm magnitude/rate "
            "penalties"
        ),
        "action": (
            "7 normalized joint-target deltas (0.04 rad scale) + "
            "1 hysteretic gripper command"
        ),
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
        if eval_env is not None:
            eval_env.close()

    print(f"model_saved={args.output / 'final_model.zip'}", flush=True)


if __name__ == "__main__":
    # Windows多进程必须把入口放在此保护块内，避免子进程重复启动训练器。
    main()
