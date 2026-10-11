"""使用Stable-Baselines3 PPO训练动态线缆抓取策略。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)

from ..scenarios.registry import list_scenario_names
from ..env.environment import ROBOT_SPECS
from ..paths import output_path
from .environment import (
    RLCableGraspEnv,
    RLConfig,
    action_interface_version,
    make_rl_env,
)
from .metrics import TrainingMetricsCallback, plot_training_curves
from .pointcloud import (
    DLOPointCloudObservation,
    PointCloudObservationConfig,
    PointNet2FeaturesExtractor,
)


RL_L1_SCENARIOS = (
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
)
RL_L1_CURRICULUM_STAGES = (
    ("id_static",),
    ("id_shape_nominal_current", "id_rigid_l1_nominal"),
    ("id_combined_l1_nominal",),
)
RL_INTERFACE_VERSION = "baseline_v4"


def build_l1_curriculum_training_mixes(
    stage1_static_replay: float = 0.20,
    stage2_static_replay: float = 0.10,
    stage2_component_replay: float = 0.20,
) -> tuple[tuple[tuple[str, float], ...], ...]:
    """Build stage-wise training distributions with easier-scene rehearsal."""
    fractions = (
        float(stage1_static_replay),
        float(stage2_static_replay),
        float(stage2_component_replay),
    )
    if not all(
        np.isfinite(value) and 0.0 <= value <= 1.0
        for value in fractions
    ):
        raise ValueError("curriculum replay fractions must be finite and in [0, 1]")
    if stage1_static_replay >= 1.0:
        raise ValueError("stage-1 static replay fraction must be less than 1")
    if stage2_static_replay + stage2_component_replay >= 1.0:
        raise ValueError(
            "stage-2 static and component replay fractions must sum to less than 1"
        )

    stage1_component = (1.0 - stage1_static_replay) / 2.0
    stage2_component = stage2_component_replay / 2.0
    return (
        (("id_static", 1.0),),
        (
            ("id_static", stage1_static_replay),
            ("id_shape_nominal_current", stage1_component),
            ("id_rigid_l1_nominal", stage1_component),
        ),
        (
            ("id_static", stage2_static_replay),
            ("id_shape_nominal_current", stage2_component),
            ("id_rigid_l1_nominal", stage2_component),
            (
                "id_combined_l1_nominal",
                1.0 - stage2_static_replay - stage2_component_replay,
            ),
        ),
    )


RL_L1_CURRICULUM_TRAINING_MIXES = build_l1_curriculum_training_mixes()


def checkpoint_interface_version(checkpoint: Path) -> str | None:
    """Find the nearest training manifest associated with a checkpoint."""
    checkpoint = Path(checkpoint).resolve()
    for directory in (checkpoint.parent, *checkpoint.parents):
        config_path = directory / "training_config.json"
        if not config_path.is_file():
            continue
        with config_path.open("r", encoding="utf-8") as source:
            return str(json.load(source).get("rl_interface_version", "")) or None
    return None


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


def cumulative_linear_schedule(initial, final, horizon, cumulative_target):
    """SB3 progress measures cumulative samples against this invocation's target."""
    if horizon <= 0 or cumulative_target <= 0:raise ValueError('Schedule targets must be positive')
    def schedule(progress_remaining):
        samples = cumulative_target * (1 - float(progress_remaining))
        fraction = float(np.clip(samples / horizon, 0, 1))
        return float(initial + (final - initial) * fraction)
    return schedule


def resolve_global_schedule(args, saved, starting_timesteps):
    if saved:
        if args.restart_schedules_on_resume:
            raise ValueError('Cannot restart a saved global schedule')
        if args.schedule_total_timesteps and args.schedule_total_timesteps != saved['total_timesteps']:
            raise ValueError('Global schedule horizon differs from checkpoint')
        result = dict(saved)
    elif args.schedule_total_timesteps:
        result = dict(version='cumulative_samples_v1', total_timesteps=args.schedule_total_timesteps,
            learning_rate_initial=args.learning_rate_initial, learning_rate_final=args.learning_rate_final,
            entropy_coef_initial=args.entropy_coef_initial, entropy_coef_final=args.entropy_coef_final)
    else:return None
    if starting_timesteps + args.timesteps > result['total_timesteps']:
        raise ValueError('Requested sample budget exceeds global schedule horizon')
    return result


def rl_config_from_args(args: argparse.Namespace) -> RLConfig:
    """Build one immutable-by-convention reward/action configuration per env."""
    reference = getattr(args, "rl_config_file", None)
    if reference is not None:
        from dataclasses import fields
        saved = json.loads(Path(reference).read_text(encoding="utf-8"))
        saved = saved.get("rl_config", saved)
        accepted = {field.name for field in fields(RLConfig)}
        values = {key: value for key, value in saved.items() if key in accepted}
        values["target_reward_profile"] = getattr(args, "target_reward_profile", "legacy")
        if values["target_reward_profile"] in {"tolerance_progress_v2", "tolerance_progress_v3", "tolerance_progress_v4", "tolerance_progress_v5", "tolerance_progress_v6"}:
            values.update(target_node_task=True, target_node_tolerance=2,
                          target_position_reward_radius=4,
                          target_wrong_event_penalty=2., target_wrong_event_cap=8.)
        if getattr(args, 'ordinary_reward_profile', 'legacy') != 'legacy':
            values['ordinary_reward_profile'] = args.ordinary_reward_profile
        if getattr(args, 'action_workspace', None):
            values['action_workspace_path'] = str(args.action_workspace.resolve())
            values['action_workspace_sha256'] = __import__('hashlib').sha256(args.action_workspace.read_bytes()).hexdigest()
        return RLConfig(**values)
    return RLConfig(
        ordinary_reward_profile=getattr(args, "ordinary_reward_profile", "legacy"),
        action_workspace_path=str(args.action_workspace.resolve()) if getattr(args, 'action_workspace', None) else None,
        action_workspace_sha256=(__import__('hashlib').sha256(args.action_workspace.read_bytes()).hexdigest() if getattr(args, 'action_workspace', None) else None),
        target_reward_profile=getattr(args, "target_reward_profile", "legacy"),
        target_node_task=getattr(args, "target_node_task", False),
        target_u_min=getattr(args, "target_u_min", 1.0 / 6.0),
        target_u_max=getattr(args, "target_u_max", 5.0 / 6.0),
        target_node_tolerance=getattr(args, "target_node_tolerance", 2),
        singularity_avoidance_enabled=(
            not args.disable_singularity_avoidance
        ),
        height_band_reward_enabled=(not args.disable_height_band_reward),
        height_band_low_m=args.height_band_low_m,
        height_band_high_m=args.height_band_high_m,
        reward_overheight_progress=args.reward_overheight_progress,
        reward_overheight_step=args.reward_overheight_step,
        overheight_full_scale_m=args.overheight_full_scale_m,
        overheight_step_penalty_cap=args.overheight_step_penalty_cap,
        success_overheight_penalty_per_m=(
            args.success_overheight_penalty_per_m
        ),
        success_overheight_penalty_cap=args.success_overheight_penalty_cap,
        height_penalty_ramp_env_steps=args.height_penalty_ramp_env_steps,
        safety_reward_enabled=args.enable_safety_penalties,
        reward_table_filter_step=args.reward_table_filter_step,
        reward_table_contact_step=args.reward_table_contact_step,
        table_safety_penalty_cap=args.table_safety_penalty_cap,
        reward_singularity_step=args.reward_singularity_step,
        singularity_safety_penalty_cap=args.singularity_safety_penalty_cap,
        safety_penalty_ramp_env_steps=args.safety_penalty_ramp_env_steps,
    )


def pointcloud_config_from_args(
    args: argparse.Namespace,
) -> PointCloudObservationConfig:
    return PointCloudObservationConfig(
        point_count=args.pointcloud_points,
        width=args.pointcloud_width,
        height=args.pointcloud_height,
        camera_update_steps=args.pointcloud_update_steps,
        sensor_delay_steps=args.pointcloud_delay_steps,
        voxel_size_m=args.pointcloud_voxel_size,
        render_mode=args.pointcloud_render_mode,
    )


def make_configured_env(
    args: argparse.Namespace,
    *,
    seed: int,
    scenario_names: tuple[str, ...],
):
    pointcloud = args.observation_mode == "pointcloud"
    env = make_rl_env(
        action_mode=args.action_mode,
        robot=getattr(args, "robot", "panda"),
        seed=seed,
        disturbance_strength=args.disturbance,
        episode_seconds=args.episode_seconds,
        dynamicvla_cameras_enabled=pointcloud,
        scenario_names=scenario_names,
        rl_config=rl_config_from_args(args),
        geometric_safety_enabled=args.geometric_safety,
        table_finger_collision_filter_enabled=(
            not args.disable_table_finger_collision_filter
        ),
        controller_profile=getattr(args, "controller_profile", "legacy"),
        table_execution_guard_enabled=getattr(args, "table_execution_guard", False),
        target_workspace=getattr(args, "target_workspace", None),
        workspace_mask_key=getattr(args, "workspace_mask_key", "interior"),
        workspace_entry_seconds=getattr(args, "workspace_entry_seconds", 8.),
        workspace_start_protocol=getattr(args, "workspace_start_protocol", "policy_from_initial_v2"),
        workspace_reset_attempts=getattr(args, "workspace_reset_attempts", 50),
        workspace_reset_seed_pool=getattr(args, "workspace_reset_seed_pool", None),
        replay_bank_dir=getattr(args, "replay_bank_dir", None),
        replay_seed_fallback=getattr(args, "replay_seed_fallback", "resample"),
    )
    env.base_env.config.contact_read_cache_enabled = getattr(args, 'contact_read_cache', True)
    if getattr(args, "workspace_curriculum", False):
        from .workspace import WorkspaceCurriculumCallback
        # SB3 resets environments before callback startup; resumed first episodes
        # must already use the restored curriculum distribution.
        curriculum = WorkspaceCurriculumCallback(
            args.training_scenario_names, args.output / "evaluation/workspace_curriculum.json",
            resume=getattr(args, "resume", None) is not None,
        )
        stage = curriculum.stages[curriculum.stage_index]
        env.set_training_scenario_distribution(tuple(args.training_scenario_names), stage[1])
        env.set_motion_difficulty(stage[2])
    if pointcloud:
        return DLOPointCloudObservation(env, pointcloud_config_from_args(args))
    return env


def make_worker(rank: int, args: argparse.Namespace):
    """返回可由Windows spawn进程安全构造的独立环境工厂。"""
    def initialize():
        from .affinity import bind_worker
        bind_worker(args, rank)
        env = make_configured_env(
            args,
            seed=args.seed + rank,
            scenario_names=args.training_scenario_names,
        )
        if (
            args.training_distribution == "l1"
            and args.curriculum_stage_steps > 0
        ):
            if getattr(args, 'rollout_ability_curriculum', False):
                from .rollout_curriculum import configure_env
                configure_env(env, args.rollout_initial_phase)
                return env
            initial_mix = args.curriculum_training_mixes[0]
            env.set_training_scenario_distribution(
                tuple(name for name, _ in initial_mix),
                tuple(probability for _, probability in initial_mix),
            )
            env.set_motion_difficulty(0.0)
        elif args.training_scenario_weights is not None:
            env.set_training_scenario_distribution(
                tuple(args.training_scenario_names),
                tuple(args.training_scenario_weights),
            )
        return env
    return initialize


def make_eval_worker(rank: int, args: argparse.Namespace):
    """Build one deterministic strict-evaluation environment."""
    def initialize():
        from .affinity import bind_worker
        bind_worker(args, rank, evaluation=True)
        from copy import copy
        eval_args = copy(args)
        if getattr(args, "eval_replay_bank_dir", None) is not None:
            eval_args.replay_bank_dir = args.eval_replay_bank_dir
            eval_args.replay_seed_fallback = "error"
        if getattr(args, "eval_seed_pool_size", 0) > 0:
            eval_args.workspace_reset_seed_pool = (args.eval_seed, args.eval_seed_pool_size)
        return make_configured_env(
            eval_args,
            seed=args.eval_seed + rank,
            scenario_names=args.eval_scenario_names,
        )
    return initialize


class MotionCurriculumCallback(BaseCallback):
    """Advance static -> component motions -> combined L1 from strict evals."""

    def __init__(
        self,
        stages: tuple[tuple[str, ...], ...],
        update_every: int = 10_000,
        required_evaluations: int = 2,
        aligned_pinch_rate: float = 0.50,
        secured_rate: float = 0.20,
        success_rate: float = 0.10,
        training_mixes: tuple[tuple[tuple[str, float], ...], ...] | None = None,
    ):
        super().__init__(verbose=0)
        if not stages:
            raise ValueError("motion curriculum requires at least one stage")
        self.stages = stages
        self.update_every = max(1, int(update_every))
        self.required_evaluations = max(1, int(required_evaluations))
        self.aligned_pinch_rate = float(aligned_pinch_rate)
        self.secured_rate = float(secured_rate)
        self.success_rate = float(success_rate)
        self.training_mixes = (
            RL_L1_CURRICULUM_TRAINING_MIXES
            if training_mixes is None
            else training_mixes
        )
        if len(self.training_mixes) != len(stages):
            raise ValueError("curriculum training mixes must match the stages")
        for stage_index, mix in enumerate(self.training_mixes):
            names = tuple(name for name, _ in mix)
            probabilities = np.asarray(
                tuple(probability for _, probability in mix), dtype=float
            )
            if not names or len(names) != len(set(names)):
                raise ValueError(
                    "each curriculum training mix must be non-empty and unique"
                )
            if not set(stages[stage_index]).issubset(names):
                raise ValueError(
                    "each training mix must include its target scenarios"
                )
            if (
                probabilities.shape != (len(names),)
                or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0.0)
                or not np.isclose(float(np.sum(probabilities)), 1.0)
            ):
                raise ValueError("curriculum training mix probabilities must sum to 1")
            if sum(
                probability
                for name, probability in mix
                if name in stages[stage_index]
            ) <= 0.0:
                raise ValueError("target scenarios must have positive training mass")
        self.phases = (
            (0, 0.0),
            *((stage, difficulty) for stage in range(1, len(stages))
              for difficulty in (1.0 / 3.0, 2.0 / 3.0, 1.0)),
        )
        self.phase_index = 0
        self.qualifying_evaluations = 0
        self._last_update = -self.update_every

    @property
    def stage_index(self) -> int:
        return int(self.phases[self.phase_index][0])

    @property
    def difficulty(self) -> float:
        return float(self.phases[self.phase_index][1])

    @property
    def current_scenarios(self) -> tuple[str, ...]:
        """Target scenarios used for strict evaluation and phase advancement."""
        return self.stages[self.stage_index]

    @property
    def current_training_mix(self) -> tuple[tuple[str, float], ...]:
        return self.training_mixes[self.stage_index]

    def _apply(self) -> None:
        self.training_env.env_method(
            "set_training_scenario_distribution",
            tuple(name for name, _ in self.current_training_mix),
            tuple(probability for _, probability in self.current_training_mix),
        )
        self.training_env.env_method("set_motion_difficulty", self.difficulty)
        self.logger.record("curriculum/phase", self.phase_index)
        self.logger.record("curriculum/stage", self.stage_index)
        self.logger.record("curriculum/motion_difficulty", self.difficulty)
        self.logger.record(
            "curriculum/target_scenario_fraction",
            sum(
                probability
                for name, probability in self.current_training_mix
                if name in self.current_scenarios
            ),
        )
        self.logger.record(
            "curriculum/qualifying_evaluations", self.qualifying_evaluations
        )
        self.model.ordinary_evaluation_scenarios = self.current_scenarios
        self.model.ordinary_evaluation_difficulty = self.difficulty
        self.model.ordinary_evaluation_phase = self.phase_index
        self._last_update = self.num_timesteps

    def configure_evaluation_env(self, env: RLCableGraspEnv | VecEnv) -> None:
        if isinstance(env, VecEnv):
            env.env_method("set_training_scenarios", self.current_scenarios)
            env.env_method("set_motion_difficulty", self.difficulty)
            return
        env.set_training_scenarios(self.current_scenarios)
        env.set_motion_difficulty(self.difficulty)

    def restore_phase(self, phase_index: int) -> None:
        phase_index = int(phase_index)
        if not 0 <= phase_index < len(self.phases):
            raise ValueError(f"invalid curriculum phase: {phase_index}")
        self.phase_index = phase_index
        self.qualifying_evaluations = 0
        self._apply()

    def evaluation_qualifies(self, metrics: dict[str, float]) -> bool:
        return bool(
            metrics["aligned_pinch_rate"] >= self.aligned_pinch_rate
            and metrics["grasp_rate"] >= self.secured_rate
            and metrics["success_rate"] >= self.success_rate
        )

    def observe_evaluation(self, metrics: dict[str, float]) -> None:
        if self.phase_index == len(self.phases) - 1:
            return
        if self.evaluation_qualifies(metrics):
            self.qualifying_evaluations += 1
        else:
            self.qualifying_evaluations = 0
        if self.qualifying_evaluations < self.required_evaluations:
            self._apply()
            return
        self.phase_index += 1
        self.qualifying_evaluations = 0
        self._apply()
        print(
            f"curriculum_phase={self.phase_index} stage={self.stage_index} "
            f"difficulty={self.difficulty:.3f} "
            f"eval_scenarios={','.join(self.current_scenarios)} "
            "training_mix="
            + ",".join(
                f"{name}:{probability:.2f}"
                for name, probability in self.current_training_mix
            ),
            flush=True,
        )

    def _on_training_start(self) -> None:
        self._apply()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self.update_every:
            self._apply()
        return True


class EntropyCoefficientScheduleCallback(BaseCallback):
    """Linearly update PPO's entropy coefficient at rollout boundaries."""

    def __init__(self, initial: float, final: float, total_timesteps: int, *, cumulative: bool = False):
        super().__init__(verbose=0)
        self.initial = float(initial)
        self.final = float(final)
        self.total_timesteps = max(1, int(total_timesteps))
        self._start_timesteps = 0
        self.cumulative = cumulative

    def _apply(self) -> None:
        elapsed = max(0, self.num_timesteps - self._start_timesteps)
        fraction = min(1.0, elapsed / self.total_timesteps)
        coefficient = self.initial + fraction * (self.final - self.initial)
        self.model.ent_coef = float(coefficient)
        self.logger.record("train/entropy_coefficient", coefficient)

    def _on_training_start(self) -> None:
        self._start_timesteps = 0 if self.cumulative else self.num_timesteps
        self._apply()

    def _on_rollout_end(self) -> None:
        # PPO reads ent_coef immediately after the rollout, during train().
        self._apply()

    def _on_step(self) -> bool:
        return True


class StrictSuccessEvalCallback(BaseCallback):
    """在固定新种子上按公共任务成功率选最佳模型并记录严格漏斗。"""

    def __init__(
        self,
        env: VecEnv,
        output_dir: Path,
        *,
        eval_freq: int,
        episodes: int,
        confirmation_episodes: int,
        seed: int,
        curriculum: MotionCurriculumCallback | None = None,
        seed_pool_size: int = 0,
    ):
        super().__init__(verbose=0)
        self.env = env
        self.output_dir = Path(output_dir)
        self.eval_freq = max(1, int(eval_freq))
        self.episodes = max(1, int(episodes))
        self.confirmation_requested_episodes = max(1, int(confirmation_episodes))
        self.seed_pool_size = int(seed_pool_size)
        if self.seed_pool_size < 0:
            raise ValueError("evaluation seed pool size must be nonnegative")
        if self.seed_pool_size and self.seed_pool_size <= self.episodes:
            raise ValueError("evaluation seed pool must leave seeds for confirmation")
        # Bank-backed evaluations must use the unused part of the finite
        # held-out pool, never fabricated +100000 seeds or routine duplicates.
        self.confirmation_seed_offset = self.episodes if self.seed_pool_size else 100_000
        self.confirmation_episodes = (
            min(self.confirmation_requested_episodes, self.seed_pool_size - self.episodes)
            if self.seed_pool_size else self.confirmation_requested_episodes
        )
        self.seed = int(seed)
        self.curriculum = curriculum
        self.eval_workers = int(env.num_envs)
        if self.eval_workers < 1:
            raise ValueError("strict evaluation requires at least one worker")
        # Do not spend confirmation episodes on a policy with no task success
        # and no secured grasp merely because its shaped return moved.
        self.best_success_rate = 0.0
        self.best_grasp_rate = 0.0
        self.best_aligned_pinch_rate = 0.0
        self.best_mean_return = -np.inf
        self.best_phase = -1
        self._last_eval = 0
        self.csv_path = self.output_dir / "strict_eval.csv"

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_eval = self.num_timesteps
        previous_rows: list[dict[str, str]] = []
        if self.csv_path.exists():
            with self.csv_path.open("r", encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    previous_rows.append(row)
                    rate = float(row["success_rate"])
                    grasp_rate = float(row["grasp_rate"])
                    aligned_pinch_rate = float(row.get("aligned_pinch_rate", 0.0))
                    mean_return = float(row["mean_return"])
                    phase = int(row.get("curriculum_phase", 0))
                    score = (
                        phase, rate, grasp_rate, aligned_pinch_rate, mean_return
                    )
                    best_score = (
                        self.best_phase, self.best_success_rate,
                        self.best_grasp_rate, self.best_aligned_pinch_rate,
                        self.best_mean_return,
                    )
                    if score > best_score:
                        self.best_phase = phase
                        self.best_success_rate = rate
                        self.best_grasp_rate = grasp_rate
                        self.best_aligned_pinch_rate = aligned_pinch_rate
                        self.best_mean_return = mean_return
            if previous_rows and "strict_success_rate" not in previous_rows[0]:
                raise RuntimeError(
                    f"Existing evaluation metrics predate the shared task-success "
                    f"contract: {self.csv_path}. Start training in a new output "
                    "directory."
                )
        else:
            with self.csv_path.open("w", encoding="utf-8", newline="") as target:
                csv.writer(target).writerow([
                    "timesteps", "curriculum_phase", "next_curriculum_phase",
                    "curriculum_stage",
                    "motion_difficulty", "success_rate", "strict_success_rate",
                    "pinch_rate",
                    "aligned_pinch_rate", "lift_attempt_rate", "loaded_lift_rate",
                    "grasp_rate", "physical_slip_rate", "mean_return",
                    "mean_length", "episodes",
                ])
        if self.curriculum is not None and previous_rows:
            self.curriculum.restore_phase(int(
                previous_rows[-1].get(
                    "next_curriculum_phase",
                    previous_rows[-1].get("curriculum_phase", 0),
                )
            ))
        confirmation_path = self.output_dir / "best" / "confirmation.json"
        if confirmation_path.is_file():
            with confirmation_path.open("r", encoding="utf-8") as source:
                confirmed = json.load(source)
            self.best_success_rate = float(confirmed["success_rate"])
            self.best_grasp_rate = float(confirmed["grasp_rate"])
            self.best_aligned_pinch_rate = float(
                confirmed.get("aligned_pinch_rate", 0.0)
            )
            self.best_mean_return = float(confirmed["mean_return"])
            self.best_phase = int(confirmed.get("curriculum_phase", 0))

    def _evaluate(self, episodes: int, *, seed_offset: int = 0) -> dict[str, float]:
        from .penalty_ramp import apply_progress
        if hasattr(self.model, 'penalty_ramp_schedule'):
            apply_progress(self.env, self.model, self.model.penalty_ramp_schedule)
        successes = 0
        strict_successes = 0
        pinches = 0
        aligned_pinches = 0
        lift_attempts = 0
        loaded_lifts = 0
        grasps = 0
        first_correct = 0
        slips = 0
        returns: list[float] = []
        lengths: list[int] = []
        completed = 0
        print(
            f"strict_eval_start episodes={episodes} workers={self.eval_workers}",
            flush=True,
        )
        while completed < episodes:
            batch_size = min(self.eval_workers, episodes - completed)
            batch_seed = self.seed + seed_offset + completed
            # VecEnv.seed(N) assigns N + rank on the following reset. Each
            # requested episode therefore keeps the exact seeds used by the
            # former serial evaluator, independent of worker count.
            self.env.seed(batch_seed)
            if self.seed_pool_size:
                if seed_offset < 0 or seed_offset + completed + batch_size > self.seed_pool_size:
                    raise ValueError("active evaluation seeds exceed the held-out pool")
                # VecEnv.reset resets inactive workers as well. SB3 stores the
                # pending per-worker reset seeds in _seeds; wrap only the unused
                # worker slots, preserving every counted episode's exact seed.
                self.env._seeds = [
                    self.seed + (seed_offset + completed + rank) % self.seed_pool_size
                    for rank in range(self.eval_workers)
                ]
            observations = self.env.reset()
            active = np.arange(self.eval_workers) < batch_size
            batch_returns = np.zeros(self.eval_workers, dtype=np.float64)
            batch_lengths = np.zeros(self.eval_workers, dtype=np.int64)

            while bool(np.any(active)):
                actions, _ = self.model.predict(
                    observations, deterministic=True
                )
                observations, rewards, dones, infos = self.env.step(actions)
                batch_returns[active] += np.asarray(rewards)[active]
                batch_lengths[active] += 1
                finished = np.flatnonzero(active & np.asarray(dones, dtype=bool))
                for worker in finished:
                    info = infos[int(worker)]
                    first_correct += int(info.get("first_grasp_correct") is True)
                    if "workspace_mask" in info or info.get("action_workspace_enabled", False):
                        with (self.output_dir / "workspace_eval_episodes.jsonl").open("a", encoding="utf-8") as stream:
                            record = {key: info.get(key) for key in (
                                "action_workspace_projection_count", "action_workspace_projection_distance_m",
                                "ever_pinched", "ever_aligned_pinch", "ever_grasped", "loaded_lift",
                                "success_hold", "strict_hold_credit_high_water",
                                "requested_seed", "actual_seed", "scenario_name", "goal_node_index",
                                "first_grasp_correct", "workspace_reset_rejections", "workspace_prefix_seconds",
                                "workspace_inside_fraction", "workspace_initial_goal_xyz", "workspace_mask",
                                "workspace_start_protocol", "workspace_goal_ever_inside",
                                "workspace_goal_first_entry_seconds")}
                            record.update(success=bool(info["success"]), seed=batch_seed+int(worker), timesteps=self.num_timesteps,
                                          seed_offset=seed_offset,
                                          phase=self.curriculum.phase_index if self.curriculum else 0)
                            stream.write(json.dumps(record) + "\n")
                    successes += int(bool(info["success"]))
                    strict_successes += int(bool(info.get("strict_success", False)))
                    pinches += int(bool(info["ever_pinched"]))
                    aligned_pinches += int(bool(
                        info.get("ever_aligned_pinch", False)
                    ))
                    lift_attempts += int(bool(info.get("lift_attempt", False)))
                    loaded_lifts += int(bool(info.get("loaded_lift", False)))
                    grasps += int(bool(info["ever_grasped"]))
                    slips += int(
                        int(info.get("physical_slip_after_secured_count", 0)) > 0
                    )
                    returns.append(float(batch_returns[worker]))
                    lengths.append(int(batch_lengths[worker]))
                    active[worker] = False

            completed += batch_size
            print(
                f"strict_eval_progress completed={completed}/{episodes}",
                flush=True,
            )
        return {
            "first_grasp_correct_rate": first_correct / episodes,
            "success_rate": successes / episodes,
            "strict_success_rate": strict_successes / episodes,
            "pinch_rate": pinches / episodes,
            "aligned_pinch_rate": aligned_pinches / episodes,
            "lift_attempt_rate": lift_attempts / episodes,
            "loaded_lift_rate": loaded_lifts / episodes,
            "grasp_rate": grasps / episodes,
            "slip_rate": slips / episodes,
            "mean_return": float(np.mean(returns)),
            "mean_length": float(np.mean(lengths)),
        }

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        if self.curriculum is not None:
            self.curriculum.configure_evaluation_env(self.env)
        metrics = self._evaluate(self.episodes)
        success_rate = metrics["success_rate"]
        strict_success_rate = metrics["strict_success_rate"]
        pinch_rate = metrics["pinch_rate"]
        aligned_pinch_rate = metrics["aligned_pinch_rate"]
        lift_attempt_rate = metrics["lift_attempt_rate"]
        loaded_lift_rate = metrics["loaded_lift_rate"]
        grasp_rate = metrics["grasp_rate"]
        slip_rate = metrics["slip_rate"]
        mean_return = metrics["mean_return"]
        mean_length = metrics["mean_length"]
        curriculum_phase = (
            self.curriculum.phase_index if self.curriculum is not None else 0
        )
        curriculum_stage = (
            self.curriculum.stage_index if self.curriculum is not None else 0
        )
        motion_difficulty = (
            self.curriculum.difficulty if self.curriculum is not None else 1.0
        )
        if self.curriculum is not None:
            self.curriculum.observe_evaluation(metrics)
        next_curriculum_phase = (
            self.curriculum.phase_index if self.curriculum is not None else 0
        )
        with self.csv_path.open("a", encoding="utf-8", newline="") as target:
            csv.writer(target).writerow([
                self.num_timesteps, curriculum_phase, next_curriculum_phase,
                curriculum_stage,
                motion_difficulty, success_rate, strict_success_rate,
                pinch_rate, aligned_pinch_rate,
                lift_attempt_rate, loaded_lift_rate, grasp_rate, slip_rate,
                mean_return, mean_length, self.episodes,
            ])
        self.logger.record("eval/task_success_rate", success_rate)
        self.logger.record("eval/strict_success_rate", strict_success_rate)
        self.logger.record("eval/pinch_rate", pinch_rate)
        self.logger.record("eval/aligned_pinch_rate", aligned_pinch_rate)
        self.logger.record("eval/lift_attempt_rate", lift_attempt_rate)
        self.logger.record("eval/loaded_lift_rate", loaded_lift_rate)
        self.logger.record("eval/secured_grasp_rate", grasp_rate)
        self.logger.record("eval/physical_slip_rate", slip_rate)
        self.logger.record("eval/mean_return", mean_return)

        routine_score = (
            curriculum_phase, success_rate, grasp_rate,
            aligned_pinch_rate, mean_return,
        )
        best_score = (
            self.best_phase, self.best_success_rate, self.best_grasp_rate,
            self.best_aligned_pinch_rate, self.best_mean_return,
        )
        has_task_progress = success_rate > 0.0 or grasp_rate > 0.0
        if has_task_progress and routine_score > best_score:
            confirmed = self._evaluate(
                self.confirmation_episodes, seed_offset=self.confirmation_seed_offset
            )
            confirmed_score = (
                curriculum_phase, confirmed["success_rate"],
                confirmed["grasp_rate"], confirmed["aligned_pinch_rate"],
                confirmed["mean_return"],
            )
            confirmed_task_progress = (
                confirmed["success_rate"] > 0.0
                or confirmed["grasp_rate"] > 0.0
            )
            if confirmed_task_progress and confirmed_score > best_score:
                self.best_phase = curriculum_phase
                self.best_success_rate = confirmed["success_rate"]
                self.best_grasp_rate = confirmed["grasp_rate"]
                self.best_aligned_pinch_rate = confirmed["aligned_pinch_rate"]
                self.best_mean_return = confirmed["mean_return"]
                best_dir = self.output_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(best_dir / "best_model")
                with (best_dir / "confirmation.json").open(
                    "w", encoding="utf-8"
                ) as target:
                    json.dump(
                        {
                            "timesteps": self.num_timesteps,
                            "episodes": self.confirmation_episodes,
                            "requested_episodes": self.confirmation_requested_episodes,
                            "seed_offset": self.confirmation_seed_offset,
                            "seed_pool_size": self.seed_pool_size,
                            "curriculum_phase": curriculum_phase,
                            "curriculum_stage": curriculum_stage,
                            "motion_difficulty": motion_difficulty,
                            **confirmed,
                        },
                        target,
                        ensure_ascii=False,
                        indent=2,
                    )
        print(
            f"strict_eval timesteps={self.num_timesteps} "
            f"task_success={success_rate:.1%} strict_success={strict_success_rate:.1%} "
            f"pinch={pinch_rate:.1%} "
            f"aligned={aligned_pinch_rate:.1%} loaded={loaded_lift_rate:.1%} "
            f"grasp={grasp_rate:.1%} slip={slip_rate:.1%} "
            f"mean_return={mean_return:.3f}",
            flush=True,
        )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for dynamic cable grasping")
    from .workspace import add_workspace_arguments, WORKSPACE_SCENE_FAMILIES
    add_workspace_arguments(parser)
    parser.add_argument("--workspace-curriculum", action="store_true")
    parser.add_argument("--eval-replay-bank-dir", type=Path,
                        help="separate held-out replay bank; uses strict bank seed matching")
    parser.add_argument("--eval-seed-pool-size", type=int, default=0,
                        help="conditioned eval reset retry pool starting at eval-seed; 0 disables")
    parser.add_argument("--workspace-scene-family", choices=tuple(WORKSPACE_SCENE_FAMILIES))
    parser.add_argument("--target-node-task", action="store_true")
    parser.add_argument("--target-reward-profile", choices=("legacy", "tolerance_progress_v2", "tolerance_progress_v3", "tolerance_progress_v4", "tolerance_progress_v5", "tolerance_progress_v6"), default=None)
    parser.add_argument("--rl-config-file", type=Path,
                        help="explicit reference RLConfig JSON; preserves its non-target reward/action values")
    parser.add_argument("--resume-legacy-full6", action="store_true",
                        help="explicitly migrate a frozen full6_workspace_target_node_v1 checkpoint")
    parser.add_argument("--target-u-min", type=float, default=1.0 / 6.0)
    parser.add_argument("--target-u-max", type=float, default=5.0 / 6.0)
    parser.add_argument("--target-node-tolerance", type=int, default=2)
    parser.add_argument("--warm-start", type=Path, default=None,
                        help="expand a compatible state99 PPO into a fresh target-node109 PPO")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument(
        "--workers", type=int, default=12,
        help="parallel MuJoCo environments; tuned for long runs on a 16 GB host",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=1,
        help="CPU threads used by the small PPO MLP in the learner process",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument('--worker-affinity', choices=('none','numa'), default='none',
                        help='Linux: pin workers to distinct physical cores balanced across NUMA nodes')
    parser.add_argument('--contact-read-cache', action=argparse.BooleanOptionalAction, default=True,
                        help='reuse contact reads within a controlled step, invalidated on physics updates')
    parser.add_argument("--disturbance", type=float, default=1.5)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument(
        "--robot", choices=tuple(sorted(ROBOT_SPECS)), default="panda",
        help="robot model used by the MuJoCo environment",
    )
    from ..env.control import add_controller_arguments
    add_controller_arguments(parser)
    parser.add_argument(
        "--action-mode",
        choices=("task_space", "task_space_vertical_down", "joint_target"),
        default="task_space",
        help=(
            "RL action interface: task_space is the historical 5D IK action; "
            "task_space_vertical_down is 5D ordinary IK with the gripper "
            "approach axis fixed vertically downward; "
            "joint_target is 7 joint target deltas plus gripper"
        ),
    )
    parser.add_argument(
        "--observation-mode",
        choices=("state", "pointcloud"),
        default="state",
        help="state uses the 99-D privileged baseline; pointcloud uses RGB-D",
    )
    parser.add_argument("--pointcloud-points", type=int, default=384)
    parser.add_argument("--pointcloud-width", type=int, default=480)
    parser.add_argument("--pointcloud-height", type=int, default=360)
    parser.add_argument(
        "--pointcloud-update-steps",
        type=int,
        default=5,
        help="render a new RGB-D observation every N 50-Hz control steps",
    )
    parser.add_argument(
        "--pointcloud-delay-steps",
        type=int,
        default=3,
        help="sensor delivery latency in 50-Hz control steps",
    )
    parser.add_argument("--pointcloud-voxel-size", type=float, default=0.002)
    parser.add_argument(
        "--pointcloud-render-mode",
        choices=("normal", "gripper_hidden", "mixed"),
        default="normal",
        help=(
            "normal renders the stock scene; gripper_hidden parks gripper "
            "geoms in a disabled render group; mixed samples per episode"
        ),
    )
    parser.add_argument(
        "--geometric-safety",
        action="store_true",
        help="enable the shared geometric robot-obstacle safety layer",
    )
    from .target_defaults import add_table_finger_filter_arguments
    add_table_finger_filter_arguments(parser)
    parser.add_argument(
        "--disable-singularity-avoidance",
        action="store_true",
        help=(
            "disable the IK damping/null-space/target-clipping guard; useful "
            "when training with the bounded singularity reward instead"
        ),
    )
    parser.add_argument(
        "--enable-safety-penalties",
        action="store_true",
        help="enable bounded table-risk and singularity reward shaping",
    )
    parser.add_argument(
        "--disable-height-band-reward",
        action="store_true",
        help=(
            "restore the legacy one-sided 0.12 m lift reward; by default PPO "
            "targets a stable tabletop-relative height band"
        ),
    )
    parser.add_argument("--height-band-low-m", type=float, default=0.18)
    parser.add_argument("--height-band-high-m", type=float, default=0.25)
    parser.add_argument("--reward-overheight-progress", type=float, default=20.0)
    parser.add_argument("--reward-overheight-step", type=float, default=-0.05)
    parser.add_argument("--overheight-full-scale-m", type=float, default=0.10)
    parser.add_argument("--overheight-step-penalty-cap", type=float, default=2.0)
    parser.add_argument(
        "--success-overheight-penalty-per-m", type=float, default=20.0
    )
    parser.add_argument("--success-overheight-penalty-cap", type=float, default=6.0)
    parser.add_argument(
        "--height-penalty-ramp-env-steps",
        type=int,
        default=10_000,
        help=(
            "per-worker control steps used to ramp only the overheight "
            "penalties; useful-height lift progress is active immediately"
        ),
    )
    parser.add_argument("--reward-table-filter-step", type=float, default=-0.02)
    parser.add_argument("--reward-table-contact-step", type=float, default=-0.05)
    parser.add_argument("--table-safety-penalty-cap", type=float, default=2.0)
    parser.add_argument("--reward-singularity-step", type=float, default=-0.01)
    parser.add_argument("--singularity-safety-penalty-cap", type=float, default=1.5)
    parser.add_argument(
        "--safety-penalty-ramp-env-steps",
        type=int,
        default=5_000,
        help="per-worker control steps used to ramp safety penalties from 0 to 1",
    )
    parser.add_argument(
        "--training-distribution", choices=("legacy", "l1", "id"), default="l1",
        help=(
            "l1 uses the four nominal L1 curriculum scenes; id samples the "
            "entire frozen ID registry; legacy preserves the old shape-only task"
        ),
    )
    parser.add_argument(
        "--eval-distribution", choices=("legacy", "l1", "id"), default="l1",
        help="strict-eval distribution; l1 is the four nominal L1 motion strata",
    )
    parser.add_argument(
        "--training-scenarios", default=None,
        help=(
            "comma-separated scenario names; overrides --training-distribution "
            "and, unless --eval-scenarios is also given, --eval-distribution"
        ),
    )
    parser.add_argument(
        "--eval-scenarios", default=None,
        help="comma-separated scenario names; overrides --eval-distribution",
    )
    parser.add_argument(
        "--output", type=Path,
        default=output_path("rl", "train", "ppo_dlo_baseline_v6")
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
        "--target-kl", type=float, default=0.0,
        help="stop a PPO update early above this approximate KL; 0 disables",
    )
    parser.add_argument(
        "--curriculum-stage-steps", "--curriculum-steps",
        dest="curriculum_stage_steps", type=int, default=500_000,
        help=(
            "positive enables evaluation-driven L1 curriculum; 0 disables it "
            "(the numeric value is retained for CLI compatibility)"
        ),
    )
    parser.add_argument("--curriculum-update-steps", type=int, default=10_000)
    parser.add_argument(
        "--curriculum-required-evals", type=int, default=2,
        help="consecutive qualifying deterministic evaluations required to advance",
    )
    parser.add_argument("--curriculum-aligned-pinch-rate", type=float, default=0.50)
    parser.add_argument("--curriculum-secured-rate", type=float, default=0.20)
    parser.add_argument("--curriculum-success-rate", type=float, default=0.10)
    parser.add_argument(
        "--curriculum-stage1-static-replay",
        type=float,
        default=0.20,
        help="fraction of stage-1 training episodes replayed on the static scene",
    )
    parser.add_argument(
        "--curriculum-stage2-static-replay",
        type=float,
        default=0.10,
        help="fraction of stage-2 training episodes replayed on the static scene",
    )
    parser.add_argument(
        "--curriculum-stage2-component-replay",
        type=float,
        default=0.20,
        help=(
            "total fraction of stage-2 training episodes replayed equally on "
            "shape-only and rigid-only scenes"
        ),
    )
    parser.add_argument("--eval-freq", type=int, default=100_000,
                        help="environment timesteps between strict evaluations; 0 disables")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-confirmation-episodes", type=int, default=50)
    parser.add_argument(
        "--eval-workers", type=int, default=4,
        help="parallel MuJoCo environments used only by strict evaluation",
    )
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
    parser.set_defaults(robot=None, action_mode=None, controller_profile=None,
                        table_execution_guard=None)
    parser.add_argument('--ordinary-reward-profile', choices=('legacy', 'physical_progress_v1', 'physical_progress_v2', 'session_height_v1'), default='legacy')
    parser.add_argument('--ordered-ability-curriculum', action='store_true')
    parser.add_argument('--rollout-ability-curriculum', action='store_true',
                        help='Use independent training-episode windows for ordered scene promotion; no eval environments')
    parser.add_argument('--rollout-gate-profile', choices=('strict', 'fast'), default='strict',
                        help='Per-scene promotion thresholds; fast uses 40/50/40%% and one complete window')
    parser.add_argument('--schedule-total-timesteps', type=int, default=0,
                        help='Global schedule horizon; preserved in checkpoint across continuations')
    parser.add_argument('--action-workspace', type=Path, help='Measured TCP execution map; does not enable target-node task')
    parser.add_argument('--ordinary-family-curriculum', action='store_true', help='Map ordinary L1 curriculum to the selected placement family')
    args = parser.parse_args()
    from .target_defaults import configure_target_training_defaults
    configure_target_training_defaults(args, sys.argv[1:])
    if args.target_reward_profile in {"tolerance_progress_v2", "tolerance_progress_v3", "tolerance_progress_v4", "tolerance_progress_v5", "tolerance_progress_v6"} and not args.target_node_task:
        parser.error("tolerance_progress_v2/v3/v4/v5 requires --target-node-task")
    if args.eval_seed_pool_size < 0:
        parser.error("eval seed pool size must be nonnegative")
    if args.eval_freq > 0 and args.eval_seed_pool_size and args.eval_seed_pool_size <= args.eval_episodes:
        parser.error("eval seed pool must leave seeds beyond routine episodes for confirmation")
    if args.rl_config_file is not None:
        try:
            rl_config_from_args(args)
        except (ValueError, OSError, TypeError) as error:
            parser.error(str(error))
    if args.workspace_scene_family:
        if not args.ordinary_family_curriculum:
            args.curriculum_stage_steps = 0
        if args.training_scenarios or args.eval_scenarios:
            parser.error("Workspace scene family cannot be combined with explicit scenario lists")
        args.training_scenarios = ",".join(WORKSPACE_SCENE_FAMILIES[args.workspace_scene_family])
    if args.action_workspace is not None:
        if args.target_node_task or args.action_mode != 'task_space_vertical_down':
            parser.error('Action workspace requires ordinary vertical-down task space')
        from .action_workspace import ActionWorkspace
        try:
            ActionWorkspace(args.action_workspace)
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
    if args.ordinary_family_curriculum and (args.target_node_task or args.workspace_scene_family != 'cable_x075_v1'):
        parser.error('Ordinary family curriculum requires ordinary cable_x075_v1')
    if args.target_workspace is not None:
        if not args.target_node_task or args.observation_mode != "state":
            parser.error("Target workspace requires target-node task and state observations")
        from .workspace import WorkspaceMask
        try:
            WorkspaceMask(args.target_workspace, args.workspace_mask_key)
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
    if not np.isfinite(args.workspace_entry_seconds) or args.workspace_entry_seconds < 0 or args.workspace_reset_attempts < 1:
        parser.error("Invalid workspace entry duration or reset attempts")
    if args.workspace_curriculum:
        if args.target_workspace is None or args.workspace_scene_family is None:
            parser.error("Workspace curriculum requires target-workspace and workspace-scene-family")
        if args.eval_freq <= 0:
            parser.error("Workspace curriculum requires periodic evaluation")
        args.curriculum_stage_steps = 0
    if args.workspace_curriculum and args.resume is None and (args.output / "training_config.json").exists():
        parser.error("Use a new output directory for a fresh workspace curriculum run")
    args.training_scenario_weights = None
    if args.training_scenarios:
        names = []
        weights = []
        has_weights = False
        for item in args.training_scenarios.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                name, _, weight_text = item.rpartition(":")
                has_weights = True
            else:
                name, weight_text = item, "1.0"
            names.append(name.strip())
            try:
                weight = float(weight_text)
            except ValueError:
                parser.error(
                    f"invalid training scenario weight in {item!r}; "
                    "expected name or name:weight"
                )
            if not np.isfinite(weight) or weight < 0.0:
                parser.error(
                    f"training scenario weight must be non-negative: {item!r}"
                )
            weights.append(weight)
        if not names:
            parser.error("--training-scenarios resolved to an empty list")
        if len(set(names)) != len(names):
            parser.error("--training-scenarios names must be unique")
        if has_weights:
            if float(sum(weights)) <= 0.0:
                parser.error("training scenario weights must have positive mass")
            args.training_scenario_weights = weights
        args.training_scenario_names = names
    elif args.training_distribution == "legacy":
        args.training_scenario_names = None
    elif args.training_distribution == "l1":
        args.training_scenario_names = list(RL_L1_SCENARIOS)
    else:
        args.training_scenario_names = list(list_scenario_names("id"))
    resolved_eval_distribution = args.eval_distribution or args.training_distribution
    if args.eval_scenarios:
        args.eval_scenario_names = [
            name.strip()
            for name in args.eval_scenarios.split(",")
            if name.strip()
        ]
    elif args.training_scenarios:
        args.eval_scenario_names = list(args.training_scenario_names)
    elif resolved_eval_distribution == "legacy":
        args.eval_scenario_names = None
    elif resolved_eval_distribution == "l1":
        args.eval_scenario_names = list(RL_L1_SCENARIOS)
    else:
        args.eval_scenario_names = list(list_scenario_names("id"))
    args.eval_distribution = resolved_eval_distribution
    if args.timesteps < 1:
        parser.error("--timesteps must be positive")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be at least 1")
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
        args.episode_seconds,
        args.height_band_low_m,
        args.height_band_high_m,
        args.reward_overheight_progress,
        args.reward_overheight_step,
        args.overheight_full_scale_m,
        args.overheight_step_penalty_cap,
        args.success_overheight_penalty_per_m,
        args.success_overheight_penalty_cap,
        args.reward_table_filter_step,
        args.reward_table_contact_step,
        args.table_safety_penalty_cap,
        args.reward_singularity_step,
        args.singularity_safety_penalty_cap,
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
    if args.disturbance < 0.0:
        parser.error("disturbance strength must be non-negative")
    if args.episode_seconds <= 0.0:
        parser.error("--episode-seconds must be positive")
    if args.height_band_low_m <= 0.0:
        parser.error("--height-band-low-m must be positive")
    if args.height_band_high_m <= args.height_band_low_m:
        parser.error("--height-band-high-m must exceed --height-band-low-m")
    if args.reward_overheight_progress < 0.0:
        parser.error("--reward-overheight-progress must be non-negative")
    if args.reward_overheight_step > 0.0:
        parser.error("--reward-overheight-step must be non-positive")
    if args.overheight_full_scale_m <= 0.0:
        parser.error("--overheight-full-scale-m must be positive")
    if (
        args.overheight_step_penalty_cap < 0.0
        or args.success_overheight_penalty_per_m < 0.0
        or args.success_overheight_penalty_cap < 0.0
    ):
        parser.error("height penalty scales and caps must be non-negative")
    if args.height_penalty_ramp_env_steps < 0:
        parser.error("--height-penalty-ramp-env-steps must be non-negative")
    if args.reward_table_filter_step > 0.0 or args.reward_table_contact_step > 0.0:
        parser.error("table safety rewards must be non-positive")
    if args.reward_singularity_step > 0.0:
        parser.error("singularity safety reward must be non-positive")
    if args.table_safety_penalty_cap < 0.0 or args.singularity_safety_penalty_cap < 0.0:
        parser.error("safety penalty caps must be non-negative")
    if args.safety_penalty_ramp_env_steps < 0:
        parser.error("--safety-penalty-ramp-env-steps must be non-negative")
    if args.pointcloud_points < 32:
        parser.error("--pointcloud-points must be at least 32")
    if args.pointcloud_width < 32 or args.pointcloud_height < 32:
        parser.error("point-cloud image dimensions must be at least 32")
    if args.pointcloud_update_steps < 1 or args.pointcloud_delay_steps < 0:
        parser.error("point-cloud update/delay steps are invalid")
    if args.pointcloud_voxel_size <= 0.0:
        parser.error("--pointcloud-voxel-size must be positive")
    if args.checkpoint_steps < 1:
        parser.error("--checkpoint-steps must be positive")
    if (
        args.curriculum_stage_steps < 0
        or args.curriculum_update_steps < 1
        or args.curriculum_required_evals < 1
    ):
        parser.error("curriculum steps must be non-negative and update steps positive")
    curriculum_rates = (
        args.curriculum_aligned_pinch_rate,
        args.curriculum_secured_rate,
        args.curriculum_success_rate,
    )
    if not all(0.0 <= value <= 1.0 for value in curriculum_rates):
        parser.error("curriculum rate thresholds must be in [0, 1]")
    try:
        args.curriculum_training_mixes = build_l1_curriculum_training_mixes(
            args.curriculum_stage1_static_replay,
            args.curriculum_stage2_static_replay,
            args.curriculum_stage2_component_replay,
        )
    except ValueError as error:
        parser.error(str(error))
    args.curriculum_stages = RL_L1_CURRICULUM_STAGES
    if args.ordinary_family_curriculum:
        variant = lambda name: 'dev_' + name.removeprefix('id_') + '_x075_v1'
        static = (variant('id_static'), variant('id_static_midshape_v1'))
        component = (variant('id_shape_nominal_current'), variant('id_rigid_l1_nominal'),
                     variant('id_rigid_l1_midshape_v1'), variant('id_rigid_replay_shape_nominal'))
        combined = (variant('id_combined_l1_nominal'),)
        args.curriculum_stages = (static, component, combined)
        a = args.curriculum_stage1_static_replay
        b = args.curriculum_stage2_static_replay
        c = args.curriculum_stage2_component_replay
        args.curriculum_training_mixes = (
            tuple((n, 1 / len(static)) for n in static),
            tuple((n, a / len(static)) for n in static) + tuple((n, (1-a) / len(component)) for n in component),
            tuple((n, b / len(static)) for n in static) + tuple((n, c / len(component)) for n in component) + ((combined[0], 1-b-c),),
        )
    if args.rollout_ability_curriculum:
        args.ordered_ability_curriculum = True
        if args.eval_freq != 0:
            parser.error('Rollout curriculum requires --eval-freq 0')
    if args.schedule_total_timesteps < 0 or (args.schedule_total_timesteps and args.restart_schedules_on_resume):
        parser.error('Global schedule horizon must be nonnegative and cannot restart on resume')
    if args.ordered_ability_curriculum:
        if not args.ordinary_family_curriculum or args.curriculum_stage_steps <= 0 or (not args.rollout_ability_curriculum and (args.eval_episodes != 20 or args.eval_seed_pool_size != 50)):
            parser.error('Ordered curriculum requires ordinary family curriculum, enabled stages, 20 episodes per target and 50 seed pool')
        from .ordered_curriculum import GROUPS, MIXES
        args.curriculum_stages, args.curriculum_training_mixes = GROUPS, MIXES
    if (
        args.training_distribution != "l1"
        and args.curriculum_stage_steps > 0
    ):
        parser.error(
            "motion curriculum is defined for --training-distribution l1; "
            "pass --curriculum-stage-steps 0 for another distribution"
        )
    if (
        args.eval_freq < 0
        or args.eval_episodes < 1
        or args.eval_confirmation_episodes < 1
        or args.eval_workers < 0 or (args.eval_freq > 0 and args.eval_workers < 1)
    ):
        parser.error(
            "evaluation frequency/workers must be non-negative; episode counts "
            "must be positive, and enabled evaluation requires positive workers"
        )
    if args.target_node_task and args.observation_mode != "state":
        parser.error("target-node task currently requires state observations")
    if args.warm_start is not None:
        if args.resume is not None or not args.target_node_task:
            parser.error("--warm-start requires --target-node-task and cannot be combined with --resume")
        if not args.warm_start.is_file():
            parser.error("--warm-start checkpoint does not exist")
    if args.resume is not None:
        resume_with_zip = Path(f"{args.resume}.zip")
        if not args.resume.is_file() and not resume_with_zip.is_file():
            parser.error(f"--resume checkpoint does not exist: {args.resume}")
        version = checkpoint_interface_version(
            args.resume if args.resume.is_file() else resume_with_zip
        )
        compatible_legacy = (
            args.resume_legacy_full6 and version == "full6_workspace_target_node_v1"
            and args.robot == "nero" and args.action_mode == "task_space_vertical_down"
            and args.target_node_task and args.observation_mode == "state"
            and args.controller_profile == "enhanced6_bounded_dls_v1"
        )
        if version != action_interface_version(args.action_mode) and not compatible_legacy:
            parser.error(
                "--resume checkpoint uses a different RL action interface and "
                "cannot be continued safely"
            )
    if args.rollout_ability_curriculum:
        from .rollout_curriculum import checkpoint_phase
        args.rollout_initial_phase = checkpoint_phase(args.resume)
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from .affinity import prepare_worker_affinity
    args.worker_affinity_plan = prepare_worker_affinity(args)
    torch.set_num_threads(args.torch_threads)
    set_random_seed(args.seed)
    target_kl = args.target_kl if args.target_kl > 0.0 else None

    # 启动大批量训练前先检查一次Gymnasium API、shape和数据类型。
    check_candidate = make_configured_env(
        args,
        seed=args.seed,
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
            "success", "strict_success", "ever_pinched", "ever_aligned_pinch", "lift_attempt",
            "loaded_lift", "ever_grasped", "lifted_fraction",
        ),
    )

    warm_start_record = None
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
        policy_name = (
            "MultiInputPolicy"
            if args.observation_mode == "pointcloud"
            else "MlpPolicy"
        )
        policy_kwargs = {
            "activation_fn": nn.ReLU,
            "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
        }
        if args.observation_mode == "pointcloud":
            policy_kwargs["features_extractor_class"] = PointNet2FeaturesExtractor
        model = PPO(
            policy_name,
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
            policy_kwargs=policy_kwargs,
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

    saved_global_schedule = getattr(model, 'training_global_schedule', None)
    global_schedule = resolve_global_schedule(args, saved_global_schedule, starting_timesteps)
    if global_schedule:
        horizon = global_schedule['total_timesteps']
        model.training_global_schedule = global_schedule
        learning_rate = cumulative_linear_schedule(global_schedule['learning_rate_initial'],
            global_schedule['learning_rate_final'], horizon, starting_timesteps + args.timesteps)
        model.learning_rate = model.lr_schedule = learning_rate
        entropy_coef_initial = global_schedule['entropy_coef_initial']
        entropy_coef_final = global_schedule['entropy_coef_final']
    if args.warm_start is not None:
        from .target_node import warm_start_policy
        warm_start_record = warm_start_policy(model, args.warm_start)
    from .penalty_ramp import resolve_schedule, apply_progress
    from .target_defaults import checkpoint_configuration
    model.penalty_ramp_schedule = resolve_schedule(model, rl_config_from_args(args), args.workers,
        checkpoint_configuration(args.resume) if args.resume is not None else {})
    apply_progress(vector_env, model, model.penalty_ramp_schedule, training_workers=args.workers)
    from .health import TrainingHealthCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_steps // args.workers),
        save_path=str(args.output / "checkpoints"),
        name_prefix="ppo_cable",
    )
    metrics_path = args.output / "training_metrics.csv"
    metrics_callback = TrainingMetricsCallback(metrics_path, window=100,
                                               goal_task=args.target_node_task)
    callback_items: list[BaseCallback] = [
        checkpoint_callback,
        metrics_callback,
        TrainingHealthCallback(args.output / "training_health.json"),
        EntropyCoefficientScheduleCallback(
            entropy_coef_initial,
            entropy_coef_final,
            global_schedule['total_timesteps'] if global_schedule else args.timesteps,
            cumulative=bool(global_schedule),
        ),
    ]
    curriculum_callback: MotionCurriculumCallback | None = None
    if getattr(args, "workspace_curriculum", False):
        from .workspace import WorkspaceCurriculumCallback
        curriculum_callback = WorkspaceCurriculumCallback(args.training_scenario_names,
            args.output / "evaluation/workspace_curriculum.json", resume=args.resume is not None)
        callback_items.append(curriculum_callback)
    elif (
        args.training_distribution == "l1"
        and args.curriculum_stage_steps > 0
    ):
        curriculum_class = MotionCurriculumCallback
        if args.rollout_ability_curriculum:
            from .rollout_curriculum import RolloutMotionCurriculum
            curriculum_class = RolloutMotionCurriculum
        elif args.ordered_ability_curriculum:
            from .ordered_curriculum import OrderedMotionCurriculum
            curriculum_class = OrderedMotionCurriculum
        curriculum_callback = curriculum_class(
            stages=args.curriculum_stages,
            update_every=args.curriculum_update_steps,
            required_evaluations=args.curriculum_required_evals,
            aligned_pinch_rate=args.curriculum_aligned_pinch_rate,
            secured_rate=args.curriculum_secured_rate,
            success_rate=args.curriculum_success_rate,
            training_mixes=args.curriculum_training_mixes,
            **(dict(output_dir=args.output, initial_phase=args.rollout_initial_phase,
                    gate_profile=args.rollout_gate_profile) if args.rollout_ability_curriculum else {}),
        )
        if args.rollout_ability_curriculum:
            callback_items.insert(0, curriculum_callback)
        else:
            callback_items.append(curriculum_callback)
    eval_env: VecEnv | None = None
    if args.eval_freq > 0:
        eval_factories = [
            make_eval_worker(rank, args) for rank in range(args.eval_workers)
        ]
        if args.eval_workers == 1:
            eval_env = DummyVecEnv(eval_factories)
        else:
            eval_env = SubprocVecEnv(eval_factories, start_method="spawn")
        evaluation_class = StrictSuccessEvalCallback
        if args.ordered_ability_curriculum:
            from .ordered_curriculum import OrderedSuccessEval
            evaluation_class = OrderedSuccessEval
        callback_items.append(evaluation_class(
            eval_env,
            args.output / "evaluation",
            eval_freq=args.eval_freq,
            episodes=args.eval_episodes,
            confirmation_episodes=args.eval_confirmation_episodes,
            seed=args.eval_seed,
            curriculum=curriculum_callback,
            seed_pool_size=args.eval_seed_pool_size,
        ))
    callbacks = CallbackList(callback_items)
    from .workspace import WorkspaceMask, workspace_stages
    resolved_workspace_stages = workspace_stages(args.training_scenario_names) if args.workspace_curriculum else None
    from .affinity import bind_learner
    bind_learner(args)
    configuration = {
        "algorithm": "PPO",
        "robot": args.robot,
        "rl_interface_version": action_interface_version(args.action_mode),
        "action_mode": args.action_mode,
        "observation_mode": args.observation_mode,
        "observation_interface_version": "target_node_state_v1" if args.target_node_task else "baseline_state_v1",
        "target_node_task": args.target_node_task,
        "warm_start": warm_start_record,
        "target_reward_profile": args.target_reward_profile,
        "target_termination_version": "public_success_or_original_timeout_v1",
        "penalty_ramp_schedule": model.penalty_ramp_schedule,
        "penalty_ramp_start_scales": {kind: vector_env.env_method('_penalty_ramp_scale', kind)[0]
                                       for kind in ('height', 'safety')},
        "lift_progress_accounting_version": (
            "physical_node_release_bounded_debit_v2" if (args.target_reward_profile == "tolerance_progress_v6" or args.ordinary_reward_profile in {"physical_progress_v1", "physical_progress_v2"})
            else "episode_credit_bounded_debit_v1" if args.target_reward_profile in {"tolerance_progress_v3", "tolerance_progress_v4", "tolerance_progress_v5"}
            else "episode_high_water_unbounded_debit_v2" if args.target_reward_profile == "tolerance_progress_v2"
            else "legacy"
        ),
        "approach_progress_accounting_version": ("episode_credit_bounded_debit_v1" if args.target_reward_profile in {"tolerance_progress_v4", "tolerance_progress_v5", "tolerance_progress_v6"} else "legacy_phase_gated"),
        "lift_reward_holding_version": (
            "confirmed_distance_contact_grace_v3_no_aperture" if (args.target_reward_profile == "tolerance_progress_v6" or args.ordinary_reward_profile in {"physical_progress_v1", "physical_progress_v2"})
            else "current_bilateral_distance_v2_no_aperture" if args.target_reward_profile == "tolerance_progress_v5"
            else "confirmed_grasp_grace_v1"),
        "lift_reward_contact_grace_seconds": (
            vector_env.env_method('_lift_reward_contact_grace_seconds')[0]
            if (args.target_reward_profile == "tolerance_progress_v6" or args.ordinary_reward_profile in {"physical_progress_v1", "physical_progress_v2"}) else 0.0),
        "secured_grasp_accounting_version": (
            "confirmed_distance_contact_grace_v3_no_aperture" if (args.target_reward_profile == "tolerance_progress_v6" or args.ordinary_reward_profile in {"physical_progress_v1", "physical_progress_v2"})
            else "raw_confirm_then_secured_grace_v2_no_aperture"),
        "secured_contact_grace_seconds": (
            vector_env.env_method('_lift_reward_contact_grace_seconds')[0]
            if (args.target_reward_profile == "tolerance_progress_v6" or args.ordinary_reward_profile in {"physical_progress_v1", "physical_progress_v2"}) else None),
        "reference_rl_config_file": str(args.rl_config_file.resolve()) if args.rl_config_file else None,
        "resume_legacy_full6_requested": args.resume_legacy_full6,
        "resume_interface_migration": ("frozen full6_workspace_target_node_v1 -> formal bounded DLS and physical_holding_v2" if args.resume_legacy_full6 else None),
        "controller_profile": args.controller_profile,
        "table_execution_guard_enabled": args.table_execution_guard,
        "grasp_judgement_version": "physical_holding_v3_no_aperture",
        "public_success_version": "confirmed_distance_grace_v4_no_aperture",
        "grasp_aperture_qualification_enabled": False,
        "hold_reward_timer_version": (
            "public_success_hold_v1" if (args.target_reward_profile == "tolerance_progress_v6" or args.ordinary_reward_profile in {"physical_progress_v1", "physical_progress_v2"})
            else "strict_height_band_hold_v1"),
        "arm_acceleration_limit_enabled": args.controller_profile == "legacy",
        "target_workspace": (None if args.target_workspace is None else WorkspaceMask(args.target_workspace, args.workspace_mask_key).manifest()),
        "workspace_start_protocol": args.workspace_start_protocol,
        "arm_motion_start_immediate": bool(args.target_workspace is not None
            and args.workspace_start_protocol == "policy_from_initial_v3"),
        "workspace_goal_sampling": (None if not args.target_node_task else "all_material_nodes_at_initial_state_v2"
            if args.workspace_start_protocol != "physical_entry_v1" else "entry_workspace_nodes_v1"),
        "workspace_entry_seconds": (0. if args.workspace_start_protocol != "physical_entry_v1" else args.workspace_entry_seconds),
        "workspace_reset_attempts": args.workspace_reset_attempts,
        "workspace_curriculum": args.workspace_curriculum,
        "workspace_scene_family": args.workspace_scene_family,
        "workspace_stages": resolved_workspace_stages,
        "replay_bank_dir": (str(args.replay_bank_dir.resolve()) if args.replay_bank_dir else None),
        "replay_seed_fallback": args.replay_seed_fallback,
        "eval_replay_bank_dir": str(args.eval_replay_bank_dir.resolve()) if args.eval_replay_bank_dir else None,
        "eval_seed_pool_size": args.eval_seed_pool_size,
        "geometric_safety_enabled": args.geometric_safety,
        "table_finger_collision_filter_enabled": (
            not args.disable_table_finger_collision_filter
        ),
        "singularity_avoidance_enabled": (
            not args.disable_singularity_avoidance
        ),
        "height_band_reward_enabled": not args.disable_height_band_reward,
        "safety_reward_enabled": args.enable_safety_penalties,
        "grasp_model": "physical_friction_v1",
        "output": str(args.output.resolve()),
        "timesteps": args.timesteps,
        "timesteps_semantics": "additional timesteps in this learn invocation",
        "starting_timesteps": starting_timesteps,
        "cumulative_target_timesteps": starting_timesteps + args.timesteps,
        "resume_checkpoint": str(args.resume.resolve()) if args.resume is not None else None,
        "reset_num_timesteps": args.resume is None,
        "workers": args.workers,
        "torch_threads": args.torch_threads,
        "worker_affinity": args.worker_affinity,
        "worker_affinity_plan": args.worker_affinity_plan,
        "contact_read_cache_enabled": args.contact_read_cache,
        "seed": args.seed,
        "disturbance": args.disturbance,
        "training_distribution": args.training_distribution,
        "training_scenario_names": args.training_scenario_names,
        "training_scenario_weights": args.training_scenario_weights,
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
        "policy_activation": "ReLU",
        "pointcloud": (
            asdict(pointcloud_config_from_args(args))
            if args.observation_mode == "pointcloud"
            else None
        ),
        "curriculum_stages": args.curriculum_stages,
        "ordinary_family_curriculum": args.ordinary_family_curriculum,
        "ordinary_reward_profile": args.ordinary_reward_profile,
        "lift_height_baseline_version": ("first_confirmed_holding_v1" if args.ordinary_reward_profile == "physical_progress_v2" else "episode_initial_max_v1"),
        "action_workspace": (dict(path=str(args.action_workspace.resolve()), sha256=__import__("hashlib").sha256(args.action_workspace.read_bytes()).hexdigest(), version="measured_polar_cells_v1") if args.action_workspace else None),
        "curriculum_training_mixes": args.curriculum_training_mixes,
        "curriculum_stage1_static_replay": args.curriculum_stage1_static_replay,
        "curriculum_stage2_static_replay": args.curriculum_stage2_static_replay,
        "curriculum_stage2_component_replay": (
            args.curriculum_stage2_component_replay
        ),
        "curriculum_stage_steps": args.curriculum_stage_steps,
        "curriculum_update_steps": args.curriculum_update_steps,
        "curriculum_required_evals": args.curriculum_required_evals,
        "curriculum_aligned_pinch_rate": args.curriculum_aligned_pinch_rate,
        "curriculum_secured_rate": args.curriculum_secured_rate,
        "curriculum_success_rate": args.curriculum_success_rate,
        "curriculum_motion_difficulty": (
            "strict-eval-driven static -> component L1 -> combined L1; easier "
            "scenes remain in the weighted training distribution while strict "
            "evaluation uses only current-stage targets; each moving stage "
            "advances through difficulty 1/3, 2/3 and 1"
        ),
        "curriculum_final_disturbance": args.disturbance,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "eval_confirmation_episodes": args.eval_confirmation_episodes,
        "eval_confirmation_effective_episodes": (
            min(args.eval_confirmation_episodes, args.eval_seed_pool_size - args.eval_episodes)
            if args.eval_seed_pool_size else args.eval_confirmation_episodes
        ),
        "eval_confirmation_seed_offset": (
            args.eval_episodes if args.eval_seed_pool_size else 100_000
        ),
        "eval_seed_protocol": "finite_pool_disjoint_confirmation_v1",
        "eval_workers": args.eval_workers,
        "eval_seed": args.eval_seed,
        "rl_grasp_definition": (
            "bilateral confirmed pinch + 30 mm lift from that cable body's "
            "episode-initial height + "
            "0.10 s continuous physical retention"
        ),
        "rl_success_definition": (
            "shared base-environment task success: confirmed grasp + body z > "
            "0.14 m + lifted fraction >= 0.18 + distance within the public pad "
            "threshold for a 500 Hz continuous 0.80 s window"
        ),
        "rl_strict_success_diagnostic": (
            "secured grasp + raw bilateral contact + distance <= 0.055 m at "
            "50 Hz for 0.80 s, intersected with the base environment's current "
            "qualification; diagnostic only for v6, historical shaping for earlier profiles"
        ),
        "observation_names": (
            {"points": [args.pointcloud_points, 3], "proprio": [16]}
            if args.observation_mode == "pointcloud"
            else list(check_candidate.unwrapped.OBSERVATION_NAMES)
        ),
        "observation_dimension": (
            args.pointcloud_points * 3 + 16
            if args.observation_mode == "pointcloud"
            else int(check_candidate.observation_space.shape[0])
        ),
        "action_names": list(check_candidate.unwrapped.ACTION_NAMES),
        "rl_config": asdict(rl_config_from_args(args)),
        "reward": (
            "pre-pinch nearest-segment reach progress + planar perpendicular "
            "alignment potential progress + centered, pad-depth-qualified capture "
            "close bonus + premature-close event/hold penalties + "
            "aligned-pinch bonus + capped unloaded-pinch penalties + "
            + (
                "legacy capped lift progress and strict-hold credit "
                if args.disable_height_band_reward
                else (
                    f"tabletop-relative {args.height_band_low_m:g}-"
                    f"{args.height_band_high_m:g} m height-band progress, bounded "
                    "overheight penalties, in-band hold credit "
                )
            )
            + "+ shared task success; "
            "post-secured active-open and physical-slip penalties remain causal"
            "; optional bounded table-filter/contact and singularity shaping"
        ),
        "action": (
            "base-frame delta xyz (0.010 m vector-norm cap), world-z yaw "
            "(0.020 rad), and one hysteretic open/close gripper command; damped resolved-rate "
            "IK with uniform joint-velocity scaling"
        ),
    }
    if args.workspace_curriculum:
        configuration["curriculum_stages"] = resolved_workspace_stages
        configuration["curriculum_training_mixes"] = [
            list(zip(args.training_scenario_names, stage[1])) for stage in resolved_workspace_stages
        ]
        configuration["curriculum_required_evals"] = "2 passing evaluations in last 3"
        configuration["curriculum_success_rate"] = .40
        configuration["curriculum_first_grasp_correct_rate"] = .60
        configuration["curriculum_motion_difficulty"] = "explicit seven-stage workspace curriculum"
    if args.controller_profile == "enhanced6_bounded_dls_v1":
        configuration["action"] = configuration["action"].split(";", 1)[0] + "; enhanced full6 bounded DLS, gravity feedforward and enhanced servo"
    if args.target_node_task:
        configuration["rl_success_definition"] = (
            "shared physical lift/retention qualification AND material node error "
            "within target_node_tolerance continuously for success_hold_seconds at physics substeps; "
            "confirmed physical geometry with existing contact-loss grace, not uninterrupted raw contacts"
        )
        configuration["observation"] = "state99 plus target material u, TCP-relative position/velocity and tangent: 109-D"
    if args.ordered_ability_curriculum:
        from .ordered_curriculum import PROTOCOL
        configuration['ordered_curriculum'] = PROTOCOL
    configuration.update(actual_eval_workers=args.eval_workers if args.eval_freq > 0 else 0,
        independent_evaluation_enabled=args.eval_freq > 0,
        rollout_ability_curriculum=args.rollout_ability_curriculum,
        training_global_schedule=global_schedule)
    if global_schedule:
        configuration['learning_rate_schedule'] = 'linear over cumulative global samples'
        configuration['entropy_coef_schedule'] = 'linear over cumulative global samples; updated per rollout'
    if args.rollout_ability_curriculum:
        from .rollout_curriculum import protocol_for, PHASES, GROUPS, MIXES
        configuration['ordered_curriculum'] = dict(protocol_for(args.rollout_gate_profile), phases=PHASES, groups=GROUPS, training_mixes=MIXES)
        configuration['rollout_gate_profile'] = args.rollout_gate_profile
        configuration['statistics_source'] = 'stochastic training episodes'
        configuration['rollout_initial_phase'] = args.rollout_initial_phase
    if args.ordinary_reward_profile == 'session_height_v1':
        configuration.update(lift_progress_accounting_version='session_reversible_bounded_balance_v1',
            lift_height_baseline_version='previous_control_step_per_session_v1',
            lift_reward_holding_version='confirmed_distance_contact_grace_v3_no_aperture',
            lift_reward_contact_grace_seconds=vector_env.env_method('_lift_reward_contact_grace_seconds')[0],
            secured_grasp_accounting_version='confirmed_distance_contact_grace_v3_no_aperture',
            secured_contact_grace_seconds=vector_env.env_method('_lift_reward_contact_grace_seconds')[0],
            hold_reward_timer_version='strict_height_band_hold_v1',
            strict_hold_reward_version='strict_in_height_band_high_water_v1',
            session_lift_node_switch='refund old balance; new node uses its own previous-step height')
    (args.output / "training_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        if args.ordinary_reward_profile == 'session_height_v1' or args.rollout_ability_curriculum:
            from .exact_budget import install_exact_budget
            install_exact_budget()
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
