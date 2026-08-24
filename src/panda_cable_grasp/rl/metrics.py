"""PPO训练期间的任务成功率记录与训练结束曲线生成。"""

from __future__ import annotations

from collections import deque
import csv
import os
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    """记录逐轮任务指标，并实时输出最近若干轮的成功率。"""

    def __init__(
        self,
        csv_path: Path,
        *,
        window: int = 100,
        print_every_episodes: int = 20,
    ):
        super().__init__(verbose=0)
        self.csv_path = Path(csv_path)
        self.window = window
        self.print_every_episodes = print_every_episodes
        self.success_window: deque[float] = deque(maxlen=window)
        self.strict_success_window: deque[float] = deque(maxlen=window)
        self.pinch_window: deque[float] = deque(maxlen=window)
        self.aligned_pinch_window: deque[float] = deque(maxlen=window)
        self.loaded_lift_window: deque[float] = deque(maxlen=window)
        self.lift_attempt_window: deque[float] = deque(maxlen=window)
        self.grasp_window: deque[float] = deque(maxlen=window)
        self.return_window: deque[float] = deque(maxlen=window)
        self.lift_window: deque[float] = deque(maxlen=window)
        self.episode_count = 0
        self._file = None
        self._writer = None
        self._rollout_actions: list[np.ndarray] = []
        self._rollout_ik_scales: list[float] = []
        self._rollout_joint_limit_ratios: list[float] = []
        self._rollout_gripper_switches: list[float] = []
        self._rollout_gripper_closed: list[float] = []
        self._rollout_capture_ready: list[float] = []
        self._rollout_capture_close_events: list[float] = []
        self._rollout_premature_close_events: list[float] = []

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        if file_exists:
            with self.csv_path.open("r", encoding="utf-8", newline="") as existing:
                previous_rows = list(csv.DictReader(existing))
            self.episode_count = len(previous_rows)
            # 继续训练时恢复最近一个窗口，避免实时成功率从空窗口重新开始。
            for row in previous_rows[-self.window:]:
                self.success_window.append(float(row["success"]))
                self.strict_success_window.append(float(
                    row.get("strict_success", row["success"])
                ))
                self.pinch_window.append(float(row.get("ever_pinched", 0.0)))
                self.aligned_pinch_window.append(float(
                    row.get("ever_aligned_pinch", 0.0)
                ))
                self.loaded_lift_window.append(float(row.get("loaded_lift", 0.0)))
                self.lift_attempt_window.append(float(row.get("lift_attempt", 0.0)))
                self.grasp_window.append(float(row["ever_grasped"]))
                self.return_window.append(float(row["episode_return"]))
                self.lift_window.append(float(row["lifted_fraction"]))
        fieldnames = [
            "episode", "timesteps", "success", "strict_success", "ever_pinched",
            "ever_aligned_pinch", "lift_attempt", "loaded_lift", "ever_grasped",
            "episode_return", "episode_length", "lifted_fraction",
        ]
        required_v3_fields = {
            "strict_success", "ever_pinched", "ever_aligned_pinch", "lift_attempt",
            "loaded_lift",
        }
        if (
            file_exists
            and previous_rows
            and not required_v3_fields.issubset(previous_rows[0])
        ):
            raise RuntimeError(
                f"Existing metrics use an older RL interface: {self.csv_path}. "
                "Start baseline v4 in a new output directory."
            )
        self._file = self.csv_path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions")
        if actions is not None:
            action_batch = np.asarray(actions, dtype=float)
            if action_batch.ndim == 1:
                action_batch = action_batch[None, :]
            self._rollout_actions.append(action_batch.copy())
        for info in infos:
            self._rollout_ik_scales.append(
                float(info.get("ik_velocity_scale", 1.0))
            )
            self._rollout_joint_limit_ratios.append(
                float(info.get("joint_velocity_limit_ratio", 0.0))
            )
            self._rollout_gripper_switches.append(
                float(bool(info.get("gripper_switch_event", False)))
            )
            self._rollout_gripper_closed.append(
                float(bool(info.get("gripper_closed", False)))
            )
            self._rollout_capture_ready.append(
                float(bool(info.get("capture_ready", False)))
            )
            self._rollout_capture_close_events.append(
                float(bool(info.get("capture_close_event", False)))
            )
            self._rollout_premature_close_events.append(
                float(bool(info.get("premature_close_event", False)))
            )
        for done, info in zip(dones, infos):
            if not done:
                continue
            episode_info = info.get("episode", {})
            success = float(bool(episode_info.get("success", info.get("success", False))))
            strict_success = float(bool(episode_info.get(
                "strict_success", info.get("strict_success", False)
            )))
            pinched = float(bool(
                episode_info.get("ever_pinched", info.get("ever_pinched", False))
            ))
            grasped = float(bool(
                episode_info.get("ever_grasped", info.get("ever_grasped", False))
            ))
            aligned_pinched = float(bool(episode_info.get(
                "ever_aligned_pinch", info.get("ever_aligned_pinch", False)
            )))
            lift_attempt = float(bool(episode_info.get(
                "lift_attempt", info.get("lift_attempt", False)
            )))
            loaded_lift = float(bool(episode_info.get(
                "loaded_lift", info.get("loaded_lift", False)
            )))
            episode_return = float(episode_info.get("r", 0.0))
            episode_length = int(episode_info.get("l", 0))
            lifted_fraction = float(
                episode_info.get("lifted_fraction", info.get("lifted_fraction", 0.0))
            )

            self.episode_count += 1
            self.success_window.append(success)
            self.strict_success_window.append(strict_success)
            self.pinch_window.append(pinched)
            self.aligned_pinch_window.append(aligned_pinched)
            self.lift_attempt_window.append(lift_attempt)
            self.loaded_lift_window.append(loaded_lift)
            self.grasp_window.append(grasped)
            self.return_window.append(episode_return)
            self.lift_window.append(lifted_fraction)
            assert self._writer is not None and self._file is not None
            self._writer.writerow({
                "episode": self.episode_count,
                "timesteps": self.num_timesteps,
                "success": int(success),
                "strict_success": int(strict_success),
                "ever_pinched": int(pinched),
                "ever_aligned_pinch": int(aligned_pinched),
                "lift_attempt": int(lift_attempt),
                "loaded_lift": int(loaded_lift),
                "ever_grasped": int(grasped),
                "episode_return": episode_return,
                "episode_length": episode_length,
                "lifted_fraction": lifted_fraction,
            })
            self._file.flush()

            success_rate = float(np.mean(self.success_window))
            strict_success_rate = float(np.mean(self.strict_success_window))
            pinch_rate = float(np.mean(self.pinch_window))
            grasp_rate = float(np.mean(self.grasp_window))
            aligned_pinch_rate = float(np.mean(self.aligned_pinch_window))
            lift_attempt_rate = float(np.mean(self.lift_attempt_window))
            loaded_lift_rate = float(np.mean(self.loaded_lift_window))
            mean_return = float(np.mean(self.return_window))
            mean_lift = float(np.mean(self.lift_window))
            # logger记录会同时进入PPO终端表格和TensorBoard。
            self.logger.record("task/episodes", self.episode_count)
            self.logger.record("task/success_rate_100", success_rate)
            self.logger.record("task/strict_success_rate_100", strict_success_rate)
            self.logger.record("task/pinch_rate_100", pinch_rate)
            self.logger.record("task/aligned_pinch_rate_100", aligned_pinch_rate)
            self.logger.record("task/lift_attempt_rate_100", lift_attempt_rate)
            self.logger.record("task/loaded_lift_rate_100", loaded_lift_rate)
            self.logger.record("task/grasp_rate_100", grasp_rate)
            self.logger.record("task/mean_return_100", mean_return)
            self.logger.record("task/mean_lifted_fraction_100", mean_lift)
            self.logger.record(
                "task/pinch_to_secured_conversion_100",
                float(np.sum(self.grasp_window))
                / max(float(np.sum(self.pinch_window)), 1.0),
            )
            self.logger.record(
                "task/secured_to_success_conversion_100",
                float(np.sum(self.success_window))
                / max(float(np.sum(self.grasp_window)), 1.0),
            )

            if self.episode_count % self.print_every_episodes == 0:
                print(
                    f"training episodes={self.episode_count} "
                    f"timesteps={self.num_timesteps} "
                    f"success_rate_{self.window}={success_rate:.1%} "
                    f"strict_success_rate_{self.window}={strict_success_rate:.1%} "
                    f"pinch_rate_{self.window}={pinch_rate:.1%} "
                    f"aligned_pinch_rate_{self.window}={aligned_pinch_rate:.1%} "
                    f"loaded_lift_rate_{self.window}={loaded_lift_rate:.1%} "
                    f"grasp_rate_{self.window}={grasp_rate:.1%} "
                    f"mean_return_{self.window}={mean_return:.3f} "
                    f"mean_lifted_fraction_{self.window}={mean_lift:.3f}",
                    flush=True,
                )
        return True

    def _on_rollout_end(self) -> None:
        if self._rollout_actions:
            actions = np.concatenate(self._rollout_actions, axis=0)
            for index in range(actions.shape[1]):
                self.logger.record(
                    f"action/dim_{index}_mean", float(np.mean(actions[:, index]))
                )
                self.logger.record(
                    f"action/dim_{index}_std", float(np.std(actions[:, index]))
                )
                self.logger.record(
                    f"action/dim_{index}_saturation",
                    float(np.mean(np.abs(actions[:, index]) >= 0.98)),
                )
        if self._rollout_ik_scales:
            self.logger.record(
                "control/ik_velocity_scale_mean",
                float(np.mean(self._rollout_ik_scales)),
            )
            self.logger.record(
                "control/ik_velocity_limited_fraction",
                float(np.mean(np.asarray(self._rollout_ik_scales) < 1.0 - 1e-9)),
            )
        if self._rollout_joint_limit_ratios:
            self.logger.record(
                "control/joint_velocity_limit_ratio_mean",
                float(np.mean(self._rollout_joint_limit_ratios)),
            )
        if self._rollout_gripper_switches:
            self.logger.record(
                "control/gripper_switch_fraction",
                float(np.mean(self._rollout_gripper_switches)),
            )
        if self._rollout_gripper_closed:
            self.logger.record(
                "control/gripper_closed_fraction",
                float(np.mean(self._rollout_gripper_closed)),
            )
        if self._rollout_capture_ready:
            self.logger.record(
                "control/capture_ready_fraction",
                float(np.mean(self._rollout_capture_ready)),
            )
        if self._rollout_capture_close_events:
            self.logger.record(
                "control/capture_close_event_fraction",
                float(np.mean(self._rollout_capture_close_events)),
            )
        if self._rollout_premature_close_events:
            self.logger.record(
                "control/premature_close_event_fraction",
                float(np.mean(self._rollout_premature_close_events)),
            )
        log_std = getattr(self.model.policy, "log_std", None)
        if log_std is not None:
            policy_std = np.exp(log_std.detach().cpu().numpy())
            self.logger.record("policy/action_std_mean", float(np.mean(policy_std)))
            for index, value in enumerate(policy_std.reshape(-1)):
                self.logger.record(f"policy/action_std_{index}", float(value))
        self._rollout_actions.clear()
        self._rollout_ik_scales.clear()
        self._rollout_joint_limit_ratios.clear()
        self._rollout_gripper_switches.clear()
        self._rollout_gripper_closed.clear()
        self._rollout_capture_ready.clear()
        self._rollout_capture_close_events.clear()
        self._rollout_premature_close_events.clear()

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """返回包含训练初期不足一个完整窗口时的滑动平均。"""
    if len(values) == 0:
        return values.copy()
    cumulative = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    result = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        result[index] = (
            cumulative[index + 1] - cumulative[start]
        ) / (index + 1 - start)
    return result


def plot_training_curves(csv_path: Path, output_dir: Path, window: int = 100) -> None:
    """从逐轮CSV生成可复算的滚动数据和PNG曲线。"""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    if not csv_path.exists():
        print(f"metrics_plot_skipped missing={csv_path}", flush=True)
        return

    with csv_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        print("metrics_plot_skipped no_completed_episodes", flush=True)
        return

    episodes = np.array([int(row["episode"]) for row in rows])
    timesteps = np.array([int(row["timesteps"]) for row in rows])
    successes = np.array([float(row["success"]) for row in rows])
    pinches = np.array([float(row.get("ever_pinched", 0.0)) for row in rows])
    aligned_pinches = np.array([
        float(row.get("ever_aligned_pinch", 0.0)) for row in rows
    ])
    loaded_lifts = np.array([float(row.get("loaded_lift", 0.0)) for row in rows])
    grasps = np.array([float(row["ever_grasped"]) for row in rows])
    returns = np.array([float(row["episode_return"]) for row in rows])
    lifted = np.array([float(row["lifted_fraction"]) for row in rows])

    success_rate = _rolling_mean(successes, window)
    pinch_rate = _rolling_mean(pinches, window)
    aligned_pinch_rate = _rolling_mean(aligned_pinches, window)
    loaded_lift_rate = _rolling_mean(loaded_lifts, window)
    grasp_rate = _rolling_mean(grasps, window)
    return_mean = _rolling_mean(returns, window)
    lifted_mean = _rolling_mean(lifted, window)

    rolling_path = output_dir / "training_curves.csv"
    with rolling_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow([
            "episode", "timesteps", f"success_rate_{window}",
            f"pinch_rate_{window}",
            f"aligned_pinch_rate_{window}", f"loaded_lift_rate_{window}",
            f"grasp_rate_{window}", f"mean_return_{window}",
            f"mean_lifted_fraction_{window}",
        ])
        writer.writerows(zip(
            episodes, timesteps, success_rate, pinch_rate, aligned_pinch_rate,
            loaded_lift_rate, grasp_rate,
            return_mean, lifted_mean
        ))

    matplotlib_cache = output_dir / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(timesteps, success_rate, label=f"Success rate (last {window})")
    axis.plot(timesteps, pinch_rate, label=f"Pinch rate (last {window})", alpha=0.75)
    axis.plot(
        timesteps, aligned_pinch_rate,
        label=f"Aligned pinch rate (last {window})", alpha=0.8,
    )
    axis.plot(
        timesteps, loaded_lift_rate,
        label=f"Loaded lift rate (last {window})", alpha=0.8,
    )
    axis.plot(timesteps, grasp_rate, label=f"Grasp rate (last {window})", alpha=0.8)
    axis.set_xlabel("Environment timesteps")
    axis.set_ylabel("Rate")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "success_rate.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(timesteps, return_mean, label=f"Mean return (last {window})")
    axis.set_xlabel("Environment timesteps")
    axis.set_ylabel("Episode return")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "reward_curve.png", dpi=180)
    plt.close(figure)
    print(
        f"metrics_plots={output_dir / 'success_rate.png'},"
        f"{output_dir / 'reward_curve.png'}",
        flush=True,
    )
