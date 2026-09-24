"""Shared, replay-oriented episode recording for simulation evaluations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from ..runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import cv2
import mujoco
import numpy as np


RECORDING_SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    """Convert numpy-rich simulator values to JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def create_unique_run_dir(root: Path, prefix: str = "run") -> Path:
    """Create a collision-free output directory using a UTC timestamp."""

    from datetime import datetime, timezone

    root = Path(root)
    stem = datetime.now(timezone.utc).strftime(f"{prefix}_%Y%m%d_%H%M%S")
    target = root / stem
    suffix = 1
    while target.exists():
        target = root / f"{stem}_{suffix:02d}"
        suffix += 1
    target.mkdir(parents=True)
    return target


@dataclass(frozen=True)
class EpisodeArtifacts:
    episode_dir: Path
    trajectory: Path
    metadata: Path
    global_video: Path
    wrist_video: Path

    def relative_to(self, root: Path) -> dict[str, str]:
        return {
            "episode_dir": str(self.episode_dir.relative_to(root)),
            "trajectory": str(self.trajectory.relative_to(root)),
            "metadata": str(self.metadata.relative_to(root)),
            "global_video": str(self.global_video.relative_to(root)),
            "wrist_video": str(self.wrist_video.relative_to(root)),
        }


class EpisodeRecorder:
    """Record every control state plus fixed-rate global and wrist videos.

    ``env`` may be either :class:`CableGraspEnv` or a wrapper exposing
    ``base_env``. FULLPHYSICS states are captured initially and after every
    environment step, so replay does not depend on policy code.
    """

    def __init__(
        self,
        env: Any,
        episode_dir: Path,
        *,
        video_fps: float = 25.0,
    ) -> None:
        if not np.isfinite(video_fps) or video_fps <= 0.0:
            raise ValueError("video_fps must be finite and positive")
        self.env = env
        self.base_env = getattr(env, "base_env", env)
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.video_fps = float(video_fps)
        self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.state_size = mujoco.mj_stateSize(self.base_env.model, self.state_spec)

        self.artifacts = EpisodeArtifacts(
            episode_dir=self.episode_dir,
            trajectory=self.episode_dir / "trajectory.npz",
            metadata=self.episode_dir / "episode.json",
            global_video=self.episode_dir / "global.mp4",
            wrist_video=self.episode_dir / "wrist.mp4",
        )
        if not self.base_env.config.dynamicvla_cameras_enabled:
            raise ValueError(
                "episode recording requires EnvConfig.dynamicvla_cameras_enabled"
            )
        width = int(self.base_env.config.dynamicvla_camera_width)
        height = int(self.base_env.config.dynamicvla_camera_height)
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        self._global_writer = cv2.VideoWriter(
            str(self.artifacts.global_video), codec, self.video_fps, (width, height)
        )
        self._wrist_writer = cv2.VideoWriter(
            str(self.artifacts.wrist_video), codec, self.video_fps, (width, height)
        )
        if not self._global_writer.isOpened() or not self._wrist_writer.isOpened():
            self.close()
            raise RuntimeError(f"failed to open video writers in {self.episode_dir}")

        self._states: list[np.ndarray] = []
        self._state_times: list[float] = []
        self._policy_actions: list[np.ndarray] = []
        self._requested_actions: list[np.ndarray] = []
        self._applied_actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._reward_components: list[dict[str, float]] = []
        self._extras: dict[str, list[np.ndarray]] = {}
        self._frame_state_indices: list[int] = []
        self._frame_times: list[float] = []
        self._next_frame_time = 0.0
        self._closed = False

    def _capture_state(self) -> int:
        state = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(
            self.base_env.model, self.base_env.data, state, self.state_spec
        )
        self._states.append(state)
        self._state_times.append(float(self.base_env.data.time))
        return len(self._states) - 1

    def _write_frame(self, state_index: int) -> None:
        frames = self.base_env.dynamicvla_camera_rgb()
        global_rgb = frames["opst_cam"]
        wrist_rgb = frames["wrist_cam"]
        self._global_writer.write(cv2.cvtColor(global_rgb, cv2.COLOR_RGB2BGR))
        self._wrist_writer.write(cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
        self._frame_state_indices.append(state_index)
        self._frame_times.append(float(self._next_frame_time))

    def capture_initial(self) -> None:
        if self._states:
            raise RuntimeError("initial state has already been captured")
        state_index = self._capture_state()
        self._write_frame(state_index)
        self._next_frame_time = 1.0 / self.video_fps

    def record_step(
        self,
        policy_action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        extras: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._states:
            raise RuntimeError("capture_initial() must be called before record_step()")
        state_index = self._capture_state()
        self._policy_actions.append(np.asarray(policy_action, dtype=np.float64).copy())
        requested = info.get(
            "requested_action", getattr(self.base_env, "_last_requested_action", [])
        )
        applied = info.get(
            "applied_action", getattr(self.base_env, "_last_applied_action", [])
        )
        self._requested_actions.append(np.asarray(requested, dtype=np.float64).copy())
        self._applied_actions.append(np.asarray(applied, dtype=np.float64).copy())
        self._rewards.append(float(reward))
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))
        self._reward_components.append({
            str(key): float(value)
            for key, value in info.items()
            if str(key).startswith("reward_") and np.isscalar(value)
        })
        for name, value in (extras or {}).items():
            self._extras.setdefault(str(name), []).append(np.asarray(value).copy())
        now = float(self.base_env.data.time)
        while now + 1e-12 >= self._next_frame_time:
            self._write_frame(state_index)
            self._next_frame_time += 1.0 / self.video_fps

    @staticmethod
    def _stack(values: list[np.ndarray]) -> np.ndarray:
        if not values:
            return np.empty((0, 0), dtype=np.float64)
        shapes = {value.shape for value in values}
        if len(shapes) != 1:
            raise RuntimeError(f"inconsistent action shapes in episode: {shapes}")
        return np.stack(values)

    def finish(self, metadata: Mapping[str, Any]) -> EpisodeArtifacts:
        if self._closed:
            raise RuntimeError("episode recorder is already closed")
        reward_names = sorted({
            key for components in self._reward_components for key in components
        })
        reward_values = np.asarray([
            [components.get(name, np.nan) for name in reward_names]
            for components in self._reward_components
        ], dtype=np.float64)
        extra_arrays = {
            f"extra_{name}": self._stack(values)
            for name, values in self._extras.items()
        }
        np.savez_compressed(
            self.artifacts.trajectory,
            schema_version=np.asarray(RECORDING_SCHEMA_VERSION, dtype=np.int64),
            state_spec=np.asarray(int(self.state_spec), dtype=np.int64),
            states=np.stack(self._states),
            state_times=np.asarray(self._state_times, dtype=np.float64),
            policy_actions=self._stack(self._policy_actions),
            requested_actions=self._stack(self._requested_actions),
            applied_actions=self._stack(self._applied_actions),
            rewards=np.asarray(self._rewards, dtype=np.float64),
            terminated=np.asarray(self._terminated, dtype=np.bool_),
            truncated=np.asarray(self._truncated, dtype=np.bool_),
            reward_component_names=np.asarray(reward_names),
            reward_components=reward_values,
            frame_state_indices=np.asarray(self._frame_state_indices, dtype=np.int64),
            frame_times=np.asarray(self._frame_times, dtype=np.float64),
            video_fps=np.asarray(self.video_fps, dtype=np.float64),
            **extra_arrays,
        )
        document = {
            "recording_schema_version": RECORDING_SCHEMA_VERSION,
            "state_format": "mujoco_mjSTATE_FULLPHYSICS",
            "state_count": len(self._states),
            "control_step_count": len(self._policy_actions),
            "video_frame_count": len(self._frame_state_indices),
            "video_fps": self.video_fps,
            "files": {
                "trajectory": self.artifacts.trajectory.name,
                "global_video": self.artifacts.global_video.name,
                "wrist_video": self.artifacts.wrist_video.name,
            },
            "result": _json_value(dict(metadata)),
        }
        self.artifacts.metadata.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.close()
        return self.artifacts

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._global_writer.release()
        self._wrist_writer.release()

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
