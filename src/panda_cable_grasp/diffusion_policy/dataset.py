"""Dataset adapter for the existing privileged-expert dual-camera recordings."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..dynamicvla.finetune.convert_dataset import JointTargetForwardKinematics
from .config import (
    ACTION_HIGH,
    ACTION_LOW,
    STATE_HIGH,
    STATE_LOW,
    normalize_array,
    split_episode_indices,
)


@dataclass(frozen=True)
class EpisodeRecord:
    trajectory: Path
    opst_video: Path
    wrist_video: Path
    state: np.ndarray
    action: np.ndarray
    scenario: str
    seed: int


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _find_trajectories(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for input_path in inputs:
        path = Path(input_path).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() != ".npz":
                raise ValueError(f"expected an .npz trajectory, got {path}")
            paths.add(path)
        elif path.is_dir():
            for pattern in ("episode_*.npz", "trajectory.npz"):
                paths.update(item.resolve() for item in path.rglob(pattern))
        else:
            raise FileNotFoundError(path)
    if not paths:
        raise ValueError("no .npz trajectories found")
    return sorted(paths)


def _read_video(path: Path, size: tuple[int, int]) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    width, height = size
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
    return np.stack(frames).astype(np.uint8, copy=False)


@lru_cache(maxsize=32)
def _forward_kinematics(model_path: str) -> JointTargetForwardKinematics:
    """Reuse the immutable FK model shared by trajectories from one scene."""

    return JointTargetForwardKinematics(Path(model_path))


def _episode_record(
    path: Path,
    action_source: str,
    gripper_threshold: float | None,
) -> EpisodeRecord:
    with np.load(path, allow_pickle=False) as data:
        required = {"hand_position", "hand_quaternion", "scenario_name", "seed"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                f"{path} is not a privileged-expert trajectory; missing {missing}"
            )
        state = np.concatenate((data["hand_position"], data["hand_quaternion"]), axis=-1)
        if action_source not in {"requested", "applied"}:
            raise ValueError("action_source must be requested or applied")
        command_key = f"{action_source}_actions"
        if command_key not in data.files:
            raise ValueError(f"{path} is missing {command_key}")
        command = np.asarray(data[command_key], dtype=np.float64)
        if command.ndim != 2 or command.shape[1] != 8:
            raise ValueError(f"{command_key} must have shape (T,8), got {command.shape}")
        model_path = path.parent / _scalar(data["model_file"])
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        opst_video = path.parent / _scalar(data["opst_video_file"])
        wrist_video = path.parent / _scalar(data["wrist_video_file"])
        scenario = _scalar(data["scenario_name"])
        seed = int(data["seed"])

    fk = _forward_kinematics(str(model_path.resolve()))
    action_position, action_quaternion = fk.poses(command[:, :7])
    threshold = (
        float(gripper_threshold)
        if gripper_threshold is not None
        else float(np.mean(fk.gripper_ctrl_range))
    )
    gripper = np.where(
        command[:, fk.gripper_actuator_id] > threshold, 1.0, -1.0
    )[:, None]
    action = np.concatenate((action_position, action_quaternion, gripper), axis=-1)
    state = np.asarray(state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    frame_count = min(len(state), len(action))
    if frame_count < 1:
        raise ValueError(f"empty trajectory: {path}")
    if not opst_video.is_file() or not wrist_video.is_file():
        raise FileNotFoundError(f"dual-camera videos missing beside {path}")
    return EpisodeRecord(
        trajectory=path,
        opst_video=opst_video,
        wrist_video=wrist_video,
        state=state[:frame_count],
        action=action[:frame_count],
        scenario=scenario,
        seed=seed,
    )


class DiffusionEpisodeDataset(Dataset):
    """Lazy dual-camera sequence dataset with action-chunk padding.

    Each item contains normalized tensors with keys ``opst_cam``, ``wrist_cam``,
    ``state`` and ``action``.  The camera/state history is padded with the first
    observation at episode boundaries, and future actions are padded with the
    final demonstrated command.  This matches the receding-horizon semantics
    used during inference.
    """

    def __init__(
        self,
        inputs: Iterable[Path],
        *,
        config,
        action_source: str = "applied",
        gripper_threshold: float | None = None,
        episode_indices: Iterable[int] | None = None,
    ) -> None:
        self.config = config
        records = [
            _episode_record(path, action_source, gripper_threshold)
            for path in _find_trajectories(inputs)
        ]
        self._initialize_from_records(records, episode_indices)

    @classmethod
    def from_records(
        cls,
        records: list[EpisodeRecord],
        *,
        config,
        episode_indices: Iterable[int] | None = None,
    ) -> "DiffusionEpisodeDataset":
        """Create a split view without reparsing trajectories or rebuilding FK."""

        dataset = cls.__new__(cls)
        dataset.config = config
        dataset._initialize_from_records(records, episode_indices)
        return dataset

    def subset(self, episode_indices: Iterable[int]) -> "DiffusionEpisodeDataset":
        """Return a split view sharing the already parsed episode records."""

        return type(self).from_records(
            self.records,
            config=self.config,
            episode_indices=episode_indices,
        )

    def _initialize_from_records(
        self,
        records: list[EpisodeRecord],
        episode_indices: Iterable[int] | None,
    ) -> None:
        selected = list(range(len(records))) if episode_indices is None else list(episode_indices)
        if not selected:
            raise ValueError("dataset split contains no episodes")
        if any(index < 0 or index >= len(records) for index in selected):
            raise IndexError("episode split index is out of range")
        self.records = records
        self.episode_indices = selected
        self.samples: list[tuple[int, int]] = [
            (episode_index, frame_index)
            for episode_index in selected
            for frame_index in range(len(records[episode_index].state))
        ]
        if not self.samples:
            raise ValueError("dataset contains no frames")
        self._video_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    @property
    def episode_count(self) -> int:
        return len(self.records)

    @property
    def action_low(self) -> np.ndarray:
        return np.asarray(ACTION_LOW, dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.asarray(ACTION_HIGH, dtype=np.float32)

    @property
    def state_low(self) -> np.ndarray:
        return np.asarray(STATE_LOW, dtype=np.float32)

    @property
    def state_high(self) -> np.ndarray:
        return np.asarray(STATE_HIGH, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _videos(self, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._video_cache.get(episode_index)
        if cached is not None:
            self._video_cache.move_to_end(episode_index)
            return cached
        record = self.records[episode_index]
        size = (self.config.image_width, self.config.image_height)
        videos = (_read_video(record.opst_video, size), _read_video(record.wrist_video, size))
        frame_count = len(record.state)
        if len(videos[0]) < frame_count or len(videos[1]) < frame_count:
            raise RuntimeError(
                f"video/trajectory alignment failed for {record.trajectory}: "
                f"states={frame_count}, videos={len(videos[0])}/{len(videos[1])}"
            )
        self._video_cache[episode_index] = videos
        self._video_cache.move_to_end(episode_index)
        while len(self._video_cache) > self.config.cache_episodes:
            self._video_cache.popitem(last=False)
        return videos

    @staticmethod
    def _history_indices(index: int, horizon: int) -> list[int]:
        return [max(0, index - horizon + 1 + offset) for offset in range(horizon)]

    @staticmethod
    def _future_indices(index: int, count: int, length: int) -> list[int]:
        return [min(length - 1, index + offset) for offset in range(count)]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, frame_index = self.samples[item]
        record = self.records[episode_index]
        opst, wrist = self._videos(episode_index)
        history = self._history_indices(frame_index, self.config.observation_horizon)
        future = self._future_indices(
            frame_index, self.config.prediction_horizon, len(record.action)
        )
        opst_tensor = torch.from_numpy(opst[history].transpose(0, 3, 1, 2)).float() / 255.0
        wrist_tensor = torch.from_numpy(wrist[history].transpose(0, 3, 1, 2)).float() / 255.0
        state = normalize_array(record.state[history], STATE_LOW, STATE_HIGH)
        action = normalize_array(record.action[future], ACTION_LOW, ACTION_HIGH)
        return {
            "opst_cam": opst_tensor,
            "wrist_cam": wrist_tensor,
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
        }
