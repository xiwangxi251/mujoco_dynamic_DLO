"""Dataset adapters for raw expert episodes and DynamicVLA LeRobot data."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import json
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


def _is_lerobot_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "meta" / "info.json").is_file()
        and (path / "data").is_dir()
        and (path / "videos").is_dir()
    )


def _lerobot_manifest(root: Path) -> dict:
    path = root / "cable_conversion_manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    """Reuse the immutable FK model shared by raw trajectories from one scene."""

    return JointTargetForwardKinematics(Path(model_path))


def _episode_record(path: Path, action_source: str, gripper_threshold: float) -> EpisodeRecord:
    from .config import wxyz_to_euler_xyz

    with np.load(path, allow_pickle=False) as data:
        required = {"hand_position", "hand_quaternion", "scenario_name", "seed"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                f"{path} is not a privileged-expert trajectory; missing {missing}"
            )
        state = np.concatenate(
            (data["hand_position"], wxyz_to_euler_xyz(data["hand_quaternion"])),
            axis=-1,
        )
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
    action_euler = wxyz_to_euler_xyz(action_quaternion)
    gripper = np.where(command[:, 7] > gripper_threshold, 1.0, -1.0)[:, None]
    action = np.concatenate((action_position, action_euler, gripper), axis=-1)
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


def _delta_timestamps(horizon: int, fps: int, include_current: bool) -> list[float]:
    step = 1.0 / float(fps)
    if include_current:
        return [round(index * step, 8) for index in range(horizon)]
    return [round(-(horizon - 1 - index) * step, 8) for index in range(horizon)]


def _normalize_tensor(value: torch.Tensor, low, high) -> torch.Tensor:
    lower = torch.as_tensor(low, dtype=value.dtype, device=value.device)
    upper = torch.as_tensor(high, dtype=value.dtype, device=value.device)
    return 2.0 * (value - lower) / (upper - lower) - 1.0


class DiffusionEpisodeDataset(Dataset):
    """Sequence dataset with direct LeRobot v2.1 support.

    A LeRobot root is consumed through ``LeRobotDataset`` using its indexed
    Parquet rows, episode boundaries, and timestamp-based video decoder.  The
    stored DynamicVLA state/action are kept as 6-D Euler state and 7-D Euler
    action.  Raw privileged ``.npz`` input remains supported as a compatibility
    fallback and is converted to the same representation once at construction.
    """

    def __init__(
        self,
        inputs: Iterable[Path],
        *,
        config,
        action_source: str | None = None,
        gripper_threshold: float = 127.5,
        episode_indices: Iterable[int] | None = None,
    ) -> None:
        self.config = config
        paths = [Path(item).expanduser().resolve() for item in inputs]
        if len(paths) == 1 and _is_lerobot_root(paths[0]):
            self._init_lerobot(paths[0], action_source, episode_indices)
            return
        if any(_is_lerobot_root(path) for path in paths):
            raise ValueError("LeRobot input must be the only dataset input")
        resolved_action_source = action_source or "applied"
        records = [
            _episode_record(path, resolved_action_source, gripper_threshold)
            for path in _find_trajectories(paths)
        ]
        self._source_kind = "raw"
        self.action_source = resolved_action_source
        self._total_episode_count = len(records)
        self._initialize_from_records(records, episode_indices)

    def _init_lerobot(
        self,
        root: Path,
        action_source: str | None,
        episode_indices: Iterable[int] | None,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        manifest = _lerobot_manifest(root)
        stored_source = manifest.get("joint_target_source")
        if stored_source not in {None, "requested", "applied"}:
            raise ValueError(f"unsupported joint_target_source in {root}: {stored_source}")
        if action_source is not None and stored_source is not None and action_source != stored_source:
            raise ValueError(
                f"LeRobot data stores action source {stored_source!r}, but "
                f"--action-source requested {action_source!r}"
            )
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        repo_id = manifest.get("repo_id", f"local/{root.name}")
        self._lerobot = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            download_videos=False,
            delta_timestamps={
                "observation.images.opst_cam": _delta_timestamps(
                    self.config.observation_horizon, int(info["fps"]), False
                ),
                "observation.images.wrist_cam": _delta_timestamps(
                    self.config.observation_horizon, int(info["fps"]), False
                ),
                "observation.state": _delta_timestamps(
                    self.config.observation_horizon, int(info["fps"]), False
                ),
                "action": _delta_timestamps(
                    self.config.prediction_horizon, int(info["fps"]), True
                ),
            },
        )
        expected_state = tuple(info["features"]["observation.state"]["shape"])
        expected_action = tuple(info["features"]["action"]["shape"])
        if expected_state != (6,) or expected_action != (7,):
            raise ValueError(
                f"expected DynamicVLA 6D/7D LeRobot data, got "
                f"state={expected_state}, action={expected_action}"
            )
        self._source_kind = "lerobot"
        self.action_source = stored_source or action_source or "unknown"
        self._total_episode_count = int(info["total_episodes"])
        self._initialize_lerobot_subset(episode_indices)

    def _initialize_lerobot_subset(self, episode_indices: Iterable[int] | None) -> None:
        selected = (
            list(range(self._total_episode_count))
            if episode_indices is None
            else list(episode_indices)
        )
        if not selected:
            raise ValueError("dataset split contains no episodes")
        if any(index < 0 or index >= self._total_episode_count for index in selected):
            raise IndexError("episode split index is out of range")
        self.episode_indices = selected
        self.samples: list[int] = []
        for episode_index in selected:
            start = int(self._lerobot.episode_data_index["from"][episode_index])
            end = int(self._lerobot.episode_data_index["to"][episode_index])
            self.samples.extend(range(start, end))
        if not self.samples:
            raise ValueError("dataset contains no frames")

    @classmethod
    def from_records(
        cls,
        records: list[EpisodeRecord],
        *,
        config,
        action_source: str,
        episode_indices: Iterable[int] | None = None,
    ) -> "DiffusionEpisodeDataset":
        dataset = cls.__new__(cls)
        dataset.config = config
        dataset._source_kind = "raw"
        dataset.action_source = action_source
        dataset._total_episode_count = len(records)
        dataset._initialize_from_records(records, episode_indices)
        return dataset

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

    def subset(self, episode_indices: Iterable[int]) -> "DiffusionEpisodeDataset":
        dataset = type(self).__new__(type(self))
        dataset.config = self.config
        dataset._source_kind = self._source_kind
        dataset.action_source = self.action_source
        dataset._total_episode_count = self._total_episode_count
        if self._source_kind == "lerobot":
            dataset._lerobot = self._lerobot
            dataset._initialize_lerobot_subset(episode_indices)
        else:
            dataset._initialize_from_records(self.records, episode_indices)
        return dataset

    @property
    def episode_count(self) -> int:
        return self._total_episode_count

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

    def _lerobot_item(self, item: int) -> dict[str, torch.Tensor]:
        value = self._lerobot[self.samples[item]]
        opst = value["observation.images.opst_cam"].float()
        wrist = value["observation.images.wrist_cam"].float()
        image_size = (self.config.image_height, self.config.image_width)
        if tuple(opst.shape[-2:]) != image_size:
            opst = torch.nn.functional.interpolate(opst, size=image_size, mode="bilinear", align_corners=False)
            wrist = torch.nn.functional.interpolate(wrist, size=image_size, mode="bilinear", align_corners=False)
        state = _normalize_tensor(value["observation.state"].float(), STATE_LOW, STATE_HIGH)
        action = _normalize_tensor(value["action"].float(), ACTION_LOW, ACTION_HIGH)
        return {"opst_cam": opst, "wrist_cam": wrist, "state": state, "action": action}

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        if self._source_kind == "lerobot":
            return self._lerobot_item(item)
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
