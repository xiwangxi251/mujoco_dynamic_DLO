"""Direct reader for canonical DynamicVLA LeRobot exports."""

from __future__ import annotations

from collections import OrderedDict
import json
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _Episode:
    index: int
    state: np.ndarray
    action: np.ndarray
    opst_video: Path
    wrist_video: Path


def _read_video(path: Path, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    frames: list[np.ndarray] = []
    container = av.open(str(path))
    try:
        for frame in container.decode(video=0):
            if frame.width != width or frame.height != height:
                frame = frame.reformat(width=width, height=height, format="rgb24")
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
    return np.stack(frames).astype(np.uint8, copy=False)


class DynamicVLAEpisodeDataset(Dataset):
    """Image-DP samples with DynamicVLA's state/action semantics.

    Parquet ``action`` is absolute ``[xyz, euler_xyz, gripper]``.  Every
    action in a prediction chunk is converted to a delta from its first/current
    state, matching DynamicVLA's ``delta_action=True`` loader.
    """

    state_format = "dynamicvla_absolute_xyz_euler_xyz"
    action_format = "dynamicvla_chunk_delta_xyz_euler_xyz_gripper"

    def __init__(self, root: Path, *, config, episode_indices=None, episodes=None,
                 state_low=None, state_high=None, action_low=None, action_high=None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.config = config
        self.episodes = self._load_episodes(self.root) if episodes is None else episodes
        if not self.episodes:
            raise ValueError("DynamicVLA dataset contains no episodes")
        self.episode_indices = list(range(len(self.episodes))) if episode_indices is None else list(episode_indices)
        if not self.episode_indices:
            raise ValueError("dataset split contains no episodes")
        self.samples = [(episode_index, frame_index) for episode_index in self.episode_indices
                        for frame_index in range(len(self.episodes[episode_index].state))]
        if not self.samples:
            raise ValueError("dataset contains no frames")
        if state_low is None:
            state_low, state_high, action_low, action_high = self._normalization_bounds()
        self._state_low = np.asarray(state_low, dtype=np.float32)
        self._state_high = np.asarray(state_high, dtype=np.float32)
        self._action_low = np.asarray(action_low, dtype=np.float32)
        self._action_high = np.asarray(action_high, dtype=np.float32)
        self._video_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    @staticmethod
    def _load_episodes(root: Path) -> list[_Episode]:
        import pandas as pd

        info_path, episodes_path = root / "meta" / "info.json", root / "meta" / "episodes.jsonl"
        if not info_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(f"not a DynamicVLA LeRobot export: {root}")
        chunk_size = int(json.loads(info_path.read_text(encoding="utf-8"))["chunks_size"])
        rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result: list[_Episode] = []
        for row in rows:
            index, chunk = int(row["episode_index"]), int(row["episode_index"]) // chunk_size
            parquet = root / "data" / f"chunk-{chunk:03d}" / f"episode_{index:06d}.parquet"
            frame = pd.read_parquet(parquet, columns=["observation.state", "action"])
            state = np.asarray(frame["observation.state"].tolist(), dtype=np.float32)
            action = np.asarray(frame["action"].tolist(), dtype=np.float32)
            if state.ndim != 2 or state.shape[1] != 6 or action.ndim != 2 or action.shape[1] != 7:
                raise ValueError(f"{parquet}: expected (T,6) state and (T,7) action, got {state.shape}/{action.shape}")
            if len(state) != len(action) or not len(state):
                raise ValueError(f"{parquet}: invalid state/action frame count")
            video_root = root / "videos" / f"chunk-{chunk:03d}"
            opst = video_root / "observation.images.opst_cam" / f"episode_{index:06d}.mp4"
            wrist = video_root / "observation.images.wrist_cam" / f"episode_{index:06d}.mp4"
            if not opst.is_file() or not wrist.is_file():
                raise FileNotFoundError(f"dual-camera video missing for episode {index}")
            result.append(_Episode(index, state, action, opst, wrist))
        return result

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def state_low(self) -> np.ndarray:
        return self._state_low

    @property
    def state_high(self) -> np.ndarray:
        return self._state_high

    @property
    def action_low(self) -> np.ndarray:
        return self._action_low

    @property
    def action_high(self) -> np.ndarray:
        return self._action_high

    def subset(self, episode_indices):
        return type(self)(self.root, config=self.config, episode_indices=episode_indices,
                          episodes=self.episodes, state_low=self._state_low, state_high=self._state_high,
                          action_low=self._action_low, action_high=self._action_high)

    def _normalization_bounds(self):
        state_values = np.concatenate([episode.state for episode in self.episodes], axis=0)
        delta_values: list[np.ndarray] = []
        offsets = np.arange(self.config.prediction_horizon, dtype=np.int64)
        for episode in self.episodes:
            starts = np.arange(0, len(episode.state), 8, dtype=np.int64)
            future = np.minimum(starts[:, None] + offsets[None, :], len(episode.state) - 1)
            target = episode.action[future].copy()
            target[:, :, :6] -= episode.state[starts, None, :6]
            delta_values.append(target.reshape(-1, 7))
        action_values = np.concatenate(delta_values, axis=0)
        return tuple(np.quantile(values, quantile, axis=0).astype(np.float32)
                     for values, quantile in ((state_values, .01), (state_values, .99),
                                               (action_values, .01), (action_values, .99)))

    @staticmethod
    def _normalized(value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        normalized = 2.0 * (value - low) / np.maximum(high - low, 1.0e-6) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _videos(self, episode_index: int):
        cached = self._video_cache.get(episode_index)
        if cached is not None:
            self._video_cache.move_to_end(episode_index)
            return cached
        episode = self.episodes[episode_index]
        videos = (_read_video(episode.opst_video, (self.config.image_width, self.config.image_height)),
                  _read_video(episode.wrist_video, (self.config.image_width, self.config.image_height)))
        frame_count = len(episode.state)
        if len(videos[0]) < frame_count or len(videos[1]) < frame_count:
            raise RuntimeError(f"video alignment failed for episode {episode.index}")
        self._video_cache[episode_index] = videos
        self._video_cache.move_to_end(episode_index)
        while len(self._video_cache) > self.config.cache_episodes:
            self._video_cache.popitem(last=False)
        return videos

    def __getitem__(self, item: int):
        episode_index, frame_index = self.samples[item]
        episode = self.episodes[episode_index]
        opst, wrist = self._videos(episode_index)
        history = [max(0, frame_index - self.config.observation_horizon + 1 + i)
                   for i in range(self.config.observation_horizon)]
        future = [min(len(episode.action) - 1, frame_index + i)
                  for i in range(self.config.prediction_horizon)]
        target = episode.action[future].copy()
        target[:, :6] -= episode.state[frame_index, :6]
        return {
            "opst_cam": torch.from_numpy(opst[history].transpose(0, 3, 1, 2)).float() / 255.0,
            "wrist_cam": torch.from_numpy(wrist[history].transpose(0, 3, 1, 2)).float() / 255.0,
            "state": torch.from_numpy(self._normalized(episode.state[history], self._state_low, self._state_high)),
            "action": torch.from_numpy(self._normalized(target, self._action_low, self._action_high)),
        }
