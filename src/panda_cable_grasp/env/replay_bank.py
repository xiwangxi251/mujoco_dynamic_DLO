"""Recorded material-node trajectories for the ``rigid_replay_v1`` profile.

A replay bank stores the free-run planar motion of every material node of the
canonical cable for a set of episode seeds of a source scenario.  The
``rigid_replay_v1`` motion profile replays one recorded node trajectory as the
rigid transform (planar COM translation plus yaw about the COM) of the whole
shape-frozen cable.  Because a paired seed produces the same initial curve and
target index, the replayed cable's tracked node has kinematics identical to
the corresponding material point in the paired deforming episode, while every
other node co-moves rigidly instead of deforming.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BANK_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ReplayTrajectory:
    """Planar reference extracted for one node of one bank entry."""

    delta_position: np.ndarray  # (T, 2) node xy minus its t=0 position
    velocity: np.ndarray        # (T, 2) finite-difference node velocity
    delta_yaw: np.ndarray       # (T,) whole-cable rigid yaw minus t=0 yaw
    yaw_rate: np.ndarray        # (T,)


class ReplayBank:
    """Lazy mmap loader for ``<name>.npz`` replay-bank files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        archive = np.load(self.path, mmap_mode="r", allow_pickle=False)
        version = int(np.asarray(archive["format_version"]))
        if version != BANK_FORMAT_VERSION:
            raise ValueError(
                f"unsupported replay bank format {version} in {self.path}"
            )
        self.meta = json.loads(str(np.asarray(archive["meta_json"])))
        self.seeds = np.asarray(archive["seeds"], dtype=np.int64)
        self.positions_xy = archive["positions_xy"]  # (E, T, N, 2) float32
        self.valid_steps = np.asarray(archive["valid_steps"], dtype=np.int64)
        self.placed_com_xy = np.asarray(
            archive["placed_com_xy"], dtype=np.float64
        )
        self.target_index = np.asarray(archive["target_index"], dtype=np.int64)
        self.cable_ids = np.asarray(archive["cable_ids"], dtype=np.int64)
        self.control_dt = float(self.meta["control_dt"])
        self._seed_to_entry = {
            int(seed): index for index, seed in enumerate(self.seeds)
        }
        self._yaw_cache: dict[int, np.ndarray] = {}

    @property
    def entry_count(self) -> int:
        return int(self.seeds.size)

    @property
    def horizon_steps(self) -> int:
        return int(self.positions_xy.shape[1])

    @property
    def node_count(self) -> int:
        return int(self.positions_xy.shape[2])

    @property
    def source_motion_mode(self) -> str:
        return str(self.meta["source_motion_mode"])

    @property
    def source_scenario(self) -> str:
        return str(self.meta["source_scenario"])

    @property
    def source_start_y_placement(self) -> bool:
        return bool(self.meta["source_start_y_placement"])

    def index_for_seed(self, seed: int | None) -> int | None:
        if seed is None:
            return None
        return self._seed_to_entry.get(int(seed))

    def check_compatible(self, cable_ids: np.ndarray, control_dt: float) -> None:
        """Reject banks recorded under a different node ordering or cadence."""
        if self.node_count != len(cable_ids) or not np.array_equal(
            self.cable_ids, np.asarray(cable_ids, dtype=np.int64)
        ):
            raise ValueError(
                "replay bank node ordering does not match this model: "
                f"bank ids={self.cable_ids.tolist()} "
                f"env ids={list(map(int, cable_ids))}"
            )
        if not math.isclose(self.control_dt, control_dt, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"replay bank control_dt={self.control_dt} does not match "
                f"env control_dt={control_dt}"
            )

    def trajectory(self, entry: int, node_index: int) -> ReplayTrajectory:
        """Return the delta trajectory of ``node_index`` within ``entry``.

        Rows beyond ``valid_steps[entry]`` are required to be padded with the
        last recorded pose by the bank writer, so the frozen tail naturally
        yields zero reference velocity/yaw rate.
        """
        node = np.asarray(
            self.positions_xy[entry, :, node_index], dtype=np.float64
        )
        delta = node - node[0]
        velocity = np.gradient(node, self.control_dt, axis=0, edge_order=1)
        yaw = self._global_yaw(entry)
        return ReplayTrajectory(
            delta_position=delta,
            velocity=velocity,
            delta_yaw=yaw - yaw[0],
            yaw_rate=np.gradient(yaw, self.control_dt, edge_order=1),
        )

    def _global_yaw(self, entry: int) -> np.ndarray:
        """整条线缆相邻帧最优刚性旋转角（Kabsch/Procrustes）的累积 yaw。

        局部切线角会把内部形变误记成整体旋转（线缆盘卷时切线可高速甩动），
        这里改为每帧求 ``X_t -> X_{t+1}`` 的最优二维旋转并累加，只保留真正
        的刚体旋转分量。
        """
        cached = self._yaw_cache.get(entry)
        if cached is not None:
            return cached
        xy = np.asarray(self.positions_xy[entry], dtype=np.float64)
        yaw = np.zeros(xy.shape[0])
        for t in range(1, xy.shape[0]):
            x0 = xy[t - 1] - xy[t - 1].mean(axis=0)
            x1 = xy[t] - xy[t].mean(axis=0)
            # 形状近似旋转对称（盘成正圆等）时整体旋转不可辨识，
            # Kabsch 的奇异向量退化、角度会随机翻转——保持 yaw 不变。
            cov = x0.T @ x0
            eigvals = np.linalg.eigvalsh(cov)
            if eigvals[-1] <= 0.0 or eigvals[0] / eigvals[-1] > 0.9:
                yaw[t] = yaw[t - 1]
                continue
            u, _, vt = np.linalg.svd(x0.T @ x1)
            rotation = u @ vt
            delta = math.atan2(rotation[0, 1], rotation[0, 0])
            # 线缆接近直线时旋转有 ±pi 歧义；50 Hz 下单帧真实旋转远小于
            # pi，按 mod-pi 取与上一帧增量最连续的分支消除翻转。
            prev_delta = yaw[t - 1] - yaw[t - 2] if t >= 2 else 0.0
            delta += math.pi * round((prev_delta - delta) / math.pi)
            # 近退化形状下 Kabsch 仍可能给出 ~pi/2 的噪声增量；真实刚性
            # 旋转在库中不超过 ~0.14 rad/帧，截断到 0.2 rad 兜底。
            delta = max(-0.2, min(0.2, delta))
            yaw[t] = yaw[t - 1] + delta
        self._yaw_cache[entry] = yaw
        return yaw
