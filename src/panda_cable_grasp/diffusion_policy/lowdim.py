"""Low-dimensional Diffusion Policy for the Panda cable task.

This module is deliberately separate from the visual policy.  It consumes the
privileged expert ``.npz`` trajectories for DLO state, while reusing the
already converted LeRobot parquet actions when available.  Consequently the
training path never decodes video and never performs per-sample FK.

The observation is the absolute end-effector pose and gripper aperture,
followed by 16 uniformly sampled DLO keypoints relative to the gripper and the
relative target position.  ``include_velocity`` adds the corresponding DLO
and target velocities.  The history length is the experimental variable H.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..dynamicvla.finetune.convert_dataset import (
    JointTargetForwardKinematics,
    _select_frame_indices,
    _wxyz_to_euler_xyz,
)
from ..env.kinematics import rotation_to_quat
from .config import (
    ACTION_HIGH,
    ACTION_LOW,
    denormalize_array,
    normalize_array,
)

# Older portability snapshots only define the quaternion-space bounds.  The
# Pi0.5-aligned path uses fitted q01/q99 bounds; these aliases keep legacy
# rotvec checkpoints loadable without requiring a config rewrite.
ACTION_LOW_ROTVEC = ACTION_LOW
ACTION_HIGH_ROTVEC = ACTION_HIGH
from .model import ConditionalUnet1D, DDPMSchedule
from scipy.spatial.transform import Rotation


def wxyz_to_euler_xyz(quaternions: np.ndarray) -> np.ndarray:
    """Use the DynamicVLA conversion convention for observations/actions."""

    return _wxyz_to_euler_xyz(quaternions)


def wxyz_to_rotvec(quaternions: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternions to 3-D rotation vectors."""

    values = np.asarray(quaternions, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError(f"quaternion array must end in 4 values, got {values.shape}")
    return Rotation.from_quat(values, scalar_first=True).as_rotvec().astype(np.float32)


def euler_xyz_to_wxyz(euler_angles: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles to scalar-first quaternions."""

    values = np.asarray(euler_angles, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError(f"Euler array must end in 3 values, got {values.shape}")
    return Rotation.from_euler("xyz", values).as_quat(
        scalar_first=True
    ).astype(np.float32)


def rotvec_to_wxyz(rotation_vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(rotation_vectors, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError(
            f"rotation-vector array must end in 3 values, got {values.shape}"
        )
    return Rotation.from_rotvec(values).as_quat(scalar_first=True).astype(np.float32)


def _pi05_quantile_normalize(
    value: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Match Pi0.5's q01/q99 normalizer, including [-1, 1] clipping."""

    value = np.asarray(value, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.maximum(np.asarray(upper, dtype=np.float32), lower + 1.0e-6)
    normalized = 2.0 * (value - lower) / (upper - lower) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


def _pi05_quantile_denormalize(
    value: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Undo Pi0.5's q01/q99 normalizer without changing the sampled range."""

    value = np.asarray(value, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.maximum(np.asarray(upper, dtype=np.float32), lower + 1.0e-6)
    return (lower + 0.5 * (value + 1.0) * (upper - lower)).astype(
        np.float32, copy=False
    )


def _make_pi05_action_target(
    observation: np.ndarray, absolute_action: np.ndarray
) -> np.ndarray:
    """Convert [absolute xyz/euler, gripper] to Pi0.5's six-delta target.

    Pi0.5 applies ``DeltaActions(mask=(True,)*6)`` and leaves the gripper
    absolute.  The subtraction intentionally follows that implementation's
    direct Euler subtraction convention.
    """

    observation = np.asarray(observation, dtype=np.float32)
    action = np.asarray(absolute_action, dtype=np.float32).copy()
    if observation.ndim != 2 or observation.shape[-1] < 6:
        raise ValueError(f"observation must have shape (T,>=6), got {observation.shape}")
    if action.ndim != 2 or action.shape != (len(observation), 7):
        raise ValueError(f"absolute action must have shape {(len(observation), 7)}, got {action.shape}")
    action[:, :6] -= observation[:, :6]
    return action


def _replay_nero_success_prefix_features(
    source: Path,
    entry: dict,
    config: "LowDimPolicyConfig",
    frame_count: int,
) -> np.ndarray:
    """Reconstruct privileged DLO observations for the rendered NERO export.

    The success-prefix parquet keeps only the 6D hand pose.  The source NPZ
    still contains the applied 8D actuator commands, so replay those commands
    in the matching NERO scene and sample the same pre-action frames.  This
    restores the cable keypoints and target-relative state without decoding
    camera images or changing the action labels.
    """

    import mujoco

    from ..env.environment import CableGraspEnv
    from ..evaluation.motion_diagnostics import env_config_for_scenario
    from ..scenarios.registry import get_scenario

    with np.load(source, allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=np.float64)
        applied = np.asarray(
            data["applied_actions"]
            if "applied_actions" in data.files
            else data["requested_actions"],
            dtype=np.float64,
        )
    if applied.shape != (len(times), 8):
        raise ValueError(f"invalid NERO replay action shape in {source}: {applied.shape}")
    selected = _select_frame_indices(times, 25)[: int(frame_count)]
    if len(selected) != int(frame_count):
        raise ValueError(
            f"invalid success-prefix length for {source}: {frame_count} from {len(selected)}"
        )

    scenario_name = str(entry["scenario"])
    seed = int(entry["seed"])
    env_config = env_config_for_scenario(
        get_scenario(scenario_name),
        seed=seed,
        episode_seconds=20.0,
        robot="nero",
    )
    # The rendered NERO trajectories and the evaluator both use the middle
    # target and 20 physics substeps per task-space control action.
    env_config.target_selection = "middle"
    env_config.frame_skip = 20
    env = CableGraspEnv(env_config)
    try:
        env.reset(seed=seed, randomize=True)
        keypoints = _keypoint_indices(len(env.cable_ids), config.num_keypoints)
        rows: list[np.ndarray] = []
        selected_cursor = 0
        quat = np.empty(4, dtype=np.float64)
        for source_index, command in enumerate(applied):
            if (
                selected_cursor < len(selected)
                and int(selected[selected_cursor]) == source_index
            ):
                rotation = env.data.xmat[env.hand_id].reshape(3, 3)
                mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
                cable_velocities = None
                target_velocity = None
                if config.include_velocity:
                    cable_velocities = np.stack(
                        [env.body_linear_velocity(body_id) for body_id in env.cable_ids],
                        axis=0,
                    )
                    target_velocity = env.target_velocity()
                rows.append(
                    make_lowdim_feature(
                        env.hand_position,
                        quat,
                        env.data.xpos[env.cable_ids],
                        env.target_position(),
                        env.finger_aperture,
                        keypoint_indices=keypoints,
                        cable_velocities=cable_velocities,
                        target_velocity=target_velocity,
                        rotation_format=config.rotation_format,
                    )
                )
                selected_cursor += 1
                if selected_cursor == len(selected):
                    break

            # applied_actions already includes the collector's velocity and
            # safety limits.  Direct MuJoCo stepping preserves cable
            # disturbance/physics while avoiding evaluator bookkeeping.
            env.data.ctrl[:] = np.clip(
                command,
                env.model.actuator_ctrlrange[:, 0],
                env.model.actuator_ctrlrange[:, 1],
            )
            for _ in range(env_config.frame_skip):
                env.data.xfrc_applied[:] = 0.0
                env._apply_cable_disturbance()
                mujoco.mj_step(env.model, env.data)
        if len(rows) != int(frame_count):
            raise ValueError(
                f"NERO replay ended early for {source}: {len(rows)} vs {frame_count} frames"
            )
        return np.stack(rows).astype(np.float32)
    finally:
        env.close()


@dataclass(frozen=True)
class LowDimPolicyConfig:
    """Architecture and optimization settings for the low-dimensional DP."""

    observation_horizon: int = 2
    prediction_horizon: int = 16
    action_horizon: int = 8
    num_keypoints: int = 16
    include_velocity: bool = False
    diffusion_step_embed_dim: int = 256
    unet_down_dims: tuple[int, ...] = (256, 512, 1024)
    unet_kernel_size: int = 5
    unet_groups: int = 8
    diffusion_steps: int = 100
    inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    batch_size: int = 100
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    seed: int = 20260804
    rotation_format: str = "euler"
    # The original low-dimensional policy uses privileged DLO features.  A
    # NERO DynamicVLA LeRobot export instead provides the 6D EE state.
    observation_dim: int | None = None

    def __post_init__(self) -> None:
        if self.rotation_format not in {"euler", "rotvec"}:
            raise ValueError("rotation_format must be 'euler' or 'rotvec'")
        if self.observation_dim is not None:
            if isinstance(self.observation_dim, bool) or self.observation_dim <= 0:
                raise ValueError("observation_dim must be positive when provided")
        for name in (
            "observation_horizon", "prediction_horizon", "action_horizon",
            "num_keypoints", "diffusion_step_embed_dim", "unet_kernel_size",
            "unet_groups", "diffusion_steps", "inference_steps", "batch_size",
            "warmup_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_horizon > self.prediction_horizon:
            raise ValueError("action_horizon cannot exceed prediction_horizon")
        if not self.unet_down_dims or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.unet_down_dims
        ):
            raise ValueError("unet_down_dims must contain positive integers")
        if any(value % self.unet_groups != 0 for value in self.unet_down_dims):
            raise ValueError("all unet_down_dims must be divisible by unet_groups")
        if self.inference_steps > self.diffusion_steps:
            raise ValueError("inference_steps cannot exceed diffusion_steps")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.grad_clip_norm < 0.0:
            raise ValueError("grad_clip_norm must be non-negative")

    @property
    def state_dim(self) -> int:
        if self.observation_dim is not None:
            return int(self.observation_dim)
        base = 3 + 3 + 1 + self.num_keypoints * 3 + 3
        return base + (self.num_keypoints * 3 + 3 if self.include_velocity else 0)

    def asdict(self) -> dict:
        value = asdict(self)
        value["unet_down_dims"] = list(self.unet_down_dims)
        return value


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _keypoint_indices(node_count: int, count: int) -> np.ndarray:
    if node_count < 1:
        raise ValueError("DLO must contain at least one node")
    if count < 1:
        raise ValueError("num_keypoints must be positive")
    return np.rint(np.linspace(0, node_count - 1, count)).astype(np.int64)


def make_lowdim_feature(
    hand_position: np.ndarray,
    hand_quaternion: np.ndarray,
    cable_positions: np.ndarray,
    target_position: np.ndarray,
    gripper_aperture: float,
    *,
    keypoint_indices: np.ndarray,
    cable_velocities: np.ndarray | None = None,
    target_velocity: np.ndarray | None = None,
    rotation_format: str = "euler",
) -> np.ndarray:
    """Build one low-dimensional observation in the train/eval shared order."""

    hand_position = np.asarray(hand_position, dtype=np.float32).reshape(3)
    hand_quaternion = np.asarray(hand_quaternion, dtype=np.float32).reshape(4)
    cable_positions = np.asarray(cable_positions, dtype=np.float32)
    target_position = np.asarray(target_position, dtype=np.float32).reshape(3)
    if cable_positions.ndim != 2 or cable_positions.shape[-1] != 3:
        raise ValueError(f"cable_positions must have shape (N,3), got {cable_positions.shape}")
    indices = np.asarray(keypoint_indices, dtype=np.int64)
    if np.any(indices >= len(cable_positions)):
        raise ValueError("keypoint index exceeds cable node count")
    if rotation_format == "euler":
        hand_rotation = wxyz_to_euler_xyz(hand_quaternion)
    elif rotation_format == "rotvec":
        hand_rotation = wxyz_to_rotvec(hand_quaternion)
    else:
        raise ValueError("rotation_format must be 'euler' or 'rotvec'")
    cable_relative = cable_positions[indices] - hand_position[None, :]
    target_relative = target_position - hand_position
    parts = [
        hand_position,
        hand_rotation,
        np.asarray([gripper_aperture], dtype=np.float32),
        cable_relative.reshape(-1),
        target_relative,
    ]
    if cable_velocities is not None or target_velocity is not None:
        if cable_velocities is None or target_velocity is None:
            raise ValueError("cable and target velocities must be provided together")
        cable_velocities = np.asarray(cable_velocities, dtype=np.float32)
        target_velocity = np.asarray(target_velocity, dtype=np.float32).reshape(3)
        if cable_velocities.shape != cable_positions.shape:
            raise ValueError("cable velocity and position shapes must match")
        parts.extend([cable_velocities[indices].reshape(-1), target_velocity])
    return np.concatenate(parts).astype(np.float32, copy=False)


@lru_cache(maxsize=32)
def _model_gripper_qpos_addresses(model_path: str) -> tuple[int, ...]:
    import mujoco

    model = mujoco.MjModel.from_binary_path(model_path)
    addresses: list[int] = []
    for names in (("finger_joint1", "gripper_joint1"), ("finger_joint2", "gripper_joint2")):
        joint_id = -1
        for name in names:
            candidate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if candidate >= 0:
                joint_id = candidate
                break
        if joint_id < 0:
            raise ValueError(f"gripper joint {names} not found in {model_path}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return tuple(addresses)


@lru_cache(maxsize=32)
def _forward_kinematics(model_path: str) -> JointTargetForwardKinematics:
    return JointTargetForwardKinematics(Path(model_path))


@dataclass
class LowDimEpisode:
    trajectory: Path
    scenario: str
    seed: int
    observation: np.ndarray
    action: np.ndarray


class _LeRobotActionCache:
    """Read preconverted 7D actions from the existing local LeRobot dataset."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        manifest_path = self.root / "cable_conversion_manifest.json"
        info_path = self.root / "meta" / "info.json"
        if not manifest_path.is_file() or not info_path.is_file():
            raise FileNotFoundError(
                f"LeRobot action cache is missing manifest/info files: {self.root}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        info = json.loads(info_path.read_text(encoding="utf-8"))
        self.chunk_size = int(info.get("chunks_size", 1000))
        self.source_kind = self.manifest.get("joint_target_source")
        self.entries = {
            str(Path(item["source"]).expanduser().resolve()): item
            for item in self.manifest.get("episodes", [])
        }
        self.identity_entries = {
            (str(item["scenario"]), int(item["seed"])): item
            for item in self.manifest.get("episodes", [])
            if "scenario" in item and "seed" in item
        }
        if not self.entries:
            raise ValueError(f"LeRobot action cache has no episode manifest: {self.root}")

    def _read_entry(self, entry: dict, expected_length: int) -> np.ndarray:
        episode_index = int(entry["episode_index"])
        parquet = (
            self.root / "data" / f"chunk-{episode_index // self.chunk_size:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        if not parquet.is_file():
            raise FileNotFoundError(parquet)
        import pandas as pd

        frame = pd.read_parquet(parquet, columns=["action"])
        action = np.asarray(frame["action"].tolist(), dtype=np.float32)
        if action.shape != (expected_length, 7):
            raise ValueError(
                f"cached action/frame mismatch for episode {entry.get('episode_index')}: "
                f"{action.shape} vs {(expected_length, 7)}"
            )
        return action

    def observation_for(self, entry: dict, expected_length: int) -> np.ndarray | None:
        """Read the converted six-dimensional EE state when present."""

        episode_index = int(entry["episode_index"])
        parquet = (
            self.root / "data" / f"chunk-{episode_index // self.chunk_size:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        if not parquet.is_file():
            return None
        import pandas as pd

        frame = pd.read_parquet(parquet, columns=["observation.state"])
        observation = np.asarray(frame["observation.state"].tolist(), dtype=np.float32)
        if observation.shape != (expected_length, 6):
            raise ValueError(
                f"cached observation/frame mismatch for episode {entry.get('episode_index')}: "
                f"{observation.shape} vs {(expected_length, 6)}"
            )
        return observation

    def action_for(self, trajectory: Path, expected_length: int) -> np.ndarray | None:
        entry = self.entries.get(str(trajectory.expanduser().resolve()))
        if entry is None:
            return None
        return self._read_entry(entry, expected_length)

    def action_for_identity(
        self, scenario: str, seed: int, expected_length: int
    ) -> np.ndarray | None:
        entry = self.identity_entries.get((str(scenario), int(seed)))
        if entry is None:
            return None
        return self._read_entry(entry, expected_length)


def _discover_trajectories(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".npz":
            paths.add(path)
        elif path.is_dir():
            # LeRobot-style exports use episode_*.npz, while the privileged
            # NERO collector stores one trajectory.npz inside each seed dir.
            paths.update(path.rglob("episode_*.npz"))
            paths.update(path.rglob("trajectory.npz"))
        else:
            raise FileNotFoundError(path)
    if not paths:
        raise ValueError("no episode_*.npz trajectories found")
    return sorted(path.resolve() for path in paths)


def _decode_calibration_token(token: str) -> float:
    """Decode the collector's filesystem-safe signed decimal token."""

    sign = -1.0 if token.startswith("n") else 1.0
    value = token[1:] if token[:1] in {"n", "p"} else token
    return sign * float(value.replace("p", "."))


def _nero_episode_metadata(path: Path) -> tuple[str, int, tuple[float, float, float, float]]:
    scenario = path.parent.parent.name
    seed_name = path.parent.name
    if seed_name.startswith("seed_"):
        seed = int(seed_name.removeprefix("seed_"))
        calibration = (0.35, 0.01, 0.0, 0.0)
    else:
        # privileged_expert writes one episode_seed*.npz directly under the
        # scenario directory.  Its EnvConfig uses the native NERO geometry:
        # base_offset=0.20 and grasp_center_local=(0.1733, 0, -0.0235).
        match = re.fullmatch(r"episode_seed(?P<seed>\d+)", path.stem)
        if match is None:
            raise ValueError(f"NERO trajectory is not under a seed directory: {path}")
        scenario = path.parent.name
        seed = int(match.group("seed"))
        calibration = (0.20, 0.0, 0.0, 0.0)
    pattern = re.compile(
        r"base_(?P<base>[np0-9]+)_tcpdx_(?P<dx>[np0-9]+)_"
        r"tcpdy_(?P<dy>[np0-9]+)_tcpdz_(?P<dz>[np0-9]+)"
    )
    for part in path.parts:
        match = pattern.fullmatch(part)
        if match:
            calibration = tuple(
                _decode_calibration_token(match.group(name))
                for name in ("base", "dx", "dy", "dz")
            )  # type: ignore[assignment]
            break
    return scenario, seed, calibration


@lru_cache(maxsize=16)
def _nero_state_decoder(
    scenario_name: str,
    base_x: float,
    tcp_dx: float,
    tcp_dy: float,
    tcp_dz: float,
):
    """Create one scene decoder for NERO FULLPHYSICS trajectories.

    The collector stores the complete MuJoCo state rather than the older
    precomputed low-dimensional fields.  Reusing the exact scenario model to
    call ``mj_setState`` keeps cable and target positions faithful to the
    recorded episode instead of attempting to reverse-engineer qpos layouts.
    """

    import mujoco

    from ..env.environment import CableGraspEnv, ROBOT_SPECS
    from ..evaluation.motion_diagnostics import env_config_for_scenario
    from ..scenarios.registry import get_scenario

    scenario = get_scenario(scenario_name)
    config = env_config_for_scenario(
        scenario, seed=20260908, episode_seconds=20.0, robot="nero"
    )
    config.target_selection = "middle"
    previous_spec = ROBOT_SPECS["nero"]
    ROBOT_SPECS["nero"] = replace(
        previous_spec, base_offset=(float(base_x), 0.0, 0.0)
    )
    try:
        env = CableGraspEnv(config)
    finally:
        ROBOT_SPECS["nero"] = previous_spec
    # The collector's tcp_dx/dy/dz calibration changes the task-space grasp
    # center, while the FULLPHYSICS state itself only contains body qpos/qvel.
    env.GRASP_CENTER_LOCAL = np.asarray(
        (0.1733 + tcp_dx, tcp_dy, -0.0235 + tcp_dz), dtype=np.float64
    )
    expected_state_size = mujoco.mj_stateSize(
        env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    return env, expected_state_size


class LowDimEpisodeDataset(Dataset):
    """In-memory frame-window dataset with episode-safe history/future padding."""

    def __init__(
        self,
        inputs: Iterable[Path],
        *,
        config: LowDimPolicyConfig,
        action_source: str = "requested",
        action_cache: Path | None = None,
        episode_indices: Iterable[int] | None = None,
        gripper_threshold: float = 127.5,
    ) -> None:
        if action_source not in {"requested", "applied"}:
            raise ValueError("action_source must be requested or applied")
        self.config = config
        self.action_source = action_source
        self.gripper_threshold = float(gripper_threshold)
        self._action_cache = (
            _LeRobotActionCache(Path(action_cache)) if action_cache is not None else None
        )
        if self._action_cache is not None and self._action_cache.source_kind not in {
            None, action_source
        }:
            raise ValueError(
                f"action cache stores {self._action_cache.source_kind!r}, "
                f"but requested {action_source!r}"
            )
        self._all_episodes = [
            self._load_episode(path) for path in _discover_trajectories(inputs)
        ]
        self._total_episode_count = len(self._all_episodes)
        if self._total_episode_count < 1:
            raise ValueError("dataset contains no episodes")
        selected = (
            list(range(self._total_episode_count))
            if episode_indices is None else list(episode_indices)
        )
        self._set_subset(selected)
        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None
        self._obs_pose_norm_low: np.ndarray | None = None
        self._obs_pose_norm_high: np.ndarray | None = None
        self._action_norm_low: np.ndarray | None = None
        self._action_norm_high: np.ndarray | None = None

    @classmethod
    def from_nero_success_prefix(
        cls,
        root: Path,
        *,
        config: LowDimPolicyConfig,
        action_source: str = "applied",
        limit_episodes: int | None = None,
        feature_cache: Path | None = None,
    ) -> "LowDimEpisodeDataset":
        """Build a 6D/7D low-dimensional dataset from a trimmed NERO export.

        The success-prefix LeRobot database records the trimmed frame counts
        and source NPZ paths in cable_conversion_manifest.json. Reading those
        source NPZs lets this policy use applied_actions and the exact
        DynamicVLA state/action conventions without decoding video or silently
        falling back to another robot.
        """

        if action_source not in {"requested", "applied"}:
            raise ValueError("action_source must be requested or applied")
        root = Path(root).expanduser().resolve()
        manifest_path = root / "cable_conversion_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("episodes", [])
        if not entries:
            raise ValueError(f"NERO success-prefix manifest has no episodes: {root}")
        if limit_episodes is not None:
            if limit_episodes <= 0:
                raise ValueError("limit_episodes must be positive")
            entries = entries[:limit_episodes]
        if feature_cache is not None:
            feature_cache = Path(feature_cache).expanduser().resolve()

        # The success-prefix export already stores the requested/applied 7D
        # task-space actions in parquet.  Reuse them when the reader is
        # available; falling back to FK keeps raw/sparse exports compatible.
        action_cache = None
        try:
            candidate_cache = _LeRobotActionCache(root)
            if candidate_cache.source_kind in {None, action_source}:
                action_cache = candidate_cache
        except (FileNotFoundError, ImportError, ValueError):
            action_cache = None

        episodes: list[LowDimEpisode] = []
        for entry in entries:
            if entry.get("robot") != "nero":
                raise ValueError(f"expected NERO episode, got {entry.get('robot')!r}")
            source = Path(entry["source"]).expanduser().resolve()
            frame_count = int(entry["converted_frames"])
            cached_action = None
            cached_observation = None
            if action_cache is not None:
                cached_action = action_cache.action_for(source, frame_count)
                cached_observation = action_cache.observation_for(entry, frame_count)
                if (
                    config.observation_dim == 6
                    and cached_action is not None
                    and cached_observation is not None
                ):
                    episodes.append(
                        LowDimEpisode(
                            source,
                            str(entry["scenario"]),
                            int(entry["seed"]),
                            cached_observation,
                            _make_pi05_action_target(cached_observation, cached_action),
                        )
                    )
                    continue
            if config.observation_dim is None:
                if not source.is_file():
                    raise FileNotFoundError(source)
                if cached_action is None:
                    raise ValueError(
                        "rich NERO success-prefix input requires a LeRobot action cache"
                    )
                if feature_cache is not None:
                    feature_path = feature_cache / f"{int(entry['episode_index']):05d}.npz"
                    if not feature_path.is_file():
                        raise FileNotFoundError(feature_path)
                    with np.load(feature_path, allow_pickle=False) as feature_data:
                        observation = np.asarray(
                            feature_data["observation"], dtype=np.float32
                        )
                else:
                    observation = _replay_nero_success_prefix_features(
                        source, entry, config, frame_count
                    )
                if (
                    observation.shape[0] != frame_count
                    or observation.shape[1] != config.state_dim
                ):
                    raise ValueError(
                        f"invalid replayed NERO feature shape in {source}: "
                        f"{observation.shape}, expected {(frame_count, config.state_dim)}"
                    )
                episodes.append(
                    LowDimEpisode(
                        source,
                        str(entry["scenario"]),
                        int(entry["seed"]),
                        observation,
                        _make_pi05_action_target(observation, cached_action),
                    )
                )
                continue
            if not source.is_file():
                raise FileNotFoundError(source)
            with np.load(source, allow_pickle=False) as data:
                required = {
                    "times", "hand_position", "hand_quaternion",
                    f"{action_source}_actions", "model_file",
                }
                missing = sorted(required.difference(data.files))
                if missing:
                    raise ValueError(f"{source} is missing NERO fields {missing}")
                times = np.asarray(data["times"], dtype=np.float64)
                selected = _select_frame_indices(times, int(manifest.get("target_fps", 25)))
                frame_count = int(entry["converted_frames"])
                if frame_count <= 0 or len(selected) < frame_count:
                    raise ValueError(
                        f"invalid success-prefix length for {source}: "
                        f"{frame_count} from {len(selected)} source frames"
                    )
                selected = selected[:frame_count]
                hand_position = np.asarray(data["hand_position"], dtype=np.float32)
                hand_quaternion = np.asarray(data["hand_quaternion"], dtype=np.float32)
                command = np.asarray(data[f"{action_source}_actions"], dtype=np.float64)
                if hand_position.shape != (len(times), 3):
                    raise ValueError(f"invalid hand_position shape in {source}: {hand_position.shape}")
                if hand_quaternion.shape != (len(times), 4):
                    raise ValueError(f"invalid hand_quaternion shape in {source}: {hand_quaternion.shape}")
                if command.shape != (len(times), 8):
                    raise ValueError(f"invalid {action_source}_actions shape in {source}: {command.shape}")
                model_path = (source.parent / _scalar(data["model_file"])).resolve()

            frame_count = int(entry["converted_frames"])
            cached_action = (
                None if action_cache is None
                else action_cache.action_for(source, frame_count)
            )
            cached_observation = (
                None if action_cache is None
                else action_cache.observation_for(entry, frame_count)
            )
            if cached_action is not None and cached_observation is not None:
                episodes.append(
                    LowDimEpisode(
                        source,
                        str(entry["scenario"]),
                        int(entry["seed"]),
                        cached_observation,
                        _make_pi05_action_target(cached_observation, cached_action),
                    )
                )
                continue

            cached_action = (
                None if action_cache is None
                else action_cache.action_for(source, frame_count)
            )
            if cached_action is not None:
                action = np.asarray(cached_action, dtype=np.float32)
            else:
                fk = _forward_kinematics(str(model_path))
                action_position, action_quaternion = fk.poses(command[selected, :7])
                action_rotation = (
                    wxyz_to_euler_xyz(action_quaternion)
                    if config.rotation_format == "euler"
                    else wxyz_to_rotvec(action_quaternion)
                )
                gripper_index = int(fk.gripper_actuator_id)
                if not 0 <= gripper_index < command.shape[1]:
                    raise ValueError(
                        f"invalid NERO gripper action index {gripper_index} in {source}"
                    )
                threshold = float(
                    entry.get("gripper_threshold", np.mean(fk.gripper_ctrl_range))
                )
                gripper = np.where(
                    command[selected, gripper_index] > threshold, 1.0, -1.0
                ).astype(np.float32)[:, None]
                action = np.concatenate(
                    (action_position, action_rotation, gripper), axis=-1
                ).astype(np.float32)
            observation = np.concatenate(
                (
                    hand_position[selected],
                    (
                        wxyz_to_euler_xyz(hand_quaternion[selected])
                        if config.rotation_format == "euler"
                        else wxyz_to_rotvec(hand_quaternion[selected])
                    ),
                ),
                axis=-1,
            ).astype(np.float32)
            action = _make_pi05_action_target(observation, action)
            episodes.append(
                LowDimEpisode(
                    source,
                    str(entry["scenario"]),
                    int(entry["seed"]),
                    observation,
                    action,
                )
            )

        dataset = cls.__new__(cls)
        dataset.config = config
        dataset.action_source = action_source
        dataset.gripper_threshold = 0.0
        dataset._action_cache = None
        dataset._all_episodes = episodes
        dataset._total_episode_count = len(episodes)
        dataset._set_subset(list(range(len(episodes))))
        dataset.obs_mean = None
        dataset.obs_std = None
        dataset._obs_pose_norm_low = None
        dataset._obs_pose_norm_high = None
        dataset._action_norm_low = None
        dataset._action_norm_high = None
        return dataset

    def _load_episode(self, path: Path) -> LowDimEpisode:
        with np.load(path, allow_pickle=False) as data:
            if (
                {
                    "state_spec", "states", f"{self.action_source}_actions",
                }.issubset(data.files)
                and ("state_times" in data.files or "times" in data.files)
            ):
                return self._load_nero_fullphysics_episode(path, data)
            required = {
                "times", "hand_position", "hand_quaternion", "cable_positions",
                "target_positions", "states", "model_file", "scenario_name", "seed",
            }
            if self.config.include_velocity:
                required.update({"cable_velocities", "target_velocities"})
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"{path} is missing low-dimensional fields {missing}")
            times = np.asarray(data["times"], dtype=np.float64)
            selected = _select_frame_indices(times, 25)
            hand_position = np.asarray(data["hand_position"], dtype=np.float32)[selected]
            hand_quaternion = np.asarray(data["hand_quaternion"], dtype=np.float32)[selected]
            cable_positions = np.asarray(data["cable_positions"], dtype=np.float32)[selected]
            target_positions = np.asarray(data["target_positions"], dtype=np.float32)[selected]
            states = np.asarray(data["states"], dtype=np.float32)[selected]
            model_path = (path.parent / _scalar(data["model_file"])).resolve()
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            addresses = _model_gripper_qpos_addresses(str(model_path))
            # mj_getState() stores time before qpos, so convert model qpos
            # addresses to indices in the serialized state vector explicitly.
            qpos_indices = np.asarray(addresses, dtype=np.int64) + 1
            if states.shape[1] <= int(np.max(qpos_indices)):
                raise ValueError(f"states do not contain gripper qpos in {path}: {states.shape}")
            # NERO's coupled finger joints have opposite signs; absolute
            # values recover the physical aperture. Panda finger positions are
            # non-negative, so this convention is valid for both robots.
            gripper = np.abs(states[:, qpos_indices]).sum(axis=1)
            keypoints = _keypoint_indices(cable_positions.shape[1], self.config.num_keypoints)
            cable_velocities = None
            target_velocities = None
            if self.config.include_velocity:
                cable_velocities = np.asarray(data["cable_velocities"], dtype=np.float32)[selected]
                target_velocities = np.asarray(data["target_velocities"], dtype=np.float32)[selected]
            feature_rows = [
                make_lowdim_feature(
                    hand_position[index], hand_quaternion[index], cable_positions[index],
                    target_positions[index], gripper[index], keypoint_indices=keypoints,
                    cable_velocities=None if cable_velocities is None else cable_velocities[index],
                    target_velocity=None if target_velocities is None else target_velocities[index],
                    rotation_format=self.config.rotation_format,
                )
                for index in range(len(selected))
            ]
            observation = np.stack(feature_rows).astype(np.float32)
            command = np.asarray(data[f"{self.action_source}_actions"], dtype=np.float64)
            if command.shape != (len(times), 8):
                raise ValueError(
                    f"{self.action_source}_actions must have shape (T,8), got {command.shape}"
                )
            action = None
            if "action" in data.files:
                action = np.asarray(data["action"], dtype=np.float32)
                if action.shape != (len(selected), 7):
                    raise ValueError(
                        f"precomputed action/frame mismatch in {path}: "
                        f"{action.shape} vs {(len(selected), 7)}"
                    )
            elif self._action_cache is not None:
                action = self._action_cache.action_for(path, len(selected))
            if action is None:
                action_position, action_quaternion = _forward_kinematics(str(model_path)).poses(
                    command[selected, :7]
                )
                action_rotation = (
                    wxyz_to_euler_xyz(action_quaternion)
                    if self.config.rotation_format == "euler"
                    else wxyz_to_rotvec(action_quaternion)
                )
                action = np.concatenate(
                    (
                        action_position,
                        action_rotation,
                        np.where(command[selected, 7] > self.gripper_threshold, 1.0, -1.0)[:, None],
                    ),
                    axis=-1,
                ).astype(np.float32)
            scenario = _scalar(data["scenario_name"])
            seed = int(data["seed"])
        if len(observation) < 1 or len(action) != len(observation):
            raise ValueError(f"empty or misaligned low-dimensional episode: {path}")
        return LowDimEpisode(
            path, scenario, seed, observation,
            _make_pi05_action_target(observation, np.asarray(action, dtype=np.float32)),
        )

    def _load_nero_fullphysics_episode(self, path: Path, data) -> LowDimEpisode:
        """Convert the current NERO recorder schema to low-dimensional frames."""

        import mujoco

        scenario, seed, calibration = _nero_episode_metadata(path)
        state_spec = int(np.asarray(data["state_spec"]).item())
        all_states = np.asarray(data["states"], dtype=np.float64)
        command = np.asarray(data[f"{self.action_source}_actions"], dtype=np.float64)
        time_key = "state_times" if "state_times" in data.files else "times"
        all_times = np.asarray(data[time_key], dtype=np.float64)
        if all_states.ndim != 2 or all_times.shape != (len(all_states),):
            raise ValueError(
                f"invalid NERO FULLPHYSICS alignment in {path}: "
                f"states={all_states.shape} {time_key}={all_times.shape}"
            )
        if len(all_states) == len(command) + 1:
            # Recorder schema with the final post-action state.
            times = all_times[:-1]
            states = all_states[:-1]
        elif len(all_states) == len(command):
            # privileged_expert collector schema: one pre-action state/time per
            # command, with no extra final state.
            times = all_times
            states = all_states
        else:
            raise ValueError(
                f"NERO trajectory state/action alignment is invalid: "
                f"states={len(all_states)} actions={len(command)} in {path}"
            )
        selected = _select_frame_indices(times, 25)
        base_x, tcp_dx, tcp_dy, tcp_dz = calibration
        env, expected_state_size = _nero_state_decoder(
            scenario, base_x, tcp_dx, tcp_dy, tcp_dz
        )
        if states.shape[1] != expected_state_size:
            raise ValueError(
                f"NERO state/model mismatch in {path}: "
                f"states={states.shape[1]} expected={expected_state_size}"
            )
        if self._action_cache is None:
            raise ValueError(
                f"raw NERO trajectory requires a LeRobot action cache: {path}"
            )
        action = self._action_cache.action_for_identity(
            scenario, seed, len(selected)
        )
        if action is None:
            action = self._action_cache.action_for(path, len(selected))
        if action is None:
            raise ValueError(
                f"action cache has no episode matching scenario={scenario} seed={seed}"
            )

        fast_fields = {
            "hand_position",
            "hand_quaternion",
            "cable_positions",
            "target_positions",
        }
        if self.config.include_velocity:
            fast_fields.update({"cable_velocities", "target_velocities"})
        if fast_fields.issubset(data.files):
            hand_position = np.asarray(data["hand_position"], dtype=np.float32)
            hand_quaternion = np.asarray(data["hand_quaternion"], dtype=np.float32)
            cable_positions = np.asarray(data["cable_positions"], dtype=np.float32)
            target_positions = np.asarray(data["target_positions"], dtype=np.float32)
            cable_velocities = (
                np.asarray(data["cable_velocities"], dtype=np.float32)
                if self.config.include_velocity else None
            )
            target_velocities = (
                np.asarray(data["target_velocities"], dtype=np.float32)
                if self.config.include_velocity else None
            )
            arrays = (
                hand_position, hand_quaternion, cable_positions, target_positions,
            )
            if cable_velocities is not None and target_velocities is not None:
                arrays += (cable_velocities, target_velocities)
            if all(array.shape[0] >= len(times) for array in arrays):
                model_path = (path.parent / _scalar(data["model_file"])).resolve()
                qpos_indices = np.asarray(
                    _model_gripper_qpos_addresses(str(model_path)),
                    dtype=np.int64,
                ) + 1
                if states.shape[1] <= int(np.max(qpos_indices)):
                    raise ValueError(
                        f"states do not contain gripper qpos in {path}: {states.shape}"
                    )
                gripper = np.abs(states[:, qpos_indices]).sum(axis=1)
                keypoints = _keypoint_indices(
                    cable_positions.shape[1], self.config.num_keypoints
                )
                observation_rows = [
                    make_lowdim_feature(
                        hand_position[index],
                        hand_quaternion[index],
                        cable_positions[index],
                        target_positions[index],
                        gripper[index],
                        keypoint_indices=keypoints,
                        cable_velocities=(
                            None if cable_velocities is None
                            else cable_velocities[index]
                        ),
                        target_velocity=(
                            None if target_velocities is None
                            else target_velocities[index]
                        ),
                        rotation_format=self.config.rotation_format,
                    )
                    for index in selected
                ]
                observation = np.stack(observation_rows).astype(np.float32)
                return LowDimEpisode(
                    path, scenario, seed, observation,
                    _make_pi05_action_target(observation, np.asarray(action, dtype=np.float32)),
                )

        keypoints = _keypoint_indices(len(env.cable_ids), self.config.num_keypoints)
        observation_rows: list[np.ndarray] = []
        quat = np.empty(4, dtype=np.float64)
        for index in selected:
            mujoco.mj_setState(env.model, env.data, states[index], state_spec)
            mujoco.mj_forward(env.model, env.data)
            rotation = env.data.xmat[env.hand_id].reshape(3, 3)
            mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
            observation_rows.append(
                make_lowdim_feature(
                    env.hand_position,
                    quat,
                    env.data.xpos[env.cable_ids],
                    env.target_position(),
                    env.finger_aperture,
                    keypoint_indices=keypoints,
                    rotation_format=self.config.rotation_format,
                )
            )
        observation = np.stack(observation_rows).astype(np.float32)
        if len(observation) != len(action):
            raise ValueError(
                f"NERO converted frame/action mismatch in {path}: "
                f"observation={len(observation)} action={len(action)}"
            )
        return LowDimEpisode(
            path, scenario, seed, observation,
            _make_pi05_action_target(observation, np.asarray(action, dtype=np.float32)),
        )

    def _set_subset(self, selected: list[int]) -> None:
        if not selected:
            raise ValueError("dataset split contains no episodes")
        if any(index < 0 or index >= self._total_episode_count for index in selected):
            raise IndexError("episode split index is out of range")
        self.episode_indices = selected
        self._sample_offsets = np.cumsum(
            [0] + [len(self._all_episodes[index].observation) for index in selected],
            dtype=np.int64,
        )

    def subset(self, episode_indices: Iterable[int]) -> "LowDimEpisodeDataset":
        dataset = self.__class__.__new__(self.__class__)
        dataset.config = self.config
        dataset.action_source = self.action_source
        dataset.gripper_threshold = self.gripper_threshold
        dataset._action_cache = self._action_cache
        dataset._all_episodes = self._all_episodes
        dataset._total_episode_count = self._total_episode_count
        dataset._set_subset(list(episode_indices))
        dataset.obs_mean = self.obs_mean
        dataset.obs_std = self.obs_std
        dataset._obs_pose_norm_low = self._obs_pose_norm_low
        dataset._obs_pose_norm_high = self._obs_pose_norm_high
        dataset._action_norm_low = self._action_norm_low
        dataset._action_norm_high = self._action_norm_high
        return dataset

    @property
    def total_episode_count(self) -> int:
        return self._total_episode_count

    @property
    def episode_count(self) -> int:
        return len(self.episode_indices)

    @property
    def state_dim(self) -> int:
        return int(self._all_episodes[0].observation.shape[-1])

    @property
    def action_low(self) -> np.ndarray:
        if self._action_norm_low is not None:
            return self._action_norm_low
        bounds = ACTION_LOW_ROTVEC if self.config.rotation_format == "rotvec" else ACTION_LOW
        return np.asarray(bounds, dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        if self._action_norm_high is not None:
            return self._action_norm_high
        bounds = ACTION_HIGH_ROTVEC if self.config.rotation_format == "rotvec" else ACTION_HIGH
        return np.asarray(bounds, dtype=np.float32)

    def fit_action_quantiles(
        self, episode_indices: Iterable[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = self.episode_indices if episode_indices is None else list(episode_indices)
        # The stored episode action is a per-frame delta (absolute action minus
        # that frame's observation). Pi0.5 supervises every action in a chunk
        # relative to the chunk's current/first observation, so fit q01/q99
        # bounds on those actual chunk targets.
        values_by_episode = []
        for index in selected:
            episode = self._all_episodes[index]
            frame_indices = np.arange(0, len(episode.action), 4, dtype=np.int64)
            offsets = np.arange(self.config.prediction_horizon, dtype=np.int64)
            future_indices = np.minimum(
                frame_indices[:, None] + offsets[None, :], len(episode.action) - 1
            )
            chunk = episode.action[future_indices].copy()
            chunk[:, :, :6] += episode.observation[future_indices, :6]
            chunk[:, :, :6] -= episode.observation[frame_indices, None, :6]
            values_by_episode.append(chunk.reshape(-1, 7))
        values = np.concatenate(values_by_episode, axis=0).astype(np.float64)
        lower = np.quantile(values, 0.01, axis=0).astype(np.float32)
        upper = np.quantile(values, 0.99, axis=0).astype(np.float32)
        upper = np.maximum(upper, lower + 1.0e-6)
        return lower, upper

    def fit_observation_pose_quantiles(
        self, episode_indices: Iterable[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = self.episode_indices if episode_indices is None else list(episode_indices)
        values = np.concatenate(
            [self._all_episodes[index].observation[:, :6] for index in selected], axis=0
        ).astype(np.float64)
        lower = np.quantile(values, 0.01, axis=0).astype(np.float32)
        upper = np.quantile(values, 0.99, axis=0).astype(np.float32)
        upper = np.maximum(upper, lower + 1.0e-6)
        return lower, upper

    def set_action_norm_bounds(self, lower: np.ndarray, upper: np.ndarray) -> None:
        lower = np.asarray(lower, dtype=np.float32)
        upper = np.asarray(upper, dtype=np.float32)
        if lower.shape != (7,) or upper.shape != (7,):
            raise ValueError("action normalization bounds must have shape (7,)")
        self._action_norm_low = lower
        self._action_norm_high = np.maximum(upper, lower + 1.0e-6)

    def set_observation_pose_norm_bounds(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> None:
        lower = np.asarray(lower, dtype=np.float32)
        upper = np.asarray(upper, dtype=np.float32)
        if lower.shape != (6,) or upper.shape != (6,):
            raise ValueError("observation pose normalization bounds must have shape (6,)")
        self._obs_pose_norm_low = lower
        self._obs_pose_norm_high = np.maximum(upper, lower + 1.0e-6)

    def fit_observation_stats(
        self, episode_indices: Iterable[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = self.episode_indices if episode_indices is None else list(episode_indices)
        total = 0
        sums = np.zeros(self.state_dim, dtype=np.float64)
        squares = np.zeros(self.state_dim, dtype=np.float64)
        for index in selected:
            values = self._all_episodes[index].observation.astype(np.float64)
            total += len(values)
            sums += values.sum(axis=0)
            squares += np.square(values).sum(axis=0)
        if total < 1:
            raise ValueError("cannot fit observation stats on an empty split")
        mean = sums / total
        variance = np.maximum(squares / total - np.square(mean), 1.0e-8)
        std = np.maximum(np.sqrt(variance), 1.0e-3)
        return mean.astype(np.float32), std.astype(np.float32)

    def set_observation_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.shape != (self.state_dim,) or std.shape != (self.state_dim,):
            raise ValueError("observation statistics have the wrong shape")
        self.obs_mean = mean
        self.obs_std = np.maximum(std, 1.0e-3)

    def __len__(self) -> int:
        return int(self._sample_offsets[-1])

    @staticmethod
    def _history_indices(index: int, horizon: int) -> list[int]:
        return [max(0, index - horizon + 1 + offset) for offset in range(horizon)]

    @staticmethod
    def _future_indices(index: int, count: int, length: int) -> list[int]:
        return [min(length - 1, index + offset) for offset in range(count)]

    def _chunk_action_target(
        self, episode: LowDimEpisode, frame_index: int
    ) -> np.ndarray:
        """Return a Pi0.5-style action chunk relative to the current state."""

        future = self._future_indices(
            frame_index, self.config.prediction_horizon, len(episode.action)
        )
        action = episode.action[future].copy()
        # Reconstruct each expert absolute pose, then reference the complete
        # chunk to the state at frame_index. Gripper remains absolute.
        action[:, :6] += episode.observation[future, :6]
        action[:, :6] -= episode.observation[frame_index, None, :6]
        return action

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        if self.obs_mean is None or self.obs_std is None:
            raise RuntimeError("call set_observation_stats before reading samples")
        if item < 0:
            item += len(self)
        episode_position = int(np.searchsorted(self._sample_offsets, item, side="right") - 1)
        frame_index = int(item - self._sample_offsets[episode_position])
        episode = self._all_episodes[self.episode_indices[episode_position]]
        history = self._history_indices(frame_index, self.config.observation_horizon)
        observation = (episode.observation[history] - self.obs_mean) / self.obs_std
        if self._obs_pose_norm_low is not None and self._obs_pose_norm_high is not None:
            observation[:, :6] = _pi05_quantile_normalize(
                episode.observation[history, :6],
                self._obs_pose_norm_low,
                self._obs_pose_norm_high,
            )
        action = _pi05_quantile_normalize(
            self._chunk_action_target(episode, frame_index),
            self.action_low,
            self.action_high,
        )
        return {
            "obs": torch.from_numpy(np.asarray(observation, dtype=np.float32)),
            "action": torch.from_numpy(np.asarray(action, dtype=np.float32)),
        }


class LowDimDiffusionPolicy(nn.Module):
    """Conditional 1-D action diffusion model for low-dimensional observations."""

    action_dim = 7

    def __init__(self, config: LowDimPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        global_condition_dim = config.observation_horizon * self.state_dim
        self.denoiser = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_condition_dim=global_condition_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=tuple(config.unet_down_dims),
            kernel_size=config.unet_kernel_size,
            groups=config.unet_groups,
        )
        self.schedule = DDPMSchedule(config)

    def forward_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3 or action.ndim != 3:
            raise ValueError("expected obs (B,H,D) and action (B,T,7)")
        timestep = torch.randint(
            0, self.config.diffusion_steps, (action.shape[0],), device=action.device
        )
        noise = torch.randn_like(action)
        noisy_action = self.schedule.add_noise(action, noise, timestep)
        prediction = self.denoiser(
            noisy_action, timestep, obs.reshape(obs.shape[0], -1)
        )
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(
        self, obs: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return self.schedule.sample(
            self.denoiser,
            obs.reshape(obs.shape[0], -1),
            (obs.shape[0], self.config.prediction_horizon, self.action_dim),
            self.config.inference_steps,
            generator,
        )


class LowDimPolicyRunner:
    """Online receding-horizon runner sharing the DynamicVLA task-space adapter."""

    def __init__(
        self,
        env,
        checkpoint: Path,
        *,
        device: str = "cpu",
        deterministic: bool = True,
        inference_steps: int | None = None,
    ) -> None:
        from ..dynamicvla.adapter import (
            DynamicVLAAdapterConfig,
            DynamicVLATaskSpaceAdapter,
        )

        self.env = env
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        try:
            payload = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            payload = torch.load(self.checkpoint_path, map_location=self.device)
        if not isinstance(payload, dict) or payload.get("format") != "panda_cable_lowdim_diffusion_policy_v1":
            raise ValueError("invalid low-dimensional Diffusion Policy checkpoint")
        values = dict(payload["config"])
        values["unet_down_dims"] = tuple(values["unet_down_dims"])
        self.config = LowDimPolicyConfig(**values)
        if inference_steps is not None:
            if isinstance(inference_steps, bool) or inference_steps <= 0:
                raise ValueError("inference_steps must be a positive integer")
            if inference_steps > self.config.diffusion_steps:
                raise ValueError(
                    "inference_steps cannot exceed the checkpoint diffusion_steps"
                )
            self.config = replace(self.config, inference_steps=int(inference_steps))
        self.model = LowDimDiffusionPolicy(self.config).to(self.device)
        self.model.load_state_dict(payload["ema_model"] or payload["model"])
        self.model.eval()
        self.obs_mean = np.asarray(payload["obs_mean"], dtype=np.float32)
        self.obs_std = np.asarray(payload["obs_std"], dtype=np.float32)
        self.obs_pose_norm_low = (
            None
            if payload.get("obs_pose_norm_low") is None
            else np.asarray(payload["obs_pose_norm_low"], dtype=np.float32)
        )
        self.obs_pose_norm_high = (
            None
            if payload.get("obs_pose_norm_high") is None
            else np.asarray(payload["obs_pose_norm_high"], dtype=np.float32)
        )
        self.action_low = np.asarray(
            payload.get("action_norm_low", payload.get("action_low", ACTION_LOW)),
            dtype=np.float32,
        )
        self.action_high = np.asarray(
            payload.get("action_norm_high", payload.get("action_high", ACTION_HIGH)),
            dtype=np.float32,
        )
        self.action_target = str(payload.get("action_target", "absolute_xyz_euler_gripper"))
        self.keypoints = (
            None
            if self.config.observation_dim == 6
            else _keypoint_indices(len(env.cable_ids), self.config.num_keypoints)
        )
        adapter_config = (
            DynamicVLAAdapterConfig(nero_pose_ik_enabled=True)
            if env.robot == "nero"
            else None
        )
        self.adapter = DynamicVLATaskSpaceAdapter(env, adapter_config)
        self.deterministic = bool(deterministic)
        self._history: deque[np.ndarray] = deque(maxlen=self.config.observation_horizon)
        self._action_queue: deque[np.ndarray] = deque()
        self._generator: torch.Generator | None = None
        self._last_raw_observation: np.ndarray | None = None
        self._inference_count = 0
        self._last_model_action = np.full(8, np.nan, dtype=np.float32)
        self.result = "running"
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.adapter.reset()
        self._history.clear()
        self._action_queue.clear()
        self._inference_count = 0
        self._last_model_action[:] = np.nan
        self._last_raw_observation = None
        self.result = "running"
        if seed is None or not self.deterministic:
            self._generator = None
        else:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(int(seed))

    def _observation(self) -> np.ndarray:
        rotation = self.env.data.xmat[self.env.hand_id].reshape(3, 3)
        quaternion = rotation_to_quat(rotation)
        if self.config.observation_dim == 6:
            hand_rotation = (
                wxyz_to_euler_xyz(quaternion)
                if self.config.rotation_format == "euler"
                else wxyz_to_rotvec(quaternion)
            )
            return np.concatenate(
                (np.asarray(self.env.hand_position, dtype=np.float32), hand_rotation),
                axis=-1,
            ).astype(np.float32)
        cable_positions = self.env.data.xpos[self.env.cable_ids].copy()
        target_position = self.env.target_position()
        cable_velocity = None
        target_velocity = None
        if self.config.include_velocity:
            cable_velocity = np.stack(
                [
                    self.env.body_linear_velocity(body_id)
                    for body_id in self.env.cable_ids
                ]
            )
            target_velocity = self.env.target_velocity()
        return make_lowdim_feature(
            self.env.hand_position,
            quaternion,
            cable_positions,
            target_position,
            float(np.abs(self.env.data.qpos[self.env.finger_qpos_adr]).sum()),
            keypoint_indices=self.keypoints,
            cable_velocities=cable_velocity,
            target_velocity=target_velocity,
            rotation_format=self.config.rotation_format,
        )

    def _sample_action_chunk(self) -> None:
        observation = np.asarray(self._history, dtype=np.float32)
        value = torch.from_numpy(observation)[None].to(self.device)
        normalized = self.model.sample(value, generator=self._generator)[0].cpu().numpy()
        model_chunk = _pi05_quantile_denormalize(
            normalized, self.action_low, self.action_high
        )
        if self.action_target in {
            "pi05_delta_xyz_euler_gripper",
            "pi05_chunk_delta_xyz_euler_gripper",
        }:
            if self._last_raw_observation is None:
                raise RuntimeError("cannot decode a delta action without a current observation")
            model_chunk[:, :6] += self._last_raw_observation[None, :6]
        task_space_chunk = np.concatenate(
            (
                model_chunk[:, :3],
                (
                    euler_xyz_to_wxyz(model_chunk[:, 3:6])
                    if self.config.rotation_format == "euler"
                    else rotvec_to_wxyz(model_chunk[:, 3:6])
                ),
                model_chunk[:, 6:7],
            ),
            axis=-1,
        )
        self._action_queue.extend(
            np.asarray(task_space_chunk[: self.config.action_horizon], dtype=np.float32)
        )
        self._inference_count += 1

    def action(self) -> np.ndarray:
        raw_observation = self._observation()
        self._last_raw_observation = raw_observation.copy()
        observation = (raw_observation - self.obs_mean) / self.obs_std
        if self.obs_pose_norm_low is not None and self.obs_pose_norm_high is not None:
            observation[:6] = _pi05_quantile_normalize(
                raw_observation[:6], self.obs_pose_norm_low, self.obs_pose_norm_high
            )
        self._history.append(observation.astype(np.float32))
        while len(self._history) < self.config.observation_horizon:
            self._history.appendleft(self._history[0].copy())
        if not self._action_queue:
            self._sample_action_chunk()
        task_space_action = self._action_queue.popleft()
        self._last_model_action[:] = task_space_action
        self.adapter.set_model_action(task_space_action)
        return self.adapter.action()

    def policy_info(self) -> dict:
        diagnostics = self.adapter.diagnostics()
        return {
            "diffusion_inference_count": self._inference_count,
            "diffusion_inference_steps": self.config.inference_steps,
            "diffusion_action_queue": len(self._action_queue),
            "diffusion_model_action": self._last_model_action.copy(),
            "diffusion_position_clipped": bool(diagnostics["position_clipped"]),
            "diffusion_quaternion_repaired": bool(diagnostics["quaternion_repaired"]),
            "action_target": self.action_target,
            "action_normalization": "pi05_q01_q99_clip" if self.action_target in {
                "pi05_delta_xyz_euler_gripper",
                "pi05_chunk_delta_xyz_euler_gripper",
            } else "legacy_fixed_bounds",
            "observation_pose_normalization": "pi05_q01_q99_clip" if self.obs_pose_norm_low is not None else "legacy_mean_std",
        }


__all__ = [
    "LowDimDiffusionPolicy", "LowDimEpisodeDataset", "LowDimPolicyConfig",
    "LowDimPolicyRunner", "make_lowdim_feature",
]
