"""Convert privileged-expert trajectories to DynamicVLA's LeRobot v2.1 format.

The collector stores joint-space commands because that is the native MuJoCo
environment interface.  DynamicVLA is trained in task space, so this converter
uses the saved ``scenario.mjb`` model to run forward kinematics for every
command and produces ``xyz + Euler(xyz) + gripper(-1/+1)`` labels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


CAMERA_KEYS = (
    "observation.images.opst_cam",
    "observation.images.wrist_cam",
)
GRASP_CENTER_LOCAL = np.array([0.0, 0.0, 0.1029], dtype=np.float64)
TASK_METADATA = {"task": "pick", "objects": ["blue cable"]}


@dataclass(frozen=True)
class EpisodeSource:
    trajectory: Path
    scenario: str
    seed: int


def _scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _wxyz_to_euler_xyz(quaternions: np.ndarray) -> np.ndarray:
    """Match DynamicVLA's Euler convention, including angle wrapping."""

    quat = np.asarray(quaternions, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"quaternion array must end in 4 values, got {quat.shape}")
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("zero-length quaternion in trajectory")
    quat = quat / norms
    euler = Rotation.from_quat(quat, scalar_first=True).as_euler(
        "xyz", degrees=False
    )
    euler[..., [0, 2]] = np.mod(euler[..., [0, 2]], 2.0 * np.pi)
    return euler.astype(np.float32)


def _select_frame_indices(times: np.ndarray, target_fps: int) -> np.ndarray:
    """Select source frames nearest a uniform target-rate time grid."""

    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times must be a non-empty 1-D array")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("trajectory times must be strictly increasing")

    period = 1.0 / float(target_fps)
    targets = times[0] + np.arange(
        int(np.floor((times[-1] - times[0]) / period + 1e-8)) + 1,
        dtype=np.float64,
    ) * period
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(times[left] - targets) <= np.abs(times[right] - targets)
    selected = np.where(choose_left, left, right)
    selected = np.unique(selected.astype(np.int64))

    if len(times) > 1:
        source_period = float(np.median(np.diff(times)))
        if source_period > period * 1.05:
            raise ValueError(
                f"source rate ({1.0 / source_period:.2f} Hz) is below "
                f"target rate ({target_fps} Hz)"
            )
    return selected


def _discover_episodes(
    inputs: Iterable[Path],
    scenarios: set[str] | None,
    allow_incomplete_run: bool = False,
) -> list[EpisodeSource]:
    paths: set[Path] = set()
    for input_path in inputs:
        resolved = input_path.expanduser().resolve()
        if resolved.is_file():
            if resolved.suffix.lower() != ".npz":
                raise ValueError(f"input file is not an .npz trajectory: {resolved}")
            paths.add(resolved)
        elif resolved.is_dir():
            direct_episodes = list(resolved.glob("episode_*.npz"))
            recursive_episodes = list(resolved.rglob("episode_*.npz"))
            if recursive_episodes and not direct_episodes and not allow_incomplete_run:
                manifest_path = resolved / "manifest.json"
                if not manifest_path.exists():
                    raise RuntimeError(
                        f"run directory has no manifest and may still be collecting: "
                        f"{resolved}; wait for completion, pass a completed scenario "
                        "directory, or use --allow-incomplete-run explicitly"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                incomplete = [
                    name
                    for name, item in manifest.get("scenario_manifests", {}).items()
                    if not item.get("complete", False)
                ]
                if incomplete:
                    raise RuntimeError(
                        f"run is incomplete for scenarios {incomplete}: {resolved}; "
                        "use --allow-incomplete-run explicitly to convert it"
                    )
            paths.update(path.resolve() for path in recursive_episodes)
        else:
            raise FileNotFoundError(resolved)

    episodes: list[EpisodeSource] = []
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as data:
            required = {
                "schema_version",
                "times",
                "requested_actions",
                "applied_actions",
                "hand_position",
                "hand_quaternion",
                "scenario_name",
                "seed",
                "opst_video_file",
                "wrist_video_file",
                "model_file",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(
                    f"{path} is not a schema-v2 dual-camera trajectory; "
                    f"missing {missing}"
                )
            schema_version = int(data["schema_version"])
            if schema_version < 2:
                raise ValueError(f"{path} has unsupported schema {schema_version}")
            scenario = _scalar_string(data["scenario_name"])
            seed = int(data["seed"])
        if scenarios is None or scenario in scenarios:
            episodes.append(EpisodeSource(path, scenario, seed))
    if not episodes:
        raise ValueError("no compatible episodes found")
    return episodes


def _deduplicate(
    episodes: list[EpisodeSource], duplicate_policy: str
) -> list[EpisodeSource]:
    unique: list[EpisodeSource] = []
    seen: dict[tuple[str, int], Path] = {}
    for episode in episodes:
        key = (episode.scenario, episode.seed)
        previous = seen.get(key)
        if previous is None or duplicate_policy == "keep":
            seen[key] = episode.trajectory
            unique.append(episode)
            continue
        if duplicate_policy == "skip":
            print(
                f"skip_duplicate scenario={episode.scenario} seed={episode.seed} "
                f"path={episode.trajectory}",
                flush=True,
            )
            continue
        raise ValueError(
            f"duplicate scenario/seed {key}: {previous} and {episode.trajectory}; "
            "pass --duplicate-policy skip or keep explicitly"
        )
    return unique


def _read_selected_rgb(path: Path, indices: np.ndarray) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    wanted = {int(index): output for output, index in enumerate(indices)}
    frames: list[np.ndarray | None] = [None] * len(indices)
    source_index = 0
    last_index = int(indices[-1])
    try:
        while source_index <= last_index:
            ok, bgr = capture.read()
            if not ok:
                break
            output_index = wanted.get(source_index)
            if output_index is not None:
                frames[output_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            source_index += 1
    finally:
        capture.release()
    missing = [int(indices[i]) for i, frame in enumerate(frames) if frame is None]
    if missing:
        raise RuntimeError(f"video {path} is missing selected frames {missing[:8]}")
    return [frame for frame in frames if frame is not None]


class JointTargetForwardKinematics:
    """Compute grasp-center poses from saved Panda or NERO joint targets."""

    def __init__(self, model_path: Path) -> None:
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError(
                "MuJoCo is required for joint-target forward kinematics; "
                "install it in the DynamicVLA environment with "
                "`python -m pip install mujoco`"
            ) from error
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        joint_ids = []
        for index in range(1, 8):
            candidates = (f"joint{index}", f"panda_joint{index}")
            joint_id = next(
                (
                    candidate_id
                    for name in candidates
                    if (
                        candidate_id := mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                        )
                    )
                    >= 0
                ),
                -1,
            )
            if joint_id < 0:
                raise ValueError(
                    f"Panda arm joint {index} not found in {model_path}; tried {candidates}"
                )
            joint_ids.append(joint_id)
        self.arm_qpos_addresses = self.model.jnt_qposadr[np.asarray(joint_ids)]
        panda_hand_id = next(
            (
                body_id
                for name in ("hand", "panda_hand")
                if (
                    body_id := mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, name
                    )
                )
                >= 0
            ),
            -1,
        )
        if panda_hand_id >= 0:
            self.robot = "panda"
            self.hand_id = panda_hand_id
            self.grasp_center_local = GRASP_CENTER_LOCAL.copy()
        else:
            self.robot = "nero"
            self.hand_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "link7"
            )
            if self.hand_id < 0:
                raise ValueError(
                    f"neither Panda hand nor NERO link7 was found in {model_path}"
                )
            self.grasp_center_local = np.array(
                [0.1733, 0.0, -0.0235], dtype=np.float64
            )
        self.gripper_actuator_id = next(
            (
                actuator_id
                for name in ("actuator8", "gripper")
                if (
                    actuator_id := mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                    )
                )
                >= 0
            ),
            7,
        )

    @property
    def gripper_ctrl_range(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[self.gripper_actuator_id].copy()

    def poses(self, joint_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joint_targets = np.asarray(joint_targets, dtype=np.float64)
        if joint_targets.ndim != 2 or joint_targets.shape[1] != 7:
            raise ValueError(f"expected joint targets (T, 7), got {joint_targets.shape}")
        positions = np.empty((len(joint_targets), 3), dtype=np.float32)
        quaternions = np.empty((len(joint_targets), 4), dtype=np.float32)
        quat = np.empty(4, dtype=np.float64)
        mujoco = self.mujoco
        for index, target in enumerate(joint_targets):
            self.data.qpos[:] = self.model.qpos0
            self.data.qpos[self.arm_qpos_addresses] = target
            mujoco.mj_forward(self.model, self.data)
            rotation = self.data.xmat[self.hand_id].reshape(3, 3)
            positions[index] = (
                self.data.xpos[self.hand_id]
                + rotation @ self.grasp_center_local
            )
            mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
            if index > 0 and np.dot(quat, quaternions[index - 1]) < 0.0:
                quat *= -1.0
            quaternions[index] = quat
        return positions, quaternions


def _episode_arrays(
    source: EpisodeSource,
    target_fps: int,
    joint_target_source: str,
    gripper_threshold: float | None,
) -> tuple[dict[str, np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    with np.load(source.trajectory, allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=np.float64)
        selected = _select_frame_indices(times, target_fps)
        state_position = np.asarray(data["hand_position"], dtype=np.float32)[selected]
        state_rotation = _wxyz_to_euler_xyz(data["hand_quaternion"])[selected]
        command = np.asarray(data[f"{joint_target_source}_actions"], dtype=np.float64)
        if len(command) != len(times) or command.shape[1] != 8:
            raise ValueError(
                f"action/time alignment is invalid in {source.trajectory}: "
                f"{command.shape} vs {times.shape}"
            )
        model_path = source.trajectory.parent / _scalar_string(data["model_file"])
        opst_path = source.trajectory.parent / _scalar_string(data["opst_video_file"])
        wrist_path = source.trajectory.parent / _scalar_string(data["wrist_video_file"])

    fk = JointTargetForwardKinematics(model_path)
    action_position, action_quaternion = fk.poses(command[selected, :7])
    action_rotation = _wxyz_to_euler_xyz(action_quaternion)
    threshold = (
        float(gripper_threshold)
        if gripper_threshold is not None
        else float(np.mean(fk.gripper_ctrl_range))
    )
    gripper = np.where(
        command[selected, fk.gripper_actuator_id] > threshold, 1.0, -1.0
    ).astype(np.float32)[:, None]
    arrays = {
        "observation.state": np.concatenate(
            [state_position, state_rotation], axis=-1
        ).astype(np.float32),
        "action": np.concatenate(
            [action_position, action_rotation, gripper], axis=-1
        ).astype(np.float32),
    }
    opst = _read_selected_rgb(opst_path, selected)
    wrist = _read_selected_rgb(wrist_path, selected)
    if not (len(opst) == len(wrist) == len(arrays["action"])):
        raise RuntimeError(f"frame alignment failed for {source.trajectory}")
    metadata = {
        "source": str(source.trajectory),
        "scenario": source.scenario,
        "seed": source.seed,
        "source_frames": int(len(times)),
        "converted_frames": int(len(selected)),
        "source_fps": float(1.0 / np.median(np.diff(times))) if len(times) > 1 else None,
        "robot": fk.robot,
        "gripper_threshold": threshold,
    }
    return arrays, opst, wrist, metadata


def _features(height: int, width: int) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz", "gripper"
            ],
        },
        CAMERA_KEYS[0]: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        CAMERA_KEYS[1]: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert cable expert trajectories to DynamicVLA LeRobot v2.1"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="run/scenario dirs or .npz files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/panda-cable-dynamicvla")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--target-fps", type=int, default=25)
    parser.add_argument(
        "--joint-target-source", choices=("requested", "applied"), default="requested"
    )
    parser.add_argument(
        "--gripper-threshold", type=float, default=None,
        help="override the model-derived gripper threshold",
    )
    parser.add_argument(
        "--duplicate-policy", choices=("error", "skip", "keep"), default="error"
    )
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-incomplete-run",
        action="store_true",
        help="allow a run root without a completed manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = _discover_episodes(
        args.inputs,
        set(args.scenarios) if args.scenarios else None,
        allow_incomplete_run=args.allow_incomplete_run,
    )
    episodes = _deduplicate(episodes, args.duplicate_policy)
    if args.limit_episodes is not None:
        if args.limit_episodes <= 0:
            raise ValueError("limit-episodes must be positive")
        episodes = episodes[: args.limit_episodes]

    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output}; use --overwrite explicitly")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    first_arrays, first_opst, first_wrist, first_metadata = _episode_arrays(
        episodes[0], args.target_fps, args.joint_target_source, args.gripper_threshold
    )
    height, width = first_opst[0].shape[:2]
    if first_wrist[0].shape[:2] != (height, width):
        raise ValueError("opposite and wrist camera dimensions differ")
    robot_type = "agilex_nero" if first_metadata["robot"] == "nero" else "franka"
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.target_fps,
        root=output,
        robot_type=robot_type,
        features=_features(height, width),
        use_videos=True,
        image_writer_threads=4,
        batch_encoding_size=1,
    )
    task = json.dumps(TASK_METADATA, ensure_ascii=False, sort_keys=True)
    conversion_rows: list[dict] = []

    for episode_index, source in enumerate(episodes):
        if episode_index == 0:
            arrays, opst, wrist, metadata = (
                first_arrays, first_opst, first_wrist, first_metadata
            )
        else:
            arrays, opst, wrist, metadata = _episode_arrays(
                source,
                args.target_fps,
                args.joint_target_source,
                args.gripper_threshold,
            )
        for frame_index in range(len(arrays["action"])):
            dataset.add_frame(
                {
                    "observation.state": arrays["observation.state"][frame_index],
                    "action": arrays["action"][frame_index],
                    CAMERA_KEYS[0]: opst[frame_index],
                    CAMERA_KEYS[1]: wrist[frame_index],
                },
                task=task,
            )
        dataset.save_episode()
        metadata["episode_index"] = episode_index
        conversion_rows.append(metadata)
        print(
            f"converted={episode_index + 1}/{len(episodes)} "
            f"scenario={source.scenario} seed={source.seed} "
            f"frames={metadata['converted_frames']}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "source_format": "privileged_expert_schema_v2",
        "dataset_format": "lerobot_v2.1",
        "repo_id": args.repo_id,
        "target_fps": args.target_fps,
        "joint_target_source": args.joint_target_source,
        "action_format": "absolute_xyz_euler_xyz_gripper_minus1_plus1",
        "state_format": "absolute_xyz_euler_xyz",
        "task_metadata": TASK_METADATA,
        "episodes": conversion_rows,
    }
    (output / "cable_conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"complete episodes={len(episodes)} output={output}", flush=True)


if __name__ == "__main__":
    main()
