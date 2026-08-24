"""验证实验场景产生的实际整体运动与形状变化。

该脚本不运行抓取策略，而是让机器人保持 ready 状态，仅采样线缆运动。输出既包含
环境施加的分解力场，也包含质心路径、刚体旋转和刚体配准后的形状残差。后者用于
确认 ``rigid`` / ``shape`` / ``combined`` 不只是配置名称不同。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import numpy as np
import mujoco

from ..env.environment import (
    CableGraspEnv,
    EnvConfig,
    PANDA_XML_PATH,
    XML_PATH,
    resolve_menagerie_panda_dir,
)
from ..scenarios.registry import (
    SCENARIO_SUITE_NAMES,
    ScenarioConfig,
    get_scenario,
    list_suite_scenarios,
)
from ..paths import output_path
from .defaults import DEFAULT_EVALUATION_SEED


ROOT = Path(__file__).resolve().parent
DIAGNOSTIC_OUTPUT_ROOT = output_path("diagnostics", "motion")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def env_config_for_scenario(
    scenario: ScenarioConfig,
    *,
    seed: int,
    episode_seconds: float,
) -> EnvConfig:
    """把方法无关的场景协议转换成底层环境配置。"""

    return EnvConfig(
        seed=seed,
        episode_seconds=episode_seconds,
        scenario_name=scenario.name,
        scenario_id=scenario.scenario_id,
        scenario_split=scenario.split.value,
        **scenario.to_env_overrides(),
    )


def _proper_rotation(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """返回使 reference 最小二乘对齐 current 的二维纯旋转。"""

    covariance = reference.T @ current
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def _kinematics(
    reference: np.ndarray,
    current: np.ndarray,
    mass: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """返回质心、平面刚体转角、配准后形状 RMS（单位 m）。"""

    reference_xy = reference[:, :2]
    current_xy = current[:, :2]
    reference_com = np.average(reference_xy, axis=0, weights=mass)
    current_com = np.average(current_xy, axis=0, weights=mass)
    reference_centered = reference_xy - reference_com
    current_centered = current_xy - current_com
    weighted_reference = reference_centered * np.sqrt(mass[:, None])
    weighted_current = current_centered * np.sqrt(mass[:, None])
    rotation = _proper_rotation(weighted_reference, weighted_current)
    aligned_reference = reference_centered @ rotation
    residual = current_centered - aligned_reference
    shape_rms = math.sqrt(
        float(np.average(np.sum(residual * residual, axis=1), weights=mass))
    )
    angle = math.atan2(float(rotation[0, 1]), float(rotation[0, 0]))
    return current_com, angle, shape_rms


@dataclass
class MotionTracker:
    reference: np.ndarray
    mass: np.ndarray

    def __post_init__(self) -> None:
        self.reference = np.asarray(self.reference, dtype=float).copy()
        self.mass = np.asarray(self.mass, dtype=float).copy()
        self.reference_com = np.average(
            self.reference[:, :2], axis=0, weights=self.mass
        )
        self.previous_com = self.reference_com.copy()
        self.previous_time = 0.0
        self.com_displacements: list[float] = []
        self.com_speeds: list[float] = []
        self.rotation_angles: list[float] = []
        self.shape_residuals: list[float] = []
        self.shape_force_rms: list[float] = []
        self.translation_force_rms: list[float] = []
        self.rotation_force_rms: list[float] = []
        self.rigid_shape_hold_force_rms: list[float] = []
        self.com_path_length = 0.0

    def record(self, time: float, positions: np.ndarray, info: dict[str, Any]) -> None:
        com, angle, shape_rms = _kinematics(
            self.reference, np.asarray(positions), self.mass
        )
        dt = float(time) - self.previous_time
        step_distance = float(np.linalg.norm(com - self.previous_com))
        self.com_path_length += step_distance
        self.com_displacements.append(float(np.linalg.norm(com - self.reference_com)))
        if dt > 0.0:
            self.com_speeds.append(step_distance / dt)
        self.rotation_angles.append(abs(angle))
        self.shape_residuals.append(shape_rms)
        self.shape_force_rms.append(float(info["shape_acceleration_rms"]))
        self.translation_force_rms.append(
            float(info["rigid_translation_acceleration_rms"])
        )
        self.rotation_force_rms.append(float(info["rigid_rotation_acceleration_rms"]))
        self.rigid_shape_hold_force_rms.append(
            float(info.get("rigid_shape_hold_acceleration_rms", 0.0))
        )
        self.previous_com = com
        self.previous_time = float(time)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _rms(values: list[float]) -> float:
        return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0

    @staticmethod
    def _max(values: list[float]) -> float:
        return float(np.max(values)) if values else 0.0

    def summary(self) -> dict[str, float | int]:
        return {
            "samples": len(self.com_displacements),
            "com_path_length_m": self.com_path_length,
            "com_displacement_rms_m": self._rms(self.com_displacements),
            "com_displacement_max_m": self._max(self.com_displacements),
            "com_speed_rms_m_s": self._rms(self.com_speeds),
            "rigid_rotation_rms_rad": self._rms(self.rotation_angles),
            "rigid_rotation_max_rad": self._max(self.rotation_angles),
            "shape_change_rms_m": self._rms(self.shape_residuals),
            "shape_change_max_m": self._max(self.shape_residuals),
            "shape_acceleration_rms_m_s2": self._rms(self.shape_force_rms),
            "rigid_translation_acceleration_rms_m_s2": self._rms(
                self.translation_force_rms
            ),
            "rigid_rotation_acceleration_rms_m_s2": self._rms(
                self.rotation_force_rms
            ),
            "rigid_shape_hold_acceleration_rms_m_s2": self._rms(
                self.rigid_shape_hold_force_rms
            ),
        }


def diagnose_scenario(
    scenario: ScenarioConfig,
    *,
    seed: int,
    seconds: float,
    sample_hz: float,
) -> dict[str, Any]:
    env = CableGraspEnv(env_config_for_scenario(
        scenario,
        seed=seed,
        episode_seconds=seconds,
    ))
    try:
        observation, initial_info = env.reset(seed=seed)
        tracker = MotionTracker(observation["cable_positions"], env.cable_mass)
        control_hz = 1.0 / (env.model.opt.timestep * env.config.frame_skip)
        sample_every = max(1, int(round(control_hz / sample_hz)))
        steps = int(math.ceil(seconds * control_hz))
        for step in range(steps):
            observation, _, _, truncated, info = env.step(env.ready_ctrl)
            if step % sample_every == 0 or truncated:
                tracker.record(
                    float(observation["time"]), observation["cable_positions"], info
                )
            if truncated:
                break
        return {
            "scenario_name": scenario.name,
            "scenario_id": scenario.scenario_id,
            "split": scenario.split.value,
            "motion_type": scenario.motion_type.value,
            "regularity": scenario.regularity.value,
            "amplitude_level": scenario.amplitude_level.value,
            "frequency_level": scenario.frequency_level.value,
            "seed": seed,
            "motion_profile_hash": initial_info["motion_profile_hash"],
            **tracker.summary(),
        }
    finally:
        del env


def validate_force_decomposition(rows: list[dict[str, Any]]) -> list[str]:
    """检查四类场景的设计分量是否精确开关。"""

    errors: list[str] = []
    tolerance = 1e-10
    for row in rows:
        motion_type = str(row["motion_type"])
        shape = float(row["shape_acceleration_rms_m_s2"])
        rigid = math.hypot(
            float(row["rigid_translation_acceleration_rms_m_s2"]),
            float(row["rigid_rotation_acceleration_rms_m_s2"]),
        )
        rigid = math.hypot(
            rigid,
            float(row.get("rigid_shape_hold_acceleration_rms_m_s2", 0.0)),
        )
        expects_shape = motion_type in {"shape", "combined"}
        expects_rigid = motion_type in {"rigid", "combined"}
        if expects_shape != (shape > tolerance):
            errors.append(f"{row['scenario_name']}: shape component mismatch")
        if expects_rigid != (rigid > tolerance):
            errors.append(f"{row['scenario_name']}: rigid component mismatch")
    return errors


def validate_core_motion_semantics(rows: list[dict[str, Any]]) -> list[str]:
    """用保守阈值检查正式L1/L2场景的实际响应，而不只检查施力标签。"""

    errors: list[str] = []
    for row in rows:
        name = str(row["scenario_name"])
        com_max = float(row["com_displacement_max_m"])
        rotation_max = float(row["rigid_rotation_max_rad"])
        shape_rms = float(row["shape_change_rms_m"])
        motion_type = str(row["motion_type"])
        if name == "id_static":
            if com_max >= 0.002 or rotation_max >= 0.02 or shape_rms >= 0.002:
                errors.append(f"{name}: actual motion exceeds static tolerance")
        elif motion_type == "shape":
            if shape_rms < 0.02:
                errors.append(f"{name}: actual shape change is too small")
        elif motion_type == "rigid":
            if float(row["com_speed_rms_m_s"]) < 0.10:
                errors.append(f"{name}: L1/L2 translation is still too slow")
            if rotation_max < 0.25:
                errors.append(f"{name}: commanded L1/L2 rotation is too small")
            if shape_rms >= 0.002:
                errors.append(f"{name}: fixed-shape rigid motion deforms too much")
        elif motion_type == "combined":
            if float(row["com_speed_rms_m_s"]) < 0.10:
                errors.append(f"{name}: combined L1/L2 translation is too slow")
            if rotation_max < 0.25:
                errors.append(f"{name}: combined L1/L2 rotation is too small")
            if shape_rms < 0.02:
                errors.append(f"{name}: combined L1/L2 lacks actual shape change")
            if float(row.get(
                "rigid_shape_hold_acceleration_rms_m_s2", 0.0
            )) > 1e-10:
                errors.append(f"{name}: shape hold must be disabled in combined motion")
    return errors


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--suite", choices=SCENARIO_SUITE_NAMES, default="core")
    selection.add_argument("--scenarios", nargs="+")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="number of consecutive seeds evaluated for every scenario",
    )
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=DIAGNOSTIC_OUTPUT_ROOT / "runs",
    )
    args = parser.parse_args()
    if args.seconds <= 0.0 or args.sample_hz <= 0.0 or args.seeds < 1:
        parser.error("--seconds, --sample-hz and --seeds must be positive")
    return args


def main() -> None:
    args = parse_args()
    scenarios = (
        tuple(get_scenario(name) for name in args.scenarios)
        if args.scenarios
        else list_suite_scenarios(args.suite)
    )
    seeds = [args.seed + index for index in range(args.seeds)]
    rows = [
        diagnose_scenario(
            scenario, seed=seed, seconds=args.seconds, sample_hz=args.sample_hz,
        )
        for scenario in scenarios
        for seed in seeds
    ]
    force_errors = validate_force_decomposition(rows)
    motion_errors = validate_core_motion_semantics(rows)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output = args.output / run_name
    output.mkdir(parents=True)
    _write_csv(output / "motion_metrics.csv", rows)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite if not args.scenarios else None,
        "scenario_names": [scenario.name for scenario in scenarios],
        "scenario_configs": [scenario.asdict() for scenario in scenarios],
        "seed": args.seed,
        "seeds": seeds,
        "seconds": args.seconds,
        "sample_hz": args.sample_hz,
        "force_decomposition_valid": not force_errors,
        "force_decomposition_errors": force_errors,
        "core_motion_semantics_valid": not motion_errors,
        "core_motion_semantics_errors": motion_errors,
        "source_xml": str(XML_PATH.resolve()),
        "source_xml_sha256": _sha256(XML_PATH),
        "panda_xml": str(PANDA_XML_PATH.resolve()),
        "panda_xml_sha256": _sha256(PANDA_XML_PATH),
        "menagerie_panda_assets": str(resolve_menagerie_panda_dir()),
        "source_files": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in {
                "motion_diagnostics": Path(__file__),
                "base_environment": ROOT / "src" / "panda_cable_grasp" / "env" / "environment.py",
                "scenario_registry": ROOT / "src" / "panda_cable_grasp" / "scenarios" / "registry.py",
            }.items()
        },
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "rows": rows,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"motion_diagnostics_output={output.resolve()}", flush=True)
    if force_errors or motion_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
