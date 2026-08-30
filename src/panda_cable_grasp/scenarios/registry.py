"""Versioned experiment scenarios for dynamic cable grasping.

This module is intentionally independent from MuJoCo and the policy code.  It
defines the experiment protocol only; environment adapters can consume these
configs later without making scenario bookkeeping depend on a simulator.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any


SCENARIO_SCHEMA_VERSION = 1


class MotionType(str, Enum):
    """How cable motion is generated during an episode."""

    STATIC = "static"
    RIGID = "rigid"
    SHAPE = "shape"
    COMBINED = "combined"


class FactorLevel(str, Enum):
    """Three-level factor used for amplitude and frequency sweeps."""

    LOW = "low"
    NOMINAL = "nominal"
    HIGH = "high"


class MotionRegularity(str, Enum):
    """Temporal structure of the commanded cable motion."""

    REGULAR = "regular"
    QUASIPERIODIC = "quasiperiodic"
    STOCHASTIC = "stochastic"


class ScenarioSplit(str, Enum):
    """Protocol partition; OOD scenes must never be used for model selection."""

    ID = "id"
    DEV = "dev"
    OOD = "ood"


class ScenarioSuite(str, Enum):
    """Named experiment bundles used by runners and paper tables."""

    CORE = "core"
    MOTION_SWEEP = "motion_sweep"
    OOD = "ood"
    PAPER = "paper"
    ALL = "all"


# The nominal strength is the current environment's disturbance_strength=1.5.
AMPLITUDE_STRENGTHS: Mapping[FactorLevel, float] = MappingProxyType({
    FactorLevel.LOW: 0.75,
    FactorLevel.NOMINAL: 1.50,
    FactorLevel.HIGH: 2.25,
})
FREQUENCY_SCALES: Mapping[FactorLevel, float] = MappingProxyType({
    FactorLevel.LOW: 2.0 / 3.0,
    FactorLevel.NOMINAL: 1.0,
    FactorLevel.HIGH: 1.5,
})

# Calibrated over seeds 20260804..20260815 using the mass-weighted mean 3-D
# node speed from t=0.8 s to t=7.4 s.  At the nominal shape settings this gives
# 0.2147 m/s, matching the measured L1/L2 rigid-motion speed (0.2128 m/s).
SHAPE_MOTION_SCALE = 0.4695

# New OOD sweeps sample a single multiplier per episode.  The bounds are
# relative to the corresponding nominal value; for amplitude this means the
# nominal disturbance_strength (1.5) is multiplied by the sampled value.
OOD_LOW_SCALE_RANGE = (0.5, 0.9)
OOD_HIGH_SCALE_RANGE = (1.1, 1.5)


# Nominal values are frozen from panda_cable_grasp.xml.  Scenario configs store
# scale factors relative to these values so an MjSpec adapter can apply them
# directly without duplicating unit conversions.
NOMINAL_CABLE_LENGTH_M = 0.80
NOMINAL_CABLE_RADIUS_M = 0.014
NOMINAL_CABLE_DENSITY_KG_M3 = 150.0
NOMINAL_CABLE_BEND_STIFFNESS = 2_000.0
NOMINAL_CABLE_TWIST_STIFFNESS = 10_000.0
NOMINAL_CABLE_DAMPING = 0.025
NOMINAL_CABLE_SLIDING_FRICTION = 2.0
NOMINAL_CABLE_TORSIONAL_FRICTION = 0.08
NOMINAL_CABLE_ROLLING_FRICTION = 0.01


_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_enum_value(item) for item in value]
    if isinstance(value, list):
        return [_enum_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _enum_value(item) for key, item in value.items()}
    return value


def _is_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


@dataclass(frozen=True)
class ScenarioConfig:
    """Validated, immutable and JSON-serializable experiment configuration.

    该对象只冻结方法无关的实验协议；``to_env_overrides``把它映射到已经实现的
    MuJoCo运动生成器和MjSpec线缆属性变体。
    """

    name: str
    split: ScenarioSplit
    motion_type: MotionType
    amplitude_level: FactorLevel = FactorLevel.NOMINAL
    frequency_level: FactorLevel = FactorLevel.NOMINAL
    regularity: MotionRegularity = MotionRegularity.QUASIPERIODIC
    motion_profile_version: str = "factorized_v2"
    disturbance_strength: float = 1.50
    frequency_scale: float = 1.0
    shape_motion_scale: float = SHAPE_MOTION_SCALE

    cable_length_scale: float = 1.0
    cable_length_ood: bool = False
    cable_material_profile: str = "nominal"
    cable_material_ood: bool = False
    cable_density_scale: float = 1.0
    cable_stiffness_scale: float = 1.0
    cable_damping_scale: float = 1.0
    cable_friction_scale: float = 1.0

    # Optional per-episode OOD ranges.  The concrete scalar fields above hold
    # the midpoint so the config remains directly usable by legacy callers;
    # ``sample_for_episode`` replaces them with a deterministic draw.
    disturbance_strength_range: tuple[float, float] | None = None
    frequency_scale_range: tuple[float, float] | None = None
    cable_length_scale_range: tuple[float, float] | None = None
    cable_material_scale_range: tuple[float, float] | None = None
    ood_factor: str | None = None
    ood_level: str | None = None

    description: str = ""
    tags: tuple[str, ...] = ()
    schema_version: int = SCENARIO_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "split", ScenarioSplit(self.split))
            object.__setattr__(self, "motion_type", MotionType(self.motion_type))
            object.__setattr__(
                self, "amplitude_level", FactorLevel(self.amplitude_level)
            )
            object.__setattr__(
                self, "frequency_level", FactorLevel(self.frequency_level)
            )
            object.__setattr__(
                self, "regularity", MotionRegularity(self.regularity)
            )
        except ValueError as error:
            raise ValueError(f"invalid categorical scenario value: {error}") from error

        if not isinstance(self.name, str) or not _NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                "scenario name must match ^[a-z][a-z0-9_]*$: "
                f"{self.name!r}"
            )
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        allowed_profiles = {
            "factorized_v1",
            "factorized_v2",
            "rigid_level1_single_pass_v2",
            "rigid_level2_single_pass_v2",
        }
        if self.motion_profile_version not in allowed_profiles:
            raise ValueError(
                "unsupported registered motion_profile_version: "
                f"{self.motion_profile_version!r}"
            )
        uses_rigid_trajectory = self.motion_profile_version.startswith("rigid_level")
        if uses_rigid_trajectory and self.motion_type not in {
            MotionType.RIGID, MotionType.COMBINED,
        }:
            raise ValueError(
                "Level-1/Level-2 trajectories require rigid or combined motion"
            )
        if (
            self.motion_type in {MotionType.RIGID, MotionType.COMBINED}
            and not uses_rigid_trajectory
        ):
            raise ValueError(
                "rigid and combined scenarios must explicitly select Level-1 or Level-2"
            )
        if not isinstance(self.cable_material_profile, str) or not _NAME_PATTERN.fullmatch(
            self.cable_material_profile
        ):
            raise ValueError("cable_material_profile must be a lowercase identifier")
        if isinstance(self.tags, str):
            raise TypeError("tags must be an iterable of strings, not one string")
        tags = tuple(self.tags)
        if any(not isinstance(tag, str) or not _NAME_PATTERN.fullmatch(tag) for tag in tags):
            raise ValueError("every tag must be a lowercase identifier")
        if len(tags) != len(set(tags)):
            raise ValueError("scenario tags must be unique")
        object.__setattr__(self, "tags", tags)

        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != SCENARIO_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be {SCENARIO_SCHEMA_VERSION}, "
                f"got {self.schema_version!r}"
            )
        for flag_name in ("cable_length_ood", "cable_material_ood"):
            if not isinstance(getattr(self, flag_name), bool):
                raise TypeError(f"{flag_name} must be bool")

        numeric_fields = (
            "disturbance_strength",
            "frequency_scale",
            "shape_motion_scale",
            "cable_length_scale",
            "cable_density_scale",
            "cable_stiffness_scale",
            "cable_damping_scale",
            "cable_friction_scale",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool):
                raise TypeError(f"{field_name} must be numeric, not bool")
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError(f"{field_name} must be numeric") from error
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)

        range_fields = (
            "disturbance_strength_range",
            "frequency_scale_range",
            "cable_length_scale_range",
            "cable_material_scale_range",
        )
        for field_name in range_fields:
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, (str, bytes)):
                raise TypeError(f"{field_name} must be a pair of numeric bounds")
            try:
                bounds = tuple(value)
            except TypeError as error:
                raise TypeError(
                    f"{field_name} must be a pair of numeric bounds"
                ) from error
            if len(bounds) != 2:
                raise ValueError(f"{field_name} must contain exactly two bounds")
            try:
                bounds = (float(bounds[0]), float(bounds[1]))
            except (TypeError, ValueError) as error:
                raise TypeError(f"{field_name} bounds must be numeric") from error
            if (
                not all(math.isfinite(bound) and bound > 0.0 for bound in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError(
                    f"{field_name} must be finite, positive and ordered"
                )
            object.__setattr__(self, field_name, bounds)

        for field_name in ("ood_factor", "ood_level"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value)
            ):
                raise ValueError(f"{field_name} must be a lowercase identifier")

        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` when fields do not form a coherent scenario."""
        if self.disturbance_strength < 0.0 or self.frequency_scale < 0.0:
            raise ValueError("motion strength and frequency scale must be non-negative")
        if self.motion_type is MotionType.STATIC:
            if not _is_close(self.disturbance_strength, 0.0):
                raise ValueError("static scenarios require disturbance_strength=0")
            if not _is_close(self.frequency_scale, 0.0):
                raise ValueError("static scenarios require frequency_scale=0")
            if self.amplitude_level is not FactorLevel.NOMINAL:
                raise ValueError("static scenarios use nominal amplitude by convention")
            if self.frequency_level is not FactorLevel.NOMINAL:
                raise ValueError("static scenarios use nominal frequency by convention")
            if self.regularity is not MotionRegularity.REGULAR:
                raise ValueError("static scenarios use regular regularity by convention")
        else:
            expected_strength = AMPLITUDE_STRENGTHS[self.amplitude_level]
            expected_frequency = FREQUENCY_SCALES[self.frequency_level]
            if (
                self.disturbance_strength_range is None
                and not _is_close(self.disturbance_strength, expected_strength)
            ):
                raise ValueError(
                    "disturbance_strength does not match amplitude_level: "
                    f"expected {expected_strength}, got {self.disturbance_strength}"
                )
            if (
                self.frequency_scale_range is None
                and not _is_close(self.frequency_scale, expected_frequency)
            ):
                raise ValueError(
                    "frequency_scale does not match frequency_level: "
                    f"expected {expected_frequency}, got {self.frequency_scale}"
                )

        positive_fields = (
            "shape_motion_scale",
            "cable_length_scale",
            "cable_density_scale",
            "cable_stiffness_scale",
            "cable_damping_scale",
            "cable_friction_scale",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")

        for field_name, value_name in (
            ("disturbance_strength_range", "disturbance_strength"),
            ("frequency_scale_range", "frequency_scale"),
            ("cable_length_scale_range", "cable_length_scale"),
        ):
            bounds = getattr(self, field_name)
            if bounds is not None and not (bounds[0] <= getattr(self, value_name) <= bounds[1]):
                raise ValueError(
                    f"{value_name} midpoint must lie inside {field_name}"
                )
            if bounds is not None and self.split is not ScenarioSplit.OOD:
                raise ValueError(f"{field_name} is only valid in the OOD split")

        material_range = self.cable_material_scale_range
        if material_range is not None:
            if self.split is not ScenarioSplit.OOD:
                raise ValueError("cable_material_scale_range is only valid in OOD")
            material_values = (
                self.cable_density_scale,
                self.cable_stiffness_scale,
                self.cable_damping_scale,
                self.cable_friction_scale,
            )
            if not all(
                material_range[0] <= value <= material_range[1]
                for value in material_values
            ) or not all(_is_close(value, material_values[0]) for value in material_values[1:]):
                raise ValueError(
                    "material range requires one shared midpoint for all material scales"
                )

        if self.ood_factor is not None or self.ood_level is not None:
            if self.split is not ScenarioSplit.OOD:
                raise ValueError("ood_factor/ood_level are only valid in OOD")
            if self.ood_factor not in {"amplitude", "frequency", "length", "material"}:
                raise ValueError(f"unsupported ood_factor: {self.ood_factor!r}")
            if self.ood_level not in {"low", "high"}:
                raise ValueError(f"unsupported ood_level: {self.ood_level!r}")

        length_is_nominal = _is_close(self.cable_length_scale, 1.0)
        if self.cable_length_ood == length_is_nominal:
            expected = "non-nominal" if self.cable_length_ood else "nominal"
            raise ValueError(
                f"cable_length_ood={self.cable_length_ood} requires {expected} length"
            )

        nominal_material_values = (
            (self.cable_density_scale, 1.0),
            (self.cable_stiffness_scale, 1.0),
            (self.cable_damping_scale, 1.0),
            (self.cable_friction_scale, 1.0),
        )
        material_is_nominal = (
            self.cable_material_profile == "nominal"
            and all(_is_close(value, nominal) for value, nominal in nominal_material_values)
        )
        if material_range is not None:
            material_is_nominal = False
        if self.cable_material_ood == material_is_nominal:
            expected = "non-nominal" if self.cable_material_ood else "nominal"
            raise ValueError(
                f"cable_material_ood={self.cable_material_ood} requires {expected} material"
            )
        if (self.cable_length_ood or self.cable_material_ood) and self.split is not ScenarioSplit.OOD:
            raise ValueError("cable length/material OOD is only valid in the OOD split")

    @property
    def uses_rigid_motion(self) -> bool:
        return self.motion_type in (MotionType.RIGID, MotionType.COMBINED)

    @property
    def uses_shape_motion(self) -> bool:
        return self.motion_type in (MotionType.SHAPE, MotionType.COMBINED)

    def _field_dict(self) -> dict[str, Any]:
        return {
            item.name: _enum_value(getattr(self, item.name))
            for item in fields(self)
        }

    def _identity_payload(self) -> dict[str, Any]:
        payload = self._field_dict()
        # Prose and search tags can evolve without changing the actual scenario.
        payload.pop("description")
        payload.pop("tags")
        # A sampled episode has different concrete scalar values, but it is
        # still the same registered scenario/paired seed.  Keep the declared
        # range in the identity and omit the sampled realization.
        for range_name, value_name in (
            ("disturbance_strength_range", "disturbance_strength"),
            ("frequency_scale_range", "frequency_scale"),
            ("cable_length_scale_range", "cable_length_scale"),
        ):
            if payload.get(range_name) is not None:
                payload.pop(value_name, None)
        if payload.get("cable_material_scale_range") is not None:
            for value_name in (
                "cable_density_scale",
                "cable_stiffness_scale",
                "cable_damping_scale",
                "cable_friction_scale",
            ):
                payload.pop(value_name, None)
        return payload

    @property
    def scenario_hash(self) -> str:
        canonical = json.dumps(
            self._identity_payload(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @property
    def scenario_id(self) -> str:
        return f"scenario-v{self.schema_version}-{self.scenario_hash[:16]}"

    def asdict(self, *, include_identity: bool = True) -> dict[str, Any]:
        """Return a plain JSON-friendly dict (enums/tuples become strings/lists)."""
        result = self._field_dict()
        if include_identity:
            result["scenario_id"] = self.scenario_id
            result["scenario_hash"] = self.scenario_hash
        return result

    def canonical_json(self, *, include_identity: bool = True) -> str:
        return json.dumps(
            self.asdict(include_identity=include_identity),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_env_overrides(self) -> dict[str, str | float | bool]:
        """返回可直接传给当前 ``EnvConfig`` 的纯标量行为字段。"""
        return {
            "disturbance_strength": self.disturbance_strength,
            "motion_mode": self.motion_type.value,
            "motion_profile_version": self.motion_profile_version,
            "motion_regularity": self.regularity.value,
            "motion_frequency_scale": self.frequency_scale,
            "shape_motion_scale": self.shape_motion_scale,
            "cable_length_scale": self.cable_length_scale,
            "cable_density_scale": self.cable_density_scale,
            "cable_stiffness_scale": self.cable_stiffness_scale,
            "cable_damping_scale": self.cable_damping_scale,
            "cable_friction_scale": self.cable_friction_scale,
        }

    def sample_for_episode(self, seed: int) -> "ScenarioConfig":
        """Return a deterministic realization of any declared OOD ranges.

        The draw is keyed by the registered scenario identity, seed and field
        name, so paired methods receive exactly the same physical parameters
        even when they run in different worker processes.
        """

        updates: dict[str, Any] = {}

        def draw(bounds: tuple[float, float], field_name: str) -> float:
            token = f"{self.scenario_id}:{int(seed)}:{field_name}".encode("ascii")
            digest = hashlib.sha256(token).digest()
            unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
            return bounds[0] + (bounds[1] - bounds[0]) * unit

        if self.disturbance_strength_range is not None:
            updates["disturbance_strength"] = draw(
                self.disturbance_strength_range, "disturbance_strength"
            )
        if self.frequency_scale_range is not None:
            updates["frequency_scale"] = draw(
                self.frequency_scale_range, "frequency_scale"
            )
        if self.cable_length_scale_range is not None:
            updates["cable_length_scale"] = draw(
                self.cable_length_scale_range, "cable_length_scale"
            )
        if self.cable_material_scale_range is not None:
            material_scale = draw(
                self.cable_material_scale_range, "cable_material_scale"
            )
            updates.update({
                "cable_density_scale": material_scale,
                "cable_stiffness_scale": material_scale,
                "cable_damping_scale": material_scale,
                "cable_friction_scale": material_scale,
            })
        return self if not updates else replace(self, **updates)

    @classmethod
    def from_dict(cls, source: Mapping[str, Any]) -> "ScenarioConfig":
        """Restore a config and verify optional serialized identity fields."""
        payload = dict(source)
        expected_id = payload.pop("scenario_id", None)
        expected_hash = payload.pop("scenario_hash", None)
        if "tags" in payload:
            payload["tags"] = tuple(payload["tags"])
        scenario = cls(**payload)
        if expected_id is not None and expected_id != scenario.scenario_id:
            raise ValueError(
                f"scenario_id mismatch: expected {expected_id}, got {scenario.scenario_id}"
            )
        if expected_hash is not None and expected_hash != scenario.scenario_hash:
            raise ValueError("scenario_hash mismatch")
        return scenario


class ScenarioRegistry(Mapping[str, ScenarioConfig]):
    """Immutable name-to-config mapping with deterministic split queries."""

    def __init__(self, scenarios: tuple[ScenarioConfig, ...] | list[ScenarioConfig]):
        by_name: dict[str, ScenarioConfig] = {}
        by_id: dict[str, str] = {}
        for scenario in scenarios:
            if not isinstance(scenario, ScenarioConfig):
                raise TypeError("ScenarioRegistry accepts ScenarioConfig values only")
            if scenario.name in by_name:
                raise ValueError(f"duplicate scenario name: {scenario.name}")
            if scenario.scenario_id in by_id:
                raise ValueError(
                    "duplicate scenario identity: "
                    f"{scenario.name} and {by_id[scenario.scenario_id]}"
                )
            by_name[scenario.name] = scenario
            by_id[scenario.scenario_id] = scenario.name
        self._by_name = MappingProxyType(by_name)

    def __getitem__(self, name: str) -> ScenarioConfig:
        try:
            return self._by_name[name]
        except KeyError as error:
            choices = ", ".join(sorted(self._by_name))
            raise KeyError(f"unknown scenario {name!r}; choices: {choices}") from error

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._by_name))

    def __len__(self) -> int:
        return len(self._by_name)

    def list(self, split: ScenarioSplit | str | None = None) -> tuple[ScenarioConfig, ...]:
        selected_split = None if split is None else ScenarioSplit(split)
        return tuple(
            self._by_name[name]
            for name in sorted(self._by_name)
            if selected_split is None or self._by_name[name].split is selected_split
        )

    def names(self, split: ScenarioSplit | str | None = None) -> tuple[str, ...]:
        return tuple(scenario.name for scenario in self.list(split))

    def asdict(self, split: ScenarioSplit | str | None = None) -> list[dict[str, Any]]:
        return [scenario.asdict() for scenario in self.list(split)]


def _scenario(
    name: str,
    split: ScenarioSplit,
    motion_type: MotionType,
    *,
    amplitude: FactorLevel = FactorLevel.NOMINAL,
    frequency: FactorLevel = FactorLevel.NOMINAL,
    regularity: MotionRegularity = MotionRegularity.QUASIPERIODIC,
    description: str = "",
    tags: tuple[str, ...] = (),
    **overrides: Any,
) -> ScenarioConfig:
    if motion_type is MotionType.STATIC:
        strength = 0.0
        frequency_scale = 0.0
    else:
        strength = AMPLITUDE_STRENGTHS[amplitude]
        frequency_scale = FREQUENCY_SCALES[frequency]
    disturbance_range = overrides.get("disturbance_strength_range")
    frequency_range = overrides.get("frequency_scale_range")
    length_range = overrides.get("cable_length_scale_range")
    material_range = overrides.get("cable_material_scale_range")
    if disturbance_range is not None:
        strength = sum(disturbance_range) / 2.0
    if frequency_range is not None:
        frequency_scale = sum(frequency_range) / 2.0
    if length_range is not None:
        overrides["cable_length_scale"] = sum(length_range) / 2.0
    if material_range is not None:
        material_midpoint = sum(material_range) / 2.0
        overrides.update({
            "cable_density_scale": material_midpoint,
            "cable_stiffness_scale": material_midpoint,
            "cable_damping_scale": material_midpoint,
            "cable_friction_scale": material_midpoint,
        })
    return ScenarioConfig(
        name=name,
        split=split,
        motion_type=motion_type,
        amplitude_level=amplitude,
        frequency_level=frequency,
        regularity=regularity,
        disturbance_strength=strength,
        frequency_scale=frequency_scale,
        description=description,
        tags=tags,
        **overrides,
    )


def _registered_scenarios() -> list[ScenarioConfig]:
    scenarios = [
        _scenario(
            "id_static",
            ScenarioSplit.ID,
            MotionType.STATIC,
            regularity=MotionRegularity.REGULAR,
            description="No commanded cable motion.",
            tags=("motion_axis",),
        )
    ]

    for level in FactorLevel:
        shape_name = (
            "id_shape_nominal_current"
            if level is FactorLevel.NOMINAL
            else f"id_shape_{level.value}"
        )
        scenarios.append(_scenario(
            shape_name,
            ScenarioSplit.ID,
            MotionType.SHAPE,
            amplitude=level,
            frequency=level,
            description=(
                "Mean-removed quasiperiodic shape motion at "
                f"{level.value} difficulty."
            ),
            tags=("main_grid", "shape"),
        ))

    # L1/L2 are formal ID factors.  Every rigid component is explicit in the
    # scenario name; no registered scenario can fall back to the removed
    # bounded quasiperiodic whole-object trajectory.
    for trajectory, profile in (
        ("l1", "rigid_level1_single_pass_v2"),
        ("l2", "rigid_level2_single_pass_v2"),
    ):
        for motion_type in (MotionType.RIGID, MotionType.COMBINED):
            for level in FactorLevel:
                scenarios.append(_scenario(
                    f"id_{motion_type.value}_{trajectory}_{level.value}",
                    ScenarioSplit.ID,
                    motion_type,
                    amplitude=level,
                    frequency=level,
                    motion_profile_version=profile,
                    description=(
                        f"{motion_type.value} ID scene with one-way "
                        f"{trajectory.upper()} translation and finite rotation "
                        f"at {level.value} speed."
                    ),
                    tags=("main_grid", f"rigid_{trajectory}"),
                ))

    scenarios.extend([
        _scenario(
            "dev_shape_amplitude_low", ScenarioSplit.DEV, MotionType.SHAPE,
            amplitude=FactorLevel.LOW,
            description="Low-amplitude shape-only development scene.",
            tags=("amplitude_sweep",),
        ),
        _scenario(
            "dev_shape_amplitude_high", ScenarioSplit.DEV, MotionType.SHAPE,
            amplitude=FactorLevel.HIGH,
            description="High-amplitude shape-only development scene.",
            tags=("amplitude_sweep",),
        ),
        _scenario(
            "dev_shape_frequency_low", ScenarioSplit.DEV, MotionType.SHAPE,
            frequency=FactorLevel.LOW,
            description="Low-frequency shape-only development scene.",
            tags=("frequency_sweep",),
        ),
        _scenario(
            "dev_shape_frequency_high", ScenarioSplit.DEV, MotionType.SHAPE,
            frequency=FactorLevel.HIGH,
            description="High-frequency shape-only development scene.",
            tags=("frequency_sweep",),
        ),
        _scenario(
            "dev_shape_regular", ScenarioSplit.DEV, MotionType.SHAPE,
            regularity=MotionRegularity.REGULAR,
            description="Regular shape-only development scene.",
            tags=("regularity_sweep",),
        ),
    ])

    # OOD is intentionally L1-only for this experiment.  Each factor has a
    # low/high interval and is sampled independently for every episode.
    trajectory = "l1"
    profile = "rigid_level1_single_pass_v2"
    common = dict(
        split=ScenarioSplit.OOD,
        motion_type=MotionType.COMBINED,
        motion_profile_version=profile,
    )
    nominal_strength = AMPLITUDE_STRENGTHS[FactorLevel.NOMINAL]
    for level, scale_range in (
        ("high", OOD_HIGH_SCALE_RANGE),
        ("low", OOD_LOW_SCALE_RANGE),
    ):
        amplitude_range = tuple(nominal_strength * value for value in scale_range)
        scenarios.append(_scenario(
            f"ood_combined_{trajectory}_amplitude_{level}",
            amplitude=FactorLevel.HIGH if level == "high" else FactorLevel.LOW,
            disturbance_strength_range=amplitude_range,
            ood_factor="amplitude",
            ood_level=level,
            description=(
                f"L1 combined OOD shape-motion amplitude sampled at "
                f"{scale_range[0]:.1f}-{scale_range[1]:.1f}x nominal."
            ),
            tags=("amplitude_ood", "rigid_l1"),
            **common,
        ))
        scenarios.append(_scenario(
            f"ood_combined_{trajectory}_frequency_{level}",
            frequency=FactorLevel.HIGH if level == "high" else FactorLevel.LOW,
            frequency_scale_range=scale_range,
            ood_factor="frequency",
            ood_level=level,
            description=(
                f"L1 combined OOD frequency sampled at "
                f"{scale_range[0]:.1f}-{scale_range[1]:.1f}x nominal."
            ),
            tags=("frequency_ood", "rigid_l1"),
            **common,
        ))
        scenarios.append(_scenario(
            f"ood_combined_{trajectory}_length_{level}",
            cable_length_ood=True,
            cable_length_scale_range=scale_range,
            ood_factor="length",
            ood_level=level,
            description=(
                f"L1 combined OOD cable length sampled at "
                f"{scale_range[0]:.1f}-{scale_range[1]:.1f}x nominal."
            ),
            tags=("length_ood", "rigid_l1"),
            **common,
        ))
        scenarios.append(_scenario(
            f"ood_combined_{trajectory}_material_{level}",
            cable_material_profile=f"range_{level}",
            cable_material_ood=True,
            cable_material_scale_range=scale_range,
            ood_factor="material",
            ood_level=level,
            description=(
                f"L1 combined OOD material scales jointly sampled at "
                f"{scale_range[0]:.1f}-{scale_range[1]:.1f}x nominal."
            ),
            tags=("material_ood", "rigid_l1"),
            **common,
        ))
    return scenarios


DEFAULT_SCENARIO_NAME = "id_shape_nominal_current"
SCENARIO_REGISTRY = ScenarioRegistry(_registered_scenarios())
SCENARIOS = SCENARIO_REGISTRY
DEFAULT_SCENARIO = SCENARIO_REGISTRY[DEFAULT_SCENARIO_NAME]

CORE_SCENARIO_NAMES = (
    "id_static",
    "id_shape_nominal_current",
    "id_rigid_l1_nominal",
    "id_rigid_l2_nominal",
    "id_combined_l1_nominal",
    "id_combined_l2_nominal",
)
SCENARIO_SUITE_NAMES = tuple(suite.value for suite in ScenarioSuite)


def get_scenario(name: str) -> ScenarioConfig:
    """Look up a registered scenario by its stable human-readable name."""
    return SCENARIO_REGISTRY[name]


def list_scenarios(
    split: ScenarioSplit | str | None = None,
) -> tuple[ScenarioConfig, ...]:
    """List all scenarios, or only one deterministic protocol split."""
    return SCENARIO_REGISTRY.list(split)


def list_scenario_names(split: ScenarioSplit | str | None = None) -> tuple[str, ...]:
    return SCENARIO_REGISTRY.names(split)


def parse_scenario_suite(suite: ScenarioSuite | str) -> ScenarioSuite:
    """Parse a suite name; ``paper/all`` is accepted as a convenience alias."""
    if isinstance(suite, ScenarioSuite):
        return suite
    if not isinstance(suite, str):
        raise TypeError("scenario suite must be a string or ScenarioSuite")
    normalized = suite.strip().lower().replace("-", "_")
    if normalized == "paper/all":
        normalized = ScenarioSuite.PAPER.value
    try:
        return ScenarioSuite(normalized)
    except ValueError as error:
        raise ValueError(
            f"unknown scenario suite {suite!r}; choices: "
            + ", ".join(SCENARIO_SUITE_NAMES)
        ) from error


def list_suite_scenarios(
    suite: ScenarioSuite | str,
    registry: ScenarioRegistry = SCENARIO_REGISTRY,
) -> tuple[ScenarioConfig, ...]:
    """Return a deterministic named experiment suite.

    ``core``, ``motion_sweep`` and ``ood`` are disjoint. ``paper`` and ``all``
    both return the full registry now that L1/L2 are explicit ID factors.
    """
    selected = parse_scenario_suite(suite)
    core_names = set(CORE_SCENARIO_NAMES)
    if selected is ScenarioSuite.CORE:
        names = core_names
    elif selected is ScenarioSuite.MOTION_SWEEP:
        names = {
            scenario.name
            for scenario in registry.list()
            if scenario.split is not ScenarioSplit.OOD
            and scenario.name not in core_names
        }
    elif selected is ScenarioSuite.OOD:
        names = set(registry.names(ScenarioSplit.OOD))
    elif selected is ScenarioSuite.PAPER:
        names = set(registry)
    else:
        names = set(registry)
    missing = names - set(registry)
    if missing:
        raise ValueError(
            "scenario registry is missing suite members: " + ", ".join(sorted(missing))
        )
    return tuple(registry[name] for name in sorted(names))


def list_suite_names(
    suite: ScenarioSuite | str,
    registry: ScenarioRegistry = SCENARIO_REGISTRY,
) -> tuple[str, ...]:
    return tuple(scenario.name for scenario in list_suite_scenarios(suite, registry))


def validate_registry(registry: ScenarioRegistry = SCENARIO_REGISTRY) -> None:
    """Validate required factor and OOD coverage for the experiment protocol."""
    scenarios = registry.list()
    required_checks = {
        "motion types": ({item.motion_type for item in scenarios}, set(MotionType)),
        "amplitude levels": (
            {item.amplitude_level for item in scenarios if item.motion_type is not MotionType.STATIC},
            set(FactorLevel),
        ),
        "frequency levels": (
            {item.frequency_level for item in scenarios if item.motion_type is not MotionType.STATIC},
            set(FactorLevel),
        ),
        # Stochastic regularity was part of the previous OOD suite.  The
        # current protocol samples continuous amplitudes/frequencies instead;
        # regular and quasiperiodic remain the registered regularities.
        "regularities": (
            {item.regularity for item in scenarios},
            {MotionRegularity.REGULAR, MotionRegularity.QUASIPERIODIC},
        ),
        "splits": ({item.split for item in scenarios}, set(ScenarioSplit)),
    }
    for label, (actual, expected) in required_checks.items():
        missing = expected - actual
        if missing:
            raise ValueError(f"registry is missing {label}: {sorted(item.value for item in missing)}")
    if not any(item.cable_length_ood for item in scenarios):
        raise ValueError("registry has no explicit cable length OOD scenario")
    if not any(item.cable_material_ood for item in scenarios):
        raise ValueError("registry has no explicit cable material OOD scenario")
    if any(
        item.regularity is MotionRegularity.STOCHASTIC
        and item.split is not ScenarioSplit.OOD
        for item in scenarios
    ):
        raise ValueError("stochastic regularity must remain unseen outside OOD")

    default = registry[DEFAULT_SCENARIO_NAME]
    if default.motion_type is not MotionType.SHAPE:
        raise ValueError("default scenario must preserve the current shape motion")
    if not _is_close(default.disturbance_strength, 1.5):
        raise ValueError("default scenario must preserve disturbance_strength=1.5")

    core = set(list_suite_names(ScenarioSuite.CORE, registry))
    sweep = set(list_suite_names(ScenarioSuite.MOTION_SWEEP, registry))
    ood = set(list_suite_names(ScenarioSuite.OOD, registry))
    if core != set(CORE_SCENARIO_NAMES):
        raise ValueError("core suite does not match CORE_SCENARIO_NAMES")
    suites = (core, sweep, ood)
    if any(left & right for index, left in enumerate(suites) for right in suites[index + 1:]):
        raise ValueError("core, motion_sweep and OOD suites must be disjoint")
    if core | sweep | ood != set(registry):
        raise ValueError("named suites do not cover the complete registry")
    if set(list_suite_names(ScenarioSuite.PAPER, registry)) != core | sweep | ood:
        raise ValueError("paper suite must cover all formal scenarios")
    if set(list_suite_names(ScenarioSuite.ALL, registry)) != set(registry):
        raise ValueError("all suite must cover the complete registry")


def _self_test() -> None:
    validate_registry()
    encoded = json.dumps(DEFAULT_SCENARIO.asdict(), allow_nan=False, sort_keys=True)
    restored = ScenarioConfig.from_dict(json.loads(encoded))
    if restored != DEFAULT_SCENARIO:
        raise AssertionError("ScenarioConfig JSON round trip changed the config")
    if restored.scenario_hash != DEFAULT_SCENARIO.scenario_hash:
        raise AssertionError("scenario hash is not stable across JSON round trip")
    for split in ScenarioSplit:
        if not list_scenarios(split):
            raise AssertionError(f"empty scenario split: {split.value}")
    for suite in ScenarioSuite:
        if not list_suite_scenarios(suite):
            raise AssertionError(f"empty scenario suite: {suite.value}")


validate_registry()


__all__ = [
    "AMPLITUDE_STRENGTHS",
    "CORE_SCENARIO_NAMES",
    "DEFAULT_SCENARIO",
    "DEFAULT_SCENARIO_NAME",
    "FREQUENCY_SCALES",
    "FactorLevel",
    "MotionRegularity",
    "MotionType",
    "SCENARIOS",
    "SCENARIO_REGISTRY",
    "SCENARIO_SCHEMA_VERSION",
    "SCENARIO_SUITE_NAMES",
    "SHAPE_MOTION_SCALE",
    "OOD_HIGH_SCALE_RANGE",
    "OOD_LOW_SCALE_RANGE",
    "ScenarioConfig",
    "ScenarioRegistry",
    "ScenarioSplit",
    "ScenarioSuite",
    "get_scenario",
    "list_scenario_names",
    "list_scenarios",
    "list_suite_names",
    "list_suite_scenarios",
    "parse_scenario_suite",
    "validate_registry",
]


if __name__ == "__main__":
    _self_test()
    print(
        f"scenario_registry_ok total={len(SCENARIO_REGISTRY)} "
        + " ".join(
            f"{split.value}={len(list_scenarios(split))}"
            for split in ScenarioSplit
        )
        + f" default={DEFAULT_SCENARIO_NAME} id={DEFAULT_SCENARIO.scenario_id}",
        flush=True,
    )
