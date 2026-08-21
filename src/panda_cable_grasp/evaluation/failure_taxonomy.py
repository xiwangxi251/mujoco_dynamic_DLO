"""跨脚本与学习方法共用的任务结果分类和场景指纹。

这里只解释已经发生的环境事件，不改变动作、扰动或成功判定。论文主表应使用
``task_success`` 和这里的公共分类；各方法内部的成功信号可作为额外诊断保留。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


TASK_OUTCOME_TYPES = (
    "success",
    "never_bilateral_candidate",
    "bilateral_not_confirmed",
    "active_open_after_confirmed_grasp",
    "physical_slip_after_confirmed_grasp",
    "open_during_contact_loss_after_confirmed_grasp",
    "confirmed_grasp_but_no_task_success",
)


def break_causal_class(event: Mapping[str, Any]) -> str:
    """兼容新旧日志，将一次抓取断开映射到互斥的物理因果类别。"""

    recorded = event.get("causal_class")
    if recorded:
        return str(recorded)
    reason = str(event.get("reason", "other"))
    if reason == "lost_physical_pad_contact":
        return "physical_slip"
    if reason == "gripper_command_open":
        return (
            "open_during_contact_loss"
            if float(event.get("no_contact_time", 0.0)) > 0.0
            else "active_open"
        )
    return "other"


def confirmed_break_times(
    events: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    """返回确认抓取后每种断开原因的首次物理时间。"""

    first: dict[str, float] = {}
    for event in events:
        if not bool(event.get("bilateral_confirmed", False)):
            continue
        cause = break_causal_class(event)
        if cause not in {"active_open", "physical_slip", "open_during_contact_loss"}:
            continue
        event_time = float(event.get("time", float("inf")))
        first[cause] = min(first.get(cause, float("inf")), event_time)
    return first


def classify_task_outcome(
    *,
    task_success: bool,
    ever_bilateral_candidate: bool,
    ever_confirmed_grasp: bool,
    break_events: Iterable[Mapping[str, Any]],
) -> str:
    """按最早实际断开事件为一个回合分配唯一公共任务结果。"""

    if task_success:
        return "success"
    if not ever_bilateral_candidate:
        return "never_bilateral_candidate"
    if not ever_confirmed_grasp:
        return "bilateral_not_confirmed"

    first = confirmed_break_times(break_events)
    if first:
        cause = min(first, key=lambda name: (first[name], name))
        return {
            "active_open": "active_open_after_confirmed_grasp",
            "physical_slip": "physical_slip_after_confirmed_grasp",
            "open_during_contact_loss": (
                "open_during_contact_loss_after_confirmed_grasp"
            ),
        }[cause]
    return "confirmed_grasp_but_no_task_success"


def scene_fingerprint(info: Mapping[str, Any]) -> str:
    """为完整场景生成稳定指纹；不同运动/材质条件不能误判为同一场景。"""

    scene = {
        "episode_seed": info.get("episode_seed"),
        "initial_cable_dx": float(info["initial_cable_dx"]),
        "initial_cable_dy": float(info["initial_cable_dy"]),
        "disturbance_phase": float(info["disturbance_phase"]),
        "disturbance_spatial_phase": float(info["disturbance_spatial_phase"]),
        "target_body_id": int(info["target_body_id"]),
        "scenario_name": info.get("scenario_name"),
        "scenario_id": info.get("scenario_id"),
        "scenario_split": info.get("scenario_split"),
        "motion_mode": info.get("motion_mode"),
        "motion_profile_version": info.get("motion_profile_version"),
        "motion_regularity": info.get("motion_regularity"),
        "motion_frequency_scale": float(info.get("motion_frequency_scale", 1.0)),
        "disturbance_strength": float(info.get("disturbance_strength", 1.5)),
        "motion_profile_hash": info.get("motion_profile_hash"),
        "cable_length_scale": float(info.get("cable_length_scale", 1.0)),
        "cable_density_scale": float(info.get("cable_density_scale", 1.0)),
        "cable_stiffness_scale": float(info.get("cable_stiffness_scale", 1.0)),
        "cable_damping_scale": float(info.get("cable_damping_scale", 1.0)),
        "cable_friction_scale": float(info.get("cable_friction_scale", 1.0)),
    }
    payload = json.dumps(
        scene, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def base_scene_fingerprint(info: Mapping[str, Any]) -> str:
    """仅描述初态和目标，用于确认不同运动 cell 复用了同一基础场景。"""

    scene = {
        "episode_seed": info.get("episode_seed"),
        "initial_cable_dx": float(info["initial_cable_dx"]),
        "initial_cable_dy": float(info["initial_cable_dy"]),
        "disturbance_phase": float(info["disturbance_phase"]),
        "disturbance_spatial_phase": float(info["disturbance_spatial_phase"]),
        "target_body_id": int(info["target_body_id"]),
    }
    payload = json.dumps(
        scene, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
