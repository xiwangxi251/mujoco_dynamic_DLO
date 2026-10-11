"""Isolated feedback/servo/gravity interventions for the tested JIK variants."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from record_retest_136_videos import create_rig, ROOT

OUT = ROOT / "outputs/vertical_workspace/scripted_136_retest_20261004/pose_diagnosis"


def run(solver, label, gain, servo, feedforward, x, y):
    version = 1 if solver == "hierarchy" else 6
    rig, env, native = create_rig(version, OUT / solver / label)
    model, data = rig.model, rig.data
    orientation = rig.policy._orientation_error
    native_gain = 2.5 if version == 1 else 6.0

    def scaled_error():
        error, angle = orientation()
        return error * (gain / native_gain), angle

    rig.policy._orientation_error = scaled_error
    if servo:
        model.actuator_gainprm[:7, 0] *= 4
        model.actuator_biasprm[:7, 1] *= 4
        model.actuator_biasprm[:7, 2] *= 2
    probe = mujoco.MjData(model)
    target = np.array([x, y, 0.005])
    rig.reset()
    reference = env.hand_position.copy()
    approach = np.array([x, y, 0.10])
    descending = False
    held = 0
    ever_held = False
    peak_torque = np.zeros(7)
    peak_ff = np.zeros(7)
    guard_count = saturation_count = 0
    records, joint_states = [], []
    reason = "budget_16s"
    max_pose = 0.0
    for step in range(800):
        delta = (target if descending else approach) - reference
        reference += delta * min(1.0, (0.03 if descending else 0.15) * 0.02 / max(np.linalg.norm(delta), 1e-12))
        if not descending and np.linalg.norm(reference - approach) < 1e-9:
            descending = True
        rig.policy.phase = native.Phase.INTERCEPT if version == 1 or descending else native.Phase.APPROACH
        request = rig.policy._ik_action(reference, env.gripper_open_ctrl)
        applied = env._limit_robot_action(np.clip(request, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]))
        for _ in range(10):
            guarded, active = env._velocity_guarded_action(applied)
            guard_count += int(active)
            data.ctrl[:] = guarded
            data.qfrc_applied[:] = 0
            pd = (model.actuator_gainprm[:7, 0] * guarded[:7]
                  + model.actuator_biasprm[:7, 0]
                  + model.actuator_biasprm[:7, 1] * data.qpos[env.arm_qpos_adr]
                  + model.actuator_biasprm[:7, 2] * data.qvel[env.arm_dof_adr])
            bounds = model.actuator_forcerange[:7]
            pd = np.clip(pd, bounds[:, 0], bounds[:, 1])
            ff = np.zeros(7)
            if feedforward:
                probe.qpos[:] = data.qpos
                probe.qvel[:] = 0
                mujoco.mj_forward(model, probe)
                desired_ff = probe.qfrc_bias[env.arm_dof_adr].copy()
                total = np.clip(pd + desired_ff, bounds[:, 0], bounds[:, 1])
                saturation_count += int(np.any(abs(total - pd - desired_ff) > 1e-9))
                ff = total - pd
                data.qfrc_applied[env.arm_dof_adr] = ff
            peak_torque = np.maximum(peak_torque, abs(pd + ff))
            peak_ff = np.maximum(peak_ff, abs(ff))
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        measurement = rig.measure(target, collision=(step % 5 == 0 or held > 0))
        pose = float(measurement[2])
        finger_floor = min(float(rig.geometry.geom_vertices(g)[:, 2].min()) for g in rig.geometry.finger_geoms)
        max_pose = max(max_pose, pose)
        ok = (measurement[0] <= 0.005 and pose <= 2.0 and 0.0002 <= finger_floor <= 0.008
              and measurement[3] >= 0.0002 and measurement[4] >= 0 and measurement[5] >= 0.001)
        held = held + 1 if ok else 0
        ever_held |= held >= 100
        records.append([data.time, float(measurement[0]) * 1000, pose,
                        float(measurement[3]) * 1000, float(measurement[4]),
                        float(np.linalg.norm(data.ctrl[:7] - data.qpos[env.arm_qpos_adr])),
                        float(np.max(abs(data.qvel[env.arm_dof_adr]))), float(ok)])
        joint_states.append(data.qpos[:9].copy())
        if measurement[3] < -0.001:
            reason = "floor_penetration"
            break
    records = np.asarray(records)
    q = data.qpos[env.arm_qpos_adr].copy()
    probe.qpos[:] = data.qpos
    probe.qvel[:] = 0
    mujoco.mj_forward(model, probe)
    jacobian = rig.policy._task_jacobian()
    jp, jr = jacobian[:3], jacobian[3:]
    jo = jr @ (np.eye(7) - np.linalg.pinv(jp, rcond=1e-5) @ jp)
    so = np.linalg.svd(jo, compute_uv=False)
    sj = np.linalg.svd(jacobian, compute_uv=False)
    lam = 0.04 + 0.12 * np.clip((0.10 - so[-1]) / 0.10, 0, 1) ** 2
    last = records[records[:, 0] >= records[-1, 0] - 2 + 1e-8]
    result = {
        "solver": solver, "intervention": label, "angular_gain": gain,
        "servo_kp_scale": 4 if servo else 1, "servo_kv_scale": 2 if servo else 1,
        "gravity_feedforward": feedforward, "goal_xy_m": [x, y], "reason": reason,
        "duration_s": float(data.time), "final_pose_deg": float(records[-1, 2]),
        "final_position_mm": float(records[-1, 1]), "max_path_pose_deg": max_pose,
        "last_2s_max_pose_deg": float(last[:, 2].max()),
        "last_2s_conditions_all_pass": bool(len(records) == 800 and last[:, 7].all()),
        "ever_held_2s": bool(ever_held), "actual_joint_velocity_peak_at_end": float(records[-1, 6]),
        "joint_target_minus_actual_rad": (data.ctrl[:7] - q).tolist(),
        "gravity_torque_nm": probe.qfrc_bias[env.arm_dof_adr].tolist(),
        "joint_position_rad": q.tolist(),
        "joint_margin_deg": float(records[-1, 4]),
        "joint_ranges_rad": model.jnt_range[env.arm_joint_ids].tolist(),
        "position_nullspace_orientation_sigma_min": float(so[-1]),
        "position_nullspace_orientation_damping": float(lam),
        "orientation_minimum_mode_transmission": float(so[-1] ** 2 / (so[-1] ** 2 + lam ** 2)),
        "full_jacobian_sigma_min": float(sj[-1]),
        "full_jacobian_minimum_mode_transmission": float(sj[-1] ** 2 / (sj[-1] ** 2 + 0.04 ** 2)),
        "peak_torque_nm": peak_torque.tolist(), "peak_ff_nm": peak_ff.tolist(),
        "torque_saturation_substeps": saturation_count, "velocity_guard_substeps": guard_count,
    }
    # Reproduce original state prefixes before interpreting interventions.
    archive_version = version if label == "original" else 3 if solver == "hierarchy" and label == "all_three" else None
    if archive_version:
        filename = f"pilot_{x}_0.npz" if archive_version in (1, 3) and y == 0 else f"pilot_{x}_{y}.npz"
        with np.load(OUT.parent / f"version_{archive_version}" / filename) as saved:
            previous = saved["history"]
        previous_q = previous[:, 10:19] if archive_version == 6 else previous[:, 1:10]
        difference = float(np.max(abs(np.asarray(joint_states)[:len(previous_q)] - previous_q)))
        result["previous_joint_state_prefix_max_difference"] = difference
        assert difference < 1e-10, difference
    name = f"{solver}_{label}_{x}_{y}"
    np.savez_compressed(OUT / f"{name}.npz", trace=records, qpos=np.asarray(joint_states))
    print(json.dumps({k: result[k] for k in ("solver", "intervention", "goal_xy_m", "final_pose_deg", "final_position_mm", "last_2s_conditions_all_pass")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("hierarchy", "full"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base_gain = 2.5 if args.solver == "hierarchy" else 6.0
    configurations = [
        ("original", base_gain, False, False),
        ("angular_only", 10.0, False, False),
        ("servo_only", base_gain, True, False),
        ("gravity_only", base_gain, False, True),
        ("gravity_servo", base_gain, True, True),
        ("all_three", 10.0, True, True),
    ]
    results = [run(args.solver, *config, x, y) for x, y in ((0.7, 0.0), (0.65, 0.3)) for config in configurations]
    (OUT / f"{args.solver}_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / f"{args.solver}_plan.json").write_text(json.dumps({
        "solver": args.solver, "maximum_duration_s": 16, "continue_after_qualification": True,
        "purpose": "isolated causal probes at two recorded goals; not a workspace rescan",
        "interventions": configurations,
        "recording_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
