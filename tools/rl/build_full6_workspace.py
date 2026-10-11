"""Actual-physics workspace scan for the new PPO execution profile."""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

RUN = Path(os.environ.get("FULL6_PPO_RUN", Path(__file__).resolve().parent))
sys.path.insert(0, str(RUN / "frozen_code/src"))
import mujoco
import numpy as np
from scipy import ndimage
from panda_cable_grasp.env.environment import EnvConfig
from panda_cable_grasp.env.kinematics import quat_error, rotation_to_quat
from full6_workspace_env import EnhancedCableEnv, PROFILE, solve_full6
from render_vertical_workspace import Workspace

OUT = RUN / "workspace"
RIG = None


class ScanRig:
    def __init__(self):
        self.geometry = Workspace()
        self.base = EnhancedCableEnv(EnvConfig(robot="nero", motion_mode="static",
            disturbance_strength=0., frame_skip=10, episode_seconds=10000.,
            geometric_safety_enabled=False, arm_acceleration_limit_enabled=False,
            hand_cartesian_velocity_limit_enabled=True, hand_linear_velocity_limit=.9,
            hand_angular_velocity_limit=2.))
        b = self.base
        original = b.model
        self.model = self.geometry.model
        for name in (*b.robot_spec.arm_joint_names, *b.robot_spec.finger_joint_names):
            a, c = original.joint(name).id, self.model.joint(name).id
            assert a == c
            for attr in ("jnt_range", "jnt_axis", "jnt_pos"):
                assert np.allclose(getattr(original, attr)[a], getattr(self.model, attr)[c])
        for attr in ("actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange"):
            getattr(self.model, attr)[:] = getattr(original, attr)
        b.model, b.data = self.model, self.geometry.data
        self.data = b.data
        b._hand_jacp = np.zeros((3, self.model.nv))
        b._hand_jacr = np.zeros((3, self.model.nv))
        self.desired_quat = rotation_to_quat(self.geometry.rotation())

    def trial(self, x, y, yaw=0.):
        b, model, data = self.base, self.model, self.data
        mujoco.mj_resetData(model, data)
        data.qpos[b.arm_qpos_adr] = b.ready_qpos[:7]
        data.qpos[b.finger_qpos_adr] = b.robot_spec.ready_finger_qpos
        data.ctrl[:] = b.ready_ctrl
        mujoco.mj_forward(model, data)
        b._previous_arm_command_velocity[:] = 0
        b._last_applied_action[:] = b.ready_ctrl
        b._last_requested_action[:] = b.ready_ctrl
        b.ff_peak[:] = 0
        b.total_torque_peak[:] = 0
        target = np.array([x, y, .005])
        reference = b.hand_position.copy()
        approach = np.array([x, y, .10])
        quat = rotation_to_quat(self.geometry.rotation(yaw))
        held = 0
        descending = False
        max_pose, peak_ratio, min_floor = 0., 0., 1.
        reason = "timeout"
        endpoint = False
        started = time.monotonic()
        for step in range(800):
            delta = (target if descending else approach) - reference
            reference += delta * min(1., (.03 if descending else .15) * .02 / max(np.linalg.norm(delta), 1e-12))
            if not descending and np.linalg.norm(reference - approach) < 1e-9:
                descending = True
            error = quat_error(rotation_to_quat(data.xmat[b.hand_id].reshape(3, 3)), quat)
            qtarget, _, _, _, _ = solve_full6(b, 6. * (reference - b.hand_position), 10. * error)
            requested = np.r_[qtarget, b.gripper_open_ctrl]
            applied = b._limit_robot_action(np.clip(requested, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]))
            for _ in range(10):
                guarded, _ = b._velocity_guarded_action(applied)
                data.ctrl[:] = guarded
                b._physics_step()
                peak_ratio = max(peak_ratio, float(np.max(np.abs(data.qvel[b.arm_dof_adr]) / b._arm_velocity_limits)))
            mujoco.mj_forward(model, data)
            error = quat_error(rotation_to_quat(data.xmat[b.hand_id].reshape(3, 3)), quat)
            pose = float(np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(error) / 2, 0, 1))))
            position = float(np.linalg.norm(b.hand_position - target))
            max_pose = max(max_pose, pose)
            floor = float(self.geometry.floor_clearance())
            min_floor = min(min_floor, floor)
            finger_floor = min(float(self.geometry.geom_vertices(g)[:, 2].min()) for g in self.geometry.finger_geoms)
            q = data.qpos[b.arm_qpos_adr]
            ranges = model.jnt_range[b.arm_joint_ids]
            margin = float(np.min(np.minimum(q - ranges[:, 0], ranges[:, 1] - q)))
            collision = self.geometry.self_collision() if step % 5 == 0 or held else False
            ok = position <= .005 and pose <= 2. and .0002 <= finger_floor <= .008 and floor >= .0002 and margin >= 0 and not collision
            held = held + 1 if ok else 0
            if held >= 100:
                endpoint, reason = True, "endpoint_held_2s"
                break
            if floor < -.001:
                reason = "floor_penetration"
                break
            if collision:
                reason = "self_collision"
                break
            if not np.isfinite(data.qpos).all():
                reason = "nonfinite"
                break
        return dict(x_m=x, y_m=y, yaw_rad=yaw, endpoint_pass=endpoint,
                    motion_pass=bool(endpoint and max_pose <= 5. and peak_ratio <= 1.01),
                    reason=reason, steps=step+1, position_error_mm=position*1000,
                    pose_deg=pose, max_path_pose_deg=max_pose, peak_velocity_ratio=peak_ratio,
                    min_floor_mm=min_floor*1000, wall_s=time.monotonic()-started,
                    torque_peak=b.total_torque_peak.tolist(), gravity_peak=b.ff_peak.tolist())


def worker(job):
    global RIG
    if RIG is None:
        RIG = ScanRig()
    return RIG.trial(*job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        rig = ScanRig()
        rows = [rig.trial(x, y, yaw) for x, y in ((.7, 0.), (.65, .3)) for yaw in (0., -.35, .35)]
        (OUT / "pilot.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(rows, indent=2), flush=True)
        return
    xs = np.round(np.arange(-.25, .951, .04), 6)
    ys = np.round(np.arange(-.6, .601, .04), 6)
    file = OUT / "cells.jsonl"
    rows = [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
    done = {(r["x_m"], r["y_m"]) for r in rows}
    jobs = [(float(x), float(y)) for y in ys for x in xs if (float(x), float(y)) not in done]
    start = time.monotonic()
    with file.open("a") as out, mp.get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(worker, jobs):
            rows.append(row)
            out.write(json.dumps(row)+"\n")
            out.flush()
            progress = dict(completed=len(rows), total=len(xs)*len(ys),
                            endpoint=sum(r["endpoint_pass"] for r in rows),
                            motion=sum(r["motion_pass"] for r in rows), wall_s=time.monotonic()-start)
            (OUT / "progress.json").write_text(json.dumps(progress))
            if len(rows) % 25 == 0:
                print(json.dumps(progress), flush=True)
    masks = {}
    for key in ("endpoint_pass", "motion_pass"):
        mask = np.zeros((len(ys), len(xs)), dtype=bool)
        for row in rows:
            mask[np.argmin(abs(ys-row["y_m"])), np.argmin(abs(xs-row["x_m"]))] = row[key]
        masks[key] = mask
    masks["interior"] = masks["motion_pass"] & (ndimage.distance_transform_edt(np.pad(masks["motion_pass"], 1))[1:-1, 1:-1] * .04 >= .06)
    if not np.any(masks["interior"]):
        raise RuntimeError("No measured interior; do not start training")
    np.savez_compressed(OUT / "workspace.npz", xs=xs, ys=ys, **masks)
    manifest = dict(profile=PROFILE, points=len(rows), counts={k:int(v.sum()) for k,v in masks.items()},
                    z_m=.005, fixed_yaw_rad=0., grid_step_m=.04, interior_margin_m=.06,
                    control_dt_s=.02, duration_limit_s=16., endpoint_hold_s=2.,
                    position_tolerance_m=.005, pose_tolerance_deg=2., path_pose_limit_deg=5.,
                    scope="actual unloaded from-home fixed-yaw near-floor screening mask; not guaranteed graspability at all yaw/heights",
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), RUN/"full6_workspace_env.py", RUN/"render_vertical_workspace.py")})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
