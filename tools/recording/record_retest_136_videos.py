"""Fresh physical reruns, with native 1280x960/50 Hz global and wrist capture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
WORKTREE = ROOT.parent / "worktrees/scripted-tracking-20261001"
OUT = ROOT / "outputs/vertical_workspace/scripted_136_retest_20261004/videos"
sys.path.insert(0, str(WORKTREE / "tools/recording"))
from human_episode_recording import RGBEncoder


def create_rig(version, directory):
    if version == 6:
        sys.path.insert(0, str(WORKTREE / "src"))
        import record_standard_jik_workspace as native
        rig = native.Rig(cartesian_linear_velocity_limit=0.90)
        return rig, rig.env, native
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools/recording"))
    import record_improved_workspace as native
    native.OUT = directory / "audit"
    native.VARIANT = "baseline" if version == 1 else "tracking_fix_orientation4"
    rig = native.ImprovedRig()
    rig.audit.update(version_number=version, arm_kp_scale=1.0 if version == 1 else 4.0,
                     arm_kv_scale=1.0 if version == 1 else 2.0,
                     orientation_error_scale=1.0 if version == 1 else 4.0,
                     gravity_feedforward="disabled" if version == 1 else "bounded at 500 Hz")
    return rig, rig.base, native


def capture(version, x, y, name, *, rig_factory=None, output_root=None, scan_directory=None, label=None):
    directory = (output_root or OUT) / f"version_{version}" / name
    directory.mkdir(parents=True, exist_ok=True)
    rig, env, native = (rig_factory(directory) if rig_factory else create_rig(version, directory))
    previous_file = (scan_directory or (ROOT / "outputs/vertical_workspace/scripted_136_retest_20261004" / f"version_{version}")) / (
        f"pilot_{x}_0.npz" if version in (1, 3) and y == 0 else f"pilot_{x}_{y}.npz")
    comparison_source = str(previous_file)
    if previous_file.exists():
        with np.load(previous_file) as previous:
            old = previous["history"]
        old_q = old[:, 10:19] if version == 6 else old[:, 1:10]
    else:
        reference_q = []
        unrecorded_control = rig.control
        def reference_control(*args):
            result = unrecorded_control(*args)
            reference_q.append(rig.data.qpos[:9].copy())
            return result
        rig.control = reference_control
        reference_result = rig.trial(x, y)
        reference_row = reference_result if version == 6 else reference_result[0]
        rig.control = unrecorded_control
        old_q = np.asarray(reference_q)
        np.savez_compressed(directory / 'unrendered_reference.npz', qpos=old_q)
        comparison_source = str(directory / 'unrendered_reference.npz')
        archived_file = previous_file.parent / 'cells.json'
        if archived_file.exists():
            archived = json.loads(archived_file.read_text())
        else:
            # Completed rows are flushed individually during an ongoing scan.
            archived = [json.loads(line) for line in
                        (previous_file.parent / 'cells.jsonl').read_text().splitlines()]
        scan_row = next(r for r in archived if abs(r['x_m']-x)<1e-8 and abs(r['y_m']-y)<1e-8)
        assert reference_row['reason'] == scan_row['reason'] and reference_row['steps'] == scan_row['steps']
        for key in ('robot_floor_mm', 'finger_floor_mm', 'position_error_mm', 'pose_error_deg'):
            assert abs(reference_row[key] - scan_row[key]) < 1e-8
    model, data = rig.model, rig.data
    model.vis.global_.offwidth = max(1280, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(960, model.vis.global_.offheight)
    renderer = mujoco.Renderer(model, height=960, width=1280, max_geom=10000)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.48, 0.04, 0.15)
    camera.distance = 1.55
    camera.azimuth = 135
    camera.elevation = -32
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 21)
    writers = {view: RGBEncoder(directory / f"{view}.mp4", 1280, 960, 50)
               for view in ("global", "wrist")}
    local_rotation = np.empty(9)
    mujoco.mju_quat2Mat(local_rotation, np.asarray(env.config.dynamicvla_wrist_camera_quat))
    local_rotation = local_rotation.reshape(3, 3)
    local_position = np.asarray(env.config.dynamicvla_wrist_camera_pos)
    target = np.array([x, y, 0.005])
    states, references = [], []
    actual_path = []
    start = time.monotonic()
    base_geom = next(g for g in rig.geometry.visual_geoms if model.mesh(int(model.geom_dataid[g])).name == 'gripper_base')

    def frame(reference):
        states.append(np.r_[data.time, data.qpos[:9], data.qvel[:9], data.ctrl])
        references.append(np.asarray(reference).copy())
        actual_path.append(env.hand_position.copy())
        position_error = float(np.linalg.norm(env.hand_position - target) * 1000)
        pose_error = rig.measure(target)[2]
        base_vertices = rig.geometry.geom_vertices(base_geom)
        base_point = base_vertices[np.argmin(base_vertices[:, 2])]
        finger_points = [v[np.argmin(v[:, 2])] for v in (rig.geometry.geom_vertices(g) for g in rig.geometry.finger_geoms)]
        finger_point = min(finger_points, key=lambda p: p[2])
        for view, writer in writers.items():
            renderer.update_scene(data, camera=camera, scene_option=option)
            if view == "wrist":
                hand_rotation = data.xmat[env.hand_id].reshape(3, 3)
                position = data.xpos[env.hand_id] + hand_rotation @ local_position
                rotation = hand_rotation @ local_rotation
                for scene_camera in renderer.scene.camera:
                    scene_camera.pos[:] = position
                    scene_camera.forward[:] = -rotation[:, 2]
                    scene_camera.up[:] = rotation[:, 1]
            # The yellow target is a visual marker, without collision geometry.
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.006, 0, 0]),
                              target, np.eye(3).ravel(), np.array([1.0, 0.72, 0.04, 1.0]))
            renderer.scene.ngeom += 1
            for point in (base_point, finger_point):
                if point[2] >= 0:
                    continue
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.005, 0, 0]),
                    point + np.array([0, 0, .01]), np.eye(3).ravel(), np.array([1, .08, .06, 1]))
                renderer.scene.ngeom += 1
            rgb = renderer.render()
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((18, 16, 1262, 124), radius=10, fill="#f6f9fc")
            draw.text((34, 23), f"{label or f'版本 {version}'} · {'全局' if view == 'global' else '腕部'}视角 · 实际物理重跑 · 1×", font=font, fill="#21354a")
            draw.text((34, 65), f"目标 ({x:.2f}, {y:+.2f}, 0.005) m  |  t={data.time:.2f}s  |  目标位置误差 {position_error:.1f} mm  |  姿态误差 {pose_error:.1f}°", font=small, fill="#42566b")
            draw.text((34, 96), f"底座离桌 {base_point[2]*1000:+.2f} mm  |  夹指离桌 {finger_point[2]*1000:+.2f} mm  |  负值＝穿入；红点标示位置", font=small, fill="#42566b")
            array = np.asarray(image)
            writer.write(array)
            if len(states) == 1:
                image.save(directory / f"{view}_poster.png")
        if len(states) % 100 == 0:
            print(f"version={version} goal={name} frames={len(states)} sim={data.time:.2f}s wall={time.monotonic()-start:.1f}s", flush=True)

    original_control = rig.control
    if version == 6:
        def recorded_control(reference, phase):
            original_control(reference, phase)
            frame(reference)
    else:
        def recorded_control(reference):
            original_control(reference)
            frame(reference)
    rig.control = recorded_control
    rig.reset()
    frame(env.hand_position)
    try:
        result = rig.trial(x, y)
        row = result if version == 6 else result[0]
    finally:
        for writer in writers.values():
            writer.release()
        renderer.close()
    states = np.asarray(states)
    np.savez_compressed(directory / "states.npz", states=states,
                        reference=np.asarray(references), actual_tcp=np.asarray(actual_path),
                        frame_index=np.arange(len(states)), frame_time=states[:, 0])
    assert len(old_q) == len(states) - 1, (len(old_q), len(states))
    difference = float(np.max(np.abs(old_q - states[1:, 1:10])))
    assert difference <= 1e-12, difference
    metadata = {
        "version": version, "goal_name": name, "goal_m": target.tolist(), "result": row,
        "label": label or f'版本 {version}',
        "capture": "fresh physical rerun; independent renderer after every actual 20 ms control state",
        "width": 1280, "height": 960, "fps": 50, "frames": len(states),
        "simulation_duration_s": float(states[-1, 0]), "playback_speed": 1,
        "codec": "H.264", "crf": 16, "gop": 12, "b_frames": 0,
        "previous_saved_joint_state_max_difference": difference,
        "joint_state_comparison_source": comparison_source,
        "controller_audit": rig.audit, "native_controller_file": str(Path(native.__file__).resolve()),
        "native_controller_file_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "recording_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wrist_camera": {"local_position": local_position.tolist(),
                         "local_rotation": local_rotation.tolist(), "attached_body": env.robot_spec.hand_body_name},
    }
    from diagnose_retest_floor import floor_details
    metadata['final_floor_details'] = floor_details(rig, env)
    if scan_directory:
        metadata['scan_manifest'] = str(scan_directory / 'manifest.json')
        metadata['scan_manifest_sha256'] = hashlib.sha256((scan_directory / 'manifest.json').read_bytes()).hexdigest()
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"version": version, "goal": name, "frames": len(states), "motion_pass": row["motion_pass"], "trajectory_difference": difference}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, choices=(1, 3, 6), required=True)
    parser.add_argument('--failures', action='store_true')
    args = parser.parse_args()
    goals = ((.55, 0., 'body_floor'), (.51, -.04, 'finger_floor_v1'), (.47, .16, 'finger_floor')) if args.failures else ((0.7, 0.0, "center"), (0.65, 0.3, "positive_y"))
    for x, y, name in goals:
        capture(args.version, x, y, name)
