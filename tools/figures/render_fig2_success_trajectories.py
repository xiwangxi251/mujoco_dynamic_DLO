"""Render three views from one recorded successful trial in each Fig. 2 scenario.

The MuJoCo states come from a completed scripted-policy experiment. Camera,
and cable geometry are replayed from those states. For the first view in the
rigid and combined scenarios, the recorded pre-contact cable state is paired
with the common robot reset pose so the full cable is visible. The caption
explicitly marks this visualization choice. Table appearance is simplified.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_paper_fig2 import make_wood_table

RUN = ROOT / "outputs" / "nero_ablation_2x2_video_20260912_both_on"
MODELS = ROOT / "outputs" / "scripted" / "nero_core_4scenes_5trials_20260912" / "models"
OUT = Path(__file__).resolve().parent / "renders"
SCENARIOS = (
    ("a", "id_static", "Static", 20280804, (0.00, 1.50, 4.36)),
    ("b", "id_rigid_l1_nominal", "Rigid motion", 20280804, (2.20, 3.20, 6.92)),
    ("c", "id_shape_nominal_current", "Shape deformation", 20280805, (0.00, 1.44, 13.52)),
    ("d", "id_combined_l1_nominal", "Combined motion", 20280806, (2.00, 3.28, 6.48)),
)
CAMERA = "dynamicvla_opst_camera"
RENDER_W, RENDER_H = 1000, 750
CROP = (120, 140, 880, 720)
PANEL_W, PANEL_H = 760, 580
GHOST_VIDEO_FRAMES = {
    "id_shape_nominal_current": 25,
    "id_combined_l1_nominal": 62,
}
CURRENT_VIDEO_FRAMES = {
    "id_shape_nominal_current": 36,
    "id_combined_l1_nominal": 82,
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def load_recordings() -> dict[tuple[str, int], dict[str, str]]:
    with (RUN / "episodes.csv").open(newline="", encoding="utf-8-sig") as handle:
        return {(row["scenario"], int(row["seed"])): row for row in csv.DictReader(handle)}


def render_scenario(name: str, seed: int, times: tuple[float, ...], record: dict[str, str]):
    if record["success"].lower() != "true" or record["outcome"] != "success":
        raise ValueError(f"recorded trial is not a success: {name}/{seed}")
    trajectory = Path(record["trajectory"])
    if not trajectory.is_absolute():
        trajectory = RUN / trajectory
    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)
    with np.load(trajectory, allow_pickle=False) as saved:
        states = saved["states"].copy()
        state_times = saved["state_times"].copy()
        frame_state_indices = saved["frame_state_indices"].copy()
        state_spec = int(saved["state_spec"])
    model = mujoco.MjModel.from_binary_path(str(MODELS / f"{name}.mjb"))
    # These reusable compiled models are from the earlier core run, where the
    # robot base was at x=0.20 m. The recorded success trials were calibrated
    # with base_x=0.35 m; full-physics states do not store fixed body offsets.
    # Restore the trial's model-space base position before state replay.
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0:
        raise ValueError(f"missing robot base in {name} model")
    model.body_pos[base_id, 0] = float(record["base_x"])
    expected_size = mujoco.mj_stateSize(model, state_spec)
    if states.shape[1] != expected_size:
        raise ValueError(f"model does not match saved states for {name}")
    data = mujoco.MjData(model)
    if not 0.0 <= times[0] < times[1] < times[2]:
        raise ValueError("figure times must be strictly increasing")
    mujoco.mj_setState(model, data, states[0], state_spec)
    initial_robot_qpos = data.qpos[:9].copy()

    # Figure 2 compares the scenario-driven cable motion before the gripper
    # affects it. Check the complete history through the second frame, rather
    # than only checking that frame for a momentary gap in contact.
    second_index = (int(frame_state_indices[CURRENT_VIDEO_FRAMES[name]])
                    if name in CURRENT_VIDEO_FRAMES
                    else int(np.argmin(np.abs(state_times - times[1]))))
    finger_geoms = {
        geom_id for geom_id in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                             int(model.geom_bodyid[geom_id])) in ("gripper_link1", "gripper_link2")
        and model.geom_contype[geom_id]
    }
    cable_geoms = {
        geom_id for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("cableG")
    }

    def gripper_cable_contact() -> bool:
        return any(
            (contact.geom1 in finger_geoms and contact.geom2 in cable_geoms)
            or (contact.geom2 in finger_geoms and contact.geom1 in cable_geoms)
            for contact in data.contact
        )

    for index in range(second_index + 1):
        mujoco.mj_setState(model, data, states[index], state_spec)
        mujoco.mj_forward(model, data)
        if gripper_cable_contact():
            raise ValueError(f"second frame follows gripper-cable contact in {name} at {state_times[index]:.2f}s")

    make_wood_table(model)
    # A uniform wood tone avoids moire from repeating texture grain at this
    # camera distance while retaining the prior figure's table color.
    table_texture = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "table_tex")
    texture_adr = int(model.tex_adr[table_texture])
    texture_size = int(model.tex_height[table_texture] * model.tex_width[table_texture] * 3)
    model.tex_data[texture_adr:texture_adr + texture_size] = np.tile(
        np.array([218, 183, 143], dtype=np.uint8), texture_size // 3
    )
    model.mat_reflectance[:] = 0.0
    model.mat_specular[:] = 0.0
    model.mat_shininess[:] = 0.0
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), RENDER_W)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), RENDER_H)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA) < 0:
        raise ValueError(f"missing {CAMERA} in {name} model")
    marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cable_marker")
    if marker >= 0:
        model.site_rgba[marker, 3] = 0.0
    cable_bodies = [
        body_id for body_id in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").startswith("cableB")
    ]

    output = []
    earlier_cable = None
    recorded_ghost = None
    if name in GHOST_VIDEO_FRAMES:
        ghost_index = int(frame_state_indices[GHOST_VIDEO_FRAMES[name]])
        if not state_times[ghost_index] < times[1]:
            raise ValueError("cable ghost must precede the second frame")
        mujoco.mj_setState(model, data, states[ghost_index], state_spec)
        mujoco.mj_forward(model, data)
        recorded_ghost = data.xpos[cable_bodies].copy()
    with mujoco.Renderer(model, height=RENDER_H, width=RENDER_W) as renderer:
        for col, target_time in enumerate(times):
            index = (second_index if col == 1 and name in CURRENT_VIDEO_FRAMES
                     else int(np.argmin(np.abs(state_times - target_time))))
            mujoco.mj_setState(model, data, states[index], state_spec)
            if col == 0 and target_time > 0.0:
                # Display-only robot pose. Cable coordinates remain the exact
                # pre-contact saved state from this successful trajectory.
                data.qpos[:9] = initial_robot_qpos
                data.qvel[:9] = 0.0
            mujoco.mj_forward(model, data)
            if col == 0 and gripper_cable_contact():
                raise ValueError(f"display reset pose contacts cable in {name}")
            if col == 2 and not gripper_cable_contact():
                raise ValueError(f"third frame does not show grasp contact in {name}")
            if col == 0:
                earlier_cable = data.xpos[cable_bodies].copy()
            renderer.update_scene(data, camera=CAMERA)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            if col == 1 and name != "id_static":
                if earlier_cable is None:
                    raise RuntimeError("missing earlier cable state")
                ghost_cable = recorded_ghost if recorded_ghost is not None else earlier_cable
                scene = renderer.scene
                if scene.ngeom + len(ghost_cable) - 1 > scene.maxgeom:
                    raise RuntimeError("render scene has insufficient room for cable ghost")
                for start, end in zip(ghost_cable[:-1], ghost_cable[1:]):
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.array([0.35, 0.67, 0.83, 0.78], dtype=np.float32),
                    )
                    mujoco.mjv_connector(
                        geom, mujoco.mjtGeom.mjGEOM_LINE, 5.0,
                        np.asarray(start, dtype=float) + (0.0, 0.0, 0.024),
                        np.asarray(end, dtype=float) + (0.0, 0.0, 0.024),
                    )
                    scene.ngeom += 1
            rgb = renderer.render().copy()
            frame = Image.fromarray(rgb).crop(CROP)
            output.append((frame, float(state_times[index]), index))
    return output, initial_robot_qpos


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_recordings()
    edge = 35
    col_gap = 20
    row_gap = 30
    title_h = 84
    row_h = title_h + PANEL_H + 48
    width = 2 * edge + 3 * PANEL_W + 2 * col_gap
    height = 2 * edge + 4 * row_h + 3 * row_gap + 85
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    metadata = []
    reference_robot_qpos = None
    for row_i, (letter, name, title, seed, times) in enumerate(SCENARIOS):
        row = records[name, seed]
        frames, initial_robot_qpos = render_scenario(name, seed, times, row)
        if reference_robot_qpos is None:
            reference_robot_qpos = initial_robot_qpos
        elif not np.allclose(initial_robot_qpos, reference_robot_qpos, atol=1e-9, rtol=0):
            raise ValueError(f"reset robot pose differs in {name}")
        y = edge + row_i * (row_h + row_gap)
        draw.text((edge, y), f"({letter})  {title}", font=font(34, True), fill="#172b3b")
        draw.text((width - edge - 310, y + 7), f"successful trial  ·  seed {seed}", font=font(22), fill="#60717c")
        for col, (frame, actual_time, state_index) in enumerate(frames):
            x = edge + col * (PANEL_W + col_gap)
            top = y + title_h
            canvas.paste(frame, (x, top))
            draw.rectangle((x, top, x + PANEL_W - 1, top + PANEL_H - 1), outline="#c6d0d5", width=2)
            stage = ("Reset arm" if col == 0 and times[0] > 0.0 else "Initial",
                     "Before contact", "Grasp phase")[col]
            draw.text((x + 10, top + PANEL_H + 11), f"t = {actual_time:.2f} s   ·   {stage}",
                      font=font(25), fill="#435867")
            metadata.append({
                "scenario": name, "seed": seed, "state_index": state_index,
                "time_s": actual_time, "base_x_m": float(row["base_x"]),
                "source_video_frame": CURRENT_VIDEO_FRAMES.get(name) if col == 1 else None,
                "ghost_video_frame": GHOST_VIDEO_FRAMES.get(name) if col == 1 else None,
                "arm_reset_for_display": bool(col == 0 and times[0] > 0.0),
                "trajectory": str(Path(row["trajectory"]).resolve()),
                "source_video": str(Path(row["global_video"]).resolve()),
                "model": str((MODELS / f"{name}.mjb").resolve()),
            })
    draw.text((edge, height - 94),
              "Second column: pale blue = earlier cable state (b: first-column time; c: frame 25; d: frame 62).",
              font=font(21), fill="#61717c")
    draw.text((edge, height - 61),
              "(b,d), first column: recorded pre-contact cable state with the robot shown at its common reset pose.",
              font=font(21), fill="#61717c")
    png = OUT / "fig2_success_trajectories_4x3_combined_ghost62.png"
    pdf = OUT / "fig2_success_trajectories_4x3_combined_ghost62.pdf"
    meta = OUT / "fig2_success_trajectories_4x3_combined_ghost62.json"
    canvas.save(png)
    canvas.save(pdf, resolution=300)
    meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("wrote", png, pdf, meta)


if __name__ == "__main__":
    main()
