"""Render the four DynaDLO core motion scenarios for paper Figure 2.

Each panel uses a seed-paired MuJoCo rollout of the same cable model. For
visual comparison, the displayed current cable centroids are translated to
the position of panel (b) after simulation; relative motion is preserved.
This figure illustrates conditions, not grasp outcomes. Run from repo root:

    $env:MUJOCO_GL = 'wgl'
    conda run -n mujoco-dlo python tools/figures/render_paper_fig2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.scenarios.registry import SCENARIO_REGISTRY

OUTPUT = Path(__file__).resolve().parent / "renders"
SEED = 20280804
FRAMES = 150
RENDER_W, RENDER_H = 1000, 750
CROP_BOX = (0, 130, 1000, 730)
PANEL_W, PANEL_H = 1000, 600
GHOST_FRAME = 100
SHAPE_SAMPLES_PER_SEGMENT = 5
SCENARIOS = (
    ("a", "id_static", "Static", "Global motion  −     Deformation  −"),
    ("b", "id_rigid_l1_nominal", "Global motion", "Global motion  +     Deformation  −"),
    ("c", "id_shape_nominal_current", "Shape deformation", "Global motion  −     Deformation  +"),
    ("d", "id_combined_l1_nominal", "Combined motion", "Global motion  +     Deformation  +"),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size)


def make_wood_table(model: mujoco.MjModel) -> None:
    """Expand the tabletop and recolor its existing texture as light oak."""
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    texture = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "table_tex")
    material = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "table_mat")
    if min(table, texture, material) < 0:
        raise RuntimeError("figure requires table, table_tex, and table_mat")
    model.geom_size[table, :2] = (8.0, 8.0)
    height, width = int(model.tex_height[texture]), int(model.tex_width[texture])
    rows, cols = np.mgrid[:height, :width]
    grain = (2.2 * np.sin(rows * 0.47 + 1.8 * np.sin(cols * 0.032))
             + 1.1 * np.sin(rows * 1.31 + 0.9 * np.sin(cols * 0.071)))
    plank = ((rows // 64) % 2) * 2.0
    seam = np.where(rows % 64 < 2, -5.0, 0.0)
    tone = grain + plank + seam
    base = np.array([218.0, 183.0, 143.0])
    rgb = np.clip(base + tone[..., None], 0, 255).astype(np.uint8)
    adr = int(model.tex_adr[texture])
    size = height * width * 3
    model.tex_data[adr:adr + size] = rgb.ravel()
    model.mat_texrepeat[material] = (8.0, 8.0)


def smooth_centerline(points: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """Interpolate the sampled cable centerline for display only."""
    points = np.asarray(points, dtype=float)
    extended = np.vstack((2 * points[0] - points[1], points,
                          2 * points[-1] - points[-2]))
    curve = []
    for index in range(len(points) - 1):
        p0, p1, p2, p3 = extended[index:index + 4]
        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            curve.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    curve.append(points[-1])
    return np.asarray(curve)


def render_condition(name: str, target_center_xy: np.ndarray | None = None) -> tuple[Image.Image, np.ndarray]:
    scenario = SCENARIO_REGISTRY[name]
    cfg = EnvConfig(
        robot="nero", seed=SEED, dynamicvla_cameras_enabled=True,
        target_selection="middle",
        scenario_name=scenario.name,
        scenario_id=scenario.scenario_id, scenario_split=scenario.split.value,
        **scenario.to_env_overrides(),
    )
    env = CableGraspEnv(cfg)
    env.reset(seed=SEED)
    held_action = env.ready_ctrl.copy()
    initial_robot_qpos = env.data.qpos[:9].copy()
    initial_robot_qvel = env.data.qvel[:9].copy()
    cable_bodies = [
        i for i in range(env.model.nbody)
        if (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("cableB")
    ]
    earlier_cable = None
    for frame in range(1, FRAMES + 1):
        env.step(held_action)
        # Freeze the actual reset pose, including both fingers, while cable
        # physics and the scenario disturbance continue to evolve.
        env.data.qpos[:9] = initial_robot_qpos
        env.data.qvel[:9] = initial_robot_qvel
        mujoco.mj_forward(env.model, env.data)
        if frame == GHOST_FRAME:
            earlier_cable = env.data.xpos[cable_bodies].copy()
    model, data = env.model, env.data
    current_center_xy = data.xpos[cable_bodies, :2].mean(axis=0).copy()
    if target_center_xy is not None:
        # Align only the displayed cable, after the physical rollout. Its
        # shape and the earlier-to-current displacement remain unchanged.
        shift_xy = np.asarray(target_center_xy) - current_center_xy
        root_joint = int(model.body_jntadr[cable_bodies[0]])
        if int(model.jnt_type[root_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError("cable root must have a free joint")
        root_qpos = int(model.jnt_qposadr[root_joint])
        data.qpos[root_qpos:root_qpos + 2] += shift_xy
        earlier_cable[:, :2] += shift_xy
        mujoco.mj_forward(model, data)

    marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cable_marker")
    if marker >= 0:
        model.site_rgba[marker, 3] = 0.0
    make_wood_table(model)
    # Use an unshadowed, non-reflective render for the paper illustration.
    # Keep the experiment's checker texture and object colors.
    model.mat_reflectance[:] = 0.0
    model.mat_specular[:] = 0.0
    model.mat_shininess[:] = 0.0

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), RENDER_W)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), RENDER_H)
    with mujoco.Renderer(model, height=RENDER_H, width=RENDER_W) as renderer:
        renderer.update_scene(data, camera=cfg.dynamicvla_opst_camera_name)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        scene = renderer.scene
        if name == "id_shape_nominal_current":
            # Panel (c) bends sharply enough that the original capsule chain
            # reads as separate segments. Draw both states at 5x the spatial
            # sampling density; this changes only the displayed contour.
            cable_color = None
            cable_radius = None
            for geom in scene.geoms[:scene.ngeom]:
                if int(geom.objtype) != int(mujoco.mjtObj.mjOBJ_GEOM):
                    continue
                geom_id = int(geom.objid)
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                if geom_name.startswith("cableG"):
                    cable_color = geom.rgba.copy()
                    cable_radius = float(model.geom_size[geom_id, 0])
                    geom.rgba[3] = 0.0
            if cable_color is None or cable_radius is None:
                raise RuntimeError("no cable geometry found for panel (c)")
            current_curve = smooth_centerline(data.xpos[cable_bodies], SHAPE_SAMPLES_PER_SEGMENT)
            if scene.ngeom + len(current_curve) - 1 > scene.maxgeom:
                raise RuntimeError("MuJoCo render scene has insufficient room for smoothed cable")
            for start, end in zip(current_curve[:-1], current_curve[1:]):
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                    np.eye(3).ravel(), cable_color,
                )
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_CAPSULE, cable_radius,
                    np.asarray(start, dtype=float), np.asarray(end, dtype=float),
                )
                scene.ngeom += 1
        # The environment visualizer supplies the experiment's blue cable.
        # One translucent earlier contour encodes motion without an arrow or
        # a separate timeline. Static has no ghost because it has no driven
        # motion. The contour is a sampled MuJoCo cable state, not artwork.
        if scenario.motion_type.value != "static":
            ghost_curve = (smooth_centerline(earlier_cable, SHAPE_SAMPLES_PER_SEGMENT)
                           if name == "id_shape_nominal_current" else earlier_cable)
            if scene.ngeom + len(ghost_curve) - 1 > scene.maxgeom:
                raise RuntimeError("MuJoCo render scene has insufficient room for cable ghost")
            for start, end in zip(ghost_curve[:-1], ghost_curve[1:]):
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                    np.eye(3).ravel(), np.array([0.57, 0.83, 0.96, 0.72], dtype=np.float32),
                )
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_LINE, 4.0,
                    np.asarray(start, dtype=float) + [0, 0, 0.023],
                    np.asarray(end, dtype=float) + [0, 0, 0.023],
                )
                scene.ngeom += 1
        rgb = renderer.render().copy()
    robot_drift = float(np.max(np.abs(data.qpos[:9] - initial_robot_qpos)))
    print(name, "time", round(float(data.time), 2), "s", "robot drift", round(robot_drift, 6),
          "display center", data.xpos[cable_bodies, :2].mean(axis=0).round(4).tolist())
    return Image.fromarray(rgb).crop(CROP_BOX), current_center_xy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[item[0] for item in SCENARIOS])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scene_images = {}
    reference_letter, reference_name = SCENARIOS[1][:2]
    reference_image, reference_center = render_condition(reference_name)
    if args.only in (None, reference_letter):
        reference_image.save(OUTPUT / f"fig2_{reference_letter}_{reference_name}.png")
        scene_images[reference_letter] = reference_image
    for letter, name, _, _ in SCENARIOS:
        if letter == reference_letter:
            continue
        if args.only and letter != args.only:
            continue
        image, _ = render_condition(name, reference_center)
        image.save(OUTPUT / f"fig2_{letter}_{name}.png")
        scene_images[letter] = image
    if args.only:
        return

    margin, gap, header, footer = 38, 22, 0, 80
    width = 2 * PANEL_W + gap + 2 * margin
    height = 2 * (PANEL_H + footer) + gap + 2 * margin + header + 25
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    label_font, detail_font = font(27, True), font(21)
    for index, (letter, name, title, factors) in enumerate(SCENARIOS):
        row, col = divmod(index, 2)
        x = margin + col * (PANEL_W + gap)
        y = margin + header + row * (PANEL_H + footer + gap)
        canvas.paste(scene_images[letter], (x, y))
        draw.rectangle((x, y, x + PANEL_W - 1, y + PANEL_H - 1), outline="#c8d0d6", width=2)
        draw.text((x + 3, y + PANEL_H + 12), f"({letter})  {title}", font=label_font, fill="#182c3c")
        draw.text((x + 3, y + PANEL_H + 47), factors, font=detail_font, fill="#526270")
    draw.text((margin, height - 37), "Current cable centers aligned to (b)    Pale blue: 2.0 s    Dark blue: 3.0 s",
              font=font(20), fill="#526270")
    png = OUTPUT / "fig2_four_motion_scenarios.png"
    pdf = OUTPUT / "fig2_four_motion_scenarios.pdf"
    canvas.save(png)
    canvas.save(pdf, resolution=300)
    print("wrote", png, pdf)


if __name__ == "__main__":
    main()
