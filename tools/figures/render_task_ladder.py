"""Render paper-figure task panels with the NERO arm.

Four scenes illustrating the manipulation difficulty ladder:
  1. task_static_cube   -- reaching for a static rigid cube
  2. task_rolling_ball  -- intercepting a rolling ball (ghost trail)
  3. task_routing       -- seating a cable in successive snap clips
  4. task_dynamic_cable -- the real env: grasping a writhing DLO

Outputs PNGs to tools/figures/renders/ (plus a 2x2 composite).
Run:  PYTHONPATH=src python tools/figures/render_task_ladder.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

NERO_XML = ROOT / "assets" / "mujoco" / "nero" / "nero.xml"
NERO_ASSETS = NERO_XML.parent / "assets"
OUT_DIR = Path(__file__).resolve().parent / "renders"

BASE_OFFSET = np.array([0.35, 0.0, 0.0])
GRASP_CENTER_LOCAL = np.array([0.1733, 0.0, -0.0235])
READY_QPOS = np.array(
    [0.546612, -0.327276, 2.381604, 1.360838, 0.217447, -0.193041, 1.539675]
)
CABLE_RGBA = "0.95 0.22 0.035 1"
CABLE_R = 0.014

# Shared scene shell: identical visual style to assets/mujoco/panda_cable_grasp.xml
SCENE_TEMPLATE = """
<mujoco model="figure scene">
  <compiler meshdir="assets" angle="radian" autolimits="true"/>
  <include file="robot.xml"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"
          solver="Newton" cone="elliptic" impratio="20" noslip_iterations="1"/>
  <size njmax="1200" nconmax="600"/>
  <statistic center="0.55 0 0.15" extent="1.6"/>
  <visual>
    <global azimuth="135" elevation="-25"/>
    <quality shadowsize="2048"/>
    <map znear="0.01" zfar="12" fogstart="4" fogend="10"/>
    <rgba haze="0.12 0.18 0.25 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.32 0.52 0.72"
             rgb2="0.02 0.04 0.08" width="512" height="512"/>
    <texture name="table_tex" type="2d" builtin="checker"
             rgb1="0.18 0.20 0.22" rgb2="0.10 0.12 0.14"
             width="256" height="256"/>
    <material name="table_mat" texture="table_tex" texrepeat="6 6"
              texuniform="true" reflectance="0.02"/>
  </asset>
  <worldbody>
    <light directional="true" pos="-2 -3 4" dir="0.3 0.35 -1"
           diffuse="0.95 0.92 0.84" specular="0.3 0.3 0.3"/>
    <light directional="true" pos="2 1 3" dir="-0.35 -0.15 -1"
           diffuse="0.25 0.35 0.45"/>
    <geom name="table" type="box" pos="0.55 0 -0.06" size="1.20 1.20 0.06"
          material="table_mat" friction="0.35 0.05 0.01"/>
    <!--OBJECTS-->
  </worldbody>
</mujoco>
"""


def nero_assets() -> dict[str, bytes]:
    return {
        p.relative_to(NERO_ASSETS).as_posix(): p.read_bytes()
        for p in NERO_ASSETS.rglob("*")
        if p.is_file()
    }


def catmull_rom(points: np.ndarray, n_per_seg: int = 14) -> np.ndarray:
    """Sample a Catmull-Rom spline through control points."""
    p = np.asarray(points, float)
    ext = np.vstack([p[0] - (p[1] - p[0]), p, p[-1] + (p[-1] - p[-2])])
    out = []
    for i in range(len(p) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for t in np.linspace(0.0, 1.0, n_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(
                0.5
                * (
                    (2 * p1)
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
            )
    out.append(p[-1])
    return np.asarray(out)


def cable_geom_xml(pts: np.ndarray, radius: float = CABLE_R, rgba: str = CABLE_RGBA) -> str:
    """Static capsule chain following a polyline (figure-only, no joints)."""
    geoms = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        geoms.append(
            f'<geom type="capsule" fromto="{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} '
            f'{b[0]:.4f} {b[1]:.4f} {b[2]:.4f}" size="{radius}" rgba="{rgba}" '
            f'contype="0" conaffinity="0"/>'
        )
    return "\n".join(geoms)


def build_scene(objects_xml: str):
    xml = SCENE_TEMPLATE.replace("<!--OBJECTS-->", objects_xml)
    spec = mujoco.MjSpec.from_string(
        xml,
        include={"robot.xml": NERO_XML.read_bytes()},
        assets=nero_assets(),
    )
    base = next(b for b in spec.bodies if b.name == "base_link")
    base.pos[:] += BASE_OFFSET
    link7 = next(b for b in spec.bodies if b.name == "link7")
    link7.add_site(name="grasp_center", pos=GRASP_CENTER_LOCAL.tolist())
    return spec.compile()


def arm_joints(model: mujoco.MjModel):
    qadr = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]) for i in range(1, 8)]
    dadr = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]) for i in range(1, 8)]
    return qadr, dadr


def set_fingers(model: mujoco.MjModel, data: mujoco.MjData, aperture_half: float):
    for name, sign in (("gripper_joint1", 1.0), ("gripper_joint2", -1.0)):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[j]] = sign * aperture_half


def solve_ik(model, data, site_id, target_pos, approach_dir,
             rot_weight=0.6, lam=1e-3, iters=600, tol=2e-4) -> float:
    """Damped least squares on the 7 arm joints; aligns local +X to approach_dir."""
    qadr, dadr = arm_joints(model)
    data.qpos[qadr] = READY_QPOS
    target = np.asarray(target_pos, float)
    ad = np.asarray(approach_dir, float)
    ad /= np.linalg.norm(ad)
    lam2 = lam * lam
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        axis = data.site_xmat[site_id].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        err = np.concatenate([
            target - data.site_xpos[site_id],
            rot_weight * np.cross(axis, ad),
        ])
        J = np.vstack([jacp, rot_weight * jacr])[:, dadr]
        if np.linalg.norm(err[:3]) < tol and np.linalg.norm(err[3:]) < 0.02:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), err)
        data.qpos[qadr] += np.clip(dq, -0.25, 0.25)
        for i, qa in enumerate(qadr):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            lo, hi = model.jnt_range[jid]
            data.qpos[qa] = np.clip(data.qpos[qa], lo, hi)
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(target - data.site_xpos[site_id]))


def render(model, data, lookat, distance, azimuth, elevation,
           width=960, height=720, cable_motion=None, hide_robot=False,
           centroid_inset_top=278, motion_kind="cable") -> np.ndarray:
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=cam)
    if cable_motion is not None:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        history, current, current_time = cable_motion[:3]
        spin_phases = cable_motion[3] if len(cable_motion) > 3 else None
        scene = renderer.scene
        # Keep only the current cable in the 3-D scene. A separate timeline
        # below shows earlier shapes without crossing the current contour.
        for geom in scene.geoms[:scene.ngeom]:
            if int(geom.objtype) != int(mujoco.mjtObj.mjOBJ_GEOM):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom.objid)) or ""
            if name.startswith("cableG"):
                geom.rgba[3] = 0.0
            elif hide_robot and name not in {"table", "ball"}:
                geom.rgba[3] = 0.0

        def add_connector(kind, start, end, radius, rgba):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("render scene has no room for motion overlays")
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, kind, np.zeros(3), np.zeros(3),
                np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom, kind, radius, np.asarray(start, dtype=float),
                np.asarray(end, dtype=float),
            )
            scene.ngeom += 1

        current_color = [0.13, 0.74, 0.82, 0.96]
        if motion_kind == "ball":
            for i, (previous, _) in enumerate(history):
                if scene.ngeom >= scene.maxgeom:
                    raise RuntimeError("render scene has no room for ball afterimages")
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.065, 0.065, 0.065]), previous[0],
                    np.eye(3).ravel(),
                    np.array([0.13, 0.74, 0.82, 0.18 + i * 0.12], dtype=np.float32),
                )
                scene.ngeom += 1
        for start, end in zip(current[:-1], current[1:]):
            add_connector(
                mujoco.mjtGeom.mjGEOM_LINE,
                start + [0, 0, 0.023], end + [0, 0, 0.023],
                4.0, current_color,
            )
        if motion_kind == "ball":
            camera = scene.camera[0]
            right = np.cross(camera.forward, camera.up)
            vertical_span = camera.frustum_top - camera.frustum_bottom
            horizontal_span = vertical_span * width / height
            ball_screen_centers = []
            for points, _ in history + [(current, current_time)]:
                offset = points[0] - camera.pos
                depth = float(np.dot(offset, camera.forward))
                ball_screen_centers.append((
                    width * (0.5 + np.dot(offset, right) * camera.frustum_near
                             / (depth * horizontal_span)),
                    height * (0.5 - np.dot(offset, camera.up) * camera.frustum_near
                              / (depth * vertical_span)),
                ))
    img = renderer.render()
    renderer.close()
    if cable_motion is not None:
        if motion_kind == "ball":
            from scipy.ndimage import label

            pixels = np.asarray(img)
            cyan = ((pixels[:449, :, 0] < 120)
                    & (pixels[:449, :, 1] > 125)
                    & (pixels[:449, :, 2] > 130))
            regions, _ = label(cyan)
            counts = np.bincount(regions.ravel())
            large_regions = [i for i in np.flatnonzero(counts > 3000) if i != 0]
            if len(large_regions) == len(history) + 1:
                ball_screen_centers = sorted(
                    [(float(np.where(regions == i)[1].mean()),
                      float(np.where(regions == i)[0].mean()))
                     for i in large_regions],
                    key=lambda point: point[0],
                )
        canvas = Image.fromarray(img).convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if motion_kind == "ball" and spin_phases is not None:
            for i, ((cx, cy), phase) in enumerate(zip(ball_screen_centers, spin_phases)):
                direction = np.array([np.sin(phase), -np.cos(phase)])
                inner = np.array([cx, cy]) + 18 * direction
                outer = np.array([cx, cy]) + 43 * direction
                alpha = 105 + i * 35
                draw.line((*inner, *outer), fill=(19, 78, 91, alpha), width=5)
                draw.ellipse((outer[0] - 5, outer[1] - 5,
                              outer[0] + 5, outer[1] + 5),
                             fill=(19, 78, 91, alpha))
        draw.rounded_rectangle((20, 449, width - 20, height - 17), radius=13,
                               fill=(248, 251, 252, 244),
                               outline=(183, 203, 210, 255), width=1)
        font_path = Path("C:/Windows/Fonts/arial.ttf")
        font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
        small = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
        samples = history + [(current, current_time)]
        centroids = np.array([points[:, :2].mean(axis=0) for points, _ in samples])
        centroid_span = np.maximum(np.ptp(centroids, axis=0), 1e-6)
        centroid_scale = min(166 / centroid_span[0], 67 / centroid_span[1])
        centroid_center = centroids.mean(axis=0)
        path = [
            (809 + float(point[0] - centroid_center[0]) * centroid_scale,
             centroid_inset_top + 88 - float(point[1] - centroid_center[1]) * centroid_scale)
            for point in centroids
        ]
        draw.rounded_rectangle((683, centroid_inset_top, 933, centroid_inset_top + 148), radius=10,
                               fill=(248, 251, 252, 237),
                               outline=(183, 203, 210, 255), width=1)
        draw.text((700, centroid_inset_top + 13), "Centroid motion", font=font,
                  fill=(28, 43, 52, 255))
        draw.line(path, fill=(53, 174, 192, 255), width=3, joint="curve")
        for i, (x, y) in enumerate(path[:-1]):
            color = (139 - 28 * i, 201 - 13 * i, 210 - 9 * i, 255)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        end = np.array(path[-1])
        direction = end - np.array(path[-2])
        direction /= max(np.linalg.norm(direction), 1e-6)
        normal = np.array([-direction[1], direction[0]])
        draw.polygon([tuple(end), tuple(end - 11 * direction + 5 * normal),
                      tuple(end - 11 * direction - 5 * normal)],
                     fill=(28, 157, 180, 255))
        draw.text((759, centroid_inset_top + 123), f"{samples[0][1]:.1f} to {current_time:.1f} s",
                  font=small, fill=(82, 111, 123, 255))
        draw.text((39, 461), "Object shape over time", font=font,
                  fill=(28, 43, 52, 255))
        detail = ("Position aligned; marker shows rotation" if motion_kind == "ball"
                  else "Translation and rotation removed")
        draw.text((510, 464), detail, font=small,
                  fill=(95, 119, 129, 255))

        card_w, card_h = 202, 166
        if motion_kind == "cable":
            reference = samples[0][0][:, :2]
            reference = reference - reference.mean(axis=0)
            aligned = []
            for points, _ in samples:
                xy = points[:, :2] - points[:, :2].mean(axis=0)
                u, _, vt = np.linalg.svd(xy.T @ reference)
                correction = np.eye(2)
                correction[1, 1] = np.linalg.det(u @ vt)
                aligned.append(xy @ (u @ correction @ vt))
            all_xy = np.concatenate(aligned)
            span = np.maximum(all_xy.max(axis=0) - all_xy.min(axis=0), 1e-6)
            scale = min((card_w - 30) / span[0], (card_h - 28) / span[1])
        for i, (_, time) in enumerate(samples):
            left = 39 + i * 221
            top = 496
            draw.rounded_rectangle((left, top, left + card_w, top + card_h),
                                   radius=7, fill=(239, 247, 249, 255),
                                   outline=(205, 223, 228, 255), width=1)
            center_x, center_y = left + card_w / 2, top + card_h / 2
            color = (40, 174, 192, 255) if i < len(samples) - 1 else (28, 157, 180, 255)
            if motion_kind == "ball":
                radius = 43
                draw.ellipse((center_x - radius, center_y - radius,
                              center_x + radius, center_y + radius),
                             outline=color, width=4)
                if spin_phases is not None:
                    direction = np.array([np.sin(spin_phases[i]),
                                          -np.cos(spin_phases[i])])
                    inner = np.array([center_x, center_y]) + 12 * direction
                    outer = np.array([center_x, center_y]) + 35 * direction
                    draw.line((*inner, *outer), fill=(24, 96, 112, 255), width=4)
                    draw.ellipse((outer[0] - 4, outer[1] - 4,
                                  outer[0] + 4, outer[1] + 4),
                                 fill=(24, 96, 112, 255))
            else:
                polyline = [
                    (center_x + float(point[0]) * scale,
                     center_y - float(point[1]) * scale)
                    for point in aligned[i]
                ]
                draw.line(polyline, fill=color, width=4, joint="curve")
                x0, y0 = polyline[0]
                draw.ellipse((x0 - 3, y0 - 3, x0 + 3, y0 + 3), fill=color)
            caption = f"t = {time:.1f} s" + ("  (now)" if i == len(samples) - 1 else "")
            draw.text((left + 52, 670), caption, font=small, fill=(44, 68, 78, 255))
            if i < len(samples) - 1:
                arrow_x = left + card_w + 10
                arrow_y = top + card_h / 2
                draw.line((arrow_x - 4, arrow_y, arrow_x + 5, arrow_y),
                          fill=(75, 112, 125, 255), width=2)
                draw.line((arrow_x + 1, arrow_y - 4, arrow_x + 5, arrow_y,
                           arrow_x + 1, arrow_y + 4),
                          fill=(75, 112, 125, 255), width=2)
        img = np.asarray(Image.alpha_composite(canvas, overlay).convert("RGB"))
    return img


# --------------------------------------------------------------------------
# Scene builders
# --------------------------------------------------------------------------

def scene_static_cube():
    objects = f"""
    <geom name="cube" type="box" pos="0.62 -0.02 0.021" size="0.021 0.021 0.021"
          rgba="0.20 0.45 0.95 1" contype="1" conaffinity="1"/>
    """
    model = build_scene(objects)
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
    err = solve_ik(model, data, site, (0.62, -0.02, 0.115), (0.0, 0.0, -1.0))
    set_fingers(model, data, 0.045)
    mujoco.mj_forward(model, data)
    print(f"cube: ik residual {err*1000:.1f} mm")
    return model, data


def scene_rolling_ball():
    ghosts = []
    for i, (y, a) in enumerate([(-0.045, 0.30), (-0.125, 0.20), (-0.215, 0.11)]):
        ghosts.append(
            f'<geom name="ghost{i}" type="sphere" pos="0.60 {y} 0.028" '
            f'size="0.028" rgba="0.95 0.75 0.10 {a}" contype="0" conaffinity="0"/>'
        )
    objects = f"""
    <geom name="ball" type="sphere" pos="0.60 0.03 0.028" size="0.028"
          rgba="0.95 0.75 0.10 1" contype="1" conaffinity="1"/>
    {''.join(ghosts)}
    """
    model = build_scene(objects)
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
    # reach toward the intercept point ahead of the ball (+Y)
    err = solve_ik(model, data, site, (0.615, 0.115, 0.115), (0.0, 0.0, -1.0))
    set_fingers(model, data, 0.045)
    mujoco.mj_forward(model, data)
    print(f"ball: ik residual {err*1000:.1f} mm")
    return model, data


def scene_routing():
    # A tabletop harness route: one snap clip holds the cable, while the
    # free end is being brought to a second empty clip.
    first = np.array([0.50, -0.14, 0.022])
    second = np.array([0.69, 0.11, 0.022])
    along = second - first
    along[2] = 0.0
    along /= np.linalg.norm(along)
    heading = math.atan2(along[1], along[0])
    tip = second - 0.065 * along

    controls = np.array([
        [0.28, -0.34, 0.016],
        [0.36, -0.28, 0.018],
        first - 0.080 * along,
        first,
        first + 0.080 * along,
        tip,
    ])
    curve = catmull_rom(controls, n_per_seg=20)
    curve[:, 2] = np.maximum(curve[:, 2], 0.016)
    # The first clip's cavity spans local y = +/-0.020 m and x = +/-0.031 m.
    assert np.linalg.norm((curve[np.argmin(np.linalg.norm(curve - first, axis=1))] - first)[:2]) < 0.002
    assert np.dot(tip - second, along) < -0.031

    def snap_clip(name: str, center: np.ndarray) -> str:
        return f"""
    <body name="{name}" pos="{center[0]:.4f} {center[1]:.4f} 0" euler="0 0 {heading:.6f}">
      <geom name="{name}_base" type="box" size="0.041 0.041 0.004" pos="0 0 0.004"
            rgba="0.18 0.22 0.27 1"/>
      <geom type="box" size="0.031 0.008 0.020" pos="0 -0.028 0.028"
            rgba="0.68 0.73 0.77 1"/>
      <geom type="box" size="0.031 0.008 0.020" pos="0 0.028 0.028"
            rgba="0.68 0.73 0.77 1"/>
      <geom type="box" size="0.031 0.004 0.004" pos="0 -0.018 0.048"
            rgba="0.86 0.89 0.91 1"/>
      <geom type="box" size="0.031 0.004 0.004" pos="0 0.018 0.048"
            rgba="0.86 0.89 0.91 1"/>
      <geom type="sphere" size="0.004" pos="-0.031 -0.032 0.009" rgba="0.09 0.11 0.14 1"/>
      <geom type="sphere" size="0.004" pos="0.031 0.032 0.009" rgba="0.09 0.11 0.14 1"/>
    </body>"""

    objects = snap_clip("secured_clip", first) + snap_clip("target_clip", second)
    objects += "\n" + cable_geom_xml(curve)
    model = build_scene(objects)
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
    grasp_pt = tip - 0.014 * along + np.array([0.0, 0.0, 0.09])
    err = solve_ik(model, data, site, grasp_pt, (0.0, 0.0, -1.0))
    set_fingers(model, data, 0.035)
    mujoco.mj_forward(model, data)
    print(f"routing: ik residual {err*1000:.1f} mm")
    return model, data


def scene_dynamic_cable(motion_mode="combined", seed=20283026,
                        sample_frames=(125, 150, 175), final_frame=200):
    from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig

    cfg = EnvConfig(
        robot="nero", motion_mode=motion_mode,
        motion_profile_version="rigid_level1_single_pass_v2",
        seed=seed,
    )
    env = CableGraspEnv(cfg)
    env.reset()
    cable_ids = [
        i for i in range(env.model.nbody)
        if (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("cableB")
    ]
    history = []
    if 0 in sample_frames:
        history.append((env.data.xpos[cable_ids].copy(), float(env.data.time)))
    for frame in range(final_frame):
        env.step(np.zeros(env.model.nu))
        if frame + 1 in sample_frames:
            history.append((env.data.xpos[cable_ids].copy(), float(env.data.time)))

    model, data = env.model, env.data
    marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cable_marker")
    if marker >= 0:
        model.site_rgba[marker, 3] = 0.0
    table_mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "table_mat")
    if table_mat >= 0:
        model.mat_reflectance[table_mat] = 0.0
        model.mat_texid[table_mat] = -1
        model.mat_rgba[table_mat] = [0.31, 0.36, 0.40, 1.0]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
    if site < 0:
        # add site is impossible post-compile; IK on link7 body instead
        site = None
    pts = np.array([data.xpos[i] for i in cable_ids])
    mid = pts[np.argmax(pts[:, 0])]           # leading edge of the writhe
    qadr, dadr = arm_joints(model)
    data.qpos[qadr] = READY_QPOS
    target = np.array([mid[0], mid[1], mid[2] + 0.10])
    # IK with mj_jacBody on link7 grasp offset — reuse site if available
    if site is not None:
        err = solve_ik(model, data, site, target, (0.0, 0.0, -1.0))
    else:
        err = _ik_body_offset(model, data, target, (0.0, 0.0, -1.0))
    set_fingers(model, data, 0.045)
    mujoco.mj_forward(model, data)
    first = history[0][0]
    shift = pts.mean(axis=0) - first.mean(axis=0)
    shape_rms = np.sqrt(np.mean(np.sum(
        ((pts - pts.mean(axis=0)) - (first - first.mean(axis=0))) ** 2,
        axis=1,
    )))
    print(f"cable ({motion_mode}): ik residual {err*1000:.1f} mm; "
          f"centroid shift {np.linalg.norm(shift)*100:.1f} cm; "
          f"shape RMS {shape_rms*100:.1f} cm")
    return model, data, (history, pts, float(data.time))


def _ik_body_offset(model, data, target_pos, approach_dir,
                    rot_weight=0.6, lam=1e-3, iters=600):
    """DLS IK on link7 with the grasp-center local offset (no site needed)."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    qadr, dadr = arm_joints(model)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        xmat = data.xmat[body].reshape(3, 3)
        pos = data.xpos[body] + xmat @ GRASP_CENTER_LOCAL
        mujoco.mj_jac(model, data, jacp, jacr, pos, body)
        axis = xmat @ np.array([1.0, 0.0, 0.0])
        err = np.concatenate([
            target_pos - pos,
            rot_weight * np.cross(axis, np.asarray(approach_dir, float)),
        ])
        J = np.vstack([jacp, rot_weight * jacr])[:, dadr]
        if np.linalg.norm(err[:3]) < 2e-4:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), err)
        data.qpos[qadr] += np.clip(dq, -0.25, 0.25)
        for i, qa in enumerate(qadr):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            lo, hi = model.jnt_range[jid]
            data.qpos[qa] = np.clip(data.qpos[qa], lo, hi)
    mujoco.mj_forward(model, data)
    xmat = data.xmat[body].reshape(3, 3)
    pos = data.xpos[body] + xmat @ GRASP_CENTER_LOCAL
    return float(np.linalg.norm(target_pos - pos))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # shared viewpoint (matches the project's opst camera side)
    view = dict(lookat=(0.60, 0.0, 0.18), distance=1.10, azimuth=90.0, elevation=-38.0)

    scenes = [
        ("task_a_static_cube", scene_static_cube),
        ("task_b_rolling_ball", scene_rolling_ball),
        ("task_c_routing", scene_routing),
        ("task_d_dynamic_cable", scene_dynamic_cable),
    ]
    images = {}
    for name, builder in scenes:
        cable_motion = None
        if name == "task_d_dynamic_cable":
            model, data, cable_motion = builder()
        else:
            model, data = builder()
        scene_view = view
        if name == "task_c_routing":
            scene_view = dict(lookat=(0.55, -0.06, 0.11), distance=0.91,
                              azimuth=90.0, elevation=-44.0)
        elif name == "task_d_dynamic_cable":
            scene_view = dict(lookat=(0.55, -0.10, 0.08), distance=1.03,
                              azimuth=90.0, elevation=-70.0)
        img = render(model, data, **scene_view, cable_motion=cable_motion)
        path = OUT_DIR / f"{name}.png"
        Image.fromarray(img).save(path)
        images[name] = img
        print("wrote", path)

    # 2x2 composite
    order = ["task_a_static_cube", "task_b_rolling_ball",
             "task_c_routing", "task_d_dynamic_cable"]
    h, w = images[order[0]].shape[:2]
    gutter = 12
    grid = np.full((h * 2 + gutter, w * 2 + gutter, 3), 255, dtype=np.uint8)
    for k, name in enumerate(order):
        r, c = divmod(k, 2)
        y, x = r * (h + gutter), c * (w + gutter)
        grid[y:y + h, x:x + w] = images[name]
    path = OUT_DIR / "task_ladder.png"
    Image.fromarray(grid).save(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
