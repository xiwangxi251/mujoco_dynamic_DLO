"""Render a continuous four-snapshot trajectory for a ball and cable.

Display spacing is schematic; cable contours come from consecutive frames.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_task_ladder import OUT_DIR, catmull_rom


WIDTH, HEIGHT = 1700, 1040
TABLE_Z = 0.045
CENTERS = [(-0.90, 0.0), (-0.30, 0.0), (0.30, 0.0), (0.90, 0.0)]
BLUE = "0.09 0.46 0.73 1"
GHOST_BALL = "0.09 0.46 0.73 0.18"
GHOST_CABLE = "0.09 0.46 0.73 0.045"
SAMPLE_FRAMES = (30, 34, 38, 42)  # 0.60, 0.68, 0.76, 0.84 seconds
CABLE_SEED = 20283029


def capsule(a, b, radius, color, name):
    return (
        f'<geom name="{name}" type="capsule" fromto="'
        f'{a[0]:.5f} {a[1]:.5f} {a[2]:.5f} '
        f'{b[0]:.5f} {b[1]:.5f} {b[2]:.5f}" '
        f'size="{radius:.5f}" rgba="{color}" contype="0" conaffinity="0"/>'
    )


def scene_xml(objects):
    return f'''<mujoco model="motion comparison">
      <option gravity="0 0 -9.81"/>
      <visual>
        <map znear="0.01" zfar="20"/>
        <rgba haze="1 1 1 1"/>
      </visual>
      <asset>
        <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="4" height="4"/>
      </asset>
      <worldbody>
        <light directional="true" pos="-2 -1 5" dir="0.2 0.1 -1" diffuse="0.58 0.58 0.58" specular="0 0 0"/>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 -0.79" rgba="1 1 1 1"/>
        <geom name="table_edge" type="box" size="1.31 0.89 0.045" pos="0 0 -0.030" rgba="0.18 0.27 0.34 1"/>
        <geom name="table_top" type="box" size="1.28 0.86 0.025" pos="0 0 0.020" rgba="0.72 0.61 0.46 1"/>
        <geom name="leg_fl" type="box" size="0.035 0.035 0.35" pos="-1.17 -0.76 -0.43" rgba="0.23 0.32 0.39 1"/>
        <geom name="leg_fr" type="box" size="0.035 0.035 0.35" pos="1.17 -0.76 -0.43" rgba="0.23 0.32 0.39 1"/>
        <geom name="leg_bl" type="box" size="0.035 0.035 0.35" pos="-1.17 0.76 -0.43" rgba="0.23 0.32 0.39 1"/>
        <geom name="leg_br" type="box" size="0.035 0.035 0.35" pos="1.17 0.76 -0.43" rgba="0.23 0.32 0.39 1"/>
        {objects}
      </worldbody>
    </mujoco>'''


def cable_snapshots():
    from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig

    env = CableGraspEnv(EnvConfig(
        robot="nero", motion_mode="combined",
        motion_profile_version="rigid_level1_single_pass_v2", seed=CABLE_SEED,
    ))
    env.reset()
    cable_ids = [
        i for i in range(env.model.nbody)
        if (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("cableB")
    ]
    states = []
    for frame in range(SAMPLE_FRAMES[-1]):
        env.step(np.zeros(env.model.nu))
        if frame + 1 in SAMPLE_FRAMES:
            states.append(env.data.xpos[cable_ids, :2].copy())
    return states


def cable_objects():
    objects = []
    for k, (xy, (cx, cy)) in enumerate(zip(cable_snapshots(), CENTERS)):
        smooth = catmull_rom(xy - xy.mean(axis=0), n_per_seg=3)
        smooth = 1.12 * smooth + np.array([cx, cy])
        color = GHOST_CABLE if k in (1, 2) else BLUE
        for j, (p, q) in enumerate(zip(smooth[:-1], smooth[1:])):
            a = np.array([*p, TABLE_Z + 0.013])
            b = np.array([*q, TABLE_Z + 0.013])
            objects.append(capsule(a, b, 0.014, color, f"cable_{k}_{j}"))
    return "\n".join(objects)


def ball_objects():
    radius = 0.095
    objects = []
    for k, (cx, cy) in enumerate(CENTERS):
        cz = TABLE_Z + radius
        color = GHOST_BALL if k in (1, 2) else BLUE
        objects.append(
            f'<geom name="ball_{k}" type="sphere" pos="{cx} {cy} {cz}" '
            f'size="{radius}" rgba="{color}" contype="0" conaffinity="0"/>'
        )
    return "\n".join(objects)


def render(objects, heading):
    model = mujoco.MjModel.from_xml_string(scene_xml(objects))
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, -0.08]
    camera.distance = 3.5
    camera.azimuth = 115
    camera.elevation = -58
    renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)
    renderer.update_scene(data, camera=camera)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    scene_camera = renderer.scene.camera[0]
    right = np.cross(scene_camera.forward, scene_camera.up)
    vertical_span = scene_camera.frustum_top - scene_camera.frustum_bottom
    horizontal_span = vertical_span * WIDTH / HEIGHT
    pixel_centers = []
    for cx, cy in CENTERS:
        offset = np.array([cx, cy, TABLE_Z]) - scene_camera.pos
        depth = float(np.dot(offset, scene_camera.forward))
        pixel_centers.append((
            WIDTH * (0.5 + np.dot(offset, right) * scene_camera.frustum_near / (depth * horizontal_span)),
            HEIGHT * (0.5 - np.dot(offset, scene_camera.up) * scene_camera.frustum_near / (depth * vertical_span)),
        ))
    image = Image.fromarray(renderer.render())
    renderer.close()
    draw = ImageDraw.Draw(image)
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    label_font = ImageFont.truetype(str(bold_path), 34)
    heading_font = ImageFont.truetype(str(bold_path), 40)
    draw.text((65, 56), heading, font=heading_font, fill=(35, 56, 69))
    for k, (px, py) in enumerate(pixel_centers):
        label = f"t{k}"
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        box_width, box_height = text_width + 24, text_height + 20
        lx, ly = int(px - box_width / 2), int(py - 115)
        draw.rounded_rectangle((lx, ly, lx + box_width, ly + box_height), radius=10,
                               fill=(248, 246, 239), outline=(124, 117, 103), width=2)
        draw.text((lx + (box_width - text_width) / 2 - bbox[0],
                   ly + (box_height - text_height) / 2 - bbox[1]),
                  label, font=label_font, fill=(39, 66, 82))
    return image


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ball = render(ball_objects(), "(a) Rigid rolling ball")
    cable = render(cable_objects(), "(b) Deforming cable")
    ball_path = OUT_DIR / "sim_rigid_ball_table.png"
    cable_path = OUT_DIR / "sim_deforming_cable_table.png"
    ball.save(ball_path)
    cable.save(cable_path)
    combined = Image.new("RGB", (WIDTH * 2 + 34, HEIGHT), (255, 255, 255))
    combined.paste(ball, (0, 0))
    combined.paste(cable, (WIDTH + 34, 0))
    combined.save(OUT_DIR / "sim_rigid_vs_deforming.png")
    combined.save(OUT_DIR / "sim_rigid_vs_deforming.pdf", "PDF", resolution=300)
    print(ball_path)
    print(cable_path)


if __name__ == "__main__":
    main()
