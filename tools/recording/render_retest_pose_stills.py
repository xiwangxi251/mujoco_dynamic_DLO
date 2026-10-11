"""Render measured robot states from the three center-goal pilot traces."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from render_vertical_workspace import Workspace, add_geom, add_line, font

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vertical_workspace/scripted_136_retest_20261004"


def history(version):
    filename = "pilot_0.7_0.npz" if version in (1, 3) else "pilot_0.7_0.0.npz"
    with np.load(OUT / f"version_{version}" / filename) as file:
        return file["history"]


def main():
    geometry = Workspace()
    geometry.model.vis.global_.offwidth = 960
    geometry.model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(geometry.model, height=720, width=960, max_geom=10000)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.48, 0.02, 0.12)
    camera.distance = 1.40
    camera.azimuth = 135
    camera.elevation = -29
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    sheet = Image.new("RGB", (2880, 1500), "#f7f9fc")
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 10), "Measured physical postures at common goal (0.70, 0.00, 0.005) m", font=font(29), fill="#23364a")
    for row, timestamp in enumerate((3.0, 6.0)):
        for col, version in enumerate((1, 3, 6)):
            h = history(version)
            sample = h[min(len(h) - 1, int(np.argmin(abs(h[:, 0] - timestamp))))]
            qpos = sample[1:10] if version in (1, 3) else sample[10:19]
            actual = sample[16:19] if version in (1, 3) else sample[1:4]
            geometry.data.qpos[:9] = qpos
            geometry.data.qvel[:] = 0
            mujoco.mj_forward(geometry.model, geometry.data)
            renderer.update_scene(geometry.data, camera=camera, scene_option=option)
            add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0],
                     [0.7, 0, 0.005], (1.0, 0.70, 0.04, 1))
            add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0],
                     actual, (0.94, 0.20, 0.36, 1))
            frame = Image.fromarray(renderer.render())
            sheet.paste(frame, (col * 960, 48 + row * 726))
            draw = ImageDraw.Draw(sheet)
            pose_deg = sample[12] if version in (1, 3) else sample[8]
            draw.rounded_rectangle((col * 960 + 12, 60 + row * 726,
                                    col * 960 + 408, 147 + row * 726),
                                   radius=8, fill="#ffffff", outline="#d7e0e9")
            draw.text((col * 960 + 25, 67 + row * 726),
                      f"Version {version}   t={sample[0]:.1f}s", font=font(25), fill="#1f3347")
            draw.text((col * 960 + 25, 105 + row * 726),
                      f"Full pose error {pose_deg:.1f} deg", font=font(20), fill="#3e5266")
    renderer.close()
    sheet.save(OUT / "pose_comparison.png")
    print(OUT / "pose_comparison.png")


if __name__ == "__main__":
    main()
