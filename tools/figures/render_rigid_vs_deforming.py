"""Render matching tabletop illustrations from simulated ball and cable states.

The four poses in each panel are laid out for visual comparison; their display
positions are not the objects' measured world positions.
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from render_task_ladder import OUT_DIR, SCENE_TEMPLATE, catmull_rom, scene_dynamic_cable


SCALE = 2
WIDTH, HEIGHT = 1800, 680
CENTERS = [(195, 390), (655, 370), (1110, 385), (1550, 395)]
TIMES = ("0.0 s", "1.3 s", "2.7 s", "4.0 s")


def font(name, size):
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size * SCALE) if path.exists() else ImageFont.load_default()


def rolling_ball_states():
    radius = 0.065
    ball_xml = f'''
    <body name="rolling_ball" pos="0.28 -0.10 {radius + 0.002}">
      <freejoint/>
      <geom name="ball" type="sphere" size="{radius}"
            rgba="0.13 0.74 0.82 1" density="600"
            friction="0.8 0.005 0.0001"/>
    </body>'''
    xml = SCENE_TEMPLATE.replace('<include file="robot.xml"/>', '')
    model = mujoco.MjModel.from_xml_string(xml.replace('<!--OBJECTS-->', ball_xml))
    data = mujoco.MjData(model)
    data.qvel[0] = 0.12
    data.qvel[4] = 0.12 / radius
    mujoco.mj_forward(model, data)
    phases = [0.0]
    phase = 0.0
    for step in range(2000):
        mujoco.mj_step(model, data)
        phase += float(data.qvel[4] * model.opt.timestep)
        if step + 1 in {670, 1330, 2000}:
            phases.append(phase)
    return phases


def cable_states():
    _, _, motion = scene_dynamic_cable(
        motion_mode="combined", seed=20283026, sample_frames=(0, 67, 133),
    )
    shapes = [points[:, :2] for points, _ in motion[0]]
    shapes.append(motion[1][:, :2])
    reference = shapes[0] - shapes[0].mean(axis=0)
    aligned = []
    for points in shapes:
        xy = points - points.mean(axis=0)
        u, _, vt = np.linalg.svd(xy.T @ reference)
        correction = np.diag([1.0, np.linalg.det(u @ vt)])
        aligned.append(xy @ (u @ correction @ vt))
    return aligned


def table_canvas(title):
    w, h = WIDTH * SCALE, HEIGHT * SCALE
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        f = y / max(h - 1, 1)
        rgb[y, :, :] = (
            int(232 + 12 * f), int(151 + 19 * f), int(139 + 16 * f),
        )
    image = Image.fromarray(rgb, "RGB").convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.polygon([(0, 585*SCALE), (400*SCALE, h), (0, h)],
                 fill=(38, 59, 78, 255))
    draw.line([(0, 585*SCALE), (400*SCALE, h)],
              fill=(83, 103, 121, 255), width=7*SCALE)
    draw.rounded_rectangle((32*SCALE, 20*SCALE, 735*SCALE, 76*SCALE),
                           radius=10*SCALE, fill=(255, 255, 255, 218))
    draw.text((49*SCALE, 27*SCALE), title, font=font("arialbd.ttf", 32),
              fill=(40, 61, 72, 255))
    return image


def draw_arrow(draw, x1, y1, x2, y2):
    a = np.array([x1, y1], dtype=float) * SCALE
    b = np.array([x2, y2], dtype=float) * SCALE
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])
    base = b - 18*SCALE*d
    draw.line((tuple(a), tuple(base)), fill=(255, 248, 240, 235),
              width=7*SCALE)
    draw.polygon([tuple(b), tuple(base + 10*SCALE*n),
                  tuple(base - 10*SCALE*n)], fill=(255, 248, 240, 235))


def decorate_times(image):
    draw = ImageDraw.Draw(image, "RGBA")
    for i, (cx, _) in enumerate(CENTERS):
        label = f"t = {TIMES[i]}"
        text_x = (cx - 77) * SCALE
        draw.text((text_x + 3*SCALE, 117*SCALE + 3*SCALE),
                  label, font=font("timesi.ttf", 43), fill=(94, 80, 76, 85))
        draw.text((text_x, 117*SCALE), label,
                  font=font("timesi.ttf", 43), fill=(255, 249, 244, 255))
    for i in range(3):
        x1 = CENTERS[i][0] + 135
        x2 = CENTERS[i+1][0] - 145
        y = 365 + 12 * (i % 2)
        draw_arrow(draw, x1, y, x2, y + 7)


def render_ball(phases):
    image = table_canvas("(a) Rigid motion · rolling ball")
    shadow = Image.new("RGBA", image.size)
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    for cx, cy in CENTERS:
        shadow_draw.ellipse(((cx-67)*SCALE, (cy+47)*SCALE,
                             (cx+74)*SCALE, (cy+78)*SCALE),
                            fill=(68, 52, 48, 85))
    image = Image.alpha_composite(image, shadow.filter(ImageFilter.GaussianBlur(12*SCALE)))
    draw = ImageDraw.Draw(image, "RGBA")
    for (cx, cy), phase in zip(CENTERS, phases):
        x, y, r = cx*SCALE, cy*SCALE, 63*SCALE
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(14, 130, 151, 255))
        draw.ellipse((x-r+5*SCALE, y-r+5*SCALE, x+r-5*SCALE, y+r-5*SCALE),
                     fill=(42, 211, 224, 255))
        draw.ellipse((x-39*SCALE, y-42*SCALE, x+6*SCALE, y+3*SCALE),
                     fill=(115, 242, 245, 115))
        direction = np.array([np.sin(phase), -np.cos(phase)])
        p1 = np.array([x, y]) + 17*SCALE*direction
        p2 = np.array([x, y]) + 48*SCALE*direction
        draw.line((tuple(p1), tuple(p2)), fill=(15, 99, 120, 245),
                  width=8*SCALE)
        draw.ellipse((p2[0]-6*SCALE, p2[1]-6*SCALE,
                      p2[0]+6*SCALE, p2[1]+6*SCALE),
                     fill=(15, 99, 120, 245))
    decorate_times(image)
    return image


def render_cable(shapes):
    image = table_canvas("(b) Deformable motion · cable")
    paths = []
    for (cx, cy), xy in zip(CENTERS, shapes):
        span = np.maximum(np.ptp(xy, axis=0), 1e-6)
        scale = min(920.0, 300.0/span[0], 300.0/span[1])
        # The common affine projection tilts the object plane slightly.
        smooth = catmull_rom(xy, n_per_seg=5)
        points = [((cx + float(x)*scale)*SCALE,
                   (cy - float(y)*scale*0.83 - float(x)*scale*0.10)*SCALE)
                  for x, y in smooth]
        paths.append(points)
    shadow = Image.new("RGBA", image.size)
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    for points in paths:
        shadow_draw.line([(x+7*SCALE, y+11*SCALE) for x, y in points],
                         fill=(70, 51, 48, 120), width=21*SCALE, joint="curve")
    image = Image.alpha_composite(image, shadow.filter(ImageFilter.GaussianBlur(7*SCALE)))
    draw = ImageDraw.Draw(image, "RGBA")
    def round_stroke(points, color, width):
        draw.line(points, fill=color, width=width, joint="curve")
        radius = width / 2
        for x, y in points:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

    for points in paths:
        round_stroke(points, (28, 149, 173, 255), 17*SCALE)
        round_stroke(points, (54, 215, 230, 255), 14*SCALE)
        for x, y in (points[0], points[-1]):
            draw.ellipse((x-7*SCALE, y-7*SCALE, x+7*SCALE, y+7*SCALE),
                         fill=(48, 212, 227, 255))
    decorate_times(image)
    return image


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ball = render_ball(rolling_ball_states())
    cable = render_cable(cable_states())
    ball = ball.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    cable = cable.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    ball.save(OUT_DIR / "rigid_rolling_ball_table.png")
    cable.save(OUT_DIR / "deforming_cable_table.png")
    combined = Image.new("RGB", (WIDTH, 2*HEIGHT + 24), "white")
    combined.paste(ball, (0, 0))
    combined.paste(cable, (0, HEIGHT + 24))
    png_path = OUT_DIR / "rigid_vs_deforming.png"
    pdf_path = OUT_DIR / "rigid_vs_deforming.pdf"
    combined.save(png_path)
    combined.save(pdf_path, "PDF", resolution=300.0)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
