"""逐个扫描机器人各关节的活动范围并录制视频。

在仓库根目录运行（本地 conda 环境 mujoco-dlo）::

    conda run -n mujoco-dlo python tools/recording/render_joint_ranges.py --robot nero

采用运动学回放（直接写 qpos + mj_forward），不依赖执行器增益，
保证每个关节都精确扫到上下限；每段配独立相机视角并在段首平滑过渡。
输出 ``outputs/joint_ranges/<robot>_joint_ranges.mp4``，
并在 ``outputs/joint_ranges/checkframes/`` 存各关节端点检查帧。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from panda_cable_grasp.runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from panda_cable_grasp.env import ROBOT_SPECS

FPS = 30
WIDTH, HEIGHT = 1280, 720
RAMP = 1.05   # home→下限 / 上限→home 时长（秒）
SWEEP = 1.5   # 下限→上限 扫掠时长（秒）
HOLD = 0.7    # 端点保持时长（秒）
CAM_BLEND = 0.5  # 段首相机过渡时长（秒）
INTRO = 1.5

FONT_SIZE = 26
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑（含中文）
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/arial.ttf",
]
FONT = None
for _fp in _FONT_CANDIDATES:
    try:
        FONT = ImageFont.truetype(_fp, FONT_SIZE)
        break
    except OSError:
        continue
if FONT is None:
    FONT = ImageFont.load_default()

REPO_ROOT = Path(__file__).resolve().parents[2]


def experiment_model(robot: str) -> mujoco.MjModel:
    """编译实验场景（panda_cable_grasp.xml + 注入机器人），与环境 _load_model 一致。

    场景自带天空盒、桌面纹理、双方向光与电缆，比裸机器人 XML 亮得多。
    演示副本会剔除 cable 插件声明：官方弹性体插件在场景层把电缆渲染成纯蓝
    平滑管，而运动学回放不需要电缆物理，剔除后按 XML 呈橙色胶囊。
    """
    import re

    from panda_cable_grasp.env.environment import XML_PATH, _robot_assets

    spec = ROBOT_SPECS[robot]
    if spec.asset_dir is not None:
        assets = _robot_assets(str(spec.asset_dir))
    else:
        from panda_cable_grasp.env.environment import _panda_assets, resolve_menagerie_panda_dir

        assets = _panda_assets(str(resolve_menagerie_panda_dir()))
    xml = XML_PATH.read_text(encoding="utf-8")
    xml = re.sub(r"<extension>.*?</extension>", "", xml, flags=re.S)
    xml = re.sub(r"<plugin plugin=\"mujoco\.elasticity\.cable\">.*?</plugin>", "", xml, flags=re.S)
    mjspec = mujoco.MjSpec.from_string(
        xml, include={"robot.xml": spec.xml_path.read_bytes()}, assets=assets)
    base_body = next(b for b in mjspec.bodies if b.name == spec.base_body_name)
    base_body.pos[:] += spec.base_offset
    # 只动演示副本的纹理亮度（色调不变）：天空盒下半部近黑（桌缘外的"虚空"）、
    # 桌面 checker 也很暗，是"太暗"观感的主要来源。
    for tex in mjspec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.rgb2[:] = (0.12, 0.19, 0.28)
        elif tex.name == "table_tex":
            tex.rgb1[:] = (0.34, 0.37, 0.40)
            tex.rgb2[:] = (0.22, 0.25, 0.28)
    model = mjspec.compile()
    # 离屏缓冲与输出分辨率对齐。
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    # 场景里桌面材质很暗（checker rgb 0.10-0.22），把跟随相机的头灯调亮，
    # 保证机器人与近景桌面在任何机位下都清晰可见。
    model.vis.headlight.diffuse[:] = (0.85, 0.85, 0.85)
    model.vis.headlight.specular[:] = (0.35, 0.35, 0.35)
    return model


def arm_actuator_uids(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> list[int]:
    """按 joint1..joint7 顺序返回对应臂执行器 id。"""
    by_joint: dict[int, int] = {}
    for uid in range(model.nu):
        if model.actuator_trntype[uid] == mujoco.mjtTrn.mjTRN_JOINT:
            by_joint[int(model.actuator_trnid[uid, 0])] = uid
    out = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or jid not in by_joint:
            raise RuntimeError(f"模型中找不到臂关节/执行器: {name}")
        out.append(by_joint[jid])
    return out


def joint_limited_range(model: mujoco.MjModel, jid: int) -> tuple[float, float]:
    lo, hi = (float(v) for v in model.jnt_range[jid])
    if not model.jnt_limited[jid]:
        raise RuntimeError(f"关节 {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)} 未设置范围")
    return lo, hi


def rad2deg(x: float) -> float:
    return math.degrees(x)


def fmt_deg(x: float) -> str:
    return f"{x:+.1f}".replace("-", "\u2212")  # 用真减号，雅黑下更美观


# ---------- 相机视角 ----------

def azel_from_dir(c: np.ndarray) -> tuple[float, float]:
    """给定 lookat→相机 的单位方向，反推 MuJoCo free 相机的 (azimuth, elevation)。

    约定: offset = dist * [sin(az)cos(-el), -cos(az)cos(-el), sin(-el)]，
    即 **负 elevation 表示相机在 lookat 上方**（与场景默认 global elevation=-25 一致）。
    """
    c = np.asarray(c, dtype=float)
    c = c / np.linalg.norm(c)
    el = -math.degrees(math.asin(float(np.clip(c[2], -1.0, 1.0))))
    az = math.degrees(math.atan2(c[0], -c[1]))
    return az, el


def view_around_axis(axis: np.ndarray, roll_like: bool) -> tuple[float, float]:
    """回转/偏摆类关节（轴接近世界 z）用斜俯视角；俯仰类用垂直于轴的侧视角。

    仰角控制在 ~20-24°：相机高于桌面，画面上缘能露出明亮天空，
    避免整幅都是暗桌面背景。
    """
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    if roll_like:
        c = a + np.array([0.0, 0.0, 0.85])  # 顺轴线看，圆轨迹在画面里
    elif abs(a[2]) > 0.85:
        c = np.array([0.55, -0.78, 0.42])   # 绕竖直轴回转：斜上方 3/4 视角
    else:
        c = np.array([-a[1], a[0], 0.0])    # 俯仰关节：垂直于轴的侧面机位
        if np.linalg.norm(c) < 1e-6:
            c = np.array([0.5, -0.8, 0.0])
        c = c / np.linalg.norm(c)
        if c[1] > 0:                        # 统一站在 -y 前侧，与片头同半球
            c = -c
        c[2] = 0.35
        c = c / np.linalg.norm(c)
    return azel_from_dir(c)


def world_joint_axis(model: mujoco.MjModel, data: mujoco.MjData, jid: int) -> np.ndarray:
    bid = int(model.jnt_bodyid[jid])
    a_local = np.asarray(model.jnt_axis).reshape(-1, 3)[jid].astype(float)
    rot = np.asarray(data.xmat[bid]).reshape(3, 3)
    return rot @ a_local


def blend_view(v0: dict, v1: dict, u: float) -> dict:
    """视角插值（azimuth 走最短路径，u 处已做 ease）。"""
    daz = ((v1["az"] - v0["az"] + 180.0) % 360.0) - 180.0
    return {
        "az": v0["az"] + daz * u,
        "el": v0["el"] + (v1["el"] - v0["el"]) * u,
        "dist": v0["dist"] + (v1["dist"] - v0["dist"]) * u,
        "lookat": v0["lookat"] + (v1["lookat"] - v0["lookat"]) * u,
    }


def apply_view(cam: mujoco.MjvCamera, v: dict) -> None:
    cam.azimuth = v["az"]
    cam.elevation = v["el"]
    cam.distance = v["dist"]
    cam.lookat = np.asarray(v["lookat"], dtype=float)


# ---------- 字幕 ----------

def make_overlay(lines: list[str], bar: tuple[float, float, float] | None) -> np.ndarray:
    """透明文字层（RGBA），文字带白色描边保证在任意背景可读。"""
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y = 10
    for ln in lines:
        d.text((16, y), ln, font=FONT, fill=(15, 15, 15, 255),
               stroke_width=3, stroke_fill=(255, 255, 255, 235))
        y += FONT_SIZE + 6
    if bar is not None:
        lo, hi, cur = bar
        x0, w, bh, by = 16, 420, 14, HEIGHT - 42
        d.rectangle([x0, by, x0 + w, by + bh], fill=(255, 255, 255, 150), outline=(0, 0, 0, 200))
        frac = 0.5 if hi <= lo else (cur - lo) / (hi - lo)
        cx = x0 + min(max(frac, 0.0), 1.0) * w
        d.rectangle([cx - 3, by - 6, cx + 3, by + bh + 6], fill=(220, 40, 40, 230))
        d.text((x0, by - 34), f"{fmt_deg(lo)}°", font=FONT, fill=(20, 20, 20, 255),
               stroke_width=2, stroke_fill=(255, 255, 255, 235))
        d.text((x0 + w - 78, by - 34), f"{fmt_deg(hi)}°", font=FONT, fill=(20, 20, 20, 255),
               stroke_width=2, stroke_fill=(255, 255, 255, 235))
    return np.asarray(layer)


def compose(rgb: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    img = Image.fromarray(rgb).convert("RGBA")
    img = Image.alpha_composite(img, Image.fromarray(overlay))
    return np.asarray(img.convert("RGB"))


def run(robot: str, output: Path) -> None:
    spec = ROBOT_SPECS[robot]
    model = experiment_model(robot)
    arm_uids = arm_actuator_uids(model, spec.arm_joint_names)
    arm_jids = [int(model.actuator_trnid[u, 0]) for u in arm_uids]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # 基准位形（运动学直接写 qpos）：臂到 ready，夹爪打开。
    qpos_base = np.array(data.qpos, dtype=float)
    for jid, q in zip(arm_jids, spec.ready_arm_qpos):
        qpos_base[model.jnt_qposadr[jid]] = float(q)
    fj_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in spec.finger_joint_names]
    open_q = 0.0
    for jid in fj_ids:
        lo, hi = joint_limited_range(model, jid)
        open_q = max(open_q, abs(lo), abs(hi))  # 全开口
    for jid in fj_ids:
        sgn = 1.0 if abs(joint_limited_range(model, jid)[1]) >= abs(joint_limited_range(model, jid)[0]) else -1.0
        qpos_base[model.jnt_qposadr[jid]] = sgn * open_q
    data.qpos[:] = qpos_base
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    # 各关节扫描时间线（单位 = 关节 qpos 值）。
    segments: list[dict] = []
    for k, (uid, jid) in enumerate(zip(arm_uids, arm_jids)):
        lo, hi = joint_limited_range(model, jid)
        home = float(qpos_base[model.jnt_qposadr[jid]])
        t = [0.0]
        v = [home]
        t.append(t[-1] + RAMP); v.append(lo)
        t.append(t[-1] + HOLD); v.append(lo)
        t.append(t[-1] + SWEEP); v.append(hi)
        t.append(t[-1] + HOLD); v.append(hi)
        t.append(t[-1] + RAMP); v.append(home)
        t.append(t[-1] + HOLD); v.append(home)
        axis = world_joint_axis(model, data, jid)
        segments.append({
            "kind": "arm", "uid": uid, "jid": jid, "addr": int(model.jnt_qposadr[jid]),
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
            "range": (lo, hi), "times": t, "values": v, "duration": t[-1],
        "axis": axis, "roll_like": k == len(arm_jids) - 1,  # 末端关节按回转处理
        "snaps": [("lo", t[1] + HOLD * 0.5), ("hi", t[3] + HOLD * 0.5)],
    })
    # 夹爪段：开 → 闭 → 开（两指按等式约束 q2 = -q1 同步）。
    t = [0.0, 0.35, 0.35 + SWEEP, 0.35 + SWEEP + HOLD, 0.35 + SWEEP * 2, 0.35 + SWEEP * 2 + HOLD]
    v = [open_q, open_q, 0.0, 0.0, open_q, open_q]
    segments.append({
        "kind": "gripper", "times": t, "values": v, "duration": t[-1],
        "name": "gripper", "open_q": open_q,
        # 夹爪时间线：开→闭(t[2]~t[3] 保持)→开(t[4]~t[5] 保持)，检查帧取保持中点。
        "snaps": [("closed", t[2] + HOLD * 0.5), ("open", t[4] + HOLD * 0.5)],
    })

    # 打印关节范围总表。
    print(f"== {robot} 关节活动范围 ==")
    for uid in arm_uids:
        jid = int(model.actuator_trnid[uid, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        lo, hi = joint_limited_range(model, jid)
        print(f"  {name:7s} [{lo:+.4f}, {hi:+.4f}] rad   "
              f"[{rad2deg(lo):+7.1f}, {rad2deg(hi):+7.1f}] deg   幅值 {rad2deg(hi - lo):.1f} deg")
    for jid in fj_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        lo, hi = joint_limited_range(model, jid)
        print(f"  {name:15s} [{lo:+.4f}, {hi:+.4f}] m（滑动）")

    # 指尖世界坐标（基准位形下），用于腕部/夹爪特写视角。
    xs = [np.array(data.xpos[int(model.jnt_bodyid[jid])]) for jid in fj_ids]
    tip = np.mean(xs, axis=0) if xs else np.array([0.4, 0.0, 0.6])
    # 臂身中心（基准位形下各臂关节体位置与指尖的均值），适配底座偏移后的取景。
    arm_bids = [int(model.jnt_bodyid[j]) for j in arm_jids]
    arm_mid = np.mean([np.array(data.xpos[b]) for b in arm_bids] + [tip], axis=0)

    def seg_view(seg: dict) -> dict:
        if seg["kind"] == "gripper":
            az, el = azel_from_dir(np.array([0.45, -0.55, 0.70]))
            return {"az": az, "el": el, "dist": 0.75, "lookat": tip.copy()}
        az, el = view_around_axis(seg["axis"], seg["roll_like"])
        jid = seg["jid"]
        near = jid in arm_jids[-3:]  # 腕部三关节特写
        if near and not seg["roll_like"]:
            look = 0.55 * tip + 0.45 * arm_mid + np.array([0.0, 0.0, 0.03])
            dist = 1.05
        elif seg["roll_like"]:
            look = tip - np.array([0.05, 0.0, 0.02])
            dist = 1.0
        else:
            # NERO 臂身中心 z≈0.33、指尖 z≈0.25，1.8m 左右距离即可充满画面；
            # 低仰角让画面上缘露出明亮天空，机器人不再被暗桌面淹没。
            look = 0.6 * arm_mid + 0.4 * tip + np.array([0.0, 0.0, 0.06])
            dist = 1.75
        return {"az": az, "el": el, "dist": dist, "lookat": look}

    intro_view = {"az": 32.0, "el": -16.0, "dist": 2.2,
                  "lookat": 0.6 * arm_mid + 0.4 * tip + np.array([0.0, 0.0, 0.06])}

    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    output.parent.mkdir(parents=True, exist_ok=True)
    check_dir = output.parent / "checkframes"
    check_dir.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频编码器写入 {output}")

    def write_frame(lines, bar, view, png_path=None):
        apply_view(cam, view)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        frame = compose(rgb, make_overlay(lines, bar))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if png_path is not None:
            cv2.imwrite(str(png_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # 片头：ready 位形。
    data.qpos[:] = qpos_base
    for f in range(int(INTRO * FPS)):
        write_frame([f"{robot.upper()} 关节活动范围演示    Ready 位形（1.5 秒后开始逐关节扫描）",
                     "运动学回放：每段 起点→下限→上限→回位"],
                    None, intro_view)

    view_prev = intro_view
    total_frames = 0
    for si, seg in enumerate(segments):
        seg["view"] = seg_view(seg)
        nseg = int(seg["duration"] * FPS)
        snap_done: set[str] = set()
        data.qpos[:] = qpos_base
        for f in range(nseg):
            tl = f / FPS
            cur = float(np.interp(tl, seg["times"], seg["values"]))
            ease = 0.0
            if tl < CAM_BLEND:
                u = (tl / CAM_BLEND) ** 2 * (3 - 2 * (tl / CAM_BLEND))
                view = blend_view(view_prev, seg["view"], u)
            else:
                view = seg["view"]
            _ = ease
            if seg["kind"] == "arm":
                data.qpos[seg["addr"]] = cur
                lo, hi = seg["range"]
                lines = [
                    f"{robot.upper()} 关节范围扫描    第 {si + 1}/8 段    {seg['name']}",
                    f"范围 [{fmt_deg(rad2deg(lo))}°, {fmt_deg(rad2deg(hi))}°]  共 {rad2deg(hi - lo):.1f}°",
                    f"当前 {fmt_deg(rad2deg(cur))}°",
                ]
                bar = (rad2deg(lo), rad2deg(hi), rad2deg(cur))
            else:
                for jid in fj_ids:
                    sgn = 1.0 if abs(joint_limited_range(model, jid)[1]) >= abs(joint_limited_range(model, jid)[0]) else -1.0
                    data.qpos[model.jnt_qposadr[jid]] = sgn * cur
                data.qvel[:] = 0.0
                lines = [
                    f"{robot.upper()} 关节范围扫描    第 {si + 1}/8 段    夹爪",
                    f"夹爪开口 0 → {open_q * 2000:.0f} mm（两指滑动行程）",
                    f"当前开口约 {cur * 2000:.1f} mm",
                ]
                bar = None
            # 端点保持中点各存一张检查帧。
            png = None
            for tag, tv in seg["snaps"]:
                if tag not in snap_done and abs(tl - tv) < 0.5 / FPS:
                    png = check_dir / f"{robot}_{seg['name']}_{tag}.png"
                    snap_done.add(tag)
            write_frame(lines, bar, view, png)
            total_frames += 1
        view_prev = seg["view"]

    writer.release()
    renderer.close()
    dur = (total_frames + int(INTRO * FPS)) / FPS
    print(f"已写入 {output}（约 {dur:.1f} 秒，{total_frames + int(INTRO * FPS)} 帧）")
    print(f"检查帧目录：{check_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="录制各关节活动范围扫描视频")
    parser.add_argument("--robot", choices=tuple(ROBOT_SPECS), default="nero")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or REPO_ROOT / "outputs" / "joint_ranges" / f"{args.robot}_joint_ranges.mp4"
    run(args.robot, output)


if __name__ == "__main__":
    main()
