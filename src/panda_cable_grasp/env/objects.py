"""可变形态操作对象：物理规格、步态公式与场景 XML 生成。

每个操作对象都是一条 MuJoCo elasticity cable composite（与主线缆同构），
区别在物理参数（长度/半径轮廓/密度/刚度/摩擦/外观）与驱动方式：

- ``cable``：现有橙色线缆，使用环境内置的正弦扰动与 L1/L2 整体运动。
- ``fish`` / ``loach`` / ``snake`` / ``worm``：由文献中的运动学模型驱动——
  鲹形（carangiform）、鳗形（anguilliform）、蛇形（serpenoid）、
  蠕动（peristaltic）行波模板 + PD 跟踪伺服，同时提供形状变化与自主移动。

步态公式（ŝ 为归一化弧长坐标，t 为时间）：

- carangiform（鲹科鱼，Lighthill 细长体理论 / Videler 1993）：
  后体占优行波  y(s,t) = A·env_c(ŝ)·sin(2π(N·ŝ − f·t))，
  env_c(ŝ) = 0.10 + 0.20ŝ + 0.70ŝ²（尾部归一化为 1）。
- anguilliform（鳗/泥鳅，Grillner & Kashin 1976）：
  全身行波，幅值向尾部平缓增长 env_a(ŝ) = 0.55 + 0.45ŝ^1.5。
- serpenoid（蛇，Hirose 1993 serpenoid curve）：
  切向角场  θ(s,t) = α·sin(2π(N·ŝ − f·t))，
  形状由 θ 沿弧长积分得到，天然满足不可伸长约束。
- peristaltic（蚯蚓/蠕虫）：轴向收缩行波
  u(s,t) = u0·sin(2π(N·ŝ − f·t))，x(s) = s + u(s,t)。

模板伺服：对象坐标系（模板质心轨迹 + 航向角）按恒速+恒定转向率解析
积分；每个节点的目标点由模板形状给出，驱动力为
a_i = Kp·(ref_i − p_i) + Kd·(ṙef_i − v_i)（限幅）。这是与
``_rigid_shape_hold_acceleration`` 同族的形状跟踪控制，区别只在目标
形状随时间行波运动。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 步态规格
# ---------------------------------------------------------------------------

GAIT_MODES = ("anguilliform", "carangiform", "serpenoid", "peristaltic")


@dataclass(frozen=True)
class GaitSpec:
    """一族公式化步态的参数（默认值来自各族的生物运动学文献量级）。"""

    mode: str
    # 行波空间频率（每体长波数）与时间频率（Hz）
    wave_count: float
    frequency_hz: float
    # 位移幅值（carangiform/anguilliform/peristaltic，米）或切向角幅值
    # （serpenoid，弧度）
    amplitude: float
    # 模板整体运动：泳动速度、初始航向、转向率与启动延迟
    swim_speed: float = 0.0
    heading_deg: float = 90.0
    turn_rate_deg: float = 0.0
    start_delay: float = 0.8
    # 竖直方向分量（鱼离水挣扎式的尾摆抬升）
    vertical_amplitude: float = 0.0
    vertical_wave_count: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in GAIT_MODES:
            raise ValueError(f"unsupported gait mode: {self.mode!r}")
        for name in ("wave_count", "frequency_hz", "amplitude"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"gait {name} must be positive")
        for name in (
            "swim_speed", "heading_deg", "turn_rate_deg", "start_delay",
            "vertical_amplitude", "vertical_wave_count",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"gait {name} must be finite")


# ---------------------------------------------------------------------------
# 对象族物理规格
# ---------------------------------------------------------------------------

RADIUS_PROFILES = ("uniform", "fish_taper", "loach_taper", "whip_taper")


@dataclass(frozen=True)
class DeformableSpec:
    """一条可变形细长对象的物理与运动规格。"""

    name: str                       # composite 前缀（须唯一）
    family: str                     # cable/fish/loach/snake/worm
    count: int = 41                 # composite 粒子数
    length: float = 0.8
    radius: float = 0.014
    density: float = 150.0
    bend: float = 2e3
    twist: float = 1e4
    damping: float = 0.025
    friction: tuple[float, float, float] = (2.0, 0.08, 0.01)
    rgba: tuple[float, float, float, float] = (0.95, 0.22, 0.035, 1.0)
    radius_profile: str = "uniform"
    gait: GaitSpec | None = None
    # 初始质心（世界 xy）；step 前由环境按布局覆写
    start_com: tuple[float, float] = (0.55, 0.0)
    # 传送带/车道驱动；None 表示不使用车道驱动
    lane: "LaneDrive | None" = None
    # 抓取确认的夹爪孔径上限；None = 按厚度自动折算（兼容旧线缆语义）。
    # 柔软易折叠对象（蠕虫）允许更大的孔径——双层折叠捏取仍算抓住。
    grasp_aperture: float | None = None
    # 为对象生成 <skin> 蒙皮网格（椭圆截面环沿链 + 族剪影），顶点按
    # 站序绑到 composite 节点 body——纯渲染层，物理/驱动完全不变；
    # 开启后物理胶囊 geom 以 alpha=0 隐藏（仍参与碰撞与抓取判定）。
    skin: bool = False
    # 真 mesh 资产文件名（assets/mujoco/objects/ 下的 .glb/.obj）：
    # 设定后蒙皮改用该网格的顶点/UV/贴图，按弧长线性混合绑骨骼；
    # 方向约定由 MESH_ASSETS 注册表给出。None = 程序化截面蒙皮。
    skin_mesh: str | None = None

    def __post_init__(self) -> None:
        if self.family not in OBJECT_FAMILIES:
            raise ValueError(f"unsupported object family: {self.family!r}")
        if self.radius_profile not in RADIUS_PROFILES:
            raise ValueError(
                f"unsupported radius_profile: {self.radius_profile!r}"
            )
        if not self.name.isidentifier():
            raise ValueError(f"invalid object name: {self.name!r}")
        for name in ("count",):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 3:
                raise ValueError(f"{name} must be an integer >= 3")
        for name in ("length", "radius", "density", "damping"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("bend", "twist"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def node_count(self) -> int:
        """composite 生成 count-1 个节点 body（B_first..B_last）。"""

        return self.count - 1

    def radius_at(self, s_hat: np.ndarray) -> np.ndarray:
        """按归一化弧长返回每个节点的胶囊半径（鱼/泥鳅的锥形身体）。"""

        s_hat = np.asarray(s_hat, dtype=float)
        if self.radius_profile == "uniform":
            return np.full_like(s_hat, self.radius)
        if self.radius_profile == "fish_taper":
            # 纺锤形：头后 1/4 处最粗，向尾部快速收窄
            profile = 0.35 + 0.65 * np.sin(
                math.pi * np.clip(0.18 + 0.75 * s_hat, 0.0, 1.0)
            ) ** 1.2
            profile = np.clip(profile, 0.30, 1.0)
            return self.radius * profile
        if self.radius_profile == "whip_taper":
            # 鞭子：柄粗尖细的近似线性锥形
            return self.radius * (1.0 - 0.75 * np.clip(s_hat, 0.0, 1.0))
        # loach_taper：整体细长，尾部缓收
        profile = 0.55 + 0.45 * np.cos(0.5 * math.pi * np.clip(s_hat, 0.0, 1.0)) ** 0.8
        return self.radius * np.clip(profile, 0.45, 1.0)


@dataclass(frozen=True)
class LaneDrive:
    """一条直线车道的进度伺服参数（传送带/交叉车道共用）。"""

    origin: tuple[float, float]     # 车道起点（世界 xy，drive 起点的质心投影）
    direction_deg: float            # 行进方向
    speed: float                    # 标称速度 m/s
    start_time: float = 0.8         # 驱动启动时刻
    travel: float = 1.40            # 越过终点线所需的投影距离
    hold_shape: bool = False        # 是否保持初始形状（类 rigid 场景）
    # 与 L1/L2 相同的伺服增益
    path_position_gain: float = 20.0
    velocity_gain: float = 32.0
    max_acceleration: float = 8.0
    shape_stiffness: float = 1000.0
    shape_damping: float = 68.0
    shape_max_acceleration: float = 45.0


# 族默认参数（均可被 DeformableSpec 覆写）
OBJECT_FAMILIES: dict[str, dict] = {
    "cable": dict(
        length=0.80, radius=0.014, density=150.0,
        bend=2e3, twist=1e4, damping=0.025,
        friction=(2.0, 0.08, 0.01),
        rgba=(0.95, 0.22, 0.035, 1.0),
        radius_profile="uniform",
        gait=None,
    ),
    # 鲹科鱼：纺锤体、后体行波、中等游速、离水时有尾摆抬升
    "fish": dict(
        length=0.60, radius=0.020, density=220.0,
        bend=3.5e3, twist=1.2e4, damping=0.05,
        # 高滚动摩擦近似扁鱼身的抗滚性，防止闭爪瞬间圆截面滚出指垫。
        friction=(2.3, 0.10, 0.10),
        rgba=(0.18, 0.45, 0.72, 1.0),
        radius_profile="fish_taper",
        gait=GaitSpec(
            mode="carangiform",
            wave_count=1.0, frequency_hz=1.4, amplitude=0.055,
            swim_speed=0.05, heading_deg=90.0,
            vertical_amplitude=0.03, vertical_wave_count=0.5,
        ),
        # 长软鱼身在夹爪间呈 V 形搭握仍属有效抓取。
        grasp_aperture=0.100,
        skin=True,
        # 真 barramundi mesh（CC0 资产），贴图+顶点蒙到骨骼
        skin_mesh="barramundi.glb",
    ),
    # 泥鳅：细长锥形、全身行波、原地扭动为主、偶尔缓慢爬行
    "loach": dict(
        length=0.45, radius=0.012, density=200.0,
        bend=1.2e3, twist=6e3, damping=0.04,
        friction=(2.2, 0.12, 0.02),
        rgba=(0.45, 0.34, 0.16, 1.0),
        radius_profile="loach_taper",
        gait=GaitSpec(
            mode="anguilliform",
            wave_count=1.25, frequency_hz=1.7, amplitude=0.045,
            swim_speed=0.0, heading_deg=90.0,
        ),
        skin=True,
    ),
    # 蛇：均匀细圆柱、serpenoid 蜿蜒、靠行波整体推进
    "snake": dict(
        length=0.85, radius=0.011, density=170.0,
        bend=1.0e3, twist=5e3, damping=0.045,
        friction=(2.4, 0.15, 0.03),
        rgba=(0.13, 0.45, 0.22, 1.0),
        radius_profile="uniform",
        gait=GaitSpec(
            mode="serpenoid",
            wave_count=1.5, frequency_hz=0.9, amplitude=math.radians(38.0),
            swim_speed=0.03, heading_deg=90.0,
        ),
        skin=True,
        # 真蛇 mesh（Google Poly CC-BY），头/鳞纹理自带
        skin_mesh="snake_poly.glb",
    ),
    # 蠕虫：短小、轴向蠕动波、缓慢爬行（半径取肥蚯蚓/毛虫量级，
    # 保证 Panda 指垫能完成干净的单点捏取）
    "worm": dict(
        length=0.40, radius=0.013, density=180.0,
        bend=2.5e3, twist=5e3, damping=0.06,
        friction=(1.8, 0.12, 0.03),
        rgba=(0.75, 0.42, 0.46, 1.0),
        radius_profile="uniform",
        gait=GaitSpec(
            mode="peristaltic",
            wave_count=2.0, frequency_hz=1.3, amplitude=0.018,
            swim_speed=0.0, heading_deg=90.0,
        ),
        # 软体被捏时容易对折：允许最多约三层的折叠捏取孔径。
        grasp_aperture=0.080,
        skin=True,
    ),
    # ---- 非生物可变形对象 ----
    # 螺旋弹簧玩具（slinky）：更慢更深的轴向压缩脉冲波
    "spring": dict(
        length=0.50, radius=0.016, density=260.0,
        bend=4.0e3, twist=1.5e4, damping=0.04,
        friction=(2.0, 0.10, 0.03),
        rgba=(0.95, 0.55, 0.10, 1.0),
        radius_profile="uniform",
        gait=GaitSpec(
            mode="peristaltic",
            wave_count=1.0, frequency_hz=0.7, amplitude=0.040,
            swim_speed=0.0, heading_deg=90.0,
        ),
        skin=True,
    ),
    # 体操彩带：轻、扁，行波幅值沿弧长放大，带垂向飘动
    "ribbon": dict(
        length=0.70, radius=0.006, density=60.0,
        bend=0.4e3, twist=2e3, damping=0.03,
        friction=(1.2, 0.06, 0.02),
        rgba=(0.93, 0.25, 0.55, 1.0),
        radius_profile="uniform",
        gait=GaitSpec(
            mode="anguilliform",
            wave_count=1.5, frequency_hz=1.1, amplitude=0.070,
            swim_speed=0.0, heading_deg=90.0,
            vertical_amplitude=0.025, vertical_wave_count=1.0,
        ),
        skin=True,
    ),
    # 花园水管：粗重硬，低波数缓慢摆动，一端带铜喷头
    "hose": dict(
        length=0.90, radius=0.022, density=380.0,
        bend=8.0e3, twist=2.0e4, damping=0.06,
        friction=(2.2, 0.12, 0.05),
        rgba=(0.10, 0.30, 0.14, 1.0),
        radius_profile="uniform",
        gait=GaitSpec(
            mode="serpenoid",
            wave_count=0.8, frequency_hz=0.5, amplitude=math.radians(16.0),
            swim_speed=0.0, heading_deg=90.0,
        ),
        grasp_aperture=0.075,
        skin=True,
    ),
    # 鞭子：强锥形（柄粗尖细），行波向尖端增幅（甩鞭能量集中）
    "whip": dict(
        length=0.85, radius=0.018, density=190.0,
        bend=1.5e3, twist=6e3, damping=0.035,
        friction=(1.9, 0.10, 0.02),
        rgba=(0.42, 0.26, 0.12, 1.0),
        radius_profile="whip_taper",
        gait=GaitSpec(
            mode="carangiform",
            wave_count=1.2, frequency_hz=1.6, amplitude=0.045,
            swim_speed=0.0, heading_deg=90.0,
        ),
        skin=True,
    ),
}


def family_spec(
    family: str,
    name: str,
    *,
    gait_amplitude_scale: float = 1.0,
    gait_frequency_scale: float = 1.0,
    gait_swim_speed: float | None = None,
    **overrides,
) -> DeformableSpec:
    """按族默认参数生成一个对象规格；name 作为 composite 前缀须唯一。"""

    if family not in OBJECT_FAMILIES:
        choices = ", ".join(sorted(OBJECT_FAMILIES))
        raise ValueError(f"unknown object family {family!r}; choices: {choices}")
    defaults = OBJECT_FAMILIES[family]
    gait = defaults["gait"]
    if gait is not None:
        gait_updates: dict[str, float] = {}
        if gait_swim_speed is not None:
            gait_updates["swim_speed"] = float(gait_swim_speed)
        if gait_amplitude_scale != 1.0:
            gait_updates["amplitude"] = gait.amplitude * gait_amplitude_scale
            gait_updates["vertical_amplitude"] = (
                gait.vertical_amplitude * gait_amplitude_scale
            )
        if gait_frequency_scale != 1.0:
            gait_updates["frequency_hz"] = (
                gait.frequency_hz * gait_frequency_scale
            )
        if gait_updates:
            gait = replace(gait, **gait_updates)
    values = {k: v for k, v in defaults.items() if k != "gait"}
    values.update(overrides)
    return DeformableSpec(name=name, family=family, gait=gait, **values)


# ---------------------------------------------------------------------------
# 步态模板（运动学公式）
# ---------------------------------------------------------------------------

def _amplitude_envelope(mode: str, s_hat: np.ndarray) -> np.ndarray:
    """各步态的侧向幅值包络（尾部 ŝ=1 归一化）。"""

    s_hat = np.asarray(s_hat, dtype=float)
    if mode == "carangiform":
        return 0.10 + 0.20 * s_hat + 0.70 * s_hat ** 2
    if mode == "anguilliform":
        return 0.55 + 0.45 * s_hat ** 1.5
    if mode == "peristaltic":
        return np.ones_like(s_hat)
    raise ValueError(f"no amplitude envelope for gait {mode!r}")


def gait_template(
    gait: GaitSpec,
    length: float,
    s_hat: np.ndarray,
    time_value: float,
    phase: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回对象局部坐标系下模板节点位置与速度（沿航向为 x）。

    局部系 x 轴为行进方向、y 为横向、z 为竖直方向；弧长方向以体长
    ``length`` 归一。返回形状 ``(n, 3)``，模板质心已对准 x=0 基准
    （调用方再叠加模板质心轨迹与航向旋转）。
    """

    s_hat = np.asarray(s_hat, dtype=float)
    n = s_hat.size
    elapsed = max(0.0, float(time_value) - gait.start_delay)
    omega = 2.0 * math.pi * gait.frequency_hz
    wave_number = 2.0 * math.pi * gait.wave_count
    arg = wave_number * s_hat - omega * elapsed + float(phase)

    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    # 弦向坐标：以体长为跨距、质心对准 0
    chord = (s_hat - 0.5) * length

    if gait.mode == "peristaltic":
        # 轴向收缩行波：x(s,t) = chord + u0·sin(arg)
        u = gait.amplitude * np.sin(arg)
        du_dt = gait.amplitude * (-omega) * np.cos(arg)
        positions[:, 0] = chord + u
        velocities[:, 0] = du_dt
        # 收缩处略微抬升，模拟环节膨起
        positions[:, 2] = 0.4 * gait.amplitude * (1.0 - np.cos(arg))
        velocities[:, 2] = 0.4 * gait.amplitude * omega * np.sin(arg)
    elif gait.mode == "serpenoid":
        # 切向角场 θ(s,t) = α·sin(arg)；形状由 θ 积分并重新居中
        theta = gait.amplitude * np.sin(arg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # 数值积分切向角场得到折线，再平移使质心过原点
        xs = np.concatenate(([0.0], np.cumsum(cos_t[:-1])))
        ys = np.concatenate(([0.0], np.cumsum(sin_t[:-1])))
        xs -= xs.mean()
        ys -= ys.mean()
        # 归一化弦长与体长一致
        span = np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2)
        if span > 1e-12:
            scale = length / max(span, 1e-12)
            xs *= scale
            ys *= scale
        positions[:, 0] = xs
        positions[:, 1] = ys
        dtheta_dt = gait.amplitude * (-omega) * np.cos(arg)
        # 速度由位置场的欧拉导数近似：对弧长参数化的形变速度
        velocities[:, 1] = dtheta_dt * (chord - chord.mean())
    else:
        envelope = _amplitude_envelope(gait.mode, s_hat)
        y = gait.amplitude * envelope * np.sin(arg)
        dy_dt = gait.amplitude * envelope * (-omega) * np.cos(arg)
        positions[:, 0] = chord
        positions[:, 1] = y
        velocities[:, 1] = dy_dt
        if gait.vertical_amplitude > 0.0:
            z_arg = (
                2.0 * math.pi * gait.vertical_wave_count * s_hat
                - omega * elapsed + float(phase)
            )
            z_env = s_hat ** 2  # 尾摆抬升集中于尾部
            positions[:, 2] = gait.vertical_amplitude * z_env * np.abs(
                np.sin(z_arg)
            )
            velocities[:, 2] = (
                gait.vertical_amplitude
                * z_env
                * (-omega)
                * np.where(np.sin(z_arg) >= 0.0, np.cos(z_arg), -np.cos(z_arg))
            )
    return positions, velocities


def template_frame_state(
    gait: GaitSpec,
    time_value: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """模板质心位置、航向角及其速率（恒速 + 恒定转向率的解析积分）。"""

    elapsed = max(0.0, float(time_value) - gait.start_delay)
    heading = math.radians(gait.heading_deg)
    turn_rate = math.radians(gait.turn_rate_deg)
    speed = float(gait.swim_speed)
    if abs(turn_rate) < 1e-12:
        offset = np.array(
            [math.cos(heading) * speed * elapsed,
             math.sin(heading) * speed * elapsed]
        )
    else:
        # 匀速率圆周运动弧线的解析积分
        angle = heading + turn_rate * elapsed
        radius = speed / turn_rate
        offset = np.array([
            radius * (math.sin(angle) - math.sin(heading)),
            -radius * (math.cos(angle) - math.cos(heading)),
        ])
    heading_now = heading + turn_rate * elapsed
    velocity = speed * np.array([math.cos(heading_now), math.sin(heading_now)])
    return offset, heading_now, velocity, turn_rate


# ---------------------------------------------------------------------------
# 视觉装饰（一眼可辨的对象外观）
# ---------------------------------------------------------------------------
#
# 每项：node 为归一化弧长位置（0=头/柄端，1=尾/尖端），geom 为传给
# MjSpec body.add_geom 的关键字；type/size/pos/quat/rgba 之外强制
# contype=0、conaffinity=0、density=0（纯视觉、不参与物理与质量）。
# composite cable 的节点胶囊轴沿局部 +Z；以下尺寸/位置均按该约定。

def decoration_geoms(spec: DeformableSpec) -> list[dict]:
    """按族返回挂在 composite 节点 body 上的装饰 geom 列表。"""

    plans: list[dict] = []
    family = spec.family
    if family == "fish":
        if not spec.skin:
            plans += [
                # 尾鳍/背鳍（无蒙皮时靠扁盒做剪影；有蒙皮时鳍在网格里）
                dict(node=0.97, geom=dict(
                    type="box", size=(0.038, 0.0035, 0.048),
                    pos=(-0.030, 0.0, 0.0), rgba=(0.10, 0.30, 0.55, 1.0),
                )),
                dict(node=0.50, geom=dict(
                    type="box", size=(0.055, 0.003, 0.022),
                    pos=(0.0, 0.0, 0.034), rgba=(0.10, 0.30, 0.55, 1.0),
                )),
            ]
        if not spec.skin_mesh:
            # 双眼（程序化蒙皮需要；真 mesh 自带眼睛纹理）
            plans += [
                dict(node=0.04, geom=dict(
                    type="sphere", size=(0.008,),
                    pos=(0.012, 0.016, 0.007), rgba=(0.02, 0.02, 0.02, 1.0),
                )),
                dict(node=0.04, geom=dict(
                    type="sphere", size=(0.008,),
                    pos=(0.012, -0.016, 0.007), rgba=(0.02, 0.02, 0.02, 1.0),
                )),
            ]
    elif family == "snake":
        if not spec.skin_mesh:
            plans += [
                # 略膨大的头 + 黄眼（真 mesh 自带头形/纹理时省略）
                dict(node=0.01, geom=dict(
                    type="ellipsoid", size=(0.026, 0.019, 0.017),
                    pos=(0.012, 0.0, 0.0), rgba=(0.10, 0.38, 0.18, 1.0),
                )),
                dict(node=0.01, geom=dict(
                    type="sphere", size=(0.0055,),
                    pos=(0.026, 0.011, 0.011), rgba=(0.95, 0.82, 0.10, 1.0),
                )),
                dict(node=0.01, geom=dict(
                    type="sphere", size=(0.0055,),
                    pos=(0.026, -0.011, 0.011), rgba=(0.95, 0.82, 0.10, 1.0),
                )),
            ]
    elif family == "loach":
        # 口周三对触须（绕局部 X 轴侧向展开）
        for side in (-1.0, 1.0):
            for index in range(3):
                plans.append(dict(node=0.02, geom=dict(
                    type="cylinder", size=(0.0015, 0.014),
                    pos=(0.008, side * (0.006 + 0.004 * index),
                         -0.006 + 0.005 * index),
                    quat=(0.9239, side * 0.3827, 0.0, 0.0),
                    rgba=(0.30, 0.22, 0.10, 1.0),
                )))
    elif family == "spring":
        # 均布螺旋环片
        for s in np.linspace(0.04, 0.96, 14):
            plans.append(dict(node=float(s), geom=dict(
                type="cylinder", size=(0.034, 0.0018),
                pos=(0.0, 0.0, 0.0), quat=(0.7071, 0.0, 0.7071, 0.0),
                rgba=(1.0, 0.62, 0.12, 1.0),
            )))
    elif family == "ribbon":
        # 每节一片薄板（彩带面）；蒙皮本身是扁带，跳过
        if not spec.skin:
            for s in np.linspace(0.0, 0.98, spec.node_count):
                plans.append(dict(node=float(s), geom=dict(
                    type="box", size=(0.014, 0.022, 0.0008),
                    pos=(0.0, 0.0, 0.0),
                    rgba=(0.93, 0.25, 0.55, 1.0),
                )))
    elif family == "hose":
        plans += [
            # 铜喷头 + 收口喷口
            dict(node=0.98, geom=dict(
                type="cylinder", size=(0.026, 0.038),
                pos=(0.030, 0.0, 0.0), quat=(0.7071, 0.0, 0.7071, 0.0),
                rgba=(0.74, 0.56, 0.22, 1.0),
            )),
            dict(node=0.98, geom=dict(
                type="cylinder", size=(0.017, 0.012),
                pos=(0.075, 0.0, 0.0), quat=(0.7071, 0.0, 0.7071, 0.0),
                rgba=(0.62, 0.46, 0.16, 1.0),
            )),
        ]
    elif family == "whip":
        plans += [
            # 握柄 + 末端响梢
            dict(node=0.0, geom=dict(
                type="cylinder", size=(0.021, 0.065),
                pos=(-0.045, 0.0, 0.0), quat=(0.7071, 0.0, 0.7071, 0.0),
                rgba=(0.28, 0.16, 0.07, 1.0),
            )),
            dict(node=0.97, geom=dict(
                type="capsule", size=(0.004, 0.022),
                pos=(0.024, 0.0, 0.0), quat=(0.7071, 0.0, 0.7071, 0.0),
                rgba=(0.86, 0.76, 0.55, 1.0),
            )),
        ]
    return plans


# ---------------------------------------------------------------------------
# 蒙皮网格（<skin>）：沿链截面环 + 族剪影，纯渲染层
# ---------------------------------------------------------------------------

SKIN_RING = 8


def _skin_profiles(spec: DeformableSpec) -> tuple[np.ndarray, np.ndarray]:
    """每节点截面 (半宽 w, 半高 h)。s=0 头/柄端，s=1 尾/尖端。"""

    s = np.linspace(0.0, 1.0, spec.node_count)
    r = spec.radius_at(s)
    fam = spec.family
    if fam == "fish":
        # 侧扁鱼：尖吻、中前段高、尾柄急收
        w = r * (0.22 + 0.62 * np.sin(
            np.pi * np.clip(s * 0.88 + 0.05, 0.0, 1.0)) ** 1.4)
        h = r * (0.35 + 1.75 * np.sin(
            np.pi * np.clip(s * 0.82 + 0.08, 0.0, 1.0)) ** 1.3)
        k = np.clip((s - 0.80) / 0.20, 0.0, 1.0)
        w = w * (1.0 - 0.80 * k)
        h = h * (1.0 - 0.82 * k)
    elif fam == "snake":
        # 头端略膨大的圆管
        head = np.exp(-((s / 0.10) ** 2))
        w = r * (1.0 + 0.35 * head)
        h = r * (0.95 + 0.45 * head)
    elif fam == "worm":
        # 环纹蚯蚓
        rib = 1.0 + 0.16 * np.sin(36 * np.pi * s)
        w = r * rib
        h = r * rib
    elif fam == "ribbon":
        # 扁宽带
        w = np.full_like(s, 0.022)
        h = np.full_like(s, 0.0012)
    elif fam == "loach":
        w = r * 0.85
        h = r * 1.15
    elif fam == "spring":
        # 细芯管（螺旋环在装饰层）
        w = r * 0.55
        h = r * 0.55
    elif fam == "hose":
        w = r * 1.10
        h = r * 1.10
    else:  # whip / cable / 其他
        w = r * 1.05
        h = r * 1.05
    return np.maximum(w, 0.0018), np.maximum(h, 0.0012)


def _node_body_name(spec: DeformableSpec, index: int) -> str:
    """composite 节点序号 → 生成 body 名（B_first/B_i/B_last）。"""

    if index <= 0:
        return f"{spec.name}B_first"
    if index >= spec.node_count - 1:
        return f"{spec.name}B_last"
    return f"{spec.name}B_{index}"


def skin_xml(spec: DeformableSpec, offset: tuple[float, float, float]) -> str:
    """生成 `<skin>` 蒙皮 asset XML。

    网格 = 每节点一个椭圆截面环（局部 +X=链切向、+Z=竖直、+Y=侧向），
    相邻环连成管面，两端封口；鱼另有尾鳍 V 形片与背鳍脊。每站顶点
    刚性绑到对应 composite 节点 body——皮肤随骨骼变形，不参与物理。
    设了 ``skin_mesh`` 时改用真 mesh 资产（`_mesh_skin_xml`）。
    """

    if spec.skin_mesh:
        return _mesh_skin_xml(spec, offset)
    n = spec.node_count
    ring = SKIN_RING
    half_w, half_h = _skin_profiles(spec)
    dx = spec.length / n
    ox, oy, oz = offset
    verts: list[tuple[float, float, float]] = []
    faces: list[int] = []
    vert_bone: list[int] = []

    for i in range(n):
        cx = ox + i * dx
        w, h = float(half_w[i]), float(half_h[i])
        for k in range(ring):
            a = 2.0 * math.pi * k / ring
            verts.append((cx, oy + w * math.sin(a), oz + h * math.cos(a)))
            vert_bone.append(i)
    for i in range(n - 1):
        for k in range(ring):
            a = i * ring + k
            b = i * ring + (k + 1) % ring
            c = (i + 1) * ring + (k + 1) % ring
            d = (i + 1) * ring + k
            faces += [a, b, c, a, c, d]
    for k in range(ring):  # 两端封口
        faces += [0, k, (k + 1) % ring]
        base = (n - 1) * ring
        faces += [base, base + (k + 1) % ring, base + k]

    if spec.family == "fish":
        # 尾鳍：末节之后外伸的 V 形叉尾（上下两尖 + 中央凹口）
        tip_x = ox + n * dx + 0.045
        i_up = len(verts)
        verts.append((tip_x, oy, oz + 0.048))
        i_dn = len(verts)
        verts.append((tip_x, oy, oz - 0.048))
        i_no = len(verts)
        verts.append((tip_x - 0.028, oy, oz))
        vert_bone += [n - 1, n - 1, n - 1]
        top = (n - 1) * ring        # k=0 → z+h 顶部顶点
        bot = (n - 1) * ring + ring // 2
        faces += [top, i_up, i_no, bot, i_no, i_dn]
        # 背鳍：中前段上方一条脊状薄片
        tips = []
        for i in range(int(0.38 * n), int(0.60 * n)):
            t = len(verts)
            verts.append((ox + i * dx, oy, oz + float(half_h[i]) + 0.024))
            vert_bone.append(i)
            tips.append((i, t))
        for (i0, t0), (i1, t1) in zip(tips, tips[1:]):
            faces += [i0 * ring, i1 * ring, t1, i0 * ring, t1, t0]

    v_txt = " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts)
    f_txt = " ".join(str(f) for f in faces)
    bones_xml = []
    for i in range(n):
        ids = [v for v, b in enumerate(vert_bone) if b == i]
        bones_xml.append(
            f'    <bone body="{_node_body_name(spec, i)}" '
            f'bindpos="{ox + i * dx:.5f} {oy:.5f} {oz:.5f}" '
            f'bindquat="1 0 0 0" '
            f'vertid="{" ".join(str(v) for v in ids)}" '
            f'vertweight="{" ".join("1" for _ in ids)}"/>'
        )
    rgba = " ".join(f"{c:g}" for c in spec.rgba)
    return (
        f'  <skin name="{spec.name}_skin" vertex="{v_txt}" face="{f_txt}" '
        f'inflate="0.0015" rgba="{rgba}">\n'
        + "\n".join(bones_xml)
        + "\n  </skin>"
    )


# ---------------------------------------------------------------------------
# 真 mesh 蒙皮（艺术家建模资产 → <skin>，物理仍走 composite 骨骼）
# ---------------------------------------------------------------------------

OBJECT_ASSETS_DIR = Path(__file__).resolve().parents[3] / \
    "assets" / "mujoco" / "objects"

# 每个网格资产的原生坐标约定：
#   length_axis  身长轴（头→尾方向所在轴）
#   head_at      头在该轴正向端（"max"）还是负向端（"min"）
#   up_axis      背/顶方向轴
#   texture      配套贴图文件名（同目录；None 表示无贴图）
MESH_ASSETS: dict[str, dict] = {
    # BarramundiFish.glb（Sketchfab CC0，trimesh 可直接加载）：
    # 原生 z=身长、头在 z_max、y=背向、x=侧向，自带 2048 贴图+UV。
    # shape_scale=(y,z) 压扁体高/体宽 → 细长鱼观感（原生高/长≈0.45）。
    "barramundi.glb": dict(
        length_axis=2, head_at="max", up_axis=1,
        texture="barramundi_tex.png",
        shape_scale=(0.78, 0.55),
    ),
    # Snake.glb（Google Poly，CC-BY）：直管状蛇姿、原生 z=身长、
    # 头在 z_max、y=背向，32² 调色板贴图。
    "snake_poly.glb": dict(
        length_axis=2, head_at="max", up_axis=1,
        texture="snake_poly_tex.png",
    ),
}


def object_assets(specs: Sequence[DeformableSpec]) -> dict[str, bytes]:
    """收集蒙皮引用的贴图等运行时资产（供 MjSpec.from_string(assets=)）。"""

    assets: dict[str, bytes] = {}
    for spec in specs:
        info = MESH_ASSETS.get(spec.skin_mesh or "") if spec.skin_mesh else None
        if info and info.get("texture"):
            tex = OBJECT_ASSETS_DIR / info["texture"]
            assets[info["texture"]] = tex.read_bytes()
    return assets


def _load_skin_mesh(spec: DeformableSpec) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray | None, dict]:
    """加载并归一化真 mesh：旋到链坐标（+x=头→尾、+z=上、+y=侧）、
    缩放到 spec.length 身长、居中于原点（偏移在生成 XML 时再加）。"""

    try:
        import trimesh  # noqa: PLC0415  仅在用到真 mesh 资产时才需要
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "skin_mesh requires trimesh: pip install trimesh"
        ) from exc
    info = MESH_ASSETS.get(spec.skin_mesh or "")
    if info is None:
        raise ValueError(
            f"unregistered skin_mesh asset: {spec.skin_mesh!r} "
            f"(add it to MESH_ASSETS with axis/texture info)"
        )
    path = OBJECT_ASSETS_DIR / spec.skin_mesh
    mesh = trimesh.load(str(path)).to_geometry()
    verts = np.asarray(mesh.vertices, dtype=float)
    axis = info["length_axis"]
    lo, hi = verts[:, axis].min(), verts[:, axis].max()
    scale = spec.length / (hi - lo)
    # 头 → x=0（s=0），尾 → x=length；横向/竖向各自居中
    along = verts[:, axis]
    if info["head_at"] == "max":
        x = (hi - along) * scale
    else:
        x = (along - lo) * scale
    axes = [0, 1, 2]
    axes.remove(axis)
    up = info["up_axis"]
    side = next(a for a in axes if a != up)
    # 侧向必须取负：x 轴已因头向映射翻转过一次，再翻转一次保持
    # 旋转手性（det=+1）——否则面片绕序反了，蒙皮外表面被背面
    # 剔除，渲染出来会像半透明一样透出桌面。
    y = -verts[:, side] * scale
    z = verts[:, up] * scale
    y -= (y.min() + y.max()) / 2.0
    z -= (z.min() + z.max()) / 2.0
    slim = info.get("shape_scale")
    if slim:
        y *= slim[0]
        z *= slim[1]
    uv = getattr(mesh.visual, "uv", None)
    return (np.column_stack([x, y, z]), np.asarray(mesh.faces),
            None if uv is None else np.asarray(uv), info)


def _mesh_skin_xml(spec: DeformableSpec,
                   offset: tuple[float, float, float]) -> str:
    """真 mesh → <skin>：顶点按弧长投影线性混合绑相邻两根骨骼。"""

    n = spec.node_count
    verts, faces, uv, info = _load_skin_mesh(spec)
    verts = verts + np.asarray(offset, dtype=float)
    dx = spec.length / n
    ox, oy, oz = offset
    # 每顶点按 x（弧长）映射到相邻两骨骼，权重线性插值（平滑变形）
    s = np.clip((verts[:, 0] - ox) / spec.length, 0.0, 1.0) * (n - 1)
    i0 = np.clip(np.floor(s).astype(int), 0, n - 2)
    frac = s - i0
    bone_ids: list[list[int]] = [[] for _ in range(n)]
    bone_w: list[list[str]] = [[] for _ in range(n)]
    for vid in range(len(verts)):
        b0, b1 = int(i0[vid]), int(i0[vid]) + 1
        w0, w1 = 1.0 - float(frac[vid]), float(frac[vid])
        bone_ids[b0].append(vid); bone_w[b0].append(f"{w0:.3f}")
        bone_ids[b1].append(vid); bone_w[b1].append(f"{w1:.3f}")
    # 低顶点密度网格在细尾/尖端区域可能出现"空骨骼"（区间内无
    # 顶点），而 <bone> 的 vertid 为必填——补一个最近顶点、权重 0，
    # 不影响变形（实际变形由邻居骨骼的插值权重承担）。
    for i in range(n):
        if not bone_ids[i]:
            near = int(np.argmin(np.abs(verts[:, 0] - (ox + i * dx))))
            bone_ids[i].append(near)
            bone_w[i].append("0")
    bones_xml = []
    for i in range(n):
        bones_xml.append(
            f'    <bone body="{_node_body_name(spec, i)}" '
            f'bindpos="{ox + i * dx:.5f} {oy:.5f} {oz:.5f}" '
            f'bindquat="1 0 0 0" '
            f'vertid="{" ".join(map(str, bone_ids[i]))}" '
            f'vertweight="{" ".join(bone_w[i])}"/>'
        )
    v_txt = " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts)
    f_txt = " ".join(str(f) for f in faces.ravel())
    prefix = ""
    attrs = ""
    if info.get("texture") and uv is not None:
        tex, mat = info["texture"], f"{spec.name}_mat"
        prefix = (
            f'  <texture name="{tex[:-4]}_t" type="2d" file="{tex}"/>\n'
            # emission 微提亮：鱼背纹理偏暗，补偿场景阴影下的可读性
            f'  <material name="{mat}" texture="{tex[:-4]}_t" '
            f'specular="0.3" shininess="0.2" emission="0.18"/>\n'
        )
        t_txt = " ".join(f"{u:.5f} {v:.5f}" for u, v in uv)
        attrs = f' material="{mat}" texcoord="{t_txt}" rgba="1 1 1 1"'
    else:
        rgba = " ".join(f"{c:g}" for c in spec.rgba)
        attrs = f' rgba="{rgba}"'
    return (
        prefix
        + f'  <skin name="{spec.name}_skin"{attrs} '
        + f'vertex="{v_txt}" face="{f_txt}" inflate="0.001">\n'
        + "\n".join(bones_xml)
        + "\n  </skin>"
    )

def _composite_xml(spec: DeformableSpec, offset: tuple[float, float, float]) -> str:
    """生成单个 cable composite 元素（schema 与 panda_cable_grasp.xml 一致）。"""

    # 蒙皮对象：物理胶囊 alpha=0 隐藏（皮肤即外观，胶囊只管碰撞/抓取）
    rgba = "0 0 0 0" if spec.skin else " ".join(f"{c:g}" for c in spec.rgba)
    friction = " ".join(f"{c:g}" for c in spec.friction)
    ox, oy, oz = offset
    return (
        f'    <composite prefix="{spec.name}" type="cable" curve="s" '
        f'count="{spec.count} 1 1" size="{spec.length:g}"\n'
        f'               offset="{ox:g} {oy:g} {oz:g}" initial="free">\n'
        f'      <plugin plugin="mujoco.elasticity.cable">\n'
        f'      <config key="twist" value="{spec.twist:g}"/>\n'
        f'      <config key="bend" value="{spec.bend:g}"/>\n'
        # vmax=0 关闭插件的应力配色：生成对象用族外观色，不再被每步刷蓝
        f'        <config key="vmax" value="0"/>\n'
        f'      </plugin>\n'
        f'      <joint kind="main" damping="{spec.damping:g}"/>\n'
        f'      <geom type="capsule" size="{spec.radius:g}" '
        f'density="{spec.density:g}"\n'
        f'            rgba="{rgba}" condim="4"\n'
        f'            friction="{friction}"/>\n'
        f'    </composite>'
    )


def _contact_pair_xml(spec: DeformableSpec) -> list[str]:
    """桌面对该对象所有 geom 的低摩擦 pair 覆盖（与既有线缆同规则）。"""

    return [
        f'    <pair geom1="table" geom2="{spec.name}G{i}" '
        f'friction="0.12 0.12 0.003 0.0001 0.0001"/>'
        for i in range(spec.node_count)
    ]


def build_scene_xml(
    objects: Sequence[DeformableSpec],
    *,
    table_half_size: tuple[float, float],
    composite_offsets: Sequence[tuple[float, float, float]],
    njmax: int,
    nconmax: int,
) -> str:
    """生成完整场景 XML（结构对齐 assets/mujoco/panda_cable_grasp.xml）。"""

    composites = "\n\n".join(
        _composite_xml(spec, offset)
        for spec, offset in zip(objects, composite_offsets)
    )
    skins = "\n".join(
        skin_xml(spec, offset)
        for spec, offset in zip(objects, composite_offsets)
        if spec.skin
    )
    pairs = "\n".join(
        line for spec in objects for line in _contact_pair_xml(spec)
    )
    tx, ty = table_half_size
    return f"""<mujoco model="Panda dynamic cable grasp">
  <compiler meshdir="assets" angle="radian" autolimits="true"/>
  <!-- The environment injects either Panda or AgileX NERO at compile time. -->
  <include file="robot.xml"/>

  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"
          solver="Newton" cone="elliptic" impratio="20"     noslip_iterations="1"/>
  <size njmax="{njmax:d}" nconmax="{nconmax:d}"/>
  <statistic center="0.45 0 0.35" extent="1.80"/>

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
              texuniform="true" reflectance="0.12"/>
{skins}
  </asset>

  <worldbody>
    <light directional="true" pos="-2 -3 4" dir="0.3 0.35 -1"
           diffuse="0.95 0.92 0.84" specular="0.3 0.3 0.3"/>
    <light directional="true" pos="2 1 3" dir="-0.35 -0.15 -1"
           diffuse="0.25 0.35 0.45"/>

    <geom name="table" type="box" pos="0.55 0 -0.06" size="{tx:g} {ty:g} 0.06"
          material="table_mat" friction="0.35 0.05 0.01"/>

{composites}

    <site name="cable_marker" pos="0.63 0 0.08" size="0.018"
          rgba="1 0.85 0.08 0.8"/>
  </worldbody>

  <!-- Isolate the two physically different contacts.  Cable-pad contacts keep
       the cable's high friction above; table-cable contacts override it with
       low sliding friction so a missed cable slides aside instead of sticking,
       buckling and being funnelled into a moving closed gripper. -->
  <contact>
{pairs}
  </contact>
</mujoco>
"""
