"""Render same-scale comparisons of the fresh physical workspace scans."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vertical_workspace/scripted_136_retest_20261004"
VERSIONS = (1, 3, 6)
LABELS = {
    1: "1  Position-first JIK / original servo",
    3: "3  Position-first JIK / enhanced servo + orientation x4",
    6: "6  Full 6D DLS JIK / original servo",
}
COLORS = ["#e7edf3", "#ed8272", "#b189c4", "#82bfee", "#48b995", "#137b62"]


def load():
    all_rows = {}
    for version in VERSIONS:
        directory = OUT / f"version_{version}"
        rows = json.loads((directory / "cells.json").read_text(encoding="utf-8"))
        positions = [(r["x_m"], r["y_m"]) for r in rows]
        if len(rows) != 961 or len(set(positions)) != 961:
            raise RuntimeError(f"Version {version}: expected 961 unique cells, got {len(rows)}")
        all_rows[version] = rows
    if any(set((r["x_m"], r["y_m"]) for r in all_rows[v]) !=
           set((r["x_m"], r["y_m"]) for r in all_rows[1]) for v in (3, 6)):
        raise RuntimeError("Grid coordinates differ across versions")
    return all_rows


def grid(rows, version):
    xs = np.round(np.arange(-0.25, 0.951, 0.04), 6)
    ys = np.round(np.arange(-0.60, 0.601, 0.04), 6)
    shape = (len(ys), len(xs))
    category = np.zeros(shape, dtype=int)
    position = np.full(shape, np.nan)
    pose = np.full(shape, np.nan)
    for row in rows:
        ix = int(np.argmin(abs(xs - row["x_m"])))
        iy = int(np.argmin(abs(ys - row["y_m"])))
        category[iy, ix] = (4 if row["motion_pass"] else 3 if row["endpoint_pass"]
                            else 1 if row["reason"] == "floor_penetration"
                            else 0 if row["reason"] == "timeout" else 2)
        position[iy, ix] = row["position_error_mm"]
        pose[iy, ix] = row["max_path_pose_deg"]
    with np.load(OUT / f"version_{version}" / "workspace.npz") as file:
        category[file["interior"]] = 5
    return category, position, pose


def format_axis(ax):
    ax.set_xlim(-0.27, 0.97)
    ax.set_ylim(-0.62, 0.62)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World X (m)")
    ax.grid(color="#8795a5", alpha=0.18, linewidth=0.5)
    ax.plot([0.35], [0], marker="o", markersize=5, color="#263c51", zorder=5)


def plot_maps(all_rows):
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(16.7, 6.2), layout="constrained")
    stats = {}
    for ax, version in zip(axes, VERSIONS):
        rows = all_rows[version]
        cat, _, _ = grid(rows, version)
        ax.imshow(cat, extent=(-0.27, 0.97, -0.62, 0.62), origin="lower",
                  interpolation="nearest", cmap=cmap, norm=norm, zorder=1)
        format_axis(ax)
        ax.set_title(LABELS[version], fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("World Y (m)")
        endpoint = sum(r["endpoint_pass"] for r in rows)
        motion = sum(r["motion_pass"] for r in rows)
        interior = int(np.sum(cat == 5))
        stats[version] = {"endpoint": endpoint, "motion": motion, "interior": interior}
        ax.text(0.02, 0.02, f"Endpoint {endpoint}/961  |  Motion {motion}/961  |  Interior {interior}/961",
                transform=ax.transAxes, fontsize=9, color="#1e3347",
                bbox=dict(facecolor="white", alpha=0.93, edgecolor="none", pad=5))
    fig.legend(handles=[Patch(color=color, label=label) for color, label in zip(COLORS, (
        "Timeout", "Floor penetration", "Other failure", "Endpoint only",
        "Motion pass", "Motion pass, >=6 cm interior"))],
        loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle("NERO near-table physical closed-loop scan | same home, trajectory and 4 cm grid", fontsize=15)
    fig.savefig(OUT / "workspace_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return stats


def plot_diagnostics(all_rows):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.4), layout="constrained")
    extent = (-0.27, 0.97, -0.62, 0.62)
    for col, version in enumerate(VERSIONS):
        _, position, pose = grid(all_rows[version], version)
        a = axes[0, col]
        im1 = a.imshow(np.maximum(position, 1), extent=extent, origin="lower",
                       cmap="magma", norm=LogNorm(vmin=1, vmax=1000), interpolation="nearest")
        format_axis(a)
        a.set_title(LABELS[version], fontsize=11)
        b = axes[1, col]
        im2 = b.imshow(pose, extent=extent, origin="lower", cmap="viridis",
                       vmin=0, vmax=180, interpolation="nearest")
        format_axis(b)
        if col == 0:
            a.set_ylabel("World Y (m)")
            b.set_ylabel("World Y (m)")
    fig.colorbar(im1, ax=axes[0, :], shrink=0.75, label="Final TCP position error (mm, log scale)")
    fig.colorbar(im2, ax=axes[1, :], shrink=0.75, label="Maximum path full-pose error (deg)")
    fig.suptitle("Failure diagnostics on identical goals and axes", fontsize=15)
    fig.savefig(OUT / "diagnostics_comparison.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def trace(version, x, y):
    path = OUT / f"version_{version}" / f"pilot_{x}_{y}.npz"
    if version in (1, 3) and not path.exists() and y == 0.0:
        path = OUT / f"version_{version}" / f"pilot_{x}_0.npz"
    with np.load(path) as file:
        h = file["history"]
    if version in (1, 3):
        return h[:, 0], h[:, 16:19], h[:, 22:25], h[:, 12]
    return h[:, 0], h[:, 1:4], h[:, 4:7], h[:, 8]


def plot_traces():
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), layout="constrained")
    palette = {1: "#d17369", 3: "#118467", 6: "#6574c4"}
    for row, (x, y) in enumerate(((0.7, 0.0), (0.65, 0.3))):
        for version in VERSIONS:
            t, actual, reference, pose = trace(version, x, y)
            label = f"Version {version}"
            axes[row, 0].plot(t, actual[:, 0], color=palette[version], label=label)
            axes[row, 1].plot(t, actual[:, 2], color=palette[version], label=label)
            axes[row, 2].plot(t, pose, color=palette[version], label=label)
        t, _, ref, _ = trace(1, x, y)
        axes[row, 0].plot(t, ref[:, 0], color="#161f2b", linestyle="--", alpha=0.75, label="Reference")
        axes[row, 1].plot(t, ref[:, 2], color="#161f2b", linestyle="--", alpha=0.75, label="Reference")
        axes[row, 0].axhline(x, color="#8794a0", linestyle=":", linewidth=0.9)
        axes[row, 1].axhline(0.005, color="#8794a0", linestyle=":", linewidth=0.9)
        axes[row, 2].axhline(2, color="#cf4747", linestyle="--", linewidth=1, label="Endpoint limit 2 deg")
        axes[row, 2].axhline(5, color="#e4a029", linestyle=":", linewidth=1, label="Path limit 5 deg")
        axes[row, 0].set_ylabel(f"Goal ({x:.2f}, {y:+.2f}, 0.005) m\nActual X (m)")
        axes[row, 1].set_ylabel("Actual TCP Z (m)")
        axes[row, 2].set_ylabel("Full pose error (deg)")
        for ax in axes[row]:
            ax.set_xlabel("Simulation time (s)")
            ax.grid(alpha=0.22)
            ax.set_xlim(0, 16)
    for ax, title in zip(axes[0], ("X tracking", "Height / floor approach", "Orientation tracking")):
        ax.set_title(title)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Actual measured TCP trajectories | common reference and home", fontsize=15)
    fig.savefig(OUT / "trajectory_comparison.png", dpi=185, bbox_inches="tight")
    plt.close(fig)


def write_report(all_rows, stats):
    lines = ["# 规则控制版本 1、3、6：近桌面工作范围复测", "",
        "三版在同一底座、home、固定末端朝向、目标高度 5 mm、31×31 网格（间距 4 cm）和参考轨迹下独立重跑；每点从 home 出发，先以 0.15 m/s 到目标上方 10 cm，再以 0.03 m/s 下降，最多 16 s。", "",
        "终点要求：位置误差 ≤5 mm、完整姿态误差 ≤2°、指尖离地 0.2–8 mm、不穿地、关节不越限、非邻接自碰间隙 ≥1 mm，连续保持 2 s。运动通过还要求全过程完整姿态误差 ≤5°、实际关节速度峰值 ≤原限制的 1.01 倍。内部余量表示离运动通过区采样边界 ≥6 cm。", "",
        "| 版本 | 终点通过 | 运动通过 | 内部余量 | 失败原因 |", "|---|---:|---:|---:|---|" ]
    for version in VERSIONS:
        rows = all_rows[version]
        reasons = {name: sum(r["reason"] == name for r in rows) for name in sorted({r["reason"] for r in rows})}
        s = stats[version]
        lines.append(f"| {version} | {s['endpoint']}/961 | {s['motion']}/961 | {s['interior']}/961 | {reasons} |")
    lines += ["", "版本 1：位置优先 JIK，原始执行器，位置增益 6、姿态增益 2.5。版本 3：同一 JIK，臂 kp×4、kv×2、有界重力前馈、姿态反馈 ×4。版本 6：完整 6D Jacobian DLS，位置/姿态增益 6，λ=0.04，线速度统一上限 0.90 m/s，原始执行器，无重力前馈和关节加速度限制。三版均使用 0.11 s 关节位置前瞻。", "",
        "本结果是固定初始状态、固定末端朝向和指定进场路径的空载闭环物理试跑范围。失败不等于几何不可达，也不是抓取成功率或真机安全范围。版本 6 来自独立开发工作树，其源码和配置以各目录 manifest 为准。", "",
        "图：`workspace_comparison.png`（同尺度分类图）、`diagnostics_comparison.png`（TCP 与姿态误差）、`trajectory_comparison.png`（两处共同目标的实测轨迹）、`pose_comparison.png`（同一目标在 3 s 和 6 s 的实测机器人姿态）。各版本目录保留原始逐点记录、摘要和运行来源。", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_provenance():
    main = ROOT
    worktree = ROOT.parent / "worktrees/scripted-tracking-20261001"
    common = [
        "src/panda_cable_grasp/policies/scripted.py",
        "src/panda_cable_grasp/env/environment.py",
        "src/panda_cable_grasp/env/kinematics.py",
        "assets/mujoco/nero/nero.xml",
    ]
    paths = {
        "versions_1_3": (main, common + [
            "tools/evaluation/compare_scripted_tracking_fix.py",
            "tools/recording/compare_project_ik_workspace.py",
            "tools/recording/record_improved_workspace.py",
            "tools/recording/retest_workspace_1_3.py",
        ]),
        "version_6": (worktree, common + [
            "tools/recording/record_standard_jik_workspace.py",
        ]),
        "visualization": (main, [
            "tools/recording/render_retest_136.py",
            "tools/recording/render_retest_pose_stills.py",
            "tools/recording/capture_retest_version6.py",
        ]),
    }
    result = {}
    for group, (base, relatives) in paths.items():
        result[group] = {str(base / relative): hashlib.sha256((base / relative).read_bytes()).hexdigest()
                         for relative in relatives}
    (OUT / "source_provenance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    rows = load()
    stats = plot_maps(rows)
    plot_diagnostics(rows)
    plot_traces()
    write_report(rows, stats)
    write_provenance()
    print(json.dumps(stats, indent=2), flush=True)
