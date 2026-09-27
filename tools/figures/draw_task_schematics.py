"""Draw restrained, editable vector task schematics for a research paper.

Run: conda run -n mujoco-dlo python tools/figures/draw_task_schematics.py
The drawings are conceptual and do not encode measured trajectories.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath


OUT = Path(__file__).resolve().parent / "schematics"
INK = "#263844"
SECONDARY = "#667782"
FAINT = "#BFCAD0"
HAIR = "#D9E0E3"
OBJECT = "#BF5B44"
OBJECT_LIGHT = "#F4DED8"
ACCENT = "#37828B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def canvas(ax, letter: str, title: str):
    ax.set_xlim(-0.05, 3.25)
    ax.set_ylim(-0.12, 2.80)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.03, 2.57, f"({letter})", ha="left", va="center",
            fontsize=9.5, weight="bold", color=INK)
    ax.text(0.50, 2.57, title, ha="left", va="center",
            fontsize=9.2, color=INK)
    ax.plot([0.02, 3.12], [2.38, 2.38], color=HAIR, lw=0.8)


def arrow(ax, start, end, *, color=ACCENT, lw=1.1, size=9,
          rad=0, style="-|>", z=8):
    ax.add_patch(patches.FancyArrowPatch(
        start, end, arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=size, linewidth=lw,
        color=color, zorder=z,
    ))


def curve(ax, vertices, *, color=OBJECT, lw=3.7,
          dashed=False, alpha=1, z=5):
    path = MplPath(vertices, [MplPath.MOVETO] + [MplPath.CURVE4] * (len(vertices) - 1))
    ax.add_patch(patches.PathPatch(
        path, facecolor="none", edgecolor=color, linewidth=lw,
        capstyle="round", joinstyle="round", alpha=alpha,
        linestyle=(0, (3.5, 3.5)) if dashed else "solid", zorder=z,
    ))


def clip(ax, x: float, y: float, *, rail_z=7):
    # Plan-view snap retainer: cable is held between the two lips.
    ax.add_patch(patches.FancyBboxPatch(
        (x - 0.32, y - 0.29), 0.64, 0.58,
        boxstyle="round,pad=0,rounding_size=0.045",
        facecolor="white", edgecolor=FAINT, lw=0.9, zorder=2,
    ))
    for dy in (-0.22, 0.15):
        ax.add_patch(patches.Rectangle(
            (x - 0.25, y + dy), 0.50, 0.07,
            facecolor="#E7ECEF", edgecolor=INK, lw=0.95, zorder=rail_z,
        ))
    for dx in (-0.20, 0.15):
        for dy in (-0.15, 0.10):
            ax.add_patch(patches.Rectangle(
                (x + dx, y + dy), 0.05, 0.05,
                facecolor=INK, edgecolor="none", zorder=rail_z + 1,
            ))


def static_cube(ax):
    canvas(ax, "a", "Stationary rigid object")
    ax.add_patch(patches.Rectangle((1.22, 0.83), 0.78, 0.78,
                                   facecolor=OBJECT_LIGHT, edgecolor=OBJECT,
                                   lw=1.45, zorder=5))
    ax.plot([1.22, 2.00], [1.61, 0.83], color=OBJECT, lw=0.75, alpha=0.5, zorder=6)
    ax.text(1.61, 1.87, r"$\mathbf{v}=0$", ha="center", color=SECONDARY, fontsize=8)
    ax.text(1.61, 0.60, "cube", ha="center", color=SECONDARY, fontsize=7.5)


def rolling_ball(ax):
    canvas(ax, "b", "Moving rigid object")
    for x, alpha in ((0.77, 0.28), (1.28, 0.53)):
        ax.add_patch(patches.Circle((x, 0.93), 0.22, facecolor="none",
                                    edgecolor=OBJECT, lw=1.1,
                                    alpha=alpha, zorder=3))
    ax.add_patch(patches.Circle((1.76, 0.93), 0.22,
                                facecolor=OBJECT_LIGHT, edgecolor=OBJECT,
                                lw=1.4, zorder=5))
    ax.add_patch(patches.Circle((2.28, 0.93), 0.25, facecolor="none",
                                edgecolor=ACCENT, lw=1.1,
                                linestyle=(0, (3, 2)), zorder=4))
    arrow(ax, (1.80, 0.45), (2.30, 0.45), size=8)
    ax.text(2.30, 0.21, "future position", ha="center", color=SECONDARY, fontsize=7.5)


def routing(ax):
    canvas(ax, "c", "Cable routing in clips")
    # Bases first, cable second, retaining lips last.
    for x in (1.06, 2.49):
        ax.add_patch(patches.FancyBboxPatch(
            (x - 0.32, 0.84), 0.64, 0.58,
            boxstyle="round,pad=0,rounding_size=0.045",
            facecolor="white", edgecolor=FAINT, lw=0.9, zorder=2))
    curve(ax, [(0.06, 0.54), (0.55, 0.56), (0.72, 1.13), (1.14, 1.13),
               (1.50, 1.13), (1.79, 1.13), (2.04, 1.13)], lw=3.7)
    clip(ax, 1.06, 1.13)
    clip(ax, 2.49, 1.13)
    ax.add_patch(patches.Circle((2.04, 1.13), 0.045,
                                facecolor=OBJECT, edgecolor="none", zorder=8))
    arrow(ax, (2.04, 1.72), (2.49, 1.72), size=8)
    ax.text(1.06, 0.59, "seated", ha="center", color=SECONDARY, fontsize=7.5)
    ax.text(2.49, 0.59, "next", ha="center", color=SECONDARY, fontsize=7.5)


def dynamic_cable(ax):
    canvas(ax, "d", "Moving deformable object")
    curve(ax, [(0.74, 0.70), (1.10, 0.49), (1.34, 1.39), (1.69, 1.03),
               (2.05, 0.77), (2.15, 1.68), (2.59, 1.47)],
          color=FAINT, lw=2.4, dashed=True, z=2)
    curve(ax, [(0.93, 0.89), (1.27, 0.62), (1.47, 1.63), (1.82, 1.34),
               (2.13, 1.04), (2.27, 1.98), (2.62, 1.83)],
          color=OBJECT, lw=3.8, z=5)
    ax.add_patch(patches.Circle((1.76, 1.34), 0.045,
                                facecolor="white", edgecolor=OBJECT,
                                lw=1.25, zorder=6))
    arrow(ax, (1.07, 2.08), (1.61, 2.08), size=8)
    ax.text(1.35, 2.22, "global motion", ha="center",
            color=SECONDARY, fontsize=7.1)
    arrow(ax, (1.93, 1.43), (2.20, 1.68),
          color=OBJECT, size=7.5, rad=-0.30)
    ax.text(2.21, 0.50, "shape change", ha="center",
            color=SECONDARY, fontsize=7.1)
    ax.text(0.84, 0.51, r"$t$", color=SECONDARY, fontsize=7.5)
    ax.text(2.58, 1.96, r"$t+\Delta t$", color=OBJECT, fontsize=7.5, ha="right")


PANELS = [
    ("a_static_object", static_cube),
    ("b_moving_object", rolling_ball),
    ("c_cable_routing", routing),
    ("d_dynamic_cable", dynamic_cable),
]


def export(fig, stem: str):
    for extension in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=400,
                    bbox_inches="tight", pad_inches=0.025)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, drawer in PANELS:
        fig, ax = plt.subplots(figsize=(3.35, 2.90))
        fig.subplots_adjust(0, 0, 1, 1)
        drawer(ax)
        export(fig, stem)
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 2.90))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.98, wspace=0.035)
    for ax, (_, drawer) in zip(axes, PANELS):
        drawer(ax)
    export(fig, "task_schematics_4panel")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.8))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.98,
                        wspace=0.05, hspace=0.06)
    for ax, (_, drawer) in zip(axes.flat, PANELS):
        drawer(ax)
    export(fig, "task_schematics_2x2")
    plt.close(fig)
    print(f"Wrote paper-style schematics to {OUT}")


if __name__ == "__main__":
    main()
