"""Render the measured workspace and summarize a synced training snapshot."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    with np.load(run / "workspace/workspace.npz") as file:
        xs, ys = file["xs"], file["ys"]
        categories = file["endpoint_pass"].astype(int)
        categories[file["motion_pass"]] = 2
        categories[file["interior"]] = 3
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(categories, origin="lower", extent=[xs[0]-.02,xs[-1]+.02,ys[0]-.02,ys[-1]+.02],
              cmap=ListedColormap(["#e6edf3","#89c0ef","#44b692","#147c66"]), vmin=0, vmax=3)
    ax.scatter([.35],[0],c="#263f52",s=30,label="Robot base")
    verification = run / "prelaunch_verification.json"
    if verification.exists():
        data = json.loads(verification.read_text())
        examples = data.get("initial_goal_positions", [])
        if examples:
            positions = np.array(examples)
            ax.scatter(positions[:,0],positions[:,1],marker="x",c="#d84b37",label="Sampled goals")
    ax.set(xlabel="World X (m)",ylabel="World Y (m)",title="Full 6D JIK + gravity + kp x4 / kv x2 + orientation gain 10\n205 endpoint / 159 motion / 84 interior cells")
    ax.grid(alpha=.15)
    ax.legend(loc="lower right")
    fig.text(.12,.035,"Light blue: endpoint | Green: motion | Dark green: sampling region (6 cm margin)",fontsize=9)
    fig.savefig(run / "workspace/workspace_sampling.png",dpi=170,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
