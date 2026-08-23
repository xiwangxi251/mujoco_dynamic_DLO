"""Validate model assets, plugins, physics, and optional offscreen rendering."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys

from _bootstrap import bootstrap

bootstrap()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panda_cable_grasp.runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import mujoco

from panda_cable_grasp.env.environment import (
    CableGraspEnv,
    EnvConfig,
    resolve_menagerie_panda_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="check model loading and physics only",
    )
    args = parser.parse_args()

    print(f"platform={platform.platform()}")
    print(f"python={sys.version.split()[0]}")
    print(f"mujoco={mujoco.__version__}")
    print(f"mujoco_gl={os.environ.get('MUJOCO_GL', '<default>')}")
    print(f"menagerie_panda={resolve_menagerie_panda_dir()}")

    env = CableGraspEnv(EnvConfig(
        seed=1,
        episode_seconds=0.1,
        dynamicvla_cameras_enabled=not args.no_render,
    ))
    try:
        observation, _ = env.reset(seed=1)
        observation, _, _, _, _ = env.step(env.ready_ctrl)
        print(f"model_bodies={env.model.nbody} model_geoms={env.model.ngeom}")
        if not args.no_render:
            frames = env.dynamicvla_camera_rgb()
            for name in ("opst_cam", "wrist_cam"):
                frame = frames[name]
                print(f"{name}={frame.shape} dtype={frame.dtype}")
        print("installation_check=OK")
    finally:
        env.close()


if __name__ == "__main__":
    main()
