"""Project regression tests with a local ``src`` checkout on ``sys.path``."""

from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
source = str(SRC_ROOT)
if source not in sys.path:
    sys.path.insert(0, source)

from panda_cable_grasp.runtime import configure_mujoco_runtime

configure_mujoco_runtime()
