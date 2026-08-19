"""Process-wide runtime defaults shared by command-line entry points.

This module must be imported before :mod:`mujoco` on Linux so the rendering
backend is selected before PyOpenGL creates a context.
"""

from __future__ import annotations

import os
import sys


def configure_mujoco_runtime() -> str | None:
    """Select a sensible headless backend without overriding user settings.

    ``MUJOCO_GL`` remains the authoritative setting.  On a Linux process with
    no display, EGL is the default because server experiments normally use a
    GPU.  CPU-only machines can set ``MUJOCO_GL=osmesa`` explicitly.
    """

    if "MUJOCO_GL" not in os.environ:
        requested = os.environ.get("PANDA_CABLE_GL_BACKEND", "").strip()
        if requested:
            os.environ["MUJOCO_GL"] = requested
        elif sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            os.environ["MUJOCO_GL"] = "egl"

    backend = os.environ.get("MUJOCO_GL")
    if backend in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    return backend


configure_mujoco_runtime()
