"""Visual Diffusion Policy for the Panda dynamic-cable task.

The package intentionally keeps the PyTorch import lazy.  The core project,
benchmark utilities, and command discovery remain usable when the optional
``diffusion`` extra has not been installed.
"""

from .config import DiffusionPolicyConfig

__all__ = ["DiffusionPolicyConfig"]
