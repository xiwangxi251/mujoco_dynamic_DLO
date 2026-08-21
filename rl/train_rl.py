"""Compatibility launcher for ``python -m rl.train_rl``."""

from _bootstrap import bootstrap
from _compat import reexport

bootstrap()
_implementation = reexport("panda_cable_grasp.rl.train", globals())


if __name__ == "__main__":
    _implementation.main()

