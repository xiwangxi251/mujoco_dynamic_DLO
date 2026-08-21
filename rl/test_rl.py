"""Compatibility launcher for ``python -m rl.test_rl``."""

from _bootstrap import bootstrap
from _compat import reexport

bootstrap()
_implementation = reexport("panda_cable_grasp.rl.evaluate", globals())


if __name__ == "__main__":
    _implementation.main()

