"""Compatibility launcher for expert experiments."""

from _bootstrap import bootstrap
from _compat import reexport

bootstrap()
_implementation = reexport("panda_cable_grasp.expert.run_experiment", globals())


if __name__ == "__main__":
    _implementation.main()

