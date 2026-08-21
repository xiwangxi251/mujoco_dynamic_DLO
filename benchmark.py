"""Legacy launcher and import bridge for the benchmark module."""

from _bootstrap import bootstrap
from _compat import reexport

bootstrap()
_implementation = reexport(
    "panda_cable_grasp.evaluation.benchmark", globals()
)


if __name__ == "__main__":
    _implementation.main()

