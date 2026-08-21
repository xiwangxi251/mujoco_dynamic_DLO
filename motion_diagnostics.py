"""Legacy launcher and import bridge for motion diagnostics."""

from _bootstrap import bootstrap
from _compat import reexport

bootstrap()
_implementation = reexport(
    "panda_cable_grasp.evaluation.motion_diagnostics", globals()
)


if __name__ == "__main__":
    _implementation.main()

