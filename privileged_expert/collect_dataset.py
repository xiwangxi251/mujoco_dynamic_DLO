"""Compatibility launcher for privileged-expert dataset collection."""

from _bootstrap import bootstrap
from _compat import reexport

bootstrap()
_implementation = reexport("panda_cable_grasp.expert.collect_dataset", globals())


if __name__ == "__main__":
    _implementation.main()

