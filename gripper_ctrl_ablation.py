"""Legacy launcher for the gripper-control ablation tool."""

from _bootstrap import bootstrap
from _compat import reexport

bootstrap()
_implementation = reexport("tools.ablations.gripper_control", globals())


if __name__ == "__main__":
    _implementation.main()

