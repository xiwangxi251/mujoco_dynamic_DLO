"""Compatibility launcher for DynamicVLA dataset conversion."""

from _bootstrap import bootstrap
from _compat import reexport

bootstrap()
_implementation = reexport(
    "panda_cable_grasp.dynamicvla.finetune.convert_dataset", globals()
)


if __name__ == "__main__":
    _implementation.main()

