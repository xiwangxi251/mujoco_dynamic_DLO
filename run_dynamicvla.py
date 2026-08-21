"""Legacy launcher for :mod:`panda_cable_grasp.cli.run_dynamicvla`."""

from _bootstrap import bootstrap

bootstrap()
from panda_cable_grasp.cli.run_dynamicvla import main


if __name__ == "__main__":
    main()

