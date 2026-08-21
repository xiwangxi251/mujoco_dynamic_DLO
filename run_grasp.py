"""Legacy launcher for :mod:`panda_cable_grasp.cli.run_grasp`."""

from _bootstrap import bootstrap

bootstrap()
from panda_cable_grasp.cli.run_grasp import main


if __name__ == "__main__":
    main()

