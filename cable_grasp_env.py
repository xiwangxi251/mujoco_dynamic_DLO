"""Compatibility import; use :mod:`panda_cable_grasp.env` in new code."""

from _bootstrap import bootstrap

bootstrap()
from panda_cable_grasp.env.environment import *

