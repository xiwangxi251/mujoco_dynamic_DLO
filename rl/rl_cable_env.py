"""Compatibility import for the reorganized RL environment."""

from _bootstrap import bootstrap

bootstrap()
from panda_cable_grasp.rl.environment import *

