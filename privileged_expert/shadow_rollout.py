"""Compatibility import for privileged shadow rollouts."""

from _bootstrap import bootstrap

bootstrap()
from panda_cable_grasp.expert.shadow_rollout import *

