"""Contact gating must not depend on confirmation, hold duration or jaw command."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from panda_cable_grasp.env.environment import CableGraspEnv, GraspState


def make_env(version="continuous_ff_v3"):
    env = CableGraspEnv.__new__(CableGraspEnv)
    env.config = SimpleNamespace(
        motion_profile_version="rigid_replay_v1",
        replay_controller_version=version,
        grasp_loss_seconds=.35,
    )
    # finger, palm, cable, table, unrelated arm geom.
    env.model = SimpleNamespace(geom_bodyid=np.array([1, 2, 3, 4, 5]))
    env.replay_gripper_geom_ids = {0, 1}
    env.cable_set = {3}
    env.data = SimpleNamespace(time=1., ncon=0, contact=[])
    env.grasp_state = None
    env.rigid_motion_released = False
    env._replay_was_suspended = False
    env._replay_last_gripper_contact_time = -np.inf
    env._replay_reanchor = Mock()
    return env


def contacts(env, *pairs):
    env.data.contact = [SimpleNamespace(geom1=a, geom2=b, dist=d) for a, b, d in pairs]
    env.data.ncon = len(pairs)


@pytest.mark.parametrize("pair", [(0, 2, -.001), (2, 0, 0.), (1, 2, -.001)])
def test_unilateral_and_palm_contacts_pause_without_grasp(pair):
    env = make_env()
    contacts(env, pair)
    env._update_rigid_motion_suspension_state()
    assert env.rigid_motion_suspended and env._replay_contact_blocked
    assert not env.grasp_confirmed
    assert not env.rigid_motion_released  # Not a confirmed grasp history event.
    env._replay_reanchor.assert_not_called()


@pytest.mark.parametrize("pair", [(0, 2, .001), (3, 2, -.001), (4, 2, -.001)])
def test_margin_table_and_unrelated_arm_contacts_do_not_pause(pair):
    env = make_env()
    contacts(env, pair)
    env._update_rigid_motion_suspension_state()
    assert not env.rigid_motion_suspended


def test_contact_chatter_and_long_hold_reanchor_only_after_clear_interval():
    env = make_env()
    contacts(env, (0, 2, -.001))
    env._update_rigid_motion_suspension_state()
    env.data.time = 11.  # Long obstruction cannot time out while contact remains.
    env._update_rigid_motion_suspension_state()
    contacts(env)
    env.data.time = 11.02
    env._update_rigid_motion_suspension_state()
    assert env.rigid_motion_suspended
    contacts(env, (0, 2, -.001))
    env._update_rigid_motion_suspension_state()
    contacts(env)
    env.data.time = 11.058
    env._update_rigid_motion_suspension_state()
    assert env.rigid_motion_suspended
    env.data.time = 11.06
    env._update_rigid_motion_suspension_state()
    assert not env.rigid_motion_suspended
    env._replay_reanchor.assert_called_once()
    env.data.time += .1
    env._update_rigid_motion_suspension_state()
    env._replay_reanchor.assert_called_once()


def test_confirmed_grasp_to_unilateral_contact_does_not_resume_early():
    env = make_env()
    env.grasp_state = GraspState(
        body_id=3, candidate_time=1., bilateral_confirmed=True,
        last_bilateral_time=1., lost_contact_time=1.,
    )
    env._update_rigid_motion_suspension_state()
    env.grasp_state = None
    contacts(env, (0, 2, -.001))
    env.data.time = 1.5
    env._update_rigid_motion_suspension_state()
    assert env.rigid_motion_suspended
    env._replay_reanchor.assert_not_called()
    contacts(env)
    env.data.time = 1.54
    env._update_rigid_motion_suspension_state()
    env._replay_reanchor.assert_called_once()


@pytest.mark.parametrize("version", ["legacy_pd_v1", "continuous_ff_v2"])
def test_saved_controller_versions_keep_contact_semantics(version):
    env = make_env(version)
    contacts(env, (0, 2, -.001))
    env._update_rigid_motion_suspension_state()
    assert not env.rigid_motion_suspended
    assert not env._replay_contact_blocked
