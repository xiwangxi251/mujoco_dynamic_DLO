"""Distinguish sustained obstruction from contact; preserve historical v3."""
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
    env._replay_contact_blocked = False
    env._replay_obstruction_started_at = None
    env._replay_obstruction_geom_pairs = set()
    env._replay_obstruction_still_near = Mock(return_value=False)
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


def test_v4_short_touch_does_not_pause_or_reanchor():
    env = make_env("continuous_ff_v4")
    env._replay_obstruction_evidence = Mock(return_value=True)
    contacts(env, (0, 2, -.001))
    env._update_rigid_motion_suspension_state()
    env.data.time = 1.098
    env._update_rigid_motion_suspension_state()
    assert not env.rigid_motion_suspended
    contacts(env)
    env.data.time = 1.1
    env._update_rigid_motion_suspension_state()
    assert not env.rigid_motion_suspended
    assert env._replay_obstruction_started_at is None
    env._replay_reanchor.assert_not_called()


def test_v4_sliding_contact_does_not_pause_even_when_persistent():
    env = make_env("continuous_ff_v4")
    env._replay_obstruction_evidence = Mock(return_value=False)
    contacts(env, (0, 2, -.001))
    for time in (1., 1.1, 2., 11.):
        env.data.time = time
        env._update_rigid_motion_suspension_state()
        assert not env.rigid_motion_suspended
    env._replay_reanchor.assert_not_called()


def test_v4_obstruction_requires_continuous_evidence_then_reanchors_on_clear():
    env = make_env("continuous_ff_v4")
    env._replay_obstruction_evidence = Mock(return_value=True)
    contacts(env, (0, 2, -.001))
    env._update_rigid_motion_suspension_state()
    env.data.time = 1.08
    env._replay_obstruction_evidence.return_value = False
    env._update_rigid_motion_suspension_state()
    env._replay_obstruction_evidence.return_value = True
    for time in (1.10, 1.198):
        env.data.time = time
        env._update_rigid_motion_suspension_state()
        assert not env.rigid_motion_suspended
    env.data.time = 1.20
    env._update_rigid_motion_suspension_state()
    assert env.rigid_motion_suspended
    env.data.time = 11.20
    env._update_rigid_motion_suspension_state()
    assert env.rigid_motion_suspended
    contacts(env)
    env.data.time = 11.238
    env._update_rigid_motion_suspension_state()
    assert env.rigid_motion_suspended
    env.data.time = 11.24
    env._update_rigid_motion_suspension_state()
    assert not env.rigid_motion_suspended
    env._replay_reanchor.assert_called_once()


@pytest.mark.parametrize("speed,actual,lag,expected", [
    (.1, .0, .02, True),     # Actually held back.
    (.1, -.1, .02, True),    # Pushed against the direction of travel.
    (.1, .08, .02, False),   # Sliding contact with good progress.
    (.1, .0, .001, False),   # Little tracking lag.
    (.01, .0, .02, False),   # Naturally slow reference.
    (.1, .0, -.02, False),   # Cable already ahead of reference.
])
def test_v4_uses_material_progress_and_lag(speed, actual, lag, expected):
    env = make_env("continuous_ff_v4")
    contacts(env, (0, 2, -.001), (0, 2, -.001))
    env.cable_index = {3: 0}
    env._rigid_reference_xy = np.zeros((1, 2))
    env._rigid_reference_com_xy = np.zeros(2)
    env.data.xpos = np.zeros((6, 3))
    env.body_linear_velocity = Mock(return_value=np.array([actual, 0., 0.]))
    env._continuous_replay_targets = lambda: (np.array([lag, 0.]), np.array([speed, 0.]), None, 0., 0., 0.)
    assert env._replay_obstruction_evidence() is expected


def test_v4_rotation_can_be_obstructed_even_when_com_is_stationary():
    env = make_env("continuous_ff_v4")
    contacts(env, (0, 2, -.001))
    env.cable_index = {3: 0}
    env._rigid_reference_xy = np.array([[.1, 0.]])
    env._rigid_reference_com_xy = np.zeros(2)
    env.data.xpos = np.zeros((6, 3))
    env.data.xpos[3, :2] = [.1, -.02]
    env.body_linear_velocity = Mock(return_value=np.zeros(3))
    env._continuous_replay_targets = lambda: (np.zeros(2), np.zeros(2), None, 0., 1., 0.)
    assert env._replay_obstruction_evidence()
    assert env._replay_motion_progress_ratio == 0.
    assert env._replay_motion_lag_m == pytest.approx(.02)
