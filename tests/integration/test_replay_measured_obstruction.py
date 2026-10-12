"""Real solver contact: a bead slides into a stationary gripper obstacle.

This minimal model isolates detection/controller interaction. It is not a
robot grasp benchmark; full robot regressions live in test_rigid_replay_env.
"""
from types import SimpleNamespace
import math

import mujoco
import numpy as np

from panda_cable_grasp.env.environment import CableGraspEnv


def make_fixture():
    env = CableGraspEnv.__new__(CableGraspEnv)
    env.model = mujoco.MjModel.from_xml_string('''
      <mujoco><option timestep="0.002"/>
        <worldbody>
          <geom name="floor" type="plane" size="1 1 .1" friction="0 0 0"/>
          <body name="finger" mocap="true" pos="0 0 .1">
            <geom name="obstacle" type="box" size=".01 .2 .1" friction="0 0 0"/>
          </body>
          <body name="cable" pos="-.05 0 .011"><freejoint/>
            <geom name="bead" type="sphere" size=".01" mass=".1" friction="0 0 0"/>
          </body>
        </worldbody>
      </mujoco>''')
    env.data = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, env.data)
    body = env.model.body('cable').id
    env.cable_ids = np.array([body])
    env.cable_set = {body}
    env.cable_index = {body: 0}
    env.cable_mass = np.array([.1])
    env.replay_gripper_geom_ids = {env.model.geom('obstacle').id}
    env.config = SimpleNamespace(
        motion_profile_version="rigid_replay_v1", replay_controller_version="continuous_ff_v4",
        grasp_loss_seconds=.35, replay_position_gain=800., replay_velocity_gain=56.,
        replay_yaw_position_gain=160., replay_yaw_velocity_gain=24.,
        rigid_translation_max_acceleration=8.,
    )
    env.grasp_state = None
    env.rigid_motion_suspended = False
    env.rigid_motion_released = False
    env._replay_contact_blocked = False
    env._replay_was_suspended = False
    env._replay_last_gripper_contact_time = -math.inf
    env._replay_obstruction_started_at = None
    env._replay_obstruction_geom_pairs = set()
    env._replay_elapsed = 0.
    env._replay_dt = .02
    env._replay_horizon = 1000
    env._replay_anchor_elapsed = 0.
    env._replay_anchor_tracked = np.array([-.05, 0.])
    env._replay_tracked_index = 0
    env._rigid_reference_xy = np.zeros((1, 2))
    env._rigid_reference_com_xy = np.zeros(2)
    env._current_rigid_pose = lambda: (np.zeros(2), 0.)
    env._continuous_replay_targets = lambda: (
        env._replay_anchor_tracked + [.1*(env._replay_elapsed-env._replay_anchor_elapsed), 0.],
        np.array([.1, 0.]), np.zeros(2), 0., 0., 0.,
    )
    return env


def advance(env):
    env.data.xfrc_applied[:] = 0.
    if not env.rigid_motion_suspended:
        velocity = np.array([env.body_linear_velocity(b)[:2] for b in env.cable_ids])
        translation, rotation = env._continuous_replay_motion_acceleration(velocity)
        env.data.xfrc_applied[env.cable_ids, :3] = .1*(translation+rotation)
        env._replay_elapsed += .002
    mujoco.mj_step(env.model, env.data)
    env._update_rigid_motion_suspension_state()


def test_sustained_real_obstruction_freezes_without_repaying_elapsed_time():
    env = make_fixture()
    first_contact = None
    first_blocked = None
    for _ in range(400):
        advance(env)
        if env._replay_gripper_contact_count and first_contact is None:
            first_contact = float(env.data.time)
            assert not env.rigid_motion_suspended  # Initial touch is not pause.
        if env._replay_contact_blocked and first_blocked is None:
            first_blocked = float(env.data.time)
            frozen = env._replay_elapsed
    assert first_contact is not None and first_blocked is not None
    assert first_blocked-first_contact >= .1
    assert env._replay_contact_blocked
    assert env._replay_elapsed == frozen
    # Remove the physical obstacle, then allow clear hysteresis and resume.
    env.data.mocap_pos[0, 1] = 1.
    mujoco.mj_forward(env.model, env.data)
    for _ in range(25):
        advance(env)
    assert not env._replay_contact_blocked
    assert env._replay_anchor_elapsed == frozen
    assert 0. < env._replay_elapsed-frozen < .03


def test_brief_real_contact_keeps_clock_and_drive_running():
    env = make_fixture()
    for _ in range(400):
        advance(env)
        if env._replay_gripper_contact_count:
            break
    else:
        raise AssertionError("fixture never contacted obstacle")
    elapsed = env._replay_elapsed
    for _ in range(10):  # 20 ms impact, then obstacle removed.
        advance(env)
        assert not env.rigid_motion_suspended
    env.data.mocap_pos[0, 1] = 1.
    mujoco.mj_forward(env.model, env.data)
    for _ in range(100):
        advance(env)
        assert not env.rigid_motion_suspended
    assert math.isclose(env._replay_elapsed-elapsed, .22, abs_tol=1e-10)
    assert env._replay_anchor_elapsed == 0.
