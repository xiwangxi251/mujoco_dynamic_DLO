from types import SimpleNamespace
import numpy as np
import pytest

from panda_cable_grasp.env.environment import CableGraspEnv
from panda_cable_grasp.env.force_replay import ForceReplayTape, supported_force_distribution


def fixture():
    env=CableGraspEnv.__new__(CableGraspEnv)
    env.config=SimpleNamespace(motion_profile_version='rigid_replay_v1', replay_controller_version='force_tape_v5', replay_time_scale=1.)
    env.cable_ids=np.array([1,2]); env.cable_mass=np.array([.1,.2])
    env.model=SimpleNamespace(opt=SimpleNamespace(timestep=.002))
    env.data=SimpleNamespace(time=.002, xfrc_applied=np.zeros((3,6)))
    forces=np.arange(24,dtype=float).reshape(4,2,3)
    forces.setflags(write=False)
    env._replay_force_tape=ForceReplayTape(forces,.002,0.,'test')
    env._last_ungated_scene_force=np.zeros((2,3))
    env._last_ungated_rigid_force=np.zeros((2,3))
    env._last_replay_force_grounded=np.zeros(2,dtype=bool)
    for name in ('_last_shape_acceleration','_last_rigid_translation_acceleration','_last_rigid_rotation_acceleration','_last_rigid_shape_hold_acceleration'):
        setattr(env,name,np.zeros((2,3)))
    env._cable_table_contact_mask=lambda: np.array([True,False])
    return env


def test_material_node_gating_and_return_use_current_force_only():
    env=fixture()
    env._apply_cable_disturbance()
    np.testing.assert_array_equal(env.data.xfrc_applied[1,:3],[6.,7.,8.])
    np.testing.assert_array_equal(env.data.xfrc_applied[2],0.)
    env.data.time=.006; env.data.xfrc_applied[:]=0.
    env._cable_table_contact_mask=lambda: np.array([True,True])
    env._apply_cable_disturbance()
    np.testing.assert_array_equal(env.data.xfrc_applied[2,:3],[21.,22.,23.])
    assert env._replay_elapsed == pytest.approx(.008)


def test_tape_integer_tick_and_copy_protect_force_identity():
    tape=fixture()._replay_force_tape
    assert tape.index(.004-1e-16)==2
    sample=tape.sample(.002); sample[:]=0.
    assert tape.forces[1].sum()>0
    with pytest.raises(ValueError): tape.sample(.008)


def test_grasp_obstruction_never_trigger_reanchor_or_suspension():
    env=fixture()
    env._replay_gripper_contact_count_now=lambda: 10
    env._replay_was_suspended=True
    env.rigid_motion_suspended=True
    env._replay_contact_blocked=True
    env._update_rigid_motion_suspension_state()
    assert not env.rigid_motion_suspended and not env._replay_contact_blocked
    assert not env._replay_was_suspended


def test_compiler_preserves_wrench_using_only_supported_nodes():
    positions=np.array([[-.1,0.,0.],[0.,0.,0.],[.1,0.,0.]])
    original=np.array([[1.,2.,0.],[3.,1.,0.],[-1.,4.,0.]])
    grounded=np.array([True,False,True])
    result=supported_force_distribution(positions,original,grounded)
    np.testing.assert_array_equal(result[1],0.)
    np.testing.assert_allclose(result.sum(axis=0),original.sum(axis=0),atol=1e-12)
    np.testing.assert_allclose(np.cross(positions,result).sum(axis=0),np.cross(positions,original).sum(axis=0),atol=1e-12)
    np.testing.assert_array_equal(supported_force_distribution(positions,original,np.zeros(3,dtype=bool)),0.)
