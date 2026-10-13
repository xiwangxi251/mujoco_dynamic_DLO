from dataclasses import replace

import numpy as np
import pytest

from panda_cable_grasp.env.environment import EnvConfig
from panda_cable_grasp.env.surface_drive import surface_accelerations
from panda_cable_grasp.scenarios import SCENARIO_FAMILIES, get_scenario


def drive(xyz, velocities, mask, **extra):
    kwargs=dict(velocity_gain=32.,max_acceleration=8.,yaw_velocity_gain=9.,reference=xyz[:,:2].copy(),
                shape_stiffness=1000.,shape_damping=68.)
    kwargs.update(extra)
    return surface_accelerations(xyz,velocities,np.array([1.,2.,3.,4.]),mask,np.array([0.,.25]),.1,**kwargs)


def test_airborne_velocity_and_position_do_not_influence_supported_feedback():
    xyz=np.array([[0.,0.,0.],[.1,0.,0.],[.2,0.,.2],[.3,0.,.3]])
    velocities=np.zeros_like(xyz)
    mask=np.array([True,True,False,False])
    before=drive(xyz,velocities,mask)
    shifted=xyz.copy();shifted[~mask]+=100.
    velocities[~mask]=[100.,-100.,200.]
    after=drive(shifted,velocities,mask,reference=xyz[:,:2])
    for a,b in zip(before,after):
        np.testing.assert_allclose(a,b,atol=1e-12)
        np.testing.assert_array_equal(b[~mask],0.)


def test_translation_does_not_create_catchup_force():
    xyz=np.array([[0.,0.,0.],[.1,0.,0.],[.2,0.,0.],[.3,0.,0.]])
    mask=np.ones(4,dtype=bool)
    velocities=np.zeros_like(xyz)
    before=drive(xyz,velocities,mask)
    xyz[:,:2]+=[-10.,10.]
    after=drive(xyz,velocities,mask)
    for a,b in zip(before,after):np.testing.assert_allclose(a,b,atol=1e-10)


def test_feedback_brakes_overspeed_and_never_exceeds_translation_limit():
    xyz=np.zeros((4,3));velocities=np.zeros_like(xyz);velocities[:,1]=2.
    translation,_,_=drive(xyz,velocities,np.ones(4,dtype=bool),reference=None)
    np.testing.assert_allclose(translation[:,1],-8.)


def test_all_airborne_is_zero_and_one_supported_node_has_no_spin():
    xyz=np.arange(12,dtype=float).reshape(4,3);velocities=np.zeros_like(xyz)
    for result in drive(xyz,velocities,np.zeros(4,dtype=bool)):np.testing.assert_array_equal(result,0.)
    _,rotation,hold=drive(xyz,velocities,np.array([True,False,False,False]))
    np.testing.assert_array_equal(rotation,0.);np.testing.assert_array_equal(hold,0.)


def test_family_is_separate_and_preserves_old_and_force_scene_ids():
    new=SCENARIO_FAMILIES['cable_x075_surface_v1']
    assert len(new)==7
    assert not set(new)&set(SCENARIO_FAMILIES['cable_x075_v1'])
    assert not set(new)&set(SCENARIO_FAMILIES['cable_x075_force_v1'])
    for old_name,new_name in zip(SCENARIO_FAMILIES['cable_x075_v1'],new):
        old,scene=get_scenario(old_name),get_scenario(new_name)
        assert old.rigid_drive_version=='feedback_v1'
        assert scene.arm_motion_start_delay==.82
        config=EnvConfig(**scene.to_env_overrides())
        if scene.motion_type.value in {'rigid','combined'}:
            assert config.rigid_drive_version=='surface_velocity_v1'
        else:assert config.rigid_drive_version=='feedback_v1'
    assert get_scenario('dev_rigid_l1_nominal_x075_force_v1').scenario_id=='scenario-v1-5b62e984d48957ad'
    assert get_scenario('dev_rigid_replay_shape_nominal_x075_force_v1').scenario_id=='scenario-v1-5326fe3c5575a9f1'


def test_surface_configuration_rejects_unsupported_drivers():
    with pytest.raises(ValueError,match='surface_velocity_v1'):EnvConfig(rigid_drive_version='surface_velocity_v1')
    replay=EnvConfig(**get_scenario('dev_rigid_replay_shape_nominal_x075_surface_v1').to_env_overrides())
    with pytest.raises(ValueError,match='surface_velocity_v1'):replace(replay,replay_controller_version='force_tape_v5')
    with pytest.raises(ValueError,match='surface_velocity_v1'):replace(replay,motion_profile_version='rigid_level2_single_pass_v2',replay_bank=None)
