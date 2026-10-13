import copy
import numpy as np
import pytest
from panda_cable_grasp.rl.motion_response import MotionResponse


def profile():
    return dict(version='measured_motion_response_v1',
        validation=dict(status='passed',independent_physics_validation=True),
        scenes={'shape':dict(axes=['shape_amplitude_mm','shape_speed_m_s'],
            bounds=dict(shape_amplitude_mm=[40,80],shape_speed_m_s=[.1,.2]),
            driver_parameters=['disturbance_strength','motion_frequency_scale'],
            measured_outputs=[[40,.1],[40,.2],[80,.1],[80,.2]],
            driver_values=[[1,.5],[1.5,.8],[2,.7],[2.5,1.]])})


def test_joint_inverse_interpolates_continuously_not_independent_knobs():
    response=MotionResponse(profile())
    a=response.solve('shape',dict(shape_amplitude_mm=60,shape_speed_m_s=.15))
    assert a==pytest.approx(dict(disturbance_strength=1.75,motion_frequency_scale=.75))
    faster=response.solve('shape',dict(shape_amplitude_mm=60,shape_speed_m_s=.151))
    assert faster['disturbance_strength']>a['disturbance_strength']
    assert faster['motion_frequency_scale']>a['motion_frequency_scale']
    assert faster['disturbance_strength']-a['disturbance_strength']<.01


def test_uncovered_corner_and_unvalidated_map_cannot_be_used():
    bad=profile();bad['validation']['status']='pending'
    with pytest.raises(ValueError,match='acceptance'): MotionResponse(bad)
    bad=profile();bad['scenes']['shape']['measured_outputs'].pop()
    bad['scenes']['shape']['driver_values'].pop()
    with pytest.raises(ValueError,match='not covered'): MotionResponse(bad)
    with pytest.raises(ValueError,match='outside'):
        MotionResponse(profile()).solve('shape',dict(shape_amplitude_mm=90,shape_speed_m_s=.15))


def test_whole_motion_speed_uses_one_dimensional_measured_response():
    p=profile();p['scenes']['rigid']=dict(axes=['rigid_speed_m_s'],
        bounds=dict(rigid_speed_m_s=[.15,.2]),driver_parameters=['rigid_motion_nominal_speed'],
        measured_outputs=[[.15],[.2]],driver_values=[[.19],[.25]])
    assert MotionResponse(p).solve('rigid',{'rigid_speed_m_s':.175})==pytest.approx(
        dict(rigid_motion_nominal_speed=.22))


def test_combined_inverse_has_independent_actual_axes_and_joint_drivers():
    import itertools
    p=profile()
    points=list(itertools.product([40,80],[.1,.2],[.15,.2]))
    p['scenes']['combined']=dict(axes=['shape_amplitude_mm','shape_speed_m_s','rigid_speed_m_s'],
        bounds=dict(shape_amplitude_mm=[40,80],shape_speed_m_s=[.1,.2],rigid_speed_m_s=[.15,.2]),
        driver_parameters=['disturbance_strength','motion_frequency_scale','rigid_motion_nominal_speed'],
        measured_outputs=points,
        driver_values=[[a/40+v,v*3+r,r*1.2] for a,v,r in points])
    result=MotionResponse(p).solve('combined',dict(shape_amplitude_mm=60,shape_speed_m_s=.15,rigid_speed_m_s=.175))
    assert result==pytest.approx(dict(disturbance_strength=1.65,motion_frequency_scale=.625,rigid_motion_nominal_speed=.21))


def test_range_episode_samples_actual_targets_and_preserves_original_drive(monkeypatch):
    from panda_cable_grasp.rl import range_curriculum as curriculum
    from panda_cable_grasp.scenarios.registry import get_scenario
    name=curriculum.ALL_SCENES[-2]
    p=profile();p['scenes'][name]=p['scenes'].pop('shape')
    monkeypatch.setattr(curriculum,'_OUTPUT_RESPONSE',MotionResponse(p))
    episode=curriculum.sample_episode(get_scenario(name),np.random.default_rng(3),
        dict(probabilities=[.25]*4),4)
    assert episode['axes']==('shape_amplitude_mm','shape_speed_m_s')
    assert set(episode['parameters'])=={'shape_amplitude_mm','shape_speed_m_s'}
    assert 40<=episode['parameters']['shape_amplitude_mm']<=80
    assert .1<=episode['parameters']['shape_speed_m_s']<=.2
    assert set(episode['internal_driver_parameters'])=={'disturbance_strength','motion_frequency_scale'}
    assert 'shape_force_gate' not in episode['overrides']
    assert 'shape_pose_hold' not in episode['overrides']


def test_shape_force_remains_active_without_table_support_and_during_grasp():
    from types import SimpleNamespace
    from panda_cable_grasp.env.environment import CableGraspEnv
    shape=np.array([[1.,2.,3.],[-1.,-2.,-3.]])
    obj=SimpleNamespace(_uses_force_replay=False,
        config=SimpleNamespace(_uses_object_scene=False,motion_profile_version='factorized_v2',
            motion_frequency_scale=1.,motion_mode='shape',rigid_motion_contact_only=True),
        data=SimpleNamespace(time=1.,xfrc_applied=np.zeros((2,6))),
        phase_offset=0.,spatial_phase=0.,cable_ids=np.array([0,1]),cable_mass=np.array([1.,2.]),
        rigid_motion_suspended=True,_shape_acceleration=lambda t,p:shape.copy())
    for name in ('_last_shape_acceleration','_last_rigid_translation_acceleration',
        '_last_rigid_rotation_acceleration','_last_rigid_shape_hold_acceleration',
        '_last_ungated_scene_force','_last_ungated_rigid_force'):
        setattr(obj,name,np.zeros((2,3)))
    def forbidden(): raise AssertionError('Shape drive must not use table contact gating')
    obj._cable_table_contact_mask=forbidden
    CableGraspEnv._apply_cable_disturbance(obj)
    assert np.array_equal(obj.data.xfrc_applied[:,:3],shape*obj.cable_mass[:,None])
