import numpy as np
import pytest
from panda_cable_grasp.env.sector_placement import STATIC_SECTOR, sample_rotated_translation
from panda_cable_grasp.scenarios.registry import SCENARIO_FAMILIES, get_scenario


def test_rotated_placement_preserves_shape_and_obeys_strict_coverage():
    x = np.linspace(.3,1.1,40)
    xy = np.column_stack((x,.05*np.sin(np.linspace(0,2*np.pi,40))))
    distances = np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    yaws=[]
    for seed in range(24):
        result=sample_rotated_translation(xy,np.random.default_rng(seed),[-.63,-1.18],[1.73,1.18])
        yaw,shift,coverage,used=result
        r=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        placed=(xy-xy.mean(0))@r.T+xy.mean(0)+shift
        np.testing.assert_allclose(np.linalg.norm(placed[:,None]-placed[None,:],axis=-1),distances,atol=1e-14)
        assert STATIC_SECTOR.contains_xy(placed).mean()==coverage>=.3
        assert used==.3 and np.all(placed>=[-.63,-1.18]) and np.all(placed<=[1.73,1.18])
        again=sample_rotated_translation(xy,np.random.default_rng(seed),[-.63,-1.18],[1.73,1.18])
        assert again[0]==yaw
        np.testing.assert_array_equal(again[1],shift)
        expected=np.random.default_rng(seed).uniform(-np.pi,np.pi)
        assert yaw==expected  # one angle before any placement rejection
        yaws.append(yaw)
    assert np.ptp(yaws)>5.


def test_new_static_protocol_does_not_redefine_old_scenes_or_other_motion():
    old=SCENARIO_FAMILIES['cable_x075_sector_speed_v1']
    new=SCENARIO_FAMILIES['cable_x075_sector_speed_v2']
    assert len(new)==7
    assert get_scenario(old[0]).scenario_id=='scenario-v1-838866fd55bf60cf'
    assert get_scenario(old[4]).scenario_id=='scenario-v1-dfe2ce5166153378'
    for old_name,new_name in zip(old,new):
        a,b=get_scenario(old_name),get_scenario(new_name)
        assert a.scenario_id!=b.scenario_id
        if b.motion_type.value=='static':
            assert b.initial_placement_version=='sector_fraction_yaw_v2'
            assert b.initial_placement_min_fraction==b.initial_placement_fallback_fraction==.3
        else:
            assert a.to_env_overrides()==b.to_env_overrides()


def test_yaw_protocol_cannot_silently_relax_coverage():
    from dataclasses import replace
    from panda_cable_grasp.env.environment import EnvConfig
    s=get_scenario(SCENARIO_FAMILIES['cable_x075_sector_speed_v2'][0])
    with pytest.raises(ValueError,match='strict 30%'):
        replace(s,initial_placement_fallback_fraction=.2)
    with pytest.raises(ValueError,match='strict 30%'):
        EnvConfig(motion_mode='static',initial_placement_version='sector_fraction_yaw_v2')
