import numpy as np
import pytest
from panda_cable_grasp.env.sector_placement import STATIC_SECTOR, sample_translation
from panda_cable_grasp.scenarios.registry import SCENARIO_FAMILIES, get_scenario


@pytest.mark.parametrize('shape', ['straight', 'curved'])
def test_sector_placement_keeps_shape_and_table_bounds(shape):
    y = np.linspace(-.45, .45, 40)
    x = .75 + (.07*np.sin(np.linspace(0, 4*np.pi, 40)) if shape == 'curved' else np.zeros(40))
    xy = np.column_stack((x,y))
    for seed in range(12):
        shift, fraction, used = sample_translation(xy, np.random.default_rng(seed), np.array([-1.1,-1.1]), np.array([1.1,1.1]))
        placed = xy + shift
        np.testing.assert_allclose(np.diff(placed,axis=0), np.diff(xy,axis=0), atol=1e-15)
        assert STATIC_SECTOR.contains_xy(placed).mean() == fraction >= used >= .4
        assert np.all(abs(placed) <= 1.1)
        again = sample_translation(xy, np.random.default_rng(seed), np.array([-1.1,-1.1]), np.array([1.1,1.1]))
        np.testing.assert_array_equal(again[0], shift)


def test_new_family_limits_only_static_placement_and_three_rigid_components():
    names = SCENARIO_FAMILIES['cable_x075_sector_speed_v1']
    assert len(names) == 7
    for old,new in zip(SCENARIO_FAMILIES['cable_x075_surface_v1'],names):
        a,b = get_scenario(old),get_scenario(new)
        assert b.rigid_drive_version == a.rigid_drive_version
        assert b.shape_motion_scale == a.shape_motion_scale
        assert b.frequency_scale == a.frequency_scale
        assert (b.initial_placement_version == 'sector_fraction_v1') == (b.motion_type.value == 'static')
        assert (b.rigid_motion_speed_range is not None) == (b.motion_profile_version == 'rigid_level1_single_pass_v2')


def test_speed_draw_is_repeatable_and_does_not_leak_into_legacy_scene():
    from panda_cable_grasp.rl.environment import make_rl_env
    names=('dev_rigid_l1_nominal_x075_sector_speed_v1','dev_rigid_l1_nominal_x075_surface_v1')
    env=make_rl_env(robot='nero',scenario_names=names)
    try:
        _,a=env.reset(seed=0)
        sampled=a['rigid_motion_nominal_speed']
        assert get_scenario(names[0]).rigid_motion_speed_range[0] <= sampled <= get_scenario(names[0]).rigid_motion_speed_range[1]
        _,b=env.reset(seed=1)
        assert b['rigid_motion_nominal_speed']==.25
        _,c=env.reset(seed=0)
        assert c['rigid_motion_nominal_speed']==sampled
    finally:
        env.close()
