import numpy as np
import pytest

from panda_cable_grasp.rl.lift_credit import HighWaterLiftCredit
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
from panda_cable_grasp.env.environment import GraspState


def credit():
    return HighWaterLiftCredit([.045, .045], 0., bounded_debits=True,
                              release_debits=True, first_holding_baseline=True)


def test_settle_first_lift_release_regrasp_and_node_switch():
    c = credit()
    assert c.update([.015, .015], 0., None, 0.) == (0., 0.)
    assert not c.height_baseline_locked
    assert c.update([.015, .015], 0., 0, 1.) == (0., 0.)
    assert c.height_baseline_locked
    assert c.update([.020, .015], 0., 0, 1.) == pytest.approx((.005, 0.))
    assert c.update([.030, .015], 0., 0, 1.) == pytest.approx((.010, 0.))
    assert c.update([.015, .015], 0., None, 0.) == pytest.approx((-.015, 0.))
    assert c.update([.015, .030], 0., 1, 1.) == (0., 0.)
    assert c.update([.030, .015], 0., 0, 1.) == (0., 0.)
    assert c.update([.035, .015], 0., 0, 1.) == pytest.approx((.005, 0.))


def test_external_motion_is_not_paid_and_grasp_jitter_cannot_rebase():
    c = credit()
    assert c.update([.080, .015], .2, None, 0.) == (0., 0.)
    assert c.update([.080, .015], .2, 0, 1.) == (0., 0.)
    assert c.update([.015, .015], 0., None, 0.) == (0., 0.)
    assert c.update([.080, .015], .2, 0, 1.) == (0., 0.)
    assert c.update([.090, .015], .3, 0, 1.) == pytest.approx((.01, .1))
    assert c.update([.015, .015], 0., None, 0.) == pytest.approx((-.01, -.1))


@pytest.mark.parametrize('profile,expected', [('physical_progress_v1', 0.),
                                           ('physical_progress_v2', .20)])
def test_real_wrapper_pays_5mm_lift_below_spawn_height(profile, expected):
    env = make_rl_env(robot='nero', action_mode='task_space_vertical_down',
                     scenario_names=['dev_static_x075_v1'],
                     rl_config=RLConfig(ordinary_reward_profile=profile))
    try:
        obs, _ = env.reset(seed=20260804)
        assert obs.shape == (99,) and env.goal is None
        b = env.base_env
        top = env.data.geom_xpos[b.table_geom_id, 2] + env.model.geom_size[b.table_geom_id, 2]
        env.data.xpos[b.cable_ids, 2] = top + .015
        status = dict(pinch_confirmed=False, raw_bilateral_contact=False,
                      secured_grasp=False, grasp_lift_delta=0.,
                      grasp_body_height_above_table=.015,
                      strict_success_hold=0., height_band_hold=0.)
        env._reward(np.zeros(5), False, {'lifted_fraction': 0.}, status)
        node = b.cable_ids[20]
        b.grasp_state = GraspState(node, 0., True, 0., 0.)
        env._grasp_center_position = lambda: env.data.xpos[node].copy()
        status.update(pinch_confirmed=True, raw_bilateral_contact=True)
        env._reward(np.zeros(5), False, {'lifted_fraction': 0.}, status)
        env.data.xpos[node, 2] += .005
        _, terms = env._reward(np.zeros(5), False, {'lifted_fraction': 0.}, status)
        assert terms['reward_lift_progress'] == pytest.approx(expected)
        assert terms['reward_new_secured_grasp'] == 0.
        assert not any(k.startswith('reward_target') for k in terms)
    finally:
        env.close()
