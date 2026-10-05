import numpy as np
import pytest

from panda_cable_grasp.env.environment import GraspState
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
from panda_cable_grasp.rl.lift_credit import HighWaterLiftCredit


def test_progress_can_be_paid_and_debited_only_once():
    credit = HighWaterLiftCredit([0., 0.], 0., bounded_debits=True)
    assert credit.update([.10, 0.], .20, 0, 1.) == pytest.approx((.10, .20))
    assert credit.update([0., 0.], 0., 0, 1.) == pytest.approx((-.10, -.20))
    for _ in range(5):
        assert credit.update([.10, 0.], .20, 0, 1.) == (0., 0.)
        assert credit.update([0., 0.], 0., 0, 1.) == (0., 0.)
    assert credit.update([.12, 0.], .25, 0, 1.) == pytest.approx((.02, .05))
    assert credit.update([0., 0.], 0., 0, 1.) == pytest.approx((-.02, -.05))


def test_release_node_switch_and_discount_cannot_reset_debit_budget():
    credit = HighWaterLiftCredit([.02, .02], 0., bounded_debits=True)
    assert credit.update([.10, .02], .20, 0, .25) == pytest.approx((.02, .05))
    assert credit.update([.02, .02], 0., 0, 1.) == pytest.approx((-.02, -.05))
    credit.update([.02, .02], 0., None, 0.)
    assert credit.update([.02, .10], .20, 1, 1.) == (0., 0.)
    assert credit.update([.02, .02], 0., 1, 1.) == (0., 0.)
    assert credit.update([.02, .12], .30, 1, .5) == pytest.approx((.01, .05))
    assert credit.update([.02, .02], 0., 1, 1.) == pytest.approx((-.01, -.05))


def test_external_motion_does_not_create_a_reward_or_debit_budget():
    credit = HighWaterLiftCredit([.02], 0., bounded_debits=True)
    assert credit.update([.12], .25, None, 0.) == (0., 0.)
    assert credit.update([.12], .25, 0, 1.) == (0., 0.)
    assert credit.update([.02], 0., 0, 1.) == (0., 0.)
    assert credit.update([.12], .25, 0, 1.) == (0., 0.)
    assert credit.update([.14], .30, 0, 1.) == pytest.approx((.02, .05))


@pytest.mark.parametrize('profile,bounded', [('tolerance_progress_v2',False),
                                          ('tolerance_progress_v3',True)])
def test_profile_selects_production_reward_accounting(profile, bounded):
    env = make_rl_env(robot='nero', action_mode='task_space_vertical_down',
                     scenario_names=['id_static'], rl_config=RLConfig(
                         target_node_task=True, target_reward_profile=profile))
    try:
        env.reset(seed=81, options={'goal_node_index':20})
        assert env._lift_progress_credit.bounded_debits is bounded
        env.base_env.grasp_state = GraspState(body_id=env.base_env.cable_ids[20],
            candidate_time=0., bilateral_confirmed=True, last_bilateral_time=0., lost_contact_time=0.)
        top=env.data.geom_xpos[env._table_geom_id,2]+env.model.geom_size[env._table_geom_id,2]
        status=dict(pinch_confirmed=True,secured_grasp=False,grasp_lift_delta=.05,
                    grasp_body_height_above_table=.09,strict_success_hold=0.,height_band_hold=0.)
        values=[]
        for height,fraction in [(.09,.1),(.02,0.),(.09,.1),(.02,0.)]:
            env.data.xpos[env.base_env.cable_ids[20],2]=top+height
            _,parts=env._reward(np.zeros(5),False,{'lifted_fraction':fraction},status)
            values.append((parts['reward_lift_progress'],parts['reward_cable_lift_progress']))
        assert values[0][0]>0 and values[0][1]>0
        if bounded:
            assert values[1]==pytest.approx(tuple(-v for v in values[0]))
            assert values[2]==values[3]==(0.,0.)
        else:
            assert values[3][0]<0 and values[3][1]<0
    finally:
        env.close()
