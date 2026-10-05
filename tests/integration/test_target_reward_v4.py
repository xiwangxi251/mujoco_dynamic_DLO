import numpy as np
import pytest
from panda_cable_grasp.rl.approach_credit import ApproachProgressCredit
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
from panda_cable_grasp.env.environment import GraspState


def test_pinch_cycles_cannot_repay_reach_or_alignment():
    credit = ApproachProgressCredit(.04, 0.)
    assert credit.update(.03, .8, eligible=True) == pytest.approx((.01, .8))
    for _ in range(10):
        assert credit.update(.04, 0., eligible=False) == (0., 0.)
        assert credit.update(.03, .8, eligible=True) == (0., 0.)
    assert credit.update(.02, .9, eligible=True) == pytest.approx((.01, .1))
    assert credit.update(.04, 0., eligible=True) == pytest.approx((-.02, -.9))
    assert credit.update(.02, .9, eligible=True) == (0., 0.)
    assert credit.update(.04, 0., eligible=True) == (0., 0.)


def test_inactive_improvement_cannot_be_claimed_on_release():
    credit = ApproachProgressCredit(.04, 0.)
    assert credit.update(.01, 1., eligible=False) == (0., 0.)
    assert credit.update(.03, .8, eligible=True) == (0., 0.)
    assert credit.update(.01, 1., eligible=True) == (0., 0.)


@pytest.mark.parametrize('profile,expected', [('tolerance_progress_v3', (.18, 4.8)),
                                           ('tolerance_progress_v4', (.06, 1.6))])
def test_production_shaping_ledger_survives_pinch_transitions(profile, expected):
    env=make_rl_env(robot='nero',action_mode='task_space_vertical_down',
                   scenario_names=['id_static'],rl_config=RLConfig(target_node_task=True,target_reward_profile=profile))
    try:
        env.reset(seed=81,options={'goal_node_index':20})
        value={'distance':.04,'alignment':0.}
        point=env.data.xpos[env.base_env.cable_ids[20]].copy()
        env._nearest_graspable_segment=lambda _: (point,value['distance'],np.array([1.,0.,0.]),20,0.)
        env._alignment_terms=lambda *_: (1.,value['alignment'])
        env._previous_distance=.04
        env._previous_alignment_potential=0.
        status=dict(pinch_confirmed=False,secured_grasp=False,grasp_lift_delta=0.,
                    grasp_body_height_above_table=.02,strict_success_hold=0.,height_band_hold=0.)
        env._reward(np.zeros(5),False,{'lifted_fraction':0.},status)
        total=np.zeros(2)
        for _ in range(3):
            # The first unheld sample reestablishes the old phase baseline.
            value.update(distance=.04,alignment=0.)
            env.base_env.grasp_state=None
            status['pinch_confirmed']=False
            env._reward(np.zeros(5),False,{'lifted_fraction':0.},status)
            value.update(distance=.03,alignment=.8)
            env.base_env.grasp_state=None
            status['pinch_confirmed']=False
            _,parts=env._reward(np.zeros(5),False,{'lifted_fraction':0.},status)
            total += [parts['reward_reach_progress'],parts['reward_alignment_progress']]
            env.base_env.grasp_state=GraspState(body_id=env.base_env.cable_ids[20],candidate_time=0.,
                bilateral_confirmed=True,last_bilateral_time=0.,lost_contact_time=0.)
            value.update(distance=.04,alignment=0.)
            status['pinch_confirmed']=True
            _,parts=env._reward(np.zeros(5),False,{'lifted_fraction':0.},status)
            assert parts['reward_reach_progress']==parts['reward_alignment_progress']==0.
        assert total==pytest.approx(expected)
        env.reset(seed=82)
        assert env._approach_progress_credit is None
    finally:
        env.close()
