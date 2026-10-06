import numpy as np
import pytest

from panda_cable_grasp.env.environment import GraspState
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
from panda_cable_grasp.rl.lift_credit import HighWaterLiftCredit


def ledger(heights=(.02, .02), fraction=0.):
    return HighWaterLiftCredit(heights, fraction, bounded_debits=True, release_debits=True)


def test_continuous_lift_drop_regrasp_and_new_record():
    credit = ledger()
    assert credit.update([.06, .02], .10, 0, 1.) == pytest.approx((.04, .10))
    assert credit.update([.12, .02], .20, 0, 1.) == pytest.approx((.06, .10))
    assert credit.update([.12, .02], .20, None, 0.) == (0., 0.)
    assert credit.update([.07, .02], .10, None, 0.) == pytest.approx((-.05, -.10))
    assert credit.update([.02, .02], 0., None, 0.) == pytest.approx((-.05, -.10))
    assert credit.update([.12, .02], .20, 0, 1.) == (0., 0.)
    assert credit.update([.02, .02], 0., None, 0.) == (0., 0.)
    assert credit.update([.14, .02], .25, 0, 1.) == pytest.approx((.02, .05))
    assert credit.update([.02, .02], 0., None, 0.) == pytest.approx((-.02, -.05))
    assert credit.height_debit_remaining == credit.fraction_debit_remaining == 0.


def test_external_raise_and_drop_have_no_credit():
    credit = ledger()
    assert credit.update([.12, .02], .20, None, 0.) == (0., 0.)
    assert credit.update([.02, .02], 0., None, 0.) == (0., 0.)
    assert credit.update([.12, .02], .20, 0, 1.) == (0., 0.)
    assert credit.update([.14, .02], .25, 0, 1.) == pytest.approx((.02, .05))
    assert credit.update([.12, .02], .20, None, 0.) == pytest.approx((-.02, -.05))
    assert credit.update([.02, .02], 0., None, 0.) == (0., 0.)


def test_node_switch_tracks_paid_material_and_original_discount():
    credit = ledger()
    assert credit.update([.10, .02], .20, 0, .25) == pytest.approx((.02, .05))
    # The new node falls without any height credit; the paid node stays high.
    assert credit.update([.10, 0.], .20, 1, 1.) == (0., 0.)
    # Losing the paid node still debits, even while another node earns credit.
    assert credit.update([.02, .12], .30, 1, .5) == pytest.approx((-.01, .05))
    assert credit.update([.02, .02], 0., 1, 0.) == pytest.approx((-.01, -.10))
    assert credit.update([.10, .12], .30, 0, 1.) == (0., 0.)


def test_partial_fall_rebound_does_not_debit_twice():
    credit = ledger()
    assert credit.update([.12, .02], .20, 0, .5) == pytest.approx((.05, .10))
    assert credit.update([.08, .02], .10, None, 0.) == pytest.approx((-.02, -.05))
    assert credit.update([.12, .02], .20, None, 0.) == (0., 0.)
    assert credit.update([.08, .02], .10, None, 0.) == (0., 0.)
    assert credit.update([.02, .02], 0., None, 0.) == pytest.approx((-.03, -.05))


@pytest.mark.parametrize('profile,repaired', [('tolerance_progress_v5', False),
                                            ('tolerance_progress_v6', True)])
def test_production_reward_debits_after_raw_contact_loss(profile, repaired):
    env = make_rl_env(robot='nero', action_mode='task_space_vertical_down',
                      scenario_names=['id_static'], rl_config=RLConfig(
                          target_node_task=True, target_reward_profile=profile))
    try:
        env.reset(seed=81, options={'goal_node_index': 20})
        assert env._lift_progress_credit.release_debits is repaired
        env.base_env.grasp_state = GraspState(body_id=env.base_env.cable_ids[20],
            candidate_time=0., bilateral_confirmed=True, last_bilateral_time=0., lost_contact_time=0.)
        table = env.base_env.table_geom_id
        top = env.data.geom_xpos[table, 2] + env.model.geom_size[table, 2]
        status = dict(pinch_confirmed=True, raw_bilateral_contact=True, secured_grasp=False,
                      grasp_lift_delta=.05, grasp_body_height_above_table=.09,
                      strict_success_hold=0., height_band_hold=0.)
        env.data.qpos[env.base_env.finger_qpos_adr] = [.012, -.012]
        env.data.xpos[env.base_env.cable_ids[20]] = [0., 0., top + .10]
        env._grasp_center_position = lambda: np.array([0., 0., top + .10])
        _, paid = env._reward(np.zeros(5), False, {'lifted_fraction': .20}, status)
        assert paid['reward_lift_progress'] > 0 and paid['reward_cable_lift_progress'] > 0
        status.update(pinch_confirmed=False, raw_bilateral_contact=False)
        env.base_env.grasp_state = None
        env.data.xpos[env.base_env.cable_ids[20], 2] = top
        _, lost = env._reward(np.zeros(5), False, {'lifted_fraction': 0.}, status)
        for key in ('reward_lift_progress', 'reward_cable_lift_progress'):
            assert lost[key] == pytest.approx(-paid[key] if repaired else 0.)
    finally:
        env.close()
