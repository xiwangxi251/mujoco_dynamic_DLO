import numpy as np
import pytest

from panda_cable_grasp.env.environment import GraspState
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env


@pytest.fixture
def env():
    e = make_rl_env(robot='nero', action_mode='task_space_vertical_down',
                    scenario_names=['id_static'], rl_config=RLConfig(
                        target_node_task=True, target_reward_profile='tolerance_progress_v6'))
    e.reset(seed=81, options={'goal_node_index': 20})
    b = e.base_env
    table = b.table_geom_id
    top = e.data.geom_xpos[table, 2] + e.model.geom_size[table, 2]
    b.grasp_state = GraspState(b.cable_ids[20], 0., True, 0., 0.)
    e.data.qpos[b.finger_qpos_adr] = [.012, -.012]
    e.data.xpos[b.cable_ids[20]] = [0., 0., top+.15]
    e._grasp_center_position = lambda: np.array([0., 0., top+.15])
    b._grasp_has_bilateral_pad_contact = lambda: False
    yield e
    e.close()


def status():
    return dict(pinch_confirmed=True, raw_bilateral_contact=False, secured_grasp=False,
                grasp_lift_delta=.10, grasp_body_height_above_table=.15,
                strict_success_hold=0., height_band_hold=0.)


def test_lift_pays_inside_grace_and_debits_after_expiry(env):
    env.data.time = .34
    assert env._lift_reward_contact_grace_seconds() == .35
    assert env._current_lift_reward_holding(status())
    _, paid = env._reward(np.zeros(5), False, {'lifted_fraction': .20}, status())
    assert paid['reward_lift_progress'] > 0 and paid['reward_cable_lift_progress'] > 0
    env.data.time = .351
    # Even a stale caller status cannot extend the physical confirmation timer.
    assert not env._current_lift_reward_holding(status())
    env.data.xpos[env.base_env.cable_ids[20], 2] = 0.
    _, lost = env._reward(np.zeros(5), False, {'lifted_fraction': 0.}, status())
    for key in ('reward_lift_progress', 'reward_cable_lift_progress'):
        assert lost[key] == pytest.approx(-paid[key])


@pytest.mark.parametrize('invalid', ['distance', 'unconfirmed', 'cleared'])
def test_geometry_or_unconfirmed_state_invalidates_immediately(env, invalid):
    env.data.time = .02
    if invalid == 'distance':
        env._grasp_center_position = lambda: np.array([1., 0., .15])
    elif invalid == 'unconfirmed':
        env.base_env.grasp_state.bilateral_confirmed = False
    else:
        env.base_env.grasp_state = None
    assert not env._current_lift_reward_holding(status())
    _, parts = env._reward(np.zeros(5), False, {'lifted_fraction': .20}, status())
    assert parts['reward_lift_progress'] == parts['reward_cable_lift_progress'] == 0.


@pytest.mark.parametrize('profile, secured', [('tolerance_progress_v5', False),
                                           ('tolerance_progress_v6', True)])
def test_loaded_confirmation_tolerates_jitter_before_first_secured(env, profile, secured):
    env.rl_config.target_reward_profile = profile
    for t in [.02, .04, .06, .08, .10]:
        env.data.time = t
        s = env._update_grasp_status({'lifted_fraction': .30})
    assert s['secured_grasp'] is secured
    assert not s['raw_bilateral_contact']
    assert s['strict_success_hold'] == 0.  # Original raw-contact diagnostic.
    env.data.time = .34
    s = env._update_grasp_status({'lifted_fraction': .30})
    assert s['secured_grasp'] is secured
    env.data.time = .351
    s = env._update_grasp_status({'lifted_fraction': .30})
    assert not s['secured_grasp']


def test_large_aperture_pays_lift_and_secured_reward_once_then_expires(env):
    env.data.qpos[env.base_env.finger_qpos_adr] = [.035, -.035]
    for t in [.02, .04, .06, .08, .10]:
        env.data.time = t
        s = env._update_grasp_status({'lifted_fraction': .30})
    _, paid = env._reward(np.zeros(5), False, {'lifted_fraction': .30}, s)
    assert paid['reward_new_secured_grasp'] == env.rl_config.reward_new_secured_grasp
    assert paid['reward_lift_progress'] > 0
    assert paid['reward_cable_lift_progress'] > 0
    _, repeated = env._reward(np.zeros(5), False, {'lifted_fraction': .30}, s)
    assert repeated['reward_new_secured_grasp'] == 0.
    env.data.time = .351
    s = env._update_grasp_status({'lifted_fraction': .30})
    assert not s['secured_grasp']
    assert not env._current_lift_reward_holding(s)
