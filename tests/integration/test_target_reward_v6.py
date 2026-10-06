import numpy as np
import pytest

from panda_cable_grasp.env.environment import GraspState
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env


def test_hold_reward_uses_public_timer_and_cannot_repay_after_reset():
    env = make_rl_env(robot='nero', action_mode='task_space_vertical_down',
        scenario_names=['id_static'],
        rl_config=RLConfig(target_node_task=True, target_reward_profile='tolerance_progress_v6'))
    try:
        env.reset(seed=81, options={'goal_node_index': 20})
        base = env.base_env
        base.grasp_state = GraspState(base.cable_ids[20], 0., True, 0., 0.)
        status = dict(pinch_confirmed=True, raw_bilateral_contact=False,
            secured_grasp=False, grasp_lift_delta=.10, grasp_body_height_above_table=.15,
            strict_success_hold=0., height_band_hold=0.)
        info = dict(lifted_fraction=.3, success_hold=.4)
        _, parts = env._reward(np.zeros(5), False, info, status)
        assert parts['reward_strict_hold'] == pytest.approx(1.)
        info['success_hold'] = 0.
        _, parts = env._reward(np.zeros(5), False, info, status)
        assert parts['reward_strict_hold'] == 0.
        info['success_hold'] = .4
        _, parts = env._reward(np.zeros(5), False, info, status)
        assert parts['reward_strict_hold'] == 0.
        info['success_hold'] = .8
        _, parts = env._reward(np.zeros(5), True, info, status)
        assert parts['reward_strict_hold'] == pytest.approx(1.)
        assert parts['reward_success'] == env.rl_config.reward_success
        # A different material node cannot earn target holding credit.
        base.grasp_state.body_id = base.cable_ids[30]
        env._strict_hold_credit_high_water = 0.
        _, parts = env._reward(np.zeros(5), False, info, status)
        assert parts['reward_strict_hold'] == 0.
    finally:
        env.close()
