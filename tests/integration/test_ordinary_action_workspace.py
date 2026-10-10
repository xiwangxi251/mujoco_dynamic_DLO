import hashlib
import json

import numpy as np
import pytest

from panda_cable_grasp.env.environment import GraspState
from panda_cable_grasp.rl.action_workspace import ActionWorkspace, DEFAULT_ACTION_WORKSPACE
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env


def test_measured_map_rejects_inner_and_top_failed_samples():
    w = ActionWorkspace(DEFAULT_ACTION_WORKSPACE)
    assert w.contains([.35 + .36, 0., .10])
    assert not w.contains([.35 + .24, 0., .005])
    assert not w.contains([.35 + .36, 0., .30])
    for p, d in [([.71, 0., .10], [-.3, 0., 0.]),
                 ([.71, 0., .10], [0., 0., .4]),
                 ([.71, 0., .10], [0., -.5, 0.])]:
        p, d = np.array(p), np.array(d)
        result = w.constrain(p, d)
        assert np.linalg.norm(result) <= np.linalg.norm(d) + 1e-9
        assert all(w.contains(p + result * t) for t in np.linspace(0, 1, 101))


def test_outside_entry_is_bounded_and_zero_command_stays_zero():
    w = ActionWorkspace(DEFAULT_ACTION_WORKSPACE)
    p = np.array([.5, 0., .5])
    assert not w.contains(p)
    assert np.array_equal(w.constrain(p, np.zeros(3)), np.zeros(3))
    d = w.constrain(p, np.array([.001, 0., 0.]))
    assert np.linalg.norm(d) <= .001 + 1e-12
    assert np.linalg.norm(w.nearest(p+d)-(p+d)) < np.linalg.norm(w.nearest(p)-p)


def test_failed_hole_is_not_bridged(tmp_path):
    path = tmp_path / 'map.json'
    path.write_text(json.dumps(dict(version='measured_polar_cells_v1', center_xy_m=[0, 0],
        cells_r_theta_z=[[.28, .32, -20, 20, .005, .2], [.36, .4, -20, 20, .005, .2]])))
    w = ActionWorkspace(path)
    p = np.array([.3, 0., .1])
    result = w.constrain(p, np.array([.09, 0., 0.]))
    assert p[0] + result[0] <= .32 + 1e-9


def ordinary_env():
    return make_rl_env(robot='nero', action_mode='task_space_vertical_down',
        scenario_names=['dev_static_x075_v1'], controller_profile='enhanced6_bounded_dls_v1',
        table_execution_guard_enabled=True, table_finger_collision_filter_enabled=False,
        rl_config=RLConfig(ordinary_reward_profile='physical_progress_v1',
            action_workspace_path=str(DEFAULT_ACTION_WORKSPACE),
            action_workspace_sha256=hashlib.sha256(DEFAULT_ACTION_WORKSPACE.read_bytes()).hexdigest()))


def test_ordinary_runtime_uses_99d_and_raw_action_is_unchanged():
    env = ordinary_env()
    try:
        obs, info = env.reset(seed=20260804)
        assert obs.shape == (99,) and env.goal is None and env.action_space.shape == (5,)
        action = np.array([1., 0., .8, .2, -.8], dtype=np.float32)
        original = action.copy()
        for _ in range(3):
            obs, reward, _, _, info = env.step(action)
            assert np.isfinite(reward) and np.isfinite(obs).all()
            assert np.array_equal(action, original)
            assert np.linalg.norm(info['workspace_executed_delta']) <= .01 + 1e-9
        assert env.base_env._arm_motion_start_time() == pytest.approx(.82)
    finally:
        env.close()


def test_ordinary_lift_grace_release_debit_and_no_reclaim():
    env = ordinary_env()
    try:
        env.reset(seed=81)
        b = env.base_env
        node = b.cable_ids[20]
        top = env.data.geom_xpos[b.table_geom_id, 2] + env.model.geom_size[b.table_geom_id, 2]
        b.grasp_state = GraspState(node, 0., True, 0., 0.)
        env.data.xpos[node] = [0., 0., top+.15]
        env._grasp_center_position = lambda: np.array([0., 0., top+.15])
        b._grasp_has_bilateral_pad_contact = lambda: False
        for t in [.02, .04, .06, .08, .10]:
            env.data.time = t
            status = env._update_grasp_status({'lifted_fraction': .3})
        assert status['secured_grasp'] and not status['raw_bilateral_contact']
        env.data.time = .34
        assert env._current_lift_reward_holding(status)
        _, paid = env._reward(np.zeros(5), False, {'lifted_fraction': .3, 'success_hold': .4}, status)
        assert paid['reward_lift_progress'] > 0 and paid['reward_strict_hold'] > 0
        assert not any(k.startswith('reward_target') for k in paid)
        env.data.time = .351
        assert not env._current_lift_reward_holding(status)
        b.grasp_state = None
        status.update(pinch_confirmed=False, secured_grasp=False)
        env.data.xpos[node, 2] = top
        _, lost = env._reward(np.zeros(5), False, {'lifted_fraction': 0.}, status)
        for k in ('reward_lift_progress', 'reward_cable_lift_progress'):
            assert lost[k] == pytest.approx(-paid[k])
        b.grasp_state = GraspState(node, 0., True, .351, .351)
        env.data.xpos[node, 2] = top+.15
        status.update(pinch_confirmed=True)
        _, again = env._reward(np.zeros(5), False, {'lifted_fraction': .3}, status)
        assert again['reward_lift_progress'] == again['reward_cable_lift_progress'] == 0.
    finally:
        env.close()
