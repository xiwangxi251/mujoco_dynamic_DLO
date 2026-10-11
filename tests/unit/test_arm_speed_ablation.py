import argparse
import json
import numpy as np
import pytest

from panda_cable_grasp.env.control import delta_pose_twist, solve_bounded_full6
from panda_cable_grasp.rl.environment import RLConfig, VerticalDownRLCableGraspEnv
from panda_cable_grasp.rl.train import rl_config_from_args
from panda_cable_grasp.rl.penalty_ramp import resolve_schedule


def test_uncapped_dls_retains_position_bounds_and_damping():
    j = np.c_[np.eye(6), np.zeros(6)]
    q = np.zeros(7)
    limited, _, _ = solve_bounded_full6(j, [2, 0, 0], [0, 0, 5], q, q,
                                       np.ones(7)*.5, -np.ones(7), np.ones(7))
    fast, target, _ = solve_bounded_full6(j, [2, 0, 0], [0, 0, 5], q, q,
                                       np.ones(7)*.5, -np.ones(7), np.ones(7),
                                       velocity_limits_enabled=False)
    assert max(abs(limited)) <= .5
    np.testing.assert_allclose(fast[:6], np.array([2,0,0,0,0,5])/(1+.04**2))
    assert np.all(target >= -1) and np.all(target <= 1)
    q[0] = .999
    _, target, _ = solve_bounded_full6(j, [20,0,0], [0,0,0], q, q,
                                      np.ones(7)*.5, -np.ones(7), np.ones(7),
                                      velocity_limits_enabled=False)
    assert target[0] <= 1
    linear, angular = delta_pose_twist([.04,0,0], .02, [0,0,2], velocity_limits_enabled=False)
    np.testing.assert_allclose(linear, [2,0,0])
    np.testing.assert_allclose(angular, [0,0,20])


def test_execution_bypass_reset_gripper_and_physical_invariants():
    kwargs = dict(robot='nero', controller_profile='enhanced6_bounded_dls_v1',
                  scenario_names=('dev_static_x075_v1',),
                  table_execution_guard_enabled=True, table_finger_collision_filter_enabled=False)
    baseline = VerticalDownRLCableGraspEnv(**kwargs)
    fast = VerticalDownRLCableGraspEnv(**kwargs, rl_config=RLConfig(
        arm_velocity_limits_enabled=False, translation_delta_scale=.04, yaw_delta_scale=.08))
    try:
        for seed in (20, 21):
            fast.reset(seed=seed)
            fast.base_env.configure_controller_profile('enhanced6_bounded_dls_v1')
            audit = fast.execution_configuration()
            assert audit['env_config']['arm_velocity_limits_enabled'] is False
            assert audit['env_config']['hand_cartesian_velocity_limit_enabled'] is False
            assert audit['arm_start_seconds'] == .82
            assert audit['compiled_hashes'] == baseline.execution_configuration()['compiled_hashes']
            env = fast.base_env
            env.data.time = 1.
            requested = env.ready_ctrl.copy()
            ranges = env.model.jnt_range[env.arm_joint_ids]
            requested[:7] = np.clip(requested[:7]+.3, ranges[:,0], ranges[:,1])
            requested[7] = 0
            before = env._last_applied_action.copy()
            applied = env._limit_robot_action(requested)
            np.testing.assert_allclose(applied[:7], requested[:7])
            assert abs(applied[7]-before[7]) <= env.config.gripper_finger_velocity_limit*.02 + 1e-9
            env.data.qvel[env.arm_dof_adr] = 30
            guarded, changed = env._velocity_guarded_action(applied)
            np.testing.assert_array_equal(guarded, applied)
            assert not changed
    finally:
        fast.close()
        baseline.close()


def test_strict_reference_and_worker_independent_ramp(tmp_path):
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'rl_config': {'arm_velocity_limits_enabled': False,
                                             'translation_delta_scale': .04, 'yaw_delta_scale': .08}}))
    args = argparse.Namespace(rl_config_file=path, target_reward_profile='legacy')
    config = rl_config_from_args(args)
    assert config.arm_velocity_limits_enabled is False and config.translation_delta_scale == .04
    for workers in (16,24,32):
        schedule = resolve_schedule(object(), config, workers, {'workers':48})
        assert schedule['height_full_samples'] == 480000
        assert schedule['safety_full_samples'] == 240000
    path.write_text(json.dumps({'typo_velocity_limit':False}))
    with pytest.raises(ValueError, match='Unknown RLConfig'):
        rl_config_from_args(args)


def test_evaluation_restores_saved_uncapped_action_configuration(tmp_path):
    from panda_cable_grasp.rl.evaluate import evaluation_rl_config
    saved = dict(arm_velocity_limits_enabled=False, translation_delta_scale=.04,
                 yaw_delta_scale=.08, ordinary_reward_profile='session_height_v1')
    (tmp_path/'training_config.json').write_text(json.dumps({'rl_config':saved}))
    restored = evaluation_rl_config(argparse.Namespace(model=tmp_path/'final_model.zip'))
    assert restored.arm_velocity_limits_enabled is False
    assert restored.translation_delta_scale == .04 and restored.yaw_delta_scale == .08
    assert restored.ordinary_reward_profile == 'session_height_v1'
