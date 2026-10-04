"""Regression contracts for the selected execution path migrated to dev."""
import mujoco
import numpy as np
import pytest

from panda_cable_grasp.env.control import (
    ENHANCED6_BOUNDED_DLS_V1, absolute_pose_twist, compose_bounded_pd_gravity,
    solve_bounded_full6,
)
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.env.kinematics import quat_error, rotation_to_quat
from panda_cable_grasp.env.table_execution_guard import TableExecutionGuard
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy
from panda_cable_grasp.rl.environment import VerticalDownRLCableGraspEnv
from panda_cable_grasp.dynamicvla.adapter import DynamicVLATaskSpaceAdapter


PROFILE = ENHANCED6_BOUNDED_DLS_V1.name


def test_inactive_bounds_preserve_original_full6_dls():
    rng = np.random.default_rng(20)
    j = rng.normal(size=(6, 7))
    q, ready = np.zeros(7), rng.normal(size=7)*.1
    twist = rng.normal(size=6)*.01
    bias = (np.eye(7)-np.linalg.pinv(j, rcond=1e-5)@j) @ (.8*(ready-q))
    expected = j.T@np.linalg.solve(j@j.T+.04**2*np.eye(6), twist)+bias
    velocity, target, _ = solve_bounded_full6(j, twist[:3], twist[3:], q, ready,
                                             np.ones(7)*3, -np.ones(7)*2, np.ones(7)*2)
    np.testing.assert_allclose(velocity, expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(target, q+.11*expected, atol=1e-10, rtol=0)


def test_bound_changes_solution_instead_of_clipping_task_component():
    j = np.zeros((6, 7))
    j[0, :2] = 1
    j[1:, 2:] = np.eye(5)
    q = np.zeros(7); q[0] = .99
    velocity, target, _ = solve_bounded_full6(j, [.9, 0, 0], [0, 0, 0], q, q,
                                             np.ones(7)*3, -np.ones(7), np.ones(7))
    assert target[0] <= 1 and np.all(target >= -1) and np.all(target <= 1)
    assert velocity[0] <= .01/.11+1e-10
    assert velocity[1] > .75
    assert abs((j@velocity)[0]-.9) < .002


def test_soft_limit_overshoot_can_command_recovery():
    j = np.c_[np.eye(6), np.zeros(6)]
    q = np.zeros(7); q[0] = 1.002
    velocity, target, _ = solve_bounded_full6(j, [.4, 0, 0], [0, 0, 0], q, np.zeros(7),
                                             np.ones(7)*3, -np.ones(7), np.ones(7))
    assert velocity[0] < 0 and target[0] <= 1
    with pytest.raises(RuntimeError, match="infeasible"):
        solve_bounded_full6(j, [.4, 0, 0], [0, 0, 0], q+np.array([2,0,0,0,0,0,0]),
                           np.zeros(7), np.ones(7)*3, -np.ones(7), np.ones(7))


def test_total_pd_external_gravity_respects_original_torque_ranges():
    bounds = np.tile([-100., 100.], (7, 1))
    pd, total, added = compose_bounded_pd_gravity(np.ones(7)*150, np.ones(7)*20,
                                                  np.ones(7)*5, bounds)
    np.testing.assert_allclose(pd+5+added, total)
    assert np.max(total) <= 100


@pytest.fixture(scope="module")
def enhanced():
    env = CableGraspEnv(EnvConfig(robot="nero", motion_mode="static",
                                  controller_profile=PROFILE,
                                  table_execution_guard_enabled=True))
    yield env
    env.close()


def test_cached_gravity_matches_scene_and_uses_independent_data(enhanced):
    env = enhanced
    env.reset(seed=12)
    probe = mujoco.MjData(env.model)
    rng = np.random.default_rng(12)
    ranges = env.model.jnt_range[env.arm_joint_ids]
    for _ in range(12):
        probe.qpos[:] = env.data.qpos
        probe.qpos[env.arm_qpos_adr] = rng.uniform(ranges[:, 0], ranges[:, 1])
        probe.qpos[env.finger_qpos_adr] = [.015, -.015]
        probe.qvel[:] = 0
        mujoco.mj_forward(env.model, probe)
        np.testing.assert_allclose(env._gravity_bias_for_qpos(probe.qpos),
                                    probe.qfrc_bias[env.arm_dof_adr], atol=1e-10, rtol=0)
    second = CableGraspEnv(EnvConfig(robot="nero", controller_profile=PROFILE))
    try:
        assert env._gravity_model is second._gravity_model
        assert env._gravity_data is not second._gravity_data
    finally:
        second.close()


def test_actuation_and_original_limits_are_preserved(enhanced):
    legacy = CableGraspEnv(EnvConfig(robot="nero"))
    try:
        env = enhanced; ids = env.arm_actuator_ids
        np.testing.assert_allclose(env.model.actuator_gainprm[ids, 0],
                                    legacy.model.actuator_gainprm[ids, 0]*4)
        np.testing.assert_allclose(env.model.actuator_biasprm[ids, 2],
                                    legacy.model.actuator_biasprm[ids, 2]*2)
        for attr in ("actuator_forcerange", "actuator_ctrlrange", "jnt_range"):
            np.testing.assert_array_equal(getattr(env.model, attr), getattr(legacy.model, attr))
        assert not env.config.arm_acceleration_limit_enabled
        assert env.config.hand_linear_velocity_limit == .9
        assert env.config.hand_angular_velocity_limit == 2.
        assert env.model.opt.timestep*env.config.frame_skip == .02
        gains = env.model.actuator_gainprm.copy()
        env.configure_controller_profile(PROFILE)
        np.testing.assert_array_equal(gains, env.model.actuator_gainprm)
    finally:
        legacy.close()


def test_guard_runs_each_physics_step_and_resets(enhanced):
    env = enhanced
    env.reset(seed=13)
    env.step(env.ready_ctrl.copy())
    assert env.info()["table_execution_guard"]["guard_steps"] == 10
    assert env.info()["physics_substep_count"] == 10
    assert env.info()["table_execution_guard"]["monitored_mesh_count"] == 5
    env.reset(seed=13)
    assert env._table_execution_guard.steps == 0
    assert env._table_execution_guard.last_min_clearance == np.inf


def test_guard_projects_only_commands_and_feedback_brakes(enhanced):
    env = enhanced; env.reset(seed=14)
    guard = TableExecutionGuard(env)
    requested = env.ready_ctrl.copy()
    requested[0] -= .1
    qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
    normal = np.zeros(7); normal[0] = 1
    target = guard.project(requested, [(0, .002, normal, -.2)])
    assert target[0] > requested[0] and guard.feedback_steps == 1
    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_array_equal(env.data.qvel, qvel)
    assert target[7] == requested[7]


def test_scripted_and_absolute_pose_adapter_use_identical_full6(enhanced):
    env = enhanced; env.reset(seed=15)
    policy = DynamicCableGraspPolicy(env)
    desired = env.hand_position + [.002, -.002, .001]
    action = policy._ik_action(desired, env.gripper_open_ctrl)
    adapter = DynamicVLATaskSpaceAdapter(env)
    adapter.has_model_action = True
    adapter.last_pose_command[:3] = desired
    adapter.last_pose_command[3:] = policy.desired_quat
    adapter._gripper_ctrl = env.gripper_open_ctrl
    np.testing.assert_allclose(adapter.action(), action, atol=1e-10, rtol=0)


def test_ppo_delta_semantics_full6_and_execution_guard():
    env = VerticalDownRLCableGraspEnv(robot="nero", controller_profile=PROFILE,
                                      table_execution_guard_enabled=True,
                                      table_finger_collision_filter_enabled=False)
    try:
        env.reset(seed=16)
        action = np.array([.1, -.2, .05, 0., 1.])
        result = env._convert_action(action)
        base = env.base_env
        jp, jr = base._hand_jacobian()
        j = np.vstack((jp, jr))[:, base.arm_dof_adr]
        ranges = base.model.jnt_range[base.arm_joint_ids]
        error = quat_error(rotation_to_quat(base.data.xmat[base.hand_id].reshape(3, 3)),
                           rotation_to_quat(env._desired_hand_rotation))
        _, expected, _ = solve_bounded_full6(j,
            env.rl_config.translation_delta_scale*action[:3]/.02, 10.*error,
            base.data.qpos[base.arm_qpos_adr], base.ready_qpos[:7],
            base._arm_velocity_limits, ranges[:,0], ranges[:,1])
        np.testing.assert_allclose(result[:7], expected, atol=1e-10, rtol=0)
        assert not base.config.arm_acceleration_limit_enabled
        env.step(action)
        assert base._table_execution_guard.steps == 10
    finally:
        env.close()


def test_invalid_profile_and_guard_combinations_fail_early():
    with pytest.raises(ValueError):
        EnvConfig(robot="panda", controller_profile=PROFILE)
    with pytest.raises(ValueError):
        EnvConfig(robot="nero", table_execution_guard_enabled=True)
