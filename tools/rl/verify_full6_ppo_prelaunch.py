"""Verify the requested execution, sampling and curriculum before training."""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from train_targetnode_full6_workspace import (RUN, CFG, NEW_CFG, OLD_CFG, CourseEnv,
    SCENES, save, promotion_update, configuration)
from full6_workspace_env import Full6TargetEnv, WorkspaceTargetNode, PROFILE, limit_norm
from panda_cable_grasp.rl.environment import RLCableGraspEnv
from panda_cable_grasp.rl.target_node import TargetNodeTask
from panda_cable_grasp.env.kinematics import quat_error, rotation_to_quat


def main():
    assert Full6TargetEnv._reward is RLCableGraspEnv._reward
    assert WorkspaceTargetNode.finish_reward is TargetNodeTask.finish_reward
    assert WorkspaceTargetNode.qualify is TargetNodeTask.qualify
    assert WorkspaceTargetNode.matches is TargetNodeTask.matches
    expected_changes = {"target_u_min", "target_u_max", "target_balanced_sampling", "graspable_end_fraction"}
    assert {key for key in OLD_CFG if OLD_CFG[key] != NEW_CFG[key]} == expected_changes
    # Endpoint nodes must remain valid goals whenever their spatial positions pass.
    synthetic = SimpleNamespace(_objects=[1], cable_ids=list(range(40)), data=SimpleNamespace(xpos=np.tile([.7,0,.005], (40,1))))
    wrapper = SimpleNamespace(base_env=synthetic, rl_config=CFG)
    workspace = SimpleNamespace(contains=lambda points:np.ones(len(np.atleast_2d(points)), dtype=bool))
    goal = WorkspaceTargetNode(wrapper, 1, workspace)
    for index in (0, 1, 38, 39):
        goal.reset(1, {"goal_node_index":index})
        assert goal.index == index and len(goal.candidates) == 40
    assert promotion_update([], .65, .45) == ([True], False)
    assert promotion_update([True], .65, .45) == ([True,True], True)
    assert promotion_update([True,False], .65, .45) == ([True,False,True], True)
    assert promotion_update([True,True], .59, .45)[0][-1] is False
    assert promotion_update([True,True], .65, .39)[0][-1] is False
    assert "bins" not in inspect.signature(promotion_update).parameters
    # Existing first-grasp reward curve is preserved.
    rewards = []
    for error in range(10):
        synthetic.cable_index = {i:i for i in range(40)}
        synthetic.grasp_state = SimpleNamespace(body_id=20+error)
        g = TargetNodeTask(wrapper, 1)
        g.index = 20
        components = {}
        g.finish_reward(components, dict(pinch_confirmed=True, secured_grasp=False))
        assert g.first_grasp_correct == (error <= 4)
        rewards.append(components["reward_first_grasp_proximity"])
    assert rewards == [2,1.75,1.5,1.25,1,.75,.5,.25,0,0]
    env = CourseEnv()
    base = env.env.base_env
    assert env.observation_space.shape == (109,) and env.action_space.shape == (5,)
    ids = base.arm_actuator_ids
    assert np.array_equal(base.model.actuator_gainprm[ids,0], 4*base.original_kp)
    assert np.array_equal(-base.model.actuator_biasprm[ids,2], 2*base.original_kv)
    assert np.array_equal(base.model.actuator_forcerange[ids], base.original_torque_bounds)
    assert not base.config.arm_acceleration_limit_enabled
    assert base.config.hand_cartesian_velocity_limit_enabled
    rows = []
    all_candidates, chosen = set(), set()
    rng = np.random.default_rng(775)
    try:
        for scene_index, scene in enumerate(SCENES):
            env.set_stage(min(scene_index, 6))
            mix = [0.]*len(SCENES)
            mix[scene_index] = 1.
            env.env.set_training_scenario_distribution(SCENES, tuple(mix))
            for repetition in range(3):
                obs, info = env.reset(seed=20261004+scene_index*100+repetition)
                assert np.isfinite(obs).all()
                index = env.env.goal.index
                assert index in env.env.goal.candidates
                assert env.workspace.contains(base.data.xpos[base.cable_ids[index]])[0]
                all_candidates.update(env.env.goal.candidates.tolist())
                chosen.add(index)
                # The adapter's target is derived from actual q, independent of old command.
                if scene_index == repetition == 0:
                    old_filter = env.env.table_finger_collision_filter_enabled
                    env.env.table_finger_collision_filter_enabled = False
                    previous = base._last_applied_action.copy()
                    action = np.array([.3,-.2,.1,0.,1.])
                    first = env.env._convert_action(action)
                    base._last_applied_action[:7] += .1
                    second = env.env._convert_action(action)
                    assert np.allclose(first,second,rtol=0,atol=1e-12)
                    expected = np.clip(base.data.qpos[base.arm_qpos_adr]+.11*env.env.last_qdot,
                        base.model.jnt_range[base.arm_joint_ids,0],base.model.jnt_range[base.arm_joint_ids,1])
                    assert np.allclose(second[:7],expected,rtol=0,atol=1e-12)
                    error = quat_error(rotation_to_quat(base.data.xmat[base.hand_id].reshape(3,3)),rotation_to_quat(env.env._desired_hand_rotation))
                    assert np.allclose(env.env.last_twist[3:],limit_norm(10*error,1.4),atol=1e-12)
                    base._last_applied_action[:] = previous
                    env.env.table_finger_collision_filter_enabled = old_filter
                for _ in range(80):
                    obs,reward,done,truncated,info = env.step(rng.uniform(-.25,.25,5))
                    assert np.isfinite(obs).all() and np.isfinite(reward)
                    assert env.env.goal.index == index
                    if done or truncated:
                        break
                rows.append(dict(scene=scene, goal=index, candidates=env.env.goal.candidates.tolist(),
                    reset_prefix_seconds=base.workspace_prefix_seconds,
                    inside_fraction=info["workspace_inside_fraction"], finite=True))
        assert base.ff_steps > 0 and np.any(base.ff_peak > 0)
        assert np.all(base.total_torque_peak <= np.max(np.abs(base.original_torque_bounds),axis=1)+1e-8)
    finally:
        env.close()
    cfg = configuration(20000000,24)
    eval_env = CourseEnv(200, evaluation=True)
    eval_rows = []
    try:
        eval_env.set_stage(6)
        for scene_index, scene in enumerate(SCENES):
            mix = [0.] * len(SCENES)
            mix[scene_index] = 1.
            eval_env.env.set_training_scenario_distribution(SCENES, tuple(mix))
            obs, initial = eval_env.reset(seed=20260804+scene_index)
            for _ in range(5):
                obs, reward, done, truncated, info = eval_env.step(np.zeros(5))
                assert np.isfinite(obs).all() and np.isfinite(reward)
            eval_rows.append(dict(scene=scene, seed=initial["actual_seed"], goal=eval_env.episode_goal))
    finally:
        eval_env.close()
    save("prelaunch_verification.json",dict(passed=True,
        execution_module_sha256=hashlib.sha256((RUN/"full6_workspace_env.py").read_bytes()).hexdigest(),
        reward_methods_identical=True, task_success_methods_identical=True,
        unchanged_reward_curve=rewards, endpoint_nodes_allowed=[0,1,38,39],
        node_gate_removed=True, profile=PROFILE, actual_physics_trials=rows,
        candidate_node_coverage=sorted(all_candidates), selected_node_coverage=sorted(chosen),
        mean_trial_workspace_inside_fraction=float(np.mean([r["inside_fraction"] for r in rows])),
        held_out_evaluation_resets=eval_rows,
        first_checkpoint_configuration=cfg))
    print(json.dumps(dict(passed=True, scenes=len(SCENES), physical_trials=len(rows),
        candidate_nodes=sorted(all_candidates), selected_nodes=sorted(chosen)),indent=2),flush=True)


if __name__ == "__main__":
    main()
