"""Target-conditioned rule state machine under the PPO's frozen execution.

The RL wrapper is retained for identical scene/goal sampling and task judgment.
Its five input slots are placeholders; the rule sends native joint commands.
Rewards are bookkeeping only and must not be compared to PPO action rewards.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path
import sys
import time

RUN = Path(os.environ['FULL6_PPO_RUN'])
sys.path.insert(0,str(RUN))
from test_full6_table_guard import trainer, Observer, state, save
from full6_workspace_env import solve_full6, limit_norm, PROFILE
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig
from panda_cable_grasp.env.kinematics import quat_error, rotation_to_quat
import numpy as np


class TargetFull6Rule(DynamicCableGraspPolicy):
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.goal_task = wrapper.goal
        self.last_linear = np.zeros(3)
        self.last_angular = np.zeros(3)
        super().__init__(wrapper.base_env,PolicyConfig())
        # Same initial vertical frame as the previous PPO test, eliminating
        # a separate gripper-yaw change from the execution comparison.
        self.desired_quat = rotation_to_quat(wrapper._desired_hand_rotation)
        self.desired_approach_axis = wrapper._desired_hand_rotation @ self.env.gripper_approach_axis_local

    def _choose_initial_target(self, desired_rotation):
        self.selected_target_index = int(self.goal_task.index)
        self.selected_target_body_id = self.env.cable_ids[self.selected_target_index]
        self.selected_target_crossing_angle_deg = float('nan')
        self.selected_target_corridor_clearance = float('nan')

    def _nearest_cable_point(self, point):
        position,distance,tangent,index,alpha = self.goal_task.nearest(point)
        return position,distance,index,alpha

    def _select_intercept_segment(self, point):
        # Descend toward the requested material node, not the globally nearest
        # cable fold. Closure/contact feedback can lock a point in its same
        # allowed +/-4 window using the existing _lock_segment_near hook.
        node = int(self.goal_task.index)
        self.locked_segment_index = min(node,len(self.env.cable_ids)-2)
        self.locked_segment_alpha = float(node-self.locked_segment_index)
        self.locked_node_ids = list(self.env.cable_ids)
        self.filtered_target = self.env.data.xpos[self.env.cable_ids[node]].copy()
        return self.filtered_target.copy()

    def _ik_action(self, desired_position, gripper):
        w,b = self.wrapper,self.env
        self.last_desired = np.asarray(desired_position,dtype=float).copy()
        self.last_linear = limit_norm(6.*(self.last_desired-b.hand_position),PROFILE['linear_cap'])
        error = quat_error(rotation_to_quat(b.data.xmat[b.hand_id].reshape(3,3)),self.desired_quat)
        self.last_angular = PROFILE['angular_gain']*error
        target,qdot,scale,jacobian,twist = solve_full6(b,self.last_linear,self.last_angular)
        w.last_qdot,w.last_twist,w._last_ik_velocity_scale = qdot,twist,scale
        w._singularity_metrics(jacobian)
        w._last_ik_singularity_damping = PROFILE['damping']
        w._last_ik_singularity_target_clipped = False
        w._last_commanded_world_translation = self.last_linear*.02
        w._desired_hand_rotation = self._quat_matrix()
        result = np.r_[target,gripper]
        if w.table_finger_collision_filter_enabled and not b.config.geometric_safety_enabled:
            result[:7] = w._project_table_safe_target(result[:7])
        return result

    def _quat_matrix(self):
        import mujoco
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat,self.desired_quat)
        return mat.reshape(3,3)


def attach_rule(env):
    wrapper = env.env
    policy = TargetFull6Rule(wrapper)
    def convert_placeholder(ignored):
        native = policy.action()
        previous = wrapper._gripper_closed
        wrapper._gripper_closed = bool(native[7] < .5*wrapper.base_env.gripper_open_ctrl)
        wrapper._gripper_switch_event = bool(previous != wrapper._gripper_closed)
        wrapper._gripper_switch_count += int(wrapper._gripper_switch_event)
        return native
    wrapper._convert_action = convert_placeholder
    return policy


DYNAMIC_SCENES = ('id_rigid_l1_nominal','id_shape_nominal_current','id_combined_l1_nominal')


def configure_scene(env, stage, scenario=None):
    env.set_stage(stage)
    if scenario:
        env.env.set_training_scenario_distribution(trainer.SCENES,tuple(float(s == scenario) for s in trainer.SCENES))


def evaluate(job):
    episode,variant,output,ppo_directory,limit,passive_verify,dynamic = job
    trainer.torch.set_num_threads(1)
    started = time.monotonic()
    seed = 20260804+episode
    directory = Path(output)/variant/f'seed_{seed}'
    directory.mkdir(parents=True,exist_ok=False)
    env = trainer.CourseEnv(1000+episode,evaluation=True)
    stage = 4 if dynamic else 0
    scenario = DYNAMIC_SCENES[episode%3] if dynamic else None
    configure_scene(env,stage,scenario)
    obs,initial = env.reset(seed=seed)
    b = env.env.base_env
    initial_state = state(b)
    initial_sha = hashlib.sha256(initial_state.tobytes()).hexdigest()
    ppo = json.loads((Path(ppo_directory)/'baseline'/f'seed_{20260804 if dynamic else seed}'/'metadata.json').read_text())
    if not dynamic:
        assert initial_sha == ppo['initial_state_sha256']
        assert initial['actual_seed'] == ppo['actual_seed']
        assert env.env.goal.index == ppo['goal_node']
    else:
        assert b.config.scenario_name == scenario
    policy = attach_rule(env)
    assert np.array_equal(state(b),initial_state), 'Attaching rule altered physical state'
    observer = Observer(b,variant == 'protected')
    trajectory,observations,actions,rewards,rows = [initial_state],[obs.copy()],[],[],[]
    done = truncated = False
    info = initial
    while not (done or truncated):
        observer.row_force = observer.row_speed = 0.
        obs,reward,done,truncated,info = env.step(np.zeros(5,dtype=np.float32))
        native = info['mujoco_action'].copy()
        trajectory.append(state(b))
        observations.append(obs.copy())
        actions.append(native)
        rewards.append(float(reward))
        rows.append(dict(time=len(actions)*.02,task_success=bool(info.get('task_success',False)),
            secured=bool(info.get('secured_grasp',False)),actual_node=env.env.goal.actual_index(),
            lifted_fraction=float(info.get('lifted_fraction',0)),success_hold=float(b.success_hold),
            surface_clearance_m=min(p[1] for p in observer.points),
            down_speed_m_s=max(max(0.,-p[3]) for p in observer.points),
            normal_force_N=observer.row_force,contact_down_speed_m_s=observer.row_speed,
            guard_active=bool(observer.guard and observer.guard.last_active),phase=policy.phase.name,
            desired_position=policy.last_desired.tolist(),linear_request=policy.last_linear.tolist(),
            node_error=info.get('goal_node_error'),first_correct=info.get('first_grasp_correct'),
            goal_xyz=b.data.xpos[b.cable_ids[env.env.goal.index]].tolist(),
            goal_speed_m_s=float(np.linalg.norm(b.body_linear_velocity(b.cable_ids[env.env.goal.index]))),
            workspace_goal_inside=bool(info.get('workspace_goal_current_inside'))))
        if len(actions)%150 == 0:
            print(f'PROGRESS {variant} {seed} {len(actions)} phase={policy.phase.name}',flush=True)
        if limit and len(actions) >= limit:
            break
    meta = dict(seed=seed,episode=episode,actual_seed=initial['actual_seed'],variant=variant,
        policy_kind='scripted',stage=stage,stage_name=trainer.STAGES[stage]['name'],
        runtime=dict(platform=platform.platform(),python=platform.python_version(),mujoco=__import__('mujoco').__version__,torch=trainer.torch.__version__),
        scenario=info.get('scenario_name',b.config.scenario_name),scenario_override=scenario,
        dynamic_test=dynamic,motion_difficulty=trainer.STAGES[stage]['motion'],goal_node=env.env.goal.index,
        goal_initial_xyz=initial['workspace_initial_goal_xyz'],initial_state_sha256=initial_sha,
        initial_matches_previous_ppo=not dynamic,reference_ppo_checkpoint_sha256=ppo['checkpoint_sha256'],
        workspace_inside_fraction=float(info.get('workspace_inside_fraction',1.)),
        workspace_reset_attempts=initial['workspace_reset_attempts'],workspace_prefix_seconds=initial['workspace_prefix_seconds'],
        goal_pregrasp_max_speed_m_s=max((r['goal_speed_m_s'] for r in rows if r['first_correct'] is None),default=0.),
        first_correct=info.get('first_grasp_correct'),first_grasp_node=info.get('first_grasp_index'),
        task_success=bool(info.get('task_success',False)),terminated=bool(done),truncated=bool(truncated),
        termination_reason=info.get('termination_reason') or ('timeout' if truncated else 'success' if done else 'pilot_limit'),
        return_=sum(rewards),reward_comparison_valid=False,
        reward_note='Original reward is bookkeeping with zero placeholder normalized actions; rule does not consume it.',
        max_lifted_fraction=max(r['lifted_fraction'] for r in rows),
        max_success_hold_s=max(r['success_hold'] for r in rows),steps=len(actions),frames=len(trajectory),duration_s=len(actions)*.02,
        checkpoint=None,checkpoint_steps=0,checkpoint_sha256=None,controller_profile=PROFILE,
        execution_source_sha256=trainer.sha(RUN/'full6_workspace_env.py'),
        frozen_reward_sha256=trainer.sha(RUN/'frozen_code/src/panda_cable_grasp/rl/environment.py'),
        frozen_target_sha256=trainer.sha(RUN/'frozen_code/src/panda_cable_grasp/rl/target_node.py'),
        frozen_rule_sha256=trainer.sha(RUN/'frozen_code/src/panda_cable_grasp/policies/scripted.py'),
        training_config_sha256=trainer.sha(RUN/'training_config.json'),
        target_node_tolerance=int(env.env.rl_config.target_node_tolerance),reward_changes=False,
        guard_source_sha256=trainer.sha(Path(__file__).parent/'table_execution_guard.py'),
        observer_source_sha256=trainer.sha(Path(__file__).parent/'test_full6_table_guard.py'),
        runner_source_sha256=trainer.sha(Path(__file__)),rule_config=asdict(policy.config),
        rule_info=policy.policy_info(),rule_phase=policy.phase.name,rule_result=policy.result,
        adapter='fixed assigned node; closest-point closure within same goal tolerance; same initial vertical frame; linear gain 6; shared solve_full6',
        pilot_limit=limit,wall_s=time.monotonic()-started,**observer.summary())
    for key in ('execution_source_sha256','frozen_reward_sha256','frozen_target_sha256','training_config_sha256','target_node_tolerance','guard_source_sha256'):
        assert meta[key] == ppo[key], key
    if passive_verify:
        assert variant == 'baseline'
        plain = trainer.CourseEnv(1000+episode,evaluation=True)
        configure_scene(plain,stage,scenario)
        plain_obs,_ = plain.reset(seed=seed)
        plain_policy = attach_rule(plain)
        difference = float(np.max(abs(state(plain.env.base_env)-trajectory[0])))
        for i in range(len(actions)):
            plain_obs,r,d,t,plain_info = plain.step(np.zeros(5,dtype=np.float32))
            assert np.array_equal(plain_info['mujoco_action'],actions[i])
            assert np.array_equal(plain_obs,observations[i+1])
            assert r == rewards[i]
            difference = max(difference,float(np.max(abs(state(plain.env.base_env)-trajectory[i+1]))))
        assert difference == 0
        meta['passive_observer_max_state_difference'] = difference
        plain.close()
    np.savez_compressed(directory/'trajectory.npz',states=np.asarray(trajectory),
        observations=np.asarray(observations),actions=np.asarray(actions),rewards=np.asarray(rewards),
        frame_times=np.arange(len(trajectory))*.02)
    meta['trajectory_sha256'] = trainer.sha(directory/'trajectory.npz')
    save(directory/'metadata.json',meta)
    save(directory/'metrics.json',rows)
    env.close()
    print(f'DONE {variant} {seed} contact={meta["table_contact_episode"]} first={meta["first_correct"]} success={meta["task_success"]} phase={meta["rule_phase"]}',flush=True)
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--ppo-directory',required=True)
    parser.add_argument('--episodes',type=int,default=16)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--start-episode',type=int,default=0)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--verify-first-baseline',action='store_true')
    parser.add_argument('--dynamic-scenes',action='store_true')
    args = parser.parse_args()
    jobs = [(e,v,args.output,args.ppo_directory,args.limit,args.verify_first_baseline and e==0 and v=='baseline',args.dynamic_scenes)
        for e in range(args.start_episode,args.start_episode+args.episodes) for v in ('baseline','protected')]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        results = list(pool.map(evaluate,jobs))
    for e in range(args.episodes):
        a,b = results[2*e:2*e+2]
        for key in ('initial_state_sha256','goal_node','actual_seed','execution_source_sha256','guard_source_sha256','frozen_rule_sha256','rule_config','scenario','stage','motion_difficulty'):
            assert a[key] == b[key], (e,key)
    save(Path(args.output)/'manifest.json',dict(policy='scripted',results=results,
        paired_verification_passed=True,initial_matches_previous_ppo=not args.dynamic_scenes,
        dynamic_test=args.dynamic_scenes,episodes_per_variant=args.episodes,workers=args.workers))
