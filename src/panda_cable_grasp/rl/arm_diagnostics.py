"""Compact control diagnostics for the explicit arm speed ablation."""
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ArmSpeedDiagnostics(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.samples = []
        self.reward_sums = {}

    def _on_step(self):
        actions = np.asarray(self.locals['actions'])
        worker_indices = self.locals.get('worker_indices', range(len(actions)))
        for rank, (info, done) in enumerate(zip(self.locals['infos'], self.locals['dones'])):
            worker = int(worker_indices[rank])
            sums = self.reward_sums.setdefault(worker, {})
            for key, value in info.items():
                if key.startswith('reward_') and isinstance(value, (int, float, np.number)):
                    sums[key] = sums.get(key, 0.) + float(value)
            actual = np.asarray(info['actual_hand_linear_velocity'])
            desired = np.asarray(info['commanded_world_translation'])/.02
            self.samples.append([
                np.linalg.norm(actual), np.linalg.norm(info['actual_hand_angular_velocity']),
                np.max(np.abs(info['actual_arm_velocity'])), np.linalg.norm(desired-actual),
                info.get('ik_task_residual_mixed_units', 0.),
                float(np.linalg.norm(actions[rank,:3]) >= 1), float(abs(actions[rank,3]) >= 1),
            ])
            if done:
                keys = ('episode_seed','scenario_name','success','strict_success','termination_reason',
                        'episode','rollout_episode_phase','rollout_episode_difficulty','ever_grasped',
                        'loaded_lift','ever_pinched','ever_aligned_pinch','lift_attempt',
                        'max_abs_actual_arm_velocity','max_actual_hand_linear_speed',
                        'max_actual_hand_angular_speed','actual_joint_velocity_exceedance_ratio',
                        'joint_velocity_limit_ratio','cartesian_velocity_limit_ratio',
                        'gripper_velocity_limit_ratio','gravity_torque_saturation_step_ratio',
                        'table_execution_guard','action_workspace_projection_count',
                        'action_workspace_projection_distance_m')
                record = dict(timesteps=self.num_timesteps, worker=worker,
                              reward_component_sums=sums.copy(),
                              metrics={k:info.get(k) for k in keys})
                with (self.output/'arm_episode_diagnostics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
                self.reward_sums[worker] = {}
        return True

    def _on_rollout_end(self):
        a = np.asarray(self.samples)
        names = ('tcp_speed_m_s','tcp_angular_speed_rad_s','max_joint_speed_rad_s',
                 'tcp_requested_actual_velocity_error_m_s','ik_residual_mixed_units',
                 'translation_action_saturation','yaw_action_saturation')
        record = dict(timesteps=self.num_timesteps, samples=len(a),
                      metrics={name:dict(mean=float(a[:,i].mean()),p95=float(np.quantile(a[:,i],.95)),
                                         max=float(a[:,i].max())) for i,name in enumerate(names)})
        with (self.output/'arm_rollout_diagnostics.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        self.samples.clear()
