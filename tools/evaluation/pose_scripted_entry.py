"""Explicit Pose IK entry for isolated scripted experiments.

--verify-entry validates routing only. It does not produce grasp success rates.
Pose uses the project's absolute-pose adapter at its original orientation
weight. The old Jacobian angular-feedback multiplier is deliberately absent.
The common environment retains its actuator and command limits. Unlike the
old velocity-level policy, the solver receives the absolute stage target;
this difference must be declared in any subsequent experiment manifest.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import compare_scripted_tracking_fix as previous
from panda_cable_grasp.dynamicvla.adapter import (
    DynamicVLAAdapterConfig, DynamicVLATaskSpaceAdapter,
)
from panda_cable_grasp.env.environment import EnvConfig

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/benchmarks/nero_pose_scripted_tracking_fix_20261001'


class PoseIKScriptedPolicy(previous.DiagnosticPolicy):
    def __init__(self, env, config=None):
        super().__init__(env,config)
        self.pose_adapter=DynamicVLATaskSpaceAdapter(
            env,DynamicVLAAdapterConfig(nero_pose_ik_enabled=True))
        self.pose_ik_calls=0

    def _ik_action(self, desired_position, gripper):
        self.pose_ik_calls+=1
        action=np.empty(8)
        action[:7]=self.pose_adapter._pose_ik_target(
            np.asarray(desired_position,dtype=float),self.desired_quat)
        action[self.env.gripper_actuator_id]=gripper
        self.last_desired=np.asarray(desired_position,dtype=float).copy()
        return action


def verify_entry():
    old_folder=previous.OUT/'main'
    manifest=json.loads((old_folder/'manifest.json').read_text(encoding='utf-8'))
    rows=json.loads((old_folder/'episodes.json').read_text(encoding='utf-8'))
    assert len(rows)==900
    for path,expected in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    assert all(r['scripted_config']['target_selection_mode']=='middle' and
        not any(r['scripted_config'][flag] for flag in (
            'nero_weighted_pose_ik','strict_vertical_gripper',
            'nero_tilt_only_orientation_control','nero_post_grasp_orientation_priority',
            'nero_intercept_orientation_priority','nero_approach_orientation_priority'))
        for r in rows)
    previous.OUT=OUT
    env=previous.DiagnosticEnv(EnvConfig(robot='nero',seed=2026100000,
        scenario_name='id_static'),variant='tracking_fix')
    env.reset(seed=2026100000)
    policy=PoseIKScriptedPolicy(env)
    qpos=env.data.qpos.copy(); qvel=env.data.qvel.copy()
    target=env.hand_position.copy()+np.array([.002,0.,0.])
    with patch.object(policy.pose_adapter,'_pose_ik_target',
        wraps=policy.pose_adapter._pose_ik_target) as solver:
        actual=policy._ik_action(target,env.gripper_open_ctrl)
        assert solver.call_count==1
    expected=policy.pose_adapter._pose_ik_target(target,policy.desired_quat)
    assert np.array_equal(actual[:7],expected)
    assert np.array_equal(qpos,env.data.qpos) and np.array_equal(qvel,env.data.qvel)
    assert np.all(np.isfinite(actual)) and actual[7]==env.gripper_open_ctrl
    result={'scope':'entry verification only; no grasp success claim',
        'previous_900_episodes_solver':'hierarchical velocity-level Jacobian IK',
        'previous_production_source_hashes_still_match':True,
        'pose_solver':'DynamicVLATaskSpaceAdapter._pose_ik_target',
        'verified_real_pose_solver_call':True,'verified_same_official_solver_result':True,
        'solver_restores_physical_qpos_and_qvel':True,
        'adapter_config':asdict(policy.pose_adapter.config),
        'old_angular_feedback_x4_applied':False,
        'corrected_entry_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'next_run_requirement':'Record absolute-stage-target Pose conversion explicitly; reuse old baselines only with matching model, scene, motion, seeds, physics and source hashes.'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'entry_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-entry',action='store_true',required=True)
    parser.parse_args()
    verify_entry()
