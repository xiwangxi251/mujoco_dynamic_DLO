import argparse
from dataclasses import replace
import json
import numpy as np
import pytest
from panda_cable_grasp.rl.environment import RLConfig, VerticalDownRLCableGraspEnv
from panda_cable_grasp.env.control import scaled_full6_profile


def test_scaled_limits_survive_reset_and_preserve_physics():
    kwargs=dict(robot='nero',controller_profile='enhanced6_bounded_dls_v1',
                scenario_names=('dev_static_x075_v1',),table_execution_guard_enabled=True)
    a=VerticalDownRLCableGraspEnv(**kwargs)
    b=VerticalDownRLCableGraspEnv(**kwargs,rl_config=RLConfig(
        arm_velocity_limit_scale=2.,translation_delta_scale=.02,yaw_delta_scale=.04))
    try:
        for seed in (20,21):
            b.reset(seed=seed)
            b.base_env.configure_controller_profile('enhanced6_bounded_dls_v1')
            np.testing.assert_allclose(b.base_env._arm_velocity_limits,2*a.base_env._arm_velocity_limits)
            assert b.base_env.config.hand_linear_velocity_limit==1.8
            assert b.base_env.config.hand_angular_velocity_limit==4.
            assert b._full6_control_profile.angular_velocity_limit==2.8
            assert b.base_env.config.hand_cartesian_velocity_limit_enabled
            assert b.base_env.config.gripper_finger_velocity_limit==.2
            assert b.execution_configuration()['compiled_hashes']==a.execution_configuration()['compiled_hashes']
            base=b.base_env
            base.data.qvel[base.arm_dof_adr]=base._arm_velocity_limits+1
            base.data.time=1.
            target=base.ready_ctrl.copy();target[:7]=base.data.qpos[base.arm_qpos_adr]+.01
            guarded,changed=base._velocity_guarded_action(target)
            assert changed
            np.testing.assert_allclose(guarded[:7],base.data.qpos[base.arm_qpos_adr])
    finally:a.close();b.close()


def test_profile_preserves_other_terms_and_saved_scale_restores(tmp_path):
    from panda_cable_grasp.rl.evaluate import evaluation_rl_config
    baseline=scaled_full6_profile();scaled=scaled_full6_profile(2.)
    assert scaled==replace(baseline,linear_velocity_limit=1.8,angular_velocity_limit=2.8)
    (tmp_path/'training_config.json').write_text(json.dumps({'rl_config':{'arm_velocity_limit_scale':2.}}))
    assert evaluation_rl_config(argparse.Namespace(model=tmp_path/'model.zip')).arm_velocity_limit_scale==2.
    for bad in (-1,0,True,float('nan'),float('inf')):
        with pytest.raises(ValueError):scaled_full6_profile(bad)


def test_reference_and_cli_scale_override(tmp_path):
    from panda_cable_grasp.rl.train import rl_config_from_args
    path=tmp_path/'reference.json'
    path.write_text(json.dumps({'rl_config':{'arm_velocity_limit_scale':2.,'arm_velocity_limits_enabled':True}}))
    args=argparse.Namespace(rl_config_file=path,target_reward_profile='legacy',arm_velocity_limit_scale=None)
    assert rl_config_from_args(args).arm_velocity_limit_scale==2.
    args.arm_velocity_limit_scale=1.
    assert rl_config_from_args(args).arm_velocity_limit_scale==1.
