import numpy as np
import pytest
from panda_cable_grasp.env.environment import GraspState
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env


@pytest.mark.parametrize('profile,paid', [('tolerance_progress_v4',True),('tolerance_progress_v5',False)])
def test_grasp_grace_does_not_pay_lift_without_live_contacts(profile, paid):
    env=make_rl_env(robot='nero',action_mode='task_space_vertical_down',scenario_names=['id_static'],
                   rl_config=RLConfig(target_node_task=True,target_reward_profile=profile))
    try:
        env.reset(seed=81,options={'goal_node_index':20})
        env.base_env.grasp_state=GraspState(body_id=env.base_env.cable_ids[20],candidate_time=0.,
            bilateral_confirmed=True,last_bilateral_time=0.,lost_contact_time=0.)
        table=env.base_env.table_geom_id
        top=env.data.geom_xpos[table,2]+env.model.geom_size[table,2]
        status=dict(pinch_confirmed=True,raw_bilateral_contact=False,secured_grasp=False,
                    grasp_lift_delta=.05,grasp_body_height_above_table=.09,strict_success_hold=0.,height_band_hold=0.)
        env.data.xpos[env.base_env.cable_ids[20]]=[0.,0.,top+.09]
        env._grasp_center_position=lambda: np.array([0.,0.,top+.09])
        _,parts=env._reward(np.zeros(5),False,{'lifted_fraction':.1},status)
        assert (parts['reward_lift_progress']>0) is paid
        assert (parts['reward_cable_lift_progress']>0) is paid
        if not paid:
                # Restoring contact at the same externally observed height cannot claim it.
            status['raw_bilateral_contact']=True
            env.data.qpos[env.base_env.finger_qpos_adr]=[.012,-.012]
            assert env._current_lift_reward_holding(status)
            _,parts=env._reward(np.zeros(5),False,{'lifted_fraction':.1},status)
            assert parts['reward_lift_progress']==parts['reward_cable_lift_progress']==0.
            env.data.xpos[env.base_env.cable_ids[20],2]=top+.10
            _,parts=env._reward(np.zeros(5),False,{'lifted_fraction':.2},status)
            assert parts['reward_lift_progress']>0 and parts['reward_cable_lift_progress']>0
            env.data.qpos[env.base_env.finger_qpos_adr]=[.05,-.05]
            assert not env._current_lift_reward_holding(status)
            env.data.qpos[env.base_env.finger_qpos_adr]=[.012,-.012]
            env._grasp_center_position=lambda: np.array([1.,0.,top+.10])
            assert not env._current_lift_reward_holding(status)
            env.data.xpos[env.base_env.cable_ids[20],2]=top+.12
            _,parts=env._reward(np.zeros(5),False,{'lifted_fraction':.3},status)
            assert parts['reward_lift_progress']==parts['reward_cable_lift_progress']==0.
    finally:
        env.close()
