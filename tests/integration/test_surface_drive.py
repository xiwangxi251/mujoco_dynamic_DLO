"""Physical driving contract, including a partly lifted cable and current clock."""
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig, GraspState
from panda_cable_grasp.scenarios import get_scenario
from tests.integration import test_rigid_replay_env as fixtures

NAMES=('dev_rigid_l1_nominal_x075_surface_v1','dev_rigid_l1_midshape_v1_x075_surface_v1',
       'dev_combined_l1_nominal_x075_surface_v1','dev_rigid_replay_shape_nominal_x075_surface_v1')


class SurfaceDriveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.ReplayPairingTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):fixtures.ReplayPairingTests.tearDownClass.__func__(cls)

    def make(self,name):
        overrides=get_scenario(name).to_env_overrides()
        if overrides.get('initial_shape_bank'):overrides['initial_shape_bank']=fixtures.BANK_NAME
        if overrides.get('replay_bank'):overrides['replay_bank']=fixtures.BANK_NAME
        return CableGraspEnv(EnvConfig(robot='panda',episode_seconds=1.2,replay_bank_dir=str(self.bank_dir),**overrides))

    def test_grasp_airborne_and_return_never_pause_or_compile(self):
        with (patch('panda_cable_grasp.env.force_replay.prepare_force_replay',side_effect=AssertionError('compiler preparation')),
              patch('panda_cable_grasp.env.force_replay.compile_force_replay',side_effect=AssertionError('force compilation'))):
            for name in NAMES:
                with self.subTest(scene=name):
                    env=self.make(name)
                    try:
                        env.reset(seed=fixtures.SEED)
                        for _ in range(43):env.step(env.ready_ctrl)
                        env.grasp_state=GraspState(body_id=int(env.target_body_id),candidate_time=env.data.time,
                            bilateral_confirmed=True,last_bilateral_time=env.data.time,lost_contact_time=env.data.time)
                        assert env.grasp_confirmed
                        env._update_rigid_motion_suspension_state()
                        assert not env.rigid_motion_suspended
                        env._apply_cable_disturbance()
                        assert np.linalg.norm(env._last_ungated_rigid_force)>0
                        initial_elapsed=env._replay_elapsed
                        env.data.qpos[env.cable_free_qadr:env.cable_free_qadr+3]+=[.02,.03,.3]
                        mujoco.mj_forward(env.model,env.data)
                        assert not env._cable_table_contact_mask().any()
                        for _ in range(2):env.step(env.ready_ctrl)
                        np.testing.assert_array_equal(env._last_ungated_rigid_force,0.)
                        assert env._replay_elapsed>initial_elapsed
                        env.grasp_state=None
                        env.data.qpos[env.cable_free_qadr+2]-=.3
                        env.data.qvel[:]=0.
                        mujoco.mj_forward(env.model,env.data)
                        env._update_rigid_motion_suspension_state()
                        assert not env.rigid_motion_suspended
                        env._apply_cable_disturbance()
                        assert env._cable_table_contact_mask().any()
                        assert np.linalg.norm(env._last_ungated_rigid_force)>0
                        assert env._replay_anchor_elapsed==0.
                        assert env._replay_force_tape is None and env._replay_force_compiler is None
                    finally:env.close()

    def test_partial_real_table_contacts_keep_force_after_grasp(self):
        env=self.make(NAMES[0])
        try:
            env.reset(seed=fixtures.SEED)
            for _ in range(43):env.step(env.ready_ctrl)
            # Bend the tail upward; front nodes retain real table contacts.
            qadr=int(env.cable_ball_qadr[len(env.cable_ball_qadr)//2])
            env.data.qpos[qadr:qadr+4]=[np.cos(np.pi/6),0.,-np.sin(np.pi/6),0.]
            mujoco.mj_forward(env.model,env.data)
            grounded=env._surface_table_support_mask()
            assert grounded.any() and not grounded.all(),grounded
            env.grasp_state=GraspState(body_id=int(env.target_body_id),candidate_time=env.data.time,
                bilateral_confirmed=True,last_bilateral_time=env.data.time,lost_contact_time=env.data.time)
            env._update_rigid_motion_suspension_state()
            env._apply_cable_disturbance()
            np.testing.assert_array_equal(env._last_ungated_rigid_force[~grounded],0.)
            assert np.linalg.norm(env._last_ungated_rigid_force[grounded])>0
            assert not env.rigid_motion_suspended
        finally:env.close()

    def test_replay_displacement_has_no_position_catchup_and_combined_shape_is_live(self):
        for name in (NAMES[2],NAMES[3]):
            with self.subTest(scene=name):
                env=self.make(name)
                try:
                    env.reset(seed=fixtures.SEED)
                    for _ in range(43):env.step(env.ready_ctrl)
                    if env.config.motion_mode=='combined':
                        expected=env._shape_acceleration(env.phase_offset+env.config.motion_frequency_scale*env.data.time,env.spatial_phase)
                        env._apply_cable_disturbance()
                        np.testing.assert_allclose(env._last_shape_acceleration,expected)
                    else:
                        # mj_step's derived xpos/velocity cache represents its
                        # last integration stage. Compare two equally refreshed
                        # states, so this check changes only world translation.
                        mujoco.mj_forward(env.model,env.data)
                        mask=env._surface_table_support_mask()
                        env._apply_cable_disturbance();before=env._last_ungated_rigid_force.copy()
                        env.data.qpos[env.cable_free_qadr:env.cable_free_qadr+2]+=[.2,-.2]
                        mujoco.mj_forward(env.model,env.data)
                        env._apply_cable_disturbance()
                        np.testing.assert_array_equal(env._last_replay_force_grounded,mask)
                        np.testing.assert_allclose(env._last_ungated_rigid_force,before,atol=1e-9)
                        assert env._replay_anchor_elapsed==0.
                finally:env.close()

    def test_near_table_without_contacts_stays_driven_and_lift_return_has_hysteresis(self):
        env=self.make(NAMES[0])
        try:
            env.reset(seed=fixtures.SEED)
            # Straight, horizontal cable: translate its real collision geometry
            # to known positive gaps, where no penetration contacts exist.
            env.data.qpos[env.cable_free_qadr+3:env.cable_free_qadr+7]=[1.,0.,0.,0.]
            for qadr in env.cable_ball_qadr:
                env.data.qpos[qadr:qadr+4]=[1.,0.,0.,0.]
            env.data.qvel[:]=0.
            env.data.time=.9
            mujoco.mj_forward(env.model,env.data)
            env._surface_table_support_mask()
            assert np.ptp(env._surface_table_gaps)<1e-10
            for gap,expected in ((.0005,True),(.002,True),(.0035,False),
                                 (.002,False),(.0005,True)):
                env.data.qpos[env.cable_free_qadr+2]+=gap-env._surface_table_gaps[0]
                mujoco.mj_forward(env.model,env.data)
                assert not env._cable_table_contact_mask().any()
                with patch.object(env,'_cable_table_contact_mask',side_effect=AssertionError('contact gate')):
                    env._apply_cable_disturbance()
                np.testing.assert_allclose(env._surface_table_gaps,gap,atol=1e-10)
                np.testing.assert_array_equal(env._last_replay_force_grounded,expected)
                if expected:assert np.linalg.norm(env._last_ungated_rigid_force)>0.
                else:np.testing.assert_array_equal(env._last_ungated_rigid_force,0.)
            # A near-height strand outside the table has no table drive.
            env.data.qpos[env.cable_free_qadr]+=10.
            mujoco.mj_forward(env.model,env.data)
            assert not env._surface_table_support_mask().any()
            env.reset(seed=fixtures.SEED)
            assert not env._surface_supported.any()
            assert np.isinf(env._surface_table_gaps).all()
        finally:env.close()
