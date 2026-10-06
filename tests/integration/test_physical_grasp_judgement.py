"""Physical grasp/qualification must not follow an actuator target alone."""
import unittest
from unittest.mock import patch
import numpy as np

from panda_cable_grasp.env.environment import GraspState
from panda_cable_grasp.rl.environment import RLCableGraspEnv


class PhysicalGraspJudgementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rl = RLCableGraspEnv(robot='nero', scenario_names=('id_static',))
        cls.base = cls.rl.base_env

    @classmethod
    def tearDownClass(cls):
        cls.rl.close()

    def setUp(self):
        self.rl.reset(seed=100)
        self.base.data.qpos[self.base.finger_qpos_adr] = [.015, -.015]
        self.base.data.ctrl[self.base.gripper_actuator_id] = .05
        self.base.grasp_state = GraspState(self.base.target_body_id, 0., True,
                                          float(self.base.data.time), 0.)
        self.pairs = [(self.base.target_body_id,self.base.left_finger_id),
                      (self.base.target_body_id,self.base.right_finger_id)]
        self.stack = []
        for name, value in [('_current_grasp_error',lambda:.01),
                            ('_physical_grasp_candidate',lambda:self.base.target_body_id),
                            ('_finger_body_contact_pairs',lambda *args:self.pairs),
                            ('_finger_contact_pairs',lambda *args:self.pairs)]:
            p = patch.object(self.base,name,value)
            p.start(); self.stack.append(p)

    def tearDown(self):
        for p in reversed(self.stack): p.stop()

    def test_open_target_keeps_physical_hold_and_does_not_emit_release_penalty(self):
        self.rl._gripper_closed = False
        for _ in range(20):
            self.base._update_physical_grasp_state(gripper_closed=False)
        self.assertTrue(self.base.grasp_confirmed)
        self.assertEqual(self.base.grasp_break_history,[])
        self.assertEqual(self.rl._classify_secured_grasp_break(),(False,False,False))

    def test_force_dropout_with_bilateral_contacts_preserves_confirmed_hold(self):
        with patch.object(self.base,'_physical_grasp_candidate',return_value=None):
            for _ in range(200):
                self.base._update_physical_grasp_state(gripper_closed=False)
        self.assertTrue(self.base.grasp_confirmed)
        self.assertEqual(self.base.grasp_state.lost_contact_time,0.)

    def test_real_opening_clears_once_and_reward_event_is_consumed_once(self):
        self.base._update_physical_grasp_state(gripper_closed=False)
        self.base.data.qpos[self.base.finger_qpos_adr] = [.025,-.025]
        self.rl._secured_session_active = True
        self.base._update_physical_grasp_state(gripper_closed=False)
        self.assertIsNone(self.base.grasp_state)
        self.assertEqual(self.base.last_grasp_break['reason'],'physical_grasp_geometry_lost')
        self.assertEqual(self.base.last_grasp_break['causal_class'],'active_open')
        self.assertEqual(self.rl._classify_secured_grasp_break(),(True,False,False))
        self.assertEqual(self.rl._classify_secured_grasp_break(),(False,False,False))

    def test_contact_loss_under_close_command_still_clears_after_grace(self):
        self.pairs = []
        with patch.object(self.base,'_physical_grasp_candidate',return_value=None):
            count = int(np.ceil(self.base.config.grasp_loss_seconds/self.base.model.opt.timestep))+1
            for _ in range(count): self.base._update_physical_grasp_state(gripper_closed=True)
        self.assertIsNone(self.base.grasp_state)
        self.assertEqual(self.base.last_grasp_break['causal_class'],'physical_slip')

    def test_success_independent_of_command_tolerates_contact_grace(self):
        self.base.data.xpos[np.asarray(self.base.cable_ids),2] = .22
        with patch.object(type(self.base),'pad_center_position',new_callable=lambda:property(lambda _:self.base.data.xpos[self.base.target_body_id].copy())):
            self.assertTrue(self.base._success_qualification(gripper_closed=False))
            self.assertTrue(self.base._success_qualification(gripper_closed=True))
            self.pairs = []
            self.assertTrue(self.base._success_qualification(gripper_closed=True))
            self.base.data.time += self.base.config.grasp_loss_seconds + .002
            self.assertFalse(self.base._success_qualification(gripper_closed=True))

    def test_public_success_rejects_real_opening_despite_contact_grace(self):
        self.base.data.xpos[np.asarray(self.base.cable_ids),2] = .22
        with patch.object(type(self.base),'pad_center_position',new_callable=lambda:property(lambda _:self.base.data.xpos[self.base.target_body_id].copy())):
            self.base.data.qpos[self.base.finger_qpos_adr] = [.025,-.025]
            self.assertFalse(self.base._success_qualification(gripper_closed=True))

    def test_secured_contact_grace_and_strict_hold_do_not_depend_on_command(self):
        self.rl._secured_candidate_hold = self.rl.rl_config.secured_confirm_seconds
        self.rl._gripper_closed = False
        self.base.data.xpos[np.asarray(self.base.cable_ids),2] = .22
        center=self.base.data.xpos[self.base.target_body_id].copy()
        with patch.object(self.rl,'_grasp_center_position',return_value=center):
            info={'lifted_fraction':1.}
            status=self.rl._update_grasp_status(info)
            self.assertTrue(status['secured_grasp'])
            self.assertGreater(self.rl._strict_success_hold,0.)
            self.pairs=[]
            status=self.rl._update_grasp_status(info)
            self.assertTrue(status['secured_grasp'])
            self.assertEqual(self.rl._strict_success_hold,0.)


if __name__=='__main__': unittest.main()
