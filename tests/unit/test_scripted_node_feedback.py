"""Node-only feedback must not read contact/grasp/evaluation truth."""
import unittest
import numpy as np
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig, Phase


class NodeOnlyEnv:
    forbidden = {'finger_contacts', 'grasp_state', 'grasp_confirmed',
                 'last_grasp_break', 'success_hold', 'success_snapshot', 'ever_success'}

    def __init__(self, env):
        self.base = env

    def __getattr__(self, name):
        if name in self.forbidden:
            raise AssertionError('Forbidden policy input: '+name)
        return getattr(self.base, name)


class NodeFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.env = CableGraspEnv(EnvConfig(robot='nero', motion_mode='static'))
        self.env.reset(seed=20260804)
        self.policy = DynamicCableGraspPolicy(NodeOnlyEnv(self.env), PolicyConfig())
        self.policy._ik_action = lambda position, gripper: np.r_[position, gripper]
        self.policy.locked_segment_index = 3
        self.policy.locked_segment_alpha = .5
        self.policy.locked_node_ids = list(self.env.cable_ids)
        self.ids = self.env.cable_ids[3:5]
        self.env.data.xpos[self.ids, 2] = .02

    def tearDown(self):
        self.env.close()

    def test_close_waits_then_lifts_without_grasp_truth(self):
        p = self.policy
        p._transition(Phase.CLOSE)
        self.env.data.time = .78
        p.action()
        self.assertEqual(p.phase, Phase.CLOSE)
        self.env.data.time = .80
        p.action()
        self.assertEqual(p.phase, Phase.LIFT)
        self.assertAlmostEqual(p._lift_reference_height, .02)
        self.assertEqual(p.locked_segment_index, 3)

    def test_failed_height_check_opens_and_recovers(self):
        p = self.policy
        p._lift_reference_height = .02
        p._transition(Phase.LIFT)
        self.env.data.time = .98
        p.action()
        self.assertEqual(p.phase, Phase.LIFT)
        self.env.data.time = 1.
        action = p.action()
        self.assertEqual(p.phase, Phase.RECOVER)
        self.assertEqual(p.last_attempt_failure, 'failed_no_lift')
        self.assertEqual(action[-1], self.env.gripper_open_ctrl)

    def test_raised_material_point_passes_height_check(self):
        p = self.policy
        p._lift_reference_height = .02
        p._transition(Phase.LIFT)
        self.env.data.xpos[self.ids, 2] = .06
        self.env.data.time = 1.
        p.action()
        self.assertEqual(p.phase, Phase.LIFT)

    def test_drop_during_carry_recovers_without_break_truth(self):
        p = self.policy
        p._lift_reference_height = .02
        p._transition(Phase.CARRY)
        p.action()
        self.assertEqual(p.phase, Phase.RECOVER)
        self.assertEqual(p.last_attempt_failure, 'failed_lost_lift')

    def test_hold_does_not_read_evaluation_success(self):
        p = self.policy
        p._lift_reference_height = .02
        self.env.data.xpos[self.ids, 2] = .06
        p._transition(Phase.HOLD)
        p.action()
        self.assertEqual(p.phase, Phase.HOLD)
        self.assertFalse(p.finished)

    def test_selected_material_point_survives_nearer_fold_and_closure(self):
        p = self.policy
        p.config.preserve_selected_material_target = True
        p.selected_target_index = 11
        points = self.env.data.xpos[self.env.cable_ids].copy()
        hand = points[7].copy()
        target = p._select_intercept_segment(hand)
        np.testing.assert_allclose(target, points[11])
        self.assertEqual(p.locked_segment_index, 11)
        np.testing.assert_allclose(p._lock_segment_near(points[7]), points[11])
        self.assertEqual(p.locked_segment_index, 11)
        self.assertEqual(p.locked_segment_alpha, 0.)

    def test_retry_explicitly_reselects_geometry_target(self):
        p = self.policy
        p.config.preserve_selected_material_target = True
        calls = []
        p._choose_initial_target = lambda rotation: calls.append(rotation)
        p.recover_start = self.env.hand_position.copy()
        p.recover_goal = p.recover_start.copy()
        p._transition(Phase.RECOVER)
        self.env.data.time = 1.02
        p.action()
        self.assertEqual(len(calls), 1)
        self.assertEqual(p.phase, Phase.APPROACH)


if __name__ == '__main__':
    unittest.main()
