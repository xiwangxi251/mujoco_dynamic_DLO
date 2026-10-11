import math
import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from panda_cable_grasp.policies.scripted import DynamicCableGraspPolicy, PolicyConfig, Phase
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig


class TargetTangentTests(unittest.TestCase):
    def setUp(self):
        self.env = CableGraspEnv(EnvConfig(robot='nero', motion_mode='static'))
        self.env.reset(seed=20260804)
        self.policy = DynamicCableGraspPolicy(self.env, PolicyConfig())

    def tearDown(self):
        self.env.close()

    def rotation(self):
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, self.policy.desired_quat)
        return matrix.reshape(3, 3)

    def set_segment(self, angle):
        self.policy.locked_segment_index = 3
        self.policy.locked_node_ids = list(self.env.cable_ids)
        ids = self.policy.locked_node_ids
        self.env.data.xpos[ids[3]] = [0., 0., .014]
        self.env.data.xpos[ids[4]] = [.02 * math.cos(angle), .02 * math.sin(angle), .014]

    def test_alignment_uses_locked_segment_and_preserves_vertical_axis(self):
        for angle in (0., .4, 1.7, -2.4):
            self.set_segment(angle)
            self.policy._update_target_tangent_orientation()
            rotation = self.rotation()
            closing = rotation @ self.env.gripper_closing_axis_local
            self.assertAlmostEqual(float(closing[:2] @ [math.cos(angle), math.sin(angle)]), 0., places=8)
            np.testing.assert_allclose(rotation @ self.env.gripper_approach_axis_local, [0.,0.,-1.], atol=1e-8)

    def test_reversed_tangent_does_not_rotate_gripper_by_pi(self):
        self.set_segment(.7)
        self.policy._update_target_tangent_orientation()
        before = self.rotation().copy()
        self.set_segment(.7 + math.pi)
        self.policy._update_target_tangent_orientation()
        np.testing.assert_allclose(self.rotation(), before, atol=1e-8)

    def test_degenerate_horizontal_tangent_keeps_previous_orientation(self):
        self.set_segment(.7)
        self.policy._update_target_tangent_orientation()
        before = self.rotation().copy()
        ids = self.policy.locked_node_ids
        self.env.data.xpos[ids[4]] = self.env.data.xpos[ids[3]] + [0.,0.,.02]
        self.policy._update_target_tangent_orientation()
        np.testing.assert_allclose(self.rotation(), before, atol=1e-8)

    def test_lift_keeps_grasp_orientation(self):
        self.set_segment(.7)
        self.policy._update_target_tangent_orientation()
        before = self.rotation().copy()
        self.policy.phase = Phase.LIFT
        self.set_segment(1.2)
        self.policy._update_target_tangent_orientation()
        np.testing.assert_allclose(self.rotation(), before, atol=1e-8)

    def test_option_off_keeps_fixed_orientation(self):
        policy = DynamicCableGraspPolicy(self.env, PolicyConfig(follow_target_tangent=False))
        before = policy.desired_quat.copy()
        policy._update_target_tangent_orientation()
        np.testing.assert_array_equal(policy.desired_quat, before)

if __name__ == '__main__':
    unittest.main()
