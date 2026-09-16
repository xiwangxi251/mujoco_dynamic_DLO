import unittest

import numpy as np
import torch

from panda_cable_grasp.rl.pointcloud import (
    PointNet2FeaturesExtractor,
    camera_matrix,
    fixed_farthest_point_sample,
)


class PointCloudRLTests(unittest.TestCase):
    def test_camera_matrix_uses_vertical_fov(self):
        matrix = camera_matrix(240, 180, 90.0)
        self.assertAlmostEqual(matrix[0, 0], 90.0)
        self.assertAlmostEqual(matrix[1, 1], 90.0)
        self.assertAlmostEqual(matrix[0, 2], 119.5)
        self.assertAlmostEqual(matrix[1, 2], 89.5)

    def test_fixed_fps_returns_384_finite_points(self):
        source = np.column_stack(
            (np.linspace(-1.0, 1.0, 73), np.zeros(73), np.ones(73))
        )
        sampled, fraction = fixed_farthest_point_sample(source, 384)
        self.assertEqual(sampled.shape, (384, 3))
        self.assertTrue(np.isfinite(sampled).all())
        self.assertAlmostEqual(fraction, 73.0 / 384.0)
        self.assertAlmostEqual(float(sampled[:, 0].min()), -1.0)
        self.assertAlmostEqual(float(sampled[:, 0].max()), 1.0)

    def test_pointnet2_output_shape(self):
        from gymnasium import spaces

        observation_space = spaces.Dict(
            {
                "points": spaces.Box(-10.0, 10.0, (384, 3), np.float32),
                "proprio": spaces.Box(-10.0, 10.0, (16,), np.float32),
            }
        )
        extractor = PointNet2FeaturesExtractor(observation_space)
        output = extractor(
            {
                "points": torch.randn(2, 384, 3),
                "proprio": torch.randn(2, 16),
            }
        )
        self.assertEqual(tuple(output.shape), (2, 320))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
