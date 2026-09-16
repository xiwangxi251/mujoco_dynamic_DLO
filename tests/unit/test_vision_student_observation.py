import unittest

import numpy as np

from panda_cable_grasp.vision_student.observation import (
    build_student_observation,
    map_visibility_to_samples,
    resample_polyline_arclength,
    stabilize_polyline_direction,
)


class VisionStudentObservationTests(unittest.TestCase):
    def test_arclength_resampling_includes_endpoints(self):
        points = np.stack((np.arange(45), np.zeros(45), np.zeros(45)), axis=1)
        sampled = resample_polyline_arclength(points)
        np.testing.assert_allclose(sampled[0], points[0])
        np.testing.assert_allclose(sampled[-1], points[-1])
        self.assertEqual(sampled.shape, (14, 3))

    def test_visibility_mapping_keeps_hidden_points_explicit(self):
        mask = map_visibility_to_samples(np.asarray([0, 1, 44]))
        self.assertEqual(mask.shape, (14,))
        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[-1], 1.0)
        self.assertLess(mask[7], 1.0)

    def test_direction_uses_previous_frame(self):
        current = np.arange(15, dtype=np.float32).reshape(5, 3)
        previous = current[::-1].copy()
        aligned = stabilize_polyline_direction(current, previous)
        np.testing.assert_allclose(aligned, previous)

    def test_student_observation_is_127d_and_scales_only_dlo_positions(self):
        points = np.ones((14, 3), dtype=np.float32) * 0.5
        observation = build_student_observation(
            points,
            points * 0.5,
            np.ones(14),
            np.zeros(14),
            np.arange(7),
            np.arange(7) + 10,
            0.25,
        )
        self.assertEqual(observation.shape, (127,))
        np.testing.assert_allclose(observation[:42], 1.0)
        np.testing.assert_allclose(observation[42:84], 0.5)
        np.testing.assert_allclose(observation[84:98], 1.0)
        np.testing.assert_allclose(observation[98:112], 0.0)
        np.testing.assert_allclose(observation[-1], 0.25)


if __name__ == "__main__":
    unittest.main()
