from __future__ import annotations

import unittest

import numpy as np

from panda_cable_grasp.dynamicvla.finetune.convert_dataset import (
    _select_frame_indices,
    _wxyz_to_euler_xyz,
)


class DynamicVLAFinetuneConversionTests(unittest.TestCase):
    def test_downsamples_50_hz_to_25_hz(self) -> None:
        times = np.arange(11, dtype=np.float64) * 0.02
        np.testing.assert_array_equal(
            _select_frame_indices(times, 25), np.array([0, 2, 4, 6, 8, 10])
        )

    def test_euler_convention_wraps_x_and_z_positive(self) -> None:
        quaternion = np.array(
            [RotationQuaternion.w, RotationQuaternion.x,
             RotationQuaternion.y, RotationQuaternion.z],
            dtype=np.float64,
        )[None, :]
        euler = _wxyz_to_euler_xyz(quaternion)[0]
        self.assertGreaterEqual(euler[0], 0.0)
        self.assertGreaterEqual(euler[2], 0.0)
        self.assertLess(euler[0], 2.0 * np.pi)
        self.assertLess(euler[2], 2.0 * np.pi)


class RotationQuaternion:
    # Quaternion for Euler xyz (-0.2, 0.1, -0.3), precomputed in wxyz order.
    w = 0.981856172866081
    x = -0.0911575493429907
    y = 0.0640713477060712
    z = -0.143572175027392


if __name__ == "__main__":
    unittest.main()
