import numpy as np

from panda_cable_grasp.rl.future_geometry import encode, material_samples


def test_remaining_frames_time_coordinates_and_disabled_mask():
    xyz = np.ones((30, 14, 3))
    result = encode(xyz, 1., np.array([1, 0, 0]), np.eye(3), 1., True).reshape(30, 44)
    assert np.count_nonzero(result[:, 43]) == 20
    assert np.array_equal(result[:10], np.zeros((10, 44)))
    assert np.array_equal(result[10, :3], [0, 1, 1])
    assert np.isclose(result[10, 42], .1/3)
    assert not encode(xyz, 3., np.zeros(3), np.eye(3), 1., True).any()
    assert not encode(xyz, 0., np.zeros(3), np.eye(3), 1., False).any()


def test_material_correspondence_is_fixed_after_deformation():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=float)
    segments, alpha = material_samples(xyz, count=4)
    mapped = xyz[segments]*(1-alpha[:, None])+xyz[segments+1]*alpha[:, None]
    assert np.allclose(mapped[:, 0], [0, 1, 2, 3])
    future = xyz.copy()
    future[1] = [1, 2, 0]
    mapped = future[segments]*(1-alpha[:, None])+future[segments+1]*alpha[:, None]
    assert np.allclose(mapped[2], [2, 1, 0])
