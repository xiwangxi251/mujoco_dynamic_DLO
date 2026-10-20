import numpy as np
import pytest
from types import SimpleNamespace

from panda_cable_grasp.rl.future_geometry import encode, material_samples
from panda_cable_grasp.rl.future_geometry import resolve_future_mode, PROTOCOL


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


def test_future_is_opt_in_and_saved_protocol_cannot_be_silently_replaced():
    assert not resolve_future_mode()
    assert resolve_future_mode(True)
    saved = dict(future_geometry=PROTOCOL)
    assert resolve_future_mode(None, saved)
    with pytest.raises(ValueError, match='requires --future'):
        resolve_future_mode(False, saved)
    with pytest.raises(ValueError, match='Current-only checkpoint'):
        resolve_future_mode(True, dict(observation_dimension=99))
    with pytest.raises(ValueError, match='Unsupported'):
        resolve_future_mode(None, dict(future_geometry=dict(version='other')))
    with pytest.raises(ValueError, match='frame times'):
        resolve_future_mode(None, dict(future_geometry=dict(PROTOCOL, frame_times_s=[.1])))


def test_standalone_checkpoint_metadata_and_dimension_guards():
    model = SimpleNamespace(observation_space=SimpleNamespace(shape=(1419,)),
                            future_geometry_protocol=PROTOCOL)
    assert resolve_future_mode(None, model=model)
    model.observation_space.shape = (99,)
    with pytest.raises(ValueError, match='1419-D'):
        resolve_future_mode(None, model=model)
    model.future_geometry_protocol = None
    model.observation_space.shape = (1419,)
    with pytest.raises(ValueError, match='lacks an enabled'):
        resolve_future_mode(None, model=model)
    model.future_geometry_protocol = dict(PROTOCOL, version='other')
    with pytest.raises(ValueError, match='Unsupported'):
        resolve_future_mode(None, dict(future_geometry=PROTOCOL), model=model)


def test_contact_detection_includes_robot_links_and_excludes_table(monkeypatch):
    from panda_cable_grasp.rl import future_geometry as module
    model = SimpleNamespace(nbody=4, body_parentid=np.array([0,0,1,0]),
                            geom_bodyid=np.array([0,1,2,3]))
    env = SimpleNamespace(model=model, robot_spec=SimpleNamespace(base_body_name='robot'),
                          cable_set={3}, data=SimpleNamespace(contact=[]))
    monkeypatch.setattr(module.mujoco, 'mj_name2id', lambda *a:1)
    env.data.contact=[SimpleNamespace(geom1=2,geom2=3)]
    assert module.robot_cable_contact(env)
    env.data.contact=[SimpleNamespace(geom1=3,geom2=1)]
    assert module.robot_cable_contact(env)
    env.data.contact=[SimpleNamespace(geom1=0,geom2=3)]
    assert not module.robot_cable_contact(env)


@pytest.mark.parametrize('extra', [[], ['--future-geometry'],
    ['--future-geometry', '--target-node-task'],
    ['--future-geometry', '--observation-mode', 'pointcloud'],
    ['--future-geometry', '--action-mode', 'joint_target']])
def test_training_cli_defaults_and_incompatible_interfaces(monkeypatch, extra):
    from panda_cable_grasp.rl.train import parse_args
    monkeypatch.setattr('sys.argv', ['train', *extra])
    if len(extra) > 1:
        with pytest.raises(SystemExit):
            parse_args()
    else:
        assert parse_args().future_geometry == bool(extra)
