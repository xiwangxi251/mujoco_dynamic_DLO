from dataclasses import replace
import numpy as np
import pytest

from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.scenarios.registry import SCENARIO_FAMILIES, get_scenario


def test_only_simple_whole_motion_gets_a_new_identity():
    old = SCENARIO_FAMILIES['cable_x075_sector_speed_v2']
    new = SCENARIO_FAMILIES['cable_x075_sector_speed_v3']
    assert len(new) == 7
    assert [i for i, pair in enumerate(zip(old, new)) if pair[0] != pair[1]] == [1]
    original, changed = get_scenario(old[1]), get_scenario(new[1])
    overrides = changed.to_env_overrides()
    overrides['initial_placement_version'] = original.initial_placement_version
    assert overrides == original.to_env_overrides()
    assert changed.scenario_id != original.scenario_id
    for i in (0, 2, 3, 4, 5, 6):
        assert get_scenario(old[i]) is get_scenario(new[i])


@pytest.mark.parametrize('changes', [
    {'motion_type': 'shape'},
    {'initial_shape_bank': 'replay_src_shape_nominal_v1'},
    {'motion_profile_version': 'rigid_level2_single_pass_v2'},
])
def test_yaw_protocol_cannot_be_applied_to_other_scene_types(changes):
    from panda_cable_grasp.scenarios.registry import MotionType
    scene = get_scenario(SCENARIO_FAMILIES['cable_x075_sector_speed_v3'][1])
    if 'motion_type' in changes:
        changes['motion_type'] = MotionType.SHAPE
    with pytest.raises(ValueError, match='simple-initial L1'):
        replace(scene, **changes)


def test_real_reset_preserves_curve_speed_and_rng_with_new_yaw():
    old = get_scenario(SCENARIO_FAMILIES['cable_x075_sector_speed_v2'][1])
    new = get_scenario(SCENARIO_FAMILIES['cable_x075_sector_speed_v3'][1])
    baseline = CableGraspEnv(EnvConfig(robot='nero', **old.to_env_overrides()))
    candidate = CableGraspEnv(EnvConfig(robot='nero', **new.to_env_overrides()))
    try:
        for seed in (20260804, 20260812, 20260853):
            baseline.reset(seed=seed)
            candidate.reset(seed=seed)
            expected_yaw = np.random.default_rng(seed ^ 0xA61E).uniform(-180., 180.)
            assert candidate.info()['initial_rigid_yaw_deg'] == expected_yaw
            angle = np.radians(expected_yaw)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            a = baseline.data.xpos[baseline.cable_ids].copy()
            b = candidate.data.xpos[candidate.cable_ids].copy()
            ca = np.average(a[:, :2], axis=0, weights=baseline.cable_mass)
            cb = np.average(b[:, :2], axis=0, weights=candidate.cable_mass)
            np.testing.assert_allclose(ca, cb, atol=1e-12)
            np.testing.assert_allclose(b[:, :2] - cb, (a[:, :2] - ca) @ rotation.T, atol=1e-12)
            np.testing.assert_array_equal(a[:, 2], b[:, 2])
            assert baseline.config.rigid_motion_nominal_speed == candidate.config.rigid_motion_nominal_speed
            assert baseline._rigid_motion_rotation_sign == candidate._rigid_motion_rotation_sign
            assert baseline.rng.bit_generator.state == candidate.rng.bit_generator.state
            assert baseline.motion_profile_hash != candidate.motion_profile_hash
        candidate.reset(seed=20260804, randomize=False)
        assert candidate.info()['initial_rigid_yaw_deg'] == 0.
    finally:
        baseline.close()
        candidate.close()


def test_ordinary_curriculum_selects_the_mixed_identity_family(monkeypatch):
    import sys
    from panda_cable_grasp.rl.train import parse_args
    monkeypatch.setattr(sys, 'argv', ['train', '--ordinary-family-curriculum',
        '--workspace-scene-family', 'cable_x075_sector_speed_v3'])
    args = parse_args()
    stages = {name for stage in args.curriculum_stages for name in stage}
    assert stages == set(SCENARIO_FAMILIES['cable_x075_sector_speed_v3'])
    assert 'dev_rigid_l1_nominal_x075_sector_speed_v2' not in stages
