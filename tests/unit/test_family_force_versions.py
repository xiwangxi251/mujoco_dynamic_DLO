from dataclasses import replace
import pytest
from panda_cable_grasp.env.environment import EnvConfig
from panda_cable_grasp.scenarios import get_scenario, SCENARIO_FAMILIES


def test_only_three_new_family_l1_scenes_enable_force_tape():
    names=("dev_rigid_l1_nominal_x075_force_v1","dev_rigid_l1_midshape_v1_x075_force_v1","dev_combined_l1_nominal_x075_force_v1")
    for name in names:
        scene=get_scenario(name)
        assert scene.rigid_drive_version=="force_tape_v1"
        assert EnvConfig(**scene.to_env_overrides()).rigid_drive_version=="force_tape_v1"
        old=replace(scene,rigid_drive_version="feedback_v1")
        assert "rigid_drive_version" not in old._identity_payload()
        assert old.scenario_id!=scene.scenario_id
    for name in ("id_rigid_l1_nominal","id_combined_l1_nominal","id_rigid_l1_midshape_v1",
                 "dev_static_x075_v1","dev_static_midshape_v1_x075_v1","dev_shape_nominal_current_x075_v1"):
        assert get_scenario(name).rigid_drive_version=="feedback_v1"


def test_old_family_preserves_pre_correction_checkpoint_scene_ids():
    expected = (
        'scenario-v1-8dc92f2a5015c979', 'scenario-v1-9b4356d5cd075ee9',
        'scenario-v1-a9e435bd5bf91005', 'scenario-v1-90dbf3eca85bbc3e',
        'scenario-v1-3cce0c0233111be5', 'scenario-v1-48ead0f573de9a62',
        'scenario-v1-027958373c34dea0',
    )
    assert tuple(get_scenario(n).scenario_id for n in SCENARIO_FAMILIES['cable_x075_v1']) == expected
    for name in SCENARIO_FAMILIES['cable_x075_v1']:
        scene = get_scenario(name)
        assert scene.rigid_drive_version == 'feedback_v1'
        assert scene.replay_controller_version == 'legacy_pd_v1'


def test_force_family_keeps_physics_and_order_but_has_separate_identity():
    old_names = SCENARIO_FAMILIES['cable_x075_v1']
    new_names = SCENARIO_FAMILIES['cable_x075_force_v1']
    assert len(new_names) == 7 and not set(old_names) & set(new_names)
    for old_name, new_name in zip(old_names, new_names):
        old, new = get_scenario(old_name), get_scenario(new_name)
        assert new_name == old_name.removesuffix('_x075_v1') + '_x075_force_v1'
        assert old.scenario_id != new.scenario_id
        old_config, new_config = old.to_env_overrides(), new.to_env_overrides()
        for key in ('scenario_name', 'scenario_id', 'rigid_drive_version', 'replay_controller_version'):
            old_config.pop(key, None)
            new_config.pop(key, None)
        assert old_config == new_config
        assert new.arm_motion_start_delay == .82
    assert get_scenario('dev_rigid_replay_shape_nominal_x075_force_v1').replay_controller_version == 'force_tape_v5'


def test_force_tape_rejects_nonrigid_and_other_object_drivers():
    with pytest.raises(ValueError,match="rigid_drive_version"):
        EnvConfig(rigid_drive_version="force_tape_v1")
    with pytest.raises(ValueError,match="rigid_drive_version"):
        EnvConfig(rigid_drive_version="typo")
