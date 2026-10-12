from dataclasses import replace
import pytest
from panda_cable_grasp.env.environment import EnvConfig
from panda_cable_grasp.scenarios import get_scenario


def test_only_three_current_l1_scenes_enable_force_tape():
    names=("dev_rigid_l1_nominal_x075_v1","dev_rigid_l1_midshape_v1_x075_v1","dev_combined_l1_nominal_x075_v1")
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


def test_force_tape_rejects_nonrigid_and_other_object_drivers():
    with pytest.raises(ValueError,match="rigid_drive_version"):
        EnvConfig(rigid_drive_version="force_tape_v1")
    with pytest.raises(ValueError,match="rigid_drive_version"):
        EnvConfig(rigid_drive_version="typo")
