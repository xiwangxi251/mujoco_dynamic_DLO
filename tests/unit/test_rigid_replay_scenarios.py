import unittest
from dataclasses import replace

from panda_cable_grasp.env.environment import EnvConfig
from panda_cable_grasp.scenarios import get_scenario, list_scenarios
from panda_cable_grasp.scenarios.registry import SCENARIO_FAMILIES


class ReplayScenarioRegistryTests(unittest.TestCase):
    def test_current_shape_replay_uses_versioned_controller_only(self):
        current=get_scenario('dev_rigid_replay_shape_nominal_x075_v1')
        self.assertEqual(current.replay_controller_version,'continuous_ff_v2')
        self.assertEqual(EnvConfig(**current.to_env_overrides()).replay_controller_version,'continuous_ff_v2')
        legacy=replace(current,replay_controller_version='legacy_pd_v1')
        self.assertNotEqual(current.scenario_id,legacy.scenario_id)
        self.assertEqual(legacy.scenario_id,'scenario-v1-48ead0f573de9a62')
        self.assertNotIn('replay_controller_version',legacy._identity_payload())
        for name in ('id_rigid_replay_shape_nominal','dev_rigid_replay_midshape_v1_x075_v1'):
            self.assertEqual(get_scenario(name).replay_controller_version,'legacy_pd_v1')

    def test_controller_version_and_gain_validation(self):
        with self.assertRaisesRegex(ValueError,'replay_controller_version'):
            EnvConfig(replay_controller_version='typo')
        with self.assertRaisesRegex(ValueError,'replay_controller_version'):
            EnvConfig(replay_controller_version='continuous_ff_v2')
        with self.assertRaises(ValueError):
            EnvConfig(replay_position_gain=float('nan'))
        with self.assertRaises(ValueError):
            EnvConfig(replay_velocity_gain=-1.)
    def test_x075_complex_replay_pairs_snapshot_shape_and_position(self) -> None:
        name = "dev_rigid_replay_midshape_v1_x075_v1"
        self.assertNotIn(name, SCENARIO_FAMILIES["cable_x075_v1"])
        replay = get_scenario(name)
        static = get_scenario("dev_static_midshape_v1_x075_v1")
        self.assertEqual(replay.initial_shape_bank, static.initial_shape_bank)
        self.assertEqual(replay.replay_bank, static.initial_shape_bank)
        self.assertEqual(replay.initial_cable_center_x, static.initial_cable_center_x)
        config = EnvConfig(**replay.to_env_overrides())
        self.assertTrue(config.replay_use_snapshot_position)
        self.assertFalse(get_scenario("id_rigid_replay_midshape_v1").replay_use_snapshot_position)

    def test_x075_seventh_scene_has_complex_shape_and_simple_motion(self) -> None:
        names = SCENARIO_FAMILIES["cable_x075_v1"]
        self.assertEqual(len(names), 7)
        self.assertEqual(names[-1], "dev_rigid_l1_midshape_v1_x075_v1")
        simple = get_scenario("dev_rigid_l1_nominal_x075_v1")
        complex_scene = get_scenario(names[-1])
        static = get_scenario("dev_static_midshape_v1_x075_v1")
        self.assertEqual(complex_scene.initial_shape_bank, static.initial_shape_bank)
        self.assertEqual(complex_scene.motion_profile_version, simple.motion_profile_version)
        self.assertEqual(complex_scene.motion_type, simple.motion_type)
        self.assertEqual(complex_scene.frequency_scale, simple.frequency_scale)
        self.assertEqual(complex_scene.amplitude_level, simple.amplitude_level)
        self.assertEqual(complex_scene.initial_cable_center_x, .75)
        self.assertIsNone(complex_scene.replay_bank)
        EnvConfig(**complex_scene.to_env_overrides())

    def test_replay_scenarios_registered_with_expected_banks(self) -> None:
        shape = get_scenario("id_rigid_replay_shape_nominal")
        combined = get_scenario("id_rigid_replay_combined_nominal")
        self.assertEqual(shape.motion_type.value, "rigid")
        self.assertEqual(combined.motion_type.value, "rigid")
        self.assertEqual(shape.motion_profile_version, "rigid_replay_v1")
        self.assertEqual(combined.motion_profile_version, "rigid_replay_v1")
        self.assertEqual(shape.replay_bank, "replay_src_shape_nominal_v1")
        self.assertEqual(
            combined.replay_bank, "replay_src_combined_l1_v1"
        )
        self.assertEqual(shape.split.value, "id")
        self.assertEqual(combined.split.value, "id")

    def test_env_overrides_carry_replay_bank(self) -> None:
        scenario = get_scenario("id_rigid_replay_shape_nominal")
        overrides = scenario.to_env_overrides()
        self.assertEqual(overrides["replay_bank"], "replay_src_shape_nominal_v1")
        self.assertEqual(overrides["motion_mode"], "rigid")
        self.assertEqual(
            overrides["motion_profile_version"], "rigid_replay_v1"
        )

    def test_replay_profile_requires_bank_and_bank_requires_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "replay_bank"):
            EnvConfig(
                motion_mode="rigid",
                motion_profile_version="rigid_replay_v1",
            )
        with self.assertRaisesRegex(ValueError, "replay_bank"):
            EnvConfig(
                motion_mode="shape",
                motion_profile_version="factorized_v2",
                replay_bank="replay_src_shape_nominal_v1",
            )
        with self.assertRaisesRegex(ValueError, "replay_seed_fallback"):
            EnvConfig(
                motion_mode="rigid",
                motion_profile_version="rigid_replay_v1",
                replay_bank="replay_src_shape_nominal_v1",
                replay_seed_fallback="explode",
            )
        with self.assertRaises(KeyError):
            get_scenario("id_rigid_replay_nominal")

    def test_registry_rejects_replay_misconfiguration(self) -> None:
        scenario = get_scenario("id_shape_nominal_current")
        payload = scenario._identity_payload()
        self.assertNotIn("replay_bank", payload)
        replay = get_scenario("id_rigid_replay_shape_nominal")
        replay_payload = replay._identity_payload()
        self.assertEqual(
            replay_payload["replay_bank"], "replay_src_shape_nominal_v1"
        )

    def test_legacy_scenario_hashes_unchanged_by_replay_field(self) -> None:
        # Adding replay_bank must not perturb existing scenario identities:
        # the field is stripped from the identity payload at its default.
        for name in (
            "id_static",
            "id_shape_nominal_current",
            "id_rigid_l1_nominal",
            "id_combined_l1_nominal",
        ):
            scenario = get_scenario(name)
            self.assertIsNone(scenario.replay_bank)
            self.assertNotIn("replay_bank", scenario._identity_payload())
            self.assertTrue(
                scenario.scenario_id.startswith("scenario-v")
            )


if __name__ == "__main__":
    unittest.main()
