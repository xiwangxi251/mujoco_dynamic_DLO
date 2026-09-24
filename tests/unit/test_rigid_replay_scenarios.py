import unittest

from panda_cable_grasp.env.environment import EnvConfig
from panda_cable_grasp.scenarios import get_scenario, list_scenarios


class ReplayScenarioRegistryTests(unittest.TestCase):
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
