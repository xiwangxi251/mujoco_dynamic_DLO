from __future__ import annotations

import unittest

import numpy as np

from panda_cable_grasp.rl.environment import RLCableGraspEnv
from panda_cable_grasp.rl.evaluate import resolve_scenario_names
from panda_cable_grasp.rl.train import RL_L1_SCENARIOS


class RLTestScenarioSelectionTests(unittest.TestCase):
    def test_l1_distribution_matches_training_evaluation(self) -> None:
        self.assertEqual(
            resolve_scenario_names(None, "l1"),
            tuple(RL_L1_SCENARIOS),
        )

    def test_single_scenario_overrides_distribution(self) -> None:
        self.assertEqual(
            resolve_scenario_names("id_static", "l1"),
            ("id_static",),
        )

    def test_legacy_requires_explicit_selection(self) -> None:
        self.assertIsNone(resolve_scenario_names(None, "legacy"))

    def test_registered_static_scenario_uses_random_curved_initial_shape(self) -> None:
        env = RLCableGraspEnv(seed=17, scenario_names=("id_static",))
        try:
            _, info = env.reset(seed=17)
            cable_xy = env.data.xpos[env.base_env.cable_ids, :2]
            singular_values = np.linalg.svd(
                cable_xy - cable_xy.mean(axis=0), compute_uv=False
            )
            self.assertEqual(info["scenario_name"], "id_static")
            self.assertEqual(info["motion_profile_version"], "factorized_v2")
            self.assertGreater(float(singular_values[1]), 0.01)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
