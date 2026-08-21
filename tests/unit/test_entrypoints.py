from __future__ import annotations

from importlib import import_module
import unittest

from _bootstrap import bootstrap


class EntrypointImportTests(unittest.TestCase):
    def test_all_command_modules_import(self) -> None:
        bootstrap()
        modules = (
            "panda_cable_grasp.cli.run_grasp",
            "panda_cable_grasp.cli.run_dynamicvla",
            "panda_cable_grasp.evaluation.benchmark",
            "panda_cable_grasp.evaluation.motion_diagnostics",
            "panda_cable_grasp.rl.train",
            "panda_cable_grasp.rl.evaluate",
            "panda_cable_grasp.expert.collect_dataset",
            "panda_cable_grasp.expert.run_experiment",
            "panda_cable_grasp.dynamicvla.finetune.convert_dataset",
            "panda_cable_grasp.dynamicvla.finetune.launch_finetune",
        )
        for module_name in modules:
            with self.subTest(module=module_name):
                module = import_module(module_name)
                self.assertTrue(callable(getattr(module, "main", None)))


if __name__ == "__main__":
    unittest.main()
