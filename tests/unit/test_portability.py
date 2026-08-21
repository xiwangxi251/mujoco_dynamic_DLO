from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cable_grasp_env import (
    MENAGERIE_ENV_VAR,
    PANDA_XML_PATH,
    ROOT,
    XML_PATH,
    resolve_menagerie_panda_dir,
)
from project_paths import OUTPUT_ROOT_ENV_VAR, output_path


class PortabilityTests(unittest.TestCase):
    def test_model_sources_are_owned_by_repository(self) -> None:
        self.assertEqual(
            XML_PATH, ROOT / "assets" / "mujoco" / "panda_cable_grasp.xml"
        )
        self.assertEqual(
            PANDA_XML_PATH, ROOT / "assets" / "mujoco" / "panda.xml"
        )
        self.assertTrue(XML_PATH.is_file())
        self.assertTrue(PANDA_XML_PATH.is_file())

    def test_menagerie_can_be_located_outside_project_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "arbitrary" / "menagerie"
            panda_dir = root / "franka_emika_panda"
            (panda_dir / "assets").mkdir(parents=True)
            with patch.dict(os.environ, {MENAGERIE_ENV_VAR: str(root)}):
                self.assertEqual(resolve_menagerie_panda_dir(), panda_dir.resolve())

    def test_output_root_can_target_server_scratch_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {OUTPUT_ROOT_ENV_VAR: temporary}):
                self.assertEqual(
                    output_path("headless_videos"),
                    Path(temporary) / "headless_videos",
                )


if __name__ == "__main__":
    unittest.main()
