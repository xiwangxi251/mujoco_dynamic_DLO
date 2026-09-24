from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_cable_grasp.env.environment import (
    MENAGERIE_ENV_VAR,
    NERO_XML_PATH,
    PANDA_XML_PATH,
    ROBOT_SPECS,
    ROOT,
    XML_PATH,
    robot_spec,
    resolve_menagerie_panda_dir,
)
from panda_cable_grasp.paths import OUTPUT_ROOT_ENV_VAR, output_path


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

    def test_nero_model_sources_are_owned_by_repository(self) -> None:
        spec = robot_spec("NERO")
        self.assertIs(spec, ROBOT_SPECS["nero"])
        self.assertEqual(spec.xml_path, NERO_XML_PATH)
        self.assertEqual(spec.hand_body_name, "link7")
        self.assertEqual(spec.finger_joint_names, ("gripper_joint1", "gripper_joint2"))
        self.assertTrue(NERO_XML_PATH.is_file())
        self.assertTrue(spec.asset_dir.is_dir())
        self.assertTrue((spec.asset_dir / "gripper_link1.stl").is_file())
        self.assertTrue((spec.asset_dir / "gripper_link2.stl").is_file())

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
