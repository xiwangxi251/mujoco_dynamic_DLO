import json
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from panda_cable_grasp.env.replay_bank import (
    BANK_FORMAT_VERSION,
    ReplayBank,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def write_bank(
    directory: Path,
    positions: np.ndarray,
    seeds: list[int],
    *,
    control_dt: float = 0.02,
    valid_steps: np.ndarray | None = None,
    version: int = BANK_FORMAT_VERSION,
) -> Path:
    """Write a synthetic replay bank with the production file layout."""
    positions = np.asarray(positions, dtype=np.float32)
    entries, horizon, node_count, _ = positions.shape
    cable_ids = np.arange(100, 100 + node_count, dtype=np.int64)
    meta = {
        "format": "replay_bank",
        "format_version": version,
        "control_dt": control_dt,
        "physics_timestep": 0.002,
        "frame_skip": 10,
        "episode_seconds": (horizon - 1) * control_dt,
        "horizon_steps": horizon,
        "node_count": node_count,
        "source_scenario": "synthetic_source",
        "source_scenario_id": "scenario-v1-synthetic",
        "source_scenario_hash": "0" * 64,
        "source_motion_mode": "shape",
        "source_start_y_placement": False,
    }
    path = directory / "synthetic.npz"
    np.savez(
        path,
        format_version=np.asarray(version, dtype=np.int32),
        meta_json=np.asarray(json.dumps(meta)),
        seeds=np.asarray(seeds, dtype=np.int64),
        positions_xy=positions,
        valid_steps=(
            np.asarray(valid_steps, dtype=np.int64)
            if valid_steps is not None
            else np.full(entries, horizon, dtype=np.int64)
        ),
        placed_com_xy=np.zeros((entries, 2), dtype=np.float64),
        target_index=np.full(entries, node_count // 2, dtype=np.int64),
        cable_ids=cable_ids,
    )
    return path


class ReplayBankTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_trajectory_returns_delta_velocity_and_yaw(self) -> None:
        horizon, node_count = 41, 5
        omega = 0.7
        control_dt = 0.02
        times = np.arange(horizon) * control_dt
        # 非对称排布的刚性节点组绕原点以 ``omega`` rad/s 旋转（对称点云
        # 会让整体旋转不可辨识，必须用非对称形状），位移/速度/全局 yaw
        # 都有解析解。
        offsets = np.array([-0.02, -0.008, 0.004, 0.011, 0.02])
        bumps = np.array([0.0, 0.006, -0.003, 0.005, 0.0])
        positions = np.empty((1, horizon, node_count, 2))
        for index in range(node_count):
            c, s = np.cos(omega * times), np.sin(omega * times)
            positions[0, :, index, 0] = offsets[index] * c - bumps[index] * s
            positions[0, :, index, 1] = offsets[index] * s + bumps[index] * c
        path = write_bank(
            self.directory, positions, [7], control_dt=control_dt
        )
        bank = ReplayBank(path)
        trajectory = bank.trajectory(0, 2)

        expected = positions[0, :, 2] - positions[0, 0, 2]
        np.testing.assert_allclose(
            trajectory.delta_position, expected, atol=1e-6
        )
        analytic_velocity = np.column_stack(
            (
                -offsets[2] * omega * np.sin(omega * times)
                - bumps[2] * omega * np.cos(omega * times),
                offsets[2] * omega * np.cos(omega * times)
                - bumps[2] * omega * np.sin(omega * times),
            )
        )
        np.testing.assert_allclose(
            trajectory.velocity[2:-2], analytic_velocity[2:-2], atol=1e-3
        )
        np.testing.assert_allclose(
            trajectory.delta_yaw, omega * times, atol=1e-6
        )
        np.testing.assert_allclose(
            trajectory.yaw_rate[2:-2],
            np.full(horizon - 4, omega),
            atol=1e-3,
        )

    def test_frozen_tail_yields_zero_reference_motion(self) -> None:
        horizon, node_count = 31, 4
        valid = 12
        positions = np.zeros((1, horizon, node_count, 2))
        ramp = np.linspace(0.0, 0.1, valid)
        positions[0, :valid, :, 0] = ramp[:, None]
        positions[0, valid:, :, :] = positions[0, valid - 1]
        path = write_bank(
            self.directory, positions, [3], valid_steps=[valid]
        )
        trajectory = ReplayBank(path).trajectory(0, 1)
        np.testing.assert_allclose(
            trajectory.velocity[valid + 1:], np.zeros((horizon - valid - 1, 2)),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            trajectory.yaw_rate[valid + 1:],
            np.zeros(horizon - valid - 1),
            atol=1e-7,
        )

    def test_seed_lookup_and_compatibility(self) -> None:
        positions = np.zeros((2, 5, 3, 2), dtype=np.float32)
        path = write_bank(self.directory, positions, [11, 22])
        bank = ReplayBank(path)
        self.assertEqual(bank.index_for_seed(11), 0)
        self.assertEqual(bank.index_for_seed(22), 1)
        self.assertIsNone(bank.index_for_seed(33))
        self.assertIsNone(bank.index_for_seed(None))
        bank.check_compatible(np.arange(100, 103), 0.02)
        with self.assertRaises(ValueError):
            bank.check_compatible(np.arange(200, 203), 0.02)
        with self.assertRaises(ValueError):
            bank.check_compatible(np.arange(100, 103), 0.05)

    def test_version_mismatch_is_rejected(self) -> None:
        path = write_bank(
            self.directory,
            np.zeros((1, 5, 3, 2), dtype=np.float32),
            [1],
            version=BANK_FORMAT_VERSION + 99,
        )
        with self.assertRaises(ValueError):
            ReplayBank(path)


GRIPPER_MESHES = {
    "gripper_flange",
    "gripper_base",
    "gripper_link1",
    "gripper_link2",
}


class NeroGripperRenderGroupTests(unittest.TestCase):
    """XML invariants the runtime geom-group toggle relies on.

    Gripper hiding is done at render time by flipping ``model.geom_group``
    on the gripper-mesh geoms, so the XML keeps its original groups and the
    lookup must find exactly the coincident geom pairs below.
    """

    def test_each_gripper_mesh_has_coincident_geom_pair(self) -> None:
        xml_path = (
            REPO_ROOT / "assets" / "mujoco" / "nero" / "nero.xml"
        )
        root = ET.parse(xml_path).getroot()
        counts = {name: 0 for name in GRIPPER_MESHES}
        for geom in root.iter("geom"):
            mesh = geom.get("mesh")
            if mesh in counts:
                counts[mesh] += 1
        self.assertEqual(
            counts,
            {name: 2 for name in GRIPPER_MESHES},
            f"gripper geom count changed: {counts}",
        )

    def test_no_geom_uses_render_group_4(self) -> None:
        xml_path = (
            REPO_ROOT / "assets" / "mujoco" / "nero" / "nero.xml"
        )
        root = ET.parse(xml_path).getroot()
        offenders = [
            geom.get("mesh")
            for geom in root.iter("geom")
            if geom.get("group") == "4"
        ]
        self.assertEqual(
            offenders,
            [],
            f"group 4 is reserved for the runtime gripper toggle: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
