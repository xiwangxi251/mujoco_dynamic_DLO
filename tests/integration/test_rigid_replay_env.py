"""Server-side integration tests for ``rigid_replay_v1`` and render modes.

These tests need MuJoCo and run on the simulation host.  A small synthetic
replay bank is generated in a temporary directory so no pre-recorded bank is
required; the analytical trajectory (rigid rotation + circular COM drift) is
fully known, which makes tracking assertions exact rather than empirical.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from panda_cable_grasp.env.environment import (
    CableGraspEnv,
    EnvConfig,
    GraspState,
)
from panda_cable_grasp.env.replay_bank import BANK_FORMAT_VERSION
from panda_cable_grasp.evaluation.motion_diagnostics import (
    env_config_for_scenario,
)
from panda_cable_grasp.perception.rendering import (
    GRIPPER_RENDER_GROUP,
    GripperVisibilityToggle,
)
from panda_cable_grasp.scenarios import get_scenario


SEED = 4242
BANK_NAME = "synthetic_test_bank"
BANK_FAST = "synthetic_test_bank_fast"
SEED_FAST = 7777
OMEGA_YAW = 0.6
# 快旋库：1.2 rad/s x 4s ≈ 4.8 rad 累积转角，必然越过 +/-pi 包裹边界。
OMEGA_YAW_FAST = 1.2
OMEGA_COM = 1.8
COM_AMPLITUDE = 0.06


def _write_synthetic_bank(
    directory: Path,
    *,
    cable_ids: np.ndarray,
    node_xy0: np.ndarray,
    placed_com: np.ndarray,
    target_index: int,
    control_dt: float,
    horizon: int,
    seed: int,
    bank_name: str = BANK_NAME,
    omega_yaw: float = OMEGA_YAW,
) -> Path:
    node_count = len(cable_ids)
    times = np.arange(horizon) * control_dt
    rel = node_xy0 - placed_com
    positions = np.empty((1, horizon, node_count, 2), dtype=np.float32)
    for row, t in enumerate(times):
        com_delta = np.array([
            COM_AMPLITUDE * math.sin(OMEGA_COM * t),
            COM_AMPLITUDE * (1.0 - math.cos(OMEGA_COM * t)),
        ])
        angle = omega_yaw * t
        rotation = np.array([
            [math.cos(angle), math.sin(angle)],
            [-math.sin(angle), math.cos(angle)],
        ])
        positions[0, row] = (
            placed_com + com_delta + rel @ rotation
        ).astype(np.float32)
    meta = {
        "format": "replay_bank",
        "format_version": BANK_FORMAT_VERSION,
        "control_dt": control_dt,
        "physics_timestep": control_dt / 10.0,
        "frame_skip": 10,
        "horizon_steps": horizon,
        "node_count": node_count,
        "source_scenario": "id_shape_nominal_current",
        "source_scenario_id": "synthetic",
        "source_scenario_hash": "0" * 64,
        "source_motion_mode": "shape",
        "source_start_y_placement": False,
    }
    path = directory / f"{bank_name}.npz"
    np.savez(
        path,
        format_version=np.asarray(BANK_FORMAT_VERSION, dtype=np.int32),
        meta_json=np.asarray(json.dumps(meta)),
        seeds=np.asarray([seed], dtype=np.int64),
        positions_xy=positions,
        valid_steps=np.asarray([horizon], dtype=np.int64),
        placed_com_xy=placed_com[None, :].astype(np.float64),
        target_index=np.asarray([target_index], dtype=np.int64),
        cable_ids=np.asarray(cable_ids, dtype=np.int64),
    )
    return path


class ReplayPairingTests(unittest.TestCase):
    """Synthetic-bank pairing/tracking against the real shape scenario."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.bank_dir = Path(cls._tmp.name)
        cls.scenario = get_scenario("id_shape_nominal_current")
        cls.shape_config = env_config_for_scenario(
            cls.scenario, seed=0, episode_seconds=4.0
        )
        probe = CableGraspEnv(cls.shape_config)
        probe.reset(randomize=True, seed=SEED)
        cls.probe = probe
        cls.cable_ids = np.asarray(probe.cable_ids, dtype=np.int64)
        cls.node_xy0 = probe.data.xpos[cls.cable_ids, :2].copy()
        cls.placed_com = np.average(
            cls.node_xy0, axis=0, weights=probe.cable_mass
        )
        cls.target_index = int(probe.cable_index[probe.target_body_id])
        cls.control_dt = float(
            probe.model.opt.timestep * max(1, probe.config.frame_skip)
        )
        _write_synthetic_bank(
            cls.bank_dir,
            cable_ids=cls.cable_ids,
            node_xy0=cls.node_xy0,
            placed_com=cls.placed_com,
            target_index=cls.target_index,
            control_dt=cls.control_dt,
            horizon=int(4.0 / cls.control_dt) + 1,
            seed=SEED,
        )
        _write_synthetic_bank(
            cls.bank_dir,
            cable_ids=cls.cable_ids,
            node_xy0=cls.node_xy0,
            placed_com=cls.placed_com,
            target_index=cls.target_index,
            control_dt=cls.control_dt,
            horizon=int(4.0 / cls.control_dt) + 1,
            seed=SEED_FAST,
            bank_name=BANK_FAST,
            omega_yaw=OMEGA_YAW_FAST,
        )
        overrides = dict(cls.scenario.to_env_overrides())
        overrides.update(
            motion_mode="rigid",
            motion_profile_version="rigid_replay_v1",
            replay_bank=BANK_NAME,
            replay_bank_dir=str(cls.bank_dir),
            replay_seed_fallback="error",
        )
        cls.replay_config = EnvConfig(
            robot=cls.shape_config.robot,
            seed=0,
            episode_seconds=4.0,
            scenario_name="synthetic_replay_test",
            scenario_id="synthetic-replay",
            scenario_split="id",
            **overrides,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.probe.close()
        cls._tmp.cleanup()

    def make_replay_env(self) -> CableGraspEnv:
        return CableGraspEnv(self.replay_config)

    def test_paired_seed_produces_identical_initial_state(self) -> None:
        env = self.make_replay_env()
        try:
            env.reset(randomize=True, seed=SEED)
            self.assertEqual(
                int(env.cable_index[env.target_body_id]), self.target_index
            )
            np.testing.assert_allclose(
                env.data.xpos[self.cable_ids, :2],
                self.node_xy0,
                atol=1e-6,
            )
            self.assertEqual(env._replay_entry_seed, SEED)
            self.assertEqual(env._replay_tracked_index, self.target_index)
            info = env.info()
            self.assertEqual(info["replay_bank"], BANK_NAME)
            self.assertEqual(info["replay_entry_seed"], SEED)
            self.assertEqual(info["replay_tracked_index"], self.target_index)
        finally:
            env.close()

    def test_missing_seed_raises_under_error_fallback(self) -> None:
        env = self.make_replay_env()
        try:
            with self.assertRaises(KeyError):
                env.reset(randomize=True, seed=SEED + 1)
        finally:
            env.close()

    def test_replay_tracks_recorded_node_trajectory(self) -> None:
        env = self.make_replay_env()
        try:
            env.reset(randomize=True, seed=SEED)
            # 回放回合线缆被 weld 成一根刚体。
            self.assertGreater(len(env._replay_weld_eq_ids), 0)
            self.assertTrue(np.all(
                env.data.eq_active[env._replay_weld_eq_ids] == 1.0
            ))
            tracked = self.target_index
            # 内部形变的位姿不变量度量：reset 后各节点两两距离。刚体化
            # 成立时该距离矩阵冻结；伺服跟踪滞后只影响 com/node 误差。
            xyz0 = env.data.xpos[self.cable_ids, :].copy()
            pair0 = np.linalg.norm(
                xyz0[:, None, :] - xyz0[None, :, :], axis=-1
            )
            max_com_error = 0.0
            max_node_error = 0.0
            max_shape_error = 0.0
            for _ in range(int(2.0 / self.control_dt)):
                env.step(env.ready_ctrl)
                t = float(env.data.time)
                index = min(int(env._replay_elapsed / self.control_dt), 200)
                times_row = index * self.control_dt
                com_delta = np.array([
                    COM_AMPLITUDE * math.sin(OMEGA_COM * times_row),
                    COM_AMPLITUDE * (1.0 - math.cos(OMEGA_COM * times_row)),
                ])
                angle = OMEGA_YAW * times_row
                rotation = np.array([
                    [math.cos(angle), math.sin(angle)],
                    [-math.sin(angle), math.cos(angle)],
                ])
                current_xy = env.data.xpos[self.cable_ids, :2]
                current_com = np.average(
                    current_xy, axis=0, weights=env.cable_mass
                )
                # The replay controller subtracts the tracked node's rotation
                # offset from the COM reference so the tracked node itself
                # lands exactly on its recorded trajectory; the COM therefore
                # follows only the recorded COM drift.
                expected_com = self.placed_com + com_delta
                expected_node = self.placed_com + com_delta + (
                    (self.node_xy0[tracked] - self.placed_com) @ rotation
                )
                max_com_error = max(
                    max_com_error,
                    float(np.linalg.norm(current_com - expected_com)),
                )
                max_node_error = max(
                    max_node_error,
                    float(np.linalg.norm(
                        current_xy[tracked] - expected_node
                    )),
                )
                pair_t = np.linalg.norm(
                    env.data.xpos[self.cable_ids, None, :]
                    - env.data.xpos[None, self.cable_ids, :],
                    axis=-1,
                )
                max_shape_error = max(
                    max_shape_error,
                    float(np.abs(pair_t - pair0).max()),
                )
            print(
                f"tracking: com={max_com_error:.4f} "
                f"node={max_node_error:.4f} shape={max_shape_error:.4f}"
            )
            self.assertLess(max_com_error, 0.05)
            self.assertLess(max_node_error, 0.05)
            # 刚化 weld 激活后形变残差只剩求解器容差量级；任何松动
            # （weld 丢失/relpose 错误）都会把该值推高几个数量级。
            self.assertLess(max_shape_error, 0.01)
        finally:
            env.close()

    def test_grasp_pauses_replay_and_release_reanchors(self) -> None:
        """确认抓取冻结回放时钟；松开后参考重锚到当前位姿，无追赶段。"""
        env = self.make_replay_env()
        try:
            env.reset(randomize=True, seed=SEED)
            tracked_id = int(env.cable_ids[self.target_index])
            for _ in range(20):
                env.step(env.ready_ctrl)
            self.assertGreater(float(env._replay_elapsed), 0.0)

            # 伪造一个已确认的双侧抓取——悬挂路径只读 grasp_confirmed，
            # 不需要真实接触；屏蔽物理抓取状态更新避免它把桩清掉。
            env._update_physical_grasp_state = (
                lambda gripper_closed: None
            )
            env.grasp_state = GraspState(
                body_id=tracked_id,
                candidate_time=float(env.data.time),
                bilateral_confirmed=True,
                last_bilateral_time=float(env.data.time),
                lost_contact_time=float(env.data.time),
            )

            # 首个子步的物理积分发生在确认之前，允许一个子步的固有滞后；
            # 悬挂生效后才冻结时钟起点。
            env.grasp_state.last_bilateral_time = float(env.data.time)
            env.step(env.ready_ctrl)
            self.assertTrue(env.rigid_motion_suspended)
            held_elapsed = float(env._replay_elapsed)
            for _ in range(15):
                env.grasp_state.last_bilateral_time = float(env.data.time)
                env.step(env.ready_ctrl)
                self.assertTrue(env.rigid_motion_suspended)
            # 按住期间回放时钟一步都不能走。
            self.assertEqual(float(env._replay_elapsed), held_elapsed)

            # 模拟夹爪把线缆拖离暂停位姿（刚体：平移根 free joint）。
            env.data.qpos[env.cable_free_qadr] += 0.04
            env.data.qpos[env.cable_free_qadr + 1] += 0.03
            env.grasp_state.last_bilateral_time = float(env.data.time)
            env.step(env.ready_ctrl)
            dragged_xy = env.data.xpos[tracked_id, :2].copy()
            paused_index = env._replay_index()

            # 松开：恢复 _update_physical_grasp_state 并清抓取状态。
            del env._update_physical_grasp_state
            env.grasp_state = None
            pre_xy = env.data.xpos[tracked_id, :2].copy()
            env.step(env.ready_ctrl)
            post_xy = env.data.xpos[tracked_id, :2].copy()

            self.assertFalse(env.rigid_motion_suspended)
            # 时钟从暂停值续走，不能一次性补齐悬挂期间的所有帧。
            self.assertGreater(float(env._replay_elapsed), held_elapsed)
            self.assertLess(
                float(env._replay_elapsed) - held_elapsed,
                2.0 * self.control_dt,
            )
            # 参考必须重锚到被拖到的实际位姿。
            self.assertEqual(env._replay_anchor_index, paused_index)
            np.testing.assert_allclose(
                env._replay_anchor_tracked, dragged_xy, atol=2e-3
            )
            # 无追赶段：恢复首帧位移不得超过录制速度的正常步进量级。
            self.assertLess(np.linalg.norm(post_xy - pre_xy), 0.02)

            for _ in range(10):
                env.step(env.ready_ctrl)
            progressed = float(env._replay_elapsed)
            self.assertGreater(progressed, held_elapsed)
            self.assertLess(progressed - held_elapsed, 0.5)
        finally:
            env.close()

    def test_replay_yaw_wrap_does_not_spin(self) -> None:
        """累积 yaw 越过 +/-pi 时伺服不得命令整圈绕行。

        回归：desired_yaw 是累积角而 current_yaw 包裹在 (-pi, pi]，
        未包裹的误差在边界处跳变 ~2pi，曾造成 15 rad/s 的自旋爆发。
        该库 4s 累计转 4.8 rad，必然过界。
        """
        import dataclasses
        config = dataclasses.replace(
            self.replay_config, replay_bank=BANK_FAST
        )
        env = CableGraspEnv(config)
        try:
            env.reset(randomize=True, seed=SEED_FAST)
            prev_xy = None
            max_speed = 0.0
            for _ in range(int(3.5 / self.control_dt)):
                env.step(env.ready_ctrl)
                xy = env.data.xpos[self.cable_ids, :2].copy()
                if prev_xy is not None:
                    speed = float(np.linalg.norm(
                        xy - prev_xy, axis=1
                    ).max() / self.control_dt)
                    max_speed = max(max_speed, speed)
                prev_xy = xy
            # 录制最大节点速度 ~0.35 m/s（1.2 rad/s x ~0.3m 半径）；
            # wrap bug 时观测到 >5 m/s。给伺服跟踪留 3 倍余量。
            self.assertLess(max_speed, 1.0)
        finally:
            env.close()


class RenderModeInvarianceTests(unittest.TestCase):
    """The gripper geom-group toggle must be an observation-only change."""

    @staticmethod
    def make_env() -> CableGraspEnv:
        scenario = get_scenario("id_shape_nominal_current")
        config = EnvConfig(
            robot="nero",
            seed=0,
            episode_seconds=4.0,
            scenario_name=scenario.name,
            scenario_id=scenario.scenario_id,
            scenario_split=scenario.split.value,
            dynamicvla_cameras_enabled=True,
            **scenario.to_env_overrides(),
        )
        return CableGraspEnv(config)

    def test_toggle_parks_and_restores_gripper_group(self) -> None:
        env = self.make_env()
        try:
            toggle = GripperVisibilityToggle(env.model)
            self.assertEqual(len(toggle.geom_ids), 8)
            original = env.model.geom_group[toggle.geom_ids].copy()
            self.assertTrue(np.all(original != GRIPPER_RENDER_GROUP))
            with toggle.hidden():
                self.assertTrue(np.all(
                    env.model.geom_group[toggle.geom_ids]
                    == GRIPPER_RENDER_GROUP
                ))
            np.testing.assert_array_equal(
                env.model.geom_group[toggle.geom_ids], original
            )
        finally:
            env.close()

    def test_rendering_does_not_touch_physics_state(self) -> None:
        env = self.make_env()
        try:
            env.reset(randomize=True, seed=SEED)
            for _ in range(10):
                env.step(env.ready_ctrl)
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            ctrl = env.data.ctrl.copy()
            contacts = [
                (int(c.geom1), int(c.geom2), float(c.dist))
                for c in env.data.contact
            ]
            geom_groups = env.model.geom_group.copy()
            toggle = GripperVisibilityToggle(env.model)

            renderer = mujoco.Renderer(env.model, height=240, width=320)
            try:
                camera = env.config.dynamicvla_opst_camera_name
                renderer.update_scene(env.data, camera=camera)
                normal_rgb = renderer.render().copy()
                with toggle.hidden():
                    renderer.update_scene(env.data, camera=camera)
                    hidden_rgb = renderer.render().copy()
            finally:
                renderer.close()

            np.testing.assert_array_equal(env.data.qpos, qpos)
            np.testing.assert_array_equal(env.data.qvel, qvel)
            np.testing.assert_array_equal(env.data.ctrl, ctrl)
            np.testing.assert_array_equal(env.model.geom_group, geom_groups)
            self.assertEqual(
                contacts,
                [
                    (int(c.geom1), int(c.geom2), float(c.dist))
                    for c in env.data.contact
                ],
            )
            self.assertGreater(env.data.time, 0.0)
            # The hidden-gripper frame must differ somewhere; a zero diff
            # would mean the toggle did not reach the renderer.
            self.assertFalse(np.array_equal(normal_rgb, hidden_rgb))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
