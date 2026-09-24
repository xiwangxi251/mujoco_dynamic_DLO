"""standing_wave / traveling_wave 规律形状运动场景的单元测试。

``dev_shape_standing_wave`` 与 ``dev_shape_traveling_wave`` 是 regularity
轴上"完全可预测"的端点：单模态形状波在空间和时间上都严格周期。
这些测试验证场景注册、驱动加速度的严格周期性，以及与其他 shape
分支一致的净平动/净转动去除。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from panda_cable_grasp.env.environment import (
    CableGraspEnv,
    SINE_SERVO_ANGULAR_FREQUENCY,
    WAVE_ANGULAR_FREQUENCY,
)
from panda_cable_grasp.evaluation.motion_diagnostics import (
    env_config_for_scenario,
)
from panda_cable_grasp.scenarios import MotionRegularity, get_scenario

WAVE_SCENARIOS = (
    "dev_shape_sine",
    "dev_shape_standing_wave",
    "dev_shape_traveling_wave",
)


def _wave_env(name: str, seconds: float = 6.0) -> CableGraspEnv:
    return CableGraspEnv(
        env_config_for_scenario(
            get_scenario(name), seed=7, episode_seconds=seconds
        )
    )


class WaveScenarioTests(unittest.TestCase):
    def test_scenarios_registered_and_overrides(self) -> None:
        expected = {
            "dev_shape_sine": MotionRegularity.SINE_SERVO,
            "dev_shape_standing_wave": MotionRegularity.STANDING_WAVE,
            "dev_shape_traveling_wave": MotionRegularity.TRAVELING_WAVE,
        }
        for name, regularity in expected.items():
            scenario = get_scenario(name)
            self.assertIs(scenario.regularity, regularity)
            overrides = scenario.to_env_overrides()
            self.assertEqual(overrides["motion_regularity"], regularity.value)
            self.assertEqual(overrides["motion_mode"], "shape")
        self.assertEqual(
            get_scenario("dev_shape_sine").to_env_overrides()[
                "arm_motion_start_delay"
            ],
            1.5,
        )

    def test_wave_acceleration_is_strictly_periodic(self) -> None:
        periods = {
            "dev_shape_sine": 2.0 * math.pi / SINE_SERVO_ANGULAR_FREQUENCY,
            "dev_shape_standing_wave": 2.0 * math.pi / WAVE_ANGULAR_FREQUENCY,
            "dev_shape_traveling_wave": 2.0 * math.pi / WAVE_ANGULAR_FREQUENCY,
        }
        for name in WAVE_SCENARIOS:
            period = periods[name]
            env = _wave_env(name)
            try:
                env.reset(seed=7)
                # 同一物理状态下相差整数个周期的驱动场必须完全一致。
                first = env._shape_acceleration(1.0, 0.3)
                second = env._shape_acceleration(1.0 + 3.0 * period, 0.3)
                np.testing.assert_allclose(first, second, atol=1e-9)
            finally:
                env.close()

    def test_wave_removes_net_translation(self) -> None:
        for name in WAVE_SCENARIOS:
            env = _wave_env(name)
            try:
                env.reset(seed=7)
                for t in (0.0, 0.4, 1.1):
                    shape = env._shape_acceleration(t, 0.0)
                    net = np.average(
                        shape, axis=0, weights=env.cable_mass
                    )
                    np.testing.assert_allclose(net, np.zeros(3), atol=1e-8)
                    self.assertGreater(np.abs(shape).max(), 1e-3)
            finally:
                env.close()

    def test_free_run_stays_finite_and_bounded(self) -> None:
        for name in WAVE_SCENARIOS:
            env = _wave_env(name)
            try:
                env.reset(seed=7)
                for _ in range(120):
                    env.step(env.ready_ctrl)
                positions = env.data.xpos[env.cable_ids]
                self.assertTrue(np.isfinite(positions).all())
                # 纯形状驱动：质心不得明显漂移出工作区。
                com = np.average(
                    positions[:, :2], axis=0, weights=env.cable_mass
                )
                self.assertLess(np.linalg.norm(com - [0.55, 0.0]), 0.45)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
