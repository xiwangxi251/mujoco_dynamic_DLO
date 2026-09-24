"""可变形态对象与多对象场景的单元测试。

覆盖：对象族规格、步态模板公式、场景 XML 编译、环境多对象运行时、
对象限定的抓取邻域、脚本策略与 RL 观测的多对象兼容性。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from panda_cable_grasp.env.environment import (
    CableGraspEnv,
    EnvConfig,
    _build_object_specs,
)
from panda_cable_grasp.env.objects import (
    OBJECT_FAMILIES,
    DeformableSpec,
    LaneDrive,
    build_scene_xml,
    family_spec,
    gait_template,
    template_frame_state,
)
from panda_cable_grasp.evaluation.motion_diagnostics import (
    env_config_for_scenario,
)
from panda_cable_grasp.policies import DynamicCableGraspPolicy
from panda_cable_grasp.scenarios import get_scenario


def _env_for(name: str, seconds: float = 6.0) -> CableGraspEnv:
    return CableGraspEnv(
        env_config_for_scenario(get_scenario(name), seed=7, episode_seconds=seconds)
    )


class ObjectFamilySpecTests(unittest.TestCase):
    def test_all_families_produce_valid_specs(self) -> None:
        for index, family in enumerate(sorted(OBJECT_FAMILIES)):
            spec = family_spec(family, f"{family}{index}")
            self.assertIsInstance(spec, DeformableSpec)
            self.assertEqual(spec.node_count, spec.count - 1)
            self.assertGreater(spec.length, 0.0)

    def test_unknown_family_rejected(self) -> None:
        with self.assertRaises(ValueError):
            family_spec("octopus", "octopus0")

    def test_radius_profiles_are_tapered(self) -> None:
        s_hat = np.linspace(0.0, 1.0, 40)
        fish = family_spec("fish", "fish0").radius_at(s_hat)
        loach = family_spec("loach", "loach0").radius_at(s_hat)
        whip = family_spec("whip", "whip0").radius_at(s_hat)
        cable = family_spec("cable", "cable0").radius_at(s_hat)
        # 鱼体中段最粗、头尾收窄；泥鳅整体缓收；鞭子柄粗尖细；线缆均匀。
        self.assertGreater(fish.max(), fish[0])
        self.assertGreater(fish.max(), fish[-1])
        self.assertGreater(loach[0], loach[-1])
        self.assertGreater(whip[0], 2.0 * whip[-1])
        self.assertTrue((np.diff(whip) <= 1e-9).all())
        np.testing.assert_allclose(cable, cable[0])

    def test_gait_scaling_overrides(self) -> None:
        spec = family_spec(
            "fish", "fish0",
            gait_amplitude_scale=0.5,
            gait_frequency_scale=0.5,
            gait_swim_speed=0.0,
        )
        base = family_spec("fish", "fish_base")
        self.assertAlmostEqual(spec.gait.amplitude, base.gait.amplitude * 0.5)
        self.assertAlmostEqual(
            spec.gait.frequency_hz, base.gait.frequency_hz * 0.5
        )
        self.assertEqual(spec.gait.swim_speed, 0.0)


class GaitTemplateTests(unittest.TestCase):
    def _template(self, family: str):
        spec = family_spec(family, f"{family}0")
        s = np.linspace(0.0, 1.0, spec.node_count)
        pos, vel = gait_template(
            spec.gait, spec.length, s, time_value=1.0, phase=0.0
        )
        return spec, s, pos, vel

    def test_carangiform_is_posterior_dominant(self) -> None:
        spec, s, pos, _ = self._template("fish")
        # 鲹形波：侧向幅值向尾部（s->1）显著增大。
        lateral = np.abs(pos[:, 1])
        head = lateral[s < 0.25].max()
        tail = lateral[s > 0.75].max()
        self.assertGreater(tail, 2.0 * head)

    def test_anguilliform_whole_body_wave(self) -> None:
        spec, s, pos, _ = self._template("loach")
        lateral = np.abs(pos[:, 1])
        # 鳗形波全身均有明显侧向位移（头段不明显小于尾段一个量级）。
        self.assertGreater(lateral[s < 0.25].max(), 0.2 * lateral.max())

    def test_serpenoid_tangent_angle_wave(self) -> None:
        spec, s, pos, _ = self._template("snake")
        # serpenoid：相邻节点切向角沿弧长正弦变化，形状平滑蜿蜒。
        tangents = np.diff(pos[:, :2], axis=0)
        angles = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
        self.assertGreater(angles.max() - angles.min(), 0.5)

    def test_peristaltic_is_axial(self) -> None:
        spec, s, pos, _ = self._template("worm")
        # 蠕动波：轴向位移幅值明显大于侧向。
        axial = np.abs(pos[:, 0] - pos[:, 0].mean())
        lateral = np.abs(pos[:, 1])
        self.assertGreater(axial.max(), 2.0 * lateral.max())

    def test_template_produces_finite_positions(self) -> None:
        for family in ("fish", "loach", "snake", "worm",
                       "spring", "ribbon", "hose", "whip"):
            _, _, pos, vel = self._template(family)
            self.assertTrue(np.isfinite(pos).all(), family)
            self.assertTrue(np.isfinite(vel).all(), family)

    def test_whip_amplitude_grows_toward_tip(self) -> None:
        spec, s, pos, _ = self._template("whip")
        lateral = np.abs(pos[:, 1])
        self.assertGreater(
            lateral[s > 0.8].max(), 1.5 * lateral[s < 0.2].max()
        )

    def test_frame_state_straight_and_turning(self) -> None:
        spec = family_spec("fish", "fish0")
        gait = spec.gait
        offset, heading, velocity, _ = template_frame_state(gait, 2.0)
        self.assertAlmostEqual(np.linalg.norm(velocity), gait.swim_speed)
        # 转向步态产生弧形轨迹。
        turning = family_spec("snake", "snake_t")
        gait_t = turning.gait.__class__(
            **{**turning.gait.__dict__, "turn_rate_deg": 30.0, "swim_speed": 0.1}
        )
        offset_t, heading_t, _, rate = template_frame_state(gait_t, 2.0)
        self.assertAlmostEqual(rate, math.radians(30.0))
        self.assertGreater(abs(heading_t - math.radians(gait_t.heading_deg)), 0.1)


class SceneXmlTests(unittest.TestCase):
    def test_generated_xml_has_unique_names(self) -> None:
        specs = [
            family_spec("cable", "cable0"),
            family_spec("fish", "fish1"),
            family_spec("snake", "snake2"),
        ]
        offsets = [(0.4, 0.0, 0.02), (0.6, -0.5, 0.03), (0.8, -1.0, 0.02)]
        xml = build_scene_xml(
            specs,
            table_half_size=(0.9, 1.1),
            composite_offsets=offsets,
            njmax=4000,
            nconmax=4000,
        )
        for spec in specs:
            self.assertIn(f'prefix="{spec.name}"', xml)
        # 每对象 40 个 pair 覆盖 + 桌面 geom。
        self.assertEqual(xml.count("<pair geom1=\"table\""), 120)


class MultiObjectEnvTests(unittest.TestCase):
    def test_single_family_scenes_build_and_reset(self) -> None:
        for name, family in (
            ("dev_fish_swim", "fish"),
            ("dev_loach_wriggle", "loach"),
            ("dev_snake_serpent", "snake"),
            ("dev_worm_crawl", "worm"),
            ("dev_spring_pulse", "spring"),
            ("dev_ribbon_wave", "ribbon"),
            ("dev_hose_swing", "hose"),
            ("dev_whip_lash", "whip"),
        ):
            env = _env_for(name)
            try:
                env.reset(seed=7)
                self.assertEqual(len(env._objects), 1)
                self.assertEqual(env._objects[0].spec.family, family)
                self.assertEqual(len(env.cable_ids), 40)
                for _ in range(60):
                    env.step(env.ready_ctrl)
                positions = env.data.xpos[env.cable_ids]
                self.assertTrue(np.isfinite(positions).all())
            finally:
                env.close()

    def test_multi_object_layouts(self) -> None:
        for name, expected in (
            ("dev_multi_conveyor3", 3),
            ("dev_multi_conveyor3_rigid", 3),
            ("dev_multi_crossing2", 2),
            ("dev_multi_parallel3", 3),
            ("dev_multi_menagerie3", 3),
            ("dev_multi_menagerie4", 4),
        ):
            env = _env_for(name)
            try:
                env.reset(seed=7)
                self.assertEqual(len(env._objects), expected)
                self.assertEqual(len(env.cable_ids), 40 * expected)
                # 扁平索引切片不重叠且连续。
                slices = env._object_slices
                for left, right in zip(slices, slices[1:]):
                    self.assertEqual(left.stop, right.start)
                for _ in range(60):
                    env.step(env.ready_ctrl)
                positions = env.data.xpos[env.cable_ids]
                self.assertTrue(np.isfinite(positions).all())
            finally:
                env.close()

    def test_body_names_are_unique(self) -> None:
        env = _env_for("dev_multi_menagerie3")
        try:
            env.reset(seed=7)
            names = [
                env.model.body(i).name
                for i in range(env.model.nbody)
                if env.model.body(i).name
            ]
            self.assertEqual(len(names), len(set(names)))
        finally:
            env.close()

    def test_grasp_neighborhood_stays_within_object(self) -> None:
        """抓取候选的邻域节点不得跨越对象边界。"""
        env = _env_for("dev_multi_conveyor3")
        try:
            env.reset(seed=7)
            for obj_index, obj in enumerate(env._objects):
                node_slice = env._object_slices[obj_index]
                middle_flat = node_slice.start + len(obj.ids) // 2
                radius = env.config.grasp_contact_index_radius
                neighborhood = env.cable_ids[
                    max(node_slice.start, middle_flat - radius):
                    min(node_slice.stop, middle_flat + radius + 1)
                ]
                for body_id in neighborhood:
                    self.assertEqual(
                        env._node_object[env.cable_index[body_id]], obj_index
                    )
        finally:
            env.close()

    def test_scripted_nearest_point_within_target_object(self) -> None:
        env = _env_for("dev_multi_conveyor3")
        try:
            env.reset(seed=7)
            policy = DynamicCableGraspPolicy(env)
            policy.reset()
            target = env._objects[env._target_object]
            point = env.data.xpos[np.asarray(target.ids)].mean(axis=0)
            _, _, index, _ = policy._nearest_cable_point(point)
            node_slice = env._object_slices[env._target_object]
            self.assertGreaterEqual(index, 0)
            self.assertLess(index, node_slice.stop - node_slice.start - 1)
        finally:
            env.close()

    def test_gait_objects_have_gait_acceleration(self) -> None:
        env = _env_for("dev_fish_swim")
        try:
            env.reset(seed=7)
            for _ in range(80):
                env.step(env.ready_ctrl)
            accel = env._last_shape_acceleration
            self.assertIsNotNone(accel)
            # 步态伺服产生非零驱动加速度。
            self.assertGreater(np.abs(accel).max(), 0.01)
        finally:
            env.close()

    def test_conveyor_queue_positions_and_exit(self) -> None:
        env = _env_for("dev_multi_conveyor3", seconds=14.0)
        try:
            env.reset(seed=7)
            lanes = [obj.lane for obj in env._objects]
            self.assertTrue(all(lane is not None for lane in lanes))
            # 排队间距沿车道反方向递增。
            starts = np.array([obj.start_com for obj in env._objects])
            direction = env._objects[0].lane_dir
            projections = (starts - starts[0]) @ direction
            self.assertTrue((projections <= 1e-6).all())
            self.assertGreater(
                abs(projections[1] - projections[0]), 0.3
            )
            # 跑到足够长时间后至少第一个对象越线。
            dt = env.model.opt.timestep * env.config.frame_skip
            for _ in range(int(13.0 / dt)):
                env.step(env.ready_ctrl)
            self.assertTrue(env._objects[0].exited)
        finally:
            env.close()

    def test_rl_observation_shape_on_multi_object(self) -> None:
        from panda_cable_grasp.rl.environment import RLCableGraspEnv

        cfg = env_config_for_scenario(
            get_scenario("dev_multi_conveyor3"), seed=5, episode_seconds=4.0
        )
        env = RLCableGraspEnv(env_config=cfg)
        try:
            obs, _ = env.reset(seed=5)
            self.assertEqual(obs.shape, (99,))
            for _ in range(20):
                obs, _, _, _, _ = env.step(np.zeros(5))
            self.assertTrue(np.isfinite(obs).all())
        finally:
            env.close()


class VisualIdentityTests(unittest.TestCase):
    """可辨识外观：装饰 geom 与族颜色。"""

    def _deco_ids(self, env: CableGraspEnv, prefix: str) -> list[int]:
        import mujoco
        return [
            geom_id for geom_id in range(env.model.ngeom)
            if (mujoco.mj_id2name(
                env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ) or "").startswith(f"{prefix}Deco")
        ]

    def test_decorations_are_visual_only(self) -> None:
        env = _env_for("dev_loach_wriggle")
        try:
            env.reset(seed=7)
            deco = self._deco_ids(env, "loach0")
            self.assertGreaterEqual(len(deco), 6)  # 三对触须
            for geom_id in deco:
                self.assertEqual(env.model.geom_contype[geom_id], 0)
                self.assertEqual(env.model.geom_conaffinity[geom_id], 0)
                # 装饰 geom 不计入对象物理胶囊集合（不参与孔径估计）。
                self.assertNotIn(geom_id, env._objects[0].geom_ids.tolist())
        finally:
            env.close()

    def test_each_family_has_distinguishing_decorations(self) -> None:
        expected = {
            "dev_fish_swim": ("fish0", 0),  # 真 mesh 蒙皮自带眼睛/鳍纹理
            "dev_snake_serpent": ("snake0", 0),  # 真蛇 mesh 自带头形/纹理
            "dev_loach_wriggle": ("loach0", 6),
            "dev_spring_pulse": ("spring0", 10),
            "dev_ribbon_wave": ("ribbon0", 0),
            "dev_hose_swing": ("hose0", 2),
            "dev_whip_lash": ("whip0", 2),
        }
        for name, (prefix, minimum) in expected.items():
            env = _env_for(name)
            try:
                env.reset(seed=7)
                self.assertGreaterEqual(
                    len(self._deco_ids(env, prefix)), minimum, name
                )
            finally:
                env.close()

    def test_family_color_survives_steps_and_scene_update(self) -> None:
        """vmax=0 关掉应力配色后，族色在 step 与 update_scene 后不变。"""
        import mujoco

        env = _env_for("dev_multi_conveyor3")  # cable 族无蒙皮，颜色在胶囊上
        try:
            env.reset(seed=7)
            spec = env._object_specs[0]
            for _ in range(10):
                env.step(env.ready_ctrl)
            geom_id = env._objects[0].geom_ids[0]
            renderer = mujoco.Renderer(env.model, height=64, width=64)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = env.data.xpos[env._objects[0].ids].mean(axis=0)
            cam.distance = 0.6
            renderer.update_scene(env.data, cam)
            np.testing.assert_allclose(
                env.model.geom_rgba[geom_id], np.asarray(spec.rgba), atol=1e-6
            )
            renderer.close()
        finally:
            env.close()

    def test_skinned_families_compile_and_hide_capsules(self) -> None:
        """蒙皮族：nskin>0、胶囊 geom alpha=0（隐藏但仍碰撞）。"""
        env = _env_for("dev_fish_swim")
        try:
            env.reset(seed=7)
            self.assertEqual(env.model.nskin, 1)
            capsule = env._objects[0].geom_ids
            np.testing.assert_allclose(
                env.model.geom_rgba[capsule, 3], 0.0
            )
            # 胶囊仍参与碰撞：contype 非零
            self.assertTrue((env.model.geom_contype[capsule] > 0).all())
            for _ in range(10):
                env.step(env.ready_ctrl)
        finally:
            env.close()

    def test_fish_uses_real_mesh_skin_with_texture(self) -> None:
        """fish 族走真 mesh 蒙皮分支：2188 顶点、带 UV 贴图。"""
        env = _env_for("dev_fish_swim")
        try:
            env.reset(seed=7)
            spec = env._object_specs[0]
            self.assertEqual(spec.skin_mesh, "barramundi.glb")
            self.assertEqual(env.model.nskin, 1)
            self.assertEqual(int(env.model.skin_vertnum[0]), 2188)
            # 贴图材质已挂上（texcoord 顶点数与顶点一致）
            self.assertEqual(int(env.model.nskintexvert), 2188)
            self.assertGreaterEqual(int(env.model.skin_matid[0]), 0)
            # 蒙皮是纯渲染层：对象 geom 集不含皮肤、仍只含胶囊
            self.assertEqual(len(env._objects[0].geom_ids), spec.node_count)
            for _ in range(10):
                env.step(env.ready_ctrl)
        finally:
            env.close()

    def test_nero_robot_builds_object_scenes(self) -> None:
        from dataclasses import replace

        cfg = replace(
            env_config_for_scenario(
                get_scenario("dev_fish_swim"), seed=7, episode_seconds=2.0
            ),
            robot="nero",
        )
        env = CableGraspEnv(cfg)
        try:
            env.reset(seed=7)
            self.assertEqual(env.config.robot, "nero")
            for _ in range(10):
                env.step(env.ready_ctrl)
        finally:
            env.close()


class LegacyPreservationTests(unittest.TestCase):
    def test_default_single_cable_uses_legacy_path(self) -> None:
        env = _env_for("id_static")
        try:
            self.assertFalse(env.config._uses_object_scene)
            self.assertEqual(len(env._objects), 1)
            self.assertEqual(env._objects[0].spec.family, "cable")
            self.assertEqual(len(env.cable_ids), 40)
        finally:
            env.close()

    def test_legacy_scenario_ids_unchanged(self) -> None:
        # 旧场景哈希在加入对象字段后保持不变。
        expected = {
            "id_static": "scenario-v1-90660fd760d9b37e",
            "id_shape_nominal_current": "scenario-v1-5ab67431c3c28afa",
            "id_rigid_l1_nominal": "scenario-v1-3019f9c8a2df824d",
            "id_combined_l1_nominal": "scenario-v1-7e3bcd56713043e5",
        }
        for name, scenario_id in expected.items():
            self.assertEqual(get_scenario(name).scenario_id, scenario_id)


if __name__ == "__main__":
    unittest.main()
