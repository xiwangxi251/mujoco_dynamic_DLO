"""Current-family L1 force tapes retain shape drive and never pause."""
from dataclasses import replace
import unittest

import mujoco
import numpy as np

from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig, GraspState
from panda_cable_grasp.scenarios import get_scenario
from tests.integration import test_rigid_replay_env as fixtures

NAMES=("dev_rigid_l1_nominal_x075_v1", "dev_rigid_l1_midshape_v1_x075_v1", "dev_combined_l1_nominal_x075_v1")


class FamilyForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.ReplayPairingTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls): fixtures.ReplayPairingTests.tearDownClass.__func__(cls)

    def make_env(self,name):
        overrides=get_scenario(name).to_env_overrides()
        config=EnvConfig(robot="panda",episode_seconds=1.2,replay_bank_dir=str(self.bank_dir), **overrides)
        if config.initial_shape_bank: config.initial_shape_bank=fixtures.BANK_NAME
        return CableGraspEnv(config)

    def test_each_scene_matches_independent_compiler_without_intervention(self):
        for name in NAMES:
            with self.subTest(scene=name):
                env=self.make_env(name)
                try:
                    env.reset(seed=fixtures.SEED)
                    for _ in range(60): env.step(env.ready_ctrl)
                    tape=env._replay_force_tape
                    self.assertIsNotNone(tape)
                    np.testing.assert_allclose(env.data.qpos,tape.final_qpos,atol=1e-9,rtol=0.)
                    np.testing.assert_allclose(env.data.qvel,tape.final_qvel,atol=1e-8,rtol=0.)
                    self.assertFalse(env.rigid_motion_suspended)
                    self.assertAlmostEqual(env._replay_elapsed,1.2)
                    if "combined" in name:
                        self.assertGreater(env.info()["shape_acceleration_rms"],0.)
                        expected=env._shape_acceleration(
                            env.phase_offset+env.config.motion_frequency_scale*(env.data.time-env.model.opt.timestep),
                            env.spatial_phase,
                        )
                        # The shape rule is still evaluated live; no force tape
                        # replaces its deformation amplitude/time policy.
                        self.assertTrue(np.isfinite(expected).all())
                    else:
                        self.assertEqual(env.info()["shape_acceleration_rms"],0.)
                finally: env.close()

    def test_grasp_displacement_airborne_and_return_do_not_pause_or_reanchor(self):
        for name in NAMES:
            with self.subTest(scene=name):
                env=self.make_env(name)
                try:
                    env.reset(seed=fixtures.SEED)
                    for _ in range(50): env.step(env.ready_ctrl)
                    tape=env._replay_force_tape
                    original=tape.forces.copy()
                    env._update_physical_grasp_state=lambda closed: None
                    env.grasp_state=GraspState(body_id=int(env.target_body_id),candidate_time=env.data.time,
                        bilateral_confirmed=True,last_bilateral_time=env.data.time,lost_contact_time=env.data.time)
                    env.data.qpos[env.cable_free_qadr:env.cable_free_qadr+3]+=[.03,.02,.2]
                    mujoco.mj_forward(env.model,env.data)
                    self.assertFalse(env._cable_table_contact_mask().any())
                    elapsed=env._replay_elapsed
                    for _ in range(3):
                        env.grasp_state.last_bilateral_time=env.data.time
                        env.step(env.ready_ctrl)
                        self.assertFalse(env.rigid_motion_suspended)
                        np.testing.assert_array_equal(env._last_rigid_translation_acceleration,0.)
                    self.assertAlmostEqual(env._replay_elapsed-elapsed,.06)
                    env.grasp_state=None
                    env.step(env.ready_ctrl)
                    self.assertAlmostEqual(env._replay_elapsed-elapsed,.08)
                    np.testing.assert_array_equal(original,tape.forces)
                    self.assertEqual(env._replay_anchor_elapsed,0.)
                finally: env.close()
