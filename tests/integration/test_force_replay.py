"""Independent pure-force playback, not a live position/velocity servo."""
from dataclasses import replace
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np

from panda_cable_grasp.env.environment import CableGraspEnv, GraspState
from tests.integration import test_rigid_replay_env as fixtures

SEED = fixtures.SEED


class ForceReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ReplayPairingTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        fixtures.ReplayPairingTests.tearDownClass.__func__(cls)

    def make_env(self, version="force_tape_v5"):
        return CableGraspEnv(replace(self.replay_config, replay_controller_version=version, episode_seconds=2.))

    def test_no_intervention_reproduces_teacher_and_reference(self):
        replay = self.make_env()
        try:
            replay.reset(seed=SEED)
            probe = mujoco.MjData(replay.model)
            jac = np.zeros((3, replay.model.nv)); jacr = np.zeros_like(jac)
            errors=[]; velocity_errors=[]; speeds=[]; reference_speeds=[]
            for _ in range(100):
                replay.step(replay.ready_ctrl)
                if replay.data.time < .82:
                    continue
                probe.qpos[:]=replay.data.qpos; probe.qvel[:]=replay.data.qvel
                mujoco.mj_kinematics(replay.model, probe); mujoco.mj_comPos(replay.model, probe)
                body=int(replay.cable_ids[replay._replay_tracked_index])
                mujoco.mj_jacBody(replay.model, probe, jac, jacr, body)
                sample=replay._replay_reference.sample(replay._replay_elapsed)
                actual_velocity=(jac@probe.qvel)[:2]
                errors.append(probe.xpos[body,:2]-replay._replay_anchor_tracked-sample.position)
                velocity_errors.append(actual_velocity-sample.velocity)
                speeds.append(np.linalg.norm(actual_velocity)); reference_speeds.append(np.linalg.norm(sample.velocity))
            self.assertLess(np.sqrt(np.mean(np.sum(np.array(errors)**2,axis=1))), .005)
            self.assertLess(np.sqrt(np.mean(np.sum(np.array(velocity_errors)**2,axis=1))), .02)
            self.assertLess(abs(np.mean(speeds)/np.mean(reference_speeds)-1), .05)
            np.testing.assert_allclose(replay.data.qpos, replay._replay_force_tape.final_qpos, atol=1e-9, rtol=0)
            np.testing.assert_allclose(replay.data.qvel, replay._replay_force_tape.final_qvel, atol=1e-8, rtol=0)
        finally:
            replay.close()

    def test_confirmed_grasp_and_displacement_never_pause_or_change_tape(self):
        env=self.make_env()
        try:
            env.reset(seed=SEED)
            for _ in range(50): env.step(env.ready_ctrl)
            tape=env._replay_force_tape
            original=tape.forces.copy()
            anchor=env._replay_anchor_tracked.copy()
            body=int(env.cable_ids[env._replay_tracked_index])
            env._update_physical_grasp_state=lambda closed: None
            env.grasp_state=GraspState(body_id=body, candidate_time=env.data.time,
                bilateral_confirmed=True, last_bilateral_time=env.data.time, lost_contact_time=env.data.time)
            env.data.qpos[env.cable_free_qadr:env.cable_free_qadr+2] += [.03, -.04]
            mujoco.mj_forward(env.model, env.data)
            elapsed=env._replay_elapsed
            for _ in range(10):
                env.grasp_state.last_bilateral_time=env.data.time
                env.step(env.ready_ctrl)
                self.assertFalse(env.rigid_motion_suspended)
            self.assertAlmostEqual(env._replay_elapsed-elapsed, .2)
            env.grasp_state=None
            env.step(env.ready_ctrl)
            self.assertAlmostEqual(env._replay_elapsed-elapsed, .22)
            np.testing.assert_array_equal(env._replay_anchor_tracked, anchor)
            np.testing.assert_array_equal(tape.forces, original)
            self.assertEqual(env._replay_anchor_elapsed, 0.)
        finally:
            env.close()

    def test_airborne_force_gates_by_node_but_time_continues(self):
        env=self.make_env()
        try:
            env.reset(seed=SEED)
            for _ in range(50): env.step(env.ready_ctrl)
            env.data.qpos[env.cable_free_qadr+2] += .2
            mujoco.mj_forward(env.model, env.data)
            self.assertFalse(env._cable_table_contact_mask().any())
            elapsed=env._replay_elapsed
            env.data.xfrc_applied[:]=0.
            env._apply_cable_disturbance()
            np.testing.assert_array_equal(env.data.xfrc_applied[env.cable_ids], 0.)
            self.assertGreater(env._replay_elapsed, elapsed)
            np.testing.assert_array_equal(env._last_ungated_scene_force, env._replay_force_tape.sample(env.data.time))
        finally:
            env.close()

    def test_compiler_uses_reset_snapshot_even_if_first_action_is_displaced(self):
        env=self.make_env()
        try:
            env.reset(seed=SEED)
            frozen=env._replay_force_compiler.data.qpos.copy()
            frozen_time=env._replay_force_compiler.data.time
            env.data.qpos[env.cable_free_qadr:env.cable_free_qadr+2] += [.03,.02]
            mujoco.mj_forward(env.model,env.data)
            np.testing.assert_array_equal(env._replay_force_compiler.data.qpos,frozen)
            self.assertEqual(env._replay_force_compiler.data.time,frozen_time)
            # Compiler destruction must not close the actual camera renderer.
            renderer=Mock()
            env._dynamicvla_camera_renderer=renderer
            env.step(env.ready_ctrl)
            renderer.close.assert_not_called()
            self.assertIsNotNone(env._replay_force_tape)
            self.assertIsNone(env._replay_force_compiler)
        finally:
            env.close()
