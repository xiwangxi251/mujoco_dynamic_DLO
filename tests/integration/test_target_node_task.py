from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from panda_cable_grasp.env.environment import GraspState
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env
from panda_cable_grasp.rl.evaluate import evaluation_rl_config
from panda_cable_grasp.rl.target_node import warm_start_policy


def make(goal=True, **config):
    return make_rl_env(robot='nero', action_mode='task_space_vertical_down',
                       scenario_names=['id_static'], episode_seconds=3.0,
                       rl_config=RLConfig(target_node_task=goal, **config))


class TargetNodeTests(unittest.TestCase):
    def setUp(self):
        self.env = make()
        self.env.reset(seed=81, options={'goal_node_index': 20})

    def tearDown(self):
        self.env.close()

    def state(self, index):
        self.env.base_env.grasp_state = GraspState(
            body_id=self.env.base_env.cable_ids[index], candidate_time=0.0,
            bilateral_confirmed=True, last_bilateral_time=0.0, lost_contact_time=0.0,
        )

    def test_default_state_prefix_and_physics_are_unchanged(self):
        ordinary = make(False)
        try:
            old, _ = ordinary.reset(seed=81)
            new, _ = self.env.reset(seed=81)
            self.assertEqual(old.shape, (99,))
            self.assertEqual(new.shape, (109,))
            self.assertEqual(len(self.env.OBSERVATION_NAMES), 109)
            np.testing.assert_array_equal(old, new[:99])
            np.testing.assert_array_equal(ordinary.data.qpos, self.env.data.qpos)
            self.assertEqual(ordinary.base_env.target_body_id, self.env.base_env.target_body_id)
        finally:
            ordinary.close()

    def test_seeded_goal_sampling_and_fixed_episode_identity(self):
        sampled = []
        for seed in range(20):
            _, first = self.env.reset(seed=seed)
            _, second = self.env.reset(seed=seed)
            self.assertEqual(first['goal_node_index'], second['goal_node_index'])
            sampled.append(first['goal_node_index'])
        self.assertGreater(len(set(sampled)), 5)
        self.assertTrue(all(7 <= i <= 32 for i in sampled))
        _, first = self.env.reset(seed=81, options={'goal_node_index': 12})
        for _ in range(5):
            _, _, _, _, info = self.env.step(np.zeros(5))
            self.assertEqual(info['goal_node_index'], first['goal_node_index'])
        for invalid in (-1, 40, True, 1.5):
            with self.assertRaises(ValueError):
                self.env.reset(options={'goal_node_index': invalid})

    def test_target_observation_frame_velocity_and_tangent(self):
        base = self.env.base_env
        base.data.xmat[base.hand_id] = np.eye(3).reshape(-1)
        base.data.xpos[base.cable_ids[19]] = [.6, .2, .4]
        base.data.xpos[base.cable_ids[20]] = [.6, .3, .4]
        base.data.xpos[base.cable_ids[21]] = [.6, .4, .4]
        with patch.object(self.env, '_grasp_center_position', return_value=np.array([.5, .1, .2])), \
             patch.object(self.env, '_hand_velocity', return_value=(np.array([.1, .2, .3]), np.zeros(3))), \
             patch.object(base, 'body_linear_velocity', return_value=np.array([.2, .4, .6])):
            observed = self.env.goal.observation()
        np.testing.assert_allclose(observed, [1/39, .2, .4, .4, .1, .2, .3, 0, 1, 0], atol=1e-7)

    def test_folded_spatial_neighbor_is_not_material_goal(self):
        base = self.env.base_env
        base.data.xpos[base.cable_ids[35]] = base.data.xpos[base.cable_ids[20]]
        self.state(35)
        self.assertFalse(self.env.goal.matches())
        self.assertFalse(self.env.goal.qualify(True))
        _, _, _, index, _ = self.env._nearest_graspable_segment(base.data.xpos[base.cable_ids[35]])
        self.assertTrue(18 <= index < 22)
        self.state(22)
        self.assertTrue(self.env.goal.matches())
        self.state(23)
        self.assertFalse(self.env.goal.matches())

    def test_goal_hold_at_physics_substeps_does_not_use_any_node_latch(self):
        base = self.env.base_env
        self.state(35)
        with patch.object(base, '_success_qualification', return_value=True), \
             patch.object(base, '_update_physical_grasp_state'), \
             patch.object(base, '_update_rigid_motion_suspension_state'):
            for _ in range(41):
                base.step(base.ready_ctrl.copy())
            self.assertTrue(self.env.goal.any_success)
            self.assertFalse(base.ever_success)
            self.state(20)
            for _ in range(20):
                base.step(base.ready_ctrl.copy())
            self.assertFalse(base.ever_success)
            self.state(35)
            base.step(base.ready_ctrl.copy())
            self.assertEqual(base.success_hold, 0.0)
            self.state(20)
            for _ in range(41):
                base.step(base.ready_ctrl.copy())
            self.assertTrue(base.ever_success)

    def test_wrong_grasp_cannot_consume_or_earn_goal_milestones(self):
        self.state(35)
        status = {'pinch_confirmed': True, 'secured_grasp': True,
                  'grasp_lift_delta': .20, 'grasp_body_height_above_table': .22,
                  'strict_success_hold': .8, 'height_band_hold': .8}
        _, parts = self.env._reward(np.zeros(5), False, {'lifted_fraction': .3}, status)
        for key in ('reward_new_pinch', 'reward_new_aligned_pinch',
                    'reward_new_secured_grasp', 'reward_lift_progress',
                    'reward_cable_lift_progress', 'reward_strict_hold', 'reward_success'):
            self.assertEqual(parts[key], 0.0, key)
        self.assertLess(parts['reward_wrong_target_grasp'], 0)
        self.assertFalse(self.env._pinch_rewarded)
        self.assertFalse(self.env._secured_rewarded)
        self.state(20)
        _, parts = self.env._reward(np.zeros(5), False, {'lifted_fraction': .3}, status)
        self.assertGreater(parts['reward_new_pinch'], 0)
        self.assertGreater(parts['reward_new_secured_grasp'], 0)

    def test_invalid_task_config(self):
        for params in ({'target_u_min': -.1}, {'target_u_max': float('nan')},
                       {'target_u_min': .8, 'target_u_max': .2},
                       {'target_node_tolerance': -1}, {'target_node_tolerance': True}):
            with self.assertRaises(ValueError):
                RLConfig(**params)
        empty = make(target_u_min=.50001, target_u_max=.50002)
        try:
            with self.assertRaises(ValueError):
                empty.reset()
        finally:
            empty.close()


class WarmStartTests(unittest.TestCase):
    def test_weights_actions_fresh_optimizer_and_saved_goal_config(self):
        old_env = DummyVecEnv([lambda: make(False)])
        new_env = DummyVecEnv([make])
        try:
            source = PPO('MlpPolicy', old_env, n_steps=8, batch_size=8, device='cpu')
            target = PPO('MlpPolicy', new_env, n_steps=8, batch_size=8, device='cpu')
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                source.save(path / 'source')
                (path / 'training_config.json').write_text(json.dumps({
                    'action_mode': 'task_space_vertical_down',
                    'rl_interface_version': 'baseline_v6_task_space_vertical_down',
                }))
                warm_start_policy(target, path / 'source.zip')
                x = np.random.default_rng(9).uniform(-1, 1, (16, 99)).astype(np.float32)
                goals = np.random.default_rng(10).uniform(-1, 1, (16, 10)).astype(np.float32)
                before, _ = source.predict(x, deterministic=True)
                after, _ = target.predict(np.concatenate([x, goals], axis=1), deterministic=True)
                np.testing.assert_allclose(before, after, atol=1e-6)
                self.assertEqual(target.num_timesteps, 0)
                self.assertFalse(target.policy.optimizer.state)
                target.save(path / 'target')
                cfg = RLConfig(target_node_task=True, target_node_tolerance=1)
                (path / 'training_config.json').write_text(json.dumps({'rl_config': asdict(cfg)}))
                args = Namespace(model=path / 'target.zip', target_node_index=None,
                                 disable_singularity_avoidance=False)
                restored = evaluation_rl_config(args)
                self.assertTrue(restored.target_node_task)
                self.assertEqual(restored.target_node_tolerance, 1)
                loaded = PPO.load(path / 'target.zip', device='cpu')
                self.assertEqual(loaded.observation_space.shape, (109,))
                np.testing.assert_allclose(loaded.predict(np.concatenate([x, goals], axis=1), deterministic=True)[0], after)
        finally:
            old_env.close()
            new_env.close()
