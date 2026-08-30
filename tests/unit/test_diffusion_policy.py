"""Fast contract tests for the optional Diffusion Policy method."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from panda_cable_grasp.diffusion_policy.config import (
    ACTION_HIGH,
    ACTION_LOW,
    DiffusionPolicyConfig,
    denormalize_array,
    normalize_array,
    split_episode_indices,
)


class DiffusionPolicyConfigTests(unittest.TestCase):
    def test_dynamicvla_action_normalization_round_trips(self) -> None:
        values = np.asarray(ACTION_LOW, dtype=np.float32)
        values[3:] = 0.0
        encoded = normalize_array(values, ACTION_LOW, ACTION_HIGH)
        decoded = denormalize_array(encoded, ACTION_LOW, ACTION_HIGH)
        np.testing.assert_allclose(decoded, values, atol=1e-6)

    def test_default_action_chunk_is_receding_horizon_compatible(self) -> None:
        config = DiffusionPolicyConfig()
        self.assertEqual(config.observation_horizon, 2)
        self.assertEqual(config.prediction_horizon, 16)
        self.assertEqual(config.action_horizon, 8)

    def test_episode_split_is_disjoint_and_deterministic(self) -> None:
        train_a, validation_a = split_episode_indices(20, 0.2, 7)
        train_b, validation_b = split_episode_indices(20, 0.2, 7)
        self.assertEqual(train_a, train_b)
        self.assertEqual(validation_a, validation_b)
        self.assertTrue(set(train_a).isdisjoint(validation_a))
        self.assertEqual(len(train_a) + len(validation_a), 20)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional PyTorch")
class DiffusionPolicyTorchShapeTests(unittest.TestCase):
    def test_model_loss_and_sampling_shapes(self) -> None:
        import torch

        from panda_cable_grasp.diffusion_policy.model import DiffusionPolicy

        config = DiffusionPolicyConfig(
            image_feature_dim=16,
            state_feature_dim=8,
            denoiser_dim=32,
            denoiser_blocks=2,
            diffusion_steps=8,
            inference_steps=3,
            prediction_horizon=4,
            action_horizon=2,
        )
        model = DiffusionPolicy(config)
        observations = {
            "opst_cam": torch.rand(2, 2, 3, 84, 84),
            "wrist_cam": torch.rand(2, 2, 3, 84, 84),
            "state": torch.rand(2, 2, 7),
        }
        actions = torch.rand(2, 4, 8) * 2.0 - 1.0
        loss = model.forward_loss(**observations, action=actions)
        sample = model.sample(**observations)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(tuple(sample.shape), (2, 4, 8))


if __name__ == "__main__":
    unittest.main()
