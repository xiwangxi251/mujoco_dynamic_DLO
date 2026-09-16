"""Fast contract tests for the optional Diffusion Policy method."""

from __future__ import annotations

import importlib.util
from pathlib import Path
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
            spatial_num_keypoints=8,
            diffusion_step_embed_dim=32,
            unet_down_dims=(32, 64, 128),
            diffusion_steps=8,
            inference_steps=3,
            prediction_horizon=4,
            action_horizon=2,
        )
        model = DiffusionPolicy(config)
        observations = {
            "opst_cam": torch.rand(2, 2, 3, 84, 84),
            "wrist_cam": torch.rand(2, 2, 3, 84, 84),
            "state": torch.rand(2, 2, 6),
        }
        actions = torch.rand(2, 4, 7) * 2.0 - 1.0
        self.assertIsNot(model.opst_encoder, model.wrist_encoder)
        loss = model.forward_loss(**observations, action=actions)
        sample = model.sample(**observations)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(tuple(sample.shape), (2, 4, 7))


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional PyTorch")
class DiffusionEpisodeDatasetContractTests(unittest.TestCase):
    """The raw-NPZ dataset must match the DynamicVLA/LeRobot contract."""

    def test_items_are_6d_euler_state_and_7d_delta_actions(self) -> None:
        import torch  # noqa: F401 - dataset returns torch tensors

        from panda_cable_grasp.diffusion_policy.dataset import (
            DiffusionEpisodeDataset,
            EpisodeRecord,
        )

        config = DiffusionPolicyConfig(
            observation_horizon=2,
            prediction_horizon=4,
            action_horizon=2,
        )
        frames = 12
        rng = np.random.default_rng(0)
        state = rng.uniform(-0.5, 0.5, size=(frames, 6)).astype(np.float32)
        action = rng.uniform(-0.5, 0.5, size=(frames, 7)).astype(np.float32)
        record = EpisodeRecord(
            trajectory=Path("episode_000.npz"),
            opst_video=Path("global.mp4"),
            wrist_video=Path("wrist.mp4"),
            state=state,
            action=action,
            scenario="id_static",
            seed=0,
        )
        dataset = DiffusionEpisodeDataset.from_records([record], config=config)
        images = np.zeros((frames, 84, 84, 3), dtype=np.uint8)
        dataset._videos = lambda _index: (images, images)

        item = dataset[3]
        self.assertEqual(tuple(item["state"].shape), (2, 6))
        self.assertEqual(tuple(item["action"].shape), (4, 7))
        self.assertEqual(dataset.state_low.shape, (6,))
        self.assertEqual(dataset.action_low.shape, (7,))

        # Every future action is a delta from the current frame's state; the
        # gripper channel stays absolute.
        expected_delta = action[3:7].copy()
        expected_delta[:, :6] -= state[3, :6]
        expected = dataset._normalized(
            expected_delta, dataset.action_low, dataset.action_high
        )
        np.testing.assert_allclose(item["action"].numpy(), expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
