from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig
from panda_cable_grasp.evaluation.recording import EpisodeRecorder
from panda_cable_grasp.evaluation.replay import load_episode


class EvaluationRecordingTests(unittest.TestCase):
    def test_fullphysics_episode_can_be_loaded_with_saved_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            models_dir = run_dir / "models"
            models_dir.mkdir(parents=True)
            episode_dir = (
                run_dir / "episodes" / "scripted" / "recording_test"
                / "seed_20280804"
            )
            env = CableGraspEnv(EnvConfig(
                seed=20280804,
                episode_seconds=0.04,
                scenario_name="recording_test",
                dynamicvla_cameras_enabled=True,
            ))
            try:
                env.reset(seed=20280804)
                model_path = models_dir / "recording_test.mjb"
                mujoco.mj_saveModel(env.model, str(model_path), None)
                recorder = EpisodeRecorder(env, episode_dir, video_fps=10.0)
                recorder.capture_initial()
                action = env.ready_ctrl.copy()
                _, reward, terminated, truncated, info = env.step(action)
                recorder.record_step(action, reward, terminated, truncated, info)
                artifacts = recorder.finish({
                    "compiled_model": "models/recording_test.mjb",
                    "method": "scripted",
                    "seed": 20280804,
                })
            finally:
                env.close()

            with np.load(artifacts.trajectory) as trajectory:
                self.assertEqual(trajectory["states"].shape[0], 2)
                self.assertEqual(trajectory["policy_actions"].shape, (1, 8))
                self.assertEqual(trajectory["requested_actions"].shape, (1, 8))
                self.assertEqual(trajectory["applied_actions"].shape, (1, 8))
                self.assertEqual(trajectory["frame_state_indices"][0], 0)
                recorded_times = trajectory["state_times"].copy()
            self.assertGreater(artifacts.global_video.stat().st_size, 0)
            self.assertGreater(artifacts.wrist_video.stat().st_size, 0)

            model, states, times, state_spec = load_episode(episode_dir)
            self.assertEqual(states.shape[1], mujoco.mj_stateSize(model, state_spec))
            np.testing.assert_allclose(times, recorded_times)


if __name__ == "__main__":
    unittest.main()
