from __future__ import annotations

import json
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

    def test_replay_resolves_model_from_three_level_episode_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            models_dir = run_dir / "models"
            models_dir.mkdir(parents=True)
            episode_dir = run_dir / "episodes" / "recording_test" / "seed_7"
            episode_dir.mkdir(parents=True)
            env = CableGraspEnv(EnvConfig(
                seed=7,
                episode_seconds=0.02,
                scenario_name="recording_test",
                dynamicvla_cameras_enabled=True,
            ))
            try:
                env.reset(seed=7)
                model_path = models_dir / "recording_test.mjb"
                mujoco.mj_saveModel(env.model, str(model_path), None)
            finally:
                env.close()
            state_size = mujoco.mj_stateSize(
                mujoco.MjModel.from_binary_path(str(model_path)),
                mujoco.mjtState.mjSTATE_FULLPHYSICS,
            )
            np.savez_compressed(
                episode_dir / "trajectory.npz",
                schema_version=np.asarray(1, dtype=np.int64),
                state_spec=np.asarray(
                    int(mujoco.mjtState.mjSTATE_FULLPHYSICS), dtype=np.int64
                ),
                states=np.zeros((2, state_size), dtype=np.float64),
                state_times=np.asarray([0.0, 0.02], dtype=np.float64),
            )
            (episode_dir / "episode.json").write_text(
                json.dumps({
                    "files": {"trajectory": "trajectory.npz"},
                    "result": {"compiled_model": "models/recording_test.mjb"},
                }),
                encoding="utf-8",
            )

            model, states, times, state_spec = load_episode(episode_dir)
            self.assertEqual(states.shape[1], mujoco.mj_stateSize(model, state_spec))
            np.testing.assert_allclose(times, [0.0, 0.02])

            (episode_dir / "episode.json").write_text(
                json.dumps({
                    "files": {"trajectory": "trajectory.npz"},
                    "result": {},
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_episode(episode_dir)


if __name__ == "__main__":
    unittest.main()
