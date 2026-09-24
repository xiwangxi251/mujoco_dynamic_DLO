"""Record global/wrist videos of a point-cloud PPO policy on given scenarios.

One episode per seed; deterministic inference; identical env construction to
``evaluate_pointcloud_ppo.py`` (same action mode, cameras, point-cloud config),
so the recorded behaviour matches the evaluation setting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from panda_cable_grasp.evaluation.recording import EpisodeRecorder  # noqa: E402
from panda_cable_grasp.rl.environment import RLConfig, make_rl_env  # noqa: E402
from panda_cable_grasp.rl.pointcloud import (  # noqa: E402
    DLOPointCloudObservation,
    PointCloudObservationConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenarios",
        required=True,
        help="comma-separated scenario names",
    )
    parser.add_argument(
        "--seeds",
        default="20280804,20280805,20280806,20280807,20280808",
        help="comma-separated episode seeds (must hit replay bank for replay scenarios)",
    )
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument(
        "--render-mode",
        choices=("normal", "gripper_hidden", "mixed"),
        default="mixed",
    )
    parser.add_argument(
        "--target-mask-radius", type=float, default=0.0,
        help="meters; drop cloud points within this radius of the target segment",
    )
    parser.add_argument(
        "--depth-noise-std", type=float, default=0.0,
        help="sigma = k*z^2 coefficient (m at z=1m)",
    )
    parser.add_argument(
        "--pixel-dropout", type=float, default=0.0,
        help="per-pixel missing-return probability",
    )
    parser.add_argument(
        "--frame-drop", type=float, default=0.0,
        help="whole-frame dropout probability",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable-table-finger-collision-filter", action="store_true"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    cloud_config = PointCloudObservationConfig(
        point_count=384,
        width=480,
        height=360,
        camera_update_steps=5,
        sensor_delay_steps=3,
        voxel_size_m=0.002,
        render_mode=args.render_mode,
        target_mask_radius_m=args.target_mask_radius,
        depth_noise_std_at_1m=args.depth_noise_std,
        pixel_dropout_p=args.pixel_dropout,
        frame_drop_p=args.frame_drop,
    )
    model = PPO.load(args.model, device=args.device)
    results: list[dict] = []

    for scenario in scenarios:
        env = make_rl_env(
            action_mode="task_space_vertical_down",
            robot="nero",
            seed=args.seed,
            disturbance_strength=1.5,
            episode_seconds=args.episode_seconds,
            dynamicvla_cameras_enabled=True,
            scenario_names=(scenario,),
            rl_config=RLConfig(singularity_avoidance_enabled=False),
            geometric_safety_enabled=False,
            table_finger_collision_filter_enabled=(
                not args.disable_table_finger_collision_filter
            ),
        )
        env.base_env.config.replay_seed_fallback = "error"
        wrapped = DLOPointCloudObservation(env, cloud_config)
        wrapped.set_training_scenarios((scenario,))
        wrapped.set_motion_difficulty(1.0)

        for seed in seeds:
            episode_dir = args.output / scenario / f"seed_{seed}"
            observation, _ = wrapped.reset(seed=seed)
            recorder = EpisodeRecorder(
                env, episode_dir, video_fps=args.video_fps
            )
            recorder.capture_initial()
            episode_return = 0.0
            steps = 0
            info = {}
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = wrapped.step(action)
                episode_return += float(reward)
                steps += 1
                recorder.record_step(action, reward, terminated, truncated, info)
            row = {
                "scenario": scenario,
                "seed": seed,
                "steps": steps,
                "return": episode_return,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                **{
                    key: info[key]
                    for key in (
                        "success", "strict_success", "pinch", "aligned_pinch",
                        "loaded_lift", "grasp", "task_success",
                        "termination_reason",
                    )
                    if key in info
                },
            }
            recorder.finish(row)
            results.append(row)
            print(json.dumps(row), flush=True)

        wrapped.close()

    (args.output / "episodes.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
