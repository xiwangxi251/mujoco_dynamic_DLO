"""Evaluate a low-dimensional Diffusion Policy, optionally recording videos."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    from ..scenarios.registry import list_scenario_names

    parser = argparse.ArgumentParser(description="Evaluate low-dimensional Diffusion Policy")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario", choices=list_scenario_names(), required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20280804)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"checkpoint not found: {args.model}")
    if args.trials < 1 or args.episode_seconds <= 0.0 or args.video_fps <= 0.0:
        parser.error("trials, episode duration, and video FPS must be positive")
    return args


def run(args: argparse.Namespace) -> Path:
    import numpy as np

    from ..diffusion_policy.lowdim import LowDimPolicyRunner
    from ..env.environment import CableGraspEnv, EnvConfig
    from ..evaluation.benchmark import base_row, sha256_file, summarize, write_csv
    from ..evaluation.motion_diagnostics import env_config_for_scenario
    from ..evaluation.recording import EpisodeRecorder
    from ..scenarios.registry import get_scenario

    scenario = get_scenario(args.scenario)
    config = env_config_for_scenario(
        scenario, seed=args.seed, episode_seconds=args.episode_seconds
    )
    config = EnvConfig(
        **{
            **vars(config),
            "frame_skip": 20,
            "dynamicvla_cameras_enabled": bool(args.record_video),
        }
    )
    env = CableGraspEnv(config)
    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    run_dir = args.output_dir.expanduser().resolve() / run_name
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    rows: list[dict] = []
    try:
        policy = LowDimPolicyRunner(
            env, args.model, device=args.device, deterministic=args.deterministic
        )
        for trial_offset in range(args.trials):
            seed = args.seed + trial_offset
            _, initial_info = env.reset(seed=seed)
            policy.reset(seed=seed)
            recorder = None
            if args.record_video:
                recorder = EpisodeRecorder(
                    env,
                    run_dir / "episodes" / scenario.name / f"seed_{seed}",
                    video_fps=args.video_fps,
                )
                recorder.capture_initial()
            terminated = False
            truncated = False
            steps = 0
            episode_return = 0.0
            min_target_distance = float(initial_info.get("target_distance", np.inf))
            info = dict(initial_info)
            try:
                while not terminated and not truncated:
                    action = policy.action()
                    _, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    steps += 1
                    min_target_distance = min(
                        min_target_distance,
                        float(np.linalg.norm(env.target_position() - env.hand_position)),
                    )
                    if recorder is not None:
                        recorder.record_step(action, reward, terminated, truncated, info)
            except BaseException:
                if recorder is not None:
                    recorder.close()
                raise
            info = env.info()
            info["base_success"] = env.ever_success
            info["success"] = bool(terminated)
            policy.result = "success" if terminated else (
                "failed_motion_boundary"
                if info.get("termination_reason") == "rigid_motion_boundary_crossed"
                else "failed_timeout"
            )
            row = base_row(
                "diffusion_policy_lowdim", trial_offset + 1, seed, scenario,
                initial_info, info, env.grasp_break_history,
            )
            row.update({
                "steps": steps,
                "sim_time": float(env.data.time),
                "episode_return": episode_return,
                "min_target_distance": min_target_distance,
                "policy_result": policy.result,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "diffusion_policy_model": str(args.model.expanduser().resolve()),
                **policy.policy_info(),
            })
            if recorder is not None:
                row.update(recorder.finish(row).relative_to(run_dir))
            else:
                row.update({
                    "episode_dir": None, "trajectory": None, "metadata": None,
                    "global_video": None, "wrist_video": None,
                })
            rows.append(row)
            print(
                f"trial={trial_offset + 1}/{args.trials} seed={seed} result={policy.result} "
                f"steps={steps} diffusion_inferences={policy.policy_info()['diffusion_inference_count']}",
                flush=True,
            )
    finally:
        env.close()

    episodes_path = run_dir / "episodes.csv"
    summary_path = run_dir / "summary.json"
    write_csv(episodes_path, rows)
    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "method": "diffusion_policy_lowdim",
        "checkpoint": str(args.model.expanduser().resolve()),
        "checkpoint_sha256": sha256_file(args.model),
        "scenario": scenario.asdict(),
        "seeds": [args.seed + index for index in range(args.trials)],
        "recording": bool(args.record_video),
        "camera_input": {
            "keys": ["opst_cam", "wrist_cam"],
            "resolution": [480, 360],
            "fps": args.video_fps,
        } if args.record_video else None,
        "state_input": "ee_xyz_euler_gripper + 16 DLO keypoints relative to EE + relative target",
        "action": "absolute_xyz_euler_xyz_gripper_through_dynamicvla_ik",
        "episodes_csv": str(episodes_path.relative_to(run_dir)),
        "summary": str(summary_path.relative_to(run_dir)),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"lowdim_eval_output={run_dir}", flush=True)
    return run_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
