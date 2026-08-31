"""Run a trained visual Diffusion Policy on the MuJoCo cable task."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from ..paths import output_path


def parse_args() -> argparse.Namespace:
    from ..scenarios.registry import list_scenario_names

    parser = argparse.ArgumentParser(
        description="Evaluate a visual Diffusion Policy on the Panda cable task"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario", choices=list_scenario_names(), required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True,
        help="use a seedable diffusion noise generator for repeatable evaluation",
    )
    parser.add_argument("--video-dir", type=Path, default=output_path("diffusion_policy", "evaluation"))
    parser.add_argument("--run-name")
    parser.add_argument(
        "--headless", action="store_true",
        help="accepted for CLI compatibility; evaluation is always offscreen",
    )
    parser.add_argument("--no-recording", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Diffusion Policy checkpoint not found: {args.model}")
    if args.trials < 1 or args.episode_seconds <= 0.0 or args.video_fps <= 0.0:
        parser.error("trials, episode duration, and video FPS must be positive")
    return args


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import mujoco

    from ..diffusion_policy.runner import DiffusionPolicyRunner
    from ..env.environment import CableGraspEnv, EnvConfig, PANDA_XML_PATH, XML_PATH, resolve_menagerie_panda_dir
    from ..evaluation.benchmark import base_row, sha256_file, summarize, write_csv
    from ..evaluation.defaults import DEFAULT_VIDEO_FPS
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
            "dynamicvla_cameras_enabled": True,
        }
    )
    env = CableGraspEnv(config)
    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    run_dir = args.video_dir.expanduser().resolve() / run_name
    suffix = 1
    while run_dir.exists():
        if args.run_name:
            raise FileExistsError(f"refusing to overwrite {run_dir}")
        run_dir = args.video_dir.expanduser().resolve() / f"{run_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    model_dir = run_dir / "models"
    model_dir.mkdir()
    compiled_model = model_dir / f"{scenario.name}.mjb"
    mujoco.mj_saveModel(env.model, str(compiled_model), None)
    rows: list[dict] = []
    recording = not args.no_recording
    try:
        policy = DiffusionPolicyRunner(
            env, args.model, device=args.device, deterministic=args.deterministic
        )
        for trial_offset in range(args.trials):
            seed = args.seed + trial_offset
            _, initial_info = env.reset(seed=seed)
            policy.reset(seed=seed)
            recorder = None
            if recording:
                recorder = EpisodeRecorder(
                    env,
                    run_dir / "episodes" / scenario.name / f"seed_{seed}",
                    video_fps=args.video_fps or DEFAULT_VIDEO_FPS,
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
                "diffusion_policy", trial_offset + 1, seed, scenario,
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
                artifacts = recorder.finish(row)
                row.update(artifacts.relative_to(run_dir))
            else:
                row.update({
                    "episode_dir": None, "trajectory": None, "metadata": None,
                    "global_video": None, "wrist_video": None,
                })
            rows.append(row)
            print(
                f"trial={trial_offset + 1} seed={seed} result={policy.result} "
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
        "method": "diffusion_policy",
        "checkpoint": str(args.model.expanduser().resolve()),
        "checkpoint_sha256": sha256_file(args.model),
        "scenario": scenario.asdict(),
        "seeds": [args.seed + index for index in range(args.trials)],
        "recording": recording,
        "camera_input": {
            "keys": ["observation.images.opst_cam", "observation.images.wrist_cam"],
            "resolution": [480, 360],
            "state": "observation.state.end_effector.pos + euler_xyz",
        },
        "action": "absolute_xyz_euler_xyz_gripper_through_dynamicvla_ik",
        "compiled_model": str(compiled_model.resolve()),
        "compiled_model_sha256": sha256_file(compiled_model),
        "source_xml": str(XML_PATH.resolve()),
        "source_xml_sha256": sha256_file(XML_PATH),
        "panda_xml": str(PANDA_XML_PATH.resolve()),
        "panda_xml_sha256": sha256_file(PANDA_XML_PATH),
        "menagerie_panda_assets": str(resolve_menagerie_panda_dir()),
        "summary": summary,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"diffusion_policy_output={run_dir}", flush=True)
    return run_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
