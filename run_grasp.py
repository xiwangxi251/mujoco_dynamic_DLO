"""以无界面模式或 MuJoCo 被动查看器运行同一个动态线缆任务。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

from cable_grasp_env import CableGraspEnv, EnvConfig, XML_PATH
from dynamic_grasp_policy import DynamicCableGraspPolicy
from experiment_scenarios import get_scenario, list_scenario_names


def env_config_from_args(args: argparse.Namespace) -> EnvConfig:
    if args.scenario is None:
        return EnvConfig(
            seed=args.seed,
            episode_seconds=args.episode_seconds,
            disturbance_strength=args.disturbance,
        )
    scenario = get_scenario(args.scenario)
    return EnvConfig(
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        scenario_name=scenario.name,
        scenario_id=scenario.scenario_id,
        scenario_split=scenario.split.value,
        **scenario.to_env_overrides(),
    )


def print_trial_start(env: CableGraspEnv) -> None:
    index = env.cable_ids.index(env.target_body_id)
    print(
        f"trial={env.trial_index} started target_segment={index}/{len(env.cable_ids) - 1} "
        f"target_body={env.target_body_id}",
        flush=True,
    )


def run_headless(args: argparse.Namespace) -> None:
    import cv2
    import mujoco
    import numpy as np

    """快速验证；使用与GUI模式完全相同的环境和策略。"""
    # 无界面模式不等待墙钟时间，因此适合批量统计成功率；物理与GUI模式完全相同。
    env = CableGraspEnv(env_config_from_args(args))
    policy = DynamicCableGraspPolicy(env)
    successes = 0

    # 每次运行使用独立子目录，避免覆盖之前各回合的视频。
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    video_dir = args.video_dir / run_name
    suffix = 1
    while video_dir.exists():
        video_dir = args.video_dir / f"{run_name}_{suffix:02d}"
        suffix += 1
    video_dir.mkdir(parents=True)

    # MuJoCo默认离屏 framebuffer 只有640x480；按请求尺寸自动扩展后再创建渲染器。
    # 这只修改当前进程中的模型，不改变物理参数，也不要求用户手工编辑XML。
    env.model.vis.global_.offwidth = max(
        int(env.model.vis.global_.offwidth), args.video_width
    )
    env.model.vis.global_.offheight = max(
        int(env.model.vis.global_.offheight), args.video_height
    )
    renderer = mujoco.Renderer(
        env.model, height=args.video_height, width=args.video_width
    )
    model_path = video_dir / "model.mjb"
    mujoco.mj_saveModel(env.model, str(model_path), None)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state_size = mujoco.mj_stateSize(env.model, state_spec)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.55, 0.0, 0.30]
    camera.distance = 1.65
    camera.azimuth = 135
    camera.elevation = -25

    def write_frame(writer: cv2.VideoWriter, recorded_states: list[np.ndarray]) -> None:
        """渲染当前物理状态，并写入本回合的视频。"""
        state = np.empty(state_size, dtype=np.float64)
        mujoco.mj_getState(env.model, env.data, state, state_spec)
        recorded_states.append(state)
        renderer.update_scene(env.data, camera=camera)
        rgb = renderer.render()
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print(f"model={XML_PATH}", flush=True)
    print(f"headless_videos={video_dir.resolve()}", flush=True)
    for trial in range(args.trials):
        episode_seed = args.seed + trial
        env.reset(seed=episode_seed)
        policy.reset()
        print_trial_start(env)
        video_path = video_dir / f"trial_{env.trial_index:03d}.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.video_fps,
            (args.video_width, args.video_height),
        )
        if not writer.isOpened():
            renderer.close()
            raise RuntimeError(f"无法创建视频文件: {video_path}")
        previous_phase = policy.phase
        next_frame_time = 0.0
        recorded_states: list[np.ndarray] = []
        # 保存重置后的初始画面；之后按仿真时间而不是计算耗时采样。
        write_frame(writer, recorded_states)
        next_frame_time += 1.0 / args.video_fps
        while not policy.finished and env.data.time < env.config.episode_seconds:
            # 标准交互循环：策略产生动作 -> 环境执行动作并推进物理。
            action = policy.action()
            _, _, _, truncated, _ = env.step(action)
            if env.data.time + 1e-9 >= next_frame_time:
                write_frame(writer, recorded_states)
                next_frame_time += 1.0 / args.video_fps
            if policy.phase is not previous_phase:
                tracking_error = ((env.hand_position - policy.last_desired) ** 2).sum() ** 0.5
                print(
                    f"  t={env.data.time:6.3f}s phase={previous_phase.name}->{policy.phase.name} "
                    f"tracking_error={tracking_error:.3f}m contacts={len(env.finger_contacts())} "
                    f"hand={env.hand_position.round(3)} desired={policy.last_desired.round(3)}",
                    flush=True,
                )
                break_diagnostics = policy.break_diagnostics_text()
                if break_diagnostics is not None and policy.phase.name == "FAILURE_OBSERVE":
                    print(f"    {break_diagnostics}", flush=True)
                previous_phase = policy.phase
            if truncated:
                policy.result = "failed_timeout"
                policy.finished = True
        writer.release()
        states_path = video_dir / f"trial_{env.trial_index:03d}_states.npz"
        np.savez_compressed(
            states_path,
            states=np.stack(recorded_states),
            state_spec=np.int64(int(state_spec)),
            frame_times=np.asarray([state[0] for state in recorded_states]),
            fps=np.float64(args.video_fps),
            width=np.int64(args.video_width),
            height=np.int64(args.video_height),
            model_file=np.asarray(model_path.name),
            source_xml=np.asarray(str(XML_PATH.resolve())),
            mujoco_version=np.asarray(mujoco.__version__),
            trial=np.int64(env.trial_index),
            seed=np.int64(episode_seed),
            scenario_name=np.asarray(env.config.scenario_name),
            scenario_id=np.asarray(env.config.scenario_id or ""),
            motion_profile_hash=np.asarray(env.motion_profile_hash),
            result=np.asarray(policy.result),
        )
        successes += int(policy.result == "success")
        print(policy.summary(), flush=True)
        print(f"  video={video_path.resolve()}", flush=True)
        print(f"  states={states_path.resolve()}", flush=True)

    renderer.close()
    print(f"completed={args.trials} successes={successes} rate={successes / args.trials:.1%}")


def run_viewer(args: argparse.Namespace) -> None:
    """实时运行：查看器显示期间持续生成新的仿真步。"""
    # GUI不是预先计算后的回放：窗口存在时才持续生成新的物理步。
    from mujoco import viewer

    env = CableGraspEnv(env_config_from_args(args))
    env.reset(seed=args.seed)
    policy = DynamicCableGraspPolicy(env)

    controls = {
        # 这些变量只控制显示调度和重置，不参与环境成功判定。
        "reset": False,
        "paused": False,
        "speed": float(args.speed),
        "speed_changed": False,
    }

    def key_callback(keycode: int) -> None:
        # GLFW按键码：Backspace=259；字母回调可能收到大写或小写编码。
        # Backspace/R/N重置，空格暂停，=/]加速，-/[减速。
        if keycode in (259, ord("R"), ord("r"), ord("N"), ord("n")):
            controls["reset"] = True
        elif keycode == ord(" "):
            controls["paused"] = not controls["paused"]
            print("paused" if controls["paused"] else "running", flush=True)
        elif keycode in (ord("="), ord("]")):
            controls["speed"] = min(8.0, controls["speed"] * 2.0)
            controls["speed_changed"] = True
            print(f"viewer speed={controls['speed']:.2f}x", flush=True)
        elif keycode in (ord("-"), ord("[")):
            controls["speed"] = max(0.25, controls["speed"] / 2.0)
            controls["speed_changed"] = True
            print(f"viewer speed={controls['speed']:.2f}x", flush=True)

    print(f"model={XML_PATH}", flush=True)
    print(
        "GUI controls: = or ] speed up, - or [ slow down, Space pause, "
        "Backspace/R reset, N new random trial",
        flush=True,
    )
    print_trial_start(env)

    # 这些时间锚点在查看器创建后才初始化。GLFW/OpenGL上下文启动可能花费数秒，
    # 不能把这段时间误算成首个可见画面需要一次性追赶的仿真积压。
    start_wall = 0.0
    wall_anchor = 0.0
    sim_anchor = float(env.data.time)
    previous_phase = policy.phase
    completed_trials = 0
    reset_after_done_wall: float | None = None
    trial_reported = False

    # 该调用之前没有隐藏预运行：窗口建立后第一轮才开始，终端报告的也是同一批实时物理步。
    with viewer.launch_passive(
        env.model,
        env.data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as handle:
        handle.cam.lookat[:] = [0.55, 0.0, 0.30]
        handle.cam.distance = 1.65
        handle.cam.azimuth = 135
        handle.cam.elevation = -25
        handle.sync()
        start_wall = time.perf_counter()
        wall_anchor = start_wall
        sim_anchor = float(env.data.time)

        while handle.is_running():
            frame_start = time.perf_counter()
            now = time.perf_counter()
            if args.duration > 0.0 and now - start_wall >= args.duration:
                break

            if controls["reset"]:
                # 将物理环境和策略作为一次操作同时重置。这里有意不修改 model.qpos0，
                # 因为它是运动学参考配置，不是用户重置书签。
                with handle.lock():
                    env.reset()
                    policy.reset()
                controls["reset"] = False
                reset_after_done_wall = None
                trial_reported = False
                previous_phase = policy.phase
                wall_anchor = now
                sim_anchor = float(env.data.time)
                print_trial_start(env)
                handle.sync()

            if controls["speed_changed"]:
                wall_anchor = now
                sim_anchor = float(env.data.time)
                controls["speed_changed"] = False

            if controls["paused"]:
                wall_anchor = now
                sim_anchor = float(env.data.time)
                handle.sync()
                time.sleep(1.0 / 60.0)
                continue

            # 根据墙钟时间计算这一帧应追到的仿真时间，从而实现0.25x到8x播放。
            target_sim_time = sim_anchor + controls["speed"] * (now - wall_anchor)
            # 限制断点或拖动窗口后的单帧追赶量；重新设置时间锚点可避免永久追赶积压，
            # 同时保留用户选择的播放倍速。
            # 最多只允许单帧追赶8个控制周期，避免拖动窗口或断点后GUI长时间假死。
            max_steps = 8
            steps = 0
            while env.data.time < target_sim_time and steps < max_steps:
                # 试次结束后停止推进物理，等待一秒后自动重置或保留最终画面。
                # 否则failed_timeout虽已打印，旧状态机仍会继续产生动作和转换日志。
                if policy.finished:
                    break
                # 同一GUI帧内可推进多个50 Hz动作，viewer仍按约60 FPS刷新。
                action = policy.action()
                _, _, _, truncated, _ = env.step(action)
                steps += 1

                if policy.phase is not previous_phase:
                    tracking_error = ((env.hand_position - policy.last_desired) ** 2).sum() ** 0.5
                    print(
                        f"  t={env.data.time:6.3f}s phase={previous_phase.name}->{policy.phase.name} "
                        f"tracking_error={tracking_error:.3f}m contacts={len(env.finger_contacts())} "
                        f"hand={env.hand_position.round(3)} desired={policy.last_desired.round(3)}",
                        flush=True,
                    )
                    break_diagnostics = policy.break_diagnostics_text()
                    if break_diagnostics is not None and policy.phase.name == "FAILURE_OBSERVE":
                        print(f"    {break_diagnostics}", flush=True)
                    previous_phase = policy.phase

                if truncated and not policy.finished:
                    policy.result = "failed_timeout"
                    policy.finished = True

                if policy.finished and not trial_reported:
                    completed_trials += 1
                    print(policy.summary(), flush=True)
                    trial_reported = True
                    reset_after_done_wall = now + 1.0
                    break

            if steps >= max_steps:
                wall_anchor = now
                sim_anchor = float(env.data.time)

            if reset_after_done_wall is not None and now >= reset_after_done_wall:
                if args.trials <= 0 or completed_trials < args.trials:
                    controls["reset"] = True
                else:
                    # 保留最后一轮画面；用户可以按N/R开始新随机试次，或直接关闭窗口。
                    reset_after_done_wall = None

            handle.sync()
            # 以稳定60 Hz渲染，而不是要求查看器每秒重建120次场景；
            # 物理积分仍为500 Hz，策略仍以50 Hz运行。
            remaining = 1.0 / 60.0 - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)

    print(
        f"viewer_closed wall_time={time.perf_counter() - start_wall:.3f}s "
        f"current_trial_sim_time={env.data.time:.3f}s",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数；默认扰动倍率1.5，默认每轮最多28仿真秒。"""

    parser = argparse.ArgumentParser(
        description="Reactive Panda grasping of a continuously deforming cable"
    )
    parser.add_argument("--headless", action="store_true",
                        help="run fast without opening the viewer")
    parser.add_argument("--trials", type=int, default=3,
                        help="number of trials; GUI value <=0 repeats forever")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="GUI wall-clock seconds; 0 waits until window close")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="initial GUI simulation-time multiplier (0.25 to 8)")
    parser.add_argument("--disturbance", type=float, default=1.5,
                        help="cable disturbance multiplier, independent of actions")
    parser.add_argument(
        "--scenario", choices=list_scenario_names(),
        help="run one frozen experiment scenario; overrides --disturbance",
    )
    parser.add_argument("--episode-seconds", type=float, default=28.0,
                        help="maximum simulated seconds per trial")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--video-dir", type=Path, default=Path("headless_videos"),
                        help="headless视频根目录；每次运行会建立独立子目录")
    parser.add_argument("--video-fps", type=float, default=25.0,
                        help="headless视频帧率")
    parser.add_argument("--video-width", type=int, default=960,
                        help="headless视频宽度")
    parser.add_argument("--video-height", type=int, default=540,
                        help="headless视频高度")
    args = parser.parse_args()
    args.speed = min(8.0, max(0.25, args.speed))
    if args.headless and args.trials <= 0:
        parser.error("--headless requires --trials greater than zero")
    if args.video_fps <= 0.0:
        parser.error("--video-fps must be greater than zero")
    if args.video_width <= 0 or args.video_height <= 0:
        parser.error("--video-width and --video-height must be greater than zero")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.headless:
        run_headless(arguments)
    else:
        run_viewer(arguments)
