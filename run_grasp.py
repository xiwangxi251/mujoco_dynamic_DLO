"""以无界面模式或 MuJoCo 被动查看器运行同一个动态线缆任务。"""

from __future__ import annotations

import argparse
import time

from cable_grasp_env import CableGraspEnv, EnvConfig, XML_PATH
from dynamic_grasp_policy import DynamicCableGraspPolicy


def print_trial_start(env: CableGraspEnv) -> None:
    index = env.cable_ids.index(env.target_body_id)
    print(
        f"trial={env.trial_index} started target_segment={index}/{len(env.cable_ids) - 1} "
        f"target_body={env.target_body_id}",
        flush=True,
    )


def run_headless(args: argparse.Namespace) -> None:
    """快速验证；使用与GUI模式完全相同的环境和策略。"""
    # 无界面模式不等待墙钟时间，因此适合批量统计成功率；物理与GUI模式完全相同。
    env = CableGraspEnv(EnvConfig(
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        disturbance_strength=args.disturbance,
    ))
    policy = DynamicCableGraspPolicy(env)
    successes = 0

    print(f"model={XML_PATH}", flush=True)
    for trial in range(args.trials):
        if trial:
            env.reset()
            policy.reset()
        print_trial_start(env)
        previous_phase = policy.phase
        while not policy.finished and env.data.time < env.config.episode_seconds:
            # 标准交互循环：策略产生动作 -> 环境执行动作并推进物理。
            action = policy.action()
            _, _, _, truncated, _ = env.step(action)
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
        successes += int(policy.result == "success")
        print(policy.summary(), flush=True)

    print(f"completed={args.trials} successes={successes} rate={successes / args.trials:.1%}")


def run_viewer(args: argparse.Namespace) -> None:
    """实时运行：查看器显示期间持续生成新的仿真步。"""
    # GUI不是预先计算后的回放：窗口存在时才持续生成新的物理步。
    from mujoco import viewer

    env = CableGraspEnv(EnvConfig(
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        disturbance_strength=args.disturbance,
    ))
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
    parser.add_argument("--episode-seconds", type=float, default=28.0,
                        help="maximum simulated seconds per trial")
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    args.speed = min(8.0, max(0.25, args.speed))
    if args.headless and args.trials <= 0:
        parser.error("--headless requires --trials greater than zero")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.headless:
        run_headless(arguments)
    else:
        run_viewer(arguments)
