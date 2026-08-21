"""交互式回放无头仿真状态，并用鼠标选定的相机重新导出视频。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

from runtime_config import configure_mujoco_runtime

configure_mujoco_runtime()

import cv2
import mujoco
from mujoco import viewer
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="交互式MuJoCo状态回放与视频导出")
    parser.add_argument("--states", type=Path, required=True,
                        help="无头模式生成的trial_XXX_states.npz")
    parser.add_argument("--model", type=Path, default=None,
                        help="对应model.mjb；默认读取状态文件旁的模型")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="按V导出视频的目录；默认与状态文件相同")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="初始回放速度")
    parser.add_argument("--loop", action="store_true",
                        help="到末尾后自动循环播放")
    parser.add_argument("--export-fps", type=float, default=None)
    parser.add_argument("--export-width", type=int, default=None)
    parser.add_argument("--export-height", type=int, default=None)
    parser.add_argument("--check", action="store_true",
                        help="只检查模型和状态是否匹配，不打开窗口")
    args = parser.parse_args()
    if not args.states.is_file():
        parser.error(f"状态文件不存在: {args.states}")
    if args.speed <= 0.0:
        parser.error("--speed必须大于0")
    return args


def load_recording(args: argparse.Namespace):
    with np.load(args.states, allow_pickle=False) as recording:
        states = recording["states"].copy()
        state_spec = int(recording["state_spec"])
        fps = float(recording["fps"])
        width = int(recording["width"])
        height = int(recording["height"])
        model_name = str(recording["model_file"])

    model_path = args.model or args.states.parent / model_name
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到对应的MuJoCo模型: {model_path}")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    expected = mujoco.mj_stateSize(model, state_spec)
    if states.ndim != 2 or states.shape[1] != expected:
        raise ValueError(f"状态维度{states.shape}与模型要求的(*, {expected})不一致")
    return model, states, state_spec, fps, width, height, model_path


def copy_camera(source: mujoco.MjvCamera) -> mujoco.MjvCamera:
    """复制查看器当前相机，导出期间继续使用这个固定视角。"""
    result = mujoco.MjvCamera()
    result.type = source.type
    result.fixedcamid = source.fixedcamid
    result.trackbodyid = source.trackbodyid
    result.lookat[:] = source.lookat
    result.distance = source.distance
    result.azimuth = source.azimuth
    result.elevation = source.elevation
    result.orthographic = source.orthographic
    return result


def export_video(
    model: mujoco.MjModel,
    states: np.ndarray,
    state_spec: int,
    camera: mujoco.MjvCamera,
    output: Path,
    fps: float,
    width: int,
    height: int,
) -> None:
    """用交互窗口中锁定的相机离线绘制全部记录帧。"""
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"无法创建视频文件: {output}")
    try:
        for state in states:
            mujoco.mj_setState(model, data, state, state_spec)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.close()


def run_viewer(args: argparse.Namespace) -> None:
    model, states, state_spec, recorded_fps, recorded_width, recorded_height, model_path = (
        load_recording(args)
    )
    export_fps = recorded_fps if args.export_fps is None else args.export_fps
    export_width = recorded_width if args.export_width is None else args.export_width
    export_height = recorded_height if args.export_height is None else args.export_height
    if export_fps <= 0.0 or export_width <= 0 or export_height <= 0:
        raise ValueError("导出fps、width和height必须大于0")
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), export_width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), export_height)
    output_dir = args.output_dir or args.states.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, states[0], state_spec)
    mujoco.mj_forward(model, data)

    controls = {
        "playing": True,
        "restart": False,
        "step": 0,
        "export": False,
        "speed": float(args.speed),
    }

    def key_callback(keycode: int) -> None:
        if keycode == ord(" "):
            controls["playing"] = not controls["playing"]
            print("replay_running" if controls["playing"] else "replay_paused", flush=True)
        elif keycode in (ord("R"), ord("r")):
            controls["restart"] = True
        elif keycode in (ord("V"), ord("v")):
            controls["export"] = True
        elif keycode == 262:  # Right arrow
            controls["step"] = 1
        elif keycode == 263:  # Left arrow
            controls["step"] = -1
        elif keycode in (ord("]"), ord("=")):
            controls["speed"] = min(8.0, controls["speed"] * 2.0)
            print(f"replay_speed={controls['speed']:.2f}x", flush=True)
        elif keycode in (ord("["), ord("-")):
            controls["speed"] = max(0.25, controls["speed"] / 2.0)
            print(f"replay_speed={controls['speed']:.2f}x", flush=True)

    print(f"states={args.states.resolve()}", flush=True)
    print(f"model={model_path.resolve()} frames={len(states)} fps={recorded_fps:g}", flush=True)
    print(
        "Replay controls: mouse drag/scroll changes camera; Space pause/play; "
        "Left/Right seek; R restart; [/- slower; ]/= faster; V export current camera",
        flush=True,
    )

    frame_index = 0
    accumulator = 0.0
    previous_wall = time.perf_counter()
    with viewer.launch_passive(
        model, data, key_callback=key_callback,
        show_left_ui=False, show_right_ui=False,
    ) as handle:
        handle.cam.lookat[:] = [0.55, 0.0, 0.30]
        handle.cam.distance = 1.65
        handle.cam.azimuth = 135
        handle.cam.elevation = -25
        handle.sync()

        while handle.is_running():
            frame_start = time.perf_counter()
            elapsed = min(frame_start - previous_wall, 0.1)
            previous_wall = frame_start

            if controls["restart"]:
                frame_index = 0
                accumulator = 0.0
                controls["playing"] = True
                controls["restart"] = False

            if controls["step"]:
                controls["playing"] = False
                frame_index = int(np.clip(
                    frame_index + controls["step"], 0, len(states) - 1
                ))
                controls["step"] = 0

            if controls["playing"]:
                accumulator += elapsed * recorded_fps * controls["speed"]
                advance = int(accumulator)
                if advance:
                    accumulator -= advance
                    frame_index += advance
                    if frame_index >= len(states):
                        if args.loop:
                            frame_index %= len(states)
                        else:
                            frame_index = len(states) - 1
                            controls["playing"] = False
                            print("replay_finished; drag camera then press V to export", flush=True)

            with handle.lock():
                mujoco.mj_setState(model, data, states[frame_index], state_spec)
                mujoco.mj_forward(model, data)
            handle.sync()

            if controls["export"]:
                with handle.lock():
                    selected_camera = copy_camera(handle.cam)
                controls["export"] = False
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output = output_dir / (
                    f"{args.states.stem.removesuffix('_states')}_view_{timestamp}.mp4"
                )
                print(
                    f"exporting camera azimuth={selected_camera.azimuth:.2f} "
                    f"elevation={selected_camera.elevation:.2f} "
                    f"distance={selected_camera.distance:.3f} "
                    f"lookat={selected_camera.lookat.round(3)}",
                    flush=True,
                )
                export_video(
                    model, states, state_spec, selected_camera, output,
                    export_fps, export_width, export_height,
                )
                print(f"exported={output.resolve()}", flush=True)

            remaining = 1.0 / 60.0 - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.check:
        loaded = load_recording(arguments)
        print(
            f"recording_ok model={loaded[6].resolve()} "
            f"frames={len(loaded[1])} state_size={loaded[1].shape[1]}",
            flush=True,
        )
    else:
        run_viewer(arguments)
