"""把无头仿真保存的完整 MuJoCo 状态从任意相机角度重新渲染。"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from panda_cable_grasp.runtime import configure_mujoco_runtime

configure_mujoco_runtime()

import cv2
import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线重新渲染MuJoCo状态记录")
    parser.add_argument("--states", type=Path, required=True,
                        help="trial_XXX_states.npz状态文件")
    parser.add_argument("--model", type=Path, default=None,
                        help="对应的model.mjb；默认读取状态文件旁的模型")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出MP4；默认在状态文件旁生成*_rerender.mp4")
    parser.add_argument(
        "--camera", choices=("opst", "wrist"), default="opst",
        help="使用模型中环境定义的 DynamicVLA 命名相机",
    )
    parser.add_argument("--fps", type=float, default=None,
                        help="输出帧率；默认沿用记录帧率")
    parser.add_argument("--width", type=int, default=None,
                        help="输出宽度；默认沿用记录宽度")
    parser.add_argument("--height", type=int, default=None,
                        help="输出高度；默认沿用记录高度")
    args = parser.parse_args()
    if not args.states.is_file():
        parser.error(f"状态文件不存在: {args.states}")
    return args


def main(args: argparse.Namespace) -> None:
    with np.load(args.states, allow_pickle=False) as recording:
        states = recording["states"].copy()
        state_spec = int(recording["state_spec"])
        recorded_fps = float(recording["fps"])
        recorded_width = int(recording["width"])
        recorded_height = int(recording["height"])
        saved_model_name = str(recording["model_file"])

    model_path = args.model or args.states.parent / saved_model_name
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到对应的MuJoCo模型: {model_path}")
    output_path = args.output or args.states.with_name(
        args.states.stem.removesuffix("_states")
        + f"_{args.camera}_rerender.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = recorded_fps if args.fps is None else args.fps
    width = recorded_width if args.width is None else args.width
    height = recorded_height if args.height is None else args.height
    if fps <= 0.0 or width <= 0 or height <= 0:
        raise ValueError("fps、width和height必须大于0")

    model = mujoco.MjModel.from_binary_path(str(model_path))
    camera_name = (
        "dynamicvla_opst_camera"
        if args.camera == "opst"
        else "dynamicvla_wrist_camera"
    )
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
        raise ValueError(
            f"模型不包含环境标准相机 {camera_name!r}; "
            "旧录像请使用保存时附带的旧工具版本"
        )
    expected_state_size = mujoco.mj_stateSize(model, state_spec)
    if states.ndim != 2 or states.shape[1] != expected_state_size:
        raise ValueError(
            f"状态维度{states.shape}与模型要求的(*, {expected_state_size})不一致"
        )
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"无法创建视频文件: {output_path}")
    try:
        for state in states:
            mujoco.mj_setState(model, data, state, state_spec)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera_name)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.close()

    print(
        f"rendered={output_path.resolve()} frames={len(states)} "
        f"fps={fps:g} size={width}x{height} camera={camera_name}",
        flush=True,
    )


if __name__ == "__main__":
    main(parse_args())
