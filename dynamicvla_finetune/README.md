# DynamicVLA 线缆数据微调

本目录将 `privileged_expert.collect_dataset` 保存的 schema-v2 成功轨迹转换为
DynamicVLA 官方训练代码使用的 LeRobot v2.1 数据集，并调用官方 `run.py` 从
`dynamic-vla-DOM` 权重开始微调。

## 数据语义

- 只读取同时包含 `opst_cam`、`wrist_cam` 的 schema-v2 轨迹。
- 采集数据为 50 Hz，转换时根据 `times` 下采样为 DynamicVLA 使用的 25 Hz。
- `observation.state` 为 6 维绝对末端位姿 `xyz + Euler(xyz)`。
- 采集器的动作是 7 个关节目标和夹爪命令。转换器使用每个场景的
  `scenario.mjb` 做前向运动学，得到 7 维绝对动作
  `xyz + Euler(xyz) + gripper(-1/+1)`。
- 默认使用 `requested_actions`，因为官方 DOM 标签也是控制器接收的期望末端
  位姿，而不是物理系统一步后实际达到的位姿。可用
  `--joint-target-source applied` 做消融。
- 数据集 task 保存为 DynamicVLA 官方 `InstructionGenerator` 能识别的
  `{"task": "pick", "objects": ["blue cable"]}`，训练时会自动生成不同的抓取措辞。
- 线缆、目标节点等特权字段不写入学生模型输入。

## 1. 安装转换依赖

在 DynamicVLA 的 Python 环境中执行：

```bash
cd /path/to/DynamicVLA
python -m pip install -r requirements.txt
python -m pip install mujoco torchcodec
```

`torchcodec` 必须与服务器的 PyTorch/CUDA 版本匹配。DynamicVLA 官方训练加载器直接
导入它，但当前官方 `requirements.txt` 未列出该依赖。

## 2. 转换数据

在 `panda_cable_grasp` 根目录执行。建议明确指定一次采集生成的 run 目录，不要
直接传整个 `privileged_expert_dataset`，否则不同批次使用相同 seed 时会被判定为重复。

```bash
python -m dynamicvla_finetune.convert_dataset \
  benchmark_runs/privileged_expert_dataset/run_YYYYMMDD_HHMMSS_seed20260804 \
  --output /data1/hxai/datasets/panda_cable_dynamicvla \
  --repo-id local/panda-cable-dynamicvla \
  --target-fps 25
```

只转换静态场景：

```bash
python -m dynamicvla_finetune.convert_dataset \
  benchmark_runs/privileged_expert_dataset/run_YYYYMMDD_HHMMSS_seed20260804 \
  --scenarios id_static \
  --output /data1/hxai/datasets/panda_cable_static_dynamicvla \
  --repo-id local/panda-cable-static-dynamicvla
```

转换器拒绝覆盖已有输出。确实要覆盖时显式增加 `--overwrite`。多个输入中出现相同
`(scenario, seed)` 时默认报错；可显式选择 `--duplicate-policy skip`。
run 根目录缺少最终 `manifest.json` 或其中存在未完成场景时也会拒绝转换，防止读取
仍在写入的数据。确实要使用部分结果时，应直接传入已经停止写入的场景子目录；也可
显式增加 `--allow-incomplete-run`。

转换完成后，先让 DynamicVLA 自己的加载器检查数据、双帧观测和20步动作块：

```bash
python -m dynamicvla_finetune.check_dataset \
  --dynamicvla-root /data1/hxai/mujoco/DynamicVLA \
  --dataset /data1/hxai/datasets/panda_cable_static_dynamicvla
```

## 3. 微调

官方训练实现依赖 NCCL、`os.sched_setaffinity` 和 Linux `libcudart.so`，应在 Linux
GPU 服务器运行：

```bash
python -m dynamicvla_finetune.launch_finetune \
  --dynamicvla-root /data1/hxai/mujoco/DynamicVLA \
  --dataset /data1/hxai/datasets/panda_cable_static_dynamicvla \
  --checkpoint /data1/hxai/mujoco/DynamicVLA/ckt/dynamic-vla-DOM \
  --experiment cable-static-50 \
  --gpus 0 \
  --epochs 30 \
  --batch-size 4
```

多 GPU 示例：

```bash
python -m dynamicvla_finetune.launch_finetune \
  --dynamicvla-root /data1/hxai/mujoco/DynamicVLA \
  --dataset /data1/hxai/datasets/panda_cable_dynamicvla \
  --experiment cable-four-scenarios \
  --gpus 0,1,2,3 \
  --batch-size 8
```

脚本会生成 resolved YAML，再通过 `torch.distributed.run` 启动官方 `run.py`。第一次
建议增加 `--dry-run` 检查路径和最终命令。checkpoint 默认使用
`<dynamicvla-root>/ckt/dynamic-vla-DOM`。

50 条静态成功轨迹属于小数据微调，默认冻结视觉模型、connector 和文本模型，学习率
为 `1e-5`，训练 30 epoch。先根据验证损失和闭环成功率选择 checkpoint，不建议只看
训练损失。
