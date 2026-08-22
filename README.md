# Panda 动态线缆抓取

这是一个基于 MuJoCo 的 Franka Panda 动态柔性线缆抓取项目，包含规则策略、
PPO 强化学习、特权专家数据采集和 DynamicVLA 接入。产品代码统一位于
`src/panda_cable_grasp/`，所有实验产物统一写入 `outputs/`。

## 快速开始

推荐使用 Python 3.10+。在仓库根目录安装：

```bash
python -m pip install -e ".[rl,dynamicvla,dev]"
```

运行一个无头规则策略实验：

```bash
panda-cable-grasp --headless --trials 1
```

运行测试：

```bash
python -m unittest discover -s tests -t . -p "test_*.py"
```

## 常用命令

规则策略与诊断：

```bash
panda-cable-grasp --headless --trials 5 --seed 20260804
panda-cable-benchmark --help
panda-cable-motion-diagnostics --help
```

PPO 训练与评估：

```bash
panda-cable-rl-train --workers 12 --eval-workers 4 --timesteps 2000000 --device cpu
panda-cable-rl-eval --model outputs/rl/train/ppo_dlo_baseline_v3/best_model.zip --headless
```

`--workers` 是并行 MuJoCo 环境进程数，不是 GPU 数。服务器上应从较小数值开始，
逐级测试 8、12、16……，根据总采样吞吐、内存和 CPU 利用率决定，而不是直接设为
CPU 线程总数。`--eval-workers` 独立控制严格评估的并行环境数，设为 1 时串行评估。

特权专家和 DynamicVLA：

```bash
panda-cable-expert-collect --help
panda-cable-dataset-convert --help
panda-cable-dataset-check --help
panda-cable-finetune --help
panda-cable-dynamicvla --help
```

这些命令由 `pyproject.toml` 注册。Python 代码应直接从
`panda_cable_grasp` 包导入；旧的根目录转发脚本和
`rl.*`、`privileged_expert.*`、`dynamicvla_finetune.*` 兼容入口已退役。

## 项目结构

```text
panda_cable_grasp/
├── assets/mujoco/               # MuJoCo XML 模型
├── configs/                     # 训练和集成配置
├── docs/                        # 当前文档、历史归档和参考结果
├── outputs/                     # 录像、模型、指标和数据集（不入 Git）
├── src/panda_cable_grasp/
│   ├── cli/                     # 用户命令入口
│   ├── env/                     # 物理环境与运动学
│   ├── policies/                # 规则策略
│   ├── scenarios/               # 固定实验场景注册表
│   ├── rl/                      # PPO 环境、训练、评估和指标
│   ├── evaluation/              # 基准测试、诊断和失败分类
│   ├── expert/                  # 特权专家
│   └── dynamicvla/              # DynamicVLA 适配与微调工具
├── tests/unit/                  # 快速、隔离的单元测试
├── tests/integration/           # MuJoCo/RL 集成测试
└── tools/                       # 安装检查、消融、录像和启动工具
```

根目录每个文件的用途见 [根目录说明](docs/root_layout.md)，完整模块关系见
[架构说明](docs/architecture.md)，当前可操作状态见
[当前状态](docs/current_status.md)。

## 输出目录

- `outputs/scripted/`：规则策略录像与状态。
- `outputs/benchmarks/`：基准实验 CSV、manifest 和模型快照。
- `outputs/rl/train/`：PPO checkpoint、TensorBoard 和评估结果。
- `outputs/rl/eval/`：PPO 评估录像与状态。
- `outputs/diagnostics/`：运动诊断。
- `outputs/datasets/`：专家数据和转换后的数据集。

旧目录到新目录的实际迁移记录保存在 `outputs/migration_manifest.json`。迁移过程没有
删除实验数据。

## 设计约束

- 成功由底层物理环境统一判定，训练封装不能降低任务门槛。
- PPO 观测不包含特权目标线段；特权信息只允许用于专家数据生成和诊断。
- 场景、随机种子、奖励分量和失败原因必须写入实验产物，保证结果可复现。
- 涉及线缆材料、碰撞尺寸、摩擦或成功阈值的改动，应作为独立实验处理。

## 文档索引

- [新会话入口](docs/start_here.md)
- [架构与代码导航](docs/architecture.md)
- [当前状态](docs/current_status.md)
- [RL 基线与奖励修正](docs/rl_baseline.md)
- [实验协议](docs/experiment_protocol.md)
- [服务器部署](docs/server_setup.md)
- [DynamicVLA 零样本接入](docs/dynamicvla_zero_shot.md)
- [DynamicVLA 微调](docs/dynamicvla_finetune.md)
- [特权专家](docs/privileged_expert.md)
- [受版本控制的参考结果](docs/reference_results/)
- [历史记录](docs/history/)
