# Panda 动态线缆抓取

这是一个基于 MuJoCo 的 Franka Panda 动态柔性线缆抓取项目，包含规则策略、PPO 强化学习、特权专家数据采集和 DynamicVLA 接入。当前代码采用 `src/` 包布局，所有实验产物统一写入 `outputs/`。

## 快速开始

推荐使用 Python 3.10+，在仓库根目录安装：

```bash
pip install -e ".[rl,dynamicvla,dev]"
```

先运行一个无头规则策略实验：

```bash
python run_grasp.py --headless --trials 1
```

运行测试：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## 常用命令

规则策略：

```bash
python run_grasp.py --headless --trials 5 --seed 20260804
python benchmark.py --help
python motion_diagnostics.py --help
```

PPO 训练与评估：

```bash
python -m rl.train_rl --workers 12 --timesteps 2000000 --device cpu
python -m rl.test_rl --model outputs/rl/train/ppo_dlo_baseline_v3/best_model.zip --headless
```

`--workers` 是并行 MuJoCo 环境进程数，不是 GPU 数。服务器上应逐级测试 8、12、16……，以总采样吞吐、内存和 CPU 利用率决定，而不是盲目设成 CPU 线程数。

特权专家和 DynamicVLA 微调：

```bash
python -m privileged_expert.collect_dataset --help
python -m dynamicvla_finetune.convert_dataset --help
python -m dynamicvla_finetune.launch_finetune --help
```

安装为可编辑包后，也可以使用 `pyproject.toml` 中定义的 `panda-cable-*` 命令。根目录脚本和 `rl.*`、`privileged_expert.*`、`dynamicvla_finetune.*` 模块是兼容入口，新代码应直接从 `panda_cable_grasp` 包导入。

## 项目结构

```text
panda_cable_grasp/
├── assets/mujoco/               # MuJoCo XML 模型
├── configs/                     # 训练和集成配置
├── docs/                        # 当前文档与历史归档
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
└── tools/                       # 安装检查、录像处理和消融工具
```

更完整的模块关系见 [架构说明](docs/architecture.md)，本轮 RL 奖励修正见 [RL 基线说明](docs/rl_baseline.md)，当前可操作状态见 [当前状态](docs/current_status.md)。

## 输出目录

- `outputs/scripted/`：规则策略录像与状态
- `outputs/benchmarks/`：基准实验 CSV、manifest 和模型快照
- `outputs/rl/train/`：PPO checkpoint、TensorBoard 和评估结果
- `outputs/rl/eval/`：PPO 评估录像与状态
- `outputs/diagnostics/`：运动诊断
- `outputs/datasets/`：专家数据和转换后的数据集

旧目录到新目录的实际迁移记录保存在 `outputs/migration_manifest.json`。迁移过程中没有删除实验数据。

## 设计约束

- 成功由底层物理环境统一判定，训练封装不能降低任务门槛。
- PPO 观测不包含特权目标线段；特权信息只允许用于专家数据生成和诊断。
- 场景、随机种子、奖励分量和失败原因必须写入实验产物，保证结果可复现。
- 任何涉及线缆材料、碰撞尺寸、摩擦或成功阈值的改动，都应作为单独实验处理。

## 文档索引

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
