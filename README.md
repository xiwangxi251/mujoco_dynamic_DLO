# MuJoCo 动态线缆抓取

这是一个基于 MuJoCo 的 Franka Panda / 松灵 NERO 动态柔性线缆抓取项目，包含规则策略、
PPO 强化学习、特权专家数据采集、DynamicVLA 和视觉 Diffusion Policy。产品代码统一位于
`src/panda_cable_grasp/`，所有实验产物统一写入 `outputs/`。默认机器人仍为 Panda；NERO
使用仓库内 `assets/mujoco/nero/` 的 AgileX 官方模型和网格资源。

## 快速开始

推荐使用 Python 3.10+。在仓库根目录安装：

```bash
python -m pip install -e ".[rl,dynamicvla,diffusion,dev]"
```

运行一个无头规则策略实验：

```bash
panda-cable-grasp --headless --trials 1
```

选择松灵 NERO：

```bash
panda-cable-grasp --headless --robot nero --trials 1
```

`panda-cable-benchmark`、`panda-cable-rl-train`、`panda-cable-rl-eval`、
`panda-cable-dynamicvla`、`panda-cable-diffusion-eval` 和运动诊断命令同样支持
`--robot nero`。已有 Panda checkpoint 不能直接视为 NERO checkpoint，NERO 训练或评估应
使用对应机器人采集/训练的模型。

运行测试：

```bash
python -m unittest discover -s tests -t . -p "test_*.py"
```

## 常用命令

规则策略与诊断：

```bash
panda-cable-grasp --headless --trials 5 --seed 20280804
panda-cable-benchmark --methods scripted expert --suite core --episodes 20 \
  --scenario-workers 2 --envs-per-scenario 4
panda-cable-replay --help
panda-cable-motion-diagnostics --help
```

PPO 训练与评估：

```bash
panda-cable-rl-train --workers 12 --eval-workers 4 --timesteps 2000000 --device cpu
panda-cable-rl-eval --model outputs/rl/train/ppo_dlo_baseline_v4/best_model.zip --headless
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

Diffusion Policy（输入与 DynamicVLA 相同：对侧/腕部两路 RGB 加末端位姿）：

```bash
panda-cable-diffusion-train outputs/datasets/privileged_expert/<run> \
  --output outputs/diffusion_policy/train/cable_dp --epochs 2000 --device cuda
panda-cable-diffusion-eval \
  --model outputs/diffusion_policy/train/cable_dp/checkpoint_best.pt \
  --scenario id_static --trials 20 --headless --device cuda
```

该实现使用专家已执行关节目标的前向运动学生成 `[xyz, quat_wxyz, gripper]`
任务空间标签，预测 16 步动作块，每次执行 8 步后重规划；执行端复用
DynamicVLA 的任务空间 IK 和安全限幅。详细说明见
[`docs/diffusion_policy.md`](docs/diffusion_policy.md)。

并行采集时，`--workers` 控制并行场景数，`--envs-per-scenario` 控制每个场景
内部的独立 MuJoCo 环境数。采集器会实时显示 attempts/h、successes/h 和 ETA，并可用
`--resume --run-name <名称>` 续跑中断任务。当前实验状态和各方法的实验文档入口见
  [项目状态](docs/project_status.md)。

这些命令由 `pyproject.toml` 注册。Python 代码应直接从
`panda_cable_grasp` 包导入；旧的根目录转发脚本和
`rl.*`、`privileged_expert.*`、`dynamicvla_finetune.*` 兼容入口已退役。

## 项目结构

```text
panda_cable_grasp/
├── assets/mujoco/               # MuJoCo XML 模型
├── configs/                     # 训练和集成配置
├── docs/                        # 当前文档、实验索引和运行说明
├── outputs/                     # 录像、模型、指标和数据集（不入 Git）
├── src/panda_cable_grasp/
│   ├── cli/                     # 用户命令入口
│   ├── env/                     # 物理环境与运动学
│   ├── policies/                # 规则策略
│   ├── scenarios/               # 固定实验场景注册表
│   ├── rl/                      # PPO 环境、训练、评估和指标
│   ├── evaluation/              # 基准测试、诊断和失败分类
│   ├── expert/                  # 特权专家
│   ├── dynamicvla/              # DynamicVLA 适配与微调工具
│   └── diffusion_policy/        # 视觉 Diffusion Policy、数据集与训练
├── tests/unit/                  # 快速、隔离的单元测试
├── tests/integration/           # MuJoCo/RL 集成测试
└── tools/                       # 安装检查、消融、录像和启动工具
```

完整模块关系见[架构说明](docs/reference/architecture.md)，当前目标、路线、进度和问题见
[项目状态](docs/project_status.md)。

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

- [项目目标、整体规划、进度和问题](docs/project_status.md)
- [环境端进度](docs/environment.md)
- [Scripted 与 Expert 实验记录](docs/scripted_policy_experiments.md)
- [RL 实验记录](docs/rl_experiments.md)
- [DynamicVLA 实验记录](docs/dynamicvla_experiments.md)

架构、实验协议、统一评测和服务器部署等辅助说明位于 `docs/reference/`；实验索引和专项诊断
位于 `docs/experiments/`。

文档分工：README 只负责安装和导航；`project_status.md` 维护当前事实；
`experiment_protocol.md` 维护冻结规则；方法实验文档保留必要的过程和结果。过期内容通过
Git 历史追溯，不在当前文档树中重复保存。
