# 架构与代码导航

## 一条 episode 如何运行

```text
CLI / train / evaluate
        │
        ├── scenarios.registry 选择冻结场景与随机种子
        │
        ├── policy 或 PPO 产生动作
        │
        ▼
rl.environment（仅 PPO 使用：观测、动作变换、奖励）
        │
        ▼
env.environment（MuJoCo 状态、接触、夹持、扰动、成功判定）
        │
        ├── evaluation 记录指标和失败分类
        └── outputs 写录像、状态、CSV、模型与 manifest
```

底层 `CableGraspEnv` 是物理事实的唯一来源。PPO 封装可以塑形奖励，但不能覆盖接触、滑脱或严格成功判定。

## 目录职责

| 目录 | 职责 | 主要入口 |
|---|---|---|
| `src/panda_cable_grasp/env/` | MuJoCo 模型、状态推进、接触与成功语义；纯运动学工具 | `CableGraspEnv`, `EnvConfig` |
| `src/panda_cable_grasp/policies/` | 可解释的规则抓取控制器 | `DynamicGraspPolicy` |
| `src/panda_cable_grasp/scenarios/` | ID/OOD 场景、物理和运动参数注册表 | `get_scenario`, `list_scenario_names` |
| `src/panda_cable_grasp/rl/` | 99 维观测、5 维动作、奖励、PPO 训练和严格评估 | `RLCableGraspEnv`, `RLConfig` |
| `src/panda_cable_grasp/evaluation/` | 多场景两级并行基准、统一轨迹/双视频记录、回放、运动诊断与失败分类 | `benchmark`, `EpisodeRecorder`, `replay` |
| `src/panda_cable_grasp/expert/` | 使用仿真特权状态的专家策略和数据采集 | `collect_dataset` |
| `src/panda_cable_grasp/dynamicvla/` | DynamicVLA 动作适配、数据转换和微调启动 | `DynamicVLATaskSpaceAdapter` |
| `src/panda_cable_grasp/cli/` | 面向用户的规则策略与 DynamicVLA 命令 | `run_grasp`, `run_dynamicvla` |

## 物理环境边界

`env/environment.py` 仍保留完整的状态型 `CableGraspEnv`。它很长，但内部状态高度耦合于同一个 MuJoCo step：扰动、500 Hz 接触采样、夹持状态机和严格成功计时需要共享一致状态。为了降低风险，本轮没有机械地拆成多个互相回调的类。

已经独立出的无状态数学代码位于 `env/kinematics.py`，策略、RL、专家和 DynamicVLA 都从这里复用四元数误差和 Jacobian 工具。后续拆分环境时，应优先提取无状态模块并保持 `CableGraspEnv` 的外部接口不变。

## 导入规则

新代码使用包导入：

```python
from panda_cable_grasp.env import CableGraspEnv, EnvConfig
from panda_cable_grasp.rl import RLCableGraspEnv, RLConfig
```

旧的根目录同名模块和 `rl.*`、`privileged_expert.*`、`dynamicvla_finetune.*`
兼容入口已经退役。命令行使用 `pyproject.toml` 注册的 `panda-cable-*` 命令，
Python 代码只从 `panda_cable_grasp` 包导入。

## 配置与路径

- XML 模型：`assets/mujoco/`
- 静态配置：`configs/`
- 默认输出：`outputs/`
- 输出根目录可以通过 `PANDA_CABLE_OUTPUT_ROOT` 覆盖。
- Menagerie 路径解析集中在 `paths.py`；MuJoCo 运行时设置集中在 `runtime.py`。

代码不能依赖开发者个人的绝对路径。路径可移植性由 `tests/unit/test_portability.py` 检查。

## 测试层次

- `tests/unit/`：配置、场景注册、转换和路径等快速测试。
- `tests/integration/`：真实 MuJoCo 环境、RL 奖励状态机和专家策略测试。

修改奖励时至少运行 integration；修改 XML、接触、动作映射或成功判定时运行全套测试，并补一个短 episode 烟测。

## 根目录约定

根目录只保存项目级元数据和入口文档。可执行实现全部位于
`src/panda_cable_grasp/`，维护脚本位于 `tools/`，生成产物位于 `outputs/`。
各根目录文件的用途见 [根目录说明](root_layout.md)。
