
# 项目结构

| 目录 | 作用 |
| ---- | ---- |
| `assets/` | 放仿真资源。目前主要是 `assets/mujoco/`，里面是 Panda、线缆、桌面、夹爪等 MuJoCo XML 模型 |
| `configs/` | 放配置文件，主要用于训练或外部方法接入。目前主要有 DynamicVLA 微调配置 |
| `docs/` | 放项目文档，包括当前实验设计、RL baseline、实验协议、服务器配置、DynamicVLA、历史记录等 |
| `outputs/` | 放实验运行产生的结果，如模型、录像、CSV、TensorBoard、专家数据集。 |
| `src/` | 项目真正的核心代码。环境、PPO、规则控制、专家策略、评测、DynamicVLA 都在这里 |
| `tests/` | 测试代码，用于检查环境、RL、场景设置、入口脚本等有没有被改坏 |
| `tools/` | 辅助工具，比如安装检查、录像回放、消融实验等，不属于核心算法代码 |

# src结构
| 目录 | 作用 |
| ---- | ---- |
| `env/` | 最底层的 MuJoCo 仿真环境。机械臂、线缆运动、接触、夹持、成功判定、运动学等核心逻辑都在这里。 |
| `scenarios/` | 场景配置中心。定义 static / rigid / shape / combined、不同运动规律、ID/OOD 等实验场景。 |
| `policies/` | 非学习的规则控制策略。例如 scripted controller，用手写规则决定机械臂怎么抓。 |
| `rl/` | 强化学习部分。把底层环境包装成 RL 环境，定义 observation、action、reward，并负责 PPO 训练和测试。 |
| `expert/` | 特权专家策略。可以直接读取仿真中的线缆真实位置、速度等信息，用于做强基线或采集专家数据。 |
| `evaluation/` | 统一实验评测。批量跑测试、统计成功率、失败原因、不同运动场景下的结果。 |
| `dynamicvla/` | DynamicVLA 相关代码。包括把 DynamicVLA 输出动作接到你当前环境，以及数据转换和微调相关功能。 |
| `cli/` | 命令行入口层。负责解析你在终端输入的参数，再调用上面的环境、策略、RL、DynamicVLA 等模块。 |

# 仿真评测架构

正式的动作成功率评测统一由 `evaluation/benchmark.py` 调度。scripted、expert 和 PPO
只实现策略适配，场景选择、seed、公共成功条件、episode 汇总、双视角录像和完整状态记录均走同一条路径。
`cli/run_grasp.py` 的 headless 模式是该入口的兼容包装；GUI、运动诊断、RL 训练诊断和专家数据采集不再作为平行的成功率评测实现。

`evaluation/recording.py` 定义统一 episode schema，`evaluation/replay.py` 负责用保存的
`.mjb` 和 `mjSTATE_FULLPHYSICS` 校验或回放。两级进程调度由 benchmark 管理：
`scenario_workers` 限制同时活跃的场景数，`envs_per_scenario` 限制单场景并行回合数。
详细命令和输出目录见 [`docs/evaluation.md`](docs/evaluation.md)。

# 相机架构

环境只有一套标准视觉传感器，由 `EnvConfig.dynamicvla_cameras_enabled` 在
MuJoCo 模型编译前启用：

- `dynamicvla_opst_camera` 固定在 Panda base/world，用作全局 opposite 视角；
- `dynamicvla_wrist_camera` 挂在 `panda_hand`，随腕部运动。

需要图像的代码统一调用 `CableGraspEnv.dynamicvla_camera_rgb()`，一次取得
`opst_cam` 和 `wrist_cam`。普通环境观测和 RL 策略观测不隐式渲染图像；
正式评测的标准录像均保存为每个 episode 的 `global.mp4` 和 `wrist.mp4`；专家数据采集器
可按数据集 schema 使用自己的文件名。交互式 MuJoCo viewer 可以保留自由相机，但它不属于传感器、
模型输入或标准录像 schema。
