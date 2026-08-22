
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
