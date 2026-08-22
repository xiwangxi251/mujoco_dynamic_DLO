
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

