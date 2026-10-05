# 指定点 PPO 默认配置（2026-10-05）

正式训练入口 `python -m panda_cable_grasp.rl.train --target-node-task` 现在默认采用：

| 配置 | 默认值 |
|---|---|
| 机器人 / 动作接口 | NERO / `task_space_vertical_down` |
| 执行控制器 | `enhanced6_bounded_dls_v1`，含增强伺服、姿态增益 10、有界 DLS、缓存重力前馈 |
| 碰桌执行保护 | 开启 `TableExecutionGuard`，每物理步执行 |
| 六场景族 | `cable_x075_v1`，训练和阶段评估一致 |
| 指定点奖励 | `tolerance_progress_v2`，含跨重抓抬升信用及延迟释放惩罚修复 |
| 成功 / 位置正奖励容差 | ±2 / ±4 节点 |

唯一默认定义位于 `src/panda_cable_grasp/rl/target_defaults.py` 的
`TARGET_PPO_DEFAULTS`，由正式训练参数解析调用，避免不同启动脚本分别维护开关。
启动配置保存最终解析值及机器人类型。工作区 NPZ、训练库、评估库路径仍须按
本机数据位置指定；七阶段课程仍用 `--workspace-curriculum` 显式开启。

命令行显式配置优先。复现旧实验须显式选择旧场景族、控制器和奖励；
需要关闭碰桌保护时使用 `--no-table-execution-guard`。
普通任意点 PPO 的默认配置保持原样。已有权重、冻结源码及历史统计不改写。

评估入口从权重旁的 `training_config.json` 继承机器人、控制器、碰桌保护
开关及指定点场景族。显式参数可以覆盖；旧配置中关闭的保护不会被新默认开启，
旧场景族也不会被替换。工作区和轨迹库路径按已有评估接口显式传入。

## 151 采用记录

新续训输出：`outputs/rl/train/ppo_targetnode_x075_guard_dev_20261005`。
沿用最新完整保存权重和优化器，累计目标仍为 2000 万步。
从旧六场景切换到新六场景并开启保护，课程阶段保持当前阶段，清空旧通过历史。
旧训练进程停止，原权重、配置、录像和统计保留。
启动记录包含 checkpoint 来源与哈希、实际提交、代码快照、配置差异和原进程身份。
