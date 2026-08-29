# 实验索引

> 更新时间：2026-08-27

本页是实验记录的统一目录。项目总体目标、路线和当前问题见
[项目目标与当前状态](../project_status.md)，实验定义和正式统计口径见
[实验协议](../reference/experiment_protocol.md)。

## 结果可信度

| 标签 | 含义 |
|---|---|
| `SMOKE` | 只验证链路能运行，不能比较性能 |
| `PILOT` | 固定设置的小样本诊断，可指导下一步，不是最终结论 |
| `EVALUATED` | 场景、seed、样本量和产物明确，可作为阶段结论 |
| `PROVISIONAL` | 有完整逐回合结果，但缺少模型哈希、manifest 或冻结代码证据 |
| `FROZEN` | 按预注册协议完成，可进入最终表格 |

## 当前实验总表

| 方法/实验 | 状态 | 样本量 | 当前结论 | 详细记录 |
|---|---|---:|---|---|
| Scripted 当前基线 | `EVALUATED` | 4×50 | 91/200（45.5%）；主要瓶颈是形成双指候选 | [Scripted/Expert](../scripted_policy_experiments.md) |
| Expert 当前基线 | `EVALUATED` | 4×50 | 130/200（65%）；rigid 41/50 | [Scripted/Expert](../scripted_policy_experiments.md) |
| 严格竖直夹爪 | `EVALUATED` | 2 方法×4×50 | 明显降低动态可达性，不作为默认设置 | [Scripted/Expert](../scripted_policy_experiments.md) |
| PPO v6 3.801M | `EVALUATED` | 4×20 | 19/80；combined 0/20 | [RL](../rl_experiments.md) |
| PPO v6 8.8M best | `PROVISIONAL` | 4×20 | 45/80；rigid 19/20，combined 4/20 | [RL](../rl_experiments.md) |
| PPO v6 19.87M | `PILOT` | 训练曲线 | 最近训练窗口约 59%；尚无固定四场景严格评估 | [RL](../rl_experiments.md) |
| PPO 课程回放 | `IMPLEMENTED` | 测试 | 代码和测试完成，训练收益待验证 | [RL](../rl_experiments.md) |
| DynamicVLA finetune | `PILOT` | 4×10 | 2/40；主要失败在双指候选之前 | [DynamicVLA](../dynamicvla_experiments.md) |
| DynamicVLA `ckt` 零样本 | `PILOT` | 40 | 0/40 | [DynamicVLA](../dynamicvla_experiments.md) |
| PPO 外工作区控制诊断 | `PILOT` | 单轨迹消融 | 位置/高度跟踪问题正在定位 | [控制诊断](control_diagnostics.md) |

## 产物位置

- 仓库默认输出：`outputs/`
- 从旧工作区迁移的映射：`outputs/migration_manifest.json`
- 本机服务器日志副本：`../linux_log/`

`outputs/` 和 `../linux_log/` 不是版本控制中的最终证据库。需要进入阶段结论的运行，应在
文档中保存最小复现信息：commit、dirty 状态、命令、场景、seeds、checkpoint SHA256、
manifest 路径和汇总表。大型视频、MJB 和完整状态继续保留在实验存储中。

## 新实验记录模板

每个新实验至少记录：

1. 问题或假设；
2. 唯一改变的变量与保持不变的变量；
3. code commit、dirty 状态和 checkpoint SHA256；
4. 场景、paired seeds、样本量和完整命令；
5. 原始产物路径；
6. `task_success`、Wilson 区间、配对比较和互斥失败类型；
7. 结论、有效性限制，以及是否进入默认配置；
8. 下一实验。

不要只根据目录名推断样本量，应以 manifest 和逐回合文件为准。
