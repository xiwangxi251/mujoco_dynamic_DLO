# 增强 6 执行流程迁入 dev（2026-10-05）

## 迁入范围

按用户选择，将以下五项迁入本地 `dev`，以 `63c3514` 为基准。
151 的 `dev` 在检查时为 `9ce2768`；本地基准比它多两个文档整理提交，产品代码一致。

| 项目 | 来源与正式入口 |
|---|---|
| 新场景族 `cable_x075_v1` | 原提交 `0de702d`；见 [场景族说明](../cable_x075_scene_family.md) |
| 实际夹持判定 `physical_holding_v2` | 原提交 `1cd0897` 的环境、RL、测试及说明；见 [判定修正](physical_grasp_judgement_fix_20261004.md) |
| 重力模型缓存 | 从 `d064c1b` 提取独立机器人模型、每环境独立 MjData、力矩合成 |
| 增强 6＋有界 DLS | 从 `tools/recording/experiment_bounded_full6.py` 的 `bounded_action` 整理到 `env/control.py` |
| 碰桌保护 | 从 `tools/rl/table_execution_guard.py` 迁到 `env/table_execution_guard.py`；保持原参数及投影，增加回合计数重置 |

没有合并整个 `feat/unified-jacobian-control-20261002` 分支。
新位置优先算法 `unified_jacobian_v1`、额外制动/预测速度保护/加速度投影/
保护锁定与释放试验没有进入此次产品代码；原 dev 的 legacy 控制路径保留。
实验录像、工作区扫描器和工作区条件目标采样不是本次控制迁移的运行依赖。

## 正式配置与使用

控制版本名为 **`enhanced6_bounded_dls_v1`**，默认仍为 `legacy`。
只支持 NERO。碰桌保护默认关闭，需显式开启。

```python
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig

env = CableGraspEnv(EnvConfig(
    robot="nero",
    controller_profile="enhanced6_bounded_dls_v1",
    table_execution_guard_enabled=True,
))
```

规则、PPO 训练/评估、DP、低维 DP、DynamicVLA 和 π0.5 的执行入口支持：

```text
--robot nero --controller-profile enhanced6_bounded_dls_v1 --table-execution-guard
```

例如只选择新族的一个场景运行规则策略：

```text
panda-cable-grasp --headless --trials 1 --scenario dev_static_x075_v1 --robot nero --controller-profile enhanced6_bounded_dls_v1 --table-execution-guard
```

场景族与控制版本独立选择；新族没有自动替换原 PPO 六场景列表。
本次没有改已有训练包、冻结源码、权重、轨迹库和实验统计。
历史结果保持旧配置归属。新运行须记录场景、控制版本、保护开关与
`grasp_judgement_version`；旧权重在新执行层/新判定下的评估属于新实验。
本文件记录的是本地迁移；服务器部署须单独记录实际提交。

## 动作流程

1. 绝对位姿策略：位置误差乘 6、姿态误差乘 10。
   PPO 保持原位移增量语义：位移除以控制周期得到线速度，姿态误差乘 10。
2. 求解请求的线速度模长上限 0.9 m/s，角速度模长上限 1.4 rad/s。
3. 完整六维雅可比共同求解位置与姿态，阻尼 0.04；
   ready 回拉增益 0.8，通过完整任务的零空间生成偏好速度。
4. 最小化 `||J v - twist||² + 0.04² ||v - ready_bias||²`。
   优化内部同时满足原关节速度边界与 `q_actual + 0.11 v` 的关节位置边界。
   当前实际状态导致约束不可行时明确报错，不返回越界指令。
5. 关节目标为 `q_actual + 0.11 v`，再经过原执行层的位置与速度处理。
   控制周期 20 ms、物理周期 2 ms；伺服 kp×4、kv×2。
6. 开启碰桌保护时，在原执行速度保护后，每个物理步修正最终关节目标。
7. 重力前馈每物理步计算，基于缓存机器人模型；实际已限幅 PD、原外力和
   重力前馈合成后仍在原执行器力矩边界内，不直接修改 qpos/qvel。

原关节速度、位置、执行器力矩边界保持。
原实验的执行层笛卡尔上限保留为线速度 0.9 m/s、角速度 2.0 rad/s；
求解器的 1.4 rad/s 角速度请求上限与执行层的 2.0 rad/s 是两个不同位置的限制。
增强 6 配置关闭额外关节指令加速度限制，与已接受的增强 6 实验一致；
没有增加参考轨迹爬坡或“实际到达上方航点才下降”的门限。
策略本来的阶段转换与任务逻辑没有被迁移代码改写。

## 保护范围与限制

保护继承 1 mm 余量、15 cm 激活距离、距离增益 5/s、制动参数
2 m/s²、响应延迟 20 ms、实际速度反馈增益 2、12 轮投影。
监测 link7、法兰、夹爪基体和两根手指的五个去重网格，未覆盖靠近底座的臂段。
保留投影残差计数；这是执行指令投影原型，不是形式化安全保证。
可选旧几何预测过滤器不会在增强 6 路径中直接提交预测物理状态：
实际积分仍通过有界 PD＋重力前馈及可选碰桌保护执行。

已有工作区和录像属于原冻结实验。本迁移不宣称这些扫描是正式产品代码的新结果。
当前选择的矩形 X=0.60～0.84 m、Y=−0.22～0.30 m 仍是原实验的工作区参考。

## 针对迁移的检查

`tests/unit/test_enhanced6_bounded_control.py` 覆盖：无激活约束时与原 Full6
等价、关节触限时重新分配速度、轻微实际越界恢复、原力矩边界、缓存重力
与完整场景一致及数据隔离、原限位/增益、500 Hz 保护与回合重置、
反馈投影不修改物理状态、规则与绝对位姿接口一致、PPO 增量语义和保护执行。
还运行实际夹持判定、RL 基线、指定节点及相关场景回归用例。
没有重新执行全工作区扫描或录制视频。

本地迁移回归：38 passed、69 deselected（21.63 s）；8 个执行入口的
`--help` 均包含控制版本和碰桌保护开关。仅验证接口可用，没有对所有策略
重新测成功率。
