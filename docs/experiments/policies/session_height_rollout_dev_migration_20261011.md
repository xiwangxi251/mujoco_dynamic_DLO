# 当前151训练配置：提交与迁移到dev

## 版本范围

比较基准是 `dev` 的 `a04a510b5e694f754fd77c21a46cf9a151a3bf07`。
训练分支为 `train/session-lift-curriculum-20261011`。

1. `87b59957db9be2f3d6daf287d4af1b4d49a284b3`：会话抬升奖励、14阶段分场景课程、精确预算。
2. 本文所在提交：正在运行的关闭评估版本，包含训练回合晋级模块、启动及监督脚本、配置和启动证据快照。

第二笔提交保存的是**实际运行脚本**，没有把运行时补丁重写为新的训练入口。代码位于
`archive/session_height_ordered_20m_20261011/rollout_only_20261011/`。
`snapshot_manifest.json` 记录逐文件SHA256；`run_rollout.py`、`rollout_gate.py`与远端已核实版本一致。
权重、视频、轨迹和日志不入Git；实际权重路径与SHA256保存在 `resume_plan.json`。
`plan.json` 是首次启动时的历史计划，当前生效参数以 `training_config.json`、`resume_plan.json`及启动器为准。

151正在运行的工作树仍冻结在87b59957；提交此快照不修改远端进程、配置或工作树。

## dev需要的源码变化

第一笔提交修改以下生产文件，不能只改奖励profile字符串：

| 文件 | 迁移内容 |
| --- | --- |
| `src/panda_cable_grasp/rl/session_lift.py` | 有效夹持会话的可逆高度余额；下降、松脱、节点切换结算 |
| `src/panda_cable_grasp/rl/environment.py` | 接入session_height_v1，重抓从前一步高度重新开始，18–25cm保持逻辑 |
| `src/panda_cable_grasp/rl/ordered_curriculum.py` | 七个具体场景组成六个能力组、14阶段及80%/20%混合比例 |
| `src/panda_cable_grasp/rl/exact_budget.py` | 缩短末轮、末步只执行必要worker，累计样本恰好达到预算 |
| `src/panda_cable_grasp/rl/train.py` | profile和课程入口、记录配置、启用精确预算 |
| `src/panda_cable_grasp/rl/metrics.py` | 会话奖励诊断字段 |

同时迁移该提交的 `test_session_lift.py` 和 `test_ordered_curriculum.py`。

关闭评估版本还必须启用本次提交的：

- `rollout_gate.py`：每目标场景100回合、每个回放场景50回合独立计数；两次不重叠窗口全部达标晋级；阶段、难度及窗口状态写入checkpoint。切课前开始的回合不计入新阶段。回放使用已学最高难度。
- `run_rollout.py`：在构建worker前安装上述课程；eval_freq=0；保持全局20M学习率、熵和惩罚渐入；记录实际0个eval worker及每轮采样FPS；独立监督训练退出、源心跳和真实进度。

**只把 `--eval-freq` 改为0，原先依赖评估的课程不会自动晋级。**
参数解析仍保留 `--eval-workers 1` 占位值，实际不创建eval环境；
配置中的 `actual_eval_workers=0` 是运行人数。

## 迁移方式

在dev的干净检出中按顺序应用上述两笔提交即可保留全部已提交代码。
本次没有操作dev，也没有移动现有训练进程。

若保留当前脚本启动方式，迁移主要是部署路径与运行身份：

1. 为新的运行选择独立输出目录；将启动器里的 `SOURCE` 指向迁移后源码，`ROOT` 指向新运行；地图文件保持SHA256 `43840a44aa6a58e2bfd36ba8c282c29646a0457004f5cfb858ef95789429a756`。
2. 更新计划中的源码提交和参考配置路径。历史PID、启动/停止回执及预检回执只作证据，不拿来启动或终止新任务。
3. 使用当前有效checkpoint继续时，加载策略、价值、优化器和rollout窗口状态；剩余预算为20000000减去实际累计样本，LR/entropy仍按原全局20M，不重置惩罚渐入。新回合的phase/difficulty必须在SB3首次reset前设置。
4. 监督脚本和原监控任务指向新的run/commit/PID出生身份；首个新更新以后继续原累计20M监控。迁移启动本身需要一次有界预检与worker-ready确认。

若希望以后直接通过通用 `rl/train.py` 启动，可再将 `rollout_gate.py` 的环境方法和callback移入包内，并给train入口增加显式的训练回合晋级选项：

- 在 `environment.py` 接入每次reset的实际phase/difficulty标签及回放难度。
- 在 `train.py` 的make_worker/课程callback分支接入rollout gate，关闭独立评估时配置明确记录0个eval worker。
- 保留checkpoint窗口状态、全局调度和精确样本预算；将采样FPS计时纳入通用health callback。

这是后续整理入口的工作，本次快照提交没有声称完成此重构。

## 必须保留的训练设置

- 99维状态、5动作、非指定节点；48个训练worker，0个独立eval worker。
- session_height_v1，高度带与安全奖励开启；18cm下每厘米上升+0.4、下降-0.4，扣回不超会话余额；松脱退余额，重抓开启新会话。
- 当前增强6有界DLS、奇异保护、TableExecutionGuard、cache和XYZ地图；旧filter与geometric safety关闭；15秒、0.82秒ready。
- 14阶段：普通静态、复杂静态；简单初态整体运动、复杂初态整体运动与shape replay、形变、组合四组各1/3、2/3、1难度。
- 每目标独立success≥0.6、loaded≥0.8、secured≥0.6；每回放独立success≥0.4；全场景足量，两次独立窗口达标才晋级。
- 累计20M；1024步/worker、batch512、5epochs；LR0.0003→0.00003、entropy0.01→0.001，全局样本调度；惩罚渐入480k/240k。

训练窗口统计是随机策略交互，不是独立确定性评估成功率。历史预检、退出判failed及完整启动证据随快照保留；本次提交不追加仿真、评估或录像。
