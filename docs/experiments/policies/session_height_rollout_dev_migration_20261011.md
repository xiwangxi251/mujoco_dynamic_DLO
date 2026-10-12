# 当前151训练配置：提交与迁移到dev

## dev原生入口已迁移（2026-10-11）

`dev` 已合入87b59957和bb301cb，并将运行时晋级模块接入通用训练入口。
新增 `src/panda_cable_grasp/rl/rollout_curriculum.py`，环境直接支持每次reset的实际阶段/难度标签；不再需要在新训练中替换类或方法。

新增参数：

- `--rollout-ability-curriculum`：启用14阶段训练回合晋级，自动采用ordered能力组；要求普通cable_x075_v1课程和 `--eval-freq 0`。
- `--eval-workers 0`：关闭评估时可显式使用0；有独立评估时仍要求正数。
- `--schedule-total-timesteps 20000000`：LR/entropy使用原全局样本计划；调度元数据写入checkpoint，后续续训自动继承，拒绝扩展预算或重置已保存的全局调度。
- `--rollout-gate-profile fast`：用户2026-10-11选择的更快晋级档。每个目标独立100回合，成功≥40%、承载≥50%、稳定夹持≥40%；每个回放独立20回合、成功≥30%。一个完整窗口全部达标即晋级。默认strict保留原两窗口规则。
- `--rollout-gate-profile consolidate`：用户2026-10-12授权加入第15阶段。前14阶段保持fast规则；组合满难度达标后，复杂初始静态及复杂初态整体运动各30%，其余五场景各8%，全部运动满难度。每场景独立100回合统计，终阶段不再晋级，持续原累计20M预算。已通过第14阶段的fast checkpoint在worker首次reset前直接恢复到第15阶段；保留策略/价值/优化器/全局调度，清空新阶段统计窗口。

从strict切到fast时，保留checkpoint课程阶段、策略、价值、优化器与全局样本数；晋级窗口清空，以新规则下的训练回合重新计数。奖励、控制、地图及14阶段顺序不变。fast训练和续训均须显式传入该profile。

例如，在新的独立输出目录从零训练：

```bash
PYTHONPATH=src python -m panda_cable_grasp.rl.train \
  --output outputs/rl/train/NEW_RUN \
  --robot nero --action-mode task_space_vertical_down --observation-mode state \
  --workspace-scene-family cable_x075_v1 --ordinary-family-curriculum \
  --rollout-ability-curriculum --ordinary-reward-profile session_height_v1 \
  --rl-config-file archive/session_height_ordered_20m_20261011/rollout_only_20261011/reference_config.json \
  --controller-profile enhanced6_bounded_dls_v1 --table-execution-guard \
  --disable-table-finger-collision-filter --contact-read-cache --enable-safety-penalties \
  --action-workspace src/panda_cable_grasp/rl/workspaces/workspace3d_20261010.json \
  --replay-bank-dir outputs/motion_banks --replay-seed-fallback resample \
  --workers 48 --eval-freq 0 --eval-workers 0 --episode-seconds 15 --seed 20261010 \
  --timesteps 20000000 --schedule-total-timesteps 20000000 \
  --n-steps 1024 --batch-size 512 --n-epochs 5 --torch-threads 1 --device cpu \
  --learning-rate-initial .0003 --learning-rate-final .00003 \
  --entropy-coef-initial .01 --entropy-coef-final .001 --checkpoint-steps 200000
```

这是入口示例，不是长任务监督器；长期训练仍须由独立supervisor启动、检查和监控。
续训增加 `--resume`，`--timesteps` 填剩余样本；从当前冻结运行的旧checkpoint首次转入原生入口时显式指定全局20M调度，上述profile及原配置保持一致。
原生入口会在worker首次reset前读checkpoint阶段，再恢复窗口、策略、价值、优化器及惩罚渐入；CSV按实际回合阶段/难度归属。
`training_health.json` 包含每轮采样FPS、采样耗时、优化耗时和策略/优化器参数有限性。

此次源码迁移不切换当前151训练：其冻结源码、运行脚本哈希、进程身份和原预算继续保持。
下文保留迁移前的差异说明；运行脚本归档仍用于对照已运行版本。

## 版本范围

比较基准是 `dev` 的 `a04a510b5e694f754fd77c21a46cf9a151a3bf07`。
训练分支为 `train/session-lift-curriculum-20261011`。

1. `87b59957db9be2f3d6daf287d4af1b4d49a284b3`：会话抬升奖励、14阶段分场景课程、精确预算。
2. `bb301cb874e434b2e07012dfa5e3e6ef65e5dce0`：正在运行的关闭评估版本，包含训练回合晋级模块、启动及监督脚本、配置和启动证据快照。

第二笔提交保存的是**实际运行脚本**；第三笔迁移提交已提供上述原生入口。归档代码位于
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
历史归档启动器仍传入 `--eval-workers 1` 占位值，实际不创建eval环境；原生入口可以直接指定0。
配置中的 `actual_eval_workers=0` 是运行人数。

## 迁移方式

在dev的干净检出中先应用上述两笔提交，再应用原生迁移提交，才能使用新的通用入口。
迁移前两笔提交已合入dev；现有训练进程没有移动。

若保留当前脚本启动方式，迁移主要是部署路径与运行身份：

1. 为新的运行选择独立输出目录；将启动器里的 `SOURCE` 指向迁移后源码，`ROOT` 指向新运行；地图文件保持SHA256 `43840a44aa6a58e2bfd36ba8c282c29646a0457004f5cfb858ef95789429a756`。
2. 更新计划中的源码提交和参考配置路径。历史PID、启动/停止回执及预检回执只作证据，不拿来启动或终止新任务。
3. 使用当前有效checkpoint继续时，加载策略、价值、优化器和rollout窗口状态；剩余预算为20000000减去实际累计样本，LR/entropy仍按原全局20M，不重置惩罚渐入。新回合的phase/difficulty必须在SB3首次reset前设置。
4. 监督脚本和原监控任务指向新的run/commit/PID出生身份；首个新更新以后继续原累计20M监控。迁移启动本身需要一次有界预检与worker-ready确认。

以下通用 `rl/train.py` 整理工作已在本次原生迁移完成：

- 在 `environment.py` 接入每次reset的实际phase/difficulty标签及回放难度。
- 在 `train.py` 的make_worker/课程callback分支接入rollout gate，关闭独立评估时配置明确记录0个eval worker。
- 保留checkpoint窗口状态、全局调度和精确样本预算；将采样FPS计时纳入通用health callback。

冻结快照保持原内容；原生代码与通用入口单独提交，避免改变历史脚本哈希。

## 必须保留的训练设置

- 99维状态、5动作、非指定节点；48个训练worker，0个独立eval worker。
- session_height_v1，高度带与安全奖励开启；18cm下每厘米上升+0.4、下降-0.4，扣回不超会话余额；松脱退余额，重抓开启新会话。
- 当前增强6有界DLS、奇异保护、TableExecutionGuard、cache和XYZ地图；旧filter与geometric safety关闭；15秒、0.82秒ready。
- 14阶段：普通静态、复杂静态；简单初态整体运动、复杂初态整体运动与shape replay、形变、组合四组各1/3、2/3、1难度。
- 每目标独立success≥0.6、loaded≥0.8、secured≥0.6；每回放独立success≥0.4；全场景足量，两次独立窗口达标才晋级。
- 累计20M；1024步/worker、batch512、5epochs；LR0.0003→0.00003、entropy0.01→0.001，全局样本调度；惩罚渐入480k/240k。

训练窗口统计是随机策略交互，不是独立确定性评估成功率。历史预检、退出判failed及完整启动证据随快照保留。

原生迁移验证通过21项单元检查和3项集成检查，包含真实spawn worker采样、14样本精确末轮、checkpoint加载后累计26样本、阶段/窗口恢复、全局调度续接及0个eval环境。未对正在运行的训练追加评估或录像。
