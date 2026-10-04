# 指定点 PPO：工作区目标采样与七阶段课程

2026-10-05，正式接入 `dev` 的 `panda-cable-rl-train` / `panda-cable-rl-eval`。
基础指定节点任务、109 维状态、5 维垂直向下动作和 state99 权重扩展入口原本已存在。
本次迁入的是实验脚本中的**工作区目标筛选和七阶段课程机制**。

## 目标如何选择

显式传入 `--target-workspace /path/to/workspace.npz`：文件包含均匀递增的
`xs`、`ys` 和形状为 `(len(ys), len(xs))` 的布尔掩码。默认使用 `interior`，
可用 `--workspace-mask-key` 选择其他字段。位置按最近网格点所属单元判断；
高度限制为 −5 mm～100 mm，与原实验筛选一致。掩码记录路径和 SHA256。

每回合在当前位于掩码内的**全部材料节点**中均匀抽一个目标，覆盖端部，
不再使用默认的 7～32 中部节点范围。因此本模式不采用 `target-u-min/max`。
目标身份整回合固定；节点随后离开工作区不会换目标或直接判失败。
显式指定的节点不在工作区时直接报错，不用其他节点替代。

如果初始化没有可用节点，机械臂保持 ready 执行指令，线缆通过正常物理仿真
推进，最多等待 `--workspace-entry-seconds`（默认 8 秒）。此段不执行 PPO 推理、
不累计 PPO 奖励。进入后才选目标，PPO 仍获得完整的 `--episode-seconds`
（默认 15 秒）；仿真绝对时间包含等待段。

等待后仍无可用节点则重新初始化，最多 `--workspace-reset-attempts` 次，默认 50。
显式请求种子失败后，第 k 次重试使用 `requested_seed + 100003*k`。
日志记录请求种子、实际种子、拒绝次数、等待时长和目标初始坐标。
设置次数为 1 可禁止这种种子重试。

这些是**有可用工作区目标条件下的训练、评估**，不能把筛选后的成功率当作
所有原始场景的无条件成功率。目标首次确认夹持前处于掩码内的时间比例也会记录。

## 七阶段课程

顺序固定为：静态、刚体运动、形变、组合运动、复杂静态、形变轨迹回放。
训练与阶段评估采用相同场景混合及运动难度；难度通过正式环境的
`set_motion_difficulty` 应用。阶段编号从 0 开始。

| 阶段 | 场景权重（按上述六场景顺序） | 运动难度 |
|---|---|---|
| 0 静态工作区目标 | 1 / 0 / 0 / 0 / 0 / 0 | 0 |
| 1 加入复杂静态 | .3 / 0 / 0 / 0 / .7 / 0 | 0 |
| 2 低难度单独运动 | .1 / .4 / .4 / 0 / .1 / 0 | 1/3 |
| 3 完整单独运动 | .1 / .4 / .4 / 0 / .1 / 0 | 1 |
| 4 组合运动 | .1 / .1 / .1 / .6 / .1 / 0 | 1 |
| 5 加入低难度回放 | .1 / .1 / .1 / .1 / .3 / .3 | 1/3 |
| 6 六场景完整难度 | .1 / .1 / .1 / .1 / .3 / .3 | 1 |

一次阶段评估同时满足**首次确认抓对目标率 ≥60%**、**指定节点任务成功率 ≥40%**
才合格；最近三次中两次合格则升一级，晋级后清空历史。
首次确认指标在 500 Hz 物理子步锁存；错误首次抓取后重新抓对仍可获得任务成功，
不会将首次指标改成正确。这个指标仅用于课程晋级，未增加奖励或提前终止规则。

`evaluation/workspace_curriculum.json` 保存阶段、晋级历史和最近评估指标；
`evaluation/workspace_eval_episodes.jsonl` 记录每次阶段/最佳模型复核评估的回合信息。
沿用正式训练器的模型保存和复核机制。续训需显式 `--resume`，并沿用对应运行目录、
场景、工作区和任务参数；从较早 checkpoint 开新运行时不能冒用较晚的课程状态。

## 新场景族与历史配置

`--workspace-scene-family cable_x075_v1` 选择为指定点训练、测试定义的新六场景；
`legacy_ppo_six` 显式选择旧六场景。此参数关闭旧 L1 课程；新课程需另加
`--workspace-curriculum`，且必须开启周期评估。
复杂静态及回放场景需要已有的形变轨迹库，可用 `--replay-bank-dir` 指定目录。
`--replay-seed-fallback error` 拒绝不在库中的种子，`resample` 沿用原场景的随机条目选择。
工作区初始化重试与轨迹库条目选择是两项独立设置。

执行层可选择已迁入的增强 6、有界 DLS、重力前馈和碰桌保护。
奖励、成功容差沿用正式 `RLConfig`：默认 ±2 节点。
“成功 ±2、位置正奖励 ±4”的优化点 1 已于 2026-10-05 实装，并增加递减抬升
正奖励和跨重抓信用，须显式选择 `--target-reward-profile tolerance_progress_v2`；
见 [奖励 v2](target_node_reward_v2.md)。默认仍是原奖励分支。
原冻结实验的 ±4 成功容差、奖励及旧 Full6 执行配置不随本迁移改变；
新代码运行是新实验，旧权重和结果仍归属旧场景与原配置。

## 使用示例

以下是启动命令示例，本次迁移没有启动正式训练。先准备工作区掩码和轨迹库，
使用新的输出目录；程序不硬编码历史实验路径或动态导入冻结源码。

```bash
PYTHONPATH=src python -m panda_cable_grasp.rl.train \
  --target-node-task --target-node-tolerance 2 \
  --robot nero --action-mode task_space_vertical_down --observation-mode state \
  --controller-profile enhanced6_bounded_dls_v1 --table-execution-guard \
  --target-workspace /path/to/workspace.npz --workspace-mask-key interior \
  --workspace-scene-family cable_x075_v1 --workspace-curriculum \
  --replay-bank-dir /path/to/motion_banks \
  --eval-freq 20000 --eval-episodes 50 --eval-confirmation-episodes 50 \
  --timesteps 3000000 --workers 12 --eval-workers 4 \
  --output outputs/rl/train/target_node_x075_workspace
```

可添加 `--warm-start /path/to/state99/final_model.zip` 扩展兼容旧模型，
或 `--resume /path/to/target_node/final_model.zip` 续训，二者互斥。

```bash
PYTHONPATH=src python -m panda_cable_grasp.rl.evaluate \
  --model outputs/rl/train/target_node_x075_workspace/final_model.zip \
  --robot nero --controller-profile enhanced6_bounded_dls_v1 --table-execution-guard \
  --workspace-scene-family cable_x075_v1 \
  --target-workspace /path/to/workspace.npz --workspace-mask-key interior \
  --replay-bank-dir /path/to/motion_banks --headless --no-video --episodes 50
```

评估从 checkpoint 配置读取指定节点奖励、容差和动作接口；场景族、执行配置、
工作区及库路径按上述命令显式选择。CSV 记录实际种子与筛选信息，manifest 标明条件评估。
不传工作区参数时，基础指定点采样保持原有默认行为。
