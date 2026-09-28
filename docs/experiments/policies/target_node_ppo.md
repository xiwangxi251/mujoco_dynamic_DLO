# 指定材料节点的状态 PPO

这是独立的可选任务，默认的任意位置抓取任务、99 维观测及线缆物理保持原样。
开启 `--target-node-task` 后，每回合在材料坐标 `[1/6,5/6]` 中均匀采样一个
离散节点，并在整个回合固定其身份。当前 40 节点模型对应 **7–32，零起始编号**。

输入为 109 维：原有 99 维状态加目标材料坐标 `2u-1`、TCP 坐标系下的位置、
相对于 TCP 的线速度及局部单位切线。目标不是“每步离夹爪最近的节点”。
此版本使用仿真真值状态，尚未实现点云中的材料身份识别；点云训练入口拒绝该模式。

默认容差为目标 **±2 节点**。接近、对齐和闭爪几何针对目标窗口；主要夹持、
secured、举起和保持奖励受材料身份匹配约束。错误抓取的第一次确认产生 0.5
惩罚，整回合上限 2；放开错误抓取不会承担正确 secured 抓取的主动张爪惩罚。

成功要求原来的物理举起/保持条件和节点匹配在 **500 Hz 物理子步**连续成立
0.8 秒。任何一次目标不匹配都会重置任务保持计时。任意节点成功在另一计时器中
记录为 `any_node_success`，不会提前终止指定节点任务。物理接触确认与外力不变。

## 训练

从匹配动作接口、网络结构的原有 state99 checkpoint 迁移：

```bash
PYTHONPATH=src python -m panda_cable_grasp.rl.train \
  --target-node-task --target-node-tolerance 2 \
  --warm-start /path/to/state99/final_model.zip \
  --robot nero --action-mode task_space_vertical_down --observation-mode state \
  --training-scenarios id_static:0.3,id_static_midshape_v1:0.3,id_rigid_l1_nominal:0.1,id_shape_nominal_current:0.1,id_combined_l1_nominal:0.1,id_rigid_replay_shape_nominal:0.1 \
  --timesteps 3000000 --workers 16 --torch-threads 1 --device cpu \
  --n-steps 1024 --batch-size 512 --n-epochs 5 \
  --learning-rate-initial 1e-4 --learning-rate-final 3e-5 \
  --entropy-coef-initial 0.005 --entropy-coef-final 0.001 \
  --curriculum-stage-steps 0 --eval-freq 0 --disable-singularity-avoidance \
  --checkpoint-steps 100000 --output outputs/rl/train/target_node_run
```

`--warm-start` 将 actor/critic 首层由 99 列扩展到 109 列，新增 10 列为零；
其余兼容参数完全复制。优化器、采样步数和学习率计划重新初始化。源 checkpoint
旁必须有记录动作接口的 `training_config.json`；不匹配时拒绝迁移。
`--resume` 用于已有的 109 维模型续训，须同时指定相同的目标任务参数；两者互斥。

此轮直接覆盖整个中部区域，不自动收紧容差。后续若改为 ±1，应单独记录训练阶段。
`--target-u-min`、`--target-u-max` 可显式调整训练目标范围。

## 评测与接口

```bash
PYTHONPATH=src python -m panda_cable_grasp.rl.evaluate \
  --model outputs/rl/train/target_node_run/final_model.zip \
  --robot nero --headless --no-video --scenario id_static \
  --episodes 50 --seed 20260804 --target-node-index 7
```

评测从 checkpoint 的训练 manifest 恢复目标范围、容差及奖励参数。省略
`--target-node-index` 时按固定评测种子采样；指定它时使用固定材料节点。
Gym 调用者也可使用 `reset(seed=..., options={"goal_node_index": 7})`；
显式固定节点可以用于范围外测试，但必须是有效的零起始材料索引。

训练 CSV 和评测 CSV 记录目标索引、材料误差、目标成功、任意节点成功以及错误抓取。
评测另记录实际抓取索引、目标匹配和材料坐标。误差为空表示当时没有抓取候选，
不能把空值当成零。成功率应按场景和指定目标分层，不能仅报告合并成功率。
已有 `tools/rl/eval_paired_bank.py` 可原样用于独立形变库的配对评测。
