# 当前状态

更新日期：2026-08-22。

## 已完成

- 项目改为 `src/panda_cable_grasp/` 包布局，按环境、策略、场景、RL、评估、专家和 DynamicVLA 分层。
- MuJoCo XML 移至 `assets/mujoco/`，训练配置移至 `configs/`。
- 所有生成结果统一迁移到 `outputs/`，原数据未删除，映射记录见 `outputs/migration_manifest.json`。
- 根目录旧命令和历史 Python 导入兼容层已退役，统一使用安装后的
  `panda-cable-*` 命令或 `panda_cable_grasp.*` 包导入。
- PPO 姿态奖励改为“夹爪闭合轴与局部绳子切向垂直”，只在接近可抓取线段时给予有效势能。
- 夹爪恢复迟滞开关；捕获区内闭爪获得一次性奖励，保持张开或过早闭爪分别受到轻量逐步惩罚。
- 有效抓取必须在双指夹持后实际抬升 30 mm 并稳定 0.10 s，才进入 secured 状态。
- 对“捏住但不抬升”加入延迟且封顶的停滞惩罚；未承载夹持结束时加入事件惩罚，避免持续刷夹持奖励。
- 主动张爪、物理滑脱和接触丢失后张爪分别记录，方便因果诊断。
- 本轮代码重组后的全量测试基线：78 项通过。

## 尚未完成

- 新奖励版本还没有正式长训练结果，不能用旧 PPO v2 的成功率评价本次修正。
- 需要从头训练 `outputs/rl/train/ppo_dlo_baseline_v4/`，然后用独立 seed 做严格评估。
- 先比较 aligned-pinch、secured-grasp、loaded-lift 和 strict-success 四级指标，再看总 reward；只看 episode return 无法判断抓取姿态是否正确。

## 推荐下一次训练

```bash
panda-cable-rl-train \
  --workers 12 \
  --eval-workers 4 \
  --timesteps 2000000 \
  --training-distribution l1 \
  --eval-distribution l1 \
  --output outputs/rl/train/ppo_dlo_baseline_v4 \
  --device cpu
```

服务器上先运行 10–20 万步 pilot。逐级提高 `--workers`，当 CPU 已接近饱和、内存压力明显或 steps/s 不再增长时停止增加。
严格评估默认使用 4 个独立进程；可通过 `--eval-workers` 调整，设为 1 时恢复串行评估。

## 评估重点

1. 首次 pinch 时 `alignment_score_at_first_pinch` 是否集中在高值。
2. `ever_aligned_pinch_rate` 提高后，`ever_secured_grasp_rate` 是否同步提高。
3. `lift_attempt_rate` 高但 `loaded_lift_rate` 低，通常意味着抓取仍会滑脱。
4. `failed_unloaded_pinch_count` 高，说明策略在利用“夹住不抬”的局部最优。
5. `active_open_after_secured` 与 `physical_slip_after_secured` 要分开分析。

奖励定义和预期验证见 [RL 基线说明](rl_baseline.md)，模块导航见 [架构说明](architecture.md)。完整历史实验上下文已归档到 [session_handoff_20260820.md](history/session_handoff_20260820.md)。
