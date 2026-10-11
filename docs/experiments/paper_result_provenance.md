# 论文结果与原始运行映射

> - 核对日期：2026-09-29
> - 当前论文：项目根目录 `docs/论文草稿/Formatting_Instructions_for_ICLR_2027_Conference_Submissions (2).pdf`
> - 原始产物服务器：151，`/data1/hxai/mujoco/mujoco_dynamic_DLO`
> - 核对时服务器分支 / commit：`dev` / `9ce2768ef09d16aaada50f0855376160f8e1e454`

本文记录当前论文表格中的数值来自服务器上的哪一次评测。它解决的是结果溯源问题，
不替代[实验协议](../reference/experiment_protocol.md)，也不把论文之后的新运行自动升级为论文结果。

核对规则如下：

1. 先从当前论文 PDF / LaTeX 读取完整的逐场景成功率向量；
2. 在服务器逐回合产物中匹配全部场景，而不是根据目录名称或日期猜测；
3. 使用公共 `task_success` 计数；
4. 论文之后的 checkpoint 或复评结果单列，除非论文被明确修订，否则不得覆盖本页的论文映射。

## 1. 论文四场景主表

每个场景 50 回合，场景顺序为 static、shape、rigid、combined。

| 方法 | 成功数 | 成功率 | 论文平均 |
|---|---:|---:|---:|
| Point-cloud PPO | 44 / 15 / 50 / 17 | 88% / 30% / 100% / 34% | 63.0% |
| Diffusion Policy | 37 / 6 / 47 / 16 | 74% / 12% / 94% / 32% | 53.0% |
| π0.5 | 43 / 9 / 50 / 17 | 86% / 18% / 100% / 34% | 59.5% |
| DynamicVLA | 39 / 19 / 50 / 22 | 78% / 38% / 100% / 44% | 65.0% |

### Point-cloud PPO

论文的四个单元格来自四个录制目录，逐目录合并五个 `shard*/episodes.json`，按
`task_success == true` 计数：

- `outputs/rl/eval/ppo_pc384_static50_recording_20260923` → 44/50；
- `outputs/rl/eval/ppo_pc384_shape50_recording_20260922` → 15/50；
- `outputs/rl/eval/ppo_pc384_rigid_l1_nominal50_recording_20260923` → 50/50；
- `outputs/rl/eval/ppo_pc384_comb50_recording_20260923` → 17/50。

这些录制目录保存了逐回合 JSON、轨迹和视频，但目录顶层没有独立 manifest，因而不能仅从
这四个目录恢复 checkpoint 哈希。结果计数与论文精确一致；checkpoint 身份仍需从启动记录或
上游运行清单补档，不能根据相邻目录推断。

### Diffusion Policy

统一根目录：
`outputs/diffusion_policy/evaluation/image_dp168_step137k_strict_4x50/`。

四个子目录 `id_static`、`id_shape_nominal_current`、`id_rigid_l1_nominal`、
`id_combined_l1_nominal` 的 `summary.json` 分别给出 37、6、47、16 次成功。manifest 记录：

- checkpoint：`outputs/diffusion_policy/image_dp168_crop160_step137k_strict_20260919/checkpoint_best.pt`；
- checkpoint SHA256：`5651fd3e6d3692b1d6e256b583ae070158c0ba1459fcfc490db04b7067af9963`；
- 双相机分辨率 480×360，diffusion inference steps = 100。

### π0.5

论文每个场景的 50 回合由同一冻结 checkpoint 的 20 回合首批和 30 回合补跑拼接而成：

| 场景 | 20 回合目录 | 30 回合目录 | 合计 |
|---|---|---|---:|
| static | `pi05_newcam_parallel_static_20260916`（16） | `pi05_newcam_parallel_static_add30_20260916`（27） | 43/50 |
| shape | `pi05_newcam_parallel_shape_20260916`（4） | `pi05_newcam_parallel_shape_add30_20260916`（5） | 9/50 |
| rigid | `pi05_newcam_parallel_rigid_20260916`（20） | `pi05_newcam_parallel_rigid_add30_20260916`（30） | 50/50 |
| combined | `pi05_newcam_parallel_combined_20260916`（6） | `pi05_newcam_parallel_combined_add30_20260916`（11） | 17/50 |

以上目录均位于 `outputs/pi05/evaluation/`。manifest 共同记录 checkpoint
`/data/hxai/panda_cable_pi05/checkpoints/pi05_nero_cable_lora_full/nero_wrist_pandalike_b32_h16_20260913/29999`，
同步无延迟执行、`execute_steps=8`、new camera，代码 commit
`3c9e4c0b58acc80f6b3def45f5557b2045f8092b`（运行时 dirty）。首批 seeds 为
20280804–20280823，补跑 seeds 为 20280824–20280853。

### DynamicVLA

统一根目录：
`outputs/dynamicvla/d2_epoch0001_poseik_oldcam_50x4_20260914/`。

四个 `d2_epoch0001_poseik_oldcam_{static,shape,rigid,combined}_50x_20260914`
子目录分别给出 39、19、50、22 次成功。manifest 记录 epoch 1、同步执行、
`execute_steps=8`、old wrist camera、Nero pose IK，代码 commit
`e91852facc41e951194dd28dcb216556c40a8cf8`（运行时 dirty）。manifest 的方法字段仍为
`dynamicvla_zero_shot`，且没有保存远端模型 checkpoint 路径 / 哈希；论文行应按该运行目录和
epoch 标识，不应从字段名推断成原始零样本权重。

## 2. 论文 OOD 表

每个条件 50 回合；列顺序为 amplitude low/high、frequency low/high、length low/high、
material low/high。

| 方法 | 成功数（8×50） | 论文平均 |
|---|---:|---:|
| Point-cloud PPO | 24 / 17 / 17 / 16 / 12 / 20 / 22 / 15 | 36.0% |
| Diffusion Policy | 8 / 11 / 6 / 9 / 9 / 3 / 7 / 7 | 15.0% |
| π0.5 | 22 / 18 / 8 / 10 / 17 / 10 / 19 / 17 | 30.3% |
| DynamicVLA | 17 / 13 / 17 / 10 / 14 / 4 / 24 / 17 | 29.0% |

对应原始目录：

- Point-cloud PPO：`outputs/rl/eval/ppo_pc384_ood8x50_recording_20260923/`；
  合并 `shard*/episodes.json` 后按 `scenario` 分组。
- Diffusion Policy：
  `outputs/diffusion_policy/evaluation/image_dp168_strict_ood8x50_20260922/`。
- π0.5：`outputs/pi05/evaluation/pi05_newcam_ood50_{amplitude,frequency,length,material}_{low,high}/`。
- DynamicVLA：
  `outputs/dynamicvla/d2_epoch0001_poseik_oldcam_ood8x50_20260922/`。

## 3. 论文之后的运行

以下结果用于跟踪后续研究，不属于当前论文主表：

- DynamicVLA `opt5820_paired50_*_20260928_105831`：static 40/50、shape 14/50、
  rigid 42/50、combined 15/50、replay-shape 22/50。
- π0.5 `outputs/pi05_5scenes_s29999_paired50/`：static 43/50、shape 10/50、
  rigid 46/50、combined 20/50、replay-shape 14/50。
- State PPO height-band paired run
  `outputs/rl/eval/ppo_heightband6_paired50_20260928_010749/`：before 201/300，
  after 215/300；after checkpoint SHA256
  `4ba154c80334cd35efef32e44661326af327c8752aa5ed0f0722b162020e4d61`。
  这是 99 维特权状态 PPO 的六场景实验，不是论文中的 point-cloud PPO 行。
- Target-node PPO 首轮
  `outputs/rl/eval/ppo_targetnode_firstattempt_paired50_20260928_223100/` 是抓取目标节点
  诊断实验，也不能替代论文 point-cloud PPO 行。

更新论文表格时，应同时修改论文 LaTeX、本页、项目状态和对应方法实验文档，并保留旧运行映射，
使“论文采用哪次结果”始终可以从逐回合产物复核。
