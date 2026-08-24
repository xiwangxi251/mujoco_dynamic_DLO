# DynamicVLA 实验记录

本文档记录 DynamicVLA 在线缆抓取四个固定场景上的实验设置与主要结果。

> 更新日期：2026-08-24

## 1. 实验设置

### 1.1 场景

| 场景 | 说明 |
|---|---|
| `id_static` | 静态线缆 |
| `id_rigid_l1_nominal` | L1 单程整体平动与旋转，保持线缆形状 |
| `id_shape_nominal_current` | 局部准周期形变 |
| `id_combined_l1_nominal` | L1 整体运动与局部形变同时存在 |

### 1.2 推理设置

```text
instruction = Pick up the blue cable.
protocol    = official_zmq_schema
action      = delta action
control_hz  = 25 Hz
rotation    = quaternion on wire (wxyz)
```

实验结果目录：

```text
outputs/dynamicvla/evaluation/finetune_latest_static
outputs/dynamicvla/evaluation/finetune_latest_rigid
outputs/dynamicvla/evaluation/finetune_latest_shape
outputs/dynamicvla/evaluation/finetune_latest_combined
outputs/dynamicvla/ckt_4x50
```

## 2. Finetune 权重评测

模型服务标识为 `cable-four-scenarios-200-latest`，对应四场景微调权重。每个场景运行 10 回合，seed 为 `20260804`。

| 场景 | 成功数 | 成功率 | Wilson 95% 区间 |
|---|---:|---:|---:|
| Static | 1/10 | 10% | 1.8%–40.4% |
| Rigid L1 | 0/10 | 0% | 0%–27.8% |
| Shape | 1/10 | 10% | 1.8%–40.4% |
| Combined L1 | 0/10 | 0% | 0%–27.8% |
| **总体** | **2/40** | **5%** | **1.4%–16.5%** |

失败模式汇总：

| 失败模式 | 次数 |
|---|---:|
| 从未形成双指候选接触 | 31 |
| 确认抓取后物理滑脱 | 5 |
| 双指候选接触未确认 | 1 |
| 确认抓取后主动张开 | 1 |

40 回合中约 9 回合形成双指候选、8 回合完成抓取确认，确认抓取后成功率为 `2/8=25%`。总体首要瓶颈是接近线缆并形成双指接触。

Shape 场景有 7/10 形成双指候选、6/10 完成抓取确认，但发生 4 次确认后物理滑脱，因此该场景还存在明显的抓取保持问题。

## 3. `ckt` 权重初步评测

加载权重：

```text
/data1/hxai/mujoco/DynamicVLA/ckt
```

seed 为 `20280804`，每回合最长 15 s。

| 场景 | 成功数 | 终止原因 | 失败诊断 |
|---|---:|---|---|
| Static | 0/1 | timeout | 从未形成双指候选接触 |
| Rigid L1 | 0/1 | motion boundary | 从未形成双指候选接触 |
| Shape | 0/1 | timeout | 从未形成双指候选接触 |
| Combined L1 | 0/1 | motion boundary | 从未形成双指候选接触 |
| **总体** | **0/4** |  | **成功率 0%** |

目录虽然命名为 `ckt_4x50`，但四个 manifest 均设置为 `--trials 1`，因此实际只有每场景 1 回合、总计 4 回合。这是 smoke test，不是完整的 4×50 评测；`0/4` 的 Wilson 95% 区间约为 0%–49%，不能据此可靠比较 `ckt` 与 finetune 权重。

## 4. 结论

1. 四场景 finetune 权重总体成功率为 5%，主要失败发生在形成双指接触之前。
2. Shape 场景较容易形成和确认抓取，但确认后的物理滑脱较多。
3. `ckt_4x50` 当前只完成 4 次 smoke test，需按每场景 50 回合重新评测后才能得出结论。
4. 四个 finetune 结果由不同 Git commit 且 dirty 工作树下的评测代码产生，后续正式对比应固定同一代码版本、seed 和评测参数。
