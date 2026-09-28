# pi0.5 动态线缆基线实验计划

本文只安排 pi0.5：数据准备、LoRA 训练、在线推理适配、checkpoint 选择和统一 benchmark
评测。不安排 Diffusion Policy、DynamicVLA、PPO 或其他方法的重训。

pi0.5 定位为通用预训练 VLA 基线。若表现较好，只能说明预训练和模型规模有效，不能声称它显式
建模了线缆形状或未来可抓区域。

## 1. 资源与隔离

- 训练服务器：`jump130`；
- 唯一允许的设备：物理 GPU1，A100 40GB；
- 所有命令显式设置 `CUDA_VISIBLE_DEVICES=1`；
- GPU0 属于其他任务，实验前后必须确认其进程和显存占用未变化；
- 实验根目录：`/data/hxai/panda_cable_pi05`；
- cache、checkpoint、数据和日志都写入 `/data`，不占用服务器根分区；
- OpenPI 使用私有副本，不修改共享 checkout 或共享 Python 环境。

后台链路已经配置为：官方 `pi05_base` 下载并校验完成后，自动启动 200-step pilot。状态检查：

```bash
tmux list-sessions | grep -E 'pi05_base_download|pi05_pilot_waiter|pi05_cable_pilot'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail -f /data/hxai/panda_cable_pi05/logs/pilot_b8_h16_20260902_r1.log
```

## 2. 数据定义

数据集为 `panda_cable_4scenes_1000_dynamicvla_parallel`，LeRobot v2.1、25 Hz，共 4000
条成功轨迹和 871645 帧：

| 场景 | 数量 | episode index |
| --- | ---: | --- |
| `id_static` | 1000 | 0--999 |
| `id_rigid_l1_nominal` | 1000 | 1000--1999 |
| `id_shape_nominal_current` | 1000 | 2000--2999 |
| `id_combined_l1_nominal` | 1000 | 3000--3999 |

每个样本包含：

- `observation.images.opst_cam`；
- `observation.images.wrist_cam`；
- 6D 末端状态 `xyz + Euler`；
- 7D 绝对动作 `xyz + Euler + gripper`；
- 固定任务指令 `Grasp and lift the blue cable.`。

数据只覆盖 L1，因此 pi0.5 的主 ID 评测也采用 L1。L2 只能作为未见整体轨迹的额外泛化条件，
不能混入 ID 平均值。

## 3. 无泄漏 train/dev 切分

四个场景内部 seed 都不重复，但不同场景会复用相同 seed。相同 seed 对应相关的初始曲线和随机
因素，所以不能简单使用“每场景前 900 条训练、后 100 条验证”。正式切分规则为：

```text
bucket = uint64(sha256("panda-cable-pi05-v1:" + str(seed))[:16], 16) % 10
bucket 0..8 -> train
bucket 9    -> imitation_dev
```

同一个 seed 在所有场景必须落入相同分区。预期约 3600 条 train、400 条 imitation-dev；实际
数量以生成后的 manifest 为准，不为凑整数人工移动样本。需要保存：

- `train_episodes.json`；
- `imitation_dev_episodes.json`；
- 切分规则、场景计数、帧数和 SHA-256；
- 数据根目录及 `cable_conversion_manifest.json` 的哈希。

归一化统计只允许使用 train。正式 simulator seed 必须完全落在数据采集 seed 范围之外。

## 4. 模型与动作接口

正式候选配置：

- 官方 `gs://openpi-assets/checkpoints/pi05_base/params`；
- `pi05=True`；
- `gemma_2b_lora` + `gemma_300m_lora`；
- prediction horizon 16；
- batch 8，OOM 时降到 4；
- 无 EMA、无 W&B；
- AdamW，gradient norm 1.0；
- 1000-step warmup 到 `5e-5`，30k steps cosine decay 到 `5e-6`；
- 每 2500 steps 保存 checkpoint。

训练时对 `xyz + Euler` 前 6 维做相对当前 state 的 delta；夹爪维保持绝对值。在线输出需要用
当前 state 恢复绝对动作，再把 Euler 转成 `quaternion_wxyz`，最后复用项目现有 DynamicVLA
IK、安全裁剪和机器人运动限制。

模型预测 16 步，但在线最多执行前 8 步后重新推理，最大开环时间 0.32 s。必须记录实际推理
延迟、deadline miss 和动作裁剪率。

## 5. 分阶段执行

### G0：数据契约检查

从四个场景各取 8 个固定样本，验证：

1. opst 和 wrist 两个图像槽有效，未使用的第三图像槽 mask 为 false；
2. state 为 6D，action 为 `16 × 7`，padding 后模型 action 为 `16 × 32`；
3. prompt token 非空；
4. state/action 没有 NaN 或 Inf；
5. `DeltaActions` 和 `AbsoluteActions` 往返误差接近 0；
6. Euler 角不存在未处理的 `-pi/pi` 跳变；若存在，训练前先 unwrap。

G0 失败时只修数据和坐标接口，不启动正式训练。

### G1：200-step 资源 pilot

- 数据：四场景各 10 条，共 40 条；
- norm stats：平衡抽样 2000 帧；
- batch 8、200 steps、每 10 steps 记录一次；
- 目标：测显存、首步编译时间、稳定 step/s 和 checkpoint 保存/恢复。

通过标准：GPU0 不变、GPU1 无 OOM、loss 全程有限、至少完成一个 checkpoint。batch 8 OOM 时
改 batch 4 重新运行；其他错误先定位，不通过缩短 horizon 或改数据掩盖。

### G2：2500-step 学习 smoke

- 使用正式 hash train split；
- 使用只读 parquet 的 state/action 计算 full norm stats，避免为统计量解码约 87 万帧视频；
- 训练 2500 steps；
- 在四个 L1 core 场景各运行 10 个新 seed，共 40 回合。

若静态场景完全不能闭合/抬升，优先检查坐标系、夹爪符号、delta 恢复和 IK，不直接跑 30k。
如果静态能工作但动态场景失败，可进入正式训练，因为这可能是模型真实能力差距。

### G3：30k LoRA 正式训练

保存 2.5k、5k、10k、15k、20k、25k、30k checkpoint。训练日志至少包含：

- step、loss、learning rate、step time；
- 峰值/当前 GPU 显存；
- 数据加载等待比例；
- 累计样本数和 GPU 小时；
- Git/OpenPI/config/data hashes。

训练 loss 不能直接决定最终 checkpoint。

### G4：dev checkpoint 选择

固定 dev simulator seeds：`20271804..20271823`。对候选 checkpoint 先做分层筛选：

1. 2.5k、5k、10k、20k、30k 在四个 L1 core 场景各 10 回合；
2. 取成功率最高的两个 checkpoint；
3. 两者各在四场景补到 20 回合；
4. 按整体 `task_success` 选择；并列时依次比较 combined、shape 成功率和动作饱和率。

选择规则执行一次后冻结。不得使用 OOD 结果选择 checkpoint。

### G5：正式 pi0.5 评测

只运行冻结的 pi0.5 checkpoint：

| 集合 | 场景 | 回合数 |
| --- | ---: | ---: |
| core pilot | static、rigid L1 nominal、shape nominal、combined L1 nominal | 4 × 10 = 40 |
| formal ID | static；shape、rigid L1、combined L1 的 low/nominal/high | 10 × 100 = 1000 |
| formal OOD | 幅度、频率、长度、材质各 high/low | 8 × 100 = 800 |
| L2 附录 | rigid/combined L2 nominal | 2 × 100 = 200 |

formal 使用从 `20280804` 开始冻结的 100 个 paired seed。pi0.5 可以先独立生成上述结果；只有
其他基线已有完全相同场景、seed、代码/XML 和成功判定的结果时，才做 paired 方法比较。本文不
安排其他方法的补训或补跑。

## 6. pi0.5 专属消融

只保留三个与 pi0.5 接入直接相关的有限消融，并且先在 dev 运行：

1. batch 8 vs batch 4：仅在 batch 8 不稳定或 OOM 时比较训练稳定性；
2. execute horizon 4 vs 8：检验动态场景的开环时间；
3. 双相机 vs 仅 opst camera：检验腕部相机对接触和遮挡阶段的作用。

不做大规模学习率、prompt 或 LoRA rank 网格搜索。消融不得查看 formal OOD 后再新增。

## 7. 指标与归档

唯一主指标是公共 `task_success`。每个场景报告：

- 成功数/总数及 95% Wilson 双侧区间；
- 首次双侧候选、首次确认抓取、成功时间、峰值抬升；
- 动作饱和率、推理延迟、deadline miss；
- 六类互斥失败原因；
- 确认抓取后物理滑脱在失败中的比例及单侧 95% Wilson 上界。

每次正式运行归档 checkpoint、norm stats、episode/seed manifest、逐回合 CSV、视频、场景及
motion hash、代码/XML/OpenPI 哈希、完整命令和环境版本。禁止覆盖旧目录或只保留汇总均值。

## 8. 单 GPU1 排期与结果定义

G1 完成后使用真实 `seconds/step` 更新训练 ETA。当前保守预算：

| 阶段 | 预计时间 |
| --- | ---: |
| base checkpoint 下载 | 受网络影响，后台完成 |
| G1 200-step pilot | 15--45 分钟 |
| 全量数据同步、切分、哈希 | 1--3 小时 |
| parquet-only norm stats | 15--60 分钟 |
| G2 2500 steps + 40 回合 | 1--3 小时（待 pilot 校准） |
| G3 30k LoRA | 暂估 10--24 小时 |
| G4 checkpoint dev 选择 | 3--8 小时，取决于推理速度 |
| G5 formal ID + OOD | 理论仿真时间下限 7.5 小时，实际通常更长 |

“pi0.5 出结果”分三级：

1. **训练链路结果**：G1 完成并保存可恢复 checkpoint；
2. **可比较 pilot**：选定 checkpoint 完成 40 回合 core pilot；
3. **论文结果**：冻结 checkpoint 完成 1000 回合 formal ID 和 800 回合 OOD，并通过归档检查。

## 9. 运行入口

pilot 启动：

```bash
bash /data/hxai/panda_cable_pi05/run_pi05_cable_jump130.sh \
  pi05_cable_lora_pilot \
  pilot_b8_h16_20260902_r1 \
  pi05_cable_pilot_20260902_r1
```

正式训练必须在 hash split 和 full norm stats 落盘后才能启动：

```bash
bash /data/hxai/panda_cable_pi05/run_pi05_cable_jump130.sh \
  pi05_cable_lora_full \
  full_b8_h16_seedhash_v1 \
  pi05_cable_full_seedhash_v1
```

当前远端 `pi05_cable_lora_full` 仍是候选的连续 episode 切分配置；在替换为本文 seed-hash
episode 清单并校验前，禁止运行正式命令。

## 10. 本地 OpenPI 兼容修改

私有 OpenPI 副本目前包含三项数据兼容修改：

1. 支持显式本地 `dataset_root`；
2. 修复 LeRobot v0.3.x 对稀疏全局 episode ID 的取样索引；
3. 使用已有 PyAV 后端，绕过当前环境中不可用的 TorchCodec/FFmpeg 组合。

这些修改只服务 pi0.5 实验，正式运行时记录补丁哈希。
