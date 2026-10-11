# 规则方法动态场景与公共执行保护（2026-10-04）

用户要求查看规则方法在动态场景中的表现，沿用上一轮同一执行层限制。本次为独立诊断，不替换论文冻结结果，不修改 PPO 训练或奖励。

## 方法和来源

- 固定规则状态机使用前一轮 `TargetFull6Rule` 适配器：指定材料节点、现有接近/下降/闭爪/抬升/恢复流程。
- Full6 DLS JIK、姿态增益 10、阻尼 0.04、线速度上限 0.9 m/s、角速度指令上限 1.4 rad/s、关节目标预测 0.11 s；臂 kp×4、kv×2、500 Hz 重力前馈。
- 保护算法和参数逐项比对前一轮规则实验；保护仅在 protected 条件启用。监控 link7、flange、palm 与两个手指；下臂未纳入保护，不声明形式安全保证。
- 使用 PPO RUN 的 frozen_code、执行模块和课程环境；强制指定动态场景，stage 4 的 motion=1，单场景 one-hot 分布，禁止掺入静态场景。
- 三类：`id_rigid_l1_nominal`、`id_shape_nominal_current`、`id_combined_l1_nominal`。`shape_nominal_current` 是旧模型兼容形变条件，不代表完整实验矩阵。
- 每类 4 组同种子配对，共 12 对/24 回合；第 e 组种子 20260804+e，场景按 e%3 轮换。原执行层/保护层的实际 seed、场景、初始 FULLPHYSICS SHA、目标节点和配置必须相同。reset 有回退时记录 actual_seed，不假定与请求值相同。
- 15 秒回合、原始目标成功容差 ±4；奖励优化点 1 未实施。规则不读取奖励，累计奖励不用于跨方法比较。
- 目标仅在 reset 的工作区内候选节点中采样，跟踪同一材料身份；本次不新增“动态全程在工作区内”的筛选。报告抓前工作区内比例，避免把不可达时段隐藏。

151 上传运行脚本起初被自动审批拒绝（未确认该批源码的目的地传输授权），等待期间本机已完成 24 回合。用户随后授权上传并运行，但又明确要求采用已完成本机结果、不要重复；因此已终止 151 的重复主进程 2268550 及全部专属 worker，并确认无该实验进程。151 的部分产物不进入本次统计。最终结果全部来自本机：Windows、MuJoCo 3.11.0（与 151 相同）、Torch 2.14.0+cu126；本次不使用神经网络推理。静态实验平台为 Linux，不能把跨平台细小数值差异当作保护效果；本次配对两条件均在同一本机运行。

输入 RUN：`outputs/rl/train/ppo_targetnode_full6_workspace_20261004/`；回放库 `outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks/`。实际执行代码 SHA 记录在每回合 metadata 与 source_manifest；工作区原有修改保留，新增脚本未提交。

## 结果

| 场景 | 原执行层成功 | 保护成功 | 原执行层撞桌 | 保护撞桌 |
|---|---:|---:|---:|---:|
| 整体运动 | 4/4 | 4/4 | 0/4 | 0/4 |
| 形变 | 2/4 | 1/4 | 0/4 | 0/4 |
| 运动＋形变 | 1/4 | 1/4 | 2/4 | 0/4 |
| 总计 | 7/12 | 6/12 | 2/12 | 0/12 |

总体成功率 95% Wilson 区间：原执行层 7/12 约 31.95%–80.67%，保护 6/12 约 25.38%–74.62%。逐场景仅 4 个样本，不将单个回合差异解释为确定的方法优势。

形变 seed 20260814 是唯一成功状态退化的配对：原执行层 4.00 秒成功，保护 15 秒超时。整体运动 seed 20260810 两条件均成功，但用时 4.22 → 14.56 秒。运动＋形变 seed 20260809 两条件均成功，原执行层发生桌面接触，保护无接触，用时 5.66 → 4.40 秒。seed 20260812 原执行层发生接触且未确认首次抓取，两条件均提前结束，失败原因以 metadata 为准。

保护消除了本组检测到的接触，但不能仅凭 12 对声称普遍安全；成功数从 7 降到 6，说明需要同时看成功率和安全干预的代价。若目标大部分离开工作区，当前任务仍保持同一材料节点而不重选。形变 seed 20260805 的抓前工作区内比例原执行层 5.33%、保护 5.86%，两者均无首次抓取且超时；工作区可用机会不足可能参与失败，不能把所有失败都归因于保护。每类仅 4 组，且未与动态 PPO 做同场景对照，不用于最终方法排名。

## 产物与复现命令

本地目录 `outputs/rl/eval/scripted_dynamic_table_guard_20261004/`，`paired/` 保存全部轨迹、时序、元数据和视频，`code/` 保存运行及页面工具。

同步视频页面 <http://127.0.0.1:8771/index.html>，覆盖三类场景的首组与补充成功例，共 5 对 × 2 视角 = 20 段。视频标注黄色目标、绿色 ±4 范围、目标速度和 WS IN/OUT（目标当前是否位于测得工作区）。原始回合不重新仿真；录像是保存轨迹重放。浏览器已核查两路播放、全局/腕部切换、拖动暂停与恢复，截图 `paired/gallery_verified.jpg`。

```powershell
$env:FULL6_PPO_RUN = Join-Path (Get-Location) 'outputs/rl/train/ppo_targetnode_full6_workspace_20261004'
$env:MUJOCO_GL = 'glfw'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python outputs/rl/eval/scripted_dynamic_table_guard_20261004/code/test_scripted_table_guard.py --output NEW_DIRECTORY --ppo-directory outputs/rl/eval/table_guard_20261004_step2433024/paired --episodes 12 --workers 4 --verify-first-baseline --dynamic-scenes
```

首个动态原执行层回合另去掉 Observer 完整复跑，确认统计无干扰。每帧重放保存的实际 20 ms 状态，原生 1280×960、50 fps，全局与腕部；H264 CRF16、GOP12、无 B 帧，不插帧或改变仿真。每类首组优先展示，另补充该类存在的不同成功/失败结果；全部 24 回合进入统计，不用挑选录像替代统计。
