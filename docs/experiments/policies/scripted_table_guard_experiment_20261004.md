# 规则方法使用与 PPO 相同的执行层限制

日期：2026-10-04。用户要求：用规则方法，使用相同的执行层限制。

## 方法和实验条件

规则策略使用训练运行所冻结的 `DynamicCableGraspPolicy` 状态机：等待、接近、下降截获、闭爪、抬升、侧移、保持以及原有恢复重试。

针对本次指定节点任务做明确适配：固定追踪与上一轮 PPO 相同的目标材料节点，闭爪距离在同一个 ±4 目标窗口内计算；下降时锁定指定节点，不改抓取接触或成功门槛。使用与 PPO 初始状态相同的竖直夹爪姿态。规则的期望位置误差乘位置增益 6，再传入公共 Full6 解算；不用 PPO 权重或奖励决策。

执行层与[上一轮 PPO 保护实验](ppo_table_guard_experiment_20261004.md)相同：

- 同一个冻结 `CourseEnv`、电缆物理、关节/夹爪速度限制、已有静态手指位置投影。
- `solve_full6`：DLS 阻尼 0.04、姿态增益 10、角速度 cap 1.4 rad/s、线速度 cap 0.9 m/s、关节目标时域 0.11 s、ready 拉回增益 0.8。
- 臂伺服 `kp×4、kv×2`，500 Hz 有界重力前馈和同一力矩范围。
- 同一个 500 Hz 近桌面保护源码及全部参数：15 cm 启用、1 mm 余量、距离增益 5 /s、制动参数 2 m/s² 与 20 ms、速度反馈增益 2、投影 12 次上限。
- 未增加关节加速度或 jerk 滤波，未改变模拟实际 qpos/qvel。

16 组原执行层 / 保护层配对，共 32 回合；种子 20260804–20260819。静态 stage 0，每回合最长 15 秒，满足原物理成功条件与材料节点判定即结束。每个回合初始 FULLPHYSICS 状态 SHA、实际种子、目标节点及初始目标位置与上一轮 PPO 完全匹配。环境和执行源码、目标判断、配置及保护参数也逐一核对一致。

成功范围沿用 **±4**。先前记录的成功 ±2 / 首次位置正奖励 ±4 优化没有用于本次实验。规则通过原任务 wrapper 的原生关节指令适配执行；5 维 RL 输入槽为零占位，只保留任务判断和奖励记账。规则不使用奖励，**累计奖励不作规则与 PPO 的比较**。

第一对先完成整回合和观测无干扰复跑，再使用 8 个 worker 补齐其余 15 对。没有根据结果调规则或保护参数。运行中的 PPO 训练未改。

## 结果

| 指标 | 规则原执行层 | 规则＋相同近桌面保护 |
|---|---:|---:|
| 首次抓取在当前目标容差内 | 15/16 | 16/16 |
| 完整任务成功 | 15/16（93.75%） | 16/16（100%） |
| 桌面接触回合 | 0/16 | 0/16 |
| 成功回合用时均值 | 4.263 s（15 个成功） | 4.751 s（16 个成功） |
| 全部回合最小离桌距离 | +10.57 mm | +14.08 mm |
| 保护介入比例，按物理步加权 | 0% | 8.48% |

成功用时均值的样本集合不同；共同成功的 15 对：原执行层均值 4.263 秒，保护层 4.723 秒，配对平均增加 0.46 秒、中位增加 0.16 秒。第 3 对为明显减速例：6.28 → 9.66 秒。第 9 对 seed 20260812：原执行层停在 CLOSE，15 秒超时，未形成首次确认抓取；保护层 5.18 秒成功。因此新增一个成功回合，不能据单个样本断言保护普遍提升成功率。

95% Wilson 区间：规则原执行层成功 15/16 为约 71.67%–98.89%；保护层成功 16/16 为 80.64%–100%。两组接触 0/16 的区间均为 0%–19.36%。本次规则原执行层已无撞桌，不能把它的 0 接触归功于新增保护。

与此前冻结 2,433,024 步 PPO 的同场景诊断：

| 方法 | 执行层 | 撞桌 | 首次抓对 | 完整成功 |
|---|---|---:|---:|---:|
| PPO | 原执行层 | 16/16 | 16/16 | 0/16 |
| PPO | 相同保护 | 0/16 | 16/16 | 0/16 |
| 规则方法 | 原执行层 | 0/16 | 15/16 | 15/16 |
| 规则方法 | 相同保护 | 0/16 | 16/16 | 16/16 |

**结论：**相同执行限制下规则方法能完成这组指定节点静态任务；限制没有使任务不可完成。保护对规则方法的干预比当前 PPO 小（8.48% 对 54.82%），但仍会延长部分回合。当前 PPO 完整成功不足还需要分析其动作输出和执行适配。这是固定静态小样本诊断，不是整个 RL 方法与规则方法的最终排名，也未测试动态课程或真机。

## 核查和可视化

同步页面：<http://127.0.0.1:8770/index.html>。本地核查 32 条轨迹 SHA、20 段视频编码/帧间隔及 HTTP Range 均通过（206）；浏览器实测全局/腕部切换、同时间播放及拖动暂停后恢复正常。规则日志中的未定义角度在页面 JSON 中转换为 null，原始数据保留。截图 `paired/gallery_verified.jpg` 显示 seed 20260812 在 5.18 秒的对比。

- 无统计工具完整复跑第一组规则原执行层，198 个状态的物理轨迹、动作、观察及记账奖励全部精确一致，状态最大差 0。
- 几何雅可比校验复用相同模型和初始状态的上一轮独立结果，最大误差 1.85e-7。
- 统计接触仍为每 2 ms 的实际机器人–桌面穿透及正法向力；最低表面点的实际速度和 `mj_contactForce` 不变。
- 视频固定展示前四个连续种子，再补充唯一规则原执行层失败 seed 20260812 的同种子保护前后对照，共 5 对 × 2 视角 = 20 段；失败例是额外诊断展示，不替换 16 组统计。
- 原生 1280×960、50 fps、H.264 CRF16、GOP12、无 B 帧、yuv420p、faststart；每帧重放保存的实际 20 ms 状态，不插帧，短成功视频在末帧停止，不拉伸到失败回合长度。
- 标注黄色目标圆圈和绿色原 ±4 范围，保留规则阶段 HUD，无目标文字框或箭头。

服务器产物：`/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/eval/scripted_table_guard_20261004/`。本地对应 `outputs/rl/eval/scripted_table_guard_20261004/`。

归档状态：服务器已保存原始回合、运行/录像代码及全部视频；最新页面、统计报告、报告脚本、source_manifest 和浏览器核查保存在本地。向 151 补传这些派生文件被自动审批拒绝，理由是未确认源码/浏览器产物向该目标的额外归档传输授权；尚未补传。

`paired/summary.json`、`episodes.csv`、`paired_manifest.json` 为规则 32 回合统计与来源；`ppo_rule_comparison.json` 记录跨方法相同执行和初始条件核查；各 seed 目录保存轨迹、状态机时序和视频。`code/` 保存冻结代码。录像、观测及本地 HTTP/浏览器核查分别记录在 verification 文件中。

151 基础提交 `9ce2768ef09d16aaada50f0855376160f8e1e454`（dev），实际环境读 RUN 冻结代码；本地 `e41ff39eced3960baf813837c6a269d37eed5dc8`（feat/unified-jacobian-control-20261002），原有 dirty 工作区保留。新增脚本尚未提交，以各回合源码 SHA 和实验 source_manifest 为准。MuJoCo 3.11.0、Torch 2.7.1+cu126、SB3 2.7.0，CPU 单线程。

复现（151，独立新目录）：

```bash
export FULL6_PPO_RUN=/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/train/ppo_targetnode_full6_workspace_20261004
experiment=/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/eval/scripted_table_guard_20261004
previous=/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/eval/table_guard_20261004_step2433024/paired
python_runtime=/data1/hxai/miniconda3/envs/dynamicvla/bin/python
$python_runtime $experiment/code/test_scripted_table_guard.py --output $experiment/reproduction --ppo-directory $previous --episodes 16 --workers 8 --verify-first-baseline
$python_runtime $experiment/code/render_table_guard_experiment.py --directory $experiment/reproduction --episodes 4 --workers 4
$python_runtime $experiment/code/render_table_guard_experiment.py --directory $experiment/reproduction --start-episode 8 --episodes 1 --workers 2
```

实际运行先单独跑并核查第一对，再复制该对到 paired 后补其余 15 对。上述整批复现使用完全相同的种子、状态机和执行参数。
