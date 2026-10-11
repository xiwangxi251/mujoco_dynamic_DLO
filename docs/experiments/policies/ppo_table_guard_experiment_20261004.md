# 指定目标节点 PPO：公共执行层近桌面保护配对实验

日期：2026-10-04。用户授权：对讨论的执行层限制做实验并测试。

## 实验范围

- 检查点：`ppo_targetnode_full6_workspace_20261004/checkpoints/step_002433024.zip`，2,433,024 步。实验开始时选定，整个实验冻结。
- SHA256：`a5e6c85e71224b9738326b404c08eef99d5d77b33a1c4a803f4bf1eef6eb5863`。
- 场景：当前课程 stage 0，`static_workspace_all_nodes`。
- 种子：20260804–20260819，16 个连续留出种子；每个种子原执行层 / 保护层各一个完整 15 秒回合，共 32 回合。
- 前两对先验证，再补 14 对，后者使用 8 个评估 worker。参数没有根据测试结果调整。
- 每对初始 FULLPHYSICS 状态 SHA、实际种子、目标节点、初始目标位置、检查点、奖励源、目标判定源、训练配置 SHA 完全一致。
- 保留原有增强伺服、重力前馈、Full6 DLS、姿态增益 10、已有位置投影和速度限制。
- **奖励和成功判定均没有修改。此次沿用运行中的成功容差 ±4。此前记录的“成功 ±2、位置正奖励 ±4”优化尚未用于这次实验。**
- 这是新的执行层诊断实验，不替换论文中冻结的主实验数值。正在运行的训练未接入此保护。
- 151 基础 checkout：`dev`，commit `9ce2768ef09d16aaada50f0855376160f8e1e454`。实验使用 RUN 下冻结的环境与执行模块，未提交的独立保护脚本以各回合记录的源码 SHA 为准。开发工作区原已有未提交改动，未清理。
- 本地开发 checkout：`feat/unified-jacobian-control-20261002`，commit `e41ff39eced3960baf813837c6a269d37eed5dc8`；本次新增实验脚本及文档尚未提交。
- 仿真运行版本：MuJoCo 3.11.0、PyTorch 2.7.1+cu126、Stable-Baselines3 2.7.0；策略推理使用 CPU、单线程。

## 保护原型

入口是最终关节位置指令，接口不依赖 PPO 的动作、目标节点或奖励；以后其他方法可以使用同一个入口。

1. 根据机器人模型和当前关节状态计算 link7、法兰、夹爪基体、两根手指共 5 个去重网格的最低表面点到桌面距离。凸包顶点保留平面方向的最低点。
2. 距离小于 15 cm 时启用下降速度包络；安全余量 1 mm，距离增益 5 /s。
3. 下降速度上限为 `min(0.9, 5*max(d-0.001,0), sqrt((a*tau)^2+2*a*max(d-0.001,0))-a*tau)`，其中 `a=2 m/s²`、`tau=0.02 s`。这里的制动加速度是设计参数，不是已经验证的真机保证值。
4. 以 `Kp*(q_target-q)/Kv` 近似伺服速度平衡参考，对各表面点法向速度约束做关节空间投影。实际下降速度超过包络时，加入增益 2 的制动反馈。
5. 每个 2 ms 物理步执行一次，共 500 Hz。只修改关节位置指令；不直接修改实际 qpos/qvel、不移动桌面、不改变碰撞几何、伺服参数、力矩边界或电缆物理。

本原型**没有**加入关节加速度/jerk 滤波，也未覆盖更靠近底座的臂段。它是经过物理回放测试的伺服指令投影，不是具有形式化安全保证的 CBF。

## 结果

峰值的平均值：先取每个回合峰值，再在 16 个回合上平均。无接触回合的撞击速度、力和冲量记为 0。

| 指标 | 原执行层 | 近桌面保护 |
|---|---:|---:|
| 发生机器人–桌面接触的回合 | 16/16 | 0/16 |
| 每回合峰值撞击下降速度的均值 | 0.622 m/s | 无接触 |
| 全部回合最高撞击下降速度 | 0.706 m/s | 无接触 |
| 每回合峰值桌面法向力的均值（仿真） | 521.5 N | 0 N |
| 每回合桌面法向冲量的均值（仿真） | 32.765 N·s | 0 N·s |
| 每回合接触时长的均值 | 0.335 s | 0 s |
| 全部回合最小表面离桌距离 | −1.31 mm | +8.74 mm |
| 首次抓取在当前 ±4 容差内 | 16/16 | 16/16 |
| 每回合最大抬升电缆比例的均值 | 63.59% | 50.63% |
| 完整任务成功 | 0/16 | 0/16 |
| 回合累计奖励均值（原奖励） | 154.75 | 21.21 |

保护在 54.82% 的物理步干预，其中 19.90% 的物理步出现实际速度制动反馈。120,000 个保护步中，100 步在有限投影迭代/位置限位后仍有约束残差，已经保留统计，不能声称严格逐步满足所有约束。

小样本 95% Wilson 区间：原执行层接触 16/16 为 80.64%–100%；保护接触 0/16 为 0%–19.36%；两组首次抓对 16/16 为 80.64%–100%，完整成功 0/16 为 0%–19.36%。32 个回合均到达 15 秒超时上限，没有提前成功终止。这些区间也表明 0/16 不能证明所有场景零碰撞。

**结论：**这个固定参数原型在本次静态小样本中消除了桌面接触，且首次抓取精度未下降；现有模型接入后的抬升表现和累计奖励明显下降，完整任务成功也未改善。它支持“执行层能抑制撞桌”的可行性，尚不支持“直接接入后整个任务更好”或“真机已经安全”。应继续检查干预对姿态与抬升保持的影响，再决定参数调整及是否在同一公共执行层下训练。

如果用于方法比较，所有方法必须接入同样的保护、场景、种子及成功条件。本次只测试了这个 PPO，未验证其他方法。

## 测量与录像核查

- 接触定义：每个 2 ms 物理步，机器人–桌面 contact.dist < 0 且法向力 > 1e-6 N。
- 撞击下降速度：实际接触手指最低表面点在该物理步之前的法向速度，不是 PPO 请求速度或 TCP 速度。
- 接触力来自 MuJoCo `mj_contactForce`，多个手指接触相加；数值不是实机力传感器测量。
- 完整 751 状态的原执行层回合去掉观测工具后，动作、观察、奖励、状态全部精确一致，最大状态差 0。
- 几何表面点雅可比与独立 probe 的有限差分最大误差 `1.85e-7`；probe 未改实际 rollout 状态。
- 前四个连续种子全部提供全局和腕部视角，2 执行版本 × 4 种子 × 2 视角 = 16 段。
- 1280×960、50 fps、H.264 CRF16、GOP12、无 B 帧、yuv420p、faststart。
- 逐帧重放同一保存的 20 ms 物理状态，没有插帧；渲染状态差 0，源轨迹 SHA 不变；编码时间戳每 20 ms，最大关键帧间距 12 帧。
- 黄色圆圈标出指定节点，绿色显示本次实际采用的 ±4 范围，无节点文字框或箭头。

## 来源与复现

服务器目录：`/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/eval/table_guard_20261004_step2433024/`。

本地同名目录：`outputs/rl/eval/table_guard_20261004_step2433024/`。

- `code/`：冻结保护、原始 pilot runner、扩展 runner、观测验证、视频渲染代码。
- `paired/summary.json`：聚合指标、配对差值与小样本 bootstrap 区间、定义与限制。
- `paired/episodes.csv`：32 个回合逐回合原始汇总。
- `paired/paired_manifest.json`：完整配对来源核查；原 `manifest.json` 单独覆盖新增 14 对。
- `paired/measurement_verification.json`：观测无干扰与雅可比核查。
- `paired/video_manifest.json`、各回合 `video_verification.json`：16 段编码及轨迹核查。
- 各 `baseline/seed_*`、`protected/seed_*`：`trajectory.npz`、`metrics.json`、`metadata.json` 和前四组视频。
- `paired/index.html`：支持同步播放、80 ms 合并寻址、暂停/恢复、视角和种子切换的对照页面。
- `paired/local_gallery_verification.json`：本地重新核验 32 条轨迹 SHA、16 段编码（无 B 帧），16 个视频 HTTP Range 均返回 206 且字节与本地文件一致。浏览器实测原生尺寸 1280×960；拖动到 7.88 秒后，两路恢复播放且时间差约 0.3 ms，另核验腕部/第4组切换；最终暂停在第一组全局视角 2.40 秒。
- `paired/gallery_verified.jpg`：本地同步对照页面截图。

执行命令（151；前两对验证通过后沿用同一参数，不再重复 rollout）：

```bash
export FULL6_PPO_RUN=/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/train/ppo_targetnode_full6_workspace_20261004
experiment=/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/rl/eval/table_guard_20261004_step2433024
python_runtime=/data1/hxai/miniconda3/envs/dynamicvla/bin/python
$python_runtime $experiment/code/pilot_runner_original.py --checkpoint $FULL6_PPO_RUN/checkpoints/step_002433024.zip --output $experiment/pilot --episodes 2 --workers 4
mkdir -p $experiment/paired
cp -a $experiment/pilot/baseline $experiment/pilot/protected $experiment/paired/
$python_runtime $experiment/code/test_full6_table_guard.py --checkpoint $FULL6_PPO_RUN/checkpoints/step_002433024.zip --output $experiment/paired --episodes 14 --start-episode 2 --workers 8
$python_runtime $experiment/code/verify_table_guard_experiment.py $experiment/paired/baseline/seed_20260804
$python_runtime $experiment/code/render_table_guard_experiment.py --directory $experiment/paired --episodes 4 --workers 4
```

实际录像按前两对、接下来两对分批启动；第二批曾因第4组保护轨迹尚未落盘失败，该回合完成后单独补渲染，其他已完成的视频不重跑策略。最终 16 段全部独立核验通过。上面的统一渲染命令可在全部回合完成后复现同一视频集合。

本地页面：<http://127.0.0.1:8769/index.html>。
