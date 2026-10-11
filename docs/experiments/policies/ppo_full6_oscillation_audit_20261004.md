# 增强 6 PPO 抬升后反复松开诊断

对象：`ppo_targetnode_full6_workspace_20261004/videos/step_003538944/` 的四个完整回合。
只核查现有源码、配置、metadata 与保存状态；没有重新跑策略，没有修改训练。

## 奖励版本核查

151 的当前运行和上一轮 `ppo_targetnode_scratch_curriculum_tol4_near8_20260929`
冻结奖励文件完全相同：

- `rl/environment.py`：`a4d30074e3ef4292d491ccaab43ff05141cef8c321e75a79ac283c9bab91ff0c`
- `rl/target_node.py`：`5a0b44c17fa31fba454fe041a2407ed5066c4318f56fa15275e18b98220cc2c6`

可逆抬升奖励和 0.18–0.25 m 高度带仍启用，并未因增强 6 丢失。
新训练配置是 `initialization=random, warm_start=null`，并未继承旧模型的已学行为。

## 现有回合证据

下表直接读取每个视频目录的 `metadata.json/final_info`。“主动张爪中断”这个
历史字段实际按 `reason == gripper_command_open` 计数：它表示开爪指令清除逻辑抓取状态，
不是实际指距已显著增加的次数；还包含接触丢失后的歧义开爪，以及未达到稳定抓牢的中断。

| seed | 最终成功 | 累计奖励 | 主动张爪中断 | 物理滑脱中断 | secured 后主动张爪事件 |
|---|---|---:|---:|---:|---:|
| 20260804 | 否 | 78.72 | 19 | 5 | 6 |
| 20260805 | 否 | 146.15 | 34 | 8 | 6 |
| 20260806 | 否 | 82.25 | 12 | 3 | 0 |
| 20260807 | 否 | 102.71 | 23 | 4 | 5 |

保存动作也包含大量正/负 z 指令，起伏包含策略主动控制。不能仅据视频将其归为
增强伺服的机械振荡。四回合没有连续满足 0.8 s 的最终成功资格。

## 遗留漏洞与因果边界

冻结 `rl/environment.py` 的高度带分支在失去 pinch 时把
`_height_below_distance_previous` 和 `_overheight_distance_previous` 清为 None。
因此持续夹持的下降会扣回抬升分，但松开后的掉落不结算这部分势能；重抓再次初始化
后，抬到先前高度仍能获得正分。线缆抬起比例的势能同样只在 pinch 时发放，
无夹持时仅更新基准。两者留下跨夹持会话反复领取进度的机会。

主动松开惩罚使用 `_active_open_penalty_applied` 每回合只扣一次，
滑脱使用 `_contact_loss_penalty_applied` 每回合只扣一次；而抬升收益可跨重抓重复。
若未达到 secured，相关松开惩罚资格还可能不成立。当前失败却高回报的轨迹与
“抬一点—松开—落下—再抓”的漏洞吻合。尚未进行奖励修正的配对消融，
因此不能给出增强 6 对漏洞发生率的独立因果贡献。

增强执行层、目标采样分布与随机重新训练均发生变化，旧模型的行为不会自动保留。
应优先补齐跨重抓的抬升信用结算，再处理重复主动松开的事件扣分；仅增加最终
成功奖励或等待更多训练不能保证消除漏洞。此诊断不实施上述修改。

补充保存状态检查存于视频目录 `oscillation_saved_state_audit.json`：
使用 `mj_setState/mj_forward` 读取几何高度，不执行 `env.step` 或策略推理；
其中按 `actual_node != None` 划分的是有抓取状态的区间，不能代替 pinch/secured 判定。
重建模型的几何高度仅作辅助，主要事件与收益结论来自原录像 metadata。

## 后续开爪与腕部视角核查

`gripper_camera_audit.json` 直接读取录像的动作与手指关节位置。动作第 5 维 ≥0.35
切换 OPEN，≤−0.35 切换 CLOSE，中间保持状态；NERO 对应执行指令 0.05 / 0 m。
底层用执行指令是否 <0.005 判闭合，一旦不闭合就立即清除 grasp_state；不等待
实际手指分开。第二回合 0.80 s 的 OPEN 指令为 +1.0，实际指距从 33.704 mm
到 20 ms 后的 33.748 mm，只增加 0.044 mm，0.90 s 已再次发 CLOSE。
因此先前“抬起后主动放掉再落下”的解释需收紧：逻辑状态清除与信用重置已证实，
但并非每次都有明显的物理释放；状态判定和实际夹持不同步也是问题的一部分。

录像器覆盖了腕部相机位姿，却沿用了自由相机的 45° 视场；源码默认相机视场为
73.7398°。默认值本身不能证明用户所指旧视频使用的实际设置。
本次另生成 `wrist_review.mp4`，原片保留；修正视场并令录像视轴朝向固定的局部
两指夹持中心，展示手指、OPEN/CLOSE 指令和真实关节开度。仅改变观看相机和 HUD，
不改策略观测、动作、物理和训练。仍是保存的 751 个 20 ms 状态，状态差异为 0；
检查记录为 `wrist_review_manifest.json` 与 `local_gallery_verification.json`。

### 纠正观看视角（用户指出手掌遮挡后）

此前新增 TCP look-at 并非恢复旧录像：它让手掌进入画面并遮挡，已经撤掉。
旧实帧来源是 `outputs/control_unification_20261002/scripted_pair_hd_20261002/raw/baseline/episodes/scripted/id_static/seed_2026100200/wrist.mp4`
的 1.00 s 帧，本地提取为 `old_recorded_wrist_reference.png`；其中指尖仅位于下方两角。
该录像 manifest 记录 commit `d70b249fd623f13c3bc1f670bdb3da2e9cd8685a`；该 commit
的相机源码与对应 HumanEpisodeRecorder 使用固定相机，不使用 TCP look-at。
这是一段已核实的旧录像参考，不能把它推断成用户记忆中所有视频的统一配置。

新 `wrist_tips.mp4` 恢复保存的局部相机位置/旋转，沿固定光轴观看，并保留实际开度 HUD。
`wrist_tips_manifest.json` 写明源录像参考、位姿、视场、状态一致性和编码验证。
上一版遮挡视频保留但页面不再使用。网页腕部播放器使用公共播放控件，以避免
暂停时原生控件覆盖下方指尖。未更改训练、策略动作或模拟轨迹。

### 相机配置来源链核实（不是从视频估计）

进一步读取上述旧录像所在 `raw/episodes.json`，匹配 baseline / id_static /
seed 2026100200 的实际 `env_config` 和 `compiled_model`，再加载归档模型
`raw/models/id_static_baseline_5e7840e12200.mjb`。模型中命名相机
`dynamicvla_wrist_camera` 挂在 `link7`：局部位置 `(0.020, -0.100, 0)` m，
四元数 wxyz `(0, -0.7071067811865477, 0, 0.7071067811865477)`，垂直视场
`73.7397952917°`。这些是当时编译模型的实际参数，不是图像估算。

原录制 worker 的 SHA256 与 manifest 中 `original_worker_sha256` 完全一致。
入口 `rerun_restored_scripted_pair.initialize` 选择 `HumanEpisodeRecorder`；
它的 `_write_frame` 直接按上述命名相机渲染，没有 TCP 转向或视场覆盖。
worker 返回当时的 `asdict(env.config)` 和所保存模型路径，形成录像、运行记录、
录制代码、编译模型之间的来源链。

当前四段 `wrist_tips` 重放的位置/旋转来自各自 metadata，视场来自冻结 EnvConfig。
与归档编译相机逐项核对：位置差为 0，旋转矩阵差小于 1e-9，视场差为 0。
核查保存为视频目录 `camera_configuration_provenance.json`；没有重新运行策略。
旧帧仅用于观看效果检查。73.74° 现已证实属于这段 10 月 2 日旧录像，仍不能
泛称所有旧视频的“原配置”；原始本次 PPO 手动录像器实际沿用 45° 的问题仍成立。

## 整回合夹爪动作与关节曲线

`tools/recording/plot_ppo_gripper_episode.py` 读取四段保存轨迹、动作、20 ms metrics，
不创建环境或运行策略。FULLPHYSICS 的 time 后为 qpos；使用原核查确认的手指地址
7、8，实际指距按冻结源码 `sum(abs(qpos[finger_qpos_adr]))` 计算，并与原保存状态
重放的 `wrist_tips_tracks.json` 每帧核对，差值不超过 1e-9 mm。
`actions[i]` 施加在 `[frame_times[i], frame_times[i+1])`，关节 `i` 为之前，`i+1` 为之后。
开合 latch 从 OPEN 开始，动作 ≥+0.35 为 OPEN、≤−0.35 为 CLOSE，其余保持。
单指执行目标 OPEN=50 mm、CLOSE=0 mm，与实际两指总间距不是同一个数值。

网页视频下新增四行同步时间曲线（动作、执行指令、实际双指关节/总间距、逻辑状态），
支持整回合/前2秒/局部放大、点击定位视频、逐次事件表、PNG与CSV。四段原数据 SHA256
保持一致，第二回合事件点击定位两视频至0.80 s，播放至约1.076 s时曲线读数同步更新。
保存记录 `gripper_episode_data.json`、`gripper_episode_summary.json`、
`gripper_episode_browser_verification.json`，截图 `gripper_episode_browser.png`。

第二回合34次切换 OPEN，31次切换动作≥0.99（91.2%），OPEN持续中位数120 ms，
切换后首个20 ms的指距变化中位数0.05876 mm。0.80 s动作+1、OPEN持续100 ms，
指距33.704→33.748 mm，期间最大增加也仅0.044 mm；0.82 s仍有逻辑抓取状态，
0.86 s记录中消失。进一步核查发现：底层先限制夹爪目标变化，
`gripper_finger_velocity_limit=0.2` m/s，代码转换系数为2，所以单指控制目标
每20 ms最多改变2 mm。0.80 s开始的三步目标分别为2、4、6 mm；第三步超过
`gripper_closed_ctrl_threshold=5` mm后才清除状态。此前“PPO OPEN立即清除”
的描述遗漏了这层限速，必须纠正：立即清除针对限速后目标达到阈值，不是原动作
跨过+0.35。新增青线为代码重建的限速目标，未直接记录ctrl；四段重建的限速比例
与metadata记录完全相同，且重建目标≥5 mm时所有记录均没有逻辑抓取状态。
因此不仅是动作阈值+0.35灵敏度问题：动作往往已饱和，而清除资格依据目标指令，
不依据实际手指是否分开。物理释放应依据实际关节与接触证据另外判定。曲线的紫色竖线仅是
20 ms metrics中抓取状态消失且该步执行OPEN，不是500 Hz事件精确时间或物理释放证明。
本次没有修改训练/判定逻辑；变更判定需要处理抓取和奖励状态一致性，不能凭图直接
指定全局固定指距阈值。
