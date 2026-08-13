# Franka Panda 动态线缆抓取演示

本项目使用 Menagerie 的 Franka Panda、MuJoCo 官方
`mujoco.elasticity.cable` 插件，以及一个持续变形的自由线缆场景。

程序现已按职责拆分：

- `cable_grasp_env.py`：模型加载、`reset/step`、连续扰动、真实碰撞/摩擦和成功判定；
- `dynamic_grasp_policy.py`：目标线段预测、逆运动学、接近/截获/闭爪/抬升/搬运状态机；
- `run_grasp.py`：无界面试验或实时 GUI 调度；
- `replay_recording.py`：打开状态回放窗口，用鼠标选视角并重新导出整段视频；
- `render_recording.py`：读取无头模式保存的完整状态，从任意相机角度离线重新渲染；
- `panda_cable_grasp.xml`：桌面、线缆和物理参数；
- `run_demo.ps1`：使用 `dynamic` Conda 环境启动。

策略动作只包含 7 个机械臂关节目标和 1 个夹爪命令。线缆扰动始终由环境施加，
不会再因为闭爪而降到 15% 或停止。

## 启动 GUI

```powershell
cd C:\Users\27642\Desktop\dynamic_cable\panda_cable_grasp
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
```

窗口会立即出现，然后才开始本轮物理仿真；不是先算完再回放。默认执行 3 个随机试次，
最后一个场景会留在窗口中。按 `N` 可继续新试次。

物理积分保持 500 Hz，逆运动学和策略动作更新为 50 Hz，GUI 固定按 60 FPS 调度；viewer
创建 OpenGL 窗口所花的启动时间不会再被错误计入待追赶的仿真时间。

扰动空间项、40个节点的外力和桌边软边界均采用批量数组计算。GUI在窗口拖动或断点后每帧
最多追赶8个控制周期，避免长时间无响应。当前机器实测环境平均步耗时约15.84 ms，90%步骤
不超过19.57 ms；7.008秒墙钟时间推进了7.000秒仿真时间。

GUI 快捷键：

- `=` 或 `]`：2 倍加速，最高 8x；
- `-` 或 `[`：减半，最低 0.25x；
- `Space`：暂停/继续；
- `Backspace` 或 `R`：完整重置到桌面上方的弯臂 ready 姿态；
- `N`：开始新的随机试次。

也可在启动时设置倍速、无限试次和扰动强度：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1 --speed 2 --trials 0 --disturbance 1.5
```

`--speed` 是“仿真时间 / 墙钟时间”目标值。若计算机无法实时完成 8x 所需的物理步，
实际速度会受 CPU 性能限制。

默认扰动倍率为 `1.5`。当前力场以横向局部弯折为主，包含四个互不整除的空间频率、快速
时间相位和两组相位调制；不会通过夹爪开合或策略阶段改变强度。

环境还使用桌边软边界：距桌边 0.16 m 时开始渐进调整环境自身的向外扰动力，并只阻尼
朝桌外的速度。它不裁剪线缆坐标、不覆盖策略动作，也不增加实体围栏，因此可以防止线缆因
合成扰动自行掉落，但机械臂仍能依靠正常 MuJoCo 接触力把线缆推出桌外。

## 无界面验证

无界面模式不做实时等待，但运行的仍是和 GUI 完全相同的环境和策略：

```powershell
& C:\ProgramData\anaconda3\envs\dynamic\python.exe .\run_grasp.py --headless --trials 3
```

无界面模式会同步离屏渲染，每个 trial 单独保存一个 MP4。默认输出到
`headless_videos/run_时间_seed随机种子/trial_001.mp4`。可通过参数修改目录、帧率和分辨率：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1 --headless --trials 3 `
  --video-dir .\headless_videos --video-fps 25 `
  --video-width 960 --video-height 540
```

每个视频同时生成同名的 `trial_XXX_states.npz`，运行目录还包含对应的编译模型
`model.mjb`。状态与视频帧一一对应，可以不重新运行物理和策略，从任意视角离线渲染：

推荐使用交互式回放窗口：

```powershell
& C:\ProgramData\anaconda3\envs\dynamic\python.exe .\replay_recording.py `
  --states .\headless_videos\run_时间_seed20260804\trial_001_states.npz
```

窗口内直接使用MuJoCo原生鼠标操作旋转、平移和缩放相机。按键如下：

- `Space`：暂停或继续；
- `←/→`：暂停并逐帧查看；
- `R`：从头播放；
- `[`/`]`：降低或提高回放速度；
- `V`：锁定当前鼠标相机，并用该视角重新渲染完整视频。

按 `V` 后，视频默认保存到状态文件所在目录，文件名类似
`trial_001_view_20260808_193932.mp4`。导出规格可用 `--export-width`、
`--export-height` 和 `--export-fps` 设置。使用 `--loop` 可以循环回放，便于反复调整相机。

下面的批处理命令仍可用于已知相机数值、不需要打开交互窗口的情况：

```powershell
& C:\ProgramData\anaconda3\envs\dynamic\python.exe .\render_recording.py `
  --states .\headless_videos\run_时间_seed20260804\trial_001_states.npz `
  --azimuth 45 --elevation -15 --distance 1.8 `
  --lookat 0.55 0 0.30 `
  --output .\headless_videos\run_时间_seed20260804\trial_001_side.mp4
```

也可以用 `--width`、`--height` 和 `--fps` 修改二次渲染的视频规格。离线渲染默认使用
状态文件旁的 `model.mjb`，因此即使之后修改XML，也不会误用新模型解释旧状态。

## 纯碰撞/摩擦抓取和成功标准

当前已经完全删除原来的弹簧阻尼承重力、重力补偿、三节点跟随带和闭爪资格快照。环境不会在
检测到抓取后对线缆写入跟随夹爪的外力，也没有焊接、等式约束或磁吸。线缆能否被拿起只取决于
MuJoCo 求解的碰撞、法向夹紧力和摩擦。

Panda 内指垫没有扩大原碰撞外轮廓，但已把原平面近似改成五段阶梯浅槽。当前接触采用
`condim=6`，滑动/扭转/滚动摩擦分别为 `4.0 / 0.10 / 0.05`。闭爪位置伺服增益和位置
偏置按 Menagerie 配置的 5 倍缩放，执行器最大 `forcerange` 不变；这些参数始终生效，
不随抓取状态、策略阶段或成功判定切换。

环境不限制策略动作。夹爪闭合后横扫线缆时，线缆只会发生普通碰撞；如果确实形成双指夹持，
也会按照相同物理规则被抓起，不再存在环境侧的“允许/禁止吸附”名单。

线缆自身保留 `2.0` 的高摩擦以支持指垫夹持；XML 为 40 个“桌面—线缆”组合单独设置
`0.12` 的滑动摩擦。因此桌面接触不会再继承线缆的高摩擦，抓空后更倾向于滑到一侧。

夹爪请求完全闭合，但最终开口由线缆—指垫碰撞决定。抓取状态只是只读任务判定：左右指垫
各自法向力至少 `0.20 N`、开口不超过 `34 mm`、接触线段位于指垫中心 `55 mm` 内，并连续
保持 `0.06 s` 后才确认。确认后只要左右指垫仍各有真实碰撞，就不会因瞬时法向力低于阈值
误报滑脱；真实双侧碰撞累计丢失 `0.35 s` 后才清除状态。接触从一个离散线缆节点自然移动
到另一个节点不会被误判为脱落。

成功必须连续 0.80 秒同时满足：

- 当前仍有经过确认的真实双指夹持；确认后的短接触丢失只按既有0.35秒滞回处理；
- 被抓线段高度超过 0.14 m；
- 至少 18% 的线缆节点高于桌面 0.055 m；
- 夹爪命令仍处于闭合状态；
- 当前或最近确认的抓取线段距指垫中心不超过 82.5 mm。

脚本在截获阶段还会检查整条线缆中心线：只要某一实际线段进入夹持中心 18 mm 内，就锁定
该线段并闭爪，不再继续追逐已经远离的随机参考节点。闭爪期间只要仍有真实指垫接触，就会
延长抓取确认窗口。双侧抓取确认后停止追踪线缆相对运动，转入名义举升和保持轨迹；抓前的
形状预测与实际线段截获不受影响。名义抬升为220 mm；之后的1.0 s CARRY阶段朝桌面中心
侧移80 mm并上移20 mm，夹爪中心不得降到闭爪位置上方140 mm以下。闭爪命令 `ctrl=20`
对应约6.3 mm无负载目标总开口，仍远小于28 mm线径，但避免零间隙目标过度挤压圆线。
HOLD只接受当前连续0.80秒举升资格，不使用历史成功掩盖后续掉落；资格不足时继续闭爪保持
直到回合上限，不会因固定计时器主动张开。

固定种子 `20260804` 的消融中，关闭抓后反馈仍为5/5严格成功，且末端高度和当前保持时间
更一致，因此正式代码已删除该反馈。无固定搬运与恢复80 mm侧移/20 mm上移均为5/5严格
成功，取消固定搬运没有显示成功率收益，因此默认恢复80 mm侧移和20 mm上移。该小样本结果
只用于定向回归，仍需录像和更多随机种子验证。

`headless_videos/run_20260809_200100_seed20260804/` 是无固定搬运消融录像，不再对应当前默认
配置。`headless_videos/run_20260809_200707_seed20260804/` 是抬升180 mm、连续保持0.55 s时的
5/5旧基线。当前抬升220 mm、连续保持0.80 s的录像位于
`headless_videos/run_20260809_202134_seed20260804/`：同一5个初态为4/5，trial 3在CARRY中真实滑脱。

纯摩擦模型不会、也不应保证在任意大外力下永不滑脱；那需要隐藏约束或粘附模型。当前参数的
目标是在默认持续扰动下显著提高夹持能力，同时让真正超过摩擦锥的滑移仍按物理方式发生。

终端会输出状态转换、追踪误差、接触数，以及成功时的抬升比例和抓取误差。

## PPO强化学习策略

RL环境完全绕过 `dynamic_grasp_policy.py` 的脚本状态机。PPO输出8维归一化动作：前7维
控制Panda关节位置目标增量，每步范围为±0.04 rad；第8维通过`-0.35/+0.35`开合滞回控制
夹爪，闭合目标为`ctrl=20`、张开目标为255，避免单一阈值附近反复开合。

48维归一化特权观测包括：

- 目标线段的世界位置和线速度；
- 目标线段相对真实指垫中心的位置；
- 7个机械臂关节的位置和速度；
- 两个手指关节的位置和速度；
- 真实指垫中心的位置、手部四元数、线速度和角速度。
- 左右指垫接触、法向力、pinch、secured grasp、相对抬升量和严格成功保持进度。

策略看不到完整线缆形状，也不读取脚本策略阶段。RL将双侧真实接触确认记为`pinch`，不再直接
记作抓取；只有线段相对pinch建立时的高度上升至少30 mm，并在真实双侧接触、开度和中心距离
均合格时连续保持0.10 s，才成为`secured grasp`。抓取率和抓取/抬升奖励只采用secured grasp。
RL成功还要求secured grasp下线段高度、整线离桌比例和55 mm中心距离连续满足0.80 s。
训练态允许已确认secured grasp吸收最多0.06 s的接触求解抖动，但这段滞回不会延长严格成功
计时；当前原始双侧接触一旦缺失，RL保持计时立即清零。最终终止还要求底层当前500 Hz
高度、整线离桌比例和中心距离连续计时同时达到0.80 s，避免50 Hz策略采样漏掉几何子步
间断。pinch形成后，目标位置、
速度和接近进度切换到当前实际夹持线段，不再追逐原随机节点。

奖励采用事件和单调高水位形式：首次单/双指接触、首次pinch和首次secured grasp只奖励一次；
抓取段相对抬升在120 mm封顶，整线抬升比例在30%封顶，掉落后重抓不会再次领取旧高度奖励。
secured后纯主动张爪、闭爪物理滑脱和“接触已丢失时又张爪”分别记录；主动张爪惩罚为-8，
后两类为-3。物理/歧义接触丢失合计每回合最多罚一次，纯主动张爪另最多罚一次，全部事件仍
写入诊断。严格保持奖励也按整回合高水位发放，总额封顶0.8分。另有抓后机械臂动作
幅值/变化率代价、严格保持小奖励和25分终止成功奖励。这样保留真实滑脱，但不让一串接触
事件主导整回合回报。

注意：此前在弹性夹持代理下训练的 `rl/runs/ppo_cable/final_model.zip` 与当前物理模型不再属于
同一个训练环境，不能用来比较新环境成功率；应在本分支重新训练。

首次使用已经在 `dynamic` 环境安装了依赖；如需重建环境可运行：

```powershell
& C:\ProgramData\anaconda3\envs\dynamic\python.exe -m pip install -r .\rl\requirements_rl.txt
```

默认用6个独立MuJoCo进程训练200万步并写入`rl/runs/ppo_cable_v3/`。前100万步把训练扰动
从0.30线性增加到完整的1.50，之后保持完整扰动；学习率从3e-4线性降到3e-5，熵系数从
0.01降到0.001，PPO每批更新5轮并用`target_kl=0.015`提前停止过大的更新。每10万步在
10个固定新种子上以完整扰动独立评估，并按严格成功率保存`evaluation/best/best_model.zip`：

```powershell
powershell -ExecutionPolicy Bypass -File .\rl\run_rl_train.ps1
```

短训练或改变并行数：

```powershell
powershell -ExecutionPolicy Bypass -File .\rl\run_rl_train.ps1 `
  --timesteps 200000 --workers 6 --output .\rl\runs\ppo_cable_v3_200k
```

训练时每完成20轮会在终端输出最近100轮的成功率、pinch率、有效抓取率、平均回报和平均线缆抬升比例；
同样的滚动指标会写入TensorBoard的 `task/` 分组。训练产物包括：

- `final_model.zip` 和定期检查点；
- `training_metrics.csv`：每轮原始成功、pinch、有效抓取、回报和抬升数据；
- `training_curves.csv`：最近100轮滑动指标；
- `success_rate.png`：成功率和抓取率曲线；
- `reward_curve.png`：平均回报曲线；
- TensorBoard日志、Monitor记录和完整训练配置。

查看实时TensorBoard曲线：

```powershell
& C:\ProgramData\anaconda3\envs\dynamic\python.exe -m tensorboard.main `
  --logdir .\rl\runs\ppo_cable_v3\tensorboard
```

仅对采用当前抓取语义的新模型继续训练可使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\rl\run_rl_train.ps1 `
  --resume .\rl\runs\ppo_cable_v3\final_model.zip --timesteps 1000000 `
  --output .\rl\runs\ppo_cable_v3
```

旧`ppo_cable`模型和`ppo_cable_v2`都不应resume到v3奖励继续正式训练；它们只用于回归测试。

无界面测试20个新随机场景：

```powershell
powershell -ExecutionPolicy Bypass -File .\rl\run_rl_test.ps1 `
  --model .\rl\runs\ppo_cable_v3\evaluation\best\best_model.zip --headless --episodes 20
```

大批量统计时可关闭录像。每回合仍写入`episodes.csv`，并保存含checkpoint/XML/源码哈希、
配置、依赖版本和seed规则的`run_manifest.json`：

```powershell
powershell -ExecutionPolicy Bypass -File .\rl\run_rl_test.ps1 `
  --model .\rl\runs\ppo_cable_v3\evaluation\best\best_model.zip `
  --headless --episodes 100 --no-video
```

CSV同时记录公共任务成功、PPO内部严格成功、场景指纹、动作饱和率，以及互斥失败类型。
纯主动张爪、物理滑脱和接触丢失过程中张爪不会混为一类；汇总会直接报告物理滑脱在失败中的
占比及其是否超过50%。

在完整core场景矩阵上，用完全相同seed配对运行脚本基线与PPO，并自动校验
`scenario_id + motion_profile_hash + scene_fingerprint`：

```powershell
& C:\ProgramData\anaconda3\envs\dynamic\python.exe .\benchmark.py `
  --suite core --methods scripted ppo --episodes 100 --seed 20280804 `
  --ppo-model .\rl\runs\ppo_cable_v3\evaluation\best\best_model.zip
```

`--episodes`表示每个场景的重复次数。可将`core`换成`motion_sweep`、`ood`或`paper`；
`paper`覆盖全部20个冻结场景。输出包含逐回合`episodes.csv`、分层`summary.json`、
`manifest.json`及每种物理配置的编译MJB。汇总同时报告逐场景、运动类型、ID/OOD、micro与
场景等权macro成功率、Wilson 95%区间；滑脱是否主导失败使用单侧95% Wilson上界判断，
不是简单看小样本点估计。

现有PPO checkpoint只对应旧`legacy_v1`形变场。正式矩阵使用严格去除shape净力/净力矩的
`factorized_v1`，必须从头覆盖ID场景训练：

```powershell
powershell -ExecutionPolicy Bypass -File .\rl\run_rl_train.ps1 `
  --training-distribution id --eval-distribution core `
  --timesteps 2000000 --workers 6 --output .\rl\runs\ppo_cable_matrix_v1
```

先运行无策略运动诊断，确认四类场景的实际响应可区分：

```powershell
& C:\ProgramData\anaconda3\envs\dynamic\python.exe .\motion_diagnostics.py `
  --suite core --seeds 10 --seconds 8 --sample-hz 10 --seed 20280804
```

完整冻结流程、统计口径和真机环境建议见[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。
π0.5和Diffusion Policy目前尚无适配器与checkpoint，不会在结果表中用替代数据冒充。

无头RL测试默认在`rl_test_videos/run_时间_seed.../`中保存：

- 每回合`episode_XXX.mp4`；
- 每回合`episode_XXX_states.npz`，包含视频帧对应的完整MuJoCo状态，以及每个50 Hz策略动作、奖励和时间；
- 本次测试实际使用的`model.mjb`。

可以修改录像参数：

```powershell
powershell -ExecutionPolicy Bypass -File .\rl\run_rl_test.ps1 `
  --model .\rl\runs\ppo_cable_v3\evaluation\best\best_model.zip `
  --headless --episodes 20 --video-dir .\rl_test_videos `
  --video-fps 25 --video-width 960 --video-height 540
```

状态文件与脚本基线格式兼容，可直接回放：

```powershell
& C:\ProgramData\anaconda3\envs\dynamic\python.exe .\replay_recording.py `
  --states .\rl_test_videos\run_...\episode_001_states.npz --loop
```

打开实时MuJoCo窗口观察5轮：

```powershell
powershell -ExecutionPolicy Bypass -File .\rl\run_rl_test.ps1 `
  --model .\rl\runs\ppo_cable_v3\evaluation\best\best_model.zip --episodes 5 --speed 1
```

`rl/runs/smoke_test`只是2048步程序链路检查，不是已经学会抓取的模型；正式效果需要运行足够长的训练。
