# Franka Panda 动态线缆抓取演示

本项目使用 Menagerie 的 Franka Panda、MuJoCo 官方
`mujoco.elasticity.cable` 插件，以及一个持续变形的自由线缆场景。

程序现已按职责拆分：

- `cable_grasp_env.py`：模型加载、`reset/step`、连续扰动、真实碰撞/摩擦和成功判定；
- `dynamic_grasp_policy.py`：目标线段预测、逆运动学、接近/截获/闭爪/抬升/搬运状态机；
- `run_grasp.py`：无界面试验或实时 GUI 调度；
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

## 纯碰撞/摩擦抓取和成功标准

当前已经完全删除原来的弹簧阻尼承重力、重力补偿、三节点跟随带和闭爪资格快照。环境不会在
检测到抓取后对线缆写入跟随夹爪的外力，也没有焊接、等式约束或磁吸。线缆能否被拿起只取决于
MuJoCo 求解的碰撞、法向夹紧力和摩擦。

Panda 内指垫使用高摩擦橡胶参数：`condim=6`，滑动/扭转/滚动摩擦分别为
`6.0 / 0.35 / 0.12`。主指垫扩大到能覆盖 28 mm 线缆，并由原有小碰撞 box 组成浅凹槽，
降低圆柱从指垫边缘滚出的概率。闭爪执行器增益为 Menagerie 配置的 5 倍；这是始终生效的
执行器/接触参数，不随抓取状态、策略阶段或成功判定切换。

环境不限制策略动作。夹爪闭合后横扫线缆时，线缆只会发生普通碰撞；如果确实形成双指夹持，
也会按照相同物理规则被抓起，不再存在环境侧的“允许/禁止吸附”名单。

线缆自身保留 `2.0` 的高摩擦以支持指垫夹持；XML 为 40 个“桌面—线缆”组合单独设置
`0.12` 的滑动摩擦。因此桌面接触不会再继承线缆的高摩擦，抓空后更倾向于滑到一侧。

夹爪请求完全闭合，但最终开口由线缆—指垫碰撞决定。抓取状态只是只读任务判定：左右指垫
各自法向力至少 `0.20 N`、开口不超过 `34 mm`、接触线段位于指垫中心 `55 mm` 内，并连续
保持 `0.06 s` 后才确认。真实双指接触丢失 `0.10 s` 后清除状态。接触从一个离散线缆节点
自然移动到另一个节点不会被误判为脱落。

成功必须连续 0.55 秒同时满足：

- 当前仍存在满足法向力阈值的真实双指接触；
- 被抓线段高度超过 0.14 m；
- 至少 18% 的线缆节点高于桌面 0.055 m；
- 指间开口处于 0–34 mm；
- 接触线段中心距指垫中心不超过 55 mm。

纯摩擦模型不会、也不应保证在任意大外力下永不滑脱；那需要隐藏约束或粘附模型。当前参数的
目标是在默认持续扰动下显著提高夹持能力，同时让真正超过摩擦锥的滑移仍按物理方式发生。

终端会输出状态转换、追踪误差、接触数，以及成功时的抬升比例和抓取误差。

## PPO强化学习策略

RL环境完全绕过 `dynamic_grasp_policy.py` 的脚本状态机。PPO直接输出8维连续动作：前7维
控制Panda关节位置目标的增量，第8维控制夹爪，`-1`表示闭合、`+1`表示张开。环境只做
执行器合法范围裁剪，不根据训练阶段修改策略动作。

40维归一化观测包括：

- 目标线段的世界位置和线速度；
- 目标线段相对真实指垫中心的位置；
- 7个机械臂关节的位置和速度；
- 两个手指关节的位置和速度；
- 真实指垫中心的位置、手部四元数、线速度和角速度。

策略看不到完整线缆形状，也不读取脚本策略阶段。奖励由接近进度、真实指垫接触、已确认抓取、
抬升高度、整条线缆离桌比例和最终成功组成，并包含很小的动作代价。奖励只读取状态，不修改
线缆扰动、接触、抓取判定或机器人动作。

注意：此前在弹性夹持代理下训练的 `runs/ppo_cable/final_model.zip` 与当前物理模型不再属于
同一个训练环境，不能用来比较新环境成功率；应在本分支重新训练。

首次使用已经在 `dynamic` 环境安装了依赖；如需重建环境可运行：

```powershell
& C:\ProgramData\anaconda3\envs\dynamic\python.exe -m pip install -r .\requirements_rl.txt
```

默认用6个独立MuJoCo进程训练200万步：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_rl_train.ps1
```

短训练或改变并行数：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_rl_train.ps1 `
  --timesteps 200000 --workers 6 --output .\runs\ppo_cable_200k
```

训练时每完成20轮会在终端输出最近100轮的成功率、抓取率、平均回报和平均线缆抬升比例；
同样的滚动指标会写入TensorBoard的 `task/` 分组。训练产物包括：

- `final_model.zip` 和定期检查点；
- `training_metrics.csv`：每轮原始成功、抓取、回报和抬升数据；
- `training_curves.csv`：最近100轮滑动指标；
- `success_rate.png`：成功率和抓取率曲线；
- `reward_curve.png`：平均回报曲线；
- TensorBoard日志、Monitor记录和完整训练配置。

查看实时TensorBoard曲线：

```powershell
& C:\ProgramData\anaconda3\envs\dynamic\python.exe -m tensorboard.main `
  --logdir .\runs\ppo_cable\tensorboard
```

继续训练可使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_rl_train.ps1 `
  --resume .\runs\ppo_cable\final_model.zip --timesteps 1000000
```

无界面测试20个新随机场景：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_rl_test.ps1 `
  --model .\runs\ppo_cable\final_model.zip --headless --episodes 20
```

打开实时MuJoCo窗口观察5轮：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_rl_test.ps1 `
  --model .\runs\ppo_cable\final_model.zip --episodes 5 --speed 1
```

`runs/smoke_test`只是2048步程序链路检查，不是已经学会抓取的模型；正式效果需要运行足够长的训练。
