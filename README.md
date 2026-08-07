# Franka Panda 动态线缆抓取演示

本项目使用 Menagerie 的 Franka Panda、MuJoCo 官方
`mujoco.elasticity.cable` 插件，以及一个持续变形的自由线缆场景。

程序现已按职责拆分：

- `cable_grasp_env.py`：模型加载、`reset/step`、连续扰动、接触抓取代理、成功判定；
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

## 抓取代理和成功标准

MuJoCo 的普通摩擦接触并不保证一条很细、持续受强扰动的柔性线缆不会从离散碰撞几何间
数值滑出。当前不再使用任何单侧接触对中或吸入引导：只有左右两指都实际接触同一线段或
相邻线段，并连续保持 0.08 秒后，才启用局部柔性夹持代理。其余线缆节点仍自由。

确认双侧接触后，抓取范围从单个节点扩展为中心节点及左右相邻节点，形成约 40 mm 的局部
夹持带。三个节点在手坐标系中按线缆 20 mm 节距排列，并锁定到指垫中心，因而同时限制局部
位置和切向，不再允许单节点绕夹爪自由旋转或穿过指垫。

环境不限制策略的机械臂动作；脚本策略在闭爪期间仍可继续横向跟踪。每次夹爪命令从张开变为
闭合时，环境只记录当时已经位于指间区域的线段。闭合后横扫新碰到的线段仍参与正常碰撞并会
被推开，但不能激活承重代理或成功判定；策略若想抓它，必须重新张开、对准并闭合。

线缆自身保留 `2.0` 的高摩擦以支持指垫夹持；XML 为 40 个“桌面—线缆”组合单独设置
`0.12` 的滑动摩擦。因此桌面接触不会再继承线缆的高摩擦，抓空后更倾向于滑到一侧。

夹爪现在请求完全闭合，但最终开口由线缆—指垫碰撞决定；环境将 Menagerie 原本较柔和的
夹爪位置增益和闭合刚度等比例放大 1.5 倍，目标位置映射不变，只适度增加夹持力。双侧接触只用于
建立抓取；确认后允许接触在左右指垫之间切换。只有代理误差连续超过 0.025 m，或者线缆中心连续
离开两指夹持区域 0.15 秒，才撤销夹持代理。这仍是局部柔性夹持代理，不是 FEM 指垫模型。

成功必须连续 0.55 秒同时满足：

- 左右两根手指都提供了真实接触证据，且抓取尚未因误差过大而断开；
- 被抓线段高度超过 0.14 m；
- 至少 18% 的线缆节点高于桌面 0.055 m；
- 抓取代理误差小于 0.045 m；
- 指间仍保留至少 0.018 m 开口；
- 被抓线段仍位于夹爪附近。

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
线缆扰动、接触、夹持代理或机器人动作。

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
