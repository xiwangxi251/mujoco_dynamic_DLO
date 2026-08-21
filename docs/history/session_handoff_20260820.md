# Franka Panda 动态线缆抓取项目交接文档

- 更新时间：2026-08-19
- 项目目录：Git 仓库根目录
- Menagerie 目录：由 `MUJOCO_MENAGERIE_PATH` 指定；仍兼容仓库旁的自动发现
- Python：当前激活环境中的 `python`

> 2026-08-20 RL baseline v2 更新：旧48维观测/8维关节动作已被99维整线DLO观测和5维TCP动作替代，
> 旧PPO checkpoint不兼容。默认训练只使用L1 nominal场景，并采用三阶段课程：`id_static` →
> `id_shape_nominal_current + id_rigid_l1_nominal` → `id_combined_l1_nominal`。两个运动阶段各用
> 50万步从现有low强度升到nominal，每阶段需要累计10次严格成功才允许进入下一阶段。
> 当前机器实测6/8/10/12/14/16环境吞吐约为62/73/80/87/89/95 steps/s；14环境长跑只剩约0.8 GB
> 可用内存，因此正式默认使用12环境。当前默认输出为`rl/runs/ppo_dlo_baseline_v2/`。下文涉及PPO v2/v3、
> 48维观测、8维动作、`target_kl=0.015`和`ppo_cable_v3`的内容仅保留为历史记录。

Linux 服务器安装以 `LINUX_SERVER_SETUP.md` 为准。当前代码加载仓库内的
`panda_cable_grasp.xml` 和 `models/panda.xml`，Menagerie 只提供 mesh 资产，不再维护
两份任务 XML，也不再依赖某台 Windows 电脑的绝对路径。

## 0. 新会话首先阅读：截至 2026-08-13 的最新状态

- 环境、浅槽指垫、纯碰撞/摩擦夹持、脚本基线、PPO接口以及无头录像/状态回放链路均已搭建完成。
- 当前研究重点仍是：**根据线缆形状变化移动夹爪，使运动线段进入夹爪中心**。不得通过扩大夹爪、碰撞盒、浅槽捕获范围或隐藏约束降低该难度。
- 当前形变由多组不同时间频率和空间频率的正弦行波外力叠加产生。不同线缆位置包含相同的时间频率分量，但空间相位不同，因此同一时刻受力大小和方向不同；每回合随机初始时间/空间相位。固定seed后外力是确定且可复现的，不是逐时刻白噪声。
- 当前规则方法在固定5个初始场景中严格成功4次（4/5，80%）。该样本很小，而且脚本与RL测试流程尚未完全对齐，**不能把80%与RL成功率直接比较**。
- 当前PPO v2已完成200万环境步训练，输出目录为`rl/runs/ppo_cable_v2/`。最佳模型在1600032步保存，固定10个完整扰动评估场景中成功3次（30%）；最终200万步模型在相同10个场景中为0次成功，说明后半程发生策略退化。
- 最佳模型另做了5局录像复核，使用的正是上述固定10个评估seed中的前5个，结果1/5成功（20%），不是另一组独立样本。录像和state位于`rl_test_videos/run_20260810_075643_seed20270804/`。
- 这5局的主要结论：策略已能学会接近、闭爪和抬升，但抓后保持不稳定；第2、5局明确主动张爪，第1局既有主动张爪也有闭爪状态下真实滑脱，第3局未形成双侧夹持，第4局成功。第5局严格保持达到0.76 s，距0.80 s成功条件只差0.04 s，随后张爪掉落。
- 2026-08-13 已完成RL v3代码修改：抬升/严格保持高水位封顶、抓后动作平滑代价、主动张爪/物理滑脱/接触丢失中张爪三类因果、接触丢失与主动张爪各自的一次性惩罚、secured训练态短滞回、PPO学习率/熵衰减、`target_kl`、配对seed评估和无录像诊断。没有修改扰动力、夹爪/碰撞尺寸、摩擦或0.80 s阈值。
- v3尚未进行正式长训练；只完成了8步新训练、4步续训和旧checkpoint兼容回归。下一步应从头训练`rl/runs/ppo_cable_v3`，不能把v2结果当成v3性能。
- 新盲测区间`20280804...`的10回合脚本定向基线为公共任务成功7/10、脚本内部成功6/10；3个公共失败中2个未形成夹持、1个闭爪物理滑脱。样本仍小，但滑脱存在且未主导该组失败。

## 1. 当前目标

在 MuJoCo 中搭建 Franka Panda + 保持原外轮廓、带浅槽指垫的平行夹爪抓取持续弯曲线缆的实验环境，并同时提供：

- 一个用于直观看效果和检查环境的脚本控制基线；
- 一个与脚本策略解耦、可训练和测试的 PPO 强化学习接口；
- GUI 实时观察、无头批量运行、逐回合视频和完整状态回放；
- 真实碰撞与摩擦夹持，不使用把线缆“粘”在夹爪上的弹簧力或隐藏约束。

后续研究重点是根据线缆形状变化移动夹爪，使运动线段进入夹爪中心。任务主要难点应保持在动态形状跟踪与对中；抓取仍使用具有真机迁移意义的碰撞、夹紧力和摩擦模型，但不应成为任务的主要难点。不能通过放大夹爪、扩大碰撞盒或增加捕获范围的方法替对中算法兜底。

## 2. 用户已经明确的设计要求

1. 线缆必须持续发生明显的局部弯曲，尤其是横向弯折，不能只是整体滚动或短时间后静止。
2. 环境扰动力与策略动作独立，不能因为闭合夹爪就降低扰动。
3. 抓取采用 MuJoCo 真实碰撞、夹紧力和摩擦承重，不使用弹性吸附、焊接约束或人为跟随力。
4. 夹爪保持 Menagerie 原始外轮廓；当前允许已有的浅槽指垫，但不能扩大指垫或碰撞范围来降低动态对中难度。
5. 指垫—线缆摩擦可以增强；线缆—桌面摩擦保持较低，使抓空后线缆更倾向被推开，而不是被桌面摩擦挤回夹爪。
6. 环境、策略和 RL 方法端必须分离。新算法应能自由输出动作，环境不能暗中改动作。
7. 线缆因为环境扰动自行掉出桌外应受软边界抑制，但机械臂仍应能够把线缆推出桌面。
8. GUI 应实时仿真；无头模式应快速运行，并按回合保存视频和可交互回放的完整状态。
9. 代码注释使用中文，保留有解释价值的注释。

## 3. Git 状态

仓库就在项目目录内，不在上一级 `dynamic_cable`：

```powershell
cd path\to\panda_cable_grasp
git status --short --branch
```

- 当前分支：`feat/linux-server-portability`
- 当前 HEAD：`c42bc37 feat: move L1/L2 motion into ID`（其后仍有未提交修改）

最近提交：

```text
65b0c72 Restore official Panda fingertip geometry
5379b80 Replace elastic grasp proxy with frictional contact
6a6502d RL
c7d5293 init
```

当前有大量**尚未提交**的修改，不能执行 `git reset --hard`、`git checkout -- .` 或批量覆盖。主要包括：

- 修改：`.gitignore`、`README.md`、`IMPLEMENTATION_REPORT.md`、`cable_grasp_env.py`、`dynamic_grasp_policy.py`、`panda_cable_grasp.xml`、`run_grasp.py`；
- 新增：`render_recording.py`、`replay_recording.py`、整个 `rl/` 子目录；
- 原根目录 RL 文件显示为删除，是因为它们已移动到 `rl/`，不是要丢弃 RL 功能。

建议新会话开始后先重新运行 `git status` 和 `git diff --stat`，然后再编辑。

## 4. 模型文件与 Menagerie 资产

路径双副本问题已经消除：

- `panda_cable_grasp.xml` 是程序实际加载的任务模型；
- `models/panda.xml` 是仓库固定的 Panda 与浅槽指垫模型；
- 外部 Menagerie 仅提供 `franka_emika_panda/assets/` 下的官方 mesh。

Menagerie 可由 `MUJOCO_MENAGERIE_PATH` 指向任意位置；未设置时依次检查仓库内、
`third_party/` 和仓库旁的常见位置。不要再复制任务 XML 到 Menagerie，也不要直接修改
Menagerie 的 `panda.xml`。Linux 安装和固定资产 commit 见 `LINUX_SERVER_SETUP.md`。

## 5. 当前软件环境

已核对版本：

```text
Python                 3.11.15
mujoco                 3.11.0
numpy                  2.4.6
opencv-python          5.0.0
gymnasium              1.3.0
stable-baselines3      2.9.0
torch                  2.13.0+cpu
torch.cuda.is_available() = False
```

PPO 当前通过多个 CPU 子进程并行环境训练，不是 MJX/JAX 或 MuJoCo Warp GPU 批量仿真。

## 6. 文件职责

### 非 RL 主体

- `cable_grasp_env.py`
  - 加载模型；
  - 环境 `reset/step`；
  - 线缆持续行波扰动力；
  - 桌面软边界；
  - 指垫接触读取、摩擦抓取状态和成功判定；
  - 不负责生成机械臂动作。
- `dynamic_grasp_policy.py`
  - 脚本基线策略；
  - 目标线段短时速度外推；
  - 阻尼最小二乘 IK；
  - `SETTLE → APPROACH → INTERCEPT → CLOSE → LIFT → CARRY → HOLD → RELEASE` 状态机；
  - 抓空时最多两次 `RECOVER`。
- `run_grasp.py`
  - GUI 与 headless 调度；
  - 打印阶段变化和失败诊断；
  - headless 每回合录制 MP4 和完整 MuJoCo 状态。
- `run_demo.sh`、`run_demo.ps1`
  - 使用当前环境中的 `python` 启动 `run_grasp.py`。
- `panda_cable_grasp.xml`
  - 程序实际加载的场景，包含桌面、线缆 composite、线缆参数和桌面—线缆接触 pair。
- `models/panda.xml`
  - 仓库固定的 Panda 与浅槽指垫定义；外部 Menagerie 只提供 mesh。
- `replay_recording.py`
  - 加载 `.npz` 状态文件并打开 MuJoCo 窗口；
  - 可用鼠标在线选择任意视角，按 `V` 用当前视角导出完整视频。
- `render_recording.py`
  - 已知相机参数时批量离线重渲染状态文件。
- `README.md`
  - 当前运行命令与功能说明。
- `IMPLEMENTATION_REPORT.md`
  - 较完整的历史实施报告，部分内容可能落后于未提交代码，应以代码为准。

### RL 子目录

- `rl/rl_cable_env.py`：Gymnasium 包装、观察、动作映射和奖励；
- `rl/train_rl.py`：Stable-Baselines3 PPO、多进程训练、断点和模型保存；
- `rl/test_rl.py`：无头成功率测试与 GUI 测试；
- `rl/rl_training_metrics.py`：最近 100 回合指标、CSV 和曲线；
- `rl/run_rl_train.ps1`、`rl/run_rl_test.ps1`：PowerShell 启动器；
- `rl/requirements_rl.txt`：RL 依赖。

旧 RL 模型是在先前物理/奖励版本下训练的，不能代表当前摩擦抓取环境的效果，应重新训练。

## 7. 当前物理模型

### 线缆

- MuJoCo `mujoco.elasticity.cable` 插件；这里的 elasticity 是线缆本体弯曲/扭转模型，不是夹爪吸附；
- `count="41 1 1"`，运行时生成 40 个线缆 body；
- 长度 `0.8 m`；
- 胶囊半径 `0.014 m`，即直径约 `28 mm`；
- `density=150`；
- `twist=1e4`、`bend=2e3`、`vmax=0.08`、主关节 `damping=0.025`；
- `timestep=0.002 s`，即 500 Hz 物理步；
- 一个策略动作执行 10 个物理步，即控制频率约 50 Hz。

线缆持续扰动由 `_apply_cable_disturbance()` 产生。默认 `disturbance_strength=1.5`，三个方向基础加速度尺度分别约为纵向 `5`、横向 `19`、竖向 `9`，横向最强。扰动力按节点质量换算为力，并且不依赖夹爪命令。

### 接触和摩擦

当前为纯物理摩擦抓取，没有任何弹簧吸附力、重力补偿、等式约束或位置跟随带。

Python 在模型加载后识别左右手指共 10 个浅槽 box 指垫碰撞 geom，并设置：

```text
priority = 1
condim = 6
friction = [4.0, 0.10, 0.05]
solref = [0.004, 1.0]
solimp = [0.98, 0.995, 0.0005, 0.5, 2.0]
```

`gripper_force_scale=5.0` 会在内存中缩放第 8 个夹爪执行器的位置增益和偏置，提高闭爪刚度；执行器最大 `forcerange` 保持不变。

线缆—桌面通过 40 个显式 `<pair>` 使用较低滑动摩擦：

```text
friction = [0.12, 0.12, 0.003, 0.0001, 0.0001]
```

当前实际 XML 求解器为 Newton + elliptic cone + `impratio=20` + `noslip_iterations=1`。

### `solref` 已核实结论

当前 `timestep=0.002` 且 `refsafe` 默认启用。正数格式 `solref=(timeconst, dampratio)` 的 `timeconst` 内部不得小于 `2*timestep=0.004`。因此旧值 `0.002` 实际也会被内部限制到 `0.004`；代码现已显式设置 `[0.004, 1.0]`。MuJoCo 不会改写 `model.geom_solref` 的原始配置值，限制发生在约束计算内部。

## 8. 最近完成的控制点统一修改

旧实现同时维护：

```python
HAND_LOCAL_POINT = [0, 0, 0.145]
PAD_CENTER_LOCAL = [0, 0, 0.1029]
```

现在已统一为：

```python
GRASP_CENTER_LOCAL = np.array([0.0, 0.0, 0.1029])
```

其含义是 `hand` body 坐标系中的实际两指主指垫夹持中心：手指根部偏移 `0.0584 m` 加主指垫局部中心 `0.0445 m`，合计 `0.1029 m`。

当前：

- `hand_position` 返回实际夹持中心的世界坐标；
- `pad_center_position` 直接返回同一个点；
- point Jacobian 使用 `GRASP_CENTER_LOCAL`；
- IK `position_error`、APPROACH、INTERCEPT、CLOSE、LIFT、CARRY、RECOVER 全部围绕该点；
- INTERCEPT/CLOSE 已删除旧的 `target + [0,0,-0.036]` 补偿，当前直接使用 `desired = target.copy()`；
- APPROACH 仍表示实际夹持中心位于目标上方 `0.20 m`。

全项目搜索已确认没有遗留的 `HAND_LOCAL_POINT`、`PAD_CENTER_LOCAL`、数值 `0.145` 或代码中的 `-0.036`。实施报告里出现 `-0.036` 仅是在说明该旧补偿已删除。

这项修改已经通过 Python 语法编译和模型加载检查，但尚未完成足够的 GUI 多回合行为验证。由于控制点从 `0.145` 改为 `0.1029` 后，初始 `hand_position` 世界坐标约为 `[0.5545, 0, 0.5216]`，轨迹视觉和抓取对齐应在新会话中优先实测。

## 9. 当前抓取检测与成功判定

### 抓取候选

`_physical_grasp_candidate()` 要求：

1. 夹爪总开口不大于 `0.034 m`；
2. 任一局部线段或其相邻节点同时与左右浅槽内指垫发生接触；当前暂不要求它与随机参考目标节点相同；
3. 左右两侧汇总法向力都至少为 `0.20 N`；
4. 候选节点距实际夹持中心不超过 `0.055 m`。

候选连续存在 `0.06 s` 后，`bilateral_confirmed=True`。首次确认必须满足两侧法向力阈值；确认后若左右指垫仍各有真实碰撞，即使瞬时法向力低于阈值也保留状态。只有真实双侧碰撞丢失累计超过 `0.35 s`，才记录 `lost_physical_pad_contact` 并清除 `grasp_state`。这个滞回只用于判定和日志，不向线缆施力。

### 成功条件

夹爪命令必须处于闭合状态，并连续 `0.80 s` 满足：

- 当前确认抓取状态尚未因真实接触丢失超过0.35 s而清除；
- 当前/最近抓取节点高度大于 `0.14 m`；
- 至少 18% 的线缆节点高度大于 `0.055 m`；
- 该节点距夹持中心不超过 `1.5 × 0.055 = 0.0825 m`。

当前成功资格不要求每个2 ms物理时刻都有原始双指碰撞，确认后的抓取检测会吸收最多0.35 s
短接触丢失；但`grasp_state`一旦因真实滑脱清除，公共0.80 s计时立即清零。脚本状态机仍在
`env.grasp_state is None`时立即进入失败观察，所以两者对0.35 s滞回窗口内的短暂失联仍有差异，
配对评估必须同时记录公共任务成功和策略内部成功。

## 10. 脚本策略现状

默认时序：

```text
SETTLE 0.8 s
APPROACH：目标上方 0.20 m，超时 6 s
INTERCEPT：追踪预测参考目标；任意实际线缆中心线进入夹持中心 0.018 m 内后锁定该段并闭爪，超时 9 s
CLOSE：基础确认时间 0.8 s；仍有真实指垫接触时额外保留 0.35 s 确认窗口
RECOVER：抓空后竖直抬高 0.18 m，最多重试 2 次
LIFT：2.4 s 内抬高 0.22 m
CARRY：1.0 s 内朝桌面中心侧移0.08 m并上移0.02 m，夹爪中心不得低于闭爪位置上方0.14 m
LIFT/CARRY/HOLD：确认抓取后只执行名义举升/保持轨迹，不再根据线缆相对运动追随夹持中的线缆
HOLD：进入后立即检查当前连续成功资格；若尚未满足则继续闭爪保持到回合上限
FAILURE_OBSERVE：确认滑脱后闭爪保持 0.6 s，然后保持闭爪结束本轮
RELEASE：仅在重试耗尽且没有形成抓取时张开 1.2 s；已确认抓取后不因计时器主动张开
```

目标预测在抓取前使用随机参考节点位置加 `0.22 s × 当前速度`，再低通滤波；一旦实际线段进入夹持中心或形成接触，就锁定并追踪该局部线段，不再被远处参考节点拉走。双侧抓取确认后停止这项线缆跟踪，转入名义举升/保持轨迹。该脚本只是基线，不代表最终论文方法，也不应成为环境对动作的限制。

脚本闭爪命令已恢复为 `ctrl=0`。标准形变场景的20-seed成对消融显示，`ctrl=20`与0的确认
抓取率同为18/20，但严格成功率由14/20降至10/20，终局物理滑脱由1/20升至7/20；“20可
稳定减小过度挤压”的解释没有得到实际接触力和穿透指标支持。HOLD只接受当前连续 `success_hold>=0.80 s`，
不再以历史 `ever_success` 掩盖后续掉落。抓后相对位置/积分/速度反馈的独立消融结果为：开启和
关闭均为5/5严格成功；关闭后末端高度 `0.157–0.177 m`、当前连续成功保持 `1.64–1.77 s`，
没有显示反馈收益，因此已从正式代码删除。固定搬运消融在同一5个初态上比较“无搬运”和
“朝桌面中心侧移80 mm并上移20 mm”，两组均为5/5严格成功，成功时间和连续保持时长逐次
相同；因此取消固定搬运也不是成功率改善的原因，默认已恢复80 mm侧移和20 mm上移。

`headless_videos/run_20260809_200100_seed20260804/` 是无固定搬运消融录像，不再对应当前默认
配置。`headless_videos/run_20260809_200707_seed20260804/` 是抬升0.18 m、连续保持0.55 s时的
5/5旧基线。当前抬升0.22 m、连续保持0.80 s的录像位于
`headless_videos/run_20260809_202134_seed20260804/`：同一5个初态为4/5，trial 3在CARRY中真实滑脱。

## 11. 运行命令

从项目目录执行。

### GUI 脚本基线

```powershell
cd path\to\panda_cable_grasp
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
```

常用参数：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1 --speed 2 --trials 0 --disturbance 1.5
```

GUI 按键：`=` 或 `]` 加速，`-` 或 `[` 减速，`Space` 暂停，`Backspace/R` 完整任务重置，`N` 新随机回合。

### 无头运行、逐回合视频和状态

```powershell
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1 --headless --trials 3
```

默认输出：

```text
headless_videos/run_时间_seed.../
  <场景名>/
    <场景名>.mjb
    trial_001.mp4
    trial_001_global.mp4
    trial_001_states.npz
    episodes.csv
    manifest.json
    ...
```

默认每次`run_grasp.py`命令建立一个新run目录；用PowerShell批量运行多个场景时，应为每条
命令传入相同的`--run-name`，从而让同一批实验共享run目录。若该run下已存在同名场景，
脚本会拒绝覆盖。

模型离屏 framebuffer 会在内存中自动扩到请求的视频尺寸，避免默认 640 宽度导致 `Image width 960 > framebuffer width 640`。

### 交互式状态回放与重新选视角

```powershell
python .\replay_recording.py `
  --states .\headless_videos\run_...\id_static\trial_001_states.npz --loop
```

回放窗口中：`Space` 播放/暂停，方向键逐帧，`R` 回到开头，`[`/`]` 调速，鼠标调整 MuJoCo 相机，`V` 以当前视角重新导出整段视频。

### PPO 训练

```powershell
powershell -ExecutionPolicy Bypass -File .\rl\run_rl_train.ps1 `
  --timesteps 2000000 --workers 6 --output .\rl\runs\ppo_cable_v2
```

这里 `timesteps` 是所有并行环境累计的环境交互步数。每步约 `0.02 s` 仿真时间；每回合上限 28 秒，即最多约 1400 步。

### PPO 测试

```powershell
powershell -ExecutionPolicy Bypass -File .\rl\run_rl_test.ps1 `
  --model .\rl\runs\ppo_cable_v2\evaluation\best\best_model.zip --headless --episodes 20
```

无头测试默认把每回合MP4、完整MuJoCo状态NPZ、逐步RL动作/奖励和本次`model.mjb`保存到
`rl_test_videos/run_时间_seed.../`。NPZ与`replay_recording.py`和`render_recording.py`兼容；
可用`--video-dir`、`--video-fps`、`--video-width`和`--video-height`修改录制参数。

## 12. RL 接口摘要

动作是 8 维归一化向量：

- 前 7 维：归一化关节目标增量，每个50 Hz动作最大±0.04 rad；
- 第 8 维：通过`-0.35/+0.35`滞回阈值选择`ctrl=20`闭合或255张开。

48维观察包含目标段位置与速度、目标相对夹持中心的位置、7轴位置/速度、两指位置/速度、
夹持中心位置、末端四元数和末端线/角速度，以及左右接触、法向力、pinch、secured grasp、
相对抬升和严格成功保持进度。pinch后目标切换到实际夹持线段。

v3奖励采用事件和整回合高水位形式：首次接触、首次pinch和首次secured grasp只奖励一次；
抓取段抬升在120 mm封顶，整线抬升比例在30%封顶，掉落/重抓不会重复领取旧高度奖励。
secured后的纯主动张爪为-8，闭爪物理滑脱和接触已丢失时张爪为-3；物理/歧义接触丢失
合计每回合最多罚一次，纯主动张爪另最多罚一次，但全部事件继续记录。严格保持奖励同样使用
整回合高水位、总额封顶0.8分。另有25分成功、时间、动作幅值/变化率和抓后
机械臂动作平滑代价。

RL现在把双侧真实接触确认单独记为`pinch`。只有接触线段相对pinch建立高度上升至少30 mm，
并在真实双侧接触、开度和中心距离合格时连续保持0.10 s，才记为`secured grasp`。
`pinch_rate_100`、`grasp_rate_100`和`success_rate_100`分别统计接触夹持、有效承载抓取和严格成功；
单纯闭爪接触不再计入抓取率、抓取奖励或成功终止。

当前新版定义下的`ppo_cable_v2`已完成200万步训练。训练日志共有1443回合，其中763回合曾形成
pinch（52.9%）、208回合曾形成secured grasp（14.4%）、28回合严格成功（1.94%）。分阶段看，
125万步后才开始出现严格成功，150万至175万步是表现较好的区间，之后又发生退化。

固定10个评估seed的最佳检查点位于约160万步：成功3/10、pinch 8/10、secured grasp 5/10，
平均回报14.759。190万和200万步的严格成功率均回落到0/10。因此继续训练并非单调改善；测试应优先
使用`rl/runs/ppo_cable_v2/evaluation/best/best_model.zip`，不要误用`final_model.zip`。

默认训练课程在前100万步将扰动从0.30线性增加到完整1.50，后100万步保持完整扰动；独立验证
始终使用完整1.50，每10万步在10个固定新种子上评估，按严格成功率保存最佳模型。

## 13. 已完成的检查

本次交接前已执行：

- 所有主要 Python 文件 `py_compile`：通过；
- `CableGraspEnv()` 模型加载：通过；
- 当前加载结果：`timestep=0.002`、40 个 cable body、控制点 `[0,0,0.1029]`；
- 当前实际求解器：Newton、elliptic、`impratio=20`、`noslip_iterations=1`；
- 当前内存中指垫 `solref`：`[(0.004, 1.0)]`；
- 旧控制点和旧 `-0.036` 代码搜索：无遗留。
- RL v2 `py_compile`、Gymnasium `check_env`、48维观察、夹爪滞回、实际线段锁定、事件奖励、
  扰动课程、严格独立验证、训练/测试脚本smoke test：通过。
- RL headless MP4/NPZ录制、MJB保存和`replay_recording.py --check`：通过。

静态/加载检查已通过；当前抬升0.22 m、连续保持0.80 s的MP4已生成于
`headless_videos/run_20260809_202134_seed20260804/`，自动判定同一5个初态为4/5；仍建议人工复核录像中的实际夹持质量。

## 14. 当前RL结果、问题和建议优先级

### 14.1 最佳模型的5局录像复核

测试命令使用`evaluation/best/best_model.zip`、完整扰动1.5和seed 20270804，结果为1/5成功：

| 回合 | 结果 | 逐步回放结论 |
| --- | --- | --- |
| 1 | 失败 | 多次形成有效承载，峰值相对抬升约375 mm；既发生主动张爪，也发生闭爪状态下的物理滑脱。 |
| 2 | 失败 | 形成短暂pinch但未secured；约3.90 s主动张爪，8.36 s后基本持续张开。 |
| 3 | 失败 | 最接近目标约28 mm，但未形成双侧夹持，反复开闭并将线缆推离。 |
| 4 | 成功 | 4.68 s pinch，随后保持闭爪并抬升，7.04 s满足连续0.80 s严格成功。 |
| 5 | 失败 | 峰值相对抬升约404 mm，严格资格最长保持0.76 s，随后状态闪断并在5.10 s主动张爪。 |

终端打印的`max_z`是**最终帧所有线缆节点的最高世界坐标**，不是整局峰值抬升。例如失败回合最终
`max_z=0.019 m`表示线缆已落回桌面附近；真正的相对抬升看`grasp_lift_delta`，但当前终端也只打印
最终帧数值。逐步复核时应从NPZ/state重放计算整局峰值。

### 14.2 已确认的算法问题

1. **抓后动作过激。** 5局中前7维动作平均绝对值约0.77–0.80，约60%的关节输出绝对值超过0.9；在每步最大±0.04 rad的映射下，抓住后仍持续给出接近饱和的关节增量，容易拉扯和甩落线缆。
2. **策略会主动张爪。** 第2、5局以及第1局部分失效均由第8维跨过开爪阈值直接触发，不全是外力导致的物理滑脱。
3. **抬升奖励未在合理高度封顶。** 策略会把线缆继续拉到约0.4–0.5 m，而成功只要求被夹线段高于0.14 m，这鼓励过度上拉而非安全高度保持。
4. **失败回合仍可获得较高回报。** 第1局失败但回报12.445；多次抬升的正奖励可抵消一次5分滑脱惩罚，重新抓取后还可能再次累计抬升进度。
5. **secured/严格保持对瞬时接触闪断较敏感。** 第5局已达到0.76/0.80 s；最终成功阈值不应随意放宽，但训练状态可考虑短时滞回，避免单帧接触噪声反复清零。
6. **PPO后期退化。** 策略标准差约从0.82升至1.17；训练后期`approx_kl≈0.03`、`clip_fraction≈0.28–0.30`，同时学习率一直3e-4、熵系数一直0.01，最佳策略被后续更新破坏。

### 14.3 原建议及实施状态

1. 将抬升进度奖励在安全目标高度附近封顶，并防止掉落/重抓反复获取同一段抬升奖励。
2. 加强secured后主动开爪和丢失夹持的惩罚或保持闭爪信号，但不要在环境端强制锁死策略动作。
3. 增强抓后动作幅值、速度或加速度代价，使控制更接近真机可执行轨迹；不要用降低控制频率掩盖问题。
4. 对已确认夹持状态增加很短的接触判定滞回，仅提高状态估计鲁棒性，不施加额外夹持力，也不修改最终0.80 s严格成功标准。
5. 降低后期学习率和熵，考虑`target_kl`、减少`n_epochs`或学习率衰减，防止160万步后的策略退化。
6. 先做短smoke test，再重新训练；最终至少使用100个独立随机seed评估，并分别报告pinch、secured grasp和strict success。
7. **仍需统一脚本与RL评估口径。** 当前规则方法固定5局4/5与RL固定10局3/10不能直接比较。
8. 暂缓但保留：RL目前使用特权状态且允许实际抓取段替代原参考段；这是当前简化方案，不是现阶段问题。
9. 暂缓但保留：以后对浅槽指垫做捕获范围消融，确认浅槽没有替动态对中算法兜底。

其中1–5已在RL v3代码中实现；第6的smoke test已完成，正式长训练和100个独立seed尚未执行；
第7已由`benchmark.py`的公共任务成功、内部成功和场景指纹配对记录解决数据链路问题，但正式
论文口径仍需在大规模实验前冻结。第8–9仍保留。

### 14.4 RL v3当前实现与验证

- `rl/rl_cable_env.py`：48维观察和8维动作保持不变；加入0.06 s secured训练态滞回，但当前
  原始双侧接触缺失会立即清零0.80 s严格计时。终止成功还要求底层当前500 Hz高度、整线离桌
  比例和中心距离连续计时达标，防止50 Hz采样漏掉几何子步间断；历史成功值不能兜底。
- 抓取断开现分为`active_open`、`physical_slip`和`open_during_contact_loss`。如果物理接触已开始
  丢失、策略随后张爪，不再错误记成纯主动张爪。事件全部计数；物理/歧义接触丢失和纯主动
  张爪分别最多处罚一次，避免重复滑脱淹没回报，也避免先滑一次后免费张爪。
- `rl/train_rl.py`默认输出`rl/runs/ppo_cable_v3`；学习率`3e-4→3e-5`、熵`0.01→0.001`、
  `n_epochs=5`、`target_kl=0.015`。新训练和`reset_num_timesteps=False`续训日程均已smoke通过。
- `rl/test_rl.py --no-video`生成逐回合CSV和manifest；同时报告公共任务成功与PPO内部成功，记录
  实际seed/场景指纹、动作饱和率、首次断开原因和滑脱在失败中的占比。默认录像模式仍保留。
- `benchmark.py`在完全相同seed上配对脚本与PPO，自动断言场景指纹一致，并保存CSV、manifest
  和实际编译后的MJB。失败分类由`failure_taxonomy.py`共用，成功不会再出现在failure_counts中。
- 最终源码下的完整旧checkpoint回归：seed `20270804`在14.94 s同时达到公共与PPO严格成功；
  会话门控后整回合只有1次`secured`后物理滑脱，底层共有2次确认抓取后的物理断开，随后重抓
  并稳定成功。该回合只触发一次-3接触丢失惩罚，所有事件仍保留在CSV。这只是兼容/诊断回归，
  不是v3训练结果；旧“17次”来自把后续未重新secured的候选断开误计入，已修正。
- 已通过：`py_compile`、Gymnasium `check_env`、显式seed复现、因果分类/去重/一次性惩罚断言、
  脚本+PPO短配对、无录像评估、8步新训练、4步续训和完整28 s旧checkpoint回归。
- 与最终源码哈希一致的10回合脚本归档位于
  `benchmark_runs/v3_scripted_final/run_20260813_111148/`；最终旧PPO完整回归位于
  `rl_test_videos_v3_regression/run_20260813_111152_seed20270804/`。前者含CSV、manifest和MJB，
  后者的`--no-video`归档只含CSV和manifest（manifest内含编译模型哈希）。

### 14.5 2026-08-13实验矩阵环境已落地

- `experiment_scenarios.py`冻结20个场景：`static / rigid / shape / combined`四类运动，
  low/nominal/high幅度与频率、regular/quasiperiodic/stochastic规律，以及长度和材质OOD；
  `core / motion_sweep / ood / paper`四组suite均有稳定`scenario_id`和哈希。
- `cable_grasp_env.py`新增`factorized_v1`：shape通过质量加权投影消除净力与净力矩；rigid用
  有界平移/旋转轨迹的纯外力反馈驱动；combined精确叠加两类分量。默认未显式选场景时仍是
  原始`legacy_v1`，因此旧checkpoint回归没有被暗中换任务。
- 长度、密度、bend/twist刚度、关节阻尼和线缆摩擦通过`MjSpec`在编译前实际缩放，OOD不只是
  日志标签。`motion_diagnostics.py`用质心、Kabsch旋转和刚体配准后的shape残差验证实际运动。
- `benchmark.py --suite ...`现在按`scenario_id × seed`配对脚本与PPO，输出逐场景/运动类型/
  split分层、Wilson区间和场景等权macro结果。滑脱只是附加约束：允许发生；是否未主导失败
  使用其占全部失败比例的单侧95% Wilson上界是否低于0.5判断，不围绕它继续改环境主任务。
- `rl/train_rl.py --training-distribution id`会在冻结ID场景中逐episode采样；旧PPO只适用于
  `legacy_v1`，必须从头训练矩阵模型。π0.5和Diffusion Policy尚未实现，不能伪造比较结果。
- 已通过7项场景单元测试、四场景运动诊断、8步ID训练smoke，以及脚本/PPO core配对smoke。
  这些smoke只验证链路，不代表性能。完整协议见`EXPERIMENT_PROTOCOL.md`。

建议下一条正式命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\rl\run_rl_train.ps1 `
  --training-distribution id --eval-distribution core `
  --timesteps 2000000 --workers 6 --output .\rl\runs\ppo_cable_matrix_v1
```

训练完成后先用固定开发集选择best checkpoint，再用从未参与调参的至少100个seed运行
`benchmark.py --suite paper`；主表报告公共成功率及95%置信区间、逐场景与ID/OOD分层，
并完整报告所有互斥失败类型。真机环境的XY台/转台、局部形变驱动、同步相机和力传感规划见
`EXPERIMENT_PROTOCOL.md`第7节。

### 14.6 2026-08-16 L1/L2整体运动pilot

- 独立`pilot` suite保留6个候选：L1从`y=-0.25 m`到`+0.25 m`单程匀速直线运动；L2使用
  同起终点、同平均目标速度的单段三次Bézier曲线。各有low/nominal/high，约
  `0.167/0.25/0.375 m/s`，不折返、不循环。
- 2026-08-17将profile升级为`rigid_level{1,2}_single_pass_v2`：每轮直接通过球关节生成C/S/低频
  样条型随机弯曲初始构型，保持相邻节点长度；平移期间同时绕质心完成seed决定方向的`±24°`
  有限旋转。相同seed的L1/L2逐节点初始构型、目标段和旋转方向完全配对；reset预检完整SE(2)
  扫掠范围并只收紧X起点，避免线缆出桌。
- 整体运动pilot在0.8 s settle后开始，首次夹爪物理接触后释放平移、旋转和构型保持力；正确原因
  是防止接触后环境驱动力继续改变线缆形状、污染固定形状条件，不是为了避免滑脱。普通`shape`
  及`combined`场景接触后仍继续施加局部形变；到终点仍未接触才失败。
- 旋转阻尼使用所有节点的世界速度计算质心速度和角速度，不能直接使用free joint局部角速度；后者
  曾在特定C形seed产生符号错误和正反馈。16项场景单测已通过。标称档6个配对seed诊断归档在
  `artifacts/motion_diagnostics/rigid_l1_l2_curved_se2_v2/run_20260817_101105/`：L1/L2速度RMS均值
  `0.203/0.202 m/s`，最大转角均值`23.80°/23.81°`，形状RMS最坏值`0.643/0.621 mm`，
  单帧最大残差最坏值`0.980/0.940 mm`，边界修正均为0，语义检查通过。
- 旧直线、无旋转的视频和3/5对2/5抓取结果均已失效，不得用于选择L1/L2。v2只完成1个同seed
  端到端smoke，L1/L2都成功，视频分别位于`headless_videos/run_20260817_101202_seed20260804/`
  和`headless_videos/run_20260817_101234_seed20260804/`；该样本不能用于选择。pilot仍不进入
  `paper`，最终整体运动只会从L1/L2中冻结一种，正式研究重点仍是独立的形状变化实验。

### 14.7 2026-08-17 combined与随机弯曲初态同步

- 注册场景默认profile升级为`factorized_v2`：`static / rigid / shape / combined`全部从按seed生成的
  C形、S形或低频样条型随机弯曲构型开始；相同seed逐节点初始形状配对。`legacy_v1`以及仅为读取
  旧manifest保留的`factorized_v1`仍是直线，旧checkpoint兼容入口没有被暗中改变。
- `pilot`从6个扩为12个：除`pilot_rigid_l1/l2_{low,nominal,high}`外，新增
  `pilot_combined_l1/l2_{low,nominal,high}`。combined直接使用相同L1/L2平移和`±24°`旋转，逐物理
  步叠加去净力、去净力矩的shape分量，不使用会抵消形变的构型保持力。
- 首次夹爪—线缆物理接触后，combined只撤掉整体平移和旋转；shape分量继续施加。rigid场景则
  同时撤掉其专用构型保持力。17项单元测试覆盖随机曲线配对、节点长度、combined叠加、接触后
  分量开关和rigid构型保持的零净力/零净转矩。
- 标称combined的3个配对seed诊断归档在
  `artifacts/motion_diagnostics/combined_l1_l2_curved_v2/run_20260817_110122/`，力场分解与运动语义检查均
  通过：L1/L2质心速度RMS约`0.207–0.216 m/s`，shape残差RMS约`0.165–0.218 m`，构型保持
  加速度严格为0，边界修正为0。combined发生大形变后，Kabsch“整体转角”会混入形变造成的主轴
  改变，不能把该数值直接当作L1/L2命令转角；命令本身仍是配对的`±24°`。
- L1/L2尚未冻结，所以新增combined仍属于dev pilot。旧`id_combined_*`的有界准周期整体轨迹只
  保留作历史矩阵兼容，不能作为最终combined定义；选定L1或L2后必须用胜出轨迹替换正式combined
  场景并更新scenario ID，再进行训练或论文统计。

### 14.8 2026-08-18 L1/L2迁入正式ID

- 删除旧的有界准周期整体平移/旋转执行路径；`rigid`和`combined`现在必须显式选择
  `rigid_level1_single_pass_v2`或`rigid_level2_single_pass_v2`，不再允许回退到`factorized_v2`
  整体轨迹。`factorized_v2`只继续承担shape生成和随机弯曲初态。
- 原12个DEV `pilot_*`场景迁为ID：`id_rigid_l{1,2}_{low,nominal,high}`与
  `id_combined_l{1,2}_{low,nominal,high}`。旧`id_rigid_{level}`、`id_combined_{level}`和
  `pilot_*`名称均不再注册。
- DEV改为5个纯shape单因子校准场景；OOD在冻结最终轨迹前显式保留L1/L2配对名称，禁止暗中
  选取某一轨迹。冻结L1或L2后，应删除另一组ID/OOD场景并重新冻结scenario ID。
- 环境内部及info/manifest字段从`rigid_pilot_*`改名为`rigid_motion_*`。确认双指稳定抓取后撤除
  整体驱动的规则保持不变；combined的shape驱动继续。32项单元测试通过。

### 14.9 2026-08-18 固定全局 RGB 相机与双路录像

- 在模型编译阶段把 `global_camera` 挂到 world body，默认分辨率为
  `320 x 240`，固定斜上方视野覆盖整条线缆及L1/L2整体运动范围。
- `CableGraspEnv` 的 observation 新增 RGB `uint8` 图像 `camera_rgb`；通过
  `EnvConfig.camera_observation_enabled=False` 可关闭渲染。
- `run_grasp.py --headless` 每回合同步保存诊断总览 `trial_XXX.mp4` 和全局相机
  `trial_XXX_global.mp4`，并将相机参数、路径写入 NPZ、CSV 和 manifest。
- 状态诊断、benchmark 和既有 48 维 RL 包装器内部显式关闭相机，避免改变其输入协议。

## 15. 给新会话的建议开场提示

可以把下面内容与本文档一起发给新会话：

> 请先完整阅读仓库根目录的 `NEW_SESSION_HANDOFF.md`，然后检查仓库当前`git status`，不要撤销或覆盖未提交修改。RL v3的奖励、PPO稳定性和配对评估代码已完成smoke，但尚未正式长训练；PPO v2的3/10只能作为历史诊断，不能代表v3。请先核对第14.4节，再从头训练`rl/runs/ppo_cable_v3`并用独立seed配对评估。除非我明确要求，不要修改线缆扰动、夹爪/碰撞尺寸、摩擦、抓取物理或0.80 s阈值，也不要通过环境端强制改写策略动作降低任务难度。
