# 可变形态操作对象与多对象动态场景

Status: implemented + locally smoke-tested (2026-09-20，v4 真 mesh 蒙皮：
fish 族用 barramundi.glb 真实鱼模型带贴图)。演示视频在
`docs/demo_media/dev_*.mp4`（NERO 机械臂）。

## 目标

把操作对象从单一 0.8 m 线缆扩充为「多个可变形细长对象族 + 多对象
动态布局」，为抓取/RL/感知/后续 VLA 实验提供更丰富的目标分布。
对象**不限于生物**，且要求渲染出来一眼可辨（族颜色 + 附着装饰
geom），不是换颜色的胶囊串。

## 1. 对象族（`src/panda_cable_grasp/env/objects.py`）

所有对象仍用 MuJoCo elasticity `composite type="cable"`（每对象 41
粒子、1 free joint + ball joints、独立命名前缀），区别在于物理参数、
半径轮廓、**运动模板步态**和**视觉装饰**。运动用文献运动学模型驱动
——运动模板 + PD 伺服跟踪（非解剖网格）：

| family | 形态 | 步态公式 | 外观标识 |
|---|---|---|---|
| `cable` | 0.80 m 均匀圆柱 | 无（沿用 shape/rigid 扰动通道） | 深蓝 |
| `fish` | 0.60 m 纺锤锥形 | **carangiform** 鲹形（后体优势行波 `0.10+0.20ŝ+0.70ŝ²`）+ 垂向尾摆 | 蓝身 + 尾鳍/背鳍/双眼 |
| `loach` | 0.45 m 细长锥形 | **anguilliform** 鳗形（全身行波 `0.55+0.45ŝ^1.5`） | 黄褐 + 口周三对触须 |
| `snake` | 0.85 m 均匀细柱 | **serpenoid**（Hirose 切向角波 `θ(s,t)=α·sin(ωt−βs)`） | 绿身 + 膨大头部 + 黄眼 |
| `worm` | 0.40 m 短柱 | **peristaltic** 蠕动（轴向收缩波） | 粉红短身 |
| `spring` | 0.50 m 均柱 | peristaltic 轴向压缩脉冲 | 橙色螺旋环片（Slinky） |
| `ribbon` | 0.70 m 细柱 | anguilliform 大行波 + 垂向甩动 | 粉色扁板条（体操彩带） |
| `hose` | 0.90 m 粗柱 | serpenoid 慢速摆动 | 青绿粗管 + 黄铜喷头 |
| `whip` | 0.85 m `whip_taper` | carangiform（幅值向尖端放大） | 蓝身 + 棕握柄 + 末端响梢 |

公式实现在 `gait_template()`（节点参考位置/速度）与
`template_frame_state()`（模板质心恒速/恒定转向率解析积分）。
`gait_amplitude_scale`/`gait_frequency_scale`/`gait_swim_speed` 在场景
级覆盖族默认。

### 蒙皮外观（`skin_xml()`，`DeformableSpec.skin`）

每个非线缆族生成 `<skin>` 蒙皮网格（`<asset>` 内，`vertex`/`face`
内联 + `<bone body vertid vertweight>`）：

- 网格 = 每节点一个椭圆截面环（8 顶点）连成的管面 + 两端封口 +
  **族剪影**（鱼=侧扁身+V 形叉尾鳍+背鳍脊；蛇=头端膨大；蠕虫=环纹；
  彩带=扁宽带；水管=粗管；鞭=锥形管）。
- 每站顶点刚性绑到对应 composite 节点 body（`bindpos`=直线绑定姿态），
  皮肤随骨骼变形——**纯渲染层**：力/质量/接触/抓取判定/观测完全不变，
  驱动仍是 `elasticity.cable` 插件 + 步态伺服。
- `skin=True` 时物理胶囊 geom 以 `rgba alpha=0` 隐藏（仍参与碰撞）。
- composite 节点局部轴：+X=链切向（头→尾）、+Z=竖直、+Y=侧向。

#### 真 mesh 蒙皮（`DeformableSpec.skin_mesh`，v4）

`skin_mesh` 指向 `assets/mujoco/objects/` 下的艺术家建模资产
（glb/obj），`_mesh_skin_xml()` 用 trimesh 加载 → 按 `MESH_ASSETS`
注册表的原生轴约定旋转归一化（+x=头→尾、+z=上、+y=侧、身长=
`spec.length`）→ 顶点按弧长投影**线性混合**绑相邻两根骨骼
（比程序化皮肤的单骨骼刚绑更平滑）→ 若资产带 UV+贴图则同时
生成 `<texture>`/`<material>` 并让 `<skin>` 带 `texcoord`。

- 已接入资产：
  - `barramundi.glb`（Sketchfab CC0 真鱼，2188 顶点/3864 面/1024²
    贴图，`shape_scale=(0.78,0.55)` 压扁成细长体型）→ `fish` 族。
  - `snake_poly.glb`（Google Poly CC-BY 真蛇，478 顶点/32² 调色板
    贴图，原生即为直管蛇姿）→ `snake` 族。
  鱼眼/鳍、蛇头/鳞纹理由 mesh 自带，对应装饰 geom 随之省略。
- 低顶点密度网格注意：`_mesh_skin_xml` 会给区间内无顶点的
  "空骨骼"补一个权重 0 的最近顶点（`<bone vertid>` 为必填）。
- 网格姿态要求：必须是**近似直线姿态**（鱼/蛇身长轴单调对应弧
  长）；盘旋/蜷曲姿态的 mesh 不能直接按轴向投影绑定。
- 贴图经 `object_assets()` 并入 `MjSpec.from_string(assets=)`；
  mesh 本体在 XML 生成期读盘转内联顶点，运行时无外部依赖。
- 新资产接入步骤：文件放进 `assets/mujoco/objects/` → 在
  `MESH_ASSETS` 登记 `length_axis/head_at/up_axis/texture` →
  族表加 `skin_mesh="<file>"`。需要 `pip install trimesh`（懒加载，
  不设置 skin_mesh 的场景无此依赖）。
- `material` 带 `emission="0.18"` 补偿鱼背暗纹在阴影下的可读性。

### 视觉装饰（`decoration_geoms()` + `_add_object_decorations()`）

蒙皮之外的补充标识 geom（`contype=0 conaffinity=0 density=0`，
命名 `{prefix}Deco*`，不进 `geom_ids`）：鱼/蛇的**眼**、蛇头、泥鳅
触须、弹簧螺旋环片、水管铜喷头、鞭子握柄与响梢。蒙皮已提供剪影的
部位（鱼鳍、彩带板）不再重复装饰。

### 渲染颜色（关键修复）

`elasticity.cable` 插件默认按应力配色（`vmax>0`），会在**每次
`mj_forward` 和 `Renderer.update_scene`** 里把生成 geom 刷成蓝色——
写 `model.geom_rgba` 无效。修复：生成场景的 plugin config 用
`vmax="0"` 彻底关闭应力配色，族 `rgba` 在编译/步进/渲染全链路保留。
legacy 单线缆 XML 仍用 `vmax=0.08` 保持原有应力蓝外观。

## 2. 机器人

主机械臂为 **NERO**（`ROBOT_SPECS["nero"]`，base_link/link1..7）。演示
与冒烟场景均显式 `replace(cfg, robot="nero")`，不依赖默认值。Panda
仍保留在 `ROBOT_SPECS` 中可切换。

## 3. 多对象布局（`EnvConfig` 字段）

`multi_object_layout`：`single | conveyor | parallel | crossing`

- **conveyor**：对象沿车道反方向按 `conveyor_spacing` 排队，共享
  `conveyor_direction_deg`/`conveyor_speed`/`conveyor_start_delay`，
  依次进入工作区——"传送带一条一条传过来"。
- **parallel**：多条车道并排（横向间距 `conveyor_spacing`），同时出发。
- **crossing**：两条对象在 `crossing_angle_deg` 夹角的车道上穿过桌心。
- 越线语义与 L1 一致：世界坐标系终点线投影
  `dot(com−world_center, lane_dir) ≥ conveyor_travel−0.70`，越线后
  车道/步态模板冻结，`rigid_motion_finished` 要求**全部**车道对象
  越线；抓取目标自动切到离拦截点最近的未越线对象。

关键 `EnvConfig` 字段：`object_family`、`object_families`（异构编队）、
`n_objects`(1–4)、`conveyor_*`、`crossing_angle_deg`、`gait_*_scale`、
`gait_swim_speed`（None=族默认，0=原地摆动）、`gait_stiffness/damping/
max_acceleration`（伺服增益）。默认单线缆仍走原 XML 路径
（`_uses_object_scene=False`），行为与场景 hash 不变。

## 4. 注册 DEV 场景（`scenarios/registry.py`）

| 场景 | 内容 | 脚本策略冒烟结果 |
|---|---|---|
| `dev_fish_swim` | 鲹形鱼原地尾摆 | **难**：能形成抓取但提举时滑脱（见 §6） |
| `dev_loach_wriggle` | 鳗形泥鳅原地扭动 | success (lifted 1.0) |
| `dev_snake_serpent` | serpenoid 蛇原地蜿蜒 | success (lifted 0.63) |
| `dev_worm_crawl` | 蠕动虫 | success (lifted 0.95–1.0) |
| `dev_spring_pulse` | 弹簧轴向压缩脉冲 | 构建/步进通过（NERO） |
| `dev_ribbon_wave` | 彩带大摆+垂向甩动 | 构建/步进通过（NERO） |
| `dev_hose_swing` | 水管慢速蛇摆+铜喷头 | 构建/步进通过（NERO） |
| `dev_whip_lash` | 鞭子尖端增幅甩动+握柄 | 构建/步进通过（NERO） |
| `dev_multi_conveyor3` | 3 条形变线缆排队输送 | success (lifted 0.8) |
| `dev_multi_conveyor3_rigid` | 3 条线缆刚性输送 | 构建/步进通过 |
| `dev_multi_crossing2` | 2 条线缆 90° 交叉 | success (lifted 1.0) |
| `dev_multi_parallel3` | 3 条线缆并行车道 | success（目标自动切换） |
| `dev_multi_menagerie3` | 缆+鱼+蛇异构传送带 | success (lifted 0.73) |
| `dev_multi_menagerie4` | 弹簧+鱼+彩带+蛇混合传送带 | 构建/步进通过（NERO） |

旧 29 个正式场景的 `scenario_id` 全部不变（默认对象字段不参与
identity payload）。

## 5. 兼容性改动

- **抓取判定对象化**（`environment.py`）：接触邻域、confirmed-grasp
  持久性、lifted_fraction 都限定在单一对象内。
- **厚度自适应孔径**：`aperture_limit = clip(0.040 + 2·(R_max−0.014),
  0.02, 0.12)` 按对象最大胶囊半径自适应；`DeformableSpec.grasp_aperture`
  可显式覆盖（蠕虫 0.08 / 鱼 0.10 / 水管 0.075）。
- **脚本策略**（`policies/scripted.py`）：`_nearest_cable_point` 只在
  目标对象节点内搜索；`locked_node_ids` 记录锁定段所属对象；闭爪
  触发距随对象半径放宽。
- **RL 环境**（`rl/environment.py`）：14 点弧长观测限定目标对象切片，
  观测形状仍为 99-D。
- **`observation()`** 新增 `target_object_index`/`object_positions`/
  `object_families`。
- **对象物理 geom 集合**只含 `{prefix}G*` 胶囊（装饰 Deco* 不混入）。

## 6. 已知限制

- **鱼的逃逸行为是真实难点**：脚本策略能闭合并短暂形成抓取
  （`failed_grasp_broke_on_lift`），但 0.6 m 摆动的软鱼在提举阶段
  滑脱——保留为困难/对抗性场景。
- **步态伺服是运动学跟踪**，不是解剖仿真：自推进效应（serpenoid
  爬行的涌现推进 ~2–3 cm/s）是物理真实的，但与模板速度叠加。
- 装饰 geom 不做动力学耦合（无质量），极端加速度下可能与物理胶囊
  有微小视觉错位；装饰随 body 节点刚性绑定，不会独立穿模碰撞。
- 蒙皮顶点刚性绑定单骨骼（非线性混合），大曲率弯曲处皮肤可能有
  轻微棱角；接触仍按胶囊半径，皮肤 inflate 部分无碰撞。
- 连续创建多个 `Renderer`/`CableGraspEnv` 必须 `close()`，否则后续
  渲染帧全黑（GL 上下文泄漏）。
- `n_objects` 上限 4、`crossing` 恰好 2 个对象（校验强制）。
- legacy 单线缆场景仍是应力蓝外观（vmax=0.08），属刻意保留。

## 7. 冒烟/验证命令

```bash
# 单测（对象族公式/布局/抓取邻域/装饰/颜色/NERO/旧场景 hash 不变）
PYTHONPATH=src python -m pytest tests/unit/test_object_scenes.py -x -q

# NERO + 对象场景直接实例化
python - <<'EOF'
from dataclasses import replace
from panda_cable_grasp.scenarios import get_scenario
from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
from panda_cable_grasp.env.environment import CableGraspEnv
cfg = replace(env_config_for_scenario(
    get_scenario("dev_multi_menagerie4"), seed=7, episode_seconds=12),
    robot="nero")
env = CableGraspEnv(cfg)
env.reset(seed=7); env.step(env.ready_ctrl)
EOF
```

验证记录：43 场景注册（14 新 DEV）、29 旧场景 ID 不变、全部新族
reset+step 通过（NERO）、蒙皮 nskin 编译正常且胶囊 alpha=0 仍碰撞、
装饰 geom 无碰撞无质量、vmax=0 后族色经 step+update_scene 不变、
RL 观测 99-D 有限、81 旧单测 + 27 新单测全过、12 段 NERO 蒙皮演示
视频已生成。
