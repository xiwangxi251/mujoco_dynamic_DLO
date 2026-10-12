# 当前工作区：近桌面环形扇区

2026-10-10，按用户提供的 `near_table_sector_inner15_5_20261009/summary.json` 更新。

| 项目 | 范围 |
| --- | --- |
| 圆心 | 世界坐标 `(0.35, 0)` m |
| 半径 | `0.28–0.56` m，包含边界 |
| 角度 | `−30°–70°`，包含边界；世界 +X 为 0°，向 +Y 为正 |
| 面积 | 约 `0.20525 m²` |
| 工作区内测量点 | 88点，全部通过；完整扫描为127/271通过 |

来源：[测量记录](../../../outputs/vertical_workspace/scripted_136_retest_20261004/near_table_sector_inner15_5_20261009/summary.json)；[查看范围](http://127.0.0.1:8766/near_table_sector_inner15_5_20261009/index.html)。

- 测量条件：增强6＋有界DLS、碰桌保护开、旧过滤器关、原限速；夹爪闭合、固定朝向、无绳缆接触。位置误差严格小于15mm、姿态误差严格小于5°，连续保持2s，总时限16s，保留原离桌、关节和自碰撞要求。
- 该范围是半径4cm、角度10°离散采样的全通过扇区，未测连续位置和其他夹爪朝向不自动视为已验证。
- 默认配置：`src/panda_cable_grasp/rl/workspaces/near_table_sector_inner15_5_20261009.npz`。点的归属按实际半径和角度判断，不用外接矩形或最近网格单元代替；4cm网格仅为辅助表示。
- PPO工作区诊断沿用Z=`−0.005–0.10m`，这不表示整个高度区间已测量。`policy_from_initial_v2`仍从初始状态执行、自由抽取固定材料节点，工作区不会增加启动等待、目标筛选或失败条件。
- 后续新指定点PPO默认及新规则录像标注使用此范围。已有权重按保存配置；跨机器按文件SHA256寻找对应内置工作区，保留旧矩形文件。
- 已冻结录像和正在运行的`scripted_tangent_20261010`批次仍标注其原矩形，不重写结果或重新定义其工作区。

## 可视化与新版边界绕行

- [测定图](../../../outputs/vertical_workspace/scripted_136_retest_20261004/near_table_sector_inner15_5_20261009/sector_range.png)：绿点通过、红点未通过，橙线为采用的环形扇区边界。
- [新版完整边界绕行（2026-10-12）](http://127.0.0.1:8766/near_table_sector_perimeter_20261012/index.html)：NERO 全局与腕部双视角，原生1280×960、50fps；−30°径向外移、外弧至70°、70°径向内移、内弧回到−30°，各角点停留2秒。橙线为参考、绿线为实际TCP轨迹、红十字为当前目标。
- 绕行采用保存的内弧录像物理配置，参考速度0.12m/s、TCP目标离桌5mm、朝下闭爪、增强6＋有界DLS、原限速与碰桌保护。它是连续跟踪演示，不新增工作区测定结论；实际误差、源文件哈希、相机定义和轨迹见录像目录的`summary.json`、`metrics.json`和`trajectory.npz`。
- 复用工具：`tools/recording/record_sector_perimeter.py --help`。全局与腕部相机直接使用环境的`dynamicvla_opst_camera`、`dynamicvla_wrist_camera`。
