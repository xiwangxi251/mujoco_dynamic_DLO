# 独立力播放场景族（2026-10-13）

用户接受当前力播放的精度限制，但要求原训练权重继续对应原来的场景定义。
因此将已经实现的模式移入独立场景族 `cable_x075_force_v1`，恢复
`cable_x075_v1` 在本轮 Shape replay 修正之前的场景配置与 ID。
主仓库继续使用 `dev`；两个族同时注册，由训练/评测显式选择。

## 场景选择

| 用户名称 | 原族 `cable_x075_v1` | 新族 `cable_x075_force_v1` |
|---|---|---|
| 普通静态 | `dev_static_x075_v1` | `dev_static_x075_force_v1` |
| 复杂初始静态 | `dev_static_midshape_v1_x075_v1` | `dev_static_midshape_v1_x075_force_v1` |
| 简单初态整体运动 | `dev_rigid_l1_nominal_x075_v1` | `dev_rigid_l1_nominal_x075_force_v1` |
| 复杂初态整体运动 | `dev_rigid_l1_midshape_v1_x075_v1` | `dev_rigid_l1_midshape_v1_x075_force_v1` |
| Shape replay | `dev_rigid_replay_shape_nominal_x075_v1` | `dev_rigid_replay_shape_nominal_x075_force_v1` |
| 形变 | `dev_shape_nominal_current_x075_v1` | `dev_shape_nominal_current_x075_force_v1` |
| 组合运动 | `dev_combined_l1_nominal_x075_v1` | `dev_combined_l1_nominal_x075_force_v1` |

表格采用用户阅读顺序。代码中的课程顺序沿用原七项顺序，课程概率、阶段、
晋级条件和成功标准不变。新族七个场景都拥有独立名称与 ID；三个无整体运动
场景的物理配置与原族相同。

训练或评测新族使用 `--workspace-scene-family cable_x075_force_v1`。
旧族继续使用 `--workspace-scene-family cable_x075_v1`；指定点新训练的缺省
仍是旧族，不自动切换。指定点工作区课程、普通分阶段课程、ordered ability
及 rollout/consolidation 课程均使用所选族的场景名与原概率。
范围课程保留原场景定义，本次不迁移到新族。

## 驱动和权重兼容

原族三个简单整体运动场景恢复 `rigid_drive_version=feedback_v1`；
原族 Shape replay 恢复 `replay_controller_version=legacy_pd_v1`，
对应修正前 ID `scenario-v1-48ead0f573de9a62`。
原七场景的名称、初态、库、机械臂等待 0.82 秒及原 ID 均恢复。
这保持旧权重的原训练环境语义，不代表重新测量了它们的成功率。

新族三个简单整体运动场景使用 `force_tape_v1`，Shape replay 使用
`force_tape_v5`。力序列始终前进，碰撞、阻挡和抓牢不暂停、不重锚、不追赶；
实际离桌节点关闭整体驱动力，落桌后接入当前时刻的力。
组合运动仅迁移整体分量，原形变规则保持。

权重文件、观测维数、动作接口和奖励不修改。保存配置中的旧场景名仍解析到
旧定义；评测继承保存的族，显式指定新族属于另一项评测。
rollout 课程 checkpoint 增加族标识，旧 checkpoint 缺省归属旧族；
跨族直接恢复晋级统计被拒绝，避免把旧模式的已掌握阶段当作新模式证据。
本轮没有运行训练或抓取成功率评测。

本轮之前中间版本 v2/v3/v4/v5 的完整参数配置和冻结源码仍可复现自身版本；
只保存旧场景名的配置现解析为本轮修正前的原定义，不能代表中间版本。
此前已生成的力播放验收数据保留原场景名和源码记录，本次不回写历史结果。

## 精度限制保留

本次拆分不改力生成或播放算法，也不继续校准，既有误差仍适用：

- 简单/复杂初态整体运动及组合运动：各两种子、每条 2 秒的小批对照。
  质心位置 RMSE 最大 1.490 mm，整体速度向量 RMSE 最大 0.004002 m/s，
  均速偏差最大 0.199%，组合运动转角 RMSE 最大 1.80°。
- Shape replay：50 条真实库轨迹、每条 15 秒，无夹爪干预；14 条至少一项
  超过原阈值。位置 RMSE 最大 9.082 mm、速度向量 RMSE 最大 0.07017 m/s、
  位置峰值误差最大 56.09 mm、回合均速偏差最大 2.84%。

用户接受这些限制；原始精度验收仍记录为未全部通过。完整证据见
[整体力播放记录](family_rigid_force_drive_20261013.md)和
[Shape replay 验收](shape_replay_force_playback_20261013.md)。

## 拆分验证

核对原注册表的所有既有场景 ID，核对新族的环境参数与拆分前力播放配置，
并检查保存配置评测、课程选择、checkpoint 族隔离以及力播放物理回归。
原注册表 62 个既有 ID 全部一致，新族七项环境参数与拆分前模式一致；
力生成和播放源码保持原样。116 项测试、38 项子检查通过，无失败或跳过。
结果保存在工作区 `outputs/force_scene_family_20261013/verification.json`。
这里验证注册与兼容性，没有新增轨迹精度实验。
