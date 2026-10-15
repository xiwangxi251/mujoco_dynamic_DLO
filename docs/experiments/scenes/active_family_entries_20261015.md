# 场景族日常入口收敛

用户确认移除中间场景族的日常入口。dev的训练/评估帮助中仅显示
`cable_x075_v1`和`cable_x075_sector_speed_v3`。

`ACTIVE_SCENARIO_FAMILIES`为日常入口；`HISTORICAL_SCENARIO_FAMILIES`
保存force_v1、surface_v1、sector_speed_v1、sector_speed_v2。
`SCENARIO_FAMILIES`仍作为完整兼容查找表；显式传旧名称仍解析原场景，
不映射到新版，不改变scenario_id、快照、视频、运行源或已启动进程。
v3复用的六个v2具体场景继续保留。历史legacy_ppo_six也不在日常帮助列出。

长度实验族仍独立；其日常入口为上述两项加`cable_x075_length_half_one_v1`。
23项入口/朝向/课程检查通过；未改物理或训练参数。
