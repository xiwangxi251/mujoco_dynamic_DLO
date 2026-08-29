# 统一仿真评测

本文定义当前评测入口和产物格式。项目总体状态见[项目状态](../project_status.md)，正式实验
样本量、统计和冻结要求见[实验协议](experiment_protocol.md)。

## 入口划分

动作策略成功率只使用一个正式入口：`panda-cable-benchmark`。它目前支持
`scripted`、`expert` 和 `ppo`，并统一调用底层 `CableGraspEnv` 的
`task_success` 作为任务成功标准。

其余入口不再被视为另一套成功率测试：

- `panda-cable-grasp --headless` 是 scripted benchmark 的兼容入口；GUI 模式仅用于交互观察。
- `panda-cable-rl-eval` 用于训练期间的 PPO 奖励、阶段和 checkpoint 诊断；正式跨方法结果用 benchmark 生成。
- `panda-cable-motion-diagnostics` 检查场景运动是否符合设计，不评价抓取策略。
- `panda-cable-expert-collect` 是数据采集器，允许使用不同的数据 schema 和筛选逻辑。
- `panda-cable-dynamicvla` 是外部模型服务适配入口，但 episode 产物采用与 benchmark 相同的记录 schema。

这样环境、RL 和 DynamicVLA 不再各自定义一套“任务成功”，训练内部的里程碑也不会替代公共成功条件。

## 多场景与两级并行

例如同时比较三个本地策略：

```bash
panda-cable-benchmark \
  --methods scripted expert ppo \
  --ppo-model outputs/rl/train/example/best_model.zip \
  --scenarios id_static id_shape_low id_combined_l1_nominal \
  --episodes 20 \
  --scenario-workers 2 \
  --envs-per-scenario 4 \
  --workers 8
```

- `--scenario-workers`：一批中最多同时活跃的场景数。
- `--envs-per-scenario`：每个活跃场景最多同时运行的独立 episode 环境数。
- `--workers`：可选的总进程上限；默认是前两者乘积。
- `--suite core|motion_sweep|ood|paper|all`：使用注册的场景集合。
- `--scenarios ...`：显式指定多个场景。

并行后每个 episode 仍由独立 MuJoCo model/data、独立 seed 和独立输出目录运行。
PPO checkpoint 会在每个 worker 进程中缓存。视频渲染和多个 PPO worker 会明显增加显存/内存占用，服务器上应逐步提高并发量。

DynamicVLA 的一个 ZMQ 客户端只有一条有状态动作流，不能安全地被多个环境共享。因此当前一个
`panda-cable-dynamicvla` 服务对应一个场景，并在该场景内串行运行回合。若要并行，需要为每个环境启动独立模型客户端和独立端口；本地 benchmark 的并行参数不适用于这个外部服务入口。

## 统一输出

benchmark 的默认 seed 是 `20280804`，scripted、PPO、expert 和 DynamicVLA 评测入口共用该默认值。第 `i` 个回合使用 `seed + i`。显式传入 `--seed` 时仍以命令行值为准。

每次运行目录包含：

```text
run_YYYYMMDD_HHMMSS/
├── manifest.json
├── episodes.csv
├── summary.json
├── models/<scenario>.mjb
└── episodes/<method>/<scenario>/seed_<seed>/
    ├── episode.json
    ├── trajectory.npz
    ├── global.mp4
    └── wrist.mp4
```

`trajectory.npz` 的公共字段为：

- `states`、`state_times`：reset 后初态以及每个控制步后的 MuJoCo `mjSTATE_FULLPHYSICS`。
- `policy_actions`：策略原始输出；不同方法允许维度不同，但同一 episode 内固定。
- `requested_actions`、`applied_actions`：底层环境限幅前后的 8 维执行器动作。
- `rewards`、`reward_component_names`、`reward_components`。
- `terminated`、`truncated`。
- `frame_state_indices`、`frame_times`、`video_fps`：视频帧到完整状态的对应关系。

DynamicVLA 的 `trajectory.npz` 还通过 `extra_*` 字段保存 task-space pose、关节动作、动作接收标记和修复/裁剪诊断。

默认会记录状态和视频。仅做快速吞吐测试时可使用 `--no-recording`；这种结果不满足完整回放要求。

## 完整状态校验与回放

以下命令会载入 episode 对应的 `.mjb`，逐帧恢复 FULLPHYSICS 状态并执行
`mj_forward`，从而检查模型与轨迹是否匹配：

```bash
panda-cable-replay outputs/benchmarks/<run>/episodes/scripted/id_static/seed_20280804 --check-only
```

去掉 `--check-only` 会打开 MuJoCo viewer；`--speed 2` 可按两倍仿真时间播放。

完整复现依赖同一运行目录中的 `.mjb`、`trajectory.npz` 和 `manifest.json`。视频用于检查视觉输入，不作为物理状态的替代品。
