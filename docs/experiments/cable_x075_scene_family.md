# 指定点策略训练、测试场景族：`cable_x075_v1`

## 用途与历史实验归属（2026-10-04）

本场景族目前用于**指定点策略的后续训练和测试**。它将线缆初始 X
摆放基准从原来的约 0.620 m 调整到 0.750 m，使线缆分布更接近当前
选定的机械臂工作区（X=0.60～0.84 m、Y=−0.22～0.30 m）。
这是独立的摆放场景族，不改变指定材料点任务、策略动作接口或成功标准。

**此前的权重、训练运行和实验结果仍使用旧场景定义。**新增场景族不改变
这些历史产物的场景归属，不修改其冻结源码、训练配置、轨迹库或已有统计。
旧权重若在本场景族测试，须记录为一次新的评测，不能归入原实验结果；
以旧权重初始化新族训练时，也须单独记录新的训练运行和场景列表。

后续使用本场景族时，训练和测试应显式选择下面六个 `dev_*_x075_v1`
名称，记录场景 ID/hash、种子及轨迹库来源。当前原 PPO 训练入口的六场景
列表保留原设置，尚未自动切换到新族。代码中的 `DEV` 分区标识这批独立
开发场景，并不妨碍显式选择它们进行指定点策略训练和测试。

## 六场景定义

Separate development family requested on 2026-10-04. Select its scene names
explicitly; the original PPO training scene list is unchanged.

| Original scene | New scene |
| --- | --- |
| `id_static` | `dev_static_x075_v1` |
| `id_rigid_l1_nominal` | `dev_rigid_l1_nominal_x075_v1` |
| `id_shape_nominal_current` | `dev_shape_nominal_current_x075_v1` |
| `id_combined_l1_nominal` | `dev_combined_l1_nominal_x075_v1` |
| `id_static_midshape_v1` | `dev_static_midshape_v1_x075_v1` |
| `id_rigid_replay_shape_nominal` | `dev_rigid_replay_shape_nominal_x075_v1` |

- New family: `initial_cable_center_x=0.75` m, retaining random X offset
  ±0.10 m. Initial random center request is 0.65–0.85 m for ordinary scenes,
  subject to the existing table sweep/placement checks.
- Original scenes and default `EnvConfig`: override is `None`, preserving
  original model and bank positions exactly.
- Bank scenes: shift the source positions by `0.75 - source_basis_x`.
  Existing banks without basis metadata use their original 0.62 m basis.
  Thus their snapshots retain source drift and random displacement; their
  individual centers are not forced to 0.75 m.
- Bank files remain intact. Newly generated banks record
  `source_initial_cable_center_x` to prevent a second shift when loaded.
- Motion definitions, random draws, cable material, Y/Z positions and robot
  controls are inherited from the corresponding original scenes.
- This family is registered under the DEV split with separate scenario
  identities. Generic registry `all`/`motion_sweep` lists include these
  development variants; the explicit PPO six-scene list is unchanged.

Example:

```python
from panda_cable_grasp.scenarios.registry import (
    SCENARIO_FAMILIES, get_scenario,
)
from panda_cable_grasp.env.environment import CableGraspEnv, EnvConfig

names = SCENARIO_FAMILIES["cable_x075_v1"]
scene = get_scenario(names[0])
env = CableGraspEnv(EnvConfig(
    robot="nero", scenario_name=scene.name,
    scenario_id=scene.scenario_id, scenario_split=scene.split.value,
    **scene.to_env_overrides(),
))
```

Bank scenes additionally require the same compatible replay bank directory
and seed selection used by their original counterparts.

Verification: all 53 pre-existing scenario hashes remain identical; the six
original scenes' initial full MuJoCo states exactly match the archived
videos at seed 20260804. Each new variant is translated +0.13 m in X,
within 4e-9 m numerical error. Report:
`outputs/vertical_workspace/scripted_136_retest_20261004/cable_ppo_six_scenes/cable_x075_v1/verification.json`.
## 当前布置录像与轨迹库来源

已录制本场景族六场景各 20 回合，共 120 回合；配对种子为
20260804～20260823，每回合提供俯视、全局和腕部三种视角。
机械臂固定，未运行指定点策略，也未执行 PPO 工作区条件筛选。
这些录像用于检查线缆活动范围与黄色工作区的关系，**不属于策略训练或
策略成功率评测结果**。

当前复杂静态和刚体回放仍使用原轨迹库
`replay_src_shape_nominal_v1`。加载时将绝对摆位向 +X 平移 0.13 m；
构型与相对运动参考沿用旧库，未从新族重新生成形变源轨迹。
本批录像使用的库目录为
`outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks`。
后续训练、测试须明确记录所用库及其摆放基准；重新生成参考源库时
应另存版本，不能覆盖旧实验使用的库。

- 新族录像：`outputs/vertical_workspace/scripted_136_retest_20261004/cable_x075_20episodes/index.html`
- 原摆放录像：`outputs/vertical_workspace/scripted_136_retest_20261004/cable_ppo_six_scenes/index.html`
