# 规则策略闭爪硬超时恢复（2026-10-09）

## 修改

恢复 `close_hard_timeout=3.0` 的无条件上限。未确认抓取时，即使仍有指垫接触或抓取候选，也必须退出长期闭爪；已确认抓取优先进入 LIFT。保留短等待窗口的接触宽限，配置参数与公共成功判定不变。

## 配对评测

- 持续形变 `dev_shape_nominal_current_x075_v1`，种子 20260804～20260813，15 秒上限。
- NERO、`enhanced6_bounded_dls_v1`、TableExecutionGuard 开启；默认 PolicyConfig、中点＋最近点锁定。
- 修改前：`outputs/scripted_seven_default_20261009/`。
- 修改后：`outputs/scripted_shape_close_deadline_20261009/`；服务器同名目录位于 `/data1/hxai/mujoco/mujoco_dynamic_DLO/outputs/`。
- 两次使用同一冻结源码，仅 `src/panda_cable_grasp/policies/scripted.py` 不同。十组初始物理状态、环境配置、规则参数、轨迹库和相机配置一致。

| 种子 | 修改前 | 修改后 |
| --- | --- | --- |
| 20260804 | 失败 | 失败 |
| 20260805 | 失败 | 失败 |
| 20260806 | 成功 | 成功 |
| 20260807 | 失败 | 失败 |
| 20260808 | 失败 | 失败 |
| 20260809 | 成功 | 成功 |
| 20260810 | 失败 | 失败 |
| 20260811 | 成功 | 成功 |
| 20260812 | 失败 | 成功 |
| 20260813 | 失败 | 失败 |

成功率由 3/10 变为 4/10。原成功回合 806、809、811 的完整物理轨迹未改变。804 首次闭爪由 1.54 s 持续至 4.56 s 后恢复重试，未能救回最终成功。812 首次闭爪在 4.50 s 后恢复重试，最终成功。此次十回合仅提供初步改善证据。

20 段新视频为原生 1280×960、50 fps，保留全局/腕部视角与工作区/目标点标注。逐视频编码及哈希校验通过，渲染前后物理状态差异为零。配置、源码哈希、逐回合配对与阶段退出时间分别见 `manifest.json`、`source/source_provenance.json`、`comparison.json`。

## 测试

`tests/unit/test_experiment_scenarios.py`：57 passed，8 subtests passed，1 failed。修改的回归覆盖接触/候选不能豁免硬超时、接触可延长短窗口、已确认抓取优先抬升，全部通过。

失败项 `test_actual_release_resumes_rigid_motion` 在修改前冻结源码上同样失败，与本次策略修改无关；未在此修改环境释放语义。
