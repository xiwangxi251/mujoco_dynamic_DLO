# 文档导航

本仓库 `docs/` 只保存实验结果类文档与代码参考；整体规划类文档（项目状态、论文计划、
仓库约定）位于仓库之外的项目根 [`../../docs/`](../../docs/)。

## 目录结构

- [`environment.md`](environment.md) — 环境端总体进度（场景、对象、观测与奖励实现）
- [`reference/`](reference/) — 协议与实现参考（不随单次实验变化）
  - [`experiment_protocol.md`](reference/experiment_protocol.md) — 冻结实验协议（场景、样本量、统计口径）
  - [`architecture.md`](reference/architecture.md) — 模块架构
  - [`evaluation.md`](reference/evaluation.md) — 评测入口与产物格式
  - [`server_setup.md`](reference/server_setup.md) — 服务器环境部署
- [`experiments/`](experiments/README.md) — 实验记录，按类别分目录
  - [`policies/`](experiments/policies/) — 策略与端到端方法（Scripted/Expert、RL/PPO、目标节点 PPO、控制诊断、Diffusion Policy、DynamicVLA、pi0.5 基线）
  - [`perception/`](experiments/perception/) — OccDyn-DLO 遮挡动态感知
  - [`scenes/`](experiments/scenes/) — 可变形对象与多对象场景

实验可信度标签与新实验记录模板见 [`experiments/README.md`](experiments/README.md)。
