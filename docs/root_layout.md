# 根目录说明

根目录只保留项目级入口和安装元数据：

| 文件 | 用途 | 是否经常修改 |
|---|---|---|
| `README.md` | 新用户入口、常用命令和文档索引 | 是 |
| `pyproject.toml` | 包信息、依赖组和 `panda-cable-*` 命令注册 | 依赖或命令变化时 |
| `requirements.txt` | 只安装核心仿真依赖的兼容清单 | 核心依赖变化时 |
| `.gitignore` | 排除虚拟环境、缓存和 `outputs/` 实验产物 | 很少 |

其余内容按职责进入目录：

- `src/panda_cable_grasp/`：唯一的产品代码实现。
- `assets/`：MuJoCo XML 和静态资产。
- `configs/`：训练、评估和集成配置。
- `docs/`：当前说明、历史记录和受版本控制的参考结果。
- `tests/`：单元测试与 MuJoCo/RL 集成测试。
- `tools/`：安装检查、消融、录像处理和启动辅助脚本。
- `outputs/`：模型、录像、数据集和运行日志，不提交 Git。

项目采用 editable install，命令由 `pyproject.toml` 暴露。不要再在根目录新增
`run_*.py` 或与包模块同名的转发文件；新入口应放到
`src/panda_cable_grasp/` 的对应子包并注册 console script，维护工具则放到
`tools/`。
