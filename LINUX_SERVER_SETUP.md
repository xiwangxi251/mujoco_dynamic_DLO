# Linux 服务器安装与运行

本项目支持 Linux 无桌面服务器。仓库保存任务 XML 和定制 Panda 指垫 XML；外部
MuJoCo Menagerie 只提供官方 mesh 资产，可以放在任意目录。代码不再要求
Menagerie 与项目互为固定的兄弟目录。

## 1. 系统要求

- Linux x86-64 或 AArch64；推荐 Ubuntu 22.04/24.04。
- Python 3.10 以上；当前验证版本为 Python 3.11、MuJoCo 3.11.0。
- 至少 8 GB 内存。批量并行或 RL 训练建议按 worker 数增加内存。
- 生成视频需要 EGL（GPU）或 OSMesa（CPU）离屏 OpenGL。

Ubuntu/Debian 基础依赖：

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv ffmpeg libegl1 libgl1
```

没有 GPU、准备使用软件渲染时，再安装：

```bash
sudo apt-get install -y libosmesa6 libgl1-mesa-dri
```

## 2. 获取项目和 Menagerie

```bash
git clone <项目仓库URL> panda_cable_grasp
cd panda_cable_grasp
git switch feat/linux-server-portability

git clone https://github.com/google-deepmind/mujoco_menagerie.git \
  ../mujoco_menagerie
git -C ../mujoco_menagerie checkout \
  71f066ad0be9cd271f7ed58c030243ef157af9f4
```

固定 Menagerie commit 是为了保证 mesh 资产可复现。Menagerie 不一定要放在项目旁边；
如果放在其他位置，只需要设置它的绝对路径：

```bash
export MUJOCO_MENAGERIE_PATH=/absolute/path/to/mujoco_menagerie
```

变量也可以直接指向 `franka_emika_panda` 子目录。不要再向 Menagerie 复制或修改本项目
XML；定制指垫已经保存在 `models/panda.xml`。

## 3. 创建 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
```

如果需要 PPO 训练和评估：

```bash
python -m pip install -r rl/requirements_rl.txt
```

有特定 CUDA/PyTorch 要求时，应先按服务器 CUDA 驱动安装对应的 PyTorch wheel，再执行
RL requirements；后者不会替换已经满足版本要求的 PyTorch。

## 4. 配置无头渲染和输出目录

NVIDIA GPU 服务器推荐 EGL：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

CPU 软件渲染：

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

未设置 `MUJOCO_GL` 且 Linux 没有 `DISPLAY` 时，项目会默认选择 EGL。CPU 服务器必须
显式选择 `osmesa`。输出默认位于仓库内；建议在服务器上放到 scratch/数据盘：

```bash
export PANDA_CABLE_OUTPUT_ROOT="${SCRATCH:-$PWD/server_outputs}"
mkdir -p "$PANDA_CABLE_OUTPUT_ROOT"
```

## 5. 安装自检

完整检查会加载 elasticity 插件、编译模型、执行一个物理控制步并渲染固定相机：

```bash
python scripts/check_install.py
```

最后应输出 `installation_check=OK`。仅检查模型和物理、不检查 OpenGL：

```bash
python scripts/check_install.py --no-render
```

然后运行代码回归：

```bash
python -m unittest discover -s tests -q
```

## 6. 运行实验

单场景三回合，同时保存视频、state、MJB、CSV 和 manifest：

```bash
bash run_demo.sh --scenario id_combined_l1_nominal --headless --trials 3
```

指定输出盘和 20 个回合：

```bash
python run_grasp.py \
  --scenario id_shape_nominal_current \
  --headless --trials 20 --seed 20260804 \
  --video-dir "$PANDA_CABLE_OUTPUT_ROOT/headless_videos"
```

运行 `id_static` 和所有 ID nominal 场景：

```bash
run_name="run_$(date +%Y%m%d_%H%M%S)_seed20260804"
scenarios=(
  id_static
  id_shape_nominal_current
  id_rigid_l1_nominal
  id_rigid_l2_nominal
  id_combined_l1_nominal
  id_combined_l2_nominal
)
for scenario in "${scenarios[@]}"; do
  python run_grasp.py \
    --scenario "$scenario" --headless --trials 20 --seed 20260804 \
    --run-name "$run_name"
done
```

RL 训练与测试：

```bash
bash rl/run_rl_train.sh --help
bash rl/run_rl_test.sh --help
```

## 7. Slurm 示例

```bash
#!/usr/bin/env bash
#SBATCH --job-name=panda-cable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00

set -euo pipefail
cd /absolute/path/to/panda_cable_grasp
source .venv/bin/activate
export MUJOCO_MENAGERIE_PATH=/absolute/path/to/mujoco_menagerie
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export PANDA_CABLE_OUTPUT_ROOT="${SLURM_TMPDIR:-/data/experiments}/panda_cable"

python run_grasp.py \
  --scenario id_combined_l1_nominal \
  --headless --trials 20 --seed 20260804
```

## 8. 常见问题

- `MuJoCo Menagerie Panda assets were not found`：检查
  `MUJOCO_MENAGERIE_PATH`，目录下应存在 `franka_emika_panda/assets/`。
- `GLFW ... DISPLAY`：无头服务器使用 `MUJOCO_GL=egl` 或 `osmesa`，不要启动 GUI。
- `EGL_NOT_INITIALIZED`：先确认 `nvidia-smi` 正常、作业分配了 GPU；否则改用 OSMesa。
- elasticity plugin 找不到：确认安装的是 `requirements.txt` 中的官方 `mujoco` wheel，
  不需要另行下载 MuJoCo SDK。
- MP4 无法编码：确认安装 `ffmpeg`；同时检查输出目录可写且磁盘空间充足。
- 多进程训练很慢：当前环境是 CPU MuJoCo，并非 MJX GPU 批量物理；GPU主要用于策略网络
  和视频渲染。
