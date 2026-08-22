# DynamicVLA 零样本适配与运行

这里采用 DynamicVLA 官方的双进程方式：MuJoCo 仿真和 DynamicVLA 模型分别使用独立 Python 环境，
通过官方 ZeroMQ 消息格式通信。这样不需要安装 Isaac Sim/Isaac Lab，也不会让
`lerobot/transformers/torch` 与现有 MuJoCo 依赖相互污染。

## 已对齐的接口

- 控制频率为 25 Hz，与 DOM 数据生成和官方评测的 `physics_time_step=0.04` 一致；MuJoCo 物理仍为 500 Hz。
- 模型输入为 DOM 的两帧时序、固定 opposite camera、Panda wrist camera 和 6 维末端位姿。
- 两路相机均为 480×360，外参、视场角和 OpenGL 光轴约定来自 DynamicVLA 源码配置。
- 模型在内部输出 7 维 `xyz + Euler + gripper`；官方推理端将其变为
  8 维 `xyz + quaternion(wxyz) + gripper` 后发送。
- 适配器以分层阻尼 IK 转为本项目的 7 个关节位置目标，最后仍经过环境端统一速度、加速度、
  笛卡尔速度和夹爪速度限制。
- 每回合保存模型实际看到的 opposite/wrist 视频、逐帧 MuJoCo state、原始模型动作、IK 动作、
  `episodes.csv` 和 `manifest.json`。

## 1. MuJoCo 环境

```bash
cd /path/to/panda_cable_grasp
source .venv/bin/activate
python -m pip install -e ".[dynamicvla]"
```

Windows 的现有 conda 环境：

```powershell
conda activate dynamic
python -m pip install -e ".[dynamicvla]"
```

先做不加载权重、不运行实验的桥接自检：

```bash
panda-cable-dynamicvla \
  --scenario id_static \
  --instruction "Pick up the blue cable." \
  --check-only
```

## 2. DynamicVLA 模型环境

官方验证组合是 Python 3.10 和 PyTorch 2.7.1。建议新建独立环境：

```bash
conda create -n dynamicvla python=3.10 -y
conda activate dynamicvla
python -m pip install --upgrade pip wheel setuptools
```

NVIDIA 服务器按其 CUDA/驱动安装 PyTorch 2.7.1；例如 CUDA 12.8 wheel：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
cd /path/to/DynamicVLA
python -m pip install -r requirements.txt
```

`pylibmc` 只用于该仓库的训练数据 memcached 缓存，模型推理不会导入它。Windows 没有官方 wheel，
若 `pip install -r requirements.txt` 因缺少 `libmemcached/memcached.h` 失败，使用：

```powershell
$packages = Get-Content requirements.txt | Where-Object { $_.Trim() -ne "pylibmc" }
python -m pip install $packages
```

Linux 若确实需要训练缓存，可先安装 `libmemcached-dev` 后保留该依赖；只做推理时同样可以跳过。

权重仍在下载时可只检查 config 和依赖，不会加载模型：

```bash
python /path/to/panda_cable_grasp/tools/installation/check_dynamicvla_install.py \
  --dynamicvla-root /path/to/DynamicVLA \
  --weights /path/to/DynamicVLA/ckt/dynamic-vla-DOM \
  --allow-incomplete-weights
```

Windows PowerShell 对应写法：

```powershell
conda activate dynamicvla
python C:\path\to\panda_cable_grasp\tools\installation\check_dynamicvla_install.py `
  --dynamicvla-root C:\path\to\DynamicVLA `
  --weights C:\path\to\DynamicVLA\ckt\dynamic-vla-DOM `
  --allow-incomplete-weights
```

## 3. 权重完成后运行

终端 A 使用 MuJoCo 环境，先启动评测服务器：

```bash
conda activate dynamic
cd /path/to/panda_cable_grasp
panda-cable-dynamicvla \
  --scenario id_static --trials 3 --seed 20260804 \
  --instruction "Pick up the blue cable."
```

终端 B 使用 DynamicVLA 环境，再启动官方推理客户端：

```bash
conda activate dynamicvla
cd /path/to/DynamicVLA
python scripts/inference.py \
  -p ckt/dynamic-vla-DOM -r euler -d \
  -a dynamic-vla-DOM-zero-shot \
  -o ../dynamicvla_model_outputs
```

默认先使用非流式推理。只有当设备能在 25 Hz 控制流中及时生成动作块时，才增加 `-s` 启用流式模式；否则过期动作块会被官方流式队列丢弃。RTX 3060 6 GB 的本机实测应使用非流式模式。

检查点配置仍引用 `HuggingFaceTB/SmolLM2-360M` 的 tokenizer/config；第一次加载需要能够访问
Hugging Face，或事先把相应文件放入服务器的 `HF_HOME` 缓存。权重主体仍使用本地
`ckt/dynamic-vla-DOM/model.safetensors`。

两进程在同一台机器时使用默认 `127.0.0.1:3186/3188`。跨机器运行时，仿真端用
`--host 0.0.0.0`，推理端增加 `--host <仿真机IP>`，同时只开放这两个实验端口。

## 解释结果时的边界

这是真正的零样本迁移，但不是同分布评测。DOM 训练对象是可由单点刚体位姿描述的日常刚体，
本项目目标是橙色 DLO；桌面纹理、物体尺度、相机成像和成功条件也不同。因此 0% 成功率不能直接说明
DynamicVLA 的动态预测能力无效。首先应看 actual-input 视频、原始动作是否有限、是否发生 workspace
裁剪以及是否至少产生接近/闭爪行为，再决定是否进行提示词、多视角或少量适配消融。
