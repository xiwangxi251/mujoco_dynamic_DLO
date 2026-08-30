# Diffusion Policy

本项目新增的 `diffusion_policy` 方法是一个面向动态线缆抓取的视觉动作扩散策略，
实现参考 DynaMimicGen 对 Diffusion Policy 的使用方式：用少量/合成成功演示训练
视觉条件策略，使用动作块预测和 receding-horizon 控制应对动态任务。

## 输入和动作

每个观测与 DynamicVLA 保持相同的字段：

- `observation.images.opst_cam`：固定对侧相机 RGB；
- `observation.images.wrist_cam`：腕部相机 RGB；
- `observation.state.end_effector.pos`：世界坐标末端位置；
- `observation.state.end_effector.quat`：世界坐标 `wxyz` 四元数。

训练时 RGB 被缩放为 84×84，状态使用固定工作空间范围归一化到约 `[-1, 1]`。
策略输出 8 维绝对任务空间命令 `[x, y, z, qw, qx, qy, qz, gripper]`。
夹爪命令为 `+1=open`、`-1=close`，再由已有
`DynamicVLATaskSpaceAdapter` 转换为 MuJoCo 的 7 关节目标加夹爪控制量。

## 训练

先采集成功的专家双相机数据：

```bash
panda-cable-expert-collect --help
```

然后安装可选依赖并训练：

```bash
python -m pip install -e ".[diffusion]"
panda-cable-diffusion-train \
  outputs/datasets/privileged_expert/<run> \
  --output outputs/diffusion_policy/train/cable_dp \
  --action-source applied \
  --epochs 2000 --batch-size 16 --device cuda
```

`--action-source applied` 是默认值，表示用环境实际执行、经过共享安全限幅后的
动作训练；如需复现 DynamicVLA 转换器的教师命令，可选 `requested`。数据集在
episode 级别划分训练/验证集，不会把同一 episode 的帧同时放入两边。

默认超参数是 2 帧观测历史、16 步预测块、执行 8 步后重规划、100 个 DDPM 训练
步和 20 步确定性 DDIM 推理步。模型使用双相机 CNN、末端状态 MLP 和带 FiLM
条件的 1-D 时序卷积去噪器。

## 评估

```bash
panda-cable-diffusion-eval \
  --model outputs/diffusion_policy/train/cable_dp/checkpoint_best.pt \
  --scenario id_static --trials 20 --headless --device cuda
```

也可以纳入统一配对 benchmark：

```bash
panda-cable-benchmark --methods expert diffusion_policy \
  --diffusion-policy-model outputs/diffusion_policy/train/cable_dp/checkpoint_best.pt \
  --suite core --episodes 20
```

注意：Diffusion Policy 是离线模仿学习方法，训练数据必须包含专家保存的
`episode_*.npz`、`*_opst.mp4`、`*_wrist.mp4` 和对应 `.mjb` 文件。仅有 PPO
checkpoint 或没有双相机视频的轨迹不能直接用于本方法。
