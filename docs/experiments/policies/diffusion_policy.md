# Diffusion Policy

本项目新增的 `diffusion_policy` 方法是一个面向动态线缆抓取的视觉动作扩散策略，
实现参考 DynaMimicGen 对 Diffusion Policy 的使用方式：用少量/合成成功演示训练
视觉条件策略，使用动作块预测和 receding-horizon 控制应对动态任务。

## 输入和动作

每个观测与 DynamicVLA 保持相同的字段：

- `observation.images.opst_cam`：固定对侧相机 RGB；
- `observation.images.wrist_cam`：腕部相机 RGB；
- `observation.state.end_effector.pos`：世界坐标末端位置；
- `observation.state.end_effector.quat`：世界坐标 `wxyz` 四元数（数据集内部
  转换为 `euler_xyz`）。

训练时 RGB 被缩放为 84×84。状态为 6 维 `[x, y, z, roll, pitch, yaw]`，动作是
相对当前状态的 7 维 chunk-delta `[dx, dy, dz, droll, dpitch, dyaw, gripper]`，
与 DynamicVLA 的任务空间契约一致。归一化边界从数据按 q01/q99 拟合（欧拉角
±π、欧拉 delta ±2π、夹爪 [-1, 1] 的固定边界仅作旧 checkpoint 的回退）。
夹爪命令为 `+1=open`、`-1=close`，策略输出经
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

中断后可用 `--resume <checkpoint.pt>` 从已有 checkpoint 恢复训练（恢复模型
权重、EMA、优化器状态和训练进度）。

默认超参数是 2 帧观测历史、16 步预测块、执行 8 步后重规划、100 个 DDPM 训练
步和 100 步 DDPM 推理步，与 DynaMimicGen 的 image-DP 配置一致。每个相机使用
独立的、未共享权重的 ResNet-18（BatchNorm 替换为 GroupNorm），接 32 个关键点的
SpatialSoftmax 和 64 维特征投影；训练时使用 76×76 随机裁剪，评估时使用中心裁剪。
动作去噪器是三层 1-D Conditional U-Net，通道为 `[512, 1024, 2048]`，卷积核为 5，
GroupNorm 分组数为 8，并在残差块中使用 FiLM 条件调制。扩散时间步嵌入维度为 256，
噪声调度为 squared-cosine DDPM，同时使用 power=0.75 的 EMA 权重进行推理。

该配置约有 2.82 亿个可训练参数，因此训练显存和速度要求明显高于普通行为克隆模型。

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
