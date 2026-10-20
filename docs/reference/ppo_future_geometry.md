# 可选的开场未来轨迹观测

这是使用仿真特权信息的研究模式，默认关闭。普通PPO保持99维当前观测；
显式开启`--future-geometry`后，actor/critic都接收1419维观测。动作、奖励、
物理、控制、工作区、成功标准和回合时限沿用该次运行的配置。

```powershell
# 从零训练，显式开启未来信息
panda-cable-rl-train --future-geometry --workers 48 --n-steps 1024

# 默认仍是普通PPO
panda-cable-rl-train --workers 48 --n-steps 1024

# 从未来checkpoint恢复，继承其未来协议
panda-cable-rl-train --resume <future-run>/final_model.zip

# 评估入口自动读取未来协议，普通checkpoint仍使用当前观测
panda-cable-rl-eval --model <future-run>/final_model.zip --headless --no-video
```

## 协议 initial_future3s_geometry_v1

- reset时从物理时间0复制完整model、integration state、驱动/回放时钟与RNG，
  在独立分支中一次计算绝对0～3秒的无机器人接触参考。
- 分支冻结机器人，禁用机器人接触，保留线缆、桌面、回放约束和当前场景的驱动。
  它包含隐藏驱动信息，是无机器人接触条件下的参考，不是策略接触后的真实未来。
- 固定30帧，时间为0.1、0.2、……、3.0秒。每帧14个固定材料对应点xyz、
  相对时间除以3、有效mask：`99 + 30 × (14×3 + 2) = 1419`。
- 世界轨迹在回合内保持不变，每步只转换到当前TCP坐标、更新相对时间和过期mask。
  不滚动延伸到t+3，不每步重新推演，也不重新选取材料点。
- 任何物理子步首次出现机器人与线缆实际接触，整段future立即锁定关闭，
  直到下一reset。松脱重抓不重新开启；3秒耗尽后原任务继续。
- mask不使用未来实际接触时间。14个材料点不等于保存全部仿真节点的无损形状。

当前模式仅支持普通5维自由动作、state99、14个线缆采样点，不支持目标节点任务、
joint_target或点云观测。网络沿用训练入口的256/256/128 ReLU，没有额外编码器，
输入变宽会增加第一层参数量。代码合入不代表已证明未来信息的因果收益。

## dev适配和隔离

复制分支保留当前dev的场景/表面驱动；对延迟力回放编译器独立复制其model/data、
gravity data和参考对象，避免预演提前消耗现场编译状态。现场model、状态和RNG
不能因预计算改变。接触检测hook只影响future有效性，不改变实际动作或抓取判定。

本次仅引入观测功能。下一reset预取helper和固定总批量采样器没有随之引入。
dev默认的固定配额异步采样保持，oracle自身的额外计算仍有成本，3秒仿真时长
不表示只需3秒墙钟时间。普通模式不执行oracle。

## checkpoint与评估

训练配置和模型同时保存未来协议；配置记录观测版本、名称和实际维数。恢复时
继承原协议，不能把99维checkpoint当作1419维模型继续训练，或反向去掉输入。
评估headless与viewer都使用同一wrapper，默认从配置/模型读取协议；
`--no-future-geometry`不能用来静默修改未来模型的接口。

旧的1419维模型若没有任何协议元数据，需要显式指定`--future-geometry`，
并自行提供其原场景/执行配置；维数、已声明的协议版本/帧时间不匹配会报错。
普通PPO的已保存模型和默认99维入口保持兼容。

本次接入训练器及其原生headless/viewer诊断评估入口；跨方法benchmark已有独立
冻结协议，不会自动改写成未来观测。运动库隔离测试通过
`PANDA_CABLE_TEST_REPLAY_BANK_DIR`指定实际已有运动库，无数据时明确跳过该组测试。
