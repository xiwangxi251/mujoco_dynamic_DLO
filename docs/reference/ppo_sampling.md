# PPO 采样后端

`panda-cable-rl-train` 默认使用 `--sampler async_fixed_quota`。多个训练
worker 根据各自完成状态继续推进，完整批次每个 worker 仍贡献 `--n-steps`
条 transition。例如48个worker、n_steps=1024，每次优化前仍收集49152条。
先完成配额的worker等待剩余worker；不会用快worker的额外样本替代慢worker。
每一步的同步屏障被去掉，完整批次末尾的同步保留。

```powershell
# 默认固定配额异步
panda-cable-rl-train --workers 48 --n-steps 1024

# 显式恢复原同步采样
panda-cable-rl-train --workers 48 --n-steps 1024 --sampler sync
```

只有一个worker时使用原DummyVecEnv同步路径，配置中记录实际选择。
采样器不修改动作接口、奖励、物理、观测、工作区、成功标准或网络结构。
随机动作生成顺序随worker就绪顺序改变，因此相同seed不保证复现同步轨迹。
是否加速取决于worker耗时差异和推理/通信开销，尚无当前dev普通PPO的加速倍数。

## 数据和更新

- 整批使用同一策略，所有派出的动作收回后才优化；各worker最多一个在途动作。
- GAE沿每个worker自己的时间链计算，终止切断，timeout使用terminal observation
  bootstrap。扁平buffer按返回顺序保存，不能把相邻返回事件当作相邻时间步。
- 尾批按剩余预算给worker分配相差最多1条的配额，允许不能整除worker数的预算。
  若会留下单样本批次，提前将上一批缩短1条以保留至少2条。整个调用至少需要2条。
- 回调实时收到相互对齐的actions/rewards/new_obs/infos/dones以及`worker_indices`。
  每次回调代表一个就绪组；回调次数不等于向量步数。保存间隔按实际返回样本计数。
- 提前停止后不再派发新动作，收回并统计已派出动作，放弃优化未完成批次。
- 标准SB3 worker仍先reset再返回终止数据。该worker的reset不会阻止其它worker推进，
  但不能完全消除最后配额或最后一次reset的等待。

## 课程与回调兼容性

训练中的`env_method('set_...', ...)`按worker排队，在其当前step/reset完成后、
下一动作前执行，覆盖普通、rollout、range、workspace课程和惩罚调度。
已启动回合保留环境记录的真实phase/difficulty。不要把当前全局课程标签覆盖到旧回合。
独立评估使用自己的环境；启动配置审计在异步收集开始前同步执行。

自定义回调不能在收集中修改策略参数。收集中对训练worker的阻塞查询（get_attr、
env_method非setter等）明确报错；应移到rollout边界，或选择`--sampler sync`。
SDE不受此后端支持；训练入口使用普通Box动作，支持Box和Dict观测。

## 记录与恢复

`training_config.json`记录requested/effective sampler和采样协议；
`training_health.json.async_rollout`记录各worker配额/实际数量、场景样本数、
最大进度差、批耗时及是否已排空。
`sampler_progress.json`每约10秒记录真实返回且尚未优化的样本进度；它不代表优化完成。

保存模型可以用普通`PPO.load`评估，也可通过训练入口从既有普通PPO checkpoint恢复。
恢复使用当前命令选择的采样器；环境、在途rollout本身仍不在checkpoint中。

实现参考了`perf/ppo-async-rollouts-20261013@9eda3ec`的固定配额方案，
并适配了实时回调、setter排队、精确尾批、停止排空、字典观测及dev指标。
没有合入其速度课程、控制层改动或未来观测。
