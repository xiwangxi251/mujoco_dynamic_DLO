# C PPO readiness-driven sampling

The sampler changes scheduling only. All 32 environments retain their own
1024-step quota and trajectory order. One policy version samples the entire
rollout; PPO updates only after all environments finish. Fast environments do
not contribute additional samples. Timeout value bootstrapping and GAE keep the
same time-by-environment layout as synchronous SB3 collection.

The C callbacks consume records in time-major/environment-major order once the
rollout has completed. They only record statistics or queue a curriculum change;
environment resets and curriculum transitions remain after optimization. This
sampler is not compatible with callbacks that modify an environment or policy
during collection. SDE and nondivisible final worker budgets are rejected.

Stochastic torch draws occur in readiness order, so a seed does not imply the
same trajectory as synchronous sampling. Policy, optimizer, scene state,
reward, physics, success criteria, 32 workers, 1024 steps, batch size 512, five
epochs and the original cumulative C schedule otherwise remain unchanged.

Unit verification compares deterministic synchronous and asynchronous buffers,
episode boundaries, timeout bootstrapping, returns/advantages, exact tail budget,
finite optimization and checkpoint reload. A deliberately slow worker verifies
that faster workers advance independently without exceeding their quota.

Deployment uses a separate frozen source and checkpoint. A real two-worker smoke
precedes a controlled sequential 32-worker throughput comparison; performance
samples and smoke updates are excluded from formal progress. Original outputs
are retained. The formal run uses async collection only if the measured gain is
at least 5%; otherwise it continues synchronously from the same saved model.
