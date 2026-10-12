"""A compact rollout-level startup record; no external monitoring job."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class TrainingHealthCallback(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.starting_updates = 0
        self.completed_episodes = 0

    def _on_training_start(self):
        self.starting_updates = int(self.model._n_updates)

    def _on_step(self):
        for key in ('rewards', 'new_obs'):
            value = self.locals.get(key)
            if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
                raise FloatingPointError(f'nonfinite training {key}')
        self.completed_episodes += int(np.count_nonzero(self.locals.get('dones', [])))
        return True

    def _on_rollout_start(self):
        now = time.monotonic()
        if hasattr(self, '_sampling_ended'):
            self.optimizer_and_bookkeeping_seconds = now - self._sampling_ended
        self._write_record()
        self._sampling_started = time.monotonic()
        self._sampling_start_steps = self.num_timesteps

    def _on_rollout_end(self):
        self._sampling_ended = time.monotonic()
        self.rollout_sampling_seconds = self._sampling_ended - self._sampling_started
        self.rollout_fps = (self.num_timesteps - self._sampling_start_steps) / self.rollout_sampling_seconds

    def _on_training_end(self):
        self._write_record()

    def _write_record(self):
        metrics = {key: float(value) for key, value in self.logger.name_to_value.items()
                   if key.startswith('train/') and isinstance(value, (int, float, np.number))}
        tensors = list(self.model.policy.parameters())
        for state in self.model.policy.optimizer.state.values():
            tensors.extend(v for v in state.values() if torch.is_tensor(v))
        finite = all(bool(torch.isfinite(p).all()) for p in tensors)
        if not finite or not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError('nonfinite PPO parameters or optimization metrics')
        record = dict(updated_at=time.time(), timesteps=self.num_timesteps,
                      starting_updates=self.starting_updates, updates=int(self.model._n_updates),
                      completed_episodes=self.completed_episodes, parameters_finite=finite,
                      optimization_metrics=metrics,
                      normal_training=int(self.model._n_updates) > self.starting_updates and bool(metrics))
        record.update(worker_ready=True, ppo_rounds=int(self.model._n_updates) // self.model.n_epochs,
                      optimizer_epochs=int(self.model._n_updates),
                      phase=getattr(self.model, 'ordinary_evaluation_phase', None),
                      difficulty=getattr(self.model, 'ordinary_evaluation_difficulty', None),
                      rollout_sampling_seconds=getattr(self, 'rollout_sampling_seconds', None),
                      rollout_fps=getattr(self, 'rollout_fps', None),
                      optimizer_and_bookkeeping_seconds=getattr(self, 'optimizer_and_bookkeeping_seconds', None))
        record['penalty_ramp_schedule'] = getattr(self.model, 'penalty_ramp_schedule', None)
        record['target_evaluation_stage'] = getattr(self.model, 'target_evaluation_stage', None)
        record['async_sampling'] = getattr(self.model, 'async_sampling_record', None)
        if getattr(self.model, 'future_reset_prefetch_enabled', False):
            states = self.training_env.env_method('future_prefetch_status')
            record['future_reset_prefetch'] = dict(
                workers=len(states), hits=sum(s['hits'] for s in states),
                misses=sum(s['misses'] for s in states), errors=sum(s['errors'] for s in states),
                wait_seconds=sum(s['wait_seconds'] for s in states),
                pending=sum(s['pending_pid'] is not None for s in states))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
        temporary.replace(self.path)
