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
        metrics = {key: float(value) for key, value in self.logger.name_to_value.items()
                   if key.startswith('train/') and isinstance(value, (int, float, np.number))}
        finite = all(bool(torch.isfinite(p).all()) for p in self.model.policy.parameters())
        if not finite or not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError('nonfinite PPO parameters or optimization metrics')
        record = dict(updated_at=time.time(), timesteps=self.num_timesteps,
                      starting_updates=self.starting_updates, updates=int(self.model._n_updates),
                      completed_episodes=self.completed_episodes, parameters_finite=finite,
                      optimization_metrics=metrics,
                      normal_training=int(self.model._n_updates) > self.starting_updates and bool(metrics))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
        temporary.replace(self.path)
