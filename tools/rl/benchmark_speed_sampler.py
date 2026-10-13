"""One frozen-policy sampling window; no optimization or formal samples."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from panda_cable_grasp.rl.async_rollouts import policy_digest
from panda_cable_grasp.rl.speed_curriculum import SpeedPPO, atomic, verify_workers
from panda_cable_grasp.rl.train_speed_curriculum import worker


class Finite(BaseCallback):
    def _on_step(self):
        assert np.isfinite(self.locals['rewards']).all() and np.isfinite(self.locals['new_obs']).all()
        return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sampler',choices=['sync','async'],required=True)
    p.add_argument('--steps',type=int,default=1024)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());cfg=plan['configuration']
    torch.set_num_threads(1);set_random_seed(cfg['seed'])
    initial=SpeedPPO.load(a.checkpoint,device='cpu')
    state=initial.c_speed_state;start=initial.num_timesteps;epochs=initial._n_updates
    digest=policy_digest(initial.policy)
    env=VecMonitor(SubprocVecEnv([worker(i,cfg,state) for i in range(32)],start_method='spawn'),
                   info_keywords=('success',))
    try:
        model=SpeedPPO.load(a.checkpoint,env=env,device='cpu',seed=cfg['seed'],n_steps=1024,batch_size=512,n_epochs=5)
        model.async_rollouts_enabled=a.sampler=='async'
        model._setup_learn(32*a.steps,reset_num_timesteps=False)
        verify_workers(env.env_method('execution_configuration'),state)
        atomic(a.output/'worker_ready.json',dict(worker_ready=True,workers=32,policy_sha256=digest,
               input_steps=start,input_epochs=epochs,updated_at=time.time()))
        cb=Finite();cb.init_callback(model);cb.on_training_start({}, {})
        before=time.monotonic()
        assert model.collect_rollouts(env,cb,model.rollout_buffer,a.steps)
        seconds=time.monotonic()-before
        assert model._n_updates==epochs and policy_digest(model.policy)==digest
        assert model.num_timesteps==start+32*a.steps
        assert np.isfinite(model.rollout_buffer.advantages).all() and np.isfinite(model.rollout_buffer.returns).all()
        report=dict(status='complete',sampler=a.sampler,workers=32,steps_per_worker=a.steps,
                    sampled_transitions=32*a.steps,elapsed_seconds=seconds,fps=32*a.steps/seconds,
                    input_steps=start,input_optimizer_epochs=epochs,policy_sha256=digest,policy_unchanged=True,
                    optimized=False,formal_samples=0,finite=True,updated_at=time.time(),
                    async_receipt=getattr(model,'async_rollout_receipt',None))
        atomic(a.output/'benchmark.json',report)
        print(json.dumps(report),flush=True)
    finally:env.close()


if __name__=='__main__':main()
