import numpy as np
import runner
from runner import ROOT, parsed, write, utc

if __name__=='__main__':
    train,args=parsed(ROOT/'throughput_smoke',workers=40,steps=5120,nsteps=128,epochs=1,batch=512)
    train.main()
    health=runner.read(ROOT/'throughput_smoke/training_health.json')
    assert health['normal_training'] and health['parameters_finite'] and health['timesteps']==5120
    write(ROOT/'throughput.json',dict(passed=True,workers=40,eval_workers=0,steps=5120,
        rollout_fps=health['rollout_fps'],rollout_sampling_seconds=health['rollout_sampling_seconds'],
        normal_training=True,completed_at=utc(),scientific_result=False))
