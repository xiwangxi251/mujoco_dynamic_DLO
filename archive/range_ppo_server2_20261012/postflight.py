"""Verify final source's resume/ramp path without repeating motion calibration."""
import json
import sys
import torch
import runner
from runner import ROOT, parsed, arguments, setup, write, utc

def main():
    setup()
    from stable_baselines3 import PPO
    from panda_cable_grasp.rl.range_curriculum import Evidence, ALL_SCENES, VERSION
    from panda_cable_grasp.rl import train
    model=PPO.load(ROOT/'smoke_train/final_model.zip',device='cpu')
    e=Evidence()
    for _ in range(50):e.add(ALL_SCENES[2],'0',dict(success=1,loaded=1,secured=1,attempt=1))
    model.range_curriculum_state=dict(version=VERSION,phase=2,evidence=e.state(),episode_counter=50)
    model.save(ROOT/'smoke_train/resume_input.zip')
    sys.argv=['train',*arguments(ROOT/'smoke_resume',workers=3,steps=12,nsteps=2,epochs=1,batch=2),
        '--resume',str(ROOT/'smoke_train/resume_input.zip')]
    args=train.parse_args()
    assert args.rollout_initial_phase==2
    probe=train.make_worker(0,args)()
    try:
        _,info=probe.reset(seed=20261010)
        assert info['range_episode']['phase']==2 and info['scenario_name']==ALL_SCENES[2]
    finally:probe.close()
    train.main()
    resumed=PPO.load(ROOT/'smoke_resume/final_model.zip',device='cpu')
    assert resumed.num_timesteps==26 and resumed.range_curriculum_state['phase']==2
    assert len(resumed.range_curriculum_state['evidence']['queues'][ALL_SCENES[2]]['0'])==50
    # The earlier smoke was before adding the explicit scratch ramp reference.
    # A separate six-sample scratch verifies the final 48-worker reference.
    train,args=parsed(ROOT/'ramp_smoke',workers=2,steps=6,nsteps=3,epochs=1,batch=2)
    train.main()
    ramp=PPO.load(ROOT/'ramp_smoke/final_model.zip',device='cpu').penalty_ramp_schedule
    assert ramp['height_full_samples']==480000 and ramp['safety_full_samples']==240000
    assert all(torch.isfinite(p).all() for p in resumed.policy.parameters())
    write(ROOT/'postflight.json',dict(passed=True,resume_steps=26,resume_phase=2,
        sampler_evidence_restored=True,penalty_ramp_samples=ramp,completed_at=utc()))
    print('postflight_passed',flush=True)

if __name__=='__main__':main()
