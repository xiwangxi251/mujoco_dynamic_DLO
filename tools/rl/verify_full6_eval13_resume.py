"""Verify checkpoint and optimizer loading before scheduling the handoff."""
import json
from pathlib import Path
import train_targetnode_full6_workspace_eval13 as trainer

checkpoint = sorted((trainer.RUN/'checkpoints').glob('step_*.zip'))[-1]
model = trainer.PPO.load(checkpoint, device='cpu')
trainer.verify_restored_checkpoint(model, checkpoint)
callback = trainer.Curriculum(20000000, eval_workers=13, resume=True)
assert callback.eval_workers == 13
jobs = [list(range(i, callback.eval_episodes, 13)) for i in range(13)]
assert len(jobs) == 13 and all(len(job) == 2 for job in jobs)
assert sorted(e for job in jobs for e in job) == list(range(26))
assert trainer.CFG == trainer.RLConfig(**trainer.NEW_CFG)
assert trainer.configuration(20000000, 24, 13)['promotion']['episodes'] == 26
result = dict(passed=True, checkpoint=str(checkpoint), steps=model.num_timesteps,
    updates=model._n_updates, policy_and_optimizer_exactly_restored=True,
    eval_workers=13, eval_episodes=26, curriculum_stage_restored=callback.stage,
    reward_source_sha256=trainer.configuration(20000000, 24, 13)['reward_source_sha256'])
trainer.save('eval13_preflight_verification.json', result)
print(json.dumps(result, indent=2))
