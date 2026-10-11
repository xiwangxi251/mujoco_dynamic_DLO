"""Inspect a live trainer, finite updates and a reloaded trained checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from train_targetnode_full6_workspace import RUN, PROFILE, save
import numpy as np
import torch
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-updates",type=int,default=10)
    args = parser.parse_args()
    state = json.loads((RUN/"training_status.json").read_text())
    pid = int((RUN/"train.pid").read_text())
    os.kill(pid,0)
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode()
    assert "train_targetnode_full6_workspace.py" in command
    assert not (RUN/"FAILED.json").exists()
    assert state["updates"] >= args.minimum_updates
    assert state["completed_episodes"] >= 24
    assert state["policy_parameters_finite"]
    assert time.time()-state["updated_at"] < 600
    assert all(np.isfinite(value) for value in state["training_metrics"].values())
    trained = PPO.load(state["first_trained_checkpoint"],device="cpu")
    initial = PPO.load(RUN/"random_initial_model.zip",device="cpu")
    assert trained.num_timesteps > 0 and trained._n_updates > 0
    assert trained.policy.optimizer.state
    changes = []
    for (name,a),(other,b) in zip(initial.policy.named_parameters(),trained.policy.named_parameters()):
        assert name == other and torch.isfinite(b).all()
        changes.append(float(torch.linalg.vector_norm(a-b)))
    assert max(changes)>0
    cfg = json.loads((RUN/"training_config.json").read_text())
    assert cfg["controller_profile"] == PROFILE
    assert not cfg["promotion"]["node_or_bin_success_gate"]
    assert cfg["rl_config"]["target_u_min"] == 0 and cfg["rl_config"]["target_u_max"] == 1
    identity = json.loads((RUN/"launch_identity.json").read_text())
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["execution_module_sha256"] == sha(RUN/"full6_workspace_env.py")
    assert identity["runner_sha256"] == sha(RUN/"train_targetnode_full6_workspace.py")
    assert all(sha(RUN/"frozen_code"/path)==value for path,value in identity["source_hashes"].items())
    result = dict(passed=True,checked_at=time.time(),pid=pid,command=command,
        current_steps=state["steps"],current_updates=state["updates"],
        completed_episodes=state["completed_episodes"],
        workspace_inside_fraction=state["mean_episode_workspace_inside_fraction"],
        trained_checkpoint_steps=trained.num_timesteps,
        trained_checkpoint_updates=trained._n_updates,
        optimizer_state_present=True,finite_model=True,finite_optimization_metrics=True,
        max_parameter_change_l2=max(changes),runtime_sources_unchanged=True,
        status_snapshot=state)
    save("startup_health_verification.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="status_snapshot"},indent=2),flush=True)


if __name__ == "__main__":
    main()
