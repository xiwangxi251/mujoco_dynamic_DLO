"""Fresh PPO training with frozen rewards and workspace-conditioned goals."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

RUN = Path(os.environ.get("FULL6_PPO_RUN", Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(RUN / "frozen_code/src"))
import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from panda_cable_grasp.rl.environment import RLConfig
from panda_cable_grasp.rl.metrics import TrainingMetricsCallback
from full6_workspace_env import Full6TargetEnv, NoWorkspaceTarget, PROFILE, WorkspaceMask, WorkspaceTargetNode

ROOT = RUN.parents[3]
BANK = ROOT / "outputs/rl/train/ppo_nero_targetnode_state6_warmstart3m_20260928/eval_banks"
SCENES = ("id_static", "id_rigid_l1_nominal", "id_shape_nominal_current", "id_combined_l1_nominal", "id_static_midshape_v1", "id_rigid_replay_shape_nominal")
STAGES = [
    dict(name="static_workspace_all_nodes", mix=[1,0,0,0,0,0], motion=0),
    dict(name="complex_static_workspace", mix=[.3,0,0,0,.7,0], motion=0),
    dict(name="component_motion_low", mix=[.1,.4,.4,0,.1,0], motion=1/3),
    dict(name="component_motion_full", mix=[.1,.4,.4,0,.1,0], motion=1),
    dict(name="combined_motion", mix=[.1,.1,.1,.6,.1,0], motion=1),
    dict(name="six_scenes_replay_low", mix=[.1,.1,.1,.1,.3,.3], motion=1/3),
    dict(name="six_scenes_full", mix=[.1,.1,.1,.1,.3,.3], motion=1),
]
SEED = 20261004
REFERENCE = json.loads((RUN / "reference_training_config.json").read_text())
OLD_CFG = REFERENCE["rl_config"]
NEW_CFG = dict(OLD_CFG, target_u_min=0., target_u_max=1., target_balanced_sampling=False, graspable_end_fraction=0.)
CFG = RLConfig(**NEW_CFG)


def save(name, value):
    path = RUN / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)), encoding="utf-8")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_restored_checkpoint(model, path):
    data, params, _ = load_from_zip_file(path, device="cpu")
    def equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal(a, b)
        else:
            assert left == right
    equal(params["policy"], model.policy.state_dict())
    equal(params["policy.optimizer"], model.policy.optimizer.state_dict())
    assert model.num_timesteps == data["num_timesteps"]
    assert model._n_updates == data["_n_updates"]


def promotion_update(history, first, success):
    passed = first >= .60 and success >= .40
    history = (list(history) + [bool(passed)])[-3:]
    return history, len(history) >= 2 and sum(history) >= 2


class CourseEnv(gym.Wrapper):
    def __init__(self, rank=0, evaluation=False):
        env = Full6TargetEnv(robot="nero", seed=SEED+rank, frame_skip=10, episode_seconds=15.,
            scenario_names=SCENES, rl_config=RLConfig(**asdict(CFG)),
            geometric_safety_enabled=False, table_finger_collision_filter_enabled=True)
        super().__init__(env)
        self.workspace = WorkspaceMask(RUN / "workspace/workspace.npz")
        env.base_env.workspace_sampling_mask = self.workspace
        env.goal = WorkspaceTargetNode(env, SEED+rank, self.workspace)
        self.stage = 0
        self.evaluation = evaluation
        self.reset_rejections = 0
        self.inside_steps = self.sample_steps = 0
        self.episode_goal = None
        self.set_stage(0)
        if evaluation:
            env.base_env.config.replay_bank_dir = str(BANK)
            env.base_env.config.replay_seed_fallback = "error"
            env.base_env._replay_banks.clear()
            env.base_env._shape_banks.clear()

    def set_stage(self, stage):
        self.stage = int(stage)
        self.env.set_training_scenario_distribution(SCENES, tuple(STAGES[self.stage]["mix"]))
        self.env.set_motion_difficulty(STAGES[self.stage]["motion"])

    def reset(self, **kwargs):
        requested_seed = kwargs.get("seed")
        for attempt in range(50):
            actual = requested_seed
            if attempt and requested_seed is not None:
                actual = 20260804 + (requested_seed - 20260804 + attempt) % 50 if self.evaluation else requested_seed + attempt * 100003
            try:
                obs, info = self.env.reset(**dict(kwargs, seed=actual))
                break
            except NoWorkspaceTarget as exc:
                self.reset_rejections += 1
                rejection = str(exc)
        else:
            raise NoWorkspaceTarget(f"50 scene resets produced no eligible goal; last rejection: {rejection}")
        self.episode_goal = self.env.goal.index
        self.inside_steps = self.sample_steps = 1
        info.update(curriculum_stage=self.stage, workspace_reset_attempts=attempt+1,
                    workspace_reset_rejections=self.reset_rejections,
                    requested_seed=requested_seed, actual_seed=actual,
                    workspace_prefix_seconds=self.env.base_env.workspace_prefix_seconds,
                    workspace_initial_goal_xyz=self.env.base_env.data.xpos[self.env.base_env.cable_ids[self.episode_goal]].tolist())
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        assert self.env.goal.index == self.episode_goal, "Material identity changed within an episode"
        if self.env.goal.first_grasp_correct is None:
            self.sample_steps += 1
            self.inside_steps += int(info["workspace_goal_current_inside"])
        info.update(curriculum_stage=self.stage,
                    workspace_inside_fraction=self.inside_steps/self.sample_steps,
                    workspace_reset_rejections=self.reset_rejections,
                    workspace_prefix_seconds=self.env.base_env.workspace_prefix_seconds,
                    execution_profile=PROFILE["name"])
        return obs, reward, done, truncated, info


def worker(rank):
    return lambda: CourseEnv(rank)


def eval_shard(checkpoint, stage, episodes, limit=None):
    torch.set_num_threads(1)
    model = PPO.load(checkpoint, device="cpu")
    env = CourseEnv(1000+episodes[0], evaluation=True)
    env.set_stage(stage)
    rows = []
    try:
        for episode in episodes:
            seed = 20260804 + episode % 50
            obs, initial = env.reset(seed=seed)
            total, count = 0., 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total += reward
                count += 1
                if done or truncated or (limit is not None and count >= limit):
                    break
            rows.append(dict(episode=episode, seed=seed, actual_seed=initial["actual_seed"],
                stage=stage, scenario=info.get("scenario_name"), goal=env.episode_goal,
                first=info.get("first_grasp_correct"), success=bool(info.get("task_success", False)),
                return_=total, control_steps=count, inside_fraction=info["workspace_inside_fraction"],
                reset_attempts=initial["workspace_reset_attempts"],
                reset_prefix_seconds=initial["workspace_prefix_seconds"],
                observation=obs.tolist() if limit else None))
    finally:
        env.close()
    return rows


class Curriculum(BaseCallback):
    def __init__(self, total, interval=200000, eval_workers=13, resume=False, eval_episodes=26):
        super().__init__()
        self.total, self.interval, self.eval_workers = total, interval, eval_workers
        self.eval_episodes = eval_episodes
        assert 1 <= self.eval_workers <= self.eval_episodes <= 50
        self.stage, self.last, self.history = 0, 0, []
        self.pool = None
        self.goal_counts = Counter()
        self.completed, self.successes, self.first_correct = 0, 0, 0
        self.inside_sum = 0.
        self.latest_trained_checkpoint = None
        self.training_started = time.time()
        self.pending_eval = None
        if resume:
            state = json.loads((RUN / "curriculum_state.json").read_text())
            status = json.loads((RUN / "training_status.json").read_text())
            self.stage, self.last = state["stage"], state["steps"]
            self.history = state["promotion_history"]
            self.completed = status["completed_episodes"]
            self.successes, self.first_correct = status["successes"], status["first_correct"]
            self.goal_counts.update({int(k): v for k, v in status["goal_counts"].items()})
            self.inside_sum = status["mean_episode_workspace_inside_fraction"] * self.completed
            self.latest_trained_checkpoint = status["first_trained_checkpoint"]
            self.training_started -= status["elapsed_s"]
            pending = RUN / "evaluation_in_progress.json"
            if pending.exists():
                self.pending_eval = json.loads(pending.read_text())["steps"]

    def _on_training_start(self):
        self.pool = ProcessPoolExecutor(max_workers=self.eval_workers, mp_context=mp.get_context("spawn"))
        self.training_env.env_method("set_stage", self.stage)
        if self.pending_eval is not None:
            assert self.pending_eval == self.model.num_timesteps
            self._evaluate(self.pending_eval)
            self.pending_eval = None

    def _on_step(self):
        if not np.isfinite(self.locals["rewards"]).all() or not np.isfinite(self.locals["new_obs"]).all():
            raise FloatingPointError("Nonfinite observations/rewards during rollout")
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.completed += 1
                self.goal_counts[int(info["goal_node_index"])] += 1
                self.successes += int(info.get("task_success", False))
                self.first_correct += int(info.get("first_grasp_correct") is True)
                self.inside_sum += info["workspace_inside_fraction"]
        return True

    def _on_rollout_start(self):
        if self.model._n_updates > 0 and self.latest_trained_checkpoint is None:
            path = RUN / "checkpoints/first_trained_model"
            self.model.save(path)
            self.latest_trained_checkpoint = str(path)+".zip"
        self._status()

    def _status(self):
        finite = all(bool(torch.isfinite(p).all()) for p in self.model.policy.parameters())
        if not finite:
            raise FloatingPointError("Nonfinite model parameters")
        train_metrics = {k:float(v) for k,v in self.model.logger.name_to_value.items() if k.startswith("train/") and isinstance(v, (int,float,np.number))}
        if not all(np.isfinite(v) for v in train_metrics.values()):
            raise FloatingPointError("Nonfinite optimization metrics")
        save("training_status.json", dict(pid=os.getpid(), updated_at=time.time(),
            steps=self.num_timesteps, updates=self.model._n_updates, stage=self.stage,
            stage_name=STAGES[self.stage]["name"], completed_episodes=self.completed,
            successes=self.successes, first_correct=self.first_correct,
            mean_episode_workspace_inside_fraction=self.inside_sum/max(1,self.completed),
            goal_counts=dict(self.goal_counts), policy_parameters_finite=finite,
            training_metrics=train_metrics, first_trained_checkpoint=self.latest_trained_checkpoint,
            elapsed_s=time.time()-self.training_started, eval_workers=self.eval_workers,
            eval_episodes=self.eval_episodes))

    def _on_rollout_end(self):
        step = self.num_timesteps
        self.model.ent_coef = .01 + (.001-.01)*min(step/self.total, 1.)
        self._status()
        if step-self.last < self.interval:
            return
        self._evaluate(step)

    def _evaluate(self, step):
        self.last = step
        checkpoint = RUN / f"checkpoints/step_{step:09d}"
        self.model.save(checkpoint)
        started = time.time()
        save("evaluation_in_progress.json", dict(steps=step, stage=self.stage,
             started=started, eval_workers=self.eval_workers, eval_episodes=self.eval_episodes))
        jobs = [list(range(i, self.eval_episodes, self.eval_workers)) for i in range(self.eval_workers)]
        futures = [self.pool.submit(eval_shard, str(checkpoint)+".zip", self.stage, job) for job in jobs]
        rows = sorted([row for future in futures for row in future.result()], key=lambda r:r["episode"])
        for row in rows:
            row.pop("observation", None)
            row["step"] = step
        n = len(rows)
        first = sum(row["first"] is True for row in rows)/n
        success = sum(row["success"] for row in rows)/n
        self.history, promote = promotion_update(self.history, first, success)
        with (RUN / "curriculum_eval.csv").open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            if file.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)
        previous = self.stage
        print(f"COURSE step={step} stage={previous} first={first:.3f} success={success:.3f} history={self.history} eligible_goals_only=True", flush=True)
        if promote and self.stage < len(STAGES)-1:
            self.stage += 1
            self.history = []
            self.training_env.env_method("set_stage", self.stage)
        save("curriculum_state.json", dict(steps=step, stage=self.stage,
            stage_name=STAGES[self.stage]["name"], promotion_history=self.history,
            last_eval_stage=previous, first_correct_rate=first, task_success_rate=success,
            goal_counts=dict(Counter(row["goal"] for row in rows)),
            node_success_gate=False, eval_episodes=n, eval_workers=self.eval_workers,
            eval_duration_seconds=time.time()-started))
        (RUN / "evaluation_in_progress.json").replace(RUN / "evaluation_last_completed.json")

    def _on_training_end(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)


def configuration(total, workers, eval_workers=13, eval_episodes=26):
    changed = {k:dict(old=OLD_CFG[k], new=NEW_CFG[k]) for k in NEW_CFG if NEW_CFG[k] != OLD_CFG[k]}
    assert set(changed) == {"target_u_min", "target_u_max", "target_balanced_sampling", "graspable_end_fraction"}
    return dict(algorithm="PPO", initialization="random", warm_start=None,
        controller_profile=PROFILE, rl_interface_version="full6_workspace_target_node_v1",
        action_mode="task_space_vertical_down", observation_dimension=109, action_dimension=5,
        target_node_task=True, rl_config=asdict(CFG), rl_config_changes=changed,
        reward_methods_inherited_unchanged=True, reference_training_config_sha256=sha(RUN/"reference_training_config.json"),
        reward_source_sha256={p:sha(RUN/"frozen_code/src/panda_cable_grasp"/p) for p in ("rl/environment.py", "rl/target_node.py")},
        target_sampling="uniform among all material nodes inside measured workspace interior at reset; fixed identity thereafter",
        workspace_reset_prefix=dict(maximum_s=8., robot_action="ready pose", motion="original physical motion; no PPO reward during prefix", policy_time_budget_after_prefix_s=15.),
        workspace_mask_sha256=sha(RUN/"workspace/workspace.npz"),
        cumulative_target_timesteps=total, workers=workers, eval_workers=eval_workers, n_steps=1024, batch_size=512,
        n_epochs=5, gamma=.995, gae_lambda=.95, stages=STAGES,
        promotion=dict(eval_every=200000, episodes=eval_episodes, first_correct=.60, task_success=.40,
                       window_evaluations=3, passes_required=2, node_or_bin_success_gate=False),
        episode_seconds=15., seed=SEED, evaluation_bank=str(BANK), training_device="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timesteps", type=int, default=20000000)
    parser.add_argument("--eval-workers", type=int, default=13)
    parser.add_argument("--eval-episodes", type=int, default=26)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not 1 <= args.eval_workers <= args.eval_episodes <= 50:
        parser.error("Require 1 <= eval workers <= eval episodes <= 50 held-out seeds")
    torch.set_num_threads(1)
    if not (RUN / "prelaunch_verification.json").exists():
        raise RuntimeError("Run prelaunch verification first")
    verification = json.loads((RUN / "prelaunch_verification.json").read_text())
    assert verification["passed"]
    assert verification["execution_module_sha256"] == sha(RUN/"full6_workspace_env.py")
    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        config_name, identity_name = "training_config_eval13.json", "launch_identity_eval13.json"
    else:
        config_name, identity_name = "training_config.json", "launch_identity.json"
    save(config_name, configuration(args.timesteps, args.workers, args.eval_workers, args.eval_episodes))
    save(identity_name, dict(pid=os.getpid(), host=os.uname().nodename,
        runner_sha256=sha(Path(__file__)), execution_module_sha256=sha(RUN/"full6_workspace_env.py"),
        source_hashes={str(p.relative_to(RUN/"frozen_code")):sha(p) for p in (RUN/"frozen_code/src").rglob("*.py")}))
    env = VecMonitor(SubprocVecEnv([worker(i) for i in range(args.workers)], start_method="spawn"), filename=str(RUN/("monitor_eval13.csv" if args.resume else "monitor.csv")))
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
        assert model.n_envs == args.workers and model.policy.optimizer.state
        assert model.num_timesteps > 0 and model._n_updates > 0
        verify_restored_checkpoint(model, args.resume)
        finite = all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
        assert finite
        save("RESUMED_eval13.json", dict(pid=os.getpid(), time=time.time(),
            checkpoint=str(args.resume), checkpoint_sha256=sha(args.resume),
            starting_steps=model.num_timesteps, updates=model._n_updates,
            optimizer_state_entries=len(model.policy.optimizer.state),
            policy_and_optimizer_exactly_restored=True,
            policy_parameters_finite=finite, eval_workers=args.eval_workers,
            eval_episodes=args.eval_episodes, total_budget=args.timesteps,
            pending_rollout_discarded=True))
    else:
        model = PPO("MlpPolicy", env, learning_rate=lambda p:3e-5+(3e-4-3e-5)*p,
        n_steps=1024, batch_size=512, n_epochs=5, gamma=.995, gae_lambda=.95,
        ent_coef=.01, target_kl=.03,
        policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256,256,128], vf=[256,256,128])),
        seed=SEED, device="cpu", verbose=1)
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
        model.save(RUN / "random_initial_model")
        save("STARTED.json", dict(pid=os.getpid(), time=time.time(), fresh_optimizer=True, starting_steps=0))
    callback = Curriculum(args.timesteps, eval_workers=args.eval_workers,
        eval_episodes=args.eval_episodes, resume=bool(args.resume))
    try:
        model.learn(args.timesteps-model.num_timesteps,
            reset_num_timesteps=not bool(args.resume),
            callback=[callback, TrainingMetricsCallback(RUN/"training_metrics.csv", window=100, goal_task=True)])
        model.save(RUN / "final_model")
        save("TRAIN_COMPLETE.json", dict(steps=model.num_timesteps, stage=callback.stage, time=time.time()))
    finally:
        if callback.pool:
            callback.pool.shutdown(wait=True, cancel_futures=True)
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        save("FAILED_eval13.json" if "--resume" in sys.argv else "FAILED.json",
             dict(time=time.time(), traceback=traceback.format_exc()))
        raise
