"""Ordered ability groups with independent scene gates and retention checks."""
import csv
import json
from .train import MotionCurriculumCallback, StrictSuccessEvalCallback

GROUPS = (
    ("dev_static_x075_v1",),
    ("dev_static_midshape_v1_x075_v1",),
    ("dev_rigid_l1_nominal_x075_v1",),
    ("dev_rigid_l1_midshape_v1_x075_v1", "dev_rigid_replay_shape_nominal_x075_v1"),
    ("dev_shape_nominal_current_x075_v1",),
    ("dev_combined_l1_nominal_x075_v1",),
)
PHASES = ((0, 0.), (1, 0.), *((g, d) for g in range(2, 6) for d in (1/3, 2/3, 1.)))
MIXES = tuple(
    tuple((n, (1. if g == 0 else .8)/len(targets)) for n in targets)
    + tuple((n, .2/sum(len(x) for x in GROUPS[:g])) for previous in GROUPS[:g] for n in previous)
    for g, targets in enumerate(GROUPS)
)
PROTOCOL = dict(version="ordered_ability_v1", phases=PHASES, groups=GROUPS, training_mixes=MIXES,
    required_consecutive_evaluations=2, per_target_success_rate=.6,
    per_target_loaded_lift_rate=.8, per_target_secured_rate=.6,
    routine_episodes_per_target=20, retention_episodes_per_scene=10,
    retention_group="previous completed ability group", retention_min_success_rate=.4,
    confirmation_episodes_per_target=20, confirmation_seed_offset=30,
    confirmation_counts_for_promotion=False)


def structure_for_family(family='cable_x075_v1'):
    """Use the same ability groups and probabilities in either explicit family."""
    from ..scenarios.registry import SCENARIO_FAMILIES
    mapping = dict(zip(SCENARIO_FAMILIES['cable_x075_v1'], SCENARIO_FAMILIES[family]))
    groups = tuple(tuple(mapping[n] for n in group) for group in GROUPS)
    mixes = tuple(tuple((mapping[n], p) for n, p in mix) for mix in MIXES)
    return groups, mixes


class OrderedMotionCurriculum(MotionCurriculumCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phases = PHASES

    def evaluation_qualifies(self, metrics):
        targets = metrics.get("targets", [])
        return (len(targets) == len(self.current_scenarios)
            and {r["scenario"] for r in targets} == set(self.current_scenarios)
            and all(r["episodes"] == 20 and r["success_rate"] >= .6
                    and r["loaded_lift_rate"] >= .8 and r["grasp_rate"] >= .6 for r in targets)
            and all(r["episodes"] == 10 and r["success_rate"] >= .4 for r in metrics.get("retention", [])))


class OrderedSuccessEval(StrictSuccessEvalCallback):
    def _scene(self, name, difficulty, episodes, offset, role):
        self.env.env_method("set_training_scenarios", (name,))
        self.env.env_method("set_motion_difficulty", difficulty)
        metrics = self._evaluate(episodes, seed_offset=offset)
        return dict(scenario=name, difficulty=difficulty, episodes=episodes, role=role,
                    counts={k.removesuffix("_rate"): round(v*episodes) for k, v in metrics.items() if k.endswith("_rate")},
                    **metrics)

    @staticmethod
    def _aggregate(rows):
        count = sum(r["episodes"] for r in rows)
        return {k: sum(r[k]*r["episodes"] for r in rows)/count
                for k in rows[0] if k.endswith("_rate") or k in ("mean_return", "mean_length")}

    def _on_step(self):
        if self.num_timesteps-self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        c = self.curriculum
        phase, group, difficulty = c.phase_index, c.stage_index, c.difficulty
        targets = [self._scene(n, difficulty, self.episodes, 0, "target") for n in c.current_scenarios]
        retention = ([] if group == 0 else [self._scene(n, 0. if group-1 < 2 else 1., 10, 20, "retention")
                                             for n in c.stages[group-1]])
        metrics = self._aggregate(targets)
        score = (phase, metrics["success_rate"], metrics["grasp_rate"], metrics["aligned_pinch_rate"], metrics["mean_return"])
        best = (self.best_phase, self.best_success_rate, self.best_grasp_rate,
                self.best_aligned_pinch_rate, self.best_mean_return)
        confirmed = []
        if (metrics["success_rate"] > 0 or metrics["grasp_rate"] > 0) and score > best:
            # Confirmation uses this evaluated phase, before any promotion.
            confirmed = [self._scene(n, difficulty, 20, 30, "confirmation") for n in c.current_scenarios]
            cm = self._aggregate(confirmed)
            cs = (phase, cm["success_rate"], cm["grasp_rate"], cm["aligned_pinch_rate"], cm["mean_return"])
            if (cm["success_rate"] > 0 or cm["grasp_rate"] > 0) and cs > best:
                (self.best_phase, self.best_success_rate, self.best_grasp_rate,
                 self.best_aligned_pinch_rate, self.best_mean_return) = cs
                directory = self.output_dir/"best"
                directory.mkdir(parents=True, exist_ok=True)
                self.model.save(directory/"best_model")
                (directory/"confirmation.json").write_text(json.dumps(dict(timesteps=self.num_timesteps,
                    curriculum_phase=phase, curriculum_stage=group, motion_difficulty=difficulty,
                    scenes=confirmed, **cm), indent=2))
        qualifies = c.evaluation_qualifies(dict(targets=targets, retention=retention))
        c.observe_evaluation(dict(targets=targets, retention=retention))
        record = dict(timesteps=self.num_timesteps, phase=phase, group=group, difficulty=difficulty,
                      targets=targets, retention=retention, confirmation=confirmed,
                      qualifies=qualifies, consecutive=c.qualifying_evaluations, next_phase=c.phase_index)
        with (self.output_dir/"ordered_rounds.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([self.num_timesteps, phase, c.phase_index, group, difficulty,
                metrics["success_rate"], metrics["strict_success_rate"], metrics["pinch_rate"],
                metrics["aligned_pinch_rate"], metrics["lift_attempt_rate"], metrics["loaded_lift_rate"],
                metrics["grasp_rate"], metrics["slip_rate"], metrics["mean_return"], metrics["mean_length"],
                sum(r["episodes"] for r in targets)])
        for k, v in metrics.items():
            self.logger.record("eval/"+k, v)
        c.configure_evaluation_env(self.env)
        print(f"ordered_eval timesteps={self.num_timesteps} phase={phase} next_phase={c.phase_index} "
              f"qualifies={qualifies} targets={json.dumps(targets)}", flush=True)
        return True
