# Target-node PPO: start from the scene initial state (2026-10-07)

## Purpose and authorization

The user identified that motion-scene resets secretly advanced physics until a cable
node entered the workspace, then selected a goal and started PPO. This prevented
learning or evaluating observation, waiting and approach before entry. The user
accepted that some assigned nodes may never enter, provided most have opportunities.

## Current default: `policy_from_initial_v3`

- Reset initializes the requested scene/seed once and immediately returns the first
  policy observation. It does not call `prepare_workspace_entry` or a hidden control
  step, and never rejects/resamples a seed because of workspace membership.
- Uniformly sample one of all material nodes (the preceding workspace protocol also
  allowed all material identities), or honor an explicit valid node index even when
  outside the workspace. Keep that identity for the entire episode. No future motion
  prediction, forced robot waiting, target replacement or new observation is added.
- The existing rectangle X=0.60–0.84/Y=-0.22–0.30/Z=-0.005–0.10 m is a diagnostic
  region. It neither starts the policy nor certifies that an outside point is
  physically unreachable. A goal that never enters remains in the total denominator.
- Observe `workspace_goal_ever_inside`, `workspace_goal_first_entry_seconds`,
  `workspace_goal_current_inside` at reset and each 50 Hz policy state, including
  after first pinch. Entry records are diagnostics, not privileged policy inputs,
  reward gates, additional failure gates or guaranteed grasp-opportunity labels.
- `workspace_prefix_seconds=0`, actual=requested seed, attempts=1, rejections=0.
  Reward ledgers initialize from the real initial cable positions. All actual policy
  steps receive the existing rewards and count as PPO samples, including pre-entry.
- The complete episode is **15 seconds from the initial state**, for all scenes,
  including observation, waiting, approach, grasp and lift. No hidden prefix or
  extra 8-second allowance remains. Training and evaluation use the same duration;
  explicit CLI episode duration remains authoritative for separate experiments.
  The first deployment incorrectly used 23 seconds by adding the old maximum
  prefix to the configured policy duration. The user corrected this to 15 seconds
  on 2026-10-07; that intermediate run is archived, not retroactively relabelled.
- Keep 109 observations, 5 actions, ordinary PPO, reward v6 and its paid-progress
  ledgers/contact grace, enhanced6 bounded DLS, TableExecutionGuard on, old filter
  off, contact cache, 48 train/13 evaluation workers, current motion/physics and
  public success thresholds. Keep the current curriculum stage and promotion rule;
  clear only pass history because the population/start protocol changed.

## Reproducibility

`--workspace-start-protocol physical_entry_v1` retains the archived conditioned
reset/goal selection. Evaluation inherits the saved protocol. A historical saved
workspace configuration lacking this field is explicitly treated as v1. Old videos,
weights and evaluations are preserved; v1 and v2 scores must not be mixed.

Training configuration records start protocol, goal sampling, effective entry
seconds and full episode duration. Evaluation JSONL/CSV records the goal entry
fields, seed, scene, material identity and result. Standard evaluation still uses
all requested episodes, including goals that never enter. Scene mixture and the
existing aggregate curriculum gate are unchanged in this change; their separate
known weakness is not silently addressed here.

## Performance and limitations

Removing the entry loop removes its reset-time simulation and the associated
synchronous wait. It does not eliminate other physics costs or SB3 synchronous
sampling. FPS is policy samples/wall time; pre-entry policy samples now count.
The broader target population and initial-state control change the learning task;
the configured episode duration remains the original 15 seconds.
Improved speed and per-scene entry/success rates must be measured after restart;
no return to 150 samples/s or majority-entry guarantee is asserted from source alone.

## Checks and deployment

Targeted regressions cover: no hidden steps or seed fallback with every cable node
outside; outside explicit material goal accepted; all-node candidates and fixed
identity; a never-entered goal still executes and times out unsuccessfully;
latched entry diagnostics; initial reward height baseline; train/eval factories;
15-second default and explicit overrides; saved historical durations/protocols.

The authorized 151 continuation archives the verified training process and newest
complete checkpoint with optimizer/config/course/code. Resume in a new output
folder, preserve the cumulative 20M sample budget and learning/entropy/penalty
schedules, and clear old-protocol promotion history. Record actual checkpoint and
commit in the new launch manifest. Old unsaved progress cannot be recovered.


## 15-second correction deployment

Only the full episode time changes from the intermediate 23-second run to 15
seconds. The initial-state protocol, all-node goal population, control, reward,
physics, stage, workers and cumulative budget remain. Archive the intermediate
run and resume its newest complete checkpoint in a new `initial_policy_15s` run.
Clear old time-budget evaluation pass history while preserving the checkpoint
stage and learning schedules. Record the actual checkpoint and code commit in
the launch manifest. Saved intermediate weights still inherit their recorded
23-second duration when reproducing that experiment.


## Executor startup correction (2026-10-07 15:30 audit)

The v2 wrapper starts PPO at time zero but the inherited base executor still
replaces rigid/combined arm commands with ready targets until 0.82 seconds.
This contradicts the initial-state control contract. `policy_from_initial_v3`
sets the task-scoped `arm_motion_start_immediate` flag so `_arm_motion_start_time`
returns zero and the first requested arm command enters the original limit/guard
pipeline. It does not remove joint speed/position limits, table protection,
change cable drive timing, extend the 15-second episode, add observations,
force an upward action or alter rewards/physics/success/curriculum gates.

Default base environments retain `arm_motion_start_immediate=False`.
`physical_entry_v1` and saved `policy_from_initial_v2` remain selectable for
archived reproduction, including their executor delay. The v3 protocol and
actual executor flag are recorded in training configuration. Tests check the
actual requested arm target at t=0 for all three protocols, plus historical
saved evaluation inheritance. Safely resume the latest complete checkpoint in
a new run, preserve stage and cumulative schedules, and clear only evaluation
pass history across this execution protocol change.

This is a proven execution inconsistency, not proof that the 0.82-second delay
causes all low success. The current stage-4 evaluations remain weak in shape,
combined and complex-static tasks; reward weights and exploration are unchanged.
