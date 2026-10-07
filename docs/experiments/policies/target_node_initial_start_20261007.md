# Target-node PPO: start from the scene initial state (2026-10-07)

## Purpose and authorization

The user identified that motion-scene resets secretly advanced physics until a cable
node entered the workspace, then selected a goal and started PPO. This prevented
learning or evaluating observation, waiting and approach before entry. The user
accepted that some assigned nodes may never enter, provided most have opportunities.

## Current default: `policy_from_initial_v2`

- Reset initializes the requested scene/seed once and immediately returns the first
  policy observation. It does not call `prepare_workspace_entry` or a hidden control
  step, and never rejects/resamples a seed because of workspace membership.
- Uniformly sample one of all material nodes (the preceding workspace protocol also
  allowed all material identities), or honor an explicit valid node index even when
  outside the workspace. Keep that identity for the entire episode. No future motion
  prediction, new forced robot waiting, target replacement or new observation is added.
  The original intentional rigid/combined arm-release timing remains: ready targets
  until cable motion start (0.80 s) plus one control cycle (0.02 s), i.e. 0.82 s.
  PPO still observes and trains from t=0 during that interval; this is independent
  of workspace entry and does not run hidden physics in reset.
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


## Preserve the intentional arm delay (user correction, 2026-10-07)

The 15:30 audit incorrectly called the original rigid/combined 0.82-second arm
release a defect and deployed `policy_from_initial_v3`. The user confirmed that
this timing was deliberately set to let the cable begin moving before releasing
the arm. The authorized removal was workspace-entry pre-simulation, not this
independent execution timing. Restore `policy_from_initial_v2` as the default
and use it for current training. Do not remove the delay again without an explicit
user instruction; low success is not evidence that this intentional timing is a bug.

PPO starts observation, action generation, rewards and samples at t=0; the base
executor retains ready arm targets until 0.82 s for rigid/combined scenes. There
is no workspace-entry startup gate, hidden reset prefix or extra episode time.
The full episode remains 15 seconds from the initial state. Cable drive timing,
original limits/guards, reward, physics, observation and curriculum gates remain.

Keep explicit v3 selection solely to reproduce the archived short immediate-arm
experiment and its saved configuration; it is not the recommended/default setup.
The base `arm_motion_start_immediate` flag defaults false. Regression tests verify
actual t=0 targets for v2/v1 (ready) and archived v3 (immediate), plus saved protocol
inheritance and 15-second defaults. Archive the erroneous v3 run and safely resume
the latest complete checkpoint in a new restored-delay output, preserving the
optimizer, stage, cumulative 20M budget and schedules; clear only cross-protocol
promotion history. Keep all historical configurations and evaluations unchanged.
