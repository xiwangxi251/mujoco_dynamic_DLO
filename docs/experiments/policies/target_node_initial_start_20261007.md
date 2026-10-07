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
- New target-workspace CLI training defaults to a full 23-second episode measured
  from the initial state. This explicitly covers the previous upper budget of an
  8-second hidden prefix plus 15-second policy period. Explicit episode duration is
  authoritative; this continuation sets 23 seconds for both train and evaluation.
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
Full 23-second episodes and a broader target population change the learning task.
Improved speed and per-scene entry/success rates must be measured after restart;
no return to 150 samples/s or majority-entry guarantee is asserted from source alone.

## Checks and deployment

Targeted regressions cover: no hidden steps or seed fallback with every cable node
outside; outside explicit material goal accepted; all-node candidates and fixed
identity; a never-entered goal still executes and times out unsuccessfully;
latched entry diagnostics; initial reward height baseline; train/eval factories;
new 23-second default and explicit overrides; historical evaluation protocol.

The authorized 151 continuation archives the verified training process and newest
complete checkpoint with optimizer/config/course/code. Resume in a new output
folder, preserve the cumulative 20M sample budget and learning/entropy/penalty
schedules, and clear old-protocol promotion history. Record actual checkpoint and
commit in the new launch manifest. Old unsaved progress cannot be recovered.
