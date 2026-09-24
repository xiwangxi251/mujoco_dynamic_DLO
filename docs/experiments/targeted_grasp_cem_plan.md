# Target-Conditioned Material-Segment Difficulty: Implementation Plan

> Status: implementation-ready design; queued behind the priority-1
> occlusion × deformation experiment. Within this study, the CEM pilot still
> precedes any goal-conditioned PPO training.
> Scientific scope: controller-bounded difficulty under complete simulator state, not policy-independent graspability.

## 1. Frozen question and reported quantity

For a saved simulator state `x_t`, immutable material target `s`, horizon `H`,
controller class `CEM`, and rollout budget `B`, estimate

```text
R_hat_CEM(x_t, s; H, B)
    = held-out perturbation success rate for the best CEM candidates.
```

Success requires the existing public strict grasp-and-lift criterion **and** a
grasped material body within the pre-registered intended-target neighborhood.
A strict success on another cable body is a wrong-target failure. The current
scripted/formula expert target-selection cost is not used.

Freeze the target discretization before coding: use the canonical physical
node/body index `i` with normalized material coordinate `s=i/(N-1)`. Compute a
centered local tangent for interior nodes and a one-sided tangent at endpoints.
The proposed scoring tolerance is the target body plus its immediate material
neighbors (`|grasped_index-i| <= 1`) to absorb capsule/contact discretization;
report exact-index and ±1 results together in the pilot so this choice remains
auditable rather than silently broadening success.

The first scientific comparison is paired `static` versus `shape`, where there
is no global sweep. Paired `rigid_l1` versus `combined_l1` is secondary.

## 2. Main implementation units

### 2.1 `src/panda_cable_grasp/evaluation/targeted_grasp.py`

Add four immutable dataclasses:

- `TargetedStateSnapshot`: MuJoCo FULLPHYSICS state plus `ctrl`, deterministic
  motion-profile/Python state, robot command history, target identity, and the
  pre-contact task bookkeeping needed for an exact restore.
- `GraspPrimitive`: contact time, pre-grasp height, tangent/normal offsets,
  tangent-relative yaw, descent duration, close onset/rate, lift onset/speed.
- `TargetedRolloutResult`: intended target, contacted/grasped material index,
  public task success, intended-target success, termination reason, realized
  contact time, safety flags, and search-only progress diagnostics.
- `TargetedCEMConfig`: parameter bounds, population, elite fraction,
  iterations, candidate count for robust re-evaluation, perturbation count,
  horizon, and RNG seed.

Add the following components:

- `capture_targeted_snapshot(env) -> TargetedStateSnapshot`
- `restore_targeted_snapshot(env, snapshot, target_index) -> None`
- `intended_target_success(info, target_index, radius) -> bool`
- `TargetedGraspPrimitiveController`
- `TargetedCEMPlanner.optimize(snapshot, target_index)`

Use NumPy CEM initially; add no optimizer dependency. Reuse NERO pose IK and
the environment's normal safety/velocity filters. The controller reads only
the fixed target material identity and its privileged current pose/tangent.
It may update the end-effector reference for that identity but may never call
the current expert candidate selector or switch targets.

### 2.2 `tools/experiments/run_targeted_grasp_cem.py`

The CLI must support:

- scenario, seed list, sample times/phases, and material-index list;
- CEM budget and perturbation overrides;
- smoke mode for one `(seed,time,target)` cell;
- deterministic resume by cell key;
- JSONL cell results plus a manifest containing Git state, environment config,
  compiled-model identity, CEM bounds/budget, and exact target-neighborhood
  definition.

Do not start with all 40 targets. The pilot uses nine uniformly spaced targets
including both endpoints, six registered phases, five paired seeds, and
`static/shape`. Expand only after the gates in Section 5 pass.

### 2.3 Tests

Add:

- `tests/unit/test_targeted_grasp.py`
- `tests/integration/test_targeted_grasp_cem.py`

The test suite must cover exact target locking, wrong-target rejection,
snapshot determinism, no mutation of the source episode, CEM RNG determinism,
parameter clipping, budget accounting, and a one-cell end-to-end smoke run.

## 3. Snapshot/restore contract

`mjSTATE_FULLPHYSICS` alone is insufficient because it does not restore all
environment-side bookkeeping or the current actuator command. The first
version is deliberately limited to the canonical single-DLO scenes and must
reject multi-object/object-family scenes.

Capture and restore at least:

- MuJoCo FULLPHYSICS state and `data.ctrl`;
- RNG bit-generator state;
- `phase_offset`, `spatial_phase`, hidden-velocity direction, and stochastic
  shape parameters;
- rigid reference shape/COM and initial cable translation;
- target body/index and target object index;
- last requested/applied actions and motion-limit state required for identical
  subsequent filtering;
- rigid-motion suspension/release flags;
- grasp/success state and counters, which must be clean for the pre-contact
  snapshot used by the pilot;
- disturbance and safety diagnostics that are mutated during stepping.

After restore, call `mj_forward` and reset reusable safety probe data. Prove
the contract by comparing an uninterrupted no-robot continuation with a
restore-and-continue trajectory over the full CEM horizon.

## 4. CEM rollout and scoring

Every candidate uses the same feedback phase structure:

1. approach the specified segment using the candidate pre-grasp offset/yaw;
2. track the same material identity until the candidate contact/close time;
3. descend and close according to the candidate timing/rate;
4. lift and hold until strict success or horizon termination.

Use a lexicographic search key so arbitrary weights cannot redefine the task:

1. intended-target strict task success;
2. intended-target bilateral grasp retained during lift;
3. target lift and lifted fraction;
4. negative target distance only when no candidate establishes contact;
5. safety/wrong-target/time-limit penalties.

Only item 1 is the reported outcome. Search progress fields are diagnostics.
The best few candidates are re-run under held-out pose, timing, friction, and
disturbance perturbations. CEM samples and robust-evaluation perturbations use
different recorded RNG streams.

## 5. Pilot gates and stopping rules

Do not begin PPO validation until all gates pass:

1. **Restore fidelity:** uninterrupted and restored no-robot target-node
   trajectories agree to numerical tolerance over the full planning horizon.
2. **Target fidelity:** an episode that lifts a non-target segment is always
   scored as intended-target failure even if public `task_success` is true.
3. **Search usefulness:** the high-budget CEM improves over the fixed initial
   population median on a mixed easy/hard smoke set.
4. **Budget stability:** segment ranking between the two largest tested CEM
   budgets has pre-declared high agreement; inspect both rank correlation and
   top-region overlap before choosing the frozen budget.
5. **Non-degeneracy:** the pilot contains a useful mixture of successful and
   unsuccessful/uncertain cells; an all-zero or all-one map does not justify a
   formal experiment.
6. **Paired integrity:** `static/shape` cells share seed, initial curve, robot
   state, target list, and optimizer/perturbation RNG keys.

If ranks remain budget-sensitive, expand the action parameterization or budget
before collecting more cells. If CEM and later goal-conditioned PPO disagree
substantially, report controller dependence rather than averaging the fields.

## 6. Goal-conditioned PPO validation (only after CEM pilot)

Extend the existing privileged RL environment with a target-locked mode. Add
normalized material coordinate, target position/velocity/tangent, and remaining
horizon to the existing state observation. Sample target material coordinates
through a train split; reserve material-coordinate intervals, phases, and seeds
for validation. Keep the existing task-space action interface.

All target-selection helpers are disabled in this mode. Reward, termination,
and evaluation use intended-target success; a public success on another segment
is a wrong-target failure. Train one policy across all training targets. Compare
its material-segment ranks, top regions, and deformation-phase changes with the
frozen CEM map; do not use PPO to tune the CEM protocol post hoc.

## 7. Execution order

1. Implement and test snapshot/restore plus intended-target scoring.
2. Implement one deterministic `GraspPrimitive` rollout without CEM.
3. Add NumPy CEM and one-cell smoke CLI.
4. Run budget-convergence smoke cells on server 151.
5. Freeze the pilot manifest and run nine targets × six phases × five paired
   seeds for `static/shape`.
6. Review field variance, ranking stability, artifacts, and compute cost.
7. Only then implement/train the target-conditioned PPO validation.

## 8. Verification environments and dirty-tree handling

The local Windows Python currently lacks MuJoCo. Run formatting/static/import
checks locally where possible, but run all simulator, restore-fidelity, CEM,
and integration tests on server 151 with
`/data1/hxai/miniconda3/envs/dynamicvla/bin/python` (MuJoCo 3.11).

Both local and server checkouts contain unrelated uncommitted experiment work,
including overlapping environment files. Preserve it. Transfer only new files
or reviewed narrow patches, compare each target path before applying it on 151,
and never synchronize or reset the whole repository to run this experiment.
