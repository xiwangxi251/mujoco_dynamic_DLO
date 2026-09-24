# Occlusion × Deformation: Priority-1 Implementation Plan

> Status: implementation-ready experiment design.
> Priority: this experiment precedes the material-segment CEM study.

## 1. Claim and minimum evidence

The claim is not merely that occlusion is harmful or that deformation is hard.
It is the interaction claim:

> For matched visible information, target kinematics, and approach conditions,
> losing observations of the target region is more damaging under internal
> deformation than under global rigid motion, because visible DLO motion no
> longer determines the hidden material motion.

The claim requires both of the following primary outcomes:

1. a positive deformation-by-occlusion interaction in hidden target/future
   contact prediction error; and
2. the same-sign interaction in strict closed-loop grasp success.

The experiment is negative if either interaction is small or uncertain after
the pre-registered matching controls. A larger raw deformation failure rate
alone is not supporting evidence.

## 2. Three-layer evidence chain

### Layer 0 — Oracle matched-mask identifiability test

This is the cheapest and cleanest mechanism test. It removes the camera, HSV
segmentation, point sampling, and gripper appearance.

Generate paired `rigid_l1` and `combined_l1` trajectories with the same initial
curve, global rigid sweep, seed, target material index, and registered motion
phase. Resample both to the same ordered 14 material nodes. Feed exact positions
of visible nodes to the predictor and hide the same contiguous target-centered
material indices for the same duration in both trajectories. The mask shape
and duration are copied from real gripper-occlusion statistics but are identical
between the paired rigid and deforming clips.

Use two predictors:

- a deterministic rigid-transform/history extrapolator fitted only to visible
  nodes; and
- one shared small temporal predictor trained on a balanced mixture of both
  motion types and mask durations.

The primary metric is future position error of the hidden intended-contact
node at the frozen contact horizon (initially 0.32 s). Secondary metrics are
hidden-node ordered MPNE, target tangent error, and error growth with
consecutive hidden duration.

This layer answers the precise mechanism question: with the same exact visible
nodes and the same missing material interval, can the hidden future be inferred
from the exposed DLO? It is not itself a grasping result.

### Layer 1 — Paired real-render perception test

For every saved physical state, render two observations without stepping the
simulator between them:

- `normal`: all gripper visual geoms enabled;
- `gripper_hidden`: only the NERO gripper visual group disabled.

Collision geoms, dynamics, contacts, cameras, lighting, and all cable geoms
remain unchanged. Define a node as **gripper-occluded** only when it is visible
in `gripper_hidden` but not in `normal`. Keep self/table occlusion as separate
labels rather than attributing it to the gripper.

Run the same frozen sequential estimator on both renders. Train it with
balanced render-mode augmentation so `gripper_hidden` is not an unseen test
distribution. The primary comparison remains `rigid_l1` versus `combined_l1`;
`static` versus `shape` is a secondary replication.

Record per frame: normal/hidden visibility; gripper-occluded indices and
duration; target-node current and 0.32-s future error; hidden-node MPNE; target
tangent error; target speed/acceleration; gripper-relative pose; distance and
time to contact; camera update age; and sensor delay.

### Layer 2 — Closed-loop consequence

Use the point-cloud PPO path first. It observes cable points and proprioception,
not raw robot pixels, so hiding the gripper adds previously occluded cable depth
samples without removing a learned robot-appearance cue.

Evaluate frozen weights in the same 2×2 design:

| Internal deformation | Observation render |
|---|---|
| off: `rigid_l1` | normal / gripper-hidden |
| on: `combined_l1` | normal / gripper-hidden |

Pair seeds, initial curves, global motion, camera cadence/delay, and policy RNG.
Because closed-loop branches diverge after their actions diverge, also replay
fixed actions up to the first close command to measure the pre-action perception
effect on identical physical states.

Include privileged-state PPO as a negative control. Its result must be
invariant to render mode. Add a second vision method only after the point-cloud
pilot works. Raw-RGB VLA/IL is excluded from the first pilot because removing
the gripper is a large appearance shift unless augmented during training.

The primary closed-loop outcome is strict task success. Mechanism-linked
secondary outcomes are target/future error immediately before first close,
missed-target distance at expected contact, and first-close timing.

## 3. Matching and causal controls

Pre-register matching or stratification on:

- gripper-occluded target-node count/fraction;
- consecutive occlusion duration;
- gripper-relative target pose and distance;
- time-to-contact;
- target instantaneous speed and acceleration magnitude;
- camera update age and sensor delay.

Freeze the intended material target at the start of each fixed-trajectory clip;
do not allow two observation branches to select different target nodes for the
perception metric. Use the same physical state twice for the render intervention,
so no post-hoc visibility bin is asked to carry the causal claim.

As a stronger offline robustness check, construct a rigidified kinematic twin
from a deforming trajectory whose designated target position and tangent are
matched over the prediction horizon. This is for perception only; do not present
a kinematically imposed curve as a physically valid closed-loop scene.

## 4. Primary estimands and statistics

For a higher-is-better quantity `S`, define

```text
P_rigid = S(rigid, gripper_hidden) - S(rigid, normal)
P_deform = S(combined, gripper_hidden) - S(combined, normal)
I_occ_x_def = P_deform - P_rigid
```

For errors, use `normal - gripper_hidden` so a positive interaction has the same
meaning: removing gripper occlusion helps deformation more.

Report paired bootstrap confidence intervals over seed/trajectory pairs; a
mixed-effects deformation × occlusion model; effect curves over consecutive
occlusion duration; and the privileged-policy render-mode difference as an
intervention sanity check.

The main figure connects the chain in one layout: matched-mask hidden-future
error, real-render hidden-future error, and paired closed-loop success.

## 5. Pilot matrix and decision gates

### Engineering smoke

- 2 seeds, 2 phases, both motion types;
- render one state normal/hidden and assert identical `qpos`, `qvel`, `ctrl`,
  contacts, and subsequent no-action physics;
- assert pixels/depth differ inside the gripper silhouette and previously
  hidden cable pixels become visible;
- verify privileged observation is bitwise independent of render mode.

### Decision pilot

- 20 paired seeds;
- 4 registered approach/contact phases per seed;
- `rigid_l1` and `combined_l1`;
- both render modes from every saved state;
- at least three pre-registered occlusion-duration bands;
- fixed-action perception replay first, then paired closed-loop point-cloud PPO.

Proceed to the formal experiment only if:

1. the render intervention changes observation but not physics;
2. paired cells have useful overlap in all matching variables;
3. Layer 0 shows a stable hidden-future error interaction;
4. Layer 1 has the same-sign interaction under real rendering; and
5. the closed-loop result is non-degenerate and the privileged control is
   invariant.

Failure of Gate 3 means the mechanism is unsupported. Failure of Gate 4
localizes the problem to the real perception pipeline. Failure of Gate 5 means
the information effect has not yet been shown to matter for grasping.

## 6. Implementation units

1. Add `src/panda_cable_grasp/perception/rendering.py` to select the dedicated
   gripper visual group and render normal/hidden RGB, depth, and segmentation
   from one state.
2. Extend `tools/perception/build_dataset.py` to store both render branches,
   normal/hidden node visibility, and a gripper-occluded-node mask.
3. Extend `src/panda_cable_grasp/rl/pointcloud.py` with a frozen render-mode
   option using the same helper; do not duplicate geom selection logic.
4. Add `tools/experiments/run_occlusion_deformation_pilot.py` for the paired
   matrix, deterministic resume, JSONL cells, and a full manifest.
5. Add `tools/experiments/analyze_occlusion_deformation.py` for paired
   interactions, matching diagnostics, confidence intervals, and figure data.
6. Add unit tests for geom selection and visibility labels, plus a server
   integration test proving observation-only intervention invariance.

All simulator/render integration tests run on server 151 with MuJoCo 3.11.
Apply only reviewed narrow patches because both checkouts contain unrelated
uncommitted environment work.

## 7. Execution order

1. Implement and test the observation-only gripper render intervention.
2. Build a small paired state/render dataset and audit visibility labels.
3. Run Layer 0 matched-mask oracle pilot.
4. Run Layer 1 fixed-trajectory real-render pilot.
5. Run Layer 2 point-cloud PPO pilot plus privileged negative control.
6. Freeze the formal seed/phase/matching manifest only after the gates pass.
7. Resume the material-segment CEM study after this priority-1 experiment has a
   clear positive or negative result.

## 8. As-built implementation status (2026-09-20, weld-rigidified)

The rigid control is NOT `rigid_l1` (deterministic constant-speed sweep). Per
user direction it is a stochastic replay control whose motion intensity matches
deformation by construction:

- `tools/experiments/build_replay_bank.py` records free-evolution trajectories
  from the paired deformation source scenario into a replay bank
  (`replay_src_shape_nominal_v1`: 100 paired + 200 anonymous entries;
  `replay_src_combined_l1_v1`: 101 paired + 200 anonymous; 15 s each,
  `valid_steps` truncated at the exit line for combined sources).
- Motion profile `rigid_replay_v1` + scenarios `id_rigid_replay_shape_nominal`
  / `id_rigid_replay_combined_nominal` replay the tracked material point
  (sampled in the middle half, material index in [N/4, 3N/4]) as whole-cable
  XY translation + yaw. No Z replay. The tracked node follows the recorded
  trajectory exactly; the servo reference subtracts the node's rotational
  offset so rotation is not double-counted.
- Rigidity is enforced by dormant MuJoCo weld equalities compiled into the
  shared model: one `mjEQ_WELD` (body type) per `cableB_i` (i>=1) to
  `cableB_first`, `active0=False`. Normal/deformation scenarios keep them
  inactive (`mj_resetData` restores `eq_active0`); a replay reset computes each
  body's relative pose to the root after placement, writes `model.eq_data`,
  sets `data.eq_active`, and calls `mj_forward`. Verified on server 151:
  replay internal-deformation RMS 0.0045 cm mean / 0.0190 cm max vs 15.7 cm
  in the paired shape scene. The earlier spring/shape-hold implementation is
  superseded and its artifacts are exploratory only.
- Gripper transparency is a runtime render intervention:
  `GripperVisibilityToggle` parks the eight gripper mesh geoms
  (`gripper_flange/base/link1/link2`, 2 geoms each) in render group 4 for the
  duration of a render, then restores `model.geom_group`. `nero.xml` is
  unmodified; physics state (qpos/qvel/ctrl/contacts) is untouched. Point-cloud
  wrapper exposes `render_mode = normal | gripper_hidden | mixed`.
- Training run: `ppo_nero_pointcloud384_occdef_replay_weld_mixed_6m_20260920`,
  resumed from the 12.07 M-step point-cloud PPO baseline
  (`ppo_nero_pointcloud384_safety_heightband_remaining_from10345696_20260916/final_model.zip`),
  +6 M steps, 16 workers, mixed six-scenario training/eval set, `render_mode=mixed`
  augmentation. The earlier `..._occdef_replay_mixed_6m_20260920` run (~1.2 M
  steps) used pre-weld soft replay physics and is void for benchmark claims.
- Paired evaluator: `tools/experiments/run_occlusion_deformation_pilot.py`
  evaluates deform-source vs rigid-replay under normal vs gripper_hidden with
  exact seed pairing (`replay_seed_fallback="error"`), bootstrap CIs, and the
  interaction `I_occ_x_def = P_deform - P_rigid`.
- Replay yaw uses whole-cable Kabsch/Procrustes increments between adjacent
  recorded frames (local-tangent yaw leaked deformation into the rigid
  rotation; near-straight frames get modulo-pi branch continuity plus a
  0.2 rad/frame cap, near-isotropic frames freeze yaw). Bank-wide |yaw_rate|:
  mean 0.8 / p99 4.5 / max 7.3 rad/s vs the old local-tangent max 11.6.
- Grasp pause semantics: a confirmed grasp suspends the rigid-style drive AND
  the replay clock (one-substep detection lag is inherent). On release the
  reference is re-anchored to the cable's actual pose (`_replay_reanchor`):
  subsequent desired poses replay only the remaining relative motion, so no
  high-speed catch-up segment exists. Regression test:
  `test_grasp_pauses_replay_and_release_reanchors`.
- Current training run (post yaw+pause fixes):
  `ppo_nero_pointcloud384_occdef_replay_pause_6m_w48_20260920` on server
  130, 48 workers / GPU1, resumed from the wkabsch run's 12.166 M-step
  checkpoint (itself resumed from the 12.07 M baseline). Prior runs
  `..._replay_mixed_6m_20260920` (soft physics), `..._weld_mixed_6m_*`
  (pre-Kabsch yaw) and `..._wkabsch_6m_w48_*` (pre-pause clock) are
  superseded for benchmark claims.
- Demo artifacts: `outputs/demo_videos/paired_deform_vs_replay.mp4`
  (regenerated under weld physics + Kabsch yaw),
  `gripper_normal_vs_hidden.mp4`, and
  `replay_source_tracked_v2.mp4` (old vs new yaw overlay).
