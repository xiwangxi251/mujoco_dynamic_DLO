# DynaDLO Benchmark Paper Plan

> Working draft: 2026-09-19  
> Status: benchmark-first fallback paper; all findings below are hypotheses until the corresponding frozen experiments are run.

## 1. Paper positioning

The final project objective is a publishable paper. Method development remains valuable, but the paper must not depend on a new method succeeding. The minimum viable paper is therefore a rigorous benchmark paper with:

1. a task definition that is missing from current benchmarks;
2. independently controlled global motion and internal deformation;
3. representative state-based, vision-based, RL, IL, and VLA baselines;
4. paired, reproducible evaluation with common task success and diagnostic logging;
5. non-obvious conclusions that can only be established through the benchmark's controlled experiments.

### Working title

**DynaDLO: Benchmarking Dynamic Grasping of Deformable Linear Objects**

### Working abstract

> Grasping moving deformable objects arises in applications such as cable and hose manipulation and live-fish grasping. Compared with grasping moving rigid objects, it poses a distinct challenge: the target geometry and feasible grasp regions can continuously change before contact is established. In this work, we introduce dynamic grasping of deformable linear objects (DLOs) as a new task setting and present DynaDLO, a benchmark that independently controls global motion and shape deformation to evaluate representative approaches under different target dynamics. Our experiments show that methods performing well in static settings or under global motion can suffer substantial performance degradation once continuous deformation is introduced. Further analysis suggests that partial observability is an important factor contributing to this degradation, particularly when the target continues to deform while being partially occluded.

### Distinction from adjacent benchmarks

- DynamicVLA's DOM benchmark and DOMINO study general dynamic manipulation, primarily motivating temporal reasoning and action streaming for moving objects.
- DLO-Lab and WireCraft benchmark DLO manipulation, but focus on material/task diversity, routing, insertion, seating, and other largely contact-rich or long-horizon settings.
- DynaDLO should claim a narrower, cleaner missing axis: **interception and grasping when global motion and internal shape deformation are independently controlled before contact**.
- Existing DLO perception work such as UniStateDLO establishes that occlusion is difficult. DynaDLO's intended contribution is not merely to repeat that fact, but to measure how occlusion interacts with ongoing deformation and downstream grasping.

Primary references for novelty checks:

- [DynamicVLA / DOM](https://arxiv.org/abs/2601.22153)
- [DOMINO / PUMA](https://arxiv.org/abs/2603.15620)
- [DLO-Lab](https://arxiv.org/abs/2606.04206)
- [WireCraft](https://arxiv.org/abs/2606.18097)
- [UniStateDLO](https://arxiv.org/abs/2512.17764)
- [Occlusion-robust deformable object tracking](https://arxiv.org/abs/2101.00733)

## 2. Benchmark-level research questions

The benchmark should answer questions that cannot be answered by a single aggregate success-rate table.

### RQ1 — Does deformation amplify the effect of partial observability?

**Hypothesis H1 (occlusion–deformation interaction).** For a matched amount and duration of target occlusion, observation loss causes a larger increase in hidden-state/predicted-contact error and a larger decrease in grasp success under shape deformation than under global rigid motion.

This is an interaction claim, not two separate main effects. It is supported only if the occlusion penalty under deformation is significantly larger than the occlusion penalty under rigid motion.

### RQ2 — Does deformation create time-varying reachable grasp opportunities?

**Hypothesis H2 (controller-bounded reachable-opportunity field).** For fixed robot, information, horizon, controller class, and optimization budget, high-success material regions are approximately persistent during global rigid motion but appear, disappear, or migrate during shape deformation. The relevant dynamic difficulty is therefore the lifetime of a reachable grasp opportunity, not only target velocity. The hypothesis must be replicated across at least two substantially different controller classes before describing it as controller-robust.

### RQ3 — What motion representation is sufficient for random rigid motion versus random deformation?

**Hypothesis H3 (shared versus spatially varying motion).** Even when both processes are equally random and the designated target segment has matched speed and acceleration statistics, global rigid motion is described by one shared object-level pose/twist, whereas deformation requires material-segment-specific motion information. For a fixed fully visible target segment, a global motion representation should be sufficient under rigid motion but insufficient under deformation; target-local or full material-coordinate motion should close that gap.

This question is separate from RQ1 and RQ2: use full privileged state to remove occlusion, fix the target material segment to remove target selection and changing grasp-region quality, and compare only the information needed to predict and intercept that same segment. The privileged-state versus visual gap remains a required control within RQ1, not a separate headline finding.

Scope this as a controlled low-dimensional mechanism ablation, not a benchmark-wide method claim. The clean implementations are state PPO and low-dimensional Diffusion Policy, whose observation vectors can be changed without altering the task or inventing an unnatural interface. Point-cloud, image, and VLA policies remain in the public benchmark table but are not evidence for this representation-sufficiency claim. Require replication in both PPO and low-dimensional DP before keeping the result in the main paper; a one-method effect belongs in the appendix.

## 3. Experiment A — Causal occlusion × deformation study

**Priority 1.** Implement and pilot this experiment before the material-segment
CEM study. The implementation-ready protocol is in
`docs/experiments/occlusion_deformation_priority_plan.md`.

### 3.1 Factorial design

Use paired seeds and the same initial curve, target material segment, rigid trajectory, and motion phase.

| Factor | Levels | Purpose |
|---|---|---|
| Internal deformation `D` | off (`rigid_l1`) / on (`combined_l1`) | Isolate shape change while preserving the same global sweep |
| Visual occlusion `O` | normal gripper-visible render / gripper-hidden counterfactual render | Restore the cable pixels/depth hidden by the gripper without changing physics |
| Observation | privileged state / vision | Privileged negative control and perception gap |
| Method | representative state policy plus selected vision policies | Test whether the interaction is method-general or architecture-specific |

An additional static/shape pair can isolate deformation without global translation, but the rigid/combined pair is the primary causal comparison because its global trajectory is matched by construction.

The required new benchmark control is an **observation-only occlusion intervention**: render the same physical state while hiding only the gripper visual geoms from the policy camera, without changing collision physics. MuJoCo 3.11 supports passing a per-render `MjvOption` to `Renderer.update_scene`. A server-side feasibility check on the current NERO model confirmed that disabling its existing visual geom group changes both RGB and depth while leaving `qpos`, `qvel`, and contacts unchanged. For the formal implementation, collect every renderable geom in the `gripper_flange`, `gripper_base`, `gripper_link1`, and `gripper_link2` body subtrees—including scattered group-0 geoms—assign them to a dedicated unused visual group, and disable only that group in the counterfactual pass.

The primary causal study should first use the point-cloud PPO/perception path. Its policy input already contains object points rather than robot appearance, so a gripper-hidden depth pass adds the otherwise occluded cable points without deleting a visual self-localization cue. Applying the same intervention directly to raw-RGB VLA/IL policies would create an out-of-training-distribution image with a missing gripper. Raw-RGB methods therefore require matched training augmentation or a separate robustness study and should not be the first causal result.

### 3.2 Two-stage evaluation

#### Stage A1: perception on fixed paired trajectories

Replay identical trajectories under each render intervention. This removes policy-induced state divergence and measures the information problem directly.

Record per frame:

- visible fraction of the 14 material nodes;
- whether the intended grasp segment and its neighbors are visible;
- strict ordered MPNE for visible and hidden nodes;
- intended contact-point position error;
- local tangent/orientation error at the intended contact segment;
- future contact-point error at the policy's expected contact horizon;
- error-growth slope as a function of consecutive occlusion duration;
- topology/identity failures such as direction flips or wrong-branch assignments.

#### Stage A2: closed-loop paired grasping

Run frozen policies with the same scenario/seed matrix under the render interventions. Report:

- public `task_success` and paired success difference;
- missed-target distance at intended contact time;
- hidden-target prediction error immediately before closing;
- action saturation and policy latency.

### 3.3 Primary interaction statistic

For a higher-is-better metric such as success, define the occlusion penalty

```text
P_rigid  = score(rigid, no_occlusion)  - score(rigid, occlusion)
P_deform = score(combined, no_occlusion) - score(combined, occlusion)
I_occ×def = P_deform - P_rigid
```

For an error metric, reverse the subtraction consistently so positive interaction still means deformation amplifies the occlusion cost.

Primary analysis:

- paired bootstrap confidence interval for `I_occ×def`;
- mixed-effects logistic model for success: `success ~ deformation * occlusion * method + (1 | seed)`;
- mixed-effects model or hierarchical bootstrap for continuous errors;
- pre-register one primary occlusion intervention and treat additional masks as robustness checks.

### 3.4 Essential controls

- Match or stratify by occluded fraction, occlusion duration, target distance, and time-to-contact. Otherwise deformation may merely create more occlusion.
- Report target-region visibility separately from total visible fraction. The same hidden percentage may have very different relevance depending on where it lies.
- Separate robot-induced occlusion from self-occlusion/crossing.
- Keep policy weights, camera calibration, action frequency, latency, and paired seeds frozen.
- Include a privileged-state policy: visual occlusion should not affect it. If it does, the intervention changed physics or another code path.
- Do not use post-hoc visibility bins alone to claim causality; bins are diagnostic until paired with the render intervention.

### 3.5 Claim and falsification

Supported claim:

> Partial observability and shape deformation interact non-additively: when the gripper hides the target region, rigid motion remains extrapolatable from visible geometry, whereas ongoing internal deformation causes hidden contact geometry to diverge.

The claim is falsified if `I_occ×def` is small/uncertain after visibility matching, or if the effect is explained entirely by greater occlusion frequency in shape scenes. In that case the paper should report the negative result rather than force the abstract's final sentence.

## 4. Experiment B — Controller-bounded reachable-grasp opportunity fields

### 4.1 Operational definition

There is no empirically measurable policy-independent dynamic “graspability” field. Whether the robot can intercept a segment depends on the admissible action space, controller information, horizon, and compute budget. Let `x_t` be the complete simulator state, `s ∈ [0,1]` the target material coordinate, `H` a time budget, and `Y_s` strict success on the intended segment. For a controller `π`, define

```text
G_π(x_t, s, H) = P(Y_s = 1 | x_t, target=s, horizon=H, controller=π).
```

For a pre-declared controller/action class `Π`, the corresponding finite-horizon reachable-opportunity field is

```text
R*_{Π,H}(x_t, s) = sup_{π ∈ Π} G_π(x_t, s, H).
```

The unconstrained optimum over “all possible strategies” is computationally inaccessible and scientifically underspecified. Every reported field must therefore name `Π`, `H`, the information available to the controller, and its optimization budget. A fixed descend-close-lift probe estimates only `G_probe`; it is a reproducible lower bound and diagnostic, not the policy-independent field.

The experiment still enumerates `s` externally. No estimator may choose the middle, nearest, easiest, or currently below-gripper segment, switch targets, or count a different contacted segment as success. Clone each saved MuJoCo state and use pre-registered perturbations of pose, timing, and friction to estimate success probabilities.

A practical pilot grid is:

- all physical cable nodes/segments in the canonical model (currently 40), or a pre-registered uniform material-coordinate subsample whose resolution has first been validated against the full grid;
- 12–20 time samples or motion phases per episode;
- 10 paired seeds per scenario;
- 3–5 micro-perturbations per `(s,t)` cell.

The existing expert candidate cost may be visualized only as a debugging aid. It must not define a reachable-opportunity field because its reach, curvature, speed, boundary, and failure penalties encode policy-specific preferences.

### 4.2 Concrete target-conditioned strategies

#### Primary strategy — privileged target-conditioned CEM rollout planner

For every saved state `x_t` and externally specified material segment `s`, restore an independent MuJoCo copy with the same robot state, object state, disturbance phase, and remaining episode horizon. The planner receives complete simulator state and the immutable target identity `s`; it never scores or switches to another segment.

Use the existing task-space/IK servo as the low-level controller, but replace all current expert target selection and phase heuristics with a CEM search over one common parameter vector:

```text
θ = [contact time,
     pre-grasp height,
     tangent/normal approach offsets,
     tangent-relative gripper yaw,
     descent duration,
     close onset and close rate,
     lift onset and lift speed]
```

Each sampled `θ` drives the same feedback primitive: move toward the specified segment's privileged pose/tangent, track that same material identity until the chosen close time, close, then lift and hold. The primitive may adapt its end-effector reference to the current ground-truth pose of `s`, but it may not choose another target. Candidate rollouts are ranked lexicographically by: strict intended-segment task success; intended bilateral grasp and retained lift; then continuous target/lift progress only as a sparse-search tie breaker. Collisions, safety violations, wrong-segment grasps, and timeouts are penalized identically for every `s`. The published outcome is never this search score—only strict binary success on the intended material neighborhood.

Use the same CEM population, iterations, initialization distribution, horizon, and IK/controller limits for every `(x_t,s)`. Report a budget-convergence pilot before freezing the rollout budget. Re-evaluate the best few parameter vectors under held-out perturbations of initial pose, timing, friction, and disturbance phase; their intended-segment success fraction is `R_hat_CEM(x_t,s)`. This is the headline controller-bounded difficulty map.

#### Validation strategy — one privileged target-conditioned PPO

Train one PPO policy, not one policy per segment. Its observation is the same complete low-dimensional object/robot state augmented with normalized material coordinate `s`, the specified segment's pose/velocity/tangent, and remaining horizon. Its action space is the existing task-space arm command plus gripper command. Training samples `s`, motion phase, and scenario randomly. Reward and success are tied only to the specified segment; touching or lifting a different segment is failure. Hold out material-coordinate intervals, motion phases, and seeds during training.

Compare CEM and PPO using material-segment rank correlation, top-region overlap, field changes across deformation phase, and scenario effects. Agreement supports robustness across these two controller classes; disagreement means the measured difficulty is controller-dependent and must be reported as such. A VLA is not appropriate for this validation because perception and pretraining would confound the physical target-conditioned question.

For both strategies, log realized first-contact time because commanded and actual contact horizons differ. Treat workspace or motion-boundary exits before the registered horizon as right-censored only when the opportunity was not fully observed.

### 4.3 Separate three causes of a bad region

Report two nested fields where feasible:

1. **Controller-bounded reachable opportunity `R_hat`**: complete state, specified target, actual arm state, IK, collision, joint limits, time-to-contact, and a declared optimizer/controller class.
2. **Observable policy success `G_π`**: add the evaluated benchmark policy's visual information, target selection, prediction error, and learned behavior.

This decomposition prevents “hard to grasp,” “hard to reach,” and “hard to see” from being conflated.

### 4.4 Primary metrics

- **Reachable arc fraction** `A_{Π,H}(t)`: fraction of material coordinates with `R_hat_{Π,H}(s,t)>θ`.
- **Best measured region trajectory** `s*(t)=argmax_s R_hat_{Π,H}(s,t)` and its migration speed.
- **Window survival** `P(R_hat(s,t+τ)>θ | R_hat(s,t)>θ)`.
- **Reachable-opportunity half-life**: smallest `τ` at which window survival falls below 0.5.
- **Field autocorrelation** in material coordinates.
- **Opportunity regret within the declared field**: best measured `R_hat` minus `R_hat` at the policy-selected segment/contact time.
- Conditional relationships with local curvature, tangent angular velocity, normal rotation, segment velocity/acceleration, radius, visibility, and reach time.

### 4.5 Comparisons

- `static` versus `shape`: deformation without global motion.
- paired `rigid_l1` versus `combined_l1`: same global sweep, deformation toggled.
- low/nominal/high amplitude and frequency as separate factors.
- regular/quasiperiodic/stochastic deformation at matched RMS displacement and, where possible, matched local speed.
- uniform cable versus tapered/heterogeneous object families only as a secondary generalization study; keep the primary benchmark on one canonical DLO.

### 4.6 Main conclusions that may emerge

Within each declared controller class and horizon, the experiment is designed to distinguish several possibilities:

- deformation reduces the total amount of reachable material;
- the average amount remains similar, but good regions become short-lived;
- the best region migrates along material coordinates;
- instantaneous deformation can sometimes create easier transient grasps.

The last case enables a particularly non-obvious follow-up:

**Hypothesis H5 (faster is not always better).** In periodic or quasiperiodic deformation, a phase-aware policy that deliberately waits for a favorable reachable-opportunity window can outperform immediate interception despite making contact later.

Test immediate interception, random waiting with the same average delay, and oracle/estimated phase-aware waiting under the same episode time budget. A periodic advantage for phase-aware waiting would show that dynamic grasping is partly a scheduling problem, not merely a faster tracking problem.

## 5. Additional high-value benchmark findings

These are prioritized candidate findings, not assumed conclusions.

### P0 — Perfect-state ceiling versus visual observation gap

The existing privileged PPO versus point-cloud PPO results are suggestive, but not a clean decomposition: their observation dimensions, camera cadence/delay, training histories, continuation checkpoints, and possibly scenario exposure differ. Run a matched PPO trio across the frozen motion grid:

1. **structured privileged state PPO**;
2. **privileged complete-object point-cloud PPO**, constructing the full DLO point set from simulator geometry while using the same point-cloud encoder as the visual policy;
3. **normal partial point-cloud PPO**, with real camera cadence, delay, and gripper occlusion.

Keep action space, reward, network capacity after the observation encoder, training scenario distribution, environment steps, checkpoint-selection rule, and formal paired seeds fixed. Then decompose the total shape penalty into:

1. the drop that remains with privileged state, attributable to control, reachability, and changing grasp mechanics;
2. structured state versus privileged complete-object point cloud, attributable to representation/learning difficulty;
3. complete-object versus camera point cloud, attributable jointly to camera sampling, delay, and partial observability.

Do not call a single-camera gripper-hidden render a full-object point cloud: it still contains self-occlusion, table occlusion, and sampling loss. Use the normal versus gripper-hidden render pair separately in Experiment A to isolate the incremental effect of gripper-induced occlusion within the same camera pipeline.

This guards against overstating partial observability when the current privileged PPO already degrades strongly in shape/combined scenes.

If the structured-state versus point-cloud gap is important, add two targeted state controls rather than another family of policies: a position-only state PPO and a delayed state PPO with cadence and staleness matched to the point-cloud stream. These separate missing velocity/history and sensor delay from the geometric visibility intervention.

A matched imitation-learning pair is useful but not required for the primary claim. The minimum convincing paper can use the matched PPO trio for causal decomposition while keeping DynamicVLA, image Diffusion Policy, and pi0.5 as representative vision baselines in the main benchmark table. If resources permit, train/evaluate the existing low-dimensional Diffusion Policy and image Diffusion Policy on identical trajectories, action chunks, epochs, and checkpoint rules as a second-algorithm robustness check. Do not require a state-input version of every VLA.

### P0 — Global, target-local, and full-shape motion representations

Construct paired stochastic motion processes with matched target-segment position, RMS speed, acceleration distribution, and prediction horizon: one applies a shared random planar rigid transform to the entire DLO; the other uses random internal modes so different material coordinates have different velocities. Use full privileged state and a fixed target material segment so neither occlusion nor target selection can explain the result.

Train or fit three matched short-horizon target predictors, then feed their predicted contact pose into the same downstream controller:

1. **global**: current target pose plus best-fit object-level rigid pose/twist;
2. **target-local**: target pose/velocity, tangent and tangent rate, and a fixed local material neighborhood;
3. **full-shape**: all material-coordinate positions and velocities.

Under random rigid motion, the three should be statistically equivalent because one common transform governs every point. Under random deformation, failure of the global representation establishes that an object-level velocity is insufficient even when randomness and target speed are matched. The comparison between target-local and full-shape inputs gives an additional actionable result: equality means dynamic grasping needs only local material tracking, whereas a full-shape advantage means nonlocal deformation modes contain information required to predict the target. Confirm any prediction result with the same fixed-target controller; do not change the grasp target or policy class across representations.

Target-region visibility should remain a covariate and control in Experiment A, not a headline contribution by itself. The basic statement that hiding the intended contact region is worse than hiding a distant region is too predictable. It becomes scientifically useful only if needed to demonstrate the less obvious interaction: after matching target-region visibility and total hidden fraction, deformation still causes faster hidden-state divergence than rigid motion.

### P1 — Offline perception metrics may not predict grasp success

Compare global MPNE, hidden-node MPNE, direction/topology error, target-segment position error, tangent error, and future contact error as predictors of closed-loop success. The likely benchmark contribution is a task-aligned perception metric rather than another estimator ranking by global MPNE alone.

### P1 — Rigidified counterfactual with matched target-segment motion

To isolate internal deformation, start both counterfactuals from the same complete DLO snapshot and robot state:

1. **deforming future**: continue the registered internal shape drive;
2. **rigidified future**: freeze the snapshot's material shape and advect the entire DLO with a time-varying rigid transform chosen so that the designated target segment matches the deforming future's position and local tangent trajectory over the relevant approach/contact horizon.

The target segment therefore has matched translation and rotation in both futures, while only the relative motion of the other material coordinates differs. For perception, compare hidden-node and future-contact prediction using identical visible fraction and camera geometry. For grasp mechanics/reachability, compare only if the target segment's pose, gripper-relative trajectory, robot state, and contact horizon meet pre-registered matching tolerances.

This experiment has two interpretable outcomes:

- if the gap disappears, shape deformation has no additional effect once target-local kinematics are matched; the original benchmark gap is caused by different local motion statistics or predictability, not by “deformation” as an independent mechanism;
- if a gap remains, analyze whether changing local curvature, neighboring-segment geometry, or hidden-shape evolution explains it. This is evidence of a genuinely non-rigid effect because a common rigid transform cannot produce those relative changes.

Run a no-robot replay/perception pilot before implementing a physics-valid closed-loop version. Reject the experiment if target-pose matching, shape freezing, or force application introduces visibly different artifacts or violates the public dynamics assumptions.

### P2 — Memory can become stale under irregular deformation

At matched network capacity, compare single-frame and history-conditioned policies/estimators as regularity decreases. The falsifiable possibility is that history helps predictable global motion but can hurt under rapidly changing or stochastic local deformation unless temporal alignment and uncertainty are modeled.

### P2 — Endpoint identity/topology is a downstream failure axis

Within matched MPNE bands, compare episodes with and without direction flips/wrong-branch correspondence. An orientation-supervised intervention is needed before claiming causality; otherwise this remains a correlation with self-crossing difficulty.

## 6. Recommended execution order

### Phase 0 — Freeze definitions before expensive runs

1. Freeze canonical DLO, camera geometry, public task success, diagnostic schema, and paired seeds.
2. Select one global trajectory family (L1 or L2) after a paired pilot.
3. Assign only the NERO gripper visual meshes to a dedicated render group; add and validate normal versus gripper-hidden RGB/depth rendering without changing physics.
4. Add trajectory fields for material coordinate, target-region visibility, local curvature/tangent, occlusion duration, and policy contact horizon.
5. Implement the no-target-switch privileged CEM rollout planner and intended-segment scoring; confirm that restored object/robot states and paired disturbance phases are identical before intervention.

### Phase 1 — Cheapest decisive pilots

1. Gripper-hidden rendering equivalence test plus visibility instrumentation.
2. Matched-mask oracle and fixed-trajectory real-render occlusion × deformation pilots.
3. Twenty paired seeds for the 2×2 deformation × primary-occlusion closed-loop intervention.
4. Enumerated-segment CEM difficulty maps plus compute-budget convergence; expert-cost maps may be inspected only for debugging.
5. Ten-seed cloned-state reachable-opportunity pilot on static/rigid/shape/combined, followed by one universal target-conditioned policy only if the optimizer yields stable structure.

Proceed only if motion diagnostics confirm factor separation and the proposed metrics have adequate variance.

### Phase 2 — Formal benchmark experiments

1. Main method × motion grid, at least 100 paired seeds per cell.
2. Formal occlusion × deformation experiment with frozen interventions.
3. Formal controller-bounded reachable-opportunity fields and lifetime analysis, with replication across the rollout optimizer and universal target-conditioned controller.
4. Latency/regularity study selected before reading OOD results.
5. Frozen amplitude/frequency/length/material OOD evaluation.

### Phase 3 — Real-world validation

Validate only the central causal signatures rather than reproducing the entire simulation grid:

- rigid versus deforming target under normal and reduced gripper occlusion;
- at least one measured transient reachable-opportunity-window example;
- fixed random-block order with synchronized target, camera, robot, and actuator logs.

## 7. Minimum paper figure/table plan

1. **Figure 1:** DynaDLO task and orthogonal decomposition of global motion versus shape deformation.
2. **Table 1:** benchmark factors, methods, observation modalities, and public protocol.
3. **Figure 2 / Table 2:** main paired success matrix with confidence intervals and method ranking changes.
4. **Figure 3:** causal occlusion × deformation interaction, including hidden-contact prediction error and success.
5. **Figure 4:** controller-labelled reachable-opportunity heatmaps `R_hat(s,t)` and window-survival curves, including controller-agreement and optimizer-budget checks.
6. **Figure 5:** fixed-target prediction and grasping with global, target-local, and full-shape motion representations under matched random rigid and deforming motion.
7. **Figure 6:** the highest-value mechanism that passes its pre-declared pilot (rigidified matched-target counterfactual, latency/predictability, or another stronger result).
8. **Table 3:** frozen OOD generalization by amplitude, frequency, length, and material.

## 8. Decision rules for an honest benchmark paper

- Do not promote an observational correlation to a causal conclusion without an intervention.
- Do not use aggregate success alone when an interaction, opportunity lifetime, or failure composition is the scientific claim.
- Do not claim partial observability is the main cause unless privileged-state and counterfactual-visibility controls support that decomposition.
- Do not claim a policy-independent graspability field. Name the controller/action class, horizon, information, and compute budget for every reachable-opportunity result.
- Do not claim deformation uniformly hurts reachable opportunities if it instead creates transient favorable windows.
- If a headline hypothesis is falsified, retain the result and select the next experiment based on the pre-declared priority list, not by searching arbitrary slices for significance.
- Method results and benchmark results must remain separable: DynaDLO should still yield a coherent paper if every proposed new method fails to beat the strongest baseline.
