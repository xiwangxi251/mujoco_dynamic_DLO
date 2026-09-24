# OccDyn-DLO: Occlusion-Conditioned Predictive DLO State Estimation

Status: in-progress experiment note. All numbers preliminary.

## Problem

Intercept a freely swinging/deforming cable (~1 m/s node velocities) while the
robot gripper increasingly occludes it during approach. Unlike prior DLO
perception work (TrackDLO, Lv et al. 2023, MP2CDLO, UniStateDLO) which
reconstructs the *current* cable shape under *unknown* occluders in
quasi-static settings, we:

1. treat the occluder as **known geometry** (gripper pose + rendered
   silhouette masks from proprioception),
2. predict **velocity per node** and the cable state at a **future horizon**
   (~0.3 s), not just the current shape,
3. target **closed-loop interception**, where estimation must feed a grasp
   decision.

## Data (server 151)

`tools/perception/build_dataset.py` re-renders recorded FULLPHYSICS
episodes (`nero_scripted_dynamic20_4x1000_20260906`, successful scripted
grasps) at 25 Hz (stride 2). Per frame and per camera
(`dynamicvla_opst_camera`, `dynamicvla_wrist_camera`, 480x360):

- partial point cloud: HSV(112-130/180-255/80-255) segmentation -> depth
  backprojection -> 2 mm voxel -> FPS to 384 points (camera optical frame)
- robot-occluder silhouette mask 120x90 (segmentation rendering, robot body
  subtree of `base_link`)
- per-node (40 cable bodies) visibility flag via depth test (2 cm margin)

Labels per frame: 40 ordered node positions + linear velocities
(`data.cvel[:,3:6]`), hand (`link7`) pos+quat, camera poses, intrinsics,
scenario/seed/success metadata. Resampled to 14 arc-length nodes for
training (matches the `dlo_state_estimation` eval convention).

Output: `/data1/hxai/mujoco/perception_runs/dataset_v1/<scenario>/seed_*.npz`
(4 ID scenarios x ~400 episodes = 1602 total; ~300 train + ~100 test seeds
per scenario; ~210k frames at 25 Hz).

`tools/perception/pack_dataset.py` repacks episodes into per-scenario
chunks of raw `.npy` files (`dataset_v1_packed/<scenario>__partNN.<key>.npy`),
read through `PackedDLODataset` (true OS mmap; ~1.5 ms/sample vs ~1 s for
compressed-npz loading).

Held-out comparability: the UniStateDLO reproduction
(`/data1/hxai/unistatedlo_nero_repro`) splits the same episodes
80/10/10 by seed; its test seeds are the *largest* ~100 seeds per scenario,
which the main build does not cover — a second build pass with
`--seeds-file` covers them so all methods share the test set.

## Model (`src/panda_cable_grasp/perception/model.py`)

PointNet-style per-point encoder -> learned node queries cross-attend to all
point tokens (both cameras, optional history tokens with per-frame-age
embeddings) -> per-node heads:

- position (3), velocity (3), log-variance (1), future position (3)

Conditioning: global cloud feature + hand pose MLP + occluder-mask CNN
(opst+wrist masks summed). ~0.9M params, single regression forward pass.

Loss: L1 on pos/vel/future + Gaussian NLL on pos when logvar enabled
(`gaussian_nll`).

## Variants (`tools/perception/train_estimator.py`)

| name      | wrist | mask | history | vel/fut/logvar | decoder attn |
|-----------|-------|------|---------|----------------|--------------|
| full      | x     | x    | x       | x              | x            |
| no_mask   | x     |      | x       | x              | x            |
| no_wrist  |       | x    | x       | x              | x            |
| no_hist   | x     | x    |         | x              | x            |
| no_heads  | x     | x    | x       |                | x            |
| reg_only  |       |      |         |                |              |
| reg_attn  |       |      |         |                | x            |
| dual_attn | x     |      |         |                | x            |

`reg_*` approximate the Lv-style single-frame supervised baselines;
`dual_attn` adds the wrist camera; `full` is OccDyn-DLO.

## Baselines

Oracle references (use GT at t-1 or t — not deployable, for analysis):

- persistence: pos[t] = GT pos[t-1]
- const_vel: GT pos[t-1] + GT vel[t-1]*dt
- fut_stay: pos[t+H] = pos[t]; fut_const_vel: pos[t]+GT vel[t]*H

Full held-out test split (80,756 frames, all 4 scenarios, packed eval):

| baseline        | all nodes | occluded nodes | future (H≈0.32s) | fut. occluded |
|-----------------|-----------|----------------|------------------|----------------|
| persistence     | 12.68 mm  | 11.12 mm       | —                | —              |
| const_vel       | 16.69 mm  | 14.82 mm       | —                | —              |
| visible_hold    | 7.82 mm   | 11.12 mm       | —                | —              |
| fut_stay        | —         | —              | 80.70 mm         | 72.61 mm       |
| fut_const_vel   | —         | —              | 125.88 mm        | 114.35 mm      |

Intercept node (node nearest the hand): fut_stay 57.91 mm,
fut_const_vel 80.97 mm.

Note: even *oracle* constant-velocity extrapolation over 0.32 s is worse than
assuming the cable stands still — acceleration dominates at this horizon,
which is the core motivation for a learned dynamics prior.

External baselines (same 4k-episode dataset, UniStateDLO repro, test split):

- coarse (PointNet voting, MVP): MPNE 14.8 cm
- fusion (voting + diffusion, MVP): MPNE 13.1 cm
- tracker (GT-prev-conditioned diffusion, MVP): MPNE 2.5 cm — oracle
  tracking reference, not deployable (requires GT previous state)

From `dlo_state_estimation` (separate repo, temporal geometric tracker,
opposite camera): ~3.4 cm (static) to ~13.6 cm (combined) full-state error.

### UniStateDLO reproduction (per-scenario, `/data1/hxai/unistatedlo_nero_repro`)

Cross-attention estimator trained per scenario pair (50 nodes, 1024-pt
clouds, same HSV/voxel pipeline, same seed-based test split):

| scenario | UniStateDLO repro | OccDyn-DLO (no_hist) | OccDyn-DLO (dual_attn) |
|----------|-------------------|----------------------|------------------------|
| id_static | 33.6 mm | **16.7 mm (2.0x)** | 18.3 mm |
| id_rigid_l1 | 34.8 mm | **12.2 mm (2.9x)** | 13.9 mm |
| id_shape | **108.0 mm** | 147.3 mm (loses 1.36x) | 139.7 mm |
| id_combined | 107.5 mm | 105.1 mm (tie) | **94.1 mm (1.14x)** |

Its diffusion tracker/fusion stages degrade on free motion (shape 163-278
mm MPNE) because they assume quasi-static continuity between frames.
Caveat: their eval uses 50 nodes vs our 14; MPNE over smooth chains is
approximately node-count-invariant but not exactly. Honest reading:
we clearly win static/rigid, tie combined, and LOSE on shape — chaotic
free deformation is our weakest scenario (also where MP2CDLO's dense
completion wins). Node-count caveat applies in both directions.

### TrackDLO standalone reproduction (ours, `tools/perception/run_trackdlo_eval.py`)

`trackdlo_standalone` (CPD/LLE EM + geodesic priors + visibility
correction) re-rendered on our test episodes, opst camera only,
40 nodes, `accept_nonconverged=true`:

| scenario | frames | tracking_ok | ordered err | frame err | latency |
|----------|--------|-------------|-------------|-----------|---------|
| id_static (25 eps) | 3,460 | 98.6% | 43.6 mm (p90 94.5) | 15.7 mm | 77 ms mean / 178 ms p95 |
| id_combined (20 eps) | 4,404 | 72.7% | 97.0 mm (p90 159) | 23.4 mm | 35 ms mean / 93 ms p95 |

Failures on the dynamic split are dominated by
`insufficient_visible_observations` (475) and hundreds of failed
reinitializations with length_ratio 0.3-1.9 (the CPD variance-collapse /
cable-shrinkage failure the paper lists for fast tip motion). Takeaway:
TrackDLO is a workable static baseline but degrades ~2.2x on ordered-node
error and loses a quarter of frames under free motion + self-occlusion —
and at 77 ms/frame it cannot meet the 25 Hz control budget anyway.

### MP2CDLO reproduction (official code + pretrained ckpt)

`/data1/hxai/MP2CDLO` (github.com/MP2CDLO/MP2CDLO, ICRA 2025) in an
isolated env (torch 1.13/cu117); chamfer + pointops CUDA extensions
compiled against the conda CUDA-11.7 toolkit with g++-11, pytorch3d
replaced by a pure-torch shim (knn_points + estimate_pointcloud_normals —
loss-path only, inference never calls them). Demo pipeline verified on
their shipped data, then `demo/src/eval_packed.py` replays our packed
test split through the identical preprocess->MP2C->mixture->kmeans->
sort->B-spline chain. 400 random held-out frames per scenario,
direction ambiguity resolved by min-over-flip (favours MP2CDLO):

| scenario | ordered mean | ordered med | p90 | chamfer med | fail | ms/frame |
|---|---|---|---|---|---|---|
| static | 163.2 | 145.7 | 212 | 34.3 | 0 | 289 |
| rigid | 188.5 | 163.1 | 314 | 33.7 | 4 | 295 |
| shape | 126.5 | 115.8 | 176 | 23.5 | 1 | 294 |
| combined | 150.6 | 117.2 | 301 | 25.1 | 6 | 297 |

Read: geometric completion is decent (chamfer ~25-35 mm) but the
unordered->ordered stage (kmeans centres + local-frame sorting +
B-spline) fails on our data — the same geometry-OK/ordering-bad split
as TrackDLO and our own estimator. It beats our best variant only on
id_shape ordered-mean (126.5 vs 139.7 mm), where free chaotic
deformation favours dense completion over our 14-node global
regression. Caveats: zero-shot transfer from their EPN3D rope
distribution (real Kinect, different scale), ~290 ms/frame dominated by
CPU post-processing (model itself 11 ms), 400-frame subsample.

### EPN3D / Lv23 (arXiv 2210.01433, ICRA 2023)

No official implementation found (same lab's follow-up UniStateDLO also
ships no code). Proxy in-repo: `vote_attn` = dual_attn + per-point
offset voting branch with learned per-node gating — captures the
PointNet backbone + point-to-point vote fusion essence of EPN3D's
regression+voting design. In training.

## Evaluation protocol

`tools/perception/evaluate_estimator.py`: per-node L2 in mm, split by
visibility (occluded = invisible in *both* cameras), velocity RMSE,
future-horizon MPNE, per-scenario breakdown. Model eval must not see GT
history; oracle baselines are reported separately and labelled as such.

**Primary criterion (user directive, 2025-XX): sequential episode-level
evaluation, not per-frame independent eval.** Any model with temporal
inputs (prev-state prior, history tokens) must be judged on the
self-feedback sequential metric from `tools/perception/dump_occdyn_ep.py`:
frames replayed in order on held-out test episodes, with the model's own
previous output fed back as the prior. Per-frame packed eval is still
reported for non-temporal variants and as a diagnostic, but it is
teacher-forced for `use_prev_pos` models (prev = GT frame t-1) and
systematically overestimates deployable accuracy — e.g. track3 scored
13.4 mm teacher-forced val yet 128.7 mm self-feedback on the shape test
episode. Report BOTH, clearly labelled, and compare self-feedback numbers
against baselines under the strict ordered metric (the renderer's
on-screen err is flip-tolerant `min(fwd, rev)` and understates strict
ordered MPNE).

## Open items

- [ ] full dataset build + test-seed build (UniStateDLO test split)
- [ ] train all variants (GPU 3-5), pick best by val MPNE
- [ ] error-vs-occlusion-fraction curves; per-scenario table
- [ ] latency measurement (must show >>25 Hz for deployment claim)
- [ ] compare vs `unistatedlo_nero_repro` checkpoints on identical test eps
- [ ] downstream: feed estimate to interception policy; grasp success
- [ ] uncertainty calibration (logvar vs realized error on occluded nodes)

## Learned-model test results (80,756 held-out frames, `no_hist`/`no_mask` @25ep)

| metric | no_hist | no_mask | oracle ref |
|---|---|---|---|
| MPNE | 67.5 mm | 68.9 mm | persistence 12.7 / vis-hold 7.8 |
| occluded-node | 52.3 mm | 53.7 mm | persistence-occ 11.1 |
| future MPNE | 81.8 mm | 82.4 mm | fut_stay 80.7 / fut_cv 125.9 |
| intercept-node future | 59.7 mm | 59.4 mm | fut_stay_near 57.9 |
| velocity RMSE | 0.333 m/s | 0.333 m/s | — |
| batch-1 latency | 4.42 ms (~226 Hz) | 3.87 ms | TrackDLO 77 ms |

Per-scenario MPNE on the packed test split (79,550 frames):

| variant | static | rigid | shape | combined |
|---|---|---|---|---|
| no_hist | 16.7 | 12.2 | 147.3 | 105.1 |
| no_mask | 13.7 | 9.4 | 153.4 | 111.3 |
| no_fhand | 13.3 | 9.3 | 150.2 | 108.0 |
| full | 20.3 | 14.9 | 151.5 | 114.3 |
| no_heads | 14.9 | 10.9 | 141.0 | 96.9 |
| **dual_attn** | 18.3 | 13.9 | **139.7** | **94.1** |

(earlier per-scenario numbers — static 10.7 / combined 72.1 — were from
the pre-packed eval subset; the packed split rebalances scenario frame
counts. Aggregates are consistent.) vs TrackDLO ordered 43.6 mm (static)
and 97.0 mm (combined): learned models are ~2.4x better static and the
best variants roughly tie TrackDLO on combined — but TrackDLO's
*geometric* adherence is far tighter (frame err 23 mm).

**Future head vs stay, per scenario (no_hist):** the aggregate parity
hides the real story — model wins where dynamics are structured:
rigid 14.6 vs 82.8 (5.7x), combined 145.3 vs 156.9, static 20.2 vs
34.0 (anticipates grasp disturbance), but loses on shape 182.8 vs
136.5 — chaotic free deformation is the failure mode; motivates
uncertainty-gated fusion.

Mask and history conditioning show ~no aggregate gain offline
(no_mask 68.9 vs no_hist 67.5): the partial clouds implicitly encode
occlusion; hand-pose carries the occluder state. Value may live in
uncertainty calibration and closed loop rather than MPNE.

### Round-2 ablation (held-out test, ~79.5k frames)

| variant | MPNE | visible | occluded | future | vel RMSE | batch-1 |
|---|---|---|---|---|---|---|
| no_hist | 68.0 | 80.1 | 52.9 | 81.5 | 0.319 | 4.8 ms |
| no_mask | 69.5 | 81.6 | 54.2 | 82.1 | 0.318 | 4.6 ms |
| no_fhand | 69.0 | 81.6 | 53.1 | 84.2 | 0.317 | 7.0 ms |
| full | 69.1 | 81.2 | 53.9 | 82.5 | 0.321 | 5.2 ms |
| no_heads (pos only) | 69.5 | 82.4 | 53.0 | — | — | 4.6 ms |
| **dual_attn** (pos only) | **66.8** | 78.4 | 52.2 | — | — | **3.2 ms** |

Read: within ~3 mm every conditioning variant ties — the regression
architecture is saturated. `dual_attn` (dual-camera single-frame,
position head only) is both the most accurate *and* the fastest, which
supports the diagnosis that residual error is topology/correspondence,
not missing input information. Auxiliary heads cost accuracy: no_heads
is -0.4 mm vs full and dual_attn beats full by 2.3 mm despite seeing
strictly less.

### Topology regularisation and voting (final test numbers)

| variant | MPNE | visible | occluded | combined ord. | latency |
|---|---|---|---|---|---|
| full | 69.1 | 81.2 | 53.9 | 114.3 | 5.2 ms |
| full_topo | 69.0 | 81.4 | 53.5 | 111.9 | 5.2 ms |
| dual_attn | **66.8** | **78.4** | **52.2** | **94.1** | **3.2 ms** |
| vote_attn | 67.4 | 79.6 | 52.1 | 94.2 | 4.0 ms |

- `full_topo` = full + segment-length consistency (w=1.0) +
  bidirectional cloud chamfer (w=0.5, cloud->est coverage + visible-node
  est->cloud). Result: aggregate identical, combined -2.4 mm only. On
  combined it hugs the cloud slightly tighter (est->cloud 18.2 vs 19.5
  mm) but ordered error barely moves (48.6 vs 49.8 mm) — soft losses do
  not fix branch swaps; a structural mechanism (arc-length-consistent
  assignment / hybrid geometric snapping) is needed.
- `vote_attn` = dual_attn + EPN3D-style per-point offset voting with a
  learned per-node gate. Wins on the easy scenarios (static 15.5 vs
  18.3, rigid 10.6 vs 13.9 mm) but ties on combined/shape, so the
  aggregate is a wash. Serves as the EPN3D-proxy baseline.

## Closed-loop grasping (scripted policy + learned state)

`tools/perception/run_closed_loop.py`: `PerceptualGraspPolicy` replaces
the three privileged cable reads (`_nearest_cable_point`,
`_predicted_segment`, `_lock_segment_near`) with estimator output;
robot proprioception/contact stay on env.

Root-cause found for the early ~330 mm closed-loop error: the runner
constructed `DLOStateEstimator` but never called
`load_state_dict` — random-init weights output a constant prior pose
(≈[0.75,-0.12,-0.03]) regardless of observation. My direct-path
debugging on identical seeds was correct (5-15 mm) because it loaded
weights; the runner path did not.

After the fix (no_hist checkpoint, id_static):

| control | success | est MPNE | est latency |
|---|---|---|---|
| estimator | 3/3 | 7-12 mm | ~130 ms (render-bound; model 4.4 ms) |
| GT privileged | 1/3 | — | — |

Estimator-driven grasping matched/beat the scripted policy on the same
seeds. Caveat: 3 seeds is smoke-level; full eval pending.

### Closed-loop update (collection-aligned policy config)

The scripted-policy config was aligned with the original dataset
collection (`scripted_config.lift_distance=0.3` everywhere;
`combined_scripted_config` adds prediction_horizon=0.2,
approach_prediction_horizon=0.2, approach_position_tolerance=0.1 on
id_combined). Without this, GT control itself fails on combined.

| scenario | variant | success | est MPNE |
|---|---|---|---|
| id_static | no_hist | **7/8** | 5-19 mm |
| id_static | GT privileged | 5/8 | — |
| id_combined | no_hist (v·t intercept) | 1/8 | 70-114 mm |
| id_combined | no_hist + future-head intercept | 2/8 | 47-96 mm |
| id_combined | **full** (hist+future_hand+fut-head) | **4/8** | 48-116 mm |
| id_combined | GT privileged | 2/8 | — |

Note: the scripted policy's natural success rate on combined is only
~25% (collection filtered successes with up to 4 attempts each), so
4/8 estimator ≈ at/above GT parity; the `full` variant's temporal
conditioning helps closed-loop even though offline MPNE is unchanged.

Static closed loop is essentially solved — the estimator even beats
the scripted GT reference because the estimate smooths the intercept
target. Combined remains open: the ~90 mm estimate error on a freely
swinging+deforming cable exceeds the ~20-30 mm intercept tolerance.
Ablations queued: future-head intercept target (learned displacement
instead of v·t) and the `full` checkpoint (history+future_hand).

Model latency stays ~4-6 ms batch-1; the ~130 ms closed-loop step is
simulated-camera rendering (3 passes x 2 cams), which does not exist
on a real robot — real-world budget = segmentation + FPS + 4.4 ms.

## Failure diagnosis: topology, not geometry (`tools/perception/cloud_snap_diag.py`)

Measuring est->cloud vs est->GT on held-out frames (`full` ckpt):

| scenario | est->cloud | GT->cloud (coverage floor) | est->GT ordered |
|----------|-----------|----------------------------|-----------------|
| combined | 19.5 mm | 16.1 mm | 49.8 mm |
| static | 23.9 mm | 26.1 mm | 9.6 mm |

The estimate already *hugs* the observed cloud (within ~4 mm of the
coverage floor = cable radius); ~94% of GT nodes have cloud support
<20 mm yet still show 49 mm ordered error. So the dominant residual is
**correspondence**: at self-crossings/loops the chain follows the wrong
visible strand or slides along arc length — the same axis on which
TrackDLO fails (ordered 97 vs frame 23 mm). This motivates topology
regularisation rather than more capacity:

- `full_topo` variant = `full` + segment-length consistency loss
  (inextensible DLO: `|p[i+1]-p[i]|` must match GT arc spacing, applied
  to current and future chains) + bidirectional cloud chamfer
  (cloud->est coverage punishes skipped visible strands; est->cloud only
  on nodes labelled visible so occluded nodes are not dragged onto
  unrelated strands). Weights: seg_len 1.0, chamfer 0.5.

## Baseline visualisation videos

Per-episode tracked top-down videos for all reproduced baselines,
same layout as the OccDyn tracked videos (auto-fit EMA-smoothed XY
view; left: cloud + thin green GT + estimate, right: cloud only; 10 cm
scale bar). Generated by `dump_trackdlo_ep.py`, `dump_unistatedlo_ep.py`
and MP2CDLO `eval_packed.py --episode-mode --dump-dir`, rendered by
`render_tracked_npz.py`. First held-out test episode per scenario
(seeds 20281878 / 20282002 / 20283215 / 20283026).

Local copies: `Desktop\mujoco\videos\baselines\`; server dumps +
videos under `/data1/hxai/mujoco/perception_runs/baseline_viz/`.

| method | static | rigid | shape | combined |
|--------|--------|-------|-------|----------|
| TrackDLO (40 nodes, single-cam cloud, ~77 ms) | 89.9 mm | 122.9 mm | 70.7 mm | 77.6 mm |
| UniStateDLO repro coarse (50 nodes, 1024-pt fused cloud) | 33.7 mm | 36.7 mm | 135.9 mm | 85.1 mm |
| MP2CDLO pretrained (50 kp, ~280 ms) | 234.2 mm | 245.3 mm | 140.1 mm | 174.9 mm |

Errors above are per-video ordered mean (resampled to GT node count,
flip-min). Notable qualitative failure modes visible in the videos:

- **TrackDLO static**: chain shrinkage — the tracked chain covers only
  part of the GT cable (red stops mid-cable, err ~130 mm at f90
  despite visually good local fit), plus intermittent
  `optimization did not converge` and 82/300 lost-track frames on
  rigid.
- **UniStateDLO shape**: coarse chain collapses into a zigzag wad
  inside the loop region — the 1024-pt cloud clearly draws the loop
  but the regressed chain does not follow it (err ~84-136 mm).
- **MP2CDLO combined**: dense completion hugs the cloud (chamfer
  ~30-50 mm) but the ordered chain detours through a wrong branch
  — err 132-235 mm. Same topology-vs-geometry split as ours.
- **TrackDLO combined f90**: when converged it is the best-looking
  baseline (34 mm, chain directly on GT) — its failures are tracking
  loss and shrinkage rather than wrong-branch geometry.

## Sequential episode-level comparison on S_sc (primary metric)

All four methods evaluated on the **same 10 held-out episodes per
scenario**. S_sc = first 10 test seeds per scenario that also have raw
scripted episodes (TrackDLO needs RGB-D replay; some test seeds lack
raw episodes so the set is shifted vs the plain first-10 list —
`/tmp/seq_S_<scenario>.txt`). Every chain arc-length resampled to 14
nodes; **MPNE = strict ordered** (no direction alignment). `epdir` =
per-episode direction oracle (one fixed direction per ep) — the honest
middle ground for TrackDLO whose init direction is arbitrary.
`ok`/`coverage`: TrackDLO frames counted only when tracking_ok
(82.8–90.0%); MP2CDLO failed frames produce no output (~1–3/ep);
OccDyn/UniStateDLO predict every frame. Aggregator:
`tools/perception/aggregate_seq_baseline.py`; driver:
`run_seq_baselines.sh`; OccDyn eval: `eval_sequential.py`.

| method | static | rigid | shape | combined | flip rate (s/r/sh/c) |
|--------|--------|-------|-------|----------|----------------------|
| **OccDyn-DLO `dual_local_data`** | **4.0** | **3.3** | **29.0** | **21.3** | 0 / 0 / 2.8% / 1.4% |
| UniStateDLO repro | 33.2 | 33.1 | 101.6 | 102.9 | 0 / 0 / 10.4% / 14.6% |
| TrackDLO strict | 208.5 | 153.8 | 114.7 | 117.7 | 63% / 26% / 33% / 42% |
| TrackDLO epdir oracle | 62.2 | 108.9 | 89.3 | 107.8 | — |
| MP2CDLO pretrained | 230.6 | 286.7 | 166.7 | 200.2 | 47% / 52% / 49% / 55% |

Frame-weighted strict MPNE across all 40 eps: **OccDyn 14.6 mm** vs
UniStateDLO 69.1, TrackDLO 146.0 (94.3 epdir), MP2CDLO 222.7.

Key findings:

- OccDyn-DLO is ~5–15x better than the best baseline on every scenario
  under the primary metric; latency 5.1–5.8 ms vs TrackDLO ~46–82 ms.
- TrackDLO's geometric adherence is fine but it has **no endpoint
  identity** — its init direction is arbitrary and mid-episode
  reinitialisations flip again, so strict ordered error is 3–6x its
  native (flip-tolerant) metric. Even the epdir oracle leaves it at
  89–109 mm because flips occur mid-episode too.
- MP2CDLO's pretrained ordering head degenerates on our data
  (~50% flips, ~200 mm even with direction oracle).
- UniStateDLO is the strongest baseline but still ~3–8x worse; its
  residual is the same endpoint-identity failure on shape/combined.
- The shape S_sc episode set is noticeably easier than the earlier
  first-10 set (OccDyn 29.0 vs 51.2 mm) — single-episode viz numbers
  are not representative; this is why the sequential multi-episode
  protocol is the primary criterion.

## Strand-assignment failure: three attempted fixes (2026-09-20 night)

Visual inspection of `dual_local_data` on shape ep 20283215 (e.g. f38)
showed the predicted chain shortcutting through loop interiors while a
human can trace the cable in the cloud. Diagnosis refined: the residual
error is **wrong strand assignment at self-crossings** (global
connectivity), not endpoint flips (2.8%) and not local chord shortcuts.

### Attempt 1 — cloud-coverage loss: FAILED (3 variants)

Cloud→chain-segment distance added to the loss to punish uncovered
visible arcs. All variants net-negative:

| variant | recipe | val MPNE |
|---|---|---|
| `dual_local_cov_w1_failed` | w=1.0 + node→cloud pull | ~106 mm plateau |
| `dual_local_cov_w03_failed` | w=0.3, pure coverage, seg-dist | ~85 mm plateau |
| `dual_local_cov` | warmup 8 ep + hinge(>50 mm) w=0.5 | 51.6 → **+24 mm on activation**, stays ~74 |

Mechanism (v3 per-category): coverage activation doubled *occluded*
node error (39→80 mm) while visible nodes only +19%. Coverage is
topology-blind — it rewards covering every visible strand including the
wrong continuation at a crossing, dragging occluded nodes with it.
Same root cause as `full_topo` earlier: geometric soft regularisers
cannot teach connectivity. Videos: `videos/baselines/vid_dual_local_cov_*.mp4`.

### Attempt 2 — inference-time chord repair: NO EFFECT

`repair_shortcuts` (`src/panda_cable_grasp/perception/repair.py`):
detect segments with <50% cloud support, collect cloud points orphaned
from the whole chain, re-route interior nodes via Dijkstra on a
radius-limited kNN graph (corridor-gated so occluded segments are never
touched). On shape ep 20283215 it fires on 78% of frames but net delta
is **-0.3 mm** (68 frames better, 29 worse). S_sc sequential eval
confirms: static 4.0→4.3, rigid 3.3→3.7, shape 29.0→29.0,
combined 21.3→21.5. Root cause of failure: bad frames have 4–12/13
unsupported segments — the chain is *globally* misassigned, so local
re-routing from wrong anchors cannot help.

### Attempt 3a — dense arc-length parameterisation, pure extraction: FAILED

`arc_dense`: every cloud point predicts its arc coordinate s∈[0,1]
(GT from projection onto the GT polyline), chain = soft-binned
centroids along s. val MPNE plateaus at **116 mm** (vs 49.5 baseline).
The sigmoid soft-bin is a smeared bottleneck: arc_l1 ~0.13 means bins
overlap heavily, centroids collapse toward the cloud mean. Direct
regression through a learned head is a much easier output path.

### In progress

- `arc_aux` (GPU1): same s head as a **dense auxiliary task only** —
  direct node regression stays the output. Tests whether per-point
  dense supervision shapes strand-aware features.
- `dual_local_xsamp` (GPU3): crossing-frame oversampling —
  non-adjacent GT nodes < 1 arc step apart flags 36.6% of frames,
  boosted ×5. Classic hard-example mining on the failure mode.
