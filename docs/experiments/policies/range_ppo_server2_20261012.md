# Range PPO on rental CPU server 2

User authorized scratch 20,000,000 samples on 2026-10-12. Base source 4d90a79,
matching the ordinary 151 training control/reward implementation; no checkpoint.
Robot speed caps remain ordinary, not the separate 4x arm-speed experiment.

## Frozen motion ranges

- Static: no drive.
- Simple or complex-initial-shape rigid motion: target speed [0.25*7/9, 0.30] m/s;
  fixed spatial path, acceleration limits, grasp suspension and reanchoring.
- Shape: drive strength [1.0, 1.5], temporal frequency multiplier [7/9, 1.0],
  independently sampled. Drive strength does not specify displacement amplitude.
- Combined: independent rigid speed, shape strength and shape frequency. Split
  clocks are opt-in; old callers retain their shared clock exactly.
- Shape replay: original trajectory, time scale [7/9, 1.0]. Advance clock and
  scale target linear/angular velocity consistently; no fictitious amplitude axis.
  Historical replay did not apply its curriculum frequency; this is an explicit
  additional change in this range experiment.

Parameters are sampled once per episode. Bounds describe commands, not measured
node speeds. Crossed corners can produce different physical speeds from the
historical coupled 1/3, 2/3 and full-strength points. Record precontact mass-weighted
mean node speed at 50 Hz from .82 s until first finger contact, sample count and
actual episode seed. Calibration holds ready controls without running a policy;
its .82..7.4 s speeds are diagnostics, not learning results or success rates.

## Ability curriculum and gates

Six groups: ordinary static; complex initial static; simple-initial rigid;
complex-initial rigid plus replay (each independently); shape; combined.
Group 1 is 100% target. Subsequent groups are 80% targets (equal within group),
20% all learned scenes (equal by scene). After group 6 qualifies, train equally
across all seven scenes for the remainder of the original budget.

Target gates: static 100 episodes; rigid/replay two parameter cells, shape four,
combined eight, each 50 episodes. Cells split each parameter at its range midpoint.
Each cell independently needs success >= .4, loaded >= .5, secured >= .4.
Every learned replay scene needs 20 episodes, success >= .3. All required cells
and scenes qualify in one fresh window. Evidence is reset for a new decision or
ability group; labels belong to the episode's starting group. Unfinished old-group
episodes cannot qualify the new group. No independent evaluation workers.

Within every scene, 20% uniform full-range sampling plus 80% adaptive cell sampling.
Sample independently and uniformly within the selected cell. For each cell:

    quality = min(1, success/.4, loaded/.5, secured/.4) * min(n/50,1)
    readiness = min(quality of immediate lower neighbours), or 1 for lowest cell
    priority = (.02 + readiness) * (.25 + 1 - quality)
    probability = .2 / cell_count + .8 * priority / sum(priorities)

Thus low cells dominate initially, accessible higher weak cells gain probability
as predecessors improve, and regressions receive practice. Combined parameters
are independently drawn within cells; adaptive selection of joint cells is
performance-conditioned and does not promise globally independent marginals.
Sampler evidence, gate evidence and phase persist in checkpoint metadata.

## Training and supervision

session_height_v1, height band and safety reward enabled. Enhanced6 bounded DLS,
singularity protection, table execution guard, cache, XYZ map SHA256
43840a44aa6a58e2bfd36ba8c282c29646a0457004f5cfb858ef95789429a756,
15 s episodes, .82 s arm ready delay, 99D state/5D actions, non-target-node task.
48-worker reference penalty ramps are preserved in sample units (480k/240k).
PPO n_steps 1024, batch 512, epochs 5, torch threads 1, CPU. Worker count follows
the server's 40 CPU quota and measured preflight throughput. Seed 20261010.
Global 20M LR .0003 -> .00003, entropy .01 -> .001; exact partial last rollout.

Detached supervisor writes atomic compact status every 120 s, freshness 300 s,
real-progress deadline 3600 s. Completion requires child exit, finite policy and
optimizer tensors, model_saved marker, correct config, valid final checkpoint
with exactly 20M samples. Monitoring reads compact status only. Smoke/calibration
data are separate and do not count toward scientific progress.

This variant changes sampling, rigid maximum, combined clocks, replay time scaling
and curriculum; comparison is a package comparison, not a one-factor causal test.
Training success rates are not independent evaluation; no manuscript table is
replaced. Other servers and active source snapshots remain unchanged.
