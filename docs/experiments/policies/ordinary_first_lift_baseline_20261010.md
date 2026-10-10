# Ordinary PPO: first holding height baseline

User authorized fixing the spawn-height lift reward gap and continuing the
existing ordinary privileged PPO on 2026-10-10. No other reward weights change.

The recorded 2,799,552-step policy's five episodes start with a 45 mm height
high-water mark. Cable settling lowers the grasped node to about 15 mm, but
the old ledger requires exceeding 45 mm before paying height progress.

New `ordinary_reward_profile=physical_progress_v2` latches the height baseline
exactly once, on the first contact-grace-qualified physical holding. Its
baseline is that material node's height at the previous control step, rather
than the spawn maximum. Acquisition of an already raised cable pays nothing;
only actual subsequent upward progress pays. This change does not fabricate
grasping, change physical motion, or provide a scripted upward command.

After the latch the original episode-wide maximum and paid physical-node
interval debit rules remain: releases and node switches never rebase; repeated
lifts below the established maximum cannot reclaim credit; real falls debit
only previously paid intervals, with their original discount and bounded total.
Unheld external motion after the latch still consumes observed high-water.
Fraction credit, contact grace, secured confirmation, public success, all
thresholds, penalties (including stall caps), approach/alignment and control
are unchanged. `physical_progress_v1` and target-node profiles reproduce their
old behavior. Configuration records `lift_height_baseline_version`.

This narrowly removes the pre-grasp spawn/settling maximum. It does not remove
the later episode-wide maximum across cable nodes; a higher different node or
external motion after first holding can still create a gap under that rule.

Verification covers the actual reward wrapper below spawn height, no reward
for acquiring externally raised cable, contact loss/regrasp without rebasing,
node switches, bounded release debits and legacy behavior. Resume retains
policy/value/optimizer, current ordinary curriculum phase, sample schedules,
the measured action map and the original cumulative 20M budget. Old results
remain frozen; learning evidence must come from new training episodes and
matched deterministic evaluations after further samples.
