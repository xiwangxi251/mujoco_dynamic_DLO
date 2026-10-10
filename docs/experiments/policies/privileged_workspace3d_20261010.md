# Ordinary privileged PPO with measured 3D action constraints

User authorization: fresh state PPO, cable_x075_v1, 20M cumulative samples,
48 training workers on 151. This is not the designated-node task.

## Source and map

Scan: outputs/vertical_workspace/scripted_136_retest_20261004/
workspace3d_fixed_down15_5_20261010. Source commit c9b1629; 1603 points,
596 passing. Six primary heights 5/50/100/150/200/300 mm, plus selective
midpoint refinements. The 300 mm layer has 0/131 passing points; 250 mm
refinements have 19/29. Three selected 5-to-200 mm paths passed their
endpoint hold and moving-frame criteria. These are finite tests, not a
certification of every pose, load, continuous interior or arbitrary path.

The retained asset is rl/workspaces/workspace3d_20261010.json. It records
the exact cells.json SHA256 and scan source. Its 131 polar cells form the
largest face-connected component. Every cell has eight measured passing
corners, and any measured failure inside or on its boundary vetoes it.
Coarse 8 cm/20 degree cells and available refined 4 cm/10 degree cells
are considered; redundant contained cells are removed. No convex hull,
bounding-box substitution or unknown corner acceptance. Two detached
components are excluded. The irregular union reaches 250 mm in selected
locations; its overall numeric bounds are not a valid rectangular workspace.

Frame: world XY centered at (0.35,0) m; +X angle zero, +Y positive;
Z is TCP height relative to the tabletop. Closed jaws, fixed downward
orientation and unloaded motion were scanned. Yaw remains a policy action;
other yaw/open-jaw/loaded configurations are not certified by these samples.

## Execution and reward

`--action-workspace` projects the requested TCP displacement before the
existing enhanced6 bounded DLS. Raw sampled actions and their log probabilities
remain SB3's rollout actions. The map clips the XYZ request, preserves yaw
and gripper hysteresis, and cannot increase the requested displacement norm.
Inside the map, exact intersections with radial/angular/height cell boundaries
stop motion at its first exit; commands cannot jump across holes. Outside the
map, requests approach its nearest boundary gradually at the original action
limit. A zero translation remains zero. This is a request constraint; tracking
errors and final robot safety still belong to the controller/guard.

Training, periodic evaluation and checkpoint evaluation use the same map hash
and reward config. The map is independent of target sampling; no target goal,
workspace reset rejection, presimulation, extra timeout or failure condition
is introduced. Ordinary observations remain 99D and actions 5D. All current
seven family scenes retain the user-required 0.82 s ready execution delay;
the whole episode remains 15 s and policy samples start at t=0.

`ordinary_reward_profile=physical_progress_v1` reuses the physical-node
release-bounded debit ledger. Holding-qualified height/fraction increments
pay once across the whole episode; a fall debits previously paid physical
intervals even after release. Regrasping cannot reclaim prior high-water
progress. Unheld external motion receives no lift payment. Positive lift and
initial secured confirmation use the existing 0.35 s confirmed-distance
contact grace; no fixed jaw-aperture threshold or CLOSE-command eligibility
is added. Hold shaping reads public success_hold. No designated-node reward,
success matching or target observation is enabled. Other ordinary reward
terms, physical parameters and public success thresholds are retained.

## Ordinary curriculum and validation

The existing ordinary motion curriculum is mapped to all seven cable_x075
scenes: static (simple and complex equally), component motions (shape change,
simple rigid motion, complex initial shape with simple rigid motion, archived
shape-trajectory replay equally), then combined. Default rehearsal remains
20% static at the component stage; combined uses 10% static, 20% component,
70% combined. Motion difficulties remain 1/3, 2/3, 1. Promotion remains the
ordinary aligned-pinch >=50%, secured >=20%, public success >=10%, two
consecutive qualifying evaluations. No designated-node first-correct metric
is used. Report individual scenes, since aggregate thresholds can hide weak
scenes. The checkpoint saves ordinary phase/scenarios/difficulty for evaluation.

Targeted tests cover inner/top failures, curved boundaries and holes, gradual
outside entry, unchanged raw action, 99D/5D runtime, ready delay, contact grace,
release debit and no reclaim; existing target-grace/release tests are retained.

## Synchronization incident

Remote scan completed in about 5406 seconds. The local collector stopped
updating after about one hour; the local summary remained `scanning`, so the
scheduled wait did not start PPO. On the user's status request, remote results
were read and downloaded intact. No scan was rerun. Future startup monitoring
uses a compact status producer that checks process identity, finite updates,
and writes a terminal state; a collector must not silently leave stale pending.
