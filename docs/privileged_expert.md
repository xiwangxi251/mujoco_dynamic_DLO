# Privileged formula expert

This directory contains an experimental teacher policy for data collection.
It does not replace or modify `dynamic_grasp_policy.py`.

The expert deliberately reads privileged simulator state and the environment's
motion equations. Rigid L1/L2 motion uses the analytic path directly. Shape and
combined motion copy the current `MjData`, apply the environment's original
force field in that copy, and roll MuJoCo forward. The cached shadow trajectory
therefore includes cable constraints, damping, and contact dynamics without
advancing or modifying the live episode. Candidate cable segments and
interception times are searched repeatedly until descent starts. A learned
student must never receive this privileged rollout or the motion parameters at
evaluation time.

Run a small four-scenario experiment from the repository root:

```powershell
python -m privileged_expert.run_experiment --episodes 3 --workers 4
```

Outputs are written under `benchmark_runs/privileged_formula_expert/`, which is
already ignored by Git. This first runner records metrics only; use the
collector below when image/action training artifacts are required.

Collect successful training trajectories, automatically replacing failed
seeds until each scenario reaches its requested count:

```powershell
python -m privileged_expert.collect_dataset `
  --successes-per-scenario 20 `
  --max-attempts-per-scenario 200 `
  --workers 4
```

`--workers 4` runs the four scenarios in four spawned processes. Each process
owns its MuJoCo environment, offscreen renderer, expert, and scenario output
directory; scenarios are never run concurrently in threads. Omit the option
or use `--workers 1` for the original serial behavior. A single scenario is
still collected sequentially so its replacement-seed order stays deterministic.

The run directory contains one subdirectory per scenario. Each successful
episode has synchronized DynamicVLA-style `*_opst.mp4` and `*_wrist.mp4`
videos. Its aligned NPZ contains full MuJoCo state, requested/applied actions,
robot state, cable state, teacher labels, frame indices, and both video file
names; the RGB frames are not duplicated inside the NPZ. A JSON metadata file
records both camera extrinsics. Failed attempts remain in `episodes.csv` for
audit but their large video/state artifacts are discarded. Run-level and
scenario-level manifests record configs, hashes, seeds, and collection
completeness. Privileged cable fields are teacher/debug labels and must not be
provided to the deployed student policy.
