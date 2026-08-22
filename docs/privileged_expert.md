# Privileged formula expert

This directory contains an experimental teacher policy for data collection.
It does not replace or modify the scripted policy in
`src/panda_cable_grasp/policies/scripted.py`.

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
panda-cable-expert-run --episodes 3 --workers 4
```

Outputs are written under `outputs/benchmarks/privileged_formula_expert/`, which is
already ignored by Git. This first runner records metrics only; use the
collector below when image/action training artifacts are required.

Collect successful training trajectories, automatically replacing failed
seeds until each scenario reaches its requested count. PowerShell example:

```powershell
panda-cable-expert-collect `
  --successes-per-scenario 20 `
  --max-attempts-per-scenario 200 `
  --workers 4 `
  --envs-per-scenario 2 `
  --progress-interval 10 `
  --run-name expert_four_20
```

Linux Bash uses `\` as its line-continuation character:

```bash
panda-cable-expert-collect \
  --scenarios id_rigid_l1_nominal id_shape_nominal_current id_combined_l1_nominal \
  --successes-per-scenario 50 \
  --max-attempts-per-scenario 500 \
  --workers 3 \
  --envs-per-scenario 2 \
  --progress-interval 10 \
  --run-name dynamic_three_50
```

`--workers` is the number of scenarios allowed to run concurrently.
`--envs-per-scenario` is the number of independent spawned MuJoCo processes
assigned to each active scenario. The upper process count is therefore
`min(workers, scenarios) * envs-per-scenario`; every process owns its environment,
renderer, and expert. Start with 1 or 2 environments per scenario and compare
`successes/h`; video encoding, CPU, memory, or storage can make larger values slower.

The coordinator reports aggregate attempts/hour, saved successes/hour, per-scenario
counts, and ETA every `--progress-interval` seconds. Successful quotas and attempt
budgets are divided exactly between workers, and seed ranges never overlap.

Each attempt is appended immediately to a worker-owned JSONL journal. Episode
metadata is written last as the commit marker, while manifests and progress JSON
use atomic replacement. If a run is interrupted, continue it with the same
collection arguments:

```bash
panda-cable-expert-collect \
  --scenarios id_rigid_l1_nominal id_shape_nominal_current id_combined_l1_nominal \
  --successes-per-scenario 50 \
  --max-attempts-per-scenario 500 \
  --seed 20260804 \
  --run-name dynamic_three_50 \
  --workers 3 \
  --envs-per-scenario 2 \
  --resume
```

`--run-name`, `--scenarios`, target count, attempt limit, seed, episode duration,
and instruction must match the original run. Worker counts and the progress interval
may be changed while resuming. Runs created by the older collector do not contain
`run_config.json` and cannot be resumed in place.

The run directory contains one subdirectory per scenario. Each successful
episode has a seed-based unique name and synchronized DynamicVLA-style
`*_opst.mp4` and `*_wrist.mp4`
videos. Its aligned NPZ contains full MuJoCo state, requested/applied actions,
robot state, cable state, teacher labels, frame indices, and both video file
names; the RGB frames are not duplicated inside the NPZ. A JSON metadata file
records both camera extrinsics. Failed attempts are journaled immediately and
consolidated into `episodes.csv` for
audit but their large video/state artifacts are discarded. Run-level and
scenario-level manifests record configs, hashes, seeds, and collection
completeness. Privileged cable fields are teacher/debug labels and must not be
provided to the deployed student policy.
