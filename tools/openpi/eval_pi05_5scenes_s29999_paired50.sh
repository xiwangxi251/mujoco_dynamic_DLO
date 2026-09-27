#!/usr/bin/env bash
# Parallel five-scenario pi0.5 evaluation against a served OpenPI checkpoint.
#
# Spawns one `panda_cable_grasp.cli.run_pi05` process per scenario; each runs
# TRIALS sequential episodes whose seeds are SEED..SEED+TRIALS-1, identical
# across scenarios (paired evaluation window). Mirrors the shard layout of
# tools/eval_dynamicvla_nact8_epoch4_mirror_parallel.sh.
#
# Usage:
#   bash tools/openpi/eval_pi05_5scenes_s29999_paired50.sh
#   POLICY_PORT=8011 RUN_ROOT=... bash tools/openpi/eval_pi05_5scenes_s29999_paired50.sh
set -euo pipefail

PROJECT=/data1/hxai/mujoco/mujoco_dynamic_DLO
PYTHON=/data1/hxai/miniconda3/envs/dynamicvla/bin/python

POLICY_HOST=${POLICY_HOST:-10.1.114.130}
POLICY_PORT=${POLICY_PORT:-8011}
POLICY_CONFIG=${POLICY_CONFIG:-pi05_nero_cable_5scenes_lora_scratch}
CHECKPOINT=${CHECKPOINT:-/data/hxai/panda_cable_pi05/checkpoints/pi05_nero_cable_5scenes_lora_scratch/nero_5scenes_pandalike_scratch_20260925/29999}
INSTRUCTION=${INSTRUCTION:-"Grasp and lift the blue cable."}
SEED=${SEED:-20260804}
TRIALS=${TRIALS:-50}
RUN_ROOT=${RUN_ROOT:-${PROJECT}/outputs/pi05_5scenes_s29999_paired50}
RUN_NAME=eval
SERVE_LOG_ON_130=${SERVE_LOG_ON_130:-/data/hxai/panda_cable_pi05/logs/serve_5scenes_29999_p8011.log}

SCENARIOS=(
  id_static
  id_rigid_l1_nominal
  id_shape_nominal_current
  id_combined_l1_nominal
  id_rigid_replay_shape_nominal_25hz
)

cd "$PROJECT"
export PYTHONPATH=src:vendor

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing run root: $RUN_ROOT" >&2
  exit 2
fi

# Fail fast when the policy service is not accepting TCP connections.
"$PYTHON" - "$POLICY_HOST" "$POLICY_PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=5):
        pass
except OSError as error:
    print(f"policy service unreachable: ws://{host}:{port} ({error})", file=sys.stderr)
    sys.exit(2)
print(f"policy service reachable: ws://{host}:{port}")
PY

mkdir -p "$RUN_ROOT/logs"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for scene in "${SCENARIOS[@]}"; do
  (
    ulimit -n 4096
    cd "$PROJECT"
    exec "$PYTHON" -m panda_cable_grasp.cli.run_pi05 \
      --scenarios "$scene" \
      --trials "$TRIALS" \
      --seed "$SEED" \
      --policy-host "$POLICY_HOST" \
      --policy-port "$POLICY_PORT" \
      --policy-config "$POLICY_CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --instruction "$INSTRUCTION" \
      --output "$RUN_ROOT/shards/$scene" \
      --run-name "$RUN_NAME"
  ) > "$RUN_ROOT/logs/${scene}.log" 2>&1 &
  PIDS+=("$!")
  echo "launched scenario=$scene pid=$! log=$RUN_ROOT/logs/${scene}.log"
done

SCENARIOS_JSON="$(printf '%s\n' "${SCENARIOS[@]}")" \
PIDS_JSON="$(printf '%s\n' "${PIDS[@]}")" \
RUN_ROOT="$RUN_ROOT" SEED="$SEED" TRIALS="$TRIALS" \
POLICY_HOST="$POLICY_HOST" POLICY_PORT="$POLICY_PORT" \
POLICY_CONFIG="$POLICY_CONFIG" CHECKPOINT="$CHECKPOINT" \
INSTRUCTION="$INSTRUCTION" SERVE_LOG_ON_130="$SERVE_LOG_ON_130" \
"$PYTHON" - <<'PY'
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

project = Path("/data1/hxai/mujoco/mujoco_dynamic_DLO")
run_root = Path(os.environ["RUN_ROOT"])
seed = int(os.environ["SEED"])
trials = int(os.environ["TRIALS"])
scenarios = os.environ["SCENARIOS_JSON"].splitlines()
pids = [int(p) for p in os.environ["PIDS_JSON"].splitlines()]

def git(*args):
    try:
        return subprocess.run(
            ["git", *args], cwd=project, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return None

seeds = [seed + i for i in range(trials)]
manifest = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_root": str(run_root),
    "runner": "panda_cable_grasp.cli.run_pi05",
    "python": os.environ.get("PYTHON_BIN") or "/data1/hxai/miniconda3/envs/dynamicvla/bin/python",
    "python_version": platform.python_version(),
    "pythonpath": "src:vendor",
    "policy": {
        "config": os.environ["POLICY_CONFIG"],
        "checkpoint": os.environ["CHECKPOINT"],
        "host": os.environ["POLICY_HOST"],
        "port": int(os.environ["POLICY_PORT"]),
        "instruction": os.environ["INSTRUCTION"],
        "action_horizon_predicted": 16,
        "execute_steps": 8,
        "serve_log_on_10.1.114.130": os.environ["SERVE_LOG_ON_130"],
    },
    "scenarios": scenarios,
    "seed_base": seed,
    "seeds": seeds,
    "trials_per_scenario": trials,
    "total_episodes": trials * len(scenarios),
    "shards": [
        {
            "scenario": scene,
            "pid": pid,
            "log": str(run_root / "logs" / f"{scene}.log"),
            "output_dir": str(run_root / "shards" / scene / "eval"),
        }
        for scene, pid in zip(scenarios, pids)
    ],
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain=v1")),
}
path = run_root / "launch_manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"launch_manifest={path}")
PY

status=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "finished scenario=${SCENARIOS[$i]} pid=${PIDS[$i]} status=0"
  else
    rc=$?
    echo "finished scenario=${SCENARIOS[$i]} pid=${PIDS[$i]} status=$rc"
    status=1
  fi
done

trap - EXIT INT TERM
echo "all shards finished status=$status run_root=$RUN_ROOT"
exit "$status"
