#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data1/hxai/mujoco/mujoco_dynamic_DLO
DYNAMICVLA=/data1/hxai/mujoco/DynamicVLA
CONDA=/data1/hxai/miniconda3/bin/conda
CHECKPOINT=${DYNAMICVLA}/runs/panda_cable_finetune/eval_weights/cable-four-scenarios-1000-original-single-epoch0004
OUTPUT=${PROJECT}/outputs/dynamicvla/finetune_4x50_epoch0004_nact8_20260901_retry3
SEED=20280804
TRIALS=50
EPISODE_SECONDS=15

SCENARIOS=(
  id_static
  id_rigid_l1_nominal
  id_shape_nominal_current
  id_combined_l1_nominal
)
GPUS=(2 3 4 5)
IMG_PORTS=(3386 3396 3406 3416)
ACT_PORTS=(3388 3398 3408 3418)

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT/logs" "$OUTPUT/model_outputs"

SERVER_PIDS=()
CLIENT_PIDS=()
cleanup() {
  for pid in "${CLIENT_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for i in "${!SCENARIOS[@]}"; do
  scene=${SCENARIOS[$i]}
  (
    cd "$PROJECT"
    "$PROJECT/.venv/bin/python" -m panda_cable_grasp.cli.run_dynamicvla \
      --scenario "$scene" \
      --trials "$TRIALS" \
      --seed "$SEED" \
      --episode-seconds "$EPISODE_SECONDS" \
      --instruction "Pick up the blue cable." \
      --client-timeout 900 \
      --ack-timeout 120 \
      --img-port "${IMG_PORTS[$i]}" \
      --act-port "${ACT_PORTS[$i]}" \
      --video-dir "$OUTPUT" \
      --run-name "${scene}_seed${SEED}_trials${TRIALS}_nact8"
  ) > "$OUTPUT/logs/${scene}_server.log" 2>&1 &
  SERVER_PIDS+=("$!")
done

sleep 5

start_client() {
  local i=$1
  local scene=${SCENARIOS[$i]}
  local log="$OUTPUT/logs/${scene}_client.log"
  (
    cd "$DYNAMICVLA"
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$CONDA" run --no-capture-output -n dynamicvla \
      python scripts/inference.py \
      --weights "$CHECKPOINT" \
      --rotation euler \
      --delta \
      --alias "dynamicvla-finetune-4k-epoch0004-nact8-${scene}" \
      --epoch 4 \
      --img_port "${IMG_PORTS[$i]}" \
      --act_port "${ACT_PORTS[$i]}" \
      --output_dir "$OUTPUT/model_outputs/$scene" \
      --n-action-steps 8
  ) > "$log" 2>&1 &
  local pid=$!
  CLIENT_PIDS+=("$pid")
  echo "client_started=$scene pid=$pid"
}

# Start all clients together. Model initialization can take several minutes,
# but the evaluation servers allow enough time for the clients to connect.
for i in "${!SCENARIOS[@]}"; do
  start_client "$i"
done
wait

status=0
for pid in "${CLIENT_PIDS[@]}"; do
  wait "$pid" || status=1
done
for pid in "${SERVER_PIDS[@]}"; do
  wait "$pid" || status=1
done

trap - EXIT INT TERM
exit "$status"
