#!/usr/bin/env bash
# OccDyn-DLO experiment matrix on server 151.
# Usage: bash tools/perception/run_matrix.sh
set -u
PY=/data1/hxai/miniconda3/envs/dynamicvla/bin/python
REPO=/data1/hxai/mujoco/mujoco_dynamic_DLO
ROOT=/data1/hxai/mujoco/perception_runs
DATA=$ROOT/dataset_v1
LOGS=$ROOT/matrix_logs
mkdir -p "$LOGS"
cd "$REPO"

RAW=/data1/hxai/mujoco/datasets/nero_scripted_dynamic20_4x1000_20260906/base_p0p350_tcpdx_p0p010_tcpdy_p0p000_tcpdz_p0p000_yawm30p0/scripted
RENDER=/data1/hxai/datasets/nero_cable_dynamicvla_rendered_20260912_wrist_pandalike
PACKED=$ROOT/dataset_v1_packed

SEEDS="$ROOT/test_seeds_id_static.txt $ROOT/test_seeds_id_rigid_l1_nominal.txt $ROOT/test_seeds_id_shape_nominal_current.txt $ROOT/test_seeds_id_combined_l1_nominal.txt"

phase() { echo; echo "=== $1 ==="; date; }

# ---------- Phase A: build test-seed episodes (UniStateDLO test split) -----
phase "build test-split episodes"
for s in id_static id_rigid_l1_nominal id_shape_nominal_current id_combined_l1_nominal; do
  n=$(find "$DATA/$s" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  if [ "$n" -ge 400 ]; then
    echo "skip build $s ($n episodes exist)"
    continue
  fi
  for shard in 0 1 2; do
    nohup $PY tools/perception_build_dataset.py \
      --raw-root "$RAW" --render-root "$RENDER" \
      --scenario "$s" --seeds-file "$ROOT/test_seeds_${s}.txt" \
      --episodes 400 --stride 2 --shard $shard --num-shards 3 \
      --out-dir "$DATA/$s" > "$LOGS/build_test_${s}_s${shard}.log" 2>&1 &
  done
done
wait
echo "test-split build done"

phase "pack dataset (mmap-friendly flat arrays)"
for s in id_static id_rigid_l1_nominal id_shape_nominal_current id_combined_l1_nominal; do
  $PY tools/perception/pack_dataset.py \
    --data-root "$DATA" --out-dir "$PACKED" --scenarios "$s" \
    --node-count 14 --future-steps 8 > "$LOGS/pack_${s}.log" 2>&1 &
done
wait
echo "pack done"

# ---------- Phase B: training matrix --------------------------------------
train() { # variant gpu [outname]
  local v=$1 g=$2 o=${3:-$1}
  nohup $PY tools/perception/train_estimator.py \
    --data-root "$DATA" --packed-dir "$PACKED" \
    --variant "$v" --history 4 --future-steps 8 \
    --epochs 25 --batch 256 --workers 4 --device "cuda:$g" \
    --exclude-seeds-files $SEEDS \
    --out "$ROOT/runs/$o" > "$LOGS/train_${o}.log" 2>&1 &
  echo "train $v -> gpu $g (out $o) pid $!"
}

phase "round 1: full / no_mask / no_hist"
train full    3
train no_mask 4
train no_hist 5
wait

phase "round 2: no_fhand / no_heads / dual_attn"
train no_fhand  3
train no_heads  4
train dual_attn 5
wait

# ---------- Phase C: evaluation on shared test split -----------------------
phase "evaluate on test split"
for v in full no_mask no_hist no_fhand no_heads dual_attn reg_attn40; do
  ck="$ROOT/runs/$v/checkpoint_best.pt"
  [ -f "$ck" ] || continue
  $PY tools/perception/evaluate_estimator.py \
    --data-root "$DATA" --checkpoint "$ck" \
    --seeds-files $SEEDS --limit-episodes 100000 \
    --history 4 --future-steps 8 --device cuda:3 \
    --out "$ROOT/runs/$v/test_metrics.json" \
    --dump-npz "$ROOT/runs/$v/test_preds.npz" \
    > "$LOGS/eval_${v}.log" 2>&1
  echo "eval $v done"
done
echo "ALL DONE"
