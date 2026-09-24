#!/bin/bash
# Run a baseline on the S_sc episode set (10 held-out test seeds per
# scenario that ALSO have raw scripted episodes, listed in
# /tmp/seq_S_<scenario>.txt).
# Usage: bash run_seq_baselines.sh <trackdlo|unistatedlo|mp2cdlo> <gpu>
set -u
WHICH=$1
GPU=$2
PY=/data1/hxai/miniconda3/envs/dynamicvla/bin/python
PYMP=/data1/hxai/miniconda3/envs/mp2cdlo/bin/python
REPO=/data1/hxai/mujoco/mujoco_dynamic_DLO
EP=/data1/hxai/mujoco/datasets/nero_scripted_dynamic20_4x1000_20260906/base_p0p350_tcpdx_p0p010_tcpdy_p0p000_tcpdz_p0p000_yawm30p0/scripted
RENDER=/data1/hxai/datasets/nero_cable_dynamicvla_rendered_20260907
OUT=/tmp/seqbase/$WHICH
mkdir -p $OUT

declare -A SHORT=( [id_static]=id_static [id_rigid_l1_nominal]=id_rigid [id_shape_nominal_current]=id_shape [id_combined_l1_nominal]=id_combined )

if [ "$WHICH" = "unistatedlo" ]; then
  $PY - <<'PYEOF' > /tmp/uni_ranks.txt
import json, yaml
from pathlib import Path
R = Path("/data1/hxai/unistatedlo_nero_repro")
for sc, cfgn in [("id_static","nero_static_rigid.yaml"),("id_rigid_l1_nominal","nero_static_rigid.yaml"),("id_shape_nominal_current","nero_shape_combined.yaml"),("id_combined_l1_nominal","nero_shape_combined.yaml")]:
    cfg = yaml.safe_load((R/"configs"/cfgn).read_text())
    man = json.loads((Path(cfg["dataset"]["cache_root"])/"manifest.json").read_text())
    recs = [r for r in man["records"] if r["scenario"]==sc and r["split"]=="test"]
    for i,r in enumerate(recs):
        print(sc, r["seed"], i)
PYEOF
fi

for sc in id_static id_rigid_l1_nominal id_shape_nominal_current id_combined_l1_nominal; do
  mkdir -p $OUT/$sc
  if [ "$WHICH" = "trackdlo" ]; then
    PYTHONPATH=$REPO/src $PY $REPO/tools/perception/run_trackdlo_eval.py \
      --raw-root $EP --render-root $RENDER \
      --scenario $sc --seeds-file /tmp/seq_S_$sc.txt \
      --episodes 10 --stride 1 \
      --out-dir /tmp/seqbase/trackdlo_rows/$sc \
      >> $OUT/${sc}.log 2>&1
    $PY $REPO/tools/perception/trackdlo_rows_to_eps.py \
      --rows /tmp/seqbase/trackdlo_rows/$sc/rows.npz \
      --out-dir $OUT/$sc >> $OUT/${sc}.log 2>&1
    echo "done $WHICH $sc"
    continue
  fi
  for seed in $(cat /tmp/seq_S_$sc.txt); do
    if [ "$WHICH" = "unistatedlo" ]; then
      rank=$(awk -v s=$sc -v d=$seed '$1==s && $2==d {print $3}' /tmp/uni_ranks.txt)
      [ -z "$rank" ] && { echo "NORANK $sc $seed"; continue; }
      PYTHONPATH=$REPO/src $PY $REPO/tools/perception/dump_unistatedlo_ep.py \
        --scenario $sc --ep-rank $rank \
        --out $OUT/$sc/seed_$seed.npz --device cuda:$GPU \
        >> $OUT/${sc}.log 2>&1
    else
      mkdir -p $OUT/$sc/seed_$seed
      (cd /data1/hxai/MP2CDLO/demo/src && $PYMP eval_packed.py \
        --scenario ${SHORT[$sc]} --episode-mode --seed $seed \
        --max-frames 400 --device cuda:$GPU \
        --dump-dir $OUT/$sc/seed_$seed) >> $OUT/${sc}.log 2>&1
    fi
    echo "done $WHICH $sc $seed"
  done
done
echo "ALL DONE $WHICH"
