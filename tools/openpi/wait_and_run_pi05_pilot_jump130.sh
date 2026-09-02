#!/usr/bin/env bash
set -euo pipefail

experiment_root="/data/hxai/panda_cable_pi05"
checkpoint_path="${experiment_root}/cache/openpi-assets/checkpoints/pi05_base/params"
download_session="pi05_base_download"

while [[ ! -d "${checkpoint_path}" ]]; do
  if ! tmux has-session -t "${download_session}" 2>/dev/null; then
    echo "download session ended without a complete checkpoint" >&2
    exit 1
  fi
  sleep 15
done

bash "${experiment_root}/run_pi05_cable_jump130.sh" \
  pi05_cable_lora_pilot \
  pilot_b8_h16_20260902_r1 \
  pi05_cable_pilot_20260902_r1
