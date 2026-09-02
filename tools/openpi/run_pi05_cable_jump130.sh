#!/usr/bin/env bash
set -euo pipefail

experiment_root="/data/hxai/panda_cable_pi05"
openpi_root="${experiment_root}/openpi"
python_bin="/data/hxai/behavior-1k-rl/.venv/bin/python"

config_name="${1:-pi05_cable_lora_pilot}"
exp_name="${2:-pilot_b8_h16_20260902}"
session_name="${3:-pi05_cable_pilot_20260902}"
log_path="${experiment_root}/logs/${exp_name}.log"

if [[ "${config_name}" != "pi05_cable_lora_pilot" && "${config_name}" != "pi05_cable_lora_full" ]]; then
  echo "Unsupported config: ${config_name}" >&2
  exit 2
fi

if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "tmux session already exists: ${session_name}"
  exit 0
fi

mkdir -p "${experiment_root}/logs"

tmux new-session -d -s "${session_name}" -c "${openpi_root}" \
  "env CUDA_VISIBLE_DEVICES=1 \
    OPENPI_DATA_HOME=${experiment_root}/cache \
    PYTHONPATH=${openpi_root}/src \
    HF_HUB_OFFLINE=1 \
    PYTHONIOENCODING=utf-8 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    ${python_bin} scripts/train.py ${config_name} \
      --exp-name ${exp_name} --overwrite >${log_path} 2>&1"

echo "session=${session_name}"
echo "log=${log_path}"
echo "physical_gpu=1"
