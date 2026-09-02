#!/usr/bin/env bash
set -euo pipefail

cache_root="/data/hxai/panda_cable_pi05/cache/openpi-assets/checkpoints/pi05_base"
partial_root="${cache_root}/params.partial"
final_root="${cache_root}/params"
remote_root="https://storage.googleapis.com/openpi-assets/checkpoints/pi05_base/params"
expected_total=12441721931

if [[ -d "${final_root}" ]]; then
  echo "checkpoint already complete: ${final_root}"
  exit 0
fi

declare -A expected_sizes=(
  ["ocdbt.process_0/d/7bc9d3296d23a6fb83a6b3778ac6e964"]=2240315383
  ["ocdbt.process_0/d/ec484cf8f02dcf59e1892180f0862e40"]=1234292390
  ["ocdbt.process_0/d/b4349aaadb7dfa45c3a53fc67c04b8f6"]=1120156687
)

pids=()
for relative_path in "${!expected_sizes[@]}"; do
  local_path="${partial_root}/${relative_path}"
  expected_size="${expected_sizes[${relative_path}]}"
  actual_size="$(stat -c %s "${local_path}" 2>/dev/null || echo 0)"
  if (( actual_size > expected_size )); then
    echo "oversized partial file: ${relative_path} (${actual_size} > ${expected_size})" >&2
    exit 1
  fi
  if (( actual_size == expected_size )); then
    echo "already complete: ${relative_path}"
    continue
  fi

  echo "resuming ${relative_path}: ${actual_size}/${expected_size}"
  curl --fail --location --retry 20 --retry-all-errors \
    --continue-at - --output "${local_path}" \
    "${remote_root}/${relative_path}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

for relative_path in "${!expected_sizes[@]}"; do
  local_path="${partial_root}/${relative_path}"
  expected_size="${expected_sizes[${relative_path}]}"
  actual_size="$(stat -c %s "${local_path}")"
  if (( actual_size != expected_size )); then
    echo "size mismatch: ${relative_path} (${actual_size} != ${expected_size})" >&2
    exit 1
  fi
done

actual_total="$(find "${partial_root}" -type f -printf '%s\n' | awk '{total += $1} END {print total}')"
if [[ "${actual_total}" != "${expected_total}" ]]; then
  echo "checkpoint total mismatch: ${actual_total} != ${expected_total}" >&2
  exit 1
fi

mv "${partial_root}" "${final_root}"
echo "checkpoint complete: ${final_root} (${actual_total} bytes)"
