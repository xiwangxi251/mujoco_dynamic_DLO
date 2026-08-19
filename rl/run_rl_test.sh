#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"

cd "${project_dir}"
exec "${python_bin}" -m rl.test_rl "$@"
