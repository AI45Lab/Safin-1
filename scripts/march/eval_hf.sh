#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
exec "${PYTHON_BIN:-python}" "$repo_root/scripts/march/evaluate.py" --backend hf "$@"
