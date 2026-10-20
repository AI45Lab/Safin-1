#!/usr/bin/env bash
set -euo pipefail
RESEARCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON_BIN:-python}" "$RESEARCH_ROOT/train_recipe.py" "$@"
