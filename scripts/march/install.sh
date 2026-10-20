#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
backend="${1:-hf}"
case "$backend" in cpu|hf|training|vllm|eval) ;; *)
  printf 'Usage: %s {cpu|hf|training|vllm|eval}\n' "$0" >&2; exit 2 ;;
esac
run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then printf '%q ' "$@"; printf '\n';
  else "$@"; fi
}
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "$python_bin" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit('Use Python 3.11 for the recorded MARCH environment')
if sys.prefix == sys.base_prefix:
    raise SystemExit('Activate a dedicated virtual environment before installing this fork')
PY
  if [[ "$backend" == training || "$backend" == vllm ]]; then
    command -v nvcc >/dev/null || { printf 'CUDA source builds require nvcc on PATH\n' >&2; exit 2; }
  fi
fi
# A supplied wheelhouse makes pip offline. Native source builds also need
# their CMake/SDK dependency sources; package wheels alone are insufficient.
if [[ -n "${WHEELHOUSE:-}" ]]; then
  [[ -d "$WHEELHOUSE" ]] || { printf 'WHEELHOUSE does not exist: %s\n' "$WHEELHOUSE" >&2; exit 2; }
  export PIP_NO_INDEX=1 PIP_FIND_LINKS="$WHEELHOUSE"
fi
if [[ "${PIP_NO_INDEX:-0}" == "1" && "$backend" == vllm ]]; then
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DFETCHCONTENT_FULLY_DISCONNECTED=ON"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$python_bin" "$repo_root/scripts/march/check_offline_build.py"
  fi
fi
constraints="$repo_root/requirements/march-constraints.txt"
run "$python_bin" -m pip install 'setuptools>=77,<81' wheel 'packaging>=24.2' ninja 'cmake>=3.26.1'
if [[ "$backend" == cpu ]]; then
  run "$python_bin" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
  run "$python_bin" -m pip install -r "$repo_root/requirements/march-cpu.txt"
else
  torch_packages=(torch==2.11.0)
  if [[ "$backend" == training ]]; then
    torch_packages+=(torchvision==0.26.0)
  fi
  run "$python_bin" -m pip install "${torch_packages[@]}" --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
  run "$python_bin" -m pip install -c "$constraints" -r "$repo_root/requirements/march-hf.txt"
  case "$backend" in
    training)
      # NVIDIA runtime wheels omit compiler headers such as crt/host_defines.h.
      # Keep the complete SDK on the search path alongside wheel cuDNN headers.
      if [[ "${DRY_RUN:-0}" != "1" ]]; then
        toolkit_include="$("$python_bin" - <<'PYSDK'
import os
from pathlib import Path
import shutil

if os.environ.get('CUDA_HOME'):
    cuda_root = Path(os.environ['CUDA_HOME'])
else:
    cuda_root = Path(shutil.which('nvcc')).resolve().parent.parent
include = cuda_root / 'include'
if not (include / 'crt/host_defines.h').is_file():
    raise SystemExit(f'CUDA development headers missing: {include}/crt/host_defines.h')
print(include)
PYSDK
)"
        export CPATH="$toolkit_include${CPATH:+:$CPATH}"
      fi
      # TE's wheel builder otherwise probes GitHub even with PIP_NO_INDEX.
      # CUDA wheels include cuDNN headers that the system toolkit may lack.
      run env NVTE_PYTORCH_FORCE_BUILD=TRUE NVTE_BUILD_USE_NVIDIA_WHEELS=1 \
        "$python_bin" -m pip install --no-build-isolation --no-binary=transformer_engine_torch \
          -c "$constraints" -r "$repo_root/requirements/march-training.txt"
      run "$python_bin" -m pip install --no-build-isolation -c "$constraints" -e "$repo_root/third_party/mcore-bridge"
      ;;
    vllm)
      run "$python_bin" -m pip install -c "$constraints" -r "$repo_root/requirements/march-vllm.txt"
      run "$python_bin" -m pip install -c "$constraints" -r "$repo_root/third_party/vllm/requirements/build/cuda.txt"
      run env VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.20.2rc1.dev92+g20cac26b1.march0.1.0}" VLLM_USE_PRECOMPILED=0 "$python_bin" -m pip install --no-build-isolation -c "$constraints" -e "$repo_root/third_party/vllm"
      ;;
    eval)
      run "$python_bin" -m pip install --no-build-isolation -c "$constraints" -e "$repo_root/third_party/lm-evaluation-harness[ifeval]"
      ;;
  esac
fi
run "$python_bin" -m pip install --no-build-isolation -c "$constraints" -e "$repo_root"
run "$python_bin" -m pip check
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  report_dir="${ENVIRONMENT_REPORT_DIR:-$repo_root/.local/environment}"
  mkdir -p "$report_dir"
  "$python_bin" -m pip freeze > "$report_dir/$backend-resolved.txt"
  check_backend="$backend"
  [[ "$backend" == cpu ]] && check_backend=hf
  "$python_bin" "$repo_root/scripts/march/check_environment.py" --backend "$check_backend" \
    --strict-versions > "$report_dir/$backend-check.json"
  printf 'Environment report: %s\n' "$report_dir"
fi
