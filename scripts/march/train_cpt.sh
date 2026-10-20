#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?Set MODEL to a Safin-1 MARCH checkpoint or base model}"
: "${DATASET:?Set DATASET to a public dataset path or id}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for training output}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
source "$repo_root/scripts/march/_training_common.sh"
export PYTHONPATH="$repo_root/third_party/mcore-bridge/src:$repo_root:${PYTHONPATH:-}"
export USE_MCORE_GDN="${USE_MCORE_GDN:-1}"
export SWIFT_USE_MCORE_GDN="${SWIFT_USE_MCORE_GDN:-1}"
export USE_HIST_LINEAR_ATTN="${USE_HIST_LINEAR_ATTN:-1}"
march_configure_training_profile "$repo_root" "$MODEL"
args=(pt --model "$MODEL" --dataset "$DATASET" --output_dir "$OUTPUT_DIR" --max_length "${MAX_LENGTH:-$MARCH_DEFAULT_LENGTH}" --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE:-1}" --pipeline_model_parallel_size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}" --context_parallel_size "${CONTEXT_PARALLEL_SIZE:-1}" --expert_model_parallel_size "${EXPERT_MODEL_PARALLEL_SIZE:-1}" --sequence_parallel "$MARCH_RESOLVED_SEQUENCE_PARALLEL" --bf16 "${BF16:-true}")
[[ -n "$MARCH_RESOLVED_RUNTIME_CONFIG" ]] && args+=(--hsd-runtime-config "$MARCH_RESOLVED_RUNTIME_CONFIG")
[[ -n "${MODEL_TYPE:-}" ]] && args+=(--model_type "$MODEL_TYPE")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' megatron "${args[@]}" "$@"; printf '\n'; exit 0
fi
exec megatron "${args[@]}" "$@"
