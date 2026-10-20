#!/usr/bin/env bash
# Dense history-read-query LoRA32 reference, preserving the original global batch.
set -euo pipefail
RESEARCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MODEL_PROFILE=hsd_gdn_readq32
export HSD_FUSED_GLOBAL_READ_BACKEND="${HSD_FUSED_GLOBAL_READ_BACKEND:-tile_online}"
export HSD_FUSED_GLOBAL_READ_STRICT="${HSD_FUSED_GLOBAL_READ_STRICT:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MAX_STEPS="${MAX_STEPS:-11921}"
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACC="${GRAD_ACC:-16}"
export WARMUP_STEPS="${WARMUP_STEPS:-256}"
export LEARNING_RATE="${LEARNING_RATE:-4e-4}"
export SAVE_STEPS="${SAVE_STEPS:-2048}"
export DEEPSPEED="${DEEPSPEED:-zero2}"
if [[ -z "${LR_SCHEDULER_KWARGS:-}" ]]; then
  decay_steps=$((MAX_STEPS / 10))
  if (( decay_steps < 1 )); then decay_steps=1; fi
  printf -v LR_SCHEDULER_KWARGS '{"num_decay_steps":%d,"decay_type":"cosine","min_lr_ratio":0.1}' "$decay_steps"
  export LR_SCHEDULER_KWARGS
fi
exec bash "$RESEARCH_ROOT/train.sh" "$@"
