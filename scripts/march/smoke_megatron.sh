#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?Set MODEL to a local Safin-1 checkpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new smoke output directory}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$repo_root/third_party/mcore-bridge/src:$repo_root:${PYTHONPATH:-}"
python_bin="${PYTHON:-python}"
export TRAIN_TASK="${TRAIN_TASK:-cpt}"
case "$TRAIN_TASK" in
  cpt|sft) ;;
  *) echo "TRAIN_TASK must be cpt or sft" >&2; exit 2 ;;
esac
export OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR already exists; choose a new directory for this smoke run" >&2; exit 2
fi
# Read-only preflight before training. Match the checkpoint's saved runtime,
# including older 35B snapshots that differ from the new-training template.
"$python_bin" "$repo_root/scripts/march/verify_checkpoint.py" "$MODEL"
"$python_bin" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Megatron smoke requires a visible CUDA GPU")
PY
mkdir -p "$OUTPUT_DIR"
"$python_bin" - <<'PY'
import json, os
from pathlib import Path
from mcore_bridge.config.parser import HSD_CONFIG_KEYS
root = Path(os.environ['OUTPUT_DIR'])
config = json.loads(Path(os.environ['MODEL'], 'config.json').read_text())
runtime = {key: value for key, value in config['text_config'].items() if key in HSD_CONFIG_KEYS}
runtime['model_type'] = config['model_type']
(root/'runtime.json').write_text(json.dumps(runtime, indent=2))
# Synthetic public text exercises at least one 512-token anchor boundary.
with (root/'train.jsonl').open('w') as stream:
    for i in range(8):
        content = ('A short public training example about memory and anchors. ' * 160) + str(i)
        row = {'text': content} if os.environ['TRAIN_TASK'] == 'cpt' else {
            'messages': [{'role': 'user', 'content': 'Explain anchor tokens.'},
                         {'role': 'assistant', 'content': content}]}
        stream.write(json.dumps(row) + '\n')
PY
export DATASET="$OUTPUT_DIR/train.jsonl"
export MARCH_RUNTIME_CONFIG="$OUTPUT_DIR/runtime.json"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export TUNER_TYPE="${TUNER_TYPE:-full}"
bash "$repo_root/scripts/march/train_${TRAIN_TASK}.sh" \
  --train_iters 1 --save_steps 1 --eval_steps 100 --add_version false --logging_steps 1 \
  --micro_batch_size 1 --global_batch_size "$NPROC_PER_NODE" \
  --save_safetensors true --no_save_optim true --no_save_rng true \
  --split_dataset_ratio 0 --truncation_strategy right --dataloader_num_workers 0 --dataset_num_proc 1 \
  --attention_backend unfused --tuner_type "$TUNER_TYPE" "$@"
# LoRA merge exports checkpoint-1-merged; use the resolved args rather than
# the environment, because callers may override tuner/merge options on the CLI.
checkpoint_dir=$("$python_bin" - "$repo_root" "$OUTPUT_DIR" <<'PYCODE'
import sys
sys.path.insert(0, sys.argv[1] + "/scripts/march")
from verify_checkpoint import resolve_training_checkpoint
print(resolve_training_checkpoint(sys.argv[2]))
PYCODE
)
"$python_bin" "$repo_root/scripts/march/verify_checkpoint.py" "$checkpoint_dir" \
  --load-weights --report "$OUTPUT_DIR/reload-validation.json"
