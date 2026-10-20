#!/usr/bin/env bash

# Resolve the model family and runtime source for the public training launchers.
# Existing Safin-1/HSD checkpoints keep their serialized runtime by default;
# base Qwen3.5 models receive the selected MARCH profile.
march_configure_training_profile() {
  local repo_root="$1"
  local model="$2"
  local saved_type="-"
  local has_saved_runtime="0"

  "${PYTHON_BIN:-python}" - "$repo_root" <<'PYCODE' || return
import sys
sys.path.insert(0, sys.argv[1])
from march.safin1.runtime_validation import validate_training_environment
validate_training_environment(require_march=True)
PYCODE
  if [[ -f "$model/config.json" ]]; then
    local metadata
    metadata=$("${PYTHON_BIN:-python}" - "$model/config.json" "$repo_root" <<'PYCODE'
import json
import sys

sys.path.insert(0, sys.argv[2])
from march.safin1.runtime_validation import validate_safin1_training_runtime
with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
text = config.get("text_config") or config
validate_safin1_training_runtime(text)
model_type = config.get("model_type") or text.get("model_type") or "-"
required = {
    "anchor_chunk_size", "anchor_token_num", "hsd_read_block_size",
    "hsd_use_global_read", "hsd_global_read_layers", "hsd_read_topk",
    "hsd_null_read_gate", "anchor_embedding_type", "anchor_embedding_init_from",
}
if text.get("anchor_embedding_init_from") == "bos":
    required.add("anchor_bos_token_id")
print(model_type, "1" if required <= set(text) else "0", sep="\t")
PYCODE
    ) || return
    IFS=$'\t' read -r saved_type has_saved_runtime <<< "$metadata"
  fi

  local inferred_profile=""
  case "$saved_type" in
    safin1_4b|safin1_4b_text|qwen3_5|qwen3_5_text|historical_qwen3_5|historical_qwen3_5_text)
      inferred_profile="4b" ;;
    safin1_35b_a3b|safin1_35b_a3b_text|qwen3_5_moe|qwen3_5_moe_text|historical_qwen3_5_moe|historical_qwen3_5_moe_text)
      inferred_profile="35ba3b" ;;
  esac

  if [[ -z "${MODEL_PROFILE:-}" && -z "$inferred_profile" ]]; then
    echo "Cannot determine model family from local config.json: $model. Set MODEL_PROFILE=4b or MODEL_PROFILE=35ba3b, or download the model first." >&2
    return 2
  fi
  MARCH_MODEL_PROFILE="${MODEL_PROFILE:-$inferred_profile}"
  case "$MARCH_MODEL_PROFILE" in
    4b)
      MARCH_PROFILE_CONFIG="$repo_root/configs/march/safin1_4b.yaml"
      MARCH_DEFAULT_LENGTH=32768
      ;;
    35b|35ba3b)
      MARCH_MODEL_PROFILE="35ba3b"
      MARCH_PROFILE_CONFIG="$repo_root/configs/march/safin1_35b_a3b.yaml"
      MARCH_DEFAULT_LENGTH=16384
      ;;
    *)
      echo "MODEL_PROFILE must be 4b or 35ba3b" >&2
      return 2
      ;;
  esac
  if [[ -n "${MODEL_PROFILE:-}" && -n "$inferred_profile" && "$MARCH_MODEL_PROFILE" != "$inferred_profile" ]]; then
    echo "MODEL_PROFILE=$MARCH_MODEL_PROFILE conflicts with checkpoint model_type=$saved_type" >&2
    return 2
  fi
  local tp_size="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
  if [[ ! "$tp_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "TENSOR_MODEL_PARALLEL_SIZE must be a positive integer" >&2
    return 2
  fi
  local default_sequence_parallel=false
  if [[ "$MARCH_MODEL_PROFILE" == 35ba3b && "$tp_size" -gt 1 ]]; then
    default_sequence_parallel=true
  fi
  case "${SEQUENCE_PARALLEL:-$default_sequence_parallel}" in
    true|True|TRUE|1|yes|on) MARCH_RESOLVED_SEQUENCE_PARALLEL=true ;;
    false|False|FALSE|0|no|off) MARCH_RESOLVED_SEQUENCE_PARALLEL=false ;;
    *) echo "SEQUENCE_PARALLEL must be true or false" >&2; return 2 ;;
  esac
  if [[ "$MARCH_MODEL_PROFILE" == 35ba3b && "$tp_size" -gt 1 && "$MARCH_RESOLVED_SEQUENCE_PARALLEL" != true ]]; then
    echo "35B MoE training with TP>1 requires SEQUENCE_PARALLEL=true" >&2
    return 2
  fi
  if [[ "$saved_type" == safin1_* && "$has_saved_runtime" != "1" ]]; then
    echo "Safin-1 checkpoint config is missing required explicit MARCH runtime fields: $model/config.json" >&2
    return 2
  fi
  if [[ -n "${MARCH_RUNTIME_CONFIG:-}" && -n "${HSD_RUNTIME_CONFIG:-}" && "$MARCH_RUNTIME_CONFIG" != "$HSD_RUNTIME_CONFIG" ]]; then
    echo "MARCH_RUNTIME_CONFIG and HSD_RUNTIME_CONFIG specify different files" >&2
    return 2
  fi
  MARCH_RESOLVED_RUNTIME_CONFIG="${MARCH_RUNTIME_CONFIG:-${HSD_RUNTIME_CONFIG:-}}"
  if [[ -z "$MARCH_RESOLVED_RUNTIME_CONFIG" && "$has_saved_runtime" != "1" ]]; then
    MARCH_RESOLVED_RUNTIME_CONFIG="$MARCH_PROFILE_CONFIG"
  fi
  "${PYTHON_BIN:-python}" - "$repo_root" "$model" "$MARCH_RESOLVED_RUNTIME_CONFIG" <<'PYCODE' || return
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from march.safin1.runtime_validation import validate_training_environment
path = Path(sys.argv[2]) / "config.json"
config = json.loads(path.read_text()) if path.is_file() else {}
config = dict(config.get("text_config") or config)
if sys.argv[3]:
    raw = sys.argv[3]
    profile = json.loads(raw) if raw.lstrip().startswith("{") else json.loads(Path(raw).read_text())
    config.update(profile)
validate_training_environment(config=config, require_march=True)
PYCODE
}
