#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?Set MODEL to a downloaded Safin-1 checkpoint}"
: "${PROMPT:?Set PROMPT for the smoke request}"
python_bin="${PYTHON_BIN:-python}"
max_new_tokens="${MAX_NEW_TOKENS:-128}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "$python_bin" - "$MODEL" "$PROMPT" "$max_new_tokens"
  printf '%s\n' "< scripts/march/infer_hf.sh:inline-python"
  exit 0
fi
"$python_bin" - "$MODEL" "$PROMPT" "$max_new_tokens" <<'PYCODE'
import sys, torch
from transformers import AutoModelForCausalLM, AutoProcessor
model_dir, prompt, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
inputs = processor(text=prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k,v in inputs.items() if hasattr(v, "to")}
with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=limit, do_sample=False)
print(processor.batch_decode(output, skip_special_tokens=True)[0])
PYCODE
