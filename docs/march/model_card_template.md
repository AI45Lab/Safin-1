---
library_name: transformers
tags:
- march
- safin1
- custom_code
---

# Safin-1 MODEL_SIZE (MARCH) — unpublished template

Fill every field below before uploading this card or weights.

- Model ID and revision: **TBD**.
- Size: **4B dense / 35B-A3B MoE**; model_type **safin1_4b / safin1_35b_a3b**.
- Weight license and base-model attribution: **TBD; separate from the code license**.
- Training data provenance, permitted use, filtering and limitations: **TBD**.
- Exact checkpoint step, training recipe, intended use and limitations: **TBD**.
- Saved runtime: memory-anchor chunk/token count, read block size, global-read layers,
  top-k, null gate and all non-default `hsd_*` fields: **copy from actual config.json**.
- Evaluation datasets, metric versions, prompts, scores and contamination policy: **TBD**.
- Validated GPU, CUDA/software versions and known backend limitations: **TBD**.
- File inventory/checksums: config.json, tokenizer/processor assets, all remote-code
  Python files, weight index and each safetensors shard: **TBD**.

Load with `trust_remote_code=True`. Use the code release's inference scripts and
explicitly document the checkpoint's chat template. Do not publish private paths,
dataset caches, optimizer states or internal training logs.
