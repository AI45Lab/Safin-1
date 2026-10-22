---
library_name: transformers
license: apache-2.0
base_model:
- Qwen/Qwen3.5-35B-A3B
datasets:
- nvidia/Nemotron-Cascade-2-SFT-Data
tags: [march, safin1, custom_code]
---

# Safin-1 35B-A3B

Hub repository: [`AI45Research/Safin-1-35B-A3B`](https://huggingface.co/AI45Research/Safin-1-35B-A3B).
Collection: [Safin-1](https://huggingface.co/collections/AI45Research/safin-1-6a90457bfa414a1453744596).

This release provides inference weights and loading code. No formal
current-release benchmark score table is supplied. The training-data and
evaluation notes below state the known scope and remaining unknowns.

Safin-1 35B-A3B uses the MARCH architecture on a Qwen3.5 backbone (40 layers).
The model type is `safin1_35b_a3b`. Standard persistent state is disabled.

| Runtime | Prepared checkpoint value |
|---|---|
| Memory anchors | 1 per 512 text tokens; learned, BOS initialized |
| HSD read block | 32768 |
| Global-read layers | 10–30 |
| History top-k / null gate | 4 / enabled |
| Saved SFT step | 7200 |
| Weight inventory | 17 safetensors shards, 1121 tensors |

The preparation used an Intern50B-named CPT initialization followed by a
NemotronSFT30B-named SFT recipe. Recorded sequence/packing length is 32768,
global batch 128, learning rate 1e-5, minimum 1e-6, cosine schedule and 0.05
warmup ratio. Seed and data seed are 42. The matched baseline has the same
data-list entries. Naming does not establish exact realized token counts or
data redistribution rights; the companion training-data note records the known recipe.

One-step full-parameter CPT/SFT and export/fresh HF reload have been recorded on
8 GPUs, EP=8 with synthetic public data at length 1024. This does not establish
convergence, optimizer-state resume, long-context quality or throughput. See
the validation matrix for cached-logits and HF/vLLM limitations.

Load reviewed remote code with `trust_remote_code=True`. Use the actual saved
runtime and chat template, and pin code/model revisions. The standard HF/vLLM
launchers are under `scripts/march/`.

Release scope: text inference and the recorded training/runtime entrypoints.
Full-file checksums are supplied. CPT corpus provenance is undocumented,
and no formal benchmark score table or universal numerical-parity claim
is supplied for these weights.

Do not attach internal `args.json`, job logs, optimizer state or private paths.
The code license does not assign the weight or dataset license.


## Load from Hugging Face

Use the reviewed repository ID and a fixed commit revision. Install the recorded
Python 3.11 / PyTorch 2.11 / Transformers 5.5.4 environment first; the code
repository contains the complete installation recipes.

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "AI45Research/Safin-1-35B-A3B"
revision = os.environ["MODEL_REVISION"]
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain memory anchors."}],
    add_generation_prompt=True, enable_thinking=False,
    return_dict=True, return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

The pinned code and model configuration define the supported workload. HF Hub
hosting does not itself enable the MARCH vLLM backend; accelerated vLLM inference
uses the fork and launchers in the code repository. See
[AI45Lab/Safin-1](https://github.com/AI45Lab/Safin-1) for software versions,
training commands and numerical limitations. Use a fixed Hub commit revision for reproducibility.
CPT provenance is undocumented and a formal quality score table is not supplied.


## License, training data and evaluation record

The weights use **Apache-2.0**, as selected by the maintainer. See LICENSE and
NOTICE in this model repository. Training data keeps its own terms; this does
not establish permissions for the unresolved CPT corpus.

Family-level ancestry: `Qwen/Qwen3.5-35B-A3B` → Intern-S2-derived CPT
step 5961 → SFT step 7200. The upstream training commit and
CPT provenance/permission record are unavailable. The listed public dataset
metadata covers the identifiable SFT source, not the entire training corpus.
See [TRAINING_DATA.md](TRAINING_DATA.md) for the ten-file recipe and accounting limits.

See [EVALUATION.md](EVALUATION.md) for recovered historical records and their
limits. They are not a verified score table for these release payloads.
The [code validation matrix](https://github.com/AI45Lab/Safin-1/blob/main/docs/march/validation.md) records separate
technical acceptance and numerical limits. SHA256SUMS covers all files in this
model root except the checksum file itself, including every complete weight shard.
