---
library_name: transformers
license: apache-2.0
base_model:
- Qwen/Qwen3.5-35B-A3B
datasets:
- nvidia/Nemotron-Cascade-2-SFT-Data
tags: [qwen3_5, sft, baseline]
---

# Qwen3.5 35B-A3B SFT baseline

Package name: `Qwen3.5-35B-A3B-Intern50B-NemotronSFT30B`.
Hub repository: [`AI45Research/Qwen3.5-35B-A3B-Intern50B-NemotronSFT30B`](https://huggingface.co/AI45Research/Qwen3.5-35B-A3B-Intern50B-NemotronSFT30B).
Collection: [Safin-1](https://huggingface.co/collections/AI45Research/safin-1-6a90457bfa414a1453744596).

This release provides matched-baseline inference weights. No formal
current-release benchmark score table is supplied; training-data provenance
and evaluation limits are disclosed below.

This is the matched Qwen3.5 baseline for Safin-1 35B-A3B, not a MARCH model.
Its model type remains `qwen3_5_moe`. Recorded inference weights contain
16 shards and 1026 tensor keys. Do not ship unrelated historical/Safin-1 remote code.

Recorded SFT step: 7200; global batch: 128; sequence/packing length: 32768.
The ten dataset-list entries, seeds 42, learning rate 1e-5 / minimum 1e-6,
cosine schedule and 0.05 warmup match the corresponding MARCH training recipe.
Both recipes represent 921600 packed samples by step×global-batch count;
30,198,988,800 tokens is an upper bound at full packing, not a measured token count.
Initialization points to the Intern50B-named CPT checkpoint at step 5961.
An inherited iteration metadata file is not the final SFT step.

The weight license and Qwen family attribution are recorded below.
No formal current-release quality score table is supplied. Config/index/header inspection is
not generation-quality or GPU correctness validation. Keep this baseline's
validation results separate from the Safin-1 training smoke reports.

Publish only inference assets, complete model card and applicable notices.
Exclude internal training arguments, paths and logs. The code repository's
license does not grant rights to these weights or datasets.


## Load from Hugging Face

Use the reviewed repository ID and a fixed commit revision. Install the recorded
Python 3.11 / PyTorch 2.11 / Transformers 5.5.4 environment first; the code
repository contains the complete installation recipes.

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

model_id = "AI45Research/Qwen3.5-35B-A3B-Intern50B-NemotronSFT30B"
revision = os.environ["MODEL_REVISION"]
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, revision=revision, dtype=torch.bfloat16, device_map="auto"
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
