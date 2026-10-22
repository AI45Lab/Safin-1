# Safin-1 vLLM inference

`scripts/march/infer_vllm.sh` defaults to the MARCH Graph V3 profile. It reads
the saved `config.json` to distinguish dense and MoE Safin-1 models; checkpoint
runtime fields are not replaced by the new-training YAML profiles. The OpenAI
API model name defaults to `Safin-1` and can be changed with
`--served-model-name`.

| Setting | Safin-1 4B | Safin-1 35B-A3B |
|---|---|---|
| Decode | Graph V3 FULL, compact slots, no-anchor fast path, fixed normalization/projections | Graph V3 FULL with numeric-safe capture |
| Prefill | Triton GDN + grouped varlen; fixed autotune, KKT and anchor reduction | Triton GDN + grouped varlen, deterministic autotune |
| History cache/read | scheduler-paged / paged-batched | scheduler-paged / paged-batched |
| MoE | Not applicable | Triton; batch-invariant components `norm` |
| Capture sizes | Scheduler domain | `1,16,32`, filtered by scheduler capacity |
| Prefix cache / async scheduling | Disabled | Disabled |
| Default TP / DP / max sequences | 1 / 1 / 32 | 1 / 1 / 32 |
| Default context / dtype / memory fraction | 8192 / BF16 / 0.85 | 8192 / BF16 / 0.85 |

For 4B, the Graph profile sets
`VLLM_HSD_DETERMINISTIC_PREFILL_AUTOTUNE=1`,
`VLLM_HSD_DETERMINISTIC_VARLEN_KKT=1`, and
`VLLM_HSD_DETERMINISTIC_ANCHOR_READ=1`, together with fixed decode normalization
and GDN projections. Serving and evaluation use the same policy.

The launcher uses the repository vLLM fork. Build it as described in
[installation](installation.md). The single-GPU defaults require enough memory for the full model weights and cache.

```bash
MODEL=/path/to/Safin-1-4B bash scripts/march/infer_vllm.sh --port 8000
MODEL=/path/to/Safin-1-35B-A3B TENSOR_PARALLEL_SIZE=4 \
  bash scripts/march/infer_vllm.sh --port 8000

# Eight independent model replicas for aggregate throughput on 8 CUDA GPUs.
MODEL=/path/to/Safin-1-35B-A3B TP_SIZE=1 DP_SIZE=8 \
  bash scripts/march/infer_vllm.sh --port 8000

# Inspect the actual command and kernel controls without loading weights/CUDA.
MODEL=/path/to/Safin-1-4B DRY_RUN=1 bash scripts/march/infer_vllm.sh

# Roll back both the environment and engine settings together.
MODEL=/path/to/Safin-1-35B-A3B MARCH_INFERENCE_PROFILE=eager \
  bash scripts/march/infer_vllm.sh
```

`MODEL` can also be a Hugging Face repository ID; `MODEL_REVISION` selects the
same revision for JSON inspection and vLLM loading. Local dry-runs need only
`config.json` and Python's standard library. Remote IDs use the Hub cache and
honor `HF_HUB_OFFLINE`. Old `historical_qwen3_5*` model types require checkpoint
migration first; directory names never determine the execution profile.

The launcher accepts `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`,
`TENSOR_PARALLEL_SIZE`/`TP_SIZE`, and `DATA_PARALLEL_SIZE`/`DP_SIZE`. Ordinary
API server options such as `--host`, `--port`, and `--served-model-name` pass
through. MoE Graph max sequences must be 1, 16, or 32. Profile-owned backend,
prefix-cache and compilation options cannot be overridden independently.
An explicit experimental prefix profile supports eager and Graph V3 decode;
see [the prefix-cache guide](prefix-cache.md). Graph prefix restore uses fixed
buffers, executes restore/suffix prefill eagerly, and resumes eligible FULL Graph
decode. The default Graph profile keeps prefix caching disabled.
`--profile eager` and `--enforce-eager` are aliases for the complete rollback.

Inherited `HSD_*` / `VLLM_HSD_*` diagnostic controls are cleared before the
profile is applied. For event counters set `MARCH_EVENT_LOG=/path/to/events.jsonl`
with an existing parent directory. Events record Graph and backend dispatch. Short/heterogeneous prefill and anchor
boundaries can dispatch to eager/segments as part of the Graph profile.

## Supported scope and limits

Single-GPU 35B runs require enough memory for the full model weights and cache. TP1/DP8 starts eight full model
replicas, one per GPU. Reduce `MAX_MODEL_LEN` or `MAX_NUM_SEQS` to lower
cache memory use. Tested parallel layouts and precision notes are collected
in [validation](validation.md).
