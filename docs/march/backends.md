# MARCH acceleration backends

| Component | Implementation | Function |
|---|---|---|
| HF | `march/safin1/` | Dense/MoE model and cached generation |
| Paged history | `march/inference/paged_history_read.py` | History storage and decode reads |
| Sparse top-k | `march/kernels/march_global_read_topk.py` | Sparse forward and backward |
| Chunk GDN | `march/kernels/march_chunk_gdn.py` | Recurrence, autograd and FlashQLA dispatch |
| Hopper kernels | `march/kernels/_flashqla/` | Licensed FlashQLA subset |
| Precision tracing | `march/diagnostics/precision_trace.py` | Optional tensor/rank diagnostics |

Training uses reference top-k unless explicitly selected with the runtime field
`hsd_topk_backend=sparse` in the runtime config or CLI; only the supported top-k
values 1/2/4/8/16 are accepted. Legacy environment values must match the runtime
config; they cannot silently override training or its saved metadata.
FlashQLA is opt-in (`hsd_train_scan_backend=flashqla_tile`) and has strict
Hopper/shape requirements. `hsd_flashqla_tile_bwd_backend` accepts only
`optimized` (default) and `reference` (correctness oracle). Dense packed global
read accepts `auto`, `reference`, or `triton`; the public Safin-1 templates
retain reference top-k. See the vendored FlashQLA `UPSTREAM.md` for provenance and details.

vLLM uses the repository fork with Graph V3 as the default. Dense Safin-1 uses
FULL CUDA graphs; MoE adds numeric-safe capture sizes `1,16,32`, norm-only batch
invariance and deterministic prefill autotune. Both use Triton, varlen and
disabled prefix cache. `MARCH_INFERENCE_PROFILE=eager` restores eager/segments
with reference history read. See [inference profiles](inference.md).

```bash
MODEL=/path/to/Safin-1-4B TENSOR_PARALLEL_SIZE=1 bash scripts/march/infer_vllm.sh
MODEL=/path/to/Safin-1-35B-A3B TENSOR_PARALLEL_SIZE=4 bash scripts/march/infer_vllm.sh
```

Tested configurations and precision notes are collected in
[validation](validation.md).

## Optional historical research backends

The 0.8B workspace uses its own HSD-GDN/FLA implementation. Its recommended
dense `hsd_gdn_readq32` recipe uses `tile_online`; older dense and sparse
comparisons have separate selectors. Those settings do not change the Safin-1
Megatron or vLLM defaults above. See the [research source/backend
map](../../research/hsd_gdn/source-layout.md).

## Explicit paged decode selection

`paged_batched_decode_history_read(..., backend="triton", validate_fused=False)`
selects policy for that call. Supported names are `auto`, `torch`, `triton` and
`fused_triton`; existing aliases remain supported. Omitting an argument uses
`HSD_PAGED_HISTORY_READ_IMPL` or `HSD_PAGED_HISTORY_VALIDATE_FUSED` respectively.
Explicit values override those defaults and are inherited by split batches.
Capabilities and fallback limits remain checked by the dispatcher; an unsupported
path returns `None` so the caller can use its existing fallback.

The persistent-memory path uses its dedicated memory-aware selector/read kernels;
`fused_triton` does not bypass memory semantics. `paged_static_decode_history_read`
is a separate fixed-shape Triton path for Graph V3 and does not use this general
selector. The fastest/default serving profile is unchanged.

Fused comparison is opt-in for eligible ordinary split-Triton decode. It calls the
same entrypoint with `backend="fused_triton", validate_fused=False`, preserving
routing/cache inputs, without modifying process environment variables. It reports
`status="compared"`, finiteness and absolute differences. This opt-in diagnostic
synchronizes to the host and runs outside CUDA graph capture.

## Optional selection/fallback diagnostics

For ordinary paged decode, pass `record_backend=True` or enable
`HSD_PAGED_HISTORY_BACKEND_EVENTS=1`. Explicit false overrides that default and
split batches inherit the policy. Events report `status=selected` with the
chosen path, or `status=fallback` with a reason such as `cuda_required`,
`routing_shape_mismatch`, `history_length_limit` or `triton_unavailable`.
Events describe the selected backend or fallback reason.

The public launcher accepts:

```bash
MODEL=/path/to/Safin-1-4B MARCH_BACKEND_EVENTS=1 MARCH_EVENT_LOG=/tmp/march-events.jsonl \
  bash scripts/march/infer_vllm.sh
```

Tracing is off by default. Static Graph decode uses its existing dispatch events;
these new per-call events cover the ordinary paged dispatcher. The three read
paths now live in `paged_decode_top4.py`, `paged_decode_readall.py` and
`paged_decode_memory.py`. See [current acceptance](validation.md#inference).

## Required correctness acceptance

Every published custom Triton kernel needs actual launch evidence in a passing
[source-bound GPU report](operator-correctness.md). Training paths additionally
check gradients; skipped/fallback execution and import-only checks are not passes.
CPU, package and formal-readiness CI reject stale/missing evidence.
