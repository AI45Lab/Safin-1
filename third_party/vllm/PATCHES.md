# MARCH / Safin-1 integration map

Upstream vLLM is pinned to `20cac26b197a2e9530a787a5c1dbf5163533b2ff`.
`UPSTREAM.md`, `NOTICE`, `LICENSE` and the repository dependency manifest retain
provenance and fingerprints. This is a vendored fork with local Python changes;
use the [root installation recipe](../../docs/march/installation.md).

## Model and helper ownership

| Source | Responsibility |
|---|---|
| `vllm/model_executor/models/safin1.py` | Stable Dense/MoE model registrations, parameter construction and state lifecycle; composes loading/audit/prefill/decode mixins |
| `vllm/model_executor/models/march/weights.py` | HF weight mapping, HSD key policy, TP slicing and inner/outer loading methods; no parameter construction |
| `vllm/model_executor/models/march/persistent_audit.py` | FP32 persistent-state/route-key load verification, checksums and named worker audit RPC |
| `vllm/model_executor/models/march/parallel.py` | Scoped MoE DP counts after anchor insertion; eager ranks synchronize actual local rows and restore scheduler metadata |
| `vllm/model_executor/models/march/layout.py` | Anchor expansion, text row filtering/scatter, normal and fixed-shape decode layouts |
| `vllm/model_executor/models/march/runtime.py` | Shared selectors, environment readers, valid-slot and persistent-memory history predicates |
| `vllm/model_executor/models/march/diagnostics.py` | One shared trace collector/counter owner; attention/decoder/model diagnostics mixins, optional events and passive tensor export |
| `vllm/model_executor/models/march/backends.py` | One owner for ten lazy kernel loaders, cached import failures and the optional reference convolution import |
| `vllm/model_executor/models/march/prefill.py` | Single-request capability checks and FlashQLA → varlen → segments dispatch, including strict producer errors |
| `vllm/model_executor/models/march/prefill_segments.py` | Segments reference/parity execution and process-local autotune warmup keys |
| `vllm/model_executor/models/march/prefill_batch.py` | Packed request admission, convolution/recurrence orchestration, history read, state writes and output scatter |
| `vllm/model_executor/models/march/prefill_history.py` | Batched paged history reads during prefill, append ordering and capacity fallbacks |
| `vllm/model_executor/models/march/decode.py` | Single-request cached recurrence, decode admission and shared projection/gate-norm helpers |
| `vllm/model_executor/models/march/decode_graph.py` | Graph V3 split/tensor-only decode execution with existing fixed sidecar tensors |
| `vllm/model_executor/models/march/decode_batch.py` | Ordered batched decode dispatch and explicit terminal/continuation results |
| `vllm/model_executor/models/march/decode_batch_state.py` | Typed, frozen per-forward references to batch metadata, policy, history selection and intermediate results |
| `vllm/model_executor/models/march/decode_batch_plan.py` | Admission, history candidates, cache views and original text-state restoration |
| `vllm/model_executor/models/march/decode_batch_core.py` | Projections, convolution, local GDN and batched anchor-boundary append |
| `vllm/model_executor/models/march/decode_batch_no_anchor.py` | No-anchor completion, including existing padded fast-path masking and backend fallthrough |
| `vllm/model_executor/models/march/decode_batch_reference.py` | Per-request reference/anchor handling, pending history reads, scatter and padding finalization |
| `vllm/model_executor/models/march/prefix_cache.py` | Experimental prefix snapshot capture, eviction, restore and recurrence replay; attention retains buffers/generations |
| `vllm/model_executor/models/march/state_cache.py` | Native state clearing, text-cache and static-sidecar synchronization; no parameter or tensor ownership |
| `vllm/transformers_utils/configs/safin1*.py` | Native Safin-1 config types built on the existing Qwen3.5 config classes |
| `vllm/model_executor/models/registry.py` | Four Safin-1 CausalLM/ConditionalGeneration architecture names |
| Repository `march/inference` and `march/kernels` | Shared paged reads, history/anchor states and accelerated kernels; imported by the backend rather than duplicated |

The original `models.safin1` helper and loader import paths are re-exported.
Loading, audit, diagnostic, prefill and decode mixins have no constructor or model parameters; the
attention module retains all weights, state dictionaries and cache tensors.
Prefill and decode arithmetic and state-write order remain unchanged after
extraction. Graph buffer allocation, request-slot lifecycle and custom-op registration remain
in the model/runner. Snapshot/restore execution and state-cache synchronization
are delegated to constructor-free mixins; their tensors and generations remain
owned by the attention/model/runner. Eager batched decode uses six ordered phases; its admission
phase still invokes the existing cache/restore helpers at their original point.
Shared convolution execution remains in the model because it is used by both
prefill and decode.

## Core runtime integration

- `vllm/v1/worker/gpu_model_runner.py`: request-slot mappings, release/remap,
  anchor-boundary dispatch, fixed buffers and prefix snapshot/restore calls.
- `vllm/v1/core/sched/scheduler.py`, KV cache manager/coordinator and output
  records: optional aligned prefix-prefill planning, epoch/generation ownership
  and coordinated invalidation. Prefix reuse remains explicit and experimental.
- `vllm/v1/core/march_prefix_cache.py`: CPU-safe prefix identity and unsupported
  combination checks. Mutable persistent memory cannot use prefix reuse.
- `vllm/v1/worker/gpu_worker.py`, engine/entrypoint hooks: history memory budget,
  cache reset and drained weight-update coordination.
- `vllm/model_executor/models/qwen3_next.py`: shared native layers plus optional
  MARCH inspection sidecars. Safin-1 directly depends on these native layers.
- Shared linear/norm/MoE and FLA files retain explicit batch-invariance and
  deterministic-prefill controls. The public launcher applies controls before
  kernel imports. Preserve their opt-in/shape boundaries when editing them.

The stable serving entrypoint is
[`scripts/march/infer_vllm.sh`](../../scripts/march/infer_vllm.sh). Default Graph
V3 keeps prefix caching and async scheduling disabled; eager is the complete
fallback profile. `hsd_*` configuration names remain supported MARCH runtime
metadata. Compression-specific settings are retained only as explicit rejection
checks, not active low-rank history implementations.

## Tests and subsequent extraction

CPU helper tests live in `tests/march/test_vllm_model_helpers.py` and
`tests/march/test_vllm_prefill.py`, `test_vllm_decode.py` and
`test_vllm_batch_decode.py` at the repository root.
`tests/march/test_vllm_weights.py` covers mapping, TP slices, strict first loads,
post-load mutation detection and invalidation before partial updates.
They call the extracted modules directly without constructing native model
parents. CUDA cases exercise fixed buffers and graph replay. Root prefix-cache,
launcher and checkpoint metadata tests cover the surrounding contracts. Existing
HSD request-release/slot/trace tests in `tests/v1/worker/test_gpu_model_runner.py`
remain available for native vLLM import regression.

Scheduler/worker adapters and the request/cache lifecycle retain their current
owners. The packed prefill method remains a single ordered pipeline for now;
further stage decomposition requires explicit contracts for its cache writes.
Keep new shared operators under repository `march` and retain model
registry/weight names. After any functional change, rerun the
relevant current-source GPU acceptance; dated operator timings alone do not
validate a changed serving model.

## Backend failures and regression boundaries

The optional varlen, FlashQLA, deterministic-anchor, paged-history and fused-read
loaders memoize their original import exceptions. Their callers retain the
existing fallback or strict-error policy. The history-scale-add loader retries
a missed import; required static/blocked paged reads, valid-mask copy and gate
RMSNorm propagate import errors. These policies are preserved, not unified into
a blanket exception handler.

Single-request dispatch tries FlashQLA, then varlen, then segments; an explicit
strict FlashQLA miss raises. The default Graph profile uses segments below its
2048-token fresh-prefill threshold. Regression must therefore include inputs
at/above that threshold and observe `grouped_varlen_hit`, as well as 511/512/513
anchor boundaries and actual completed Graph replays.

## Decode state contracts

- Single-request cached decode consumes the current text token, updates the
  attention-owned native SSM cache in place and returns HSD `[B,H,K,V]` snapshots
  from the native `[slots,H,V,K]` layout. Anchors follow that token; persistent
  memory replacement/history append remains in the caller.
- Graph helpers receive existing convolution/recurrent sidecars and pass them
  through unchanged to the registered operators. Fixed-buffer creation, slot
  remapping, cache release, prefix restore and anchor-boundary eager dispatch
  retain their existing owners. The helpers do not allocate replacement sidecars.
- `_forward_hsd_batched_decode_light` now lives in `march/decode_batch.py` and
  remains the attention method through inheritance. Its phase objects hold the
  original tensor/cache references. Frozen fields prevent rebinding; tensor,
  list and cache contents can still change in place. They have no checkpoint,
  parameter or request-lifecycle ownership.

### Eager batched decode phases

| Phase | Inputs and result | State writes and return contract |
|---|---|---|
| Plan | Hidden/output tensors and layout → `PreparedDecode` or `False` | Existing cache allocation/view and text-state restoration remain in order; admission is not a read-only transaction |
| Core | Prepared batch → `DecodeCore` | Original projections, convolution and local GDN; native state updates remain at their existing points |
| Boundary | Prepared/core → `DecodeBoundary` | Existing batched anchor read/state append after the current text token |
| No-anchor | Prepared/core → `True`, `False` or `None` | `True` completes output/state finalization; `False` preserves the original terminal outer-fallback result; `None` continues with the same already-updated core |
| Reference | Prepared/core/boundary → `DecodeReference` or `False` | Original per-request history/anchor and persistent-state work; explicitly returns a regenerated query, boundary lists and deferred history-finalize metadata |
| Finalize | Prepared/core/reference → boolean | Pending history read/merge, normalization/projection, scatter, padding mask and counters |

The dispatcher tests `handled is not None`, so `False` never accidentally enters
the reference continuation. On `None`, it does not repeat convolution/GDN. A
raw-query backend miss can regenerate Q in the reference stage; that Q is passed
to finalization explicitly. Padding remains part of the original admitted
fast/reference routes and keeps NULL-slot filtering and output masking.

`False` does **not** promise rollback or untouched caches: a restore can occur
in planning, and a later sidecar failure can follow the native state update.
The existing caller's fallback behavior is preserved. This batch does not
redesign retry semantics or make cache writes transactional.

The [batched decode report](../../docs/march/vllm-batch-decode-acceptance-20261016.md)
records ordered-statement equivalence, visible CPU cache comparisons and the
separate current-source GPU status. The earlier
[decode acceptance](../../docs/march/vllm-decode-acceptance-20261015.md) covers its
own frozen source. Python reorganization does not establish a throughput
improvement; per-forward phase-object overhead has not been measured.

## Memory-anchor rows and MoE data parallelism

The scheduler counts text tokens. Eager Safin-1 MoE execution inserts memory
anchors before its decoder stack; its DP gather/scatter must use those expanded
row counts. `hsd_moe_dp_metadata` exchanges actual local counts once per eager
forward, including idle ranks and ranks without new anchors. The original
forward metadata is restored even when a decoder fails. Dense/DP1 paths have
no DP metadata and perform no extra collective. Graph decode keeps its original
fixed row counts and captured collectives; inserting anchors into that path is
an error. HSD microbatching remains unsupported by the runner.

On MoE DP steps whose globally synchronized runtime mode is NONE, both the
real-request and idle dummy runner bypass cached model compilation. Their
Python ForwardContext scope must execute on every step, including anchor
boundaries with unequal remote row counts. AOT tracing can capture the count
tensor operations without preserving the runtime metadata-object rebinding.
FULL/PIECEWISE Graph decode, Dense and DP1 keep their existing compilation path.

The Graph row/descriptor defensive comparison runs only outside Dynamo. Doing
that comparison during tracing pins a marked dynamic batch to its current size.
CPU TorchDynamo regression uses one FULL/PIECEWISE graph per mode for batches
32, 8 and 4; this does not replace production-size GPU revalidation.

MoE DP Graph V3 capture additionally runs one compiled warmup per descriptor,
including when `cudagraph_num_of_warmups=0`. Ordinary NONE-mode warmups still
bypass compiled model code and cannot warm Inductor/Triton kernels. A scoped
`CUDAGraphWrapper.warmup()` bypasses capture **and replay** in all nested or
lazily created wrappers while retaining the actual Graph mode, descriptor,
padding, LoRA and attention-metadata recipe. After synchronization outside
capture, the normal capture call runs with the scope restored. Context-local
state is restored on exceptions and nested scopes. Dense, DP1 and non-Graph-V3
warmup policy remains unchanged. CPU tests execute the native runner, model
compile dispatch and Graph wrapper with a cold-autotune synchronization spy;
they establish control-flow ordering, not GPU capture success or numerics.


2026-10-20 distribution cleanup removes upstream-only `.buildkite/` and `.github/`
automation plus large PNG/JPEG/HTML documentation assets. Retained documentation
links those assets to the recorded upstream commit. Runtime Python, native build
inputs, tests and all license/notice files are unchanged by this distribution
cleanup. `DISTRIBUTION.md` records the boundary.
