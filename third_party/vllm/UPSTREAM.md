# Upstream

This vendored copy of vLLM was converted from an embedded Git checkout into
regular repository files so it can be committed directly in this repository.

- Upstream repository: https://github.com/vllm-project/vllm.git
- Upstream commit: 20cac26b197a2e9530a787a5c1dbf5163533b2ff

Local modifications: Safin-1 dense/MoE registration, MARCH history caching and prefill/decode kernels. Public-scope cleanup removes OLMo3 model/config registration, serving guards, tool/reasoning parsers, dedicated tests and documentation. Shared OLMo2 and MARCH kernels remain. Current contents are pinned in `third_party/dependency-manifest.json`.

Removed experimental low-rank history storage, factor-read kernels and compression-specific worker budgeting/telemetry. Dense history, persistent-memory state and Graph state sidecars are retained.


2026-10-08 public-scope cleanup removes inactive opaque Graph islands,
fixed-custom-op/shared-expert/direct-MoE probes, forced FlashAttention split and
single-request scheduling, and FP32 TP row-parallel diagnostics. Standard
Graph/eager arithmetic, numeric-safe norm controls, persistent state and kernel
implementations remain. Safin-1 rejects prefix-cache opt-in; snapshot/restore
internals and precision tracing remain pending dependency/lifecycle review.

2026-10-09 retains prefix snapshot/restore for explicit experimental eager
rollout. Adds full-runtime cache fingerprints, scheduler epochs, layer
generations, stale-notification filtering, coordinated native/HSD invalidation
and drained weight-update checks. Persistent-memory/prefix and Graph/prefix
combinations remain unsupported. GPU production acceptance is pending.

2026-10-10 adds an opt-in Graph V3 prefix profile. Restore writes preallocated
state/history buffers in place and validates their identity; restore and suffix
prefill execute eagerly before FULL Graph decode. Completed-replay telemetry
supports production GPU acceptance. Mutable persistent state remains excluded
from prefix reuse, while persistent-state functionality itself is retained.

Both prefix profiles use compact state slots/KV geometry so native cache blocks
remain compatible with 512-token anchor boundaries. Grouped paged prefill
snapshots filter the already-appended current anchors from prior history before
merging. Worker acceptance uses named, JSON-safe RPC payloads, and weight-copy
paths run under inference mode for learned anchor parameters.

Dense prefix reuse now partitions cold and cached prefills at memory-anchor
boundaries, while MoE retains its 1024-token partition. Default prefix-disabled
dense scheduling is unchanged. Acceptance preserves strict numerical tolerances
rather than relaxing them for grouped-prefill partition differences.

The experimental prefix profile serializes both cold and cache-hit prefills,
independent of request arrival batches, and defers decode while an admissible
prefill remains. Full decode batches can drain when waiting requests have no
slot. Completed prefills retain concurrent Graph decode. Failure reports retain
selected distributions and actual hit/restore/replay counts for diagnosis.

2026-10-13 MARCH source organization: shared HF/runtime validation, retired
inactive architecture variants and paged-inference imports. The bridge
delegates chunk GDN/profiling to the single shared implementation.
Persistent memory, request state, reference/eager and Graph paths remain.

2026-10-14 shared Safin-1 schema cleanup: retain the retired independent
history-query LoRA rank in checkpoint/config metadata, but remove unused
constructor copies and its positive-rank validation. Shared-local query
projection and all parameter/compute paths are unchanged.

2026-10-15 Safin-1 entrypoint cleanup extracts unchanged anchor layouts,
shared selectors/state predicates and optional telemetry into
`vllm/model_executor/models/march/`. Diagnostic mixins retain one collector and
counter owner, while `models/safin1.py` re-exports existing helper names and
retains all registered model classes. Remove literal retired-query branches,
no-op statements and an unused MLP helper local. No custom kernel, checkpoint
parameter name, serving profile or cache lifecycle policy changes in this batch.
See `PATCHES.md` and the dated layout acceptance report for validation scope.

2026-10-15 prefill extraction moves single-request dispatch, segments parity,
packed prefill and paged-prefill history reads into separate MARCH modules.
Ten lazy backend loaders and their success/error caches have one owner in
`march/backends.py`; the model re-exports their import paths. Prefill mixins own
no parameters or state tensors. Arithmetic, persistent memory, cache write order,
Graph/eager defaults, request lifecycle and native kernels are unchanged.
See `PATCHES.md` and the dated prefill acceptance report.

2026-10-15 decode extraction moves eleven unchanged single-request recurrence,
projection/admission and Graph V3 execution methods into `march/decode.py` and
`march/decode_graph.py`. Model classes, custom-op registrations, parameter/cache
ownership, fixed-buffer lifecycle and batched decode routing remain in place.
All model/helper function ASTs are identical across this extraction. See
`PATCHES.md` and the dated decode acceptance report for validation boundaries.

2026-10-16 batched decode extraction replaces the 3,438-line attention method
with six ordered phases and typed per-forward state references. Ordinary
no-anchor completion, anchor-boundary append and reference/padding finalization
have explicit inputs/outputs; None continuation reuses the already-updated GDN
state, while False preserves the original caller fallback. Planning restores
and late failures are not transactional. Parameters, cache ownership, arithmetic,
write order, native kernels and serving defaults remain unchanged. See
`PATCHES.md` and the dated batched decode report for current GPU status.

2026-10-16 lifecycle maintenance: move 12 unchanged prefix snapshot/restore
methods and 11 unchanged native/text/static-sidecar methods into constructor-free
`march/prefix_cache.py` and `march/state_cache.py` mixins. The model retains
parameters, buffer allocation, slot ownership and invalidation generations; all
183 original model/helper functions retain identical ASTs.


2026-10-20 distribution cleanup removes upstream-only `.buildkite/` and `.github/`
automation plus large PNG/JPEG/HTML documentation assets. Retained documentation
links those assets to the recorded upstream commit. Runtime Python, native build
inputs, tests and all license/notice files are unchanged by this distribution
cleanup. `DISTRIBUTION.md` records the boundary.
