# MARCH source layout

The `march` package owns model mathematics and reusable runtime components.
Megatron construction/conversion and the vLLM model integration remain in the
vendored bridge and vLLM trees.

| Directory | Responsibility |
|---|---|
| `safin1/` | Standalone HF remote code: model/config entrypoints, shared anchor layout/cache/model helpers, GDN and persistent-memory artifacts |
| `training/` | Packed training batches, anchor placeholders, padding and collation |
| `inference/` | History-page allocation/storage, PyTorch reference reads, Triton reads/updates, dispatch and vLLM varlen prefill |
| `kernels/` | Differentiable shared chunk-GDN/global-read/top-k operators and the licensed FlashQLA subset |
| `diagnostics/` | Precision trace orchestration, rank metadata and tensor statistics |

## Model and checkpoint boundary

`modeling_safin1.py` and `modeling_safin1_moe.py` retain their public class names,
model types and parameter names. Both import common layout/cache/decoder helpers;
the dense model no longer imports the MoE model implementation. GDN still shares
the underlying Qwen3.5 recurrence between dense and MoE.

`config_utils.py` records 18 remote-code files, including the shared configuration fields,
runtime validation and persistent-memory utilities. Model entrypoints declare
their relative dependencies directly so a fresh Transformers cache is complete.
HF exports do not include paged cache or vLLM varlen implementation files and do
not require importing the repository's `march.inference` or `march.training`.

Historical inactive architecture metadata remains accepted for loading existing
checkpoints. Enabling text-only MLP/projection, isolated anchor projection,
single-pass delta, anchor MLP or full-attention anchors now fails during config
validation. Only `hsd_local_state_mode=full` and
`hsd_history_read_q_mode=shared_local` are supported. Parameter names and
`hsd_*` / `anchor_*` schema namespaces remain stable. This is separate from the
already rejected pre-anchor schema.

Persistent state, dedicated route keys and artifact save/load remain supported;
normal state/history caches and prefix snapshots are separate runtime features.

## Training and inference boundary

The bridge's `march_chunk_gdn.py` is an adapter to
`march.kernels.march_chunk_gdn`, including shared profiling state. There is one
chunk-GDN implementation. The FLA state-producer launch uses a two-dimensional
grid matching its value-tile and sequence/head indices. Reference backward and
FlashQLA dispatch remain available.

Paged inference exposes its public API in `paged_history_read.py`. Execution is
split into `paged_static.py`, `paged_decode.py`, `paged_prefill.py`,
`paged_memory.py` and `paged_epilogue.py`. `cache_metadata.py` owns logical page
accounting/specification; `history_cache.py` owns device storage and metadata.
`paged_reference.py` remains the numerical oracle. Five `paged_*_triton.py`
modules own selection/read, persistent-memory read, prefill, fused decode and
epilogue kernels. `paged_triton.py` retains compatibility exports.

The initial 45-function extraction and the later kernel/dispatch moves preserve
arithmetic. Backend policy/diagnostic changes are documented separately below.
`varlen_segments.py` contains the packed inference producer. Existing helper
import paths remain thin compatibility modules; internal training consumers use
`march.training.anchor_batch`.

Training anchor batches and HF generation layouts remain distinct adapters:
packed document boundaries, labels and padding differ from cached generation.
They should share invariants/tests rather than be forced into a single collator.

## vLLM checkpoint loading and persistent-state audit

`models/safin1.py` retains the public model classes and all parameter/module
construction. Constructor-free `models/march/weights.py` mixins implement the
unchanged inner/outer loading methods, HF-to-vLLM mapping, optional-key policy
and TP slicing. `models/march/persistent_audit.py` owns exact local-shard checks,
checksums and the worker audit RPC. The audit imports the owning attention class
only when called, avoiding a module initialization cycle.

No new parameter/module owner is introduced. Original helper import paths are
re-exported by `models.safin1`. Strict first-load validation, invalidation before
partial weight updates, FP32 persistent state and route-key checks, and zero-mode
validation-before-zeroing retain their original order.

The [script index](../../scripts/march/README.md) is the entrypoint map;
[dated evidence](evidence-index.md) is separate from user guides. Script and
evidence paths remain stable for callers and source-bound reports.

## Diagnostics and acceptance

Tensor/rank statistics are independent of the training tracer. Parameter-chain
tracing requires `MEGATRON_PRECISION_TRACE_PARAM_CHAIN_REGEX` or
`MEGATRON_PRECISION_TRACE_TARGET_PARAM_REGEX`; no particular model layer is
selected implicitly. Reference implementations, fallback controls, numerical
precision modes and required Graph sidecars remain available.

The 2026-10-13 refactor passed 185 CPU tests and 21 subtests (52 optional tests
skipped), and 229 tests and 21 subtests on one H200 (8 optional tests skipped).
The GPU difference includes 44 CUDA cases: recurrence/gradients, sparse top-k,
paged cache and Graph reuse, and persistent-memory train/save/reload fixtures.
Four fixed-seed tiny model cases compared 408 tensors before/after with zero
difference, including weights, logits, loss, all parameter gradients and cached
logits. The four production-family checkpoint configs and weight indexes were
read-only audited before retiring variants; they had no enabled retired variant
or matching extra projection/MLP weights.

This batch did not rerun production-size 4B/35B Megatron training, TP/DP serving,
prefix engine acceptance or throughput. The existing production reports retain
their original scope. No original checkpoint files were changed.


## Shared schema/generation cleanup (2026-10-14)

`configuration_common.py` defines the common typed runtime fields and the
validation mixin. Both variants use the same defaults; saved values take
precedence, including legacy dimension aliases with different saved values.
`anchor_`/`hsd_` fields select current dimensions. The three historical HF
control switches remain active. Retired architecture metadata and the old
query LoRA rank remain serializable; the rank is no longer used or validated
by HF, the bridge model/config or vLLM constructors.

Two mixins in `modeling_common.py` own the six identical anchor-embedding and
generation methods. Public model/config class names, model types and parameter
names are retained. The methods keep their original AST, including normal
upstream generation delegation and cache row transformations. Both model
entrypoints explicitly declare the new shared config and atomic-file helper
imports so fresh HF remote-code caches are complete.

Compact persistent artifacts validate all matched keys/shapes, finite values,
metadata and device/dtype conversions before changing model parameters. New
sidecars record model type and a tensor checksum. Tensor-only and old sidecar
artifacts remain loadable. Each output file is replaced atomically; the checksum
rejects inconsistent file pairs from interrupted or concurrent saves.


## Backend and large-file cleanup (2026-10-14)

`gdn_cache.py` owns HF row-cache slicing/merging and recurrent-state cache adapters.
`gdn_recurrence.py` owns convolution, decode recurrence, segmented recurrence and
anchor-state production. These mixins contain 18 unchanged method ASTs; the GDN
entrypoint retains parameter construction, history read and forward orchestration.
Both model entrypoints declare these helpers directly for fresh remote-code caches.

`chunk_anchor_triton.py` owns boundary reads/scatter kernels;
`chunk_scan_triton.py` owns the external-gradient backward scan. The shared
`march_chunk_gdn.py` retains autograd/FlashQLA orchestration and profiling state,
including the bridge adapter's shared state. No kernel formula was changed.

`backend_policy.py` resolves per-call backend and fused-validation settings;
`diagnostics/history_read.py` compares outputs with local kwargs. Diagnostics no
longer switch process environment variables. Inactive schema metadata remains
serializable, while fixed unused GDN instance attributes and vacuous retired-branch
statements are removed. See [the acceptance report](validation.md#inference)
for CPU/packaging results and the pending CUDA gate.


## Capacity and dispatch cleanup

Single append/replacement validates both storage and graph metadata capacities
before changing cache state. Three dedicated ordinary-decode paths own Top4,
read-all and persistent-memory execution; opt-in selection/fallback events remain
separate from computation. The chunk reference oracle and shared optional imports
are split from the autograd/FlashQLA orchestrator; reference method ASTs and bridge
profiling ownership remain intact.

Diagnostic contexts restore nested state on exceptions and isolate concurrent
host collectors. Active runtime metadata comes from the common config fields,
including persistent state. Internal source snapshots account for untracked
modules and record file fingerprints. Latest results and the outstanding GPU gate
are in [cache/runtime acceptance](validation.md#inference).
