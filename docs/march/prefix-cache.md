# Experimental prefix cache for RL rollout

Prefix reuse can reduce repeated prompt computation when GRPO samples several
answers for the same prompt. The implementation retains the existing HSD
snapshot/restore path and adds explicit opt-in and cache invalidation. It is
experimental; the bounded production GPU results are recorded
[here](validation.md#inference). Graph remains the default profile with
prefix caching disabled unless explicitly enabled.

## Supported experimental boundary

- Standard Safin-1 4B and 35B-A3B text checkpoints.
- Eager or Graph V3 decode, synchronous scheduling, pipeline parallel size 1.
- Graph prefix restore uses preallocated conv/recurrent sidecars, fixed history
  pages and fixed page-table metadata. Restore, suffix prefill and memory-anchor
  boundary updates run eagerly; eligible subsequent decode uses FULL Graph replay.
- Native full-attention KV reuse plus HSD conv/recurrent/anchor snapshots.
- Canonical prefill partitions for both cold and cached prompts: one
  memory-anchor chunk for 4B (512 tokens in the tested checkpoints), 1024 tokens
  for MoE. Both cold and cached prefills execute one request at a time, preventing
  request-arrival timing from changing BF16 prefill batch shapes. Completed
  requests still decode together through Graph V3. The scheduler lets a full
  decode batch progress when queued prefills cannot acquire a request slot.
- No persistent memory / Safety State, LoRA, speculative decoding, quantization
  or cross-engine KV connectors. Persistent-memory support itself is retained.

The persistent-state restriction applies even when its serving mode is zero or
removed. Prompt tokens alone are not a sufficient identity for mutable memory.
The default launcher clears inherited experimental controls; unsafe unaligned
restore and recurrent-replay probes are not part of this profile.

## Serving

Start at max sequences 1 for the first GPU check. Request batches 1 and 4
have passed the recorded single-H200 regression; larger batches and TP/DP
require separate acceptance.

~~~bash
MODEL=/path/to/Safin-1-4B MAX_NUM_SEQS=1 \
MARCH_EVENT_LOG=/tmp/march-prefix-events.jsonl \
bash scripts/march/infer_vllm.sh --profile graph --experimental-prefix-cache

MODEL=/path/to/Safin-1-35B-A3B MAX_NUM_SEQS=1 TENSOR_PARALLEL_SIZE=4 \
bash scripts/march/infer_vllm.sh --profile graph --experimental-prefix-cache --dry-run
~~~

The environment equivalent is MARCH_INFERENCE_PROFILE=graph and
MARCH_EXPERIMENTAL_PREFIX_CACHE=1. Snapshot storage defaults to at most 2 GiB
per worker, divided across configured layers; change it with
--prefix-cache-max-bytes or MARCH_PREFIX_CACHE_MAX_BYTES.
Use --profile eager for the correctness reference. Fixed prefill partitions
and single-request prefills can increase cold-prompt overhead; throughput
requires a separate benchmark. With the tested 512/1024-token partitions, fresh
prefills use the segments fallback under the default 2048-token varlen threshold.
The default prefix-disabled Graph profile retains its existing prefill schedule.
Native KV and paged history storage have separate budgets. A snapshot that is too large is rejected;
evicted or unavailable snapshots cause full prefill instead of accepting an
unrestorable KV hit.

## Weight updates and reset

Only reuse a prefix while model weights and runtime configuration are unchanged.
The scheduler key contains a full text-config/dtype fingerprint and a reset
epoch. Layer snapshots also record a generation, checked before restore.
Delayed ready/eviction notifications from an earlier epoch are ignored.

Use this fork's engine APIs:

1. Stop new rollout submissions and drain active and queued requests.
2. Synchronize weights through collective_rpc("update_weights", ...) or
   collective_rpc("reload_weights", ...).
3. Resume rollout only after the update succeeds. A failed or partially copied
   update requires restoring a valid policy before generation resumes.

EngineCore rejects those weight updates while requests remain. Before copying,
it resets native KV and worker HSD snapshots/history together. Both checkpoint
and direct kernel-format loading invalidate HSD state, including load failures.
Graph-visible storage is zeroed in place. Storage addresses are checked before
and after snapshot restore, reset and the validation weight-update probe;
learned persistent-memory parameters
are not deleted or zeroed.

For a custom callable RPC, external trainer hook or direct parameter mutation,
the caller must enforce the same drained boundary and call
engine.reset_prefix_cache() successfully before generation resumes. An arbitrary
in-place tensor write cannot be detected automatically. This document does not
certify an external RL framework's weight-synchronization implementation.

For a manual reset, check its boolean result: a failed reset retains the epoch
and must not be treated as a completed policy change. The RPC
get_hsd_prefix_cache_status returns entries, bytes and generations for each worker.

## GPU acceptance runner

Build/install the vendored vLLM fork first, then use its source on PYTHONPATH:

~~~bash
PYTHONPATH="$PWD/third_party/vllm:$PWD" .venv/bin/python \
  scripts/march/check_prefix_cache.py \
  --model /path/to/Safin-1-4B --profile graph --full-logprobs \
  --output /tmp/safin1-prefix-4b.json

PYTHONPATH="$PWD/third_party/vllm:$PWD" .venv/bin/python \
  scripts/march/check_prefix_cache.py \
  --model /path/to/Safin-1-35B-A3B --profile graph --tensor-parallel-size 4 \
  --output /tmp/safin1-prefix-35b.json
~~~

The runner compares cold prefill against a warmed shared-prefix request at
511/512/513/1024/1025/2049 prompt tokens, using fixed tokens, greedy decoding and
a fixed seed. It requires actual scheduler-hit and worker-restore events,
matching generated IDs and selected-token logprobs within explicitly recorded
tolerances. --full-logprobs additionally compares the entire vocabulary
log-probability distribution. --batch-size 4 exercises concurrent rollout
requests with a padded Graph batch; --reference-report compares cold generated
IDs and selected logprobs against an earlier eager report. It changes a learned anchor embedding in memory through kernel-format
reload, checks cache invalidation, repeats the comparison, and restores the
original parameter in a finally block. Checkpoint files are never overwritten.

For opt-in pytest execution set MARCH_TEST_VLLM_PREFIX_CACHE=1 together with
MARCH_PREFIX_TEST_4B_MODEL and MARCH_PREFIX_TEST_35B_MODEL; optional
MARCH_PREFIX_TEST_4B_TP / MARCH_PREFIX_TEST_35B_TP select parallel sizes.
Then run tests/march/test_vllm_prefix_cuda.py. The pytest matrix includes both
models, both profiles and batch sizes 1 and 4.

The JSON reports software/GPU information, IDs, selected-logprob errors, restore
metadata and cache reset status. Raw events are also retained beside the report
as an .events.jsonl file, including numerical comparison failures. This is
profile-specific validation. Graph acceptance requires completed FULL replay
events after prefix hits, fixed buffer addresses and an exercised anchor-boundary
eager fallback. Full-vocabulary logprobs are compared only with --full-logprobs;
raw logits and an end-to-end GRPO job remain outside this runner.

CPU tests execute the actual snapshot, restore, scheduler, engine and transfer
methods with CPU tensors and real paged history storage. They cover cloning,
top-4 history-read equivalence, all-layer preflight, stale generations,
malformed snapshots, delayed notifications and failed weight loads. The
[production GPU regression](validation.md#inference) covers native
CUDA execution and request batches 1/4 on one H200. Eviction under pressure,
TP/DP, larger batches, abort/preemption and full RL integration remain untested.
The public training CLI continues to expose CPT/SFT/export.

## Verification record

On 2026-10-10, the full repository CPU suite completed with **174 passed,
21 subtests passed and 48 skipped**. The skipped tests require explicit
production-model or optional-backend acceptance settings; this result does not
include a CUDA run. All 17 changed Python files parsed successfully, the
focused syntax/name checks passed, and the patch passed git diff --check.

Use this repository's mcore bridge on PYTHONPATH so an editable installation
from a different checkout cannot supply the training bridge:

~~~bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/third_party/mcore-bridge/src:$PWD" \
MARCH_TEST_MEGATRON=1 MARCH_TEST_VLLM_PREFIX_CACHE=0 \
.venv/bin/python -m pytest -q tests/march
~~~

All eight model/profile/batch groups passed the recorded H200 regression
with four generated tokens per request. Batch 1 compares full-vocabulary
logprobs; batch 4 compares selected-token logprobs. Every group checks real
weight-update invalidation. Graph groups require completed FULL replay and
unchanged captured storage. See [the executed matrix and cross-profile
limits](gpu-prefix-regression-20261010.md). The opt-in pytest matrix uses the
longer 16-token runner default, which was not executed in this run. Both prefix
profiles remain experimental; default Graph remains prefix-off.
