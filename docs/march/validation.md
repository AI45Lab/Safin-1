# Validation and supported scope

This page describes the supported workloads and their limits. Use the
[acceptance commands](acceptance.md) to repeat checks on your environment.
The [machine-readable evidence index](evidence-index.md) lists the sanitized
receipts consumed by release checks. Each receipt retains its own source hashes,
software versions, workload and outcome; these results do not certify every
hardware or model configuration.

## Training

The supported Safin-1 entrypoints are Megatron CPT and SFT for 4B dense and
35B-A3B MoE. Training exports must retain the saved `safin1_4b` or
`safin1_35b_a3b` model type and every HSD/anchor runtime field. Existing checkpoint
runtime takes precedence over the new-training templates.

H200 checks have exercised real optimizer steps, export and strict HF loading
for both families. Full-parameter 4B and LoRA 35B have also been exercised in
bounded continuous-versus-resumed comparisons. The 35B LoRA comparison uses an
explicit BF16 weight tolerance; it does not certify bitwise MoE reproducibility
or full-parameter 35B optimizer-state resume. Short synthetic checks do not
establish convergence or training throughput. See the
[training and reload commands](acceptance.md#real-checkpoint-training-and-reload).

Persistent-state training requires context parallelism 1. Packed sequences with
context parallelism, MTP with anchors, padded/nonuniform anchor layouts and
Megatron deterministic mode are not generally supported. Capability guards
must reject unsupported combinations.

## HF generation and persistent state

Dense/MoE CPU fixtures cover save/load, remote-code imports, memory-anchor
insertion, cached generation, batching, beam expansion, cache reorder/prune and
request isolation. Manual cached decoding must call
`prepare_inputs_for_generation` at every step, as `generate` does, so anchors
and text positions are prepared correctly. HF loading requires
`trust_remote_code=True`.

Persistent state remains optional. Its small-model CPU isolation and artifact
checks do not certify production-size concurrent vLLM use. See
[persistent state](persistent-state.md) for its format and limits.

## Inference

The default vLLM policy is Graph V3 with Triton GDN/varlen prefill, paged history
and prefix caching disabled. MoE uses the numeric-safe capture policy. Use
`MARCH_INFERENCE_PROFILE=eager` for the complete eager rollback; independent
overrides of profile-owned controls are rejected. See
[inference settings](inference.md).

H200 serving smoke checks cover 4B TP1/DP2, 35B TP2/DP1 and 35B TP1/DP2 on
two GPUs, 39 API responses, 511/512/513-token boundaries, prompts up to 8,000
tokens and eight submitted concurrent requests. That matrix reused native
binaries; the native source-build evidence is separate. It is not a serving
throughput/stress benchmark or a full-logit parallel-parity result.

The tested 4B fixed prefill/decode policy preserves selected token IDs and
logprobs across independent processes/caches in the recorded batch-32
workloads. It does not establish full-vocabulary Graph/eager/HF parity.

## Numerical limits

35B MoE full-forward and cached logits can differ at anchor boundaries under
strict tolerances even when greedy token IDs agree. Near-tied expert routing
can amplify numerical differences. Matching generated IDs does not establish
logprob parity. Full-vocabulary HF/vLLM, Graph/eager and arbitrary TP/DP parity
are not certified. GRPO/RL is outside the supported release scope.

Prefix reuse is experimental and disabled by default. Its within-profile
boundary and invalidation tests do not establish cross-profile parity or
arbitrary-context concurrency. See [prefix caching](prefix-cache.md).

## Operators

The source-bound MARCH correctness receipt covers 322 CUDA cases on H200 with
no skips and observed launches of all 43 custom Triton kernels. Its covered
source fingerprints must match before publication. FlashQLA acceptance applies
only to the admitted Hopper geometry and backward paths. Safin-1 operator
evidence does not cover the separate historical research implementation.

Use the [operator gate](operator-correctness.md) and reproducible
`scripts/march/benchmark_operators.py` commands for correctness and performance
work. Speed depends on workload, compilation, precision and batching; no
maximum-throughput claim is made from short regression runs.

## Installation and packaging

The tested environment includes Python 3.11, PyTorch 2.11.0+cu129,
Transformers 5.5.4, Megatron-Core 0.16.1, Transformer Engine 2.14.1,
FLA 0.5.0 and Triton 3.6.0. GPU acceptance used NVIDIA H200. These are observed
versions; follow the [installation profiles](installation.md) and check their
receipts for exact build scopes. Other hardware requires separate validation.

Fresh-environment acceptance includes native TE/vLLM source builds and
full-size one-step 4B full/35B LoRA SFT, save and strict HF reload. Individual
receipts retain separate build/training environments and source boundaries.
Packaging checks cover source/sdist, wheel construction, installation outside
the source tree and HF remote-code loading.

## Optional 0.8B research

The [HSD-GDN/FLA workspace](../../research/hsd_gdn/README.md) retains model
configurations, baseline implementations and executable training recipes.
Its [validation scope](../../research/hsd_gdn/validation.md) is separate from
Safin-1. Original per-step curves, run identifiers and personal execution
reports are not part of the public source distribution.

## Release checks

```bash
python -m pytest tests/march -q
MARCH_TEST_MEGATRON=1 PYTHONPATH="$PWD/third_party/mcore-bridge/src:$PWD" \
  python -m pytest tests/march/test_training_contract.py -q
python scripts/march/check_publication.py --scope code
python scripts/march/check_operator_correctness.py
python scripts/march/check_serving_evidence.py
python scripts/march/check_installation_evidence.py
python scripts/march/check_vendor_manifest.py --strict-provenance
```

CPU skips do not renew CUDA acceptance. A source hash update alone cannot
renew a measured GPU result. Weight uploads, published revisions, model/data
licensing and download verification have their own release requirements.
