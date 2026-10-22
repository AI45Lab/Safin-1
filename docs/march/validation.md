# Validation and supported scope

This page collects tested configurations and compatibility notes. Repeatable
commands are in [acceptance](acceptance.md); detailed source hashes and results
are in the [evidence index](evidence-index.md).

## Training

Safin-1 4B and 35B-A3B have H200 CPT/SFT optimizer-step, checkpoint-export and
strict HF-reload checks. Bounded resume comparisons cover 4B full training and
35B LoRA; the latter uses `atol=1e-4`, `rtol=0` for BF16 weights.

Exports preserve the `safin1_4b` / `safin1_35b_a3b` model type and saved HSD/anchor
runtime. Existing checkpoint settings take precedence over new-training templates.

## HF generation and persistent state

Dense/MoE tests cover remote-code loading, save/load, anchor boundaries, cached
generation, batching, beam expansion and cache reorder/prune. Manual cached
decoding uses `prepare_inputs_for_generation` on every step. HF loading uses
`trust_remote_code=True`.

Persistent-state fixtures cover artifact roundtrips and request isolation at
batch 1/4/8/32. Configuration and examples are in
[persistent state](persistent-state.md).

## Inference

The default vLLM profile uses Graph V3, Triton GDN/varlen prefill and paged
history; MoE adds numeric-safe capture. Prefix caching is disabled. Set
`MARCH_INFERENCE_PROFILE=eager` for eager execution.

The two-H200 API matrix covers 4B TP1/DP2, 35B TP2/DP1 and 35B TP1/DP2:
39 responses, 511/512/513-token boundaries, prompts up to 8,000 tokens and eight
submitted concurrent requests. This matrix reused native binaries; native
source-build results are recorded in the installation receipt.

Independent-process tests of the 4B fixed prefill/decode policy cover matching
selected token IDs and logprobs with Graph concurrency 32.

## Numerical limits

| Area | Current condition |
|---|---|
| 35B MoE logits | Full-forward/cached logits can differ near expert-routing ties, including cases with matching greedy token IDs. |
| Cross-backend precision | Full-vocabulary HF/vLLM and Graph/eager parity is an open verification item. |
| Persistent-state training | Use context parallelism 1. |
| Other training combinations | CP with packing, anchor MTP, nonuniform anchor layouts and Megatron deterministic mode are restricted by runtime checks. |
| Persistent-state serving | Production-size vLLM concurrency is an open verification item. |
| Prefix reuse | Experimental opt-in; incompatible with mutable persistent state. |

The current release covers CPT/SFT and inference. RL, XTuner and LMDeploy are
future integration work.

## Operators

The H200 correctness receipt records **322 passed CUDA cases**, zero skips and
observed launches of **43 custom Triton kernels**. It covers the recorded MARCH
geometry and Hopper FlashQLA paths. The checker verifies the associated source
fingerprints:

```bash
python scripts/march/check_operator_correctness.py
```

Use `scripts/march/benchmark_operators.py` and the
[inference benchmark guide](inference-benchmark.md) for latency and throughput
measurements on a chosen workload.

## Installation and packaging

The tested environment uses Python 3.11, PyTorch 2.11.0+cu129,
Transformers 5.5.4, Megatron-Core 0.16.1, Transformer Engine 2.14.1,
FLA 0.5.0, Triton 3.6.0 and NVIDIA H200 GPUs.

Fresh-environment checks cover TE/vLLM native builds, one-step 4B full / 35B LoRA
SFT, save and strict HF reload. Package checks cover source/sdist, wheel
installation outside the source tree and HF remote-code loading. See
[installation](installation.md) for the build requirements and profiles.

## Optional 0.8B research

The [HSD-GDN/FLA workspace](../../research/hsd_gdn/README.md) provides model
configurations, baselines and training recipes. Its own validation results are
described [here](../../research/hsd_gdn/validation.md).

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

GPU receipts retain the source, environment, workload and tolerance used by
each run. Model publication uses its own revision and download records.
