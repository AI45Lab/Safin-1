# MARCH / Safin-1

[中文说明](README_CN.md) · [Source repository](https://github.com/AI45Lab/Safin-1)

MARCH extends Qwen3.5's gated delta network with **memory anchors** and HSD
historical-state reading. Safin-1 is the model family: 4B dense and 35B-A3B MoE.
This ms-swift fork provides Megatron CPT/SFT, Transformers inference and a
vendored vLLM backend. Persistent state is an optional feature.

This checkout is a **public-release preparation**. The planned code version is
`v0.1.0`. Model IDs are assigned; weight publication and the public release
tag are tracked in the [release checklist](docs/march/release.md). XTuner,
LMDeploy and MARCH RL are outside the current support scope. See the [first-release scope](docs/march/first-release-scope.md)
for the validated paths and remaining boundaries.

| Model | Model type | Layers | Default memory-anchor interval | HSD history read |
|---|---|---:|---:|---|
| Safin-1 4B | `safin1_4b` | 32 | 512 text tokens | layers 10–30, block 32768, top-k 4 |
| Safin-1 35B-A3B | `safin1_35b_a3b` | 40 | 512 text tokens | layers 10–30, block 32768, top-k 4 |

These profiles match the prepared checkpoints. Loading an existing Safin-1
checkpoint preserves its saved runtime; template overrides are explicit.
`hsd_*` remains the history-read namespace. Memory-anchor settings and weight
keys use `anchor_*`; old summary-based keys require checkpoint migration.

Use Python 3.11 in a dedicated environment. Install **this checkout**, since
the upstream `pip install ms-swift` package does not contain MARCH:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/march/install.sh hf
```

See [installation](docs/march/installation.md) for CPU tests, CUDA training and
building the repository's vLLM fork. Fresh isolated training installation,
TE/vLLM native source builds and bounded model checks have recorded acceptance.
Use the documented toolchain; these recipes do not require a private image.

Download an approved checkpoint and tokenizer to a local directory following
[weight delivery](docs/march/model-release.md). Then run:

```bash
MODEL=/path/to/Safin-1-4B PROMPT='Explain memory anchors in one sentence.' \
  bash scripts/march/infer_hf.sh

MODEL=/path/to/Safin-1-4B bash scripts/march/infer_vllm.sh --port 8000

MODEL=/path/to/Safin-1-4B DATASET=examples/march/sft.jsonl \
OUTPUT_DIR=/path/to/output MAX_LENGTH=1024 TUNER_TYPE=full \
  bash scripts/march/train_sft.sh --train_iters 1

MODEL=/path/to/Safin-1-4B DATASET=examples/march/cpt.jsonl \
OUTPUT_DIR=/path/to/output MAX_LENGTH=1024 \
  bash scripts/march/train_cpt.sh --train_iters 1
```

The [quick start](docs/march/quickstart.md) includes 35B parallelism, API requests,
dry-runs and evaluation. vLLM defaults to Graph V3, Triton/varlen and disabled
prefix caching; `MARCH_INFERENCE_PROFILE=eager` selects the complete fallback.

Both models have recorded one-step CPT/SFT, checkpoint export and fresh HF
reload checks on H200. This establishes basic operation, not convergence or
peak throughput. Separate [fresh-install](docs/march/validation.md#installation-and-packaging)
and [TP/DP API](docs/march/validation.md#inference) checks passed within
their recorded workloads. Full numerical parity, production-scale stress and
other hardware remain separate. Further full-parameter 35B continuous-versus-resumed comparison is
optional and does not block this release; existing bounded results remain in
[the training/resume report](docs/march/validation.md#training).
The 35B full/cached logits limitation is documented in
[validation](docs/march/validation.md).

The [repository scope](docs/march/repository-layout.md) distinguishes supported
MARCH paths, retained framework dependencies and remaining experimental cleanup.

| Directory | Purpose |
|---|---|
| `march/safin1/` | dense/MoE remote code, history read and cache |
| `march/kernels/` | memory-anchor utilities and training kernels |
| `swift/` | retained ms-swift CLI and training framework |
| `configs/march/` | model-specific runtime profiles |
| `scripts/march/` | training, inference, migration and release tools |
| `tests/march/` | CPU contracts, tiny fixtures and optional GPU regressions |
| `third_party/` | vendored bridge, vLLM and evaluator |

Start with the [user guide index](docs/march/README.md). See the
[architecture](docs/march/architecture.md), [configuration reference](docs/march/configuration.md),
[persistent state](docs/march/persistent-state.md), [backend limits](docs/march/backends.md)
and [contribution guide](CONTRIBUTING.md).

The code is Apache-2.0; vendored components retain their own notices and licenses.
See the [code licensing guide](docs/march/licensing.md) for retained landing-page
notices and the publication layout.
Safin-1 weight licensing is assigned Apache-2.0; dataset provenance and terms
are disclosed separately in [model delivery](docs/march/model-release.md). Preserve the
attribution to [ModelScope ms-swift](https://github.com/modelscope/ms-swift),
Qwen3.5 and the [vendored projects](third_party/VENDORED_COMPONENTS.md).

Historical HSD-GDN 0.8B / FLA baseline source, launchers and training records are available in the [optional research workspace](research/hsd_gdn/README.md).
