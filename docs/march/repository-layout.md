# Repository layout

Safin-1 training, HF inference and vLLM serving share the `march` package and
the ms-swift framework.

| Directory | Purpose |
|---|---|
| `configs/march/` | Model-specific runtime templates |
| `examples/march/` | CPT/SFT example data |
| `march/safin1/` | HF remote code, anchor layout, history read and cache |
| `march/training/` | Packed batches, labels, padding and anchor collation |
| `march/inference/` | History pages, reference/Triton dispatch and varlen prefill |
| `march/kernels/` | Training operators and the licensed FlashQLA subset |
| `march/diagnostics/` | Optional precision tracing and tensor statistics |
| `scripts/march/` | Training, inference, evaluation and maintenance entrypoints |
| `tests/march/` | CPU fixtures, GPU operators and integration checks |
| `docs/march/` | User and maintainer guides |
| `swift/` | Shared ms-swift CLI, model, dataset and training framework |
| `third_party/` | Pinned mcore-bridge, vLLM, evaluator and research FLA sources |
| `research/hsd_gdn/` | Independent 0.8B models, configurations and launchers |

## Entrypoints

`megatron` exposes `pt`, `sft` and `export`. The `swift` CLI provides `pt`, `sft`,
`infer`, `merge-lora`, `deploy`, `export` and `eval`. The supported Safin-1 commands
are collected under `scripts/march/`; deployment serves an HTTP API.

The retained ms-swift registrations, data/template utilities and shared training
classes provide the framework dependencies. Third-party licenses and source
inventories are listed in [vendored components](../../third_party/VENDORED_COMPONENTS.md).

## Model and runtime components

The bridge constructs and trains Safin-1, HF exports carry the standalone model
code, and vLLM uses shared recurrence/history operators with scheduler-owned
request caches. Learned persistent state belongs to the model. Optional prefix
reuse snapshots request state; its configuration is described in the
[prefix-cache guide](prefix-cache.md).

See the [source map](source-layout.md) for module ownership and the
[research workspace](../../research/hsd_gdn/README.md) for 0.8B HSD-GDN and FLA
baselines. The research launcher selects its FLA fork and historical checkpoint
schema within its own process.
