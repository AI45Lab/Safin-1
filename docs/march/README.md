# MARCH / Safin-1 user guides

Install from this source checkout, prepare a local Safin-1 checkpoint and follow
the quick start. Normal training and inference use `scripts/march/`.

| Task | Guide |
|---|---|
| Install HF, training or the vendored vLLM backend | [Installation](installation.md) |
| Run CPT/SFT, HF generation, vLLM service and evaluation | [Quick start](quickstart.md) |
| Select a model/runtime profile | [Configuration](configuration.md) |
| Select Graph or eager inference and understand limits | [Inference](inference.md), [backends](backends.md) |
| Prepare/download checkpoint files | [Model delivery](model-release.md) |
| Use optional learned persistent state | [Persistent state](persistent-state.md) |
| Evaluate with fixed task/template settings | [Evaluation](evaluation.md) |
| Understand memory anchors and HSD | [Architecture](architecture.md) |
| Inspect optional prefix reuse | [Prefix cache](prefix-cache.md) |
| Check tested configurations and compatibility | [Release scope](first-release-scope.md), [validation](validation.md) |

The optional [0.8B HSD-GDN/FLA research workspace](../../research/hsd_gdn/README.md)
has its own model types, environment selection and training recipes. Its
`tile_online` reader and historical `summary_*` schema are separate from the
Safin-1 checkpoint/backend settings.

For source changes and release tooling, use the [maintainer guide](maintenance.md).
