# Maintainer guide

User-facing installation, training and inference start in the [guide
index](README.md). This page groups implementation and release material.

| Work | Entry |
|---|---|
| Understand model/framework boundaries | [Repository layout](repository-layout.md), [source layout](source-layout.md) |
| Work on historical research code | [Research source/backend map](../../research/hsd_gdn/source-layout.md) |
| Find launchers, regression and packaging tools | [Script index](../../scripts/march/README.md) |
| Read current supported validation scope | [Validation matrix](validation.md) |
| Inspect sanitized release receipts | [Evidence index](evidence-index.md) |
| Check custom CUDA operators | [Correctness gate](operator-correctness.md), [operator measurements](validation.md#operators) |
| Run repeatable acceptance workloads | [Acceptance commands](acceptance.md) |
| Migrate an older checkpoint copy | [Naming/migration](naming.md) |
| Check licenses and vendored sources | [Licensing](licensing.md), [vendor inventory](../../third_party/VENDORED_COMPONENTS.md) |
| Prepare a source-only public distribution | [Release procedure](release.md), [source checks](public-source-validation.md) |
| Review model-card drafts | [Model-card index](model-cards/README.md) |
| Review implementation changes | [Source layout](source-layout.md), [changelog](CHANGELOG.md) |
| Measure inference workloads | [Benchmark method](inference-benchmark.md) |

Evidence JSON files and source-bound tools keep stable paths. Moving or editing
them requires checking their consumers; replacing fingerprints alone cannot
renew GPU acceptance. Ordinary user guides describe current commands, while
retained machine-readable receipts preserve their original measured scopes.

Local sharing overlays, sharing-only tools and build outputs are excluded
from public source. Public installation uses the published checkout and local
model paths supplied by the user.
