# Recovered historical evaluation records

The maintainer supplied no formal paper score table. A bounded search of the
existing result directories recovered the records in
[historical-evaluation-20261017.json](historical-evaluation-20261017.json).
They preserve metrics, effective/original sample counts, task versions/hashes,
seeds, thinking, context, TP/DP, generation limits and available software fields.
Original result hashes are retained; internal paths and prompt/sample text are not.

| Model | Distinct matching result files |
|---|---:|
| `Safin-1-4B` | 9 |
| `Safin-1-35B-A3B` | 12 |
| `Qwen3.5-4B-Intern50B-NemotronSFT30B` | 10 |
| `Qwen3.5-35B-A3B-Intern50B-NemotronSFT30B` | 14 |

Only records whose configured model contains the original run identifier and
the intended final checkpoint step are included. Byte-identical latest-file
aliases and earlier checkpoints are excluded. Every distinct matching record is
retained; no best-score selection or cross-protocol ranking is performed.

These historical results are **not a formal quality report for the current
release candidates**. They do not bind the evaluated payloads to the newly
computed release SHA-256 values. Some source/task/data revisions and dependency
versions are missing, and generation/context/backend settings vary between runs.
Missing settings remain missing; current defaults are not substituted for them.

No new benchmark or GPU inference was run for this documentation change.
Operator tests, synthetic evaluator fixtures, training smokes and these historical
quality measurements have separate meanings. Current numerical limitations remain
in [validation.md](validation.md) and [inference.md](inference.md).

To complete formal model-quality acceptance, use [evaluation.md](evaluation.md)
with the exact release weight hashes, pinned code/task/data revisions, explicit
thinking/prompt/generation settings and full dataset splits. Report sample counts,
metric definitions, seeds, software/hardware and failures; review the resulting
report before setting evaluation_report in publication.json.
