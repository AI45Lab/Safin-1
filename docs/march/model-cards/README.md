# Model cards

These cards describe the prepared inference releases. The maintainer chose
to disclose undocumented CPT provenance and share weights without a formal
current-release benchmark table. Hub commit revisions and upload verification
are recorded in `configs/release/publication.json` and the
[HF publication receipt](../hf-publication-20261018.json).

| Card | Role |
|---|---|
| [Safin-1 4B](Safin-1-4B.md) | standard MARCH dense model |
| [Safin-1 35B-A3B](Safin-1-35B-A3B.md) | standard MARCH MoE model |
| [Qwen3.5 4B SFT baseline](Qwen3.5-4B-SFT-Baseline.md) | matched training-data comparison |
| [Qwen3.5 35B-A3B SFT baseline](Qwen3.5-35B-A3B-SFT-Baseline.md) | matched training-data comparison |

Safety State B needs a separate experimental card. Its persistent-state runtime
must not be copied into the standard model cards.

The four models are assigned to the [Safin-1 collection](https://huggingface.co/collections/AI45Research/safin-1-6a90457bfa414a1453744596);
upload and verification status are recorded in the receipt above.

All four weights use Apache-2.0. See [weight licensing](../weight-licensing.md),
[training data](../training-data.md), [historical evaluation](../historical-evaluation.md)
and the [full weight checksum record](../model-weight-integrity-20261017.json).

The relative TRAINING_DATA.md / EVALUATION.md links inside each card target
its prepared HF model root. The source-tree copies are upload templates;
use the shared training/evaluation documentation linked above while browsing this directory.
