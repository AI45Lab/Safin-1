# Sanitized release evidence

The public repository retains the machine-readable receipts required by
publication checks, model delivery and licensing. They contain source hashes,
software/build scope and bounded results. Personal run diaries, per-step
training curves, diagnostic dumps and dated narrative reports are excluded.

Use the [validation summary](validation.md) for supported behavior and limits,
and [acceptance commands](acceptance.md) to repeat the checks.

| Evidence | Receipt |
|---|---|
| `checkpoint-delivery` | [checkpoint-delivery-20261017.json](checkpoint-delivery-20261017.json) |
| `hf-publication` | [hf-publication-20261018.json](hf-publication-20261018.json) |
| `historical-evaluation` | [historical-evaluation-20261017.json](historical-evaluation-20261017.json) |
| `installation-acceptance` | [installation-acceptance-20261017.json](installation-acceptance-20261017.json) |
| `license-release` | [license-release-20261017.json](license-release-20261017.json) |
| `model-release-policy` | [model-release-policy-20261018.json](model-release-policy-20261018.json) |
| `model-weight-integrity` | [model-weight-integrity-20261017.json](model-weight-integrity-20261017.json) |
| `operator-correctness` | [operator-correctness-20261015.json](operator-correctness-20261015.json) |
| `serving-acceptance` | [serving-acceptance-20261017.json](serving-acceptance-20261017.json) |
| `source-audit` | [source-audit-20261017.json](source-audit-20261017.json) |
| `training-data` | [training-data-20261017.json](training-data-20261017.json) |
| `weight-licensing` | [weight-licensing-20261017.json](weight-licensing-20261017.json) |

Each receipt refers to its measured snapshot. A source-fingerprint update alone
cannot certify new code or a broader workload. GPU counts, contexts and numerical
limits must remain attached to the actual checks.
