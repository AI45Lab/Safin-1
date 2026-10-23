# Model delivery

Prepared model cards are in [model-cards](model-cards/README.md). They distinguish
known training/checkpoint facts from unresolved publication information. Final
model IDs, revisions, weight license, data review and evaluation report are
recorded in `configs/release/publication.json`.

The code's Apache-2.0 license does not assign a license to derivative weights or
training datasets. The maintainer selected Apache-2.0 for all four weights. CPT provenance is
undocumented and no formal benchmark score table is supplied; both are disclosed
under the 2026-10-18 publication decision. Standard
Safin-1 weights and matched Qwen3.5 SFT baselines are separate model repositories.
Safety State B is an optional experimental release with its own runtime and card.

Publish only the reviewed inference files:

- `config.json`, `generation_config.json` when used, tokenizer/processor assets,
  and the actual checkpoint's chat template.
- The complete Safin-1 remote-code dependency closure selected by `auto_map`.
- The safetensors index and every referenced shard, with SHA-256 checksums.
- A completed model card, applicable weight license and upstream attribution.

Exclude internal `args.json`, optimizer/RNG/trainer states, absolute-path caches,
job logs and unrelated model code. Review JSON metadata such as `_name_or_path`
for internal paths. Baseline Qwen3.5 configs retain `qwen3_5` / `qwen3_5_moe`;
do not migrate baseline model types into Safin-1.

Before delivery, refresh the loading code in an independent Safin-1 checkpoint
copy. An existing `config.json` and complete weight index do not prove the
remote-code dependency closure is current:

```bash
python scripts/march/migrate_checkpoint.py /path/to/Safin-1-copy --dry-run
python scripts/march/migrate_checkpoint.py /path/to/Safin-1-copy
python scripts/march/verify_checkpoint.py /path/to/Safin-1-copy
```

For an already migrated anchor-schema checkpoint, the dry run should report zero
renamed tensor keys and zero rewritten weight shards. Retain its saved runtime;
do not replace it with a new-training template. The current first-release closure
contains 18 Python files. Native Qwen3.5 baselines keep their original model types
and require no Safin-1/HSD Python files in their delivery copies.

The native baselines save a full Qwen3.5 conditional-generation configuration.
Use `AutoModelForImageTextToText` for that top-level config; Safin-1 exposes its
conditional-generation wrapper through the checkpoint's custom
`AutoModelForCausalLM` mapping. Do not change a baseline config/model type to
force it through the Safin-1 loading entrypoint.

The [2026-10-17 formal-checkpoint delivery check](checkpoint-delivery-20261017.json)
records both refreshed Safin-1 copies and both clean native baseline copies.
All four passed fresh-process configuration/tokenizer/processor loading and full
meta parameter-name/shape comparison against their weight headers. The formal
step-3600/7200 inputs were used, not one-step acceptance-training outputs.
This repair did not rerun numerical GPU inference or compute full weight-payload
checksums; it preserves those separate acceptance boundaries.

Before publication:

```bash
python scripts/march/verify_checkpoint.py /path/to/Safin-1-copy
python scripts/march/check_public_release.py /path/to/reviewed-model --model-files
python scripts/march/check_publication.py --scope models
```

The source scanner's default rejects model-weight artifacts. `--model-files`
permits safetensors shards in a model directory while scanning textual metadata.
It skips their binary payloads without reading multi-GB files into memory;
checkpoint header/index validation remains a separate check. The flag does not
exempt credentials, private paths or training state files. Run actual
HF loading and generation in a fresh process/cache before approving each model.

## Published checkpoints

Both Safin-1 checkpoints (4B and 35B-A3B) and both matched Qwen3.5 SFT baselines are public
in the [Safin-1 collection](https://huggingface.co/collections/AI45Research/safin-1-6a90457bfa414a1453744596).

| Model | Hugging Face repository | Revision |
|---|---|---|
| Safin-1-4B | [AI45Research/Safin-1-4B](https://huggingface.co/AI45Research/Safin-1-4B) | `b6a3d21ed3c57cda78a2c0f2b2be92c516abd6f0` |
| Safin-1-35B-A3B | [AI45Research/Safin-1-35B-A3B](https://huggingface.co/AI45Research/Safin-1-35B-A3B) | `41c218d270bfd9100ee60e557dd6216097f36357` |
| Qwen3.5-4B-Intern50B-NemotronSFT30B | [AI45Research/Qwen3.5-4B-Intern50B-NemotronSFT30B](https://huggingface.co/AI45Research/Qwen3.5-4B-Intern50B-NemotronSFT30B) | `001d23108dc17acbaf520c774f87e8c119101c6d` |
| Qwen3.5-35B-A3B-Intern50B-NemotronSFT30B | [AI45Research/Qwen3.5-35B-A3B-Intern50B-NemotronSFT30B](https://huggingface.co/AI45Research/Qwen3.5-35B-A3B-Intern50B-NemotronSFT30B) | `77a406b1a6442b3524fb2d8307f6351b5450091e` |

All four checkpoints are publicly downloadable. The table pins a revision for each model.

See also the original release metadata in [publication.json](../../configs/release/publication.json).
The [Hub publication record](hf-publication-20261023.json) records the public
file and index checks. All 35 published weight shards have Hub SHA-256 values
matching the prepared checkpoint inventory.

Download a pinned checkpoint and run HF inference:

```bash
MODEL_ID=AI45Research/Safin-1-35B-A3B
MODEL_REVISION=41c218d270bfd9100ee60e557dd6216097f36357
hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir /path/to/Safin-1-35B-A3B
MODEL=/path/to/Safin-1-35B-A3B PROMPT='Explain MARCH.' bash scripts/march/infer_hf.sh
```

Safin-1 loading uses `trust_remote_code=True`. The two native Qwen3.5 baselines
use the conditional-generation loader described above. Their download commands
use the same pattern with the model IDs and revisions in the table.

## Hugging Face upload layout

Each checkpoint has its own model repository. Its root contains `config.json`,
tokenizer/processor assets, the weight index and referenced shards, the model
card, license files and `SHA256SUMS`. Safin-1 also includes the complete
remote-code dependency closure selected by `auto_map`.

Upload one prepared model directory to its repository:

```bash
MODEL_ID=AI45Research/Safin-1-4B
: "${HF_MODEL_DIR:?Set one model directory}"
hf upload "$MODEL_ID" "$HF_MODEL_DIR" .
```

Keep local environment files and delivery receipts outside model roots. Refresh
`SHA256SUMS` whenever changing a model card or another listed file. Verify a
downloaded model directory with `sha256sum -c SHA256SUMS`.

## Prepared release materials — 2026-10-17

Each of the four HF candidates now includes LICENSE (the unchanged upstream
Qwen Apache-2.0 text), a model-specific NOTICE, TRAINING_DATA.md, EVALUATION.md
and SHA256SUMS. The [weight-license record](weight-licensing.md),
[sanitized training inventory](training-data.md) and
[historical result review](historical-evaluation.md) explain their separate scopes.
All 38 weight payloads were read for the [full SHA-256 inventory](model-weight-integrity-20261017.json).
The earlier meta-loading check did not perform these full-payload hashes; this
later material-preparation step adds them without modifying weights or configs.

Verify an uploaded/downloaded model root with `sha256sum -c SHA256SUMS`. This
reads every listed file, including weights. Model-card changes require refreshing
their checksums before upload. Integrity is separate from numerical correctness.

CPT provenance remains undocumented and a formal benchmark table is not supplied.
The 2026-10-18 maintainer decision permits weight sharing with those disclosures.
Published Hub revisions are listed in the table above. Historical metrics retain
their own protocols and limits; they are not current-release quality acceptance.
The preparation record retains its original date and scope.

## Publication decision — 2026-10-18

The maintainer chose to proceed with HF weight upload. CPT provenance is
disclosed as undocumented and a formal current-release benchmark score table
is optional. Neither unknown field is changed into a completed review or a score.
The [explicit decision](model-release-policy-20261018.json) controls those two
metadata requirements; model IDs/revisions, licenses and technical checks remain.
The [current Hub publication receipt](hf-publication-20261023.json) records
public model revisions and file checks. Release tags are created explicitly
after selecting code and weight revisions.
