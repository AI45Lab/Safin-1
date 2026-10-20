# Runtime configuration

`configs/march/safin1_4b.yaml` and `safin1_35b_a3b.yaml` use JSON syntax, which is
also valid YAML. The launchers and bridge currently parse these files as JSON;
keep JSON syntax when editing a profile.

| Field | Standard value | Purpose |
|---|---|---|
| `model_type` | `safin1_4b` / `safin1_35b_a3b` | selects the model family; not a directory-name heuristic |
| `anchor_chunk_size` | 512 | text chunk interval for memory anchors |
| `anchor_token_num` | 1 | anchors per boundary |
| `anchor_embedding_type` | `learned` | learned anchor embedding |
| `anchor_embedding_init_from` | `bos` | initialization when starting from a base checkpoint |
| `hsd_read_block_size` | 32768 | historical-read block; distinct from launcher sequence length |
| `hsd_global_read_layers` | `10-30` | selected global-read layers |
| `hsd_read_topk` | 4 | selected historical states |
| `hsd_null_read_gate` | true | enables the null-read gate |
| `hsd_topk_backend` | `reference` | default training read implementation |
| `hsd_persistent_memory_enabled` | false | optional learned persistent state |

The complete profile files include additional route, local-state and anchor
controls. A loaded checkpoint can contain further explicitly saved `hsd_*`,
`hist_*` and `anchor_*` fields; those values must survive training export.

For an existing Safin-1 model, the training launchers read `config.json`, select
the family and preserve saved runtime. To deliberately override it:

```bash
MODEL=/path/to/Safin-1-4B DATASET=examples/march/sft.jsonl \
OUTPUT_DIR=/path/to/output MARCH_RUNTIME_CONFIG=/path/to/runtime.json \
DRY_RUN=1 bash scripts/march/train_sft.sh
```

If a local `config.json` cannot identify the family (including an undownloaded
Hub model ID), `MODEL_PROFILE=4b` or `35ba3b` is required. The launcher never
guesses 4B. An explicit family that conflicts with a local config is rejected. Legacy
`HSD_RUNTIME_CONFIG` remains supported, but conflicting variable values fail.

Training length is `MAX_LENGTH`; serving context is `MAX_MODEL_LEN`. Neither
automatically replaces checkpoint history-read block size. Changing anchor
layout or persistent-state fields changes model behavior and needs separate
training and cache validation.

Do not add runtime aliases for the former summary-based keys. Migrate an older
checkpoint copy with `scripts/march/migrate_checkpoint.py`, then run
`scripts/march/verify_checkpoint.py`.

Only `hsd_local_state_mode=full` and `hsd_history_read_q_mode=shared_local` are
supported. Retired enabled variants fail during runtime merge/config parsing,
before model construction. Inactive historical metadata remains loadable.
Training rejects inherited `HSD_DEBUG_DETACH_TEXT_O`, `HSD_DEBUG_ZERO_TEXT_O`
and enabled retired architecture environment switches; unset them explicitly.

35B MoE training with `TENSOR_MODEL_PARALLEL_SIZE>1` requires sequence parallel.
The CPT/SFT launchers enable it by default for that combination and reject an
explicit `SEQUENCE_PARALLEL=false` before training. Direct `megatron` commands
must pass `--sequence_parallel true`. Dense and TP1 defaults remain false.


## Training configuration and checkpoint integrity

Safin-1 model types or an explicit positive anchor layout select MARCH in the
Megatron loader, training collation and padding helpers. `USE_HIST_LINEAR_ATTN`
is no longer required to select the architecture. Explicitly disabling it,
`USE_MCORE_GDN` or `SWIFT_USE_MCORE_GDN` fails before model construction.
Base Qwen3.5 without an anchor profile retains its ordinary GDN path.

Serialized HSD runtime fields are authoritative during training. A legacy
`HSD_*` environment variable (including `HIST_READ_BLOCK_SIZE`) may repeat a
matching field, but conflicting or invalid values fail at launcher/config/layer
setup and export. For example, `HSD_READ_TOPK=8` cannot override a checkpoint
with `hsd_read_topk=4`. Unset it and pass a runtime file containing the desired
value through `MARCH_RUNTIME_CONFIG` / `--hsd_runtime_config`, or use an explicit
CLI field. Kernel tuning variables without a serialized runtime field, such as
`HSD_TOPK_EXECUTION_MODE`, remain available.

Full Safin-1 imports validate the active anchor, routing, gate, normalization
and persistent-state weight keys before copying any parameter. Missing active
weights raise an error instead of initializing a replacement. Starting from a
base Qwen3.5 checkpoint with no MARCH tensors explicitly initializes new MARCH
projections from QKV; partial MARCH weights cannot use this path. Ordinary LoRA
adapters remain partial imports. Standalone projection adapters use the same
generic PEFT converter for export/import, preserving selected A/B weights and
modules_to_save metadata. Frozen gates/norms/persistent state remain in the base
checkpoint. If any raw MARCH gate/norm/state is separately trainable, unmerged
export fails and requires `--merge_lora true` instead of dropping those weights.
See [adapter export/load acceptance](validation.md#training). Enabling persistent memory on an explicitly
pre-memory model retains the documented zero initialization; a source with memory
already enabled must contain its saved state. Dedicated memory route keys still
require the initialization artifact described in [persistent state](persistent-state.md).

The vendored bridge's upstream model-download integrations are excluded from
default collection. They are developer-only and can be selected with
`MCORE_BRIDGE_RUN_UPSTREAM_TESTS=1`; they retain upstream Hub model references,
require separately prepared local assets and honor the caller's assigned GPUs.
Public offline regression uses `tests/march/` instead.

## Shared HF schema and historical fields

`march/safin1/configuration_common.py` owns the Dense/MoE MARCH defaults and
validation. Both currently default `hsd_read_block_size` and the legacy
`hist_read_block_size` alias to 32768. Loading preserves explicit saved values
for both keys; it does not synchronize or rewrite a checkpoint's historical
metadata. Current history dimensions use `hsd_` fields and anchor layout uses
`anchor_` fields. `hist_use_history_read`, `hist_disable_cache` and
`hist_replace_linear_attn` remain active HF control switches.

The old `hsd_history_read_q_lora_rank` is compatibility metadata: only
`hsd_history_read_q_mode=shared_local` is supported, and changing the old rank
does not construct a LoRA projection or change parameter dimensions. Retired
architecture flags remain readable when disabled and fail validation when
enabled. Training-only gradient/LR fields remain saved for reproducibility.
