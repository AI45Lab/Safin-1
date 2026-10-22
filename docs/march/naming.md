# MARCH / Safin-1 naming and checkpoint migration

MARCH names the architecture. Safin-1 names the 4B dense and 35B-A3B MoE models,
which retain their Qwen3.5 backbone implementations and upstream attribution.

The paper's **memory anchor** is the learned boundary token that carries a
chunk's recurrent state into HSD history read. Public code shortens this to
`anchor`: configuration uses `anchor_*`, HSD-specific controls use
`hsd_anchor_*`, and parameters use names such as `anchor_embedding`,
`anchor_q_proj` and `anchor_k_proj`.

| Interface | 4B | 35B-A3B |
|---|---|---|
| model_type | `safin1_4b` | `safin1_35b_a3b` |
| text model_type | `safin1_4b_text` | `safin1_35b_a3b_text` |
| Config class | `Safin1Config` | `Safin1MoeConfig` |
| architectures | `Safin1ForConditionalGeneration` | `Safin1MoeForConditionalGeneration` |
| Config module | `configuration_safin1` | `configuration_safin1_moe` |
| Model module | `modeling_safin1` | `modeling_safin1_moe` |

The HF remote-code package is `march.safin1`. The shared training helpers are
in `march.kernels`; the Megatron layer is `MarchGatedDeltaNet` in
`mcore_bridge.model.modules.march`. vLLM registers both variants in its `safin1`
model module and provides matching dense/MoE configuration modules.

The top-level Python package is `march`:

```text
march/
├── __init__.py
├── kernels/  # shared MARCH training kernels and anchor utilities
└── safin1/   # Safin-1 dense/MoE remote-code implementation
```

The former `loglinear` package has been removed. Its nested `march` package is
now `march.kernels`, and its model package is now `march.safin1`. The checkpoint
migration utility translates legacy module names in metadata. Remote-code
`auto_map` entries remain relative to the checkpoint's own files.

## Compatibility

The HSD architecture name and the `hsd_*` / `hist_*` namespaces are retained.
Anchor-related interfaces were changed as one schema migration:

| Surface | Current form |
|---|---|
| Layout config | `anchor_chunk_size`, `anchor_token_num`, `anchor_*` |
| HSD config and CLI | `hsd_anchor_*` |
| Environment variables | `HSD_ANCHOR_*`, `VLLM_HSD_*ANCHOR*` |
| Python and cache metadata | `anchor_mask`, `anchor_indices`, `anchor_*` |
| State dict | `anchor_embedding`, `anchor_q_proj`, `anchor_k_proj` |

The previous summary-based names are unsupported and have no fallback aliases.
Tensor shapes, dtypes, offsets and numeric payloads are unchanged by the schema
migration. Upstream Qwen3.5 imports, tokenizer/processor classes and licensing
remain unchanged.

`MARCH_TRITON_CACHE_BASE` is the public cache-root variable. The former
`LOGLINEAR_TRITON_CACHE_BASE` remains only as a compatibility fallback.

Old Python import paths and `HistoricalQwen3_5*` class names are replaced in
this fork. Existing remote-code checkpoints contain their own Python files;
use the migration command on a checkpoint copy to update it for this fork.
For local AutoConfig/AutoModel construction call
`from march.safin1 import register_safin1_models; register_safin1_models()`.
Published models load with `trust_remote_code=True` through `auto_map`.

## Migrating a checkpoint copy

```bash
python scripts/march/migrate_checkpoint.py /path/to/checkpoint-copy --dry-run
python scripts/march/migrate_checkpoint.py /path/to/checkpoint-copy
```

The command updates `config.json`, optional `args.json`, the safetensors index,
matching parameter names in safetensors headers, and the complete local
remote-code file set. It removes superseded remote-code filenames. Affected
weight shards are copied to temporary files and validated before replacement,
so hardlinked source checkpoints are not modified. Tensor payload bytes,
shapes, dtypes, tokenizer files and runtime values are preserved. The filesystem
must have free space at least equal to the affected shards during migration.
Repeating the command is safe. The supplied path must be the intended copy: the
command updates that path in place.

The standalone experimental GatedDeltaNet model, external-state injection tools,
prompt-baseline helper and obsolete seed-model initializer were removed from
`march`. Optional persistent-memory state, dedicated route keys and Graph
runtime state sidecars remain in the shared HF/Megatron/vLLM implementations.
Route-action RL and low-rank history storage have been removed; normal history
routing and dense history storage remain. This naming change does not claim a
complete experiment-code audit.

## Validation

Run CPU metadata, hardlink isolation, remote-code dependency, vLLM registration,
Megatron checkpoint-finalizer and launcher checks with:

```bash
python -m unittest discover -s tests/march -p 'test_checkpoint_metadata.py' -v
```

Full model loading and GPU training/inference require the runtime dependencies
and hardware. CPU metadata checks do not establish logits parity or GPU support.

With torch and Transformers installed, run the CPU remote-code loading,
local model registration and anchor-layout boundary tests with:

```bash
HF_HUB_OFFLINE=1 HF_MODULES_CACHE=/tmp/safin1-hf-modules \
python -m unittest discover -s tests/march -p 'test_runtime_loading.py' -v
```

Validation on 2026-09-11 first used Python 3.11.15, PyTorch 2.11.0+cu129 and
Transformers 5.5.4 on a CPU node. On 2026-09-11 to 2026-09-12, GPU runs loaded
both production checkpoints, completed CPT/SFT optimizer steps and HF reloads,
and generated through the default vLLM Graph profile. Source metadata and all
original training shards remained unchanged. The prepared copies use only
anchor config/index/header names; 18 affected shards were detached from their
hardlinked originals. Tensor counts, file sizes, offsets and saved runtime
values were preserved. See the [post-migration GPU record](validation.md#inference).

The local Transformers loader required direct imports for the model entrypoints'
transitive dependencies; those declarations are now included and covered by the
fresh-cache loading tests. Package discovery includes `march.kernels` and
`march.safin1`.
