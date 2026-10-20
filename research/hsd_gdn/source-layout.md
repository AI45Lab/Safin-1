# Research source and backend map

This workspace restores HSD-GDN 0.8B and its FLA baselines independently of
Safin-1 4B/35B. Use the [README](README.md) for local inputs and the
[32-GPU guide](reference-full32.md) for the complete dense training recipe.

## Entrypoints

| Purpose | Entry | Reader policy |
|---|---|---|
| Recommended dense architecture, short or 8-GPU training | `train_reference.sh` → `run.py train`, profile `hsd_gdn_readq32` | `tile_online`, strict dispatch by default |
| Complete distributed recipe | `train_full32.sh` → `train_recipe.py` | JSON owns training/environment settings, including `tile_online` |
| Earlier dense configuration | `run.py train --profile hsd_gdn` | Historical `tile_speed` default |
| Sparse historical comparison | `train_topk4_reference.sh` | Serialized top-k4 configuration |
| GDN / DeltaNet / KDA / Transformer baselines | `run.py train --profile …` | Matching vendored FLA model implementation |
| Initialize/load historical weights | `run.py init`, `infer.sh`, `tools/` | Local inputs; historical schema retained |

For the recommended dense profile, the call chain is the model implementation
in `src/march_research/hsd/modeling_hsd_gated_deltanet.py`, the global-read
dispatcher in `hsd_global_read.py`, online forward/autograd in
`hsd_global_read_tile.py`, and backward in `hsd_global_read_tile_online_bwd.py`.
`hsd_chunk_gdn.py` owns the recurrence/anchor-state producer, with the licensed
FlashQLA subset under `_flashqla/`. `tile_speed` remains an explicit comparison
via `HSD_FUSED_GLOBAL_READ_BACKEND=tile_speed` for `run.py`/`train_reference.sh`.
For the distributed launcher, edit a copy of the JSON recipe instead.

## Retained implementation variants

The model retains the historical HSD GDN2/KDA producers and sparse routing
modules for separate comparisons. They are outside the recommended dense recipe;
their presence does not extend the supported Safin-1 paths.

OLMo3 model profiles, reordered normalization, vocabulary z-loss and low-rank
history storage have been removed. Inactive default fields in historical HSD
checkpoints are consumed for loading; enabled retired options fail before model
allocation. Rank-32 history-query projection is an active architecture component
and remains supported. It is independent of low-rank cache compression.

The frozen reader/producer kernels keep their restoration fingerprints, including
historical geometry labels in their internal tuning tables. Those labels do not
enable an OLMo3 model profile. `SOURCE_MANIFEST.json` retains the original
restoration hashes and records model/config cleanup as a separate downstream
change; it does not claim that edited model wrappers are the original snapshot.

## Checkpoint and environment boundaries

Historical serialized `summary_*` keys describe memory anchors in this workspace.
Safin-1 uses its own `anchor_*` schema and separate model types. The research
launchers select the vendored FLA source only in their subprocess environment;
normal Safin-1 training and inference keep their own implementations.

The public workspace retains configuration and launch instructions. Personal
training curves and execution diaries are excluded; see the
[validation scope](validation.md) for supported checks.
