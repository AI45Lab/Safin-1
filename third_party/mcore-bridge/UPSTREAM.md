# Source provenance

- Upstream repository: https://github.com/modelscope/mcore-bridge
- Upstream commit: `3591e434b7d5c3a4f1ae755794a5f83cd1b487cf`.
- Local import commit: `42e6247ae98961a464b416a4e05cdf04dafde4ad` (this is not an upstream SHA).
- Local modifications: MARCH GDN memory-anchor and history-read training, per-model runtime parsing, Safin-1 checkpoint export and registry cleanup. The Qwen3.5 bridge explicitly retains the singular `shared_expert` checkpoint prefix after model-type renaming.

The exact current vendored contents are pinned by per-file SHA-256 hashes in
`third_party/dependency-manifest.json`. The original import tree exactly matches the upstream tree
`a2c6ae3ece24168f6464991498d9cf6f6ae1ec2f` at the recorded revision;
this was independently recovered from public upstream Git metadata on 2026-10-06.

Public-scope cleanup: removed unsupported standalone backbone profiles, custom
initialization, padded-vocabulary loss and model registrations. Shared MARCH
GDN and the Safin-1 FlashQLA local-scan dispatch remain available.

Removed route-action RL capture/replay hooks; ordinary history top-k/null routing and persistent-memory parameters, reads and training/export support are retained.

Removed unreleased dense self-attention replacement, state-output injection,
Direct-A/native/split backward selection and dense Tile global-read branches.
The optimized and reference FlashQLA backward paths remain.

2026-10-13 MARCH source organization: shared HF/runtime validation, retired
inactive architecture variants and paged-inference imports. The bridge
delegates chunk GDN/profiling to the single shared implementation.
Persistent memory, request state, reference/eager and Graph paths remain.

2026-10-13 training entrypoint hardening: reject retired modes during runtime
merge and before ModelConfig initialization; keep inactive checkpoint metadata.
Remove output-detach/output-zero debug calculations and reject their environment
switches before training/model construction.

2026-10-14 shared Safin-1 schema cleanup: retain the retired independent
history-query LoRA rank in checkpoint/config metadata, but remove unused
constructor copies and its positive-rank validation. Shared-local query
projection and all parameter/compute paths are unchanged.

2026-10-15 Safin-1 training cleanup: reject non-additive history mixing and
nonzero route-weight momentum before profile merge/model construction because
neither option has a training implementation. Preserve their supported default
metadata and reject inherited retired state/query modes. Remove unreachable
independent history-query adapter initialization, conversion and sharding,
isolated/text-only projection helpers and constant retired-branch conditions.
Reject actual retired query-adapter tensors before any layer weight copies.
Retain standard LoRA, active weight names, anchor layout/document isolation,
persistent state and the Triton/FlashQLA/reference training backends.
Repair the shared Qwen3-Next HF GDN adapter's stale attention-mask helper name;
both absent and explicit masks have CPU wrapper regression coverage.

2026-10-15 training import/runtime integrity: select MARCH from Safin-1 config
or a positive anchor profile; reject conflicting architecture/runtime environment
values before construction and export. Preflight active full-checkpoint keys
before copies, while retaining explicit base-Qwen QKV initialization, ordinary
LoRA and documented new persistent-state zero initialization. Incomplete Safin
weights cannot silently reinitialize routing or gates. Upstream model-download
integrations are opt-in and no longer overwrite assigned CUDA devices.

2026-10-15 standalone adapter export: route all four active MARCH projection
weights through the generic bidirectional PEFT converter so selected-adapter
A/B tensors, modules_to_save and target-module metadata survive HF export.
Exclude frozen base/state tensors from adapter-only files; reject unrepresented
trainable raw MARCH state with an explicit merged-export requirement. Preserve
full/merged export arithmetic and existing persistent state on adapter import.

2026-10-16 maintenance: move 22 unchanged debug/profile/precision methods to
`model/modules/march_diagnostics.py` and eight unchanged selectors to
`march_runtime.py`. The attention class still owns all parameters, statistics
and training state; all 91 original functions retain identical ASTs.
