# Repository scope

The public preparation tree is independent of the internal development tree.
Its supported paths are Safin-1 4B / 35B-A3B CPT, SFT, HF inference and vendored
vLLM Graph inference with an eager fallback. Persistent state remains optional.

```text
configs/march/       model-specific runtime templates
examples/march/      CPT/SFT smoke data
march/safin1/        HF remote code, anchor layout, history read and cache
march/training/      packed training batches and anchor collation
march/inference/     paged cache/reference/Triton dispatch and varlen prefill
march/kernels/       shared autograd kernels and licensed FlashQLA subset
march/diagnostics/   minimal precision reproduction tooling
scripts/march/      supported launchers, migration, verification and export
tests/march/        Safin-1 CPU/GPU and release contracts
docs/march/         user guides, evidence and release records
swift/              shared ms-swift framework
third_party/        pinned bridge, vLLM and evaluator source
```

`megatron` exposes `pt`, `sft` and `export`. The `swift` CLI retains `pt`, `sft`,
`infer`, `merge-lora`, `deploy`, `export` and `eval` for framework integration;
the documented MARCH launchers select the supported configurations. RLHF,
rollout, sample and UI commands are omitted from the executable public CLI.
The UI/app source and argument/API exports are removed, and the default
installation has no Gradio dependency. Deployment serves the HTTP API.
Shared upstream RL classes, other model registrations and general data/template
utilities still exist as framework code; their presence does not certify MARCH
support. They need an import/dependency audit before further deletion.

The 2026-10-08 cleanup removes unrelated root examples/tests/docs and old CI,
installation and benchmark helpers. It also removes optional opaque Graph
islands, fixed-custom-op/MoE probes, forced FlashAttention split/schedule probes
and the FP32 TP row-parallel diagnostic. Default Graph and eager profiles keep
their selected controls; normal projection, norm and reduction operations remain.
Safin-1 prefix caching requires an explicit experimental opt-in for eager or Graph V3 decode. Graph
prefix uses fixed buffers; persistent-state/prefix combinations remain rejected. See [prefix reuse](prefix-cache.md).

Further work remains inside the inference implementation: prefix-snapshot /
restore machinery is retained for optional RL reuse; broad tensor tracing and a large model module should be
separated or deleted after checking scheduler/state-lifecycle dependencies.
The shared scheduler cache and learned persistent memory are separate features
and remain required. Full vendored upstream trees are retained with their
licenses and source inventories. See [Source layout](source-layout.md).

The [source layout](source-layout.md) describes the 2026-10-13 shared-helper,
kernel and paged-runtime refactor and its verification boundaries.

## Historical research training

The [HSD-GDN 0.8B research workspace](../../research/hsd_gdn/README.md) retains the earlier HSD model and its GDN, DeltaNet, KDA and Transformer baselines. Its vendored FLA fork is selected only by the research launchers. Historical configuration/weight schemas remain isolated from Safin-1, and sanitized historical metrics are included under research/hsd_gdn/records/.
