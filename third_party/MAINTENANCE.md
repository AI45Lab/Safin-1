# Vendored runtime maintenance

The public source keeps pinned upstream snapshots with explicit MARCH changes.
LICENSE/NOTICE/UPSTREAM records and the per-file manifest cover provenance;
`PATCHES.md` maps vLLM integration ownership. Update the manifest after a source
change and renew any source-bound GPU evidence that no longer matches.

## Supported integration scope

| Component | Public MARCH path | Other retained upstream contents |
|---|---|---|
| mcore-bridge | Safin-1 Dense/MoE CPT/SFT, ordinary LoRA and full/merged export | Generic bridges/model registrations support the ms-swift fork; their presence does not claim MARCH acceptance |
| vLLM | Safin-1 Graph/eager profiles, dense history and persistent state | Generic model/backend/worker code supplies native Qwen and scheduler dependencies |
| lm-evaluation-harness | HF/vLLM wrappers and reference `ifeval`/`mmlu_pro` tasks | The upstream task/model catalog is optional; its own extras/resources and correctness scope apply |

Retained upstream model names and task YAMLs are not MARCH experiment branches.
Do not prune them by keyword. Qwen3.5 uses Qwen3-Next and shared bridge/worker
layers. Any catalog/registration reduction needs a dependency closure and
installed import/task-loader regression. Persistent model state is retained;
experimental prefix reuse stays opt-in and incompatible with mutable state.

## Local ownership

- `mcore_bridge/model/modules/march.py`: parameters, training runtime and math.
  `march_runtime.py` owns selectors; `march_diagnostics.py` owns optional
  counters, gradient/precision traces and profiling methods. The mixin has no
  constructor and introduces no parameters or metric stores.
- `vllm/model_executor/models/safin1.py`: model registration and ownership
  of state tensors, generations and request slots. `march/prefix_cache.py`
  owns snapshot/capture/restore execution; `march/state_cache.py` owns native,
  text and static-sidecar synchronization. Arithmetic and mutation ordering
  remain in the original method bodies. `march/weights.py` owns HF mapping,
  TP slicing and constructor-free loading methods. `march/persistent_audit.py`
  owns strict load verification/checksums and the audit RPC; parameters and
  audit attributes remain on the model.
- Repository `march/inference/vllm_profile.py`: the single serving/evaluation
  profile definition, loaded before CUDA imports. Evaluation configuration is
  handed to the harness as typed JSON, preserving nested compilation/backend
  settings. The runner does not patch arbitrary upstream model constructors.
- `lm_eval/models/utils_thinking.py`: boolean thinking precedence and validation.
  `lm_eval/tasks/_nltk_resources.py`: local-only IFEval corpus validation.

The numerical parity, clean native build, TP/DP serving and full-size persistent
state gates remain separate from source organization and synthetic eval smokes.
See the root validation matrix for current evidence and limitations.

## Optional historical research

`flash-linear-attention/` and `research/hsd_gdn/src/march_research/hsd/_flashqla/`
are attributed research dependencies restored by a separate change. The research
launcher selects them explicitly; Safin-1 launchers keep their existing backend
resolution. Their manifest entries use `optional_research`. The FLA 0.5.0 source revision
is verified against both official wheels and the upstream namespace generator;
`PROVENANCE.json` retains the artifact and historical overlay hashes. See the separate
[research validation](../research/hsd_gdn/validation.md); the Safin-1 43-kernel
correctness report does not certify these research kernels.

## Distribution-only omissions

The vLLM snapshot omits upstream CI automation and large binary/generated
documentation assets. Retained documentation links images to its pinned upstream
commit. `vllm/DISTRIBUTION.md` and the dependency manifest record the omissions;
runtime/native-build source and acceptance fingerprints are unaffected.
