# Operator correctness release gate

Publication requires a passing, source-bound GPU correctness report. Operator
speed measurements, import checks, CPU fallbacks and a successful server startup
do not replace numerical acceptance. The supported scope is MARCH's custom
operators used by Safin-1 training and HF/vLLM inference, on the recorded GPU and
software environment. This is not certification of every upstream PyTorch,
vLLM or unused vendored FlashQLA context-parallel primitive.

**Current status — 2026-10-17:** source-bound renewal for the new 4B Graph
normalization/projection policy passes **322 tests, zero skipped**. All **43**
retained custom Triton kernels launched, and the raw report matches the current
171 source/test files. The report was copied without editing its fingerprints.
An additional 202 helper and 10 worker cases pass with zero skips. See the
[completed reproducibility follow-up](validation.md#numerical-limits).
The earlier maintenance and adapter/export reports retain their original scope.

The run used one CUDA GPU with PyTorch 2.11.0+cu129 / CUDA runtime 12.9,
Triton 3.6.0, FLA 0.5.0 and TileLang 0.1.9. All **43 retained custom Triton
kernels** have actual launch evidence associated with passing tests. The report
binds 171 source/test files; it includes admitted FlashQLA training/inference,
gradient, persistent-state and Graph replay checks. The test total includes host
dispatch and cache contracts executed in the same GPU environment. The matrix
also covers full Safin checkpoint preflight, live per-layer weight restoration,
base-Qwen initialization, partial LoRA, model-driven MARCH selection, runtime
environment conflicts and 16 standalone adapter export/load cases. Selected Swift
training/export and launcher sources are included in the source fingerprints.
The adapter cases use tiny conversion fixtures, actual CUDA metadata/TP1
transfers and fresh Dense/MoE HF/PEFT loads; they do not certify multi-rank
adapter export or production-size training.

Acceptance found and fixed a one-dimensional FLA varlen launch grid that left
some heads uninitialized, and an unsafe small-cache prefill selection for later
anchor slots. Prefill now uses selected-state reads when addressable capacity
exceeds four slots, without device-to-host metadata readback. A tighter isolated
history-contribution comparison supplements the residual-fusion check. An old
value-reconstruction helper/kernel had no caller and was removed; AST comparison
confirmed the remaining function/class bodies were unchanged by that deletion.
The completed renewal validates the extracted training runtime/diagnostics
and shared inference profile, including all existing adapter-import and
standalone export roundtrip/HF-loader cases. No failing, incomplete or stale
intermediate report is accepted for release.

The [checked-in report](operator-correctness-20261015.json) records the exact
software versions, test IDs, per-kernel launch associations and source hashes.
Verify it without Torch or a GPU:

```bash
python scripts/march/check_operator_correctness.py
```

The checker inventories every custom Triton definition under `march/` (excluding
the attributed upstream FlashQLA dependency), records actual launches by test,
and rejects unreviewed operator modules. It fingerprints the MARCH source,
bridge source, vLLM FLA source, numerical tests and acceptance runner. A source
change, missing launch, missing required case, skipped/xfail/failed case, nonzero
pytest exit or missing CUDA execution invalidates acceptance. Source changes
during the run also invalidate the report. A manually edited publication
checkbox cannot override a failed evidence check.

| Operator family | Required oracle/check |
|---|---|
| Training chunk GDN and anchor states | Independent FP32 token recurrence; text/anchor outputs, states and all six input gradients; packed and batched layouts |
| Hopper FlashQLA training | FP32 recurrence; optimized and reference backward, BF16/FP16, admitted H16/K96/V96 and H16/K128/V128 geometries |
| FlashQLA inference | Text, anchor and final states, with and without initial states |
| Dense packed global read | PyTorch reference, all differentiable input gradients, two-document isolation and null probability |
| Sparse top-k | PyTorch selected-state reference; forward and query/state/weight gradients, grouped/token, top-k 1/2/4/8/16 |
| HF top-4 prefill/decode | Independent FP32 masked top-k read; no visible history, fewer than four and more than four candidates |
| Varlen GDN prefill | Independent per-document recurrence; irregular lengths, initial state, local anchors, FLA/vLLM and deterministic KKT paths |
| Paged decode and Graph | PyTorch read oracle; ordinary/read-all/persistent memory, padding, split/fused Graph, replay after query/page/metadata changes |
| Paged prefill | PyTorch paged read; blocked/arbitrary/uniform selection and persistent state |
| Fused output processing | FP32 scale/add, mask copy and strided RMSNorm/gate oracles, including padded rows |
| Training import/configuration | Active full-checkpoint key preflight before copies; live route/gate restore, base-Qwen QKV initialization, LoRA partial imports, new/existing memory state, architecture selection and runtime drift rejection |
| Standalone adapters | FP32/BF16 safetensors roundtrips, selected A/B and modules_to_save, fresh Dense/MoE HF/PEFT loading and merge comparison; reject unsupported trainable raw-state export |
| Persistent-state lifecycle | Tiny Dense/MoE optimizer step, save/fresh remote-code reload, finite outputs, 511/512/513-token cache checks |

Numerical tolerances are declared beside each comparison in the source tests;
source fingerprints bind the evidence to those tolerances. They must not be
relaxed to conceal a failure. Unsupported shapes must fail capability checks or
return the documented fallback sentinel; they are not counted as validated
accelerated execution. Training gradient checks and inference-only forward
checks remain separate. Passing the matrix does not guarantee arbitrary shapes,
hardware or dtypes, training convergence, or full-model HF/vLLM parity.

To renew acceptance, use an existing Hopper CUDA environment with all local
dependencies available. Keep the job offline and put fixtures/caches/output
outside the checkout. No checkpoints, datasets or model downloads are needed.

```bash
export OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/third_party/vllm:$PWD/third_party/mcore-bridge/src:$PWD"
python scripts/march/check_operator_correctness.py --run \
  --output /path/to/operator-correctness.json
python scripts/march/check_operator_correctness.py \
  --evidence /path/to/operator-correctness.json
```

Only after the checker reports `ready: true`, review and copy the JSON into the
release evidence location. Retain the full private test log separately. Formal
readiness, CPU CI and package build workflows all validate that evidence before
producing a release artifact. Other publication requirements remain separate.
