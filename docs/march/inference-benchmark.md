# Batch inference correctness and timing

Use this guide to measure your own workload. The
[validation summary](validation.md#inference) describes the supported
scope and numerical limits; historical diagnostic dumps are not
distributed with the public source.

Use `scripts/march/benchmark_inference.py` with a local Safin-1 checkpoint in an
existing environment. Each process uses one GPU, TP1/DP1 and one profile. Keep
Graph and eager runs in separate processes with the same checkpoint and settings.
The default engine capacity is 32; actual submitted batches are 1, 4, 8 and 32.
This preserves the MoE Graph capacity contract while testing intermediate loads.

```bash
export OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/third_party/vllm:$PWD"
python scripts/march/benchmark_inference.py \
  --model /path/to/Safin-1 --profile graph --native-build reused \
  --output /path/to/new-output/graph.json --dry-run
```

Remove `--dry-run` after reviewing the plan. Run again with `--profile eager`
and a new output path. Source-built extensions can be recorded with
`--native-build source` after completing that build; the argument records the
operator's declared provenance and does not independently prove a source build.

The default workload uses synthetic token prompts at 511/512/513/1025/2048/2049/
4096/8000 tokens and a maximum of 64 generated tokens. Requests have distinct
content and heterogeneous output limits. Each batch has a uniform warmup,
measured batch A, unrelated batch B, then batch A again. The report retains
per-request token IDs, selected-token logprobs and the A-before/after-B comparison
at `atol=1e-3, rtol=1e-3`. Changed IDs and failed numerical comparisons remain
explicit and make the result conditional. Invalid outputs, live-slot collisions,
missing Graph execution or mutated persistent parameters fail execution.

The in-process probe records:

- Engine TTFT using `first_token_latency`; decode duration uses the two monotonic
  token timestamps. It never subtracts a wall-clock arrival from a monotonic time.
- Synchronized batch wall time, aggregate generated tokens/second and per-request
  decode rates. Initialization and the per-size warmup are outside measured waves.
- Absolute and incremental peak PyTorch allocated memory, plus peak reserved
  memory. Resident weights/cache and allocator reservations remain distinguishable.
- Actual completed CUDA Graph replay calls, model dispatch counters, ordered
  compact-slot allocation/release and observed peak Graph request count.
- Persistent-state and route-key checksums before/after inference when those
  parameters exist. Standard checkpoints report that persistent state is disabled.
- Source fingerprints and local checkpoint config/index hashes, shard sizes and
  modification times. Full checkpoint hashes are not recomputed by this tool.

Slot releases can be deferred until the next scheduler iteration. The audit
permits the final deferred slots, checks ownership whenever a slot is allocated,
and requires actual slot reuse for the Graph path. A requested batch size alone
does not prove Graph coverage: the report separately records whether the largest
requested batch was observed in Graph dispatch.

These are bounded offline-engine measurements with diagnostic telemetry enabled.
They do not measure HTTP transport, server DP/TP scale, production prompt quality,
maximum throughput or independent persistent-state checkpoints. Use the
[serving matrix](acceptance.md#tpdp-service-and-long-contexts) for API/parallel
acceptance and [persistent-state guidance](persistent-state.md) for that feature.


## Persistent-state reference coverage

The [persistent-state isolation regression](persistent-state.md#并发与缓存隔离验证2026-10-16)
passes eight CPU cases: Dense/MoE tiny fixtures at batch 1/4/8/32, nonzero learned
state and dedicated keys, interleaved unrelated requests, cache duplication,
reordering and pruning. Each continuation is checked against independent
full-forward rows; persistent parameters remain exactly unchanged. This coverage
is separate from full-size vLLM Graph/eager timing and GPU concurrency acceptance.
