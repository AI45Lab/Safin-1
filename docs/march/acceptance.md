# Repeating the production acceptance

The CPU suite checks model-family selection, unsupported runtime/environment
rejection, cache expansion/pruning and batched beam generation, in addition to
the model/save/load and operator fixtures:

```bash
python -m pytest tests/march -q
MARCH_TEST_MEGATRON=1 PYTHONPATH="$PWD/third_party/mcore-bridge/src:$PWD" \
  python -m pytest tests/march/test_training_contract.py -q
```

When using the existing shared environment, include the vendored bridge path
in `PYTHONPATH` for training contracts. An editable install can otherwise point
to another checkout; the origin assertion intentionally rejects that mix. The
public training launchers set this path themselves.

Unknown or undownloaded model IDs require `MODEL_PROFILE=4b` or `35ba3b`. Local
Safin-1 configs select their own family and retain saved runtime. Enabled retired
architecture options fail before construction; output-detach/output-zero debug
switches are rejected before training. The cache delegates native KV/conv/state
reordering once and separately selects MARCH sidecars, including shared anchor
metadata, expansion, pruning and int32/int64 beam indices.

## Real checkpoint training and reload

Run each CPT/SFT smoke with synthetic public text on the requested GPUs. Use
new output directories, a checkpoint copy migrated with the current source,
and the [installation recipe](installation.md). The original weights are never
updated by these commands.

```bash
MODEL=/path/to/Safin-1-4B OUTPUT_DIR=/path/to/4b-cpt TRAIN_TASK=cpt \
  bash scripts/march/smoke_megatron.sh
MODEL=/path/to/Safin-1-4B OUTPUT_DIR=/path/to/4b-sft TRAIN_TASK=sft \
  bash scripts/march/smoke_megatron.sh
MODEL=/path/to/Safin-1-35B-A3B OUTPUT_DIR=/path/to/35b-sft TRAIN_TASK=sft \
  NPROC_PER_NODE=2 TENSOR_MODEL_PARALLEL_SIZE=2 SEQUENCE_PARALLEL=true TUNER_TYPE=lora \
  bash scripts/march/smoke_megatron.sh --target_modules linear_qkv linear_proj \
    --lora_rank 8 --merge_lora true
```

For the 35B CPT smoke, change `TRAIN_TASK=cpt` and choose another output directory.
The smoke checks a real optimizer step, exports checkpoint-1 (checkpoint-1-merged for merged LoRA), validates runtime
and weight files, then loads all weights through HF remote code and generates.
35B full training requires more resources than the two-GPU LoRA recipe; the
historical full-training evidence and its GPU setup remain in
[validation](validation.md). One step does not establish convergence or
optimizer/RNG-state resume behavior.

## Multi-step training and state resume

Continuous-versus-resumed comparison is optional debugging acceptance. Further
full-parameter 35B comparison is not required for this release, by project
decision. Basic CPT/SFT, checkpoint export, fresh HF load and inference checks
remain in scope. Existing completed comparison results are retained as evidence.

Use the separate bounded acceptance tool when investigating optimizer/RNG resume.
It keeps MCore model/adapter state, optimizer moments/master parameters,
scheduler and RNG, runs three optimizer steps, then starts another process from
checkpoint-2
and completes step 3. The input checkpoint is read-only; choose a new output
folder. Refresh the remote-code files on a checkpoint copy with
`python scripts/march/migrate_checkpoint.py /path/to/copy` first if it predates
the current source layout. Offline mode is mandatory and set by the tool.
The tool uses `/tmp` for child-process temporary IPC; MCore checkpoint saving
also creates Unix sockets and cannot use a long shared-storage TMPDIR. Model
weights, checkpoint output, data and persistent caches keep their explicit paths.

```bash
python scripts/march/smoke_training_resume.py \
  --model /path/to/Safin-1-4B --output /path/to/4b-cpt-resume --task cpt
python scripts/march/smoke_training_resume.py \
  --model /path/to/Safin-1-4B --output /path/to/4b-sft-resume --task sft
python scripts/march/smoke_training_resume.py \
  --model /path/to/Safin-1-35B-A3B --output /path/to/35b-sft-lora-resume \
  --task sft --tuner lora --atol 0.0001
```

The 35B example declares a BF16 weight-comparison tolerance (`atol=1e-4`,
`rtol=0`). The tool's default remains exact. The measured 35B runs have matching
load-state fingerprints but residual numerical differences across processes;
this smoke does not certify bitwise MoE training reproducibility. See the
[current resume report](validation.md#training).

For 35B CPT, use `--task cpt` and another output folder. LoRA acceptance does
not certify full-parameter 35B training. For a planned parallel run, use
`--nproc`, `--tp`, `--ep` and `--dry-run` to review the command before allocation.
Do not assume that a dry-run or an available GPU count proves the full model
fits in memory.

Before creating output or starting training, the runner checks free space on the
output filesystem. The default budget counts all periodic/final checkpoints in
both runs: about 8 times input weight bytes per full-tuning checkpoint (model,
optimizer/master state and HF export), or 1.1 times for LoRA with merged exports,
plus 25% headroom. This is an estimate, not a reservation or a guaranteed upper
bound. A measured configuration may supply `--storage-budget-gib` for its total
output budget. The selected budget/free-space report is retained in
`storage-preflight.json` and the final validation report. This budget is specific
to the optional comparison's multiple retained checkpoints; it is not a storage
requirement for ordinary CPT/SFT or the first release. Dry-run defers the actual
filesystem/model inspection. See the
[capacity/parallel preflight report](validation.md#training).

The opt-in plugin records the actual MCore save/load boundary for each rank.
Acceptance requires equal restored step/data cursor, scheduler, full RNG hash
and sampled optimizer moments/master parameters. It checks finite metrics and
that resumed logging starts at the next iteration. Every final HF tensor is
compared with the continuous control; the default tolerance is exact. Any
explicit `--atol`/`--rtol` is recorded in the report. Model type and all serialized
anchor/HSD runtime fields must remain unchanged; strict HF reload then requires
complete weights, finite logits and greedy generation. Logs, source fingerprints
and JSON reports remain in the output folder. This short fixture does not
establish convergence or throughput.

## TP/DP service and long contexts

The serving matrix starts and stops its own local servers. It checks 4B TP1/DP2,
35B TP2/DP1 and 35B TP1/DP2, with the default Graph profile, 511/512/513-token
boundaries, prompts up to 8000 tokens and eight concurrent requests. At least
two GPUs are required. It checks valid responses and finite generated logprobs;
it records generated token IDs and requires one finite logprob per output token;
it does not certify HF/vLLM numerical parity.

```bash
python scripts/march/smoke_serving_matrix.py \
  --model-4b /path/to/Safin-1-4B --model-35b-a3b /path/to/Safin-1-35B-A3B \
  --output /path/to/serving-acceptance --dry-run
```

Remove `--dry-run` to execute. Record `--native-build source` only after building
native extensions from this source in the accepted environment. Use
`--native-build reused` when reusing existing binaries; the default is unknown.
The report includes configuration, generated IDs/logprobs, response usage,
Graph dispatch counts and elapsed time. Missing Graph decode dispatch fails
the default-profile acceptance. Event JSONL files are retained with the logs. Its short
synthetic workload is not a maximum-throughput benchmark.

## Clean environment acceptance

Keep clean installation separate from an existing image regression. A complete
result needs a new virtual environment, dependency installation, a source CUDA
extension build, `pip check`, strict environment reports and the actual model
smokes above. Retain `pip freeze`, build logs and the source fingerprints with
that result. Pending checks must remain pending in the publication manifest.

If only one GPU is available, add `--single-gpu` to the serving matrix. This
checks both models at TP1/DP1 with the same boundary, concurrent-request and 8K
workload. Its result does not close the two-GPU TP/DP acceptance. The 35B LoRA
training smoke can also use `NPROC_PER_NODE=1 TENSOR_MODEL_PARALLEL_SIZE=1`.
Merged LoRA HF exports are named `checkpoint-1-merged`; the smoke chooses this
path from resolved `args.json`, including CLI overrides.


When using a preinstalled environment, verify that its matching vLLM native
extensions are available to this source checkout, including the nested
`vllm.vllm_flash_attn` extensions. An editable installation pointing at another
checkout does not provide those files merely by changing PYTHONPATH. Keep native
reuse explicit and record the binary hashes; do not label that run a source
build. See the [existing-environment report](validation.md#training).
For service runs, use a short local TMPDIR so IPC socket paths remain below the
platform limit. Checkpoints and logs can remain on shared storage.
