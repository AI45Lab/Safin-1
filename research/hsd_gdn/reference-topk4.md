# HSD-GDN top-k4 reference training recipe

This is the sparse-training comparison; use the
[dense LoRA32 recipe](reference-recipe.md) as the main reference.

## Model and training schedule

The unchanged [model configuration](configs/hsd_gdn_0p8b_topk4.json) constructs
843,892,980 parameters: 21 layers, hidden size 1,536, one memory anchor per
512 text tokens, one 64-dimensional route-key head, sparse top-k 4, null gate,
and history read in layers 1–20. Historical checkpoint keys retain `summary_*`.


| Setting | Reference value |
|---|---|
| Initialization | Random seed 42; original 32,000-token tokenizer |
| Training | BF16, full tuning, packing, DeepSpeed ZeRO-2, no activation checkpointing |
| Sequence length | 16,384 |
| Original batch layout | 32 GPUs × micro batch 2 × accumulation 4 = 256 sequences/step |
| Optimizer | AdamW; LR 4e-4; betas (0.9, 0.95); epsilon 1e-8; weight decay 0.1; grad clip 1.0 |
| Schedule | WSD; 256 warmup steps; 1,192 cosine decay steps; minimum LR ratio 0.1 |
| Budget | 11,921 optimizer steps; nominal 50,000,297,984 packed token slots |
| Checkpoints | Every 2,048 steps; public wrapper retains the latest two |

Nominal token slots include padding/packing effects and are not an audited
count of unique corpus tokens. The original full corpus and tokenizer are
external inputs; weights and data are not embedded in this repository.

## Reference entrypoint

The public wrapper defaults to **one node with 8 GPUs, micro batch 2 and
accumulation 16**. This preserves the historical global batch of 256 and
optimizer-step schedule. The distributed reduction order and hardware layout
change; numerical identity with the historical 32-GPU run is not claimed.
Dataset-loader worker counts and checkpoint retention are reduced for sharing.
The script does not allocate GPUs. Run it inside an allocation that owns eight
GPUs, using a new output directory:

```bash
TOKENIZER=/path/to/original-32000-token-tokenizer \
DATASET='/path/to/original-corpus/data-*.arrow' \
OUTPUT_DIR=/path/to/outputs/hsd-topk4-reference \
bash research/hsd_gdn/train_topk4_reference.sh
```

To inspect the exact command first, add `DRY_RUN=1`. The wrapper explicitly
retains the historical cosine WSD decay and minimum LR ratio; these must not
be lost when transferring only the model JSON.

## Short single-GPU regression

This uses the same architecture, optimizer family and kernels, with a smaller
batch and training budget for a training/save/load check:

```bash
TOKENIZER=/path/to/original-32000-token-tokenizer \
DATASET=/path/to/local-subset.jsonl \
OUTPUT_DIR=/path/to/outputs/hsd-topk4-smoke \
NPROC_PER_NODE=1 BATCH_SIZE=1 GRAD_ACC=1 \
MAX_STEPS=20 WARMUP_STEPS=2 SAVE_STEPS=10 \
bash research/hsd_gdn/train_topk4_reference.sh

MODEL=/path/to/outputs/hsd-topk4-smoke/training/checkpoint-20 \
bash research/hsd_gdn/infer.sh --max-new-tokens 16
```

The short command uses two cosine decay steps and the same minimum LR ratio
0.1. Supply `LR_SCHEDULER_KWARGS` explicitly for a different decay schedule.
Use CLI overrides for training options not exposed as environment variables.
When changing `--steps` directly, also pass `--lr-scheduler-kwargs` if the
scheduler should change; the wrapper derives its default from `MAX_STEPS`.
Do not interpret a 20-step result as a repeat of the full training curve.
