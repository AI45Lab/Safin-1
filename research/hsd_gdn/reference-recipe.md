# Dense HSD-GDN reference training recipe

Use the dense HSD-GDN configuration with a rank-32 history-query projection
as the main 0.8B reference. The complete model and training inputs are
provided below; datasets and tokenizers are supplied locally.

## Model and training schedule

The unchanged [model configuration](configs/hsd_gdn_0p8b_readq32.json) has
845,367,540 parameter elements in the archived checkpoint: 21 layers, hidden
size 1,536, one memory anchor per 512 text tokens, one 64-dimensional route-key
head, dense history read (`hsd_read_topk=0`), null gate, and history read in
layers 1–20. Each value head uses its local query plus a rank-32 residual LoRA
projection for history reads. This LoRA is an architecture component; training
still updates all parameters. Historical checkpoint keys retain `summary_*`.

The executable [32-GPU training configuration](configs/hsd_gdn_0p8b_readq32_full32.json)
contains the optimizer, scheduler, batch and backend settings.

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
The recommended reader is `tile_online`, with strict dispatch, matching the
32-GPU recipe. `HSD_FUSED_GLOBAL_READ_BACKEND=tile_speed` explicitly selects the
historical reader for comparison. The script does not allocate GPUs. Run it
inside an allocation that owns eight GPUs, using a new output directory:

```bash
TOKENIZER=/path/to/original-32000-token-tokenizer \
DATASET='/path/to/original-corpus/data-*.arrow' \
OUTPUT_DIR=/path/to/outputs/hsd-dense-readq32-reference \
bash research/hsd_gdn/train_reference.sh
```

To inspect the exact command first, add `DRY_RUN=1`. The wrapper explicitly
retains the historical cosine WSD decay and minimum LR ratio; these must not
be lost when transferring only the model JSON.

For the original **4-node × 8-GPU** layout, see the complete
[32-GPU launch example](reference-full32.md). Its micro batch 2 and
accumulation 4 preserve the original global batch and full 11,921-step budget.
The [32-GPU training configuration](configs/hsd_gdn_0p8b_readq32_full32.json)
is read directly by `train_full32.sh`, including the optimizer/WSD arguments
and fixed `tile_online` reader backend. The launch guide includes initialization
and dataset preparation.

## Short single-GPU regression

This uses the same architecture, optimizer family and kernels, with a smaller
batch and training budget for a training/save/load check:

```bash
TOKENIZER=/path/to/original-32000-token-tokenizer \
DATASET=/path/to/local-subset.jsonl \
OUTPUT_DIR=/path/to/outputs/hsd-dense-readq32-smoke \
NPROC_PER_NODE=1 BATCH_SIZE=1 GRAD_ACC=1 \
MAX_STEPS=20 WARMUP_STEPS=2 SAVE_STEPS=10 \
bash research/hsd_gdn/train_reference.sh

MODEL=/path/to/outputs/hsd-dense-readq32-smoke/training/checkpoint-20 \
bash research/hsd_gdn/infer.sh --max-new-tokens 16
```

The short command uses two cosine decay steps and the same minimum LR ratio
0.1. Supply `LR_SCHEDULER_KWARGS` explicitly for a different decay schedule.
Use CLI overrides for training options not exposed as environment variables.
When changing `--steps` directly, also pass `--lr-scheduler-kwargs` if the
scheduler should change; the wrapper derives its default from `MAX_STEPS`.
Do not interpret a 20-step result as a repeat of the full training curve.

## Sparse comparison

`train_topk4_reference.sh` remains a separate historical sparse-training
comparison. It is not the main recommendation. Training sparse top-k4 and
applying top-4 only at inference to a dense-trained checkpoint are different
experiments. See [the sparse recipe](reference-topk4.md) for that configuration.

See the [validation scope](validation.md) for measured coverage and limits.
