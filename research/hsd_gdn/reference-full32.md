# Train MARCH 0.8B on 32 GPUs

Use the [model configuration](configs/hsd_gdn_0p8b_readq32.json) for the
21-layer dense research model and the [training configuration](configs/hsd_gdn_0p8b_readq32_full32.json)
for the optimizer, batch layout and runtime settings. Safin-1 4B/35B training
uses the separate entrypoints in `scripts/march/`.

| Setting | Value |
|---|---|
| Model | 21 layers, hidden size 1536, 6 value heads |
| Memory anchors | One per 512 text tokens |
| History read | Dense (`hsd_read_topk=0`), layers 1–20, per-head query LoRA rank 32 |
| Router Q/K L2Norm | Disabled |
| Precision / optimizer sharding | BF16 full-parameter training, DeepSpeed ZeRO-2 |
| GPUs | 4 nodes × 8 Hopper GPUs |
| Global batch | 32 × micro batch 2 × accumulation 4 = 256 |
| Sequence length | 16384 |
| Training budget | 11921 optimizer steps |
| Optimizer | Fused AdamW, LR 4e-4, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1 |
| Schedule | WSD, 256 warmup steps, 1192 cosine decay steps, minimum LR ratio 0.1 |
| Reader backend | `tile_online` |
| Gradient clipping | 1.0 |
| Saving | Every 2048 steps, retain two checkpoints |

## Prepare the model and data

Use a CUDA training environment with the dependencies in
[requirements.txt](requirements.txt). This backend targets Hopper GPUs; the
training configuration lists the software versions and GPU used for validation.
Run the commands from the repository root. Model, data and output directories
must be accessible on every node.

Prepare a local tokenizer with exactly 32000 tokens and initialize the model
once, before starting the distributed job:

```bash
export TOKENIZER=/path/to/local-32000-token-tokenizer
export MODEL=/path/to/new-model-initialization
export DATASET=/path/to/train.jsonl
export OUTPUT_DIR=/path/to/new-training-output

python research/hsd_gdn/run.py init \
  --profile hsd_gdn_readq32 \
  --tokenizer "$TOKENIZER" \
  --output "$MODEL" \
  --seed 42
```

The dataset can contain JSONL rows such as `{"text": "A training document."}`.
A local Arrow file/directory or quoted Arrow glob is selected through the
research dataset plugin.
For a pretokenized dataset, unset `DATASET` and set
`CACHED_DATASET=/path/to/encoded-dataset`. The encoded dataset must use the
same tokenizer and contain `input_ids`, `labels` and `lengths`.

## Launch training

Run the following on each node. Set `NODE_RANK` to 0, 1, 2 or 3; all nodes use
the same reachable `MASTER_ADDR` and free `MASTER_PORT`. The caller allocates
the machines. Use this command once per node with its eight GPUs available.

```bash
export NODE_RANK=0
export MASTER_ADDR=192.0.2.10
export MASTER_PORT=29500

# Inspect the resolved torchrun command and selected kernel without loading CUDA.
DRY_RUN=1 bash research/hsd_gdn/train_full32.sh

# Launch once per node, changing NODE_RANK on each node.
bash research/hsd_gdn/train_full32.sh
```

Replace the example address with the address of your rank-0 node. The launcher
reads the JSON recipe directly and generates all training and backend arguments;
do not maintain a second copy of them in a shell command. Recipe environment
settings take precedence over inherited backend variables. Each node records
the resolved command and environment under `OUTPUT_DIR/launch/`.

To use prepared encoded data, unset `DATASET` and set
`CACHED_DATASET=/path/to/encoded-dataset`. Supply exactly one dataset source.
For a different batch layout or training budget, copy and edit the recipe, then
pass `--recipe /path/to/training-recipe.json`; its `model_config` path is resolved
relative to that JSON file. The launcher checks the world size, global batch,
nominal token budget and initialized model configuration before training.

Initialize `MODEL` once before launching any nodes. For an ordinary training
resume, pass `--resume /path/to/complete-training-checkpoint` on every node;
optimizer, scheduler and RNG files must be retained. The launcher starts
`torchrun` itself, so invoke it once per node rather than once per GPU.

The nominal budget is 50,000,297,984 packed token slots, calculated from global
batch × sequence length × optimizer steps. The actual count of non-padding text
tokens depends on the supplied dataset and packing.
