# Historical HSD-GDN and FLA baselines

This optional research workspace restores the historical 21-layer HSD-GDN
experiment and its FLA baselines. It has its own configuration, registration,
training, inference and provenance. Safin-1 4B/35B launchers continue to use
`scripts/march/` and `configs/march/`.

| Profile | Configuration | Parameters constructed from the restored code |
|---|---|---:|
| `hsd_gdn` | `configs/hsd_gdn_0p8b.json` | 900,909,300 |
| `hsd_gdn_readq32` | `configs/hsd_gdn_0p8b_readq32.json` | 845,367,540 |
| `hsd_gdn_topk4` | `configs/hsd_gdn_0p8b_topk4.json` | 843,892,980 |
| `gdn` | `configs/gdn_0p8b.json` | 792,785,532 |
| `delta_net` | `configs/delta_net_0p8b.json` | 718,273,920 |
| `kda` | `configs/kda_0p8b.json` | 814,292,094 |
| `transformer` | `configs/transformer_0p8b.json` | 692,912,640 |

“0.8B” is the historical experiment label, not an assertion that these models
have equal parameter counts. All three HSD profiles have 21 layers and one
memory anchor per 512 text tokens. The original dense profile uses six
192-dimensional route keys. The recommended dense LoRA32 and sparse top-k
profiles use one 64-dimensional route key; only the sparse profile limits
reads to four history states. Use the full saved checkpoint configuration
when resuming.

The historical serialized fields and tensor names retain `summary_*` for old
HSD checkpoint compatibility. In this experiment those tokens are memory
anchors. These are separate model types from `safin1_4b` and
`safin1_35b_a3b`; do not apply these JSON files or the research FLA fork to
Safin-1 checkpoints. Safin-1 keeps its canonical `anchor_*` schema.

OLMo3 model variants and low-rank history-cache storage are outside this workspace.
Old checkpoints with those options enabled fail explicitly; inactive default
metadata remains loadable. The recommended rank-32 history-query projection is
retained. See the [source/backend map](source-layout.md).

## Environment and inputs

Run these commands from a source checkout or extracted source distribution.
The base Safin-1 wheel does not install this optional research workspace.
Use the existing CUDA training environment, or install the framework and the
additional packages listed in `requirements.txt`. The version list records the
environment used for restoration; it does not certify a clean native build.

`run.py` selects the source under
`third_party/flash-linear-attention/fla/` in its own process and its training
child. This preserves the original FLA changes without modifying a shared
Python environment. Normal Safin-1 entrypoints do not prepend this directory.

All model, tokenizer and dataset inputs must be local. Execution sets the HF,
Datasets, Transformers and evaluator offline switches. Prepare any missing
dependencies and files before starting a GPU job. The FlashQLA producer is a
Hopper path; the original 0.8B experiment is being checked on H200. Other
hardware requires separate validation.

## Train

Use the [dense history-read-query LoRA32 reference recipe](reference-recipe.md)
and `train_reference.sh` for the main 0.8B example with dense history reads
(`hsd_read_topk=0`). The complete optimizer/WSD
schedule and an 8-GPU batch layout matching the original global batch are
provided, alongside a single-GPU short command. The top-k4 training profile
remains a separate [sparse comparison](reference-topk4.md).

The [32-GPU training configuration](configs/hsd_gdn_0p8b_readq32_full32.json)
provides the batch layout, optimizer/WSD settings and `tile_online` backend.
The [training guide](reference-full32.md) covers model initialization and local
datasets. `train_full32.sh` reads this JSON directly and launches once per node;
`DRY_RUN=1` prints its command and selected backend before reserving GPUs.

Supply a tokenizer with the original 32,000-token vocabulary for a comparable
from-scratch experiment. Safin-1/Qwen tokenizers are not interchangeable with
the historical tokenizer. A JSONL dataset can contain `{"text": "..."}` rows;
an Arrow path/glob can contain the same text column.

```bash
TOKENIZER=/path/to/original-32000-token-tokenizer \
DATASET=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/outputs/hsd-smoke \
MODEL_PROFILE=hsd_gdn \
MAX_STEPS=20 MAX_LENGTH=16384 BATCH_SIZE=1 GRAD_ACC=1 \
NPROC_PER_NODE=1 \
bash research/hsd_gdn/train.sh
```

This initializes a fresh seed-42 model under `OUTPUT_DIR/seed42`, trains under
`OUTPUT_DIR/training`, and retains at most two checkpoints. It defaults to
BF16, full tuning, packing, DeepSpeed ZeRO-2, AdamW, LR `4e-4`, WSD, no warmup
for the short smoke, and no activation checkpointing. Default HSD kernels are
the original FlashQLA producer with optimized backward. The recommended
`hsd_gdn_readq32` profile defaults to `tile_online` with strict dispatch; the
older `hsd_gdn` profile retains `tile_speed`, and top-k uses its serialized
sparse reader. Set `HSD_FUSED_GLOBAL_READ_BACKEND=tile_speed` explicitly for
the historical dense comparison. They are not the
Safin-1 vLLM serving defaults.

To train an existing historical checkpoint, supply `MODEL` and its matching
`MODEL_PROFILE`; the tokenizer is then read from that checkpoint. To select a
baseline, set `MODEL_PROFILE=gdn`, `delta_net`, `kda` or `transformer`.

```bash
# Inspect the resolved command without importing the CUDA model runtime.
TOKENIZER=/path/to/tokenizer DATASET=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/new-output DRY_RUN=1 \
bash research/hsd_gdn/train.sh

# Eight local GPUs; parallel allocation is managed by the caller.
MODEL_PROFILE=gdn TOKENIZER=/path/to/tokenizer DATASET=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/new-gdn-run NPROC_PER_NODE=8 BATCH_SIZE=2 GRAD_ACC=4 \
MAX_STEPS=11921 WARMUP_STEPS=256 \
bash research/hsd_gdn/train.sh
```

The eight-GPU example has fewer tokens per optimizer step than the original
32-GPU 50B-token schedule. Match total GPUs × micro batch × accumulation ×
sequence length and the actual packed text-token count before comparing budgets.
The short default is a regression workload, not a 50B-token reproduction.

Resume uses a new output directory and a complete checkpoint, including
optimizer/RNG/trainer state:

```bash
MODEL=/path/to/run/seed42 DATASET=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/new-resume-run MAX_STEPS=20 \
bash research/hsd_gdn/train.sh --resume /path/to/run/training/checkpoint-10
```

Use `python research/hsd_gdn/run.py train --help` for sequence length, batches,
packing, scheduler and optional `--deepspeed none`. Additional Swift arguments
can follow `--`. Output directories are never overwritten.

## Load and infer

```bash
MODEL=/path/to/run/training/checkpoint-20 \
bash research/hsd_gdn/infer.sh \
  --prompt 'Explain how memory anchors retain historical information.' \
  --max-new-tokens 32
```

The inference entrypoint registers the restored model type, strictly checks
loading keys, uses BF16 CUDA and greedy cached generation, and prints generated
token IDs and text. This is the historical HF inference implementation; no
0.8B vLLM integration is implied.

The [source and backend map](source-layout.md) separates the recommended
recipe, baselines, historical comparisons and retained implementation variants.

## Source provenance and validation

Model configurations, executable training recipes and baseline code are retained.
Personal run identifiers, training curves and execution diaries are not
distributed with the public source.

`SOURCE_MANIFEST.json` records the original HSD restoration fingerprints and
import-only namespace rewrite, plus explicit model/config cleanup deltas. The
reader/producer kernels retain their original fingerprints. FLA `PROVENANCE.json` retains the wheel hashes and historical
overlays. FLA and FlashQLA preserve their MIT licenses and notices. Upstream FLA
commit `3a9ce1c83a13994d824dbb3421e2989d330bb38b` is verified by content against
the original wheels and upstream split-package generator; see its
[`UPSTREAM.md`](../../third_party/flash-linear-attention/UPSTREAM.md).

The supported validation scope is described in
[`validation.md`](validation.md). Run the repository contract checks with:

```bash
python -m pytest tests/march/test_research_hsd.py -q
# With the optional research dependencies installed, check all seven model
# geometries on meta and strict tiny-model loading on CPU.
MARCH_TEST_RESEARCH=1 python -m pytest tests/march/test_research_cleanup.py -q
```

## Matched initialization from the GDN baseline

For matched initialization, the KDA mapper uses a GDN-derived seed. The tensor mapping tools are retained:

```bash
python research/hsd_gdn/tools/init_kda_from_gdn_seed.py \
  --gdn-seed-dir /path/to/random-gdn-seed \
  --output-dir /path/to/new-kda-seed

python research/hsd_gdn/tools/init_hsd_from_gdn.py \
  --gdn-model-dir /path/to/gdn-checkpoint \
  --hsd-config-json research/hsd_gdn/configs/hsd_gdn_0p8b.json \
  --output-dir /path/to/new-hsd-initialization
```

The KDA mapper rejects trained checkpoints, copies compatible GDN tensors exactly and initializes only KDA-specific parameters. The HSD mapper can initialize compatible backbone weights from a GDN checkpoint. Both require local source weights. These preparation tools are distinct from the default from-scratch `run.py init` operation.
