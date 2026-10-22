# Safin-1 · MARCH

**MARCH: Scaling Recurrent Memory with Content-Routed State Anchors**

[Paper](https://arxiv.org/abs/2608.12435) ·
[Technical document (PDF)](Safin-1.pdf) ·
[Models](https://huggingface.co/collections/AI45Research/safin-1-6a90457bfa414a1453744596) ·
[Quick start](docs/march/quickstart.md) ·
[Arch Space](https://github.com/InternLM/archspace) ·
[中文](README_CN.md)

Training and inference code is available for **Safin-1 4B** and **Safin-1 35B-A3B**.

MARCH extends Qwen3.5's Gated Delta Network with **memory anchors** and HSD
historical-state reading. Safin-1 includes a 4B dense model and a 35B-A3B MoE
model. This ms-swift fork provides Megatron CPT/SFT, Transformers inference,
a vendored vLLM backend and optional persistent state.

| Model | Model type | Layers | Memory-anchor interval | HSD history read |
|---|---|---:|---|---|
| Safin-1 4B | `safin1_4b` | 32 | 512 text tokens | layers 10–30, block 32768, top-k 4 |
| Safin-1 35B-A3B | `safin1_35b_a3b` | 40 | 512 text tokens | layers 10–30, block 32768, top-k 4 |

Existing checkpoints retain their saved runtime. History-read settings use
`hsd_*`; memory-anchor settings and weights use `anchor_*`.

## Get started

Use Linux, Python 3.11 and a dedicated environment. Install from this checkout:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/march/install.sh hf

MODEL=/path/to/Safin-1-4B PROMPT='Explain memory anchors in one sentence.' \
  bash scripts/march/infer_hf.sh
```

Prepare the model and tokenizer using the [model guide](docs/march/model-release.md).
HF loading uses `trust_remote_code=True`.

For vLLM serving, build the repository's backend with the
[vLLM installation profile](docs/march/installation.md), then start the API:

```bash
MODEL=/path/to/Safin-1-4B bash scripts/march/infer_vllm.sh --port 8000
```

The default profile uses Graph V3, Triton/varlen and paged history with prefix
caching disabled. Set `MARCH_INFERENCE_PROFILE=eager` for eager execution.

For training, install the `training` profile and run CPT or SFT:

```bash
MODEL=/path/to/Safin-1-4B DATASET=examples/march/cpt.jsonl \
OUTPUT_DIR=/path/to/output/cpt MAX_LENGTH=1024 \
  bash scripts/march/train_cpt.sh --train_iters 1

MODEL=/path/to/Safin-1-4B DATASET=examples/march/sft.jsonl \
OUTPUT_DIR=/path/to/output/sft MAX_LENGTH=1024 TUNER_TYPE=full \
  bash scripts/march/train_sft.sh --train_iters 1
```

The examples contain small datasets for a first run. The
[quick start](docs/march/quickstart.md) covers 35B parallelism, LoRA exports,
API requests, dry-runs and evaluation.

## Guides

| Topic | Guide |
|---|---|
| Environment and CUDA dependencies | [Installation](docs/march/installation.md) |
| Training and inference commands | [Quick start](docs/march/quickstart.md) |
| Model/runtime parameters | [Configuration](docs/march/configuration.md) |
| Graph and eager inference | [Inference](docs/march/inference.md) |
| Memory anchors and HSD | [Architecture](docs/march/architecture.md) |
| Learned persistent state | [Persistent state](docs/march/persistent-state.md) |
| Tested setups and compatibility | [Validation](docs/march/validation.md) |
| HSD-GDN 0.8B and FLA baselines | [Research workspace](research/hsd_gdn/README.md) |
| Source structure and contributions | [Repository layout](docs/march/repository-layout.md), [contributing](CONTRIBUTING.md) |

GPU validation covers training/export/HF reload, native backend builds and
TP/DP API serving. Detailed versions and results are in the validation guide.
The [release guide](docs/march/release.md) tracks the initial `v0.1.0` code release.

## License and attribution

Code and Safin-1 weights use Apache-2.0. Vendored projects retain their licenses
and notices; model/data terms are described in the model materials. See
[licensing](docs/march/licensing.md) and [model delivery](docs/march/model-release.md).

Built on [ModelScope ms-swift](https://github.com/modelscope/ms-swift), Qwen3.5
and the [vendored projects](third_party/VENDORED_COMPONENTS.md).

## Citation

```bibtex
@misc{march2026,
  title={MARCH: Scaling Recurrent Memory with Content-Routed State Anchors},
  author={Ming Zhang and Kaisen Yang and Shu Yu and Ermo Hua and Ning Ding and Xia Hu and Bowen Zhou and Chaochao Lu and Youbang Sun},
  year={2026},
  eprint={2608.12435},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2608.12435}
}
```
