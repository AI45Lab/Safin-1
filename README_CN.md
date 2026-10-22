# MARCH / Safin-1

公开仓库：[AI45Lab/Safin-1](https://github.com/AI45Lab/Safin-1)。

[English](README.md)

MARCH 在 Qwen3.5 的 Gated Delta Network 上加入 **memory anchor（记忆锚点）**
和 HSD 历史状态读取。Safin-1 是对应模型家族，包含 4B dense 与 35B-A3B MoE。
本仓库保留 ms-swift fork 形态，提供 Megatron CPT/SFT、HF 推理和仓库内 vLLM 推理。
Persistent state 作为可选功能保留。

当前为公开发行准备版本，代码版本暂定 `v0.1.0`；模型 ID 与 Apache-2.0 许可已确定，
权重发布和正式 tag 的状态见发行清单。
XTuner、LMDeploy 与 MARCH RL 不在当前支持范围。状态见[发布清单](docs/march/release.md)。

| 模型 | model_type | 层数 | Memory anchor | HSD 配置 |
|---|---|---:|---|---|
| Safin-1 4B | `safin1_4b` | 32 | 每 512 个文本 token 插入一个 | global read 10–30，block 32768，top-k 4 |
| Safin-1 35B-A3B | `safin1_35b_a3b` | 40 | 每 512 个文本 token 插入一个 | global read 10–30，block 32768，top-k 4 |

加载已有 Safin-1 checkpoint 时保留其保存的 runtime。`hsd_*` 保留，记忆锚点相关
配置与权重统一使用 `anchor_*`；旧键需在 checkpoint 副本上[迁移](docs/march/naming.md)。

使用 Python 3.11 和独立环境，从当前 checkout 安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/march/install.sh hf

MODEL=/path/to/Safin-1-4B PROMPT='用一句话介绍 memory anchor。' \
  bash scripts/march/infer_hf.sh

MODEL=/path/to/Safin-1-4B bash scripts/march/infer_vllm.sh --port 8000
```

vLLM 需先构建本仓库 fork；默认 Graph V3、Triton/varlen、关闭 prefix cache。
使用 `MARCH_INFERENCE_PROFILE=eager` 完整回退。

```bash
MODEL=/path/to/Safin-1-4B DATASET=examples/march/sft.jsonl \
OUTPUT_DIR=/path/to/output MAX_LENGTH=1024 TUNER_TYPE=full \
  bash scripts/march/train_sft.sh --train_iters 1
```

详细步骤见[快速开始](docs/march/quickstart.md)和[安装说明](docs/march/installation.md)，
包含 CPT、35B 多卡、API 调用、评测、dry-run 和训练后加载。无需内部盘地址或调度系统。

两款模型已记录 H200 上一步 CPT/SFT、保存和 HF 重载验证；这不代表收敛、长时间稳定性
或吞吐上限。[独立环境安装与原生源码构建](docs/march/validation.md#installation-and-packaging)、
4B 全参 / 35B LoRA 有限步续训及[双卡 TP/DP、8K API 验收](docs/march/validation.md#inference)
已有通过记录；完整数值对齐、生产规模压力与其他硬件仍需分别验证。
35B full/cached logits 差异见[验证矩阵](docs/march/validation.md)。

[用户文档目录](docs/march/README.md)集中列出安装、训练和推理指南。
[仓库范围](docs/march/repository-layout.md)列出主链路、共享框架依赖和独立研究代码。

| 位置 | 内容 |
|---|---|
| `march/safin1/` | HF dense/MoE 模型、history read 和缓存 |
| `march/kernels/` | MARCH 训练算子与 anchor 工具 |
| `swift/` | ms-swift CLI 和通用训练框架 |
| `configs/march/`、`scripts/march/` | runtime 模板与公开使用入口 |
| `tests/march/` | CPU、小模型与可选 GPU 回归 |
| `third_party/` | mcore-bridge、vLLM、lm-eval fork |

[架构说明](docs/march/architecture.md) · [配置说明](docs/march/configuration.md) ·
[Persistent state](docs/march/persistent-state.md) · [模型交付](docs/march/model-release.md) ·
[贡献指南](CONTRIBUTING_CN.md)

代码及 Safin-1 权重许可为 Apache-2.0，第三方保留各自许可证与 attribution。
训练数据的来源与使用条款在模型交付材料中单独披露。
原公开页面的 MIT 声明单独保留，合入安排见[代码许可证说明](docs/march/licensing.md)。
本项目基于 ms-swift、Qwen3.5 和[第三方组件](third_party/VENDORED_COMPONENTS.md)。

旧 HSD-GDN 0.8B 与 FLA baseline 的模型配置和训练/推理入口见 [独立研究目录](research/hsd_gdn/README.md)。
