# Safin-1 · MARCH

**MARCH: Scaling Recurrent Memory with Content-Routed State Anchors**

[论文](https://arxiv.org/abs/2608.12435) ·
[技术文档 PDF](Safin-1.pdf) ·
[模型](https://huggingface.co/collections/AI45Research/safin-1-6a90457bfa414a1453744596) ·
[快速开始](docs/march/quickstart.md) ·
[Arch Space](https://github.com/InternLM/archspace) ·
[English](README.md)

**Safin-1 4B 与 35B-A3B 的训练和推理代码已提供。**

MARCH 在 Qwen3.5 的 Gated Delta Network 上加入 **memory anchor（记忆锚点）**
和 HSD 历史状态读取。Safin-1 包含 4B dense 与 35B-A3B MoE 两款模型。
本仓库提供 Megatron CPT/SFT、HF 推理、vLLM 加速推理和可选的 persistent state。

| 模型 | model_type | 层数 | Memory anchor | HSD 配置 |
|---|---|---:|---|---|
| Safin-1 4B | `safin1_4b` | 32 | 每 512 个文本 token 一个 | global read 10–30，block 32768，top-k 4 |
| Safin-1 35B-A3B | `safin1_35b_a3b` | 40 | 每 512 个文本 token 一个 | global read 10–30，block 32768，top-k 4 |

加载已有 checkpoint 时保留其 runtime。历史读取使用 `hsd_*` 配置，记忆锚点相关
配置与权重使用 `anchor_*`。

已公开 [Safin-1 35B-A3B 权重](https://huggingface.co/AI45Research/Safin-1-35B-A3B)
和[配套 SFT baseline](docs/march/model-release.md#published-checkpoints)，固定版本下载命令见模型指南。
Safin-1 4B 权重目前需要仓库访问权限。

## 快速使用

使用 Linux、Python 3.11 和独立环境，从当前仓库安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/march/install.sh hf

MODEL=/path/to/Safin-1-4B PROMPT='用一句话介绍 memory anchor。' \
  bash scripts/march/infer_hf.sh
```

模型与 tokenizer 的准备方式见[模型指南](docs/march/model-release.md)。
HF 加载使用 `trust_remote_code=True`。

按[安装说明](docs/march/installation.md)构建仓库内的 vLLM 后，启动 API 服务：

```bash
MODEL=/path/to/Safin-1-4B bash scripts/march/infer_vllm.sh --port 8000
```

默认使用 Graph V3、Triton/varlen 和 paged history，关闭 prefix cache。
设置 `MARCH_INFERENCE_PROFILE=eager` 可切换为 eager。

训练前安装 `training` 环境，然后运行 CPT 或 SFT：

```bash
MODEL=/path/to/Safin-1-4B DATASET=examples/march/cpt.jsonl \
OUTPUT_DIR=/path/to/output/cpt MAX_LENGTH=1024 \
  bash scripts/march/train_cpt.sh --train_iters 1

MODEL=/path/to/Safin-1-4B DATASET=examples/march/sft.jsonl \
OUTPUT_DIR=/path/to/output/sft MAX_LENGTH=1024 TUNER_TYPE=full \
  bash scripts/march/train_sft.sh --train_iters 1
```

示例数据用于首次运行。完整的 35B 多卡、LoRA 导出、API 调用和评测命令见
[快速开始](docs/march/quickstart.md)。

## 使用指南

| 内容 | 文档 |
|---|---|
| 环境安装与 CUDA 依赖 | [安装说明](docs/march/installation.md) |
| 训练和推理命令 | [快速开始](docs/march/quickstart.md) |
| 模型与 runtime 参数 | [配置说明](docs/march/configuration.md) |
| Graph 与 eager 推理 | [推理指南](docs/march/inference.md) |
| Memory anchor 和 HSD | [架构说明](docs/march/architecture.md) |
| 可训练持久状态 | [Persistent state](docs/march/persistent-state.md) |
| 已验证环境与兼容说明 | [验证说明](docs/march/validation.md) |
| 0.8B 实验和 FLA baseline | [独立研究目录](research/hsd_gdn/README.md) |
| 代码结构和贡献方式 | [仓库结构](docs/march/repository-layout.md)、[贡献指南](CONTRIBUTING_CN.md) |

已有 GPU 上的训练、保存、HF 重载、原生后端构建和 TP/DP API 验证。
具体软件版本与结果见验证说明，首个 `v0.1.0` 代码版本的发布流程见
[发行指南](docs/march/release.md)。

## 许可与致谢

代码及 Safin-1 权重使用 Apache-2.0；第三方保留各自许可证与署名。
模型和数据材料见[模型指南](docs/march/model-release.md)，代码归属与保留声明见
[许可证说明](docs/march/licensing.md)。

本项目基于 [ModelScope ms-swift](https://github.com/modelscope/ms-swift)、Qwen3.5
和[第三方组件](third_party/VENDORED_COMPONENTS.md)。
