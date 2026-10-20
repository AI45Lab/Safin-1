# MARCH / Safin-1 使用

从仓库根目录运行命令。先按[安装说明](installation.md)安装所需 backend，再准备
[审核过的模型目录](model-release.md)。这里的路径由使用者自行设置，不依赖内部盘。

```bash
export SAFIN1_4B_MODEL=/path/to/Safin-1-4B
export SAFIN1_35B_A3B_MODEL=/path/to/Safin-1-35B-A3B
python scripts/march/verify_checkpoint.py "$SAFIN1_4B_MODEL"
python scripts/march/check_environment.py --backend hf --model "$SAFIN1_4B_MODEL"
```

HF 是正确性参考，加载使用 `trust_remote_code=True`：

```bash
MODEL="$SAFIN1_4B_MODEL" PROMPT='介绍一下 MARCH。' \
  bash scripts/march/infer_hf.sh
MODEL="$SAFIN1_35B_A3B_MODEL" PROMPT='介绍一下 memory anchor。' \
  bash scripts/march/infer_hf.sh
```

当前 HF shell 示例使用原始文本输入和 greedy generation。对话评测必须明确记录
chat template、thinking 设置、采样参数与 token IDs，不能将原始文本 smoke 当成对话评测。

构建 vendored vLLM 后，启动 OpenAI 兼容服务：

```bash
# 4B 单卡；单卡 35B 的已有记录使用 H200 级显存。
MODEL="$SAFIN1_4B_MODEL" bash scripts/march/infer_vllm.sh --port 8000

# 35B 多卡配置示例；具体 TP 组合需按验证矩阵验收。
MODEL="$SAFIN1_35B_A3B_MODEL" TP_SIZE=4 \
  bash scripts/march/infer_vllm.sh --port 8000
```

服务名默认 `Safin-1`：

```bash
python scripts/march/smoke_vllm_api.py --base-url http://127.0.0.1:8000
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Safin-1","messages":[{"role":"user","content":"介绍 MARCH"}],"temperature":0}'
```

Graph V3 默认启用，anchor 边界按设计可以回退 eager。完整回退与查看命令：

```bash
MODEL="$SAFIN1_4B_MODEL" MARCH_INFERENCE_PROFILE=eager \
  bash scripts/march/infer_vllm.sh
MODEL="$SAFIN1_35B_A3B_MODEL" DRY_RUN=1 bash scripts/march/infer_vllm.sh
```

训练前先安装 [training 环境](installation.md)，在 CPU 上检查依赖、实际处理器和 checkpoint 文件：

```bash
CUDA_VISIBLE_DEVICES='' python scripts/march/check_environment.py \
  --backend training --strict-versions --model "$SAFIN1_4B_MODEL"
python scripts/march/verify_checkpoint.py "$SAFIN1_4B_MODEL"
```

检查 35B 时替换模型路径。旧 checkpoint 的 remote-code 刷新方式见[验收说明](acceptance.md)。

CPT 示例每行包含 `text`，SFT 示例每行包含 `messages`：

```bash
MODEL="$SAFIN1_4B_MODEL" DATASET=examples/march/cpt.jsonl \
OUTPUT_DIR=/path/to/output/cpt MAX_LENGTH=1024 DRY_RUN=1 \
  bash scripts/march/train_cpt.sh --train_iters 1

MODEL="$SAFIN1_4B_MODEL" DATASET=examples/march/sft.jsonl \
OUTPUT_DIR=/path/to/output/sft MAX_LENGTH=1024 TUNER_TYPE=full \
  bash scripts/march/train_sft.sh --train_iters 1

# 35B 已有一步 full-parameter smoke 使用 8×H200、EP=8。
MODEL="$SAFIN1_35B_A3B_MODEL" DATASET=examples/march/sft.jsonl \
OUTPUT_DIR=/path/to/output/35b-sft MAX_LENGTH=1024 TUNER_TYPE=full \
NPROC_PER_NODE=8 EXPERT_MODEL_PARALLEL_SIZE=8 \
  bash scripts/march/train_sft.sh --train_iters 1
```

SFT 脚本的默认 tuner 是 LoRA；上述已验证 full-parameter 路径显式传 `TUNER_TYPE=full`。
LoRA 合并导出的 HF 权重位于 `checkpoint-<step>-merged`；全参数导出位于
`checkpoint-<step>`。使用 HF/vLLM 推理时指定含 `config.json` 和 safetensors 的
导出目录。需要单独分享 HF adapter 时，可显式设置 `--save_safetensors true
--merge_lora false`；目录中应同时包含 `adapter_model.safetensors` 和
`adapter_config.json`，加载时仍需同一份 Safin-1 基础模型及 runtime。MARCH 投影
的 A/B 与 modules_to_save 权重会保存；单独训练的原始 gate/norm/persistent state
需要合并导出，不能通过 adapter 文件静默省略。见[独立 adapter 验收](validation.md#training)。
LoRA 与并行服务的最新验收范围见[验收报告](validation.md)。
已有 checkpoint 默认保留自身 runtime；
从基础 Qwen3.5 初始化才默认应用 `configs/march/`。

完整训练—保存—加载 smoke 会生成公开合成数据并重载导出 checkpoint：

```bash
MODEL="$SAFIN1_4B_MODEL" OUTPUT_DIR=/path/to/output/smoke \
TRAIN_TASK=sft bash scripts/march/smoke_megatron.sh
```

运行 HF 评测前安装 evaluator：

```bash
bash scripts/march/install.sh eval
# Run resource preparation before the GPU/offline task.
python -m nltk.downloader -d /path/to/nltk_data punkt_tab
export NLTK_DATA=/path/to/nltk_data
MODEL="$SAFIN1_4B_MODEL" TASKS=ifeval OUTPUT_DIR=/path/to/evaluation \
LIMIT=16 bash scripts/march/eval_hf.sh
```

以上 LIMIT 是使用 smoke，正式结果需要完整任务集与固定版本、seed、prompt 配置。
OOM 时先降低 `MAX_MODEL_LEN` / `MAX_NUM_SEQS` 或训练 `MAX_LENGTH`。
已执行的资源、精度边界与仍待覆盖的组合见[验证矩阵](validation.md)。

HF/vLLM 共用的评测配置、thinking 开关和离线资源要求见[评测说明](evaluation.md)。
