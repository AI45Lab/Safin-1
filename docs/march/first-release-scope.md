# 支持范围

Safin-1 提供 4B dense 与 35B-A3B MoE 的训练和推理，首个代码版本按 `v0.1.0`
准备。模型卡与权重在模型仓库发布。

| 功能 | 使用方式 |
|---|---|
| CPT / SFT | Megatron CLI，支持全参数与 LoRA，导出 HF checkpoint |
| HF 推理 | `trust_remote_code=True`，支持缓存生成与批处理 |
| vLLM 推理 | 仓库内 fork，默认 Graph V3，提供 eager profile |
| Persistent state | 可选模型状态，配套训练配置与 artifact 保存/加载 |
| 0.8B / FLA | 独立研究目录，包含 HSD-GDN 和 GDN、DeltaNet、KDA、Transformer baseline |

环境安装见[安装说明](installation.md)，训练和推理命令见[快速开始](quickstart.md)。
已验证环境、并行布局和兼容条件集中在[验证说明](validation.md)。

XTuner、LMDeploy 和 MARCH RL 列入后续适配方向。
发布配置与步骤见[发行指南](release.md)。
