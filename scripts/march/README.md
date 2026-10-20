# MARCH / Safin-1 scripts

从仓库根目录执行。首次使用先读[快速开始](../../docs/march/quickstart.md)与
[安装说明](../../docs/march/installation.md)，模型路径由 `MODEL` 指定。
脚本的目录和命令保持稳定；下面按用途列出入口。
用户指南集中在[文档目录](../../docs/march/README.md)，维护与证据材料见
[维护指南](../../docs/march/maintenance.md)。0.8B 的训练入口独立放在
[研究目录](../../research/hsd_gdn/README.md)，不使用这里的 Safin-1 runtime 模板。

## 训练、推理和评测

| 入口 | 用途 | 使用说明 |
| --- | --- | --- |
| [install.sh](install.sh) | 安装所需 backend | [安装](../../docs/march/installation.md) |
| [check_environment.py](check_environment.py) | 检查依赖版本、模型和 backend | [安装](../../docs/march/installation.md) |
| [verify_checkpoint.py](verify_checkpoint.py) | 检查 checkpoint 配置、remote code 和权重文件 | [模型交付](../../docs/march/model-release.md) |
| [train_cpt.sh](train_cpt.sh) | Megatron CPT | [训练示例](../../docs/march/quickstart.md) |
| [train_sft.sh](train_sft.sh) | Megatron SFT；默认 LoRA，full 使用 `TUNER_TYPE=full` | [训练示例](../../docs/march/quickstart.md) |
| [infer_hf.sh](infer_hf.sh) | HF 文本推理基线 | [快速开始](../../docs/march/quickstart.md) |
| [infer_vllm.sh](infer_vllm.sh) | vLLM OpenAI 兼容服务；默认 Graph profile | [推理配置](../../docs/march/inference.md) |
| [eval_hf.sh](eval_hf.sh)、[eval_vllm.sh](eval_vllm.sh) | 固定任务、seed、模板与 thinking 的评测入口 | [评测](../../docs/march/evaluation.md) |

`serve_vllm.py` 实现服务 CLI，`evaluate.py` 实现两个评测入口；它们共用
`march/inference/vllm_profile.py` 的后端策略。shell 入口负责启动，profile
负责默认值和受控参数，不在文档中维护另一套默认配置。
`_training_common.sh` 是 CPT/SFT 的共享配置解析器，不能直接启动训练。

## 回归与性能工具

| 工具 | 验证范围 |
| --- | --- |
| [smoke_hf.py](smoke_hf.py) | HF 加载与 greedy generation；由 HF shell 入口调用 |
| [smoke_megatron.sh](smoke_megatron.sh) | 合成数据训练、保存并重载导出模型 |
| [smoke_training_resume.py](smoke_training_resume.py) | 训练续训验收；`_training_resume_probe.py` 为其内部探针 |
| [smoke_vllm_api.py](smoke_vllm_api.py) | 已启动服务的 API smoke |
| [smoke_serving_matrix.py](smoke_serving_matrix.py) | 显式指定组合的服务验收矩阵 |
| [check_cached_decode.py](check_cached_decode.py) | HF full-forward / cached-decode 对比 |
| [check_prefix_cache.py](check_prefix_cache.py) | 显式开启的实验性 prefix cache 验收 |
| [benchmark_inference.py](benchmark_inference.py) | 模型级推理、批处理、缓存隔离与 backend 观测 |
| [benchmark_operators.py](benchmark_operators.py) | 算子计时、峰值分配与数值对比 |
| [check_operator_correctness.py](check_operator_correctness.py) | 执行 CUDA 正确性门禁或核对证据绑定的源码 |
| [check_offline_build.py](check_offline_build.py) | 离线原生构建所需资源预检 |

工具名称代表检查内容，运行成功的具体范围以输出报告为准。模型级、算子级、
服务级证据分别记录；相互不能替代。当前结果见[验证矩阵](../../docs/march/validation.md)，
发布检查使用的脱敏验收数据见[证据索引](../../docs/march/evidence-index.md)。

## 模型迁移与发行维护

| 工具 | 用途 |
| --- | --- |
| [migrate_checkpoint.py](migrate_checkpoint.py) | 按[命名/迁移说明](../../docs/march/naming.md)迁移 checkpoint |
| [check_public_release.py](check_public_release.py) | 公开源码内容和敏感路径检查 |
| [check_vendor_manifest.py](check_vendor_manifest.py) | vendored 文件哈希、许可证和来源检查 |
| [verify_fla_provenance.py](verify_fla_provenance.py) | 可选 FLA 研究源码的独立来源核验 |
| [check_publication.py](check_publication.py) | 检查发行元数据与待补字段 |
| [export_source.py](export_source.py) | 导出经过筛选的源码包 |
| [verify_package.py](verify_package.py) | 检查安装包内容和独立安装后的 HF 加载 |

发行步骤见[发行说明](../../docs/march/release.md)。内部模型位置、镜像与环境
不写进公开入口。Python 工具可用 `--help` 查看参数；shell 入口的必要环境变量
和示例在对应指南中。
