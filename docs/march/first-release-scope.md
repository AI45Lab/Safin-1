# 首次开源的支持范围

首次开源围绕 Safin-1 4B / 35B-A3B 的 Megatron CPT/SFT、HF 加载生成和仓库内
vLLM 推理。模型卡和权重可以独立发布，代码仓库链接到最终模型 ID 与固定版本。
首个代码版本按 `v0.1.0` 准备，联系入口为公开仓库 Issues。当前仍是发行准备，
尚未创建公开 tag。正式发布状态以 [publication.json](../../configs/release/publication.json) 为准；
范围说明不将未完成的验收改成通过。

| 路径 | 已有证据与首发边界 |
|---|---|
| 4B / 35B CPT、SFT | 已有生产尺寸优化器步、导出和 HF 重载；4B 全参数及 35B LoRA 的多步续训记录单独保留。无需再做 35B 全参数连续/恢复双跑。 |
| HF 推理 | 使用 `trust_remote_code=True` 和实际 checkpoint 的 runtime。支持边界、缓存和生成用例，以对应记录的精度、硬件与输入范围为限。 |
| vLLM 推理 | 使用本仓库 fork；默认 Graph profile，prefix caching 关闭；eager 为备用 profile。模型加载、Graph replay、请求隔离和重复性证据分开记录。 |
| Persistent state | 保留训练和推理代码及文档；可选功能。小模型隔离测试不等同于生产尺寸多请求 GPU 验收。 |
| 历史 0.8B / FLA | 可选研究工作区。旧 HSD-GDN 配方、baseline 和训练记录独立维护；其 32 卡实验不替代 Safin-1 4B/35B 的验收。 |

支持的硬件和软件组合以 [验收矩阵](validation.md) 为准。已执行的主要 GPU 验收
使用 H200 和 CUDA 12.9。其他 GPU、软件升级和未测试的并行布局没有自动继承这些结果。

## 数值与使用限制

- 相同 profile 的重复生成、Graph/eager 比较、HF/vLLM 比较是不同的检查。
  生成 token 一致不能证明 logits/logprob 一致；相关失败和条件通过记录保留。
- 35B MoE 的 full-forward/cached logits 与部分跨 profile 差异仍有明确记录。
  不承诺逐位一致、完整 HF/vLLM logits 对齐或 GRPO/RL 正确性。
- TP/DP API 服务、生产尺寸 persistent-state 并发和长上下文覆盖，按各自验收记录
  声明。尚未通过的组合仍为待验证，不能因基础推理通过而扩大支持范围。
- CP 配合 persistent memory、anchor MTP、非均匀 anchor 布局及其他受限训练组合
  必须服从已有能力检查；见 [后端限制](backends.md)。
- XTuner、LMDeploy 和 MARCH RL 是后续适配方向，不属于首发已支持后端。

## 发布前最后检查

1. 固定源码和模型版本，核对最新 4B/35B 加载、训练导出与推理证据的源码范围。
2. 完成全新环境的依赖安装、原生扩展构建、`pip check` 和模型使用闭环。
   已有镜像回归与干净构建是不同证据，见 [安装](installation.md)。
3. 对最终源码包和 Git 历史执行敏感信息检查，并核对第三方许可证与来源。
4. 补全公开仓库地址、维护者、发布版本；发布权重时补上模型链接及相应许可和数据审查。

未完成项保留在 [发行清单](release.md)，不通过修改状态或放宽比较阈值跳过。
