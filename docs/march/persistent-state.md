# Persistent state

Persistent state 是 MARCH 的可选模型状态。标准 Safin-1 checkpoint 默认关闭；
只有使用对应 runtime 字段训练的 checkpoint 才应启用。它与 vLLM Graph 的请求
sidecar cache 是两个不同概念。

训练配置至少需要：

```json
{
  "hsd_persistent_memory_enabled": true,
  "hsd_persistent_memory_route_key_enabled": true,
  "hsd_persistent_memory_route_key_mode": "dedicated"
}
```

将这些字段加入模型对应的 runtime 配置副本，通过
`MARCH_RUNTIME_CONFIG=/path/to/runtime.json` 启动训练。不要修改标准
`configs/march/safin1_*.yaml`，避免把实验状态误设为默认值。

HF 模型可单独保存和加载 persistent state：

```python
from transformers import AutoModelForCausalLM
from march.safin1.march_gdn import (
    load_persistent_memory_artifact,
    save_persistent_memory_artifact,
)

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/persistent-safin1",
    trust_remote_code=True,
    local_files_only=True,
)
save_persistent_memory_artifact(model, "/path/to/state-artifact")
load_persistent_memory_artifact(model, "/path/to/state-artifact", strict=True)
```

保存目录包含 `persistent_memory.safetensors` 和
`persistent_memory_config.json`。加载时会检查 tensor 名称和形状；标准 checkpoint
没有对应参数时会明确失败。共享 artifact 时必须同时记录基础 checkpoint、runtime
配置、训练步数及是否使用 dedicated route key。


独立 artifact 的加载会先校验所有匹配参数的键名、形状、有限浮点值和
设备/类型转换，再更新模型；这些检查失败时参数保持原值。
新保存的 sidecar 包含模型类型与 tensor 文件校验值，加载会检查 metadata
与 tensor 内容是否一致。两个输出文件分别使用原子替换，校验值可识别
保存中断或并发保存留下的不一致文件对。旧 sidecar 和仅含 tensor 的
artifact 继续支持，`strict=False` 仍允许只加载匹配的部分参数。


## 并发与缓存隔离验证（2026-10-16）

CPU/PyTorch reference 回归 **8 项全部通过**：Dense/MoE 两个模型 fixture，
分别覆盖 batch 1/4/8/32。测试使用非零 persistent state 和 dedicated route key，
并通过将 memory state 置零的独立模型确认原 fixture 实际使用了 memory read。

A 请求保留自己的 cache 时，先运行另一批 B 请求，再复制/重排 A 的缓存行，
继续跨 anchor decode，裁剪缓存后再插入另一批请求。每个保留请求的输出均与
独立 full-forward 参考保持一致，沿用 `atol=2e-5, rtol=2e-4`；每行 token
计数正确，persistent state 与 route key 参数在推理前后完全一致。

```bash
PYTHONPATH="$PWD" python -m pytest tests/march/test_persistent_batch_isolation.py -q
```

完整模型的并发配置与当前验证进度见[验证说明](validation.md#numerical-limits)。
