# 贡献指南

请在托管当前 checkout 的 MARCH / Safin-1 仓库提交 issue 或 PR。
本 fork 的维护入口与上游 ms-swift 分开；许可证与 attribution 继续保留。

问题报告包含：模型版本、源码 revision、backend、启动命令、环境检查报告和最小复现。
公开日志前去除个人路径、内部数据集地址和凭据。不要把 skipped 测试写成通过。

```bash
bash -n scripts/march/*.sh
PYTHONPATH="$PWD" python -m pytest tests/march -q
python scripts/march/check_public_release.py .
python scripts/march/check_vendor_manifest.py
```

训练变更补 Megatron contract 和训练—保存—加载回归；算子与缓存变更补 GPU
精度、反向梯度和 anchor 边界回归。第三方改动同步更新 UPSTREAM.md 与文件 hash。
详细要求见[英文指南](CONTRIBUTING.md)。
