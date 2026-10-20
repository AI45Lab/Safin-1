# MARCH regressions

Run `python -m pytest tests/march -q` in the installed HF environment. The suite
covers both Safin-1 families, training/export contracts, anchor layout and
persistent state, paged history, inference profiles and release tooling.

Set `MARCH_TEST_MEGATRON=1` with the vendored bridge on `PYTHONPATH` for the
Megatron contracts. CUDA checks skip when no GPU is available; report those
skips separately. See [validation](../docs/march/validation.md) for GPU evidence.
Unrelated upstream download/integration tests are omitted from this fork.
