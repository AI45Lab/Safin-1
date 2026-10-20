# Vendored FlashQLA Hopper kernels

This directory contains the Hopper implementation from QwenLM/FlashQLA,
pinned at commit `821fd9d37ede18fdc2a4e707fefe3770bfc32e58`.

It is used only by the opt-in HSD FlashQLA/TileLang training backend. The
default HSD implementation remains unchanged. Local changes are documented in
the source and in the HSD backend wrapper.

Current integration status:

- Hopper forward and KKT solve accept the HSD `DK=192, DV=384` layout.
- Packed producer forward returns local output plus chunk states; HSD selects
  only the chunk-512 boundary states for summary/global read.
- Backward currently reuses HSD's optimized reverse scan, which already
  injects every summary boundary's `dH`. Set
  `HSD_FLASHQLA_TILE_REFERENCE_BWD=1` for the slow correctness oracle.
- `HSD_FUSED_GLOBAL_READ_BACKEND=tile_online` selects the grouped TileLang
  online-softmax forward. Its backward recomputes routing probabilities and
  reuses the validated HSD gradient kernels.

These fallbacks are intentional while numerical and performance parity are
being established. Neither experimental path is the default backend.

Upstream: https://github.com/QwenLM/FlashQLA

FlashQLA is distributed under the MIT License. Copyright (c) 2026 Qwen,
Alibaba. See `LICENSE` in this directory.
