# Vendored FlashQLA Hopper kernels

This directory contains the Hopper implementation from QwenLM/FlashQLA,
pinned at commit `821fd9d37ede18fdc2a4e707fefe3770bfc32e58`.

It is used only by the opt-in HSD FlashQLA local recurrent-scan training
backend. Local changes are documented in the source and in the HSD backend
wrapper.

Current integration status:

- Hopper forward and KKT solve accept the Safin-1 H16/K128/V128 layout. The
  admitted H16/K96/V96 profile is zero-padded internally for the forward pass.
- Packed producer forward returns local output plus chunk states; HSD selects
  only the chunk-512 boundary states for anchor/global read.
- Backward uses HSD's optimized reverse scan by default, which injects every
  memory-anchor boundary's `dH`. Set
  `HSD_FLASHQLA_TILE_REFERENCE_BWD=1` for the slow correctness oracle.
- The unfinished Direct-A/native/split backward prototypes and dense TileLang
  global-read experiment are excluded from the Safin-1 release tree.

Upstream: https://github.com/QwenLM/FlashQLA

FlashQLA is distributed under the MIT License. Copyright (c) 2026 Qwen,
Alibaba. See `LICENSE` in this directory.
