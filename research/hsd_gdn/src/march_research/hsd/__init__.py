from __future__ import annotations

import sys

from transformers import AutoConfig

sys.modules.setdefault("hsd", sys.modules[__name__])

from .configuration_hsd_gated_deltanet import HSDGatedDeltaNetConfig  # noqa: E402

AutoConfig.register(HSDGatedDeltaNetConfig.model_type, HSDGatedDeltaNetConfig, exist_ok=True)

__all__ = ["HSDGatedDeltaNetConfig"]
