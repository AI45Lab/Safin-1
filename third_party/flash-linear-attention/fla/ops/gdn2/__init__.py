# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA Source Code License-NC included in this directory.

from .naive import naive_recurrent_gdn2

try:
    from .chunk import chunk_gdn2
    from .fused_recurrent import fused_recurrent_gdn2
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    chunk_gdn2 = None
    fused_recurrent_gdn2 = None

__all__ = [
    "chunk_gdn2",
    "fused_recurrent_gdn2",
    "naive_recurrent_gdn2",
]
