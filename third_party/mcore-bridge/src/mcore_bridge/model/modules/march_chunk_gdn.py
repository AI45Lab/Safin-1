# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron adapter to the single shared MARCH chunk-GDN implementation."""

from march.kernels.march_chunk_gdn import (
    clear_hsd_chunk_profile_stats,
    hsd_chunk_gated_delta_rule_with_anchor_states,
    pop_hsd_chunk_profile_stats,
)

__all__ = [
    "hsd_chunk_gated_delta_rule_with_anchor_states",
    "clear_hsd_chunk_profile_stats",
    "pop_hsd_chunk_profile_stats",
]
