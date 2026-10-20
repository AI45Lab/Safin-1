"""Public paged-history API; execution lives in focused prefill/decode modules."""

from __future__ import annotations

import os
import time
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU reference remains available without Triton.
    triton = None
    tl = None
from .paged_utils import (
    _cuda_elapsed_us,
    _env_truthy,
    _history_read_impl,
    _history_read_num_warps,
    _next_power_of_2,
    _normalize_value_block_size,
    _paged_history_num_warps,
    _paged_history_timing_enabled,
    _record_paged_history_event,
    _record_prefill_top4_locality,
    _value_block_size,
)
from .history_cache import (
    HSDAnchorPageCache,
    HSDHistoryCacheManager,
    HSDHistoryCacheSpec,
)
from .paged_reference import (
    _paged_decode_read_all_history_read_torch,
    _paged_decode_top4_history_read_torch,
    _paged_decode_top4_memory_history_read_torch,
    _paged_dynamic_state_rms_median,
    _record_paged_memory_metrics_reference,
    _record_pre_history_residual_rms,
)
if triton is not None:
    from .paged_triton import (
        _history_scale_add_kernel,
        _paged_decode_memory_top4_read_kernel,
        _paged_decode_memory_top4_select_kernel,
        _paged_decode_read_all_fused_kernel,
        _paged_decode_top4_fused_read_kernel,
        _paged_decode_top4_read_kernel,
        _paged_decode_top4_select_kernel,
        _paged_prefill_top4_blocked_read_kernel,
        _paged_prefill_top4_selected_blocked_read_kernel,
        _strided_gate_rmsnorm_kernel,
        _valid_mask_copy_kernel,
    )


from .paged_epilogue import (
    fused_history_scale_add_,
    fused_valid_mask_copy_,
    fused_strided_gate_rmsnorm,
)
from .paged_memory import (
    _launch_paged_decode_memory_top4_dense,
)
from .paged_static import (
    paged_static_decode_history_read,
)
from .paged_prefill import (
    paged_blocked_prefill_history_read,
)
from .paged_decode import (
    paged_batched_decode_history_read,
)

__all__ = [
    "HSDHistoryCacheSpec", "HSDHistoryCacheManager", "HSDAnchorPageCache",
    "paged_batched_decode_history_read", "paged_static_decode_history_read",
    "paged_blocked_prefill_history_read", "fused_history_scale_add_",
    "fused_valid_mask_copy_", "fused_strided_gate_rmsnorm",
]
