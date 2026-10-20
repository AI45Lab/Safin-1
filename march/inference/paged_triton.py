"""MARCH paged history kernel compatibility exports; implementations live in focused modules."""

from __future__ import annotations

try:
    import triton
except ImportError:
    triton = None

if triton is not None:
    from .paged_topk_triton import (
        _paged_decode_top4_select_kernel,
        _paged_decode_top4_read_kernel,
    )
    from .paged_memory_triton import (
        _paged_decode_memory_top4_select_kernel,
        _paged_decode_memory_top4_read_kernel,
    )
    from .paged_prefill_triton import (
        _paged_prefill_top4_blocked_read_kernel,
        _paged_prefill_top4_selected_blocked_read_kernel,
    )
    from .paged_fused_triton import (
        _paged_decode_top4_fused_read_kernel,
        _paged_decode_read_all_fused_kernel,
    )
    from .paged_epilogue_triton import (
        _history_scale_add_kernel,
        _valid_mask_copy_kernel,
        _strided_gate_rmsnorm_kernel,
    )
