# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared optional FLA/Triton dependencies for MARCH chunk recurrence."""

from __future__ import annotations

try:
    import triton
    import triton.language as tl

    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
    from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.op import exp2
    from fla.utils import custom_device_ctx
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    triton = None
    tl = None
    chunk_gated_delta_rule_fwd_h = None
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64 = None
    chunk_bwd_dqkwg = None
    chunk_bwd_dv_local = None
    chunk_fwd_o = None
    chunk_gated_delta_rule = None
    chunk_gated_delta_rule_fwd_intra = None
    chunk_gated_delta_rule_fwd_kkt_solve_kernel = None
    prepare_wy_repr_bwd = None
    recompute_w_u_fwd = None
    chunk_local_cumsum = None
    prepare_chunk_indices = None
    prepare_chunk_offsets = None
    RCP_LN2 = None
    exp2 = None
    custom_device_ctx = None

# This private FLA symbol is needed only by the optimized compact backward.
# Keep it out of the broad dependency import so an older FLA build does not
# disable the established reference paths at module import time.
if triton is not None:
    try:
        from fla.ops.gated_delta_rule.chunk_fwd import (
            chunk_gated_delta_rule_fwd_kkt_solve_kernel,
        )
    except (ImportError, AttributeError):  # pragma: no cover - FLA version dependent.
        chunk_gated_delta_rule_fwd_kkt_solve_kernel = None


FLA_CHUNK_SIZE = 64
