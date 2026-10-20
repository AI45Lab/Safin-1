# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One owner for optional MARCH backend imports and cached failures.

Optional loaders keep their original success/error memoization. Mandatory paged
loaders propagate import errors; do not silently turn those into fallback paths.
No backend is imported by a lazy loader until its first call.
"""

from __future__ import annotations

try:
    from causal_conv1d import causal_conv1d_fn as reference_causal_conv1d_fn
except Exception:
    reference_causal_conv1d_fn = None


_HSD_VARLEN_PREFILL = None
_HSD_VARLEN_PREFILL_IMPORT_ERROR: Exception | None = None
_HSD_FLASHQLA_PREFILL = None
_HSD_FLASHQLA_PREFILL_IMPORT_ERROR: Exception | None = None
_HSD_DETERMINISTIC_ANCHOR_READ = None
_HSD_DETERMINISTIC_ANCHOR_READ_IMPORT_ERROR: Exception | None = None
_HSD_PAGED_HISTORY_READ = None
_HSD_PAGED_HISTORY_READ_IMPORT_ERROR: Exception | None = None
_HSD_PAGED_STATIC_HISTORY_READ = None
_HSD_PAGED_BLOCKED_PREFILL_HISTORY_READ = None
_HSD_FUSED_GLOBAL_READ = None
_HSD_FUSED_GLOBAL_READ_IMPORT_ERROR: Exception | None = None
_HSD_HISTORY_SCALE_ADD = None
_HSD_VALID_MASK_COPY = None
_HSD_STRIDED_GATE_RMSNORM = None


def _load_hsd_varlen_prefill():
    global _HSD_VARLEN_PREFILL, _HSD_VARLEN_PREFILL_IMPORT_ERROR
    if _HSD_VARLEN_PREFILL is not None:
        return _HSD_VARLEN_PREFILL
    if _HSD_VARLEN_PREFILL_IMPORT_ERROR is not None:
        return None
    try:
        from march.inference.varlen_segments import (
            chunk_gated_delta_rule_with_anchor_states_varlen_forward,
        )
    except Exception as exc:
        _HSD_VARLEN_PREFILL_IMPORT_ERROR = exc
        return None
    _HSD_VARLEN_PREFILL = chunk_gated_delta_rule_with_anchor_states_varlen_forward
    return _HSD_VARLEN_PREFILL


def _load_hsd_flashqla_prefill():
    global _HSD_FLASHQLA_PREFILL, _HSD_FLASHQLA_PREFILL_IMPORT_ERROR
    if _HSD_FLASHQLA_PREFILL is not None:
        return _HSD_FLASHQLA_PREFILL
    if _HSD_FLASHQLA_PREFILL_IMPORT_ERROR is not None:
        return None
    try:
        from march.kernels.march_chunk_gdn import (
            hsd_flashqla_tile_gated_delta_rule_inference,
        )
    except Exception as exc:
        _HSD_FLASHQLA_PREFILL_IMPORT_ERROR = exc
        return None
    _HSD_FLASHQLA_PREFILL = hsd_flashqla_tile_gated_delta_rule_inference
    return _HSD_FLASHQLA_PREFILL


def _load_hsd_deterministic_anchor_read():
    global _HSD_DETERMINISTIC_ANCHOR_READ
    global _HSD_DETERMINISTIC_ANCHOR_READ_IMPORT_ERROR
    if _HSD_DETERMINISTIC_ANCHOR_READ is not None:
        return _HSD_DETERMINISTIC_ANCHOR_READ
    if _HSD_DETERMINISTIC_ANCHOR_READ_IMPORT_ERROR is not None:
        return None
    try:
        from march.inference.varlen_segments import (
            deterministic_anchor_state_read,
        )
    except Exception as exc:
        _HSD_DETERMINISTIC_ANCHOR_READ_IMPORT_ERROR = exc
        return None
    _HSD_DETERMINISTIC_ANCHOR_READ = deterministic_anchor_state_read
    return _HSD_DETERMINISTIC_ANCHOR_READ


def _load_hsd_paged_history_read():
    global _HSD_PAGED_HISTORY_READ, _HSD_PAGED_HISTORY_READ_IMPORT_ERROR
    if _HSD_PAGED_HISTORY_READ is not None:
        return _HSD_PAGED_HISTORY_READ
    if _HSD_PAGED_HISTORY_READ_IMPORT_ERROR is not None:
        return None
    try:
        from march.inference.paged_history_read import (
            HSDAnchorPageCache,
            HSDHistoryCacheSpec,
            paged_batched_decode_history_read,
        )
    except Exception as exc:
        _HSD_PAGED_HISTORY_READ_IMPORT_ERROR = exc
        return None
    _HSD_PAGED_HISTORY_READ = (
        HSDHistoryCacheSpec,
        HSDAnchorPageCache,
        paged_batched_decode_history_read,
    )
    return _HSD_PAGED_HISTORY_READ


def _load_hsd_paged_static_history_read():
    global _HSD_PAGED_STATIC_HISTORY_READ
    if _HSD_PAGED_STATIC_HISTORY_READ is None:
        from march.inference.paged_history_read import (
            paged_static_decode_history_read,
        )

        _HSD_PAGED_STATIC_HISTORY_READ = paged_static_decode_history_read
    return _HSD_PAGED_STATIC_HISTORY_READ


def _load_hsd_paged_blocked_prefill_history_read():
    global _HSD_PAGED_BLOCKED_PREFILL_HISTORY_READ
    if _HSD_PAGED_BLOCKED_PREFILL_HISTORY_READ is None:
        from march.inference.paged_history_read import (
            paged_blocked_prefill_history_read,
        )

        _HSD_PAGED_BLOCKED_PREFILL_HISTORY_READ = paged_blocked_prefill_history_read
    return _HSD_PAGED_BLOCKED_PREFILL_HISTORY_READ


def _load_hsd_fused_global_read():
    global _HSD_FUSED_GLOBAL_READ, _HSD_FUSED_GLOBAL_READ_IMPORT_ERROR
    if _HSD_FUSED_GLOBAL_READ is not None:
        return _HSD_FUSED_GLOBAL_READ
    if _HSD_FUSED_GLOBAL_READ_IMPORT_ERROR is not None:
        return None
    try:
        from march.kernels.march_global_read import hsd_fused_global_read
    except Exception as exc:
        _HSD_FUSED_GLOBAL_READ_IMPORT_ERROR = exc
        return None
    _HSD_FUSED_GLOBAL_READ = hsd_fused_global_read
    return _HSD_FUSED_GLOBAL_READ


def _load_hsd_history_scale_add():
    global _HSD_HISTORY_SCALE_ADD
    if _HSD_HISTORY_SCALE_ADD is not None:
        return _HSD_HISTORY_SCALE_ADD
    try:
        from march.inference.paged_history_read import (
            fused_history_scale_add_,
        )
    except Exception:
        return None
    _HSD_HISTORY_SCALE_ADD = fused_history_scale_add_
    return _HSD_HISTORY_SCALE_ADD


def _load_hsd_valid_mask_copy():
    global _HSD_VALID_MASK_COPY
    if _HSD_VALID_MASK_COPY is None:
        from march.inference.paged_history_read import (
            fused_valid_mask_copy_,
        )

        _HSD_VALID_MASK_COPY = fused_valid_mask_copy_
    return _HSD_VALID_MASK_COPY


def _load_hsd_strided_gate_rmsnorm():
    global _HSD_STRIDED_GATE_RMSNORM
    if _HSD_STRIDED_GATE_RMSNORM is None:
        from march.inference.paged_history_read import (
            fused_strided_gate_rmsnorm,
        )

        _HSD_STRIDED_GATE_RMSNORM = fused_strided_gate_rmsnorm
    return _HSD_STRIDED_GATE_RMSNORM
