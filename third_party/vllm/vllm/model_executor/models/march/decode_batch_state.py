# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed phase snapshots for eager batched decode.

Frozen fields hold the original objects, never copies or casts. Tensor/cache
contents may change in place at their original execution point. These objects
are local to one forward; they do not own parameters, slots or persistent state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .layout import HistoricalHSDLayout

if TYPE_CHECKING:
    from march.inference.paged_history_read import HSDAnchorPageCache


@dataclass(frozen=True, slots=True)
class DecodeBatch:
    hidden_states: torch.Tensor
    output: torch.Tensor
    layout: HistoricalHSDLayout
    anchor_counts_by_req: list[int]
    anchor_offsets: list[int]
    anchor_positions: torch.Tensor
    boundary_counts_by_req: list[torch.Tensor] | None
    conv_cache: torch.Tensor
    decode_valid_mask: torch.Tensor | None
    device: torch.device
    history_state_indices_tensor: torch.Tensor | None
    local_k_heads: int
    local_key_dim: int
    local_v_heads: int
    local_value_dim: int
    padded_static_decode: bool
    past_text_lens: list[int]
    past_text_lens_tensor: torch.Tensor
    qkv_size: int
    req_count: int
    ssm_state: torch.Tensor
    state_indices: list[int]
    state_indices_tensor: torch.Tensor
    total_anchor: int
    z_size: int


@dataclass(frozen=True, slots=True)
class DecodePolicy:
    global_read_needed_by_req: list[bool]
    graphable_decode_requested: bool
    needs_router_q: bool
    no_anchor_common_fast_path: bool
    no_anchor_fast_path: bool
    no_anchor_read_only_enabled: bool
    no_anchor_requests: int
    no_history_skip_requests: int
    persistent_text_cache: bool
    read_only_requests: int
    trace_requested: bool
    use_static_sidecar: bool


@dataclass(frozen=True, slots=True)
class DecodeHistorySelection:
    common_history_candidate_count: int
    common_history_candidate_state_key: tuple[int, ...] | None
    common_history_candidate_tensor: torch.Tensor | None
    common_history_use_full_batch: bool
    common_history_valid_bools: list[bool] | None
    common_history_valid_count: int
    common_history_valid_tensor: torch.Tensor | None
    common_router_project_full_batch: bool
    max_history_batch: int | None
    page_cache_for_common: HSDAnchorPageCache | None


@dataclass(frozen=True, slots=True)
class DecodeObservers:
    common_parity_debug: bool
    common_parity_debug_layers: set[int]
    decode_timing_enabled: bool
    decode_timing_payload: dict[str, int]
    fallback: Callable[[str], bool]
    mark_decode_fine_timing: Callable[[str], None]
    mark_decode_timing: Callable[[str], None]
    maybe_sync_decode_stage: Callable[[str], None]
    record_common_parity_debug: Callable[[str, torch.Tensor | None], None]
    record_no_anchor_common_fallback: Callable[..., None]


@dataclass(frozen=True, slots=True)
class DecodeCore:
    a_text: torch.Tensor
    anchor_hidden: torch.Tensor | None
    anchor_k_all: torch.Tensor | None
    anchor_o_all: torch.Tensor
    anchor_q_all: torch.Tensor | None
    b_text: torch.Tensor
    batched_anchor_boundary: bool
    batched_anchor_states_all: torch.Tensor | None
    beta: torch.Tensor | None
    core_attn_out: torch.Tensor | None
    expanded_hidden: torch.Tensor
    fused_projection: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
        | None
    )
    g: torch.Tensor | None
    history_read_query: torch.Tensor | None
    include_history_projection: bool
    key: torch.Tensor | None
    mixed_qkv_text: torch.Tensor
    null_logits_all: torch.Tensor | None
    paged_history_pending: list[
        tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor | None]
    ]
    query: torch.Tensor | None
    query_only_post_conv: bool
    raw_q_paged_read: bool
    recurrent_state_all: torch.Tensor | None
    router_q_all: torch.Tensor | None
    skip_redundant_recurrent_state_store: bool
    text_hidden: torch.Tensor | None
    text_o_all: torch.Tensor
    trace_backend: str
    trace_conv_states_before: torch.Tensor | None
    trace_pre_conv_inputs: torch.Tensor | None
    use_batched_recurrent_decode: bool
    use_text_only_no_anchor_finalize: bool
    value: torch.Tensor | None
    z_all: torch.Tensor
    z_all_flat: torch.Tensor


@dataclass(frozen=True, slots=True)
class DecodeBoundary:
    batched_anchor_boundary: bool
    batched_anchor_states_all: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class DecodeReference:
    boundary_counts_by_req: list[torch.Tensor] | None
    no_anchor_batched_history_count: int
    no_anchor_batched_history_tensor: torch.Tensor | None
    query: torch.Tensor | None
    use_no_anchor_batched_history_finalize: bool


@dataclass(frozen=True, slots=True)
class PreparedDecode:
    batch: DecodeBatch
    policy: DecodePolicy
    history: DecodeHistorySelection
    observers: DecodeObservers
