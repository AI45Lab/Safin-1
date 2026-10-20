# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler text/anchor layouts, including fixed-shape decode and memory anchors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .runtime import (
    _env_truthy,
    _hsd_history_available,
    _is_valid_hsd_state_index,
    _persistent_memory_insert_enabled,
)

if TYPE_CHECKING:
    from vllm.transformers_utils.configs.safin1_moe import Safin1MoeTextConfig


@dataclass
class HistoricalHSDLayout:
    """Packed anchor-token expansion plan for one vLLM forward step."""

    text_to_expanded: torch.Tensor
    expanded_to_text: torch.Tensor
    anchor_mask: torch.Tensor
    persistent_memory_mask: torch.Tensor
    anchor_ids: torch.Tensor
    anchor_boundaries: torch.Tensor
    anchor_text_indices: torch.Tensor
    anchor_positions: torch.Tensor
    positions: torch.Tensor
    req_text_starts: list[int]
    req_text_lens: list[int]
    req_expanded_starts: list[int]
    req_expanded_lens: list[int]
    req_anchor_starts: list[int]
    req_anchor_lens: list[int]
    past_text_lens: list[int]
    state_indices: torch.Tensor | None
    state_indices_list: list[int]
    history_state_indices: torch.Tensor | None
    history_state_indices_list: list[int] | None
    has_anchor_tokens_value: bool
    decode_valid_mask: torch.Tensor | None = None
    decode_valid_flags: list[bool] | None = None
    decode_valid_count_value: int | None = None
    past_text_lens_tensor: torch.Tensor | None = None
    decode_history_valid_mask: torch.Tensor | None = None
    decode_history_valid_flags: list[bool] | None = None
    decode_history_valid_count_value: int | None = None
    decode_history_candidate_indices: torch.Tensor | None = None
    decode_history_candidate_indices_list: list[int] | None = None
    decode_history_max_anchor_len_value: int | None = None
    history_state_indices_key: tuple[int, ...] | None = None
    decode_only_no_anchor_value: bool = False
    padded_static_decode: bool = False
    hsd_prefix_snapshot_requests: list[list[tuple[str, int, int]]] = field(
        default_factory=list
    )
    hsd_prefix_restored_flags: list[bool] = field(default_factory=list)

    @property
    def num_text_tokens(self) -> int:
        return int(self.text_to_expanded.numel())

    @property
    def num_expanded_tokens(self) -> int:
        return int(self.anchor_mask.numel())

    @property
    def has_anchor_tokens(self) -> bool:
        return self.has_anchor_tokens_value

    def text_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.index_select(0, self.text_to_expanded)

    def text_positions(self) -> torch.Tensor:
        return self.positions.index_select(0, self.text_to_expanded)

    def scatter_text_hidden(
        self,
        base_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if text_hidden_states.shape[0] != self.num_text_tokens:
            raise ValueError(
                "HSD text hidden shape mismatch: "
                f"got {tuple(text_hidden_states.shape)}, "
                f"expected first dim {self.num_text_tokens}."
            )
        output = base_hidden_states.clone()
        output.index_copy_(0, self.text_to_expanded, text_hidden_states)
        return output


@dataclass
class HSDPreparedBatch:
    text_offsets: list[int]
    anchor_offsets: list[int]
    state_indices: list[int]
    past_text_lens: list[int]
    boundary_counts: list[torch.Tensor]


def _to_cpu_int_list(
    tensor: torch.Tensor | None, length: int | None = None
) -> list[int]:
    if tensor is None:
        return []
    values = tensor.detach().cpu().tolist()
    if length is not None:
        values = values[:length]
    return [int(v) for v in values]


def _text_only_hsd_positions(positions: torch.Tensor) -> torch.Tensor:
    if positions.ndim == 1:
        return positions
    if positions.ndim == 2 and positions.shape[0] == 3:
        text_positions = positions[0]
        if torch.compiler.is_compiling() or _env_truthy(
            "VLLM_HSD_GRAPH_V3_FULL_CUDAGRAPH"
        ):
            return text_positions
        if torch.equal(text_positions, positions[1]) and torch.equal(
            text_positions, positions[2]
        ):
            return text_positions
        raise NotImplementedError(
            "Safin-1 vLLM HSD path currently supports text-only "
            "M-RoPE position tensors with identical temporal/height/width rows."
        )
    raise NotImplementedError(
        "Safin-1 vLLM HSD path currently supports text-only "
        "1-D position tensors or text-only M-RoPE tensors shaped (3, seq_len)."
    )


def _build_hsd_layout(
    *,
    config: Safin1MoeTextConfig,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    history_state_indices: torch.Tensor | None = None,
    hsd_prefix_snapshot_requests: list[list[tuple[str, int, int]]] | None = None,
    hsd_prefix_restored_flags: list[bool] | None = None,
) -> HistoricalHSDLayout:
    chunk_size = int(config.anchor_chunk_size)
    anchor_token_num = int(config.anchor_token_num)
    if chunk_size <= 0 or anchor_token_num <= 0:
        raise ValueError("HSD layout requires positive anchor_chunk_size/token_num.")
    if positions.ndim != 1:
        positions = _text_only_hsd_positions(positions)

    qsl = _to_cpu_int_list(query_start_loc)
    if len(qsl) < 2:
        raise ValueError("HSD layout requires query_start_loc with at least 2 items.")
    num_reqs = len(qsl) - 1
    prefix_restored_flags = list(hsd_prefix_restored_flags or [])
    if prefix_restored_flags and len(prefix_restored_flags) != num_reqs:
        raise ValueError(
            "HSD layout requires prefix restore flags for every request: "
            f"got {len(prefix_restored_flags)}, expected {num_reqs}."
        )
    if not prefix_restored_flags:
        prefix_restored_flags = [False] * num_reqs
    past_text_lens = _to_cpu_int_list(num_computed_tokens, num_reqs)
    if len(past_text_lens) != num_reqs:
        raise ValueError(
            "HSD layout requires num_computed_tokens for every request: "
            f"got {len(past_text_lens)}, expected {num_reqs}."
        )

    expanded_to_text: list[int] = []
    text_to_expanded: list[int] = []
    anchor_mask: list[bool] = []
    persistent_memory_mask: list[bool] = []
    anchor_ids: list[int] = []
    anchor_boundaries: list[int] = []
    anchor_text_indices: list[int] = []
    anchor_position_values: list[int] = []
    req_text_starts: list[int] = []
    req_text_lens: list[int] = []
    req_expanded_starts: list[int] = []
    req_expanded_lens: list[int] = []
    req_anchor_starts: list[int] = []
    req_anchor_lens: list[int] = []
    expanded_position_values: list[int] = []
    anchor_position_type = str(getattr(config, "anchor_position_ids_type", "zeros"))
    if anchor_position_type not in {"zeros", "chunk_end"}:
        raise ValueError(
            "anchor_position_ids_type must be either 'zeros' or 'chunk_end'."
        )

    positions_cpu = _to_cpu_int_list(positions)
    if state_indices is None:
        state_indices_list = list(range(num_reqs))
    else:
        state_indices_list = _to_cpu_int_list(state_indices, num_reqs)
        if len(state_indices_list) != num_reqs:
            raise ValueError(
                "HSD layout requires state_indices for every request: "
                f"got {len(state_indices_list)}, expected {num_reqs}."
            )
    if history_state_indices is None:
        history_state_indices_list = None
    else:
        history_state_indices_list = _to_cpu_int_list(history_state_indices, num_reqs)
        if len(history_state_indices_list) != num_reqs:
            raise ValueError(
                "HSD layout requires history_state_indices for every request: "
                f"got {len(history_state_indices_list)}, expected {num_reqs}."
            )
    for req_idx in range(num_reqs):
        text_start = qsl[req_idx]
        text_end = qsl[req_idx + 1]
        text_len = text_end - text_start
        if text_len < 0:
            raise ValueError(f"Invalid query_start_loc: {qsl}")
        past_text = past_text_lens[req_idx]

        req_text_starts.append(text_start)
        req_text_lens.append(text_len)
        req_expanded_start = len(expanded_to_text)
        req_anchor_start = len(anchor_ids)
        req_expanded_starts.append(req_expanded_start)
        req_anchor_starts.append(req_anchor_start)

        # The memory anchor is a virtual boundary-zero row.  It is inserted
        # only by the first scheduler chunk and is deliberately not represented
        # in scheduler/KV token counts or past_text_lens.
        if (
            _persistent_memory_insert_enabled(config)
            and past_text == 0
            and text_len > 0
        ):
            anchor_position_values.append(len(expanded_to_text))
            expanded_to_text.append(-1)
            anchor_mask.append(True)
            persistent_memory_mask.append(True)
            anchor_ids.append(0)
            anchor_boundaries.append(0)
            anchor_text_indices.append(text_start)
            expanded_position_values.append(0)

        for local_idx in range(text_len):
            text_idx = text_start + local_idx
            text_to_expanded.append(len(expanded_to_text))
            expanded_to_text.append(text_idx)
            anchor_mask.append(False)
            persistent_memory_mask.append(False)
            expanded_position_values.append(positions_cpu[text_idx])

            abs_text_count = past_text + local_idx + 1
            if abs_text_count % chunk_size != 0:
                continue

            anchor_position = positions_cpu[text_idx]
            for anchor_id in range(anchor_token_num):
                anchor_position_values.append(len(expanded_to_text))
                expanded_to_text.append(-1)
                anchor_mask.append(True)
                persistent_memory_mask.append(False)
                anchor_ids.append(anchor_id)
                anchor_boundaries.append(abs_text_count)
                anchor_text_indices.append(text_idx)
                expanded_position_values.append(
                    anchor_position if anchor_position_type == "chunk_end" else 0
                )

        req_expanded_lens.append(len(expanded_to_text) - req_expanded_start)
        req_anchor_lens.append(len(anchor_ids) - req_anchor_start)

    device = positions.device
    decode_valid_flags = [
        _is_valid_hsd_state_index(state_idx) for state_idx in state_indices_list
    ]
    decode_history_valid_flags = [
        valid and _hsd_history_available(config, past_text_len, chunk_size)
        for valid, past_text_len in zip(
            decode_valid_flags,
            past_text_lens,
            strict=True,
        )
    ]
    decode_history_candidate_values = [
        req_idx for req_idx, valid in enumerate(decode_history_valid_flags) if valid
    ]
    return HistoricalHSDLayout(
        text_to_expanded=torch.tensor(
            text_to_expanded, dtype=torch.long, device=device
        ),
        expanded_to_text=torch.tensor(
            expanded_to_text, dtype=torch.long, device=device
        ),
        anchor_mask=torch.tensor(anchor_mask, dtype=torch.bool, device=device),
        persistent_memory_mask=torch.tensor(
            persistent_memory_mask,
            dtype=torch.bool,
            device=device,
        ),
        anchor_ids=torch.tensor(anchor_ids, dtype=torch.long, device=device),
        anchor_boundaries=torch.tensor(
            anchor_boundaries, dtype=torch.long, device=device
        ),
        anchor_text_indices=torch.tensor(
            anchor_text_indices, dtype=torch.long, device=device
        ),
        anchor_positions=torch.tensor(
            anchor_position_values, dtype=torch.long, device=device
        ),
        positions=torch.tensor(
            expanded_position_values, dtype=positions.dtype, device=device
        ),
        req_text_starts=req_text_starts,
        req_text_lens=req_text_lens,
        req_expanded_starts=req_expanded_starts,
        req_expanded_lens=req_expanded_lens,
        req_anchor_starts=req_anchor_starts,
        req_anchor_lens=req_anchor_lens,
        past_text_lens=past_text_lens,
        state_indices=state_indices,
        state_indices_list=state_indices_list,
        history_state_indices=history_state_indices,
        history_state_indices_list=history_state_indices_list,
        has_anchor_tokens_value=bool(anchor_ids),
        decode_valid_flags=decode_valid_flags,
        decode_valid_count_value=sum(decode_valid_flags),
        past_text_lens_tensor=num_computed_tokens[:num_reqs].to(
            device=device,
            dtype=torch.long,
        ),
        decode_history_valid_mask=torch.tensor(
            decode_history_valid_flags,
            dtype=torch.bool,
            device=device,
        ),
        decode_history_valid_flags=decode_history_valid_flags,
        decode_history_valid_count_value=sum(decode_history_valid_flags),
        decode_history_candidate_indices=torch.tensor(
            decode_history_candidate_values,
            dtype=torch.long,
            device=device,
        ),
        decode_history_candidate_indices_list=decode_history_candidate_values,
        decode_history_max_anchor_len_value=max(
            (
                int(past_text_len) // chunk_size
                + int(
                    _persistent_memory_insert_enabled(config) and int(past_text_len) > 0
                )
                for past_text_len, valid in zip(
                    past_text_lens,
                    decode_history_valid_flags,
                    strict=True,
                )
                if valid
            ),
            default=0,
        ),
        history_state_indices_key=(
            None
            if history_state_indices_list is None
            else tuple(int(value) for value in history_state_indices_list)
        ),
        decode_only_no_anchor_value=(
            num_reqs > 0
            and not anchor_ids
            and all(
                int(text_len) == 1 and int(past_text_len) > 0
                for text_len, past_text_len in zip(
                    req_text_lens,
                    past_text_lens,
                    strict=True,
                )
            )
        ),
        hsd_prefix_snapshot_requests=hsd_prefix_snapshot_requests or [],
        hsd_prefix_restored_flags=prefix_restored_flags,
    )


def _build_hsd_padded_static_decode_layout(
    *,
    config: Safin1MoeTextConfig,
    positions: torch.Tensor,
    hsd_num_computed_tokens: torch.Tensor,
    hsd_state_indices: torch.Tensor,
    hsd_history_state_indices: torch.Tensor | None,
    hsd_decode_valid_mask: torch.Tensor,
) -> HistoricalHSDLayout:
    if positions.ndim != 1:
        positions = _text_only_hsd_positions(positions)
    req_count = int(positions.shape[0])
    device = positions.device
    text_to_expanded = torch.arange(req_count, dtype=torch.long, device=device)
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    state_indices = hsd_state_indices.to(device=device, dtype=torch.int32)
    state_indices_list = _to_cpu_int_list(state_indices, req_count)
    past_text_lens_tensor = hsd_num_computed_tokens.to(
        device=device,
        dtype=torch.long,
    )
    past_text_lens = _to_cpu_int_list(past_text_lens_tensor, req_count)
    decode_valid_flags = [
        _is_valid_hsd_state_index(state_idx) for state_idx in state_indices_list
    ]
    anchor_chunk_size = int(getattr(config, "anchor_chunk_size", 0) or 0)
    decode_history_valid_flags = [
        valid
        and _hsd_history_available(
            config,
            past_text_len,
            anchor_chunk_size,
        )
        for valid, past_text_len in zip(
            decode_valid_flags,
            past_text_lens,
            strict=True,
        )
    ]
    decode_history_valid_mask = torch.tensor(
        decode_history_valid_flags,
        dtype=torch.bool,
        device=device,
    )
    history_state_indices = hsd_history_state_indices.to(
        device=device,
        dtype=torch.int32,
    )
    return HistoricalHSDLayout(
        text_to_expanded=text_to_expanded,
        expanded_to_text=text_to_expanded,
        anchor_mask=torch.zeros(req_count, dtype=torch.bool, device=device),
        persistent_memory_mask=torch.zeros(
            req_count,
            dtype=torch.bool,
            device=device,
        ),
        anchor_ids=empty_long,
        anchor_boundaries=empty_long,
        anchor_text_indices=empty_long,
        anchor_positions=empty_long,
        positions=positions,
        req_text_starts=list(range(req_count)),
        req_text_lens=[1] * req_count,
        req_expanded_starts=list(range(req_count)),
        req_expanded_lens=[1] * req_count,
        req_anchor_starts=[0] * req_count,
        req_anchor_lens=[0] * req_count,
        past_text_lens=past_text_lens,
        state_indices=state_indices,
        state_indices_list=state_indices_list,
        history_state_indices=history_state_indices,
        history_state_indices_list=_to_cpu_int_list(
            history_state_indices,
            req_count,
        ),
        has_anchor_tokens_value=False,
        decode_valid_mask=hsd_decode_valid_mask.to(device=device, dtype=torch.bool),
        decode_valid_flags=decode_valid_flags,
        decode_valid_count_value=sum(decode_valid_flags),
        past_text_lens_tensor=past_text_lens_tensor,
        decode_history_valid_mask=decode_history_valid_mask,
        decode_history_valid_flags=decode_history_valid_flags,
        decode_history_valid_count_value=sum(decode_history_valid_flags),
        decode_history_candidate_indices=torch.tensor(
            [
                req_idx
                for req_idx, valid in enumerate(decode_history_valid_flags)
                if valid
            ],
            dtype=torch.long,
            device=device,
        ),
        decode_history_candidate_indices_list=[
            req_idx for req_idx, valid in enumerate(decode_history_valid_flags) if valid
        ],
        padded_static_decode=True,
    )
