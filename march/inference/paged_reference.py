"""MARCH paged history: paged reference."""

from __future__ import annotations

import os
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU reference remains available without Triton.
    triton = None
    tl = None
from .history_cache import (
    HSDAnchorPageCache,
)


def _route_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(logits.float().masked_fill(~valid, float("-inf")), dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights * valid.to(weights.dtype)
    denom = weights.sum(dim=-1, keepdim=True)
    return torch.where(denom > 0, weights / denom, torch.zeros_like(weights))


def _paged_decode_top4_history_read_torch(
    *,
    cache: HSDAnchorPageCache,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    page_table: torch.Tensor,
    anchor_lengths: torch.Tensor,
    start_indices: torch.Tensor,
    route_scale: float,
    read_scale: float,
    null_logits: torch.Tensor | None,
) -> torch.Tensor:
    assert cache.initialized
    assert cache.keys_pages is not None
    assert cache.boundary_pages is not None
    assert cache.value_dim is not None
    batch_size, heads, key_dim = read_q.shape
    page_m = int(cache.page_m)
    max_pages = int(page_table.shape[1])
    flat_anchor_count = max_pages * page_m
    selected_count = min(4, flat_anchor_count)
    chunk_size = int(os.environ.get("HSD_PAGED_HISTORY_TORCH_BATCH_CHUNK", "8"))
    if chunk_size <= 0:
        raise ValueError(
            "HSD_PAGED_HISTORY_TORCH_BATCH_CHUNK must be positive, "
            f"got {chunk_size}."
        )
    anchor_offsets = torch.arange(flat_anchor_count, device=read_q.device).view(1, -1)
    head_range = torch.arange(heads, device=read_q.device).view(1, heads, 1)
    out = torch.empty(
        batch_size,
        heads,
        cache.value_dim,
        device=read_q.device,
        dtype=read_q.dtype,
    )

    for batch_start in range(0, batch_size, chunk_size):
        batch_end = min(batch_size, batch_start + chunk_size)
        local_batch = batch_end - batch_start
        page_table_b = page_table[batch_start:batch_end]
        safe_page_table = page_table_b.clamp_min(0)
        flat_page_ids = safe_page_table.reshape(-1)
        keys = cache.keys_pages.index_select(0, flat_page_ids).view(
            local_batch,
            max_pages,
            page_m,
            heads,
            key_dim,
        )
        boundaries = cache.boundary_pages.index_select(0, flat_page_ids).view(
            local_batch,
            max_pages,
            page_m,
        )
        keys = keys.reshape(local_batch, flat_anchor_count, heads, key_dim)
        boundaries = boundaries.reshape(local_batch, flat_anchor_count)

        page_valid = page_table_b.ge(0).unsqueeze(-1).expand(
            local_batch,
            max_pages,
            page_m,
        )
        valid = (
            page_valid.reshape(local_batch, flat_anchor_count)
            & (anchor_offsets < anchor_lengths[batch_start:batch_end].view(local_batch, 1))
            & (boundaries <= start_indices[batch_start:batch_end].view(local_batch, 1))
        )
        keys_bhmk = keys.permute(0, 2, 1, 3).reshape(
            local_batch * heads,
            flat_anchor_count,
            key_dim,
        )
        router_bhk = router_q[batch_start:batch_end].reshape(local_batch * heads, key_dim, 1)
        logits = torch.bmm(keys_bhmk.float(), router_bhk.float()).view(
            local_batch,
            heads,
            flat_anchor_count,
        )
        logits = logits * float(route_scale)
        valid_by_head = valid.unsqueeze(1).expand(local_batch, heads, flat_anchor_count)
        masked_logits = logits.masked_fill(~valid_by_head, float("-inf"))
        top_scores, top_indices = torch.topk(masked_logits, k=selected_count, dim=-1)
        top_valid = torch.isfinite(top_scores)

        top_page_pos = torch.div(top_indices, page_m, rounding_mode="floor")
        top_slots = top_indices - top_page_pos * page_m
        flat_top_page_pos = top_page_pos.reshape(local_batch, heads * selected_count).clamp(
            min=0,
            max=max(0, max_pages - 1),
        )
        selected_page_ids = page_table_b.gather(1, flat_top_page_pos).clamp_min(0).view(
            local_batch,
            heads,
            selected_count,
        )
        head_indices = head_range.expand(local_batch, heads, selected_count)
        assert cache.states_pages is not None
        selected_states = cache.states_pages[
            selected_page_ids,
            top_slots,
            head_indices,
        ]
        reads = torch.einsum(
            "bhk,bhtvk->bhtv",
            read_q[batch_start:batch_end].float(),
            selected_states.float(),
        )
        reads = reads * float(read_scale)
        reads = reads.masked_fill(~top_valid.unsqueeze(-1), 0.0)

        if null_logits is not None:
            null_logits_b = null_logits[batch_start:batch_end]
            logits_with_null = torch.cat(
                [top_scores, null_logits_b.float().unsqueeze(-1)],
                dim=-1,
            )
            valid_with_null = torch.cat(
                [
                    top_valid,
                    torch.ones(local_batch, heads, 1, device=read_q.device, dtype=torch.bool),
                ],
                dim=-1,
            )
            weights = _route_softmax(logits_with_null, valid_with_null)[..., :selected_count]
        else:
            weights = _route_softmax(top_scores, top_valid)
        out[batch_start:batch_end] = torch.einsum(
            "bht,bhtv->bhv",
            weights.to(reads.dtype),
            reads,
        ).to(dtype=read_q.dtype)
    return out


def _masked_scalar_mean(
    value: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return a finite FP32 scalar mean without a host synchronization."""
    value = value.float()
    valid = valid.to(device=value.device, dtype=torch.bool)
    count = valid.sum(dtype=torch.float32)
    total = torch.where(valid, value, torch.zeros_like(value)).sum(
        dtype=torch.float32
    )
    return torch.where(count > 0, total / count, total.new_zeros(()))


def _paged_unique_valid_dynamic_slots(
    *,
    cache: HSDAnchorPageCache,
    page_table: torch.Tensor,
    anchor_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique physical ``(page, slot)`` dynamic-cache locations.

    Prefill metadata contains one page-table row per query token, so directly
    gathering pages from every row can duplicate the same large state tensor
    thousands of times.  Valid slots form a prefix within each logical page.
    Reducing the prefix length by physical page therefore computes the union
    without materializing a ``[query_tokens, pages, page_m, ...]`` state view.
    """
    assert cache.boundary_pages is not None
    max_pages = int(page_table.shape[1])
    total_pages = int(cache.boundary_pages.shape[0])
    if (
        max_pages == 0
        or int(page_table.shape[0]) == 0
        or total_pages == 0
    ):
        empty = page_table.new_empty((0,), dtype=torch.long)
        return empty, empty

    page_m = int(cache.page_m)
    logical_page_offsets = (
        torch.arange(
            max_pages,
            device=page_table.device,
            dtype=torch.long,
        )
        * page_m
    )
    valid_slot_counts = (
        anchor_lengths.to(dtype=torch.long).view(-1, 1)
        - logical_page_offsets.view(1, -1)
    ).clamp(min=0, max=page_m)
    page_occurrence_valid = (
        page_table.ge(0)
        & page_table.lt(total_pages)
        & valid_slot_counts.gt(0)
    )
    safe_page_ids = page_table.to(dtype=torch.long).clamp(
        min=0,
        max=total_pages - 1,
    )
    valid_slot_counts = torch.where(
        page_occurrence_valid,
        valid_slot_counts,
        torch.zeros_like(valid_slot_counts),
    )

    # The union of valid slots for one physical page is a prefix whose length
    # is the maximum prefix length among its logical occurrences.
    slots_per_page = torch.zeros(
        total_pages,
        device=page_table.device,
        dtype=torch.long,
    )
    slots_per_page.scatter_reduce_(
        0,
        safe_page_ids.reshape(-1),
        valid_slot_counts.reshape(-1),
        reduce="amax",
        include_self=True,
    )
    active_page_ids = torch.nonzero(
        slots_per_page.gt(0), as_tuple=False
    ).flatten()
    if int(active_page_ids.numel()) == 0:
        empty = page_table.new_empty((0,), dtype=torch.long)
        return empty, empty

    slot_ids = torch.arange(
        page_m,
        device=page_table.device,
        dtype=torch.long,
    )
    active_boundaries = cache.boundary_pages.index_select(
        0, active_page_ids
    )
    dynamic_valid = (
        slot_ids.view(1, -1)
        < slots_per_page.index_select(0, active_page_ids).view(-1, 1)
    ) & active_boundaries.ne(0)
    expanded_page_ids = active_page_ids.view(-1, 1).expand(-1, page_m)
    expanded_slot_ids = slot_ids.view(1, -1).expand_as(expanded_page_ids)
    return (
        expanded_page_ids.masked_select(dynamic_valid),
        expanded_slot_ids.masked_select(dynamic_valid),
    )


def _paged_logical_valid_dynamic_slots(
    *,
    cache: HSDAnchorPageCache,
    page_table: torch.Tensor,
    anchor_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return valid dynamic slots while retaining logical-row multiplicity."""
    assert cache.boundary_pages is not None
    max_pages = int(page_table.shape[1])
    total_pages = int(cache.boundary_pages.shape[0])
    if (
        max_pages == 0
        or int(page_table.shape[0]) == 0
        or total_pages == 0
    ):
        empty = page_table.new_empty((0,), dtype=torch.long)
        return empty, empty
    page_m = int(cache.page_m)
    slot_ids = torch.arange(
        page_m, device=page_table.device, dtype=torch.long
    )
    logical_offsets = (
        torch.arange(
            max_pages, device=page_table.device, dtype=torch.long
        ).view(1, -1, 1)
        * page_m
        + slot_ids.view(1, 1, -1)
    )
    page_ids = page_table.to(dtype=torch.long).unsqueeze(-1).expand(
        -1, -1, page_m
    )
    valid = (
        page_ids.ge(0)
        & page_ids.lt(total_pages)
        & logical_offsets.lt(anchor_lengths.view(-1, 1, 1))
    )
    safe_page_ids = page_ids.clamp(min=0, max=total_pages - 1)
    boundaries = cache.boundary_pages[
        safe_page_ids,
        slot_ids.view(1, 1, -1),
    ]
    valid &= boundaries.ne(0)
    expanded_slot_ids = slot_ids.view(1, 1, -1).expand_as(page_ids)
    return (
        page_ids.masked_select(valid),
        expanded_slot_ids.masked_select(valid),
    )


def _paged_dynamic_state_rms_median(
    *,
    cache: HSDAnchorPageCache,
    page_table: torch.Tensor,
    anchor_lengths: torch.Tensor,
    deduplicate_physical_slots: bool = False,
) -> torch.Tensor:
    """Median RMS over cached, non-memory dynamic anchor states.

    Generic callers retain logical-row multiplicity.  Request-level audit
    callers may explicitly deduplicate physical slots; persistent-memory
    evaluation disables prefix caching, so that mode measures each actual
    cache entry exactly once.
    """
    assert cache.boundary_pages is not None
    assert cache.heads is not None
    assert cache.value_dim is not None and cache.key_dim is not None
    slot_selector = (
        _paged_unique_valid_dynamic_slots
        if deduplicate_physical_slots
        else _paged_logical_valid_dynamic_slots
    )
    dynamic_page_ids, dynamic_slot_ids = slot_selector(
        cache=cache,
        page_table=page_table,
        anchor_lengths=anchor_lengths,
    )
    dynamic_count = int(dynamic_page_ids.numel())
    if dynamic_count == 0:
        return anchor_lengths.new_zeros((), dtype=torch.float32)

    # A local TP4 state is roughly 0.5 MiB once converted to FP32.  Keep the
    # diagnostics-only reconstruction bounded independently of query length.
    state_chunk_size = 16
    rms_chunks: list[torch.Tensor] = []
    for chunk_start in range(0, dynamic_count, state_chunk_size):
        chunk_end = min(dynamic_count, chunk_start + state_chunk_size)
        page_ids = dynamic_page_ids[chunk_start:chunk_end]
        slot_ids = dynamic_slot_ids[chunk_start:chunk_end]
        assert cache.states_pages is not None
        states = cache.states_pages[page_ids, slot_ids].float()
        rms_chunks.append(
            states.square().mean(dim=(-3, -2, -1)).sqrt()
        )
    sorted_rms = torch.cat(rms_chunks, dim=0).sort().values
    return sorted_rms[(dynamic_count - 1) // 2].float()


def _paged_decode_top4_memory_history_read_torch(
    *,
    cache: HSDAnchorPageCache,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    page_table: torch.Tensor,
    anchor_lengths: torch.Tensor,
    start_indices: torch.Tensor,
    route_scale: float,
    read_scale: float,
    null_logits: torch.Tensor | None,
    memory_metrics: dict[str, torch.Tensor] | None = None,
    record_dynamic_state_rms_median: bool = True,
) -> torch.Tensor:
    """Torch reference for the fixed ``[memory, dynamic Top4, null]`` route.

    A boundary of zero identifies the single persistent-memory row.  It is
    selected independently, while all boundary-zero rows are excluded from
    dynamic Top4.  The five real candidates and the optional null candidate
    share one FP32 masked softmax.
    """
    assert cache.initialized
    assert cache.keys_pages is not None
    assert cache.boundary_pages is not None
    assert cache.value_dim is not None
    batch_size, heads, key_dim = read_q.shape
    page_m = int(cache.page_m)
    max_pages = int(page_table.shape[1])
    flat_anchor_count = max_pages * page_m
    out = read_q.new_zeros(batch_size, heads, cache.value_dim)
    if flat_anchor_count == 0:
        if memory_metrics is not None:
            zero = read_q.new_zeros((), dtype=torch.float32)
            metric_updates = dict(
                memory_logit=zero,
                memory_probability=zero,
                dynamic_probability=zero,
                null_probability=zero,
                dynamic_rank1_probability=zero,
                dynamic_rank2_probability=zero,
                dynamic_rank3_probability=zero,
                dynamic_rank4_probability=zero,
                memory_read_rms=zero,
                dynamic_read_rms=zero,
                total_history_read_rms=zero,
                state_rms=zero,
            )
            if record_dynamic_state_rms_median:
                metric_updates["dynamic_state_rms_median"] = zero
            memory_metrics.update(metric_updates)
        return out

    chunk_size = int(os.environ.get("HSD_PAGED_HISTORY_TORCH_BATCH_CHUNK", "8"))
    if chunk_size <= 0:
        raise ValueError(
            "HSD_PAGED_HISTORY_TORCH_BATCH_CHUNK must be positive, "
            f"got {chunk_size}."
        )
    anchor_offsets = torch.arange(
        flat_anchor_count, device=read_q.device
    ).view(1, -1)
    head_range = torch.arange(heads, device=read_q.device).view(1, heads, 1)
    metric_logits: list[torch.Tensor] = []
    metric_probabilities: list[torch.Tensor] = []
    metric_dynamic_probabilities: list[torch.Tensor] = []
    metric_null_probabilities: list[torch.Tensor] = []
    metric_dynamic_rank_probabilities: list[torch.Tensor] = []
    metric_reads: list[torch.Tensor] = []
    metric_dynamic_reads: list[torch.Tensor] = []
    metric_total_reads: list[torch.Tensor] = []
    metric_states: list[torch.Tensor] = []
    metric_state_valid: list[torch.Tensor] = []
    metric_memory_valid: list[torch.Tensor] = []

    for batch_start in range(0, batch_size, chunk_size):
        batch_end = min(batch_size, batch_start + chunk_size)
        local_batch = batch_end - batch_start
        page_table_b = page_table[batch_start:batch_end]
        safe_page_table = page_table_b.clamp_min(0)
        flat_page_ids = safe_page_table.reshape(-1)
        keys = cache.keys_pages.index_select(0, flat_page_ids).view(
            local_batch,
            max_pages,
            page_m,
            heads,
            key_dim,
        )
        boundaries = cache.boundary_pages.index_select(0, flat_page_ids).view(
            local_batch,
            max_pages,
            page_m,
        )
        keys = keys.reshape(local_batch, flat_anchor_count, heads, key_dim)
        boundaries = boundaries.reshape(local_batch, flat_anchor_count)

        page_valid = page_table_b.ge(0).unsqueeze(-1).expand(
            local_batch,
            max_pages,
            page_m,
        )
        valid = (
            page_valid.reshape(local_batch, flat_anchor_count)
            & (
                anchor_offsets
                < anchor_lengths[batch_start:batch_end].view(local_batch, 1)
            )
            & (
                boundaries
                <= start_indices[batch_start:batch_end].view(local_batch, 1)
            )
        )
        keys_bhmk = keys.permute(0, 2, 1, 3).reshape(
            local_batch * heads,
            flat_anchor_count,
            key_dim,
        )
        router_bhk = router_q[batch_start:batch_end].reshape(
            local_batch * heads,
            key_dim,
            1,
        )
        logits = torch.bmm(keys_bhmk.float(), router_bhk.float()).view(
            local_batch,
            heads,
            flat_anchor_count,
        )
        logits = logits * float(route_scale)
        valid_by_head = valid.unsqueeze(1).expand(
            local_batch,
            heads,
            flat_anchor_count,
        )
        memory_valid_all = valid_by_head & boundaries.eq(0).unsqueeze(1)
        # There is exactly one boundary-zero row in a well-formed cache.  Max
        # is deterministic and keeps malformed duplicate rows from entering
        # the dynamic lanes; cache-order tests enforce uniqueness separately.
        memory_score, memory_index = logits.masked_fill(
            ~memory_valid_all, float("-inf")
        ).max(dim=-1, keepdim=True)
        memory_valid = torch.isfinite(memory_score)

        dynamic_valid = valid_by_head & ~boundaries.eq(0).unsqueeze(1)
        dynamic_logits = logits.masked_fill(dynamic_valid.logical_not(), float("-inf"))
        if flat_anchor_count < 4:
            dynamic_logits = torch.cat(
                (
                    dynamic_logits,
                    dynamic_logits.new_full(
                        (local_batch, heads, 4 - flat_anchor_count),
                        float("-inf"),
                    ),
                ),
                dim=-1,
            )
        dynamic_scores, dynamic_indices = torch.topk(
            dynamic_logits, k=4, dim=-1
        )
        dynamic_top_valid = torch.isfinite(dynamic_scores)
        dynamic_indices = torch.where(
            dynamic_top_valid,
            dynamic_indices,
            torch.zeros_like(dynamic_indices),
        )
        candidate_indices = torch.cat((memory_index, dynamic_indices), dim=-1)
        candidate_scores = torch.cat((memory_score, dynamic_scores), dim=-1)
        candidate_valid = torch.cat((memory_valid, dynamic_top_valid), dim=-1)

        candidate_page_pos = torch.div(
            candidate_indices, page_m, rounding_mode="floor"
        )
        candidate_slots = candidate_indices - candidate_page_pos * page_m
        flat_candidate_page_pos = candidate_page_pos.reshape(
            local_batch, heads * 5
        ).clamp(min=0, max=max(0, max_pages - 1))
        selected_page_ids = page_table_b.gather(
            1, flat_candidate_page_pos
        ).clamp_min(0).view(local_batch, heads, 5)
        head_indices = head_range.expand(local_batch, heads, 5)
        assert cache.states_pages is not None
        selected_states = cache.states_pages[
            selected_page_ids,
            candidate_slots,
            head_indices,
        ].float()
        reads = torch.einsum(
            "bhk,bhtvk->bhtv",
            read_q[batch_start:batch_end].float(),
            selected_states,
        )
        reads = reads * float(read_scale)
        reads = reads.masked_fill(~candidate_valid.unsqueeze(-1), 0.0)

        if null_logits is not None:
            logits_with_null = torch.cat(
                (
                    candidate_scores,
                    null_logits[batch_start:batch_end].float().unsqueeze(-1),
                ),
                dim=-1,
            )
            valid_with_null = torch.cat(
                (
                    candidate_valid,
                    torch.ones(
                        local_batch,
                        heads,
                        1,
                        device=read_q.device,
                        dtype=torch.bool,
                    ),
                ),
                dim=-1,
            )
            joint_weights = _route_softmax(logits_with_null, valid_with_null)
            weights = joint_weights[..., :5]
            null_weights = joint_weights[..., 5]
        else:
            weights = _route_softmax(candidate_scores, candidate_valid)
            null_weights = torch.zeros_like(weights[..., 0])
        out[batch_start:batch_end] = torch.einsum(
            "bht,bhtv->bhv",
            weights.to(reads.dtype),
            reads,
        ).to(dtype=read_q.dtype)

        if memory_metrics is not None:
            memory_valid_2d = memory_valid.squeeze(-1)
            state_rms = selected_states[..., 0, :, :].float().square().mean(
                dim=(-3, -2, -1)
            ).sqrt()
            memory_contribution = (
                reads[..., 0, :].float() * weights[..., 0, None].float()
            )
            dynamic_contribution = torch.einsum(
                "bht,bhtv->bhv",
                weights[..., 1:5].float(),
                reads[..., 1:5, :].float(),
            )
            total_contribution = memory_contribution + dynamic_contribution
            metric_logits.append(memory_score.squeeze(-1))
            metric_probabilities.append(weights[..., 0])
            dynamic_weights = weights[..., 1:5]
            metric_dynamic_probabilities.append(dynamic_weights.sum(dim=-1))
            metric_null_probabilities.append(null_weights)
            metric_dynamic_rank_probabilities.append(dynamic_weights)
            metric_reads.append(memory_contribution)
            metric_dynamic_reads.append(dynamic_contribution)
            metric_total_reads.append(total_contribution)
            metric_states.append(state_rms)
            metric_state_valid.append(memory_valid_2d.any(dim=-1))
            metric_memory_valid.append(memory_valid_2d)

    if memory_metrics is not None:
        memory_valid = torch.cat(metric_memory_valid, dim=0)
        memory_logit = torch.cat(metric_logits, dim=0)
        memory_probability = torch.cat(metric_probabilities, dim=0)
        dynamic_probability = torch.cat(metric_dynamic_probabilities, dim=0)
        null_probability = torch.cat(metric_null_probabilities, dim=0)
        dynamic_rank_probability = torch.cat(
            metric_dynamic_rank_probabilities, dim=0
        )
        memory_contribution = torch.cat(metric_reads, dim=0)
        dynamic_contribution = torch.cat(metric_dynamic_reads, dim=0)
        total_contribution = torch.cat(metric_total_reads, dim=0)
        state_rms = torch.cat(metric_states, dim=0)
        state_valid = torch.cat(metric_state_valid, dim=0)
        contribution_mask = memory_valid.unsqueeze(-1).expand_as(
            memory_contribution
        )
        contribution_count = contribution_mask.sum(dtype=torch.float32)
        contribution_square_sum = torch.where(
            contribution_mask,
            memory_contribution.float().square(),
            torch.zeros_like(memory_contribution, dtype=torch.float32),
        ).sum(dtype=torch.float32)
        memory_read_rms = torch.where(
            contribution_count > 0,
            (contribution_square_sum / contribution_count).sqrt(),
            contribution_square_sum.new_zeros(()),
        )
        dynamic_contribution_square_sum = torch.where(
            contribution_mask,
            dynamic_contribution.float().square(),
            torch.zeros_like(dynamic_contribution, dtype=torch.float32),
        ).sum(dtype=torch.float32)
        dynamic_read_rms = torch.where(
            contribution_count > 0,
            (dynamic_contribution_square_sum / contribution_count).sqrt(),
            dynamic_contribution_square_sum.new_zeros(()),
        )
        total_contribution_square_sum = torch.where(
            contribution_mask,
            total_contribution.float().square(),
            torch.zeros_like(total_contribution, dtype=torch.float32),
        ).sum(dtype=torch.float32)
        total_history_read_rms = torch.where(
            contribution_count > 0,
            (total_contribution_square_sum / contribution_count).sqrt(),
            total_contribution_square_sum.new_zeros(()),
        )
        metric_updates = dict(
            memory_logit=_masked_scalar_mean(memory_logit, memory_valid),
            memory_probability=_masked_scalar_mean(
                memory_probability, memory_valid
            ),
            dynamic_probability=_masked_scalar_mean(
                dynamic_probability, memory_valid
            ),
            null_probability=_masked_scalar_mean(
                null_probability, memory_valid
            ),
            dynamic_rank1_probability=_masked_scalar_mean(
                dynamic_rank_probability[..., 0], memory_valid
            ),
            dynamic_rank2_probability=_masked_scalar_mean(
                dynamic_rank_probability[..., 1], memory_valid
            ),
            dynamic_rank3_probability=_masked_scalar_mean(
                dynamic_rank_probability[..., 2], memory_valid
            ),
            dynamic_rank4_probability=_masked_scalar_mean(
                dynamic_rank_probability[..., 3], memory_valid
            ),
            memory_read_rms=memory_read_rms,
            dynamic_read_rms=dynamic_read_rms,
            total_history_read_rms=total_history_read_rms,
            state_rms=_masked_scalar_mean(state_rms, state_valid),
        )
        if record_dynamic_state_rms_median:
            metric_updates["dynamic_state_rms_median"] = (
                _paged_dynamic_state_rms_median(
                    cache=cache,
                    page_table=page_table,
                    anchor_lengths=anchor_lengths,
                ).float()
            )
        memory_metrics.update(metric_updates)
    return out


def _paged_decode_read_all_history_read_torch(
    *,
    cache: HSDAnchorPageCache,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    page_table: torch.Tensor,
    anchor_lengths: torch.Tensor,
    start_indices: torch.Tensor,
    route_scale: float,
    read_scale: float,
    null_logits: torch.Tensor | None,
) -> torch.Tensor:
    assert cache.states_pages is not None
    assert cache.keys_pages is not None
    assert cache.boundary_pages is not None
    assert cache.value_dim is not None
    assert cache.route_heads is not None and cache.route_dim is not None
    batch_size, heads, state_key_dim = read_q.shape
    route_dim = int(router_q.shape[-1])
    if route_dim != cache.route_dim:
        raise ValueError(
            f"route dim mismatch: query={route_dim}, cache={cache.route_dim}."
        )
    page_m = int(cache.page_m)
    max_pages = int(page_table.shape[1])
    anchor_count = max_pages * page_m
    out = read_q.new_zeros(batch_size, heads, cache.value_dim)
    if anchor_count == 0:
        return out
    chunk_size = max(
        1, int(os.environ.get("HSD_PAGED_HISTORY_TORCH_BATCH_CHUNK", "8"))
    )
    anchor_offsets = torch.arange(anchor_count, device=read_q.device)
    for batch_start in range(0, batch_size, chunk_size):
        batch_end = min(batch_size, batch_start + chunk_size)
        local_batch = batch_end - batch_start
        table = page_table[batch_start:batch_end]
        flat_page_ids = table.clamp_min(0).reshape(-1)
        keys = cache.keys_pages.index_select(0, flat_page_ids).view(
            local_batch,
            max_pages * page_m,
            cache.route_heads,
            route_dim,
        )
        states = cache.states_pages.index_select(0, flat_page_ids).view(
            local_batch,
            max_pages * page_m,
            heads,
            cache.value_dim,
            state_key_dim,
        )
        boundaries = cache.boundary_pages.index_select(0, flat_page_ids).view(
            local_batch, anchor_count
        )
        page_valid = table.ge(0).unsqueeze(-1).expand(
            local_batch, max_pages, page_m
        ).reshape(local_batch, anchor_count)
        valid = (
            page_valid
            & (
                anchor_offsets.view(1, -1)
                < anchor_lengths[batch_start:batch_end].view(-1, 1)
            )
            & (
                boundaries
                <= start_indices[batch_start:batch_end].view(-1, 1)
            )
        )
        router = router_q[batch_start:batch_end].float()
        if cache.route_heads == 1:
            logits = torch.einsum(
                "bhr,bmr->bhm", router, keys[:, :, 0].float()
            )
        else:
            logits = torch.einsum(
                "bhr,bmhr->bhm", router, keys.float()
            )
        logits = logits * float(route_scale)
        valid_h = valid.unsqueeze(1).expand(local_batch, heads, anchor_count)
        if null_logits is not None:
            logits = torch.cat(
                (logits, null_logits[batch_start:batch_end].float().unsqueeze(-1)),
                dim=-1,
            )
            valid_h = torch.cat(
                (
                    valid_h,
                    torch.ones(
                        local_batch,
                        heads,
                        1,
                        device=read_q.device,
                        dtype=torch.bool,
                    ),
                ),
                dim=-1,
            )
            weights = _route_softmax(logits, valid_h)[..., :-1]
        else:
            weights = _route_softmax(logits, valid_h)
        states = states.masked_fill(
            ~valid.view(local_batch, anchor_count, 1, 1, 1),
            0,
        )
        weighted_states = torch.einsum(
            "bhm,bmhvk->bhvk", weights.float(), states.float()
        )
        out[batch_start:batch_end] = (
            torch.einsum(
                "bhk,bhvk->bhv",
                read_q[batch_start:batch_end].float(),
                weighted_states,
            )
            * float(read_scale)
        ).to(read_q.dtype)
    return out


def _record_paged_memory_metrics_reference(
    *,
    cache: HSDAnchorPageCache,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    page_table: torch.Tensor,
    anchor_lengths: torch.Tensor,
    start_indices: torch.Tensor,
    route_scale: float,
    read_scale: float,
    null_logits: torch.Tensor | None,
    memory_metrics: dict[str, torch.Tensor] | None,
    read_q_is_raw: bool = False,
    read_q_num_heads: int | None = None,
    normalize_router_q: bool = False,
    null_bias: torch.Tensor | None = None,
    record_dynamic_state_rms_median: bool = True,
) -> None:
    """Populate optional audit tensors using the exact Torch route oracle."""
    if memory_metrics is None:
        return
    batch_size, heads, key_dim = router_q.shape
    metric_read_q = read_q
    if read_q_is_raw:
        if read_q_num_heads is None or int(read_q_num_heads) <= 0:
            raise ValueError("read_q_num_heads is required for raw metric read_q.")
        read_heads = int(read_q_num_heads)
        metric_read_q = read_q[:, : read_heads * key_dim].reshape(
            batch_size, read_heads, key_dim
        ).float()
        metric_read_q = metric_read_q * torch.rsqrt(
            metric_read_q.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        metric_read_q = metric_read_q.to(read_q.dtype).float()
        metric_read_q = metric_read_q.repeat_interleave(
            heads // read_heads, dim=1
        )
    if normalize_router_q:
        metric_router_q = router_q.float()
        metric_router_q = metric_router_q * torch.rsqrt(
            metric_router_q.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        metric_router_q = metric_router_q.to(router_q.dtype)
    else:
        metric_router_q = router_q
    metric_null_logits = null_logits
    if metric_null_logits is not None and null_bias is not None:
        metric_null_logits = metric_null_logits.float() + null_bias.float().view(
            1, -1
        )
    _paged_decode_top4_memory_history_read_torch(
        cache=cache,
        read_q=metric_read_q,
        router_q=metric_router_q,
        page_table=page_table,
        anchor_lengths=anchor_lengths,
        start_indices=start_indices,
        route_scale=route_scale,
        read_scale=read_scale,
        null_logits=metric_null_logits,
        memory_metrics=memory_metrics,
        record_dynamic_state_rms_median=record_dynamic_state_rms_median,
    )


def _record_pre_history_residual_rms(
    *,
    memory_metrics: dict[str, torch.Tensor] | None,
    base_out: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> None:
    """Record the RMS of the branch tensor immediately before history add."""
    if memory_metrics is None:
        return
    values = base_out.detach().float()
    if valid_mask is None:
        memory_metrics["residual_rms"] = values.square().mean().sqrt()
        return
    mask = valid_mask.to(device=values.device, dtype=torch.bool)
    mask = mask.view(mask.shape[0], *([1] * (values.ndim - 1))).expand_as(values)
    count = mask.sum(dtype=torch.float32)
    square_sum = torch.where(
        mask,
        values.square(),
        torch.zeros_like(values),
    ).sum(dtype=torch.float32)
    memory_metrics["residual_rms"] = torch.where(
        count > 0,
        (square_sum / count).sqrt(),
        square_sum.new_zeros(()),
    )
