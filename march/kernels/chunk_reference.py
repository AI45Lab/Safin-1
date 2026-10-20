# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reference chunk recurrence, anchor reads and autograd oracle."""

from __future__ import annotations

import torch

from .chunk_dependencies import FLA_CHUNK_SIZE, chunk_gated_delta_rule


def _run_segmented_reference(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    anchor_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, text_len, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    text_outputs = []
    anchor_outputs = []
    anchor_states = []
    state = None
    cursor = 0
    zero_state = query.new_zeros(batch, heads, key_dim, value_dim)

    for anchor_idx, boundary in enumerate(boundary_counts.tolist()):
        boundary = int(boundary)
        if boundary < cursor:
            raise ValueError("anchor positions must be ordered after non-decreasing text prefixes.")
        if boundary > text_len:
            raise ValueError("anchor positions must not exceed text length.")
        if boundary > cursor:
            out, state = chunk_gated_delta_rule(
                query[:, cursor:boundary],
                key[:, cursor:boundary],
                value[:, cursor:boundary],
                g=g[:, cursor:boundary],
                beta=beta[:, cursor:boundary],
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=None,
            )
            text_outputs.append(out)
            cursor = boundary

        state_for_anchor = zero_state if state is None else state
        anchor_states.append(state_for_anchor)
        anchor_outputs.append(
            torch.einsum(
                "bhk,bhkv->bhv",
                anchor_q[:, anchor_idx].to(state_for_anchor.dtype),
                state_for_anchor,
            )
            * scale
        )

    if cursor < text_len:
        out, _ = chunk_gated_delta_rule(
            query[:, cursor:text_len],
            key[:, cursor:text_len],
            value[:, cursor:text_len],
            g=g[:, cursor:text_len],
            beta=beta[:, cursor:text_len],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
        )
        text_outputs.append(out)

    text_o = torch.cat(text_outputs, dim=1) if text_outputs else query.new_zeros(batch, 0, heads, value_dim)
    anchor_o = torch.stack(anchor_outputs, dim=1).to(text_o.dtype) if anchor_outputs else query.new_zeros(
        batch, 0, heads, value_dim
    )
    states = torch.stack(anchor_states, dim=1) if anchor_states else query.new_zeros(
        batch, 0, heads, key_dim, value_dim
    )
    return text_o, anchor_o, states


def _run_segmented_packed_reference(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    anchor_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    anchor_cu_seqlens: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query.shape[0] != 1:
        raise NotImplementedError("HSD packed reference path requires flattened batch size 1.")
    text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    anchor_cu_cpu = anchor_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    if text_cu_cpu.numel() != anchor_cu_cpu.numel():
        raise ValueError("packed text and anchor cu_seqlens must describe the same documents.")

    text_outputs = []
    anchor_outputs = []
    state_outputs = []
    for doc_idx in range(int(text_cu_cpu.numel()) - 1):
        text_start = int(text_cu_cpu[doc_idx].item())
        text_end = int(text_cu_cpu[doc_idx + 1].item())
        anchor_start = int(anchor_cu_cpu[doc_idx].item())
        anchor_end = int(anchor_cu_cpu[doc_idx + 1].item())
        local_boundaries = boundary_counts[anchor_start:anchor_end] - text_start
        text_o_i, anchor_o_i, states_i = _run_segmented_reference(
            query=query[:, text_start:text_end],
            key=key[:, text_start:text_end],
            value=value[:, text_start:text_end],
            beta=beta[:, text_start:text_end],
            g=g[:, text_start:text_end],
            anchor_q=anchor_q[:, anchor_start:anchor_end],
            boundary_counts=local_boundaries,
            scale=scale,
        )
        text_outputs.append(text_o_i)
        if anchor_end > anchor_start:
            anchor_outputs.append(anchor_o_i)
            state_outputs.append(states_i)

    batch, _, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    text_o = torch.cat(text_outputs, dim=1) if text_outputs else query.new_zeros(batch, 0, heads, value_dim)
    anchor_o = (
        torch.cat(anchor_outputs, dim=1)
        if anchor_outputs else query.new_zeros(batch, 0, heads, value_dim)
    )
    states = (
        torch.cat(state_outputs, dim=1)
        if state_outputs else query.new_zeros(batch, 0, heads, key_dim, value_dim)
    )
    return text_o, anchor_o, states


def _is_packed_cu(text_cu_seqlens: torch.Tensor | None, anchor_cu_seqlens: torch.Tensor | None) -> bool:
    return (
        text_cu_seqlens is not None
        and anchor_cu_seqlens is not None
        and text_cu_seqlens.numel() > 0
        and anchor_cu_seqlens.numel() > 0
    )


def _packed_chunk_index(
    *,
    doc_idx: int,
    local_boundary: int,
    doc_len: int,
    chunk_offsets_cpu: torch.Tensor,
) -> int | None:
    if local_boundary == doc_len:
        return None
    return int(chunk_offsets_cpu[doc_idx].item()) + local_boundary // FLA_CHUNK_SIZE


def _run_anchor_read_reference(
    *,
    query: torch.Tensor,
    text_o: torch.Tensor,
    anchor_q: torch.Tensor,
    chunk_states_fp32: torch.Tensor,
    final_state: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor | None,
    anchor_cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, text_len, heads, key_dim = query.shape
    value_dim = text_o.shape[-1]
    states = []
    anchor_outputs = []
    packed = _is_packed_cu(text_cu_seqlens, anchor_cu_seqlens)
    if packed:
        text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        anchor_cu_cpu = anchor_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        chunk_offsets_cpu = chunk_offsets.detach().to(device="cpu", dtype=torch.long)
        boundary_cpu = boundary_counts.detach().to(device="cpu", dtype=torch.long)
        zero_state = chunk_states_fp32.new_zeros(batch, heads, key_dim, value_dim)
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            doc_len = text_end - text_start
            anchor_start = int(anchor_cu_cpu[doc_idx].item())
            anchor_end = int(anchor_cu_cpu[doc_idx + 1].item())
            for anchor_idx in range(anchor_start, anchor_end):
                local_boundary = int(boundary_cpu[anchor_idx].item()) - text_start
                if local_boundary == 0 and doc_len == 0:
                    state = zero_state
                elif local_boundary == doc_len:
                    state = final_state[doc_idx:doc_idx + 1]
                else:
                    chunk_idx = _packed_chunk_index(
                        doc_idx=doc_idx,
                        local_boundary=local_boundary,
                        doc_len=doc_len,
                        chunk_offsets_cpu=chunk_offsets_cpu,
                    )
                    state = chunk_states_fp32[:, chunk_idx]
                states.append(state)
                anchor_outputs.append(
                    torch.einsum(
                        "bhk,bhkv->bhv",
                        anchor_q[:, anchor_idx].to(state.dtype),
                        state,
                    )
                    * scale
                )
    else:
        for anchor_idx, boundary in enumerate(boundary_counts.tolist()):
            boundary = int(boundary)
            if boundary == text_len:
                state = final_state
            else:
                chunk_idx = boundary // FLA_CHUNK_SIZE
                state = chunk_states_fp32[:, chunk_idx]
            states.append(state)
            anchor_outputs.append(
                torch.einsum(
                    "bhk,bhkv->bhv",
                    anchor_q[:, anchor_idx].to(state.dtype),
                    state,
                )
                * scale
            )

    anchor_o = torch.stack(anchor_outputs, dim=1).to(text_o.dtype) if anchor_outputs else query.new_zeros(
        batch, 0, heads, value_dim
    )
    anchor_states = torch.stack(states, dim=1) if states else query.new_zeros(
        batch, 0, heads, key_dim, value_dim
    )
    return anchor_o, anchor_states


def _run_reference_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    anchor_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor | None,
    anchor_cu_seqlens: torch.Tensor | None,
    dtext_o: torch.Tensor | None,
    danchor_o: torch.Tensor | None,
    danchor_states: torch.Tensor | None,
    scale: float,
    needs_grad,
) -> list[torch.Tensor | None]:
    query_ref = query.detach().requires_grad_(needs_grad[0])
    key_ref = key.detach().requires_grad_(needs_grad[1])
    value_ref = value.detach().requires_grad_(needs_grad[2])
    beta_ref = beta.detach().requires_grad_(needs_grad[3])
    g_ref = g.detach().requires_grad_(needs_grad[4])
    anchor_q_ref = anchor_q.detach().requires_grad_(needs_grad[5])

    with torch.enable_grad():
        if _is_packed_cu(text_cu_seqlens, anchor_cu_seqlens):
            text_o, anchor_o, anchor_states = _run_segmented_packed_reference(
                query=query_ref,
                key=key_ref,
                value=value_ref,
                beta=beta_ref,
                g=g_ref,
                anchor_q=anchor_q_ref,
                boundary_counts=boundary_counts,
                text_cu_seqlens=text_cu_seqlens,
                anchor_cu_seqlens=anchor_cu_seqlens,
                scale=scale,
            )
        else:
            text_o, anchor_o, anchor_states = _run_segmented_reference(
                query=query_ref,
                key=key_ref,
                value=value_ref,
                beta=beta_ref,
                g=g_ref,
                anchor_q=anchor_q_ref,
                boundary_counts=boundary_counts,
                scale=scale,
            )
        if dtext_o is None:
            dtext_o = torch.zeros_like(text_o)
        if danchor_o is None:
            danchor_o = torch.zeros_like(anchor_o)
        if danchor_states is None:
            danchor_states = torch.zeros_like(anchor_states)

        grad_inputs = []
        input_refs = [query_ref, key_ref, value_ref, beta_ref, g_ref, anchor_q_ref]
        active_refs = [tensor for tensor, need in zip(input_refs, needs_grad[:6]) if need]
        if active_refs:
            active_grads = torch.autograd.grad(
                outputs=(text_o, anchor_o, anchor_states),
                inputs=active_refs,
                grad_outputs=(dtext_o, danchor_o, danchor_states),
                allow_unused=True,
                retain_graph=False,
                create_graph=False,
            )
            active_iter = iter(active_grads)
        else:
            active_iter = iter(())
        for need in needs_grad[:6]:
            grad_inputs.append(next(active_iter) if need else None)
    return grad_inputs

