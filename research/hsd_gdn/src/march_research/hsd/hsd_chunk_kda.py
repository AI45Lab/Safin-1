from __future__ import annotations

from typing import Optional

import torch

try:
    from fla.ops.kda import chunk_kda
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    chunk_kda = None


FLA_CHUNK_SIZE = 64


def hsd_chunk_kda_with_summary_states(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    summary_q: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: Optional[torch.Tensor] = None,
    summary_cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run one KDA scan and expose differentiable states at HSD boundaries.

    KDA's native chunk scan already materializes the state at the beginning of
    every 64-token chunk. HSD summary boundaries are chunk-aligned (except for
    a document's final boundary), so the summary states can be gathered from
    those native states without splitting the recurrent scan.
    """
    if chunk_kda is None or not query.is_cuda:
        return None
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or g.ndim != 4:
        return None
    if query.shape != key.shape:
        return None
    batch, text_len, heads, key_dim = query.shape
    if value.shape[:3] != (batch, text_len, heads):
        return None
    if beta.shape != (batch, text_len, heads):
        return None
    if g.shape != (batch, text_len, heads, key_dim):
        return None
    if summary_q.ndim != 4 or summary_q.shape[:1] != (batch,):
        return None
    if summary_q.shape[2:] != (heads, key_dim):
        return None
    summary_count = int(summary_q.shape[1])
    if boundary_counts.ndim != 1 or int(boundary_counts.shape[0]) != summary_count:
        return None
    if key_dim > 256 or text_len <= 0:
        return None

    packed = text_cu_seqlens is not None or summary_cu_seqlens is not None
    if not packed:
        # The HSD model uses this helper only for packed training. Non-packed
        # prefill/decode continues through the segmented KDA path.
        return None
    if packed:
        if text_cu_seqlens is None or summary_cu_seqlens is None or batch != 1:
            return None
        if text_cu_seqlens.ndim != 1 or summary_cu_seqlens.ndim != 1:
            return None
        if text_cu_seqlens.numel() < 2 or text_cu_seqlens.numel() != summary_cu_seqlens.numel():
            return None
        text_cu = text_cu_seqlens.to(device=query.device, dtype=torch.long).contiguous()
        summary_cu = summary_cu_seqlens.to(device=query.device, dtype=torch.long).contiguous()
        text_cu_cpu = text_cu.detach().to(device="cpu", dtype=torch.long)
        summary_cu_cpu = summary_cu.detach().to(device="cpu", dtype=torch.long)
        if int(text_cu_cpu[0]) != 0 or int(text_cu_cpu[-1]) != text_len:
            return None
        if int(summary_cu_cpu[0]) != 0 or int(summary_cu_cpu[-1]) != summary_count:
            return None
        if bool((text_cu_cpu[1:] < text_cu_cpu[:-1]).any()):
            return None
        if bool((summary_cu_cpu[1:] < summary_cu_cpu[:-1]).any()):
            return None
    boundaries_cpu = boundary_counts.detach().to(device="cpu", dtype=torch.long)
    gather_spec: list[tuple[int, int, bool]] = []
    if packed:
        assert text_cu_cpu is not None and summary_cu_cpu is not None
        chunk_offsets_cpu = torch.nn.functional.pad(
            torch.div(
                text_cu_cpu[1:] - text_cu_cpu[:-1] + FLA_CHUNK_SIZE - 1,
                FLA_CHUNK_SIZE,
                rounding_mode="floor",
            ),
            (1, 0),
            value=0,
        ).cumsum(0)
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx])
            text_end = int(text_cu_cpu[doc_idx + 1])
            doc_len = text_end - text_start
            summary_start = int(summary_cu_cpu[doc_idx])
            summary_end = int(summary_cu_cpu[doc_idx + 1])
            local_boundaries = boundaries_cpu[summary_start:summary_end] - text_start
            if bool((local_boundaries < 0).any()) or bool((local_boundaries > doc_len).any()):
                return None
            if local_boundaries.numel() > 1 and bool((local_boundaries[1:] < local_boundaries[:-1]).any()):
                return None
            for boundary in local_boundaries.tolist():
                is_final = int(boundary) == doc_len
                if not is_final and int(boundary) % FLA_CHUNK_SIZE:
                    return None
                chunk_idx = int(chunk_offsets_cpu[doc_idx]) + int(boundary) // FLA_CHUNK_SIZE
                gather_spec.append((doc_idx, chunk_idx, is_final))
    if scale is None:
        scale = key_dim**-0.5
    selected_indices = torch.tensor(
        [chunk_idx for _, chunk_idx, is_final in gather_spec if not is_final],
        device=query.device,
        dtype=torch.long,
    )
    text_o, final_state, selected_chunk_states = chunk_kda(
        q=query.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=float(scale),
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        cu_seqlens=text_cu,
        cu_seqlens_cpu=text_cu_cpu,
        return_intermediate_states=True,
        intermediate_state_indices=selected_indices,
    )
    if final_state is None or selected_chunk_states is None:
        return None

    if gather_spec:
        selected_cursor = 0
        gathered_states = []
        for doc_idx, _, is_final in gather_spec:
            if is_final:
                gathered_states.append(final_state[doc_idx])
            else:
                gathered_states.append(selected_chunk_states[selected_cursor])
                selected_cursor += 1
        states = torch.stack(gathered_states, dim=0).unsqueeze(0).float()
    else:
        states = query.new_zeros(
            batch,
            0,
            heads,
            key_dim,
            value.shape[-1],
            dtype=torch.float32,
        )
    summary_o = torch.einsum(
        "bmhk,bmhkv->bmhv",
        summary_q.to(dtype=states.dtype),
        states,
    ) * float(scale)
    return text_o, summary_o.to(dtype=text_o.dtype), states


__all__ = ["hsd_chunk_kda_with_summary_states"]
