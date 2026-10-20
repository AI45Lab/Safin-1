"""Small PyTorch reference for the GDN2 recurrence.

This file is original integration code. The optimized kernels in this package
are derived from NVIDIA GatedDeltaNet-2 commit 95709fc and remain under the
NVIDIA Source Code License-NC included alongside them.
"""

from __future__ import annotations

import torch


def naive_recurrent_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate GDN2 token by token with an FP32 recurrent state.

    All token tensors use ``[B, T, H, D]`` layout. ``g`` is the activated
    log-decay, while ``b`` and ``w`` are the sigmoid erase/write vectors.
    """
    if q.shape != k.shape or q.shape != g.shape or q.shape != b.shape:
        raise ValueError("q, k, g, and b must have identical [B,T,H,K] shapes.")
    if v.shape != w.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("v and w must share [B,T,H,V] shape and match q heads.")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    state_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    state = (
        q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1], dtype=state_dtype)
        if initial_state is None
        else initial_state.to(dtype=state_dtype).clone()
    )
    outputs = []
    for token_idx in range(q.shape[1]):
        q_t = q[:, token_idx].to(state_dtype) * float(scale)
        k_t = k[:, token_idx].to(state_dtype)
        v_t = v[:, token_idx].to(state_dtype)
        g_t = g[:, token_idx].to(state_dtype)
        b_t = b[:, token_idx].to(state_dtype)
        w_t = w[:, token_idx].to(state_dtype)
        decayed_state = g_t.exp().unsqueeze(-1) * state
        erase = torch.einsum("bhkv,bhk->bhv", decayed_state, b_t * k_t)
        delta = w_t * v_t - erase
        state = decayed_state + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state))
    output = torch.stack(outputs, dim=1).to(v.dtype)
    return output, state if output_final_state else None


__all__ = ["naive_recurrent_gdn2"]
