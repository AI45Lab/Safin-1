# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

r"""Production-faithful direct-A backward oracle with an external saved A.

The production Hopper forward solves the chunk-local KKT system once and
stores

.. math::

    A = M^{-1},\qquad
    M = I + \operatorname{tril}_{-1}
        \left(\operatorname{diag}(\beta) K K^\top\right)

in the q/k activation dtype.  Kernel B consumes that saved tensor; it does not
solve the KKT system again.  Its inverse VJP is nevertheless

.. math::

    dM = -A^\top dA A^\top.

This standalone CPU/GPU oracle models exactly that contract.  ``saved_a`` is
the value used by both the forward equations and the inverse VJP.  In the
production-faithful BF16 case callers should pass

``exact_a.to(torch.bfloat16).to(torch.float32)``.

The implementation uses a first-order surrogate for the inverse:

.. math::

    \widetilde A(M)
      = A_{\rm saved}
        - A_{\rm saved}(M-\operatorname{stopgrad}(M))A_{\rm saved}.

It has value ``A_saved`` and derivative ``dA=-A_saved dM A_saved`` at the
evaluation point.  Ordinary PyTorch autograd can therefore serve as an
independent VJP oracle without re-solving the KKT system in backward.

This file is intentionally not imported by the production runtime.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class SavedADirectAChunkForward(NamedTuple):
    output: torch.Tensor
    end_state: torch.Tensor


class SavedADirectAChunkBackward(NamedTuple):
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    dbeta: torch.Tensor
    dg: torch.Tensor
    dh0: torch.Tensor


def _check_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    h0: torch.Tensor,
    saved_a: torch.Tensor,
) -> tuple[int, int, int]:
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or h0.ndim != 2:
        raise ValueError("q/k/v/h0 must be rank-2 single-chunk tensors.")
    sequence_length, key_dim = q.shape
    value_dim = v.shape[1]
    if sequence_length < 1:
        raise ValueError("A chunk must contain at least one valid token.")
    if k.shape != (sequence_length, key_dim):
        raise ValueError(f"k must have shape {(sequence_length, key_dim)}.")
    if v.shape[0] != sequence_length:
        raise ValueError("q/k/v must have the same sequence length.")
    if beta.shape != (sequence_length,):
        raise ValueError("beta must have shape [sequence_length].")
    if g_cumsum.shape != (sequence_length,):
        raise ValueError("g_cumsum must have shape [sequence_length].")
    if h0.shape != (key_dim, value_dim):
        raise ValueError(f"h0 must have shape {(key_dim, value_dim)}.")
    if saved_a.shape != (sequence_length, sequence_length):
        raise ValueError(
            "saved_a must have shape "
            f"{(sequence_length, sequence_length)}."
        )
    return sequence_length, key_dim, value_dim


def solve_direct_a_inverse_reference(
    k: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Solve the exact chunk-local inverse before storage quantization."""

    if k.ndim != 2:
        raise ValueError("k must be rank 2.")
    sequence_length = k.shape[0]
    if sequence_length < 1 or beta.shape != (sequence_length,):
        raise ValueError("beta must match the non-empty token dimension.")
    gram = k @ k.mT
    identity = torch.eye(
        sequence_length,
        dtype=gram.dtype,
        device=gram.device,
    )
    kkt = identity + torch.tril(
        beta[:, None] * gram,
        diagonal=-1,
    )
    return torch.linalg.solve(kkt, identity)


def quantize_saved_a_bf16_fp32(saved_a: torch.Tensor) -> torch.Tensor:
    """Round an inverse through its production BF16 storage representation."""

    if not saved_a.is_floating_point():
        raise TypeError("saved_a must be floating point.")
    return saved_a.to(torch.bfloat16).to(torch.float32)


def _forward_with_linearized_saved_a(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    h0: torch.Tensor,
    saved_a: torch.Tensor,
    *,
    scale: float,
) -> SavedADirectAChunkForward:
    sequence_length, _, _ = _check_shapes(
        q,
        k,
        v,
        beta,
        g_cumsum,
        h0,
        saved_a,
    )

    gram = k @ k.mT
    identity = torch.eye(
        sequence_length,
        dtype=gram.dtype,
        device=gram.device,
    )
    kkt = identity + torch.tril(
        beta[:, None] * gram,
        diagonal=-1,
    )

    # Value: saved_a.  Differential: dA=-saved_a*dM*saved_a.
    a = saved_a + (
        -saved_a
        @ (kkt - kkt.detach())
        @ saved_a
    )

    exp_z = torch.exp(g_cumsum)
    decay = exp_z[-1]
    reverse_decay = decay / exp_z
    causal = torch.tril(
        exp_z[:, None] / exp_z[None, :],
        diagonal=0,
    )
    write = causal * a * beta[None, :]

    state_read = k @ h0
    residual_value = v - exp_z[:, None] * state_read
    delta = write @ residual_value
    normalized_delta = reverse_decay[:, None] * delta
    end_state = decay * h0 + k.mT @ normalized_delta

    output_read = float(scale) * causal * (q @ k.mT)
    output = (
        float(scale) * exp_z[:, None] * (q @ h0)
        + output_read @ delta
    )
    return SavedADirectAChunkForward(output, end_state)


def direct_a_chunk_backward_saved_a_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    h0: torch.Tensor,
    saved_a: torch.Tensor,
    output_grad: torch.Tensor,
    end_state_grad: torch.Tensor,
    *,
    scale: float,
) -> SavedADirectAChunkBackward:
    """Return the VJP used by Kernel B for an externally stored inverse.

    ``g_cumsum`` is the saved chunk-local cumulative log gate.  The returned
    ``dg`` is with respect to the original raw per-token log gate, so the
    chunk-local suffix sum performed by the production kernel is included.
    """

    sequence_length, key_dim, value_dim = _check_shapes(
        q,
        k,
        v,
        beta,
        g_cumsum,
        h0,
        saved_a,
    )
    if output_grad.shape != (sequence_length, value_dim):
        raise ValueError(
            f"output_grad must have shape {(sequence_length, value_dim)}."
        )
    if end_state_grad.shape != (key_dim, value_dim):
        raise ValueError(
            f"end_state_grad must have shape {(key_dim, value_dim)}."
        )

    inputs = tuple(
        tensor.detach().requires_grad_(True)
        for tensor in (q, k, v, beta, g_cumsum, h0)
    )
    q_var, k_var, v_var, beta_var, z_var, h0_var = inputs
    saved_a_value = saved_a.detach().to(dtype=q_var.dtype)
    with torch.enable_grad():
        forward = _forward_with_linearized_saved_a(
            q_var,
            k_var,
            v_var,
            beta_var,
            z_var,
            h0_var,
            saved_a_value,
            scale=scale,
        )
        loss = (
            torch.sum(forward.output * output_grad)
            + torch.sum(forward.end_state * end_state_grad)
        )
        dq, dk, dv, dbeta, dz, dh0 = torch.autograd.grad(loss, inputs)

    # z_i=sum_{j<=i}g_j => dg_i=sum_{j>=i}dz_j.
    dg = torch.flip(
        torch.cumsum(torch.flip(dz, dims=(0,)), dim=0),
        dims=(0,),
    )
    return SavedADirectAChunkBackward(dq, dk, dv, dbeta, dg, dh0)


__all__ = [
    "SavedADirectAChunkBackward",
    "SavedADirectAChunkForward",
    "direct_a_chunk_backward_saved_a_reference",
    "quantize_saved_a_bf16_fp32",
    "solve_direct_a_inverse_reference",
]
