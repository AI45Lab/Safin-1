# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Reference equations for the direct-A chunk-local HSD backward.

This module is deliberately independent from the production dispatch.  It is
the correctness oracle for the planned chunk-parallel TileLang Kernel B:

* one invocation owns one valid ``(chunk64, head)``;
* ``end_state_grad`` already includes the compact HSD boundary injection;
* no WY representation is built;
* the KKT inverse is differentiated in the same invocation;
* the returned gate gradient is with respect to the *raw* per-token log gate,
  not the chunk-local cumulative log gate.

The reference uses ordinary PyTorch matrix operations and is not intended for
training performance.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class DirectAChunkForward(NamedTuple):
    output: torch.Tensor
    end_state: torch.Tensor


class DirectAChunkBackward(NamedTuple):
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    dbeta: torch.Tensor
    dg: torch.Tensor
    dh0: torch.Tensor


class DirectAOutputOnlyDKComponents(NamedTuple):
    """Output-only ``dK`` split by the three direct-A graph paths."""

    direct_state: torch.Tensor
    dp: torch.Tensor
    kkt: torch.Tensor
    total: torch.Tensor


def _check_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
) -> tuple[int, int, int]:
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or h0.ndim != 2:
        raise ValueError("q/k/v/h0 must be rank-2 single-chunk tensors.")
    sequence_length, key_dim = q.shape
    value_dim = v.shape[1]
    if k.shape != (sequence_length, key_dim):
        raise ValueError(f"k must have shape {(sequence_length, key_dim)}.")
    if v.shape[0] != sequence_length:
        raise ValueError("q/k/v must have the same sequence length.")
    if beta.shape != (sequence_length,) or g.shape != (sequence_length,):
        raise ValueError("beta/g must have shape [sequence_length].")
    if h0.shape != (key_dim, value_dim):
        raise ValueError(f"h0 must have shape {(key_dim, value_dim)}.")
    if sequence_length < 1:
        raise ValueError("A chunk must contain at least one valid token.")
    return sequence_length, key_dim, value_dim


def _direct_a_intermediates(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    scale: float,
) -> dict[str, torch.Tensor]:
    sequence_length, _, _ = _check_shapes(q, k, v, beta, g, h0)
    z = torch.cumsum(g, dim=0)
    exp_z = torch.exp(z)
    decay = exp_z[-1]
    reverse_decay = decay / exp_z

    gram = k @ k.mT
    identity = torch.eye(
        sequence_length,
        dtype=gram.dtype,
        device=gram.device,
    )
    kkt = identity + torch.tril(beta[:, None] * gram, diagonal=-1)
    inverse = torch.linalg.solve(kkt, identity)

    causal = torch.tril(
        exp_z[:, None] / exp_z[None, :],
        diagonal=0,
    )
    write = causal * inverse * beta[None, :]

    state_read = k @ h0
    residual_value = v - exp_z[:, None] * state_read
    delta = write @ residual_value
    normalized_delta = reverse_decay[:, None] * delta
    end_state = decay * h0 + k.mT @ normalized_delta

    qk = q @ k.mT
    output_read = float(scale) * causal * qk
    output = (
        float(scale) * exp_z[:, None] * (q @ h0)
        + output_read @ delta
    )
    return {
        "z": z,
        "exp_z": exp_z,
        "decay": decay,
        "reverse_decay": reverse_decay,
        "gram": gram,
        "inverse": inverse,
        "causal": causal,
        "write": write,
        "state_read": state_read,
        "residual_value": residual_value,
        "delta": delta,
        "normalized_delta": normalized_delta,
        "qk": qk,
        "output_read": output_read,
        "output": output,
        "end_state": end_state,
    }


def direct_a_chunk_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    *,
    scale: float,
) -> DirectAChunkForward:
    """Evaluate the exact direct-A chunk graph for one head."""

    saved = _direct_a_intermediates(q, k, v, beta, g, h0, scale)
    return DirectAChunkForward(saved["output"], saved["end_state"])


def direct_a_chunk_delta_grad_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    output_grad: torch.Tensor,
    end_state_grad: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    r"""Return the direct-A ``d_delta`` used by the fused local backward.

    This is the quantity that the final TileLang Kernel B computes on chip:

    .. math::

       dD = r \odot (K\,dH_{\mathrm{end}}) + C^\top dO.

    Keeping this expression as an explicit oracle prevents the optimized
    implementation from silently depending on a materialized WY/``du``
    intermediate.
    """

    sequence_length, key_dim, value_dim = _check_shapes(
        q,
        k,
        v,
        beta,
        g,
        h0,
    )
    if output_grad.shape != (sequence_length, value_dim):
        raise ValueError(
            f"output_grad must have shape {(sequence_length, value_dim)}."
        )
    if end_state_grad.shape != (key_dim, value_dim):
        raise ValueError(
            f"end_state_grad must have shape {(key_dim, value_dim)}."
        )
    saved = _direct_a_intermediates(q, k, v, beta, g, h0, scale)
    return (
        saved["reverse_decay"][:, None] * (k @ end_state_grad)
        + saved["output_read"].mT @ output_grad
    )


def direct_a_chunk_output_only_dk_components_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    output_grad: torch.Tensor,
    *,
    scale: float,
) -> DirectAOutputOnlyDKComponents:
    r"""Split the output-only key VJP into direct, ``dP`` and KKT terms.

    This is a diagnostic oracle for the fused TileLang backward.  It sets the
    end-state gradient to zero and exposes the only three paths from the local
    output to ``K``:

    * ``direct_state``: ``dU @ H0.T`` through
      ``W = V - exp(z) * K @ H0``;
    * ``dp``: ``dP.T @ Q`` through
      ``C = scale * G * (Q @ K.T)``;
    * ``kkt``: the inverse-system VJP through
      ``A = KKT(K, beta)^{-1}``.

    Their sum is exactly the ``dk`` returned by
    :func:`direct_a_chunk_backward_reference` when ``end_state_grad == 0``.
    """

    sequence_length, _, value_dim = _check_shapes(
        q,
        k,
        v,
        beta,
        g,
        h0,
    )
    if output_grad.shape != (sequence_length, value_dim):
        raise ValueError(
            f"output_grad must have shape {(sequence_length, value_dim)}."
        )

    saved = _direct_a_intermediates(q, k, v, beta, g, h0, scale)
    exp_z = saved["exp_z"]
    causal = saved["causal"]
    inverse = saved["inverse"]
    gram = saved["gram"]
    write = saved["write"]
    residual_value = saved["residual_value"]
    delta = saved["delta"]
    output_read = saved["output_read"]

    # O = direct_state_output + output_read @ delta.
    d_output_read = output_grad @ delta.mT
    d_delta = output_read.mT @ output_grad

    # delta = write @ residual_value;
    # residual_value = V - exp(z) * (K @ H0).
    d_write = d_delta @ residual_value.mT
    d_residual_value = write.mT @ d_delta
    d_state_read = -exp_z[:, None] * d_residual_value
    direct_state = d_state_read @ h0.mT

    # output_read = scale * causal * (Q @ K.T).
    d_qk = float(scale) * causal * d_output_read
    dp = d_qk.mT @ q

    # write = causal * inverse * beta[column],
    # inverse = KKT^{-1},
    # KKT = I + strict_lower(diag(beta) @ (K @ K.T)).
    d_inverse = causal * beta[None, :] * d_write
    d_kkt = -inverse.mT @ d_inverse @ inverse.mT
    d_kkt = torch.tril(d_kkt, diagonal=-1)
    d_gram = beta[:, None] * d_kkt
    kkt = (d_gram + d_gram.mT) @ k

    return DirectAOutputOnlyDKComponents(
        direct_state,
        dp,
        kkt,
        direct_state + dp + kkt,
    )


def direct_a_chunk_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    output_grad: torch.Tensor,
    end_state_grad: torch.Tensor,
    *,
    scale: float,
) -> DirectAChunkBackward:
    """Manual VJP for one valid direct-A chunk.

    ``g`` is the raw per-token log gate.  Internally the forward uses
    ``z = cumsum(g)``; consequently this function performs the final reverse
    cumsum before returning ``dg``.
    """

    sequence_length, key_dim, value_dim = _check_shapes(
        q,
        k,
        v,
        beta,
        g,
        h0,
    )
    if output_grad.shape != (sequence_length, value_dim):
        raise ValueError(
            f"output_grad must have shape {(sequence_length, value_dim)}."
        )
    if end_state_grad.shape != (key_dim, value_dim):
        raise ValueError(
            f"end_state_grad must have shape {(key_dim, value_dim)}."
        )

    saved = _direct_a_intermediates(q, k, v, beta, g, h0, scale)
    exp_z = saved["exp_z"]
    decay = saved["decay"]
    reverse_decay = saved["reverse_decay"]
    gram = saved["gram"]
    inverse = saved["inverse"]
    causal = saved["causal"]
    write = saved["write"]
    state_read = saved["state_read"]
    residual_value = saved["residual_value"]
    delta = saved["delta"]
    normalized_delta = saved["normalized_delta"]
    output_read = saved["output_read"]

    # H1 = decay H0 + K.T @ normalized_delta
    d_normalized_delta = k @ end_state_grad
    dk = normalized_delta @ end_state_grad.mT
    dh0 = decay * end_state_grad
    d_decay = torch.sum(end_state_grad * h0)

    # normalized_delta = reverse_decay * delta
    d_delta = reverse_decay[:, None] * d_normalized_delta
    d_reverse_decay = torch.sum(d_normalized_delta * delta, dim=1)

    # O = scale * exp(z) * (Q H0) + output_read @ delta
    q_state = q @ h0
    scaled_output_grad = float(scale) * exp_z[:, None] * output_grad
    dq = scaled_output_grad @ h0.mT
    dh0 = dh0 + q.mT @ scaled_output_grad
    d_exp_z = float(scale) * torch.sum(output_grad * q_state, dim=1)
    d_output_read = output_grad @ delta.mT
    d_delta = d_delta + output_read.mT @ output_grad

    # delta = write @ residual_value
    d_write = d_delta @ residual_value.mT
    d_residual_value = write.mT @ d_delta

    # residual_value = V - exp(z) * (K H0)
    dv = d_residual_value
    d_state_read = -exp_z[:, None] * d_residual_value
    d_exp_z = d_exp_z - torch.sum(d_residual_value * state_read, dim=1)
    dk = dk + d_state_read @ h0.mT
    dh0 = dh0 + k.mT @ d_state_read

    # output_read = scale * causal * (Q K.T)
    d_qk = float(scale) * causal * d_output_read
    dq = dq + d_qk @ k
    dk = dk + d_qk.mT @ q

    # write = causal * inverse * beta[column]
    d_inverse = causal * beta[None, :] * d_write
    dbeta = torch.sum(causal * inverse * d_write, dim=0)

    # inverse = KKT^{-1};
    # KKT = I + strict_lower(diag(beta) @ (K K.T)).
    d_kkt = -inverse.mT @ d_inverse @ inverse.mT
    d_kkt = torch.tril(d_kkt, diagonal=-1)
    dbeta = dbeta + torch.sum(d_kkt * gram, dim=1)
    d_gram = beta[:, None] * d_kkt
    dk = dk + (d_gram + d_gram.mT) @ k

    # z-gradient from reverse_decay = decay / exp(z).
    ratio_log_grad = reverse_decay * d_reverse_decay
    dz = -ratio_log_grad
    dz[-1] = dz[-1] + torch.sum(ratio_log_grad) + decay * d_decay

    # Direct exp(z) paths in the output and residual value.
    dz = dz + exp_z * d_exp_z

    # Both causal matrices have d log(causal_ij) / dz =
    # e_i - e_j.  Invalid/future entries are already zero.
    causal_log_grad = (
        output_read * d_output_read
        + write * d_write
    )
    dz = (
        dz
        + torch.sum(causal_log_grad, dim=1)
        - torch.sum(causal_log_grad, dim=0)
    )

    # z_i = sum_{j<=i} g_j.
    dg = torch.flip(
        torch.cumsum(torch.flip(dz, dims=(0,)), dim=0),
        dims=(0,),
    )
    return DirectAChunkBackward(dq, dk, dv, dbeta, dg, dh0)
