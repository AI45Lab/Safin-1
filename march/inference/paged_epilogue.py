"""MARCH paged epilogue: execution and capability checks."""

from __future__ import annotations

import torch

try:
    import triton
except ImportError:  # CPU reference remains available without Triton.
    triton = None

if triton is not None:
    from .paged_triton import (
        _history_scale_add_kernel,
        _strided_gate_rmsnorm_kernel,
        _valid_mask_copy_kernel,
    )
from .paged_utils import (
    _next_power_of_2,
)


def fused_history_scale_add_(
    base_out: torch.Tensor,
    history_out: torch.Tensor,
    scale: torch.Tensor | float,
) -> bool:
    """Fuse fixed-shape history scaling and full-batch accumulation in-place."""
    if triton is None or not base_out.is_cuda or not history_out.is_cuda:
        return False
    if base_out.ndim != 3 or tuple(base_out.shape) != tuple(history_out.shape):
        return False
    if base_out.device != history_out.device or base_out.dtype != history_out.dtype:
        return False
    if not base_out.is_contiguous() or not history_out.is_contiguous():
        return False
    batch_size, heads, value_dim = base_out.shape
    if batch_size <= 0 or heads <= 0 or value_dim <= 0 or value_dim > 128:
        return False
    if torch.is_tensor(scale):
        if scale.numel() != 1:
            return False
        scale_tensor = scale.to(device=base_out.device, dtype=torch.float32).reshape(1)
    else:
        scale_tensor = torch.tensor(
            [float(scale)],
            device=base_out.device,
            dtype=torch.float32,
        )
    block_v = _next_power_of_2(int(value_dim))
    with torch.cuda.device(base_out.device):
        _history_scale_add_kernel[(batch_size, heads)](
            base_out,
            history_out,
            scale_tensor,
            value_dim=int(value_dim),
            base_stride_b=base_out.stride(0),
            base_stride_h=base_out.stride(1),
            base_stride_v=base_out.stride(2),
            history_stride_b=history_out.stride(0),
            history_stride_h=history_out.stride(1),
            history_stride_v=history_out.stride(2),
            BLOCK_V=block_v,
            num_warps=4,
        )
    return True


def fused_valid_mask_copy_(
    output: torch.Tensor,
    source: torch.Tensor,
    valid_mask: torch.Tensor,
) -> bool:
    """Copy a fixed decode output and zero padded rows in one launch."""
    if triton is None or not output.is_cuda or not source.is_cuda:
        return False
    if output.ndim != 2 or tuple(output.shape) != tuple(source.shape):
        return False
    if output.device != source.device or output.dtype != source.dtype:
        return False
    if not output.is_contiguous() or not source.is_contiguous():
        return False
    if (
        valid_mask.ndim != 1
        or int(valid_mask.shape[0]) != int(output.shape[0])
        or valid_mask.device != output.device
        or valid_mask.dtype != torch.bool
    ):
        return False
    numel = int(output.numel())
    hidden_dim = int(output.shape[1])
    if numel <= 0 or hidden_dim <= 0:
        return False
    block = 256
    with torch.cuda.device(output.device):
        _valid_mask_copy_kernel[(triton.cdiv(numel, block),)](
            source,
            valid_mask,
            output,
            numel=numel,
            hidden_dim=hidden_dim,
            BLOCK=block,
            num_warps=4,
        )
    return True


def fused_strided_gate_rmsnorm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    activation: str,
) -> torch.Tensor | None:
    """RMSNorm + output gate without materializing a packed gate copy."""
    if triton is None or not x.is_cuda or not gate.is_cuda:
        return None
    if x.ndim != 3 or gate.ndim != 3 or tuple(x.shape) != tuple(gate.shape):
        return None
    if x.device != gate.device or x.dtype != gate.dtype:
        return None
    if x.stride(-1) != 1 or gate.stride(-1) != 1:
        return None
    if weight.ndim != 1 or int(weight.shape[0]) != int(x.shape[2]):
        return None
    if weight.device != x.device or weight.stride(0) != 1:
        return None
    activation = activation.lower()
    if activation not in {"silu", "swish", "sigmoid"}:
        return None
    batch_size, heads, value_dim = map(int, x.shape)
    if batch_size <= 0 or heads <= 0 or value_dim <= 0 or value_dim > 128:
        return None
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    block_v = _next_power_of_2(value_dim)
    with torch.cuda.device(x.device):
        _strided_gate_rmsnorm_kernel[(batch_size * heads,)](
            x,
            gate,
            weight,
            output,
            heads=heads,
            value_dim=value_dim,
            eps=float(eps),
            x_stride_b=x.stride(0),
            x_stride_h=x.stride(1),
            x_stride_v=x.stride(2),
            gate_stride_b=gate.stride(0),
            gate_stride_h=gate.stride(1),
            gate_stride_v=gate.stride(2),
            out_stride_b=output.stride(0),
            out_stride_h=output.stride(1),
            out_stride_v=output.stride(2),
            SIGMOID_GATE=activation == "sigmoid",
            BLOCK_V=block_v,
            num_warps=max(1, min(8, block_v // 256)),
        )
    return output

