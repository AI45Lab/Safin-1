# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch

from .contiguous import input_guard


@torch.compile
def l2norm_fwd_compiled(
    x: torch.Tensor,
    eps: float = 1e-6,
    dim: int = -1,
):
    x_f32 = x.to(torch.float32)
    sum_sq = (x_f32 * x_f32).sum(dim=dim, keepdim=True) + eps
    rstd = torch.rsqrt(sum_sq)
    y = (x * rstd).to(x.dtype)
    return y, rstd


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    dim: int = -1,
):
    assert dim == -1 or dim == len(x.shape) - 1
    assert x.stride(-1) == 1
    raw_shape = x.shape
    x = x.view(-1, raw_shape[-1])
    torch._dynamo.mark_dynamic(x, 0)
    y, rstd = l2norm_fwd_compiled(x, eps, dim)
    y = y.view(raw_shape)
    rstd = rstd.view(*raw_shape[:-1])
    return y, rstd


@torch.compile
def l2norm_bwd_compiled(
    dy: torch.Tensor,
    y: torch.Tensor,
    rstd: torch.Tensor,
    eps: float = 1e-6,
    dim: int = -1,
):
    y_f32 = y.to(torch.float32)
    dy_f32 = dy.to(torch.float32)
    dot = (dy_f32 * y_f32).sum(dim=-1, keepdim=True)
    dx = (dy_f32 - dot * y_f32) * rstd
    return dx


def l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
    dim: int = -1,
):
    assert dim == -1 or dim == len(dy.shape) - 1
    assert y.stride(-1) == 1
    assert dy.stride(-1) == 1
    raw_shape = dy.shape
    y = y.view(-1, raw_shape[-1])
    dy = dy.view(-1, raw_shape[-1])
    rstd = rstd.view(-1, 1)
    torch._dynamo.mark_dynamic(y, 0)
    torch._dynamo.mark_dynamic(dy, 0)
    torch._dynamo.mark_dynamic(rstd, 0)
    dx = l2norm_bwd_compiled(dy, y, rstd, eps, dim)
    dx = dx.view(raw_shape)
    return dx


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        eps=1e-6,
    ):
        y, rstd = l2norm_fwd(x, eps)
        ctx.eps = eps
        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors
        dx = l2norm_bwd(y, rstd, dy, ctx.eps)
        return dx, None, None


def l2norm(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps)
