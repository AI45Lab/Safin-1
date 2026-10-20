"""Small helpers and kernel imports for Safin-1-MoE."""

import torch


def route_softmax_weights(logits: torch.Tensor, valid: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if valid.dtype is not torch.bool:
        valid = valid.bool()

    weights = torch.softmax(logits.float().masked_fill(~valid, float("-inf")), dim=dim)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights * valid.to(dtype=weights.dtype)
    denom = weights.sum(dim=dim, keepdim=True)
    weights = torch.where(denom > 0, weights / denom, torch.zeros_like(weights))
    return weights.to(dtype=logits.dtype)


__all__ = ["route_softmax_weights"]
