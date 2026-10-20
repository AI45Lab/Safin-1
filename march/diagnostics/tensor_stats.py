"""Rank metadata and tensor statistics shared by MARCH diagnostics."""

from __future__ import annotations

import hashlib
import os
from typing import Any
import torch


def _dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0") or 0)

def _dist_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1") or 1)

def _mpu_rank(fn_name: str, default: int = 0) -> int:
    try:
        from megatron.core import mpu

        return int(getattr(mpu, fn_name)())
    except Exception:
        return default

def rank_info() -> dict[str, int]:
    return {
        "rank": _dist_rank(),
        "world_size": _dist_world_size(),
        "data_parallel_rank": _mpu_rank("get_data_parallel_rank"),
        "tensor_model_parallel_rank": _mpu_rank("get_tensor_model_parallel_rank"),
        "pipeline_model_parallel_rank": _mpu_rank("get_pipeline_model_parallel_rank"),
        "context_parallel_rank": _mpu_rank("get_context_parallel_rank"),
    }

def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().contiguous()
    if contiguous.dim() == 0:
        contiguous = contiguous.reshape(1)
    raw = contiguous.view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()

def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        detached = tensor.detach()
        numel = int(detached.numel())
        if numel == 0:
            return {"finite": True, "norm": 0.0, "max_abs": 0.0, "mean_abs": 0.0}
        if detached.is_floating_point() or detached.is_complex():
            finite = bool(torch.isfinite(detached).all().item())
        else:
            finite = True
        if detached.is_complex():
            values = detached.abs().float()
        elif detached.dtype == torch.bool:
            values = detached.to(dtype=torch.float32)
        else:
            values = detached.float()
        abs_values = values.abs()
        return {
            "finite": finite,
            "norm": float(torch.linalg.vector_norm(values).item()),
            "max_abs": float(abs_values.max().item()),
            "mean_abs": float(abs_values.mean().item()),
        }

def _jsonable_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_metadata(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable_metadata(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _jsonable_metadata(item) for key, item in value.items()}
    return repr(value)
