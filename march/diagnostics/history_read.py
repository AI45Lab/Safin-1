"""Opt-in paged history comparison without changing process execution policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch


def compare_fused_history_read(
    output: torch.Tensor,
    *,
    read: Callable[..., torch.Tensor | None],
    read_kwargs: Mapping[str, Any],
    record_event: Callable[..., None],
    batch: int,
    max_pages: int,
    max_len: int,
) -> None:
    """Compare the split result with a fused call using the same semantic inputs.

    This diagnostic synchronizes tensors to the host and is unsuitable for CUDA
    graph capture. It records differences; it does not certify a tolerance.
    Backend overrides are local kwargs, including when the nested call fails.
    """
    if output.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Fused history diagnostics must run outside CUDA graph capture")
    kwargs = dict(read_kwargs)
    kwargs.update(backend="fused_triton", validate_fused=False)
    fused_output = read(**kwargs)
    metadata = dict(batch=int(batch), max_pages=int(max_pages), max_len=int(max_len))
    if fused_output is None:
        record_event("paged_history_fused_validate", status="fused_none", **metadata)
        return
    diff = (output.float() - fused_output.float()).abs()
    record_event(
        "paged_history_fused_validate", status="compared", **metadata,
        finite=bool(torch.isfinite(diff).all().item()),
        mean_abs=float(diff.mean().item()) if diff.numel() else 0.0,
        max_abs=float(diff.max().item()) if diff.numel() else 0.0,
    )
