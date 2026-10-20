"""MARCH paged history: paged utils."""

from __future__ import annotations

import json
import os
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU reference remains available without Triton.
    triton = None
    tl = None


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _normalize_value_block_size(value_dim: int, block: int, name: str) -> int:
    if block <= 0:
        raise ValueError(f"{name} must be positive, got {block}.")
    block = min(block, _next_power_of_2(value_dim))
    return _next_power_of_2(block)


def _value_block_size(value_dim: int) -> int:
    env_value = os.environ.get("HSD_PAGED_HISTORY_VALUE_BLOCK_SIZE", "").strip()
    block = int(env_value) if env_value else (8 if value_dim >= 64 else 16)
    return _normalize_value_block_size(
        value_dim,
        block,
        "HSD_PAGED_HISTORY_VALUE_BLOCK_SIZE",
    )


def _paged_history_num_warps(name: str, default: int = 4) -> int:
    raw = os.environ.get(name, str(default)).strip()
    value = int(raw)
    if value not in {1, 2, 4, 8}:
        raise ValueError(f"{name} must be one of 1, 2, 4, or 8; got {value}.")
    return value


def _history_read_num_warps() -> int:
    return _paged_history_num_warps("HSD_PAGED_HISTORY_READ_NUM_WARPS", default=4)


def _history_read_impl(backend: str | None = None) -> str:
    """Compatibility resolver; explicit calls should pass their resolved policy."""
    from .backend_policy import PagedHistoryReadPolicy

    return PagedHistoryReadPolicy.resolve(backend=backend, validate_fused=False).backend


def _paged_history_timing_enabled() -> bool:
    value = os.environ.get("VLLM_HSD_PAGED_HISTORY_TIMING", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _export_cache_events_enabled() -> bool:
    return _env_truthy("VLLM_HSD_PAGED_HISTORY_EXPORT_CACHE_EVENTS")


def _record_paged_history_event(event: str, **payload) -> None:
    counter_path = os.environ.get("VLLM_HSD_VARLEN_COUNTER_FILE")
    if not counter_path:
        return
    record = {
        "event": event,
        "pid": os.getpid(),
        **payload,
    }
    try:
        with open(counter_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        pass


def _record_prefill_top4_locality(
    top_indices: torch.Tensor,
    text_offsets: torch.Tensor | None,
    *,
    tag: str | None,
    max_history_len: int,
) -> None:
    """Record the state-load reuse upper bound for real prefill selections.

    For each token block, head, and top-k rank, the selected anchor state can
    be loaded once for every distinct index instead of once per token.  This
    diagnostic computes that idealized ratio without changing selection or
    read order.  It is intentionally opt-in because it synchronizes small
    aggregate tensors back to the host.
    """
    output_path = os.environ.get("VLLM_HSD_PREFILL_TOP4_LOCALITY_FILE", "").strip()
    if not output_path or text_offsets is None:
        return
    if top_indices.ndim != 3 or int(top_indices.shape[-1]) != 4:
        return
    if text_offsets.ndim != 1 or int(text_offsets.numel()) < 2:
        return

    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
    raw_ranks = os.environ.get("VLLM_HSD_PREFILL_TOP4_LOCALITY_RANKS", "0").strip()
    if raw_ranks.lower() not in {"", "all", "*"}:
        selected_ranks = {int(value) for value in raw_ranks.split(",") if value.strip()}
        if rank not in selected_ranks:
            return

    raw_blocks = os.environ.get(
        "VLLM_HSD_PREFILL_TOP4_LOCALITY_BLOCK_SIZES", "4,8,16,32,64"
    )
    block_sizes = sorted(
        {int(value) for value in raw_blocks.split(",") if int(value) > 0}
    )
    if not block_sizes:
        return
    offsets = [int(value) for value in text_offsets.detach().cpu().tolist()]
    if offsets[0] != 0 or offsets[-1] != int(top_indices.shape[0]):
        return

    zero = torch.zeros((), device=top_indices.device, dtype=torch.int64)
    aggregates = {
        block_size: {
            "selected": zero.clone(),
            "unique": zero.clone(),
            "active_rows": zero.clone(),
            "single_unique_rows": zero.clone(),
            "blocks": 0,
        }
        for block_size in block_sizes
    }
    for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
        token_count = end - begin
        if token_count <= 0:
            continue
        selected = top_indices[begin:end]
        for block_size in block_sizes:
            block_count = (token_count + block_size - 1) // block_size
            padded_count = block_count * block_size
            if padded_count != token_count:
                padded = torch.full(
                    (
                        padded_count - token_count,
                        int(selected.shape[1]),
                        int(selected.shape[2]),
                    ),
                    -1,
                    device=selected.device,
                    dtype=selected.dtype,
                )
                blocked = torch.cat((selected, padded), dim=0)
            else:
                blocked = selected
            # [block, head, topk_rank, token]
            blocked = blocked.view(
                block_count,
                block_size,
                int(selected.shape[1]),
                int(selected.shape[2]),
            ).permute(0, 2, 3, 1)
            sorted_indices = blocked.sort(dim=-1).values
            valid = sorted_indices >= 0
            unique = valid[..., 0].to(torch.int64)
            if block_size > 1:
                unique = unique + (
                    valid[..., 1:]
                    & (sorted_indices[..., 1:] != sorted_indices[..., :-1])
                ).sum(dim=-1, dtype=torch.int64)
            active = unique > 0
            row = aggregates[block_size]
            row["selected"] = row["selected"] + valid.sum(dtype=torch.int64)
            row["unique"] = row["unique"] + unique.sum(dtype=torch.int64)
            row["active_rows"] = row["active_rows"] + active.sum(dtype=torch.int64)
            row["single_unique_rows"] = row["single_unique_rows"] + (
                active & (unique == 1)
            ).sum(dtype=torch.int64)
            row["blocks"] = int(row["blocks"]) + block_count

    scalar_tensors: list[torch.Tensor] = []
    for block_size in block_sizes:
        row = aggregates[block_size]
        scalar_tensors.extend(
            [
                row["selected"],
                row["unique"],
                row["active_rows"],
                row["single_unique_rows"],
            ]
        )
    values = torch.stack(scalar_tensors).detach().cpu().tolist()
    block_rows: dict[str, dict[str, float | int]] = {}
    cursor = 0
    for block_size in block_sizes:
        selected_count, unique_count, active_rows, single_unique_rows = map(
            int, values[cursor : cursor + 4]
        )
        cursor += 4
        block_rows[str(block_size)] = {
            "blocks": int(aggregates[block_size]["blocks"]),
            "selected_state_loads": selected_count,
            "unique_state_loads": unique_count,
            "ideal_state_load_speedup": (
                float(selected_count) / float(unique_count) if unique_count else 1.0
            ),
            "ideal_state_load_reduction": (
                1.0 - float(unique_count) / float(selected_count)
                if selected_count
                else 0.0
            ),
            "single_unique_row_share": (
                float(single_unique_rows) / float(active_rows) if active_rows else 0.0
            ),
        }

    record = {
        "event": "hsd_prefill_top4_locality",
        "pid": os.getpid(),
        "rank": rank,
        "tag": tag,
        "requests": len(offsets) - 1,
        "tokens": int(top_indices.shape[0]),
        "heads": int(top_indices.shape[1]),
        "topk": int(top_indices.shape[2]),
        "max_history_len": int(max_history_len),
        "blocks": block_rows,
    }
    try:
        line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except OSError:
        pass


def _cuda_elapsed_us(start: torch.cuda.Event, end: torch.cuda.Event) -> int:
    end.synchronize()
    return int(max(0.0, start.elapsed_time(end)) * 1000.0)


def _normalize_state_format(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized != "dense":
        raise ValueError(
            "MARCH history cache supports only state_format='dense'; "
            f"got {value!r}."
        )
    return normalized
