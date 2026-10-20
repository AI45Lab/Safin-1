# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared training runtime selectors, independent of attention/diagnostic owners."""

import os
from typing import Optional

import torch


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw if raw else default


def _rank_allowed_from_env(env_name: str, default: str = "0") -> bool:
    ranks = (
        os.environ.get(env_name, os.environ.get("HSD_GRAD_DEBUG_RANKS", default))
        .strip()
        .lower()
    )
    if ranks in {"all", "*"}:
        return True
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return ranks in {"", "0", "rank0"}
    rank = torch.distributed.get_rank()
    if ranks in {"last", "last_rank"}:
        return rank == torch.distributed.get_world_size() - 1
    allowed = {item.strip() for item in ranks.split(",") if item.strip()}
    return str(rank) in allowed or f"rank{rank}" in allowed


def _precision_trace_internal_layer_allowed(layer_number: int) -> bool:
    layers = _parse_layer_selector(
        os.environ.get("MEGATRON_PRECISION_TRACE_INTERNAL_LAYERS") or "1"
    )
    return layers is None or int(layer_number) in layers


def _parse_layer_selector(value):
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "*", "all"}:
            return None
        if raw in {"0", "false", "none", "off"}:
            return set()
        layers = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                start = int(start.strip())
                end = int(end.strip())
                if start <= 0 or end <= 0 or end < start:
                    raise ValueError(f"Invalid HSD layer range: {part}")
                layers.update(range(start, end + 1))
            else:
                layer = int(part)
                if layer <= 0:
                    raise ValueError(f"Invalid HSD layer number: {part}")
                layers.add(layer)
        return layers
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"Invalid HSD layer number: {value}")
        return {value}
    if isinstance(value, (list, tuple, set)):
        layers = set()
        for item in value:
            layer = int(item)
            if layer <= 0:
                raise ValueError(f"Invalid HSD layer number: {item}")
            layers.add(layer)
        return layers
    raise ValueError(f"Unsupported HSD layer selector: {value!r}")
