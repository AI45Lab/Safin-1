# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared selectors and state predicates without native vLLM imports."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.transformers_utils.configs.safin1 import Safin1TextConfig
    from vllm.transformers_utils.configs.safin1_moe import Safin1MoeTextConfig


def _parse_layer_selector(
    value: str | list[int] | tuple[int, ...] | set[int] | int | None,
) -> set[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "*", "all"}:
            return None
        if raw in {"0", "false", "none", "off"}:
            return set()
        layers: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s.strip()), int(end_s.strip())
                if start <= 0 or end < start:
                    raise ValueError(f"Invalid HSD layer selector range: {part!r}")
                layers.update(range(start, end + 1))
            else:
                layer = int(part)
                if layer <= 0:
                    raise ValueError(f"Invalid HSD layer number: {part!r}")
                layers.add(layer)
        return layers
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"Invalid HSD layer number: {value}")
        return {int(value)}
    layers = {int(v) for v in value}
    invalid = sorted(layer for layer in layers if layer <= 0)
    if invalid:
        raise ValueError(f"Invalid HSD layer number(s): {invalid}")
    return layers


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw.strip())


def _parse_csv_selector(value: str | None) -> set[str] | None:
    if value is None:
        return None
    raw = value.strip().lower()
    if raw in {"", "*", "all"}:
        return None
    if raw in {"0", "false", "none", "off"}:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def _parse_int_csv_selector(value: str | None) -> set[int] | None:
    if value is None:
        return None
    raw = value.strip().lower()
    if raw in {"", "*", "all"}:
        return None
    if raw in {"false", "none", "off"}:
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _anchor_enabled(config: Safin1MoeTextConfig) -> bool:
    return (
        (
            bool(getattr(config, "hist_replace_linear_attn", True))
            or bool(getattr(config, "hist_replace_self_attn", False))
        )
        and int(getattr(config, "anchor_chunk_size", 0) or 0) > 0
        and int(getattr(config, "anchor_token_num", 0) or 0) > 0
    )


def _persistent_memory_mode(
    config: Safin1TextConfig | Safin1MoeTextConfig,
) -> str:
    configured = getattr(config, "_vllm_hsd_persistent_memory_mode", None)
    if configured is not None:
        return str(configured)
    return (
        "enabled"
        if bool(getattr(config, "hsd_persistent_memory_enabled", False))
        else "removed"
    )


def _persistent_memory_insert_enabled(
    config: Safin1TextConfig | Safin1MoeTextConfig,
) -> bool:
    return bool(getattr(config, "hsd_persistent_memory_enabled", False)) and (
        _persistent_memory_mode(config) in {"enabled", "zero"}
    )


def _hsd_history_available(
    config: Safin1TextConfig | Safin1MoeTextConfig,
    past_text_len: int,
    anchor_chunk_size: int | None = None,
) -> bool:
    """Return whether decode has a cached row that global read may consume."""

    past_text_len = int(past_text_len)
    chunk_size = int(
        anchor_chunk_size
        if anchor_chunk_size is not None
        else (getattr(config, "anchor_chunk_size", 0) or 0)
    )
    dynamic_history = chunk_size > 0 and past_text_len >= chunk_size
    memory_history = _persistent_memory_insert_enabled(config) and past_text_len > 0
    return dynamic_history or memory_history


def _is_valid_hsd_state_index(state_idx: int) -> bool:
    # vLLM reserves block 0 as NULL_BLOCK_ID for padding rows.
    return int(state_idx) > 0


def _advance_hsd_conv_state(
    previous_state: torch.Tensor,
    token_inputs: torch.Tensor,
    *,
    has_initial_state: bool = True,
) -> torch.Tensor:
    """Return the conv cache state after appending pre-conv token inputs."""
    if previous_state.ndim != 3:
        raise ValueError(
            "HSD conv state snapshot must be [1, dim, state_len], got "
            f"{tuple(previous_state.shape)}."
        )
    if token_inputs.ndim != 2:
        raise ValueError(
            "HSD conv token inputs must be [tokens, dim], got "
            f"{tuple(token_inputs.shape)}."
        )
    state_len = int(previous_state.shape[-1])
    if not has_initial_state:
        previous_state = torch.zeros_like(previous_state)
    if state_len <= 0 or int(token_inputs.shape[0]) == 0:
        return previous_state.contiguous()
    tokens = token_inputs.to(
        device=previous_state.device,
        dtype=previous_state.dtype,
    ).transpose(0, 1)
    if int(tokens.shape[0]) != int(previous_state.shape[1]):
        raise ValueError(
            "HSD conv token dim does not match state dim: "
            f"tokens={tuple(tokens.shape)}, state={tuple(previous_state.shape)}."
        )
    if int(tokens.shape[1]) >= state_len:
        return tokens[:, -state_len:].unsqueeze(0).contiguous()
    keep = previous_state[:, :, int(tokens.shape[1]) :]
    return torch.cat([keep, tokens.unsqueeze(0)], dim=-1).contiguous()


def _l2norm(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((tensor * tensor).sum(dim=-1, keepdim=True) + eps)
    return tensor * inv_norm


def _skip_redundant_recurrent_state_store_enabled() -> bool:
    """Return the decode-state-store policy with an explicit disable override."""
    if _env_truthy("VLLM_HSD_DISABLE_SKIP_REDUNDANT_RECURRENT_STATE_STORE"):
        return False
    return _env_truthy(
        "VLLM_HSD_SKIP_REDUNDANT_RECURRENT_STATE_STORE",
        default=True,
    )


def _compact_all_layers_fast_path_enabled(req_count: int) -> bool:
    del req_count
    return not _env_truthy("VLLM_HSD_DISABLE_COMPACT_ALL_LAYERS_FAST_PATH")
