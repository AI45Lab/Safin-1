"""Bitwise tensor tracing helpers for Megatron precision comparisons."""

from __future__ import annotations

from .tensor_stats import (
    _dist_rank,
    _dist_world_size,
    _jsonable_metadata,
    _mpu_rank,
    _tensor_sha256,
    _tensor_stats,
    rank_info,
)

import contextlib
from contextvars import ContextVar
from dataclasses import fields

from march.safin1.configuration_common import Safin1RuntimeFields

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


_ACTIVE_TRACER: ContextVar["PrecisionTracer | None"] = ContextVar("march_precision_tracer", default=None)

# Consume the model schema; inactive aliases/retired options are not runtime diagnostics.
HSD_RUNTIME_METADATA_KEYS = tuple(
    field.name for field in fields(Safin1RuntimeFields)
    if field.name not in Safin1RuntimeFields.HSD_COMPATIBILITY_FIELDS + Safin1RuntimeFields.HSD_RETIRED_FIELDS
)
HSD_TRAINING_ONLY_METADATA_FIELDS = Safin1RuntimeFields.HSD_TRAINING_ONLY_FIELDS


def set_active_precision_tracer(tracer: "PrecisionTracer | None") -> None:
    _ACTIVE_TRACER.set(tracer)


def get_active_precision_tracer() -> "PrecisionTracer | None":
    return _ACTIVE_TRACER.get()


@contextlib.contextmanager
def active_precision_tracer(tracer: "PrecisionTracer | None"):
    """Scope a tracer without redirecting concurrent or nested model forwards."""
    token = _ACTIVE_TRACER.set(tracer)
    try:
        yield tracer
    finally:
        _ACTIVE_TRACER.reset(token)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in {None, ""}:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_hsd_extra_param(name: str) -> bool:
    return (
        name.endswith("anchor_embedding.weight")
        or name.endswith(".anchor_q_proj.weight")
        or name.endswith(".anchor_q_proj.layer_norm_weight")
        or name.endswith(".anchor_k_proj.weight")
        or name.endswith(".anchor_k_proj.layer_norm_weight")
        or name.endswith(".router_q_proj.weight")
        or name.endswith(".router_q_proj.layer_norm_weight")
        or name.endswith(".null_gate_proj.weight")
        or name.endswith(".null_gate_proj.layer_norm_weight")
        or name.endswith(".hsd_null_read_gate_bias")
        or name.endswith(".persistent_memory_state")
        or name.endswith(".persistent_memory_route_key")
    )


def _parse_steps(raw: str) -> set[int] | None:
    raw = raw.strip().lower()
    if raw in {"", "all", "*"}:
        return None if raw in {"all", "*"} else set()
    steps: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid MEGATRON_PRECISION_TRACE_STEPS range: {item}")
            steps.update(range(start, end + 1))
        else:
            steps.add(int(item))
    return steps


def _parse_layers(raw: str) -> set[int] | None:
    raw = raw.strip().lower()
    if raw in {"", "all", "*"}:
        return None
    layers: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid MEGATRON_PRECISION_TRACE_LAYERS range: {item}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    return layers


def _parse_ordinals(raw: str) -> set[int] | None:
    """Parse sidecar ordinals, with an empty/all specification meaning no filter."""
    raw = raw.strip().lower()
    if raw in {"", "all", "*"}:
        return None
    if raw == "none":
        return set()

    ordinals: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start < 0 or end < start:
                raise ValueError(f"Invalid MEGATRON_PRECISION_TRACE_SAVE_TENSOR_ORDINALS range: {item}")
            ordinals.update(range(start, end + 1))
        else:
            ordinal = int(item)
            if ordinal < 0:
                raise ValueError(f"Invalid MEGATRON_PRECISION_TRACE_SAVE_TENSOR_ORDINALS value: {item}")
            ordinals.add(ordinal)
    return ordinals


def precision_trace_runtime_step() -> int:
    value = os.environ.get("HSD_RUNTIME_STEP", "0")
    try:
        return max(0, int(value or 0)) + 1
    except ValueError:
        return 1


def _rank_allowed(spec: str, info: dict[str, int]) -> bool:
    spec = spec.strip().lower()
    if spec in {"", "dp0"}:
        return info["data_parallel_rank"] == 0
    if spec in {"all", "*"}:
        return True

    allowed = {item.strip() for item in spec.split(",") if item.strip()}
    rank = info["rank"]
    dp_rank = info["data_parallel_rank"]
    for item in allowed:
        if item in {str(rank), f"rank{rank}", f"global{rank}", f"global:{rank}"}:
            return True
        if item in {f"dp{dp_rank}", f"dp:{dp_rank}", f"data{dp_rank}", f"data:{dp_rank}"}:
            return True
    return False


def _load_runtime_config_value(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(key): _jsonable_metadata(item) for key, item in value.items() if item is not None}
    if not isinstance(value, str):
        return {}

    stripped = value.strip()
    if not stripped:
        return {}
    parsed = None
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
    else:
        path = Path(stripped)
        if path.exists():
            parsed = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(parsed, dict):
        return {str(key): _jsonable_metadata(item) for key, item in parsed.items() if item is not None}
    return {}


def _collect_hsd_runtime_metadata(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    runtime_config = _load_runtime_config_value(getattr(args, "hsd_runtime_config", None))
    for key in HSD_RUNTIME_METADATA_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            runtime_config[key] = _jsonable_metadata(value)
    return {
        key: runtime_config[key]
        for key in HSD_RUNTIME_METADATA_KEYS
        if key in runtime_config and runtime_config[key] is not None
    }


def _text_view(tensor: torch.Tensor, anchor_mask: torch.Tensor | None) -> torch.Tensor:
    if anchor_mask is None or not isinstance(anchor_mask, torch.Tensor):
        return tensor
    if tensor.shape[-1] != anchor_mask.shape[-1]:
        return tensor
    mask = ~anchor_mask.to(device=tensor.device, dtype=torch.bool)
    batch = anchor_mask.shape[0]
    text_len = int(mask[0].sum().item())
    if tensor.ndim == 2 and tuple(tensor.shape) == tuple(anchor_mask.shape):
        return tensor[mask].view(batch, text_len)
    if tensor.ndim == 3 and tuple(tensor.shape[1:]) == tuple(anchor_mask.shape):
        return tensor[:, mask].view(tensor.shape[0], batch, text_len)
    return tensor


def _anchor_layout_sidecars(anchor_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_mask = anchor_mask.to(dtype=torch.bool)
    if anchor_mask.ndim != 2:
        raise ValueError(f"anchor_mask must be [batch, expanded_seq], got {tuple(anchor_mask.shape)}.")
    if anchor_mask.shape[0] > 0 and not torch.equal(anchor_mask, anchor_mask[:1].expand_as(anchor_mask)):
        raise ValueError("anchor_mask must be identical across the traced batch for HSD trace alignment.")
    text_counts = (~anchor_mask).sum(dim=1)
    anchor_counts = anchor_mask.sum(dim=1)
    if not bool((text_counts == text_counts[:1]).all().item()):
        raise ValueError(f"anchor_mask has per-row text counts that differ: {text_counts.tolist()}.")
    if not bool((anchor_counts == anchor_counts[:1]).all().item()):
        raise ValueError(f"anchor_mask has per-row anchor counts that differ: {anchor_counts.tolist()}.")

    batch_size, expanded_len = anchor_mask.shape
    text_len = int(text_counts[0].item()) if batch_size else 0
    anchor_count = int(anchor_counts[0].item()) if batch_size else 0
    text_seen = (~anchor_mask).long().cumsum(dim=1)
    if anchor_count:
        anchor_boundaries = text_seen[anchor_mask].view(batch_size, anchor_count)
        text_index = torch.arange(text_len, dtype=torch.long, device=anchor_mask.device)
        anchor_read_valid_mask = anchor_boundaries[:, None, :] <= text_index.view(1, text_len, 1)
    else:
        anchor_boundaries = torch.empty(batch_size, 0, dtype=torch.long, device=anchor_mask.device)
        anchor_read_valid_mask = torch.zeros(batch_size, text_len, 0, dtype=torch.bool, device=anchor_mask.device)
    return anchor_boundaries, anchor_read_valid_mask


def _anchor_position_sidecars(anchor_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_mask = anchor_mask.to(dtype=torch.bool)
    if anchor_mask.ndim != 2:
        raise ValueError(f"anchor_mask must be [batch, expanded_seq], got {tuple(anchor_mask.shape)}.")
    if anchor_mask.shape[0] > 0 and not torch.equal(anchor_mask, anchor_mask[:1].expand_as(anchor_mask)):
        raise ValueError("anchor_mask must be identical across the traced batch for HSD trace alignment.")
    batch_size, expanded_len = anchor_mask.shape
    expanded_index = torch.arange(expanded_len, dtype=torch.long, device=anchor_mask.device)
    text_positions = expanded_index[~anchor_mask[0]].view(1, -1).expand(batch_size, -1)
    anchor_positions = expanded_index[anchor_mask[0]].view(1, -1).expand(batch_size, -1)
    return text_positions, anchor_positions


def _anchor_config_sidecars(anchor_mask: torch.Tensor, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    anchor_mask = anchor_mask.to(device=input_ids.device, dtype=torch.bool)
    if anchor_mask.ndim != 2 or input_ids.shape != anchor_mask.shape:
        raise ValueError(
            f"anchor config sidecars require input_ids and anchor_mask with matching [batch, seq] shapes: "
            f"input_ids={tuple(input_ids.shape)} anchor_mask={tuple(anchor_mask.shape)}"
        )
    if not bool(anchor_mask.any().item()):
        return {}

    first_mask = anchor_mask[0]
    anchor_indices = torch.nonzero(first_mask, as_tuple=False).flatten()
    run_lengths: list[int] = []
    run_start = int(anchor_indices[0].item())
    previous = run_start
    for raw_index in anchor_indices[1:].tolist():
        index = int(raw_index)
        if index == previous + 1:
            previous = index
            continue
        run_lengths.append(previous - run_start + 1)
        run_start = index
        previous = index
    run_lengths.append(previous - run_start + 1)
    if not run_lengths or any(length != run_lengths[0] for length in run_lengths):
        raise ValueError(f"anchor_mask has non-uniform anchor-token runs: {run_lengths}.")
    anchor_token_num = int(run_lengths[0])

    anchor_boundaries, _ = _anchor_layout_sidecars(anchor_mask)
    first_boundaries = anchor_boundaries[0]
    if first_boundaries.numel() % anchor_token_num != 0:
        raise ValueError(
            f"anchor boundary count {first_boundaries.numel()} is not divisible by anchor_token_num "
            f"{anchor_token_num}."
        )
    boundary_groups = first_boundaries.view(-1, anchor_token_num)
    if not bool((boundary_groups == boundary_groups[:, :1]).all().item()):
        raise ValueError(f"anchor boundaries are not constant within anchor-token groups: {first_boundaries.tolist()}.")
    chunk_boundaries = boundary_groups[:, 0]
    anchor_chunk_size = int(chunk_boundaries[0].item())
    if chunk_boundaries.numel() > 1:
        expected = torch.arange(
            anchor_chunk_size,
            anchor_chunk_size * (chunk_boundaries.numel() + 1),
            anchor_chunk_size,
            dtype=chunk_boundaries.dtype,
            device=chunk_boundaries.device,
        )
        if not torch.equal(chunk_boundaries, expected):
            raise ValueError(
                f"anchor boundaries do not form a regular chunk grid: "
                f"actual={chunk_boundaries.tolist()} expected={expected.tolist()}"
            )

    placeholder_values = input_ids[anchor_mask]
    if placeholder_values.numel() == 0:
        return {}
    if not bool((placeholder_values == placeholder_values[:1]).all().item()):
        raise ValueError(f"anchor placeholder ids differ across anchor positions: {placeholder_values.tolist()}.")
    return {
        "hsd_anchor_chunk_size": torch.tensor(anchor_chunk_size, dtype=torch.long, device=input_ids.device),
        "hsd_anchor_token_num": torch.tensor(anchor_token_num, dtype=torch.long, device=input_ids.device),
        "hsd_anchor_placeholder_token_id": placeholder_values[:1].to(dtype=torch.long).reshape(()),
    }


class PrecisionTracer:
    """Writes one JSONL manifest per rank for bitwise precision checks."""

    def __init__(self, trace_dir: str | os.PathLike[str], steps: set[int] | None, ranks: str):
        self.trace_dir = Path(trace_dir)
        self.steps = steps
        self.info = rank_info()
        self.allowed_rank = _rank_allowed(ranks, self.info)
        self.layer_io_enabled = env_flag("MEGATRON_PRECISION_TRACE_LAYER_IO", False)
        self.layer_filter = _parse_layers(os.environ.get("MEGATRON_PRECISION_TRACE_LAYERS", ""))
        target_param_regex = os.environ.get("MEGATRON_PRECISION_TRACE_TARGET_PARAM_REGEX", "").strip()
        self.target_param_pattern = re.compile(target_param_regex) if target_param_regex else None
        self.param_chain_enabled = env_flag("MEGATRON_PRECISION_TRACE_PARAM_CHAIN", False)
        param_chain_regex = os.environ.get("MEGATRON_PRECISION_TRACE_PARAM_CHAIN_REGEX", "").strip()
        if self.param_chain_enabled and not param_chain_regex:
            param_chain_regex = target_param_regex
            if not param_chain_regex:
                raise ValueError("Parameter-chain tracing requires MEGATRON_PRECISION_TRACE_PARAM_CHAIN_REGEX or TARGET_PARAM_REGEX")
        self.param_chain_pattern = re.compile(param_chain_regex) if param_chain_regex else None
        self.param_grads_enabled = env_flag("MEGATRON_PRECISION_TRACE_PARAM_GRADS", True)
        tensor_include_regex = os.environ.get("MEGATRON_PRECISION_TRACE_TENSOR_INCLUDE_REGEX", "").strip()
        self.tensor_include_pattern = re.compile(tensor_include_regex) if tensor_include_regex else None
        self.save_tensors = env_flag("MEGATRON_PRECISION_TRACE_SAVE_TENSORS", False)
        save_tensor_regex = os.environ.get("MEGATRON_PRECISION_TRACE_SAVE_TENSOR_REGEX", "").strip()
        self.save_tensor_pattern = re.compile(save_tensor_regex) if save_tensor_regex else None
        self.save_tensor_ordinals = _parse_ordinals(
            os.environ.get("MEGATRON_PRECISION_TRACE_SAVE_TENSOR_ORDINALS", "")
        )
        sha256_regex = os.environ.get("MEGATRON_PRECISION_TRACE_SHA256_REGEX", "").strip()
        self.sha256_pattern = re.compile(sha256_regex) if sha256_regex else None
        tensor_dir = os.environ.get("MEGATRON_PRECISION_TRACE_TENSOR_DIR", "").strip()
        self.tensor_dir = Path(tensor_dir) if tensor_dir else self.trace_dir / "tensors"
        self._counts: dict[tuple[int, str], int] = {}
        self._metadata_steps: set[int] = set()
        self._lock = threading.Lock()
        self._fh = None

    @classmethod
    def from_env(cls, default_dir: str | os.PathLike[str]) -> "PrecisionTracer | None":
        steps_raw = os.environ.get("MEGATRON_PRECISION_TRACE_STEPS", "")
        enabled = env_flag("MEGATRON_PRECISION_TRACE", False) or bool(steps_raw.strip())
        if not enabled:
            set_active_precision_tracer(None)
            return None
        trace_dir = os.environ.get("MEGATRON_PRECISION_TRACE_DIR") or str(default_dir)
        steps = _parse_steps(steps_raw or "1")
        ranks = os.environ.get("MEGATRON_PRECISION_TRACE_RANKS", "dp0")
        tracer = cls(trace_dir=trace_dir, steps=steps, ranks=ranks)
        if tracer.allowed_rank:
            set_active_precision_tracer(tracer)
            return tracer
        set_active_precision_tracer(None)
        return None

    def should_trace(self, step: int) -> bool:
        return self.steps is None or int(step) in self.steps

    def should_trace_layer(self, layer_number: int) -> bool:
        return self.layer_filter is None or int(layer_number) in self.layer_filter

    def should_trace_param_chain(self, param_name: str) -> bool:
        return (
            self.param_chain_enabled
            and self.param_chain_pattern is not None
            and bool(self.param_chain_pattern.search(param_name))
        )

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if get_active_precision_tracer() is self:
            set_active_precision_tracer(None)

    def _writer(self):
        if self._fh is None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.trace_dir / f"trace_rank{self.info['rank']:05d}.jsonl"
            self._fh = path.open("a", encoding="utf-8", buffering=1)
        return self._fh

    def _next_ordinal(self, step: int, name: str) -> int:
        key = (int(step), name)
        ordinal = self._counts.get(key, 0)
        self._counts[key] = ordinal + 1
        return ordinal

    @staticmethod
    def _safe_tensor_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name).strip("_")[:180] or "tensor"

    @staticmethod
    def _tensor_match_key(row: dict[str, Any]) -> str:
        return f"{row.get('category', '')}:{row.get('name', '')}"

    def _should_include_tensor(self, name: str, category: str) -> bool:
        return self.tensor_include_pattern is None or bool(
            self.tensor_include_pattern.search(f"{category}:{name}")
        )

    def _should_compute_sha256(self, row: dict[str, Any]) -> bool:
        return self.sha256_pattern is None or bool(self.sha256_pattern.search(self._tensor_match_key(row)))

    def _save_tensor_sidecar(self, row: dict[str, Any], tensor: torch.Tensor) -> None:
        if not self.save_tensors:
            return
        match_key = self._tensor_match_key(row)
        if self.save_tensor_pattern is not None and not self.save_tensor_pattern.search(match_key):
            return
        if self.save_tensor_ordinals is not None and int(row["ordinal"]) not in self.save_tensor_ordinals:
            return
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._safe_tensor_name(str(row.get("name") or "tensor"))
        filename = (
            f"step{int(row['step']):06d}_rank{int(row['rank']):05d}_"
            f"ord{int(row['ordinal']):04d}_{safe_name}.pt"
        )
        path = self.tensor_dir / filename
        torch.save(tensor.detach().cpu(), path)
        try:
            row["tensor_path"] = str(path.relative_to(self.trace_dir))
        except ValueError:
            row["tensor_path"] = str(path)

    def write_row(self, row: dict[str, Any]) -> None:
        with self._lock:
            json.dump(row, self._writer(), sort_keys=True)
            self._writer().write("\n")
            self._writer().flush()

    def trace_metadata(self, name: str, value: Any, *, step: int, category: str = "metadata") -> None:
        if not self.should_trace(step):
            return
        row: dict[str, Any] = {
            **self.info,
            "step": int(step),
            "name": name,
            "category": category,
            "ordinal": self._next_ordinal(int(step), name),
            "status": "metadata",
            "value": _jsonable_metadata(value),
        }
        self.write_row(row)

    def trace_hsd_runtime_metadata(self, args: Any, *, step: int) -> None:
        step = int(step)
        if step in self._metadata_steps:
            return
        self._metadata_steps.add(step)
        runtime_config = _collect_hsd_runtime_metadata(args)
        if runtime_config:
            self.trace_metadata("metadata/hsd_runtime_config", runtime_config, step=step)
        self.trace_metadata(
            "metadata/hsd_training_only_fields",
            list(HSD_TRAINING_ONLY_METADATA_FIELDS),
            step=step,
        )

    def trace_tensor(
        self,
        name: str,
        tensor: torch.Tensor | None,
        *,
        step: int,
        category: str,
        status: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.should_trace(step) or not self._should_include_tensor(name, category):
            return
        row: dict[str, Any] = {
            **self.info,
            "step": int(step),
            "name": name,
            "category": category,
            "ordinal": self._next_ordinal(int(step), name),
        }
        if extra:
            row.update(extra)
        if tensor is None:
            row.update({
                "status": status or "none",
                "shape": None,
                "dtype": None,
                "numel": 0,
                "sha256": None,
                "finite": None,
                "norm": None,
                "max_abs": None,
                "mean_abs": None,
            })
            self.write_row(row)
            return

        detached = tensor.detach()
        row.update({
            "status": status or "tensor",
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "numel": int(detached.numel()),
            "sha256": _tensor_sha256(detached) if self._should_compute_sha256(row) else None,
        })
        row.update(_tensor_stats(detached))
        self._save_tensor_sidecar(row, detached)
        self.write_row(row)

    def trace_layer_tensor(
        self,
        name: str,
        tensor: torch.Tensor | None,
        *,
        step: int,
        category: str,
        layer_number: int,
        status: str | None = None,
    ) -> None:
        if not self.layer_io_enabled or not self.should_trace_layer(layer_number):
            return
        self.trace_tensor(
            name,
            tensor,
            step=step,
            category=category,
            status=status,
            extra={"layer_number": int(layer_number)},
        )

    def trace_tensor_backward(
        self,
        name: str,
        tensor: torch.Tensor | None,
        *,
        step: int,
        category: str,
        layer_number: int | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
    ) -> bool:
        if (
            not isinstance(tensor, torch.Tensor)
            or not self.should_trace(step)
            or not self._should_include_tensor(name, category)
            or not tensor.requires_grad
        ):
            return False

        extra = {"layer_number": int(layer_number)} if layer_number is not None else None

        def _trace_grad(grad: torch.Tensor):
            traced = transform(grad) if transform is not None else grad
            if isinstance(traced, torch.Tensor):
                self.trace_tensor(name, traced, step=step, category=category, extra=extra)
            return grad

        tensor.register_hook(_trace_grad)
        return True

    def trace_batch(self, batch: dict[str, Any], *, labels: torch.Tensor | None, loss_scale: torch.Tensor | None,
                    step: int) -> None:
        if not self.should_trace(step):
            return
        anchor_mask = batch.get("anchor_mask")
        if isinstance(anchor_mask, torch.Tensor):
            self.trace_tensor("batch/anchor_mask", anchor_mask, step=step, category="hsd_extra")
            anchor_boundaries, anchor_read_valid_mask = _anchor_layout_sidecars(anchor_mask)
            text_positions, anchor_positions = _anchor_position_sidecars(anchor_mask)
            self.trace_tensor(
                "batch/text_positions",
                text_positions,
                step=step,
                category="hsd_extra",
            )
            self.trace_tensor(
                "batch/anchor_positions",
                anchor_positions,
                step=step,
                category="hsd_extra",
            )
            self.trace_tensor(
                "batch/anchor_boundaries",
                anchor_boundaries,
                step=step,
                category="hsd_extra",
            )
            self.trace_tensor(
                "batch/anchor_read_valid_mask",
                anchor_read_valid_mask,
                step=step,
                category="hsd_extra",
            )

        input_ids = batch.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            self.trace_tensor(
                "batch/input_ids_text",
                _text_view(input_ids, anchor_mask),
                step=step,
                category="batch",
            )
            if isinstance(anchor_mask, torch.Tensor):
                self.trace_tensor("batch/input_ids_raw", input_ids, step=step, category="hsd_extra")
                for sidecar_name, sidecar_tensor in _anchor_config_sidecars(anchor_mask, input_ids).items():
                    self.trace_tensor(
                        f"batch/{sidecar_name}",
                        sidecar_tensor,
                        step=step,
                        category="hsd_extra",
                    )

        mm_token_type_ids = batch.get("mm_token_type_ids")
        if not isinstance(mm_token_type_ids, torch.Tensor) and isinstance(input_ids, torch.Tensor):
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        if isinstance(mm_token_type_ids, torch.Tensor):
            self.trace_tensor(
                "batch/mm_token_type_ids_text",
                _text_view(mm_token_type_ids, anchor_mask),
                step=step,
                category="batch",
            )
            if isinstance(anchor_mask, torch.Tensor):
                self.trace_tensor(
                    "batch/mm_token_type_ids_raw",
                    mm_token_type_ids,
                    step=step,
                    category="hsd_extra",
                )

        attention_mask = batch.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            self.trace_tensor(
                "batch/attention_mask_text",
                _text_view(attention_mask, anchor_mask),
                step=step,
                category="batch",
            )
            if isinstance(anchor_mask, torch.Tensor):
                self.trace_tensor("batch/attention_mask_raw", attention_mask, step=step, category="hsd_extra")

        position_ids = batch.get("position_ids")
        if isinstance(position_ids, torch.Tensor):
            self.trace_tensor(
                "batch/position_ids_text",
                _text_view(position_ids, anchor_mask),
                step=step,
                category="batch",
            )
            if isinstance(anchor_mask, torch.Tensor):
                self.trace_tensor("batch/position_ids_raw", position_ids, step=step, category="hsd_extra")

        if isinstance(labels, torch.Tensor):
            self.trace_tensor("batch/labels", labels, step=step, category="batch")
        if isinstance(loss_scale, torch.Tensor):
            self.trace_tensor("batch/loss_scale", loss_scale, step=step, category="batch")
        else:
            self.trace_tensor("batch/loss_scale", None, step=step, category="batch", status="none")

    def trace_output_tensor(self, output_tensor: Any, *, step: int) -> None:
        if not isinstance(output_tensor, torch.Tensor) or not self.should_trace(step):
            return
        self.trace_tensor("forward/output_tensor", output_tensor, step=step, category="forward")
        if output_tensor.requires_grad:

            def _trace_grad(grad: torch.Tensor):
                self.trace_tensor("backward/grad_output_tensor", grad, step=step, category="backward")
                return grad

            output_tensor.register_hook(_trace_grad)

    def trace_model_gradients(self, models: Iterable[torch.nn.Module], get_grad, *, step: int) -> None:
        if not self.param_grads_enabled or not self.should_trace(step):
            return
        for model_idx, model in enumerate(models):
            for param_name, param in model.named_parameters():
                extra_param = is_hsd_extra_param(param_name)
                category = "hsd_extra_grad" if extra_param else "param_grad"
                name = f"{category}/model{model_idx}/{param_name}"
                grad = get_grad(param)
                if self.target_param_pattern is not None and self.target_param_pattern.search(param_name):
                    self._trace_target_param_grad(
                        model_idx=model_idx,
                        param_name=param_name,
                        grad=getattr(param, "grad", None),
                        category="param_grad_raw",
                        grad_source="param.grad",
                        step=step,
                    )
                    self._trace_target_param_grad(
                        model_idx=model_idx,
                        param_name=param_name,
                        grad=getattr(param, "main_grad", None),
                        category="param_grad_main",
                        grad_source="main_grad",
                        step=step,
                    )
                if grad is None:
                    self.trace_tensor(name, None, step=step, category=category, status="none")
                    continue
                with torch.no_grad():
                    is_zero = bool((grad.detach() == 0).all().item()) if grad.numel() else True
                self.trace_tensor(
                    name,
                    grad,
                    step=step,
                    category=category,
                    status="zero" if is_zero else "nonzero",
                )

    def _trace_target_param_grad(
        self,
        *,
        model_idx: int,
        param_name: str,
        grad: torch.Tensor | None,
        category: str,
        grad_source: str,
        step: int,
    ) -> None:
        name = f"{category}/model{model_idx}/{param_name}"
        extra = {"grad_source": grad_source}
        if grad is None:
            self.trace_tensor(name, None, step=step, category=category, status="none", extra=extra)
            return
        with torch.no_grad():
            is_zero = bool((grad.detach() == 0).all().item()) if grad.numel() else True
        self.trace_tensor(
            name,
            grad,
            step=step,
            category=category,
            status="zero" if is_zero else "nonzero",
            extra=extra,
        )

    @staticmethod
    def _param_chain_metadata(param: torch.nn.Parameter) -> dict[str, Any]:
        main_grad = getattr(param, "main_grad", None)
        return {
            "param_shape": list(param.shape),
            "param_dtype": str(param.dtype),
            "requires_grad": bool(param.requires_grad),
            "sequence_parallel": bool(getattr(param, "sequence_parallel", False)),
            "tensor_model_parallel": bool(getattr(param, "tensor_model_parallel", False)),
            "allreduce": bool(getattr(param, "allreduce", True)),
            "average_gradients_across_tp_domain": bool(
                getattr(param, "average_gradients_across_tp_domain", False)
            ),
            "has_main_grad": main_grad is not None,
            "main_grad_shape": list(main_grad.shape) if isinstance(main_grad, torch.Tensor) else None,
            "main_grad_dtype": str(main_grad.dtype) if isinstance(main_grad, torch.Tensor) else None,
        }

    @staticmethod
    def _param_chain_grad_status(grad: torch.Tensor | None) -> str:
        if grad is None:
            return "none"
        with torch.no_grad():
            is_zero = bool((grad.detach() == 0).all().item()) if grad.numel() else True
        return "zero" if is_zero else "nonzero"

    def _trace_param_chain_tensor(
        self,
        *,
        model_idx: int,
        param_name: str,
        param: torch.nn.Parameter,
        stage: str,
        grad_source: str,
        grad: torch.Tensor | None,
        step: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row_extra: dict[str, Any] = {
            "chain_stage": stage,
            "model_idx": int(model_idx),
            "param_name": param_name,
            "grad_source": grad_source,
            "grad_added_to_main_grad": getattr(param, "grad_added_to_main_grad", None),
        }
        if extra:
            row_extra.update(extra)
        name = f"param_chain/{stage}/model{model_idx}/{param_name}/{grad_source}"
        self.trace_tensor(
            name,
            grad,
            step=step,
            category="param_chain",
            status=self._param_chain_grad_status(grad),
            extra=row_extra,
        )

    def _trace_param_chain_metadata(
        self,
        *,
        model_idx: int,
        param_name: str,
        param: torch.nn.Parameter,
        stage: str,
        step: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row_extra: dict[str, Any] = {
            "chain_stage": stage,
            "model_idx": int(model_idx),
            "param_name": param_name,
            "grad_source": "meta",
        }
        row_extra.update(self._param_chain_metadata(param))
        if extra:
            row_extra.update(extra)
        self.trace_tensor(
            f"param_chain/{stage}/model{model_idx}/{param_name}/meta",
            None,
            step=step,
            category="param_chain",
            status="meta",
            extra=row_extra,
        )

    def trace_param_chain_models(
        self,
        models: Iterable[torch.nn.Module],
        *,
        stage: str,
        step: int,
        include_meta: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.param_chain_enabled or not self.should_trace(step):
            return
        for model_idx, model in enumerate(models):
            for param_name, param in model.named_parameters():
                if not self.should_trace_param_chain(param_name):
                    continue
                if include_meta:
                    self._trace_param_chain_metadata(
                        model_idx=model_idx,
                        param_name=param_name,
                        param=param,
                        stage=stage,
                        step=step,
                        extra=extra,
                    )
                self._trace_param_chain_tensor(
                    model_idx=model_idx,
                    param_name=param_name,
                    param=param,
                    stage=stage,
                    grad_source="param.grad",
                    grad=getattr(param, "grad", None),
                    step=step,
                    extra=extra,
                )
                self._trace_param_chain_tensor(
                    model_idx=model_idx,
                    param_name=param_name,
                    param=param,
                    stage=stage,
                    grad_source="main_grad",
                    grad=getattr(param, "main_grad", None),
                    step=step,
                    extra=extra,
                )

    def register_param_chain_hooks(
        self,
        models: Iterable[torch.nn.Module],
        *,
        step: int,
    ) -> list[Any]:
        if not self.param_chain_enabled or not self.should_trace(step):
            return []

        handles = []
        for model_idx, model in enumerate(models):
            for param_name, param in model.named_parameters():
                if not self.should_trace_param_chain(param_name) or not param.requires_grad:
                    continue
                self._trace_param_chain_metadata(
                    model_idx=model_idx,
                    param_name=param_name,
                    param=param,
                    stage="hook_registration",
                    step=step,
                )

                def _trace_hook(
                    grad: torch.Tensor,
                    *,
                    model_idx: int = model_idx,
                    param_name: str = param_name,
                    param: torch.nn.Parameter = param,
                ):
                    self._trace_param_chain_tensor(
                        model_idx=model_idx,
                        param_name=param_name,
                        param=param,
                        stage="autograd_param_hook_grad",
                        grad_source="autograd_param_hook_grad",
                        grad=grad,
                        step=step,
                    )
                    return grad

                handles.append(param.register_hook(_trace_hook))
        return handles
