# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional tracing and telemetry, with one shared collector/counter owner."""

from __future__ import annotations

import atexit
import json
import os
import threading
from itertools import islice
from math import isfinite
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

from .runtime import (
    _env_int,
    _env_truthy,
    _parse_csv_selector,
    _parse_int_csv_selector,
    _parse_layer_selector,
)

if TYPE_CHECKING:
    from .layout import HistoricalHSDLayout

logger = init_logger("vllm.model_executor.models.safin1")


_HSD_TRACE_COLLECTOR: dict[str, torch.Tensor] | None = None


_HSD_TRACE_COUNTS: dict[str, int] = {}


_HSD_TRACE_LAYER_ACTIVE: dict[int, bool] = {}


_HSD_DECODE_TRACE_COUNTS: dict[str, int] = {}


_HSD_FINAL_OUTPUT_TRACE_COUNTS: dict[str, int] = {}


_HSD_LAYER_OUTPUT_TRACE_COUNTS: dict[str, int] = {}


_HSD_VARLEN_EVENT_AGGREGATES: dict[tuple[object, ...], dict[str, object]] = {}


_HSD_VARLEN_EVENT_AGGREGATE_LOCK = threading.Lock()


_HSD_VARLEN_EVENT_AGGREGATE_REGISTERED = False


_HSD_VARLEN_EVENT_AGGREGATE_UPDATES = 0


def _hsd_varlen_event_aggregate_enabled() -> bool:
    return _env_truthy("VLLM_HSD_VARLEN_COUNTER_AGGREGATE")


def _hsd_varlen_event_aggregate_flush_interval() -> int:
    raw = os.environ.get("VLLM_HSD_VARLEN_COUNTER_AGGREGATE_FLUSH_EVENTS", "")
    if not raw.strip():
        return 50000
    try:
        value = int(raw)
    except ValueError:
        return 50000
    return max(value, 0)


def _hsd_varlen_event_payload_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_shape": list(value.shape),
            "tensor_dtype": str(value.dtype),
            "tensor_device": str(value.device),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value, sort_keys=True)
        return value
    except TypeError:
        return str(value)


def _hsd_varlen_event_signature_value(value: object) -> object:
    value = _hsd_varlen_event_payload_value(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def _flush_hsd_varlen_event_aggregates() -> None:
    global _HSD_VARLEN_EVENT_AGGREGATE_UPDATES
    with _HSD_VARLEN_EVENT_AGGREGATE_LOCK:
        if not _HSD_VARLEN_EVENT_AGGREGATES:
            return
        records = list(_HSD_VARLEN_EVENT_AGGREGATES.values())
        _HSD_VARLEN_EVENT_AGGREGATES.clear()
        _HSD_VARLEN_EVENT_AGGREGATE_UPDATES = 0
    by_path: dict[str, list[dict[str, object]]] = {}
    for record in records:
        path = str(record.pop("_path"))
        by_path.setdefault(path, []).append(record)
    for path, path_records in by_path.items():
        try:
            with open(path, "a", encoding="utf-8") as f:
                for record in sorted(
                    path_records,
                    key=lambda item: (
                        str(item.get("event", "")),
                        int(item.get("layer", -1)),
                        int(item.get("pid", -1)),
                        str(item.get("reason", "")),
                    ),
                ):
                    f.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            logger.warning_once(
                "Failed to flush aggregated HSD varlen counter events to %s: %s",
                path,
                exc,
            )


def set_vllm_hsd_trace_collector(
    collector: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    previous = _HSD_TRACE_COLLECTOR
    globals()["_HSD_TRACE_COLLECTOR"] = collector
    return previous


def _hsd_graph_v3_debug_sync(stage: str) -> None:
    """Synchronize selected Graph V3 stages for ordering diagnostics only."""
    configured = os.environ.get("VLLM_HSD_GRAPH_V3_DEBUG_SYNC", "")
    if not configured:
        return
    stages = {item.strip().lower() for item in configured.split(",") if item.strip()}
    if "all" in stages or stage.lower() in stages:
        torch.cuda.synchronize()


def _hsd_trace_requested() -> bool:
    return _HSD_TRACE_COLLECTOR is not None or _env_truthy("VLLM_HSD_TRACE")


def _hsd_decode_trace_requested() -> bool:
    if not os.environ.get("VLLM_HSD_DECODE_TRACE_FILE"):
        return False
    trigger_file = os.environ.get("VLLM_HSD_DECODE_TRACE_TRIGGER_FILE")
    return not trigger_file or os.path.exists(trigger_file)


def _hsd_final_output_trace_requested() -> bool:
    return bool(os.environ.get("VLLM_HSD_FINAL_OUTPUT_TRACE_FILE"))


def _hsd_layer_output_trace_requested() -> bool:
    return bool(os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_FILE"))


def _hsd_decode_tp_rank() -> int | None:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        return None


_HSD_DECODE_TRACE_TEXT_STAGES = {
    "decoder_input",
    "residual_before_input_layernorm",
    "decoder_input_full",
    "decoder_input_full_fp32",
    "input_layernorm_out",
    "residual_after_input_layernorm",
    "hidden_states_text_before_proj",
    "qkv_text_before_conv",
    "qkvzba_text_after_proj",
    "conv_state_before_text_conv",
    "qkvzba_text_after_conv",
    "conv_state_after_text_conv",
    "recurrent_state_before_decode",
    "q_text",
    "k_text",
    "v_text",
    "beta_text",
    "g_text",
    "anchor_q",
    "anchor_k",
    "text_o_before_history",
    "anchor_states",
    "router_q",
    "read_q",
    "null_gate_logits",
    "history_out_text",
    "history_out_rmsnorm_text",
    "delta_o_text",
    "core_attn_out_before_norm",
    "core_attn_out_after_norm",
    "projected_out_text",
    "final_norm_input",
    "final_norm_out",
    "sample_hidden_states",
    "sample_logits",
}


def _hsd_decode_trace_sample(flat: torch.Tensor) -> dict[str, object]:
    sample_tensor = flat[: min(8, int(flat.numel()))].detach().cpu()
    return {
        "sample": [float(x) for x in sample_tensor.tolist()],
        "sample_sum": float(sample_tensor.sum().item()),
        "sample_absmax": float(sample_tensor.abs().max().item()),
    }


def _hsd_decode_trace_stats(
    tensor: torch.Tensor | None,
    *,
    stage: str | None = None,
) -> dict[str, object]:
    if not torch.is_tensor(tensor):
        return {"shape": None, "dtype": None, "numel": 0}
    value = tensor.detach()
    row: dict[str, object] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
    }
    if int(value.numel()) == 0:
        return row
    flat = value.reshape(-1).float()
    row.update(_hsd_decode_trace_sample(flat))
    if stage in _HSD_DECODE_TRACE_TEXT_STAGES and value.ndim >= 2:
        token_value = value[:, -1:].contiguous()
        token_flat = token_value.reshape(-1).float()
        token_sample = _hsd_decode_trace_sample(token_flat)
        row.update(
            {
                "token_shape": list(token_value.shape),
                "token_numel": int(token_value.numel()),
                "token_sample": token_sample["sample"],
                "token_sample_sum": token_sample["sample_sum"],
                "token_sample_absmax": token_sample["sample_absmax"],
            }
        )
    if _env_truthy("VLLM_HSD_DECODE_TRACE_LITE_STATS"):
        return row
    row.update(
        {
            "mean": float(flat.mean().detach().cpu().item()),
            "std": float(flat.std(unbiased=False).detach().cpu().item()),
            "min": float(flat.min().detach().cpu().item()),
            "max": float(flat.max().detach().cpu().item()),
            "absmax": float(flat.abs().max().detach().cpu().item()),
        }
    )
    return row


def _safe_decode_trace_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def trace_vllm_hsd_decode_stage(
    layer_idx: int,
    *,
    backend: str,
    stage: str,
    tensor: torch.Tensor | None,
    req_idx: int,
    state_idx: int,
    past_text_len: int,
    anchor_count: int,
    boundary_counts: torch.Tensor | None,
) -> None:
    trace_file = os.environ.get("VLLM_HSD_DECODE_TRACE_FILE")
    if not trace_file or not torch.is_tensor(tensor):
        return
    selected_backends = _parse_csv_selector(
        os.environ.get("VLLM_HSD_DECODE_TRACE_BACKENDS")
    )
    if selected_backends is not None and backend not in selected_backends:
        return
    layer_num = int(layer_idx) + 1
    selected_layers = _parse_layer_selector(
        os.environ.get("VLLM_HSD_DECODE_TRACE_LAYERS")
    )
    if selected_layers is not None and layer_num not in selected_layers:
        return
    selected_stages = _parse_csv_selector(
        os.environ.get("VLLM_HSD_DECODE_TRACE_STAGES")
    )
    if selected_stages is not None and stage not in selected_stages:
        return
    target_past_lens = _parse_int_csv_selector(
        os.environ.get("VLLM_HSD_DECODE_TRACE_TARGET_PAST_LENS")
    )
    if target_past_lens is not None and int(past_text_len) not in target_past_lens:
        return
    selected_req_idxs = _parse_int_csv_selector(
        os.environ.get("VLLM_HSD_DECODE_TRACE_REQ_IDXS")
    )
    if selected_req_idxs is not None and int(req_idx) not in selected_req_idxs:
        return
    selected_state_idxs = _parse_int_csv_selector(
        os.environ.get("VLLM_HSD_DECODE_TRACE_STATE_IDXS")
    )
    if selected_state_idxs is not None and int(state_idx) not in selected_state_idxs:
        return

    tp_rank = _hsd_decode_tp_rank()
    global_rank = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        global_rank = int(torch.distributed.get_rank())

    row: dict[str, object] = {
        "pid": int(os.getpid()),
        "trace_label": os.environ.get("VLLM_HSD_TRACE_LABEL"),
        "global_rank": global_rank,
        "tp_rank": tp_rank,
        "backend": backend,
        "layer": layer_num,
        "stage": stage,
        "req_idx": int(req_idx),
        "state_idx": int(state_idx),
        "past_text_len": int(past_text_len),
        "anchor_count": int(anchor_count),
        "boundary_counts": (
            None
            if boundary_counts is None
            else [int(x) for x in boundary_counts.detach().cpu().reshape(-1).tolist()]
        ),
    }
    row.update(_hsd_decode_trace_stats(tensor, stage=stage))

    if _env_truthy("VLLM_HSD_DECODE_TRACE_SAVE_TENSORS"):
        tensor_dir = os.environ.get("VLLM_HSD_DECODE_TRACE_TENSOR_DIR")
        if not tensor_dir:
            tensor_dir = f"{trace_file}.tensors"
        os.makedirs(tensor_dir, exist_ok=True)
        key = (
            f"pid{os.getpid()}__tp{tp_rank if tp_rank is not None else 'x'}"
            f"__layer{layer_num}__{_safe_decode_trace_name(backend)}"
            f"__state{int(state_idx)}__past{int(past_text_len)}"
            f"__req{int(req_idx)}__{_safe_decode_trace_name(stage)}"
        )
        count = _HSD_DECODE_TRACE_COUNTS.get(key, 0)
        _HSD_DECODE_TRACE_COUNTS[key] = count + 1
        tensor_path = os.path.join(tensor_dir, f"{key}__{count:04d}.pt")
        torch.save(tensor.detach().cpu(), tensor_path)
        row["tensor_file"] = tensor_path

    try:
        os.makedirs(os.path.dirname(trace_file), exist_ok=True)
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning_once("Failed to write HSD decode trace: %s", exc)


def trace_vllm_hsd_internal(
    layer_idx: int,
    name: str,
    tensor: torch.Tensor | None,
) -> None:
    if not torch.is_tensor(tensor):
        return
    if not _hsd_trace_requested():
        return
    selected_layers = _parse_layer_selector(os.environ.get("VLLM_HSD_TRACE_LAYERS"))
    if selected_layers is not None and (int(layer_idx) + 1) not in selected_layers:
        return
    target_past_len = _env_int("VLLM_HSD_TRACE_TARGET_PAST_LEN")
    if target_past_len is not None:
        layer_key = int(layer_idx)
        if name in {
            "decoder_input",
            "decoder_input_full",
            "input_layernorm_out",
        }:
            _HSD_TRACE_LAYER_ACTIVE[layer_key] = False
            return
        if name == "hsd_past_text_len":
            active = int(tensor.reshape(-1)[0].detach().cpu().item()) == int(
                target_past_len
            )
            _HSD_TRACE_LAYER_ACTIVE[layer_key] = active
            if not active:
                return
        elif not _HSD_TRACE_LAYER_ACTIVE.get(layer_key, False):
            return
    selected_stages = os.environ.get("VLLM_HSD_TRACE_STAGES")
    if selected_stages:
        stage_names = {
            stage.strip() for stage in selected_stages.split(",") if stage.strip()
        }
        if name not in stage_names:
            return
    max_numel = _env_int("VLLM_HSD_TRACE_MAX_NUMEL")
    if max_numel is not None and int(tensor.numel()) > int(max_numel):
        return
    tp_rank = _hsd_decode_tp_rank()
    selected_ranks = _parse_int_csv_selector(os.environ.get("VLLM_HSD_TRACE_TP_RANKS"))
    if selected_ranks is not None:
        rank_for_filter = -1 if tp_rank is None else int(tp_rank)
        if rank_for_filter not in selected_ranks:
            return
    key = f"layer_{int(layer_idx) + 1}/{name}"
    count = _HSD_TRACE_COUNTS.get(key, 0)
    _HSD_TRACE_COUNTS[key] = count + 1

    stats_file = os.environ.get("VLLM_HSD_TRACE_STATS_FILE")
    if stats_file:
        row = {
            "key": key,
            "layer": int(layer_idx) + 1,
            "stage": name,
            "count": count,
            "tp_rank": tp_rank,
        }
        row.update(_hsd_decode_trace_stats(tensor, stage=name))
        try:
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            with open(stats_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning_once("Failed to write HSD trace stats: %s", exc)

    slice_dir = os.environ.get("VLLM_HSD_TRACE_SLICE_DIR")
    if slice_dir:
        slice_stages = _parse_csv_selector(
            os.environ.get("VLLM_HSD_TRACE_SLICE_STAGES")
        )
        target_tokens = _env_int("VLLM_HSD_TRACE_SLICE_TARGET_TOKENS")
        token_target_exempt_stages = {
            "anchor_hidden_pre",
            "anchor_q",
            "anchor_k",
            "anchor_o",
            "anchor_states",
            "anchor_boundaries",
            "persistent_memory_anchor_o",
        }
        batch_token_stages = {
            "router_q",
            "read_q",
            "null_gate_logits",
            "history_out_text",
            "history_out_rmsnorm_text",
            "persistent_memory_logit",
            "persistent_memory_probability",
            "persistent_memory_raw_read",
            "persistent_memory_read",
            "persistent_memory_joint_probability_sum",
        }
        anchor_row_stages = {
            "anchor_hidden_pre",
            "anchor_q",
            "anchor_k",
            "anchor_o",
            "anchor_states",
            "persistent_memory_anchor_o",
        }
        token_target_applies = (
            not name.endswith("_weight")
            and not name.endswith("_bias")
            and name not in token_target_exempt_stages
        )
        text_length_dim = None
        if token_target_applies and target_tokens is not None:
            # Static vLLM prefill tensors may be padded to max_model_len, so
            # their token dimension need not equal the request's true length.
            # These named stages have an explicit [B, T, ...] contract.
            if name in batch_token_stages and tensor.ndim >= 3:
                text_length_dim = 1
            elif tensor.ndim >= 3 and int(tensor.shape[1]) == int(target_tokens):
                text_length_dim = 1
            elif tensor.ndim >= 2 and int(tensor.shape[0]) == int(target_tokens):
                text_length_dim = 0
        token_dim_matches = (
            not token_target_applies
            or target_tokens is None
            or text_length_dim is not None
        )
        if (slice_stages is None or name in slice_stages) and token_dim_matches:
            value = tensor.detach().cpu().contiguous()
            anchor_index = _env_int("VLLM_HSD_TRACE_SLICE_ANCHOR_INDEX")
            if (
                anchor_index is not None
                and name in anchor_row_stages
                and value.ndim >= 3
                and int(value.shape[1]) > 0
            ):
                idx = int(anchor_index)
                if idx < 0:
                    idx += int(value.shape[1])
                if not 0 <= idx < int(value.shape[1]):
                    raise IndexError(
                        f"HSD trace anchor index {anchor_index} is outside "
                        f"stage {name} shape {tuple(value.shape)}."
                    )
                value = value[:, idx : idx + 1].clone()
            if (
                token_target_applies
                and target_tokens is not None
                and text_length_dim == 0
                and value.ndim == 2
            ):
                value = value.unsqueeze(0).contiguous()
                text_length_dim = 1
            token_index = _env_int("VLLM_HSD_TRACE_SLICE_TOKEN_INDEX")
            if (
                token_index is not None
                and token_target_applies
                and value.ndim >= 3
                and text_length_dim == 1
                and int(value.shape[1]) > 0
            ):
                idx = int(token_index)
                if idx < 0:
                    idx += int(value.shape[1])
                if 0 <= idx < int(value.shape[1]):
                    # A narrow contiguous view may still retain the full token
                    # storage when serialized. Clone so a one-token diagnostic
                    # slice stays one token on disk.
                    value = value[:, idx : idx + 1].clone()
            os.makedirs(slice_dir, exist_ok=True)
            safe_key = key.replace("/", "__")
            rank_name = "unknown" if tp_rank is None else str(int(tp_rank))
            filename = f"tp{rank_name}__{safe_key}__{count:04d}.pt"
            tensor_path = os.path.join(slice_dir, filename)
            torch.save(value, tensor_path)
            manifest_row = {
                "key": key,
                "layer": int(layer_idx) + 1,
                "stage": name,
                "count": count,
                "tp_rank": tp_rank,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "file": tensor_path,
            }
            try:
                with open(
                    os.path.join(slice_dir, "manifest.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning_once("Failed to write HSD trace slice manifest: %s", exc)

    trace_dir = os.environ.get("VLLM_HSD_TRACE_DIR")
    if trace_dir and _env_truthy("VLLM_HSD_TRACE_SAVE_TENSORS", default=True):
        value = tensor.detach().cpu()
        os.makedirs(trace_dir, exist_ok=True)
        safe_key = key.replace("/", "__")
        if _env_truthy("VLLM_HSD_TRACE_WITH_TP_RANK"):
            try:
                from vllm.distributed import get_tensor_model_parallel_rank

                safe_key = f"tp{get_tensor_model_parallel_rank()}__{safe_key}"
            except Exception:
                safe_key = f"tp_unknown__{safe_key}"
        torch.save(value, os.path.join(trace_dir, f"{safe_key}__{count:04d}.pt"))
    if _HSD_TRACE_COLLECTOR is not None:
        value = tensor.detach().cpu()
        _HSD_TRACE_COLLECTOR[key] = value
    elif _env_truthy("VLLM_HSD_TRACE_LOG", default=not bool(trace_dir or stats_file)):
        value = tensor.detach().cpu()
        logger.info(
            "VLLM_HSD_TRACE %s shape=%s dtype=%s",
            key,
            tuple(value.shape),
            value.dtype,
        )


def _hsd_state_cache_log_enabled(layer_idx: int | None = None) -> bool:
    if not _env_truthy("VLLM_HSD_STATE_CACHE_LOG"):
        return False
    if layer_idx is None:
        return True
    selected_layers = _parse_layer_selector(
        os.environ.get("VLLM_HSD_STATE_CACHE_LOG_LAYERS", "1")
    )
    return selected_layers is None or (int(layer_idx) + 1) in selected_layers


class AttentionDiagnosticsMixin:
    """Existing optional diagnostics; no parameter or execution policy ownership."""

    @staticmethod
    def _apply_decode_trace_additional_config(
        additional_config: dict[str, object],
    ) -> None:
        trace_env = {
            "hsd_decode_trace_file": "VLLM_HSD_DECODE_TRACE_FILE",
            "hsd_decode_trace_layers": "VLLM_HSD_DECODE_TRACE_LAYERS",
            "hsd_decode_trace_stages": "VLLM_HSD_DECODE_TRACE_STAGES",
            "hsd_decode_trace_target_past_lens": (
                "VLLM_HSD_DECODE_TRACE_TARGET_PAST_LENS"
            ),
            "hsd_decode_trace_req_idxs": "VLLM_HSD_DECODE_TRACE_REQ_IDXS",
            "hsd_decode_trace_state_idxs": "VLLM_HSD_DECODE_TRACE_STATE_IDXS",
            "hsd_decode_trace_save_tensors": ("VLLM_HSD_DECODE_TRACE_SAVE_TENSORS"),
            "hsd_decode_trace_tensor_dir": "VLLM_HSD_DECODE_TRACE_TENSOR_DIR",
            "hsd_decode_trace_backends": "VLLM_HSD_DECODE_TRACE_BACKENDS",
            "hsd_decode_trace_lite_stats": ("VLLM_HSD_DECODE_TRACE_LITE_STATS"),
        }
        for config_key, env_key in trace_env.items():
            value = additional_config.get(config_key)
            if value is None:
                continue
            if os.environ.get(env_key):
                continue
            os.environ[env_key] = str(value)

    def _trace_internal(self, name: str, tensor: torch.Tensor | None) -> None:
        trace_vllm_hsd_internal(int(self.layer_idx), name, tensor)

    def _capture_hsd_prefill_passive_sidecar(
        self,
        stage: str,
        tensor: torch.Tensor | None,
        *,
        past_text_len: int,
        text_len: int,
    ) -> None:
        """Queue a GPU clone for prefill diagnostics without synchronizing."""
        if not torch.is_tensor(tensor) or not os.environ.get(
            "VLLM_HSD_PREFILL_PASSIVE_SIDECAR_DIR"
        ):
            return
        selected_layers = _parse_layer_selector(
            os.environ.get("VLLM_HSD_PREFILL_PASSIVE_SIDECAR_LAYERS")
        )
        layer_num = int(self.layer_idx) + 1
        if selected_layers is not None and layer_num not in selected_layers:
            return
        selected_ranks = _parse_int_csv_selector(
            os.environ.get("VLLM_HSD_PREFILL_PASSIVE_SIDECAR_TP_RANKS")
        )
        tp_rank = _hsd_decode_tp_rank()
        if (
            selected_ranks is not None
            and (-1 if tp_rank is None else int(tp_rank)) not in selected_ranks
        ):
            return
        selected_stages = _parse_csv_selector(
            os.environ.get("VLLM_HSD_PREFILL_PASSIVE_SIDECAR_STAGES")
        )
        if selected_stages is not None and stage not in selected_stages:
            return
        target_tokens = _env_int("VLLM_HSD_PREFILL_PASSIVE_SIDECAR_TARGET_TOKENS")
        if target_tokens is not None and int(text_len) != int(target_tokens):
            return
        target_past_len = _env_int("VLLM_HSD_PREFILL_PASSIVE_SIDECAR_TARGET_PAST_LEN")
        if target_past_len is not None and int(past_text_len) != int(target_past_len):
            return
        self._hsd_prefill_passive_sidecars[stage] = tensor.detach().clone()
        self._hsd_prefill_passive_meta = {
            "past_text_len": int(past_text_len),
            "text_len": int(text_len),
        }

    def _trace_decode_stage(
        self,
        *,
        backend: str,
        stage: str,
        tensor: torch.Tensor | None,
        req_idx: int,
        state_idx: int,
        past_text_len: int,
        anchor_count: int,
        boundary_counts: torch.Tensor | None,
    ) -> None:
        trace_vllm_hsd_decode_stage(
            int(self.layer_idx),
            backend=backend,
            stage=stage,
            tensor=tensor,
            req_idx=req_idx,
            state_idx=state_idx,
            past_text_len=past_text_len,
            anchor_count=anchor_count,
            boundary_counts=boundary_counts,
        )

    def _record_hsd_varlen_counter(self, name: str, value: int = 1) -> None:
        self.hsd_prefill_varlen_counters[name] = self.hsd_prefill_varlen_counters.get(
            name, 0
        ) + int(value)

    def _record_hsd_varlen_event(self, event: str, **payload) -> None:
        global _HSD_VARLEN_EVENT_AGGREGATE_REGISTERED
        global _HSD_VARLEN_EVENT_AGGREGATE_UPDATES
        counter_path = os.environ.get("VLLM_HSD_VARLEN_COUNTER_FILE")
        if not counter_path:
            return
        if _hsd_varlen_event_aggregate_enabled():
            if not _HSD_VARLEN_EVENT_AGGREGATE_REGISTERED:
                with _HSD_VARLEN_EVENT_AGGREGATE_LOCK:
                    if not _HSD_VARLEN_EVENT_AGGREGATE_REGISTERED:
                        atexit.register(_flush_hsd_varlen_event_aggregates)
                        _HSD_VARLEN_EVENT_AGGREGATE_REGISTERED = True
            pid = os.getpid()
            layer = int(self.layer_idx) + 1
            static_payload: dict[str, object] = {}
            numeric_payload: dict[str, float] = {}
            for key, value in payload.items():
                if isinstance(value, bool):
                    numeric_payload[key] = float(int(value))
                elif isinstance(value, int):
                    numeric_payload[key] = float(value)
                elif isinstance(value, float) and isfinite(value):
                    numeric_payload[key] = float(value)
                else:
                    static_payload[key] = _hsd_varlen_event_payload_value(value)
            signature = tuple(
                sorted(
                    (
                        key,
                        _hsd_varlen_event_signature_value(value),
                    )
                    for key, value in static_payload.items()
                )
            )
            aggregate_key: tuple[object, ...] = (
                counter_path,
                pid,
                layer,
                event,
                signature,
            )
            should_flush = False
            with _HSD_VARLEN_EVENT_AGGREGATE_LOCK:
                record = _HSD_VARLEN_EVENT_AGGREGATES.get(aggregate_key)
                if record is None:
                    record = {
                        "_path": counter_path,
                        "_aggregated": True,
                        "_count": 0,
                        "event": event,
                        "pid": pid,
                        "layer": layer,
                        **static_payload,
                    }
                    _HSD_VARLEN_EVENT_AGGREGATES[aggregate_key] = record
                record["_count"] = int(record.get("_count", 0)) + 1
                for key, value in numeric_payload.items():
                    sum_key = f"{key}_sum"
                    max_key = f"{key}_max"
                    min_key = f"{key}_min"
                    record[sum_key] = float(record.get(sum_key, 0.0)) + value
                    record[max_key] = max(float(record.get(max_key, value)), value)
                    record[min_key] = min(float(record.get(min_key, value)), value)
                    # Keep the base field as the sum so existing lightweight
                    # readers can aggregate payload totals without special cases.
                    record[key] = record[sum_key]
                _HSD_VARLEN_EVENT_AGGREGATE_UPDATES += 1
                flush_interval = _hsd_varlen_event_aggregate_flush_interval()
                should_flush = (
                    flush_interval > 0
                    and _HSD_VARLEN_EVENT_AGGREGATE_UPDATES >= flush_interval
                )
            if should_flush:
                _flush_hsd_varlen_event_aggregates()
            return
        record = {
            "event": event,
            "pid": os.getpid(),
            "layer": int(self.layer_idx) + 1,
            **payload,
        }
        try:
            with open(counter_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            logger.warning_once(
                "Failed to write HSD varlen counter event to %s: %s",
                counter_path,
                exc,
            )

    def _hsd_text_state_cache_debug_enabled(self) -> bool:
        return _env_truthy("VLLM_HSD_TEXT_STATE_CACHE_DEBUG") or _env_truthy(
            "VLLM_HSD_PREFIX_CACHE_DEBUG"
        )

    @staticmethod
    def _hsd_state_debug_stats(
        prefix: str,
        tensor: torch.Tensor | None,
    ) -> dict[str, object]:
        if not torch.is_tensor(tensor):
            return {
                f"{prefix}_present": 0,
            }
        value = tensor.detach()
        stats: dict[str, object] = {
            f"{prefix}_present": 1,
            f"{prefix}_shape": [int(x) for x in value.shape],
            f"{prefix}_dtype": str(value.dtype),
        }
        if int(value.numel()) == 0:
            stats[f"{prefix}_numel"] = 0
            return stats
        flat = value.float().reshape(-1)
        stats.update(
            {
                f"{prefix}_numel": int(flat.numel()),
                f"{prefix}_norm": float(flat.norm().item()),
                f"{prefix}_sum": float(flat.sum().item()),
                f"{prefix}_absmax": float(flat.abs().max().item()),
            }
        )
        return stats

    @staticmethod
    def _hsd_tensor_diff_stats(
        prefix: str,
        actual: torch.Tensor | None,
        expected: torch.Tensor | None,
    ) -> dict[str, object]:
        if not torch.is_tensor(actual) or not torch.is_tensor(expected):
            return {
                f"{prefix}_present": 0,
            }
        actual_value = actual.detach()
        expected_value = expected.detach()
        stats: dict[str, object] = {
            f"{prefix}_present": 1,
            f"{prefix}_actual_shape": [int(x) for x in actual_value.shape],
            f"{prefix}_expected_shape": [int(x) for x in expected_value.shape],
            f"{prefix}_actual_dtype": str(actual_value.dtype),
            f"{prefix}_expected_dtype": str(expected_value.dtype),
        }
        if tuple(actual_value.shape) != tuple(expected_value.shape):
            stats[f"{prefix}_shape_match"] = 0
            return stats
        stats[f"{prefix}_shape_match"] = 1
        if int(actual_value.numel()) == 0:
            stats[f"{prefix}_numel"] = 0
            return stats
        actual_flat = actual_value.float().reshape(-1)
        expected_flat = expected_value.float().reshape(-1)
        diff = (actual_flat - expected_flat).abs()
        denom = actual_flat.norm() * expected_flat.norm()
        cosine = (
            float(torch.dot(actual_flat, expected_flat).div(denom).item())
            if float(denom.item()) > 0.0
            else None
        )
        stats.update(
            {
                f"{prefix}_numel": int(diff.numel()),
                f"{prefix}_max_abs": float(diff.max().item()),
                f"{prefix}_mean_abs": float(diff.mean().item()),
                f"{prefix}_actual_norm": float(actual_flat.norm().item()),
                f"{prefix}_expected_norm": float(expected_flat.norm().item()),
                f"{prefix}_cosine": cosine,
            }
        )
        return stats

    def _record_hsd_text_state_cache_debug(
        self,
        phase: str,
        state_idx: int,
        *,
        cache_hit: bool | None = None,
        cached_conv_state: torch.Tensor | None = None,
        cached_recurrent_state: torch.Tensor | None = None,
        native_conv_state: torch.Tensor | None = None,
        native_recurrent_state: torch.Tensor | None = None,
        **payload,
    ) -> None:
        if not self._hsd_text_state_cache_debug_enabled():
            return
        record: dict[str, object] = {
            "phase": phase,
            "state_idx": int(state_idx),
        }
        if cache_hit is not None:
            record["cache_hit"] = int(bool(cache_hit))
        record.update(payload)
        record.update(self._hsd_state_debug_stats("cached_conv", cached_conv_state))
        record.update(
            self._hsd_state_debug_stats(
                "cached_recurrent",
                cached_recurrent_state,
            )
        )
        record.update(self._hsd_state_debug_stats("native_conv", native_conv_state))
        record.update(
            self._hsd_state_debug_stats(
                "native_recurrent",
                native_recurrent_state,
            )
        )
        self._record_hsd_varlen_event("hsd_text_state_cache_debug", **record)

    def _record_hsd_varlen_fallback(self, reason: str) -> None:
        self.hsd_prefill_varlen_fallback_counts[reason] = (
            self.hsd_prefill_varlen_fallback_counts.get(reason, 0) + 1
        )
        self._record_hsd_varlen_event("fallback", reason=reason)
        if _env_truthy("VLLM_HSD_VARLEN_LOG_FALLBACK"):
            logger.info(
                "HSD varlen prefill fallback layer=%s reason=%s",
                int(self.layer_idx) + 1,
                reason,
            )


class DecoderDiagnosticsMixin:
    """Existing optional diagnostics; no parameter or execution policy ownership."""

    def _trace_hsd_layer_stage(
        self,
        stage: str,
        text_tensor: torch.Tensor | None,
        layout: HistoricalHSDLayout,
    ) -> None:
        trace_file = os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_FILE")
        if not trace_file or not torch.is_tensor(text_tensor):
            return
        layer_num = int(self.layer_idx) + 1
        selected_layers = _parse_layer_selector(
            os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_LAYERS")
        )
        if selected_layers is not None and layer_num not in selected_layers:
            return
        selected_stages = _parse_csv_selector(
            os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_STAGES")
        )
        if selected_stages is not None and stage not in selected_stages:
            return
        selected_req_idxs = _parse_int_csv_selector(
            os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_REQ_IDXS")
        )
        target_past_lens = _parse_int_csv_selector(
            os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_TARGET_PAST_LENS")
        )
        mode = os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_MODE", "last").lower()
        if mode not in {"last", "full"}:
            mode = "last"

        tp_rank = _hsd_decode_tp_rank()
        global_rank = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_rank = int(torch.distributed.get_rank())

        for req_idx, (text_start, text_len) in enumerate(
            zip(layout.req_text_starts, layout.req_text_lens)
        ):
            if selected_req_idxs is not None and req_idx not in selected_req_idxs:
                continue
            past_text_len = int(layout.past_text_lens[req_idx])
            if target_past_lens is not None and past_text_len not in target_past_lens:
                continue
            text_len = int(text_len)
            if text_len <= 0:
                continue
            req_anchor_start = layout.req_anchor_starts[req_idx]
            req_anchor_len = layout.req_anchor_lens[req_idx]
            boundary_counts = layout.anchor_boundaries[
                req_anchor_start : req_anchor_start + req_anchor_len
            ]
            req_tensor = text_tensor[text_start : text_start + text_len]
            if mode == "last":
                req_tensor = req_tensor[-1:].view(1, 1, -1)
            else:
                req_tensor = req_tensor.view(1, text_len, -1)

            row: dict[str, object] = {
                "pid": int(os.getpid()),
                "trace_label": os.environ.get("VLLM_HSD_TRACE_LABEL"),
                "global_rank": global_rank,
                "tp_rank": tp_rank,
                "backend": "decoder_layer_stage",
                "layer": layer_num,
                "stage": stage,
                "layer_type": self.layer_type,
                "mode": mode,
                "req_idx": int(req_idx),
                "state_idx": int(layout.state_indices_list[req_idx]),
                "past_text_len": past_text_len,
                "text_len": text_len,
                "anchor_count": int(req_anchor_len),
                "boundary_counts": [
                    int(x) for x in boundary_counts.detach().cpu().reshape(-1).tolist()
                ],
            }
            row.update(_hsd_decode_trace_stats(req_tensor))

            if _env_truthy("VLLM_HSD_LAYER_OUTPUT_TRACE_SAVE_TENSORS"):
                tensor_dir = os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_TENSOR_DIR")
                if not tensor_dir:
                    tensor_dir = f"{trace_file}.tensors"
                os.makedirs(tensor_dir, exist_ok=True)
                key = (
                    f"pid{os.getpid()}__tp{tp_rank if tp_rank is not None else 'x'}"
                    f"__layer{layer_num}__req{int(req_idx)}"
                    f"__state{int(layout.state_indices_list[req_idx])}"
                    f"__past{int(layout.past_text_lens[req_idx])}"
                    f"__{_safe_decode_trace_name(stage)}__{mode}"
                )
                count = _HSD_LAYER_OUTPUT_TRACE_COUNTS.get(key, 0)
                _HSD_LAYER_OUTPUT_TRACE_COUNTS[key] = count + 1
                tensor_path = os.path.join(tensor_dir, f"{key}__{count:04d}.pt")
                torch.save(req_tensor.detach().cpu(), tensor_path)
                row["tensor_file"] = tensor_path

            try:
                os.makedirs(os.path.dirname(trace_file), exist_ok=True)
                with open(trace_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning_once("Failed to write HSD layer stage trace: %s", exc)

    def _trace_hsd_graph_v3_layer_stage(
        self,
        stage: str,
        tensor: torch.Tensor | None,
        *,
        state_indices: torch.Tensor,
        past_lens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """Trace Graph V3 eager tensors without rebuilding a dynamic layout."""
        # This helper performs CPU synchronization and file I/O. Layer-output
        # tracing is still useful for the eager prefill/mixed path, but it must
        # not become part of the compiled static-decode graph.
        if torch.compiler.is_compiling():
            return
        trace_file = os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_FILE")
        if not trace_file or not torch.is_tensor(tensor):
            return
        layer_num = int(self.layer_idx) + 1
        selected_layers = _parse_layer_selector(
            os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_LAYERS")
        )
        if selected_layers is not None and layer_num not in selected_layers:
            return
        selected_stages = _parse_csv_selector(
            os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_STAGES")
        )
        if selected_stages is not None and stage not in selected_stages:
            return
        selected_req_idxs = _parse_int_csv_selector(
            os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_REQ_IDXS")
        )
        target_past_lens = _parse_int_csv_selector(
            os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_TARGET_PAST_LENS")
        )
        tp_rank = _hsd_decode_tp_rank()
        global_rank = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_rank = int(torch.distributed.get_rank())

        valid_rows = valid_mask.detach().to(device="cpu", dtype=torch.bool).tolist()
        state_rows = state_indices.detach().to(device="cpu").tolist()
        past_rows = past_lens.detach().to(device="cpu").tolist()
        for req_idx, is_valid in enumerate(valid_rows):
            if not is_valid:
                continue
            if selected_req_idxs is not None and req_idx not in selected_req_idxs:
                continue
            past_text_len = int(past_rows[req_idx])
            if target_past_lens is not None and past_text_len not in target_past_lens:
                continue
            req_tensor = tensor[req_idx : req_idx + 1].view(1, 1, -1)
            row: dict[str, object] = {
                "pid": int(os.getpid()),
                "trace_label": os.environ.get("VLLM_HSD_TRACE_LABEL"),
                "global_rank": global_rank,
                "tp_rank": tp_rank,
                "backend": "graph_v3_layer_stage",
                "layer": layer_num,
                "stage": stage,
                "layer_type": self.layer_type,
                "mode": "last",
                "req_idx": int(req_idx),
                "state_idx": int(state_rows[req_idx]),
                "past_text_len": past_text_len,
                "text_len": 1,
                "anchor_count": 0,
                "boundary_counts": [],
            }
            row.update(_hsd_decode_trace_stats(req_tensor))
            if _env_truthy("VLLM_HSD_LAYER_OUTPUT_TRACE_SAVE_TENSORS"):
                tensor_dir = os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_TENSOR_DIR")
                if not tensor_dir:
                    tensor_dir = f"{trace_file}.tensors"
                os.makedirs(tensor_dir, exist_ok=True)
                key = (
                    f"pid{os.getpid()}__tp{tp_rank if tp_rank is not None else 'x'}"
                    f"__layer{layer_num}__req{int(req_idx)}"
                    f"__state{int(state_rows[req_idx])}"
                    f"__past{int(past_rows[req_idx])}"
                    f"__{_safe_decode_trace_name(stage)}__last"
                )
                count = _HSD_LAYER_OUTPUT_TRACE_COUNTS.get(key, 0)
                _HSD_LAYER_OUTPUT_TRACE_COUNTS[key] = count + 1
                tensor_path = os.path.join(tensor_dir, f"{key}__{count:04d}.pt")
                torch.save(req_tensor.detach().cpu(), tensor_path)
                row["tensor_file"] = tensor_path
            try:
                os.makedirs(os.path.dirname(trace_file), exist_ok=True)
                with open(trace_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning_once(
                    "Failed to write HSD Graph V3 layer stage trace: %s", exc
                )


class ModelDiagnosticsMixin:
    """Existing optional diagnostics; no parameter or execution policy ownership."""

    def dump_hsd_prefill_passive_sidecars(self) -> None:
        dump_dir = os.environ.get("VLLM_HSD_PREFILL_PASSIVE_SIDECAR_DIR")
        if not dump_dir:
            return
        layers: dict[int, dict[str, torch.Tensor]] = {}
        metadata: dict[int, dict[str, int]] = {}
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            module = layer._active_hsd_attn()
            sidecars = getattr(module, "_hsd_prefill_passive_sidecars", None)
            if not isinstance(sidecars, dict) or not sidecars:
                continue
            layer_num = int(layer.layer_idx) + 1
            layers[layer_num] = {
                str(stage): tensor.detach().cpu()
                for stage, tensor in sidecars.items()
                if torch.is_tensor(tensor)
            }
            metadata[layer_num] = dict(getattr(module, "_hsd_prefill_passive_meta", {}))
            sidecars.clear()
            module._hsd_prefill_passive_meta = {}
        if not layers:
            return
        step = int(getattr(self, "_hsd_prefill_passive_sidecar_step", 0))
        self._hsd_prefill_passive_sidecar_step = step + 1
        global_rank = (
            int(torch.distributed.get_rank())
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        os.makedirs(dump_dir, exist_ok=True)
        torch.save(
            {
                "step": step,
                "global_rank": global_rank,
                "layers": layers,
                "metadata": metadata,
            },
            os.path.join(
                dump_dir,
                f"prefill__rank{global_rank}__pid{os.getpid()}__step{step:04d}.pt",
            ),
        )

    def _trace_hsd_model_stage(
        self,
        stage: str,
        tensor: torch.Tensor | None,
        layout: HistoricalHSDLayout,
    ) -> None:
        if not _hsd_decode_trace_requested() or not torch.is_tensor(tensor):
            return
        final_layer_idx = int(getattr(self.config, "num_hidden_layers", 0) or 0)
        for req_idx, (text_start, text_len) in enumerate(
            zip(layout.req_text_starts, layout.req_text_lens)
        ):
            text_len = int(text_len)
            if text_len <= 0:
                continue
            req_anchor_start = layout.req_anchor_starts[req_idx]
            req_anchor_len = layout.req_anchor_lens[req_idx]
            boundary_counts = layout.anchor_boundaries[
                req_anchor_start : req_anchor_start + req_anchor_len
            ]
            trace_vllm_hsd_decode_stage(
                final_layer_idx,
                backend="model",
                stage=stage,
                tensor=tensor[text_start : text_start + text_len].view(1, text_len, -1),
                req_idx=req_idx,
                state_idx=int(layout.state_indices_list[req_idx]),
                past_text_len=int(layout.past_text_lens[req_idx]),
                anchor_count=int(req_anchor_len),
                boundary_counts=boundary_counts,
            )

    def _trace_hsd_sample_stage(
        self,
        stage: str,
        tensor: torch.Tensor | None,
        layout: HistoricalHSDLayout,
    ) -> None:
        if not _hsd_decode_trace_requested() or not torch.is_tensor(tensor):
            return
        selected_req_idxs = _parse_int_csv_selector(
            os.environ.get("VLLM_HSD_DECODE_TRACE_REQ_IDXS")
        )
        if selected_req_idxs is None:
            req_indices = list(range(int(tensor.shape[0])))
        else:
            req_indices = sorted(selected_req_idxs)
        final_layer_idx = int(getattr(self.config, "num_hidden_layers", 0) or 0)
        for row_idx, req_idx in enumerate(req_indices[: int(tensor.shape[0])]):
            if req_idx < 0 or req_idx >= len(layout.req_text_lens):
                continue
            req_anchor_start = layout.req_anchor_starts[req_idx]
            req_anchor_len = layout.req_anchor_lens[req_idx]
            boundary_counts = layout.anchor_boundaries[
                req_anchor_start : req_anchor_start + req_anchor_len
            ]
            trace_vllm_hsd_decode_stage(
                final_layer_idx,
                backend="model",
                stage=stage,
                tensor=tensor[row_idx : row_idx + 1].view(1, 1, -1),
                req_idx=req_idx,
                state_idx=int(layout.state_indices_list[req_idx]),
                past_text_len=int(layout.past_text_lens[req_idx]),
                anchor_count=int(req_anchor_len),
                boundary_counts=boundary_counts,
            )

    def _trace_hsd_final_output_stage(
        self,
        stage: str,
        tensor: torch.Tensor | None,
        layout: HistoricalHSDLayout,
    ) -> None:
        trace_file = os.environ.get("VLLM_HSD_FINAL_OUTPUT_TRACE_FILE")
        if not trace_file or not torch.is_tensor(tensor):
            return
        selected_req_idxs = _parse_int_csv_selector(
            os.environ.get("VLLM_HSD_FINAL_OUTPUT_TRACE_REQ_IDXS")
        )
        target_past_lens = _parse_int_csv_selector(
            os.environ.get("VLLM_HSD_FINAL_OUTPUT_TRACE_TARGET_PAST_LENS")
        )
        final_layer_idx = int(getattr(self.config, "num_hidden_layers", 0) or 0)
        tp_rank = _hsd_decode_tp_rank()
        global_rank = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_rank = int(torch.distributed.get_rank())

        for req_idx, (text_start, text_len) in enumerate(
            zip(layout.req_text_starts, layout.req_text_lens)
        ):
            if selected_req_idxs is not None and req_idx not in selected_req_idxs:
                continue
            past_text_len = int(layout.past_text_lens[req_idx])
            if target_past_lens is not None and past_text_len not in target_past_lens:
                continue
            text_len = int(text_len)
            if text_len <= 0:
                continue
            req_anchor_start = layout.req_anchor_starts[req_idx]
            req_anchor_len = layout.req_anchor_lens[req_idx]
            boundary_counts = layout.anchor_boundaries[
                req_anchor_start : req_anchor_start + req_anchor_len
            ]
            req_tensor = tensor[text_start : text_start + text_len].view(
                1, text_len, -1
            )
            row: dict[str, object] = {
                "pid": int(os.getpid()),
                "global_rank": global_rank,
                "tp_rank": tp_rank,
                "backend": "model_final_output",
                "layer": final_layer_idx,
                "stage": stage,
                "req_idx": int(req_idx),
                "state_idx": int(layout.state_indices_list[req_idx]),
                "past_text_len": past_text_len,
                "anchor_count": int(req_anchor_len),
                "boundary_counts": [
                    int(x) for x in boundary_counts.detach().cpu().reshape(-1).tolist()
                ],
            }
            row.update(_hsd_decode_trace_stats(req_tensor))
            if _env_truthy("VLLM_HSD_FINAL_OUTPUT_TRACE_SAVE_TENSORS"):
                tensor_dir = os.environ.get("VLLM_HSD_FINAL_OUTPUT_TRACE_TENSOR_DIR")
                if not tensor_dir:
                    tensor_dir = f"{trace_file}.tensors"
                os.makedirs(tensor_dir, exist_ok=True)
                key = (
                    f"pid{os.getpid()}__tp{tp_rank if tp_rank is not None else 'x'}"
                    f"__layer{final_layer_idx}__req{int(req_idx)}"
                    f"__state{int(layout.state_indices_list[req_idx])}"
                    f"__past{int(layout.past_text_lens[req_idx])}"
                    f"__{_safe_decode_trace_name(stage)}"
                )
                count = _HSD_FINAL_OUTPUT_TRACE_COUNTS.get(key, 0)
                _HSD_FINAL_OUTPUT_TRACE_COUNTS[key] = count + 1
                tensor_path = os.path.join(tensor_dir, f"{key}__{count:04d}.pt")
                torch.save(req_tensor.detach().cpu(), tensor_path)
                row["tensor_file"] = tensor_path
            try:
                os.makedirs(os.path.dirname(trace_file), exist_ok=True)
                with open(trace_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning_once("Failed to write HSD final output trace: %s", exc)

    def _trace_hsd_layer_output_stage(
        self,
        layer_num: int,
        stage: str,
        tensor: torch.Tensor | None,
        layout: HistoricalHSDLayout,
    ) -> None:
        trace_file = os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_FILE")
        if not trace_file or not torch.is_tensor(tensor):
            return
        selected_layers = _parse_layer_selector(
            os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_LAYERS")
        )
        if selected_layers is not None and int(layer_num) not in selected_layers:
            return
        selected_req_idxs = _parse_int_csv_selector(
            os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_REQ_IDXS")
        )
        target_past_lens = _parse_int_csv_selector(
            os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_TARGET_PAST_LENS")
        )
        mode = os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_MODE", "last").lower()
        if mode not in {"last", "full"}:
            mode = "last"
        tp_rank = _hsd_decode_tp_rank()
        global_rank = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_rank = int(torch.distributed.get_rank())

        for req_idx, (text_start, text_len) in enumerate(
            zip(layout.req_text_starts, layout.req_text_lens)
        ):
            if selected_req_idxs is not None and req_idx not in selected_req_idxs:
                continue
            past_text_len = int(layout.past_text_lens[req_idx])
            if target_past_lens is not None and past_text_len not in target_past_lens:
                continue
            text_len = int(text_len)
            if text_len <= 0:
                continue
            req_anchor_start = layout.req_anchor_starts[req_idx]
            req_anchor_len = layout.req_anchor_lens[req_idx]
            boundary_counts = layout.anchor_boundaries[
                req_anchor_start : req_anchor_start + req_anchor_len
            ]
            req_tensor = tensor[text_start : text_start + text_len]
            if mode == "last":
                req_tensor = req_tensor[-1:].view(1, 1, -1)
            else:
                req_tensor = req_tensor.view(1, text_len, -1)
            row: dict[str, object] = {
                "pid": int(os.getpid()),
                "trace_label": os.environ.get("VLLM_HSD_TRACE_LABEL"),
                "global_rank": global_rank,
                "tp_rank": tp_rank,
                "backend": "model_layer_output",
                "layer": int(layer_num),
                "stage": stage,
                "mode": mode,
                "req_idx": int(req_idx),
                "state_idx": int(layout.state_indices_list[req_idx]),
                "past_text_len": past_text_len,
                "text_len": text_len,
                "anchor_count": int(req_anchor_len),
                "boundary_counts": [
                    int(x) for x in boundary_counts.detach().cpu().reshape(-1).tolist()
                ],
            }
            row.update(_hsd_decode_trace_stats(req_tensor))
            if _env_truthy("VLLM_HSD_LAYER_OUTPUT_TRACE_SAVE_TENSORS"):
                tensor_dir = os.environ.get("VLLM_HSD_LAYER_OUTPUT_TRACE_TENSOR_DIR")
                if not tensor_dir:
                    tensor_dir = f"{trace_file}.tensors"
                os.makedirs(tensor_dir, exist_ok=True)
                key = (
                    f"pid{os.getpid()}__tp{tp_rank if tp_rank is not None else 'x'}"
                    f"__layer{int(layer_num)}__req{int(req_idx)}"
                    f"__state{int(layout.state_indices_list[req_idx])}"
                    f"__past{int(layout.past_text_lens[req_idx])}"
                    f"__{_safe_decode_trace_name(stage)}__{mode}"
                )
                count = _HSD_LAYER_OUTPUT_TRACE_COUNTS.get(key, 0)
                _HSD_LAYER_OUTPUT_TRACE_COUNTS[key] = count + 1
                tensor_path = os.path.join(tensor_dir, f"{key}__{count:04d}.pt")
                torch.save(req_tensor.detach().cpu(), tensor_path)
                row["tensor_file"] = tensor_path
            try:
                os.makedirs(os.path.dirname(trace_file), exist_ok=True)
                with open(trace_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning_once("Failed to write HSD layer output trace: %s", exc)
