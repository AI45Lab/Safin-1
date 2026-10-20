# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-only vLLM entrypoint for MARCH / Safin-1-MoE checkpoints.

The class intentionally registers the HF ``ForConditionalGeneration``
architecture as a text-generation model.  The native Historical path keeps
vLLM scheduler-visible tokens text-only, expands HSD anchor tokens inside the
model executor, and filters those synthetic rows before logits are produced.
"""

import os
from collections import OrderedDict
from collections.abc import Iterable
from itertools import islice
from math import isfinite, prod

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3_5RMSNorm,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn_linear_attn import (
    GatedDeltaNetAttention,
    fused_post_conv_prep,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.safin1 import (
    Safin1Config,
    Safin1TextConfig,
)
from vllm.transformers_utils.configs.safin1_moe import (
    Safin1MoeConfig,
    Safin1MoeTextConfig,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    get_dtype_size,
)
from vllm.v1.core.march_prefix_cache import validate_march_prefix_cache
from vllm.v1.kv_cache_interface import MambaSpec

from .interfaces import (
    HasInnerState,
    SupportsEagle3,
    SupportsMRoPE,
    SupportsPP,
)
from .march import backends
from .march.parallel import hsd_moe_dp_metadata
from .march.backends import (
    _load_hsd_deterministic_anchor_read as _load_hsd_deterministic_anchor_read,
)
from .march.backends import (
    _load_hsd_flashqla_prefill as _load_hsd_flashqla_prefill,
)
from .march.backends import (
    _load_hsd_fused_global_read as _load_hsd_fused_global_read,
)
from .march.backends import (
    _load_hsd_history_scale_add as _load_hsd_history_scale_add,
)
from .march.backends import (
    _load_hsd_paged_blocked_prefill_history_read as _load_hsd_paged_blocked_prefill_history_read,
)
from .march.backends import (
    _load_hsd_paged_history_read as _load_hsd_paged_history_read,
)
from .march.backends import (
    _load_hsd_paged_static_history_read as _load_hsd_paged_static_history_read,
)
from .march.backends import (
    _load_hsd_strided_gate_rmsnorm as _load_hsd_strided_gate_rmsnorm,
)
from .march.backends import (
    _load_hsd_valid_mask_copy as _load_hsd_valid_mask_copy,
)
from .march.backends import (
    _load_hsd_varlen_prefill as _load_hsd_varlen_prefill,
)
from .march.backends import (
    reference_causal_conv1d_fn as reference_causal_conv1d_fn,
)
from .march.decode import CachedDecodeMixin
from .march.decode_batch import BatchedDecodeMixin
from .march.decode_graph import GraphDecodeMixin
from .march.diagnostics import (
    AttentionDiagnosticsMixin as AttentionDiagnosticsMixin,
)
from .march.diagnostics import (
    DecoderDiagnosticsMixin as DecoderDiagnosticsMixin,
)
from .march.diagnostics import (
    ModelDiagnosticsMixin as ModelDiagnosticsMixin,
)
from .march.diagnostics import (
    _flush_hsd_varlen_event_aggregates as _flush_hsd_varlen_event_aggregates,
)
from .march.diagnostics import (
    _hsd_decode_tp_rank as _hsd_decode_tp_rank,
)
from .march.diagnostics import (
    _hsd_decode_trace_requested as _hsd_decode_trace_requested,
)
from .march.diagnostics import (
    _hsd_decode_trace_sample as _hsd_decode_trace_sample,
)
from .march.diagnostics import (
    _hsd_decode_trace_stats as _hsd_decode_trace_stats,
)
from .march.diagnostics import (
    _hsd_final_output_trace_requested as _hsd_final_output_trace_requested,
)
from .march.diagnostics import (
    _hsd_graph_v3_debug_sync as _hsd_graph_v3_debug_sync,
)
from .march.diagnostics import (
    _hsd_layer_output_trace_requested as _hsd_layer_output_trace_requested,
)
from .march.diagnostics import (
    _hsd_state_cache_log_enabled as _hsd_state_cache_log_enabled,
)
from .march.diagnostics import (
    _hsd_trace_requested as _hsd_trace_requested,
)
from .march.diagnostics import (
    _hsd_varlen_event_aggregate_enabled as _hsd_varlen_event_aggregate_enabled,
)
from .march.diagnostics import (
    _hsd_varlen_event_aggregate_flush_interval as _hsd_varlen_event_aggregate_flush_interval,
)
from .march.diagnostics import (
    _hsd_varlen_event_payload_value as _hsd_varlen_event_payload_value,
)
from .march.diagnostics import (
    _hsd_varlen_event_signature_value as _hsd_varlen_event_signature_value,
)
from .march.diagnostics import (
    _safe_decode_trace_name as _safe_decode_trace_name,
)
from .march.diagnostics import (
    set_vllm_hsd_trace_collector as set_vllm_hsd_trace_collector,
)
from .march.diagnostics import (
    trace_vllm_hsd_decode_stage as trace_vllm_hsd_decode_stage,
)
from .march.diagnostics import (
    trace_vllm_hsd_internal as trace_vllm_hsd_internal,
)
from .march.layout import (
    HistoricalHSDLayout as HistoricalHSDLayout,
)
from .march.layout import (
    HSDPreparedBatch as HSDPreparedBatch,
)
from .march.layout import (
    _build_hsd_layout as _build_hsd_layout,
)
from .march.layout import (
    _build_hsd_padded_static_decode_layout as _build_hsd_padded_static_decode_layout,
)
from .march.layout import (
    _text_only_hsd_positions as _text_only_hsd_positions,
)
from .march.layout import (
    _to_cpu_int_list as _to_cpu_int_list,
)
from .march.persistent_audit import (
    PersistentMemoryAuditMixin,
    _HSD_PERSISTENT_MEMORY_METRIC_NAMES as _HSD_PERSISTENT_MEMORY_METRIC_NAMES,
    _persistent_memory_tensor_checksum as _persistent_memory_tensor_checksum,
)
from .march.weights import (
    CausalLMWeightLoadingMixin,
    ModelWeightLoadingMixin,
    _HISTORICAL_HF_TO_VLLM_MAPPER as _HISTORICAL_HF_TO_VLLM_MAPPER,
    _HISTORICAL_HSD_PARAM_FRAGMENTS as _HISTORICAL_HSD_PARAM_FRAGMENTS,
    _is_historical_hsd_param as _is_historical_hsd_param,
    _is_optional_unmaterialized_hsd_param as _is_optional_unmaterialized_hsd_param,
    _is_optional_missing_hsd_param as _is_optional_missing_hsd_param,
    _maybe_shard_hsd_null_gate_bias as _maybe_shard_hsd_null_gate_bias,
    _load_hsd_persistent_memory_state as _load_hsd_persistent_memory_state,
    _load_hsd_persistent_memory_route_key as _load_hsd_persistent_memory_route_key,
    _is_hsd_conv1d_weight as _is_hsd_conv1d_weight,
    _maybe_shard_hsd_conv1d_weight as _maybe_shard_hsd_conv1d_weight,
)
from .march.prefill import PrefillMixin
from .march.prefill_batch import BatchedPrefillMixin
from .march.prefill_history import PagedPrefillMixin
from .march.prefill_segments import SegmentPrefillMixin
from .march.prefix_cache import PrefixSnapshotMixin
from .march.state_cache import StateCacheMixin
from .march.runtime import (
    _advance_hsd_conv_state as _advance_hsd_conv_state,
)
from .march.runtime import (
    _anchor_enabled as _anchor_enabled,
)
from .march.runtime import (
    _compact_all_layers_fast_path_enabled as _compact_all_layers_fast_path_enabled,
)
from .march.runtime import (
    _env_int as _env_int,
)
from .march.runtime import (
    _env_truthy as _env_truthy,
)
from .march.runtime import (
    _hsd_history_available as _hsd_history_available,
)
from .march.runtime import (
    _is_valid_hsd_state_index as _is_valid_hsd_state_index,
)
from .march.runtime import (
    _l2norm as _l2norm,
)
from .march.runtime import (
    _parse_csv_selector as _parse_csv_selector,
)
from .march.runtime import (
    _parse_int_csv_selector as _parse_int_csv_selector,
)
from .march.runtime import (
    _parse_layer_selector as _parse_layer_selector,
)
from .march.runtime import (
    _persistent_memory_insert_enabled as _persistent_memory_insert_enabled,
)
from .march.runtime import (
    _persistent_memory_mode as _persistent_memory_mode,
)
from .march.runtime import (
    _skip_redundant_recurrent_state_store_enabled as _skip_redundant_recurrent_state_store_enabled,
)
from .qwen3_5 import Qwen3_5Model
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
    capture_hsd_graph_v3_sidecar_row0,
)
from .utils import (
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter as is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


_HSD_PERSISTENT_MEMORY_MODES = frozenset({"enabled", "zero", "removed"})


def _configure_persistent_memory_mode(
    config: Safin1TextConfig | Safin1MoeTextConfig,
    vllm_config: VllmConfig,
) -> str:
    """Resolve the per-engine persistent-memory ablation exactly once."""

    declared = bool(getattr(config, "hsd_persistent_memory_enabled", False))
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    raw_mode = (
        additional_config.get("hsd_persistent_memory_mode")
        if isinstance(additional_config, dict)
        else None
    )
    mode = str(raw_mode or ("enabled" if declared else "removed")).strip().lower()
    if mode not in _HSD_PERSISTENT_MEMORY_MODES:
        raise ValueError(
            "additional_config['hsd_persistent_memory_mode'] must be one of "
            f"{sorted(_HSD_PERSISTENT_MEMORY_MODES)}, got {mode!r}."
        )
    if mode != "removed" and not declared:
        raise ValueError(
            f"hsd_persistent_memory_mode={mode!r} requires the checkpoint "
            "config field hsd_persistent_memory_enabled=true."
        )
    configured = getattr(config, "_vllm_hsd_persistent_memory_mode", None)
    if configured is not None and configured != mode:
        raise RuntimeError(
            "Persistent-memory mode is immutable for an engine: "
            f"configured={configured!r}, requested={mode!r}."
        )
    config._vllm_hsd_persistent_memory_mode = mode
    return mode


def _validate_persistent_memory_prefix_cache(
    config: Safin1TextConfig | Safin1MoeTextConfig,
    cache_config: object,
) -> None:
    from march.safin1.runtime_validation import validate_safin1_runtime
    validate_safin1_runtime(config)
    if bool(getattr(config, "hsd_persistent_memory_enabled", False)) and bool(
        getattr(cache_config, "enable_prefix_caching", False)
    ):
        raise NotImplementedError(
            "Safin-1 persistent memory explicitly requires prefix "
            "caching to be disabled in enabled, zero, and removed modes."
        )


def _validate_persistent_memory_architecture(
    config: Safin1TextConfig | Safin1MoeTextConfig,
) -> None:
    """Fail closed unless persistent state has the supported HSD geometry.

    Persistent memory lives entirely in the shared GatedDeltaNet attention
    implementation.  Dense and MoE Safin-1 therefore use the same
    state layout; the feed-forward architecture does not alter this contract.
    """

    if not bool(getattr(config, "hsd_persistent_memory_enabled", False)):
        return
    if not isinstance(
        config,
        (Safin1TextConfig, Safin1MoeTextConfig),
    ):
        raise NotImplementedError(
            "Persistent HSD memory requires a Safin-1 dense or "
            "MoE text configuration."
        )
    geometry = {
        "linear_num_value_heads": int(
            getattr(config, "linear_num_value_heads", 0) or 0
        ),
        "linear_key_head_dim": int(
            getattr(config, "linear_key_head_dim", 0) or 0
        ),
        "linear_value_head_dim": int(
            getattr(config, "linear_value_head_dim", 0) or 0
        ),
    }
    invalid = {name: value for name, value in geometry.items() if value <= 0}
    if invalid:
        raise ValueError(
            "Persistent HSD memory requires positive attention-state "
            f"geometry, got {invalid}."
        )


def _self_attention_layer_uses_hsd(
    config: Safin1TextConfig | Safin1MoeTextConfig,
    layer_idx: int,
) -> bool:
    if not bool(getattr(config, "hist_replace_self_attn", False)):
        return False
    layer_types = list(getattr(config, "layer_types", ()))
    if (
        int(layer_idx) < 0
        or int(layer_idx) >= len(layer_types)
        or layer_types[int(layer_idx)] != "full_attention"
    ):
        return False
    selected = _parse_layer_selector(
        getattr(config, "hist_replace_self_attn_layers", None)
    )
    return selected is None or (int(layer_idx) + 1) in selected


def _validate_native_hsd_config(config: Safin1MoeTextConfig) -> None:
    if bool(getattr(config, "hist_replace_self_attn", False)):
        selected = _parse_layer_selector(
            getattr(config, "hist_replace_self_attn_layers", None)
        )
        layer_types = list(getattr(config, "layer_types", ()))
        if selected is not None:
            invalid = sorted(
                layer_num
                for layer_num in selected
                if layer_num > len(layer_types)
                or layer_types[layer_num - 1] != "full_attention"
            )
            if invalid:
                raise ValueError(
                    "hist_replace_self_attn_layers must select only original "
                    f"full_attention layers; invalid layers: {invalid}."
                )
    route_position_encoding = str(
        getattr(config, "hsd_route_position_encoding", "none")
    ).lower()
    if route_position_encoding != "none":
        raise NotImplementedError(
            "Safin-1 vLLM HSD supports "
            "hsd_route_position_encoding='none' only."
        )
    route_position_mode = str(
        getattr(config, "hsd_route_position_mode", "chunk_end")
    ).lower()
    if route_position_mode != "chunk_end":
        raise NotImplementedError(
            "Safin-1 vLLM HSD supports "
            "hsd_route_position_mode='chunk_end' only."
        )
    if bool(getattr(config, "hist_disable_cache", False)):
        raise NotImplementedError(
            "Safin-1 vLLM HSD does not support hist_disable_cache=True."
        )
    value_heads = int(getattr(config, "linear_num_value_heads", 0) or 0)
    route_query_heads = int(
        getattr(config, "hsd_route_query_heads", None) or value_heads
    )
    route_key_heads = int(
        getattr(config, "hsd_route_key_heads", None) or value_heads
    )
    route_head_dim = int(
        getattr(config, "hsd_route_head_dim", None)
        or getattr(config, "linear_key_head_dim", 0)
        or 0
    )
    if route_head_dim <= 0:
        raise ValueError("hsd_route_head_dim must be positive.")
    if route_query_heads != value_heads:
        raise NotImplementedError(
            "Safin-1 vLLM currently requires "
            "hsd_route_query_heads=linear_num_value_heads."
        )
    if route_key_heads not in {1, value_heads}:
        raise NotImplementedError(
            "Safin-1 vLLM currently supports "
            "hsd_route_key_heads=1 or linear_num_value_heads."
        )


def _native_hsd_forward_error() -> NotImplementedError:
    return NotImplementedError(
        "Native vLLM HSD anchor-token forward requires scheduler/GDN metadata. "
        "Run through the v1 GPU runner so hsd_query_start_loc, "
        "hsd_num_computed_tokens, and hsd_state_indices are provided."
    )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _historical_hsd_enable_torch_compile(vllm_config: VllmConfig) -> bool:
    config = getattr(vllm_config.model_config, "hf_text_config", None)
    if config is not None and _anchor_enabled(config):
        if _env_truthy("VLLM_HSD_ENABLE_GRAPH_V3"):
            return True
        return not _env_truthy("VLLM_HSD_ALLOW_CUDAGRAPH")
    return True


def _route_softmax_weights(
    logits: torch.Tensor,
    valid: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    if valid.dtype is not torch.bool:
        valid = valid.bool()
    weights = torch.softmax(logits.float().masked_fill(~valid, float("-inf")), dim=dim)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights * valid.to(dtype=weights.dtype)
    denom = weights.sum(dim=dim, keepdim=True)
    weights = torch.where(denom > 0, weights / denom, torch.zeros_like(weights))
    return weights.to(dtype=logits.dtype)


def _validate_hsd_history_cache_options(additional_config: dict) -> str:
    """Keep explicit dense legacy settings, reject removed compression options."""
    state_format = str(os.environ.get(
        "VLLM_HSD_HISTORY_STATE_FORMAT",
        additional_config.get("hsd_history_state_format", "dense"),
    )).strip().lower()
    if state_format != "dense":
        raise ValueError("MARCH history cache supports only hsd_history_state_format='dense'.")
    for option, variable in (
        ("hsd_history_state_rank", "VLLM_HSD_HISTORY_STATE_RANK"),
        ("hsd_history_factor_power_iters", "VLLM_HSD_HISTORY_FACTOR_POWER_ITERS"),
    ):
        if option in additional_config or variable in os.environ:
            raise ValueError(f"{option} / {variable} has been removed; MARCH stores dense history states.")
    # Older dense launchers may explicitly set this formerly experimental knob.
    policy = str(os.environ.get(
        "VLLM_HSD_HISTORY_CACHE_CAPACITY_POLICY",
        additional_config.get("hsd_history_cache_capacity_policy", "same_dense_pages"),
    )).strip().lower()
    if policy != "same_dense_pages":
        raise ValueError("Only the legacy hsd_history_cache_capacity_policy='same_dense_pages' is accepted.")
    return state_format


class HistoricalGatedDeltaNetAttention(
    PrefixSnapshotMixin,
    StateCacheMixin,
    BatchedDecodeMixin,
    CachedDecodeMixin,
    GraphDecodeMixin,
    AttentionDiagnosticsMixin,
    PrefillMixin,
    BatchedPrefillMixin,
    PagedPrefillMixin,
    SegmentPrefillMixin,
    GatedDeltaNetAttention,
):
    """GDN layer with HSD side parameters registered for HF checkpoint loading."""

    def __init__(
        self,
        config: Safin1MoeTextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        create_in_proj_qkvz: bool = True,
    ) -> None:
        persistent_memory_mode = _configure_persistent_memory_mode(
            config,
            vllm_config,
        )
        _validate_persistent_memory_architecture(config)
        if _anchor_enabled(config):
            _validate_native_hsd_config(config)
            additional_config = getattr(vllm_config, "additional_config", None)
            if isinstance(additional_config, dict):
                additional_config.setdefault("gdn_prefill_backend", "triton")
        super().__init__(
            config=config,
            vllm_config=vllm_config,
            prefix=prefix,
            create_in_proj_qkvz=create_in_proj_qkvz,
            gqa_interleaved_layout=False,
        )
        quant_config = vllm_config.quant_config
        self._hsd_decode_projection_fusion_supported = (
            quant_config is None and create_in_proj_qkvz
        )
        self._hsd_decode_fused_proj_weight: torch.Tensor | None = None
        self._hsd_decode_fused_proj_sizes: tuple[int, ...] | None = None
        self.hsd_route_head_dim = int(
            getattr(config, "hsd_route_head_dim", None) or self.head_k_dim
        )
        self.hsd_route_query_heads = int(
            getattr(config, "hsd_route_query_heads", None) or self.num_v_heads
        )
        self.hsd_route_key_heads = int(
            getattr(config, "hsd_route_key_heads", None) or self.num_v_heads
        )
        if self.hsd_route_query_heads != self.num_v_heads:
            raise NotImplementedError(
                "vLLM HSD requires route query heads to match value heads."
            )
        if self.hsd_route_key_heads not in {1, self.num_v_heads}:
            raise NotImplementedError(
                "vLLM HSD supports one shared route key head or one route key "
                "per value head."
            )
        self.hsd_local_route_query_heads = self.hsd_route_query_heads // self.tp_size
        self.hsd_local_route_key_heads = (
            1
            if self.hsd_route_key_heads == 1
            else self.hsd_route_key_heads // self.tp_size
        )
        anchor_q_dim = self.num_v_heads * self.head_k_dim
        route_q_dim = self.hsd_route_query_heads * self.hsd_route_head_dim
        route_k_dim = self.hsd_route_key_heads * self.hsd_route_head_dim
        self.hsd_global_read_layers = _parse_layer_selector(
            getattr(config, "hsd_global_read_layers", None)
        )
        self.hsd_use_global_read = bool(getattr(config, "hsd_use_global_read", True))
        self.hist_use_history_read = bool(
            getattr(config, "hist_use_history_read", True)
        )
        self.hsd_layer_uses_global_read = self._global_read_enabled_for_layer()
        self.hsd_history_read_q_mode = "shared_local"
        if bool(getattr(config, "hsd_learnable_history_scale", False)):
            self.hsd_history_gate_logit = nn.Parameter(torch.zeros(()))
        else:
            self.hsd_history_gate_logit = None
        self.hsd_history_scale = float(getattr(config, "hsd_history_scale", 1.0))
        self.register_buffer(
            "hsd_history_scale_static",
            torch.tensor(self.hsd_history_scale, dtype=torch.float32),
            persistent=False,
        )
        self.hsd_history_out_rmsnorm = bool(
            getattr(config, "hsd_history_out_rmsnorm", False)
        )
        self.hsd_history_out_rmsnorm_eps = float(
            getattr(config, "hsd_history_out_rmsnorm_eps", 1e-6)
        )
        if self.hsd_history_out_rmsnorm_eps <= 0:
            raise ValueError("hsd_history_out_rmsnorm_eps must be positive.")
        self.hsd_learnable_history_scale_max = float(
            getattr(config, "hsd_learnable_history_scale_max", 1.0) or 1.0
        )
        self.hsd_use_route_scale = bool(getattr(config, "hsd_use_route_scale", True))
        self.hsd_route_logit_scale = float(
            getattr(config, "hsd_route_logit_scale", 1.0)
        )
        self.hsd_route_qk_l2norm = bool(
            getattr(config, "hsd_route_qk_l2norm", False)
        )
        self.hsd_read_topk = max(0, int(getattr(config, "hsd_read_topk", 0) or 0))
        self.hsd_null_read_gate = bool(getattr(config, "hsd_null_read_gate", False))
        self.hsd_persistent_memory_enabled = bool(
            getattr(config, "hsd_persistent_memory_enabled", False)
        )
        self.hsd_persistent_memory_route_key_enabled = bool(
            getattr(config, "hsd_persistent_memory_route_key_enabled", False)
        )
        self.hsd_persistent_memory_route_key_mode = str(
            getattr(config, "hsd_persistent_memory_route_key_mode", "legacy")
        ).strip().lower()
        if self.hsd_persistent_memory_route_key_mode not in {"dedicated", "legacy"}:
            raise ValueError(
                "hsd_persistent_memory_route_key_mode must be 'dedicated' or 'legacy'."
            )
        if self.hsd_persistent_memory_route_key_enabled and not self.hsd_persistent_memory_enabled:
            raise ValueError(
                "hsd_persistent_memory_route_key_enabled requires hsd_persistent_memory_enabled."
            )
        if (
            self.hsd_persistent_memory_route_key_mode == "dedicated"
            and not self.hsd_persistent_memory_route_key_enabled
        ):
            raise ValueError(
                "route-key mode 'dedicated' requires hsd_persistent_memory_route_key_enabled."
            )
        if self.hsd_persistent_memory_route_key_enabled and (
            self.hsd_route_key_heads != self.num_v_heads
            or self.hsd_route_head_dim != self.head_k_dim
        ):
            raise ValueError(
                "Dedicated persistent-memory route keys require one full key per value head."
            )
        self.hsd_persistent_memory_mode = persistent_memory_mode
        self.hsd_null_read_gate_init_bias = float(
            getattr(config, "hsd_null_read_gate_init_bias", 0.0)
        )
        if not isfinite(self.hsd_null_read_gate_init_bias):
            raise ValueError("hsd_null_read_gate_init_bias must be finite.")
        self.hsd_local_state_mode = "full"
        self.hsd_anchor_cache: dict[int, dict[str, torch.Tensor]] = {}
        self.hsd_anchor_page_cache = None
        self.hsd_text_state_cache: dict[int, dict[str, torch.Tensor]] = {}
        self.hsd_use_persistent_state_sidecar = _env_truthy(
            "VLLM_HSD_USE_PERSISTENT_STATE_SIDECAR"
        )
        self.hsd_use_compact_state_slots = (
            self.hsd_use_persistent_state_sidecar
            and _env_truthy("VLLM_HSD_ENABLE_COMPACT_STATE_SLOTS")
        )
        self.hsd_persistent_state_sidecar: dict[str, torch.Tensor] = {}
        self._hsd_prefill_passive_sidecars: dict[str, torch.Tensor] = {}
        self._hsd_prefill_passive_meta: dict[str, int] = {}
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        self.hsd_persistent_state_sidecar_rows = max(
            2,
            int(getattr(scheduler_config, "max_num_seqs", 128) or 128) + 1,
        )
        self.hsd_static_text_state_sidecar: dict[str, torch.Tensor] = {}
        self.hsd_static_text_state_sidecar_state_indices: list[int] = []
        self.hsd_static_text_state_sidecar_past_lens: list[int] = []
        self.hsd_prefix_snapshot_cache: OrderedDict[
            str, dict[str, torch.Tensor | int]
        ] = OrderedDict()
        self.hsd_prefix_recurrent_replay: dict[
            int, dict[str, torch.Tensor | int]
        ] = {}
        self.hsd_prefix_snapshot_bytes = 0
        self.hsd_prefix_cache_generation = 0
        self.hsd_prefix_snapshot_step_evicted_keys: set[str] = set()
        additional_config = getattr(vllm_config, "additional_config", {}) or {}
        self._apply_decode_trace_additional_config(additional_config)
        self.hsd_decode_backend = str(
            additional_config.get("hsd_decode_backend", "packed")
        ).strip().lower()
        if self.hsd_decode_backend not in {"packed", "recurrent"}:
            raise ValueError(
                "hsd_decode_backend must be either 'packed' or 'recurrent', "
                f"got {self.hsd_decode_backend!r}."
            )
        self.hsd_decode_prepare_backend = str(
            additional_config.get("hsd_decode_prepare_backend", "batched")
        ).strip().lower()
        if self.hsd_decode_prepare_backend not in {"loop", "batched"}:
            raise ValueError(
                "hsd_decode_prepare_backend must be either 'loop' or 'batched', "
                f"got {self.hsd_decode_prepare_backend!r}."
            )
        default_prefill_backend = (
            "segments"
            if str(getattr(self.config, "model_type", ""))
            == "safin1_35b_a3b_text"
            else "varlen"
        )
        self.hsd_prefill_backend = str(
            additional_config.get(
                "hsd_prefill_backend",
                default_prefill_backend,
            )
        ).strip().lower()
        if self.hsd_prefill_backend not in {"segments", "varlen"}:
            raise ValueError(
                "hsd_prefill_backend must be either 'segments' or 'varlen', "
                f"got {self.hsd_prefill_backend!r}."
            )
        self.hsd_local_prefill_producer_backend = str(
            getattr(
                self.config,
                "hsd_local_prefill_producer_backend",
                "triton",
            )
        ).strip().lower()
        if self.hsd_local_prefill_producer_backend not in {
            "triton",
            "flashqla_tile",
        }:
            raise ValueError(
                "hsd_local_prefill_producer_backend must be either 'triton' "
                "or 'flashqla_tile', got "
                f"{self.hsd_local_prefill_producer_backend!r}."
            )
        self.hsd_local_prefill_producer_strict = bool(
            getattr(
                self.config,
                "hsd_local_prefill_producer_strict",
                False,
            )
        )
        history_read_backend = os.environ.get(
            "VLLM_HSD_HISTORY_READ_BACKEND",
            additional_config.get("hsd_history_read_backend", "auto"),
        )
        self.hsd_history_read_backend = str(history_read_backend).strip().lower()
        if self.hsd_history_read_backend not in {"auto", "loop", "paged_batched"}:
            raise ValueError(
                "hsd_history_read_backend must be one of 'auto', 'loop', or "
                f"'paged_batched', got {self.hsd_history_read_backend!r}."
            )
        history_cache_backend = os.environ.get(
            "VLLM_HSD_HISTORY_CACHE_BACKEND",
            additional_config.get("hsd_history_cache_backend", "auto"),
        )
        self.hsd_history_cache_backend = str(history_cache_backend).strip().lower()
        if self.hsd_history_cache_backend not in {
            "auto",
            "scheduler_paged",
            "legacy_side_paged",
        }:
            raise ValueError(
                "hsd_history_cache_backend must be one of 'auto', "
                "'scheduler_paged', or 'legacy_side_paged', got "
                f"{self.hsd_history_cache_backend!r}."
            )
        self.hsd_history_state_format = _validate_hsd_history_cache_options(additional_config)
        self.hsd_history_page_alloc_failures = 0
        self.hsd_prefill_varlen_fallback_counts: dict[str, int] = {}
        self.hsd_prefill_varlen_counters: dict[str, int] = {
            "batched_prepare_calls": 0,
            "grouped_varlen_calls": 0,
            "grouped_varlen_requests": 0,
            "grouped_varlen_text_tokens": 0,
            "grouped_varlen_anchor_tokens": 0,
        }
        self.hsd_decode_prepare_fallback_counts: dict[str, int] = {}
        self.hsd_decode_prepare_counters: dict[str, int] = {
            "batched_decode_light_calls": 0,
            "batched_decode_light_requests": 0,
            "batched_decode_light_anchor_tokens": 0,
        }
        self.hsd_post_conv_backend = str(
            additional_config.get("hsd_post_conv_backend", "fused")
        ).strip().lower()
        if self.hsd_post_conv_backend not in {"fused", "torch"}:
            raise ValueError(
                "hsd_post_conv_backend must be either 'fused' or 'torch', "
                f"got {self.hsd_post_conv_backend!r}."
            )
        # Keep Graph V3 on the same raw-Q normalization path as eager decode.
        # Pre-normalizing Q is mathematically equivalent, but changes BF16
        # reduction order enough for MoE routing to amplify the difference.
        self.hsd_graph_v3_exact_read_q = _env_truthy(
            "VLLM_HSD_GRAPH_V3_EXACT_READ_Q"
        ) and not _env_truthy("VLLM_HSD_ALLOW_INEXACT_RAW_Q_PAGED_READ")

        self.anchor_q_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=anchor_q_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.anchor_q_proj",
        )
        if self.hsd_layer_uses_global_read:
            self.history_read_q_down_proj = None
            self.history_read_q_up_proj = None
            anchor_k_cls = (
                ReplicatedLinear
                if self.hsd_route_key_heads == 1
                else ColumnParallelLinear
            )
            self.anchor_k_proj = anchor_k_cls(
                input_size=self.hidden_size,
                output_size=route_k_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.anchor_k_proj",
            )
            self.router_q_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=route_q_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.router_q_proj",
            )
            if self.hsd_null_read_gate:
                self.null_gate_proj = ColumnParallelLinear(
                    input_size=self.hidden_size,
                    output_size=self.num_v_heads,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.null_gate_proj",
                )
                null_gate_weight = getattr(self.null_gate_proj, "weight", None)
                if null_gate_weight is not None:
                    with torch.no_grad():
                        null_gate_weight.zero_()
                self.hsd_null_read_gate_bias = nn.Parameter(
                    torch.full(
                        (self.num_v_heads // self.tp_size,),
                        self.hsd_null_read_gate_init_bias,
                        dtype=torch.float32,
                    )
                )
            else:
                self.null_gate_proj = None
                self.hsd_null_read_gate_bias = None
        else:
            self.history_read_q_down_proj = None
            self.history_read_q_up_proj = None
            self.anchor_k_proj = None
            self.router_q_proj = None
            self.null_gate_proj = None
            self.hsd_null_read_gate_bias = None
        if self.hsd_layer_uses_global_read and self.hsd_history_out_rmsnorm:
            self.hsd_history_out_rmsnorm_weight = nn.Parameter(torch.ones(self.head_v_dim))
        else:
            self.hsd_history_out_rmsnorm_weight = None
        if self._persistent_memory_state_required():
            # Shard only the value-head axis.  This intentionally bypasses the
            # generic flatten sharder used by some parameters.
            self.persistent_memory_state = nn.Parameter(
                torch.zeros(
                    self.num_v_heads // self.tp_size,
                    self.head_k_dim,
                    self.head_v_dim,
                    dtype=torch.float32,
                )
            )
            if self.hsd_persistent_memory_route_key_enabled:
                self.persistent_memory_route_key = nn.Parameter(
                    torch.zeros(
                        self.num_v_heads // self.tp_size,
                        self.head_k_dim,
                        dtype=torch.float32,
                    )
                )
            else:
                self.register_parameter("persistent_memory_route_key", None)
        else:
            self.register_parameter("persistent_memory_state", None)
            self.register_parameter("persistent_memory_route_key", None)
        self._hsd_persistent_memory_metrics: dict[str, torch.Tensor] = {}
        self._hsd_persistent_memory_metric_sums: dict[str, torch.Tensor] = {}
        self._hsd_persistent_memory_metric_count = 0
        # Eager paged helpers write their latest scalar tensors here. The dict
        # is allocated before graph capture and is never passed while a CUDA
        # stream is capturing. These values intentionally have latest-sample
        # semantics; they must not be mixed into the cumulative eager mean.
        self._hsd_paged_persistent_memory_metrics: dict[str, torch.Tensor] = {}
        for projection in (
            getattr(self, "in_proj_qkvz", None),
            getattr(self, "in_proj_ba", None),
            getattr(self, "out_proj", None),
            self.anchor_q_proj,
            self.history_read_q_down_proj,
            self.history_read_q_up_proj,
            self.anchor_k_proj,
            self.router_q_proj,
            self.null_gate_proj,
        ):
            if projection is not None:
                projection._hsd_batch_invariant_linear = True
        self._hsd_graph_v3_tensor_sidecars: dict[str, torch.Tensor] = {}

    def _apply(self, fn, recurse: bool = True):
        """Move the layer while retaining the checkpoint memory in FP32."""

        result = super()._apply(fn, recurse=recurse)
        for name in ("persistent_memory_state", "persistent_memory_route_key"):
            parameter = self._parameters.get(name)
            if parameter is not None and parameter.dtype != torch.float32:
                with torch.no_grad():
                    parameter.data = parameter.data.float()
                    if parameter.grad is not None:
                        parameter.grad.data = parameter.grad.data.float()
        return result

    def prepare_hsd_graph_v3_tensor_sidecars(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Allocate optional HSD attention snapshots before graph capture."""
        if not os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_DIR"):
            return
        layer_num = int(extract_layer_index(self.prefix)) + 1
        selected_layers = _parse_layer_selector(
            os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_LAYERS")
        )
        if selected_layers is not None and layer_num not in selected_layers:
            return
        stages = {
            "hsd_recurrent_out",
            "hsd_history_raw",
            "hsd_history_out",
            "hsd_gate_norm_out",
            "hsd_projected_out",
        }
        selected_stages = _parse_csv_selector(
            os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_STAGES")
        )
        if selected_stages is not None:
            stages.intersection_update(selected_stages)
        local_value_dim = self.value_dim // self.tp_size
        for stage in stages:
            width = (
                int(self.hidden_size)
                if stage == "hsd_projected_out"
                else int(local_value_dim)
            )
            if stage in self._hsd_graph_v3_tensor_sidecars:
                continue
            target = torch.empty(width, device=device, dtype=dtype)
            self.register_buffer(
                f"_hsd_graph_v3_tensor_sidecar_{stage}",
                target,
                persistent=False,
            )
            self._hsd_graph_v3_tensor_sidecars[stage] = target


    def _capture_hsd_graph_v3_tensor_sidecar(
        self,
        stage: str,
        tensor: torch.Tensor | None,
    ) -> None:
        if (
            not os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_DIR")
            or not torch.is_tensor(tensor)
        ):
            return
        target = self._hsd_graph_v3_tensor_sidecars.get(stage)
        if target is None:
            return
        capture_hsd_graph_v3_sidecar_row0(
            tensor.reshape(tensor.shape[0], -1),
            target,
        )


    @property
    def route_q_proj(self):
        return self.router_q_proj

    @property
    def route_k_proj(self):
        return self.anchor_k_proj

    @staticmethod
    def _route_logits(
        router_q: torch.Tensor,
        anchor_k: torch.Tensor,
        value_heads: int,
    ) -> torch.Tensor:
        query_heads = int(router_q.shape[2])
        key_heads = int(anchor_k.shape[2])
        if query_heads != int(value_heads):
            raise RuntimeError(
                "HSD route query heads must match local value heads: "
                f"query_heads={query_heads}, value_heads={value_heads}."
            )
        if router_q.shape[-1] != anchor_k.shape[-1]:
            raise RuntimeError(
                "HSD route q/k dimensions must match: "
                f"q={router_q.shape[-1]}, k={anchor_k.shape[-1]}."
            )
        if key_heads == query_heads:
            return torch.einsum(
                "bthk,bmhk->bthm", router_q.float(), anchor_k.float()
            )
        if key_heads == 1:
            return torch.einsum(
                "bthk,bmk->bthm", router_q.float(), anchor_k[:, :, 0].float()
            )
        raise RuntimeError(
            "HSD route key heads must be one or match query heads: "
            f"key_heads={key_heads}, query_heads={query_heads}."
        )

    def _project_null_gate_logits(self, text_hidden: torch.Tensor) -> torch.Tensor | None:
        if self.null_gate_proj is None:
            return None
        if self.hsd_null_read_gate_bias is None:
            raise RuntimeError(
                "hsd_null_read_gate_bias is required when HSD null read gate is enabled."
            )
        logits, _ = self.null_gate_proj(text_hidden)
        expected_heads = self.num_v_heads // self.tp_size
        if logits.shape[-1] != expected_heads:
            raise RuntimeError(
                "HSD null read gate output size must match local value heads: "
                f"logits={logits.shape[-1]}, local_value_heads={expected_heads}."
            )
        bias = self.hsd_null_read_gate_bias.float().to(device=logits.device).view(1, -1)
        return logits.float() + bias


    def _global_read_enabled_for_layer(self) -> bool:
        if not self.hsd_use_global_read or not self.hist_use_history_read:
            return False
        if self.hsd_global_read_layers is None:
            return True
        return (int(self.layer_idx) + 1) in self.hsd_global_read_layers

    def _persistent_memory_state_required(self) -> bool:
        """Mirror the HF causal-path rule for persistent state parameters."""

        if not self.hsd_persistent_memory_enabled or not self.hsd_use_global_read:
            return False
        # With an absolute dedicated key, non-reader states can no longer
        # affect a later legacy key.  Materialize only real state/key pairs.
        if bool(getattr(self, "hsd_persistent_memory_route_key_enabled", False)):
            return self.hsd_layer_uses_global_read
        layer_types = list(getattr(self.config, "layer_types", ()) or ())
        num_layers = int(
            getattr(self.config, "num_hidden_layers", len(layer_types))
            or len(layer_types)
        )
        if not layer_types:
            return True
        if (
            int(self.layer_idx) >= len(layer_types)
            or layer_types[int(self.layer_idx)] != "linear_attention"
        ):
            return False
        candidates = [
            idx
            for idx, layer_type in enumerate(layer_types[:num_layers])
            if layer_type == "linear_attention"
            and (
                self.hsd_global_read_layers is None
                or (idx + 1) in self.hsd_global_read_layers
            )
        ]
        return bool(candidates) and int(self.layer_idx) <= max(candidates)

    def _replace_persistent_anchor_route_key(
        self,
        anchor_k: torch.Tensor,
        persistent_anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace only persistent-slot keys before the existing normalization."""

        if self.hsd_persistent_memory_route_key_mode != "dedicated":
            return anchor_k
        parameter = self.persistent_memory_route_key
        if parameter is None:
            raise RuntimeError(
                "Dedicated persistent-memory routing requires persistent_memory_route_key."
            )
        mask = persistent_anchor_mask.to(device=anchor_k.device, dtype=torch.bool)
        if anchor_k.ndim == 3:
            mask = mask.reshape(-1)
            if int(mask.numel()) != int(anchor_k.shape[0]):
                raise ValueError(
                    "persistent route-key mask does not match packed anchor keys."
                )
            view_mask = mask.view(-1, 1, 1)
            expanded = parameter.to(device=anchor_k.device).view(1, *parameter.shape)
            expanded = expanded.expand(anchor_k.shape[0], *parameter.shape)
        elif anchor_k.ndim == 4:
            if tuple(mask.shape) != tuple(anchor_k.shape[:2]):
                raise ValueError(
                    "persistent route-key mask does not match batched anchor keys."
                )
            view_mask = mask.view(*mask.shape, 1, 1)
            expanded = parameter.to(device=anchor_k.device).view(1, 1, *parameter.shape)
            expanded = expanded.expand(*anchor_k.shape[:2], *parameter.shape)
        else:
            raise ValueError(
                f"anchor_k must be rank 3 or 4, got shape={tuple(anchor_k.shape)}."
            )
        if tuple(parameter.shape) != tuple(anchor_k.shape[-2:]):
            raise RuntimeError(
                "persistent_memory_route_key shape mismatch: "
                f"parameter={tuple(parameter.shape)}, keys={tuple(anchor_k.shape[-2:])}."
            )
        return torch.where(view_mask, expanded.to(dtype=anchor_k.dtype), anchor_k)

    def hsd_history_page_size_bytes(
        self,
        page_m: int = 16,
    ) -> int:
        if not self.hsd_layer_uses_global_read:
            return 0
        local_v_heads = self.num_v_heads // self.tp_size
        state_dtype = self.get_state_dtype()[1]
        key_dtype = next(self.parameters()).dtype
        state_elements_per_head = self.head_v_dim * self.head_k_dim
        return int(page_m) * (
            local_v_heads * state_elements_per_head * get_dtype_size(state_dtype)
            + self.hsd_local_route_key_heads
            * self.hsd_route_head_dim
            * get_dtype_size(key_dtype)
            + get_dtype_size(torch.long)
        )

    def initialize_hsd_history_page_cache(
        self,
        *,
        num_pages: int,
        device: torch.device,
        page_m: int = 16,
        max_model_len: int | None = None,
    ) -> int:
        if not self.hsd_layer_uses_global_read or num_pages <= 0:
            return 0
        loaded = backends._load_hsd_paged_history_read()
        if loaded is None:
            return 0
        spec_cls, cache_cls, _ = loaded
        local_v_heads = self.num_v_heads // self.tp_size
        state_dtype = self.get_state_dtype()[1]
        key_dtype = next(self.parameters()).dtype
        page_cache = cache_cls(
            page_m=page_m,
            state_format=self.hsd_history_state_format,
        )
        page_cache.preallocate(
            spec=spec_cls(
                page_m=int(page_m),
                num_pages=int(num_pages),
                heads=int(local_v_heads),
                value_dim=int(self.head_v_dim),
                key_dim=int(self.head_k_dim),
                states_dtype=state_dtype,
                keys_dtype=key_dtype,
                route_heads=int(self.hsd_local_route_key_heads),
                route_dim=int(self.hsd_route_head_dim),
                state_format=self.hsd_history_state_format,
            ),
            device=device,
        )
        if _env_truthy("VLLM_HSD_ENABLE_GRAPH_V3"):
            max_model_len = int(
                max_model_len
                or getattr(self.config, "max_position_embeddings", 0)
                or 1
            )
            chunk_size = int(getattr(self.config, "anchor_chunk_size", 0) or 1)
            max_anchors = max(1, (max_model_len + chunk_size - 1) // chunk_size)
            if self.hsd_persistent_memory_mode != "removed":
                max_anchors += 1
            max_pages_per_state = max(
                1,
                (max_anchors + int(page_m) - 1) // int(page_m),
            )
            page_cache.preallocate_metadata(
                state_rows=int(self.hsd_persistent_state_sidecar_rows),
                max_pages_per_state=max_pages_per_state,
                device=device,
            )
        self.hsd_anchor_page_cache = page_cache
        return page_cache.total_pages

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        if (
            _env_truthy("VLLM_HSD_ENABLE_COMPACT_STATE_KV_SPEC")
            and _env_truthy("VLLM_HSD_USE_PERSISTENT_STATE_SIDECAR")
            and _env_truthy("VLLM_HSD_ENABLE_COMPACT_STATE_SLOTS")
        ):
            state_dtypes = self.get_state_dtype()
            attention_kv_heads = max(
                1,
                int(self.config.num_key_value_heads) // self.tp_size,
            )
            attention_head_dim = int(
                getattr(self.config, "head_dim", None)
                or self.config.hidden_size
                // max(1, int(self.config.num_attention_heads) // self.tp_size)
            )
            attention_page_size = (
                2
                * int(vllm_config.cache_config.block_size)
                * attention_kv_heads
                * attention_head_dim
                * get_dtype_size(vllm_config.model_config.dtype)
            )
            return MambaSpec(
                shapes=((1, 4), (1, 4)),
                dtypes=state_dtypes,
                block_size=int(vllm_config.cache_config.block_size),
                page_size_padded=attention_page_size,
                mamba_type=self.mamba_type,
                mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
                num_speculative_blocks=0,
            )
        if vllm_config.cache_config.mamba_block_size is not None:
            return super().get_kv_cache_spec(vllm_config)
        state_shapes = tuple(self.get_state_shape())
        state_dtypes = self.get_state_dtype()
        real_page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for shape, dtype in zip(state_shapes, state_dtypes)
        )
        attention_kv_heads = max(1, int(self.config.num_key_value_heads) // self.tp_size)
        attention_head_dim = int(
            getattr(self.config, "head_dim", None)
            or self.config.hidden_size
            // max(1, int(self.config.num_attention_heads) // self.tp_size)
        )
        attention_page_size = (
            2
            * int(vllm_config.cache_config.block_size)
            * attention_kv_heads
            * attention_head_dim
            * get_dtype_size(vllm_config.model_config.dtype)
        )
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        if page_size_padded is None:
            page_size_padded = _round_up_to_multiple(
                real_page_size, attention_page_size
            )
        block_ratio = max(1, page_size_padded // max(1, attention_page_size))
        block_size = int(vllm_config.cache_config.block_size) * block_ratio
        return MambaSpec(
            shapes=state_shapes,
            dtypes=state_dtypes,
            block_size=block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
        )

    def ensure_hsd_graph_v3_state_sidecars(self) -> bool:
        if not self._use_hsd_persistent_state_sidecar():
            return False
        required_rows = int(self.hsd_persistent_state_sidecar_rows)
        conv_cache = self._conv_cache_for_kernel(required_rows)
        recurrent_cache = self._ssm_cache_for_kernel(required_rows)
        return conv_cache is not None and recurrent_cache is not None

    def _hsd_graph_v3_state_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        conv_cache = self.hsd_persistent_state_sidecar.get("conv_state")
        recurrent_cache = self.hsd_persistent_state_sidecar.get("recurrent_state")
        if not torch.is_tensor(conv_cache) or not torch.is_tensor(recurrent_cache):
            raise RuntimeError(
                "hsd_graph_v3_state_sidecar_missing: Graph V3 requires fixed "
                "conv/recurrent sidecars to be initialized before compilation."
            )
        return conv_cache, recurrent_cache


    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        hsd_layout: HistoricalHSDLayout | None = None,
    ):
        if _anchor_enabled(self.config):
            if hsd_layout is None:
                raise _native_hsd_forward_error()
            self.forward_hsd(
                hidden_states=hidden_states,
                output=output,
                layout=hsd_layout,
            )
            return
        return super().forward(hidden_states=hidden_states, output=output)

    def _layout_has_anchor_cache(self, layout: HistoricalHSDLayout) -> bool:
        for req_idx, past_text_len in enumerate(layout.past_text_lens):
            if int(past_text_len) <= 0:
                continue
            state_idx = self._state_index_for_request(layout, req_idx)
            if not _is_valid_hsd_state_index(state_idx):
                continue
            page_cache = getattr(self, "hsd_anchor_page_cache", None)
            if page_cache is not None and page_cache.has_state(state_idx):
                return True
            cached = self.hsd_anchor_cache.get(state_idx)
            if cached is None:
                continue
            states = cached.get("states")
            if torch.is_tensor(states) and int(states.shape[1]) > 0:
                return True
        return False

    def _state_has_anchor_history(self, state_idx: int) -> bool:
        if not _is_valid_hsd_state_index(state_idx):
            return False
        page_cache = getattr(self, "hsd_anchor_page_cache", None)
        if page_cache is not None and page_cache.has_state(state_idx):
            return True
        cached = self.hsd_anchor_cache.get(state_idx)
        if cached is None:
            return False
        states = cached.get("states")
        keys = cached.get("keys")
        boundaries = cached.get("boundaries")
        return (
            torch.is_tensor(states)
            and torch.is_tensor(keys)
            and torch.is_tensor(boundaries)
            and int(states.shape[1]) > 0
            and int(keys.shape[1]) > 0
            and int(boundaries.shape[0]) > 0
        )

    def _legacy_anchor_history(
        self,
        state_idx: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not _is_valid_hsd_state_index(state_idx):
            return None
        cached = self.hsd_anchor_cache.get(state_idx)
        if cached is None:
            return None
        states = cached.get("states")
        keys = cached.get("keys")
        boundaries = cached.get("boundaries")
        if not (
            torch.is_tensor(states)
            and torch.is_tensor(keys)
            and torch.is_tensor(boundaries)
            and int(states.shape[1]) > 0
            and int(keys.shape[1]) > 0
        ):
            return None
        return (
            states.to(device=device),
            keys.to(device=device),
            boundaries.to(device=device),
        )

    def _get_hsd_anchor_page_cache(self, *, allow_legacy_create: bool = False):
        page_cache = getattr(self, "hsd_anchor_page_cache", None)
        if page_cache is not None:
            return page_cache
        if not allow_legacy_create:
            return None
        loaded = backends._load_hsd_paged_history_read()
        if loaded is None:
            return None
        _, cache_cls, _ = loaded
        page_cache = cache_cls()
        self.hsd_anchor_page_cache = page_cache
        return page_cache

    def _using_scheduler_history_cache(self) -> bool:
        page_cache = getattr(self, "hsd_anchor_page_cache", None)
        return bool(page_cache is not None and getattr(page_cache, "fixed_capacity", False))

    def _allow_paged_prefill_history_read(self) -> bool:
        if int(getattr(self, "hsd_read_topk", 0) or 0) == 0:
            return False
        if str(getattr(self.config, "model_type", "")) != (
            "safin1_35b_a3b_text"
        ):
            return True
        # The paged Triton reduction is stable, but its BF16 accumulation order
        # differs enough from the reference prefill read to flip MoE top-k
        # routing. Keep decode paged; prefill stays on the canonical arithmetic
        # unless an experiment explicitly opts back in.
        return _env_truthy("VLLM_HSD_ALLOW_MOE_PAGED_PREFILL_HISTORY_READ")

    def _can_use_paged_history_read(
        self,
        *,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        null_logits: torch.Tensor | None,
        trace_requested: bool,
    ) -> bool:
        backend = str(getattr(self, "hsd_history_read_backend", "auto")).strip().lower()
        if backend == "loop":
            return False
        if backend not in {"auto", "paged_batched"}:
            return False
        cache_backend = str(
            getattr(self, "hsd_history_cache_backend", "auto")
        ).strip().lower()
        if cache_backend in {"auto", "scheduler_paged"} and not self._using_scheduler_history_cache():
            return False
        if cache_backend != "legacy_side_paged" and not self._using_scheduler_history_cache():
            return False
        if trace_requested or torch.is_grad_enabled():
            return False
        read_topk = int(getattr(self, "hsd_read_topk", 0) or 0)
        if read_topk not in {0, 4}:
            return False
        if read_topk == 0:
            if (
                self.hsd_route_query_heads != 32
                or self.hsd_route_key_heads != 1
                or self.hsd_route_head_dim != 64
            ):
                logger.warning_once(
                    "HSD paged read-all auto dispatch requires Q32/K1 d64; "
                    "using canonical history read for this configuration."
                )
                return False
        if read_q.ndim != 4 or router_q.ndim != 4:
            return False
        if int(read_q.shape[0]) != 1 or int(read_q.shape[1]) != 1:
            return False
        if null_logits is not None:
            if null_logits.ndim != 4 or tuple(null_logits.shape[:2]) != (1, 1):
                return False
            if int(null_logits.shape[2]) != int(read_q.shape[2]):
                return False
            if int(null_logits.shape[3]) != 1:
                return False
            if null_logits.device != read_q.device:
                return False
        if not read_q.is_cuda or router_q.device != read_q.device:
            return False
        if int(read_q.shape[-1]) > 128 or int(getattr(self, "head_v_dim", 0) or 0) > 128:
            return False
        return backends._load_hsd_paged_history_read() is not None

    def _migrate_legacy_anchor_cache_to_pages(self, state_idx: int) -> bool:
        page_cache = self._get_hsd_anchor_page_cache(
            allow_legacy_create=(
                str(getattr(self, "hsd_history_cache_backend", "auto")).strip().lower()
                == "legacy_side_paged"
            )
        )
        if page_cache is None:
            return False
        cached = self.hsd_anchor_cache.get(state_idx)
        if cached is None:
            return True
        cached_states = cached.get("states")
        if torch.is_tensor(cached_states) and int(cached_states.shape[1]) == 0:
            # Segments/reference prefill can leave an empty legacy placeholder
            # before the first anchor boundary. There is nothing to migrate,
            # and treating it as a migration failure prevents the first real
            # anchor from ever reaching the scheduler-paged cache.
            self.hsd_anchor_cache.pop(state_idx, None)
            return True
        paged_cached = dict(cached)
        if torch.is_tensor(cached_states):
            paged_cached["states"] = self._anchor_states_to_paged_cache(
                cached_states
            )
        if page_cache.migrate_from_legacy(state_idx, paged_cached):
            self.hsd_anchor_cache.pop(state_idx, None)
            return True
        return page_cache.has_state(state_idx)

    @staticmethod
    def _anchor_states_to_paged_cache(states: torch.Tensor) -> torch.Tensor:
        """Convert model recurrent states [..., H, K, V] to [..., H, V, K]."""
        return states.transpose(-1, -2).contiguous()

    @staticmethod
    def _anchor_states_from_paged_cache(states: torch.Tensor) -> torch.Tensor:
        """Convert paged states [..., H, V, K] to model [..., H, K, V]."""
        return states.transpose(-1, -2).contiguous()

    def _materialize_anchor_page_history(
        self,
        page_cache,
        state_idx: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        materialized = page_cache.materialize(state_idx, device=device)
        if materialized is None:
            return None
        states, keys, boundaries = materialized
        return (
            self._anchor_states_from_paged_cache(states),
            keys,
            boundaries,
        )

    def _state_index_for_request(
        self,
        layout: HistoricalHSDLayout,
        req_idx: int,
    ) -> int:
        return int(layout.state_indices_list[req_idx])

    def _history_state_index_for_request(
        self,
        layout: HistoricalHSDLayout,
        req_idx: int,
    ) -> int:
        history_indices = layout.history_state_indices_list
        if history_indices is None:
            return self._state_index_for_request(layout, req_idx)
        return int(history_indices[req_idx])


    def _single_state_index_tensor(
        self,
        state_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.tensor([state_idx], dtype=torch.int32, device=device)

    def _use_hsd_persistent_state_sidecar(self) -> bool:
        return bool(getattr(self, "hsd_use_persistent_state_sidecar", False))

    def _native_conv_cache_for_kernel(self) -> torch.Tensor | None:
        if not hasattr(self, "kv_cache") or len(self.kv_cache) == 0:
            return None
        conv_cache = self.kv_cache[0]
        if conv_cache.numel() == 0:
            return None
        if is_conv_state_dim_first():
            return conv_cache
        return conv_cache.transpose(-1, -2)

    def _conv_cache_for_kernel(
        self,
        min_rows: int = 0,
    ) -> torch.Tensor | None:
        conv_cache = self._native_conv_cache_for_kernel()
        if conv_cache is None or not self._use_hsd_persistent_state_sidecar():
            return conv_cache
        conv_shape = tuple(int(dim) for dim in self.get_state_shape()[0])
        if not is_conv_state_dim_first():
            conv_shape = (conv_shape[1], conv_shape[0])
        conv_dtype = self.get_state_dtype()[0]
        if (
            self.hsd_use_compact_state_slots
            and int(min_rows) > self.hsd_persistent_state_sidecar_rows
        ):
            raise RuntimeError(
                "hsd_compact_state_slot_out_of_range: "
                f"required_rows={int(min_rows)}, "
                f"capacity={self.hsd_persistent_state_sidecar_rows}."
            )
        target_rows = (
            self.hsd_persistent_state_sidecar_rows
            if self.hsd_use_compact_state_slots
            else min(
                int(conv_cache.shape[0]),
                max(self.hsd_persistent_state_sidecar_rows, int(min_rows)),
            )
        )
        sidecar = self.hsd_persistent_state_sidecar.get("conv_state")
        if (
            not torch.is_tensor(sidecar)
            or int(sidecar.shape[0]) < target_rows
            or tuple(sidecar.shape[1:]) != conv_shape
            or sidecar.dtype != conv_dtype
            or sidecar.device != conv_cache.device
        ):
            new_sidecar = torch.zeros(
                target_rows,
                *conv_shape,
                dtype=conv_dtype,
                device=conv_cache.device,
            )
            if (
                torch.is_tensor(sidecar)
                and tuple(sidecar.shape[1:]) == conv_shape
            ):
                copy_rows = min(int(sidecar.shape[0]), target_rows)
                new_sidecar[:copy_rows].copy_(sidecar[:copy_rows])
            sidecar = new_sidecar
            self.hsd_persistent_state_sidecar["conv_state"] = sidecar
            if self.hsd_use_compact_state_slots:
                self._record_hsd_varlen_event(
                    "hsd_compact_state_sidecar_allocate",
                    component="conv_state",
                    rows=int(sidecar.shape[0]),
                    native_rows=int(conv_cache.shape[0]),
                    bytes=int(sidecar.numel() * sidecar.element_size()),
                )
        return sidecar

    def _conv_state_for_request(self, state_idx: int) -> torch.Tensor | None:
        conv_cache = self._conv_cache_for_kernel(state_idx + 1)
        if conv_cache is None or conv_cache.numel() == 0:
            return None
        if (
            not _is_valid_hsd_state_index(state_idx)
            or state_idx >= conv_cache.shape[0]
        ):
            return None
        return conv_cache[state_idx : state_idx + 1]

    def _store_conv_state(self, state_idx: int, conv_state: torch.Tensor) -> None:
        conv_cache = self._conv_cache_for_kernel(state_idx + 1)
        if (
            conv_cache is None
            or conv_cache.numel() == 0
            or not _is_valid_hsd_state_index(state_idx)
            or state_idx >= conv_cache.shape[0]
        ):
            return
        value = conv_state.squeeze(0).to(dtype=conv_cache.dtype)
        conv_cache[state_idx].copy_(value)

    def _recurrent_state_for_request(
        self,
        state_idx: int,
        past_text_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        local_v_heads = self.num_v_heads // self.tp_size
        state = torch.zeros(
            1,
            local_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            dtype=self.get_state_dtype()[1],
            device=device,
        )
        ssm_cache = self._ssm_cache_for_kernel(state_idx + 1)
        if (
            past_text_len <= 0
            or ssm_cache is None
            or not _is_valid_hsd_state_index(state_idx)
            or state_idx >= ssm_cache.shape[0]
        ):
            return state
        return ssm_cache[state_idx : state_idx + 1].transpose(
            -1, -2
        ).contiguous()

    def _native_ssm_cache_for_kernel(self) -> torch.Tensor | None:
        if not hasattr(self, "kv_cache") or len(self.kv_cache) < 2:
            return None
        ssm_cache = self.kv_cache[1]
        if ssm_cache.numel() == 0:
            return None
        return ssm_cache

    def _ssm_cache_for_kernel(
        self,
        min_rows: int = 0,
    ) -> torch.Tensor | None:
        ssm_cache = self._native_ssm_cache_for_kernel()
        if ssm_cache is None or not self._use_hsd_persistent_state_sidecar():
            return ssm_cache
        recurrent_shape = tuple(int(dim) for dim in self.get_state_shape()[1])
        recurrent_dtype = self.get_state_dtype()[1]
        if (
            self.hsd_use_compact_state_slots
            and int(min_rows) > self.hsd_persistent_state_sidecar_rows
        ):
            raise RuntimeError(
                "hsd_compact_state_slot_out_of_range: "
                f"required_rows={int(min_rows)}, "
                f"capacity={self.hsd_persistent_state_sidecar_rows}."
            )
        target_rows = (
            self.hsd_persistent_state_sidecar_rows
            if self.hsd_use_compact_state_slots
            else min(
                int(ssm_cache.shape[0]),
                max(self.hsd_persistent_state_sidecar_rows, int(min_rows)),
            )
        )
        sidecar = self.hsd_persistent_state_sidecar.get("recurrent_state")
        if (
            not torch.is_tensor(sidecar)
            or int(sidecar.shape[0]) < target_rows
            or tuple(sidecar.shape[1:]) != recurrent_shape
            or sidecar.dtype != recurrent_dtype
            or sidecar.device != ssm_cache.device
        ):
            new_sidecar = torch.zeros(
                target_rows,
                *recurrent_shape,
                dtype=recurrent_dtype,
                device=ssm_cache.device,
            )
            if (
                torch.is_tensor(sidecar)
                and tuple(sidecar.shape[1:]) == recurrent_shape
            ):
                copy_rows = min(int(sidecar.shape[0]), target_rows)
                new_sidecar[:copy_rows].copy_(sidecar[:copy_rows])
            sidecar = new_sidecar
            self.hsd_persistent_state_sidecar["recurrent_state"] = sidecar
            if self.hsd_use_compact_state_slots:
                self._record_hsd_varlen_event(
                    "hsd_compact_state_sidecar_allocate",
                    component="recurrent_state",
                    rows=int(sidecar.shape[0]),
                    native_rows=int(ssm_cache.shape[0]),
                    bytes=int(sidecar.numel() * sidecar.element_size()),
                )
        return sidecar

    def _store_recurrent_state(self, state_idx: int, recurrent_state: torch.Tensor) -> None:
        ssm_cache = self._ssm_cache_for_kernel(state_idx + 1)
        if (
            ssm_cache is None
            or ssm_cache.numel() == 0
            or not _is_valid_hsd_state_index(state_idx)
            or state_idx >= ssm_cache.shape[0]
        ):
            return
        value = recurrent_state.squeeze(0).transpose(-1, -2).contiguous()
        ssm_cache[state_idx].copy_(value.to(dtype=ssm_cache.dtype))

    def _store_recurrent_state_batched(
        self,
        state_indices: torch.Tensor,
        recurrent_states: torch.Tensor,
    ) -> bool:
        if _env_truthy("VLLM_HSD_DISABLE_BATCHED_STATE_STORE"):
            return False
        required_rows = (
            int(state_indices.max().item()) + 1
            if int(state_indices.numel()) > 0
            else 0
        )
        ssm_cache = self._ssm_cache_for_kernel(required_rows)
        if (
            ssm_cache is None
            or ssm_cache.numel() == 0
            or int(state_indices.numel()) == 0
        ):
            return False
        if recurrent_states.ndim == 5 and int(recurrent_states.shape[1]) == 1:
            recurrent_states = recurrent_states[:, 0]
        if recurrent_states.ndim != 4:
            return False
        if int(recurrent_states.shape[0]) != int(state_indices.numel()):
            return False
        indices = state_indices.to(device=ssm_cache.device, dtype=torch.long)
        values = recurrent_states.transpose(-1, -2).contiguous().to(
            device=ssm_cache.device,
            dtype=ssm_cache.dtype,
        )
        ssm_cache.index_copy_(0, indices, values)
        self._record_hsd_varlen_event(
            "batched_recurrent_state_store_hit",
            requests=int(state_indices.numel()),
        )
        return True

    def _stored_recurrent_state_for_request(
        self,
        state_idx: int,
    ) -> torch.Tensor | None:
        ssm_cache = self._ssm_cache_for_kernel(state_idx + 1)
        if (
            ssm_cache is None
            or ssm_cache.numel() == 0
            or not _is_valid_hsd_state_index(state_idx)
            or state_idx >= ssm_cache.shape[0]
        ):
            return None
        return ssm_cache[state_idx : state_idx + 1].transpose(
            -1, -2
        ).contiguous()


    def _is_hsd_moe_model(self) -> bool:
        return str(getattr(self.config, "model_type", "")) == (
            "safin1_35b_a3b_text"
        )

    def _use_deterministic_varlen_kkt(self) -> bool:
        if "VLLM_HSD_DETERMINISTIC_VARLEN_KKT" in os.environ:
            return _env_truthy("VLLM_HSD_DETERMINISTIC_VARLEN_KKT")
        return self._is_hsd_moe_model()

    def _use_deterministic_prefill_autotune(self) -> bool:
        return _env_truthy("VLLM_HSD_DETERMINISTIC_PREFILL_AUTOTUNE")

    def _use_deterministic_anchor_read(self, anchor_count: int) -> bool:
        if anchor_count <= 0:
            return False
        if "VLLM_HSD_DETERMINISTIC_ANCHOR_READ" in os.environ:
            return _env_truthy("VLLM_HSD_DETERMINISTIC_ANCHOR_READ")
        return self._is_hsd_moe_model()

    def _use_request_loop_conv(self, req_count: int) -> bool:
        if self._use_reference_conv_explicit_state_prefix():
            return False
        if _env_truthy("VLLM_HSD_DETERMINISTIC_VARLEN_CONV"):
            return True
        return int(req_count) > 1 and _env_truthy("VLLM_HSD_DISABLE_BATCHED_CONV")

    def _use_vllm_varlen_reference_kernel(self) -> bool:
        override = os.environ.get("VLLM_HSD_VARLEN_VLLM_REFERENCE_KERNEL")
        if override is not None:
            return override.strip().lower() not in {
                "0",
                "false",
                "no",
                "n",
                "off",
            }
        # The reference recurrence with the fixed-order KKT kernel is exact
        # against segments for MoE and remains the accepted dense path.
        return True

    def _use_reference_conv_prefill(self) -> bool:
        if _env_truthy("VLLM_HSD_DISABLE_REFERENCE_CONV_PREFILL"):
            return False
        override = os.environ.get("VLLM_HSD_ENABLE_REFERENCE_CONV_PREFILL")
        if override is not None:
            return override.strip().lower() not in {
                "0",
                "false",
                "no",
                "n",
                "off",
            }
        # Preserve reference prefill convolution arithmetic for the MoE model.
        # Decode uses the stateful update kernel after prefill.
        return str(getattr(self.config, "model_type", "")) == "safin1_35b_a3b_text"

    def _use_reference_conv_explicit_state_prefix(self) -> bool:
        override = os.environ.get(
            "VLLM_HSD_REFERENCE_CONV_EXPLICIT_STATE_PREFIX"
        )
        if override is not None:
            return override.strip().lower() not in {
                "0",
                "false",
                "no",
                "n",
                "off",
            }
        # causal-conv's initial_states branch has slightly different BF16
        # arithmetic from one-shot prefill. Prefix the restored raw state and
        # reuse the one-shot operator; MoE top-k can amplify a one-ulp conv
        # difference. Keep dense checkpoints unchanged.
        return str(getattr(self.config, "model_type", "")) == (
            "safin1_35b_a3b_text"
        )

    def _store_batched_conv_states(
        self,
        *,
        state_indices: list[int],
        token_offsets: list[int],
        conv_state_snapshot: torch.Tensor,
        token_inputs: torch.Tensor,
        past_text_lens: list[int] | None = None,
    ) -> None:
        for req_pos, state_idx in enumerate(state_indices):
            token_begin = token_offsets[req_pos]
            token_end = token_offsets[req_pos + 1]
            has_initial_state = (
                True
                if past_text_lens is None
                else int(past_text_lens[req_pos]) > 0
            )
            if _env_truthy("VLLM_HSD_PREFIX_CACHE_DEBUG"):
                self._record_hsd_varlen_event(
                    "hsd_prefix_conv_state_advance",
                    state_idx=int(state_idx),
                    past_text_len=(
                        None
                        if past_text_lens is None
                        else int(past_text_lens[req_pos])
                    ),
                    token_count=int(token_end - token_begin),
                    conv_initial_state_used=bool(has_initial_state),
                    source="batched_prefill_store",
                )
            self._store_conv_state(
                state_idx,
                _advance_hsd_conv_state(
                    conv_state_snapshot[req_pos : req_pos + 1],
                    token_inputs[token_begin:token_end],
                    has_initial_state=has_initial_state,
                ),
            )

    def _store_batched_conv_states_decode_tensor(
        self,
        *,
        state_indices: torch.Tensor,
        conv_state_snapshot: torch.Tensor,
        token_inputs: torch.Tensor,
    ) -> bool:
        if not hasattr(self, "kv_cache") or len(self.kv_cache) == 0:
            return False
        conv_cache = self.kv_cache[0]
        if conv_cache.numel() == 0 or int(state_indices.numel()) == 0:
            return False
        if conv_state_snapshot.ndim != 3 or token_inputs.ndim != 2:
            return False
        if int(conv_state_snapshot.shape[0]) != int(state_indices.numel()):
            return False
        if int(token_inputs.shape[0]) != int(state_indices.numel()):
            return False
        next_state = self._expected_batched_decode_conv_state_tensor(
            conv_state_snapshot=conv_state_snapshot,
            token_inputs=token_inputs,
        )
        if next_state is None:
            return False
        values = next_state.to(device=conv_cache.device, dtype=conv_cache.dtype)
        indices = state_indices.to(device=conv_cache.device, dtype=torch.long)
        if is_conv_state_dim_first():
            conv_cache.index_copy_(0, indices, values)
        else:
            conv_cache.index_copy_(0, indices, values.transpose(-1, -2))
        self._record_hsd_varlen_event(
            "batched_conv_state_store_hit",
            requests=int(state_indices.numel()),
        )
        return True

    def _expected_batched_decode_conv_state_tensor(
        self,
        *,
        conv_state_snapshot: torch.Tensor,
        token_inputs: torch.Tensor,
    ) -> torch.Tensor | None:
        if conv_state_snapshot.ndim != 3 or token_inputs.ndim != 2:
            return None
        if int(conv_state_snapshot.shape[0]) != int(token_inputs.shape[0]):
            return None
        state_len = int(conv_state_snapshot.shape[-1])
        if state_len <= 0:
            return None
        token_values = token_inputs.to(
            device=conv_state_snapshot.device,
            dtype=conv_state_snapshot.dtype,
        ).unsqueeze(-1)
        if state_len == 1:
            return token_values
        return torch.cat([conv_state_snapshot[:, :, 1:], token_values], dim=-1)

    def _validate_native_decode_conv_state_update(
        self,
        *,
        state_indices: torch.Tensor,
        conv_state_snapshot: torch.Tensor,
        token_inputs: torch.Tensor,
    ) -> None:
        if not _env_truthy("VLLM_HSD_VALIDATE_NATIVE_CONV_STATE_UPDATE"):
            return
        expected = self._expected_batched_decode_conv_state_tensor(
            conv_state_snapshot=conv_state_snapshot,
            token_inputs=token_inputs,
        )
        if expected is None:
            self._record_hsd_varlen_event(
                "native_conv_state_update_validate",
                requests=int(state_indices.numel()),
                valid=0,
                reason="bad_shape",
            )
            return
        if not hasattr(self, "kv_cache") or len(self.kv_cache) == 0:
            self._record_hsd_varlen_event(
                "native_conv_state_update_validate",
                requests=int(state_indices.numel()),
                valid=0,
                reason="missing_conv_cache",
            )
            return
        conv_cache = self.kv_cache[0]
        if conv_cache.numel() == 0 or int(state_indices.numel()) == 0:
            self._record_hsd_varlen_event(
                "native_conv_state_update_validate",
                requests=int(state_indices.numel()),
                valid=0,
                reason="empty_conv_cache",
            )
            return
        indices = state_indices.to(device=conv_cache.device, dtype=torch.long)
        native = conv_cache.index_select(0, indices)
        if not is_conv_state_dim_first():
            native = native.transpose(-1, -2)
        native = native.to(device=expected.device, dtype=expected.dtype)
        diff = (native - expected).abs()
        self._record_hsd_varlen_event(
            "native_conv_state_update_validate",
            requests=int(state_indices.numel()),
            valid=1,
            max_abs=float(diff.max().item()) if diff.numel() else 0.0,
            mean_abs=float(diff.mean().item()) if diff.numel() else 0.0,
        )

    def _run_reference_varlen_conv_prefill(
        self,
        *,
        packed_inputs: torch.Tensor,
        token_offsets: list[int],
        conv_state_snapshot: torch.Tensor,
        past_text_lens: list[int],
        conv_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run external causal-conv once for a packed varlen request batch.

        The external operator has no per-sequence initial-state argument for a
        packed stream. Prefix each logical sequence with its three raw conv
        state tokens and use ``seq_idx`` to reset causality between requests.
        The prefixed outputs are discarded. This keeps full-prefill and
        prefix-hit suffix prefill on the same arithmetic implementation while
        avoiding a kernel launch per request.
        """
        if reference_causal_conv1d_fn is None:
            raise RuntimeError("causal-conv1d reference operator is unavailable")
        request_count = len(past_text_lens)
        if len(token_offsets) != request_count + 1:
            raise ValueError(
                "HSD reference varlen conv token offsets do not match requests: "
                f"offsets={len(token_offsets)}, requests={request_count}."
            )
        if int(conv_state_snapshot.shape[0]) != request_count:
            raise ValueError(
                "HSD reference varlen conv state rows do not match requests: "
                f"state_rows={int(conv_state_snapshot.shape[0])}, "
                f"requests={request_count}."
            )

        state_len = int(conv_state_snapshot.shape[-1])
        sequence_inputs: list[torch.Tensor] = []
        sequence_ids: list[torch.Tensor] = []
        sequence_lengths: list[int] = []
        for req_pos, past_text_len in enumerate(past_text_lens):
            token_begin = int(token_offsets[req_pos])
            token_end = int(token_offsets[req_pos + 1])
            token_values = packed_inputs[token_begin:token_end]
            prefix_state = conv_state_snapshot[req_pos].transpose(0, 1)
            if int(past_text_len) <= 0:
                prefix_state = torch.zeros_like(prefix_state)
            sequence_input = torch.cat([prefix_state, token_values], dim=0)
            sequence_inputs.append(sequence_input)
            sequence_lengths.append(int(token_values.shape[0]))
            sequence_ids.append(
                torch.full(
                    (int(sequence_input.shape[0]),),
                    req_pos,
                    dtype=torch.int32,
                    device=packed_inputs.device,
                )
            )

        packed_with_state = torch.cat(sequence_inputs, dim=0)
        seq_idx = torch.cat(sequence_ids, dim=0).view(1, -1)
        conv_output = reference_causal_conv1d_fn(
            # Keep channels stride-1; causal-conv requires this layout when
            # seq_idx or initial_states is supplied.
            x=packed_with_state.unsqueeze(0).transpose(1, 2),
            weight=conv_weights,
            bias=self.conv1d.bias,
            seq_idx=seq_idx,
            activation=self.activation,
        ).squeeze(0).transpose(0, 1)

        output_chunks: list[torch.Tensor] = []
        cursor = 0
        for text_len in sequence_lengths:
            output_chunks.append(
                conv_output[cursor + state_len : cursor + state_len + text_len]
            )
            cursor += state_len + text_len
        return torch.cat(output_chunks, dim=0).contiguous()

    def _run_reference_single_conv_prefill_with_explicit_state(
        self,
        *,
        token_inputs: torch.Tensor,
        conv_state_snapshot: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run a restored suffix through the one-shot causal-conv branch."""
        if reference_causal_conv1d_fn is None:
            raise RuntimeError(
                "causal-conv1d reference operator is unavailable"
            )
        if token_inputs.ndim != 2 or conv_state_snapshot.ndim != 3:
            raise ValueError(
                "HSD cached-state conv expects [tokens, dim] inputs and "
                f"[1, dim, state] cache, got {tuple(token_inputs.shape)} and "
                f"{tuple(conv_state_snapshot.shape)}."
            )
        if int(conv_state_snapshot.shape[0]) != 1:
            raise ValueError(
                "HSD cached-state single-request conv requires one cache row, "
                f"got {int(conv_state_snapshot.shape[0])}."
            )
        # Prefix the suffix with the exact raw-input window and call the same
        # no-initial-state operator branch as one-shot prefill. Outputs for the
        # synthetic prefix are discarded; canonical cache writeback remains in
        # _store_batched_conv_states.
        state_len = int(conv_state_snapshot.shape[-1])
        conv_input = torch.cat(
            [conv_state_snapshot[0].transpose(0, 1), token_inputs],
            dim=0,
        )
        output = reference_causal_conv1d_fn(
            x=conv_input.transpose(0, 1).unsqueeze(0).contiguous(),
            weight=conv_weights,
            bias=self.conv1d.bias,
            activation=self.activation,
        )
        return (
            output.squeeze(0)
            .transpose(0, 1)[state_len:]
            .contiguous()
        )

    def _run_text_conv(
        self,
        mixed_qkv_text: torch.Tensor,
        state_idx: int,
        past_text_len: int,
        decode_step: bool,
    ) -> torch.Tensor:
        # mixed_qkv_text: [1, T, C]
        text_len = int(mixed_qkv_text.shape[1])
        conv_dim = int(mixed_qkv_text.shape[-1])
        conv_cache = (
            self._conv_cache_for_kernel(state_idx + 1)
            if _is_valid_hsd_state_index(state_idx)
            else None
        )
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        state_indices = self._single_state_index_tensor(
            state_idx, mixed_qkv_text.device
        )
        if conv_cache is not None and decode_step:
            if text_len != 1:
                raise NotImplementedError(
                    "Safin-1 HSD vLLM decode expects one text token "
                    "per request."
                )
            conv_state_snapshot = conv_cache.index_select(
                0,
                state_indices.long(),
            ).detach().clone()
            mixed_qkv_token = mixed_qkv_text.squeeze(0).contiguous()
            # causal_conv1d_update writes its output into x in-place; keep the
            # pre-conv projection for the conv-state cache update below.
            token_inputs_for_state = mixed_qkv_token.detach().clone()
            mixed_qkv = causal_conv1d_update(
                mixed_qkv_token,
                conv_cache,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices,
                validate_data=False,
            )
            self._store_conv_state(
                state_idx,
                _advance_hsd_conv_state(
                    conv_state_snapshot,
                    token_inputs_for_state,
                    has_initial_state=past_text_len > 0,
                ),
            )
            return mixed_qkv.view(1, text_len, conv_dim)

        if conv_cache is not None:
            mixed_qkv_t = mixed_qkv_text.squeeze(0).transpose(0, 1)
            query_start_loc = torch.tensor(
                [0, text_len], dtype=torch.int32, device=mixed_qkv_text.device
            )
            has_initial_state = torch.tensor(
                [past_text_len > 0], dtype=torch.bool, device=mixed_qkv_text.device
            )
            conv_state_snapshot = conv_cache.index_select(
                0,
                state_indices.long(),
            ).detach().clone()
            if (
                reference_causal_conv1d_fn is not None
                and self._use_reference_conv_prefill()
            ):
                reference_initial_state = None
                if past_text_len > 0:
                    reference_initial_state = (
                        conv_state_snapshot.transpose(1, 2)
                        .contiguous()
                        .transpose(1, 2)
                    )
                mixed_qkv = reference_causal_conv1d_fn(
                    x=mixed_qkv_text.contiguous().transpose(1, 2),
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    initial_states=reference_initial_state,
                    activation=self.activation,
                )
                self._store_conv_state(
                    state_idx,
                    _advance_hsd_conv_state(
                        conv_state_snapshot,
                        mixed_qkv_text.squeeze(0).contiguous(),
                        has_initial_state=past_text_len > 0,
                    ),
                )
                self._record_hsd_varlen_event(
                    "reference_conv_prefill_hit",
                    requests=1,
                    text_tokens=text_len,
                    has_initial_state=bool(past_text_len > 0),
                    source="request_loop",
                )
                return mixed_qkv.transpose(1, 2).contiguous()
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv_t,
                conv_weights,
                self.conv1d.bias,
                conv_cache,
                query_start_loc=query_start_loc,
                cache_indices=state_indices,
                has_initial_state=has_initial_state,
                activation=self.activation,
            )
            self._store_conv_state(
                state_idx,
                _advance_hsd_conv_state(
                    conv_state_snapshot,
                    mixed_qkv_text.squeeze(0).contiguous(),
                    has_initial_state=past_text_len > 0,
                ),
            )
            return mixed_qkv.transpose(0, 1).contiguous().view(1, text_len, conv_dim)

        conv_state = self._conv_state_for_request(state_idx)
        if past_text_len <= 0 or conv_state is None:
            conv_prefix = mixed_qkv_text.new_zeros(
                1, conv_dim, self.conv_kernel_size - 1
            )
        else:
            conv_prefix = conv_state.to(device=mixed_qkv_text.device, dtype=mixed_qkv_text.dtype)
        conv_input = torch.cat([conv_prefix, mixed_qkv_text.transpose(1, 2)], dim=-1)
        new_state = conv_input[:, :, -(self.conv_kernel_size - 1) :].detach()
        self._store_conv_state(state_idx, new_state)

        weight = self.conv1d.weight.view(
            self.conv1d.weight.size(0), 1, self.conv1d.weight.size(-1)
        )
        conv_out = F.conv1d(
            conv_input,
            weight.to(dtype=mixed_qkv_text.dtype),
            bias=self.conv1d.bias,
            groups=conv_dim,
        )
        conv_out = F.silu(conv_out[:, :, -text_len:])
        return conv_out.transpose(1, 2).contiguous()


    def _post_conv_prep_hsd(
        self,
        mixed_qkv_text: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        local_k_heads: int,
        local_v_heads: int,
        local_key_dim: int,
        local_value_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.hsd_post_conv_backend == "fused":
            return fused_post_conv_prep(
                conv_output=mixed_qkv_text,
                a=a.contiguous(),
                b=b.contiguous(),
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=local_k_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )

        text_len = int(mixed_qkv_text.shape[0])
        query, key, value = torch.split(
            mixed_qkv_text,
            [local_key_dim, local_key_dim, local_value_dim],
            dim=-1,
        )
        query = query.reshape(text_len, local_k_heads, self.head_k_dim)
        key = key.reshape(text_len, local_k_heads, self.head_k_dim)
        value = value.reshape(text_len, local_v_heads, self.head_v_dim)
        query = _l2norm(query.contiguous())
        key = _l2norm(key.contiguous())
        beta = b.contiguous().sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        return query, key, value, g, beta

    def _build_history_read_query(
        self,
        *,
        local_query: torch.Tensor,
        text_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Build the query used only by the global history read.

        The recurrent GDN continues to consume ``local_query``.  Dense-NewOps
        checkpoints optionally add a hidden-state LoRA residual and normalize
        again for the history read, matching the Transformers reference model.
        """
        return local_query


    def _record_hsd_decode_prepare_counter(self, name: str, value: int = 1) -> None:
        self.hsd_decode_prepare_counters[name] = (
            self.hsd_decode_prepare_counters.get(name, 0) + int(value)
        )

    def _record_hsd_decode_prepare_fallback(self, reason: str) -> None:
        self.hsd_decode_prepare_fallback_counts[reason] = (
            self.hsd_decode_prepare_fallback_counts.get(reason, 0) + 1
        )
        self._record_hsd_varlen_event("decode_prepare_fallback", reason=reason)
        if _env_truthy("VLLM_HSD_VARLEN_LOG_FALLBACK"):
            logger.info(
                "HSD decode prepare fallback layer=%s reason=%s",
                int(self.layer_idx) + 1,
                reason,
            )


    def _replace_persistent_anchor_state(
        self,
        *,
        anchor_q: torch.Tensor,
        anchor_o: torch.Tensor,
        anchor_states: torch.Tensor,
        persistent_anchor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace boundary-zero snapshots after recurrence and pre-read Q/K."""

        memory_mask = persistent_anchor_mask.to(
            device=anchor_states.device,
            dtype=torch.bool,
        )
        if not bool(memory_mask.any().item()):
            return anchor_o, anchor_states
        state = self.persistent_memory_state
        if state is None:
            # Layers with no causal path to a global read intentionally retain
            # their ordinary anchor snapshot, matching the HF target rule.
            return anchor_o, anchor_states
        if tuple(memory_mask.shape) != tuple(anchor_states.shape[:2]):
            raise ValueError(
                "persistent anchor mask shape mismatch: "
                f"got {tuple(memory_mask.shape)}, expected "
                f"{tuple(anchor_states.shape[:2])}."
            )
        counts = memory_mask.long().sum(dim=1)
        if not bool((counts == 1).all().item()):
            raise ValueError(
                "vLLM HSD persistent memory requires exactly one boundary-zero "
                f"anchor per fresh request, got {counts.detach().cpu().tolist()}."
            )

        state = state.to(device=anchor_states.device)
        state_expanded = state.view(1, 1, *state.shape).expand(
            anchor_states.shape[0],
            anchor_states.shape[1],
            *state.shape,
        )
        state_mask = memory_mask.view(*memory_mask.shape, 1, 1, 1)
        anchor_states = torch.where(
            state_mask,
            state_expanded.to(anchor_states.dtype),
            anchor_states,
        )
        memory_o = (
            torch.einsum(
                "bmhk,bmhkv->bmhv",
                anchor_q.to(state_expanded.dtype),
                state_expanded,
            )
            * (self.head_k_dim**-0.5)
        )
        anchor_o = torch.where(
            memory_mask.view(*memory_mask.shape, 1, 1),
            memory_o.to(anchor_o.dtype),
            anchor_o,
        )
        self._trace_internal("persistent_memory_state", state)
        self._trace_internal("persistent_memory_anchor_o", memory_o)
        return anchor_o, anchor_states

    def _restore_persistent_varlen_anchor_states(
        self,
        *,
        dynamic_anchor_states: torch.Tensor,
        anchor_q: torch.Tensor,
        anchor_offsets: list[int],
        kernel_anchor_offsets: list[int],
        memory_counts: list[int],
    ) -> torch.Tensor:
        """Restore boundary-zero rows omitted from the recurrence kernel.

        The grouped varlen recurrence sees dynamic boundaries only.  This
        helper reconstructs the full per-request cache order and performs the
        same post-recurrence memory replacement used by the segments path.
        ``anchor_q`` has already been projected from anchor hidden states,
        so replacing cache state here cannot change the current layer's
        pre-read anchor key.
        """

        request_count = len(memory_counts)
        if len(anchor_offsets) != request_count + 1 or len(
            kernel_anchor_offsets
        ) != request_count + 1:
            raise ValueError("Invalid HSD varlen persistent-memory offsets.")
        if any(count not in (0, 1) for count in memory_counts):
            raise ValueError(
                "Each HSD varlen request may contain at most one memory row."
            )
        total_anchor = int(anchor_offsets[-1])
        total_dynamic = int(kernel_anchor_offsets[-1])
        if int(dynamic_anchor_states.shape[0]) != total_dynamic:
            raise RuntimeError(
                "HSD varlen dynamic anchor count mismatch before persistent "
                f"memory insertion: got={int(dynamic_anchor_states.shape[0])}, "
                f"expected={total_dynamic}."
            )
        if int(anchor_q.shape[0]) != total_anchor:
            raise RuntimeError(
                "HSD varlen anchor-Q count mismatch: "
                f"got={int(anchor_q.shape[0])}, expected={total_anchor}."
            )
        if not any(memory_counts):
            if total_anchor != total_dynamic:
                raise RuntimeError(
                    "HSD varlen offsets disagree without persistent memory."
                )
            return dynamic_anchor_states

        full_anchor_states = dynamic_anchor_states.new_zeros(
            total_anchor,
            *dynamic_anchor_states.shape[1:],
        )
        for req_idx, memory_count in enumerate(memory_counts):
            full_begin = int(anchor_offsets[req_idx])
            full_end = int(anchor_offsets[req_idx + 1])
            kernel_begin = int(kernel_anchor_offsets[req_idx])
            kernel_end = int(kernel_anchor_offsets[req_idx + 1])
            if full_end - full_begin != kernel_end - kernel_begin + memory_count:
                raise RuntimeError(
                    "HSD varlen per-request anchor offsets disagree with the "
                    "persistent-memory count."
                )
            if kernel_end > kernel_begin:
                full_anchor_states[
                    full_begin + memory_count : full_end
                ].copy_(dynamic_anchor_states[kernel_begin:kernel_end])
            if not memory_count:
                continue
            req_states = full_anchor_states[full_begin:full_end].unsqueeze(0)
            req_q = anchor_q[full_begin:full_end].unsqueeze(0)
            req_o = req_q.new_zeros(
                1,
                full_end - full_begin,
                int(req_states.shape[-3]),
                int(req_states.shape[-1]),
            )
            req_mask = torch.zeros(
                1,
                full_end - full_begin,
                dtype=torch.bool,
                device=req_states.device,
            )
            req_mask[:, 0] = True
            _, replaced_states = self._replace_persistent_anchor_state(
                anchor_q=req_q,
                anchor_o=req_o,
                anchor_states=req_states,
                persistent_anchor_mask=req_mask,
            )
            full_anchor_states[full_begin:full_end].copy_(
                replaced_states.squeeze(0)
            )
        return full_anchor_states

    def _history_gate_tensor(self, device: torch.device) -> torch.Tensor:
        if self.hsd_history_gate_logit is None:
            return torch.ones((), device=device, dtype=torch.float32)
        return (
            torch.sigmoid(self.hsd_history_gate_logit.float())
            * self.hsd_learnable_history_scale_max
        ).to(device=device)

    def _history_scale_tensor(self, device: torch.device) -> torch.Tensor:
        if _env_truthy("VLLM_HSD_DISABLE_HISTORY_OUTPUT"):
            return torch.zeros((), device=device, dtype=torch.float32)
        if (
            self.hsd_history_gate_logit is None
            and not _env_truthy("VLLM_HSD_DISABLE_STATIC_HISTORY_SCALE_BUFFER")
        ):
            return self.hsd_history_scale_static.to(device=device)
        return self._history_gate_tensor(device) * float(self.hsd_history_scale)

    def _apply_history_out_rmsnorm(self, history_out: torch.Tensor) -> torch.Tensor:
        if self.hsd_history_out_rmsnorm_weight is None:
            return history_out
        rms = history_out.float().pow(2).mean(dim=-1, keepdim=True).add(self.hsd_history_out_rmsnorm_eps).rsqrt()
        weight = self.hsd_history_out_rmsnorm_weight.float().view(1, 1, 1, -1)
        return (history_out.float() * rms * weight).to(history_out.dtype)

    def _merge_anchor_cache(
        self,
        state_idx: int,
        past_text_len: int,
        new_anchor_states: torch.Tensor,
        new_anchor_keys: torch.Tensor | None,
        new_anchor_boundaries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if not _is_valid_hsd_state_index(state_idx):
            return new_anchor_states, new_anchor_keys, new_anchor_boundaries
        if past_text_len <= 0:
            self.hsd_anchor_cache.pop(state_idx, None)
            page_cache = getattr(self, "hsd_anchor_page_cache", None)
            if page_cache is not None:
                page_cache.clear({state_idx})
        old = self.hsd_anchor_cache.get(state_idx)
        if old is None or int(old["states"].shape[1]) == 0:
            states, keys, boundaries = (
                new_anchor_states,
                new_anchor_keys,
                new_anchor_boundaries,
            )
        elif int(new_anchor_states.shape[1]) == 0:
            states = old["states"].to(new_anchor_states.device)
            keys = old.get("keys")
            keys = keys.to(new_anchor_states.device) if keys is not None else None
            boundaries = old["boundaries"].to(new_anchor_boundaries.device)
        else:
            states = torch.cat(
                [old["states"].to(new_anchor_states.device), new_anchor_states],
                dim=1,
            )
            old_keys = old.get("keys")
            if old_keys is None or new_anchor_keys is None:
                keys = None
            else:
                keys = torch.cat([old_keys.to(new_anchor_keys.device), new_anchor_keys], dim=1)
            boundaries = torch.cat(
                [old["boundaries"].to(new_anchor_boundaries.device), new_anchor_boundaries],
                dim=0,
            )

        self.hsd_anchor_cache[state_idx] = {
            "states": states.detach(),
            "boundaries": boundaries.detach(),
        }
        if keys is not None:
            self.hsd_anchor_cache[state_idx]["keys"] = keys.detach()
        return states, keys, boundaries

    def _append_new_anchor_to_history_cache(
        self,
        *,
        state_idx: int,
        past_text_len: int,
        new_anchor_states: torch.Tensor,
        new_anchor_keys: torch.Tensor | None,
        new_anchor_boundaries: torch.Tensor,
        use_paged_history: bool,
    ):
        if use_paged_history:
            cache_backend = str(
                getattr(self, "hsd_history_cache_backend", "auto")
            ).strip().lower()
            page_cache = self._get_hsd_anchor_page_cache(
                allow_legacy_create=cache_backend == "legacy_side_paged"
            )
            fail_closed = self.hsd_persistent_memory_mode != "removed" and (
                cache_backend == "scheduler_paged"
                or _env_truthy("VLLM_HSD_ENABLE_GRAPH_V3")
                or bool(
                    page_cache is not None
                    and getattr(page_cache, "fixed_capacity", False)
                )
            )
            if page_cache is None:
                if fail_closed:
                    raise RuntimeError(
                        "Persistent HSD memory requires the scheduler-paged "
                        "history cache, but no page cache is available."
                    )
                return False, None, None
            if not self._migrate_legacy_anchor_cache_to_pages(state_idx):
                self.hsd_history_page_alloc_failures += 1
                if fail_closed:
                    raise RuntimeError(
                        "Persistent HSD memory cannot split one request's "
                        "history between legacy and scheduler-paged caches: "
                        f"legacy migration failed for state {state_idx}."
                    )
                return False, page_cache, None
            appended = (
                page_cache.try_append(
                    state_idx,
                    self._anchor_states_to_paged_cache(new_anchor_states),
                    new_anchor_keys,
                    new_anchor_boundaries,
                )
                if getattr(page_cache, "fixed_capacity", False)
                else True
            )
            if not appended:
                lengths_by_state = getattr(page_cache, "_lengths_by_state", None)
                page_table_by_state = getattr(
                    page_cache, "_page_table_by_state", None
                )
                self._record_hsd_varlen_event(
                    "hsd_history_page_append_fallback",
                    reason="capacity_or_metadata",
                    state_idx=int(state_idx),
                    anchor_tokens=int(new_anchor_states.shape[1]),
                    fixed_capacity=int(
                        bool(getattr(page_cache, "fixed_capacity", False))
                    ),
                    initialized=int(
                        bool(getattr(page_cache, "initialized", False))
                    ),
                    free_pages=len(getattr(page_cache, "free_pages", ())),
                    total_pages=int(getattr(page_cache, "total_pages", 0)),
                    metadata_state_rows=(
                        0
                        if lengths_by_state is None
                        else int(lengths_by_state.shape[0])
                    ),
                    metadata_page_columns=(
                        0
                        if page_table_by_state is None
                        else int(page_table_by_state.shape[1])
                    ),
                )
                self.hsd_history_page_alloc_failures += 1
                if fail_closed:
                    raise RuntimeError(
                        "Persistent HSD memory scheduler-paged append failed "
                        f"for state {state_idx}; refusing a split history."
                    )
                return False, page_cache, None
            if not getattr(page_cache, "fixed_capacity", False):
                page_cache.append(
                    state_idx,
                    self._anchor_states_to_paged_cache(new_anchor_states),
                    new_anchor_keys,
                    new_anchor_boundaries,
                )
            return True, page_cache, None

        return (
            False,
            None,
            self._merge_anchor_cache(
                state_idx=state_idx,
                past_text_len=past_text_len,
                new_anchor_states=new_anchor_states,
                new_anchor_keys=new_anchor_keys,
                new_anchor_boundaries=new_anchor_boundaries,
            ),
        )

    def _history_read(
        self,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        anchor_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_len: int,
        start_index: int,
        null_logits: torch.Tensor | None = None,
        persistent_memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, heads, _ = read_q.shape
        anchor_count = int(states.shape[1])
        value_dim = int(states.shape[-1])
        if anchor_count == 0:
            return read_q.new_zeros(batch_size, text_len, heads, value_dim)
        if null_logits is not None and tuple(null_logits.shape) != (
            batch_size,
            text_len,
            heads,
            1,
        ):
            raise RuntimeError(
                "HSD null read logits shape mismatch: "
                f"got {tuple(null_logits.shape)}, expected {(batch_size, text_len, heads, 1)}."
            )

        if persistent_memory_mask is None:
            persistent_memory_mask = (
                boundary_counts.to(device=states.device, dtype=torch.long) == 0
                if self.hsd_persistent_memory_mode != "removed"
                else torch.zeros(
                    anchor_count,
                    dtype=torch.bool,
                    device=states.device,
                )
            )
        if bool(persistent_memory_mask.any().item()):
            return self._persistent_history_read(
                read_q=read_q,
                router_q=router_q,
                states=states,
                anchor_k=anchor_k,
                boundary_counts=boundary_counts,
                text_len=text_len,
                start_index=start_index,
                null_logits=null_logits,
                persistent_memory_mask=persistent_memory_mask,
            )

        route_scale = (
            self.hsd_route_head_dim**-0.5
            if self.hsd_use_route_scale
            else 1.0
        )
        route_scale *= self.hsd_route_logit_scale
        if self.hsd_read_topk == 0:
            fused_global_read = backends._load_hsd_fused_global_read()
            fused_strict = bool(
                getattr(self.config, "hsd_fused_global_read_strict", False)
            )
            if fused_global_read is None and fused_strict:
                raise RuntimeError(
                    "Strict HSD fused global read could not be imported; "
                    "refusing the dense reference fallback."
                ) from backends._HSD_FUSED_GLOBAL_READ_IMPORT_ERROR
            if fused_global_read is not None:
                text_cu_seqlens = torch.tensor(
                    [0, text_len],
                    device=read_q.device,
                    dtype=torch.long,
                )
                anchor_cu_seqlens = torch.tensor(
                    [0, anchor_count],
                    device=read_q.device,
                    dtype=torch.long,
                )
                fused = fused_global_read(
                    read_q=read_q,
                    router_q=router_q,
                    states=states,
                    anchor_k=anchor_k,
                    boundary_counts=boundary_counts - int(start_index),
                    text_cu_seqlens=text_cu_seqlens,
                    anchor_cu_seqlens=anchor_cu_seqlens,
                    null_logits=null_logits,
                    route_scale=route_scale,
                    read_scale=self.head_k_dim**-0.5,
                    backend=getattr(
                        self.config, "hsd_fused_global_read_backend", None
                    ),
                    profile=getattr(
                        self.config, "hsd_fused_global_read_profile", None
                    ),
                    strict=getattr(
                        self.config, "hsd_fused_global_read_strict", None
                    ),
                )
                if fused is not None:
                    return fused
                if fused_strict:
                    raise RuntimeError(
                        "Strict HSD fused global read returned no result; "
                        "refusing the dense reference fallback."
                    )
        logits = self._route_logits(router_q, anchor_k, heads)
        logits = logits * route_scale
        token_index = torch.arange(
            start_index,
            start_index + text_len,
            dtype=torch.long,
            device=read_q.device,
        )
        valid = boundary_counts.to(read_q.device).view(1, 1, 1, anchor_count)
        valid = valid <= token_index.view(1, text_len, 1, 1)
        valid = valid.expand(batch_size, text_len, heads, anchor_count)

        topk = self.hsd_read_topk
        if 0 < topk < anchor_count:
            _, topk_idx = logits.masked_fill(~valid, float("-inf")).topk(
                k=topk, dim=-1
            )
            topk_mask = torch.zeros_like(valid)
            topk_mask.scatter_(-1, topk_idx, True)
            valid = valid & topk_mask

        if null_logits is not None:
            null_valid = torch.ones(
                batch_size,
                text_len,
                heads,
                1,
                dtype=torch.bool,
                device=valid.device,
            )
            logits_with_null = torch.cat([logits, null_logits.float()], dim=-1)
            valid_with_null = torch.cat([valid, null_valid], dim=-1)
            weights_with_null = _route_softmax_weights(
                logits_with_null,
                valid_with_null,
                dim=-1,
            )
            weights = weights_with_null[..., :anchor_count]
        else:
            weights = _route_softmax_weights(logits, valid, dim=-1)
        reads = torch.einsum(
            "bthk,bmhkv->bthmv",
            read_q.to(states.dtype),
            states,
        )
        reads = reads * (self.head_k_dim**-0.5)
        reads = reads.masked_fill(~valid.unsqueeze(-1), 0)
        return torch.einsum("bthm,bthmv->bthv", weights.to(reads.dtype), reads).to(
            read_q.dtype
        )

    def _persistent_history_read(
        self,
        *,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        anchor_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_len: int,
        start_index: int,
        null_logits: torch.Tensor | None,
        persistent_memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Read ``[memory, dynamic Top-K, null]`` with one FP32 softmax."""

        batch_size, _, heads, _ = read_q.shape
        anchor_count = int(states.shape[1])
        memory_mask = persistent_memory_mask.to(
            device=states.device,
            dtype=torch.bool,
        )
        if memory_mask.ndim == 1:
            memory_mask = memory_mask.view(1, -1).expand(batch_size, -1)
        if tuple(memory_mask.shape) != (batch_size, anchor_count):
            raise RuntimeError(
                "HSD persistent memory mask shape mismatch: "
                f"got {tuple(memory_mask.shape)}, expected "
                f"{(batch_size, anchor_count)}."
            )
        if not bool((memory_mask.long().sum(dim=-1) == 1).all().item()):
            raise RuntimeError(
                "HSD persistent history read requires exactly one memory row."
            )
        # Boundary-zero layout and append order make the fixed lane index zero.
        if not bool(memory_mask[:, 0].all().item()) or bool(
            memory_mask[:, 1:].any().item()
        ):
            raise RuntimeError(
                "HSD persistent memory cache order must be [memory@0, dynamic...]."
            )

        boundaries = boundary_counts.to(device=read_q.device, dtype=torch.long)
        if boundaries.ndim == 1:
            boundaries = boundaries.view(1, -1).expand(batch_size, -1)
        route_scale = (
            self.hsd_route_head_dim**-0.5
            if self.hsd_use_route_scale
            else 1.0
        ) * self.hsd_route_logit_scale
        logits = self._route_logits(
            router_q.float(),
            anchor_k.float(),
            heads,
        ).float() * route_scale
        token_index = torch.arange(
            start_index,
            start_index + text_len,
            dtype=torch.long,
            device=read_q.device,
        )
        valid = boundaries[:, None, None, :] <= token_index[None, :, None, None]
        valid = valid.expand(batch_size, text_len, heads, anchor_count)

        memory_idx = torch.zeros(
            batch_size,
            text_len,
            heads,
            1,
            dtype=torch.long,
            device=read_q.device,
        )
        memory_valid = valid[..., :1]
        memory_logits = logits[..., :1]

        dynamic_count = max(0, anchor_count - 1)
        if dynamic_count:
            dynamic_logits_all = logits[..., 1:]
            dynamic_valid_all = valid[..., 1:]
            dynamic_topk = min(int(self.hsd_read_topk), dynamic_count)
            if dynamic_topk > 0 and dynamic_topk < dynamic_count:
                dynamic_idx_local = dynamic_logits_all.masked_fill(
                    ~dynamic_valid_all,
                    float("-inf"),
                ).topk(k=dynamic_topk, dim=-1).indices
                dynamic_logits = dynamic_logits_all.gather(-1, dynamic_idx_local)
                dynamic_valid = dynamic_valid_all.gather(-1, dynamic_idx_local)
                dynamic_idx = dynamic_idx_local + 1
            else:
                dynamic_idx = torch.arange(
                    1,
                    anchor_count,
                    dtype=torch.long,
                    device=read_q.device,
                ).view(1, 1, 1, dynamic_count).expand(
                    batch_size,
                    text_len,
                    heads,
                    dynamic_count,
                )
                dynamic_logits = dynamic_logits_all
                dynamic_valid = dynamic_valid_all
        else:
            dynamic_idx = memory_idx[..., :0]
            dynamic_logits = memory_logits[..., :0]
            dynamic_valid = memory_valid[..., :0]

        candidate_idx = torch.cat([memory_idx, dynamic_idx], dim=-1)
        candidate_logits = torch.cat([memory_logits, dynamic_logits], dim=-1)
        candidate_valid = torch.cat([memory_valid, dynamic_valid], dim=-1)
        if null_logits is not None:
            joint_logits = torch.cat(
                [candidate_logits.float(), null_logits.float()],
                dim=-1,
            )
            joint_valid = torch.cat(
                [candidate_valid, torch.ones_like(null_logits, dtype=torch.bool)],
                dim=-1,
            )
            joint_weights = _route_softmax_weights(
                joint_logits,
                joint_valid,
                dim=-1,
            ).float()
            candidate_weights = joint_weights[..., : candidate_idx.shape[-1]]
        else:
            joint_weights = _route_softmax_weights(
                candidate_logits.float(),
                candidate_valid,
                dim=-1,
            ).float()
            candidate_weights = joint_weights

        batch_idx = torch.arange(batch_size, device=states.device).view(
            batch_size, 1, 1, 1
        )
        head_idx = torch.arange(heads, device=states.device).view(1, 1, heads, 1)
        selected_states = states[
            batch_idx,
            candidate_idx.to(states.device),
            head_idx,
        ]
        candidate_reads = torch.einsum(
            "bthk,bthmkv->bthmv",
            read_q.to(selected_states.dtype),
            selected_states,
        ) * (self.head_k_dim**-0.5)
        candidate_reads = candidate_reads.masked_fill(
            ~candidate_valid.unsqueeze(-1),
            0,
        )
        result = torch.einsum(
            "bthm,bthmv->bthv",
            candidate_weights.to(candidate_reads.dtype),
            candidate_reads,
        ).to(read_q.dtype)

        memory_contribution = candidate_reads[..., 0, :] * candidate_weights[
            ..., 0, None
        ].to(candidate_reads.dtype)
        dynamic_contribution = torch.einsum(
            "bthm,bthmv->bthv",
            candidate_weights[..., 1:].to(candidate_reads.dtype),
            candidate_reads[..., 1:, :],
        )
        total_history_contribution = memory_contribution + dynamic_contribution
        dynamic_weights = candidate_weights[..., 1:]
        dynamic_weights = dynamic_weights.sort(dim=-1, descending=True).values
        zero_probability = candidate_weights[..., 0].new_zeros(())
        dynamic_rank_probabilities = [
            (
                dynamic_weights[..., rank].detach().float().mean()
                if rank < dynamic_weights.shape[-1]
                else zero_probability
            )
            for rank in range(4)
        ]
        null_probability = (
            joint_weights[..., -1].detach().float().mean()
            if null_logits is not None
            else zero_probability
        )
        state_rms_by_anchor = states.detach().float().square().mean(
            dim=(-3, -2, -1)
        ).sqrt()
        dynamic_rms = state_rms_by_anchor.masked_select(~memory_mask)
        metrics = {
            "memory_logit": memory_logits.detach().float().mean(),
            "memory_probability": candidate_weights[..., 0].detach().float().mean(),
            "dynamic_probability": dynamic_weights.detach().float().sum(dim=-1).mean(),
            "null_probability": null_probability,
            "dynamic_rank1_probability": dynamic_rank_probabilities[0],
            "dynamic_rank2_probability": dynamic_rank_probabilities[1],
            "dynamic_rank3_probability": dynamic_rank_probabilities[2],
            "dynamic_rank4_probability": dynamic_rank_probabilities[3],
            "memory_read_rms": memory_contribution.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "dynamic_read_rms": dynamic_contribution.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "total_history_read_rms": total_history_contribution.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "state_rms": state_rms_by_anchor.masked_select(memory_mask).mean(),
            "dynamic_state_rms_median": (
                dynamic_rms.median()
                if dynamic_rms.numel()
                else state_rms_by_anchor.new_zeros(())
            ),
        }
        self._record_persistent_memory_metrics(metrics)
        self._trace_internal("persistent_memory_logit", memory_logits)
        self._trace_internal(
            "persistent_memory_probability",
            candidate_weights[..., :1],
        )
        self._trace_internal(
            "persistent_memory_raw_read",
            candidate_reads[..., 0, :],
        )
        self._trace_internal("persistent_memory_read", memory_contribution)
        self._trace_internal(
            "persistent_memory_joint_probability_sum",
            joint_weights.sum(dim=-1),
        )
        return result

    def _record_persistent_memory_metrics(
        self,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        detached = {name: value.detach() for name, value in metrics.items()}
        self._hsd_persistent_memory_metrics = detached
        for name, value in detached.items():
            previous = self._hsd_persistent_memory_metric_sums.get(name)
            self._hsd_persistent_memory_metric_sums[name] = (
                value if previous is None else previous + value
            )
        self._hsd_persistent_memory_metric_count += 1

    def get_persistent_memory_metrics(
        self,
        *,
        cumulative: bool = True,
        reset: bool = False,
    ) -> dict[str, float]:
        if cumulative and self._hsd_persistent_memory_metric_count:
            count = float(self._hsd_persistent_memory_metric_count)
            source = {
                name: value / count
                for name, value in self._hsd_persistent_memory_metric_sums.items()
            }
        else:
            source = self._hsd_persistent_memory_metrics
        # A paged helper records one coherent latest sample. Prefer it as a
        # unit instead of overwriting individual fields of a cumulative eager
        # mean, which would imply a false dataset-average interpretation.
        source = (
            self._hsd_paged_persistent_memory_metrics
            if self._hsd_paged_persistent_memory_metrics
            else source
        )
        result = {
            name: float(value.detach().float().cpu().item())
            for name, value in source.items()
        }
        if reset:
            self._hsd_persistent_memory_metrics = {}
            self._hsd_persistent_memory_metric_sums = {}
            self._hsd_persistent_memory_metric_count = 0
            self._hsd_paged_persistent_memory_metrics.clear()
        return result

    def _paged_persistent_memory_metric_sink(
        self,
        reference: torch.Tensor,
        *,
        residual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if self.hsd_persistent_memory_mode == "removed":
            return None
        if reference.is_cuda and torch.cuda.is_current_stream_capturing():
            return None
        if residual is not None:
            self._hsd_paged_persistent_memory_metrics["residual_rms"] = (
                residual.detach().float().square().mean().sqrt()
            )
        return self._hsd_paged_persistent_memory_metrics

    def forward_hsd(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layout: HistoricalHSDLayout,
    ) -> None:
        if hidden_states.ndim != 2:
            raise ValueError("Historical vLLM HSD GDN expects flat hidden states.")
        self._hsd_force_segments_fallback = False
        local_v_heads = self.num_v_heads // self.tp_size
        local_k_heads = self.num_k_heads // self.tp_size
        local_key_dim = self.key_dim // self.tp_size
        local_value_dim = self.value_dim // self.tp_size
        is_cached_decode = self._layout_is_cached_decode(layout)
        if (
            is_cached_decode
            and self._forward_hsd_batched_decode_light(
            hidden_states=hidden_states,
            output=output,
            layout=layout,
            )
        ):
            return
        if is_cached_decode and layout.padded_static_decode:
            raise RuntimeError(
                "hsd_padded_static_decode_fallback: padded static decode requires "
                "the batched no-anchor decode path; see decode_prepare_fallback "
                "events for the rejected condition."
            )
        if (
            not is_cached_decode
            and self.hsd_prefill_backend == "varlen"
            and self.hsd_local_prefill_producer_backend != "flashqla_tile"
            and self._forward_hsd_batched_varlen(
                hidden_states=hidden_states,
                output=output,
                layout=layout,
            )
        ):
            return

        req_outputs: list[torch.Tensor] = []
        trace_requested = _hsd_trace_requested() or _hsd_decode_trace_requested()
        for req_idx, (expanded_start, expanded_len) in enumerate(
            zip(layout.req_expanded_starts, layout.req_expanded_lens)
        ):
            if expanded_len == 0:
                continue
            expanded_end = expanded_start + expanded_len
            req_hidden = hidden_states[expanded_start:expanded_end]
            req_mask = layout.anchor_mask[expanded_start:expanded_end]
            text_positions = (~req_mask).nonzero(as_tuple=False).flatten()
            anchor_positions = req_mask.nonzero(as_tuple=False).flatten()
            text_len = int(text_positions.numel())
            anchor_count = int(anchor_positions.numel())
            if text_len <= 0:
                raise ValueError("Historical vLLM HSD requires at least one text token.")

            state_idx = self._state_index_for_request(layout, req_idx)
            history_state_idx = self._history_state_index_for_request(
                layout, req_idx
            )
            past_text_len = int(layout.past_text_lens[req_idx])
            decode_step = past_text_len > 0 and text_len == 1
            trace_backend = "decode_loop" if decode_step else "prefill_loop"
            persistent_text_cache = (
                get_forward_context().attn_metadata is not None
                and not _env_truthy("VLLM_HSD_DISABLE_TEXT_STATE_CACHE")
            )
            if _hsd_state_cache_log_enabled(int(self.layer_idx)):
                logger.info(
                    "HSD forward layer=%s req_idx=%s state_idx=%s "
                    "past_text_len=%s text_len=%s anchor_count=%s "
                    "expanded_len=%s decode_step=%s persistent_text_cache=%s",
                    int(self.layer_idx) + 1,
                    req_idx,
                    state_idx,
                    past_text_len,
                    text_len,
                    anchor_count,
                    expanded_len,
                    decode_step,
                    persistent_text_cache,
            )
            if past_text_len <= 0:
                self.hsd_anchor_cache.pop(history_state_idx, None)
                page_cache = getattr(self, "hsd_anchor_page_cache", None)
                if page_cache is not None:
                    page_cache.clear({history_state_idx})
                self.hsd_text_state_cache.pop(state_idx, None)
                self.hsd_prefix_recurrent_replay.pop(state_idx, None)
                self._clear_native_state_for_request(state_idx)
            elif persistent_text_cache:
                self._restore_hsd_text_state_cache(state_idx, req_hidden.device)

            req_anchor_start = layout.req_anchor_starts[req_idx]
            req_anchor_len = layout.req_anchor_lens[req_idx]
            boundary_counts = layout.anchor_boundaries[
                req_anchor_start : req_anchor_start + req_anchor_len
            ]
            persistent_anchor_mask = layout.persistent_memory_mask[
                expanded_start:expanded_end
            ].index_select(0, anchor_positions).view(1, -1)
            prefix_snapshot_requests = (
                layout.hsd_prefix_snapshot_requests[req_idx]
                if req_idx < len(layout.hsd_prefix_snapshot_requests)
                else []
            )
            prefix_replay_pending = (
                not decode_step
                and _env_truthy("VLLM_HSD_PREFIX_CACHE_PAD_SUFFIX_TO_FULL")
                and state_idx in self.hsd_prefix_recurrent_replay
            )
            prefix_text_padding = int(past_text_len) if prefix_replay_pending else 0
            prefix_expanded_padding = 0
            prefix_anchor_padding = 0
            if prefix_replay_pending:
                anchor_chunk_size = int(
                    getattr(self.config, "anchor_chunk_size", 0) or 0
                )
                anchor_token_num = int(
                    getattr(self.config, "anchor_token_num", 0) or 0
                )
                if anchor_chunk_size <= 0 or anchor_token_num <= 0:
                    raise RuntimeError(
                        "hsd_prefix_suffix_padding_invalid_anchor_config: "
                        f"chunk={anchor_chunk_size} tokens={anchor_token_num}"
                    )
                prefix_anchor_padding = (
                    int(past_text_len) // anchor_chunk_size
                ) * anchor_token_num
                prefix_expanded_padding = (
                    prefix_text_padding + prefix_anchor_padding
                )

            def run_prefix_padded_linear(
                layer: nn.Module,
                tensor: torch.Tensor,
                prefix_rows: int,
            ) -> tuple[torch.Tensor, torch.Tensor | None]:
                if prefix_rows <= 0:
                    return layer(tensor)
                padded = tensor.new_zeros(
                    prefix_rows + int(tensor.shape[0]),
                    *tensor.shape[1:],
                )
                padded[prefix_rows:].copy_(tensor)
                projected, bias = layer(padded)
                return projected[prefix_rows:].contiguous(), bias

            if trace_requested:
                self._trace_internal(
                    "hsd_past_text_len",
                    req_hidden.new_tensor([past_text_len], dtype=torch.int64),
                )
                self._trace_internal(
                    "hsd_state_idx",
                    req_hidden.new_tensor([state_idx], dtype=torch.int64),
                )
                self._trace_internal(
                    "hidden_states_text_before_proj",
                    req_hidden.index_select(0, text_positions).view(
                        1, text_len, req_hidden.shape[-1]
                    ),
                )
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="hidden_states_text_before_proj",
                    tensor=req_hidden.index_select(0, text_positions).view(
                        1, text_len, req_hidden.shape[-1]
                    ),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )

            mixed_qkvz, _ = run_prefix_padded_linear(
                self.in_proj_qkvz,
                req_hidden,
                prefix_expanded_padding,
            )
            ba, _ = run_prefix_padded_linear(
                self.in_proj_ba,
                req_hidden,
                prefix_expanded_padding,
            )
            qkv_size = local_key_dim * 2 + local_value_dim
            z_size = local_value_dim
            mixed_qkv, z_all = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            b_all, a_all = ba.chunk(2, dim=-1)

            mixed_qkv_text = mixed_qkv.index_select(0, text_positions).view(
                1, text_len, qkv_size
            )
            mixed_qkv_text_before_conv = mixed_qkv_text.squeeze(0).contiguous()
            if not decode_step:
                self._capture_hsd_prefill_passive_sidecar(
                    "hidden_states_text_before_proj",
                    req_hidden.index_select(0, text_positions),
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
                self._capture_hsd_prefill_passive_sidecar(
                    "mixed_qkv_text_before_conv",
                    mixed_qkv_text_before_conv,
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
            conv_state_snapshot = None
            if prefix_snapshot_requests and not decode_step:
                current_conv_state = self._conv_state_for_request(state_idx)
                if current_conv_state is not None:
                    conv_state_snapshot = current_conv_state.detach().clone()
            z = z_all.view(1, expanded_len, local_v_heads, self.head_v_dim)
            b = b_all.index_select(0, text_positions)
            a = a_all.index_select(0, text_positions)
            if trace_requested:
                z_text = z_all.index_select(0, text_positions).view(
                    1, text_len, z_size
                )
                qkvzba_text = torch.cat(
                    [
                        mixed_qkv_text,
                        z_text,
                        b.view(1, text_len, -1),
                        a.view(1, text_len, -1),
                    ],
                    dim=-1,
                )
                if qkvzba_text.shape[-1] % max(1, local_k_heads) == 0:
                    qkvzba_text = qkvzba_text.view(
                        1, text_len, local_k_heads, -1
                    )
                self._trace_internal("qkvzba_text_after_proj", qkvzba_text)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="qkvzba_text_after_proj",
                    tensor=qkvzba_text,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("gate_text", z.index_select(1, text_positions))

            if trace_requested:
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="conv_state_before_text_conv",
                    tensor=self._conv_state_for_request(state_idx),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
            if not decode_step:
                self._capture_hsd_prefill_passive_sidecar(
                    "conv_state_before_text_conv",
                    self._conv_state_for_request(state_idx),
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
            mixed_qkv_text = self._run_text_conv(
                mixed_qkv_text=mixed_qkv_text,
                state_idx=state_idx,
                past_text_len=past_text_len,
                decode_step=decode_step,
            ).view(text_len, qkv_size)
            if not decode_step:
                self._capture_hsd_prefill_passive_sidecar(
                    "mixed_qkv_text_after_conv",
                    mixed_qkv_text,
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
                self._capture_hsd_prefill_passive_sidecar(
                    "conv_state_after_text_conv",
                    self._conv_state_for_request(state_idx),
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
            if trace_requested:
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="conv_state_after_text_conv",
                    tensor=self._conv_state_for_request(state_idx),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal(
                    "conv_state_after_text_conv",
                    self._conv_state_for_request(state_idx),
                )
                self._trace_internal(
                    "qkvzba_text_after_conv",
                    mixed_qkv_text.view(1, text_len, qkv_size),
                )
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="qkvzba_text_after_conv",
                    tensor=mixed_qkv_text.view(1, text_len, qkv_size),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )

            query, key, value, g, beta = self._post_conv_prep_hsd(
                mixed_qkv_text=mixed_qkv_text,
                a=a,
                b=b,
                local_k_heads=local_k_heads,
                local_v_heads=local_v_heads,
                local_key_dim=local_key_dim,
                local_value_dim=local_value_dim,
            )
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            g = g.unsqueeze(0)
            beta = beta.unsqueeze(0)

            if local_v_heads // max(1, local_k_heads) > 1:
                repeat = local_v_heads // local_k_heads
                query = query.repeat_interleave(repeat, dim=2)
                key = key.repeat_interleave(repeat, dim=2)
            if not decode_step:
                for stage, tensor in (
                    ("q_text", query),
                    ("k_text", key),
                    ("v_text", value),
                    ("beta_text", beta),
                    ("g_text", g),
                ):
                    self._capture_hsd_prefill_passive_sidecar(
                        stage,
                        tensor,
                        past_text_len=past_text_len,
                        text_len=text_len,
                    )
            if trace_requested:
                self._trace_internal("q_text", query)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="q_text",
                    tensor=query,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("k_text", key)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="k_text",
                    tensor=key,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("v_text", value)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="v_text",
                    tensor=value,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("beta_text", beta)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="beta_text",
                    tensor=beta,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("g_text", g)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="g_text",
                    tensor=g,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )

            if anchor_count:
                anchor_hidden = req_hidden.index_select(0, anchor_positions)
                anchor_q, _ = run_prefix_padded_linear(
                    self.anchor_q_proj,
                    anchor_hidden,
                    prefix_anchor_padding,
                )
                anchor_q = anchor_q.view(
                    1, anchor_count, local_v_heads, self.head_k_dim
                )
                anchor_q = _l2norm(anchor_q.contiguous())
            else:
                anchor_hidden = req_hidden.new_zeros(0, req_hidden.shape[-1])
                anchor_q = req_hidden.new_zeros(
                    1, 0, local_v_heads, self.head_k_dim
                )
            if trace_requested:
                self._trace_internal(
                    "anchor_hidden_pre",
                    anchor_hidden.unsqueeze(0),
                )
                self._trace_internal("anchor_q", anchor_q)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="anchor_q",
                    tensor=anchor_q,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )

            anchor_k = None
            if self.hsd_layer_uses_global_read:
                if self.anchor_k_proj is None:
                    raise RuntimeError("anchor_k_proj is required for HSD global read.")
                if anchor_count:
                    anchor_k, _ = run_prefix_padded_linear(
                        self.anchor_k_proj,
                        anchor_hidden,
                        prefix_anchor_padding,
                    )
                    anchor_k = anchor_k.view(
                        1,
                        anchor_count,
                        self.hsd_local_route_key_heads,
                        self.hsd_route_head_dim,
                    )
                    anchor_k = self._replace_persistent_anchor_route_key(
                        anchor_k,
                        persistent_anchor_mask,
                    )
                    if self.hsd_route_qk_l2norm:
                        anchor_k = _l2norm(anchor_k.contiguous())
                else:
                    anchor_k = req_hidden.new_zeros(
                        1,
                        0,
                        self.hsd_local_route_key_heads,
                        self.hsd_route_head_dim,
                    )
                if not decode_step:
                    self._capture_hsd_prefill_passive_sidecar(
                        "anchor_keys",
                        anchor_k,
                        past_text_len=past_text_len,
                        text_len=text_len,
                    )
                if trace_requested:
                    self._trace_internal("anchor_k", anchor_k)
                    self._trace_decode_stage(
                        backend=trace_backend,
                        stage="anchor_k",
                        tensor=anchor_k,
                        req_idx=req_idx,
                        state_idx=state_idx,
                        past_text_len=past_text_len,
                        anchor_count=anchor_count,
                        boundary_counts=boundary_counts,
                    )

            if trace_requested:
                self._trace_internal(
                    "anchor_boundaries",
                    boundary_counts.view(1, -1).to(device=req_hidden.device),
                )
            if decode_step:
                text_o, anchor_o, new_anchor_states, recurrent_state = (
                    self._run_hsd_decode_vllm(
                        mixed_qkv_text=mixed_qkv_text.view(1, text_len, qkv_size),
                        a=a,
                        b=b,
                        query=query,
                        key=key,
                        value=value,
                        g=g,
                        beta=beta,
                        anchor_q=anchor_q,
                        boundary_counts=boundary_counts,
                        boundary_start=past_text_len,
                        state_idx=state_idx,
                    )
                )
            else:
                initial_state = self._recurrent_state_for_request(
                    state_idx, past_text_len, req_hidden.device
                )
                self._capture_hsd_prefill_passive_sidecar(
                    "initial_recurrent_state",
                    initial_state,
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
                replay_result = self._run_hsd_prefix_recurrent_replay(
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    query=query,
                    key=key,
                    value=value,
                    g=g,
                    beta=beta,
                    anchor_q=anchor_q,
                    boundary_counts=boundary_counts,
                )
                if replay_result is None:
                    replay_result = self._run_hsd_prefill_vllm(
                        query=query,
                        key=key,
                        value=value,
                        g=g,
                        beta=beta,
                        anchor_q=anchor_q,
                        boundary_counts=boundary_counts,
                        boundary_start=past_text_len,
                        initial_state=initial_state,
                    )
                text_o, anchor_o, new_anchor_states, recurrent_state = (
                    replay_result
                )
                self._capture_hsd_prefill_passive_sidecar(
                    "anchor_states",
                    new_anchor_states,
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
                self._capture_hsd_prefill_passive_sidecar(
                    "text_o_before_history",
                    text_o,
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
                self._capture_hsd_prefill_passive_sidecar(
                    "final_recurrent_state",
                    recurrent_state,
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
            anchor_o, new_anchor_states = self._replace_persistent_anchor_state(
                anchor_q=anchor_q,
                anchor_o=anchor_o,
                anchor_states=new_anchor_states,
                persistent_anchor_mask=persistent_anchor_mask,
            )
            # Capture only after the boundary-zero state replacement. Prefix
            # caching is rejected for persistent-memory checkpoints, but this
            # ordering also keeps the snapshot helper correct by construction.
            if not decode_step and prefix_snapshot_requests:
                if conv_state_snapshot is None:
                    self._record_hsd_varlen_event(
                        "hsd_prefix_cache_fallback",
                        reason="snapshot_store_missing_conv_state",
                        state_idx=int(state_idx),
                    )
                else:
                    self.capture_hsd_prefix_snapshot_from_tensors(
                        state_idx=state_idx,
                        history_state_idx=history_state_idx,
                        past_text_len=past_text_len,
                        snapshot_requests=prefix_snapshot_requests,
                        conv_state_snapshot=conv_state_snapshot,
                        pre_conv_inputs=mixed_qkv_text_before_conv,
                        query=query,
                        key_tensor=key,
                        value=value,
                        g=g,
                        beta=beta,
                        initial_state=initial_state,
                        final_recurrent_state=recurrent_state,
                        new_anchor_states=new_anchor_states,
                        new_anchor_keys=anchor_k,
                        new_anchor_boundaries=boundary_counts,
                    )
            if trace_requested:
                self._trace_internal("text_o_before_history", text_o)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="text_o_before_history",
                    tensor=text_o,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_internal("anchor_o", anchor_o)
                self._trace_internal("anchor_states", new_anchor_states)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="anchor_states",
                    tensor=new_anchor_states,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )

            if self.hsd_layer_uses_global_read:
                assert anchor_k is not None
                materialized = None
                if self._using_scheduler_history_cache():
                    appended, page_cache, _ = (
                        self._append_new_anchor_to_history_cache(
                            state_idx=history_state_idx,
                            past_text_len=past_text_len,
                            new_anchor_states=new_anchor_states,
                            new_anchor_keys=anchor_k,
                            new_anchor_boundaries=boundary_counts,
                            use_paged_history=True,
                        )
                    )
                    if (
                        appended
                        and page_cache is not None
                        and page_cache.length(history_state_idx) > 0
                    ):
                        materialized = self._materialize_anchor_page_history(
                            page_cache,
                            history_state_idx,
                            device=hidden_states.device,
                        )
                if materialized is None:
                    materialized = self._merge_anchor_cache(
                        state_idx=history_state_idx,
                        past_text_len=past_text_len,
                        new_anchor_states=new_anchor_states,
                        new_anchor_keys=anchor_k,
                        new_anchor_boundaries=boundary_counts,
                    )
                anchor_states, anchor_keys, anchor_boundaries = materialized
                if anchor_states.shape[1] > 0 and anchor_keys is not None:
                    if self.router_q_proj is None:
                        raise RuntimeError("router_q_proj is required for HSD global read.")
                    text_hidden = req_hidden.index_select(0, text_positions)
                    read_q = query
                    router_q, _ = run_prefix_padded_linear(
                        self.router_q_proj,
                        text_hidden,
                        prefix_text_padding,
                    )
                    router_q = router_q.view(
                        1,
                        text_len,
                        self.hsd_local_route_query_heads,
                        self.hsd_route_head_dim,
                    )
                    if self.hsd_route_qk_l2norm:
                        router_q = _l2norm(router_q.contiguous())
                    if prefix_text_padding > 0:
                        padded_text_hidden = text_hidden.new_zeros(
                            prefix_text_padding + text_len,
                            *text_hidden.shape[1:],
                        )
                        padded_text_hidden[prefix_text_padding:].copy_(
                            text_hidden
                        )
                        padded_null_logits = self._project_null_gate_logits(
                            padded_text_hidden
                        )
                        null_logits_flat = (
                            None
                            if padded_null_logits is None
                            else padded_null_logits[
                                prefix_text_padding:
                            ].contiguous()
                        )
                    else:
                        null_logits_flat = self._project_null_gate_logits(text_hidden)
                    null_logits = None
                    if null_logits_flat is not None:
                        null_logits = (
                            null_logits_flat.view(1, text_len, local_v_heads, 1)
                            .contiguous()
                        )
                    if trace_requested:
                        self._trace_internal("router_q", router_q)
                        self._trace_internal("read_q", read_q)
                        self._trace_internal("null_gate_logits", null_logits)
                        self._trace_decode_stage(
                            backend=trace_backend,
                            stage="router_q",
                            tensor=router_q,
                            req_idx=req_idx,
                            state_idx=state_idx,
                            past_text_len=past_text_len,
                            anchor_count=anchor_count,
                            boundary_counts=anchor_boundaries,
                        )
                        self._trace_decode_stage(
                            backend=trace_backend,
                            stage="read_q",
                            tensor=read_q,
                            req_idx=req_idx,
                            state_idx=state_idx,
                            past_text_len=past_text_len,
                            anchor_count=anchor_count,
                            boundary_counts=anchor_boundaries,
                        )
                        self._trace_decode_stage(
                            backend=trace_backend,
                            stage="null_gate_logits",
                            tensor=null_logits,
                            req_idx=req_idx,
                            state_idx=state_idx,
                            past_text_len=past_text_len,
                            anchor_count=anchor_count,
                            boundary_counts=anchor_boundaries,
                        )
                    if prefix_text_padding > 0:
                        full_text_len = prefix_text_padding + text_len

                        def pad_text_tensor(
                            tensor: torch.Tensor | None,
                        ) -> torch.Tensor | None:
                            if tensor is None:
                                return None
                            padded = tensor.new_zeros(
                                tensor.shape[0],
                                full_text_len,
                                *tensor.shape[2:],
                            )
                            padded[:, prefix_text_padding:].copy_(tensor)
                            return padded

                        full_history_out = self._history_read(
                            read_q=pad_text_tensor(read_q),
                            router_q=pad_text_tensor(router_q),
                            states=anchor_states,
                            anchor_k=anchor_keys,
                            boundary_counts=anchor_boundaries,
                            text_len=full_text_len,
                            start_index=0,
                            null_logits=pad_text_tensor(null_logits),
                        )
                        history_out = full_history_out[
                            :, prefix_text_padding:
                        ].contiguous()
                        self._record_hsd_varlen_event(
                            "hsd_prefix_history_read_full_shape_hit",
                            state_idx=int(state_idx),
                            prefix_len=int(prefix_text_padding),
                            text_tokens=int(text_len),
                            full_text_tokens=int(full_text_len),
                        )
                    else:
                        history_out = self._history_read(
                            read_q=read_q,
                            router_q=router_q,
                            states=anchor_states,
                            anchor_k=anchor_keys,
                            boundary_counts=anchor_boundaries,
                            text_len=text_len,
                            start_index=past_text_len,
                            null_logits=null_logits,
                        )
                    if trace_requested:
                        self._trace_internal("history_out_text", history_out)
                        self._trace_decode_stage(
                            backend=trace_backend,
                            stage="history_out_text",
                            tensor=history_out,
                            req_idx=req_idx,
                            state_idx=state_idx,
                            past_text_len=past_text_len,
                            anchor_count=anchor_count,
                            boundary_counts=anchor_boundaries,
                        )
                    history_out = self._apply_history_out_rmsnorm(history_out)
                    if (
                        trace_requested
                        and self.hsd_history_out_rmsnorm_weight is not None
                    ):
                        self._trace_internal("history_out_rmsnorm_text", history_out)
                        self._trace_decode_stage(
                            backend=trace_backend,
                            stage="history_out_rmsnorm_text",
                            tensor=history_out,
                            req_idx=req_idx,
                            state_idx=state_idx,
                            past_text_len=past_text_len,
                            anchor_count=anchor_count,
                            boundary_counts=anchor_boundaries,
                        )
                    text_o = text_o + (
                        history_out
                        * self._history_scale_tensor(history_out.device).to(
                            dtype=history_out.dtype
                        )
                    ).to(text_o.dtype)
            if not decode_step:
                self._capture_hsd_prefill_passive_sidecar(
                    "delta_o_text",
                    text_o,
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
            if trace_requested:
                self._trace_internal("delta_o_text", text_o)
                self._trace_internal("recurrent_state_to_store", recurrent_state)
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="delta_o_text",
                    tensor=text_o,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="recurrent_state_to_store",
                    tensor=recurrent_state,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )

            self._store_recurrent_state(state_idx, recurrent_state)
            if trace_requested:
                recurrent_state_readback = self._stored_recurrent_state_for_request(
                    state_idx
                )
                self._trace_internal(
                    "recurrent_state_after_store_readback",
                    recurrent_state_readback,
                )
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="recurrent_state_after_store_readback",
                    tensor=recurrent_state_readback,
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
            if persistent_text_cache:
                self._update_hsd_text_state_cache(state_idx)

            core_attn_out = req_hidden.new_zeros(
                expanded_len, local_v_heads, self.head_v_dim
            )
            core_attn_out.index_copy_(0, text_positions, text_o.squeeze(0))
            if anchor_count:
                core_attn_out.index_copy_(0, anchor_positions, anchor_o.squeeze(0))
            if not decode_step:
                self._capture_hsd_prefill_passive_sidecar(
                    "core_attn_out_before_norm",
                    core_attn_out.index_select(0, text_positions),
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
            if trace_requested:
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="core_attn_out_before_norm",
                    tensor=core_attn_out.index_select(0, text_positions).view(
                        1, text_len, local_v_heads, self.head_v_dim
                    ),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
            if prefix_expanded_padding > 0:
                full_expanded_len = prefix_expanded_padding + expanded_len
                padded_core_attn_out = core_attn_out.new_zeros(
                    full_expanded_len,
                    local_v_heads,
                    self.head_v_dim,
                )
                padded_z = z.new_zeros(
                    1,
                    full_expanded_len,
                    local_v_heads,
                    self.head_v_dim,
                )
                padded_core_attn_out[prefix_expanded_padding:].copy_(
                    core_attn_out
                )
                padded_z[:, prefix_expanded_padding:].copy_(z)
                padded_norm_input = padded_core_attn_out.reshape(
                    -1, self.head_v_dim
                )
                padded_norm_gate = padded_z.reshape(-1, self.head_v_dim)
                if decode_step:
                    padded_core_attn_out = self._apply_hsd_decode_gate_norm(
                        padded_norm_input,
                        padded_norm_gate,
                    ).reshape(full_expanded_len, -1)
                else:
                    padded_core_attn_out = self.norm(
                        padded_norm_input,
                        padded_norm_gate,
                    ).reshape(full_expanded_len, -1)
                padded_req_out, _ = self.out_proj(padded_core_attn_out)
                core_attn_out = padded_core_attn_out[
                    prefix_expanded_padding:
                ].contiguous()
                req_out = padded_req_out[
                    prefix_expanded_padding:
                ].contiguous()
                self._record_hsd_varlen_event(
                    "hsd_prefix_output_full_shape_hit",
                    state_idx=int(state_idx),
                    prefix_text_tokens=int(prefix_text_padding),
                    prefix_anchor_tokens=int(prefix_anchor_padding),
                    expanded_tokens=int(expanded_len),
                    full_expanded_tokens=int(full_expanded_len),
                )
            else:
                core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
                z_flat = z.reshape(-1, self.head_v_dim)
                if decode_step:
                    core_attn_out = self._apply_hsd_decode_gate_norm(
                        core_attn_out,
                        z_flat,
                    )
                else:
                    core_attn_out = self.norm(core_attn_out, z_flat)
                core_attn_out = core_attn_out.reshape(expanded_len, -1)
                req_out, _ = self.out_proj(core_attn_out)
            if not decode_step:
                self._capture_hsd_prefill_passive_sidecar(
                    "core_attn_out_after_norm",
                    core_attn_out.index_select(0, text_positions),
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
            if trace_requested:
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="core_attn_out_after_norm",
                    tensor=core_attn_out.index_select(0, text_positions).view(
                        1, text_len, -1
                    ),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                        boundary_counts=boundary_counts,
                    )
            if not decode_step:
                self._capture_hsd_prefill_passive_sidecar(
                    "projected_out_text",
                    req_out.index_select(0, text_positions),
                    past_text_len=past_text_len,
                    text_len=text_len,
                )
            if trace_requested:
                self._trace_decode_stage(
                    backend=trace_backend,
                    stage="projected_out_text",
                    tensor=req_out.index_select(0, text_positions).view(
                        1, text_len, -1
                    ),
                    req_idx=req_idx,
                    state_idx=state_idx,
                    past_text_len=past_text_len,
                    anchor_count=anchor_count,
                    boundary_counts=boundary_counts,
                )
            req_outputs.append(req_out)

        output[: layout.num_expanded_tokens] = torch.cat(req_outputs, dim=0)


def hsd_static_decode_core(
    mixed_qkv: torch.Tensor,
    ba: torch.Tensor,
    router_q: torch.Tensor,
    null_logits: torch.Tensor,
    output: torch.Tensor,
    state_indices: torch.Tensor,
    history_state_indices: torch.Tensor,
    past_lens: torch.Tensor,
    valid_mask: torch.Tensor,
    conv_cache: torch.Tensor,
    recurrent_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Graph V3 stateful core with projections and output transforms outside."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    if layer.hsd_persistent_state_sidecar.get("conv_state") is not conv_cache:
        raise RuntimeError("hsd_graph_v3_conv_state_alias_mismatch")
    if (
        layer.hsd_persistent_state_sidecar.get("recurrent_state")
        is not recurrent_cache
    ):
        raise RuntimeError("hsd_graph_v3_recurrent_state_alias_mismatch")
    layer._forward_hsd_graph_v3_core_tensor_only(
        mixed_qkv=mixed_qkv,
        ba=ba,
        router_q=router_q,
        null_logits=null_logits,
        output=output,
        state_indices=state_indices,
        history_state_indices=history_state_indices,
        past_lens=past_lens,
        valid_mask=valid_mask,
        conv_cache=conv_cache,
        recurrent_cache=recurrent_cache,
    )


def hsd_static_decode_core_fake(
    mixed_qkv: torch.Tensor,
    ba: torch.Tensor,
    router_q: torch.Tensor,
    null_logits: torch.Tensor,
    output: torch.Tensor,
    state_indices: torch.Tensor,
    history_state_indices: torch.Tensor,
    past_lens: torch.Tensor,
    valid_mask: torch.Tensor,
    conv_cache: torch.Tensor,
    recurrent_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="hsd_static_decode_core",
    op_func=hsd_static_decode_core,
    mutates_args=["output", "conv_cache", "recurrent_cache"],
    fake_impl=hsd_static_decode_core_fake,
)


def hsd_static_decode_attention(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    state_indices: torch.Tensor,
    history_state_indices: torch.Tensor,
    past_lens: torch.Tensor,
    valid_mask: torch.Tensor,
    conv_cache: torch.Tensor,
    recurrent_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Graph V3 no-anchor HSD attention island.

    The fixed-shape tensor inputs remain visible to Dynamo while the stateful
    recurrent/cache implementation stays opaque. Conv and recurrent sidecars
    are explicit mutable operands so compile/cudagraph ordering cannot discard
    their updates.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    if layer.hsd_persistent_state_sidecar.get("conv_state") is not conv_cache:
        raise RuntimeError("hsd_graph_v3_conv_state_alias_mismatch")
    if (
        layer.hsd_persistent_state_sidecar.get("recurrent_state")
        is not recurrent_cache
    ):
        raise RuntimeError("hsd_graph_v3_recurrent_state_alias_mismatch")

    layer._forward_hsd_graph_v3_tensor_only(
        hidden_states=hidden_states,
        output=output,
        state_indices=state_indices,
        history_state_indices=history_state_indices,
        past_lens=past_lens,
        valid_mask=valid_mask,
        conv_cache=conv_cache,
        recurrent_cache=recurrent_cache,
    )


def hsd_static_decode_attention_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    state_indices: torch.Tensor,
    history_state_indices: torch.Tensor,
    past_lens: torch.Tensor,
    valid_mask: torch.Tensor,
    conv_cache: torch.Tensor,
    recurrent_cache: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="hsd_static_decode_attention",
    op_func=hsd_static_decode_attention,
    mutates_args=["output", "conv_cache", "recurrent_cache"],
    fake_impl=hsd_static_decode_attention_fake,
)


def _run_hsd_layer_mlp_delta(
    layer: "Safin1DecoderLayer",
    hidden_states: torch.Tensor,
    layout: HistoricalHSDLayout,
) -> torch.Tensor:
    return layer.mlp(hidden_states)


class Safin1DecoderLayer(DecoderDiagnosticsMixin, Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config: Safin1TextConfig | Safin1MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        self.config = config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.is_hsd_layer = self.layer_type == "linear_attention" or (
            self.layer_type == "full_attention"
            and _self_attention_layer_uses_hsd(config, self.layer_idx)
        )
        self.hsd_graph_v3_layer_name = f"{prefix}.hsd_graph_v3_layer"
        static_context = vllm_config.compilation_config.static_forward_context
        if self.hsd_graph_v3_layer_name in static_context:
            raise ValueError(
                f"Duplicate HSD Graph V3 layer name: {self.hsd_graph_v3_layer_name}"
            )
        static_context[self.hsd_graph_v3_layer_name] = self

        if self.layer_type == "linear_attention":
            self.linear_attn = HistoricalGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                create_in_proj_qkvz=vllm_config.lora_config is None,
            )
        elif self.layer_type == "full_attention":
            if self.is_hsd_layer:
                self.self_attn = HistoricalGatedDeltaNetAttention(
                    config=config,
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.self_attn",
                    create_in_proj_qkvz=vllm_config.lora_config is None,
                )
            else:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if config.model_type == "safin1_35b_a3b_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "safin1_4b_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size)
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(1, 1, config.hidden_size)
            )
        # Optional graph-safe diagnostic snapshots. Warmup allocates these
        # tensors before CUDA graph capture; replay only performs fixed-shape
        # device copies. The runner persists them after replay, outside the
        # captured region.
        self._hsd_graph_v3_tensor_sidecars: dict[str, torch.Tensor] = {}

    def _active_hsd_attn(self) -> HistoricalGatedDeltaNetAttention | None:
        if not self.is_hsd_layer:
            return None
        name = "linear_attn" if self.layer_type == "linear_attention" else "self_attn"
        module = getattr(self, name, None)
        return module if isinstance(module, HistoricalGatedDeltaNetAttention) else None

    def prepare_hsd_graph_v3_tensor_sidecars(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Allocate trace buffers before Dynamo or CUDA graph capture."""
        if not os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_DIR"):
            return
        layer_num = int(self.layer_idx) + 1
        selected_layers = _parse_layer_selector(
            os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_LAYERS")
        )
        if selected_layers is not None and layer_num not in selected_layers:
            return
        selected_stages = _parse_csv_selector(
            os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_STAGES")
        )
        layer_stages = {
            "decoder_input",
            "decoder_input_full",
            "input_layernorm_out",
            "attention_raw_out",
            "post_attention_layernorm_out",
            "mlp_input_inside",
            "mlp_shared_input_inside",
            "mlp_shared_gate_up_inside",
            "mlp_shared_activated_inside",
            "mlp_shared_down_inside",
            "mlp_shared_gate_inside",
            "mlp_shared_output_inside",
            "mlp_routed_output_inside",
            "mlp_output_inside",
            "mlp_out",
            "decoder_output",
            "final_norm_out",
        }
        if selected_stages is not None:
            layer_stages.intersection_update(selected_stages)
        hidden_shape = (int(self.config.hidden_size),)
        stage_shapes = {stage: hidden_shape for stage in layer_stages}
        experts = getattr(getattr(self, "mlp", None), "experts", None)
        runner = getattr(experts, "runner", None)
        shared_experts = getattr(runner, "_shared_experts", None)
        shared_layer = getattr(shared_experts, "_layer", None)
        if shared_layer is not None:
            gate_up_weight = getattr(
                getattr(shared_layer, "gate_up_proj", None),
                "weight",
                None,
            )
            if gate_up_weight is not None:
                gate_up_size = int(gate_up_weight.shape[0])
                stage_shapes["mlp_shared_gate_up_inside"] = (
                    gate_up_size,
                )
                stage_shapes["mlp_shared_activated_inside"] = (
                    gate_up_size // 2,
                )
            expert_gate_weight = getattr(
                getattr(shared_layer, "expert_gate", None),
                "weight",
                None,
            )
            if expert_gate_weight is not None:
                stage_shapes["mlp_shared_gate_inside"] = (
                    int(expert_gate_weight.shape[0]),
                )
        for stage in layer_stages:
            shape = stage_shapes[stage]
            target = self._hsd_graph_v3_tensor_sidecars.get(stage)
            if target is None:
                target = torch.empty(
                    shape,
                    device=device,
                    dtype=dtype,
                )
                self.register_buffer(
                    f"_hsd_graph_v3_tensor_sidecar_{stage}",
                    target,
                    persistent=False,
                )
                self._hsd_graph_v3_tensor_sidecars[stage] = target
        full_attn = getattr(self, "self_attn", None)
        prepare_full_attn = getattr(
            full_attn,
            "prepare_hsd_graph_v3_tensor_sidecars",
            None,
        )
        if prepare_full_attn is not None:
            prepare_full_attn(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        linear_attn = getattr(self, "linear_attn", None)
        prepare_linear_attn = getattr(
            linear_attn,
            "prepare_hsd_graph_v3_tensor_sidecars",
            None,
        )
        if prepare_linear_attn is not None:
            prepare_linear_attn(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

    def _capture_hsd_graph_v3_tensor_sidecar(
        self,
        stage: str,
        tensor: torch.Tensor | None,
    ) -> None:
        if (
            not os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_DIR")
            or not torch.is_tensor(tensor)
        ):
            return
        layer_num = int(self.layer_idx) + 1
        selected_layers = _parse_layer_selector(
            os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_LAYERS")
        )
        if selected_layers is not None and layer_num not in selected_layers:
            return
        selected_stages = _parse_csv_selector(
            os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_STAGES")
        )
        if selected_stages is not None and stage not in selected_stages:
            return

        target = self._hsd_graph_v3_tensor_sidecars.get(stage)
        if target is None:
            if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "hsd_graph_v3_tensor_sidecar_missing_before_capture: "
                    f"layer={layer_num} stage={stage} shape={tuple(tensor.shape)}"
                )
            target = torch.empty_like(tensor[0])
            self.register_buffer(
                f"_hsd_graph_v3_tensor_sidecar_{stage}",
                target,
                persistent=False,
            )
            self._hsd_graph_v3_tensor_sidecars[stage] = target
        capture_hsd_graph_v3_sidecar_row0(tensor, target)


    def forward_hsd_graph_v3(
        self,
        *,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        state_indices: torch.Tensor,
        history_state_indices: torch.Tensor,
        past_lens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trace_layer_outputs = (
            _hsd_layer_output_trace_requested()
            and not _env_truthy(
                "VLLM_HSD_DISABLE_GRAPH_V3_LAYER_OUTPUT_TRACE"
            )
        )
        input_full = hidden_states + residual if residual is not None else hidden_states
        self._capture_hsd_graph_v3_tensor_sidecar("decoder_input", hidden_states)
        self._capture_hsd_graph_v3_tensor_sidecar(
            "decoder_input_full", input_full
        )
        if trace_layer_outputs:
            self._trace_hsd_graph_v3_layer_stage(
                "decoder_input",
                hidden_states,
                state_indices=state_indices,
                past_lens=past_lens,
                valid_mask=valid_mask,
            )
            self._trace_hsd_graph_v3_layer_stage(
                "decoder_input_full",
                input_full,
                state_indices=state_indices,
                past_lens=past_lens,
                valid_mask=valid_mask,
            )
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states, None)
            assert torch.is_tensor(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        self._capture_hsd_graph_v3_tensor_sidecar(
            "input_layernorm_out", hidden_states
        )
        if trace_layer_outputs:
            self._trace_hsd_graph_v3_layer_stage(
                "input_layernorm_out",
                hidden_states,
                state_indices=state_indices,
                past_lens=past_lens,
                valid_mask=valid_mask,
            )
            self._trace_hsd_graph_v3_layer_stage(
                "residual_after_input_layernorm",
                residual,
                state_indices=state_indices,
                past_lens=past_lens,
                valid_mask=valid_mask,
            )

        attention_output = torch.empty_like(hidden_states)
        hsd_attn = self._active_hsd_attn()
        if hsd_attn is not None:
            hsd_attn.forward_hsd_graph_v3(
                hidden_states=hidden_states,
                output=attention_output,
                state_indices=state_indices,
                history_state_indices=history_state_indices,
                past_lens=past_lens,
                valid_mask=valid_mask,
            )
            _hsd_graph_v3_debug_sync("linear_attention")
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=attention_output,
                positions=positions,
                capture_hsd_graph_v3_sidecar=True,
                hsd_graph_v3_layer_name=_encode_layer_name(
                    self.hsd_graph_v3_layer_name
                ),
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = attention_output
        self._capture_hsd_graph_v3_tensor_sidecar(
            "attention_raw_out", hidden_states
        )
        if trace_layer_outputs:
            self._trace_hsd_graph_v3_layer_stage(
                "attention_raw_out",
                hidden_states,
                state_indices=state_indices,
                past_lens=past_lens,
                valid_mask=valid_mask,
            )

        if self.layer_scale:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
            )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        self._capture_hsd_graph_v3_tensor_sidecar(
            "post_attention_layernorm_out", hidden_states
        )
        if trace_layer_outputs:
            self._trace_hsd_graph_v3_layer_stage(
                "post_attention_layernorm_out",
                hidden_states,
                state_indices=state_indices,
                past_lens=past_lens,
                valid_mask=valid_mask,
            )
        hidden_states = self.mlp(hidden_states)
        self._capture_hsd_graph_v3_tensor_sidecar("mlp_out", hidden_states)
        if trace_layer_outputs:
            self._trace_hsd_graph_v3_layer_stage(
                "mlp_out",
                hidden_states,
                state_indices=state_indices,
                past_lens=past_lens,
                valid_mask=valid_mask,
            )
        if self.layer_scale:
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        decoder_output = hidden_states + residual
        self._capture_hsd_graph_v3_tensor_sidecar(
            "decoder_output", decoder_output
        )
        if trace_layer_outputs:
            self._trace_hsd_graph_v3_layer_stage(
                "decoder_output",
                decoder_output,
                state_indices=state_indices,
                past_lens=past_lens,
                valid_mask=valid_mask,
            )
        _hsd_graph_v3_debug_sync("layer")
        return hidden_states, residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        hsd_layout: HistoricalHSDLayout | None = None,
        **kwargs: object,
    ):
        if hsd_layout is None:
            return Qwen3NextDecoderLayer.forward(
                self,
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                **kwargs,
            )

        decode_trace_requested = _hsd_decode_trace_requested()
        trace_requested = _hsd_trace_requested() or decode_trace_requested
        if trace_requested:
            decode_trace_backend = (
                (
                    "decoder_decode"
                    if all(
                        int(past_text_len) > 0 and int(text_len) == 1
                        for past_text_len, text_len in zip(
                            hsd_layout.past_text_lens,
                            hsd_layout.req_text_lens,
                        )
                    )
                    else "decoder_prefill"
                )
                if decode_trace_requested
                else ""
            )

            def trace_decode_text_stage(
                stage: str,
                text_tensor: torch.Tensor | None,
            ) -> None:
                if not decode_trace_requested or not torch.is_tensor(text_tensor):
                    return
                for req_idx, (text_start, text_len) in enumerate(
                    zip(hsd_layout.req_text_starts, hsd_layout.req_text_lens)
                ):
                    text_len = int(text_len)
                    if text_len <= 0:
                        continue
                    req_anchor_start = hsd_layout.req_anchor_starts[req_idx]
                    req_anchor_len = hsd_layout.req_anchor_lens[req_idx]
                    boundary_counts = hsd_layout.anchor_boundaries[
                        req_anchor_start : req_anchor_start + req_anchor_len
                    ]
                    trace_vllm_hsd_decode_stage(
                        int(self.layer_idx),
                        backend=decode_trace_backend,
                        stage=stage,
                        tensor=text_tensor[text_start : text_start + text_len].view(
                            1, text_len, -1
                        ),
                        req_idx=req_idx,
                        state_idx=int(hsd_layout.state_indices_list[req_idx]),
                        past_text_len=int(hsd_layout.past_text_lens[req_idx]),
                        anchor_count=int(req_anchor_len),
                        boundary_counts=boundary_counts,
                    )

            input_full = hidden_states + residual if residual is not None else hidden_states
            input_full_fp32 = (
                hidden_states.float() + residual.float()
                if residual is not None
                else hidden_states.float()
            )
            decoder_input = hsd_layout.text_hidden(hidden_states)
            decoder_input_full = hsd_layout.text_hidden(input_full)
            decoder_input_full_fp32 = hsd_layout.text_hidden(input_full_fp32)
            residual_before_input_layernorm = (
                hsd_layout.text_hidden(residual) if residual is not None else None
            )
            trace_vllm_hsd_internal(
                int(self.layer_idx),
                "decoder_input",
                decoder_input,
            )
            trace_decode_text_stage("decoder_input", decoder_input)
            trace_vllm_hsd_internal(
                int(self.layer_idx),
                "residual_before_input_layernorm",
                residual_before_input_layernorm,
            )
            trace_decode_text_stage(
                "residual_before_input_layernorm",
                residual_before_input_layernorm,
            )
            trace_vllm_hsd_internal(
                int(self.layer_idx),
                "decoder_input_full",
                decoder_input_full,
            )
            trace_decode_text_stage("decoder_input_full", decoder_input_full)
            trace_vllm_hsd_internal(
                int(self.layer_idx),
                "decoder_input_full_fp32",
                decoder_input_full_fp32,
            )
            trace_decode_text_stage("decoder_input_full_fp32", decoder_input_full_fp32)
        if _hsd_layer_output_trace_requested():
            input_full = hidden_states + residual if residual is not None else hidden_states
            self._trace_hsd_layer_stage(
                "decoder_input",
                hsd_layout.text_hidden(hidden_states),
                hsd_layout,
            )
            self._trace_hsd_layer_stage(
                "decoder_input_full",
                hsd_layout.text_hidden(input_full),
                hsd_layout,
            )
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if _hsd_layer_output_trace_requested():
            self._trace_hsd_layer_stage(
                "input_layernorm_out",
                hsd_layout.text_hidden(hidden_states),
                hsd_layout,
            )
            self._trace_hsd_layer_stage(
                "residual_after_input_layernorm",
                hsd_layout.text_hidden(residual),
                hsd_layout,
            )
        if trace_requested:
            input_layernorm_out = hsd_layout.text_hidden(hidden_states)
            residual_after_input_layernorm = hsd_layout.text_hidden(residual)
            trace_vllm_hsd_internal(
                int(self.layer_idx),
                "input_layernorm_out",
                input_layernorm_out,
            )
            trace_decode_text_stage("input_layernorm_out", input_layernorm_out)
            trace_vllm_hsd_internal(
                int(self.layer_idx),
                "residual_after_input_layernorm",
                residual_after_input_layernorm,
            )
            trace_decode_text_stage(
                "residual_after_input_layernorm",
                residual_after_input_layernorm,
            )

        self_attention_output = torch.empty_like(hidden_states)
        hsd_attn = self._active_hsd_attn()
        if hsd_attn is not None:
            hsd_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                hsd_layout=hsd_layout,
            )
        elif self.layer_type == "full_attention":
            if positions is None:
                raise ValueError("HSD full-attention path requires text positions.")
            text_hidden_states = hsd_layout.text_hidden(hidden_states)
            prefix_projection_pad_rows = 0
            if _env_truthy("VLLM_HSD_PREFIX_CACHE_PAD_SUFFIX_TO_FULL"):
                prefix_projection_pad_rows = sum(
                    int(past_text_len)
                    for restored, past_text_len, text_len in zip(
                        hsd_layout.hsd_prefix_restored_flags,
                        hsd_layout.past_text_lens,
                        hsd_layout.req_text_lens,
                        strict=True,
                    )
                    if restored and int(past_text_len) > 0 and int(text_len) > 0
                )
            if _hsd_layer_output_trace_requested():
                self._trace_hsd_layer_stage(
                    "attention_input",
                    text_hidden_states,
                    hsd_layout,
                )
            text_positions = hsd_layout.text_positions()
            text_output = torch.empty_like(text_hidden_states)
            self.self_attn(
                hidden_states=text_hidden_states,
                output=text_output,
                positions=text_positions,
                hsd_prefix_pad_rows=prefix_projection_pad_rows,
            )
            if _hsd_layer_output_trace_requested():
                self._trace_hsd_layer_stage(
                    "attention_raw_out",
                    text_output,
                    hsd_layout,
                )
            self_attention_output = hsd_layout.scatter_text_hidden(
                hidden_states, text_output
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output
        if _hsd_layer_output_trace_requested() and hsd_attn is not None:
            self._trace_hsd_layer_stage(
                "attention_raw_out",
                hsd_layout.text_hidden(hidden_states),
                hsd_layout,
            )
        if trace_requested:
            attention_raw_out = hsd_layout.text_hidden(hidden_states)
            trace_vllm_hsd_internal(
                int(self.layer_idx),
                "attention_raw_out",
                attention_raw_out,
            )
            trace_decode_text_stage("attention_raw_out", attention_raw_out)

        if self.layer_scale:
            scale = self.attn_layer_scale.to(hidden_states.dtype)
            hidden_states = hidden_states * (scale[0] + 1)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if _hsd_layer_output_trace_requested():
            self._trace_hsd_layer_stage(
                "post_attention_layernorm_out",
                hsd_layout.text_hidden(hidden_states),
                hsd_layout,
            )
        if trace_requested:
            post_attention_layernorm_out = hsd_layout.text_hidden(hidden_states)
            trace_vllm_hsd_internal(
                int(self.layer_idx),
                "post_attention_layernorm_out",
                post_attention_layernorm_out,
            )
            trace_decode_text_stage(
                "post_attention_layernorm_out",
                post_attention_layernorm_out,
            )
        hidden_states = _run_hsd_layer_mlp_delta(
            self,
            hidden_states=hidden_states,
            layout=hsd_layout,
            )
        if _hsd_layer_output_trace_requested():
            self._trace_hsd_layer_stage(
                "mlp_out",
                hsd_layout.text_hidden(hidden_states),
                hsd_layout,
            )
        if trace_requested:
            mlp_out = hsd_layout.text_hidden(hidden_states)
            trace_vllm_hsd_internal(
                int(self.layer_idx),
                "mlp_out",
                mlp_out,
            )
            trace_decode_text_stage("mlp_out", mlp_out)

        if self.layer_scale:
            scale = self.ffn_layer_scale.to(hidden_states.dtype)
            hidden_states = hidden_states * (scale[0] + 1)

        if _hsd_layer_output_trace_requested():
            decoder_output = hidden_states + residual
            self._trace_hsd_layer_stage(
                "decoder_output",
                hsd_layout.text_hidden(decoder_output),
                hsd_layout,
            )
        if trace_requested:
            decoder_output = hidden_states + residual
            decoder_output_text = hsd_layout.text_hidden(decoder_output)
            trace_vllm_hsd_internal(
                int(self.layer_idx),
                "decoder_output",
                decoder_output_text,
            )
            trace_decode_text_stage("decoder_output", decoder_output_text)
            trace_vllm_hsd_internal(
                int(self.layer_idx),
                "decoder_output_full",
                decoder_output_text,
            )
            trace_decode_text_stage("decoder_output_full", decoder_output_text)
        return hidden_states, residual


@support_torch_compile(
    enable_if=_historical_hsd_enable_torch_compile,
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hsd_num_computed_tokens": 0,
        "hsd_state_indices": 0,
        "hsd_history_state_indices": 0,
        "hsd_decode_valid_mask": 0,
    }
)
class Safin1Model(ModelWeightLoadingMixin, ModelDiagnosticsMixin, Qwen3_5Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        config: Safin1TextConfig | Safin1MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        self.hsd_persistent_memory_mode = _configure_persistent_memory_mode(
            config,
            vllm_config,
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.enable_lora = vllm_config.lora_config is not None
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        if (
            _anchor_enabled(config)
            and str(getattr(config, "anchor_embedding_type", "learned")) == "learned"
        ):
            self.anchor_embedding = nn.Embedding(
                int(config.anchor_token_num), config.hidden_size
            )

        def get_layer(prefix: str):
            return Safin1DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        first_hsd_linear_layer = True
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hsd_attn = layer._active_hsd_attn()
            if hsd_attn is not None:
                hsd_attn.hsd_graph_v3_first_linear_layer = first_hsd_linear_layer
                first_hsd_linear_layer = False
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self._hsd_prefix_snapshot_ready: list[str] = []
        self._hsd_prefix_snapshot_evicted: list[str] = []
        if os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_DIR"):
            max_batch = max(
                1,
                int(
                    _env_int(
                        "VLLM_HSD_PADDED_STATIC_CAPTURE_MAX_BATCH",
                        1,
                    )
                    or 1
                ),
            )
            for batch_size in range(1, max_batch + 1):
                self.prepare_hsd_graph_v3_tensor_sidecars(
                    batch_size=batch_size
                )

    def _iter_hsd_linear_attn_modules(self):
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hsd_attn = layer._active_hsd_attn()
            if hsd_attn is not None:
                yield hsd_attn

    def ensure_hsd_graph_v3_state_sidecars(self) -> bool:
        modules = list(self._iter_hsd_linear_attn_modules())
        return bool(modules) and all(
            module.ensure_hsd_graph_v3_state_sidecars() for module in modules
        )

    def prepare_hsd_graph_v3_tensor_sidecars(self, *, batch_size: int) -> None:
        if not os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_DIR"):
            return
        weight = self.embed_tokens.weight
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            prepare = getattr(
                layer,
                "prepare_hsd_graph_v3_tensor_sidecars",
                None,
            )
            if prepare is not None:
                prepare(
                    batch_size=batch_size,
                    device=weight.device,
                    dtype=weight.dtype,
                )

    def dump_hsd_graph_v3_tensor_sidecars(
        self,
        *,
        state_indices: torch.Tensor,
        past_lens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        dump_dir = os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_DIR")
        if not dump_dir:
            return
        step = int(getattr(self, "_hsd_graph_v3_tensor_sidecar_dump_step", 0))
        self._hsd_graph_v3_tensor_sidecar_dump_step = step + 1
        global_rank = (
            int(torch.distributed.get_rank())
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else 0
        )
        payload: dict[str, object] = {
            "step": step,
            "global_rank": global_rank,
            "state_indices": state_indices.detach().cpu(),
            "past_lens": past_lens.detach().cpu(),
            "valid_mask": valid_mask.detach().cpu(),
            "captured_row": int(
                os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_ROW", "0")
            ),
            "layers": {},
        }
        layers: dict[int, dict[str, torch.Tensor]] = {}
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            sidecars = getattr(layer, "_hsd_graph_v3_tensor_sidecars", None)
            hsd_attention_sidecars = getattr(
                layer._active_hsd_attn(),
                "_hsd_graph_v3_tensor_sidecars",
                None,
            )
            selected: dict[str, torch.Tensor] = {}
            for source in (
                sidecars,
                hsd_attention_sidecars,
            ):
                if not isinstance(source, dict):
                    continue
                selected.update(
                    {
                        str(stage): tensor.unsqueeze(0).detach().cpu()
                        for stage, tensor in source.items()
                        if torch.is_tensor(tensor)
                        and tensor.ndim == 1
                    }
                )
            if selected:
                layers[int(layer.layer_idx) + 1] = selected
        payload["layers"] = layers
        os.makedirs(dump_dir, exist_ok=True)
        torch.save(
            payload,
            os.path.join(
                dump_dir,
                f"rank{global_rank}__pid{os.getpid()}__step{step:04d}.pt",
            ),
        )

    def dump_hsd_graph_v3_state_sidecars(
        self,
        *,
        phase: str,
        state_indices: torch.Tensor,
        past_lens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        dump_dir = os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_DIR")
        if not dump_dir or not _env_truthy(
            "VLLM_HSD_GRAPH_V3_STATE_SIDECAR"
        ):
            return
        valid_rows = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if valid_rows.numel() == 0:
            return
        selected_state_indices = state_indices.index_select(
            0, valid_rows
        ).long()
        selected_layers = _parse_layer_selector(
            os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_LAYERS")
        )
        layers: dict[int, dict[str, torch.Tensor]] = {}
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer_num = int(layer.layer_idx) + 1
            if selected_layers is not None and layer_num not in selected_layers:
                continue
            sidecar = getattr(
                layer._active_hsd_attn(),
                "hsd_persistent_state_sidecar",
                None,
            )
            if not isinstance(sidecar, dict):
                continue
            selected: dict[str, torch.Tensor] = {}
            for name in ("conv_state", "recurrent_state"):
                tensor = sidecar.get(name)
                if not torch.is_tensor(tensor):
                    continue
                selected[name] = tensor.index_select(
                    0,
                    selected_state_indices.to(tensor.device),
                ).detach().cpu()
            if selected:
                layers[layer_num] = selected
        step_attr = f"_hsd_graph_v3_state_sidecar_{phase}_step"
        step = int(getattr(self, step_attr, 0))
        setattr(self, step_attr, step + 1)
        global_rank = (
            int(torch.distributed.get_rank())
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else 0
        )
        os.makedirs(dump_dir, exist_ok=True)
        torch.save(
            {
                "phase": phase,
                "step": step,
                "global_rank": global_rank,
                "state_indices": selected_state_indices.detach().cpu(),
                "past_lens": past_lens.index_select(
                    0, valid_rows
                ).detach().cpu(),
                "layers": layers,
            },
            os.path.join(
                dump_dir,
                f"state_{phase}__rank{global_rank}__pid{os.getpid()}"
                f"__step{step:04d}.pt",
            ),
        )


    def _begin_hsd_prefix_snapshot_step(self) -> None:
        self._hsd_prefix_snapshot_ready = []
        self._hsd_prefix_snapshot_evicted = []
        for module in self._iter_hsd_linear_attn_modules():
            module.hsd_prefix_snapshot_step_ready_keys = set()
            module.hsd_prefix_snapshot_step_evicted_keys = set()

    def _finish_hsd_prefix_snapshot_step(
        self,
        layout: HistoricalHSDLayout,
    ) -> None:
        modules = list(self._iter_hsd_linear_attn_modules())
        if not modules:
            return
        requested_keys = {
            key
            for req_items in (layout.hsd_prefix_snapshot_requests or [])
            for key, _, _ in req_items
        }
        evicted_keys = {
            key
            for module in modules
            for key in getattr(
                module, "hsd_prefix_snapshot_step_evicted_keys", set()
            )
        }
        self._hsd_prefix_snapshot_evicted.extend(
            sorted(
                key
                for key in evicted_keys
                if not all(
                    key in module.hsd_prefix_snapshot_cache for module in modules
                )
            )
        )
        if not requested_keys:
            return
        ready: list[str] = []
        for key in sorted(requested_keys):
            if all(
                key in getattr(module, "hsd_prefix_snapshot_step_ready_keys", set())
                and key in module.hsd_prefix_snapshot_cache
                for module in modules
            ):
                ready.append(key)
            else:
                recorder = getattr(modules[0], "_record_hsd_varlen_event", None)
                if recorder is not None:
                    recorder(
                        "hsd_prefix_cache_fallback",
                        reason="snapshot_store_failed",
                        key=key,
                    )
        self._hsd_prefix_snapshot_ready.extend(ready)

    def take_hsd_prefix_snapshot_ready(self) -> list[str]:
        ready = list(dict.fromkeys(self._hsd_prefix_snapshot_ready))
        self._hsd_prefix_snapshot_ready = []
        return ready

    def take_hsd_prefix_snapshot_evicted(self) -> list[str]:
        evicted = list(dict.fromkeys(self._hsd_prefix_snapshot_evicted))
        self._hsd_prefix_snapshot_evicted = []
        return evicted

    def capture_hsd_prefix_snapshot(
        self,
        *,
        key: str,
        state_idx: int,
        prefix_len: int,
    ) -> bool:
        modules = list(self._iter_hsd_linear_attn_modules())
        if not modules:
            return False
        ok = True
        for module in modules:
            ok = bool(
                module.capture_hsd_prefix_snapshot(
                    key=key,
                    state_idx=state_idx,
                    prefix_len=prefix_len,
                )
            ) and ok
        return ok

    def restore_hsd_prefix_snapshot(
        self,
        *,
        key: str,
        state_idx: int,
        history_state_idx: int | None = None,
        prefix_len: int,
    ) -> bool:
        modules = list(self._iter_hsd_linear_attn_modules())
        if not modules:
            return False
        for module in modules:
            reason = module._hsd_prefix_snapshot_restore_preflight(
                key=key,
                state_idx=state_idx,
                history_state_idx=history_state_idx,
                prefix_len=prefix_len,
            )
            if reason is not None:
                page_cache = getattr(module, "hsd_anchor_page_cache", None)
                free_pages = (
                    len(page_cache.free_pages) if page_cache is not None else -1
                )
                total_pages = (
                    int(page_cache.total_pages) if page_cache is not None else -1
                )
                raise RuntimeError(
                    "hsd_prefix_snapshot_restore_preflight_failed: "
                    f"reason={reason}, key={key!r}, layer={module.layer_idx}, "
                    f"state_idx={state_idx}, history_state_idx="
                    f"{history_state_idx}, free_pages={free_pages}, "
                    f"total_pages={total_pages}."
                )
        ok = True
        for module in modules:
            ok = bool(
                module.restore_hsd_prefix_snapshot(
                    key=key,
                    state_idx=state_idx,
                    history_state_idx=history_state_idx,
                    prefix_len=prefix_len,
                )
            ) and ok
        return ok

    def prepare_hsd_static_decode_sidecars(
        self,
        *,
        state_indices: torch.Tensor,
        valid_mask: torch.Tensor | None,
        past_lens: torch.Tensor | None = None,
    ) -> bool:
        modules = list(self._iter_hsd_linear_attn_modules())
        if not modules:
            return False
        ok = True
        for module in modules:
            prepare = getattr(module, "prepare_hsd_static_decode_sidecar", None)
            ok = bool(
                prepare(
                    state_indices=state_indices,
                    valid_mask=valid_mask,
                    past_lens=past_lens,
                )
                if prepare is not None
                else False
            ) and ok
        return ok

    def sync_hsd_static_decode_sidecars_to_text_cache(
        self,
        *,
        state_indices: torch.Tensor,
    ) -> None:
        for module in self._iter_hsd_linear_attn_modules():
            sync = getattr(
                module,
                "sync_hsd_static_decode_sidecar_to_text_cache",
                None,
            )
            if sync is not None:
                sync(state_indices=state_indices)

    def _build_layout_from_kwargs(
        self,
        positions: torch.Tensor,
        hsd_query_start_loc: torch.Tensor | None,
        hsd_num_computed_tokens: torch.Tensor | None,
        hsd_state_indices: torch.Tensor | None,
        hsd_history_state_indices: torch.Tensor | None = None,
        hsd_prefix_snapshot_requests: list[list[tuple[str, int, int]]] | None = None,
        hsd_prefix_restored_flags: list[bool] | None = None,
    ) -> HistoricalHSDLayout:
        if hsd_query_start_loc is None or hsd_num_computed_tokens is None:
            raise _native_hsd_forward_error()
        return _build_hsd_layout(
            config=self.config,
            positions=positions,
            query_start_loc=hsd_query_start_loc,
            num_computed_tokens=hsd_num_computed_tokens,
            state_indices=hsd_state_indices,
            history_state_indices=hsd_history_state_indices,
            hsd_prefix_snapshot_requests=hsd_prefix_snapshot_requests,
            hsd_prefix_restored_flags=hsd_prefix_restored_flags,
        )

    def _build_padded_static_decode_layout(
        self,
        *,
        positions: torch.Tensor,
        hsd_num_computed_tokens: torch.Tensor,
        hsd_state_indices: torch.Tensor,
        hsd_history_state_indices: torch.Tensor | None,
        hsd_decode_valid_mask: torch.Tensor,
    ) -> HistoricalHSDLayout:
        if hsd_history_state_indices is None:
            hsd_history_state_indices = hsd_state_indices
        return _build_hsd_padded_static_decode_layout(
            config=self.config,
            positions=positions,
            hsd_num_computed_tokens=hsd_num_computed_tokens,
            hsd_state_indices=hsd_state_indices,
            hsd_history_state_indices=hsd_history_state_indices,
            hsd_decode_valid_mask=hsd_decode_valid_mask,
        )

    def _expand_inputs_with_anchor(
        self,
        text_hidden_states: torch.Tensor,
        layout: HistoricalHSDLayout,
    ) -> torch.Tensor:
        if int(text_hidden_states.shape[0]) != layout.num_text_tokens:
            raise ValueError(
                "hsd_layout_input_padding_mismatch: "
                f"text_hidden_tokens={int(text_hidden_states.shape[0])}, "
                f"layout_text_tokens={layout.num_text_tokens}, "
                f"layout_expanded_tokens={layout.num_expanded_tokens}. "
                "Historical HSD prefill/mixed/anchor-boundary batches must "
                "run without cudagraph padding; force CUDAGraphMode.NONE for "
                "non-graphable HSD batches."
            )
        expanded = text_hidden_states.new_zeros(
            layout.num_expanded_tokens, text_hidden_states.shape[-1]
        )
        expanded.index_copy_(0, layout.text_to_expanded, text_hidden_states)
        if not layout.has_anchor_tokens:
            return expanded

        anchor_positions = layout.anchor_positions
        anchor_embedding_type = str(
            getattr(self.config, "anchor_embedding_type", "learned")
        )
        if anchor_embedding_type == "learned":
            if not hasattr(self, "anchor_embedding"):
                raise RuntimeError("learned HSD anchors require anchor_embedding.")
            anchor_hidden = self.anchor_embedding(
                layout.anchor_ids.to(self.anchor_embedding.weight.device)
            )
            anchor_hidden = anchor_hidden.to(
                device=expanded.device, dtype=expanded.dtype
            )
        elif anchor_embedding_type == "chunk_end":
            anchor_hidden = text_hidden_states.index_select(
                0, layout.anchor_text_indices
            )
        elif anchor_embedding_type == "placeholder":
            placeholder_id = int(
                getattr(self.config, "anchor_placeholder_token_id", 0) or 0
            )
            placeholder_ids = torch.full(
                (anchor_positions.numel(),),
                placeholder_id,
                dtype=torch.long,
                device=expanded.device,
            )
            anchor_hidden = self.embed_input_ids(placeholder_ids)
        else:
            raise ValueError(
                f"Unsupported HSD anchor_embedding_type: {anchor_embedding_type}"
            )
        expanded.index_copy_(0, anchor_positions, anchor_hidden)
        return expanded

    def _maybe_add_hsd_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        layout: HistoricalHSDLayout,
    ) -> None:
        if layer_idx not in self.aux_hidden_state_layers:
            return
        value = hidden_states + residual if residual is not None else hidden_states
        aux_hidden_states.append(layout.text_hidden(value))


    def _forward_hsd_graph_v3(
        self,
        *,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        state_indices: torch.Tensor,
        history_state_indices: torch.Tensor,
        past_lens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = None
        last_layer = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            last_layer = layer
            hidden_states, residual = layer.forward_hsd_graph_v3(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                state_indices=state_indices,
                history_state_indices=history_state_indices,
                past_lens=past_lens,
                valid_mask=valid_mask,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if last_layer is not None:
            last_layer._capture_hsd_graph_v3_tensor_sidecar(
                "final_norm_out", hidden_states
            )
        _hsd_graph_v3_debug_sync("model")
        return hidden_states.masked_fill(~valid_mask.view(-1, 1), 0)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        hsd_query_start_loc: torch.Tensor | None = None,
        hsd_num_computed_tokens: torch.Tensor | None = None,
        hsd_state_indices: torch.Tensor | None = None,
        hsd_history_state_indices: torch.Tensor | None = None,
        hsd_decode_valid_mask: torch.Tensor | None = None,
        hsd_prefix_snapshot_requests: list[list[tuple[str, int, int]]] | None = None,
        hsd_prefix_restored_flags: list[bool] | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if _anchor_enabled(self.config):
            if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
                raise NotImplementedError(
                    "Safin-1 vLLM HSD anchor path currently supports "
                    "tensor parallel inference without pipeline parallelism."
                )
            text_positions = _text_only_hsd_positions(positions)
            if inputs_embeds is not None:
                text_hidden_states = inputs_embeds
            else:
                text_hidden_states = self.embed_input_ids(input_ids)
            use_padded_static_decode = (
                hsd_decode_valid_mask is not None
                and hsd_num_computed_tokens is not None
                and hsd_state_indices is not None
                and _env_truthy("VLLM_HSD_ENABLE_PADDED_STATIC_DECODE")
                and not (
                    (
                        _hsd_trace_requested()
                        and not _env_truthy("VLLM_HSD_TRACE_PREFILL_ONLY")
                    )
                    or _hsd_decode_trace_requested()
                )
                and int(text_hidden_states.shape[0]) == int(text_positions.shape[-1])
            )
            if use_padded_static_decode and _env_truthy(
                "VLLM_HSD_ENABLE_GRAPH_V3"
            ):
                assert hsd_num_computed_tokens is not None
                assert hsd_state_indices is not None
                assert hsd_decode_valid_mask is not None
                history_state_indices = (
                    hsd_history_state_indices
                    if hsd_history_state_indices is not None
                    else hsd_state_indices
                )
                with hsd_moe_dp_metadata(int(text_hidden_states.shape[0])):
                    return self._forward_hsd_graph_v3(
                        hidden_states=text_hidden_states,
                        positions=text_positions,
                        state_indices=hsd_state_indices,
                        history_state_indices=history_state_indices,
                        past_lens=hsd_num_computed_tokens,
                        valid_mask=hsd_decode_valid_mask,
                    )
            if use_padded_static_decode:
                layout = self._build_padded_static_decode_layout(
                    positions=text_positions,
                    hsd_num_computed_tokens=hsd_num_computed_tokens,
                    hsd_state_indices=hsd_state_indices,
                    hsd_history_state_indices=hsd_history_state_indices,
                    hsd_decode_valid_mask=hsd_decode_valid_mask,
                )
                hidden_states = text_hidden_states
            else:
                layout = self._build_layout_from_kwargs(
                    positions=text_positions,
                    hsd_query_start_loc=hsd_query_start_loc,
                    hsd_num_computed_tokens=hsd_num_computed_tokens,
                    hsd_state_indices=hsd_state_indices,
                    hsd_history_state_indices=hsd_history_state_indices,
                    hsd_prefix_snapshot_requests=hsd_prefix_snapshot_requests,
                    hsd_prefix_restored_flags=hsd_prefix_restored_flags,
                )
                hidden_states = self._expand_inputs_with_anchor(
                    text_hidden_states, layout
                )
            residual = None
            if _hsd_layer_output_trace_requested():
                self._trace_hsd_layer_output_stage(
                    0,
                    "model_input",
                    layout.text_hidden(hidden_states),
                    layout,
                )

            self._begin_hsd_prefix_snapshot_step()
            aux_hidden_states: list[torch.Tensor] = []
            self._maybe_add_hsd_hidden_state(
                aux_hidden_states, 0, hidden_states, residual, layout
            )
            with hsd_moe_dp_metadata(int(hidden_states.shape[0])):
                for layer_idx, layer in enumerate(
                    islice(self.layers, self.start_layer, self.end_layer),
                    start=self.start_layer,
                ):
                    hidden_states, residual = layer(
                        positions=text_positions,
                        hidden_states=hidden_states,
                        residual=residual,
                        hsd_layout=layout,
                    )
                    self._maybe_add_hsd_hidden_state(
                        aux_hidden_states, layer_idx + 1, hidden_states, residual, layout
                    )
                    if _hsd_layer_output_trace_requested():
                        layer_output = (
                            hidden_states + residual
                            if residual is not None
                            else hidden_states
                        )
                        self._trace_hsd_layer_output_stage(
                            layer_idx + 1,
                            "decoder_output",
                            layout.text_hidden(layer_output),
                            layout,
                        )

            self._finish_hsd_prefix_snapshot_step(layout)
            if _hsd_decode_trace_requested():
                final_norm_input = (
                    hidden_states + residual if residual is not None else hidden_states
                )
                self._trace_hsd_model_stage(
                    "final_norm_input",
                    layout.text_hidden(final_norm_input),
                    layout,
                )
            if _hsd_trace_requested():
                final_trace_layer = int(
                    getattr(self.config, "num_hidden_layers", 0) or 0
                )
                final_norm_input = (
                    hidden_states + residual if residual is not None else hidden_states
                )
                trace_vllm_hsd_internal(
                    final_trace_layer,
                    "final_norm_input",
                    layout.text_hidden(final_norm_input),
                )
            hidden_states, _ = self.norm(hidden_states, residual)
            hidden_states = layout.text_hidden(hidden_states)
            if layout.padded_static_decode and layout.decode_valid_mask is not None:
                hidden_states = hidden_states.masked_fill(
                    ~layout.decode_valid_mask.view(-1, 1),
                    0,
                )
            if _hsd_final_output_trace_requested():
                self._trace_hsd_final_output_stage(
                    "final_norm_out",
                    hidden_states,
                    layout,
                )
            if _hsd_layer_output_trace_requested():
                self._trace_hsd_layer_output_stage(
                    int(getattr(self.config, "num_hidden_layers", 0) or 0) + 1,
                    "final_norm_out",
                    hidden_states,
                    layout,
                )
            if _hsd_decode_trace_requested():
                self._trace_hsd_model_stage("final_norm_out", hidden_states, layout)
                self._last_hsd_trace_layout = layout
            if _hsd_trace_requested():
                final_trace_layer = int(
                    getattr(self.config, "num_hidden_layers", 0) or 0
                )
                trace_vllm_hsd_internal(
                    final_trace_layer,
                    "final_norm_out",
                    hidden_states,
                )
            if aux_hidden_states:
                return hidden_states, aux_hidden_states
            return hidden_states
        if _hsd_decode_trace_requested():
            self._last_hsd_trace_layout = None
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )


class Safin1MoeForCausalLM(
    CausalLMWeightLoadingMixin,
    PersistentMemoryAuditMixin,
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsMRoPE,
    SupportsPP,
    QwenNextMixtureOfExperts,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    hf_to_vllm_mapper = _HISTORICAL_HF_TO_VLLM_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config: Safin1TextConfig | Safin1MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        persistent_memory_mode = _configure_persistent_memory_mode(
            config,
            vllm_config,
        )
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        if bool(getattr(config, "hsd_persistent_memory_enabled", False)):
            _validate_persistent_memory_architecture(config)
            if int(vllm_config.parallel_config.pipeline_parallel_size) != 1:
                raise NotImplementedError(
                    "Persistent HSD memory currently requires pipeline "
                    "parallel size 1."
                )

        if _anchor_enabled(config):
            _validate_native_hsd_config(config)
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Safin-1 currently does not support 'all' prefix "
                "caching; experimental HSD snapshots require '--mamba-cache-mode=none'."
            )
        _validate_persistent_memory_prefix_cache(config, cache_config)
        if _anchor_enabled(config):
            validate_march_prefix_cache(config, vllm_config)
        if vllm_config.lora_config is not None:
            raise NotImplementedError(
                "Safin-1 vLLM path does not support LoRA."
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Safin-1 vLLM path does not support speculative decoding."
            )

        super().__init__()
        self.config = config
        self.hsd_persistent_memory_mode = persistent_memory_mode
        self.scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config
        self.model = Safin1Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
            if config.model_type == "safin1_35b_a3b_text":
                self.lm_head._hsd_moe_batch_invariant_linear = True
                self.lm_head._hsd_moe_batch_invariant_component = "lm_head"
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        if config.model_type == "safin1_35b_a3b_text":
            self.set_moe_parameters()
        else:
            self.expert_weights = []
            self.moe_layers = []
            self.num_moe_layers = 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[object],
    ) -> tuple[torch.Tensor, int]:
        if mm_features:
            raise NotImplementedError(
                "Safin-1 vLLM HSD path is text-only and does not "
                "support multimodal M-RoPE inputs."
            )
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1).contiguous(), 0


    def hsd_history_cache_budget_for_logical_pages(
        self,
        num_pages: int,
        page_m: int = 16,
    ) -> int:
        """Return the rank-local bytes required for logical history pages."""
        if int(num_pages) <= 0:
            return 0
        modules = [
            module
            for module in self.modules()
            if isinstance(module, HistoricalGatedDeltaNetAttention)
            and module.hsd_layer_uses_global_read
            and str(getattr(module, "hsd_history_cache_backend", "auto"))
            .strip()
            .lower()
            in {"auto", "scheduler_paged"}
        ]
        bytes_per_logical_page = sum(
            module.hsd_history_page_size_bytes(page_m=page_m)
            for module in modules
        )
        return int(num_pages) * int(bytes_per_logical_page)

    def initialize_hsd_history_page_cache(
        self,
        *,
        budget_bytes: int,
        device: torch.device,
        page_m: int = 16,
        max_model_len: int | None = None,
    ) -> dict[str, object]:
        modules = [
            module
            for module in self.modules()
            if isinstance(module, HistoricalGatedDeltaNetAttention)
            and module.hsd_layer_uses_global_read
            and str(getattr(module, "hsd_history_cache_backend", "auto")).strip().lower()
            in {"auto", "scheduler_paged"}
        ]
        if budget_bytes <= 0 or not modules:
            return {
                "num_layers": len(modules),
                "num_pages": 0,
                "budget_bytes": int(max(0, budget_bytes)),
                "allocated_bytes": 0,
            }
        bytes_per_logical_page = sum(
            module.hsd_history_page_size_bytes(page_m=page_m) for module in modules
        )
        if bytes_per_logical_page <= 0:
            return {
                "num_layers": len(modules),
                "num_pages": 0,
                "budget_bytes": int(max(0, budget_bytes)),
                "allocated_bytes": 0,
            }
        num_pages = int(budget_bytes) // int(bytes_per_logical_page)
        if num_pages <= 0:
            logger.warning(
                "HSD history cache budget %s bytes is too small for one "
                "logical page (%s bytes); scheduler_paged history read disabled.",
                int(budget_bytes),
                int(bytes_per_logical_page),
            )
            return {
                "num_layers": len(modules),
                "num_pages": 0,
                "budget_bytes": int(max(0, budget_bytes)),
                "allocated_bytes": 0,
            }
        initialized = 0
        for module in modules:
            initialized += module.initialize_hsd_history_page_cache(
                num_pages=num_pages,
                device=device,
                page_m=page_m,
                max_model_len=max_model_len,
            )
        allocated_bytes = int(num_pages) * int(bytes_per_logical_page)
        state_format = str(modules[0].hsd_history_state_format)
        logger.info(
            "Initialized scheduler-paged HSD history cache: layers=%s "
            "logical_pages=%s format=%s "
            "allocated=%.3f GiB budget=%.3f GiB",
            len(modules),
            num_pages,
            state_format,
            allocated_bytes / (1024**3),
            int(budget_bytes) / (1024**3),
        )
        return {
            "num_layers": len(modules),
            "num_pages": int(num_pages),
            "budget_bytes": int(max(0, budget_bytes)),
            "allocated_bytes": allocated_bytes,
            "layer_pages_initialized": int(initialized),
            "state_format": state_format,
            "allocated_page_bytes": int(bytes_per_logical_page),
        }

    def hsd_history_cache_runtime_stats(self) -> dict[str, object]:
        caches = [
            module.hsd_anchor_page_cache
            for module in self.modules()
            if isinstance(module, HistoricalGatedDeltaNetAttention)
            and module.hsd_anchor_page_cache is not None
        ]
        if not caches:
            return {}
        first = caches[0].stats()
        return {
            "state_format": first["state_format"],
            "allocated_page_bytes": sum(
                int(cache.allocated_page_bytes) for cache in caches
            ),
            "history_read_time_us": sum(
                int(cache.history_read_time_us) for cache in caches
            ),
            "materialize_fallback_count": sum(
                int(cache.materialize_fallback_count) for cache in caches
            ),
        }

    def capture_hsd_prefix_snapshot(
        self,
        *,
        key: str,
        state_idx: int,
        prefix_len: int,
    ) -> bool:
        return bool(
            self.model.capture_hsd_prefix_snapshot(
                key=key,
                state_idx=state_idx,
                prefix_len=prefix_len,
            )
        )

    def restore_hsd_prefix_snapshot(
        self,
        *,
        key: str,
        state_idx: int,
        history_state_idx: int | None = None,
        prefix_len: int,
    ) -> bool:
        return bool(
            self.model.restore_hsd_prefix_snapshot(
                key=key,
                state_idx=state_idx,
                history_state_idx=history_state_idx,
                prefix_len=prefix_len,
            )
        )

    def take_hsd_prefix_snapshot_ready(self) -> list[str]:
        return self.model.take_hsd_prefix_snapshot_ready()

    def take_hsd_prefix_snapshot_evicted(self) -> list[str]:
        return self.model.take_hsd_prefix_snapshot_evicted()

    def prepare_hsd_static_decode_sidecars(
        self,
        *,
        state_indices: torch.Tensor,
        valid_mask: torch.Tensor | None,
        past_lens: torch.Tensor | None = None,
    ) -> bool:
        return bool(
            self.model.prepare_hsd_static_decode_sidecars(
                state_indices=state_indices,
                valid_mask=valid_mask,
                past_lens=past_lens,
            )
        )

    def ensure_hsd_graph_v3_state_sidecars(self) -> bool:
        return self.model.ensure_hsd_graph_v3_state_sidecars()

    def prepare_hsd_graph_v3_tensor_sidecars(self, *, batch_size: int) -> None:
        self.model.prepare_hsd_graph_v3_tensor_sidecars(batch_size=batch_size)

    def dump_hsd_graph_v3_tensor_sidecars(
        self,
        *,
        state_indices: torch.Tensor,
        past_lens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        self.model.dump_hsd_graph_v3_tensor_sidecars(
            state_indices=state_indices,
            past_lens=past_lens,
            valid_mask=valid_mask,
        )

    def dump_hsd_graph_v3_state_sidecars(
        self,
        *,
        phase: str,
        state_indices: torch.Tensor,
        past_lens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        self.model.dump_hsd_graph_v3_state_sidecars(
            phase=phase,
            state_indices=state_indices,
            past_lens=past_lens,
            valid_mask=valid_mask,
        )

    def dump_hsd_prefill_passive_sidecars(self) -> None:
        self.model.dump_hsd_prefill_passive_sidecars()

    def sync_hsd_static_decode_sidecars_to_text_cache(
        self,
        *,
        state_indices: torch.Tensor,
    ) -> None:
        self.model.sync_hsd_static_decode_sidecars_to_text_cache(
            state_indices=state_indices
        )

    def remap_hsd_state_slots(self, mapping: dict[int, int]) -> None:
        """Atomically move model-owned HSD side state between cache rows."""
        normalized = {
            int(source): int(target)
            for source, target in mapping.items()
            if int(source) != int(target)
        }
        if not normalized:
            return
        sources = list(normalized)
        targets = list(normalized.values())
        if len(set(targets)) != len(targets):
            raise ValueError("HSD state slot remap targets must be unique.")

        def remap_dict(cache: dict[int, object]) -> None:
            occupied_targets = {
                target
                for target in targets
                if target in cache and target not in normalized
            }
            if occupied_targets:
                raise RuntimeError(
                    "hsd_state_slot_remap_target_occupied: "
                    f"{sorted(occupied_targets)}"
                )
            values = {source: cache.get(source) for source in sources}
            for source in sources:
                cache.pop(source, None)
            for source, target in normalized.items():
                value = values[source]
                if value is not None:
                    cache[target] = value

        for module in self.modules():
            anchor_cache = getattr(module, "hsd_anchor_cache", None)
            if isinstance(anchor_cache, dict):
                remap_dict(anchor_cache)
            text_state_cache = getattr(module, "hsd_text_state_cache", None)
            if isinstance(text_state_cache, dict):
                remap_dict(text_state_cache)
            prefix_recurrent_replay = getattr(
                module, "hsd_prefix_recurrent_replay", None
            )
            if isinstance(prefix_recurrent_replay, dict):
                remap_dict(prefix_recurrent_replay)
            anchor_page_cache = getattr(module, "hsd_anchor_page_cache", None)
            if anchor_page_cache is not None:
                remap_fn = getattr(anchor_page_cache, "remap_state_indices", None)
                if remap_fn is None:
                    raise RuntimeError(
                        "hsd_state_slot_page_cache_remap_unsupported"
                    )
                remap_fn(normalized)

            persistent_state_sidecar = getattr(
                module, "hsd_persistent_state_sidecar", None
            )
            if isinstance(persistent_state_sidecar, dict):
                for tensor in persistent_state_sidecar.values():
                    if not torch.is_tensor(tensor):
                        continue
                    capacity = int(tensor.shape[0])
                    if max((*sources, *targets)) >= capacity:
                        raise RuntimeError(
                            "hsd_state_slot_remap_out_of_range: "
                            f"capacity={capacity} mapping={normalized}"
                        )
                    source_indices = torch.tensor(
                        sources,
                        device=tensor.device,
                        dtype=torch.long,
                    )
                    target_indices = torch.tensor(
                        targets,
                        device=tensor.device,
                        dtype=torch.long,
                    )
                    affected_indices = torch.tensor(
                        sorted(set(sources) | set(targets)),
                        device=tensor.device,
                        dtype=torch.long,
                    )
                    values = tensor.index_select(0, source_indices).clone()
                    tensor.index_fill_(0, affected_indices, 0)
                    tensor.index_copy_(0, target_indices, values)

            static_state_indices = getattr(
                module,
                "hsd_static_text_state_sidecar_state_indices",
                None,
            )
            if isinstance(static_state_indices, list):
                for row, state_idx in enumerate(static_state_indices):
                    if int(state_idx) in normalized:
                        static_state_indices[row] = normalized[int(state_idx)]

    def remap_hsd_compact_state_slots(self, mapping: dict[int, int]) -> None:
        self.remap_hsd_state_slots(mapping)

    def clear_hsd_state_caches(
        self,
        state_indices: Iterable[int] | None = None,
        *,
        sidecar_state_indices: Iterable[int] | None = None,
    ) -> None:
        """Clear per-state HSD side caches.

        Native vLLM KV/Mamba caches are block-managed by the scheduler, while
        HSD anchor/history caches live in Python dicts keyed by state slot.
        When vLLM releases a request, clear the same block-table state slots
        from the Python side caches before those slots are reused.
        """
        state_index_set = (
            None
            if state_indices is None
            else {
                int(state_idx)
                for state_idx in state_indices
                if _is_valid_hsd_state_index(int(state_idx))
            }
        )
        sidecar_state_index_set = (
            state_index_set
            if sidecar_state_indices is None
            else {
                int(state_idx)
                for state_idx in sidecar_state_indices
                if _is_valid_hsd_state_index(int(state_idx))
            }
        )
        # Compact HSD slots key both legacy and paged anchor history even
        # outside Graph V3. Release/abort must therefore clear by the compact
        # sidecar slot whenever any layer uses compact state ownership; tying
        # this to the Graph environment leaks history across slot reuse.
        uses_compact_anchor_slots = any(
            bool(getattr(module, "hsd_use_compact_state_slots", False))
            for module in self.modules()
        )
        anchor_state_index_set = (
            sidecar_state_index_set
            if uses_compact_anchor_slots
            or _env_truthy("VLLM_HSD_ENABLE_GRAPH_V3")
            else state_index_set
        )
        anchor_hits = 0
        anchor_page_hits = 0
        text_state_hits = 0
        for module in self.modules():
            anchor_cache = getattr(module, "hsd_anchor_cache", None)
            if isinstance(anchor_cache, dict):
                if anchor_state_index_set is None:
                    anchor_hits += len(anchor_cache)
                    anchor_cache.clear()
                else:
                    for state_idx in anchor_state_index_set:
                        if anchor_cache.pop(state_idx, None) is not None:
                            anchor_hits += 1
            anchor_page_cache = getattr(module, "hsd_anchor_page_cache", None)
            if anchor_page_cache is not None:
                anchor_page_hits += anchor_page_cache.clear(
                    anchor_state_index_set
                )
            text_state_cache = getattr(module, "hsd_text_state_cache", None)
            if isinstance(text_state_cache, dict):
                if sidecar_state_index_set is None:
                    text_state_hits += len(text_state_cache)
                    text_state_cache.clear()
                else:
                    for state_idx in sidecar_state_index_set:
                        if text_state_cache.pop(state_idx, None) is not None:
                            text_state_hits += 1
            prefix_recurrent_replay = getattr(
                module, "hsd_prefix_recurrent_replay", None
            )
            if isinstance(prefix_recurrent_replay, dict):
                if sidecar_state_index_set is None:
                    prefix_recurrent_replay.clear()
                else:
                    for state_idx in sidecar_state_index_set:
                        prefix_recurrent_replay.pop(state_idx, None)
            persistent_state_sidecar = getattr(
                module, "hsd_persistent_state_sidecar", None
            )
            if isinstance(persistent_state_sidecar, dict):
                for tensor in persistent_state_sidecar.values():
                    if not torch.is_tensor(tensor):
                        continue
                    if sidecar_state_index_set is None:
                        tensor.zero_()
                    else:
                        valid_indices = [
                            state_idx
                            for state_idx in sidecar_state_index_set
                            if state_idx < tensor.shape[0]
                        ]
                        if valid_indices:
                            tensor.index_fill_(
                                0,
                                torch.tensor(
                                    valid_indices,
                                    dtype=torch.long,
                                    device=tensor.device,
                                ),
                                0,
                            )
            if state_index_set is None:
                prefix_snapshot_cache = getattr(
                    module, "hsd_prefix_snapshot_cache", None
                )
                if isinstance(prefix_snapshot_cache, dict):
                    prefix_snapshot_cache.clear()
                    if hasattr(module, "hsd_prefix_snapshot_bytes"):
                        module.hsd_prefix_snapshot_bytes = 0
        if _hsd_state_cache_log_enabled():
            logger.info(
                "HSD clear side caches state_indices=%s anchor_hits=%s "
                "sidecar_state_indices=%s anchor_page_hits=%s "
                "text_state_hits=%s",
                None if state_index_set is None else sorted(state_index_set),
                anchor_hits,
                (
                    None
                    if sidecar_state_index_set is None
                    else sorted(sidecar_state_index_set)
                ),
                anchor_page_hits,
                text_state_hits,
            )

    @torch.inference_mode()
    def get_hsd_prefix_cache_status(self) -> dict[str, object]:
        modules = list(self.model._iter_hsd_linear_attn_modules())
        return {
            "layers": len(modules),
            "entries": sum(len(m.hsd_prefix_snapshot_cache) for m in modules),
            "bytes": sum(int(m.hsd_prefix_snapshot_bytes) for m in modules),
            "generations": [int(m.hsd_prefix_cache_generation) for m in modules],
            "graph_storage": [
                {name: value.data_ptr() for name, value in m._hsd_graph_prefix_storage().items()}
                for m in modules
            ] if _env_truthy("VLLM_HSD_ENABLE_GRAPH_PREFIX_CACHE") else [],
        }

    @torch.inference_mode()
    def invalidate_hsd_prefix_cache(self) -> dict[str, int]:
        """Invalidate request state before copying weights at a drained boundary.

        EngineCore resets native KV separately. Graph allocations and learned
        persistent-memory parameters are retained.
        """
        stats = self.reset_hsd_runtime_state_after_warmup()
        for module in self.model._iter_hsd_linear_attn_modules():
            module.hsd_prefix_cache_generation += 1
        return stats

    @torch.inference_mode()
    def _clear_hsd_moe_shared_expert_outputs(self) -> int:
        cleared = 0
        seen: set[int] = set()
        for module in self.modules():
            for owner in (module, getattr(module, "runner", None)):
                shared_experts = getattr(owner, "_shared_experts", None)
                if shared_experts is None or id(shared_experts) in seen:
                    continue
                outputs = getattr(shared_experts, "_output", None)
                if isinstance(outputs, list):
                    cleared += sum(output is not None for output in outputs)
                    outputs[:] = [None] * len(outputs)
                    seen.add(id(shared_experts))
        return cleared

    @torch.inference_mode()
    def reset_hsd_runtime_state_after_warmup(self) -> dict[str, int]:
        """Reset HSD request state without reallocating graph-visible tensors."""
        stats = {
            "layers": 0,
            "legacy_anchor_entries": 0,
            "paged_anchor_entries": 0,
            "text_state_entries": 0,
            "prefix_snapshot_entries": 0,
            "moe_shared_expert_outputs": 0,
            "free_pages": 0,
            "total_pages": 0,
        }
        stats["moe_shared_expert_outputs"] = (
            self._clear_hsd_moe_shared_expert_outputs()
        )
        for module in self.modules():
            anchor_cache = getattr(module, "hsd_anchor_cache", None)
            if isinstance(anchor_cache, dict):
                stats["legacy_anchor_entries"] += len(anchor_cache)
                anchor_cache.clear()

            anchor_page_cache = getattr(module, "hsd_anchor_page_cache", None)
            if anchor_page_cache is not None:
                reset_fn = getattr(anchor_page_cache, "reset_runtime_state", None)
                if reset_fn is None:
                    raise RuntimeError("hsd_history_cache_runtime_reset_unsupported")
                stats["layers"] += 1
                stats["paged_anchor_entries"] += int(reset_fn())
                stats["free_pages"] += len(anchor_page_cache.free_pages)
                stats["total_pages"] += int(anchor_page_cache.total_pages)

            text_state_cache = getattr(module, "hsd_text_state_cache", None)
            if isinstance(text_state_cache, dict):
                stats["text_state_entries"] += len(text_state_cache)
                text_state_cache.clear()
            prefix_recurrent_replay = getattr(
                module, "hsd_prefix_recurrent_replay", None
            )
            if isinstance(prefix_recurrent_replay, dict):
                prefix_recurrent_replay.clear()

            for sidecar_name in (
                "hsd_persistent_state_sidecar",
                "hsd_static_text_state_sidecar",
            ):
                sidecar = getattr(module, sidecar_name, None)
                if isinstance(sidecar, dict):
                    for tensor in sidecar.values():
                        if torch.is_tensor(tensor):
                            tensor.zero_()

            # Dummy Graph V3 warmup can exercise the paged metric oracle even
            # though its rows are inactive. Do not expose those synthetic zero
            # samples through the post-initialization audit getter.
            for metric_name in (
                "_hsd_persistent_memory_metrics",
                "_hsd_persistent_memory_metric_sums",
                "_hsd_paged_persistent_memory_metrics",
            ):
                metrics = getattr(module, metric_name, None)
                if isinstance(metrics, dict):
                    metrics.clear()
            if hasattr(module, "_hsd_persistent_memory_metric_count"):
                module._hsd_persistent_memory_metric_count = 0

            static_state_indices = getattr(
                module,
                "hsd_static_text_state_sidecar_state_indices",
                None,
            )
            if isinstance(static_state_indices, list):
                static_state_indices[:] = [-1] * len(static_state_indices)
            static_past_lens = getattr(
                module,
                "hsd_static_text_state_sidecar_past_lens",
                None,
            )
            if isinstance(static_past_lens, list):
                static_past_lens[:] = [-1] * len(static_past_lens)

            prefix_snapshot_cache = getattr(
                module, "hsd_prefix_snapshot_cache", None
            )
            if isinstance(prefix_snapshot_cache, dict):
                stats["prefix_snapshot_entries"] += len(prefix_snapshot_cache)
                prefix_snapshot_cache.clear()
                if hasattr(module, "hsd_prefix_snapshot_bytes"):
                    module.hsd_prefix_snapshot_bytes = 0
            ready_keys = getattr(
                module, "hsd_prefix_snapshot_step_ready_keys", None
            )
            if isinstance(ready_keys, set):
                ready_keys.clear()
            evicted_keys = getattr(
                module, "hsd_prefix_snapshot_step_evicted_keys", None
            )
            if isinstance(evicted_keys, set):
                evicted_keys.clear()

        inner_model = getattr(self, "model", None)
        if inner_model is not None and hasattr(
            inner_model, "_hsd_prefix_snapshot_ready"
        ):
            inner_model._hsd_prefix_snapshot_ready = []
        if inner_model is not None and hasattr(
            inner_model, "_hsd_prefix_snapshot_evicted"
        ):
            inner_model._hsd_prefix_snapshot_evicted = []
        return stats

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        try:
            return self.model(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                hsd_query_start_loc=kwargs.get("hsd_query_start_loc"),
                hsd_num_computed_tokens=kwargs.get("hsd_num_computed_tokens"),
                hsd_state_indices=kwargs.get("hsd_state_indices"),
                hsd_history_state_indices=kwargs.get("hsd_history_state_indices"),
                hsd_decode_valid_mask=kwargs.get("hsd_decode_valid_mask"),
                hsd_prefix_snapshot_requests=kwargs.get(
                    "hsd_prefix_snapshot_requests"
                ),
                hsd_prefix_restored_flags=kwargs.get(
                    "hsd_prefix_restored_flags"
                ),
            )
        finally:
            if (
                _env_truthy("VLLM_HSD_ENABLE_GRAPH_V3")
                and self.model.config.model_type == "safin1_35b_a3b_text"
                and kwargs.get("hsd_decode_valid_mask") is None
            ):
                cleared = self._clear_hsd_moe_shared_expert_outputs()
                if cleared:
                    logger.warning_once(
                        "Cleared %d stale MoE shared-expert outputs after an "
                        "HSD Graph V3 model forward.",
                        cleared,
                    )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        layout = getattr(self.model, "_last_hsd_trace_layout", None)
        if _hsd_decode_trace_requested() and layout is not None:
            self.model._trace_hsd_sample_stage(
                "sample_hidden_states", hidden_states, layout
            )
        logits = self.logits_processor(self.lm_head, hidden_states)
        if _hsd_decode_trace_requested() and torch.is_tensor(logits):
            if layout is not None:
                self.model._trace_hsd_sample_stage("sample_logits", logits, layout)
        return logits


    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if (
            _env_truthy("VLLM_HSD_ENABLE_COMPACT_STATE_KV_SPEC")
            and _env_truthy("VLLM_HSD_USE_PERSISTENT_STATE_SIDECAR")
            and _env_truthy("VLLM_HSD_ENABLE_COMPACT_STATE_SLOTS")
        ):
            return (1, 4), (1, 4)
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


class Safin1MoeForConditionalGeneration(
    Safin1MoeForCausalLM
):
    """HF architecture name kept for converted remote-code checkpoints."""

    config_class = Safin1MoeConfig


class Safin1ForCausalLM(Safin1MoeForCausalLM):
    """Dense Safin-1 architecture name for converted checkpoints."""

    config_class = Safin1Config


class Safin1ForConditionalGeneration(Safin1ForCausalLM):
    """HF dense architecture name kept for converted remote-code checkpoints."""

    config_class = Safin1Config


__all__ = [
    "Safin1ForCausalLM",
    "Safin1ForConditionalGeneration",
    "Safin1MoeForCausalLM",
    "Safin1MoeForConditionalGeneration",
]
