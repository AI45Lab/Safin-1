from __future__ import annotations

import math
import os
import warnings
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.layers.gated_deltanet import GatedDeltaNet
try:
    from fla.layers.utils import get_layer_cache, get_packed_cu_seqlens, update_layer_cache
except ImportError:
    from fla.layers.utils import get_layer_cache, update_layer_cache

    def get_packed_cu_seqlens(kwargs: dict, device: torch.device | None = None):
        for key in ("cu_seqlens", "cu_seq_lens_q", "cu_seqlens_q", "cu_seq_lens_k", "cu_seqlens_k"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                return value.to(device=device, dtype=torch.long) if device is not None else value.to(dtype=torch.long), None
        return None, None

from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as GatedDeltaNetMLP
from fla.modules.l2norm import l2norm
from fla.modules.l2warp import l2_warp
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

try:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    fused_recurrent_gated_delta_rule = None

try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    chunk_kda = None
    fused_recurrent_kda = None

try:
    from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    chunk_gdn2 = None
    fused_recurrent_gdn2 = None

from .configuration_hsd_gated_deltanet import HSDGatedDeltaNetConfig
from .log_anchor import (
    LogAnchorBlock,
    fold_recent_events,
    fold_weak_fenwick_events,
    plan_recent_blocks,
    plan_weak_fenwick_blocks,
    reindex_compacted_blocks,
    validate_weak_fenwick_blocks,
)
from .summary_utils import (
    build_summary_anchor_indices,
    build_summary_embedding_ids,
    maybe_build_hsd_summary_batch,
    prepare_hsd_hybrid_attention_mask,
    prepare_hsd_hybrid_text_attention_inputs,
    scatter_hsd_hybrid_text_attention_output,
)

try:
    from .hsd_chunk_gdn import (
        hsd_chunk_gated_delta_rule_with_summary_states,
        hsd_flashqla_tile_gated_delta_rule_inference,
        hsd_flashqla_tile_gated_delta_rule_with_summary_states,
    )
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    hsd_chunk_gated_delta_rule_with_summary_states = None
    hsd_flashqla_tile_gated_delta_rule_inference = None
    hsd_flashqla_tile_gated_delta_rule_with_summary_states = None

try:
    from .hsd_chunk_kda import hsd_chunk_kda_with_summary_states
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    hsd_chunk_kda_with_summary_states = None

try:
    from .hsd_chunk_gdn2 import hsd_chunk_gdn2_with_summary_states
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    hsd_chunk_gdn2_with_summary_states = None

try:
    from .hsd_global_read import (
        hsd_fused_global_read,
        hsd_fused_global_read_decode,
        hsd_weighted_global_read,
        route_weight_momentum_scan,
    )
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    hsd_fused_global_read = None
    hsd_fused_global_read_decode = None
    hsd_weighted_global_read = None
    route_weight_momentum_scan = None

try:
    from .hsd_global_read_topk import hsd_sparse_topk_global_read
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    hsd_sparse_topk_global_read = None

try:
    from .hsd_route_topk import fused_route_topk_enabled, hsd_fused_route_topk
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    fused_route_topk_enabled = None
    hsd_fused_route_topk = None

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


_HSD_LOG_ANCHOR_CACHE_KEYS = (
    "hsd_log_anchor_counts",
    "hsd_log_anchor_levels",
    "hsd_log_anchor_start_chunks",
    "hsd_log_anchor_end_chunks",
    "hsd_log_anchor_total_chunks",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _apply_hsd_route_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Apply non-interleaved (NeoX-style) RoPE to the leading route dimensions."""
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError(f"route rope_dim must be a positive even integer, got {rope_dim}.")
    if rope_dim > x.shape[-1]:
        raise ValueError(f"route rope_dim={rope_dim} exceeds route head dim={x.shape[-1]}.")
    if cos.shape != sin.shape or cos.ndim != 2 or cos.shape != (x.shape[1], rope_dim // 2):
        raise ValueError(
            "route RoPE cos/sin must have shape [sequence, rope_dim / 2]; "
            f"got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, x={tuple(x.shape)}."
        )
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(0).unsqueeze(2)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(0).unsqueeze(2)
    x_rope = x[..., :rope_dim]
    x_first, x_second = x_rope.chunk(2, dim=-1)
    x_rotated_half = torch.cat((-x_second, x_first), dim=-1)
    rotated = x_rope * cos + x_rotated_half * sin
    return torch.cat((rotated, x[..., rope_dim:]), dim=-1)


class HSDRouteRotaryCache(nn.Module):
    """Shared, parameter-free FP32 frequency cache for route Partial RoPE."""

    def __init__(self, rope_dim: int, theta: float):
        super().__init__()
        self.rope_dim = int(rope_dim)
        self.theta = float(theta)
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)
        self._cache_seq_len = 0
        self._cache_dtype = None

    def _apply(self, fn):
        super()._apply(fn)
        # Module dtype conversion must not reduce the precision of the frequencies.
        self.inv_freq = self.inv_freq.float()
        self._cos_cached = self.inv_freq.new_empty(0)
        self._sin_cached = self.inv_freq.new_empty(0)
        self._cache_seq_len = 0
        self._cache_dtype = None
        return self

    def _ensure_cache(self, max_position: int, *, device: torch.device, dtype: torch.dtype) -> None:
        cache_valid = (
            self._cache_seq_len >= max_position
            and self._cos_cached.device == device
            and self._cache_dtype == dtype
        )
        if cache_valid:
            return
        positions = torch.arange(max_position, device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self._cos_cached = angles.cos().to(dtype=dtype)
        self._sin_cached = angles.sin().to(dtype=dtype)
        self._cache_seq_len = max_position
        self._cache_dtype = dtype

    def forward(
        self,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_positions.ndim != 1 or key_positions.ndim != 1:
            raise ValueError("route RoPE positions must be 1D tensors.")
        positions = query_positions if key_positions.numel() == 0 else torch.cat((query_positions, key_positions))
        max_position = int(positions.max().item()) + 1 if positions.numel() else 0
        self._ensure_cache(max_position, device=query_positions.device, dtype=dtype)
        query_positions = query_positions.to(device=self._cos_cached.device, dtype=torch.long)
        key_positions = key_positions.to(device=self._cos_cached.device, dtype=torch.long)
        return (
            self._cos_cached.index_select(0, query_positions),
            self._sin_cached.index_select(0, query_positions),
            self._cos_cached.index_select(0, key_positions),
            self._sin_cached.index_select(0, key_positions),
        )


class _HSDInitRNGReservation(nn.Module):
    """Reserve the initialization draws of a removed parameter without storing it."""

    def __init__(self, shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.shape = tuple(int(dim) for dim in shape)
        self.register_buffer(
            "_device_anchor",
            torch.empty(0, dtype=dtype, device=device),
            persistent=False,
        )


class HSDGatedDeltaNet(GatedDeltaNet):
    """Gated DeltaNet with summary-token anchored state routing."""

    def __init__(self, *args, hsd_read_block_size: int = 1024, hsd_read_chunk_block_size: int = 16,
                 hsd_use_global_read: bool = True, hsd_collect_read_cache: bool = True, summary_chunk_size: int = 0,
                 summary_token_num: int = 0, hsd_use_route_scale: bool = True, hsd_route_logit_scale: float = 1.0,
                 hsd_route_head_dim: int | None = None, hsd_route_query_heads: int | None = None,
                 hsd_route_key_heads: int | None = None, hsd_router_q_local_lora_rank: int = 0,
                 hsd_router_q_local_lora_source: str = "local_q",
                 hsd_router_q_local_lora_init_std: float = 0.02,
                 hsd_route_qk_l2norm: bool = True, hsd_read_topk: int = 0,
                 hsd_topk_backend: str = "reference",
                 hsd_route_weight_momentum_gamma: float = 0.0,
                 hsd_decode_momentum_init_mode: str = "carry",
                 hsd_route_position_encoding: str = "none", hsd_route_rope_dim: int = 0,
                 hsd_null_read_gate: bool = False, hsd_null_read_gate_init_bias: float = 0.0,
                 hsd_null_logm_scale: float = 0.0,
                 hsd_history_mix_mode: str = "additive",
                 hsd_history_gate_floor_init: float = 0.0,
                 hsd_history_gate_floor_warmup_steps: int = 0,
                 hsd_history_head_scale: bool = False,
                 hsd_local_core: str = "gdn",
                 hsd_kda_gate_rank: int | None = None,
                 hsd_kda_gate_init_std: float = 0.02,
                 hsd_summary_q_mode: str = "independent",
                 hsd_summary_q_lora_rank: int = 64,
                 hsd_summary_q_lora_init_std: float = 0.02,
                 hsd_history_read_q_mode: str = "shared_local",
                 hsd_history_read_q_lora_rank: int = 32,
                 hsd_history_read_q_lora_init_std: float = 0.02,
                 hsd_summary_key_source: str = "summary_hidden",
                 hsd_summary_key_content_mode: str = "none",
                 hsd_summary_key_content_scale_init: float = 0.0,
                 hsd_state_decay_mode: str = "gated",
                 hsd_history_state_mode: str = "continuous",
                 hsd_history_candidate_mode: str = "snapshot",
                 hsd_chunk_beta_mode: str = "shared",
                 hsd_train_selective_scan: bool = True, hsd_train_scan_backend: str = "triton",
                 hsd_train_selective_scan_packed: bool | None = None,
                 hsd_train_selective_scan_packed_strict: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.hsd_read_block_size = int(hsd_read_block_size)
        self.hsd_read_chunk_block_size = int(hsd_read_chunk_block_size)
        self.hsd_use_global_read = bool(hsd_use_global_read)
        self.hsd_collect_read_cache = bool(hsd_collect_read_cache)
        self.summary_chunk_size = int(summary_chunk_size)
        self.summary_token_num = int(summary_token_num)
        self.summary_enabled = (
            self.summary_chunk_size > 0 and self.summary_token_num > 0
        )
        self.hsd_use_route_scale = bool(hsd_use_route_scale)
        self.hsd_route_logit_scale = float(hsd_route_logit_scale)
        self.hsd_route_head_dim = self.head_k_dim if hsd_route_head_dim is None else int(hsd_route_head_dim)
        if self.hsd_route_head_dim <= 0:
            raise ValueError("hsd_route_head_dim must be positive.")
        self.hsd_route_query_heads = self.num_v_heads if hsd_route_query_heads is None else int(hsd_route_query_heads)
        if self.hsd_route_query_heads <= 0:
            raise ValueError("hsd_route_query_heads must be positive.")
        if self.hsd_route_query_heads not in {1, self.num_v_heads}:
            raise NotImplementedError("HSD route query heads currently supports 1 or num_v_heads.")
        self.hsd_router_q_local_lora_rank = int(hsd_router_q_local_lora_rank)
        if self.hsd_router_q_local_lora_rank < 0:
            raise ValueError("hsd_router_q_local_lora_rank must be non-negative.")
        self.hsd_router_q_local_lora_source = str(hsd_router_q_local_lora_source).strip().lower()
        if self.hsd_router_q_local_lora_source not in {"local_q", "history_read_q"}:
            raise ValueError(
                "hsd_router_q_local_lora_source must be 'local_q' or 'history_read_q'."
            )
        if (
            self.hsd_use_global_read
            and self.hsd_router_q_local_lora_rank > 0
            and self.hsd_route_query_heads != self.num_heads
        ):
            raise ValueError(
                "hsd_router_q_local_lora_rank > 0 requires hsd_route_query_heads "
                "to equal num_heads, because it adapts each local-Q head."
            )
        self.hsd_router_q_local_lora_init_std = float(hsd_router_q_local_lora_init_std)
        if (
            not math.isfinite(self.hsd_router_q_local_lora_init_std)
            or self.hsd_router_q_local_lora_init_std <= 0.0
        ):
            raise ValueError("hsd_router_q_local_lora_init_std must be finite and positive.")
        self.hsd_route_key_heads = self.num_v_heads if hsd_route_key_heads is None else int(hsd_route_key_heads)
        if self.hsd_route_key_heads <= 0:
            raise ValueError("hsd_route_key_heads must be positive.")
        if self.num_v_heads % self.hsd_route_key_heads != 0:
            raise ValueError(
                "HSD route key heads must divide num_v_heads so each key is shared by an equal "
                f"query-head group: got key_heads={self.hsd_route_key_heads}, "
                f"num_v_heads={self.num_v_heads}."
            )
        self.hsd_route_qk_l2norm = bool(hsd_route_qk_l2norm)
        self.hsd_route_position_encoding = str(hsd_route_position_encoding).strip().lower()
        self.hsd_route_rope_dim = int(hsd_route_rope_dim)
        if self.hsd_route_position_encoding not in {"none", "rope"}:
            raise ValueError("hsd_route_position_encoding must be either 'none' or 'rope'.")
        if self.hsd_route_position_encoding == "rope" and not 2 <= self.hsd_route_rope_dim <= self.hsd_route_head_dim:
            raise ValueError("route RoPE requires 2 <= rope_dim <= route_head_dim.")
        self.hsd_read_topk = int(hsd_read_topk)
        self.hsd_topk_backend = str(hsd_topk_backend).strip().lower()
        if self.hsd_topk_backend not in {"reference", "sparse"}:
            raise ValueError("hsd_topk_backend must be either 'reference' or 'sparse'.")
        if self.hsd_topk_backend == "sparse" and self.hsd_read_topk not in {0, 1, 2, 4, 8, 16}:
            raise ValueError("Sparse HSD top-k supports hsd_read_topk in {1, 2, 4, 8, 16} (or 0 when dormant).")
        self.hsd_route_weight_momentum_gamma = float(hsd_route_weight_momentum_gamma)
        if (
            not math.isfinite(self.hsd_route_weight_momentum_gamma)
            or not 0.0 <= self.hsd_route_weight_momentum_gamma < 1.0
        ):
            raise ValueError("hsd_route_weight_momentum_gamma must be finite and in [0, 1).")
        if self.hsd_route_weight_momentum_gamma > 0.0 and self.hsd_read_topk != 0:
            raise ValueError("HSD route-weight momentum requires hsd_read_topk=0.")
        self.hsd_decode_momentum_init_mode = str(hsd_decode_momentum_init_mode).strip().lower()
        if self.hsd_decode_momentum_init_mode not in {"carry", "last_prompt_raw"}:
            raise ValueError(
                "hsd_decode_momentum_init_mode must be either 'carry' or 'last_prompt_raw'."
            )
        self.hsd_null_read_gate = bool(hsd_null_read_gate)
        self.hsd_null_read_gate_init_bias = float(hsd_null_read_gate_init_bias)
        self.hsd_null_logm_scale = float(hsd_null_logm_scale)
        self.hsd_history_mix_mode = str(hsd_history_mix_mode).strip().lower()
        if self.hsd_history_mix_mode not in {"additive", "convex", "joint_softmax_logm", "gated_additive"}:
            raise ValueError(
                "hsd_history_mix_mode must be 'additive', 'convex', "
                "'joint_softmax_logm', or 'gated_additive'."
            )
        if self.hsd_history_mix_mode in {"convex", "joint_softmax_logm", "gated_additive"} \
                and not self.hsd_null_read_gate:
            raise ValueError(f"hsd_history_mix_mode={self.hsd_history_mix_mode!r} requires hsd_null_read_gate=True.")
        if not math.isfinite(self.hsd_null_logm_scale) or self.hsd_null_logm_scale < 0.0:
            raise ValueError("hsd_null_logm_scale must be a non-negative finite value.")
        if self.hsd_null_logm_scale > 0.0:
            if not self.hsd_null_read_gate:
                raise ValueError("hsd_null_logm_scale > 0 requires hsd_null_read_gate=True.")
            if self.hsd_history_mix_mode != "additive":
                raise ValueError(
                    "hsd_null_logm_scale > 0 currently requires "
                    "hsd_history_mix_mode='additive'."
                )
            if self.hsd_read_topk != 0:
                raise ValueError("hsd_null_logm_scale > 0 currently requires hsd_read_topk=0.")
        self.hsd_history_gate_floor_init = float(hsd_history_gate_floor_init)
        self.hsd_history_gate_floor_warmup_steps = int(hsd_history_gate_floor_warmup_steps)
        if (
            not math.isfinite(self.hsd_history_gate_floor_init)
            or not 0.0 <= self.hsd_history_gate_floor_init < 1.0
        ):
            raise ValueError("hsd_history_gate_floor_init must be finite and in [0, 1).")
        if self.hsd_history_gate_floor_warmup_steps < 0:
            raise ValueError("hsd_history_gate_floor_warmup_steps must be non-negative.")
        if self.hsd_history_gate_floor_init > 0.0:
            if self.hsd_history_mix_mode != "gated_additive":
                raise ValueError(
                    "A positive hsd_history_gate_floor_init requires "
                    "hsd_history_mix_mode='gated_additive'."
                )
            if self.hsd_history_gate_floor_warmup_steps <= 0:
                raise ValueError(
                    "A positive hsd_history_gate_floor_init requires "
                    "hsd_history_gate_floor_warmup_steps > 0."
                )
        self.hsd_history_head_scale = bool(hsd_history_head_scale)
        if self.hsd_history_head_scale and self.hsd_history_mix_mode not in {
            "additive",
            "gated_additive",
        }:
            raise ValueError(
                "hsd_history_head_scale=True requires hsd_history_mix_mode to be "
                "'additive' or 'gated_additive'."
            )
        self.hsd_local_core = str(hsd_local_core).strip().lower()
        if self.hsd_local_core not in {"gdn", "kda", "gdn2"}:
            raise ValueError("hsd_local_core must be 'gdn', 'kda', or 'gdn2'.")
        self.hsd_kda_inference_decay_mode = os.environ.get(
            "HSD_KDA_INFERENCE_DECAY_MODE", "learned"
        ).strip().lower()
        if self.hsd_kda_inference_decay_mode not in {"learned", "mean_broadcast"}:
            raise ValueError(
                "HSD_KDA_INFERENCE_DECAY_MODE must be 'learned' or "
                f"'mean_broadcast', got {self.hsd_kda_inference_decay_mode!r}."
            )
        self.hsd_kda_gate_rank = (
            self.head_v_dim if hsd_kda_gate_rank is None else int(hsd_kda_gate_rank)
        )
        if self.hsd_kda_gate_rank <= 0:
            raise ValueError("hsd_kda_gate_rank must be positive.")
        self.hsd_kda_gate_init_std = float(hsd_kda_gate_init_std)
        if not math.isfinite(self.hsd_kda_gate_init_std) or self.hsd_kda_gate_init_std <= 0.0:
            raise ValueError("hsd_kda_gate_init_std must be finite and positive.")
        self.hsd_summary_q_mode = str(hsd_summary_q_mode).strip().lower()
        if self.hsd_summary_q_mode not in {"independent", "shared_read_q_lora"}:
            raise ValueError(
                "hsd_summary_q_mode must be either 'independent' or 'shared_read_q_lora'."
            )
        self.hsd_summary_q_lora_rank = int(hsd_summary_q_lora_rank)
        if self.hsd_summary_q_lora_rank <= 0:
            raise ValueError("hsd_summary_q_lora_rank must be positive.")
        self.hsd_summary_q_lora_init_std = float(hsd_summary_q_lora_init_std)
        if not math.isfinite(self.hsd_summary_q_lora_init_std) or self.hsd_summary_q_lora_init_std <= 0:
            raise ValueError("hsd_summary_q_lora_init_std must be finite and positive.")
        if self.hsd_summary_q_mode == "shared_read_q_lora" and self.num_v_heads != self.num_heads:
            raise ValueError(
                "hsd_summary_q_mode='shared_read_q_lora' requires num_v_heads == num_heads."
            )
        self.hsd_history_read_q_mode = str(hsd_history_read_q_mode).strip().lower()
        if self.hsd_history_read_q_mode not in {
            "shared_local",
            "per_head_residual_lora",
            "independent_projection",
            "full_residual_projection",
        }:
            raise ValueError(
                "hsd_history_read_q_mode must be 'shared_local', "
                "'per_head_residual_lora', 'independent_projection', or "
                "'full_residual_projection'."
            )
        self.hsd_history_read_q_lora_rank = int(hsd_history_read_q_lora_rank)
        if self.hsd_history_read_q_lora_rank <= 0:
            raise ValueError("hsd_history_read_q_lora_rank must be positive.")
        self.hsd_history_read_q_lora_init_std = float(hsd_history_read_q_lora_init_std)
        if (
            not math.isfinite(self.hsd_history_read_q_lora_init_std)
            or self.hsd_history_read_q_lora_init_std <= 0.0
        ):
            raise ValueError("hsd_history_read_q_lora_init_std must be finite and positive.")
        self.hsd_summary_key_source = str(hsd_summary_key_source).strip().lower()
        if self.hsd_summary_key_source not in {"summary_hidden", "state_probe"}:
            raise ValueError(
                "hsd_summary_key_source must be 'summary_hidden' or 'state_probe'."
            )
        self.hsd_summary_key_content_mode = str(hsd_summary_key_content_mode).strip().lower()
        if self.hsd_summary_key_content_mode not in {
            "none",
            "chunk_mean_residual",
            "chunk_mean_only",
        }:
            raise ValueError(
                "hsd_summary_key_content_mode must be 'none', "
                "'chunk_mean_residual', or 'chunk_mean_only'."
            )
        self.hsd_summary_key_content_scale_init = float(hsd_summary_key_content_scale_init)
        if not math.isfinite(self.hsd_summary_key_content_scale_init):
            raise ValueError("hsd_summary_key_content_scale_init must be finite.")
        if (
            self.hsd_summary_key_source == "state_probe"
            and self.hsd_summary_key_content_mode != "none"
        ):
            raise ValueError(
                "hsd_summary_key_source='state_probe' is incompatible with "
                "hsd_summary_key_content_mode != 'none'."
            )
        self.hsd_state_decay_mode = str(hsd_state_decay_mode).strip().lower()
        if self.hsd_state_decay_mode not in {"gated", "none"}:
            raise ValueError("hsd_state_decay_mode must be 'gated' or 'none'.")
        self.hsd_history_state_mode = str(hsd_history_state_mode).strip().lower()
        if self.hsd_history_state_mode not in {"continuous", "chunk_reset_no_decay"}:
            raise ValueError(
                "hsd_history_state_mode must be 'continuous' or 'chunk_reset_no_decay'."
            )
        self.hsd_history_candidate_mode = str(hsd_history_candidate_mode).strip().lower()
        if self.hsd_history_candidate_mode not in {"snapshot", "delta_from_continuous"}:
            raise ValueError(
                "hsd_history_candidate_mode must be 'snapshot' or 'delta_from_continuous'."
            )
        if (
            self.hsd_history_candidate_mode == "delta_from_continuous"
            and self.hsd_history_state_mode != "continuous"
        ):
            raise ValueError(
                "hsd_history_candidate_mode='delta_from_continuous' requires "
                "hsd_history_state_mode='continuous'."
            )
        self.hsd_log_anchor_mode = os.environ.get("HSD_LOG_ANCHOR_MODE", "off").strip().lower()
        self.hsd_log_anchor_count_bias = os.environ.get(
            "HSD_LOG_ANCHOR_COUNT_BIAS", "none"
        ).strip().lower()
        if self.hsd_log_anchor_mode not in {"off", "simulate", "cache", "recent"}:
            raise ValueError(
                "HSD_LOG_ANCHOR_MODE must be 'off', 'simulate', 'cache', or 'recent', "
                f"got {self.hsd_log_anchor_mode!r}."
            )
        if self.hsd_null_logm_scale > 0.0 and self.hsd_log_anchor_mode != "off":
            raise NotImplementedError(
                "HSD null LogMeanExp correction currently requires HSD_LOG_ANCHOR_MODE=off."
            )
        if self.hsd_log_anchor_count_bias not in {"none", "log"}:
            raise ValueError(
                "HSD_LOG_ANCHOR_COUNT_BIAS must be 'none' or 'log', got "
                f"{self.hsd_log_anchor_count_bias!r}."
            )
        if self.hsd_log_anchor_count_bias == "log" and self.hsd_log_anchor_mode not in {
            "simulate", "cache"
        }:
            raise ValueError(
                "HSD_LOG_ANCHOR_COUNT_BIAS='log' requires "
                "HSD_LOG_ANCHOR_MODE='simulate' or 'cache'."
            )
        if self.hsd_log_anchor_mode != "off" and self.hsd_collect_read_cache:
            if self.summary_token_num != 1:
                raise NotImplementedError("HSD Log-Anchor V1 requires summary_token_num=1.")
            if self.hsd_history_mix_mode != "additive":
                raise NotImplementedError(
                    "HSD Log-Anchor V1 currently supports only hsd_history_mix_mode='additive'."
                )
            if self.hsd_read_topk != 0 or self.hsd_topk_backend != "reference":
                raise NotImplementedError(
                    "HSD Log-Anchor V1 requires reference routing with hsd_read_topk=0."
                )
            if self.hsd_route_weight_momentum_gamma != 0.0:
                raise NotImplementedError("HSD Log-Anchor V1 does not support route-weight momentum.")
            if self.hsd_history_candidate_mode != "snapshot":
                raise NotImplementedError("HSD Log-Anchor V1 requires cumulative snapshot candidates.")
            if _env_bool("HSD_FUSED_GLOBAL_READ_DECODE_STRICT", False):
                raise NotImplementedError(
                    "HSD Log-Anchor V1 disables the fused route/read decode fastpath; "
                    "unset HSD_FUSED_GLOBAL_READ_DECODE_STRICT."
                )
        self.hsd_chunk_beta_mode = str(hsd_chunk_beta_mode).strip().lower()
        self.hsd_train_scan_backend = os.environ.get(
            "HSD_TRAIN_SCAN_BACKEND",
            str(hsd_train_scan_backend),
        ).strip().lower()
        if self.hsd_train_scan_backend not in {"triton", "flashqla_tile"}:
            raise ValueError(
                "HSD_TRAIN_SCAN_BACKEND/hsd_train_scan_backend must be "
                "'triton' or 'flashqla_tile'."
            )
        if self.hsd_train_scan_backend == "flashqla_tile":
            if self.hsd_local_core != "gdn":
                raise ValueError("FlashQLA TileLang scan currently requires hsd_local_core='gdn'.")
            if self.hsd_history_state_mode != "continuous":
                raise ValueError(
                    "FlashQLA TileLang scan currently requires "
                    "hsd_history_state_mode='continuous'."
                )
        if self.hsd_chunk_beta_mode not in {"shared", "independent"}:
            raise ValueError("hsd_chunk_beta_mode must be 'shared' or 'independent'.")
        if self.hsd_chunk_beta_mode == "independent" and self.hsd_history_state_mode != "chunk_reset_no_decay":
            raise ValueError(
                "hsd_chunk_beta_mode='independent' requires "
                "hsd_history_state_mode='chunk_reset_no_decay'."
            )
        if self.hsd_local_core == "kda":
            if self.head_k_dim > 256:
                raise ValueError("HSD-KDA currently requires head_dim <= 256.")
            if self.hsd_history_state_mode != "continuous":
                raise ValueError(
                    "The HSD-KDA single-variable control currently requires "
                    "hsd_history_state_mode='continuous'."
                )
            if self.hsd_history_candidate_mode != "snapshot":
                raise ValueError(
                    "The HSD-KDA single-variable control currently requires "
                    "hsd_history_candidate_mode='snapshot'."
                )
        if self.hsd_local_core == "gdn2":
            if self.num_heads != self.num_v_heads or self.num_v_heads != 6:
                raise ValueError("HSD-GDN2 V1 requires num_heads=num_v_heads=6.")
            if self.head_k_dim > 256:
                raise ValueError("HSD-GDN2 V1 requires head_dim <= 256.")
            if self.allow_neg_eigval:
                raise ValueError("HSD-GDN2 V1 requires allow_neg_eigval=False.")
            if self.hsd_state_decay_mode != "gated":
                raise ValueError("HSD-GDN2 V1 requires hsd_state_decay_mode='gated'.")
            if self.hsd_history_state_mode != "continuous":
                raise ValueError("HSD-GDN2 V1 requires hsd_history_state_mode='continuous'.")
            if self.hsd_history_candidate_mode != "snapshot":
                raise ValueError("HSD-GDN2 V1 requires hsd_history_candidate_mode='snapshot'.")
            if self.hsd_chunk_beta_mode != "shared":
                raise ValueError("HSD-GDN2 V1 does not support an independent chunk beta.")
        self.kda_f_proj = None
        if self.hsd_local_core == "kda":
            gate_device = self.a_proj.weight.device
            fork_devices = (
                [gate_device.index if gate_device.index is not None else torch.cuda.current_device()]
                if gate_device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=fork_devices):
                self.kda_f_proj = nn.Sequential(
                    nn.Linear(
                        self.hidden_size,
                        self.hsd_kda_gate_rank,
                        bias=False,
                        device=gate_device,
                        dtype=self.a_proj.weight.dtype,
                    ),
                    nn.Linear(
                        self.hsd_kda_gate_rank,
                        self.num_v_heads * self.head_k_dim,
                        bias=False,
                        device=gate_device,
                        dtype=self.a_proj.weight.dtype,
                    ),
                )
                # Match native KDA parameterization: both factors are dense at
                # initialization, so every bottleneck channel receives a
                # gradient from the first step. Keep these draws inside the
                # fork so adding KDA does not perturb common GDN/HSD weights.
                for projection in self.kda_f_proj:
                    nn.init.normal_(
                        projection.weight,
                        mean=0.0,
                        std=self.hsd_kda_gate_init_std,
                    )
                gate_dim = self.num_v_heads * self.head_k_dim
                dt = torch.exp(
                    torch.rand(
                        gate_dim,
                        device=self.dt_bias.device,
                        dtype=torch.float32,
                    ) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                ).clamp(min=1e-4)
                kda_inv_dt = dt + torch.log(-torch.expm1(-dt))
            for projection in self.kda_f_proj:
                # The factors were initialized explicitly in the isolated RNG
                # scope above; post_init must not initialize them a second time.
                projection._hsd_kda_gate_initialized = True
            self.kda_dt_bias = nn.Parameter(
                kda_inv_dt.to(device=self.dt_bias.device, dtype=self.dt_bias.dtype)
            )
            self.kda_dt_bias._no_weight_decay = True
        else:
            self.register_parameter("kda_dt_bias", None)
        self.gdn2_f_proj = None
        self.gdn2_erase_proj = None
        self.gdn2_write_proj = None
        if self.hsd_local_core == "gdn2":
            legacy_a_proj = self.a_proj
            legacy_b_proj = self.b_proj
            gate_device = legacy_a_proj.weight.device
            gate_dtype = legacy_a_proj.weight.dtype
            fork_devices = (
                [gate_device.index if gate_device.index is not None else torch.cuda.current_device()]
                if gate_device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=fork_devices):
                self.gdn2_f_proj = nn.Sequential(
                    nn.Linear(
                        self.hidden_size,
                        self.head_v_dim,
                        bias=False,
                        device=gate_device,
                        dtype=gate_dtype,
                    ),
                    nn.Linear(
                        self.head_v_dim,
                        self.num_v_heads * self.head_k_dim,
                        bias=False,
                        device=gate_device,
                        dtype=gate_dtype,
                    ),
                )
                self.gdn2_erase_proj = nn.Linear(
                    self.hidden_size,
                    self.num_v_heads * self.head_k_dim,
                    bias=False,
                    device=gate_device,
                    dtype=gate_dtype,
                )
                self.gdn2_write_proj = nn.Linear(
                    self.hidden_size,
                    self.num_v_heads * self.head_v_dim,
                    bias=False,
                    device=gate_device,
                    dtype=gate_dtype,
                )
                for projection in (
                    *self.gdn2_f_proj,
                    self.gdn2_erase_proj,
                    self.gdn2_write_proj,
                ):
                    nn.init.xavier_uniform_(projection.weight, gain=2 ** -2.5)
                    projection._hsd_gdn2_initialized = True
                gate_dim = self.num_v_heads * self.head_k_dim
                dt = torch.exp(
                    torch.rand(
                        gate_dim,
                        device=gate_device,
                        dtype=torch.float32,
                    ) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                ).clamp(min=1e-4)
                gdn2_inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.gdn2_dt_bias = nn.Parameter(
                gdn2_inv_dt.to(device=gate_device, dtype=torch.float32)
            )
            self.gdn2_dt_bias._no_weight_decay = True
            # Remove the inherited scalar GDN projections without changing the
            # common-parameter RNG stream used by a fresh canonical seed.
            self.a_proj = _HSDInitRNGReservation(
                tuple(legacy_a_proj.weight.shape),
                dtype=legacy_a_proj.weight.dtype,
                device=legacy_a_proj.weight.device,
            )
            self.b_proj = _HSDInitRNGReservation(
                tuple(legacy_b_proj.weight.shape),
                dtype=legacy_b_proj.weight.dtype,
                device=legacy_b_proj.weight.device,
            )
            self.dt_bias = None
            self._hsd_gdn2_scalar_dt_rng_shape = (self.num_v_heads,)
        else:
            self.register_parameter("gdn2_dt_bias", None)
            self._hsd_gdn2_scalar_dt_rng_shape = None
        self.hsd_train_selective_scan = _env_bool("HSD_TRAIN_SELECTIVE_SCAN", bool(hsd_train_selective_scan))
        packed_default = self.hsd_train_selective_scan if hsd_train_selective_scan_packed is None else bool(
            hsd_train_selective_scan_packed
        )
        self.hsd_train_selective_scan_packed = _env_bool("HSD_TRAIN_SELECTIVE_SCAN_PACKED", packed_default)
        self.hsd_train_selective_scan_packed_strict = _env_bool(
            "HSD_TRAIN_SELECTIVE_SCAN_PACKED_STRICT",
            bool(hsd_train_selective_scan_packed_strict),
        )
        summary_q_dim = self.num_v_heads * self.head_k_dim
        self.summary_q_proj = None
        self.summary_q_down_proj = None
        self.summary_q_up_proj = None
        self.summary_q_init_rng_reservation = None
        if not self.summary_enabled:
            # A local-only GDN2 baseline must not retain the per-layer summary
            # query projection as a dormant trainable parameter.  Consume the
            # same constructor and post_init RNG draws as the ordinary
            # independent summary projection so removing it does not perturb
            # parameters constructed later in the model under the same seed.
            summary_q_reference = nn.Linear(
                self.hidden_size,
                summary_q_dim,
                bias=False,
            )
            reference_device = summary_q_reference.weight.device
            reference_dtype = summary_q_reference.weight.dtype
            del summary_q_reference
            self.summary_q_init_rng_reservation = _HSDInitRNGReservation(
                (summary_q_dim, self.hidden_size),
                dtype=reference_dtype,
                device=reference_device,
            )
        elif self.hsd_summary_q_mode == "independent":
            self.summary_q_proj = nn.Linear(self.hidden_size, summary_q_dim, bias=False)
        else:
            # Consume exactly the constructor RNG used by the independent
            # summary_q_proj. Adapter construction and initialization are
            # isolated so all parameters shared with the control keep the same
            # model_init_seed values.
            summary_q_reference = nn.Linear(self.hidden_size, summary_q_dim, bias=False)
            reference_device = summary_q_reference.weight.device
            reference_dtype = summary_q_reference.weight.dtype
            del summary_q_reference
            fork_devices = (
                [reference_device.index if reference_device.index is not None else torch.cuda.current_device()]
                if reference_device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=fork_devices):
                self.summary_q_down_proj = nn.Linear(
                    self.hidden_size,
                    self.hsd_summary_q_lora_rank,
                    bias=False,
                    device=reference_device,
                    dtype=reference_dtype,
                )
                self.summary_q_up_proj = nn.Linear(
                    self.hsd_summary_q_lora_rank,
                    summary_q_dim,
                    bias=False,
                    device=reference_device,
                    dtype=reference_dtype,
                )
                nn.init.normal_(
                    self.summary_q_down_proj.weight,
                    mean=0.0,
                    std=self.hsd_summary_q_lora_init_std,
                )
                nn.init.zeros_(self.summary_q_up_proj.weight)
            # HF post_init must not reinitialize these children or perturb the
            # common RNG stream. The reservation below consumes the draws that
            # independent summary_q_proj would have used during post_init.
            self.summary_q_down_proj._is_hf_initialized = True
            self.summary_q_up_proj._is_hf_initialized = True
            self.summary_q_init_rng_reservation = _HSDInitRNGReservation(
                (summary_q_dim, self.hidden_size),
                dtype=reference_dtype,
                device=reference_device,
            )
        self.chunk_b_proj = None
        if self.hsd_chunk_beta_mode == "independent":
            # Keep every common parameter bitwise matched to the shared-beta
            # control under the same model_init_seed. Both nn.Linear's
            # constructor and the HF initialization traversal normally consume
            # RNG, so isolate construction here and skip the child re-init in
            # _init_weights below. The enclosing HSD module copies the final
            # initialized local beta projection after its children are visited.
            beta_device = self.b_proj.weight.device
            fork_devices = (
                [beta_device.index if beta_device.index is not None else torch.cuda.current_device()]
                if beta_device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=fork_devices):
                self.chunk_b_proj = nn.Linear(
                    self.hidden_size,
                    self.num_v_heads,
                    bias=False,
                    device=beta_device,
                    dtype=self.b_proj.weight.dtype,
                )
            self.chunk_b_proj._hsd_copy_beta_init = True
        if self.chunk_b_proj is not None:
            with torch.no_grad():
                self.chunk_b_proj.weight.copy_(self.b_proj.weight)
        self.summary_k_proj = None
        self.summary_k_init_rng_reservation = None
        if self.hsd_collect_read_cache:
            summary_k_dim = self.hsd_route_key_heads * self.hsd_route_head_dim
            if 1 < self.hsd_route_key_heads < self.num_v_heads:
                # Grouped K2/K3 experiments are compared against the established
                # K1 control. Consume only the K1 constructor RNG on the main
                # stream and build the wider projection in an isolated stream.
                # _init_weights below follows the same rule during HF post_init.
                base_summary_k_dim = self.hsd_route_head_dim
                summary_k_reference = nn.Linear(
                    self.hidden_size,
                    base_summary_k_dim,
                    bias=False,
                )
                reference_device = summary_k_reference.weight.device
                reference_dtype = summary_k_reference.weight.dtype
                del summary_k_reference
                fork_devices = (
                    [reference_device.index if reference_device.index is not None else torch.cuda.current_device()]
                    if reference_device.type == "cuda" else []
                )
                with torch.random.fork_rng(devices=fork_devices):
                    self.summary_k_proj = nn.Linear(
                        self.hidden_size,
                        summary_k_dim,
                        bias=False,
                        device=reference_device,
                        dtype=reference_dtype,
                    )
                self.summary_k_proj._hsd_grouped_key_base_rows = base_summary_k_dim
            else:
                self.summary_k_proj = nn.Linear(
                    self.hidden_size,
                    summary_k_dim,
                    bias=False,
                )
        self.register_parameter("summary_state_value_probe", None)
        self.summary_state_k_proj = None
        if self.hsd_collect_read_cache and self.hsd_summary_key_source == "state_probe":
            # Derive each route key from the exact recurrent state that the
            # global reader will consume:
            #   u_m = mean_head(S_m) @ p_v
            #   k_m = W_r L2Norm(u_m)
            # Construction and initialization are isolated so enabling this
            # optional branch does not perturb any common seeded parameters.
            state_device = self.q_proj.weight.device
            state_dtype = self.q_proj.weight.dtype
            fork_devices = (
                [state_device.index if state_device.index is not None else torch.cuda.current_device()]
                if state_device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=fork_devices):
                value_probe = torch.empty(
                    self.head_v_dim,
                    device=state_device,
                    dtype=state_dtype,
                )
                nn.init.normal_(value_probe, mean=0.0, std=self.head_v_dim ** -0.5)
                self.summary_state_k_proj = nn.Linear(
                    self.head_k_dim,
                    self.hsd_route_head_dim,
                    bias=False,
                    device=state_device,
                    dtype=state_dtype,
                )
                nn.init.normal_(
                    self.summary_state_k_proj.weight,
                    mean=0.0,
                    std=self.head_k_dim ** -0.5,
                )
            self.summary_state_value_probe = nn.Parameter(value_probe)
            self.summary_state_k_proj._hsd_state_route_initialized = True
        self.router_q_proj = (
            nn.Linear(self.hidden_size, self.hsd_route_query_heads * self.hsd_route_head_dim, bias=False)
            if self.hsd_use_global_read else None
        )
        self.register_parameter("router_q_local_lora_down_weight", None)
        self.register_parameter("router_q_local_lora_up_weight", None)
        if self.hsd_use_global_read and self.hsd_router_q_local_lora_rank > 0:
            # The adapter is isolated from the common initialization stream;
            # its zero-up initialization leaves the baseline function exact.
            q_device = self.q_proj.weight.device
            q_dtype = self.q_proj.weight.dtype
            fork_devices = (
                [q_device.index if q_device.index is not None else torch.cuda.current_device()]
                if q_device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=fork_devices):
                down_weight = torch.empty(
                    self.num_heads,
                    self.hsd_router_q_local_lora_rank,
                    self.head_k_dim,
                    device=q_device,
                    dtype=q_dtype,
                )
                up_weight = torch.empty(
                    self.num_heads,
                    self.hsd_route_head_dim,
                    self.hsd_router_q_local_lora_rank,
                    device=q_device,
                    dtype=q_dtype,
                )
                nn.init.normal_(
                    down_weight,
                    mean=0.0,
                    std=self.hsd_router_q_local_lora_init_std,
                )
                nn.init.zeros_(up_weight)
            self.router_q_local_lora_down_weight = nn.Parameter(down_weight)
            self.router_q_local_lora_up_weight = nn.Parameter(up_weight)
        self.register_parameter("history_read_q_down_weight", None)
        self.register_parameter("history_read_q_up_weight", None)
        self.history_read_q_proj = None
        if (
            self.hsd_use_global_read
            and self.hsd_history_read_q_mode == "per_head_residual_lora"
        ):
            # Each GDN query head owns an independent dk -> rank -> dk adapter.
            # Construction is isolated so common baseline parameters retain
            # their exact model_init_seed values.
            q_device = self.q_proj.weight.device
            q_dtype = self.q_proj.weight.dtype
            fork_devices = (
                [q_device.index if q_device.index is not None else torch.cuda.current_device()]
                if q_device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=fork_devices):
                down_weight = torch.empty(
                    self.num_heads,
                    self.hsd_history_read_q_lora_rank,
                    self.head_k_dim,
                    device=q_device,
                    dtype=q_dtype,
                )
                up_weight = torch.empty(
                    self.num_heads,
                    self.head_k_dim,
                    self.hsd_history_read_q_lora_rank,
                    device=q_device,
                    dtype=q_dtype,
                )
                nn.init.normal_(
                    down_weight,
                    mean=0.0,
                    std=self.hsd_history_read_q_lora_init_std,
                )
                nn.init.zeros_(up_weight)
            self.history_read_q_down_weight = nn.Parameter(down_weight)
            self.history_read_q_up_weight = nn.Parameter(up_weight)
        elif self.hsd_use_global_read and self.hsd_history_read_q_mode in {
            "independent_projection",
            "full_residual_projection",
        }:
            # A complete hidden -> query projection dedicated to historical
            # state reads. The residual variant is zero initialized so its
            # initial function is exactly the shared-local query while still
            # receiving a full-rank gradient on the first step. Isolate its
            # constructor RNG so adding this optional branch does not shift
            # any subsequently initialized common parameters.
            if self.hsd_history_read_q_mode == "full_residual_projection":
                q_device = self.q_proj.weight.device
                q_dtype = self.q_proj.weight.dtype
                fork_devices = (
                    [q_device.index if q_device.index is not None else torch.cuda.current_device()]
                    if q_device.type == "cuda" else []
                )
                with torch.random.fork_rng(devices=fork_devices):
                    self.history_read_q_proj = nn.Linear(
                        self.hidden_size,
                        self.num_heads * self.head_k_dim,
                        bias=False,
                        device=q_device,
                        dtype=q_dtype,
                    )
                    nn.init.zeros_(self.history_read_q_proj.weight)
                # Preserve the zero initialization through HF post_init.
                self.history_read_q_proj._hsd_zero_init = True
            else:
                self.history_read_q_proj = nn.Linear(
                    self.hidden_size,
                    self.num_heads * self.head_k_dim,
                    bias=False,
                )
        self.null_gate_proj = (
            nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
            if self.hsd_use_global_read and self.hsd_null_read_gate else None
        )
        if self.null_gate_proj is not None:
            self.null_gate_proj._hsd_zero_init = True
            self.hsd_null_read_gate_bias = nn.Parameter(
                torch.full((self.num_v_heads,), self.hsd_null_read_gate_init_bias, dtype=torch.float32)
            )
        else:
            self.register_parameter("hsd_null_read_gate_bias", None)
        if self.hsd_history_head_scale and self.hsd_use_global_read:
            self.history_head_log_scale = nn.Parameter(
                torch.zeros(self.num_v_heads, dtype=torch.float32)
            )
        else:
            self.register_parameter("history_head_log_scale", None)
        if self.hsd_summary_key_content_mode == "chunk_mean_residual" and self.hsd_collect_read_cache:
            # A deterministic zero scalar adds no RNG draw and makes the fresh
            # model exactly match the K1 control before the first optimizer step.
            self.summary_key_content_scale = nn.Parameter(
                torch.full((), self.hsd_summary_key_content_scale_init, dtype=torch.float32)
            )
        else:
            self.register_parameter("summary_key_content_scale", None)
        self._hsd_debug_stats = {}
        self._hsd_debug_forward_count = 0
        self._warned_hsd_fallback_to_gdn = False

    def _project_decay_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.hsd_local_core == "gdn2":
            if self.gdn2_f_proj is None:
                raise RuntimeError("HSD-GDN2 decay projection is missing.")
            return self.gdn2_f_proj(hidden_states).view(
                *hidden_states.shape[:-1],
                self.num_v_heads,
                self.head_k_dim,
            )
        scalar_gate = self.a_proj(hidden_states)
        if self.hsd_local_core == "gdn":
            return scalar_gate
        if self.kda_f_proj is None:
            raise RuntimeError("HSD-KDA decay projection is missing.")
        vector_gate = self.kda_f_proj(hidden_states).view(
            *hidden_states.shape[:-1],
            self.num_v_heads,
            self.head_k_dim,
        )
        # Keep the legacy scalar gate in the graph with an exact zero gradient.
        # This preserves checkpoint compatibility and avoids DDP unused params.
        return vector_gate + scalar_gate.unsqueeze(-1) * 0.0

    @staticmethod
    def _hsd_debug_enabled() -> bool:
        return os.environ.get('HSD_DEBUG_STATS', '').lower() in {'1', 'true', 'yes', 'on'}

    @staticmethod
    def _hsd_debug_stats_interval() -> int:
        raw = os.environ.get("HSD_DEBUG_STATS_INTERVAL", "").strip()
        if not raw:
            raw = os.environ.get("LOGGING_STEPS", "").strip()
        if not raw:
            return 1
        try:
            return max(1, int(raw))
        except ValueError:
            return 1

    @staticmethod
    def _hsd_debug_stats_max_elements() -> int:
        raw = os.environ.get("HSD_DEBUG_STATS_MAX_ELEMENTS", "").strip()
        if not raw:
            return 1 << 18
        try:
            return max(1, int(raw))
        except ValueError:
            return 1 << 18

    @classmethod
    def _sample_hsd_debug_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        max_elements: int | None = None,
    ) -> torch.Tensor:
        """Deterministically sample token positions for bounded debug stats.

        The complete batch/head/channel slices are retained at every sampled
        position. Sampling only the token axis avoids consuming RNG state and
        prevents CUDA quantile from sorting production-sized gate tensors.
        """
        if tensor.ndim != 4:
            raise ValueError(
                "HSD gate debug sampling expects [batch, time, heads, channels], "
                f"got shape={tuple(tensor.shape)}."
            )
        if tensor.shape[1] == 0:
            return tensor.detach().float()
        if max_elements is None:
            max_elements = cls._hsd_debug_stats_max_elements()
        max_elements = max(1, int(max_elements))
        elements_per_token = tensor.shape[0] * tensor.shape[2] * tensor.shape[3]
        max_tokens = max(1, max_elements // max(1, elements_per_token))
        if tensor.shape[1] > max_tokens:
            token_indices = torch.linspace(
                0,
                tensor.shape[1] - 1,
                steps=max_tokens,
                device=tensor.device,
            ).round().to(dtype=torch.long)
            tensor = tensor.index_select(1, token_indices)
        return tensor.detach().float()

    def _hsd_should_record_debug_stats(self) -> bool:
        if not self._hsd_debug_enabled():
            return False
        interval = self._hsd_debug_stats_interval()
        count = int(getattr(self, "_hsd_debug_forward_count", 0)) + 1
        self._hsd_debug_forward_count = count
        return count % interval == 0

    def _history_gate_floor(self, training_step: int | None) -> float:
        """Return the training-only linearly decayed minimum history weight."""
        if not self.training or self.hsd_history_gate_floor_init <= 0.0:
            return 0.0
        if training_step is None:
            raise RuntimeError(
                "HSD history gate-floor curriculum requires hsd_training_step during training. "
                "The ms-swift trainer injects the optimizer global step automatically."
            )
        step = int(training_step)
        if step < 0:
            raise ValueError(f"hsd_training_step must be non-negative, got {step}.")
        progress = min(1.0, step / float(self.hsd_history_gate_floor_warmup_steps))
        return self.hsd_history_gate_floor_init * (1.0 - progress)

    def _project_summary_query(self, summary_hidden: torch.Tensor) -> torch.Tensor:
        if self.hsd_summary_q_mode == "independent":
            if self.summary_q_proj is None:
                raise RuntimeError("summary_q_proj is required in independent summary-query mode.")
            summary_q_raw = self.summary_q_proj(summary_hidden)
        else:
            if self.summary_q_down_proj is None or self.summary_q_up_proj is None:
                raise RuntimeError("summary-query LoRA projections are missing.")
            # Summary tokens reuse the text/read q projection weights, but do
            # not enter the causal short convolution or mutate its cache.
            summary_q_raw = self.q_proj(summary_hidden)
            summary_q_raw = summary_q_raw + self.summary_q_up_proj(
                self.summary_q_down_proj(summary_hidden)
            )
        return l2norm(
            summary_q_raw.view(
                *summary_hidden.shape[:-1],
                self.num_v_heads,
                self.head_k_dim,
            ).contiguous()
        )

    def _build_history_read_query_raw(
        self,
        local_q: torch.Tensor,
        text_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the pre-normalization per-head query for historical reads."""
        if self.hsd_use_global_read and self.hsd_history_read_q_mode in {
            "independent_projection",
            "full_residual_projection",
        }:
            if self.history_read_q_proj is None:
                raise RuntimeError("full history-read query projection is missing.")
            if text_hidden is None:
                raise ValueError(
                    f"text_hidden is required for hsd_history_read_q_mode="
                    f"'{self.hsd_history_read_q_mode}'."
                )
            projected_history_q = self.history_read_q_proj(text_hidden).view(
                *text_hidden.shape[:-1],
                self.num_heads,
                self.head_k_dim,
            )
            history_q = (
                local_q + projected_history_q
                if self.hsd_history_read_q_mode == "full_residual_projection"
                else projected_history_q
            )
        else:
            history_q = local_q.contiguous()
        if (
            self.hsd_use_global_read
            and self.hsd_history_read_q_mode == "per_head_residual_lora"
        ):
            if self.history_read_q_down_weight is None or self.history_read_q_up_weight is None:
                raise RuntimeError("history-read query LoRA projections are missing.")
            low_rank_q = torch.einsum(
                "bthd,hrd->bthr",
                history_q,
                self.history_read_q_down_weight,
            )
            history_q = history_q + torch.einsum(
                "bthr,hdr->bthd",
                low_rank_q,
                self.history_read_q_up_weight,
            )
        return history_q.contiguous()

    def _project_history_read_query(
        self,
        local_q: torch.Tensor,
        text_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the normalized query used only to read historical states."""
        history_q = self._build_history_read_query_raw(local_q, text_hidden=text_hidden)
        return self._expand_query_to_value_heads(l2norm(history_q))

    def _project_router_query(
        self,
        text_hidden: torch.Tensor,
        local_q: torch.Tensor,
        history_read_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project route-Q and optionally add a per-head local/history-Q LoRA."""
        if self.router_q_proj is None:
            raise RuntimeError("router_q_proj is required when HSD global read is enabled.")
        router_q = self.router_q_proj(text_hidden).view(
            *text_hidden.shape[:-1],
            self.hsd_route_query_heads,
            self.hsd_route_head_dim,
        )
        if self.hsd_router_q_local_lora_rank > 0:
            if (
                self.router_q_local_lora_down_weight is None
                or self.router_q_local_lora_up_weight is None
            ):
                raise RuntimeError("router local-Q LoRA parameters are missing.")
            source_q = local_q
            if self.hsd_router_q_local_lora_source == "history_read_q":
                if history_read_q is None:
                    history_read_q = self._build_history_read_query_raw(
                        local_q,
                        text_hidden=text_hidden,
                    )
                source_q = history_read_q
            if source_q.shape[-2:] != (self.num_heads, self.head_k_dim):
                raise ValueError(
                    "router-Q LoRA source has incompatible head geometry: "
                    f"got {tuple(source_q.shape[-2:])}, expected "
                    f"{(self.num_heads, self.head_k_dim)}."
                )
            normalized_source_q = l2norm(source_q.contiguous())
            low_rank_q = torch.einsum(
                "bthd,hrd->bthr",
                normalized_source_q,
                self.router_q_local_lora_down_weight,
            )
            router_q = router_q + torch.einsum(
                "bthr,hkr->bthk",
                low_rank_q,
                self.router_q_local_lora_up_weight,
            )
        return router_q

    def _project_summary_key_from_state(
        self,
        summary_states: torch.Tensor,
    ) -> torch.Tensor:
        """Build route keys from the exact dense states consumed by global read."""
        if self.hsd_summary_key_source != "state_probe":
            raise RuntimeError("state route-key projection requested while the mode is disabled.")
        if self.summary_state_value_probe is None or self.summary_state_k_proj is None:
            raise RuntimeError("state route-key probe/projection parameters are missing.")
        if not isinstance(summary_states, torch.Tensor) or summary_states.ndim != 5:
            raise ValueError(
                "state-probe route keys require dense [B, M, H, DK, DV] summary states."
            )
        batch_size, summary_count, state_heads, state_k_dim, state_v_dim = (
            int(dim) for dim in summary_states.shape
        )
        expected = (self.num_v_heads, self.head_k_dim, self.head_v_dim)
        if (state_heads, state_k_dim, state_v_dim) != expected:
            raise ValueError(
                "state-probe route-key geometry mismatch: "
                f"got {(state_heads, state_k_dim, state_v_dim)}, expected {expected}."
            )
        if state_heads % self.hsd_route_key_heads != 0:
            raise ValueError(
                "state heads must be divisible by route key heads for state-probe routing."
            )

        heads_per_key = state_heads // self.hsd_route_key_heads
        grouped_states = summary_states.reshape(
            batch_size,
            summary_count,
            self.hsd_route_key_heads,
            heads_per_key,
            state_k_dim,
            state_v_dim,
        ).float().mean(dim=3)
        # A single learned value probe is shared across heads, matching the
        # proposed u_m = S_m p_v factorization. K1 averages all value-head
        # states before the probe; grouped K partitions heads evenly.
        state_features = torch.einsum(
            "bmgkv,v->bmgk",
            grouped_states,
            self.summary_state_value_probe.float(),
        )
        state_features = F.normalize(state_features, p=2.0, dim=-1, eps=1e-6)
        summary_k = F.linear(
            state_features,
            self.summary_state_k_proj.weight.float(),
        )
        return summary_k.to(dtype=self.summary_state_k_proj.weight.dtype)

    @staticmethod
    def _tensor_rms(value: torch.Tensor) -> torch.Tensor:
        if value.numel() == 0:
            return value.new_tensor(0.0)
        return value.pow(2).mean().sqrt()

    def _record_hsd_history_debug_stats(
        self,
        text_o: torch.Tensor,
        history_read: torch.Tensor,
        record_debug_stats: bool | None = None,
    ) -> None:
        if record_debug_stats is None:
            record_debug_stats = self._hsd_should_record_debug_stats()
        if not record_debug_stats:
            return
        with torch.no_grad():
            text = text_o.detach().float()
            history = history_read.detach().float()
            text_plus_history = text + history
            history_norm = history.norm()
            text_norm = text.norm()
            sum_norm = text_plus_history.norm()
            eps = torch.finfo(torch.float32).eps
            history_flat = history.reshape(-1)
            text_flat = text.reshape(-1)
            history_text_cos = torch.dot(history_flat, text_flat) / history_norm.clamp_min(eps) / text_norm.clamp_min(eps)
            history_tokens = history.flatten(2)
            text_tokens = text.flatten(2)
            sum_tokens = text_plus_history.flatten(2)
            token_history_norm = history_tokens.norm(dim=-1)
            token_text_norm = text_tokens.norm(dim=-1)
            history_token_ratio = token_history_norm / token_text_norm.clamp_min(eps)
            flat_history_token_ratio = history_token_ratio.flatten()
            p90_idx = min(
                flat_history_token_ratio.numel() - 1,
                max(0, math.ceil(0.90 * flat_history_token_ratio.numel()) - 1),
            )
            token_cos = (history_tokens * text_tokens).sum(dim=-1)
            token_cos = token_cos / token_history_norm.clamp_min(eps) / token_text_norm.clamp_min(eps)
            token_history_rms = history_tokens.pow(2).mean(dim=-1).sqrt()
            token_text_rms = text_tokens.pow(2).mean(dim=-1).sqrt()
            token_sum_rms = sum_tokens.pow(2).mean(dim=-1).sqrt()
            self._hsd_debug_stats.update({
                'hsd/history_text_norm_ratio': history_norm / text_norm.clamp_min(eps),
                'hsd/history_effective_token_share_gt_0p10': (history_token_ratio > 0.10).float().mean(),
                'hsd/history_token_ratio_p90': flat_history_token_ratio.kthvalue(p90_idx + 1).values,
                'hsd/history_norm': history_norm,
                'hsd/history_rms': self._tensor_rms(history),
                'hsd/text_norm': text_norm,
                'hsd/text_rms': self._tensor_rms(text),
                'hsd/text_o_norm': text_norm,
                'hsd/text_o_rms': self._tensor_rms(text),
                'hsd/text_plus_history_norm': sum_norm,
                'hsd/text_plus_history_rms': self._tensor_rms(text_plus_history),
                'hsd/text_plus_history_text_rms_ratio': (
                    self._tensor_rms(text_plus_history) / self._tensor_rms(text).clamp_min(eps)
                ),
                'hsd/history_text_cos': history_text_cos,
                'hsd/history_token_cos_mean': token_cos.mean(),
                'hsd/history_token_norm_mean': token_history_norm.mean(),
                'hsd/history_token_norm_max': token_history_norm.max(),
                'hsd/text_token_norm_mean': token_text_norm.mean(),
                'hsd/text_token_norm_max': token_text_norm.max(),
                'hsd/history_token_rms_mean': token_history_rms.mean(),
                'hsd/history_token_rms_max': token_history_rms.max(),
                'hsd/text_token_rms_mean': token_text_rms.mean(),
                'hsd/text_token_rms_max': token_text_rms.max(),
                'hsd/text_plus_history_token_rms_mean': token_sum_rms.mean(),
                'hsd/text_plus_history_token_rms_max': token_sum_rms.max(),
                'hsd/history_to_sum_norm_ratio': history_norm / sum_norm.clamp_min(eps),
            })

    def _record_hsd_summary_key_content_debug_stats(
        self,
        *,
        base_key: torch.Tensor | None,
        raw_content_key: torch.Tensor | None,
        content_residual: torch.Tensor | None,
        combined_key: torch.Tensor | None,
        record_debug_stats: bool,
    ) -> None:
        if (
            not record_debug_stats
            or self.hsd_summary_key_content_mode
            not in {"chunk_mean_residual", "chunk_mean_only"}
        ):
            return
        layer_suffix = f"l{int(self.layer_idx):02d}"
        with torch.no_grad():
            self._hsd_debug_stats.update({
                f"hsd/summary_key_content_mode_{self.hsd_summary_key_content_mode}": 1.0,
            })
            if self.summary_key_content_scale is not None:
                self._hsd_debug_stats[f"hsd/summary_key_content_scale_{layer_suffix}"] = (
                    self.summary_key_content_scale.detach().float()
                )
            if raw_content_key is not None:
                self._hsd_debug_stats["hsd/summary_key_content_raw_norm"] = (
                    raw_content_key.detach().float().norm()
                )
            if combined_key is not None:
                self._hsd_debug_stats["hsd/summary_key_combined_norm"] = (
                    combined_key.detach().float().norm()
                )
            if (
                base_key is None
                or raw_content_key is None
                or content_residual is None
                or combined_key is None
                or base_key.numel() == 0
            ):
                return
            eps = torch.finfo(torch.float32).eps
            base = base_key.detach().float()
            raw_content = raw_content_key.detach().float()
            residual = content_residual.detach().float()
            combined = combined_key.detach().float()
            base_norm = base.norm()
            raw_content_norm = raw_content.norm()
            residual_norm = residual.norm()
            combined_norm = combined.norm()
            cosine = F.cosine_similarity(
                base.reshape(-1, base.shape[-1]),
                raw_content.reshape(-1, raw_content.shape[-1]),
                dim=-1,
            ).mean()
            self._hsd_debug_stats.update({
                "hsd/summary_key_base_norm": base_norm,
                "hsd/summary_key_content_raw_norm": raw_content_norm,
                "hsd/summary_key_content_norm": residual_norm,
                "hsd/summary_key_combined_norm": combined_norm,
                "hsd/summary_key_content_base_norm_ratio": (
                    residual_norm / base_norm.clamp_min(eps)
                ),
                "hsd/summary_key_content_base_cos": cosine,
                "hsd/summary_key_combined_base_norm_ratio": (
                    combined_norm / base_norm.clamp_min(eps)
                ),
            })

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        summary_mask = kwargs.pop("summary_mask", None)
        return_hsd_aux_history = bool(kwargs.pop("return_hsd_aux_history", False))
        hsd_training_step = kwargs.pop("hsd_training_step", None)
        route_q_cos = kwargs.pop("hsd_route_q_cos", None)
        route_q_sin = kwargs.pop("hsd_route_q_sin", None)
        route_k_cos = kwargs.pop("hsd_route_k_cos", None)
        route_k_sin = kwargs.pop("hsd_route_k_sin", None)
        text_cu_seqlens = kwargs.get("text_cu_seqlens")
        summary_cu_seqlens = kwargs.get("summary_cu_seqlens")
        explicit_packed_hsd = (
            kwargs.get("cu_seqlens") is not None
            or kwargs.get("cu_seq_lens_q") is not None
            or text_cu_seqlens is not None
            or summary_cu_seqlens is not None
        )
        packed_kwargs = kwargs if explicit_packed_hsd else {
            key: value for key, value in kwargs.items() if key not in {"position_ids", "text_position_ids"}
        }
        expanded_cu_seqlens, _ = get_packed_cu_seqlens(packed_kwargs, device=hidden_states.device)
        packed_hsd = (
            expanded_cu_seqlens is not None
            or text_cu_seqlens is not None
            or summary_cu_seqlens is not None
        )
        last_state = get_layer_cache(self, past_key_values)
        if (use_cache or last_state is not None) and packed_hsd and not explicit_packed_hsd:
            expanded_cu_seqlens = None
            packed_hsd = False
        has_summary_layout = summary_mask is not None
        has_summary_mask = summary_mask is not None and bool(summary_mask.any().item())
        has_hsd_cache = last_state is not None and (
            last_state.get("hsd_summary_states") is not None
            or last_state.get("hsd_chunk_memory_state") is not None
        )
        should_run_hsd = (
            self.hsd_local_core in {"kda", "gdn2"}
            or self.hsd_state_decay_mode == "none"
            or has_summary_mask
            or (self.summary_chunk_size > 0 and (has_summary_layout or use_cache or has_hsd_cache))
        )

        if not should_run_hsd:
            if not self._warned_hsd_fallback_to_gdn:
                logger.warning(
                    "HSDGatedDeltaNet layer %s is falling back to the base GDN path: "
                    "summary_mask_present=%s, summary_mask_has_tokens=%s, use_cache=%s, "
                    "has_hsd_cache=%s, summary_chunk_size=%s.",
                    self.layer_idx,
                    has_summary_layout,
                    has_summary_mask,
                    use_cache,
                    has_hsd_cache,
                    self.summary_chunk_size,
                )
                self._warned_hsd_fallback_to_gdn = True
            base_output = super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )
            if not return_hsd_aux_history:
                return base_output
            aux_history = hidden_states.new_zeros(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.num_v_heads,
                self.head_v_dim,
            )
            aux_history_available = torch.zeros(
                hidden_states.shape[0],
                hidden_states.shape[1],
                1,
                1,
                dtype=torch.bool,
                device=hidden_states.device,
            )
            return (*base_output, aux_history, aux_history_available)
        if attention_mask is not None and not bool(attention_mask.all().item()):
            raise NotImplementedError("HSDGatedDeltaNet V1 does not support padded batches.")
        if summary_mask is None:
            summary_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if summary_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"summary_mask shape {tuple(summary_mask.shape)} must match hidden states {tuple(hidden_states.shape[:2])}"
            )
        if not torch.equal(summary_mask, summary_mask[:1].expand_as(summary_mask)):
            raise NotImplementedError("HSDGatedDeltaNet V1 requires the same summary layout for every batch row.")

        batch_size, seq_len, _ = hidden_states.shape
        text_mask = ~summary_mask
        text_len = int(text_mask[0].sum().item())
        summary_count = int(summary_mask[0].sum().item())
        if text_len == 0:
            raise ValueError("HSDGatedDeltaNet requires at least one text token.")
        log_anchor_enabled = (
            self.hsd_log_anchor_mode != "off" and self.hsd_collect_read_cache
        )
        if log_anchor_enabled:
            if self.training:
                raise RuntimeError("HSD Log-Anchor modes are inference-only.")
            if batch_size != 1:
                raise NotImplementedError("HSD Log-Anchor V1 requires batch_size=1.")
            if packed_hsd:
                raise NotImplementedError("HSD Log-Anchor V1 does not support packed inputs.")
            if self.hsd_log_anchor_mode == "cache" and not use_cache:
                raise RuntimeError(
                    f"HSD_LOG_ANCHOR_MODE={self.hsd_log_anchor_mode!r} requires use_cache=True."
                )
        if packed_hsd:
            if batch_size != 1:
                raise NotImplementedError("HSD packed path requires flattened batch_size=1 inputs.")
            if use_cache or last_state is not None:
                raise NotImplementedError("HSD packed path is training/prefill-only and does not support cache state.")
            if text_cu_seqlens is None or summary_cu_seqlens is None:
                if expanded_cu_seqlens is None:
                    raise ValueError("HSD packed path requires text/summary cu_seqlens or expanded cu_seqlens.")
                text_cu_seqlens, summary_cu_seqlens = self._packed_cu_from_expanded_cu(
                    summary_mask,
                    expanded_cu_seqlens,
                )
            text_cu_seqlens = self._validate_packed_cu_seqlens(
                text_cu_seqlens,
                expected_end=text_len,
                name="text_cu_seqlens",
            )
            summary_cu_seqlens = self._validate_packed_cu_seqlens(
                summary_cu_seqlens,
                expected_end=summary_count,
                name="summary_cu_seqlens",
            )
            if text_cu_seqlens.numel() != summary_cu_seqlens.numel():
                raise ValueError("HSD packed text_cu_seqlens and summary_cu_seqlens must describe the same documents.")
            text_cu_for_kernels = text_cu_seqlens.to(device=hidden_states.device, dtype=torch.long)
        else:
            text_cu_for_kernels = None

        text_hidden = hidden_states[text_mask].view(batch_size, text_len, self.hidden_size)
        summary_hidden = hidden_states[summary_mask]
        summary_hidden = summary_hidden.view(batch_size, summary_count, self.hidden_size)
        past_text_len = 0 if packed_hsd else (
            past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        )
        boundary_counts = self._summary_text_counts(summary_mask) + int(past_text_len)
        record_hsd_debug_stats = (
            self._hsd_should_record_debug_stats() if self.hsd_use_global_read else False
        )

        q, k, v, conv_state = self._project_text(
            text_hidden=text_hidden,
            use_cache=use_cache,
            last_state=last_state,
            cu_seqlens=text_cu_for_kernels,
        )

        erase = None
        write = None
        if self.hsd_local_core == "gdn2":
            if self.gdn2_erase_proj is None or self.gdn2_write_proj is None:
                raise RuntimeError("HSD-GDN2 erase/write projections are missing.")
            erase = self.gdn2_erase_proj(text_hidden).view(
                batch_size, text_len, self.num_v_heads, self.head_k_dim
            ).sigmoid()
            write = self.gdn2_write_proj(text_hidden).view(
                batch_size, text_len, self.num_v_heads, self.head_v_dim
            ).sigmoid()
            beta = None
            chunk_beta = None
        else:
            beta = self.b_proj(text_hidden).sigmoid()
            chunk_beta = (
                self.chunk_b_proj(text_hidden).sigmoid()
                if self.chunk_b_proj is not None else beta
            )
            if self.allow_neg_eigval:
                beta = beta * 2.
                if self.chunk_b_proj is not None:
                    chunk_beta = chunk_beta * 2.
        g = self._project_decay_gate(text_hidden)
        if record_hsd_debug_stats and erase is not None and write is not None:
            with torch.no_grad():
                erase_f = self._sample_hsd_debug_tensor(erase)
                write_f = self._sample_hsd_debug_tensor(write)
                decay_g = self._sample_hsd_debug_tensor(g)
                decay_factor = self._g_decay_from_raw(decay_g).exp()
                erase_q = torch.quantile(
                    erase_f.flatten(),
                    erase_f.new_tensor([0.1, 0.5, 0.9]),
                )
                write_q = torch.quantile(
                    write_f.flatten(),
                    write_f.new_tensor([0.1, 0.5, 0.9]),
                )
                decay_q = torch.quantile(
                    decay_factor.flatten(),
                    decay_factor.new_tensor([0.1, 0.9]),
                )
                self._hsd_debug_stats.update({
                    "hsd/gdn2_erase_mean": erase_f.mean(),
                    "hsd/gdn2_erase_p10": erase_q[0],
                    "hsd/gdn2_erase_p50": erase_q[1],
                    "hsd/gdn2_erase_p90": erase_q[2],
                    "hsd/gdn2_write_mean": write_f.mean(),
                    "hsd/gdn2_write_p10": write_q[0],
                    "hsd/gdn2_write_p50": write_q[1],
                    "hsd/gdn2_write_p90": write_q[2],
                    "hsd/gdn2_decay_factor_mean": decay_factor.mean(),
                    "hsd/gdn2_decay_factor_p10": decay_q[0],
                    "hsd/gdn2_decay_factor_p90": decay_q[1],
                    "hsd/gdn2_decay_anisotropy_std": (
                        decay_factor.std(dim=-1, unbiased=False).mean()
                    ),
                })
        if self.summary_key_content_scale is not None:
            # Keep the zero-initialized scalar in the graph even for rare
            # batches without a complete chunk, so DDP does not see it as
            # conditionally unused.
            g = g + self.summary_key_content_scale.to(dtype=g.dtype) * 0.0
        if self.hsd_summary_key_source == "state_probe" and self.hsd_collect_read_cache:
            if (
                self.summary_k_proj is None
                or self.summary_state_value_probe is None
                or self.summary_state_k_proj is None
            ):
                raise RuntimeError("state route-key parameters are not initialized.")
            # summary_k_proj remains in the checkpoint to preserve the exact
            # common baseline schema/initialization. It is intentionally
            # inactive in this mode, while the scalar zero edge also keeps all
            # route-key parameters DDP-safe on ranks with no complete chunk.
            state_route_zero = (
                self.summary_k_proj.weight.reshape(-1)[0]
                + self.summary_state_value_probe.reshape(-1)[0]
                + self.summary_state_k_proj.weight.reshape(-1)[0]
            ) * 0.0
            g = g + state_route_zero.to(dtype=g.dtype)

        collect_summary_cache = self.hsd_collect_read_cache
        content_pending_hidden_sum = None
        content_pending_token_count = 0
        content_base_key = None
        content_raw_key = None
        content_residual = None
        content_combined_key = None
        chunk_mean_hidden = None
        if (
            collect_summary_cache
            and self.hsd_summary_key_content_mode
            in {"chunk_mean_residual", "chunk_mean_only"}
        ):
            if packed_hsd:
                chunk_mean_hidden = self._chunk_mean_hidden_packed(
                    text_hidden,
                    boundary_counts,
                    text_cu_seqlens=text_cu_seqlens,
                    summary_cu_seqlens=summary_cu_seqlens,
                    summary_chunk_size=self.summary_chunk_size,
                    summary_token_num=self.summary_token_num,
                )
            else:
                cached_hidden_sum = (
                    last_state.get("hsd_route_content_hidden_sum")
                    if last_state is not None else None
                )
                cached_token_count = (
                    int(last_state.get("hsd_route_content_token_count", 0))
                    if last_state is not None else 0
                )
                if int(past_text_len) > 0 and cached_hidden_sum is None:
                    raise RuntimeError(
                        "Cached HSD chunk-mean route keys require hsd_route_content_hidden_sum "
                        "from the prefill cache."
                    )
                expected_cached_count = int(past_text_len) % int(self.summary_chunk_size)
                if cached_token_count != expected_cached_count:
                    raise RuntimeError(
                        "Cached HSD chunk-mean route-key token count is inconsistent with text length: "
                        f"cached={cached_token_count}, expected={expected_cached_count}, "
                        f"past_text_len={int(past_text_len)}."
                    )
                (
                    chunk_mean_hidden,
                    content_pending_hidden_sum,
                    content_pending_token_count,
                ) = self._chunk_mean_hidden_stream(
                    text_hidden,
                    boundary_counts,
                    summary_chunk_size=self.summary_chunk_size,
                    summary_token_num=self.summary_token_num,
                    boundary_start=int(past_text_len),
                    initial_hidden_sum=cached_hidden_sum,
                    initial_token_count=cached_token_count,
                )
        if summary_count:
            summary_q = self._project_summary_query(summary_hidden)
            if collect_summary_cache:
                if self.summary_k_proj is None:
                    raise RuntimeError("summary_k_proj is required when HSD summary cache collection is enabled.")
                if self.hsd_summary_key_content_mode in {
                    "chunk_mean_residual",
                    "chunk_mean_only",
                }:
                    if chunk_mean_hidden is None:
                        raise RuntimeError("HSD chunk-mean summary key is not initialized.")
                    if chunk_mean_hidden.shape[:2] != summary_hidden.shape[:2]:
                        raise RuntimeError(
                            "HSD chunk-mean content count must match summary count: "
                            f"content={tuple(chunk_mean_hidden.shape[:2])}, "
                            f"summary={tuple(summary_hidden.shape[:2])}."
                        )
                    content_raw_key = self.summary_k_proj(
                        chunk_mean_hidden.to(dtype=summary_hidden.dtype)
                    ).view(
                        batch_size,
                        summary_count,
                        self.hsd_route_key_heads,
                        self.hsd_route_head_dim,
                    )
                    if self.hsd_summary_key_content_mode == "chunk_mean_residual":
                        if self.summary_key_content_scale is None:
                            raise RuntimeError(
                                "HSD chunk-mean summary-key residual scale is not initialized."
                            )
                        content_base_key = self.summary_k_proj(summary_hidden).view(
                            batch_size,
                            summary_count,
                            self.hsd_route_key_heads,
                            self.hsd_route_head_dim,
                        )
                        content_residual = content_raw_key * self.summary_key_content_scale.to(
                            dtype=content_raw_key.dtype
                        )
                        content_combined_key = content_base_key + content_residual
                    else:
                        content_combined_key = content_raw_key
                    summary_k_flat = content_combined_key.reshape(
                        batch_size, summary_count, -1
                    )
                else:
                    summary_k_flat = self.summary_k_proj(summary_hidden)
                summary_k = summary_k_flat.view(
                    batch_size, summary_count, self.hsd_route_key_heads, self.hsd_route_head_dim
                )
                self._record_hsd_summary_key_content_debug_stats(
                    base_key=content_base_key,
                    raw_content_key=content_raw_key,
                    content_residual=content_residual,
                    combined_key=content_combined_key,
                    record_debug_stats=record_hsd_debug_stats,
                )
                if self.hsd_route_qk_l2norm:
                    summary_k = l2norm(summary_k.contiguous())
            else:
                summary_k = None
        else:
            summary_q = q.new_zeros(batch_size, 0, self.num_v_heads, self.head_k_dim)
            summary_k = q.new_zeros(
                batch_size, 0, self.hsd_route_key_heads, self.hsd_route_head_dim
            ) if collect_summary_cache else None
            self._record_hsd_summary_key_content_debug_stats(
                base_key=None,
                raw_content_key=None,
                content_residual=None,
                combined_key=None,
                record_debug_stats=record_hsd_debug_stats,
            )
        if (
            self.hsd_use_global_read
            and self.hsd_route_weight_momentum_gamma > 0.0
            and self.hsd_decode_momentum_init_mode == "last_prompt_raw"
            and bool(use_cache)
            and int(past_text_len) > 0
            and text_len > 1
        ):
            raise RuntimeError(
                "hsd_decode_momentum_init_mode='last_prompt_raw' does not support chunked prefill: "
                "received more than one text token with a non-empty cache. Use a single-call prefill "
                "or hsd_decode_momentum_init_mode='carry'."
            )
        if self.hsd_route_position_encoding == "rope" and collect_summary_cache:
            if route_k_cos is None or route_k_sin is None:
                raise RuntimeError("route RoPE is enabled but summary-key cos/sin sidecars are missing.")
            summary_k = _apply_hsd_route_rope(
                summary_k,
                route_k_cos,
                route_k_sin,
                self.hsd_route_rope_dim,
            )
        initial_state = last_state.get("recurrent_state") if last_state is not None else None
        chunk_memory_state = None
        if self.hsd_history_state_mode == "chunk_reset_no_decay":
            chunk_initial_state = (
                last_state.get("hsd_chunk_memory_state") if last_state is not None else None
            )
            text_o, summary_o, new_summary_states, recurrent_state, chunk_memory_state = (
                self._run_dual_state_segments(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    chunk_beta=chunk_beta,
                    summary_q=summary_q,
                    boundary_counts=boundary_counts,
                    boundary_start=int(past_text_len),
                    initial_state=initial_state,
                    chunk_initial_state=chunk_initial_state,
                    collect_summary_states=collect_summary_cache,
                    text_cu_seqlens=text_cu_for_kernels,
                    summary_cu_seqlens=summary_cu_seqlens,
                    record_debug_stats=record_hsd_debug_stats,
                )
            )
        else:
            text_o, summary_o, new_summary_states, recurrent_state = self._run_hsd_segments(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                erase=erase,
                write=write,
                summary_q=summary_q,
                boundary_counts=boundary_counts,
                boundary_start=int(past_text_len),
                initial_state=initial_state,
                collect_summary_states=collect_summary_cache,
                text_cu_seqlens=text_cu_for_kernels,
                summary_cu_seqlens=summary_cu_seqlens,
            )

        delta_previous_boundary_state = None
        delta_pending_log_decay = None
        if (
            collect_summary_cache
            and self.hsd_history_candidate_mode == "delta_from_continuous"
        ):
            if new_summary_states is None:
                raise RuntimeError("HSD delta-state candidates require collected continuous summary states.")
            continuous_summary_states = new_summary_states
            previous_boundary_state = (
                last_state.get("hsd_delta_prev_boundary_state")
                if last_state is not None else None
            )
            previous_pending_log_decay = (
                last_state.get("hsd_delta_pending_log_decay")
                if last_state is not None else None
            )
            (
                new_summary_states,
                delta_gammas,
                delta_previous_boundary_state,
                delta_pending_log_decay,
            ) = self._continuous_states_to_delta_candidates(
                continuous_states=continuous_summary_states,
                effective_g=self._effective_g_decay(g),
                boundary_counts=boundary_counts,
                boundary_start=int(past_text_len),
                text_cu_seqlens=text_cu_for_kernels,
                summary_cu_seqlens=summary_cu_seqlens,
                previous_boundary_state=previous_boundary_state,
                pending_log_decay=previous_pending_log_decay,
            )
            self._record_delta_candidate_debug_stats(
                continuous_states=continuous_summary_states,
                delta_states=new_summary_states,
                gammas=delta_gammas,
                record_debug_stats=record_hsd_debug_stats,
            )
        elif record_hsd_debug_stats:
            self._hsd_debug_stats["hsd/history_candidate_delta"] = 0.0

        if collect_summary_cache and self.hsd_summary_key_source == "state_probe":
            if new_summary_states is None:
                raise RuntimeError("state-probe route keys require collected summary states.")
            summary_k = self._project_summary_key_from_state(new_summary_states)
            if self.hsd_route_qk_l2norm:
                summary_k = l2norm(summary_k.contiguous())
            if self.hsd_route_position_encoding == "rope":
                if route_k_cos is None or route_k_sin is None:
                    raise RuntimeError(
                        "route RoPE is enabled but summary-key cos/sin sidecars are missing."
                    )
                summary_k = _apply_hsd_route_rope(
                    summary_k,
                    route_k_cos,
                    route_k_sin,
                    self.hsd_route_rope_dim,
                )
            if record_hsd_debug_stats:
                with torch.no_grad():
                    self._hsd_debug_stats.update({
                        "hsd/summary_key_source_state_probe": 1.0,
                        "hsd/summary_state_value_probe_norm": (
                            self.summary_state_value_probe.detach().float().norm()
                        ),
                        "hsd/summary_state_route_key_norm": (
                            summary_k.detach().float().norm(dim=-1).mean()
                            if summary_k.numel() > 0 else summary_k.new_tensor(0.0)
                        ),
                    })

        text_output_scale = _env_float(
            "HSD_TEXT_OUTPUT_SCALE",
            _env_float("HSD_TEXT_O_OUTPUT_SCALE", 1.0),
        )
        if text_output_scale < 0 or not math.isfinite(text_output_scale):
            raise ValueError("HSD_TEXT_OUTPUT_SCALE must be a non-negative finite value.")
        if text_output_scale != 1.0:
            text_o = text_o * text_output_scale

        summary_states, summary_keys, summary_boundaries = None, None, None
        cache_summary_states, cache_summary_keys, cache_summary_boundaries = None, None, None
        log_anchor_active_counts = None
        log_anchor_cache_blocks = None
        log_anchor_total_chunks = 0
        if collect_summary_cache:
            summary_states, summary_keys, summary_boundaries = self._merge_summary_cache(
                last_state=last_state,
                new_summary_states=new_summary_states,
                new_summary_keys=summary_k,
                new_summary_boundaries=boundary_counts,
            )
            cache_summary_states = summary_states
            cache_summary_keys = summary_keys
            cache_summary_boundaries = summary_boundaries
            if log_anchor_enabled:
                if not isinstance(summary_states, torch.Tensor):
                    raise RuntimeError("HSD Log-Anchor V1 requires dense tensor states.")
                old_states = (
                    last_state.get("hsd_summary_states")
                    if last_state is not None else None
                )
                old_bank_size = 0 if old_states is None else int(old_states.shape[1])
                (
                    log_anchor_active_counts,
                    final_blocks,
                    log_anchor_total_chunks,
                ) = self._build_log_anchor_plan(
                    last_state=last_state,
                    old_bank_size=old_bank_size,
                    new_summary_boundaries=boundary_counts,
                    total_bank_size=int(summary_states.shape[1]),
                    text_len=text_len,
                    start_index=int(past_text_len),
                    device=summary_states.device,
                )
                if record_hsd_debug_stats:
                    with torch.no_grad():
                        represented = log_anchor_active_counts.sum(dim=-1).float()
                        active = log_anchor_active_counts.gt(0).sum(dim=-1).float()
                        visible_rows = represented.gt(0)
                        self._hsd_debug_stats.update({
                            "hsd/log_anchor_mode_active": 1.0,
                            "hsd/log_anchor_persistent_anchor_count": float(len(final_blocks)),
                            "hsd/log_anchor_total_chunks": float(log_anchor_total_chunks),
                        })
                        if bool(visible_rows.any().item()):
                            self._hsd_debug_stats.update({
                                "hsd/log_anchor_visible_anchor_count_mean": active[visible_rows].mean(),
                                "hsd/log_anchor_visible_represented_chunks_mean": represented[visible_rows].mean(),
                            })
                if self.hsd_log_anchor_mode == "cache":
                    (
                        cache_summary_states,
                        cache_summary_keys,
                        cache_summary_boundaries,
                        log_anchor_cache_blocks,
                    ) = self._compact_log_anchor_bank(
                        summary_states,
                        summary_keys,
                        summary_boundaries,
                        final_blocks,
                    )
            if (
                record_hsd_debug_stats
                and self.hsd_local_core == "gdn2"
                and new_summary_states is not None
                and new_summary_states.numel() > 0
            ):
                with torch.no_grad():
                    state_norm = torch.linalg.vector_norm(
                        new_summary_states.detach().float(),
                        dim=(-2, -1),
                    )
                    self._hsd_debug_stats.update({
                        "hsd/gdn2_snapshot_state_norm_mean": state_norm.mean(),
                        "hsd/gdn2_snapshot_state_norm_p90": state_norm.quantile(0.9),
                    })

        aux_history = text_o * 0.0 if return_hsd_aux_history else None
        aux_history_available = (
            torch.zeros(
                batch_size,
                text_len,
                1,
                1,
                dtype=torch.bool,
                device=text_o.device,
            )
            if return_hsd_aux_history else None
        )
        momentum_final_summary = None
        momentum_final_null = None
        if self.hsd_use_global_read and summary_states is not None and summary_states.shape[1] > 0:
            if self.router_q_proj is None:
                raise RuntimeError("router_q_proj is required when HSD global read is enabled.")
            if record_hsd_debug_stats:
                self._hsd_debug_stats['hsd/state_decay_none'] = float(
                    self.hsd_state_decay_mode == "none"
                )
            raw_history_read_q = self._build_history_read_query_raw(
                q,
                text_hidden=text_hidden,
            )
            read_q = self._expand_query_to_value_heads(l2norm(raw_history_read_q))
            if record_hsd_debug_stats:
                with torch.no_grad():
                    local_read_q = self._expand_query_to_value_heads(l2norm(q.contiguous()))
                    delta = read_q.detach().float() - local_read_q.detach().float()
                    baseline = local_read_q.detach().float()
                    delta_ratio = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(
                        baseline
                    ).clamp_min(1e-12)
                    cosine = torch.nn.functional.cosine_similarity(
                        read_q.detach().float(),
                        baseline,
                        dim=-1,
                    ).mean()
                    self._hsd_debug_stats.update({
                        "hsd/history_read_q_per_head_residual_lora": float(
                            self.hsd_history_read_q_mode == "per_head_residual_lora"
                        ),
                        "hsd/history_read_q_independent_projection": float(
                            self.hsd_history_read_q_mode == "independent_projection"
                        ),
                        "hsd/history_read_q_full_residual_projection": float(
                            self.hsd_history_read_q_mode == "full_residual_projection"
                        ),
                        "hsd/history_read_q_delta_norm_ratio": delta_ratio,
                        "hsd/history_read_q_local_cos": cosine,
                    })
            router_q = self._project_router_query(
                text_hidden,
                q,
                history_read_q=raw_history_read_q,
            )
            if record_hsd_debug_stats:
                with torch.no_grad():
                    base_router_q = self.router_q_proj(text_hidden).view_as(router_q)
                    route_delta = router_q.detach().float() - base_router_q.detach().float()
                    base_router_q = base_router_q.detach().float()
                    self._hsd_debug_stats.update({
                        "hsd/router_q_local_lora": float(
                            self.hsd_router_q_local_lora_rank > 0
                        ),
                        "hsd/router_q_local_lora_source_history_read_q": float(
                            self.hsd_router_q_local_lora_source == "history_read_q"
                        ),
                        "hsd/router_q_local_lora_delta_norm_ratio": (
                            torch.linalg.vector_norm(route_delta)
                            / torch.linalg.vector_norm(base_router_q).clamp_min(1e-12)
                        ),
                        "hsd/router_q_local_lora_base_cos": torch.nn.functional.cosine_similarity(
                            router_q.detach().float(),
                            base_router_q,
                            dim=-1,
                        ).mean(),
                    })
            if self.hsd_route_qk_l2norm:
                router_q = l2norm(router_q.contiguous())
            if self.hsd_route_position_encoding == "rope":
                if route_q_cos is None or route_q_sin is None:
                    raise RuntimeError("route RoPE is enabled but query cos/sin sidecars are missing.")
                router_q = _apply_hsd_route_rope(
                    router_q,
                    route_q_cos,
                    route_q_sin,
                    self.hsd_route_rope_dim,
                )
            null_logits = self._null_gate_logits(text_hidden)
            convex_mix = self.hsd_history_mix_mode == "convex"
            joint_softmax_mix = self.hsd_history_mix_mode == "joint_softmax_logm"
            gated_additive_mix = self.hsd_history_mix_mode == "gated_additive"
            null_logm_enabled = self.hsd_null_logm_scale > 0.0
            if (convex_mix or joint_softmax_mix or gated_additive_mix) and null_logits is None:
                raise RuntimeError(f"HSD {self.hsd_history_mix_mode} history mix requires local gate logits.")
            if null_logm_enabled and null_logits is None:
                raise RuntimeError("HSD null LogMeanExp correction requires local/null logits.")
            candidate_counts = None
            history_null_logits = None if (convex_mix or gated_additive_mix) else null_logits
            momentum_enabled = self.hsd_route_weight_momentum_gamma > 0.0
            reset_last_prompt_token = bool(
                momentum_enabled
                and use_cache
                and int(past_text_len) == 0
                and text_len > 1
                and self.hsd_decode_momentum_init_mode == "last_prompt_raw"
            )
            momentum_initial_summary = (
                last_state.get("hsd_route_momentum_summary_weights")
                if momentum_enabled and last_state is not None else None
            )
            momentum_initial_null = (
                last_state.get("hsd_route_momentum_null_weight")
                if momentum_enabled and last_state is not None else None
            )
            if joint_softmax_mix or null_logm_enabled:
                candidate_counts = self._history_candidate_counts(
                    boundary_counts=summary_boundaries,
                    text_len=text_len,
                    batch_size=batch_size,
                    start_index=int(past_text_len),
                    topk=(max(0, int(self.hsd_read_topk)) if joint_softmax_mix else 0),
                    text_cu_seqlens=text_cu_seqlens if packed_hsd else None,
                    summary_cu_seqlens=summary_cu_seqlens if packed_hsd else None,
                )
            if joint_softmax_mix:
                history_null_logits = self._joint_softmax_logm_logits(null_logits, candidate_counts)
                if record_hsd_debug_stats:
                    with torch.no_grad():
                        self._hsd_debug_stats.update({
                            'hsd/router_valid_candidate_count': candidate_counts.float().mean(),
                            'hsd/router_local_logit_raw': null_logits.detach().float().mean(),
                            'hsd/router_local_logit_logm_corrected': history_null_logits.detach().float().mean(),
                        })
            elif null_logm_enabled:
                history_null_logits = self._null_logm_corrected_logits(
                    null_logits,
                    candidate_counts,
                    scale=self.hsd_null_logm_scale,
                )
            return_router_local_prob = bool(
                joint_softmax_mix or (null_logm_enabled and record_hsd_debug_stats)
            )
            if packed_hsd:
                history_result = self._history_read_packed(
                    read_q=read_q,
                    router_q=router_q,
                    states=summary_states,
                    summary_k=summary_keys,
                    boundary_counts=summary_boundaries,
                    text_cu_seqlens=text_cu_seqlens,
                    summary_cu_seqlens=summary_cu_seqlens,
                    null_logits=history_null_logits,
                    record_debug_stats=record_hsd_debug_stats,
                    return_local_prob=return_router_local_prob,
                )
            else:
                history_result = self._history_read(
                    read_q=read_q,
                    router_q=router_q,
                    states=summary_states,
                    summary_k=summary_keys,
                    boundary_counts=summary_boundaries,
                    text_len=text_len,
                    start_index=int(past_text_len),
                    null_logits=history_null_logits,
                    record_debug_stats=record_hsd_debug_stats,
                    try_fused_decode=bool(use_cache and text_len == 1),
                    return_local_prob=return_router_local_prob,
                    momentum_initial_summary=momentum_initial_summary,
                    momentum_initial_null=momentum_initial_null,
                    return_momentum_state=bool(momentum_enabled and use_cache),
                    reset_last_token=reset_last_prompt_token,
                    log_anchor_counts=log_anchor_active_counts,
                )
            if momentum_enabled and use_cache and not packed_hsd:
                history_read, local_prob, momentum_final_summary, momentum_final_null = history_result
            elif return_router_local_prob:
                history_read, local_prob = history_result
            else:
                history_read = history_result
                local_prob = None
            if null_logm_enabled and record_hsd_debug_stats:
                if local_prob is None or candidate_counts is None:
                    raise RuntimeError("HSD null LogMeanExp debug statistics require local probabilities and counts.")
                self._record_hsd_null_logm_debug_stats(
                    candidate_counts=candidate_counts,
                    local_prob=local_prob,
                    raw_null_logits=null_logits,
                    corrected_null_logits=history_null_logits,
                )
            if return_hsd_aux_history:
                if not gated_additive_mix:
                    raise RuntimeError(
                        "History-only auxiliary loss currently requires "
                        "hsd_history_mix_mode='gated_additive'."
                    )
                aux_history = history_read
                aux_history_available = self._history_available_mask(
                    boundary_counts=summary_boundaries,
                    text_len=text_len,
                    batch_size=batch_size,
                    start_index=int(past_text_len),
                    text_cu_seqlens=text_cu_seqlens if packed_hsd else None,
                    summary_cu_seqlens=summary_cu_seqlens if packed_hsd else None,
                )
            history_read_scale = _env_float(
                "HSD_GLOBAL_READ_OUTPUT_SCALE",
                _env_float("HSD_HISTORY_READ_OUTPUT_SCALE", 1.0),
            )
            if history_read_scale < 0 or not math.isfinite(history_read_scale):
                raise ValueError("HSD_GLOBAL_READ_OUTPUT_SCALE must be a non-negative finite value.")
            if history_read_scale != 1.0:
                history_read = history_read * history_read_scale
            if joint_softmax_mix:
                history_available = candidate_counts > 0
                mixed_o = local_prob.to(text_o.dtype) * text_o + history_read.to(text_o.dtype)
                self._record_hsd_history_debug_stats(
                    text_o=local_prob.to(text_o.dtype) * text_o,
                    history_read=history_read,
                    record_debug_stats=record_hsd_debug_stats,
                )
                self._record_hsd_history_mix_debug_stats(
                    local_o=text_o,
                    conditional_history=history_read,
                    mixed_o=mixed_o,
                    local_alpha=local_prob,
                    history_available=history_available,
                    record_debug_stats=record_hsd_debug_stats,
                    history_is_joint_weighted=True,
                    mix_mode="joint_softmax_logm",
                )
                text_o = mixed_o
            elif convex_mix:
                history_available = self._history_available_mask(
                    boundary_counts=summary_boundaries,
                    text_len=text_len,
                    batch_size=batch_size,
                    start_index=int(past_text_len),
                    text_cu_seqlens=text_cu_seqlens if packed_hsd else None,
                    summary_cu_seqlens=summary_cu_seqlens if packed_hsd else None,
                )
                mixed_o, local_alpha = self._mix_local_and_history(
                    local_o=text_o,
                    history_o=history_read,
                    local_logits=null_logits,
                    history_available=history_available,
                )
                actual_history = (1.0 - local_alpha) * history_read.to(text_o.dtype)
                self._record_hsd_history_debug_stats(
                    text_o=text_o,
                    history_read=actual_history,
                    record_debug_stats=record_hsd_debug_stats,
                )
                self._record_hsd_history_mix_debug_stats(
                    local_o=text_o,
                    conditional_history=history_read,
                    mixed_o=mixed_o,
                    local_alpha=local_alpha,
                    history_available=history_available,
                    record_debug_stats=record_hsd_debug_stats,
                    mix_mode="convex",
                )
                text_o = mixed_o
            elif gated_additive_mix:
                history_available = self._history_available_mask(
                    boundary_counts=summary_boundaries,
                    text_len=text_len,
                    batch_size=batch_size,
                    start_index=int(past_text_len),
                    text_cu_seqlens=text_cu_seqlens if packed_hsd else None,
                    summary_cu_seqlens=summary_cu_seqlens if packed_hsd else None,
                )
                if self.history_head_log_scale is not None:
                    history_read = self._scale_history_by_head(
                        history_read,
                        self.history_head_log_scale.exp(),
                    )
                history_gate_floor = self._history_gate_floor(hsd_training_step)
                mixed_o, local_alpha = self._mix_local_with_gated_history(
                    local_o=text_o,
                    history_o=history_read,
                    local_logits=null_logits,
                    history_available=history_available,
                    history_gate_floor=history_gate_floor,
                )
                effective_history_alpha = self._history_alpha_with_floor(
                    local_alpha=local_alpha,
                    history_available=history_available,
                    history_gate_floor=history_gate_floor,
                )
                actual_history = effective_history_alpha * history_read.to(text_o.dtype)
                self._record_hsd_history_debug_stats(
                    text_o=text_o,
                    history_read=actual_history,
                    record_debug_stats=record_hsd_debug_stats,
                )
                self._record_hsd_history_mix_debug_stats(
                    local_o=text_o,
                    conditional_history=history_read,
                    mixed_o=mixed_o,
                    local_alpha=local_alpha,
                    effective_history_alpha=effective_history_alpha,
                    history_available=history_available,
                    record_debug_stats=record_hsd_debug_stats,
                    mix_mode="gated_additive",
                    history_gate_floor=history_gate_floor,
                )
                text_o = mixed_o
            else:
                if self.history_head_log_scale is not None:
                    history_read = self._scale_history_by_head(
                        history_read,
                        self.history_head_log_scale.exp(),
                    )
                    self._record_hsd_additive_head_scale_debug_stats(
                        local_o=text_o,
                        history_o=history_read,
                        record_debug_stats=record_hsd_debug_stats,
                    )
                self._record_hsd_history_debug_stats(
                    text_o=text_o,
                    history_read=history_read,
                    record_debug_stats=record_hsd_debug_stats,
                )
                text_o = text_o + history_read.to(text_o.dtype)

        if self.hsd_use_global_read and self.hsd_history_read_q_mode == "per_head_residual_lora":
            if self.history_read_q_down_weight is None or self.history_read_q_up_weight is None:
                raise RuntimeError("history-read query LoRA projections are missing.")
            # A packed rank can occasionally contain no complete summary chunk.
            # Keep both adapter parameters in the graph in that case so ZeRO/DDP
            # observes the same parameter set on every rank.  The dependency is
            # exactly zero and therefore does not change the model output.
            history_read_q_zero = (
                self.history_read_q_down_weight.float().sum()
                + self.history_read_q_up_weight.float().sum()
            ) * 0.0
            text_o = text_o + history_read_q_zero.to(dtype=text_o.dtype)
        elif (
            self.hsd_use_global_read
            and self.hsd_history_read_q_mode in {
                "independent_projection",
                "full_residual_projection",
            }
            and (summary_states is None or summary_states.shape[1] == 0)
        ):
            if self.history_read_q_proj is None:
                raise RuntimeError("independent history-read query projection is missing.")
            # Some packed ranks may contain no complete summary chunk. Keep
            # the full projection in the graph so ZeRO/DDP sees the same
            # parameter set on every rank without changing the output. One
            # element is sufficient; avoid reducing the full 35M-parameter
            # projection on ordinary training batches.
            history_read_q_zero = self.history_read_q_proj.weight.reshape(-1)[0] * 0.0
            text_o = text_o + history_read_q_zero.to(dtype=text_o.dtype)

        if self.hsd_use_global_read and self.hsd_router_q_local_lora_rank > 0:
            if (
                self.router_q_local_lora_down_weight is None
                or self.router_q_local_lora_up_weight is None
            ):
                raise RuntimeError("router local-Q LoRA parameters are missing.")
            # Packed ranks may contain no complete summary chunk and therefore
            # never evaluate router-Q. Keep these adapter parameters in every
            # rank's graph with an exact zero dependency.
            router_q_local_lora_zero = (
                self.router_q_local_lora_down_weight.float().sum()
                + self.router_q_local_lora_up_weight.float().sum()
            ) * 0.0
            text_o = text_o + router_q_local_lora_zero.to(dtype=text_o.dtype)

        if use_cache:
            cache_state = update_layer_cache(
                self,
                past_key_values,
                recurrent_state=recurrent_state,
                conv_state=conv_state if self.use_short_conv else None,
                offset=text_len,
            )
            if cache_state is not None and collect_summary_cache:
                cache_state["hsd_summary_states"] = cache_summary_states
                cache_state["hsd_summary_keys"] = cache_summary_keys
                cache_state["hsd_summary_boundaries"] = cache_summary_boundaries
                if log_anchor_enabled and self.hsd_log_anchor_mode == "cache":
                    if log_anchor_cache_blocks is None or cache_summary_boundaries is None:
                        raise RuntimeError("HSD Log-Anchor cache metadata was not prepared.")
                    self._write_log_anchor_metadata(
                        cache_state,
                        log_anchor_cache_blocks,
                        total_chunks=log_anchor_total_chunks,
                        device=cache_summary_boundaries.device,
                    )
                elif log_anchor_enabled:
                    for key in _HSD_LOG_ANCHOR_CACHE_KEYS:
                        cache_state.pop(key, None)
                if self.hsd_summary_key_content_mode in {
                    "chunk_mean_residual",
                    "chunk_mean_only",
                }:
                    if content_pending_hidden_sum is None:
                        raise RuntimeError(
                            "Cached HSD chunk-mean route keys require a pending hidden-sum accumulator."
                        )
                    cache_state["hsd_route_content_hidden_sum"] = content_pending_hidden_sum
                    cache_state["hsd_route_content_token_count"] = int(
                        content_pending_token_count
                    )
                if self.hsd_route_weight_momentum_gamma > 0.0 and momentum_final_summary is not None:
                    cache_state["hsd_route_momentum_summary_weights"] = momentum_final_summary
                    if momentum_final_null is not None:
                        cache_state["hsd_route_momentum_null_weight"] = momentum_final_null
                    else:
                        cache_state.pop("hsd_route_momentum_null_weight", None)
                if self.hsd_history_candidate_mode == "delta_from_continuous":
                    if delta_previous_boundary_state is not None:
                        cache_state["hsd_delta_prev_boundary_state"] = delta_previous_boundary_state
                    else:
                        cache_state.pop("hsd_delta_prev_boundary_state", None)
                    if delta_pending_log_decay is None:
                        raise RuntimeError("Cached HSD delta-state mode requires pending decay metadata.")
                    cache_state["hsd_delta_pending_log_decay"] = delta_pending_log_decay
            if cache_state is not None and self.hsd_history_state_mode == "chunk_reset_no_decay":
                cache_state["hsd_chunk_memory_state"] = chunk_memory_state

        expanded_o = hidden_states.new_zeros(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        expanded_o[text_mask] = text_o.reshape(-1, self.num_v_heads, self.head_v_dim)
        expanded_o[summary_mask] = summary_o.reshape(-1, self.num_v_heads, self.head_v_dim)

        if self.use_gate:
            gate = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            expanded_o = self.o_norm(expanded_o, gate)
        else:
            expanded_o = self.o_norm(expanded_o)
        expanded_o = rearrange(expanded_o, 'b t h d -> b t (h d)')
        output = self.o_proj(expanded_o)
        if return_hsd_aux_history:
            return output, None, past_key_values, aux_history, aux_history_available
        return output, None, past_key_values

    def _project_text(
        self,
        text_hidden: torch.Tensor,
        use_cache: bool = False,
        last_state: dict | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None and last_state.get("conv_state") is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(text_hidden),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(text_hidden),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(text_hidden),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            conv_state = (conv_state_q, conv_state_k, conv_state_v)
        else:
            q = F.silu(self.q_proj(text_hidden))
            k = F.silu(self.k_proj(text_hidden))
            v = F.silu(self.v_proj(text_hidden))
            conv_state = None
        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        return q, k, v, conv_state

    @staticmethod
    def _merge_summary_cache(
        last_state: dict | None,
        new_summary_states: torch.Tensor,
        new_summary_keys: torch.Tensor,
        new_summary_boundaries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        old_states = last_state.get("hsd_summary_states") if last_state is not None else None
        old_keys = last_state.get("hsd_summary_keys") if last_state is not None else None
        old_boundaries = last_state.get("hsd_summary_boundaries") if last_state is not None else None
        if new_summary_states.shape[1] == 0:
            if old_states is None:
                return new_summary_states, new_summary_keys, new_summary_boundaries
            return old_states, old_keys, old_boundaries
        if old_states is None or old_states.shape[1] == 0:
            return new_summary_states, new_summary_keys, new_summary_boundaries
        return (
            torch.cat([old_states, new_summary_states], dim=1),
            torch.cat([old_keys, new_summary_keys], dim=1),
            torch.cat([old_boundaries.to(new_summary_boundaries.device), new_summary_boundaries], dim=0),
        )

    def _load_log_anchor_blocks(
        self,
        last_state: dict | None,
        *,
        old_bank_size: int,
    ) -> tuple[tuple[LogAnchorBlock, ...], int]:
        """Load compact cache metadata or derive a plan over a dense control bank."""

        old_bank_size = int(old_bank_size)
        if self.hsd_log_anchor_mode == "simulate":
            return plan_weak_fenwick_blocks(old_bank_size), old_bank_size
        if self.hsd_log_anchor_mode == "recent":
            return plan_recent_blocks(old_bank_size), old_bank_size
        if old_bank_size == 0:
            return (), 0
        if last_state is None:
            raise RuntimeError("Non-empty HSD Log-Anchor bank requires a layer cache state.")

        missing = [key for key in _HSD_LOG_ANCHOR_CACHE_KEYS if key not in last_state]
        if missing:
            raise RuntimeError(
                "Cached HSD Log-Anchor bank is missing metadata: " + ", ".join(missing)
            )

        def values(name: str) -> list[int]:
            raw = last_state[name]
            if isinstance(raw, torch.Tensor):
                return [int(value) for value in raw.detach().cpu().reshape(-1).tolist()]
            if isinstance(raw, (tuple, list)):
                return [int(value) for value in raw]
            raise RuntimeError(f"Cached HSD Log-Anchor metadata {name!r} must be a tensor or list.")

        starts = values("hsd_log_anchor_start_chunks")
        ends = values("hsd_log_anchor_end_chunks")
        counts = values("hsd_log_anchor_counts")
        levels = values("hsd_log_anchor_levels")
        if not all(len(items) == old_bank_size for items in (starts, ends, counts, levels)):
            raise RuntimeError(
                "Cached HSD Log-Anchor metadata length must match the compact summary bank: "
                f"bank={old_bank_size}, starts={len(starts)}, ends={len(ends)}, "
                f"counts={len(counts)}, levels={len(levels)}."
            )
        blocks = tuple(
            LogAnchorBlock(
                start=starts[index],
                end=ends[index],
                count=counts[index],
                level=levels[index],
                anchor_index=index,
            )
            for index in range(old_bank_size)
        )
        total_chunks = int(last_state["hsd_log_anchor_total_chunks"])
        if self.hsd_log_anchor_mode == "cache":
            represented = validate_weak_fenwick_blocks(blocks)
            if represented != total_chunks:
                raise RuntimeError(
                    "Cached weak-Fenwick metadata disagrees with total_chunks: "
                    f"represented={represented}, total={total_chunks}."
                )
        else:
            raise RuntimeError(f"Unexpected compact Log-Anchor mode {self.hsd_log_anchor_mode!r}.")
        return blocks, total_chunks

    def _build_log_anchor_plan(
        self,
        *,
        last_state: dict | None,
        old_bank_size: int,
        new_summary_boundaries: torch.Tensor,
        total_bank_size: int,
        text_len: int,
        start_index: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[LogAnchorBlock, ...], int]:
        """Return causal per-token counts and the final persistent cache layout."""

        existing_blocks, old_total_chunks = self._load_log_anchor_blocks(
            last_state,
            old_bank_size=old_bank_size,
        )
        new_event_count = int(new_summary_boundaries.numel())
        visible_prefixes = tuple(range(new_event_count + 1))
        if self.hsd_log_anchor_mode == "recent":
            folded = fold_recent_events(
                existing_blocks,
                old_total_chunks,
                new_event_count,
                visible_prefixes,
                existing_bank_size=old_bank_size,
            )
        else:
            folded = fold_weak_fenwick_events(
                existing_blocks,
                new_event_count,
                visible_prefixes,
                existing_bank_size=old_bank_size,
            )
        if folded.bank_size != int(total_bank_size):
            raise RuntimeError(
                "HSD Log-Anchor ephemeral bank size mismatch: "
                f"planned={folded.bank_size}, actual={int(total_bank_size)}."
            )

        token_index = torch.arange(
            int(start_index),
            int(start_index) + int(text_len),
            device=device,
            dtype=torch.long,
        )
        if new_event_count:
            new_boundaries = new_summary_boundaries.to(device=device, dtype=torch.long)
            visible_new = (
                new_boundaries.view(1, new_event_count)
                <= token_index.view(int(text_len), 1)
            ).sum(dim=-1)
        else:
            visible_new = torch.zeros(int(text_len), device=device, dtype=torch.long)
        prefix_count_table = torch.tensor(
            folded.active_count_matrix,
            device=device,
            dtype=torch.long,
        )
        active_counts = prefix_count_table.index_select(0, visible_new)
        return (
            active_counts,
            folded.final_blocks,
            int(old_total_chunks + new_event_count),
        )

    @staticmethod
    def _compact_log_anchor_bank(
        states: torch.Tensor,
        keys: torch.Tensor,
        boundaries: torch.Tensor,
        blocks: tuple[LogAnchorBlock, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[LogAnchorBlock, ...]]:
        indices = torch.tensor(
            [block.anchor_index for block in blocks],
            device=states.device,
            dtype=torch.long,
        )
        compact_states = states.index_select(1, indices)
        compact_keys = keys.index_select(1, indices.to(device=keys.device))
        compact_boundaries = boundaries.index_select(0, indices.to(device=boundaries.device))
        return (
            compact_states,
            compact_keys,
            compact_boundaries,
            reindex_compacted_blocks(blocks),
        )

    @staticmethod
    def _write_log_anchor_metadata(
        cache_state: dict,
        blocks: tuple[LogAnchorBlock, ...],
        *,
        total_chunks: int,
        device: torch.device,
    ) -> None:
        cache_state["hsd_log_anchor_counts"] = torch.tensor(
            [block.count for block in blocks], device=device, dtype=torch.long
        )
        cache_state["hsd_log_anchor_levels"] = torch.tensor(
            [block.level for block in blocks], device=device, dtype=torch.long
        )
        cache_state["hsd_log_anchor_start_chunks"] = torch.tensor(
            [block.start for block in blocks], device=device, dtype=torch.long
        )
        cache_state["hsd_log_anchor_end_chunks"] = torch.tensor(
            [block.end for block in blocks], device=device, dtype=torch.long
        )
        cache_state["hsd_log_anchor_total_chunks"] = int(total_chunks)

    @staticmethod
    def _route_keys_for_query_heads(summary_k: torch.Tensor, query_heads: int) -> torch.Tensor:
        key_heads = int(summary_k.shape[2])
        if key_heads == int(query_heads):
            return summary_k
        if key_heads == 1:
            return summary_k.expand(summary_k.shape[0], summary_k.shape[1], int(query_heads), summary_k.shape[3])
        if int(query_heads) % key_heads == 0:
            return summary_k.repeat_interleave(int(query_heads) // key_heads, dim=2)
        raise RuntimeError(
            "HSD summary route key heads must divide query heads: "
            f"got key_heads={key_heads}, query_heads={int(query_heads)}."
        )

    @staticmethod
    def _route_queries_for_value_heads(router_q: torch.Tensor, value_heads: int) -> torch.Tensor:
        query_heads = int(router_q.shape[2])
        if query_heads == int(value_heads):
            return router_q
        if query_heads == 1:
            return router_q.expand(router_q.shape[0], router_q.shape[1], int(value_heads), router_q.shape[3])
        raise RuntimeError(
            "HSD route query heads must be 1 or match value heads: "
            f"got query_heads={query_heads}, value_heads={int(value_heads)}."
        )

    @staticmethod
    def _summary_text_counts(summary_mask: torch.Tensor) -> torch.Tensor:
        text_seen = (~summary_mask[0]).long().cumsum(dim=0)
        return text_seen[summary_mask[0]]

    @staticmethod
    def _chunk_mean_hidden_stream(
        text_hidden: torch.Tensor,
        boundary_counts: torch.Tensor,
        *,
        summary_chunk_size: int,
        summary_token_num: int,
        boundary_start: int = 0,
        initial_hidden_sum: torch.Tensor | None = None,
        initial_token_count: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Pool the completed chunk before each summary and retain the tail.

        The accumulator is float32 so a cached decode can reproduce the same
        chunk mean as a single full forward. All rows share one summary layout,
        which is already an HSD V1 invariant.
        """
        if text_hidden.ndim != 3:
            raise ValueError("HSD route-key content pooling expects text_hidden=[B,T,D].")
        chunk_size = int(summary_chunk_size)
        summary_tokens = int(summary_token_num)
        if chunk_size <= 0 or summary_tokens <= 0:
            raise ValueError("HSD route-key content pooling requires positive chunk and summary sizes.")
        if boundary_counts.ndim != 1:
            raise ValueError("HSD route-key content pooling requires 1D boundary_counts.")
        if int(boundary_counts.numel()) % summary_tokens != 0:
            raise ValueError(
                "HSD route-key content pooling requires exactly summary_token_num summaries per boundary."
            )
        batch_size, text_len, hidden_size = text_hidden.shape
        count = int(initial_token_count)
        if count < 0 or count >= chunk_size:
            raise ValueError(
                f"HSD route-key content cached token count must be in [0, {chunk_size}); got {count}."
            )
        if initial_hidden_sum is None:
            running_sum = torch.zeros(
                batch_size,
                hidden_size,
                dtype=torch.float32,
                device=text_hidden.device,
            )
        else:
            if tuple(initial_hidden_sum.shape) != (batch_size, hidden_size):
                raise ValueError(
                    "HSD route-key content cached hidden sum shape must be "
                    f"{(batch_size, hidden_size)}; got {tuple(initial_hidden_sum.shape)}."
                )
            running_sum = initial_hidden_sum.to(device=text_hidden.device, dtype=torch.float32)

        group_count = int(boundary_counts.numel()) // summary_tokens
        if initial_hidden_sum is None and count == 0 and group_count > 0:
            expected_boundaries = (
                torch.arange(
                    1,
                    group_count + 1,
                    dtype=boundary_counts.dtype,
                    device=boundary_counts.device,
                )
                * chunk_size
                + int(boundary_start)
            ).repeat_interleave(summary_tokens)
            if torch.equal(boundary_counts, expected_boundaries):
                completed_text_len = group_count * chunk_size
                if completed_text_len > text_len:
                    raise ValueError(
                        "HSD route-key content boundary exceeds the current text segment: "
                        f"completed={completed_text_len}, text_len={text_len}."
                    )
                pooled_hidden = (
                    text_hidden[:, :completed_text_len]
                    .reshape(batch_size, group_count, chunk_size, hidden_size)
                    .float()
                    .mean(dim=2)
                    .repeat_interleave(summary_tokens, dim=1)
                )
                tail = text_hidden[:, completed_text_len:]
                tail_count = int(tail.shape[1])
                if tail_count >= chunk_size:
                    raise ValueError(
                        "HSD route-key content pooling reached a full tail chunk without a summary boundary; "
                        f"pending_count={tail_count}, chunk_size={chunk_size}."
                    )
                tail_sum = tail.float().sum(dim=1)
                return pooled_hidden, tail_sum, tail_count

        cursor = 0
        pooled_groups: list[torch.Tensor] = []
        for group_idx in range(group_count):
            group = boundary_counts[
                group_idx * summary_tokens:(group_idx + 1) * summary_tokens
            ]
            local_end = int(group[0].item()) - int(boundary_start)
            if bool((group != group[:1]).any().item()):
                raise ValueError("Repeated summary tokens at one boundary must share the same text count.")
            if local_end < cursor or local_end > text_len:
                raise ValueError(
                    "HSD route-key content boundary must be non-decreasing and inside the current text segment: "
                    f"cursor={cursor}, local_end={local_end}, text_len={text_len}."
                )
            if local_end > cursor:
                running_sum = running_sum + text_hidden[:, cursor:local_end].float().sum(dim=1)
                count += local_end - cursor
            if count != chunk_size:
                raise ValueError(
                    "Each HSD summary boundary must close exactly one full content chunk: "
                    f"expected={chunk_size}, got={count}, boundary={int(group[0].item())}."
                )
            pooled = running_sum / float(chunk_size)
            pooled_groups.append(
                pooled[:, None, :].expand(batch_size, summary_tokens, hidden_size)
            )
            running_sum = torch.zeros_like(running_sum)
            count = 0
            cursor = local_end

        if cursor < text_len:
            running_sum = running_sum + text_hidden[:, cursor:].float().sum(dim=1)
            count += text_len - cursor
        if count >= chunk_size:
            raise ValueError(
                "HSD route-key content pooling reached a full chunk without a summary boundary; "
                f"pending_count={count}, chunk_size={chunk_size}."
            )
        if pooled_groups:
            pooled_hidden = torch.cat(pooled_groups, dim=1)
        else:
            pooled_hidden = text_hidden.new_zeros(
                batch_size, 0, hidden_size, dtype=torch.float32
            )
        return pooled_hidden, running_sum, count

    @classmethod
    def _chunk_mean_hidden_packed(
        cls,
        text_hidden: torch.Tensor,
        boundary_counts: torch.Tensor,
        *,
        text_cu_seqlens: torch.Tensor,
        summary_cu_seqlens: torch.Tensor,
        summary_chunk_size: int,
        summary_token_num: int,
    ) -> torch.Tensor:
        if text_hidden.shape[0] != 1:
            raise ValueError("Packed HSD route-key content pooling requires batch_size=1.")
        text_cu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        summary_cu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        pooled_docs: list[torch.Tensor] = []
        for doc_idx in range(int(text_cu.numel()) - 1):
            text_start = int(text_cu[doc_idx].item())
            text_end = int(text_cu[doc_idx + 1].item())
            summary_start = int(summary_cu[doc_idx].item())
            summary_end = int(summary_cu[doc_idx + 1].item())
            pooled, _, _ = cls._chunk_mean_hidden_stream(
                text_hidden[:, text_start:text_end],
                boundary_counts[summary_start:summary_end],
                summary_chunk_size=summary_chunk_size,
                summary_token_num=summary_token_num,
                boundary_start=text_start,
            )
            pooled_docs.append(pooled)
        if pooled_docs:
            return torch.cat(pooled_docs, dim=1)
        return text_hidden.new_zeros(1, 0, text_hidden.shape[-1], dtype=torch.float32)

    @staticmethod
    def _validate_packed_cu_seqlens(
        cu_seqlens: torch.Tensor,
        *,
        expected_end: int,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1:
            raise ValueError(f"HSD packed sequence requires {name} to be a 1D tensor.")
        if cu_seqlens.numel() < 2:
            raise ValueError(f"HSD packed sequence requires {name} to contain at least 2 entries.")
        cu_cpu = cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        if int(cu_cpu[0].item()) != 0 or int(cu_cpu[-1].item()) != int(expected_end):
            raise ValueError(
                f"HSD packed {name} must span expected length {expected_end}: "
                f"got first={int(cu_cpu[0].item())}, last={int(cu_cpu[-1].item())}."
            )
        if bool((cu_cpu[1:] < cu_cpu[:-1]).any().item()):
            raise ValueError(f"HSD packed {name} must be non-decreasing.")
        return cu_seqlens

    @staticmethod
    def _packed_cu_from_expanded_cu(
        summary_mask: torch.Tensor,
        expanded_cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if summary_mask.shape[0] != 1:
            raise NotImplementedError("HSD packed path requires flattened batch_size=1 inputs.")
        if not isinstance(expanded_cu_seqlens, torch.Tensor) or expanded_cu_seqlens.ndim != 1:
            raise ValueError("HSD packed cu_seqlens must be a 1D tensor.")
        cu_cpu = expanded_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        expanded_len = int(summary_mask.shape[1])
        if int(cu_cpu[0].item()) != 0 or int(cu_cpu[-1].item()) != expanded_len:
            raise ValueError(
                f"HSD packed expanded cu_seqlens must span expanded length {expanded_len}: "
                f"got first={int(cu_cpu[0].item())}, last={int(cu_cpu[-1].item())}."
            )
        if bool((cu_cpu[1:] < cu_cpu[:-1]).any().item()):
            raise ValueError("HSD packed expanded cu_seqlens must be non-decreasing.")
        mask_cpu = summary_mask[0].detach().to(device="cpu", dtype=torch.bool)
        text_counts = [0]
        summary_counts = [0]
        for start, end in zip(cu_cpu[:-1].tolist(), cu_cpu[1:].tolist()):
            doc_mask = mask_cpu[int(start):int(end)]
            summary_count = int(doc_mask.sum().item())
            text_counts.append(text_counts[-1] + int(doc_mask.numel()) - summary_count)
            summary_counts.append(summary_counts[-1] + summary_count)
        device = expanded_cu_seqlens.device
        return (
            torch.tensor(text_counts, dtype=torch.long, device=device),
            torch.tensor(summary_counts, dtype=torch.long, device=device),
        )

    @staticmethod
    def _packed_local_boundary_counts(
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        summary_cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if boundary_counts.numel() == 0:
            return boundary_counts
        text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        summary_cu_cpu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        local_boundaries = []
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            summary_start = int(summary_cu_cpu[doc_idx].item())
            summary_end = int(summary_cu_cpu[doc_idx + 1].item())
            if summary_end > summary_start:
                local_boundaries.append(boundary_counts[summary_start:summary_end] - text_start)
        if not local_boundaries:
            return boundary_counts.new_zeros((0,), dtype=torch.long)
        return torch.cat(local_boundaries, dim=0)

    def _expand_query_to_value_heads(self, q: torch.Tensor) -> torch.Tensor:
        if self.num_v_heads == self.num_heads:
            return q
        repeat = self.num_v_heads // self.num_heads
        return q.repeat_interleave(repeat, dim=2)

    def _null_gate_logits(self, text_hidden: torch.Tensor) -> torch.Tensor | None:
        if self.null_gate_proj is None:
            return None
        if self.hsd_null_read_gate_bias is None:
            raise RuntimeError("hsd_null_read_gate_bias is required when HSD null read gate is enabled.")
        logits = self.null_gate_proj(text_hidden)
        if logits.shape[-1] != self.hsd_null_read_gate_bias.numel():
            raise RuntimeError(
                "HSD null read gate output size must match bias size: "
                f"logits={logits.shape[-1]}, bias={self.hsd_null_read_gate_bias.numel()}."
            )
        bias = self.hsd_null_read_gate_bias.float().to(device=logits.device).view(1, 1, -1)
        return (logits.float() + bias).unsqueeze(-1)

    @staticmethod
    def _history_available_mask(
        *,
        boundary_counts: torch.Tensor,
        text_len: int,
        batch_size: int,
        start_index: int = 0,
        text_cu_seqlens: torch.Tensor | None = None,
        summary_cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [B, T, 1, 1] indicating whether each token can read any summary."""
        device = boundary_counts.device
        if text_cu_seqlens is None and summary_cu_seqlens is None:
            token_index = torch.arange(start_index, start_index + text_len, device=device, dtype=torch.long)
            boundaries = boundary_counts.to(device=device, dtype=torch.long)
            available = (boundaries.view(1, -1) <= token_index.view(-1, 1)).any(dim=-1)
            return available.view(1, text_len, 1, 1).expand(batch_size, -1, -1, -1)

        if text_cu_seqlens is None or summary_cu_seqlens is None:
            raise ValueError("Packed HSD history availability requires both text and summary cu_seqlens.")
        if batch_size != 1:
            raise NotImplementedError("Packed HSD history availability requires batch_size=1.")
        text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        summary_cu_cpu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        available = torch.zeros(text_len, device=device, dtype=torch.bool)
        boundaries = boundary_counts.to(device=device, dtype=torch.long)
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            summary_start = int(summary_cu_cpu[doc_idx].item())
            summary_end = int(summary_cu_cpu[doc_idx + 1].item())
            if text_end <= text_start or summary_end <= summary_start:
                continue
            token_index = torch.arange(text_start, text_end, device=device, dtype=torch.long)
            doc_boundaries = boundaries[summary_start:summary_end]
            available[text_start:text_end] = (
                doc_boundaries.view(1, -1) <= token_index.view(-1, 1)
            ).any(dim=-1)
        return available.view(1, text_len, 1, 1)

    @staticmethod
    def _history_candidate_counts(
        *,
        boundary_counts: torch.Tensor,
        text_len: int,
        batch_size: int,
        start_index: int = 0,
        topk: int = 0,
        text_cu_seqlens: torch.Tensor | None = None,
        summary_cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the number of causal history candidates as [B, T, 1, 1]."""
        device = boundary_counts.device
        boundaries = boundary_counts.to(device=device, dtype=torch.long).contiguous()
        if text_cu_seqlens is None and summary_cu_seqlens is None:
            token_index = torch.arange(start_index, start_index + text_len, device=device, dtype=torch.long)
            counts = torch.searchsorted(boundaries, token_index, right=True)
            counts = counts.view(1, text_len, 1, 1).expand(batch_size, -1, -1, -1)
        else:
            if text_cu_seqlens is None or summary_cu_seqlens is None:
                raise ValueError("Packed HSD candidate counts require both text and summary cu_seqlens.")
            if batch_size != 1:
                raise NotImplementedError("Packed HSD candidate counts require batch_size=1.")
            text_cu = text_cu_seqlens.to(device=device, dtype=torch.long)
            summary_cu = summary_cu_seqlens.to(device=device, dtype=torch.long)
            token_index = torch.arange(text_len, device=device, dtype=torch.long)
            token_doc = torch.searchsorted(text_cu[1:], token_index, right=True)
            cumulative = torch.searchsorted(boundaries, token_index, right=True)
            doc_summary_start = summary_cu.index_select(0, token_doc)
            doc_summary_end = summary_cu.index_select(0, token_doc + 1)
            counts = (cumulative - doc_summary_start).clamp_min(0)
            counts = torch.minimum(counts, doc_summary_end - doc_summary_start)
            counts = counts.view(1, text_len, 1, 1)
        if topk > 0:
            counts = counts.clamp_max(int(topk))
        return counts

    @staticmethod
    def _joint_softmax_logm_logits(
        local_logits: torch.Tensor,
        candidate_counts: torch.Tensor,
    ) -> torch.Tensor:
        return HSDGatedDeltaNet._null_logm_corrected_logits(
            local_logits,
            candidate_counts,
            scale=1.0,
        )

    @staticmethod
    def _null_logm_corrected_logits(
        local_logits: torch.Tensor,
        candidate_counts: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        if candidate_counts.shape not in {
            (1, local_logits.shape[1], 1, 1),
            (local_logits.shape[0], local_logits.shape[1], 1, 1),
        }:
            raise RuntimeError(
                "HSD joint-softmax candidate counts must be [1|B, T, 1, 1], got "
                f"{tuple(candidate_counts.shape)}."
            )
        scale = float(scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("HSD null log-M correction scale must be non-negative and finite.")
        log_m = candidate_counts.to(device=local_logits.device, dtype=torch.float32).clamp_min(1).log()
        return local_logits.float() + scale * log_m

    def _record_hsd_null_logm_debug_stats(
        self,
        *,
        candidate_counts: torch.Tensor,
        local_prob: torch.Tensor,
        raw_null_logits: torch.Tensor,
        corrected_null_logits: torch.Tensor,
    ) -> None:
        """Record actual null probability by causal candidate-count bucket."""
        with torch.no_grad():
            counts = candidate_counts.detach().to(device=local_prob.device, dtype=torch.long)
            probs = local_prob.detach().float().squeeze(-1)
            if counts.shape[0] == 1 and probs.shape[0] != 1:
                counts = counts.expand(probs.shape[0], -1, -1, -1)
            token_counts = counts[..., 0, 0]
            active_tokens = token_counts > 0
            head_counts = token_counts.unsqueeze(-1).expand_as(probs)
            active_heads = head_counts > 0
            raw = raw_null_logits.detach().float().squeeze(-1)
            corrected = corrected_null_logits.detach().float().squeeze(-1)
            correction = corrected - raw
            stats = {
                "hsd/null_logm_scale": probs.new_tensor(self.hsd_null_logm_scale),
                "hsd/router_candidate_share_m0": (token_counts == 0).float().mean(),
            }
            zero_heads = head_counts == 0
            if bool(zero_heads.any().item()):
                stats["hsd/router_null_prob_m0"] = probs[zero_heads].mean()
            if bool(active_tokens.any().item()):
                stats.update({
                    "hsd/router_valid_candidate_count": token_counts[active_tokens].float().mean(),
                    "hsd/router_null_logm_correction": correction[active_heads].mean(),
                    "hsd/router_null_prob": probs[active_heads].mean(),
                })
            buckets = (
                (1, 1, "m01"),
                (2, 3, "m02_03"),
                (4, 7, "m04_07"),
                (8, 15, "m08_15"),
                (16, 31, "m16_31"),
                (32, 63, "m32_63"),
                (64, 127, "m64_127"),
                (128, None, "m128_plus"),
            )
            token_count = token_counts.new_tensor(float(token_counts.numel()), dtype=torch.float32)
            for lower, upper, suffix in buckets:
                token_mask = token_counts >= lower
                if upper is not None:
                    token_mask = token_mask & (token_counts <= upper)
                if not bool(token_mask.any().item()):
                    continue
                head_mask = token_mask.unsqueeze(-1).expand_as(probs)
                stats[f"hsd/router_null_prob_{suffix}"] = probs[head_mask].mean()
                stats[f"hsd/router_candidate_share_{suffix}"] = (
                    token_mask.float().sum() / token_count
                )
            self._hsd_debug_stats.update(stats)

    @staticmethod
    def _mix_local_and_history(
        *,
        local_o: torch.Tensor,
        history_o: torch.Tensor,
        local_logits: torch.Tensor,
        history_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_o.shape != history_o.shape:
            raise RuntimeError(
                f"HSD convex mix requires matching local/history shapes: {local_o.shape} vs {history_o.shape}."
            )
        expected_logits = (*local_o.shape[:3], 1)
        if tuple(local_logits.shape) != expected_logits:
            raise RuntimeError(
                f"HSD convex local logits shape {tuple(local_logits.shape)} must be {expected_logits}."
            )
        if history_available.shape not in {
            (1, local_o.shape[1], 1, 1),
            (local_o.shape[0], local_o.shape[1], 1, 1),
        }:
            raise RuntimeError(
                "HSD convex history availability must be [1|B, T, 1, 1], got "
                f"{tuple(history_available.shape)}."
            )
        local_alpha = torch.sigmoid(local_logits.float()).to(local_o.dtype)
        local_alpha = torch.where(
            history_available.to(device=local_o.device, dtype=torch.bool),
            local_alpha,
            torch.ones_like(local_alpha),
        )
        mixed = local_alpha * local_o + (1.0 - local_alpha) * history_o.to(local_o.dtype)
        return mixed, local_alpha

    @staticmethod
    def _mix_local_with_gated_history(
        *,
        local_o: torch.Tensor,
        history_o: torch.Tensor,
        local_logits: torch.Tensor,
        history_available: torch.Tensor,
        history_gate_floor: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_o.shape != history_o.shape:
            raise RuntimeError(
                "HSD gated-additive mix requires matching local/history shapes: "
                f"{local_o.shape} vs {history_o.shape}."
            )
        expected_logits = (*local_o.shape[:3], 1)
        if tuple(local_logits.shape) != expected_logits:
            raise RuntimeError(
                f"HSD gated-additive local logits shape {tuple(local_logits.shape)} must be {expected_logits}."
            )
        if history_available.shape not in {
            (1, local_o.shape[1], 1, 1),
            (local_o.shape[0], local_o.shape[1], 1, 1),
        }:
            raise RuntimeError(
                "HSD gated-additive history availability must be [1|B, T, 1, 1], got "
                f"{tuple(history_available.shape)}."
            )
        if not math.isfinite(history_gate_floor) or not 0.0 <= history_gate_floor < 1.0:
            raise ValueError("history_gate_floor must be finite and in [0, 1).")
        raw_local_alpha = torch.sigmoid(local_logits.float()).to(local_o.dtype)
        local_alpha = torch.where(
            history_available.to(device=local_o.device, dtype=torch.bool),
            raw_local_alpha,
            torch.ones_like(raw_local_alpha),
        )
        history_alpha = HSDGatedDeltaNet._history_alpha_with_floor(
            local_alpha=local_alpha,
            history_available=history_available,
            history_gate_floor=history_gate_floor,
        )
        mixed = local_o + history_alpha * history_o.to(local_o.dtype)
        return mixed, local_alpha

    @staticmethod
    def _history_alpha_with_floor(
        *,
        local_alpha: torch.Tensor,
        history_available: torch.Tensor,
        history_gate_floor: float,
    ) -> torch.Tensor:
        floor = local_alpha.new_tensor(float(history_gate_floor))
        history_alpha = floor + (1.0 - floor) * (1.0 - local_alpha)
        return torch.where(
            history_available.to(device=local_alpha.device, dtype=torch.bool),
            history_alpha,
            torch.zeros_like(history_alpha),
        )

    @staticmethod
    def _scale_history_by_head(
        history_o: torch.Tensor,
        head_scale: torch.Tensor,
    ) -> torch.Tensor:
        if history_o.ndim != 4:
            raise RuntimeError(
                f"HSD per-head history scale requires history [B, T, H, D], got {tuple(history_o.shape)}."
            )
        if head_scale.ndim != 1 or head_scale.shape[0] != history_o.shape[-2]:
            raise RuntimeError(
                "HSD per-head history scale must have shape [H] matching the history heads: "
                f"{tuple(head_scale.shape)} vs H={history_o.shape[-2]}."
            )
        scale = head_scale.to(device=history_o.device, dtype=history_o.dtype)
        return history_o * scale.view(1, 1, -1, 1)

    def _record_hsd_history_mix_debug_stats(
        self,
        *,
        local_o: torch.Tensor,
        conditional_history: torch.Tensor,
        mixed_o: torch.Tensor,
        local_alpha: torch.Tensor,
        effective_history_alpha: torch.Tensor | None = None,
        history_available: torch.Tensor,
        record_debug_stats: bool,
        history_is_joint_weighted: bool = False,
        mix_mode: str = "convex",
        history_gate_floor: float = 0.0,
    ) -> None:
        if not record_debug_stats:
            return
        with torch.no_grad():
            local = local_o.detach().float()
            history = conditional_history.detach().float()
            mixed = mixed_o.detach().float()
            alpha = local_alpha.detach().float()
            raw_history_alpha = 1.0 - alpha
            if effective_history_alpha is None:
                used_history_alpha = raw_history_alpha
            else:
                used_history_alpha = effective_history_alpha.detach().float()
            actual_history = history if history_is_joint_weighted else used_history_alpha * history
            eps = torch.finfo(torch.float32).eps
            flat_alpha = alpha.flatten()
            self._hsd_debug_stats.update({
                'hsd/history_mix_mode_convex': local.new_tensor(1.0 if mix_mode == "convex" else 0.0),
                'hsd/history_mix_mode_joint_softmax_logm': local.new_tensor(
                    1.0 if mix_mode == "joint_softmax_logm" else 0.0
                ),
                'hsd/history_mix_mode_gated_additive': local.new_tensor(
                    1.0 if mix_mode == "gated_additive" else 0.0
                ),
                'hsd/local_alpha_mean': flat_alpha.mean(),
                'hsd/local_alpha_p10': torch.quantile(flat_alpha, 0.10),
                'hsd/local_alpha_p50': torch.quantile(flat_alpha, 0.50),
                'hsd/local_alpha_p90': torch.quantile(flat_alpha, 0.90),
                'hsd/local_alpha_max': flat_alpha.max(),
                'hsd/history_alpha_mean': used_history_alpha.flatten().mean(),
                'hsd/history_alpha_raw_mean': raw_history_alpha.flatten().mean(),
                'hsd/history_alpha_effective_mean': used_history_alpha.flatten().mean(),
                'hsd/history_gate_floor': local.new_tensor(float(history_gate_floor)),
                'hsd/history_available_share': history_available.detach().float().mean(),
                'hsd/conditional_history_text_norm_ratio': history.norm() / local.norm().clamp_min(eps),
                'hsd/conditional_history_local_norm_ratio': history.norm() / local.norm().clamp_min(eps),
                'hsd/history_text_norm_ratio': actual_history.norm() / local.norm().clamp_min(eps),
                'hsd/gated_history_local_norm_ratio': actual_history.norm() / local.norm().clamp_min(eps),
                'hsd/mixed_text_norm_ratio': mixed.norm() / local.norm().clamp_min(eps),
                'hsd/mixed_local_norm_ratio': mixed.norm() / local.norm().clamp_min(eps),
                # In convex mode the old null probability is exactly the local mixture weight.
                'hsd/router_null_prob': flat_alpha.mean(),
                'hsd/router_summary_mass': raw_history_alpha.flatten().mean(),
                'hsd/router_local_prob': flat_alpha.mean(),
                'hsd/history_group_mass': used_history_alpha.flatten().mean(),
            })
            history_head_log_scale = getattr(self, "history_head_log_scale", None)
            if history_head_log_scale is not None:
                layer_idx = int(getattr(self, "layer_idx", -1))
                head_scale = history_head_log_scale.detach().float().exp()
                local_per_head = local.movedim(-2, 0).reshape(self.num_v_heads, -1)
                history_per_head = history.movedim(-2, 0).reshape(self.num_v_heads, -1)
                actual_history_per_head = actual_history.movedim(-2, 0).reshape(self.num_v_heads, -1)
                local_head_norm = local_per_head.norm(dim=-1).clamp_min(eps)
                conditional_ratio = history_per_head.norm(dim=-1) / local_head_norm
                gated_ratio = actual_history_per_head.norm(dim=-1) / local_head_norm
                for head_idx in range(self.num_v_heads):
                    suffix = f"l{layer_idx:02d}_h{head_idx}"
                    self._hsd_debug_stats.update({
                        f"hsd/history_head_scale_{suffix}": head_scale[head_idx],
                        f"hsd/conditional_history_local_norm_ratio_{suffix}": conditional_ratio[head_idx],
                        f"hsd/gated_history_local_norm_ratio_{suffix}": gated_ratio[head_idx],
                    })

    def _record_hsd_additive_head_scale_debug_stats(
        self,
        *,
        local_o: torch.Tensor,
        history_o: torch.Tensor,
        record_debug_stats: bool,
    ) -> None:
        if not record_debug_stats or self.history_head_log_scale is None:
            return
        with torch.no_grad():
            local = local_o.detach().float()
            history = history_o.detach().float()
            eps = torch.finfo(torch.float32).eps
            layer_idx = int(getattr(self, "layer_idx", -1))
            head_scale = self.history_head_log_scale.detach().float().exp()
            local_per_head = local.movedim(-2, 0).reshape(self.num_v_heads, -1)
            history_per_head = history.movedim(-2, 0).reshape(self.num_v_heads, -1)
            ratio = history_per_head.norm(dim=-1) / local_per_head.norm(dim=-1).clamp_min(eps)
            for head_idx in range(self.num_v_heads):
                suffix = f"l{layer_idx:02d}_h{head_idx}"
                self._hsd_debug_stats.update({
                    f"hsd/history_head_scale_{suffix}": head_scale[head_idx],
                    f"hsd/conditional_history_local_norm_ratio_{suffix}": ratio[head_idx],
                    f"hsd/scaled_history_local_norm_ratio_{suffix}": ratio[head_idx],
                })

    def _g_decay_from_raw(self, g: torch.Tensor) -> torch.Tensor:
        if getattr(self, "hsd_local_core", "gdn") == "gdn2":
            if self.gdn2_dt_bias is None:
                raise RuntimeError("HSD-GDN2 decay bias is missing.")
            a = self.A_log.float().exp().to(device=g.device).view(1, 1, -1, 1)
            dt_bias = self.gdn2_dt_bias.float().to(device=g.device).view(
                1, 1, self.num_v_heads, self.head_k_dim
            )
            return -a * F.softplus(g.float() + dt_bias)
        if getattr(self, "hsd_local_core", "gdn") == "kda":
            if self.kda_dt_bias is None:
                raise RuntimeError("HSD-KDA decay bias is missing.")
            a = self.A_log.float().exp().to(device=g.device).view(1, 1, -1, 1)
            dt_bias = self.kda_dt_bias.float().to(device=g.device).view(
                1, 1, self.num_v_heads, self.head_k_dim
            )
            decay = -a * F.softplus(g.float() + dt_bias)
            # The legacy scalar bias remains checkpoint-compatible and receives
            # an explicit zero gradient in the KDA treatment.
            return decay + self.dt_bias.float().sum() * 0.0
        a = self.A_log.float().exp().to(device=g.device).view(1, 1, -1)
        dt_bias = self.dt_bias.float().to(device=g.device).view(1, 1, -1)
        return -a * F.softplus(g.float() + dt_bias)

    def _effective_g_decay(self, g: torch.Tensor) -> torch.Tensor:
        decay = self._g_decay_from_raw(g)
        if self.hsd_state_decay_mode == "none":
            # Keep a_proj/A_log/dt_bias connected to DDP with exact zero gradients.
            return decay * 0.0
        if (
            getattr(self, "hsd_local_core", "gdn") == "kda"
            and not self.training
            and self.hsd_kda_inference_decay_mode == "mean_broadcast"
        ):
            # Inference-only diagnostic: preserve each token/head's learned
            # mean log-decay while removing per-key-dimension anisotropy.
            # expand_as is a view, so this does not materialize another gate.
            decay = decay.mean(dim=-1, keepdim=True).expand_as(decay)
        return decay

    def _zero_g_decay(self, g: torch.Tensor) -> torch.Tensor:
        # The chunk-memory branch intentionally has no decay, but retaining this
        # zero-valued dependency makes its gradient contribution explicit.
        return self._g_decay_from_raw(g) * 0.0

    def _continuous_states_to_delta_candidates(
        self,
        *,
        continuous_states: torch.Tensor,
        effective_g: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int = 0,
        text_cu_seqlens: torch.Tensor | None = None,
        summary_cu_seqlens: torch.Tensor | None = None,
        previous_boundary_state: torch.Tensor | None = None,
        pending_log_decay: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Convert cumulative boundary states into exact per-chunk net changes.

        For every value head independently,
        ``delta_m = H_m - exp(sum(g_m)) * H_{m-1}``.
        Repeated summary tokens at one boundary share the same delta candidate.
        The two returned cache tensors are only meaningful for non-packed
        prefill/decode; packed documents reset independently and do not cache.
        """
        if continuous_states.ndim != 5:
            raise ValueError(
                "HSD continuous summary states must have shape [B,M,H,K,V], "
                f"got {tuple(continuous_states.shape)}."
            )
        if effective_g.ndim != 3:
            raise ValueError(
                "HSD effective decay must have shape [B,T,H], "
                f"got {tuple(effective_g.shape)}."
            )
        batch_size, summary_count, value_heads = continuous_states.shape[:3]
        if tuple(effective_g.shape[:1]) != (batch_size,) or int(effective_g.shape[2]) != value_heads:
            raise ValueError("HSD delta-state decay batch/head dimensions must match summary states.")
        if int(boundary_counts.numel()) != int(summary_count):
            raise ValueError(
                "HSD delta-state boundary count must match summary states: "
                f"boundaries={int(boundary_counts.numel())}, states={int(summary_count)}."
            )

        state_dtype = continuous_states.dtype
        state_device = continuous_states.device
        g_fp32 = effective_g.float()
        zero_state = torch.zeros(
            batch_size,
            value_heads,
            continuous_states.shape[-2],
            continuous_states.shape[-1],
            dtype=state_dtype,
            device=state_device,
        )
        packed = text_cu_seqlens is not None or summary_cu_seqlens is not None
        delta_by_summary: list[torch.Tensor | None] = [None] * int(summary_count)
        gamma_by_summary: list[torch.Tensor | None] = [None] * int(summary_count)

        def append_boundary_group(
            *,
            summary_start: int,
            summary_end: int,
            previous_state: torch.Tensor,
            previous_log_decay: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            current_state = continuous_states[:, summary_start]
            gamma = previous_log_decay.exp()
            delta_fp32 = (
                current_state.float()
                - gamma[..., None, None] * previous_state.float()
            )
            delta = delta_fp32.to(state_dtype)
            for summary_idx in range(summary_start, summary_end):
                delta_by_summary[summary_idx] = delta
                gamma_by_summary[summary_idx] = gamma
            return current_state, g_fp32.new_zeros(batch_size, value_heads)

        if packed:
            if text_cu_seqlens is None or summary_cu_seqlens is None:
                raise ValueError("Packed HSD delta-state conversion requires both text and summary cu_seqlens.")
            if int(boundary_start) != 0:
                raise ValueError("Packed HSD delta-state conversion requires boundary_start=0.")
            if previous_boundary_state is not None or pending_log_decay is not None:
                raise ValueError("Packed HSD delta-state conversion cannot inherit cache metadata.")
            if batch_size != 1:
                raise ValueError("Packed HSD delta-state conversion requires flattened batch_size=1.")
            text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
            summary_cu_cpu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
            boundaries_cpu = boundary_counts.detach().to(device="cpu", dtype=torch.long)
            if text_cu_cpu.numel() != summary_cu_cpu.numel():
                raise ValueError("Packed HSD delta-state text/summary cu_seqlens must describe the same documents.")
            for doc_idx in range(int(text_cu_cpu.numel()) - 1):
                text_start = int(text_cu_cpu[doc_idx].item())
                text_end = int(text_cu_cpu[doc_idx + 1].item())
                summary_idx = int(summary_cu_cpu[doc_idx].item())
                summary_end = int(summary_cu_cpu[doc_idx + 1].item())
                cursor = text_start
                previous_state = zero_state
                while summary_idx < summary_end:
                    boundary = int(boundaries_cpu[summary_idx].item())
                    if boundary < cursor or boundary > text_end:
                        raise ValueError(
                            "Packed HSD delta-state boundaries must be ordered inside each document: "
                            f"cursor={cursor}, boundary={boundary}, document_end={text_end}."
                        )
                    group_end = summary_idx + 1
                    while (
                        group_end < summary_end
                        and int(boundaries_cpu[group_end].item()) == boundary
                    ):
                        group_end += 1
                    segment_log_decay = g_fp32[:, cursor:boundary].sum(dim=1)
                    previous_state, _ = append_boundary_group(
                        summary_start=summary_idx,
                        summary_end=group_end,
                        previous_state=previous_state,
                        previous_log_decay=segment_log_decay,
                    )
                    cursor = boundary
                    summary_idx = group_end
            final_previous_state = None
            final_pending_log_decay = None
        else:
            boundaries_cpu = boundary_counts.detach().to(device="cpu", dtype=torch.long)
            if previous_boundary_state is not None:
                if tuple(previous_boundary_state.shape) != tuple(zero_state.shape):
                    raise ValueError(
                        "Cached HSD delta previous-boundary state has incompatible shape: "
                        f"expected={tuple(zero_state.shape)}, got={tuple(previous_boundary_state.shape)}."
                    )
                previous_state = previous_boundary_state
            else:
                previous_state = zero_state
            had_previous_boundary = previous_boundary_state is not None
            if pending_log_decay is None:
                pending = g_fp32.new_zeros(batch_size, value_heads)
            else:
                if tuple(pending_log_decay.shape) != (batch_size, value_heads):
                    raise ValueError(
                        "Cached HSD delta pending decay has incompatible shape: "
                        f"expected={(batch_size, value_heads)}, got={tuple(pending_log_decay.shape)}."
                    )
                pending = pending_log_decay.float()

            cursor = 0
            summary_idx = 0
            text_len = int(effective_g.shape[1])
            while summary_idx < summary_count:
                boundary = int(boundaries_cpu[summary_idx].item())
                local_boundary = boundary - int(boundary_start)
                if local_boundary < cursor or local_boundary > text_len:
                    raise ValueError(
                        "HSD delta-state boundaries must be ordered inside the current text segment: "
                        f"cursor={cursor}, local_boundary={local_boundary}, text_len={text_len}."
                    )
                group_end = summary_idx + 1
                while (
                    group_end < summary_count
                    and int(boundaries_cpu[group_end].item()) == boundary
                ):
                    group_end += 1
                segment_log_decay = pending + g_fp32[:, cursor:local_boundary].sum(dim=1)
                previous_state, pending = append_boundary_group(
                    summary_start=summary_idx,
                    summary_end=group_end,
                    previous_state=previous_state,
                    previous_log_decay=segment_log_decay,
                )
                had_previous_boundary = True
                cursor = local_boundary
                summary_idx = group_end
            pending = pending + g_fp32[:, cursor:].sum(dim=1)
            final_previous_state = previous_state if had_previous_boundary else None
            final_pending_log_decay = pending

        if summary_count:
            if any(value is None for value in delta_by_summary) or any(
                value is None for value in gamma_by_summary
            ):
                raise RuntimeError("HSD delta-state conversion did not cover every summary state.")
            delta_states = torch.stack(
                [value for value in delta_by_summary if value is not None],
                dim=1,
            )
            gammas = torch.stack(
                [value for value in gamma_by_summary if value is not None],
                dim=1,
            )
        else:
            delta_states = continuous_states
            gammas = g_fp32.new_zeros(batch_size, 0, value_heads)
        return delta_states, gammas, final_previous_state, final_pending_log_decay

    def _record_delta_candidate_debug_stats(
        self,
        *,
        continuous_states: torch.Tensor,
        delta_states: torch.Tensor,
        gammas: torch.Tensor,
        record_debug_stats: bool,
    ) -> None:
        if not record_debug_stats:
            return
        with torch.no_grad():
            self._hsd_debug_stats["hsd/history_candidate_delta"] = 1.0
            if int(delta_states.shape[1]) == 0:
                return
            eps = torch.finfo(torch.float32).eps
            snapshot_norm = continuous_states.detach().float().flatten(start_dim=-2).norm(dim=-1)
            delta_norm = delta_states.detach().float().flatten(start_dim=-2).norm(dim=-1)
            gamma = gammas.detach().float().flatten()
            self._hsd_debug_stats.update({
                "hsd/delta_snapshot_norm_ratio": (
                    delta_norm / snapshot_norm.clamp_min(eps)
                ).mean(),
                "hsd/delta_state_norm_mean": delta_norm.mean(),
                "hsd/snapshot_state_norm_mean": snapshot_norm.mean(),
                "hsd/chunk_gamma_mean": gamma.mean(),
                "hsd/chunk_gamma_p10": torch.quantile(gamma, 0.1),
                "hsd/chunk_gamma_p50": torch.quantile(gamma, 0.5),
                "hsd/chunk_gamma_p90": torch.quantile(gamma, 0.9),
            })
            if int(delta_states.shape[1]) > 1:
                left = delta_states[:, :-1].detach().float().flatten(start_dim=-2)
                right = delta_states[:, 1:].detach().float().flatten(start_dim=-2)
                self._hsd_debug_stats["hsd/delta_adjacent_cosine"] = F.cosine_similarity(
                    left,
                    right,
                    dim=-1,
                ).mean()

    @staticmethod
    def _chunk_reset_packed_cu_seqlens(
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        summary_cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Treat each interval ending at a summary boundary as a packed sequence."""
        text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        summary_cu_cpu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        boundaries_cpu = boundary_counts.detach().to(device="cpu", dtype=torch.long)
        if text_cu_cpu.numel() != summary_cu_cpu.numel():
            raise ValueError("HSD packed text and summary cu_seqlens must describe the same documents.")

        chunk_text_cu = [0]
        chunk_summary_cu = [0]
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            summary_start = int(summary_cu_cpu[doc_idx].item())
            summary_end = int(summary_cu_cpu[doc_idx + 1].item())
            if text_end <= text_start:
                raise ValueError("HSD chunk-reset packed path requires every document to contain text tokens.")

            cursor = text_start
            summary_idx = summary_start
            while summary_idx < summary_end:
                boundary = int(boundaries_cpu[summary_idx].item())
                if boundary <= cursor or boundary > text_end:
                    raise ValueError(
                        "HSD chunk-reset summaries must end non-empty, ordered text chunks: "
                        f"cursor={cursor}, boundary={boundary}, document_end={text_end}."
                    )
                group_end = summary_idx + 1
                while (
                    group_end < summary_end
                    and int(boundaries_cpu[group_end].item()) == boundary
                ):
                    group_end += 1
                chunk_text_cu.append(boundary)
                chunk_summary_cu.append(chunk_summary_cu[-1] + group_end - summary_idx)
                cursor = boundary
                summary_idx = group_end

            if cursor < text_end:
                chunk_text_cu.append(text_end)
                chunk_summary_cu.append(chunk_summary_cu[-1])

        device = text_cu_seqlens.device
        return (
            torch.tensor(chunk_text_cu, dtype=torch.long, device=device),
            torch.tensor(chunk_summary_cu, dtype=torch.long, device=device),
        )

    def _run_hsd_segments_packed_ref(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor | None,
        erase: torch.Tensor | None,
        write: torch.Tensor | None,
        summary_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        summary_cu_seqlens: torch.Tensor,
        collect_summary_states: bool = True,
        force_no_decay: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if q.shape[0] != 1:
            raise NotImplementedError("HSD packed reference path requires flattened batch_size=1 inputs.")
        text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        summary_cu_cpu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        if text_cu_cpu.numel() != summary_cu_cpu.numel():
            raise ValueError("HSD packed text_cu_seqlens and summary_cu_seqlens must describe the same documents.")

        text_outputs = []
        summary_outputs = []
        state_outputs = []
        recurrent_state = q.new_zeros(q.shape[0], self.num_v_heads, self.head_k_dim, self.head_v_dim)
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            summary_start = int(summary_cu_cpu[doc_idx].item())
            summary_end = int(summary_cu_cpu[doc_idx + 1].item())
            local_boundaries = boundary_counts[summary_start:summary_end] - text_start
            text_o_i, summary_o_i, states_i, state_i = self._run_hsd_segments(
                q=q[:, text_start:text_end],
                k=k[:, text_start:text_end],
                v=v[:, text_start:text_end],
                g=g[:, text_start:text_end],
                beta=None if beta is None else beta[:, text_start:text_end],
                erase=None if erase is None else erase[:, text_start:text_end],
                write=None if write is None else write[:, text_start:text_end],
                summary_q=summary_q[:, summary_start:summary_end],
                boundary_counts=local_boundaries,
                boundary_start=0,
                initial_state=None,
                collect_summary_states=collect_summary_states,
                force_no_decay=force_no_decay,
            )
            text_outputs.append(text_o_i)
            recurrent_state = state_i
            if summary_end > summary_start:
                summary_outputs.append(summary_o_i)
                if collect_summary_states and states_i is not None:
                    state_outputs.append(states_i)

        text_o = torch.cat(text_outputs, dim=1) if text_outputs else q.new_zeros(
            q.shape[0], 0, self.num_v_heads, self.head_v_dim
        )
        summary_o = torch.cat(summary_outputs, dim=1) if summary_outputs else q.new_zeros(
            q.shape[0], 0, self.num_v_heads, self.head_v_dim
        )
        if collect_summary_states:
            states = torch.cat(state_outputs, dim=1) if state_outputs else q.new_zeros(
                q.shape[0], 0, self.num_v_heads, self.head_k_dim, self.head_v_dim
            )
        else:
            states = None
        return text_o, summary_o, states, recurrent_state

    def _run_hsd_segments(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor | None,
        summary_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int = 0,
        initial_state: torch.Tensor | None = None,
        collect_summary_states: bool = True,
        text_cu_seqlens: torch.Tensor | None = None,
        summary_cu_seqlens: torch.Tensor | None = None,
        force_no_decay: bool = False,
        reset_state_after_summary_boundary: bool = False,
        erase: torch.Tensor | None = None,
        write: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        packed = text_cu_seqlens is not None or summary_cu_seqlens is not None
        if packed:
            if text_cu_seqlens is None or summary_cu_seqlens is None:
                raise ValueError("HSD packed segment path requires both text_cu_seqlens and summary_cu_seqlens.")
            if boundary_start != 0 or initial_state is not None:
                raise NotImplementedError("HSD packed segment path is training/prefill-only.")
            strict = bool(getattr(self, "hsd_train_selective_scan_packed_strict", False))
            local_core = getattr(self, "hsd_local_core", "gdn")
            if local_core == "kda":
                packed_scan = hsd_chunk_kda_with_summary_states
            elif local_core == "gdn2":
                packed_scan = hsd_chunk_gdn2_with_summary_states
            else:
                packed_scan = (
                    hsd_flashqla_tile_gated_delta_rule_with_summary_states
                    if getattr(self, "hsd_train_scan_backend", "triton") == "flashqla_tile"
                    else hsd_chunk_gated_delta_rule_with_summary_states
                )
            fastpath_enabled = (
                bool(getattr(self, "hsd_train_selective_scan", False))
                and bool(getattr(self, "hsd_train_selective_scan_packed", False))
                and packed_scan is not None
                and q.is_cuda
            )
            if fastpath_enabled:
                q_fast = self._expand_query_to_value_heads(l2norm(q.contiguous()))
                k_fast = self._expand_query_to_value_heads(l2norm(k.contiguous()))
                packed_scan_kwargs = dict(
                    query=q_fast,
                    key=k_fast,
                    value=v,
                    g=self._zero_g_decay(g) if force_no_decay else self._effective_g_decay(g),
                    summary_q=summary_q,
                    boundary_counts=boundary_counts,
                    text_cu_seqlens=text_cu_seqlens,
                    summary_cu_seqlens=summary_cu_seqlens,
                    scale=self.head_k_dim ** -0.5,
                )
                if local_core == "gdn2":
                    if erase is None or write is None:
                        raise RuntimeError("HSD-GDN2 packed path requires erase and write gates.")
                    packed_scan_kwargs.update(erase=erase, write=write)
                else:
                    if beta is None:
                        raise RuntimeError(f"HSD-{local_core} packed path requires beta.")
                    packed_scan_kwargs["beta"] = beta
                outputs = packed_scan(**packed_scan_kwargs)
                if outputs is not None:
                    text_o, summary_o, states = outputs
                    if (
                        local_core == "gdn"
                        and getattr(self, "hsd_train_scan_backend", "triton")
                        == "flashqla_tile"
                        and self._hsd_debug_enabled()
                    ):
                        self._hsd_debug_stats[
                            "hsd/train_scan_backend_flashqla_tile"
                        ] = 1.0
                    recurrent_state = q.new_zeros(q.shape[0], self.num_v_heads, self.head_k_dim, self.head_v_dim)
                    return text_o, summary_o, states if collect_summary_states else None, recurrent_state
                if strict:
                    raise RuntimeError(
                        "HSD packed selective-scan strict mode requested a fastpath hit, "
                        f"but hsd_chunk_{local_core} backend="
                        f"{getattr(self, 'hsd_train_scan_backend', 'triton')} returned None."
                    )
            elif strict:
                raise RuntimeError(
                    "HSD packed selective-scan strict mode requested a fastpath hit, "
                    "but packed fastpath conditions were not met."
                )
            return self._run_hsd_segments_packed_ref(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                erase=erase,
                write=write,
                summary_q=summary_q,
                boundary_counts=boundary_counts,
                text_cu_seqlens=text_cu_seqlens,
                summary_cu_seqlens=summary_cu_seqlens,
                collect_summary_states=collect_summary_states,
                force_no_decay=force_no_decay,
            )

        batch_size, text_len, _, _ = q.shape
        # The exported Stage4 profile was trained with the FlashQLA producer,
        # but the original HF single-request inference path split the prompt
        # into FLA segments.  Those implementations are mathematically
        # equivalent yet produce different FP32 recurrent states, and the
        # difference is amplified across layers during autoregressive decode.
        # Use the same producer as training/vLLM for the initial contiguous
        # prefill.  Cached one-token calls keep using the recurrent path below.
        flashqla_initial_prefill = (
            self.hsd_local_core == "gdn"
            and getattr(self, "hsd_train_scan_backend", "triton")
            == "flashqla_tile"
            and not self.training
            and q.is_cuda
            and batch_size == 1
            and text_len > 0
            and int(boundary_start) == 0
            and initial_state is None
            and not force_no_decay
            and not reset_state_after_summary_boundary
        )
        if flashqla_initial_prefill:
            if hsd_flashqla_tile_gated_delta_rule_inference is None:
                raise RuntimeError(
                    "Stage4 FlashQLA HF inference prefill requested the shared "
                    "training producer, but it is unavailable."
                )
            if beta is None:
                raise RuntimeError("Stage4 FlashQLA HF prefill requires beta.")
            summary_count = int(boundary_counts.numel())
            text_cu = torch.tensor(
                [0, text_len],
                dtype=torch.int32,
                device=q.device,
            )
            summary_cu = torch.tensor(
                [0, summary_count],
                dtype=torch.int32,
                device=q.device,
            )
            q_fast = self._expand_query_to_value_heads(l2norm(q.contiguous()))
            k_fast = self._expand_query_to_value_heads(l2norm(k.contiguous()))
            text_o, states, final_states = (
                hsd_flashqla_tile_gated_delta_rule_inference(
                    query=q_fast,
                    key=k_fast,
                    value=v,
                    beta=beta,
                    g=self._effective_g_decay(g),
                    boundary_counts=boundary_counts,
                    text_cu_seqlens=text_cu,
                    summary_cu_seqlens=summary_cu,
                    initial_state=None,
                    scale=self.head_k_dim**-0.5,
                )
            )
            expected_states = (
                batch_size,
                summary_count,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
            if tuple(states.shape) != expected_states:
                raise RuntimeError(
                    "Stage4 FlashQLA HF prefill returned invalid summary-state "
                    f"shape {tuple(states.shape)}; expected {expected_states}."
                )
            expected_final = (
                batch_size,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
            if tuple(final_states.shape) != expected_final:
                raise RuntimeError(
                    "Stage4 FlashQLA HF prefill returned invalid final-state "
                    f"shape {tuple(final_states.shape)}; expected {expected_final}."
                )
            if summary_count:
                summary_o = (
                    torch.einsum(
                        "bmhk,bmhkv->bmhv",
                        summary_q.to(states.dtype),
                        states,
                    )
                    * (self.head_k_dim**-0.5)
                ).to(text_o.dtype)
            else:
                summary_o = q.new_zeros(
                    batch_size,
                    0,
                    self.num_v_heads,
                    self.head_v_dim,
                )
            if self._hsd_debug_enabled():
                self._hsd_debug_stats[
                    "hsd/inference_prefill_backend_flashqla_tile"
                ] = 1.0
            return (
                text_o,
                summary_o,
                states if collect_summary_states else None,
                final_states,
            )

        text_outputs = []
        summary_outputs = []
        summary_states = []
        state = initial_state
        cursor = 0
        zero_state = q.new_zeros(batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim)

        boundary_list = boundary_counts.tolist()
        for summary_idx, boundary in enumerate(boundary_list):
            local_boundary = int(boundary) - int(boundary_start)
            if local_boundary < cursor:
                raise ValueError("summary positions must be ordered after non-decreasing text prefixes.")
            if local_boundary > cursor:
                out, state = self._run_delta_segment(
                    q, k, v, g, beta, cursor, local_boundary, state,
                    force_no_decay=force_no_decay,
                    erase=erase,
                    write=write,
                )
                text_outputs.append(out)
                cursor = local_boundary

            state_for_summary = zero_state if state is None else state
            if collect_summary_states:
                summary_states.append(state_for_summary)
            summary_outputs.append(torch.einsum(
                "bhk,bhkv->bhv",
                summary_q[:, summary_idx].to(state_for_summary.dtype),
                state_for_summary,
            ) * (self.head_k_dim ** -0.5))
            is_last_at_boundary = (
                summary_idx + 1 == len(boundary_list)
                or int(boundary_list[summary_idx + 1]) != int(boundary)
            )
            if reset_state_after_summary_boundary and is_last_at_boundary:
                state = None

        if cursor < text_len:
            out, state = self._run_delta_segment(
                q, k, v, g, beta, cursor, text_len, state,
                force_no_decay=force_no_decay,
                erase=erase,
                write=write,
            )
            text_outputs.append(out)

        text_o = torch.cat(text_outputs, dim=1) if text_outputs else q.new_zeros(
            batch_size, 0, self.num_v_heads, self.head_v_dim
        )
        summary_o = torch.stack(summary_outputs, dim=1).to(text_o.dtype) if summary_outputs else q.new_zeros(
            batch_size, 0, self.num_v_heads, self.head_v_dim
        )
        if collect_summary_states:
            states = torch.stack(summary_states, dim=1) if summary_states else q.new_zeros(
                batch_size, 0, self.num_v_heads, self.head_k_dim, self.head_v_dim
            )
        else:
            states = None
        if state is None:
            state = zero_state
        return text_o, summary_o, states, state

    def _record_dual_state_debug_stats(
        self,
        continuous_states: torch.Tensor | None,
        chunk_states: torch.Tensor | None,
        boundary_counts: torch.Tensor,
        *,
        record_debug_stats: bool,
    ) -> None:
        if not record_debug_stats:
            return
        self._hsd_debug_stats["hsd/history_state_chunk_reset_no_decay"] = 1.0
        if continuous_states is None or chunk_states is None or chunk_states.numel() == 0:
            return
        with torch.no_grad():
            continuous = continuous_states.detach().float()
            chunk = chunk_states.detach().float()
            continuous_norm = torch.linalg.vector_norm(continuous, dim=(-2, -1)).mean()
            chunk_norm = torch.linalg.vector_norm(chunk, dim=(-2, -1)).mean()
            cross_cos = F.cosine_similarity(
                continuous.flatten(start_dim=3),
                chunk.flatten(start_dim=3),
                dim=-1,
            ).mean()
            stats = {
                "hsd/local_snapshot_norm": continuous_norm,
                "hsd/continuous_snapshot_norm": continuous_norm,
                "hsd/chunk_snapshot_norm": chunk_norm,
                "hsd/continuous_chunk_snapshot_cos": cross_cos,
            }
            if chunk.shape[1] > 1:
                unique_boundary = torch.ones(
                    boundary_counts.shape[0], dtype=torch.bool, device=boundary_counts.device
                )
                unique_boundary[1:] = boundary_counts[1:] != boundary_counts[:-1]
                unique_chunk = chunk[:, unique_boundary]
                if unique_chunk.shape[1] > 1:
                    stats["hsd/adjacent_chunk_snapshot_cos"] = F.cosine_similarity(
                        unique_chunk[:, 1:].flatten(start_dim=3),
                        unique_chunk[:, :-1].flatten(start_dim=3),
                        dim=-1,
                    ).mean()
            self._hsd_debug_stats.update(stats)

    def _record_chunk_beta_debug_stats(
        self,
        local_beta: torch.Tensor,
        chunk_beta: torch.Tensor,
        *,
        record_debug_stats: bool,
    ) -> None:
        if not record_debug_stats:
            return
        with torch.no_grad():
            local = local_beta.detach().float()
            chunk = chunk_beta.detach().float()
            eps = torch.finfo(torch.float32).eps
            local_flat = local.flatten()
            chunk_flat = chunk.flatten()
            stats = {
                "hsd/chunk_beta_mode_independent": float(self.hsd_chunk_beta_mode == "independent"),
                "hsd/local_beta_mean": local_flat.mean(),
                "hsd/local_beta_p10": torch.quantile(local_flat, 0.1),
                "hsd/local_beta_p50": torch.quantile(local_flat, 0.5),
                "hsd/local_beta_p90": torch.quantile(local_flat, 0.9),
                "hsd/chunk_beta_mean": chunk_flat.mean(),
                "hsd/chunk_beta_p10": torch.quantile(chunk_flat, 0.1),
                "hsd/chunk_beta_p50": torch.quantile(chunk_flat, 0.5),
                "hsd/chunk_beta_p90": torch.quantile(chunk_flat, 0.9),
                "hsd/chunk_local_beta_mean_ratio": chunk_flat.mean() / local_flat.mean().clamp_min(eps),
                "hsd/chunk_local_beta_abs_diff_mean": (chunk_flat - local_flat).abs().mean(),
                "hsd/chunk_local_beta_cos": F.cosine_similarity(
                    chunk.flatten(start_dim=1),
                    local.flatten(start_dim=1),
                    dim=-1,
                ).mean(),
            }
            self._hsd_debug_stats.update(stats)

    def _run_dual_state_segments(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        summary_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int,
        initial_state: torch.Tensor | None,
        chunk_initial_state: torch.Tensor | None,
        collect_summary_states: bool,
        text_cu_seqlens: torch.Tensor | None,
        summary_cu_seqlens: torch.Tensor | None,
        record_debug_stats: bool,
        chunk_beta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        chunk_beta = beta if chunk_beta is None else chunk_beta
        self._record_chunk_beta_debug_stats(
            beta,
            chunk_beta,
            record_debug_stats=record_debug_stats,
        )
        collect_continuous_states = bool(record_debug_stats)
        text_o, _, continuous_states, recurrent_state = self._run_hsd_segments(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            summary_q=summary_q,
            boundary_counts=boundary_counts,
            boundary_start=boundary_start,
            initial_state=initial_state,
            collect_summary_states=collect_continuous_states,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
        )

        packed = text_cu_seqlens is not None or summary_cu_seqlens is not None
        chunk_text_cu = text_cu_seqlens
        chunk_summary_cu = summary_cu_seqlens
        if packed:
            if text_cu_seqlens is None or summary_cu_seqlens is None:
                raise ValueError("HSD dual-state packed path requires both text and summary cu_seqlens.")
            if chunk_initial_state is not None:
                raise NotImplementedError("HSD dual-state packed path does not support cached chunk state.")
            chunk_text_cu, chunk_summary_cu = self._chunk_reset_packed_cu_seqlens(
                boundary_counts,
                text_cu_seqlens,
                summary_cu_seqlens,
            )

        collect_chunk_states = bool(collect_summary_states or record_debug_stats)
        _, summary_o, chunk_states, chunk_memory_state = self._run_hsd_segments(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=chunk_beta,
            summary_q=summary_q,
            boundary_counts=boundary_counts,
            boundary_start=boundary_start,
            initial_state=chunk_initial_state,
            collect_summary_states=collect_chunk_states,
            text_cu_seqlens=chunk_text_cu,
            summary_cu_seqlens=chunk_summary_cu,
            force_no_decay=True,
            reset_state_after_summary_boundary=not packed,
        )
        self._record_dual_state_debug_stats(
            continuous_states,
            chunk_states,
            boundary_counts,
            record_debug_stats=record_debug_stats,
        )
        return (
            text_o,
            summary_o,
            chunk_states if collect_summary_states else None,
            recurrent_state,
            chunk_memory_state,
        )

    def _run_delta_segment(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor | None,
        start: int,
        end: int,
        initial_state: torch.Tensor | None,
        force_no_decay: bool = False,
        erase: torch.Tensor | None = None,
        write: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        segment_len = int(end) - int(start)
        if self.hsd_local_core == "gdn2":
            if chunk_gdn2 is None:
                raise RuntimeError("HSD-GDN2 requires the FLA GDN2 kernels.")
            if erase is None or write is None:
                raise RuntimeError("HSD-GDN2 requires erase and write gates.")
            segment_g = self._effective_g_decay(g[:, start:end])
            if (
                fused_recurrent_gdn2 is not None
                and segment_len == 1
                and q.is_cuda
                and not self.training
            ):
                return fused_recurrent_gdn2(
                    q=q[:, start:end],
                    k=k[:, start:end],
                    v=v[:, start:end],
                    g=segment_g,
                    b=erase[:, start:end],
                    w=write[:, start:end],
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=False,
                )
            return chunk_gdn2(
                q=q[:, start:end],
                k=k[:, start:end],
                v=v[:, start:end],
                g=segment_g,
                b=erase[:, start:end],
                w=write[:, start:end],
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
            )
        if self.hsd_local_core == "kda":
            if chunk_kda is None:
                raise RuntimeError("HSD-KDA requires the FLA KDA kernels.")
            if beta is None:
                raise RuntimeError("HSD-KDA requires beta.")
            segment_g = (
                self._zero_g_decay(g[:, start:end])
                if force_no_decay else self._effective_g_decay(g[:, start:end])
            )
            if (
                fused_recurrent_kda is not None
                and segment_len == 1
                and q.is_cuda
                and not self.training
            ):
                return fused_recurrent_kda(
                    q=q[:, start:end],
                    k=k[:, start:end],
                    v=v[:, start:end],
                    g=segment_g,
                    beta=beta[:, start:end],
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=False,
                )
            return chunk_kda(
                q=q[:, start:end],
                k=k[:, start:end],
                v=v[:, start:end],
                g=segment_g,
                beta=beta[:, start:end],
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
            )
        recurrent_decode_requested = _env_bool("HSD_FUSED_GDN_RECURRENT_DECODE", False)
        if beta is None:
            raise RuntimeError("HSD-GDN requires beta.")
        recurrent_decode_strict = _env_bool("HSD_FUSED_GDN_RECURRENT_DECODE_STRICT", False)
        can_try_recurrent_decode = (
            recurrent_decode_requested
            and fused_recurrent_gated_delta_rule is not None
            and segment_len == 1
            and q.is_cuda
            and not self.training
        )
        if can_try_recurrent_decode:
            try:
                if force_no_decay or self.hsd_state_decay_mode == "none":
                    return fused_recurrent_gated_delta_rule(
                        q=q[:, start:end],
                        k=k[:, start:end],
                        v=v[:, start:end],
                        g=(
                            self._zero_g_decay(g[:, start:end])
                            if force_no_decay else self._effective_g_decay(g[:, start:end])
                        ),
                        beta=beta[:, start:end],
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        use_gate_in_kernel=False,
                    )
                return fused_recurrent_gated_delta_rule(
                    q=q[:, start:end],
                    k=k[:, start:end],
                    v=v[:, start:end],
                    g=g[:, start:end],
                    beta=beta[:, start:end],
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                )
            except Exception:
                if recurrent_decode_strict:
                    raise
        elif recurrent_decode_requested and recurrent_decode_strict and segment_len == 1 and not self.training:
            raise RuntimeError("HSD fused GDN recurrent decode strict mode requested a fastpath hit, but it missed.")
        if force_no_decay or self.hsd_state_decay_mode == "none":
            return chunk_gated_delta_rule(
                q=q[:, start:end],
                k=k[:, start:end],
                v=v[:, start:end],
                g=(
                    self._zero_g_decay(g[:, start:end])
                    if force_no_decay else self._effective_g_decay(g[:, start:end])
                ),
                beta=beta[:, start:end],
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
            )
        return chunk_gated_delta_rule(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    def _route_logits_for_value_heads(
        self,
        router_q: torch.Tensor,
        summary_k: torch.Tensor,
        value_heads: int,
    ) -> torch.Tensor:
        route_q = self._route_queries_for_value_heads(router_q.float(), value_heads)
        route_k = self._route_keys_for_query_heads(summary_k.float(), value_heads)
        route_scale = self.hsd_route_head_dim ** -0.5 if self.hsd_use_route_scale else 1.0
        route_scale = route_scale * self.hsd_route_logit_scale
        return torch.einsum("bthk,bmhk->bthm", route_q, route_k) * route_scale

    @staticmethod
    def _history_valid_prefix(
        *,
        boundary_counts: torch.Tensor,
        text_len: int,
        start_index: int = 0,
        text_cu_seqlens: torch.Tensor | None = None,
        summary_cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = boundary_counts.device
        summary_count = int(boundary_counts.numel())
        token_index = torch.arange(start_index, start_index + text_len, device=device, dtype=torch.long)
        boundaries = boundary_counts.to(device=device, dtype=torch.long)
        causal = boundaries.view(1, summary_count) <= token_index.view(text_len, 1)
        if text_cu_seqlens is None and summary_cu_seqlens is None:
            return causal
        if text_cu_seqlens is None or summary_cu_seqlens is None:
            raise ValueError("Packed HSD route momentum requires both text and summary cu_seqlens.")
        text_cu = text_cu_seqlens.to(device=device, dtype=torch.long)
        summary_cu = summary_cu_seqlens.to(device=device, dtype=torch.long)
        local_token_index = torch.arange(text_len, device=device, dtype=torch.long)
        token_doc = torch.searchsorted(text_cu[1:], local_token_index, right=True)
        summary_index = torch.arange(summary_count, device=device, dtype=torch.long)
        summary_doc = torch.searchsorted(summary_cu[1:], summary_index, right=True)
        return causal & token_doc.view(text_len, 1).eq(summary_doc.view(1, summary_count))

    @staticmethod
    def _route_momentum_reset_mask(
        has_history: torch.Tensor,
        *,
        has_initial_state: bool,
        text_cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, text_len = has_history.shape
        previous = torch.zeros_like(has_history)
        if text_len > 1:
            previous[:, 1:] = has_history[:, :-1]
        if has_initial_state and text_len > 0:
            previous[:, 0] = True
        reset = has_history & ~previous
        if text_cu_seqlens is not None:
            if batch_size != 1:
                raise NotImplementedError("Packed route momentum reset requires batch_size=1.")
            starts = text_cu_seqlens[:-1].to(device=has_history.device, dtype=torch.long)
            starts = starts[(starts >= 0) & (starts < text_len)]
            if starts.numel() > 0:
                reset[0, starts] = has_history[0, starts]
        return reset

    @staticmethod
    def _align_route_momentum_state(
        summary_state: torch.Tensor | None,
        null_state: torch.Tensor | None,
        *,
        batch_size: int,
        heads: int,
        summary_count: int,
        include_null: bool,
        device: torch.device,
    ) -> torch.Tensor | None:
        if summary_state is None:
            return None
        expected_prefix = (batch_size, heads)
        if tuple(summary_state.shape[:2]) != expected_prefix or summary_state.ndim != 3:
            raise RuntimeError(f"Invalid cached route momentum summary state shape: {tuple(summary_state.shape)}.")
        old_count = int(summary_state.shape[-1])
        if old_count > summary_count:
            raise RuntimeError(
                f"Cached route momentum has {old_count} summaries, but current cache only has {summary_count}."
            )
        summary = summary_state.to(device=device, dtype=torch.float32)
        if old_count < summary_count:
            summary = F.pad(summary, (0, summary_count - old_count))
        if not include_null:
            return summary
        if null_state is None:
            null = torch.zeros(batch_size, heads, 1, device=device, dtype=torch.float32)
        else:
            if tuple(null_state.shape) != (batch_size, heads, 1):
                raise RuntimeError(f"Invalid cached route momentum null state shape: {tuple(null_state.shape)}.")
            null = null_state.to(device=device, dtype=torch.float32)
        return torch.cat([summary, null], dim=-1)

    def _history_read_momentum(
        self,
        *,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        summary_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_len: int,
        start_index: int,
        null_logits: torch.Tensor | None,
        record_debug_stats: bool,
        return_local_prob: bool,
        initial_summary_state: torch.Tensor | None,
        initial_null_state: torch.Tensor | None,
        reset_last_token: bool = False,
        text_cu_seqlens: torch.Tensor | None = None,
        summary_cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        if route_weight_momentum_scan is None or hsd_weighted_global_read is None:
            raise RuntimeError("HSD route-weight momentum helpers are unavailable.")
        batch_size, _, heads, _ = read_q.shape
        summary_count = int(states.shape[1])
        logits = self._route_logits_for_value_heads(router_q, summary_k, heads)
        valid_prefix = self._history_valid_prefix(
            boundary_counts=boundary_counts,
            text_len=text_len,
            start_index=start_index,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
        )
        valid = valid_prefix.view(1, text_len, 1, summary_count).expand(batch_size, -1, heads, -1)
        has_valid = valid_prefix.any(dim=-1).view(1, text_len, 1, 1).expand(batch_size, -1, -1, -1)
        if null_logits is not None:
            if tuple(null_logits.shape) != (batch_size, text_len, heads, 1):
                raise RuntimeError(
                    f"HSD null read logits shape mismatch: got {tuple(null_logits.shape)}, "
                    f"expected {(batch_size, text_len, heads, 1)}."
                )
            null_valid = torch.ones(batch_size, text_len, heads, 1, device=valid.device, dtype=torch.bool)
            all_logits = torch.cat([logits, null_logits.float()], dim=-1)
            all_valid = torch.cat([valid, null_valid], dim=-1)
            raw_weights = torch.softmax(all_logits.masked_fill(~all_valid, -torch.inf), dim=-1)
        else:
            masked_logits = logits.masked_fill(~valid, -torch.inf)
            masked_logits = torch.where(has_valid, masked_logits, torch.zeros_like(masked_logits))
            raw_weights = torch.softmax(masked_logits, dim=-1)
            raw_weights = torch.where(has_valid, raw_weights, torch.zeros_like(raw_weights))

        initial_state = self._align_route_momentum_state(
            initial_summary_state,
            initial_null_state,
            batch_size=batch_size,
            heads=heads,
            summary_count=summary_count,
            include_null=null_logits is not None,
            device=read_q.device,
        )
        # A decode boundary can create a momentum cache before that summary is
        # causally visible.  Tensor existence is therefore insufficient: use
        # the previous token's causal boundary to decide whether EMA continues.
        initial_has_history = bool(
            initial_state is not None
            and start_index > 0
            and (boundary_counts <= start_index - 1).any().item()
        )
        has_history = has_valid.squeeze(-1).squeeze(-1)
        reset_mask = self._route_momentum_reset_mask(
            has_history,
            has_initial_state=initial_has_history,
            text_cu_seqlens=text_cu_seqlens,
        )
        last_token_reset_applied = bool(reset_last_token and text_len > 0)
        if last_token_reset_applied:
            reset_mask = reset_mask.clone()
            reset_mask[:, -1] |= has_history[:, -1]
        smoothed, final_state = route_weight_momentum_scan(
            raw_weights=raw_weights,
            has_history=has_history,
            reset_mask=reset_mask,
            gamma=self.hsd_route_weight_momentum_gamma,
            initial_state=initial_state,
        )
        summary_weights = smoothed[..., :summary_count]
        null_prob = smoothed[..., summary_count:summary_count + 1] if null_logits is not None else None
        history = hsd_weighted_global_read(
            read_q=read_q,
            states=states,
            summary_weights=summary_weights,
            read_scale=self.head_k_dim ** -0.5,
        )
        summary_mass = summary_weights.sum(dim=-1, keepdim=True)
        debug_weights = torch.where(
            summary_mass > 0,
            summary_weights / summary_mass.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(summary_weights),
        )
        boundaries = boundary_counts.to(device=read_q.device, dtype=torch.long)
        token_index = torch.arange(start_index, start_index + text_len, device=read_q.device, dtype=torch.long)
        self._record_hsd_router_debug_stats(
            weights=debug_weights,
            logits=logits.masked_fill(~valid, -torch.inf),
            summary_k=summary_k,
            valid_prefix=valid_prefix,
            has_valid=has_valid,
            boundaries=boundaries,
            token_index=token_index,
            null_prob=null_prob,
            summary_mass=summary_mass,
            record_debug_stats=record_debug_stats,
        )
        if record_debug_stats:
            self._record_route_momentum_debug_stats(
                raw_weights=raw_weights,
                smoothed_weights=smoothed,
                has_history=has_history,
                reset_mask=reset_mask,
                include_null=null_logits is not None,
                last_prompt_reset=last_token_reset_applied,
            )
        final_summary = final_state[..., :summary_count]
        final_null = final_state[..., summary_count:summary_count + 1] if null_logits is not None else None
        return history, null_prob if return_local_prob else None, final_summary, final_null

    def _history_read_sparse_topk(
        self,
        *,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        summary_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_len: int,
        start_index: int,
        null_logits: torch.Tensor | None,
        record_debug_stats: bool,
        return_local_prob: bool,
        text_cu_seqlens: torch.Tensor | None = None,
        summary_cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Route densely, then read only the selected summary states."""
        if hsd_sparse_topk_global_read is None:
            raise RuntimeError("HSD sparse top-k global-read helper is unavailable.")
        batch_size, _, heads, _ = read_q.shape
        summary_count = int(states.shape[1])
        topk = int(self.hsd_read_topk)
        if topk not in {1, 2, 4, 8, 16}:
            raise RuntimeError(f"HSD sparse top-k supports K in {{1, 2, 4, 8, 16}}, got {topk}.")

        fused_route = None
        if (
            hsd_fused_route_topk is not None
            and fused_route_topk_enabled is not None
            and fused_route_topk_enabled()
        ):
            route_scale = self.hsd_route_head_dim ** -0.5 if self.hsd_use_route_scale else 1.0
            route_scale *= self.hsd_route_logit_scale
            fused_route = hsd_fused_route_topk(
                route_q=router_q,
                route_k=summary_k,
                boundary_counts=boundary_counts,
                value_heads=heads,
                topk=topk,
                start_index=start_index,
                route_scale=route_scale,
                null_logits=null_logits,
                text_cu_seqlens=text_cu_seqlens,
                summary_cu_seqlens=summary_cu_seqlens,
            )
            if fused_route is None and read_q.is_cuda and _env_bool("HSD_TOPK_FUSED_ROUTE_STRICT", False):
                raise RuntimeError("HSD fused sparse top-k route strict mode requested a hit, but it missed.")

        if fused_route is not None:
            safe_indices, selected_weights, selected_valid, null_prob = fused_route
            summary_mass = (
                selected_weights.sum(dim=-1, keepdim=True) if null_prob is not None else None
            )
            logits = valid_prefix = valid = masked_logits = has_valid = None
        else:
            logits = self._route_logits_for_value_heads(router_q, summary_k, heads)
            valid_prefix = self._history_valid_prefix(
                boundary_counts=boundary_counts,
                text_len=text_len,
                start_index=start_index,
                text_cu_seqlens=text_cu_seqlens,
                summary_cu_seqlens=summary_cu_seqlens,
            )
            valid = valid_prefix.view(1, text_len, 1, summary_count).expand(batch_size, -1, heads, -1)
            masked_logits = logits.masked_fill(~valid, -torch.inf)

            # Keep the configured K as a compile-time kernel bucket even before a
            # document has accumulated K summaries. Padded candidates are invalid
            # and use safe state index zero with exactly zero probability.
            if summary_count < topk:
                pad = topk - summary_count
                ranked_logits = F.pad(masked_logits, (0, pad), value=-torch.inf)
                ranked_valid = F.pad(valid, (0, pad), value=False)
            else:
                ranked_logits = masked_logits
                ranked_valid = valid
            selected_logits, selected_indices = ranked_logits.topk(k=topk, dim=-1)
            selected_valid = ranked_valid.gather(-1, selected_indices)
            safe_indices = torch.where(
                selected_valid,
                selected_indices.clamp(0, summary_count - 1),
                torch.zeros_like(selected_indices),
            )
            has_valid = valid.any(dim=-1, keepdim=True)

            null_prob = None
            summary_mass = None
            if null_logits is not None:
                if tuple(null_logits.shape) != (batch_size, text_len, heads, 1):
                    raise RuntimeError(
                        f"HSD null read logits shape mismatch: got {tuple(null_logits.shape)}, "
                        f"expected {(batch_size, text_len, heads, 1)}."
                    )
                logits_with_null = torch.cat([selected_logits, null_logits.float()], dim=-1)
                valid_with_null = torch.cat(
                    [selected_valid, torch.ones_like(null_logits, dtype=torch.bool)], dim=-1
                )
                weights_with_null = torch.softmax(
                    logits_with_null.masked_fill(~valid_with_null, -torch.inf), dim=-1
                )
                selected_weights = weights_with_null[..., :topk]
                null_prob = weights_with_null[..., topk:topk + 1]
                summary_mass = selected_weights.sum(dim=-1, keepdim=True)
            else:
                stable_logits = torch.where(has_valid, selected_logits, torch.zeros_like(selected_logits))
                selected_weights = torch.softmax(
                    stable_logits.masked_fill(has_valid & ~selected_valid, -torch.inf), dim=-1
                )
                selected_weights = torch.where(has_valid, selected_weights, torch.zeros_like(selected_weights))
            selected_weights = selected_weights * selected_valid.to(selected_weights.dtype)

        history = hsd_sparse_topk_global_read(
            read_q=read_q,
            states=states,
            topk_indices=safe_indices,
            topk_weights=selected_weights.float(),
            topk_valid=selected_valid,
            read_scale=self.head_k_dim ** -0.5,
        )

        if record_debug_stats:
            if logits is None:
                # The fused route intentionally does not materialize dense
                # [B,T,H,M] logits. Recompute them only on debug sampling
                # steps, without attaching this diagnostic path to autograd.
                with torch.no_grad():
                    logits = self._route_logits_for_value_heads(router_q, summary_k, heads)
                    valid_prefix = self._history_valid_prefix(
                        boundary_counts=boundary_counts,
                        text_len=text_len,
                        start_index=start_index,
                        text_cu_seqlens=text_cu_seqlens,
                        summary_cu_seqlens=summary_cu_seqlens,
                    )
                    valid = valid_prefix.view(1, text_len, 1, summary_count).expand(
                        batch_size, -1, heads, -1
                    )
                    masked_logits = logits.masked_fill(~valid, -torch.inf)
                    has_valid = valid.any(dim=-1, keepdim=True)
            conditional_weights = selected_weights
            if summary_mass is not None:
                conditional_weights = torch.where(
                    summary_mass > 0,
                    selected_weights / summary_mass.clamp_min(torch.finfo(torch.float32).tiny),
                    torch.zeros_like(selected_weights),
                )
            debug_weights = torch.zeros_like(logits, dtype=torch.float32)
            debug_weights.scatter_add_(-1, safe_indices, conditional_weights.float())
            boundaries = boundary_counts.to(device=read_q.device, dtype=torch.long)
            token_index = torch.arange(
                start_index, start_index + text_len, device=read_q.device, dtype=torch.long
            )
            self._record_hsd_router_debug_stats(
                weights=debug_weights,
                logits=masked_logits,
                summary_k=summary_k,
                valid_prefix=valid_prefix,
                has_valid=has_valid,
                boundaries=boundaries,
                token_index=token_index,
                null_prob=null_prob,
                summary_mass=summary_mass,
                record_debug_stats=True,
            )
            self._record_sparse_topk_debug_stats(
                logits=logits,
                valid=valid,
                has_valid=has_valid,
                selected_indices=safe_indices,
                selected_valid=selected_valid,
                selected_weights=selected_weights,
                null_logits=null_logits,
                null_prob=null_prob,
            )

        if return_local_prob:
            if null_prob is None:
                raise RuntimeError("HSD local probability was requested but no local/null logits were provided.")
            return history, null_prob
        return history

    def _record_sparse_topk_debug_stats(
        self,
        *,
        logits: torch.Tensor,
        valid: torch.Tensor,
        has_valid: torch.Tensor,
        selected_indices: torch.Tensor,
        selected_valid: torch.Tensor,
        selected_weights: torch.Tensor,
        null_logits: torch.Tensor | None,
        null_prob: torch.Tensor | None,
    ) -> None:
        with torch.no_grad():
            active = has_valid.squeeze(-1).expand_as(logits[..., 0])
            stats = {
                "hsd/router_topk_k": logits.new_tensor(float(self.hsd_read_topk)),
            }
            if not bool(active.any().item()):
                stats["hsd/router_topk_active_count_mean"] = logits.new_zeros(())
                stats["hsd/router_topk_retained_mass"] = logits.new_zeros(())
                if null_prob is not None:
                    stats["hsd/router_topk_null_prob"] = logits.new_tensor(1.0)
                self._hsd_debug_stats.update(stats)
                return
            stats["hsd/router_topk_active_count_mean"] = selected_valid.float().sum(dim=-1)[active].mean()
            dense_logits = logits.detach().float().masked_fill(~valid, -torch.inf)
            stable_dense = torch.where(has_valid, dense_logits, torch.zeros_like(dense_logits))
            dense_conditional = torch.softmax(stable_dense, dim=-1)
            dense_conditional = torch.where(has_valid, dense_conditional, torch.zeros_like(dense_conditional))
            selected_dense_mass = dense_conditional.gather(-1, selected_indices)
            retained_mass = (selected_dense_mass * selected_valid.float()).sum(dim=-1)
            stats["hsd/router_topk_retained_mass"] = retained_mass[active].mean()

            usage = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.float32)
            usage.scatter_add_(0, selected_indices.reshape(-1), selected_weights.float().reshape(-1))
            usage_sum = usage.sum()
            if bool((usage_sum > 0).item()):
                usage = usage / usage_sum
                entropy = -(usage * usage.clamp_min(torch.finfo(torch.float32).tiny).log()).sum()
                stats["hsd/router_topk_selection_usage_effective_count"] = entropy.exp()
            if null_prob is not None and null_logits is not None:
                valid_null = null_prob.detach().float().squeeze(-1)[active]
                stats["hsd/router_topk_null_prob"] = valid_null.mean()
                dense_with_null = torch.cat([dense_logits, null_logits.detach().float()], dim=-1)
                dense_valid = torch.cat([valid, torch.ones_like(null_logits, dtype=torch.bool)], dim=-1)
                dense_null = torch.softmax(dense_with_null.masked_fill(~dense_valid, -torch.inf), dim=-1)[..., -1]
                stats["hsd/router_topk_null_prob_delta_vs_dense"] = (
                    null_prob.detach().float().squeeze(-1)[active] - dense_null[active]
                ).mean()
            self._hsd_debug_stats.update(stats)

    def _history_read(
        self,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        summary_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_len: int,
        start_index: int = 0,
        null_logits: torch.Tensor | None = None,
        record_debug_stats: bool | None = None,
        try_fused_decode: bool = False,
        return_local_prob: bool = False,
        momentum_initial_summary: torch.Tensor | None = None,
        momentum_initial_null: torch.Tensor | None = None,
        return_momentum_state: bool = False,
        reset_last_token: bool = False,
        log_anchor_counts: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        batch_size, _, heads, _ = read_q.shape
        summary_count = states.shape[1]
        if summary_count == 0:
            history = read_q.new_zeros(batch_size, text_len, heads, self.head_v_dim)
            local_prob = read_q.new_ones(batch_size, text_len, heads, 1) if return_local_prob else None
            if return_momentum_state:
                final_summary = torch.zeros(batch_size, heads, 0, device=read_q.device, dtype=torch.float32)
                final_null = (
                    torch.ones(batch_size, heads, 1, device=read_q.device, dtype=torch.float32)
                    if null_logits is not None else None
                )
                return history, local_prob, final_summary, final_null
            if return_local_prob:
                return history, local_prob
            return history
        if return_local_prob and null_logits is None:
            raise RuntimeError("Returning HSD local probability requires local/null logits.")
        if null_logits is not None and tuple(null_logits.shape) != (batch_size, text_len, heads, 1):
            raise RuntimeError(
                "HSD null read logits shape mismatch: "
                f"got {tuple(null_logits.shape)}, expected {(batch_size, text_len, heads, 1)}."
            )

        if record_debug_stats is None:
            record_debug_stats = self._hsd_should_record_debug_stats()
        if float(getattr(self, "hsd_route_weight_momentum_gamma", 0.0)) > 0.0:
            history, local_prob, final_summary, final_null = self._history_read_momentum(
                read_q=read_q,
                router_q=router_q,
                states=states,
                summary_k=summary_k,
                boundary_counts=boundary_counts,
                text_len=text_len,
                start_index=start_index,
                null_logits=null_logits,
                record_debug_stats=bool(record_debug_stats),
                return_local_prob=return_local_prob,
                initial_summary_state=momentum_initial_summary,
                initial_null_state=momentum_initial_null,
                reset_last_token=reset_last_token,
            )
            if return_momentum_state:
                return history, local_prob, final_summary, final_null
            if return_local_prob:
                return history, local_prob
            return history
        if self.hsd_read_topk > 0 and getattr(self, "hsd_topk_backend", "reference") == "sparse":
            if return_momentum_state:
                raise RuntimeError("Sparse top-k cannot return route-momentum cache state.")
            return self._history_read_sparse_topk(
                read_q=read_q,
                router_q=router_q,
                states=states,
                summary_k=summary_k,
                boundary_counts=boundary_counts,
                text_len=text_len,
                start_index=start_index,
                null_logits=null_logits,
                record_debug_stats=bool(record_debug_stats),
                return_local_prob=return_local_prob,
            )
        route_scale = self.hsd_route_head_dim ** -0.5 if self.hsd_use_route_scale else 1.0
        route_scale = route_scale * self.hsd_route_logit_scale
        read_scale = self.head_k_dim ** -0.5
        topk = max(0, int(self.hsd_read_topk))
        decode_fused_requested = _env_bool("HSD_FUSED_GLOBAL_READ_DECODE", True)
        decode_fused_strict = _env_bool("HSD_FUSED_GLOBAL_READ_DECODE_STRICT", False)
        can_try_decode_fused = (
            try_fused_decode
            and decode_fused_requested
            and log_anchor_counts is None
            and hsd_fused_global_read_decode is not None
            and topk == 0
            and not record_debug_stats
        )
        if can_try_decode_fused:
            fused = hsd_fused_global_read_decode(
                read_q=read_q,
                router_q=self._route_queries_for_value_heads(router_q, heads),
                states=states,
                summary_k=self._route_keys_for_query_heads(summary_k, heads),
                boundary_counts=boundary_counts,
                start_index=int(start_index),
                null_logits=null_logits,
                route_scale=route_scale,
                read_scale=read_scale,
                debug_stats_callback=None,
                return_local_prob=return_local_prob,
            )
            if fused is not None:
                return fused
        if (
            try_fused_decode
            and decode_fused_requested
            and decode_fused_strict
            and log_anchor_counts is None
            and topk == 0
            and not record_debug_stats
        ):
            raise RuntimeError("HSD fused global-read decode strict mode requested a fastpath hit, but it missed.")

        token_index = torch.arange(start_index, start_index + text_len, device=read_q.device, dtype=torch.long)
        boundaries = boundary_counts.to(device=read_q.device, dtype=torch.long)
        anchor_count_matrix = None
        if log_anchor_counts is not None:
            anchor_count_matrix = log_anchor_counts.to(device=read_q.device)
            if anchor_count_matrix.ndim == 1:
                anchor_count_matrix = anchor_count_matrix.view(1, summary_count).expand(text_len, summary_count)
            if tuple(anchor_count_matrix.shape) != (text_len, summary_count):
                raise RuntimeError(
                    "HSD Log-Anchor count shape mismatch: "
                    f"got {tuple(anchor_count_matrix.shape)}, expected {(text_len, summary_count)}."
                )
            if not bool(torch.isfinite(anchor_count_matrix.float()).all().item()):
                raise RuntimeError("HSD Log-Anchor counts must be finite.")
            if bool((anchor_count_matrix < 0).any().item()):
                raise RuntimeError("HSD Log-Anchor counts must be non-negative.")
        route_q = router_q.float()
        route_keys = summary_k.float()
        route_q_heads = int(route_q.shape[2])
        route_key_heads = int(route_keys.shape[2])
        if route_q.shape[:2] != (batch_size, text_len):
            raise RuntimeError(
                "HSD router query shape mismatch: "
                f"got {tuple(router_q.shape[:2])}, expected {(batch_size, text_len)}."
            )
        if route_q_heads not in {1, heads}:
            raise RuntimeError(
                "HSD route query heads must be 1 or match value heads: "
                f"got query_heads={route_q_heads}, value_heads={heads}."
            )
        if heads % route_key_heads != 0:
            raise RuntimeError(
                "HSD summary route key heads must divide value heads: "
                f"got key_heads={route_key_heads}, value_heads={heads}."
            )
        if route_q.shape[-1] != route_keys.shape[-1]:
            raise RuntimeError(
                "HSD route q/k dim mismatch: "
                f"router_q dim={route_q.shape[-1]}, summary_k dim={route_keys.shape[-1]}."
            )
        valid_prefix = boundaries.view(1, summary_count) <= token_index.view(text_len, 1)
        if anchor_count_matrix is not None:
            valid_prefix = valid_prefix & anchor_count_matrix.gt(0)

        if route_q_heads == heads and route_key_heads == heads:
            logits = torch.einsum("bthk,bmhk->bthm", route_q, route_keys) * route_scale
        elif route_q_heads == heads and route_key_heads == 1:
            logits = torch.einsum("bthk,bmk->bthm", route_q, route_keys[:, :, 0]) * route_scale
        elif route_q_heads == heads:
            grouped_keys = self._route_keys_for_query_heads(route_keys, heads)
            logits = torch.einsum("bthk,bmhk->bthm", route_q, grouped_keys) * route_scale
        elif route_key_heads == heads:
            logits = torch.einsum("btk,bmhk->bthm", route_q[:, :, 0], route_keys) * route_scale
        elif route_key_heads == 1:
            shared_logits = torch.einsum("btk,bmk->btm", route_q[:, :, 0], route_keys[:, :, 0]) * route_scale
            logits = shared_logits.view(batch_size, text_len, 1, summary_count).expand(
                batch_size, text_len, heads, summary_count
            )
        else:
            grouped_keys = self._route_keys_for_query_heads(route_keys, heads)
            logits = torch.einsum("btk,bmhk->bthm", route_q[:, :, 0], grouped_keys) * route_scale
        if anchor_count_matrix is not None and self.hsd_log_anchor_count_bias == "log":
            logits = logits + anchor_count_matrix.float().clamp_min(1).log().view(
                1, text_len, 1, summary_count
            )
        valid = valid_prefix.view(1, text_len, 1, summary_count).expand(batch_size, text_len, heads, summary_count)
        router_debug_logits = logits.masked_fill(~valid, -torch.inf)
        if 0 < topk < summary_count:
            topk = min(topk, summary_count)
            _, topk_idx = router_debug_logits.topk(k=topk, dim=-1)
            topk_mask = torch.zeros_like(logits, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_idx, True)
            valid = valid & topk_mask

        has_valid = valid_prefix.any(dim=-1).view(1, text_len, 1, 1)
        null_prob = None
        summary_mass = None
        debug_weights = None
        if null_logits is not None:
            null_valid = torch.ones(batch_size, text_len, heads, 1, device=valid.device, dtype=torch.bool)
            logits_with_null = torch.cat([logits, null_logits.float()], dim=-1)
            valid_with_null = torch.cat([valid, null_valid], dim=-1)
            weights_with_null = torch.softmax(logits_with_null.masked_fill(~valid_with_null, -torch.inf), dim=-1)
            weights = weights_with_null[..., :summary_count].to(states.dtype)
            null_prob = weights_with_null[..., summary_count:summary_count + 1].to(states.dtype)
            summary_mass = weights.sum(dim=-1, keepdim=True)
            debug_weights = torch.where(
                summary_mass > 0,
                weights / summary_mass.clamp_min(torch.finfo(weights.dtype).tiny),
                torch.zeros_like(weights),
            )
        else:
            masked_logits = logits.masked_fill(~valid, -torch.inf)
            masked_logits = torch.where(has_valid, masked_logits, torch.zeros_like(masked_logits))
            weights = torch.softmax(masked_logits, dim=-1)
            weights = torch.where(has_valid, weights, torch.zeros_like(weights)).to(states.dtype)
            debug_weights = weights
        self._record_hsd_router_debug_stats(
            weights=debug_weights,
            logits=router_debug_logits,
            summary_k=summary_k,
            valid_prefix=valid_prefix,
            has_valid=has_valid,
            boundaries=boundaries,
            token_index=token_index,
            null_prob=null_prob,
            summary_mass=summary_mass,
            record_debug_stats=record_debug_stats,
        )

        # Do not materialize every per-summary read for dense states.  That
        # temporary has shape [B, T, H, M, V] and becomes prohibitively large
        # for long prefill (for example, about 71 GiB at 64K with 128
        # summaries).  The weighted-read operator contracts the summary axis
        # inside the kernel and only materializes [B, T, H, V].
        if hsd_weighted_global_read is not None:
            history = hsd_weighted_global_read(
                read_q=read_q,
                states=states,
                summary_weights=weights,
                read_scale=read_scale,
            )
        else:
            values = torch.einsum("bthk,bmhkv->bthmv", read_q.to(states.dtype), states) * float(read_scale)
            history = torch.einsum("bthm,bthmv->bthv", weights.to(values.dtype), values)
        if return_local_prob:
            if null_prob is None:
                raise RuntimeError("HSD local probability was requested but not computed.")
            return history, null_prob
        return history

    def _history_read_packed(
        self,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        summary_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        summary_cu_seqlens: torch.Tensor,
        null_logits: torch.Tensor | None = None,
        record_debug_stats: bool | None = None,
        return_local_prob: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if record_debug_stats is None:
            record_debug_stats = self._hsd_should_record_debug_stats()
        if read_q.shape[0] != 1:
            raise NotImplementedError("HSD packed history read requires flattened batch_size=1 inputs.")
        if null_logits is not None and tuple(null_logits.shape[:3]) != tuple(read_q.shape[:3]):
            raise RuntimeError(
                "HSD packed null read logits shape mismatch: "
                f"got {tuple(null_logits.shape)}, expected prefix {tuple(read_q.shape[:3])}."
            )
        if float(getattr(self, "hsd_route_weight_momentum_gamma", 0.0)) > 0.0:
            history, local_prob, _, _ = self._history_read_momentum(
                read_q=read_q,
                router_q=router_q,
                states=states,
                summary_k=summary_k,
                boundary_counts=boundary_counts,
                text_len=int(read_q.shape[1]),
                start_index=0,
                null_logits=null_logits,
                record_debug_stats=bool(record_debug_stats),
                return_local_prob=return_local_prob,
                initial_summary_state=None,
                initial_null_state=None,
                text_cu_seqlens=text_cu_seqlens,
                summary_cu_seqlens=summary_cu_seqlens,
            )
            return (history, local_prob) if return_local_prob else history
        if self.hsd_read_topk > 0 and getattr(self, "hsd_topk_backend", "reference") == "sparse":
            return self._history_read_sparse_topk(
                read_q=read_q,
                router_q=router_q,
                states=states,
                summary_k=summary_k,
                boundary_counts=boundary_counts,
                text_len=int(read_q.shape[1]),
                start_index=0,
                null_logits=null_logits,
                record_debug_stats=bool(record_debug_stats),
                return_local_prob=return_local_prob,
                text_cu_seqlens=text_cu_seqlens,
                summary_cu_seqlens=summary_cu_seqlens,
            )
        route_scale = self.hsd_route_head_dim ** -0.5 if self.hsd_use_route_scale else 1.0
        route_scale = route_scale * self.hsd_route_logit_scale
        read_scale = self.head_k_dim ** -0.5
        fused_requested = _env_bool("HSD_FUSED_GLOBAL_READ", True)
        fused_strict = _env_bool("HSD_FUSED_GLOBAL_READ_STRICT", False)
        can_try_fused = (
            fused_requested
            and hsd_fused_global_read is not None
            and max(0, int(self.hsd_read_topk)) == 0
        )
        if can_try_fused:
            fused = hsd_fused_global_read(
                read_q=read_q,
                router_q=self._route_queries_for_value_heads(router_q, read_q.shape[2]),
                states=states,
                summary_k=self._route_keys_for_query_heads(summary_k, read_q.shape[2]),
                boundary_counts=boundary_counts,
                text_cu_seqlens=text_cu_seqlens,
                summary_cu_seqlens=summary_cu_seqlens,
                null_logits=null_logits,
                route_scale=route_scale,
                read_scale=read_scale,
                debug_stats_callback=self._record_hsd_router_debug_stats if record_debug_stats else None,
                return_local_prob=return_local_prob,
            )
            if fused is not None:
                return fused
        if (
            fused_strict
            and fused_requested
            and max(0, int(self.hsd_read_topk)) == 0
            and not record_debug_stats
        ):
            raise RuntimeError("HSD fused packed global read strict mode requested a fastpath hit, but it missed.")

        text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        summary_cu_cpu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        outputs = []
        local_probs = []
        value_dim = states.shape[-1]
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            summary_start = int(summary_cu_cpu[doc_idx].item())
            summary_end = int(summary_cu_cpu[doc_idx + 1].item())
            doc_text_len = text_end - text_start
            if doc_text_len <= 0:
                continue
            if summary_end <= summary_start:
                outputs.append(read_q.new_zeros(1, doc_text_len, read_q.shape[2], value_dim))
                if return_local_prob:
                    local_probs.append(
                        read_q.new_ones(1, doc_text_len, read_q.shape[2], 1, dtype=torch.float32)
                    )
                continue
            local_boundaries = boundary_counts[summary_start:summary_end] - text_start
            result = self._history_read(
                    read_q=read_q[:, text_start:text_end],
                    router_q=router_q[:, text_start:text_end],
                    states=states[:, summary_start:summary_end],
                    summary_k=summary_k[:, summary_start:summary_end],
                    boundary_counts=local_boundaries,
                    text_len=doc_text_len,
                    start_index=0,
                    null_logits=None if null_logits is None else null_logits[:, text_start:text_end],
                    record_debug_stats=record_debug_stats,
                    return_local_prob=return_local_prob,
                )
            if return_local_prob:
                history_i, local_prob_i = result
                outputs.append(history_i)
                local_probs.append(local_prob_i)
            else:
                outputs.append(result)
        if not outputs:
            history = read_q.new_zeros(read_q.shape[0], 0, read_q.shape[2], value_dim)
            if return_local_prob:
                return history, read_q.new_zeros(read_q.shape[0], 0, read_q.shape[2], 1)
            return history
        history = torch.cat(outputs, dim=1)
        if return_local_prob:
            return history, torch.cat(local_probs, dim=1)
        return history

    def _record_route_momentum_debug_stats(
        self,
        *,
        raw_weights: torch.Tensor,
        smoothed_weights: torch.Tensor,
        has_history: torch.Tensor,
        reset_mask: torch.Tensor,
        include_null: bool = False,
        last_prompt_reset: bool = False,
    ) -> None:
        with torch.no_grad():
            raw = raw_weights.detach().float()
            smooth = smoothed_weights.detach().float()
            active = has_history.to(device=raw.device, dtype=torch.bool)
            eps = torch.finfo(torch.float32).tiny
            stats = {
                "hsd/router_momentum_gamma": raw.new_tensor(self.hsd_route_weight_momentum_gamma),
                "hsd/decode_momentum_last_prompt_reset": raw.new_tensor(float(last_prompt_reset)),
            }
            if bool(active.any().item()):
                active_heads = active.unsqueeze(-1).expand(-1, -1, raw.shape[2])
                raw_smooth_l1 = (raw - smooth).abs().mean(dim=-1)
                raw_entropy = -(raw * raw.clamp_min(eps).log()).sum(dim=-1)
                smooth_entropy = -(smooth * smooth.clamp_min(eps).log()).sum(dim=-1)
                stats.update({
                    "hsd/router_raw_smoothed_l1": raw_smooth_l1[active_heads].mean(),
                    "hsd/router_raw_entropy": raw_entropy[active_heads].mean(),
                    "hsd/router_momentum_entropy": smooth_entropy[active_heads].mean(),
                })
                if include_null:
                    raw_null = raw[..., -1]
                    momentum_null = smooth[..., -1]
                    stats.update({
                        "hsd/router_raw_null_prob": raw_null[active_heads].mean(),
                        "hsd/router_momentum_null_prob": momentum_null[active_heads].mean(),
                        "hsd/router_raw_summary_mass": (1.0 - raw_null)[active_heads].mean(),
                        "hsd/router_momentum_summary_mass": (1.0 - momentum_null)[active_heads].mean(),
                    })
                else:
                    stats.update({
                        "hsd/router_raw_summary_mass": raw.sum(dim=-1)[active_heads].mean(),
                        "hsd/router_momentum_summary_mass": smooth.sum(dim=-1)[active_heads].mean(),
                    })
            if raw.shape[1] > 1:
                transition = active[:, 1:] & active[:, :-1] & ~reset_mask[:, 1:]
                if bool(transition.any().item()):
                    transition_heads = transition.unsqueeze(-1).expand(-1, -1, raw.shape[2])
                    raw_delta = (raw[:, 1:] - raw[:, :-1]).abs().mean(dim=-1)
                    smooth_delta = (smooth[:, 1:] - smooth[:, :-1]).abs().mean(dim=-1)
                    raw_switch = raw[:, 1:].argmax(dim=-1).ne(raw[:, :-1].argmax(dim=-1))
                    smooth_switch = smooth[:, 1:].argmax(dim=-1).ne(smooth[:, :-1].argmax(dim=-1))
                    stats.update({
                        "hsd/router_raw_weight_delta_l1": raw_delta[transition_heads].mean(),
                        "hsd/router_momentum_weight_delta_l1": smooth_delta[transition_heads].mean(),
                        "hsd/router_raw_top1_switch_rate": raw_switch[transition_heads].float().mean(),
                        "hsd/router_momentum_top1_switch_rate": smooth_switch[transition_heads].float().mean(),
                    })
            self._hsd_debug_stats.update(stats)

    def _record_hsd_router_debug_stats(
        self,
        weights: torch.Tensor,
        logits: torch.Tensor,
        summary_k: torch.Tensor,
        valid_prefix: torch.Tensor,
        has_valid: torch.Tensor,
        boundaries: torch.Tensor,
        token_index: torch.Tensor,
        null_prob: torch.Tensor | None = None,
        summary_mass: torch.Tensor | None = None,
        record_debug_stats: bool | None = None,
    ) -> None:
        if record_debug_stats is None:
            record_debug_stats = self._hsd_should_record_debug_stats()
        if not record_debug_stats:
            return
        with torch.no_grad():
            weights_f = weights.detach().float()
            valid_mask = has_valid.squeeze(-1).expand_as(weights_f[..., 0])
            if not bool(valid_mask.any().item()):
                return
            eps = torch.finfo(torch.float32).tiny
            extra_stats = {}
            summary_mass_f = None
            null_prob_f = None
            if null_prob is not None:
                null_prob_f = null_prob.detach().float()
                if null_prob_f.ndim == 4 and null_prob_f.shape[-1] == 1:
                    null_prob_f = null_prob_f.squeeze(-1)
                summary_mass_f = (
                    summary_mass.detach().float()
                    if summary_mass is not None else weights_f.sum(dim=-1, keepdim=True)
                )
                valid_null_prob = null_prob_f[valid_mask]
                valid_summary_mass = summary_mass_f.squeeze(-1)[valid_mask]
                extra_stats.update({
                    'hsd/router_null_prob': valid_null_prob.mean(),
                    'hsd/router_null_prob_max': valid_null_prob.max(),
                    'hsd/router_summary_mass': valid_summary_mass.mean(),
                    'hsd/router_history_head_share_gt_0p10': (valid_summary_mass > 0.10).float().mean(),
                })
                final_valid_mask = valid_mask[:, -1]
                if bool(final_valid_mask.any().item()):
                    extra_stats['hsd/router_null_prob_last_token'] = (
                        null_prob_f[:, -1][final_valid_mask].mean()
                    )
                token_valid_mask = valid_mask.any(dim=-1)
                if bool(token_valid_mask.any().item()):
                    summary_mass_bth = summary_mass_f.squeeze(-1)
                    valid_mask_f = valid_mask.float()
                    token_summary_mass_max = summary_mass_bth.masked_fill(~valid_mask, 0.0).max(dim=-1).values
                    token_summary_mass_mean = (
                        summary_mass_bth * valid_mask_f).sum(dim=-1) / valid_mask_f.sum(dim=-1).clamp_min(1.0)
                    extra_stats.update({
                        'hsd/router_history_token_share_any_head_gt_0p10': (
                            token_summary_mass_max[token_valid_mask] > 0.10).float().mean(),
                        'hsd/router_history_token_share_mean_head_gt_0p10': (
                            token_summary_mass_mean[token_valid_mask] > 0.10).float().mean(),
                    })
            summary_k_norm = torch.linalg.vector_norm(summary_k.detach().float(), dim=-1).mean(dim=(0, 2))
            if summary_k_norm.numel() > 0:
                extra_stats.update({
                    'hsd/router_summary_k_norm_oldest': summary_k_norm[0],
                    'hsd/router_summary_k_norm_latest': summary_k_norm[-1],
                    'hsd/router_summary_k_norm_max_over_mean': (
                        summary_k_norm.max() / summary_k_norm.mean().clamp_min(eps)
                    ),
                })
            logits_f = logits.detach().float()
            pre_weights = torch.softmax(torch.where(has_valid, logits_f, torch.zeros_like(logits_f)), dim=-1)
            pre_weights = torch.where(has_valid, pre_weights, torch.zeros_like(pre_weights))
            top4 = min(4, weights_f.shape[-1])
            top4_mass = pre_weights.topk(k=top4, dim=-1).values.sum(dim=-1)
            pre_entropy = -(pre_weights * pre_weights.clamp_min(eps).log()).sum(dim=-1)
            pre_top1_prob = pre_weights.max(dim=-1).values
            top2_logits = logits_f.topk(k=min(2, logits_f.shape[-1]), dim=-1).values
            if top2_logits.shape[-1] == 1:
                top1_margin = torch.zeros_like(top2_logits[..., 0])
            else:
                top1_margin = top2_logits[..., 0] - top2_logits[..., 1]
            weighted_boundary = (weights_f * boundaries.float().view(1, 1, 1, -1)).sum(dim=-1)
            weighted_age = token_index.float().view(1, -1, 1) - weighted_boundary
            entropy = -(weights_f * weights_f.clamp_min(eps).log()).sum(dim=-1)
            top1_prob = weights_f.max(dim=-1).values
            top1_idx = weights_f.argmax(dim=-1)
            if null_prob_f is not None:
                summary_top1_prob = top1_prob
                if summary_mass_f is not None:
                    summary_top1_prob = summary_top1_prob * summary_mass_f.squeeze(-1)
                null_top1 = null_prob_f >= summary_top1_prob
                extra_stats['hsd/router_null_top1_share'] = null_top1[valid_mask].float().mean()
            valid_entropy = entropy[valid_mask]
            valid_top1_prob = top1_prob[valid_mask]
            valid_top4_mass = top4_mass[valid_mask]
            valid_pre_entropy = pre_entropy[valid_mask]
            valid_pre_top1_prob = pre_top1_prob[valid_mask]
            valid_weighted_age = weighted_age[valid_mask]
            valid_counts_per_token = valid_prefix.sum(dim=-1)
            valid_counts = valid_counts_per_token.float()
            margin_mask = valid_mask & (valid_counts.to(device=valid_mask.device).view(1, -1, 1) >= 2)
            valid_counts = valid_counts[valid_counts > 0]
            stats = {
                'hsd/router_entropy': valid_entropy.mean(),
                'hsd/router_top1_prob': valid_top1_prob.mean(),
                'hsd/router_valid_summary_count': valid_counts.mean(),
                'hsd/router_effective_summary_count': valid_entropy.exp().mean(),
                'hsd/router_top4_mass': valid_top4_mass.mean(),
                'hsd/router_pre_topk_entropy': valid_pre_entropy.mean(),
                'hsd/router_pre_topk_effective_summary_count': valid_pre_entropy.exp().mean(),
                'hsd/router_pre_topk_top1_prob': valid_pre_top1_prob.mean(),
                'hsd/router_weighted_age': valid_weighted_age.mean(),
            }
            stats.update(extra_stats)
            if bool(margin_mask.any().item()):
                stats['hsd/router_top1_margin'] = top1_margin[margin_mask].mean()
            valid_top1_idx = top1_idx[valid_mask].long()
            valid_row_count = valid_top1_idx.numel()
            if valid_row_count > 0:
                row_count = weights_f.new_tensor(float(valid_row_count))
                top1_usage = torch.bincount(
                    valid_top1_idx,
                    minlength=weights_f.shape[-1],
                ).to(device=weights_f.device, dtype=weights_f.dtype) / row_count
                top1_usage_entropy = -(top1_usage * top1_usage.clamp_min(eps).log()).sum()
                mass_usage = weights_f[valid_mask].sum(dim=0) / row_count
                mass_usage_entropy = -(mass_usage * mass_usage.clamp_min(eps).log()).sum()
                bank_indices = torch.arange(
                    valid_prefix.shape[-1], device=valid_prefix.device, dtype=torch.long
                ).view(1, -1)
                valid_oldest_per_token = bank_indices.masked_fill(
                    ~valid_prefix, valid_prefix.shape[-1]
                ).amin(dim=-1)
                valid_latest_per_token = bank_indices.masked_fill(~valid_prefix, -1).amax(dim=-1)
                valid_oldest_idx = valid_oldest_per_token.view(1, -1, 1).expand_as(top1_idx)[valid_mask]
                valid_latest_idx = valid_latest_per_token.view(1, -1, 1).expand_as(top1_idx)[valid_mask]
                stats.update({
                    'hsd/router_top1_usage_effective_summary_count': top1_usage_entropy.exp(),
                    'hsd/router_top1_max_share': top1_usage.max(),
                    'hsd/router_mass_usage_effective_summary_count': mass_usage_entropy.exp(),
                    'hsd/router_mass_max_share': mass_usage.max(),
                    'hsd/router_top1_oldest_share': (valid_top1_idx == valid_oldest_idx).float().mean(),
                    'hsd/router_top1_latest_share': (valid_top1_idx == valid_latest_idx).float().mean(),
                })
                token_valid_mask = valid_mask.any(dim=-1)
                if bool(token_valid_mask.any().item()):
                    head_all_same = (top1_idx == top1_idx[..., :1]).all(dim=-1)
                    stats['hsd/router_head_all_same_share'] = head_all_same[token_valid_mask].float().mean()
                if top1_idx.shape[1] > 1:
                    switch_mask = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]
                    if bool(switch_mask.any().item()):
                        switched = top1_idx[:, 1:, :] != top1_idx[:, :-1, :]
                        stats['hsd/router_top1_switch_rate'] = switched[switch_mask].float().mean()
            self._hsd_debug_stats.update(stats)


class HSDGatedDeltaNetBlock(GradientCheckpointingLayer):

    def __init__(self, config: HSDGatedDeltaNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hsd_hybrid_summary_mode = getattr(
            config, "hsd_hybrid_summary_mode", "all_tokens"
        )
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                qkv_bias=config.attn['qkv_bias'],
                window_size=config.attn['window_size'],
                rope_theta=config.attn['rope_theta'],
                use_rope=config.attn.get('use_rope', True),
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx,
            )
        else:
            hsd_global_read_enabled = bool(config.hsd_use_global_read) and not _env_bool(
                "HSD_DISABLE_GLOBAL_READ", False
            )
            if config.hsd_read_layers is None:
                hsd_use_global_read = hsd_global_read_enabled
                hsd_collect_read_cache = hsd_global_read_enabled
            else:
                hsd_use_global_read = hsd_global_read_enabled and layer_idx in config.hsd_read_layers
                hsd_collect_read_cache = hsd_use_global_read
            self.attn = HSDGatedDeltaNet(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                num_v_heads=config.num_v_heads,
                use_gate=config.use_gate,
                use_short_conv=config.use_short_conv,
                allow_neg_eigval=config.allow_neg_eigval,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
                hsd_read_block_size=config.hsd_read_block_size,
                hsd_read_chunk_block_size=config.hsd_read_chunk_block_size,
                hsd_use_global_read=hsd_use_global_read,
                hsd_collect_read_cache=hsd_collect_read_cache,
                summary_chunk_size=config.summary_chunk_size,
                summary_token_num=config.summary_token_num,
                hsd_use_route_scale=getattr(config, "hsd_use_route_scale", True),
                hsd_route_logit_scale=getattr(config, "hsd_route_logit_scale", 1.0),
                hsd_route_head_dim=getattr(config, "hsd_route_head_dim", None),
                hsd_route_query_heads=getattr(config, "hsd_route_query_heads", None),
                hsd_route_key_heads=getattr(config, "hsd_route_key_heads", None),
                hsd_router_q_local_lora_rank=getattr(
                    config, "hsd_router_q_local_lora_rank", 0
                ),
                hsd_router_q_local_lora_source=getattr(
                    config, "hsd_router_q_local_lora_source", "local_q"
                ),
                hsd_router_q_local_lora_init_std=config.initializer_range,
                hsd_route_qk_l2norm=getattr(config, "hsd_route_qk_l2norm", True),
                hsd_route_position_encoding=getattr(config, "hsd_route_position_encoding", "none"),
                hsd_route_rope_dim=(
                    config.hsd_route_rope_dim
                    if getattr(config, "hsd_route_position_encoding", "none") == "rope" else 0
                ),
                hsd_read_topk=config.hsd_read_topk,
                hsd_topk_backend=getattr(config, "hsd_topk_backend", "reference"),
                hsd_route_weight_momentum_gamma=_env_float(
                    "HSD_ROUTE_WEIGHT_MOMENTUM_GAMMA",
                    getattr(config, "hsd_route_weight_momentum_gamma", 0.0),
                ),
                hsd_decode_momentum_init_mode=getattr(
                    config, "hsd_decode_momentum_init_mode", "carry"
                ),
                hsd_null_read_gate=getattr(config, "hsd_null_read_gate", False),
                hsd_null_read_gate_init_bias=getattr(config, "hsd_null_read_gate_init_bias", 0.0),
                hsd_null_logm_scale=getattr(config, "hsd_null_logm_scale", 0.0),
                hsd_history_mix_mode=getattr(config, "hsd_history_mix_mode", "additive"),
                hsd_history_gate_floor_init=getattr(
                    config, "hsd_history_gate_floor_init", 0.0
                ),
                hsd_history_gate_floor_warmup_steps=getattr(
                    config, "hsd_history_gate_floor_warmup_steps", 0
                ),
                hsd_history_head_scale=getattr(config, "hsd_history_head_scale", False),
                hsd_local_core=getattr(config, "hsd_local_core", "gdn"),
                hsd_kda_gate_rank=getattr(config, "hsd_kda_gate_rank", None),
                hsd_kda_gate_init_std=config.initializer_range,
                hsd_summary_q_mode=getattr(config, "hsd_summary_q_mode", "independent"),
                hsd_summary_q_lora_rank=getattr(config, "hsd_summary_q_lora_rank", 64),
                hsd_summary_q_lora_init_std=config.initializer_range,
                hsd_history_read_q_mode=getattr(
                    config, "hsd_history_read_q_mode", "shared_local"
                ),
                hsd_history_read_q_lora_rank=getattr(
                    config, "hsd_history_read_q_lora_rank", 32
                ),
                hsd_history_read_q_lora_init_std=config.initializer_range,
                hsd_summary_key_source=getattr(
                    config, "hsd_summary_key_source", "summary_hidden"
                ),
                hsd_summary_key_content_mode=getattr(
                    config, "hsd_summary_key_content_mode", "none"
                ),
                hsd_summary_key_content_scale_init=getattr(
                    config, "hsd_summary_key_content_scale_init", 0.0
                ),
                hsd_state_decay_mode=getattr(config, "hsd_state_decay_mode", "gated"),
                hsd_history_state_mode=getattr(config, "hsd_history_state_mode", "continuous"),
                hsd_history_candidate_mode=getattr(config, "hsd_history_candidate_mode", "snapshot"),
                hsd_chunk_beta_mode=getattr(config, "hsd_chunk_beta_mode", "shared"),
                hsd_train_selective_scan=getattr(config, "hsd_train_selective_scan", True),
                hsd_train_scan_backend=getattr(config, "hsd_train_scan_backend", "triton"),
                hsd_train_selective_scan_packed=getattr(config, "hsd_train_selective_scan_packed", None),
                hsd_train_selective_scan_packed_strict=getattr(
                    config, "hsd_train_selective_scan_packed_strict", False
                ),
            )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedDeltaNetMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        summary_mask = kwargs.get("summary_mask")
        text_only_hybrid_attention = (
            isinstance(self.attn, Attention)
            and self.hsd_hybrid_summary_mode == "text_only"
            and summary_mask is not None
            and bool(summary_mask.any().item())
        )
        if text_only_hybrid_attention:
            text_hidden_states, text_mask, text_kwargs = (
                prepare_hsd_hybrid_text_attention_inputs(
                    hidden_states=hidden_states,
                    summary_mask=summary_mask,
                    kwargs=kwargs,
                )
            )
            attn_outputs = self.attn(
                hidden_states=text_hidden_states,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **text_kwargs,
            )
            hidden_states = scatter_hsd_hybrid_text_attention_output(
                text_output=attn_outputs[0],
                text_mask=text_mask,
            )
            attn_outputs = (hidden_states, *attn_outputs[1:])
        else:
            attention_mask_for_attn = attention_mask
            attn_kwargs = kwargs
            if isinstance(self.attn, Attention):
                if self.hsd_hybrid_summary_mode == "all_tokens":
                    # HF generation supplies ordinary text position_ids.  After
                    # summary insertion, summary positions may also be zero, so
                    # Attention.get_packed_cu_seqlens() would mistake every
                    # summary for a packed-document boundary.  Packed HSD inputs
                    # already carry explicit expanded cu_seqlens; keep those and
                    # prevent the ambiguous expanded position_ids from being
                    # used as an implicit packed layout.
                    attn_kwargs = dict(kwargs)
                    attn_kwargs.pop("position_ids", None)
                    attn_kwargs.pop("text_position_ids", None)
                all_tokens_hybrid_cache = (
                    self.hsd_hybrid_summary_mode == "all_tokens"
                    and bool(use_cache)
                )
                if all_tokens_hybrid_cache:
                    if attention_mask is not None and (
                        attention_mask.ndim != 2
                        or not bool(attention_mask.all().item())
                    ):
                        raise NotImplementedError(
                            "Cached all-token HSD hybrid attention does not support "
                            "padded batches."
                        )
                    # Generation keeps a text-length attention mask, while the
                    # Full Attention KV cache also contains inserted summary
                    # tokens.  An all-ones mask is redundant and has the wrong
                    # length after the first summary boundary, so let the
                    # causal attention kernel infer the expanded KV length.
                    attention_mask_for_attn = None
                else:
                    attention_mask_for_attn = prepare_hsd_hybrid_attention_mask(
                        attention_mask=attention_mask,
                        hidden_states=hidden_states,
                        kwargs=kwargs,
                    )
            attn_outputs = self.attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask_for_attn,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **attn_kwargs,
            )
        hidden_states, attentions, past_key_values = attn_outputs[:3]
        aux_outputs = attn_outputs[3:]
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        return (hidden_states, attentions, past_key_values, *aux_outputs)


class HSDGatedDeltaNetPreTrainedModel(PreTrainedModel):

    config_class = HSDGatedDeltaNetConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['HSDGatedDeltaNetBlock']
    _supports_cache_class = True

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, GatedDeltaNet) and next(module.parameters()).device.type != 'meta':
            with torch.no_grad():
                if not getattr(module.A_log, '_is_hf_initialized', False):
                    a_min = 1.0 if getattr(module, "hsd_local_core", "gdn") == "gdn2" else 0.0
                    module.A_log.copy_(nn.init.uniform_(module.A_log, a=a_min, b=16).log())
                module.A_log._no_weight_decay = True
                if module.dt_bias is None:
                    reservation_shape = getattr(module, "_hsd_gdn2_scalar_dt_rng_shape", None)
                    if reservation_shape is not None:
                        # Reserve the exact canonical scalar-dt RNG draw after
                        # A_log, while keeping the legacy parameter absent.
                        nn.init.uniform_(
                            torch.empty(
                                reservation_shape,
                                device=module.A_log.device,
                                dtype=module.A_log.dtype,
                            )
                        )
                else:
                    if not getattr(module.dt_bias, '_is_hf_initialized', False):
                        dt = torch.exp(
                            nn.init.uniform_(module.dt_bias) * (math.log(0.1) - math.log(0.001)) + math.log(0.001),
                        ).clamp(min=1e-4)
                        inv_dt = dt + torch.log(-torch.expm1(-dt))
                        module.dt_bias.copy_(inv_dt)
                    module.dt_bias._no_weight_decay = True
        elif isinstance(module, _HSDInitRNGReservation):
            if module._device_anchor.device.type != "meta":
                reservation = torch.empty(
                    module.shape,
                    dtype=module._device_anchor.dtype,
                    device=module._device_anchor.device,
                )
                self._init_common_weight(reservation)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            if getattr(module, '_hsd_gdn2_initialized', False):
                # Official Xavier initialization was performed in an isolated
                # RNG scope by the owning HSD-GDN2 layer.
                pass
            elif getattr(module, '_hsd_state_route_initialized', False):
                # The state-aware route projection was initialized in an
                # isolated RNG scope so common seeded parameters stay exact.
                pass
            elif getattr(module, '_hsd_kda_gate_initialized', False):
                # Initialized explicitly in HSDGatedDeltaNet under an isolated
                # RNG scope, preserving all common GDN/HSD seeded parameters.
                pass
            elif getattr(module, '_hsd_copy_beta_init', False):
                # The owning HSDGatedDeltaNet copies b_proj after child
                # initialization. Do not consume RNG here, otherwise adding the
                # optional projection changes all later common weights.
                pass
            elif getattr(module, '_hsd_grouped_key_base_rows', None) is not None:
                base_rows = int(module._hsd_grouped_key_base_rows)
                self._init_common_weight(module.weight[:base_rows])
                weight_device = module.weight.device
                fork_devices = (
                    [weight_device.index if weight_device.index is not None else torch.cuda.current_device()]
                    if weight_device.type == "cuda" else []
                )
                with torch.random.fork_rng(devices=fork_devices):
                    self._init_common_weight(module.weight[base_rows:])
            elif getattr(module, '_hsd_zero_init', False):
                nn.init.zeros_(module.weight)
            else:
                self._init_common_weight(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            self._init_common_weight(module.weight)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                if prenorm_residual_strategy == 'rescale':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")

    def _init_common_weight(self, tensor: torch.Tensor) -> None:
        if getattr(self.config, "initializer_type", "normal") == "truncated_normal":
            cutoff = float(self.config.initializer_range) * float(
                getattr(self.config, "initializer_cutoff_factor", 3.0)
            )
            nn.init.trunc_normal_(
                tensor,
                mean=0.0,
                std=float(self.config.initializer_range),
                a=-cutoff,
                b=cutoff,
            )
        else:
            nn.init.normal_(tensor, mean=0.0, std=float(self.config.initializer_range))


class HSDGatedDeltaNetModel(HSDGatedDeltaNetPreTrainedModel):

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        if all_submodels:
            return super().get_expanded_tied_weights_keys(all_submodels=True)
        shared_router_ties = getattr(self, "_hsd_shared_router_tied_weights", None)
        if shared_router_ties is not None:
            return shared_router_ties.copy()
        return super().get_expanded_tied_weights_keys(all_submodels=False)

    def _configure_shared_router_ties(self) -> None:
        self._hsd_shared_router_tied_weights = {}
        self._tied_weights_keys = {}
        self.hsd_shared_router_layer_indices = ()
        if not bool(getattr(self.config, "hsd_share_router_across_read_layers", False)):
            return

        router_layers = []
        for layer in self.layers:
            attn = layer.attn
            if (
                isinstance(attn, HSDGatedDeltaNet)
                and attn.router_q_proj is not None
                and attn.summary_k_proj is not None
            ):
                router_layers.append(layer)
        if not router_layers:
            # HSD_DISABLE_GLOBAL_READ is an intentional runtime ablation. It
            # removes the router modules entirely, so there is nothing to tie.
            return
        if len(router_layers) < 2:
            raise ValueError(
                "hsd_share_router_across_read_layers=True requires at least two "
                "constructed HSD read layers."
            )

        source_layer = router_layers[0]
        source_attn = source_layer.attn
        assert isinstance(source_attn, HSDGatedDeltaNet)
        source_q_shape = source_attn.router_q_proj.weight.shape
        source_k_shape = source_attn.summary_k_proj.weight.shape
        source_uses_state_key = source_attn.hsd_summary_key_source == "state_probe"
        if source_uses_state_key and (
            source_attn.summary_state_value_probe is None
            or source_attn.summary_state_k_proj is None
        ):
            raise ValueError("Shared HSD state route-key parameters are missing.")
        source_prefix = f"layers.{source_layer.layer_idx}.attn"
        ties = {}
        for target_layer in router_layers[1:]:
            target_attn = target_layer.attn
            assert isinstance(target_attn, HSDGatedDeltaNet)
            if target_attn.router_q_proj.weight.shape != source_q_shape:
                raise ValueError(
                    "Shared HSD route-Q projections must have identical shapes; "
                    f"layer {source_layer.layer_idx} has {tuple(source_q_shape)} "
                    f"but layer {target_layer.layer_idx} has "
                    f"{tuple(target_attn.router_q_proj.weight.shape)}."
                )
            if target_attn.summary_k_proj.weight.shape != source_k_shape:
                raise ValueError(
                    "Shared HSD route-K projections must have identical shapes; "
                    f"layer {source_layer.layer_idx} has {tuple(source_k_shape)} "
                    f"but layer {target_layer.layer_idx} has "
                    f"{tuple(target_attn.summary_k_proj.weight.shape)}."
                )
            if (target_attn.hsd_summary_key_source == "state_probe") != source_uses_state_key:
                raise ValueError("Shared HSD routers must use the same summary-key source.")
            target_prefix = f"layers.{target_layer.layer_idx}.attn"
            ties[f"{target_prefix}.router_q_proj.weight"] = (
                f"{source_prefix}.router_q_proj.weight"
            )
            if source_attn.hsd_router_q_local_lora_rank > 0:
                if (
                    source_attn.router_q_local_lora_down_weight is None
                    or source_attn.router_q_local_lora_up_weight is None
                    or target_attn.router_q_local_lora_down_weight is None
                    or target_attn.router_q_local_lora_up_weight is None
                ):
                    raise ValueError("Shared HSD router local-Q LoRA parameters are missing.")
                if (
                    target_attn.router_q_local_lora_down_weight.shape
                    != source_attn.router_q_local_lora_down_weight.shape
                    or target_attn.router_q_local_lora_up_weight.shape
                    != source_attn.router_q_local_lora_up_weight.shape
                ):
                    raise ValueError("Shared HSD router local-Q LoRA parameters must have identical shapes.")
                ties[f"{target_prefix}.router_q_local_lora_down_weight"] = (
                    f"{source_prefix}.router_q_local_lora_down_weight"
                )
                ties[f"{target_prefix}.router_q_local_lora_up_weight"] = (
                    f"{source_prefix}.router_q_local_lora_up_weight"
                )
            ties[f"{target_prefix}.summary_k_proj.weight"] = (
                f"{source_prefix}.summary_k_proj.weight"
            )
            if source_uses_state_key:
                if (
                    target_attn.summary_state_value_probe is None
                    or target_attn.summary_state_k_proj is None
                    or target_attn.summary_state_value_probe.shape
                    != source_attn.summary_state_value_probe.shape
                    or target_attn.summary_state_k_proj.weight.shape
                    != source_attn.summary_state_k_proj.weight.shape
                ):
                    raise ValueError(
                        "Shared HSD state route-key parameters must have identical shapes."
                    )
                ties[f"{target_prefix}.summary_state_value_probe"] = (
                    f"{source_prefix}.summary_state_value_probe"
                )
                ties[f"{target_prefix}.summary_state_k_proj.weight"] = (
                    f"{source_prefix}.summary_state_k_proj.weight"
                )

        # HF post_init() consumes the ordinary per-layer initialization stream
        # first and then applies these ties. Consequently all non-router
        # parameters remain seed-compatible with the independent-router control.
        self._hsd_shared_router_tied_weights = ties
        self._tied_weights_keys = ties.copy()
        self.hsd_shared_router_layer_indices = tuple(
            layer.layer_idx for layer in router_layers
        )

    def __init__(self, config: HSDGatedDeltaNetConfig):
        max_position_embeddings_override = os.environ.get(
            "HSD_MAX_POSITION_EMBEDDINGS", ""
        ).strip()
        if max_position_embeddings_override:
            try:
                max_position_embeddings = int(max_position_embeddings_override)
            except ValueError as exc:
                raise ValueError(
                    "HSD_MAX_POSITION_EMBEDDINGS must be a positive integer."
                ) from exc
            if max_position_embeddings <= 0:
                raise ValueError(
                    "HSD_MAX_POSITION_EMBEDDINGS must be a positive integer."
                )
            # This is a runtime capacity override, not a checkpoint mutation.
            # Apply it before constructing either HSD or hybrid-attention layers
            # so summary-length validation and any RoPE caches agree.
            config.max_position_embeddings = max_position_embeddings
        train_scan_backend = os.environ.get(
            "HSD_TRAIN_SCAN_BACKEND",
            getattr(config, "hsd_train_scan_backend", "triton"),
        ).strip().lower()
        if train_scan_backend not in {"triton", "flashqla_tile"}:
            raise ValueError(
                "HSD_TRAIN_SCAN_BACKEND must be 'triton' or 'flashqla_tile'."
            )
        config.hsd_train_scan_backend = train_scan_backend
        read_topk_override = os.environ.get("HSD_READ_TOPK", "").strip()
        topk_backend_override = os.environ.get("HSD_TOPK_BACKEND", "").strip()
        try:
            read_topk = int(
                read_topk_override
                if read_topk_override
                else getattr(config, "hsd_read_topk", 0)
            )
        except ValueError as exc:
            raise ValueError("HSD_READ_TOPK must be a non-negative integer.") from exc
        topk_backend = (
            topk_backend_override
            or getattr(config, "hsd_topk_backend", "reference")
        ).lower()
        if read_topk < 0:
            raise ValueError("HSD_READ_TOPK must be a non-negative integer.")
        if topk_backend not in {"reference", "sparse"}:
            raise ValueError("HSD_TOPK_BACKEND must be 'reference' or 'sparse'.")
        if topk_backend == "sparse" and read_topk not in {0, 1, 2, 4, 8, 16}:
            raise ValueError(
                "Sparse HSD top-k requires HSD_READ_TOPK in "
                "{1, 2, 4, 8, 16} (or 0 when dormant)."
            )
        config.hsd_read_topk = read_topk
        config.hsd_topk_backend = topk_backend
        history_mix_override = os.environ.get("HSD_HISTORY_MIX_MODE", "").strip()
        history_mix_mode = (
            history_mix_override or getattr(config, "hsd_history_mix_mode", "additive")
        ).strip().lower()
        if history_mix_mode not in {"additive", "convex", "joint_softmax_logm", "gated_additive"}:
            raise ValueError(
                "HSD_HISTORY_MIX_MODE must be 'additive', 'convex', "
                "'joint_softmax_logm', or 'gated_additive'."
            )
        if history_mix_mode in {"convex", "joint_softmax_logm", "gated_additive"} and not bool(
            getattr(config, "hsd_null_read_gate", False)
        ):
            raise ValueError(f"HSD_HISTORY_MIX_MODE={history_mix_mode} requires hsd_null_read_gate=True.")
        config.hsd_history_mix_mode = history_mix_mode
        null_logm_scale = _env_float(
            "HSD_NULL_LOGM_SCALE",
            getattr(config, "hsd_null_logm_scale", 0.0),
        )
        if not math.isfinite(null_logm_scale) or null_logm_scale < 0.0:
            raise ValueError("HSD_NULL_LOGM_SCALE must be a non-negative finite value.")
        if null_logm_scale > 0.0:
            if not bool(getattr(config, "hsd_null_read_gate", False)):
                raise ValueError("HSD_NULL_LOGM_SCALE > 0 requires hsd_null_read_gate=True.")
            if history_mix_mode != "additive":
                raise ValueError(
                    "HSD_NULL_LOGM_SCALE > 0 currently requires "
                    "HSD_HISTORY_MIX_MODE=additive."
                )
            if read_topk != 0:
                raise ValueError("HSD_NULL_LOGM_SCALE > 0 currently requires HSD_READ_TOPK=0.")
        config.hsd_null_logm_scale = null_logm_scale
        history_aux_loss_weight = _env_float(
            "HSD_HISTORY_AUX_LOSS_WEIGHT",
            getattr(config, "hsd_history_aux_loss_weight", 0.0),
        )
        if not math.isfinite(history_aux_loss_weight) or history_aux_loss_weight < 0.0:
            raise ValueError(
                "HSD_HISTORY_AUX_LOSS_WEIGHT must be a non-negative finite value."
            )
        if history_aux_loss_weight > 0.0:
            if not bool(getattr(config, "hsd_use_global_read", False)):
                raise ValueError(
                    "A positive HSD_HISTORY_AUX_LOSS_WEIGHT requires hsd_use_global_read=True."
                )
            if history_mix_mode != "gated_additive" or not bool(
                getattr(config, "hsd_null_read_gate", False)
            ):
                raise ValueError(
                    "A positive HSD_HISTORY_AUX_LOSS_WEIGHT requires "
                    "HSD_HISTORY_MIX_MODE=gated_additive and hsd_null_read_gate=True."
                )
        config.hsd_history_aux_loss_weight = history_aux_loss_weight
        state_decay_override = os.environ.get("HSD_STATE_DECAY_MODE", "").strip()
        state_decay_mode = (
            state_decay_override or getattr(config, "hsd_state_decay_mode", "gated")
        ).strip().lower()
        if state_decay_mode not in {"gated", "none"}:
            raise ValueError("HSD_STATE_DECAY_MODE must be 'gated' or 'none'.")
        config.hsd_state_decay_mode = state_decay_mode
        history_state_override = os.environ.get("HSD_HISTORY_STATE_MODE", "").strip()
        history_state_mode = (
            history_state_override or getattr(config, "hsd_history_state_mode", "continuous")
        ).strip().lower()
        if history_state_mode not in {"continuous", "chunk_reset_no_decay"}:
            raise ValueError(
                "HSD_HISTORY_STATE_MODE must be 'continuous' or 'chunk_reset_no_decay'."
            )
        config.hsd_history_state_mode = history_state_mode
        history_candidate_override = os.environ.get(
            "HSD_HISTORY_CANDIDATE_MODE", ""
        ).strip()
        history_candidate_mode = (
            history_candidate_override
            or getattr(config, "hsd_history_candidate_mode", "snapshot")
        ).strip().lower()
        if history_candidate_mode not in {"snapshot", "delta_from_continuous"}:
            raise ValueError(
                "HSD_HISTORY_CANDIDATE_MODE must be 'snapshot' or 'delta_from_continuous'."
            )
        if (
            history_candidate_mode == "delta_from_continuous"
            and history_state_mode != "continuous"
        ):
            raise ValueError(
                "HSD_HISTORY_CANDIDATE_MODE='delta_from_continuous' requires "
                "HSD_HISTORY_STATE_MODE='continuous'."
            )
        config.hsd_history_candidate_mode = history_candidate_mode
        chunk_beta_override = os.environ.get("HSD_CHUNK_BETA_MODE", "").strip()
        chunk_beta_mode = (
            chunk_beta_override or getattr(config, "hsd_chunk_beta_mode", "shared")
        ).strip().lower()
        if chunk_beta_mode not in {"shared", "independent"}:
            raise ValueError("HSD_CHUNK_BETA_MODE must be 'shared' or 'independent'.")
        if chunk_beta_mode == "independent" and history_state_mode != "chunk_reset_no_decay":
            raise ValueError(
                "HSD_CHUNK_BETA_MODE='independent' requires "
                "HSD_HISTORY_STATE_MODE='chunk_reset_no_decay'."
            )
        config.hsd_chunk_beta_mode = chunk_beta_mode
        decode_momentum_init_override = os.environ.get(
            "HSD_DECODE_MOMENTUM_INIT_MODE", ""
        ).strip()
        decode_momentum_init_mode = (
            decode_momentum_init_override
            or getattr(config, "hsd_decode_momentum_init_mode", "carry")
        ).strip().lower()
        if decode_momentum_init_mode not in {"carry", "last_prompt_raw"}:
            raise ValueError(
                "HSD_DECODE_MOMENTUM_INIT_MODE must be either 'carry' or 'last_prompt_raw'."
            )
        config.hsd_decode_momentum_init_mode = decode_momentum_init_mode
        route_position_encoding = os.environ.get(
            "HSD_ROUTE_POSITION_ENCODING",
            getattr(config, "hsd_route_position_encoding", "none"),
        ).strip().lower()
        route_position_mode = os.environ.get(
            "HSD_ROUTE_POSITION_MODE",
            getattr(config, "hsd_route_position_mode", "chunk_end"),
        ).strip().lower()
        route_rope_fraction = _env_float(
            "HSD_ROUTE_ROPE_FRACTION",
            getattr(config, "hsd_route_rope_fraction", 0.5),
        )
        route_rope_theta = _env_float(
            "HSD_ROUTE_ROPE_THETA",
            getattr(config, "hsd_route_rope_theta", 10000.0),
        )
        config.hsd_route_position_encoding = route_position_encoding
        config.hsd_route_position_mode = route_position_mode
        config.hsd_route_rope_fraction = route_rope_fraction
        config.hsd_route_rope_theta = route_rope_theta
        config._validate_hsd_route_position()
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.summary_embedding = None
        self.summary_embedding_init_rng_reservation = None
        if bool(getattr(config, "summary_enabled", False)):
            self.summary_embedding = nn.Embedding(
                int(config.summary_token_num),
                config.hidden_size,
            )
        else:
            # Match the constructor and post_init RNG consumption of the
            # one-row embedding used by the corresponding HSD configuration,
            # without storing an unused baseline parameter.
            summary_embedding_reference = nn.Embedding(1, config.hidden_size)
            reference_device = summary_embedding_reference.weight.device
            reference_dtype = summary_embedding_reference.weight.dtype
            del summary_embedding_reference
            self.summary_embedding_init_rng_reservation = _HSDInitRNGReservation(
                (1, config.hidden_size),
                dtype=reference_dtype,
                device=reference_device,
            )
        self.route_rotary_cache = (
            HSDRouteRotaryCache(config.hsd_route_rope_dim, config.hsd_route_rope_theta)
            if config.hsd_route_position_encoding == "rope" else None
        )
        self.layers = nn.ModuleList([HSDGatedDeltaNetBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._configure_shared_router_ties()
        history_aux_layers = [
            layer.layer_idx
            for layer in self.layers
            if isinstance(layer.attn, HSDGatedDeltaNet) and layer.attn.hsd_use_global_read
        ]
        self.hsd_history_aux_layer_idx = (
            max(history_aux_layers) if history_aux_loss_weight > 0.0 and history_aux_layers else None
        )
        if history_aux_loss_weight > 0.0 and self.hsd_history_aux_layer_idx is None:
            raise ValueError(
                "History-only auxiliary loss requires at least one HSD layer with global read enabled."
            )
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self._warned_auto_summary_insert = False
        self._warned_missing_summary_mask = False
        self.post_init()
        # The chunk projection deliberately skips the normal Linear
        # initialization so adding it does not perturb the common RNG stream.
        # Copy only while constructing the fresh nested model, before
        # from_pretrained() loads checkpoint tensors. Copying from
        # _init_weights is unsafe because HF may call it again while
        # initializing missing parent modules after checkpoint loading.
        with torch.no_grad():
            for layer in self.layers:
                attn = layer.attn
                if (
                    isinstance(attn, HSDGatedDeltaNet)
                    and attn.chunk_b_proj is not None
                    and attn.chunk_b_proj.weight.device.type != "meta"
                ):
                    attn.chunk_b_proj.weight.copy_(attn.b_proj.weight)

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def _prepare_route_rotary_sidecars(
        self,
        *,
        summary_mask: torch.Tensor | None,
        sequence_shape: torch.Size,
        past_seen_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if self.route_rotary_cache is None:
            return {}
        if summary_mask is None:
            summary_mask = torch.zeros(sequence_shape, device=device, dtype=torch.bool)
        if summary_mask.shape != sequence_shape:
            raise ValueError("summary_mask must match input sequence shape for route RoPE.")
        text_len = int((~summary_mask[0]).sum().item())
        query_positions = torch.arange(
            past_seen_tokens,
            past_seen_tokens + text_len,
            device=device,
            dtype=torch.long,
        )
        text_seen = (~summary_mask[0]).to(dtype=torch.long).cumsum(dim=0)
        key_positions = (text_seen[summary_mask[0]] + int(past_seen_tokens) - 1).clamp_min(0)
        query_cos, query_sin, key_cos, key_sin = self.route_rotary_cache(
            query_positions,
            key_positions,
            dtype=dtype,
        )
        return {
            "hsd_route_q_cos": query_cos,
            "hsd_route_q_sin": query_sin,
            "hsd_route_k_cos": key_cos,
            "hsd_route_k_sin": key_sin,
        }

    @staticmethod
    def _filter_summary_hidden(hidden_states: torch.Tensor, summary_mask: torch.Tensor | None) -> torch.Tensor:
        if summary_mask is None or not bool(summary_mask.any().item()):
            return hidden_states
        text_mask = ~summary_mask.to(device=hidden_states.device, dtype=torch.bool)
        if not torch.equal(text_mask.sum(dim=1), text_mask.sum(dim=1)[:1].expand(text_mask.shape[0])):
            raise NotImplementedError("HSD V1 requires the same summary layout for every batch row.")
        return hidden_states[text_mask].view(
            hidden_states.shape[0],
            int(text_mask[0].sum().item()),
            hidden_states.shape[-1],
        )

    def _get_hsd_text_cache_length(self, past_key_values: Cache | None) -> int:
        """Return the generated text length, independent of hybrid FA summaries."""
        if past_key_values is None:
            return 0
        for layer in self.layers:
            if isinstance(layer.attn, HSDGatedDeltaNet):
                return int(past_key_values.get_seq_length(layer.layer_idx))
        cached_length = int(past_key_values.get_seq_length())
        if getattr(self.config, "hsd_hybrid_summary_mode", "all_tokens") != "all_tokens":
            return cached_length
        chunk_size = int(self.config.summary_chunk_size)
        summary_token_num = int(self.config.summary_token_num)
        completed_chunks, within_chunk = divmod(
            cached_length,
            chunk_size + summary_token_num,
        )
        if within_chunk >= chunk_size:
            raise RuntimeError(
                "The all-token Full Attention cache length is inconsistent with "
                "the configured HSD summary layout."
            )
        return completed_chunks * chunk_size + within_chunk

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        summary_mask: torch.Tensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple | BaseModelOutputWithPast:
        output_hsd_aux_history = bool(kwargs.pop("output_hsd_aux_history", False))
        if output_hsd_aux_history and self.hsd_history_aux_layer_idx is None:
            raise RuntimeError(
                "output_hsd_aux_history=True requires a configured history auxiliary layer."
            )
        if output_attentions:
            warnings.warn("`HSDGatedDeltaNetModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if past_key_values is not None:
            use_cache = True
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        past_seen_tokens = self._get_hsd_text_cache_length(past_key_values)
        is_prefill = past_key_values is None or past_seen_tokens == 0
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if summary_mask is None and getattr(self.config, "summary_enabled", False) and is_prefill:
            if inputs_embeds is None and input_ids is not None:
                hsd_batch = {"input_ids": input_ids}
                if attention_mask is not None:
                    hsd_batch["attention_mask"] = attention_mask
                if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                    hsd_batch["position_ids"] = kwargs["position_ids"]
                hsd_batch, hsd_ctx = maybe_build_hsd_summary_batch(hsd_batch, self.config)
                input_ids = hsd_batch["input_ids"]
                attention_mask = hsd_batch.get("attention_mask", attention_mask)
                summary_mask = hsd_batch.get("summary_mask")
                for hsd_sidecar_name in ("text_cu_seqlens", "summary_cu_seqlens", "cu_seqlens"):
                    if hsd_sidecar_name in hsd_batch:
                        kwargs[hsd_sidecar_name] = hsd_batch[hsd_sidecar_name]
                if "position_ids" in hsd_batch:
                    kwargs["position_ids"] = hsd_batch["position_ids"]
                if hsd_ctx is not None and hsd_ctx.enabled and not self._warned_auto_summary_insert:
                    logger.warning(
                        "HSD summary_mask was not provided; automatically inserted summary tokens for this model."
                    )
                    self._warned_auto_summary_insert = True
            elif not self._warned_missing_summary_mask:
                logger.warning(
                    "HSD summary is enabled, but summary_mask was not provided and inputs_embeds cannot be "
                    "expanded automatically; falling back to the base GDN path."
                )
                self._warned_missing_summary_mask = True
        if summary_mask is not None:
            summary_mask = summary_mask.to(dtype=torch.bool, device=input_ids.device if input_ids is not None else inputs_embeds.device)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        if summary_mask is not None and bool(summary_mask.any().item()):
            if summary_mask.shape != inputs_embeds.shape[:2]:
                raise ValueError("summary_mask must match input sequence shape.")
            summary_embedding_type = getattr(self.config, "summary_embedding_type", "learned")
            summary_replacements = torch.zeros_like(inputs_embeds)
            if summary_embedding_type == "learned":
                if self.summary_embedding is None:
                    raise RuntimeError(
                        "Learned HSD summary embeddings are unavailable because "
                        "summary insertion is disabled in the model config."
                    )
                summary_ids = build_summary_embedding_ids(summary_mask, self.config.summary_token_num)
                summary_replacements[summary_mask] = self.summary_embedding(summary_ids).repeat(inputs_embeds.shape[0], 1)
            elif summary_embedding_type == "chunk_end":
                anchor_indices = build_summary_anchor_indices(summary_mask)
                anchor_embeds = inputs_embeds.gather(
                    1,
                    anchor_indices.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1]),
                )
                summary_replacements[summary_mask] = anchor_embeds.reshape(-1, inputs_embeds.shape[-1])
            else:
                raise ValueError(f"Unsupported summary_embedding_type: {summary_embedding_type}")
            inputs_embeds = torch.where(summary_mask.unsqueeze(-1), summary_replacements, inputs_embeds)
        kwargs.update(self._prepare_route_rotary_sidecars(
            summary_mask=summary_mask,
            sequence_shape=inputs_embeds.shape[:2],
            past_seen_tokens=int(past_seen_tokens),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ))
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        hsd_aux_history = None
        hsd_aux_history_available = None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            capture_aux_history = bool(
                output_hsd_aux_history
                and layer.layer_idx == self.hsd_history_aux_layer_idx
            )
            layer_kwargs = kwargs
            if capture_aux_history:
                layer_kwargs = {**kwargs, "return_hsd_aux_history": True}
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                summary_mask=summary_mask,
                **layer_kwargs,
            )
            hidden_states, attentions, past_key_values = layer_outputs[:3]
            if capture_aux_history:
                if len(layer_outputs) != 5:
                    raise RuntimeError(
                        "The selected HSD auxiliary layer did not return conditional history."
                    )
                hsd_aux_history, hsd_aux_history_available = layer_outputs[3:]
            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)
        hidden_states = self._filter_summary_hidden(hidden_states, summary_mask)
        if output_hidden_states:
            all_hidden_states = tuple(self._filter_summary_hidden(h, summary_mask) for h in all_hidden_states)
            all_hidden_states += (hidden_states,)

        if output_hsd_aux_history and (
            hsd_aux_history is None or hsd_aux_history_available is None
        ):
            raise RuntimeError("History-only auxiliary outputs were not captured.")
        if not return_dict:
            values = [hidden_states, past_key_values, all_hidden_states, all_attns]
            if output_hsd_aux_history:
                values.extend([hsd_aux_history, hsd_aux_history_available])
            return tuple(i for i in values if i is not None)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )
        if output_hsd_aux_history:
            output["hsd_aux_history"] = hsd_aux_history
            output["hsd_aux_history_available"] = hsd_aux_history_available
        return output


class HSDGatedDeltaNetForCausalLM(HSDGatedDeltaNetPreTrainedModel, FLAGenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = HSDGatedDeltaNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @staticmethod
    def _select_logits_positions(
        tensor: torch.Tensor,
        logits_to_keep: int | torch.Tensor | None,
        summary_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(logits_to_keep, torch.Tensor) and logits_to_keep.dtype == torch.bool:
            logits_mask = logits_to_keep.to(device=tensor.device)
            if logits_mask.ndim != 1:
                raise NotImplementedError("HSD logits_to_keep bool mask must be 1D.")
            if (
                summary_mask is not None
                and bool(summary_mask.any().item())
                and logits_mask.numel() == summary_mask.shape[1]
            ):
                text_mask = ~summary_mask[0].to(device=logits_mask.device, dtype=torch.bool)
                logits_mask = logits_mask[text_mask]
            if logits_mask.numel() != tensor.shape[1]:
                raise ValueError(
                    f"HSD logits_to_keep bool mask length {logits_mask.numel()} must match "
                    f"text tensor length {tensor.shape[1]}."
                )
            return tensor[:, logits_mask]
        if logits_to_keep is None:
            return tensor
        n_logits_to_keep = (
            int(logits_to_keep.item())
            if isinstance(logits_to_keep, torch.Tensor)
            else int(logits_to_keep)
        )
        return tensor[:, -n_logits_to_keep:]

    def _compute_history_aux_loss(
        self,
        *,
        history: torch.Tensor,
        history_available: torch.Tensor,
        labels: torch.LongTensor,
        ignore_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_idx = self.model.hsd_history_aux_layer_idx
        if layer_idx is None:
            raise RuntimeError("History auxiliary loss layer is not configured.")
        history_layer = self.model.layers[layer_idx].attn
        if not isinstance(history_layer, HSDGatedDeltaNet):
            raise RuntimeError("History auxiliary loss layer must be an HSDGatedDeltaNet layer.")
        if history.ndim != 4 or history.shape[2:] != (
            history_layer.num_v_heads,
            history_layer.head_v_dim,
        ):
            raise ValueError(
                "History auxiliary tensor must have shape [B, T, num_v_heads, head_v_dim]."
            )
        if history_available.shape[:2] != history.shape[:2]:
            raise ValueError("History availability mask must match the auxiliary history sequence.")
        if labels.shape != history.shape[:2]:
            raise ValueError("History auxiliary labels must match the auxiliary history sequence.")

        history_flat = history.reshape(history.shape[0], history.shape[1], -1)
        o_proj_bias = history_layer.o_proj.bias
        projected = F.linear(
            history_flat,
            history_layer.o_proj.weight.detach(),
            None if o_proj_bias is None else o_proj_bias.detach(),
        )
        rms = torch.rsqrt(
            projected.float().square().mean(dim=-1, keepdim=True)
            + float(self.config.norm_eps)
        )
        normalized = projected * rms.to(dtype=projected.dtype)

        available = history_available.to(device=labels.device, dtype=torch.bool)
        while available.ndim > 2:
            available = available.squeeze(-1)
        valid = available & labels.ne(ignore_index)
        valid_share = valid.float().mean()
        aux_labels = labels.masked_fill(~valid, ignore_index)
        if not bool(valid.any().item()):
            return history.sum() * 0.0, valid_share

        lm_bias = self.lm_head.bias
        detached_lm_bias = None if lm_bias is None else lm_bias.detach()
        if normalized.is_cuda:
            aux_criterion = FusedLinearCrossEntropyLoss(use_l2warp=False)
            aux_loss = aux_criterion(
                normalized,
                aux_labels,
                self.lm_head.weight.detach(),
                detached_lm_bias,
            )
        else:
            aux_logits = F.linear(
                normalized,
                self.lm_head.weight.detach(),
                detached_lm_bias,
            )
            aux_loss = F.cross_entropy(
                aux_logits.reshape(-1, aux_logits.shape[-1]),
                aux_labels.reshape(-1),
                ignore_index=ignore_index,
            )
        return aux_loss, valid_share

    @staticmethod
    def _filter_summary_hidden(hidden_states: torch.Tensor, summary_mask: torch.Tensor | None) -> torch.Tensor:
        if summary_mask is None or not bool(summary_mask.any().item()):
            return hidden_states
        text_mask = ~summary_mask.to(device=hidden_states.device, dtype=torch.bool)
        if not torch.equal(text_mask.sum(dim=1), text_mask.sum(dim=1)[:1].expand(text_mask.shape[0])):
            raise NotImplementedError("HSD V1 requires the same summary layout for every batch row.")
        return hidden_states[text_mask].view(hidden_states.shape[0], int(text_mask[0].sum().item()), hidden_states.shape[-1])

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool = True,
        logits_to_keep: int | None = None,
        **kwargs,
    ):
        if not getattr(self.config, "summary_enabled", False):
            return super().prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                cache_position=cache_position,
                **kwargs,
            )

        past_text_len = self.model._get_hsd_text_cache_length(past_key_values)
        is_prefill = past_key_values is None or past_text_len == 0
        if is_prefill:
            model_inputs = {
                "input_ids": None if inputs_embeds is not None else input_ids,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
            if position_ids is not None:
                model_inputs["position_ids"] = position_ids
            if logits_to_keep is not None:
                model_inputs["logits_to_keep"] = logits_to_keep
            return model_inputs

        if inputs_embeds is not None:
            inputs_embeds = None
        if input_ids is None:
            raise ValueError("HSD cache generation requires input_ids during decode.")
        if cache_position is not None:
            cache_position = cache_position[-1:]
        if input_ids.shape[1] > 1:
            new_token_count = input_ids.shape[1] - past_text_len
            if new_token_count <= 0:
                input_ids = input_ids[:, -1:]
            elif new_token_count == 1:
                input_ids = input_ids[:, -1:]
            else:
                raise NotImplementedError("HSD cache generation V1 supports one new decode token per step.")

        summary_mask = None
        summary_chunk_size = int(self.config.summary_chunk_size)
        summary_token_num = int(self.config.summary_token_num)
        if (past_text_len + 1) % summary_chunk_size == 0:
            batch_size = input_ids.shape[0]
            summary_ids = torch.full(
                (batch_size, summary_token_num),
                int(self.config.summary_placeholder_token_id),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, summary_ids], dim=1)
            summary_mask = torch.zeros(batch_size, 1 + summary_token_num, dtype=torch.bool, device=input_ids.device)
            summary_mask[:, 1:] = True
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
                summary_pos = torch.zeros(batch_size, summary_token_num, dtype=position_ids.dtype, device=position_ids.device)
                position_ids = torch.cat([position_ids, summary_pos], dim=1)
        elif position_ids is not None:
            position_ids = position_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids.contiguous(),
            "inputs_embeds": None,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
        }
        if position_ids is not None:
            model_inputs["position_ids"] = position_ids
        if summary_mask is not None:
            model_inputs["summary_mask"] = summary_mask
        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        summary_mask: torch.Tensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        history_aux_weight = float(
            getattr(self.config, "hsd_history_aux_loss_weight", 0.0)
        )
        compute_history_aux = bool(
            self.training and labels is not None and history_aux_weight > 0.0
        )
        model_return_dict = True if compute_history_aux else return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=model_return_dict,
            summary_mask=summary_mask,
            output_hsd_aux_history=compute_history_aux,
            **kwargs,
        )
        hidden_states = outputs[0]

        hidden_states_for_logits = self._select_logits_positions(
            hidden_states,
            logits_to_keep,
            summary_mask,
        )

        loss, logits = None, None
        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states_for_logits)
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            if (labels.shape[1] == hidden_states_for_logits.shape[1] + 1
                    and bool((labels[:, :1] == criterion.ignore_index).all().item())):
                labels = labels[:, 1:]
            if labels.shape[1] != hidden_states_for_logits.shape[1]:
                raise ValueError(
                    f"HSD labels length {labels.shape[1]} must match text hidden length {hidden_states_for_logits.shape[1]}."
                )
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states_for_logits, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss
            if compute_history_aux:
                aux_history = self._select_logits_positions(
                    outputs.hsd_aux_history,
                    logits_to_keep,
                    summary_mask,
                )
                aux_history_available = self._select_logits_positions(
                    outputs.hsd_aux_history_available,
                    logits_to_keep,
                    summary_mask,
                )
                history_aux_loss, history_aux_valid_share = self._compute_history_aux_loss(
                    history=aux_history,
                    history_available=aux_history_available,
                    labels=labels,
                    ignore_index=criterion.ignore_index,
                )
                weighted_history_aux_loss = history_aux_loss * history_aux_weight
                loss = loss + weighted_history_aux_loss
                history_aux_layer = self.model.layers[
                    self.model.hsd_history_aux_layer_idx
                ].attn
                if history_aux_layer._hsd_debug_enabled():
                    history_aux_layer._hsd_debug_stats.update({
                        "hsd/history_aux_loss": history_aux_loss.detach(),
                        "hsd/history_aux_weighted_loss": weighted_history_aux_loss.detach(),
                        "hsd/history_aux_valid_token_share": history_aux_valid_share.detach(),
                        "hsd/history_aux_layer_idx": history_aux_loss.new_tensor(
                            float(self.model.hsd_history_aux_layer_idx)
                        ),
                    })

        if not return_dict:
            if compute_history_aux:
                output = (logits,) + tuple(
                    value
                    for value in (
                        outputs.past_key_values,
                        outputs.hidden_states,
                        outputs.attentions,
                    )
                    if value is not None
                )
            else:
                output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        model_output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return model_output


__all__ = [
    "HSDGatedDeltaNet",
    "HSDGatedDeltaNetBlock",
    "HSDGatedDeltaNetForCausalLM",
    "HSDGatedDeltaNetModel",
    "HSDGatedDeltaNetPreTrainedModel",
    "HSDRouteRotaryCache",
]

AutoConfig.register(HSDGatedDeltaNetConfig.model_type, HSDGatedDeltaNetConfig, exist_ok=True)
AutoModel.register(HSDGatedDeltaNetConfig, HSDGatedDeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(HSDGatedDeltaNetConfig, HSDGatedDeltaNetForCausalLM, exist_ok=True)
