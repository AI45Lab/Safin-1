# Copyright (c) ModelScope Contributors. All rights reserved.
from march.safin1.runtime_validation import validate_safin1_training_runtime, validate_training_environment

import math
import os
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
import transformer_engine
from megatron.core import parallel_state
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (
    all_gather_last_dim_from_tensor_parallel_region,
    gather_from_sequence_parallel_region,
)
from megatron.core.transformer.spec_utils import build_module

from mcore_bridge.config import ModelConfig

from .march_diagnostics import MarchDiagnosticsMixin
from .march_runtime import (
    _env_int as _env_int,
    _env_float as _env_float,
    _env_optional_float as _env_optional_float,
    _env_bool as _env_bool,
    _env_str as _env_str,
    _rank_allowed_from_env as _rank_allowed_from_env,
    _precision_trace_internal_layer_allowed as _precision_trace_internal_layer_allowed,
    _parse_layer_selector as _parse_layer_selector,
)
from ..rope import dynamic_rope_update, get_rope_inv_freq
from .gated_delta_net import (
    GatedDeltaNet,
    causal_conv1d,
    chunk_gated_delta_rule,
    get_parameter_local_cp,
    l2norm,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)

if TYPE_CHECKING:
    from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules

try:
    from .march_chunk_gdn import hsd_chunk_gated_delta_rule_with_anchor_states
except Exception:  # pragma: no cover - optional Triton/FLA training fastpath.
    hsd_chunk_gated_delta_rule_with_anchor_states = None

try:
    from march.kernels.march_chunk_gdn import (
        hsd_flashqla_tile_gated_delta_rule_with_anchor_states,
    )
except Exception:  # pragma: no cover - optional Hopper/TileLang training fastpath.
    hsd_flashqla_tile_gated_delta_rule_with_anchor_states = None

try:
    from march.kernels.march_global_read import hsd_fused_global_read
except Exception:  # pragma: no cover - optional Triton/TileLang global-read fastpath.
    hsd_fused_global_read = None

try:
    from march.kernels.march_global_read_topk import hsd_sparse_topk_global_read
except Exception:  # pragma: no cover - optional Triton sparse Top-K fastpath.
    hsd_sparse_topk_global_read = None


_HSD_NONPACKED_SIDECAR_CACHE_MAX_ENTRIES = 16
_HSD_NONPACKED_SIDECAR_CACHE: dict[
    tuple,
    tuple[
        torch.Tensor,
        int,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
] = {}


def _all_gather_hsd_shared_projection(input_: torch.Tensor, *, group) -> torch.Tensor:
    """Gather a shared route projection and sum its backward TP contributions.

    ``anchor_k_proj`` is column-parallel even when the model owns one shared
    route-key head.  Every TP rank consumes the gathered K1 vector with its
    own value/query-head shard, so the gradient of each original projection
    shard is the sum of the reader losses from *all* TP ranks.  Megatron's
    legacy ``gather_from_tensor_model_parallel_region`` only splits the
    backward gradient and silently drops those cross-rank contributions.
    This all-gather uses reduce-scatter(sum) in backward, which is the exact
    adjoint required by the replicated consumer graph.
    """

    return all_gather_last_dim_from_tensor_parallel_region(input_, group=group)


def _route_softmax_weights(logits: torch.Tensor, valid: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if valid.dtype is not torch.bool:
        valid = valid.bool()

    weights = torch.softmax(logits.float().masked_fill(~valid, float('-inf')), dim=dim)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights * valid.to(dtype=weights.dtype)
    denom = weights.sum(dim=dim, keepdim=True)
    weights = torch.where(denom > 0, weights / denom, torch.zeros_like(weights))
    return weights.to(dtype=logits.dtype)


def _blocked_history_read_output(
    read_q: torch.Tensor,
    states: torch.Tensor,
    weights: torch.Tensor,
    *,
    read_scale: float,
    block_m: int,
    block_t: int,
) -> torch.Tensor:
    return _BlockedHistoryReadFunction.apply(read_q, states, weights, float(read_scale), int(block_m), int(block_t))


class _BlockedHistoryReadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        read_q: torch.Tensor,
        states: torch.Tensor,
        weights: torch.Tensor,
        read_scale: float,
        block_m: int,
        block_t: int,
    ) -> torch.Tensor:
        ctx.read_scale = float(read_scale)
        ctx.block_m = int(block_m)
        ctx.block_t = int(block_t)
        ctx.save_for_backward(read_q, states, weights)

        with torch.no_grad():
            return _blocked_history_read_forward_impl(
                read_q=read_q,
                states=states,
                weights=weights,
                read_scale=float(read_scale),
                block_m=int(block_m),
                block_t=int(block_t),
            )

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        read_q, states, weights = ctx.saved_tensors
        dread_q, dstates, dweights = _blocked_history_read_backward_impl(
            dout=dout,
            read_q=read_q,
            states=states,
            weights=weights,
            read_scale=ctx.read_scale,
            block_m=ctx.block_m,
            block_t=ctx.block_t,
        )
        return dread_q, dstates, dweights, None, None, None


def _blocked_history_read_forward_impl(
    *,
    read_q: torch.Tensor,
    states: torch.Tensor,
    weights: torch.Tensor,
    read_scale: float,
    block_m: int,
    block_t: int,
) -> torch.Tensor:
    batch, text_len, heads, _ = read_q.shape
    anchor_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    block_m = max(1, int(block_m))
    block_t = text_len if block_t <= 0 else max(1, int(block_t))
    read_q = read_q.to(states.dtype)
    out = states.new_zeros(batch, text_len, heads, value_dim)

    for t_start in range(0, text_len, block_t):
        t_stop = min(text_len, t_start + block_t)
        read_q_t = read_q[:, t_start:t_stop]
        weights_t = weights[:, t_start:t_stop]
        out_t = out[:, t_start:t_stop]
        for start in range(0, anchor_count, block_m):
            stop = min(anchor_count, start + block_m)
            reads = torch.einsum('bthk,bmhkv->bthmv', read_q_t, states[:, start:stop]) * read_scale
            out_t.add_(torch.einsum('bthm,bthmv->bthv', weights_t[..., start:stop].to(reads.dtype), reads))

    return out


def _blocked_history_read_backward_impl(
    *,
    dout: torch.Tensor,
    read_q: torch.Tensor,
    states: torch.Tensor,
    weights: torch.Tensor,
    read_scale: float,
    block_m: int,
    block_t: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, text_len, _, _ = read_q.shape
    anchor_count = int(states.shape[1])
    block_m = max(1, int(block_m))
    block_t = text_len if block_t <= 0 else max(1, int(block_t))

    read_q_f = read_q.to(states.dtype)
    dout_f = dout.to(states.dtype)
    weights_f = weights.to(states.dtype)
    dread_q = torch.zeros_like(read_q_f)
    dstates = torch.zeros_like(states)
    dweights = torch.zeros_like(weights_f)

    for t_start in range(0, text_len, block_t):
        t_stop = min(text_len, t_start + block_t)
        read_q_t = read_q_f[:, t_start:t_stop]
        dout_t = dout_f[:, t_start:t_stop]
        weights_t = weights_f[:, t_start:t_stop]
        dread_q_t = dread_q[:, t_start:t_stop]
        dweights_t = dweights[:, t_start:t_stop]
        for start in range(0, anchor_count, block_m):
            stop = min(anchor_count, start + block_m)
            states_m = states[:, start:stop]
            weights_m = weights_t[..., start:stop]
            reads = torch.einsum('bthk,bmhkv->bthmv', read_q_t, states_m) * read_scale
            dweights_t[..., start:stop].add_(torch.einsum('bthv,bthmv->bthm', dout_t, reads))
            dread_q_t.add_(
                torch.einsum('bthm,bthv,bmhkv->bthk', weights_m, dout_t, states_m) * read_scale
            )
            dstates[:, start:stop].add_(
                torch.einsum('bthk,bthm,bthv->bmhkv', read_q_t, weights_m, dout_t) * read_scale
            )

    return dread_q.to(read_q.dtype), dstates.to(states.dtype), dweights.to(weights.dtype)


@contextmanager
def _module_sequence_parallel(module, enabled: bool):
    if not hasattr(module, 'sequence_parallel'):
        yield
        return
    old_value = module.sequence_parallel
    module.sequence_parallel = enabled
    try:
        yield
    finally:
        module.sequence_parallel = old_value


class MarchGatedDeltaNet(MarchDiagnosticsMixin, GatedDeltaNet):
    """Anchor-token historical-state read path for Qwen3.5 linear-attention layers.

    This module intentionally supports only the training path:
    CP=1, packed_seq_params=None, and no inference cache.
    """

    _HSD_BACKEND_COUNTER_NAMES = (
        'local_flashqla_hit',
        'local_flashqla_miss',
        'local_reference_hit',
        'dense_reference_hit',
        'fallback',
    )

    def __init__(self, config: ModelConfig, submodules: 'GatedDeltaNetSubmodules', *args, **kwargs):
        validate_training_environment(config=config, require_march=True)
        validate_safin1_training_runtime(config)
        super().__init__(config, submodules, *args, **kwargs)
        read_block_default = int(getattr(config, 'hsd_read_block_size', 1024))
        read_block_default = _env_int('HIST_READ_BLOCK_SIZE', read_block_default)
        self.hist_read_block_size = max(1, _env_int('HSD_READ_BLOCK_SIZE', read_block_default))
        self.hsd_use_global_read = _env_bool(
            'HSD_USE_GLOBAL_READ', bool(getattr(config, 'hsd_use_global_read', True)))
        selective_scan_default = bool(getattr(config, 'hsd_train_selective_scan', True))
        self.hsd_train_selective_scan = _env_bool('HSD_TRAIN_SELECTIVE_SCAN', selective_scan_default)
        packed_selective_scan_default = bool(
            getattr(config, 'hsd_train_selective_scan_packed', selective_scan_default))
        self.hsd_train_selective_scan_packed = _env_bool(
            'HSD_TRAIN_SELECTIVE_SCAN_PACKED', packed_selective_scan_default)
        self.hsd_train_selective_scan_packed_strict = _env_bool(
            'HSD_TRAIN_SELECTIVE_SCAN_PACKED_STRICT',
            bool(getattr(config, 'hsd_train_selective_scan_packed_strict', False)))
        # The production local/global kernel selections are part of the model
        # contract.  Do not allow process-local HSD_* environment variables to
        # silently change the graph after the contract has been fingerprinted.
        self.hsd_train_scan_backend = str(
            getattr(config, 'hsd_train_scan_backend', 'triton')).strip().lower()
        self.hsd_train_scan_strict = bool(getattr(config, 'hsd_train_scan_strict', False))
        self.hsd_flashqla_tile_bwd_backend = str(
            getattr(config, 'hsd_flashqla_tile_bwd_backend', 'optimized')).strip().lower()
        self.hsd_fused_global_read_backend = str(
            getattr(config, 'hsd_fused_global_read_backend', 'auto')).strip().lower()
        self.hsd_fused_global_read_strict = bool(
            getattr(config, 'hsd_fused_global_read_strict', False))
        if self.hsd_train_scan_backend not in {'triton', 'flashqla_tile'}:
            raise ValueError(
                "hsd_train_scan_backend must be either 'triton' or 'flashqla_tile'.")
        if self.hsd_flashqla_tile_bwd_backend not in {'optimized', 'reference'}:
            raise ValueError(
                "hsd_flashqla_tile_bwd_backend must be 'optimized' or 'reference'.")
        if self.hsd_fused_global_read_backend not in {'auto', 'reference', 'triton'}:
            raise ValueError(f'Unsupported hsd_fused_global_read_backend: '
                             f'{self.hsd_fused_global_read_backend!r}.')
        self.hsd_block_history_read = _env_bool(
            'HSD_BLOCK_HISTORY_READ', bool(getattr(config, 'hsd_block_history_read', False)))
        self.hsd_history_read_block_m = max(
            1, _env_int('HSD_HISTORY_READ_BLOCK_M', int(getattr(config, 'hsd_history_read_block_m', 4) or 4)))
        self.hsd_history_read_block_t = _env_int(
            'HSD_HISTORY_READ_BLOCK_T', int(getattr(config, 'hsd_history_read_block_t', 0) or 0))
        global_read_layers = os.environ.get('HSD_GLOBAL_READ_LAYERS', '').strip()
        if not global_read_layers:
            global_read_layers = getattr(config, 'hsd_global_read_layers', None)
        self.hsd_global_read_layers = _parse_layer_selector(global_read_layers)
        self.hsd_layer_uses_global_read = self._global_read_enabled_for_layer()
        self.hsd_persistent_memory_enabled = bool(
            getattr(config, 'hsd_persistent_memory_enabled', False))
        self.hsd_persistent_memory_route_key_enabled = bool(
            getattr(config, 'hsd_persistent_memory_route_key_enabled', False))
        self.hsd_persistent_memory_route_key_mode = str(
            getattr(config, 'hsd_persistent_memory_route_key_mode', 'legacy')).strip().lower()
        if self.hsd_persistent_memory_route_key_mode not in {'dedicated', 'legacy'}:
            raise ValueError(
                "hsd_persistent_memory_route_key_mode must be 'dedicated' or 'legacy'.")
        if self.hsd_persistent_memory_route_key_enabled and not self.hsd_persistent_memory_enabled:
            raise ValueError(
                'hsd_persistent_memory_route_key_enabled requires hsd_persistent_memory_enabled.')
        if self.hsd_persistent_memory_route_key_mode == 'dedicated' \
                and not self.hsd_persistent_memory_route_key_enabled:
            raise ValueError(
                "route-key mode 'dedicated' requires hsd_persistent_memory_route_key_enabled.")
        self.hsd_layer_has_persistent_memory = self._persistent_memory_enabled_for_layer()
        if self.hsd_layer_has_persistent_memory:
            if self.config.context_parallel_size > 1:
                raise NotImplementedError('Persistent HSD memory currently requires context_parallel_size=1.')
            if self.num_value_heads % self.tp_size != 0:
                raise RuntimeError(
                    'Persistent HSD memory requires num_value_heads divisible by tensor parallel size: '
                    f'num_value_heads={self.num_value_heads}, tp_size={self.tp_size}.')
            local_value_heads = self.num_value_heads // self.tp_size
            self.persistent_memory_state = torch.nn.Parameter(torch.zeros(
                local_value_heads,
                self.key_head_dim,
                self.value_head_dim,
                dtype=torch.float32,
                device=self.A_log.device,
            ))
            # Mark the leading value-head dimension as tensor-parallel.  DDP
            # still averages each corresponding shard across data-parallel
            # ranks; TP ranks own disjoint heads.
            self.persistent_memory_state.tensor_model_parallel = True
            self.persistent_memory_state.partition_dim = 0
            self.persistent_memory_state.partition_stride = 1
        else:
            self.register_parameter('persistent_memory_state', None)
        self.hsd_use_route_scale = _env_bool(
            'HSD_USE_ROUTE_SCALE', bool(getattr(config, 'hsd_use_route_scale', True)))
        self.hsd_route_logit_scale = _env_float(
            'HSD_ROUTE_LOGIT_SCALE', float(getattr(config, 'hsd_route_logit_scale', 1.0)))
        if self.hsd_route_logit_scale <= 0:
            raise ValueError('HSD_ROUTE_LOGIT_SCALE must be positive.')
        self.hsd_route_head_dim = _env_int(
            'HSD_ROUTE_HEAD_DIM', int(getattr(config, 'hsd_route_head_dim', None) or self.key_head_dim))
        if self.hsd_route_head_dim <= 0:
            raise ValueError('HSD_ROUTE_HEAD_DIM must be positive.')
        self.hsd_route_query_heads = _env_int(
            'HSD_ROUTE_QUERY_HEADS',
            int(getattr(config, 'hsd_route_query_heads', None) or self.num_value_heads),
        )
        if self.hsd_route_query_heads not in {1, self.num_value_heads}:
            raise NotImplementedError('HSD_ROUTE_QUERY_HEADS currently supports 1 or num_value_heads.')
        self.hsd_route_key_heads = _env_int(
            'HSD_ROUTE_KEY_HEADS',
            int(getattr(config, 'hsd_route_key_heads', None) or self.num_value_heads),
        )
        if self.hsd_route_key_heads <= 0:
            raise ValueError('HSD_ROUTE_KEY_HEADS must be positive.')
        if self.num_value_heads % self.hsd_route_key_heads != 0:
            raise ValueError(
                'HSD_ROUTE_KEY_HEADS must divide num_value_heads so each route key is shared by an equal '
                f'query-head group: route_key_heads={self.hsd_route_key_heads}, '
                f'num_value_heads={self.num_value_heads}.'
            )
        if self.hsd_persistent_memory_route_key_enabled and (
                self.hsd_route_key_heads != self.num_value_heads
                or self.hsd_route_head_dim != self.key_head_dim):
            raise ValueError(
                'Dedicated persistent-memory route keys require one full key per value head: '
                f'route_key_heads={self.hsd_route_key_heads}, value_heads={self.num_value_heads}, '
                f'route_head_dim={self.hsd_route_head_dim}, key_head_dim={self.key_head_dim}.')
        if self.hsd_layer_has_persistent_memory and self.hsd_persistent_memory_route_key_enabled:
            self.persistent_memory_route_key = torch.nn.Parameter(torch.zeros(
                self.num_value_heads // self.tp_size,
                self.key_head_dim,
                dtype=torch.float32,
                device=self.A_log.device,
            ))
            self.persistent_memory_route_key.tensor_model_parallel = True
            self.persistent_memory_route_key.partition_dim = 0
            self.persistent_memory_route_key.partition_stride = 1
        else:
            self.register_parameter('persistent_memory_route_key', None)
        local_value_heads = self.num_value_heads // self.tp_size
        if self.hsd_route_key_heads != self.num_value_heads and local_value_heads % self.hsd_route_key_heads != 0:
            raise ValueError(
                'Grouped/shared HSD route keys must divide the local value heads under tensor parallelism: '
                f'route_key_heads={self.hsd_route_key_heads}, local_value_heads={local_value_heads}, '
                f'tp_size={self.tp_size}.'
            )
        self.hsd_route_qk_l2norm = _env_bool(
            'HSD_ROUTE_QK_L2NORM', bool(getattr(config, 'hsd_route_qk_l2norm', True)))
        self.hsd_route_position_encoding = _env_str(
            'HSD_ROUTE_POSITION_ENCODING', str(getattr(config, 'hsd_route_position_encoding', 'none'))).lower()
        if self.hsd_route_position_encoding not in {'none', 'rope'}:
            raise ValueError("HSD_ROUTE_POSITION_ENCODING must be either 'none' or 'rope'.")
        self.hsd_route_position_mode = _env_str(
            'HSD_ROUTE_POSITION_MODE', str(getattr(config, 'hsd_route_position_mode', 'chunk_end'))).lower()
        if self.hsd_route_position_mode not in {'chunk_end'}:
            raise ValueError("HSD_ROUTE_POSITION_MODE must be 'chunk_end'.")
        self.hsd_route_attention_scaling = 1.0
        self.hsd_route_rotary_pos_emb = None
        if self.hsd_layer_uses_global_read and self.hsd_route_position_encoding == 'rope':
            new_inv_freq, self.hsd_route_attention_scaling = get_rope_inv_freq(config)
            route_rope_channels = int(new_inv_freq.numel() * 2)
            if route_rope_channels > self.hsd_route_head_dim:
                raise ValueError(
                    'HSD route RoPE dim cannot exceed route q/k head dim: '
                    f'route_rope_channels={route_rope_channels}, route_head_dim={self.hsd_route_head_dim}.')
            self.hsd_route_rotary_pos_emb = RotaryEmbedding(
                kv_channels=route_rope_channels,
                rotary_percent=1,
                rotary_interleaved=self.config.rotary_interleaved,
                rotary_base=self.config.rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
            if new_inv_freq.numel() != self.hsd_route_rotary_pos_emb.inv_freq.numel():
                raise ValueError(
                    'HSD route RoPE dim must match model attention RoPE dim: '
                    f'route_inv_freq={self.hsd_route_rotary_pos_emb.inv_freq.numel()}, '
                    f'model_inv_freq={new_inv_freq.numel()}.')
            self.hsd_route_rotary_pos_emb.inv_freq = new_inv_freq.to(
                self.hsd_route_rotary_pos_emb.inv_freq.device)
        self.hsd_read_topk = max(0, _env_int('HSD_READ_TOPK', int(getattr(config, 'hsd_read_topk', 0))))
        self.hsd_topk_backend = _env_str(
            'HSD_TOPK_BACKEND', str(getattr(config, 'hsd_topk_backend', 'reference'))).lower()
        if self.hsd_topk_backend not in {'reference', 'sparse'}:
            raise ValueError("HSD_TOPK_BACKEND must be either 'reference' or 'sparse'.")
        if self.hsd_topk_backend == 'sparse' and self.hsd_read_topk not in {1, 2, 4, 8, 16}:
            raise ValueError('Sparse HSD top-k requires HSD_READ_TOPK in {1, 2, 4, 8, 16}.')
        self.hsd_null_read_gate = _env_bool(
            'HSD_NULL_READ_GATE', bool(getattr(config, 'hsd_null_read_gate', False)))
        self.hsd_null_read_gate_init_bias = _env_float(
            'HSD_NULL_READ_GATE_INIT_BIAS', float(getattr(config, 'hsd_null_read_gate_init_bias', 0.0)))
        if not math.isfinite(self.hsd_null_read_gate_init_bias):
            raise ValueError('HSD_NULL_READ_GATE_INIT_BIAS must be finite.')
        self.hsd_history_scale = _env_float(
            'HSD_HISTORY_SCALE', float(getattr(config, 'hsd_history_scale', 1.0)))
        if self.hsd_history_scale < 0:
            raise ValueError('HSD_HISTORY_SCALE must be non-negative.')
        self.hsd_history_out_rmsnorm = _env_bool(
            'HSD_HISTORY_OUT_RMSNORM', bool(getattr(config, 'hsd_history_out_rmsnorm', False)))
        self.hsd_history_out_rmsnorm_eps = _env_float(
            'HSD_HISTORY_OUT_RMSNORM_EPS', float(getattr(config, 'hsd_history_out_rmsnorm_eps', 1e-6)))
        if self.hsd_history_out_rmsnorm_eps <= 0:
            raise ValueError('HSD_HISTORY_OUT_RMSNORM_EPS must be positive.')
        self.hsd_learnable_history_scale = _env_bool(
            'HSD_LEARNABLE_HISTORY_SCALE', bool(getattr(config, 'hsd_learnable_history_scale', False)))
        self.hsd_learnable_history_scale_init = _env_float(
            'HSD_LEARNABLE_HISTORY_SCALE_INIT',
            float(getattr(config, 'hsd_learnable_history_scale_init', 0.01)),
        )
        self.hsd_learnable_history_scale_max = _env_float(
            'HSD_LEARNABLE_HISTORY_SCALE_MAX',
            float(getattr(config, 'hsd_learnable_history_scale_max', 1.0)),
        )
        if self.hsd_learnable_history_scale_init < 0:
            raise ValueError('HSD_LEARNABLE_HISTORY_SCALE_INIT must be non-negative.')
        if self.hsd_learnable_history_scale_max <= 0:
            raise ValueError('HSD_LEARNABLE_HISTORY_SCALE_MAX must be positive.')
        if self.hsd_learnable_history_scale_init > self.hsd_learnable_history_scale_max:
            raise ValueError('HSD_LEARNABLE_HISTORY_SCALE_INIT must not exceed HSD_LEARNABLE_HISTORY_SCALE_MAX.')
        self.hsd_history_gate_schedule_start = _env_optional_float(
            'HSD_HISTORY_GATE_SCHEDULE_START',
            getattr(config, 'hsd_history_gate_schedule_start', None),
        )
        self.hsd_history_gate_schedule_end = _env_optional_float(
            'HSD_HISTORY_GATE_SCHEDULE_END',
            getattr(config, 'hsd_history_gate_schedule_end', None),
        )
        self.hsd_history_gate_schedule_steps = _env_int(
            'HSD_HISTORY_GATE_SCHEDULE_STEPS',
            int(getattr(config, 'hsd_history_gate_schedule_steps', 0) or 0),
        )
        if self.hsd_history_gate_schedule_steps < 0:
            raise ValueError('HSD_HISTORY_GATE_SCHEDULE_STEPS must be non-negative.')
        for name, value in (
                ('HSD_HISTORY_GATE_SCHEDULE_START', self.hsd_history_gate_schedule_start),
                ('HSD_HISTORY_GATE_SCHEDULE_END', self.hsd_history_gate_schedule_end)):
            if value is None:
                continue
            if value < 0:
                raise ValueError(f'{name} must be non-negative.')
            if value > self.hsd_learnable_history_scale_max:
                raise ValueError(f'{name} must not exceed HSD_LEARNABLE_HISTORY_SCALE_MAX.')
        self.hsd_history_gate_schedule_enabled = (
            self.hsd_history_gate_schedule_steps > 0
            and self.hsd_history_gate_schedule_start is not None
            and self.hsd_history_gate_schedule_end is not None
        )
        if self.hsd_history_gate_schedule_steps > 0 and not self.hsd_history_gate_schedule_enabled:
            raise ValueError(
                'HSD_HISTORY_GATE_SCHEDULE_START and HSD_HISTORY_GATE_SCHEDULE_END are required when '
                'HSD_HISTORY_GATE_SCHEDULE_STEPS > 0.')
        self.hsd_history_out_grad_scale = _env_optional_float(
            'HSD_HISTORY_OUT_GRAD_SCALE',
            getattr(config, 'hsd_history_out_grad_scale', None),
        )
        self.hsd_history_out_grad_scale_start = _env_optional_float(
            'HSD_HISTORY_OUT_GRAD_SCALE_START',
            getattr(config, 'hsd_history_out_grad_scale_start', None),
        )
        self.hsd_history_out_grad_scale_end = _env_optional_float(
            'HSD_HISTORY_OUT_GRAD_SCALE_END',
            getattr(config, 'hsd_history_out_grad_scale_end', None),
        )
        self.hsd_history_out_grad_scale_steps = _env_int(
            'HSD_HISTORY_OUT_GRAD_SCALE_STEPS',
            int(getattr(config, 'hsd_history_out_grad_scale_steps', 0) or 0),
        )
        if self.hsd_history_out_grad_scale is not None and self.hsd_history_out_grad_scale < 0:
            raise ValueError('HSD_HISTORY_OUT_GRAD_SCALE must be non-negative.')
        if self.hsd_history_out_grad_scale_steps < 0:
            raise ValueError('HSD_HISTORY_OUT_GRAD_SCALE_STEPS must be non-negative.')
        for name, value in (
                ('HSD_HISTORY_OUT_GRAD_SCALE_START', self.hsd_history_out_grad_scale_start),
                ('HSD_HISTORY_OUT_GRAD_SCALE_END', self.hsd_history_out_grad_scale_end)):
            if value is not None and value < 0:
                raise ValueError(f'{name} must be non-negative.')
        self.hsd_history_out_grad_scale_schedule_enabled = (
            self.hsd_history_out_grad_scale_steps > 0
            and self.hsd_history_out_grad_scale_start is not None
            and self.hsd_history_out_grad_scale_end is not None
        )
        if self.hsd_history_out_grad_scale_steps > 0 and not self.hsd_history_out_grad_scale_schedule_enabled:
            raise ValueError(
                'HSD_HISTORY_OUT_GRAD_SCALE_START and HSD_HISTORY_OUT_GRAD_SCALE_END are required when '
                'HSD_HISTORY_OUT_GRAD_SCALE_STEPS > 0.')
        self.hsd_anchor_content_grad_scale = _env_optional_float(
            'HSD_ANCHOR_CONTENT_GRAD_SCALE',
            getattr(config, 'hsd_anchor_content_grad_scale', None),
        )
        self.hsd_anchor_content_grad_scale_start = _env_optional_float(
            'HSD_ANCHOR_CONTENT_GRAD_SCALE_START',
            getattr(config, 'hsd_anchor_content_grad_scale_start', None),
        )
        self.hsd_anchor_content_grad_scale_end = _env_optional_float(
            'HSD_ANCHOR_CONTENT_GRAD_SCALE_END',
            getattr(config, 'hsd_anchor_content_grad_scale_end', None),
        )
        self.hsd_anchor_content_grad_scale_steps = _env_int(
            'HSD_ANCHOR_CONTENT_GRAD_SCALE_STEPS',
            int(getattr(config, 'hsd_anchor_content_grad_scale_steps', 0) or 0),
        )
        if self.hsd_anchor_content_grad_scale is not None and self.hsd_anchor_content_grad_scale < 0:
            raise ValueError('HSD_ANCHOR_CONTENT_GRAD_SCALE must be non-negative.')
        if self.hsd_anchor_content_grad_scale_steps < 0:
            raise ValueError('HSD_ANCHOR_CONTENT_GRAD_SCALE_STEPS must be non-negative.')
        for name, value in (
                ('HSD_ANCHOR_CONTENT_GRAD_SCALE_START', self.hsd_anchor_content_grad_scale_start),
                ('HSD_ANCHOR_CONTENT_GRAD_SCALE_END', self.hsd_anchor_content_grad_scale_end)):
            if value is not None and value < 0:
                raise ValueError(f'{name} must be non-negative.')
        self.hsd_anchor_content_grad_scale_schedule_enabled = (
            self.hsd_anchor_content_grad_scale_steps > 0
            and self.hsd_anchor_content_grad_scale_start is not None
            and self.hsd_anchor_content_grad_scale_end is not None
        )
        if (self.hsd_anchor_content_grad_scale_steps > 0
                and not self.hsd_anchor_content_grad_scale_schedule_enabled):
            raise ValueError(
                'HSD_ANCHOR_CONTENT_GRAD_SCALE_START and HSD_ANCHOR_CONTENT_GRAD_SCALE_END are required when '
                'HSD_ANCHOR_CONTENT_GRAD_SCALE_STEPS > 0.')

        anchor_q_dim = self.num_value_heads * self.key_head_dim
        self.anchor_q_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            anchor_q_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='anchor_q',
            tp_group=self.pg_collection.tp,
        )
        if self.hsd_layer_uses_global_read:
            route_key_dim = self.hsd_route_key_heads * self.hsd_route_head_dim
            anchor_k_gather_tp_output = self.hsd_route_key_heads != self.num_value_heads
            self.anchor_k_proj = build_module(
                submodules.in_proj,
                self.hidden_size,
                route_key_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='anchor_k',
                tp_group=self.pg_collection.tp,
            )
            self.anchor_k_proj._hsd_gather_tp_output = anchor_k_gather_tp_output
            route_query_dim = self.hsd_route_query_heads * self.hsd_route_head_dim
            router_q_gather_tp_output = self.hsd_route_query_heads != self.num_value_heads
            self.router_q_proj = build_module(
                submodules.in_proj,
                self.hidden_size,
                route_query_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='router_q',
                tp_group=self.pg_collection.tp,
            )
            self.router_q_proj._hsd_gather_tp_output = router_q_gather_tp_output
            if self.hsd_null_read_gate:
                if self.num_value_heads % self.tp_size != 0:
                    raise RuntimeError(
                        'HSD null read gate requires num_value_heads to be divisible by tensor parallel size: '
                        f'num_value_heads={self.num_value_heads}, tp_size={self.tp_size}.')
                self.null_gate_proj = build_module(
                    submodules.in_proj,
                    self.hidden_size,
                    self.num_value_heads,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=False,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name='null_gate',
                    tp_group=self.pg_collection.tp,
                )
                local_value_heads = self.num_value_heads // self.tp_size
                self.hsd_null_read_gate_bias = torch.nn.Parameter(
                    torch.full((local_value_heads,), self.hsd_null_read_gate_init_bias, dtype=torch.float32))
            else:
                self.null_gate_proj = None
                self.register_parameter('hsd_null_read_gate_bias', None)
        else:
            self.anchor_k_proj = None
            self.router_q_proj = None
            self.null_gate_proj = None
            self.register_parameter('hsd_null_read_gate_bias', None)
        if self.hsd_layer_uses_global_read and self.hsd_learnable_history_scale:
            eps = 1e-6
            ratio = self.hsd_learnable_history_scale_init / self.hsd_learnable_history_scale_max
            ratio = min(max(ratio, eps), 1.0 - eps)
            init_logit = math.log(ratio / (1.0 - ratio))
            self.hsd_history_gate_logit = torch.nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.register_parameter('hsd_history_gate_logit', None)
        if self.hsd_layer_uses_global_read and self.hsd_history_out_rmsnorm:
            self.hsd_history_out_rmsnorm_weight = torch.nn.Parameter(torch.ones(self.value_head_dim))
        else:
            self.register_parameter('hsd_history_out_rmsnorm_weight', None)
        self._init_hsd_projection_weights()
        self._hsd_debug_stats = {}
        self._hsd_debug_stat_counts = {}
        self._hsd_profile_stats = {}
        self._persistent_memory_metrics = {}
        self._hsd_backend_counters = {
            name: 0 for name in self._HSD_BACKEND_COUNTER_NAMES
        }
        self._persistent_memory_grad_hook_registered = False

        def _supply_missing_persistent_memory(
            module,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            parameter = getattr(module, 'persistent_memory_state', None)
            key = f'{prefix}persistent_memory_state'
            if parameter is not None and key not in state_dict:
                state_dict[key] = torch.zeros_like(parameter.detach())

        # Torch-style checkpoints created before this feature load without a
        # missing-key warning and deterministically receive the zero state.
        self.register_load_state_dict_pre_hook(_supply_missing_persistent_memory)

    @property
    def route_q_proj(self):
        return self.router_q_proj

    def _apply(self, fn, recurse: bool = True):
        """Apply device moves while keeping persistent memory and its grad FP32.

        Megatron's ``Float16Module`` calls ``module.half()`` or
        ``module.bfloat16()`` over the complete model.  The persistent state is
        an optimizer/master-quality parameter and must be the one exception;
        forward paths cast only its computed read at the residual boundary.
        """
        result = super()._apply(fn, recurse=recurse)
        for name in ('persistent_memory_state', 'persistent_memory_route_key'):
            parameter = self._parameters.get(name)
            if parameter is not None and parameter.dtype != torch.float32:
                with torch.no_grad():
                    parameter.data = parameter.data.float()
                    if parameter.grad is not None:
                        parameter.grad.data = parameter.grad.data.float()
        return result

    def _increment_hsd_backend_counter(self, name: str) -> None:
        if name not in self._HSD_BACKEND_COUNTER_NAMES:
            raise KeyError(f'Unknown HSD backend counter: {name}.')
        counters = getattr(self, '_hsd_backend_counters', None)
        if counters is None:
            counters = {key: 0 for key in self._HSD_BACKEND_COUNTER_NAMES}
            self._hsd_backend_counters = counters
        counters[name] += 1

    def get_hsd_backend_counters(self) -> dict[str, int]:
        """Return per-layer kernel dispatch counts for startup validation."""
        counters = getattr(self, '_hsd_backend_counters', {})
        return {
            name: int(counters.get(name, 0))
            for name in self._HSD_BACKEND_COUNTER_NAMES
        }

    def get_persistent_memory_metrics(self, reset: bool = False) -> dict[str, torch.Tensor]:
        """Return the latest local persistent-memory health metrics.

        Values are detached FP32 scalar tensors so the trainer can aggregate
        them with its normal DP metric path without forcing a host sync.
        """
        metrics = {
            name: value.detach().clone()
            for name, value in getattr(self, '_persistent_memory_metrics', {}).items()
        }
        if reset:
            self._persistent_memory_metrics.clear()
        return metrics

    def _set_persistent_memory_metric(self, name: str, value) -> None:
        if not hasattr(self, '_persistent_memory_metrics'):
            self._persistent_memory_metrics = {}
        if isinstance(value, torch.Tensor):
            value = value.detach().float()
            if value.numel() != 1:
                value = value.mean()
            value = value.reshape(())
        else:
            value = torch.tensor(float(value), dtype=torch.float32, device=self.A_log.device)
        self._persistent_memory_metrics[name] = value

    def _init_hsd_projection_weights(self):
        for module in (self.anchor_q_proj, self.anchor_k_proj, self.router_q_proj):
            weight = getattr(module, 'weight', None)
            if weight is None:
                continue
            with torch.no_grad():
                self.config.init_method(weight)
        weight = getattr(self.null_gate_proj, 'weight', None)
        if weight is not None:
            with torch.no_grad():
                weight.zero_()

    def _check_supported_path(self, packed_seq_params, inference_context):
        if inference_context is not None:
            raise NotImplementedError('MarchGatedDeltaNet does not support inference yet.')
        if packed_seq_params is not None:
            if getattr(packed_seq_params, 'qkv_format', None) != 'thd':
                raise NotImplementedError('MarchGatedDeltaNet packed sequence support requires qkv_format="thd".')
        if bool(getattr(self.config, 'hsd_require_packed_seq_params', False)):
            if not bool(getattr(self.config, 'hsd_packed_document_isolation', False)):
                raise RuntimeError(
                    'HSD packed_seq_params are required but packed document isolation is disabled.')
            if packed_seq_params is None:
                raise RuntimeError(
                    'Stage-3 document-isolated HSD requires packed_seq_params on every layer forward.')
        if self.config.context_parallel_size > 1:
            if packed_seq_params is not None:
                raise NotImplementedError('MarchGatedDeltaNet CP support does not cover packed sequences yet.')
        if getattr(self.config, 'mtp_num_layers', None):
            raise NotImplementedError('MarchGatedDeltaNet does not support MTP with anchor tokens yet.')
        if self.config.deterministic_mode:
            raise NotImplementedError('MarchGatedDeltaNet does not support deterministic_mode yet.')
        if chunk_gated_delta_rule is None or l2norm is None:
            raise ImportError('MarchGatedDeltaNet requires fla gated-delta-rule and l2norm kernels.')

    def _global_read_enabled_for_layer(self) -> bool:
        if not self.hsd_use_global_read:
            return False
        if self.hsd_global_read_layers is None:
            return True
        layer_number = getattr(self, 'layer_number', None)
        if layer_number is None:
            raise RuntimeError('HSD_GLOBAL_READ_LAYERS requires layer_number to be set.')
        return int(layer_number) in self.hsd_global_read_layers

    def _persistent_memory_enabled_for_layer(self) -> bool:
        if not bool(getattr(self.config, 'hsd_persistent_memory_enabled', False)):
            return False
        if not self.hsd_use_global_read:
            return False
        # Absolute dedicated keys remove the only causal route from an early
        # non-reader state to text logits.  Allocate state/key pairs only on
        # actual global-read layers; retain the historical wider allocation
        # when dedicated keys are disabled for old state-only checkpoints.
        if bool(getattr(self.config, 'hsd_persistent_memory_route_key_enabled', False)):
            return self.hsd_layer_uses_global_read
        layer_number = getattr(self, 'layer_number', None)
        if layer_number is None:
            raise RuntimeError('Persistent HSD memory requires layer_number to be set.')
        if self.hsd_global_read_layers is None:
            # Every HSD layer reads globally, so its state is directly useful.
            return True
        if not self.hsd_global_read_layers:
            return False
        # A non-global HSD layer still needs a memory state because its virtual
        # anchor output changes keys in later global-read layers. The first HSD
        # layer strictly after the final global read has no causal path to text
        # logits and intentionally receives no parameter.
        return int(layer_number) <= max(self.hsd_global_read_layers)

    @staticmethod
    def _route_keys_for_query_heads(anchor_k: torch.Tensor, query_heads: int) -> torch.Tensor:
        key_heads = int(anchor_k.shape[2])
        query_heads = int(query_heads)
        if key_heads == query_heads:
            return anchor_k
        if key_heads == 1:
            return anchor_k.expand(anchor_k.shape[0], anchor_k.shape[1], query_heads, anchor_k.shape[3])
        if query_heads % key_heads == 0:
            return anchor_k.repeat_interleave(query_heads // key_heads, dim=2)
        raise RuntimeError(
            'HSD anchor route key heads must divide query heads: '
            f'key_heads={key_heads}, query_heads={query_heads}.'
        )

    @staticmethod
    def _route_queries_for_value_heads(router_q: torch.Tensor, value_heads: int) -> torch.Tensor:
        query_heads = int(router_q.shape[2])
        value_heads = int(value_heads)
        if query_heads == value_heads:
            return router_q
        if query_heads == 1:
            return router_q.expand(router_q.shape[0], router_q.shape[1], value_heads, router_q.shape[3])
        raise RuntimeError(
            'HSD route query heads must be 1 or match value heads: '
            f'query_heads={query_heads}, value_heads={value_heads}.'
        )

    def _history_gate_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = self.A_log.device
        schedule_gate = self._history_gate_schedule_tensor(device=device)
        if self.hsd_history_gate_logit is None:
            if schedule_gate is not None:
                return schedule_gate
            return torch.ones((), device=device, dtype=torch.float32)
        param_gate = torch.sigmoid(self.hsd_history_gate_logit.float()).to(device) * float(
            self.hsd_learnable_history_scale_max)
        if schedule_gate is not None:
            return torch.maximum(param_gate, schedule_gate)
        return param_gate

    def _history_gate_param_tensor(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if self.hsd_history_gate_logit is None:
            return None
        if device is None:
            device = self.A_log.device
        return torch.sigmoid(self.hsd_history_gate_logit.float()).to(device) * float(
            self.hsd_learnable_history_scale_max)

    def _history_gate_schedule_tensor(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if not self.hsd_history_gate_schedule_enabled:
            return None
        if device is None:
            device = self.A_log.device
        step = max(0, _env_int('HSD_RUNTIME_STEP', 0))
        if self.hsd_history_gate_schedule_steps <= 1:
            progress = 1.0
        else:
            progress = min(1.0, float(step) / float(self.hsd_history_gate_schedule_steps - 1))
        start = float(self.hsd_history_gate_schedule_start)
        end = float(self.hsd_history_gate_schedule_end)
        value = start + (end - start) * progress
        return torch.tensor(value, device=device, dtype=torch.float32)

    def _history_scale_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = self.A_log.device
        return self._history_gate_tensor(device=device) * float(self.hsd_history_scale)

    def _apply_history_out_rmsnorm(self, history_out: torch.Tensor) -> torch.Tensor:
        if self.hsd_history_out_rmsnorm_weight is None:
            return history_out
        rms = history_out.float().pow(2).mean(dim=-1, keepdim=True).add(self.hsd_history_out_rmsnorm_eps).rsqrt()
        weight = self.hsd_history_out_rmsnorm_weight.float().view(1, 1, 1, -1)
        return (history_out.float() * rms * weight).to(history_out.dtype)

    def _history_out_grad_scale_tensor(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if self.hsd_history_out_grad_scale is None and not self.hsd_history_out_grad_scale_schedule_enabled:
            return None
        if device is None:
            device = self.A_log.device
        if self.hsd_history_out_grad_scale_schedule_enabled:
            if self.hsd_history_out_grad_scale_steps <= 1:
                value = float(self.hsd_history_out_grad_scale_end)
            else:
                step = max(0, _env_int('HSD_RUNTIME_STEP', 0))
                progress = min(1.0, float(step) / float(self.hsd_history_out_grad_scale_steps - 1))
                start = float(self.hsd_history_out_grad_scale_start)
                end = float(self.hsd_history_out_grad_scale_end)
                value = start + (end - start) * progress
        else:
            value = float(self.hsd_history_out_grad_scale)
        if value < 0:
            raise RuntimeError('HSD history_out grad scale must be non-negative.')
        return torch.tensor(value, device=device, dtype=torch.float32)

    def _apply_history_out_grad_scale(self, history_out: torch.Tensor) -> torch.Tensor:
        grad_scale = self._history_out_grad_scale_tensor(device=history_out.device)
        if grad_scale is None:
            return history_out
        grad_scale = grad_scale.to(dtype=history_out.dtype)
        return history_out.detach() + grad_scale * (history_out - history_out.detach())

    def _anchor_content_grad_scale_tensor(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if self.hsd_anchor_content_grad_scale is None and not self.hsd_anchor_content_grad_scale_schedule_enabled:
            return None
        if device is None:
            device = self.A_log.device
        if self.hsd_anchor_content_grad_scale_schedule_enabled:
            if self.hsd_anchor_content_grad_scale_steps <= 1:
                value = float(self.hsd_anchor_content_grad_scale_end)
            else:
                step = max(0, _env_int('HSD_RUNTIME_STEP', 0))
                progress = min(1.0, float(step) / float(self.hsd_anchor_content_grad_scale_steps - 1))
                start = float(self.hsd_anchor_content_grad_scale_start)
                end = float(self.hsd_anchor_content_grad_scale_end)
                value = start + (end - start) * progress
        else:
            value = float(self.hsd_anchor_content_grad_scale)
        if value < 0:
            raise RuntimeError('HSD anchor content grad scale must be non-negative.')
        return torch.tensor(value, device=device, dtype=torch.float32)

    def _apply_anchor_content_grad_scale(
        self,
        out: torch.Tensor,
        local_anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        grad_scale = self._anchor_content_grad_scale_tensor(device=out.device)
        if grad_scale is None or not bool(local_anchor_mask.any().item()):
            return out
        grad_scale = grad_scale.to(dtype=out.dtype)
        anchor_mask_sbh = local_anchor_mask.to(device=out.device, dtype=torch.bool).transpose(0, 1).unsqueeze(-1)
        scaled_out = out.detach() + grad_scale * (out - out.detach())
        return torch.where(anchor_mask_sbh, scaled_out, out)


    @staticmethod
    def _anchor_text_counts(anchor_mask: torch.Tensor) -> torch.Tensor:
        text_seen = (~anchor_mask[0]).long().cumsum(dim=0)
        return text_seen[anchor_mask[0]]

    def _persistent_anchor_slots(
        self,
        anchor_mask: torch.Tensor,
        persistent_memory_mask: Optional[torch.Tensor],
        anchor_cu_seqlens: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return an anchor-index mask with one persistent slot per document."""
        if not self.hsd_persistent_memory_enabled:
            if persistent_memory_mask is not None and bool(persistent_memory_mask.any().item()):
                raise ValueError('persistent_memory_mask was provided while persistent HSD memory is disabled.')
            return None

        if persistent_memory_mask is None:
            persistent_memory_mask = torch.zeros_like(anchor_mask, dtype=torch.bool)
            anchor_positions = torch.nonzero(anchor_mask[0], as_tuple=False).flatten()
            if anchor_cu_seqlens is None:
                if anchor_positions.numel() == 0:
                    raise RuntimeError('Persistent HSD memory requires one anchor slot.')
                persistent_memory_mask[:, anchor_positions[0]] = True
            else:
                anchor_cu = anchor_cu_seqlens.detach().to(device='cpu', dtype=torch.long)
                starts, ends = anchor_cu[:-1], anchor_cu[1:]
                if bool((ends <= starts).any().item()):
                    raise RuntimeError('Every packed document must contain one persistent memory anchor.')
                memory_positions = anchor_positions.index_select(
                    0, starts.to(device=anchor_positions.device))
                persistent_memory_mask[:, memory_positions] = True

        persistent_memory_mask = persistent_memory_mask.to(device=anchor_mask.device, dtype=torch.bool)
        if tuple(persistent_memory_mask.shape) != tuple(anchor_mask.shape):
            raise ValueError(
                'persistent_memory_mask shape must match anchor_mask: '
                f'{tuple(persistent_memory_mask.shape)} vs {tuple(anchor_mask.shape)}.')
        if not torch.equal(
                persistent_memory_mask,
                persistent_memory_mask[:1].expand_as(persistent_memory_mask)):
            raise NotImplementedError('Persistent HSD memory requires the same layout for every batch row.')
        if bool((persistent_memory_mask & ~anchor_mask).any().item()):
            raise ValueError('persistent_memory_mask must be a subset of anchor_mask.')

        anchor_slots = persistent_memory_mask[0, anchor_mask[0]]
        if anchor_cu_seqlens is None:
            expected = 1
            actual = int(anchor_slots.sum().item())
            if actual != expected:
                raise ValueError(f'Non-packed HSD requires exactly one persistent memory slot, got {actual}.')
        else:
            anchor_cu = anchor_cu_seqlens.detach().to(device='cpu', dtype=torch.long)
            expected = int(anchor_cu.numel()) - 1
            if int(anchor_slots.sum().item()) != expected:
                raise ValueError(
                    f'Packed HSD requires exactly one persistent memory slot per document ({expected}).')
            slots_cpu = anchor_slots.detach().to(device='cpu', dtype=torch.bool)
            for start, end in zip(anchor_cu[:-1].tolist(), anchor_cu[1:].tolist()):
                if int(slots_cpu[int(start):int(end)].sum().item()) != 1:
                    raise ValueError('Each packed document must contain exactly one persistent memory slot.')
        return anchor_slots

    def _apply_persistent_memory_anchor_output(
        self,
        anchor_o: torch.Tensor,
        anchor_q: torch.Tensor,
        persistent_anchor_slots: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parameter = self.persistent_memory_state
        if parameter is None or persistent_anchor_slots is None:
            return anchor_o
        if tuple(parameter.shape) != (
                int(anchor_q.shape[2]), self.key_head_dim, self.value_head_dim):
            raise RuntimeError(
                'Persistent HSD memory state shape mismatch: '
                f'parameter={tuple(parameter.shape)}, anchor_q={tuple(anchor_q.shape)}.')
        memory_o = torch.einsum(
            'bmhk,hkv->bmhv',
            anchor_q.float(),
            parameter,
        ) * (self.key_head_dim ** -0.5)
        mask = persistent_anchor_slots.to(device=anchor_o.device, dtype=torch.bool).view(1, -1, 1, 1)
        output = torch.where(mask, memory_o.to(dtype=anchor_o.dtype), anchor_o)
        state_rms = self._tensor_rms(parameter.detach().float())
        self._set_persistent_memory_metric('state_rms', state_rms)
        if self._hsd_debug_enabled():
            self._record_debug_scalar('persistent_memory_state_rms', state_rms)
            self._tensor_debug_stats('persistent_memory_anchor_o', memory_o)
        if not self._persistent_memory_grad_hook_registered:
            self._register_debug_grad('persistent_memory_state', parameter)
            parameter.register_hook(
                lambda grad: (
                    self._set_persistent_memory_metric(
                        'gradient_rms', self._tensor_rms(grad.detach().float())),
                    grad,
                )[1]
            )
            self._persistent_memory_grad_hook_registered = True
        return output

    def _replace_persistent_anchor_route_key(
        self,
        anchor_k: torch.Tensor,
        persistent_anchor_slots: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Use the per-layer key only for the boundary-zero memory slot."""

        if self.hsd_persistent_memory_route_key_mode != 'dedicated':
            return anchor_k
        parameter = self.persistent_memory_route_key
        if parameter is None:
            raise RuntimeError(
                'Dedicated persistent-memory routing requires persistent_memory_route_key.')
        if persistent_anchor_slots is None:
            return anchor_k
        slots = persistent_anchor_slots.to(device=anchor_k.device, dtype=torch.bool)
        if slots.ndim != 1 or int(slots.numel()) != int(anchor_k.shape[1]):
            raise ValueError(
                'persistent anchor route-key slots shape mismatch: '
                f'slots={tuple(slots.shape)}, keys={tuple(anchor_k.shape)}.')
        if tuple(parameter.shape) != tuple(anchor_k.shape[2:]):
            raise RuntimeError(
                'persistent_memory_route_key shape mismatch: '
                f'parameter={tuple(parameter.shape)}, keys={tuple(anchor_k.shape[2:])}.')
        expanded = parameter.to(device=anchor_k.device).view(1, 1, *parameter.shape)
        expanded = expanded.expand(anchor_k.shape[0], anchor_k.shape[1], *parameter.shape)
        return torch.where(
            slots.view(1, -1, 1, 1),
            expanded.to(dtype=anchor_k.dtype),
            anchor_k,
        )


    def _persistent_memory_read(
        self,
        read_q: torch.Tensor,
        *,
        read_scale: float,
    ) -> torch.Tensor:
        if self.persistent_memory_state is None:
            raise RuntimeError('Persistent-memory routing requires persistent_memory_state on this layer.')
        return torch.einsum(
            'bthk,hkv->bthv',
            read_q.float(),
            self.persistent_memory_state,
        ) * float(read_scale)


    @staticmethod
    def _validate_packed_cu_seqlens(
        cu_seqlens: torch.Tensor,
        *,
        expected_end: int,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1:
            raise ValueError(f'HSD packed sequence requires {name} to be a 1D tensor.')
        if cu_seqlens.numel() < 2:
            raise ValueError(f'HSD packed sequence requires {name} to contain at least 2 entries.')
        cu_cpu = cu_seqlens.detach().to(device='cpu', dtype=torch.long)
        if int(cu_cpu[0].item()) != 0 or int(cu_cpu[-1].item()) != int(expected_end):
            raise ValueError(
                f'HSD packed {name} must span expected length {expected_end}: '
                f'got first={int(cu_cpu[0].item())}, last={int(cu_cpu[-1].item())}.')
        if bool((cu_cpu[1:] < cu_cpu[:-1]).any().item()):
            raise ValueError(f'HSD packed {name} must be non-decreasing.')
        return cu_seqlens

    @staticmethod
    def _packed_local_positions_from_cu(
        cu_seqlens: torch.Tensor,
        total_len: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        cu_cpu = cu_seqlens.detach().to(device='cpu', dtype=torch.long)
        positions = torch.empty(int(total_len), device=device, dtype=torch.long)
        for start, end in zip(cu_cpu[:-1].tolist(), cu_cpu[1:].tolist()):
            start = int(start)
            end = int(end)
            if end > start:
                positions[start:end] = torch.arange(end - start, device=device, dtype=torch.long)
        return positions

    @staticmethod
    def _packed_local_boundary_counts(
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        anchor_cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if boundary_counts.numel() == 0:
            return boundary_counts
        text_cu_cpu = text_cu_seqlens.detach().to(device='cpu', dtype=torch.long)
        anchor_cu_cpu = anchor_cu_seqlens.detach().to(device='cpu', dtype=torch.long)
        local_boundaries = []
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            anchor_start = int(anchor_cu_cpu[doc_idx].item())
            anchor_end = int(anchor_cu_cpu[doc_idx + 1].item())
            if anchor_end > anchor_start:
                local_boundaries.append(boundary_counts[anchor_start:anchor_end] - text_start)
        if not local_boundaries:
            return boundary_counts.new_zeros((0,), dtype=torch.long)
        return torch.cat(local_boundaries, dim=0)

    @staticmethod
    def _nonpacked_batch_sidecars(
        boundary_counts: torch.Tensor,
        *,
        batch: int,
        text_len: int,
        anchor_count: int,
        device: torch.device,
        layout_cache_source: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten a regular batch into B1 packed-kernel metadata.

        Each batch item is one independent segment.  Dataset EOD tokens remain
        ordinary tokens inside that segment: they never create an additional
        boundary.  For B=2/T=8192/M=16 this yields text cu-seqlens
        ``[0, 8192, 16384]``, anchor cu-seqlens ``[0, 16, 32]``, and the second
        sample's anchor boundaries offset by 8192.
        """
        batch = int(batch)
        text_len = int(text_len)
        anchor_count = int(anchor_count)
        if batch <= 0 or text_len < 0 or anchor_count < 0:
            raise ValueError(
                'HSD nonpacked sidecars require batch>0, text_len>=0, and '
                f'anchor_count>=0; got B={batch}, T={text_len}, M={anchor_count}.')
        if boundary_counts.ndim != 1 or int(boundary_counts.numel()) != anchor_count:
            raise ValueError(
                'HSD nonpacked boundary count must match per-sample anchors: '
                f'boundaries={tuple(boundary_counts.shape)}, M={anchor_count}.')
        device = torch.device(device)
        cache_key = None
        if layout_cache_source is not None:
            # Keep the source tensor itself alive in the cache entry.  A key
            # based only on data_ptr/shape/version is unsafe: after the old
            # tensor dies, CUDA's caching allocator can reuse the same pointer
            # for a different anchor layout whose version also starts at
            # zero.  Identity plus a strong reference makes that ABA collision
            # impossible; the version check still invalidates in-place edits.
            cache_key = (
                device.type,
                device.index,
                torch.long,
                batch,
                text_len,
                anchor_count,
                id(layout_cache_source),
            )
            cache_entry = _HSD_NONPACKED_SIDECAR_CACHE.get(cache_key)
            if cache_entry is not None:
                cached_source, cached_source_version, cached = cache_entry
                expected_shapes = (
                    (batch * anchor_count,),
                    (batch + 1,),
                    (batch + 1,),
                )
                cache_valid = (
                    cached_source is layout_cache_source
                    and cached_source_version == int(layout_cache_source._version)
                    and all(
                        tensor.device == device
                        and tensor.dtype == torch.long
                        and tensor.is_contiguous()
                        and tuple(tensor.shape) == expected_shape
                        for tensor, expected_shape in zip(cached, expected_shapes)
                    )
                )
                if cache_valid:
                    # Python dicts preserve insertion order.  Reinsert on a
                    # hit so the bounded cache evicts the least-recently-used
                    # layout rather than the first layout ever inserted.
                    del _HSD_NONPACKED_SIDECAR_CACHE[cache_key]
                    _HSD_NONPACKED_SIDECAR_CACHE[cache_key] = cache_entry
                    return cached
                del _HSD_NONPACKED_SIDECAR_CACHE[cache_key]
        boundaries = boundary_counts.to(device=device, dtype=torch.long).contiguous()
        # Value validation is intentionally CPU-only.  The production CUDA
        # path is called in every layer/step; a Tensor.item() check here would
        # introduce a host synchronization in the hot path.
        if anchor_count and boundaries.device.type == 'cpu':
            if bool(((boundaries < 0) | (boundaries > text_len)).any().item()):
                raise ValueError(
                    f'HSD nonpacked boundaries must lie in [0, {text_len}].')
            if bool((boundaries[1:] < boundaries[:-1]).any().item()):
                raise ValueError('HSD nonpacked boundaries must be non-decreasing.')
        text_cu = torch.arange(batch + 1, device=device, dtype=torch.long) * text_len
        anchor_cu = torch.arange(batch + 1, device=device, dtype=torch.long) * anchor_count
        if anchor_count:
            sample_offsets = (
                torch.arange(batch, device=device, dtype=torch.long).unsqueeze(1) * text_len
            )
            boundaries = (boundaries.unsqueeze(0) + sample_offsets).reshape(-1).contiguous()
        result = (boundaries, text_cu.contiguous(), anchor_cu.contiguous())
        if cache_key is not None:
            if len(_HSD_NONPACKED_SIDECAR_CACHE) >= _HSD_NONPACKED_SIDECAR_CACHE_MAX_ENTRIES:
                oldest_key = next(iter(_HSD_NONPACKED_SIDECAR_CACHE))
                del _HSD_NONPACKED_SIDECAR_CACHE[oldest_key]
            _HSD_NONPACKED_SIDECAR_CACHE[cache_key] = (
                layout_cache_source,
                int(layout_cache_source._version),
                result,
            )
        return result

    def _project_input(self, hidden_states: torch.Tensor, num_key_heads_per_device: int) -> torch.Tensor:
        if self.config.linear_decoupled_in_proj:
            qkvz, _ = self.in_proj_qkvz(hidden_states)
            if self.config.fp8_param:
                fp8_context = transformer_engine.pytorch.fp8_autocast(enabled=False)
            else:
                fp8_context = nullcontext()
            with fp8_context:
                ba, _ = self.in_proj_ba(hidden_states)
            qkvz = qkvz.view(qkvz.shape[:-1] + (num_key_heads_per_device, qkvz.shape[-1] // num_key_heads_per_device))
            ba = ba.view(ba.shape[:-1] + (num_key_heads_per_device, ba.shape[-1] // num_key_heads_per_device))
            qkvzba = torch.concat([qkvz, ba], dim=-1).view(*qkvz.shape[:2], -1)
        else:
            qkvzba, _ = self.in_proj(hidden_states)
        if self.config.context_parallel_size > 1:
            qkvzba = tensor_a2a_cp2hp(
                qkvzba,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.pg_collection.cp,
            )
        qkvzba = qkvzba.view(qkvzba.shape[:-1]
                             + (num_key_heads_per_device, qkvzba.shape[-1] // num_key_heads_per_device))
        return qkvzba.transpose(0, 1).contiguous()


    def _linear_bsh(self, module, hidden_states_bsh: torch.Tensor) -> torch.Tensor:
        if hidden_states_bsh.shape[1] == 0:
            # Transformer Engine's fused LayerNormLinear does not accept a
            # zero-row input (it tries to validate an unallocated rsigma
            # buffer).  An all-short packed batch legitimately has no anchor
            # tokens, though, and the mathematical projection is simply an
            # empty tensor.  Keep every module parameter in the autograd graph
            # so tensor/data-parallel ranks that do have anchors can still
            # reduce gradients with ranks that do not.
            weight = getattr(module, 'weight', None)
            if weight is None or weight.ndim < 1:
                raise RuntimeError('HSD empty projection requires a module weight with an output dimension.')
            output_dim = int(weight.shape[0])
            if getattr(module, '_hsd_gather_tp_output', False):
                output_dim *= int(self.tp_size)

            empty = hidden_states_bsh.sum(dim=-1, keepdim=True) * 0
            parameters = module.parameters() if callable(getattr(module, 'parameters', None)) else ()
            for parameter in parameters:
                if parameter.requires_grad and parameter.numel() > 0:
                    empty = empty + parameter.reshape(-1)[0].to(
                        device=empty.device,
                        dtype=empty.dtype,
                    ) * 0
            return empty.expand(*hidden_states_bsh.shape[:-1], output_dim).contiguous()

        # `hidden_states_bsh` is already full sequence. Column-parallel linears with
        # sequence_parallel=True would all-gather again, so disable it only here.
        with _module_sequence_parallel(module, False):
            out, _ = module(hidden_states_bsh.transpose(0, 1).contiguous())
        if getattr(module, '_hsd_gather_tp_output', False):
            out = _all_gather_hsd_shared_projection(out, group=self.pg_collection.tp)
        return out.transpose(0, 1).contiguous()

    def _linear_heads_bsh(self, module, hidden_states_bsh: torch.Tensor, head_dim: Optional[int] = None) -> torch.Tensor:
        out = self._linear_bsh(module, hidden_states_bsh)
        dim = self.key_head_dim if head_dim is None else int(head_dim)
        if out.shape[-1] % dim != 0:
            raise RuntimeError(f'HSD projection output dim {out.shape[-1]} is not divisible by head_dim={dim}.')
        heads = out.shape[-1] // dim
        return out.view(out.shape[0], out.shape[1], heads, dim)

    def _build_history_read_query(
        self,
        local_query: torch.Tensor,
        text_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return normalize(q_i + W_up W_down h_i) for historical reads."""
        return l2norm(local_query.contiguous())

    def _linear_bsh_from_local_sequence(self, module, hidden_states: torch.Tensor) -> torch.Tensor:
        out, _ = module(hidden_states)
        if getattr(module, '_hsd_gather_tp_output', False):
            out = _all_gather_hsd_shared_projection(out, group=self.pg_collection.tp)
        if self.config.context_parallel_size > 1:
            out = tensor_a2a_cp2hp(
                out,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.pg_collection.cp,
            )
        return out.transpose(0, 1).contiguous()

    def _linear_heads_from_local_sequence(
        self,
        module,
        hidden_states: torch.Tensor,
        head_dim: Optional[int] = None,
    ) -> torch.Tensor:
        out = self._linear_bsh_from_local_sequence(module, hidden_states)
        dim = self.key_head_dim if head_dim is None else int(head_dim)
        if out.shape[-1] % dim != 0:
            raise RuntimeError(f'HSD projection output dim {out.shape[-1]} is not divisible by head_dim={dim}.')
        heads = out.shape[-1] // dim
        return out.view(out.shape[0], out.shape[1], heads, dim)

    def _null_gate_logits(self, text_hidden: torch.Tensor) -> Optional[torch.Tensor]:
        if self.null_gate_proj is None:
            return None
        if self.hsd_null_read_gate_bias is None:
            raise RuntimeError('hsd_null_read_gate_bias is required when HSD null read gate is enabled.')
        logits = self._linear_bsh(self.null_gate_proj, text_hidden)
        if logits.shape[-1] != self.hsd_null_read_gate_bias.numel():
            raise RuntimeError(
                'HSD null read gate output size must match bias size: '
                f'logits={logits.shape[-1]}, bias={self.hsd_null_read_gate_bias.numel()}.')
        bias = self.hsd_null_read_gate_bias.float().to(device=logits.device).view(1, 1, -1)
        logits = logits.float() + bias
        return logits.unsqueeze(-1)

    def _null_gate_logits_from_local_sequence(
        self,
        hidden_states: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.null_gate_proj is None:
            return None
        if self.hsd_null_read_gate_bias is None:
            raise RuntimeError('hsd_null_read_gate_bias is required when HSD null read gate is enabled.')
        logits = self._linear_bsh_from_local_sequence(self.null_gate_proj, hidden_states)
        logits = self._select_bld_masked(logits, text_mask)
        bias = get_parameter_local_cp(
            self.hsd_null_read_gate_bias,
            dim=0,
            cp_group=self.pg_collection.cp,
        ).float().to(device=logits.device).view(1, 1, -1)
        if logits.shape[-1] != bias.numel():
            raise RuntimeError(
                'HSD null read gate output size must match CP-local bias size: '
                f'logits={logits.shape[-1]}, bias={bias.numel()}.')
        logits = logits.float() + bias
        return logits.unsqueeze(-1)

    def _apply_route_position_encoding(self, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.hsd_route_position_encoding == 'none':
            return tensor
        if self.hsd_route_position_encoding == 'rope':
            if positions.numel() == 0:
                return tensor
            if self.hsd_route_rotary_pos_emb is None:
                raise RuntimeError('HSD route RoPE is not initialized.')
            max_pos = int(positions.max().item()) + 1
            if getattr(self.config, 'rope_scaling', None) is not None:
                attention_scaling = dynamic_rope_update(
                    self, self.hsd_route_rotary_pos_emb.inv_freq, max_pos)
                if attention_scaling is not None and attention_scaling != self.hsd_route_attention_scaling:
                    raise ValueError(
                        'Currently does not support changing HSD route attention_scaling during training. '
                        f'self.hsd_route_attention_scaling: {self.hsd_route_attention_scaling}, '
                        f'current_attention_scaling: {attention_scaling}.')
            rotary_pos_emb = self.hsd_route_rotary_pos_emb(max_pos)
            pos = positions.to(device=rotary_pos_emb.device, dtype=torch.long)
            rotary_pos_emb = rotary_pos_emb.index_select(0, pos)
            encoded = apply_rotary_pos_emb(
                tensor.transpose(0, 1).contiguous(),
                rotary_pos_emb,
                self.config,
                mscale=self.hsd_route_attention_scaling,
            )
            return encoded.transpose(0, 1).contiguous()
        raise RuntimeError(f'Unsupported HSD route position encoding: {self.hsd_route_position_encoding}.')

    def _gather_full_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.sp_size == 1:
            return hidden_states
        return gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=False,
            group=self.pg_collection.tp,
        )

    def _anchor_mask_hp2cp(self, anchor_mask: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        mask = anchor_mask.to(device=device, dtype=torch.float32).transpose(0, 1).contiguous().unsqueeze(-1)
        mask = tensor_a2a_hp2cp(
            mask,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
        )
        return mask.any(dim=-1).transpose(0, 1).contiguous().to(dtype=torch.bool)

    def _local_anchor_mask(
        self,
        anchor_mask: torch.Tensor,
        local_seq_len: int,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, int]:
        anchor_mask = anchor_mask.to(dtype=torch.bool)
        full_seq_len = anchor_mask.shape[1]
        if local_seq_len == full_seq_len:
            return anchor_mask.to(device=device, dtype=torch.bool), 0
        cp_size = getattr(self.config, 'context_parallel_size', 1)
        if cp_size > 1:
            cp_anchor_mask = self._anchor_mask_hp2cp(anchor_mask, device=device)
            cp_seq_len = cp_anchor_mask.shape[1]
            if local_seq_len == cp_seq_len:
                return cp_anchor_mask, 0
            if not getattr(self.config, 'sequence_parallel', False):
                raise ValueError(
                    f'HSD local sequence length {local_seq_len} does not match CP-local anchor mask length '
                    f'{cp_seq_len}.')
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            if cp_seq_len % tp_size != 0:
                raise ValueError(
                    f'HSD CP-local sequence length {cp_seq_len} must be divisible by tensor parallel size {tp_size}.')
            shard_len = cp_seq_len // tp_size
            if local_seq_len != shard_len:
                raise ValueError(
                    f'HSD local sequence length {local_seq_len} must match CP/SP shard length {shard_len}.')
            start = tp_rank * shard_len
            return cp_anchor_mask[:, start:start + shard_len], start
        if not getattr(self.config, 'sequence_parallel', False):
            raise ValueError(
                f'HSD local sequence length {local_seq_len} does not match anchor_mask length {full_seq_len}.')
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        if full_seq_len % tp_size != 0:
            raise ValueError(
                f'HSD expanded sequence length {full_seq_len} must be divisible by tensor parallel size {tp_size}.')
        shard_len = full_seq_len // tp_size
        if local_seq_len != shard_len:
            raise ValueError(
                f'HSD local sequence length {local_seq_len} must match SP shard length {shard_len}.')
        start = tp_rank * shard_len
        return anchor_mask[:, start:start + shard_len].to(device=device, dtype=torch.bool), start


    def _run_delta_segment(self, query, key, value, beta, g, start: int, end: int, initial_state):
        return chunk_gated_delta_rule(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
        )


    def _run_hsd_segments_ref(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
    ):
        batch, text_len, heads, key_dim = query.shape
        value_dim = value.shape[-1]
        text_outputs = []
        anchor_outputs = []
        anchor_states = []
        state = None
        cursor = 0
        zero_state = query.new_zeros(batch, heads, key_dim, value_dim)

        with self._profile_cuda_timer('run_hsd_segments_ref'):
            for anchor_idx, boundary in enumerate(boundary_counts.tolist()):
                boundary = int(boundary)
                if boundary < cursor:
                    raise ValueError('anchor positions must be ordered after non-decreasing text prefixes.')
                if boundary > text_len:
                    raise ValueError('anchor positions must not exceed text length.')
                if boundary > cursor:
                    out, state = self._run_delta_segment(query, key, value, beta, g, cursor, boundary, state)
                    text_outputs.append(out)
                    cursor = boundary

                state_for_anchor = zero_state if state is None else state
                anchor_states.append(state_for_anchor)
                anchor_outputs.append(torch.einsum(
                    'bhk,bhkv->bhv',
                    anchor_q[:, anchor_idx].to(state_for_anchor.dtype),
                    state_for_anchor,
                ) * (self.key_head_dim ** -0.5))

            if cursor < text_len:
                initial_state = (state)
                out, state = self._run_delta_segment(query, key, value, beta, g, cursor, text_len, initial_state)
                text_outputs.append(out)

            text_o = torch.cat(text_outputs, dim=1) if text_outputs else query.new_zeros(batch, 0, heads, value_dim)
            anchor_o = torch.stack(anchor_outputs, dim=1).to(text_o.dtype) if anchor_outputs else query.new_zeros(
                batch, 0, heads, value_dim)
            states = torch.stack(anchor_states, dim=1) if anchor_states else query.new_zeros(
                batch, 0, heads, key_dim, value_dim)
        return text_o, anchor_o, states

    def _run_hsd_segments_packed_ref(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        anchor_cu_seqlens: torch.Tensor,
    ):
        if query.shape[0] != 1:
            raise NotImplementedError('MarchGatedDeltaNet packed reference path requires flattened batch size 1.')
        text_cu_cpu = text_cu_seqlens.detach().to(device='cpu', dtype=torch.long)
        anchor_cu_cpu = anchor_cu_seqlens.detach().to(device='cpu', dtype=torch.long)
        if text_cu_cpu.numel() != anchor_cu_cpu.numel():
            raise ValueError(
                'HSD packed text_cu_seqlens and anchor_cu_seqlens must have the same number of entries, '
                f'got {text_cu_cpu.numel()} and {anchor_cu_cpu.numel()}.')

        text_outputs = []
        anchor_outputs = []
        state_outputs = []
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            anchor_start = int(anchor_cu_cpu[doc_idx].item())
            anchor_end = int(anchor_cu_cpu[doc_idx + 1].item())
            local_boundaries = boundary_counts[anchor_start:anchor_end] - text_start
            text_o_i, anchor_o_i, states_i = self._run_hsd_segments_ref(
                query=query[:, text_start:text_end],
                key=key[:, text_start:text_end],
                value=value[:, text_start:text_end],
                beta=beta[:, text_start:text_end],
                g=g[:, text_start:text_end],
                anchor_q=anchor_q[:, anchor_start:anchor_end],
                boundary_counts=local_boundaries,
            )
            text_outputs.append(text_o_i)
            if anchor_end > anchor_start:
                anchor_outputs.append(anchor_o_i)
                state_outputs.append(states_i)

        batch, _, heads, key_dim = query.shape
        value_dim = value.shape[-1]
        text_o = torch.cat(text_outputs, dim=1) if text_outputs else query.new_zeros(batch, 0, heads, value_dim)
        anchor_o = (
            torch.cat(anchor_outputs, dim=1)
            if anchor_outputs else query.new_zeros(batch, 0, heads, value_dim)
        )
        states = (
            torch.cat(state_outputs, dim=1)
            if state_outputs else query.new_zeros(batch, 0, heads, key_dim, value_dim)
        )
        return text_o, anchor_o, states

    def _run_hsd_segments(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: Optional[torch.Tensor] = None,
        anchor_cu_seqlens: Optional[torch.Tensor] = None,
        nonpacked_layout_cache_source: Optional[torch.Tensor] = None,
        sp_padding_anchor_count: int = 0,
    ):
        packed = text_cu_seqlens is not None or anchor_cu_seqlens is not None
        # Honor explicit FlashQLA producer selection from the model config.
        if getattr(self, 'hsd_train_scan_backend', 'triton') == 'flashqla_tile':
            flashqla_eligible = (
                bool(getattr(self, 'hsd_train_selective_scan', False))
                and hsd_flashqla_tile_gated_delta_rule_with_anchor_states is not None
            )
            if packed and (text_cu_seqlens is None or anchor_cu_seqlens is None):
                raise ValueError(
                    'HSD packed FlashQLA path requires both text_cu_seqlens '
                    'and anchor_cu_seqlens.')
            outputs = None
            regular_shape = None
            flash_sp_padding_count = 0
            flash_query = None
            flash_anchor_q = None
            flash_boundaries = None
            flash_text_cu = None
            flash_anchor_cu = None
            if flashqla_eligible:
                if packed:
                    flash_query = query
                    flash_key = key
                    flash_value = value
                    flash_beta = beta
                    flash_g = g
                    flash_anchor_q = anchor_q
                    flash_boundaries = boundary_counts
                    flash_text_cu = text_cu_seqlens
                    flash_anchor_cu = anchor_cu_seqlens
                    flash_sp_padding_count = int(sp_padding_anchor_count)
                    if flash_sp_padding_count:
                        if flash_sp_padding_count < 0:
                            raise ValueError('HSD layout padding anchor count must be non-negative.')
                        anchor_count = int(anchor_q.shape[1])
                        if flash_sp_padding_count > anchor_count:
                            raise ValueError(
                                'HSD layout padding anchor count exceeds the available anchors: '
                                f'padding={flash_sp_padding_count}, anchors={anchor_count}.')
                        anchor_cu_end = int(anchor_cu_seqlens[-1].detach().cpu())
                        text_cu_end = int(text_cu_seqlens[-1].detach().cpu())
                        padding_boundaries = boundary_counts[
                            anchor_count - flash_sp_padding_count:
                        ].detach().to(device='cpu', dtype=torch.long)
                        if anchor_cu_end != anchor_count:
                            raise ValueError(
                                'HSD packed anchor cu-seqlens must span all anchors before SP padding is '
                                f'stripped: cu_end={anchor_cu_end}, anchors={anchor_count}.')
                        if not bool((padding_boundaries == text_cu_end).all().item()):
                            raise ValueError(
                                'HSD layout padding anchors must be a trailing suffix at the packed text end: '
                                f'padding_boundaries={padding_boundaries.tolist()}, text_end={text_cu_end}.')
                        # Fixed-M Stage-3 padding and sequence-parallel padding
                        # both append anchor-only slots at the absolute end of
                        # the packed stream. They are not semantic chunk
                        # anchors and must not enter the local recurrence.
                        # Strip exactly the explicitly reported suffix while
                        # retaining strict validation of every semantic anchor
                        # in the FlashQLA wrapper.
                        flash_anchor_q = anchor_q[:, :anchor_count - flash_sp_padding_count]
                        flash_boundaries = boundary_counts[:anchor_count - flash_sp_padding_count]
                        flash_anchor_cu = anchor_cu_seqlens.clone()
                        flash_anchor_cu[-1] = flash_anchor_cu[-1] - flash_sp_padding_count
                else:
                    if int(sp_padding_anchor_count) != 0:
                        raise ValueError(
                            'HSD layout padding anchors are only valid for model-side packed layouts.')
                    batch, text_len, heads, key_dim = query.shape
                    anchor_count = int(anchor_q.shape[1])
                    value_dim = int(value.shape[-1])
                    flash_boundaries, flash_text_cu, flash_anchor_cu = (
                        self._nonpacked_batch_sidecars(
                            boundary_counts,
                            batch=batch,
                            text_len=text_len,
                            anchor_count=anchor_count,
                            device=query.device,
                            layout_cache_source=nonpacked_layout_cache_source,
                        )
                    )
                    # Batch-major flattening keeps every sample contiguous and
                    # the sidecars above prevent state/history leakage between
                    # samples.  EOD tokens do not alter these two segments.
                    flash_query = query.reshape(1, batch * text_len, heads, key_dim)
                    flash_key = key.reshape(1, batch * text_len, heads, key_dim)
                    flash_value = value.reshape(1, batch * text_len, heads, value_dim)
                    flash_beta = beta.reshape(1, batch * text_len, heads)
                    flash_g = g.reshape(1, batch * text_len, heads)
                    flash_anchor_q = anchor_q.reshape(
                        1, batch * anchor_count, heads, key_dim)
                    regular_shape = (batch, text_len, anchor_count, heads, key_dim, value_dim)
                with self._profile_cuda_timer('run_hsd_segments_flashqla_tile'):
                    outputs = hsd_flashqla_tile_gated_delta_rule_with_anchor_states(
                        query=flash_query,
                        key=flash_key,
                        value=flash_value,
                        beta=flash_beta,
                        g=flash_g,
                        anchor_q=flash_anchor_q,
                        boundary_counts=flash_boundaries,
                        text_cu_seqlens=flash_text_cu,
                        anchor_cu_seqlens=flash_anchor_cu,
                        scale=self.key_head_dim ** -0.5,
                        backward_backend=self.hsd_flashqla_tile_bwd_backend,
                    )
                    if outputs is not None and flash_sp_padding_count:
                        text_o, anchor_o, states = outputs
                        value_dim = int(flash_value.shape[-1])
                        pad_anchor_o = anchor_o.new_zeros(
                            1, flash_sp_padding_count, anchor_o.shape[2], value_dim)
                        pad_states = states.new_zeros(
                            1,
                            flash_sp_padding_count,
                            states.shape[2],
                            states.shape[3],
                            states.shape[4],
                        )
                        outputs = (
                            text_o,
                            torch.cat((anchor_o, pad_anchor_o), dim=1),
                            torch.cat((states, pad_states), dim=1),
                        )
            if outputs is not None:
                self._increment_hsd_backend_counter('local_flashqla_hit')
                self._record_profile_scalar('run_hsd_segments_flashqla_tile_hit', 1.0)
                if regular_shape is None:
                    return outputs
                batch, text_len, anchor_count, heads, key_dim, value_dim = regular_shape
                text_o, anchor_o, states = outputs
                return (
                    text_o.reshape(batch, text_len, heads, value_dim),
                    anchor_o.reshape(batch, anchor_count, heads, value_dim),
                    states.reshape(batch, anchor_count, heads, key_dim, value_dim),
                )
            self._increment_hsd_backend_counter('local_flashqla_miss')
            self._record_profile_scalar('run_hsd_segments_flashqla_tile_miss', 1.0)
            if self.hsd_train_scan_strict:
                details = [
                    f'eligible={flashqla_eligible}',
                    f'sp_padding_anchor_count={flash_sp_padding_count}',
                    f'query_shape={tuple(query.shape)}',
                    f'query_is_cuda={query.is_cuda}',
                    f'query_device={query.device}',
                    f'query_dtype={query.dtype}',
                    f'key_shape={tuple(key.shape)}',
                    f'value_shape={tuple(value.shape)}',
                    f'beta_shape={tuple(beta.shape)}',
                    f'g_shape={tuple(g.shape)}',
                    f'anchor_q_shape={tuple(anchor_q.shape)}',
                    f'boundary_count={int(boundary_counts.numel())}',
                    f'flash_query_shape={None if flash_query is None else tuple(flash_query.shape)}',
                    f'flash_anchor_q_shape={None if flash_anchor_q is None else tuple(flash_anchor_q.shape)}',
                    f'flash_boundary_count={None if flash_boundaries is None else int(flash_boundaries.numel())}',
                    f'flash_text_cu_end={None if flash_text_cu is None else int(flash_text_cu[-1].detach().cpu())}',
                    f'flash_anchor_cu_end={None if flash_anchor_cu is None else int(flash_anchor_cu[-1].detach().cpu())}',
                ]
                if text_cu_seqlens is None or anchor_cu_seqlens is None:
                    details.append('packed_metadata=missing')
                else:
                    try:
                        text_cu_cpu = text_cu_seqlens.detach().to(
                            device='cpu', dtype=torch.long)
                        anchor_cu_cpu = anchor_cu_seqlens.detach().to(
                            device='cpu', dtype=torch.long)
                        boundary_cpu = boundary_counts.detach().to(
                            device='cpu', dtype=torch.long)
                        details.extend((
                            f'text_cu_count={int(text_cu_cpu.numel())}',
                            f'text_cu_end={int(text_cu_cpu[-1])}',
                            f'anchor_cu_count={int(anchor_cu_cpu.numel())}',
                            f'anchor_cu_end={int(anchor_cu_cpu[-1])}',
                        ))
                        if text_cu_cpu.numel() != anchor_cu_cpu.numel():
                            details.append('first_bad=cu_count_mismatch')
                        else:
                            first_bad = None
                            for doc_idx in range(int(text_cu_cpu.numel()) - 1):
                                text_start = int(text_cu_cpu[doc_idx])
                                text_end = int(text_cu_cpu[doc_idx + 1])
                                anchor_start = int(anchor_cu_cpu[doc_idx])
                                anchor_end = int(anchor_cu_cpu[doc_idx + 1])
                                expected_count = (text_end - text_start) // 512
                                actual_count = anchor_end - anchor_start
                                local_boundaries = (
                                    boundary_cpu[anchor_start:anchor_end] - text_start)
                                expected_boundaries = torch.arange(
                                    1,
                                    expected_count + 1,
                                    dtype=torch.long,
                                ) * 512
                                if (
                                    actual_count != expected_count
                                    or not torch.equal(local_boundaries, expected_boundaries)
                                ):
                                    first_bad = (
                                        f'doc={doc_idx},text=[{text_start},{text_end}),'
                                        f'length={text_end - text_start},'
                                        f'anchors=[{anchor_start},{anchor_end}),'
                                        f'actual_count={actual_count},'
                                        f'expected_count={expected_count},'
                                        f'local_boundaries={local_boundaries.tolist()},'
                                        f'expected_boundaries={expected_boundaries.tolist()}'
                                    )
                                    break
                            details.append(f'first_bad={first_bad or "none"}')
                    except Exception as exc:  # pragma: no cover - diagnostic only.
                        details.append(
                            f'packed_metadata_diagnostic_error={type(exc).__name__}:{exc}')
                raise RuntimeError(
                    'HSD FlashQLA Tile local selective-scan strict mode '
                    'requested a fastpath hit, but it missed; '
                    + '; '.join(details))
            self._increment_hsd_backend_counter('fallback')

        if packed:
            if text_cu_seqlens is None or anchor_cu_seqlens is None:
                raise ValueError('HSD packed segment path requires both text_cu_seqlens and anchor_cu_seqlens.')
            packed_fastpath_enabled = (
                bool(getattr(self, 'hsd_train_selective_scan', False))
                and bool(getattr(self, 'hsd_train_selective_scan_packed', False))
                and hsd_chunk_gated_delta_rule_with_anchor_states is not None
            )
            packed_strict = bool(getattr(self, 'hsd_train_selective_scan_packed_strict', False))
            if packed_fastpath_enabled:
                with self._profile_cuda_timer('run_hsd_segments_packed_fastpath'):
                    outputs = hsd_chunk_gated_delta_rule_with_anchor_states(
                        query=query,
                        key=key,
                        value=value,
                        beta=beta,
                        g=g,
                        anchor_q=anchor_q,
                        boundary_counts=boundary_counts,
                        text_cu_seqlens=text_cu_seqlens,
                        anchor_cu_seqlens=anchor_cu_seqlens,
                        scale=self.key_head_dim ** -0.5,
                    )
                if outputs is not None:
                    # This is the established Triton/FLA producer used as the
                    # validation reference.  Count successful dispatches as
                    # reference hits as well as true Python fallbacks, so a
                    # fail-closed reference-vs-new run can prove that every
                    # HSD layer actually exercised its requested baseline.
                    self._increment_hsd_backend_counter('local_reference_hit')
                    self._record_profile_scalar('run_hsd_segments_packed_fastpath_hit', 1.0)
                    return outputs
                self._record_profile_scalar('run_hsd_segments_packed_fastpath_miss', 1.0)
                if packed_strict:
                    raise RuntimeError(
                        'HSD packed selective-scan strict mode requested a fastpath hit, '
                        'but hsd_chunk_gdn returned None.')
            elif (
                bool(getattr(self, 'hsd_train_selective_scan', False))
                and bool(getattr(self, 'hsd_train_selective_scan_packed', False))
                and packed_strict
            ):
                raise RuntimeError(
                    'HSD packed selective-scan strict mode requested a fastpath hit, '
                    'but packed fastpath hit conditions were not met.')
            self._increment_hsd_backend_counter('local_reference_hit')
            self._record_profile_scalar('run_hsd_segments_packed_ref_hit', 1.0)
            with self._profile_cuda_timer('run_hsd_segments_packed_ref'):
                return self._run_hsd_segments_packed_ref(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    g=g,
                    anchor_q=anchor_q,
                    boundary_counts=boundary_counts,
                    text_cu_seqlens=text_cu_seqlens,
                    anchor_cu_seqlens=anchor_cu_seqlens,
                )

        if (
            bool(getattr(self, 'hsd_train_selective_scan', False))
            and 'full' == 'full'
            and not bool(False)
            and hsd_chunk_gated_delta_rule_with_anchor_states is not None
        ):
            with self._profile_cuda_timer('run_hsd_segments_fastpath'):
                outputs = hsd_chunk_gated_delta_rule_with_anchor_states(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    g=g,
                    anchor_q=anchor_q,
                    boundary_counts=boundary_counts,
                    scale=self.key_head_dim ** -0.5,
                )
            if outputs is not None:
                self._record_profile_scalar('run_hsd_segments_fastpath_hit', 1.0)
                return outputs

        self._increment_hsd_backend_counter('local_reference_hit')
        self._record_profile_scalar('run_hsd_segments_ref_hit', 1.0)
        return self._run_hsd_segments_ref(query, key, value, beta, g, anchor_q, boundary_counts)

    def _history_read(
        self,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        anchor_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_len: int,
        null_logits: Optional[torch.Tensor] = None,
        persistent_anchor_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, _, heads, _ = read_q.shape
        anchor_count = states.shape[1]
        value_dim = states.shape[-1]
        if anchor_count == 0:
            return read_q.new_zeros(batch, text_len, heads, value_dim)

        if router_q.shape[:2] != read_q.shape[:2]:
            raise RuntimeError(
                f'HSD router query shape mismatch: got {tuple(router_q.shape[:2])}, '
                f'expected {tuple(read_q.shape[:2])}.'
            )
        route_q_heads = int(router_q.shape[2])
        route_key_heads = int(anchor_k.shape[2])
        if route_q_heads not in {1, heads}:
            raise RuntimeError(
                'HSD route query heads must be 1 or match value heads: '
                f'route_query_heads={route_q_heads}, value_heads={heads}.'
            )
        if heads % route_key_heads != 0:
            raise RuntimeError(
                'HSD anchor route key heads must divide value heads: '
                f'route_key_heads={route_key_heads}, value_heads={heads}.'
            )
        if router_q.shape[-1] != anchor_k.shape[-1]:
            raise RuntimeError(
                'HSD route q/k dim mismatch: '
                f'router_q dim={router_q.shape[-1]}, anchor_k dim={anchor_k.shape[-1]}.'
            )
        assert states.shape[:3] == (batch, anchor_count, heads), (states.shape, read_q.shape)
        assert anchor_k.shape[:2] == (batch, anchor_count), (anchor_k.shape, states.shape)
        if null_logits is not None and tuple(null_logits.shape) != (batch, text_len, heads, 1):
            raise RuntimeError(
                'HSD null read logits shape mismatch: '
                f'got {tuple(null_logits.shape)}, expected {(batch, text_len, heads, 1)}.')

        route_scale = self.hsd_route_head_dim ** -0.5 if self.hsd_use_route_scale else 1.0
        route_scale = route_scale * self.hsd_route_logit_scale
        read_scale = self.key_head_dim ** -0.5
        boundaries = boundary_counts.to(device=read_q.device, dtype=torch.long)
        token_index = torch.arange(text_len, device=read_q.device, dtype=torch.long)
        if self._hsd_debug_enabled():
            self._record_debug_scalar('router_use_route_scale', float(self.hsd_use_route_scale))
            self._record_debug_scalar('router_route_logit_scale', float(self.hsd_route_logit_scale))
            self._record_debug_scalar('router_effective_logit_scale', float(route_scale))
            self._record_debug_scalar('router_key_head_dim', float(self.hsd_route_head_dim))
            self._record_debug_scalar('router_query_heads', float(route_q_heads))
            self._record_debug_scalar('router_key_heads', float(route_key_heads))
            self._record_debug_scalar('router_qk_l2norm_enabled', float(self.hsd_route_qk_l2norm))
        route_q = router_q.float()
        route_keys = anchor_k.float()
        if route_q_heads == heads and route_key_heads == heads:
            logits = torch.einsum('bthk,bmhk->bthm', route_q, route_keys)
        elif route_q_heads == heads and route_key_heads == 1:
            logits = torch.einsum('bthk,bmk->bthm', route_q, route_keys[:, :, 0])
        elif route_q_heads == heads:
            grouped_keys = self._route_keys_for_query_heads(route_keys, heads)
            logits = torch.einsum('bthk,bmhk->bthm', route_q, grouped_keys)
        elif route_key_heads == heads:
            logits = torch.einsum('btk,bmhk->bthm', route_q[:, :, 0], route_keys)
        elif route_key_heads == 1:
            shared_logits = torch.einsum('btk,bmk->btm', route_q[:, :, 0], route_keys[:, :, 0])
            logits = shared_logits.view(batch, text_len, 1, anchor_count).expand(batch, text_len, heads, anchor_count)
        else:
            grouped_keys = self._route_keys_for_query_heads(route_keys, heads)
            logits = torch.einsum('btk,bmhk->bthm', route_q[:, :, 0], grouped_keys)
        logits = logits * route_scale
        self._tensor_debug_stats('route_logits', logits)
        self._register_debug_grad('route_logits', logits)
        valid_2d = boundaries.view(1, anchor_count) <= token_index.view(text_len, 1)
        valid = valid_2d.view(1, text_len, 1, anchor_count).expand(batch, text_len, heads, anchor_count)
        router_debug_logits = logits.masked_fill(~valid, float('-inf'))

        memory_index = None
        memory_slot_mask = None
        if persistent_anchor_slots is not None:
            persistent_anchor_slots = persistent_anchor_slots.to(device=logits.device, dtype=torch.bool)
            if persistent_anchor_slots.ndim != 1 or int(persistent_anchor_slots.numel()) != anchor_count:
                raise ValueError(
                    'persistent_anchor_slots must be [anchor_count], got '
                    f'{tuple(persistent_anchor_slots.shape)} for {anchor_count} anchors.')
            memory_indices = torch.nonzero(persistent_anchor_slots, as_tuple=False).flatten()
            if memory_indices.numel() != 1:
                raise ValueError(
                    f'Each history-read segment requires exactly one persistent memory, got {memory_indices.numel()}.')
            if self.persistent_memory_state is None:
                raise RuntimeError('Persistent-memory route is active but this layer has no memory state.')
            memory_index = int(memory_indices[0].item())
            memory_slot_mask = persistent_anchor_slots.view(1, 1, 1, anchor_count)

        topk = self.hsd_read_topk
        if getattr(self, 'hsd_topk_backend', 'reference') == 'sparse' and topk > 0:
            if memory_index is not None:
                return self._history_read_sparse_topk_with_persistent_memory_from_logits(
                    read_q=read_q,
                    states=states,
                    logits=logits,
                    valid=valid,
                    valid_prefix=valid_2d,
                    anchor_k=anchor_k,
                    boundaries=boundaries,
                    token_index=token_index,
                    null_logits=null_logits,
                    read_scale=read_scale,
                    memory_index=memory_index,
                )
            return self._history_read_sparse_topk_from_logits(
                read_q=read_q,
                states=states,
                logits=logits,
                valid=valid,
                valid_prefix=valid_2d,
                anchor_k=anchor_k,
                boundaries=boundaries,
                token_index=token_index,
                null_logits=null_logits,
                read_scale=read_scale,
            )
        if memory_slot_mask is not None:
            dynamic_valid = valid & ~memory_slot_mask
            if topk > 0:
                select_k = min(int(topk), anchor_count)
                _, topk_idx = logits.masked_fill(~dynamic_valid, float('-inf')).topk(k=select_k, dim=-1)
                topk_mask = torch.zeros_like(logits, dtype=torch.bool)
                topk_mask.scatter_(-1, topk_idx, True)
                dynamic_valid = dynamic_valid & topk_mask
            valid = dynamic_valid | (valid & memory_slot_mask)
        elif 0 < topk < anchor_count:
            _, topk_idx = logits.masked_fill(~valid, float('-inf')).topk(k=topk, dim=-1)
            topk_mask = torch.zeros_like(logits, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_idx, True)
            valid = valid & topk_mask

        null_prob = None
        anchor_mass = None
        debug_weights = None
        if null_logits is not None:
            null_valid = torch.ones(batch, text_len, heads, 1, device=valid.device, dtype=torch.bool)
            logits_with_null = torch.cat([logits, null_logits.float()], dim=-1)
            valid_with_null = torch.cat([valid, null_valid], dim=-1)
            weights_with_null = _route_softmax_weights(logits_with_null, valid_with_null, dim=-1)
            weights = weights_with_null[..., :anchor_count].to(states.dtype)
            null_prob = weights_with_null[..., anchor_count].to(states.dtype)
            anchor_mass = weights.sum(dim=-1, keepdim=True)
            debug_weights = torch.where(
                anchor_mass > 0,
                weights / anchor_mass.clamp_min(torch.finfo(weights.dtype).tiny),
                torch.zeros_like(weights),
            )
        else:
            weights = _route_softmax_weights(logits, valid, dim=-1).to(states.dtype)
            debug_weights = weights
        self._tensor_debug_stats('route_weights', weights)
        self._register_debug_grad('route_weights', weights)
        if null_prob is not None:
            self._tensor_debug_stats('route_null_prob', null_prob)
            self._register_debug_grad('route_null_prob', null_prob)
        has_valid = valid_2d.any(dim=-1).view(1, text_len, 1, 1)
        self._record_hsd_router_debug_stats(
            weights=debug_weights,
            logits=router_debug_logits,
            anchor_k=anchor_k,
            valid_prefix=valid_2d,
            has_valid=has_valid,
            boundaries=boundaries,
            token_index=token_index,
            null_prob=null_prob,
            anchor_mass=anchor_mass,
        )
        if bool(getattr(self, 'hsd_block_history_read', False)):
            dynamic_weights = weights
            if memory_index is not None:
                dynamic_weights = weights.clone()
                dynamic_weights[..., memory_index] = 0
            history = _blocked_history_read_output(
                read_q=read_q,
                states=states,
                weights=dynamic_weights,
                read_scale=read_scale,
                block_m=getattr(self, 'hsd_history_read_block_m', 4),
                block_t=getattr(self, 'hsd_history_read_block_t', 0),
            )
        else:
            dynamic_weights = weights
            if memory_index is not None:
                dynamic_weights = weights.clone()
                dynamic_weights[..., memory_index] = 0
            reads = torch.einsum('bthk,bmhkv->bthmv', read_q.to(states.dtype), states) * read_scale
            reads = reads.masked_fill(~valid.unsqueeze(-1), 0)
            history = torch.einsum('bthm,bthmv->bthv', dynamic_weights, reads)

        if memory_index is not None:
            memory_prob = weights[..., memory_index]
            memory_read = self._persistent_memory_read(read_q, read_scale=read_scale)
            weighted_memory_read = memory_read * memory_prob.float().unsqueeze(-1)
            self._record_persistent_memory_route_stats(
                memory_logits=logits[..., memory_index],
                memory_prob=memory_prob,
                weighted_memory_read=weighted_memory_read,
            )
            history = history + weighted_memory_read.to(dtype=history.dtype)
        return history

    def _should_dispatch_history_read(self, use_global_read: bool, anchor_count: int) -> bool:
        return bool(use_global_read) and int(anchor_count) > 0

    def _history_read_sparse_topk_from_logits(
        self,
        *,
        read_q: torch.Tensor,
        states: torch.Tensor,
        logits: torch.Tensor,
        valid: torch.Tensor,
        valid_prefix: torch.Tensor,
        anchor_k: torch.Tensor,
        boundaries: torch.Tensor,
        token_index: torch.Tensor,
        null_logits: Optional[torch.Tensor],
        read_scale: float,
    ) -> torch.Tensor:
        """Select routes densely, then materialize only K selected state reads."""
        if hsd_sparse_topk_global_read is None:
            raise RuntimeError('HSD sparse top-k global-read helper is unavailable.')
        batch, text_len, heads, anchor_count = logits.shape
        topk = int(self.hsd_read_topk)
        if topk not in {1, 2, 4, 8, 16}:
            raise RuntimeError(f'HSD sparse top-k supports K in {{1, 2, 4, 8, 16}}, got {topk}.')

        masked_logits = logits.masked_fill(~valid, float('-inf'))
        # Preserve K as the kernel bucket before K anchors exist. Invalid
        # padded slots use safe state index zero and exactly zero probability.
        if anchor_count < topk:
            pad = topk - anchor_count
            ranked_logits = F.pad(masked_logits, (0, pad), value=float('-inf'))
            ranked_valid = F.pad(valid, (0, pad), value=False)
        else:
            ranked_logits = masked_logits
            ranked_valid = valid
        selected_logits, selected_indices = ranked_logits.topk(k=topk, dim=-1)
        selected_valid = ranked_valid.gather(-1, selected_indices)
        safe_indices = torch.where(
            selected_valid,
            selected_indices.clamp(0, anchor_count - 1),
            torch.zeros_like(selected_indices),
        )
        has_valid = valid.any(dim=-1, keepdim=True)

        null_prob = None
        anchor_mass = None
        if null_logits is not None:
            if tuple(null_logits.shape) != (batch, text_len, heads, 1):
                raise RuntimeError(
                    'HSD null read logits shape mismatch: '
                    f'got {tuple(null_logits.shape)}, expected {(batch, text_len, heads, 1)}.')
            logits_with_null = torch.cat([selected_logits, null_logits.float()], dim=-1)
            valid_with_null = torch.cat(
                [selected_valid, torch.ones_like(null_logits, dtype=torch.bool)], dim=-1)
            weights_with_null = _route_softmax_weights(logits_with_null, valid_with_null, dim=-1)
            selected_weights = weights_with_null[..., :topk]
            null_prob = weights_with_null[..., topk]
            anchor_mass = selected_weights.sum(dim=-1, keepdim=True)
        else:
            stable_logits = torch.where(has_valid, selected_logits, torch.zeros_like(selected_logits))
            selected_weights = _route_softmax_weights(stable_logits, selected_valid, dim=-1)
        selected_weights = selected_weights * selected_valid.to(selected_weights.dtype)

        history = hsd_sparse_topk_global_read(
            read_q=read_q,
            states=states,
            topk_indices=safe_indices,
            topk_weights=selected_weights.float(),
            topk_valid=selected_valid,
            read_scale=float(read_scale),
        )

        if self._hsd_debug_enabled():
            conditional_weights = selected_weights
            if anchor_mass is not None:
                conditional_weights = torch.where(
                    anchor_mass > 0,
                    selected_weights / anchor_mass.clamp_min(torch.finfo(selected_weights.dtype).tiny),
                    torch.zeros_like(selected_weights),
                )
            debug_weights = torch.zeros_like(logits, dtype=torch.float32)
            debug_weights.scatter_add_(-1, safe_indices, conditional_weights.float())
            self._record_hsd_router_debug_stats(
                weights=debug_weights,
                logits=masked_logits,
                anchor_k=anchor_k,
                valid_prefix=valid_prefix,
                has_valid=has_valid,
                boundaries=boundaries,
                token_index=token_index,
                null_prob=null_prob,
                anchor_mass=anchor_mass,
            )
        return history

    def _history_read_sparse_topk_with_persistent_memory_from_logits(
        self,
        *,
        read_q: torch.Tensor,
        states: torch.Tensor,
        logits: torch.Tensor,
        valid: torch.Tensor,
        valid_prefix: torch.Tensor,
        anchor_k: torch.Tensor,
        boundaries: torch.Tensor,
        token_index: torch.Tensor,
        null_logits: Optional[torch.Tensor],
        read_scale: float,
        memory_index: int,
    ) -> torch.Tensor:
        """Sparse dynamic Top-K plus a reserved persistent-memory lane.

        The existing sparse kernel still materializes exactly K dynamic state
        reads. Memory is excluded before Top-K, participates in the same final
        softmax (with null), and is read separately from its FP32 parameter.
        """
        if hsd_sparse_topk_global_read is None:
            raise RuntimeError('HSD sparse top-k global-read helper is unavailable.')
        batch, text_len, heads, anchor_count = logits.shape
        topk = int(self.hsd_read_topk)
        if topk not in {1, 2, 4, 8, 16}:
            raise RuntimeError(f'HSD sparse top-k supports K in {{1, 2, 4, 8, 16}}, got {topk}.')
        if not 0 <= int(memory_index) < anchor_count:
            raise ValueError(f'Persistent memory index {memory_index} is outside {anchor_count} anchors.')

        dynamic_valid = valid.clone()
        dynamic_valid[..., memory_index] = False
        masked_dynamic_logits = logits.masked_fill(~dynamic_valid, float('-inf'))
        if anchor_count < topk:
            pad = topk - anchor_count
            ranked_logits = F.pad(masked_dynamic_logits, (0, pad), value=float('-inf'))
            ranked_valid = F.pad(dynamic_valid, (0, pad), value=False)
        else:
            ranked_logits = masked_dynamic_logits
            ranked_valid = dynamic_valid
        selected_logits, selected_indices = ranked_logits.topk(k=topk, dim=-1)
        selected_valid = ranked_valid.gather(-1, selected_indices)
        safe_indices = torch.where(
            selected_valid,
            selected_indices.clamp(0, anchor_count - 1),
            torch.zeros_like(selected_indices),
        )

        memory_logits = logits[..., memory_index:memory_index + 1]
        memory_valid = valid[..., memory_index:memory_index + 1]
        candidate_logits = torch.cat([memory_logits, selected_logits], dim=-1)
        candidate_valid = torch.cat([memory_valid, selected_valid], dim=-1)
        null_prob = None
        if null_logits is not None:
            if tuple(null_logits.shape) != (batch, text_len, heads, 1):
                raise RuntimeError(
                    'HSD null read logits shape mismatch: '
                    f'got {tuple(null_logits.shape)}, expected {(batch, text_len, heads, 1)}.')
            candidate_logits = torch.cat([candidate_logits, null_logits.float()], dim=-1)
            candidate_valid = torch.cat(
                [candidate_valid, torch.ones_like(null_logits, dtype=torch.bool)], dim=-1)
            candidate_weights = _route_softmax_weights(candidate_logits, candidate_valid, dim=-1)
            null_prob = candidate_weights[..., -1]
        else:
            candidate_weights = _route_softmax_weights(candidate_logits, candidate_valid, dim=-1)

        memory_prob = candidate_weights[..., 0] * memory_valid.squeeze(-1).to(candidate_weights.dtype)
        selected_weights = candidate_weights[..., 1:1 + topk]
        selected_weights = selected_weights * selected_valid.to(selected_weights.dtype)
        anchor_mass = memory_prob.unsqueeze(-1) + selected_weights.sum(dim=-1, keepdim=True)

        history = hsd_sparse_topk_global_read(
            read_q=read_q,
            states=states,
            topk_indices=safe_indices,
            topk_weights=selected_weights.float(),
            topk_valid=selected_valid,
            read_scale=float(read_scale),
        )
        memory_read = self._persistent_memory_read(read_q, read_scale=read_scale)
        weighted_memory_read = memory_read * memory_prob.float().unsqueeze(-1)
        history = history + weighted_memory_read.to(dtype=history.dtype)
        self._record_persistent_memory_route_stats(
            memory_logits=memory_logits.squeeze(-1),
            memory_prob=memory_prob,
            weighted_memory_read=weighted_memory_read,
        )

        if self._hsd_debug_enabled():
            conditional_memory = memory_prob
            conditional_dynamic = selected_weights
            conditional_memory = torch.where(
                anchor_mass.squeeze(-1) > 0,
                conditional_memory / anchor_mass.squeeze(-1).clamp_min(
                    torch.finfo(conditional_memory.dtype).tiny),
                torch.zeros_like(conditional_memory),
            )
            conditional_dynamic = torch.where(
                anchor_mass > 0,
                conditional_dynamic / anchor_mass.clamp_min(torch.finfo(conditional_dynamic.dtype).tiny),
                torch.zeros_like(conditional_dynamic),
            )
            debug_weights = torch.zeros_like(logits, dtype=torch.float32)
            debug_weights.scatter_add_(-1, safe_indices, conditional_dynamic.float())
            debug_weights[..., memory_index] += conditional_memory.float()
            has_valid = valid.any(dim=-1, keepdim=True)
            self._record_hsd_router_debug_stats(
                weights=debug_weights,
                logits=logits.masked_fill(~valid, float('-inf')),
                anchor_k=anchor_k,
                valid_prefix=valid_prefix,
                has_valid=has_valid,
                boundaries=boundaries,
                token_index=token_index,
                null_prob=null_prob,
                anchor_mass=anchor_mass,
            )
        return history

    def _history_read_packed(
        self,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        anchor_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        anchor_cu_seqlens: torch.Tensor,
        null_logits: Optional[torch.Tensor] = None,
        persistent_anchor_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if read_q.shape[0] != 1:
            raise NotImplementedError('MarchGatedDeltaNet packed history read requires flattened batch size 1.')
        if null_logits is not None and tuple(null_logits.shape[:3]) != tuple(read_q.shape[:3]):
            raise RuntimeError(
                'HSD packed null read logits shape mismatch: '
                f'got {tuple(null_logits.shape)}, expected prefix {tuple(read_q.shape[:3])}.')
        configured_global_read_backend = getattr(
            self,
            'hsd_fused_global_read_backend',
            'auto',
        )
        configured_backend = configured_global_read_backend != 'auto'
        if configured_backend:
            fused_backend = configured_global_read_backend
            fused_requested = True
            fused_strict = bool(getattr(self, 'hsd_fused_global_read_strict', False))
        else:
            # Auto selection retains environment overrides and FP32 route tensors.
            fused_backend = None
            fused_requested = _env_bool('HSD_FUSED_GLOBAL_READ', True)
            fused_strict = _env_bool('HSD_FUSED_GLOBAL_READ_STRICT', False)
        can_try_fused = (
            fused_requested
            and hsd_fused_global_read is not None
            and int(self.hsd_read_topk) == 0
            and persistent_anchor_slots is None
        )
        if can_try_fused:
            heads = int(read_q.shape[2])
            route_scale = self.hsd_route_head_dim ** -0.5 if self.hsd_use_route_scale else 1.0
            route_scale = route_scale * self.hsd_route_logit_scale
            fused_kwargs = {}
            if configured_backend:
                fused_kwargs['backend'] = fused_backend
                # Keep route Q/K in the model dtype at the kernel ABI.  The
                # online kernel performs its own FP32 accumulation; casting
                # here would miss the admitted profile and allocate a full-size
                # extra tensor.
                fused_router_q = self._route_queries_for_value_heads(router_q, heads)
                fused_anchor_k = self._route_keys_for_query_heads(anchor_k, heads)
            else:
                fused_router_q = self._route_queries_for_value_heads(router_q.float(), heads)
                fused_anchor_k = self._route_keys_for_query_heads(anchor_k.float(), heads)
            fused = hsd_fused_global_read(
                read_q=read_q,
                router_q=fused_router_q,
                states=states,
                anchor_k=fused_anchor_k,
                boundary_counts=boundary_counts,
                text_cu_seqlens=text_cu_seqlens,
                anchor_cu_seqlens=anchor_cu_seqlens,
                null_logits=null_logits,
                route_scale=route_scale,
                read_scale=self.key_head_dim ** -0.5,
                debug_stats_callback=(
                    self._record_hsd_router_debug_stats if self._hsd_debug_enabled() else None
                ),
                return_local_prob=False,
                **fused_kwargs,
            )
            if fused is not None:
                if (
                    configured_backend
                    and fused_backend in {'auto', 'reference'}
                    and int(self.hsd_read_topk) == 0
                ):
                    self._increment_hsd_backend_counter('dense_reference_hit')
                return fused
        if (
            fused_requested
            and fused_strict
            and int(self.hsd_read_topk) == 0
            and persistent_anchor_slots is None
        ):
            raise RuntimeError('HSD fused packed global-read strict mode requested a fastpath hit, but it missed.')
        if persistent_anchor_slots is not None:
            persistent_anchor_slots = persistent_anchor_slots.to(device=states.device, dtype=torch.bool)
            if persistent_anchor_slots.ndim != 1 or int(persistent_anchor_slots.numel()) != int(states.shape[1]):
                raise RuntimeError(
                    'persistent_anchor_slots must be [packed_anchor_count], got '
                    f'{tuple(persistent_anchor_slots.shape)} for {int(states.shape[1])} anchors.')
        if configured_backend and int(self.hsd_read_topk) == 0:
            if fused_backend in {'auto', 'reference'}:
                self._increment_hsd_backend_counter('dense_reference_hit')
            else:
                self._increment_hsd_backend_counter('fallback')
        text_cu_cpu = text_cu_seqlens.detach().to(device='cpu', dtype=torch.long)
        anchor_cu_cpu = anchor_cu_seqlens.detach().to(device='cpu', dtype=torch.long)
        outputs = []
        value_dim = states.shape[-1]
        for doc_idx in range(int(text_cu_cpu.numel()) - 1):
            text_start = int(text_cu_cpu[doc_idx].item())
            text_end = int(text_cu_cpu[doc_idx + 1].item())
            anchor_start = int(anchor_cu_cpu[doc_idx].item())
            anchor_end = int(anchor_cu_cpu[doc_idx + 1].item())
            doc_text_len = text_end - text_start
            if doc_text_len <= 0:
                continue
            if anchor_end <= anchor_start:
                outputs.append(read_q.new_zeros(1, doc_text_len, read_q.shape[2], value_dim))
                continue
            local_boundaries = boundary_counts[anchor_start:anchor_end] - text_start
            outputs.append(
                self._history_read(
                    read_q=read_q[:, text_start:text_end],
                    router_q=router_q[:, text_start:text_end],
                    states=states[:, anchor_start:anchor_end],
                    anchor_k=anchor_k[:, anchor_start:anchor_end],
                    boundary_counts=local_boundaries,
                    text_len=doc_text_len,
                    null_logits=None if null_logits is None else null_logits[:, text_start:text_end],
                    persistent_anchor_slots=(
                        None
                        if persistent_anchor_slots is None
                        else persistent_anchor_slots[anchor_start:anchor_end]
                    ),
                ))
        if not outputs:
            return read_q.new_zeros(read_q.shape[0], 0, read_q.shape[2], value_dim)
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [seq_len, batch, hidden_size]
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

        inference_context = deprecate_inference_params(inference_context, inference_params)
        anchor_mask = kwargs.pop('anchor_mask', None)
        kwargs.pop('has_anchor_tokens', None)
        text_cu_seqlens = kwargs.pop('text_cu_seqlens', None)
        anchor_cu_seqlens = kwargs.pop('anchor_cu_seqlens', None)
        persistent_memory_mask = kwargs.pop('persistent_memory_mask', None)
        sp_padding_anchor_count = int(kwargs.pop('sp_padding_anchor_count', 0))
        # An explicit all-false anchor mask is a valid M=0 HSD layout.  It
        # must still exercise the configured local backend (and its strict
        # contract) instead of silently becoming an ordinary parent GDN call.
        # Only the absence of HSD layout metadata selects the parent path;
        # ``has_anchor_tokens`` remains a compatibility hint, not a bypass.
        if anchor_mask is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                **kwargs,
            )

        del key_value_states, attention_bias, sequence_len_offset, kwargs
        self._check_supported_path(packed_seq_params, inference_context)
        packed_hsd = packed_seq_params is not None
        if packed_hsd:
            if text_cu_seqlens is None or anchor_cu_seqlens is None:
                raise ValueError('HSD packed forward requires text_cu_seqlens and anchor_cu_seqlens.')
        nonpacked_layout_cache_source = (
            anchor_mask
            if not packed_hsd and getattr(self, 'hsd_train_scan_backend', 'triton') == 'flashqla_tile'
            else None
        )

        local_seq_len, batch, _ = hidden_states.shape
        cp_size = self.config.context_parallel_size
        seq_len = local_seq_len * self.sp_size * cp_size
        cp_local_hidden_states = self._gather_full_hidden_states(hidden_states)
        if cp_size == 1:
            if cp_local_hidden_states.shape[0] != seq_len:
                raise RuntimeError(
                    f'HSD full hidden length {cp_local_hidden_states.shape[0]} '
                    f'does not match expected sequence length {seq_len}.')
            hidden_states_bld = cp_local_hidden_states.transpose(0, 1).contiguous()
        else:
            expected_cp_local_seq_len = seq_len // cp_size
            if cp_local_hidden_states.shape[0] != expected_cp_local_seq_len:
                raise RuntimeError(
                    f'HSD CP-local hidden length {cp_local_hidden_states.shape[0]} '
                    f'does not match expected CP-local sequence length {expected_cp_local_seq_len}.')
            hidden_states_bld = None
        anchor_mask = anchor_mask.to(device=hidden_states.device, dtype=torch.bool)
        if anchor_mask.shape != (batch, seq_len):
            raise ValueError(
                f'anchor_mask shape {tuple(anchor_mask.shape)} must match hidden states {(batch, seq_len)}.')
        if not torch.equal(anchor_mask, anchor_mask[:1].expand_as(anchor_mask)):
            raise NotImplementedError('MarchGatedDeltaNet requires the same anchor layout for every batch row.')
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2 and attention_mask.shape == anchor_mask.shape:
            if not bool(attention_mask.all().item()):
                raise NotImplementedError('MarchGatedDeltaNet does not support padded batches yet.')

        text_mask = ~anchor_mask
        text_len = int(text_mask[0].sum().item())
        if text_len == 0:
            raise ValueError('MarchGatedDeltaNet requires at least one text token.')
        anchor_count = int(anchor_mask[0].sum().item())
        if packed_hsd:
            if batch != 1:
                raise NotImplementedError('MarchGatedDeltaNet packed HSD requires flattened batch size 1.')
            text_cu_seqlens = self._validate_packed_cu_seqlens(
                text_cu_seqlens,
                expected_end=text_len,
                name='text_cu_seqlens',
            )
            anchor_cu_seqlens = self._validate_packed_cu_seqlens(
                anchor_cu_seqlens,
                expected_end=anchor_count,
                name='anchor_cu_seqlens',
            )
            if text_cu_seqlens.numel() != anchor_cu_seqlens.numel():
                raise ValueError(
                    'HSD packed text_cu_seqlens and anchor_cu_seqlens must describe the same documents.')
        persistent_anchor_slots = self._persistent_anchor_slots(
            anchor_mask=anchor_mask,
            persistent_memory_mask=persistent_memory_mask,
            anchor_cu_seqlens=anchor_cu_seqlens if packed_hsd else None,
        )
        if cp_size == 1:
            text_hidden = self._select_bld_masked(hidden_states_bld, text_mask)
            anchor_hidden = self._select_bld_masked(hidden_states_bld, anchor_mask)
            self._trace_precision_internal_tensor('hidden_states_text_before_proj', text_hidden)
            self._tensor_debug_stats('anchor_hidden', anchor_hidden)
            self._register_debug_grad('anchor_hidden', anchor_hidden)
        else:
            text_hidden = None
            anchor_hidden = None
        self._record_layout_debug_stats(
            local_seq_len=local_seq_len,
            full_seq_len=seq_len,
            full_hidden_seq_len=cp_local_hidden_states.shape[0],
            anchor_mask=anchor_mask,
            text_len=text_len,
            anchor_count=anchor_count,
        )
        num_key_heads_per_device = self.num_key_heads // self.tp_size // cp_size

        nvtx_range_push(suffix='in_proj')
        qkvzba = self._project_input(hidden_states, num_key_heads_per_device)
        nvtx_range_pop(suffix='in_proj')
        if qkvzba.shape[0] != batch or qkvzba.shape[1] != seq_len:
            raise RuntimeError(
                'HSD projected input sequence shape mismatch after TP/SP projection: '
                f'got {tuple(qkvzba.shape[:2])}, expected {(batch, seq_len)}.')
        self._trace_precision_internal_tensor('qkvzba_text_after_proj', self._select_bld_masked(qkvzba, text_mask))
        self._trace_precision_internal_backward_tensor(
            'qkvzba_text_after_proj_grad',
            qkvzba,
            transform=lambda grad: self._select_bld_masked(grad, text_mask),
        )
        self._record_layout_debug_stats(
            local_seq_len=local_seq_len,
            full_seq_len=seq_len,
            full_hidden_seq_len=cp_local_hidden_states.shape[0],
            anchor_mask=anchor_mask,
            text_len=text_len,
            anchor_count=anchor_count,
            projected_seq_len=qkvzba.shape[1],
        )

        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim * 2 + self.v_dim) // self.num_key_heads,
                self.v_dim // self.num_key_heads,
                self.num_value_heads // self.num_key_heads,
                self.num_value_heads // self.num_key_heads,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        self._trace_precision_internal_tensor('gate_text', self._select_bld_masked(gate, text_mask))
        qkv = self._select_bld_masked(qkv, text_mask).reshape(batch, text_len, -1)
        beta = self._select_bld_masked(beta, text_mask).reshape(batch, text_len, -1)
        alpha = self._select_bld_masked(alpha, text_mask).reshape(batch, text_len, -1)

        nvtx_range_push(suffix='conv1d')
        if causal_conv1d is None:
            raise ImportError('MarchGatedDeltaNet requires fla.modules.convolution.causal_conv1d.')
        assert self.activation in ['silu', 'swish']
        text_cu_for_kernels = (
            text_cu_seqlens.to(device=qkv.device)
            if packed_hsd and text_cu_seqlens.device != qkv.device else text_cu_seqlens
        )
        if cp_size > 1:
            conv1d_weight = get_parameter_local_cp(
                self.conv1d.weight,
                dim=0,
                cp_group=self.pg_collection.cp,
            )
            conv1d_bias = (
                get_parameter_local_cp(
                    self.conv1d.bias,
                    dim=0,
                    cp_group=self.pg_collection.cp,
                ) if self.conv_bias else None)
        else:
            conv1d_weight = self.conv1d.weight
            conv1d_bias = self.conv1d.bias
        qkv = causal_conv1d(
            x=qkv,
            weight=conv1d_weight.squeeze(1),
            bias=conv1d_bias,
            activation=self.activation,
            cu_seqlens=text_cu_for_kernels,
        )[0]
        nvtx_range_pop(suffix='conv1d')
        self._trace_precision_internal_tensor('qkvzba_text_after_conv', qkv)
        self._trace_precision_internal_backward_tensor('qkvzba_text_after_conv_grad', qkv)

        qkv = qkv.view(qkv.shape[:-1] + (num_key_heads_per_device, qkv.shape[-1] // num_key_heads_per_device))
        query, key, value = torch.split(
            qkv,
            [self.qk_dim // self.num_key_heads, self.qk_dim // self.num_key_heads, self.v_dim // self.num_key_heads],
            dim=-1,
        )
        query = query.reshape(batch, text_len, -1, self.key_head_dim)
        key = key.reshape(batch, text_len, -1, self.key_head_dim)
        value = value.reshape(batch, text_len, -1, self.value_head_dim)
        if self.use_qk_l2norm:
            query = l2norm(query.contiguous())
            key = l2norm(key.contiguous())
        if self.num_value_heads // self.num_key_heads > 1:
            query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
            key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
        if gate.shape[2] != query.shape[2]:
            raise RuntimeError(f'HSD gate head count mismatch: query={query.shape[2]}, gate={gate.shape[2]}.')

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous().sigmoid()
        alpha = alpha.contiguous()
        if cp_size > 1:
            A_log = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
            dt_bias = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=self.pg_collection.cp)
        else:
            A_log, dt_bias = self.A_log, self.dt_bias
        g = -A_log.float().exp() * F.softplus(alpha.float() + dt_bias)
        self._trace_precision_internal_tensor('q_text', query)
        self._trace_precision_internal_tensor('k_text', key)
        self._trace_precision_internal_tensor('v_text', value)
        self._trace_precision_internal_tensor('beta_text', beta)
        self._trace_precision_internal_tensor('g_text', g)
        self._trace_precision_internal_backward_tensor(
            'gate_text_grad',
            gate,
            transform=lambda grad: self._select_bld_masked(grad, text_mask),
        )
        self._trace_precision_internal_backward_tensor('q_text_grad', query)
        self._trace_precision_internal_backward_tensor('k_text_grad', key)
        self._trace_precision_internal_backward_tensor('v_text_grad', value)
        self._trace_precision_internal_backward_tensor('beta_text_grad', beta)
        self._trace_precision_internal_backward_tensor('g_text_grad', g)

        if cp_size > 1:
            anchor_q_full = self._linear_heads_from_local_sequence(self.anchor_q_proj, hidden_states)
            anchor_q = self._select_bld_masked(anchor_q_full, anchor_mask)
        else:
            anchor_q = self._linear_heads_bsh(self.anchor_q_proj, anchor_hidden)
        anchor_q = l2norm(anchor_q.contiguous())
        use_global_read = self.hsd_layer_uses_global_read
        anchor_k = None
        if use_global_read:
            if self.anchor_k_proj is None:
                raise RuntimeError('anchor_k_proj is required when HSD global read is enabled.')
            if cp_size > 1:
                anchor_k_full = self._linear_heads_from_local_sequence(
                    self.anchor_k_proj, hidden_states, self.hsd_route_head_dim)
                anchor_k = self._select_bld_masked(anchor_k_full, anchor_mask)
            else:
                anchor_key_hidden = anchor_hidden
                anchor_k = self._linear_heads_bsh(self.anchor_k_proj, anchor_key_hidden,
                                                   self.hsd_route_head_dim)
            anchor_k = self._replace_persistent_anchor_route_key(
                anchor_k,
                persistent_anchor_slots,
            )
            if self.hsd_route_qk_l2norm:
                anchor_k = l2norm(anchor_k.contiguous())

        if anchor_q.shape[2] != query.shape[2]:
            raise RuntimeError(
                f'HSD projection head count mismatch: query={query.shape[2]}, '
                f'anchor_q={anchor_q.shape[2]}.')
        if anchor_k is not None and query.shape[2] % anchor_k.shape[2] != 0:
            raise RuntimeError(
                f'HSD anchor route key heads must divide value heads: query={query.shape[2]}, '
                f'anchor_k={anchor_k.shape[2]}.')

        boundary_counts = self._anchor_text_counts(anchor_mask)
        route_boundary_counts = (
            self._packed_local_boundary_counts(boundary_counts, text_cu_seqlens, anchor_cu_seqlens)
            if packed_hsd else boundary_counts
        )
        if anchor_k is not None and self.hsd_route_position_encoding != 'none':
            anchor_positions = route_boundary_counts.to(device=anchor_k.device, dtype=torch.long).sub(1).clamp_min(0)
            anchor_k = self._apply_route_position_encoding(anchor_k, anchor_positions)
        self._trace_precision_internal_tensor('anchor_q', anchor_q)
        self._trace_precision_internal_tensor('anchor_k', anchor_k)
        self._tensor_debug_stats('anchor_q', anchor_q)
        self._register_debug_grad('anchor_q', anchor_q)
        if anchor_k is not None:
            self._tensor_debug_stats('anchor_k', anchor_k)
            self._register_debug_grad('anchor_k', anchor_k)

        nvtx_range_push(suffix='historical_anchor_gated_delta_rule')
        text_o, anchor_o, anchor_states = self._run_hsd_segments(
            query=query,
            key=key,
            value=value,
            beta=beta,
            g=g,
            anchor_q=anchor_q,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_for_kernels if packed_hsd else None,
            anchor_cu_seqlens=anchor_cu_seqlens,
            nonpacked_layout_cache_source=nonpacked_layout_cache_source,
            sp_padding_anchor_count=sp_padding_anchor_count,
        )
        nvtx_range_pop(suffix='historical_anchor_gated_delta_rule')
        anchor_o = self._apply_persistent_memory_anchor_output(
            anchor_o,
            anchor_q,
            persistent_anchor_slots,
        )
        self._record_persistent_memory_state_stats(anchor_states, persistent_anchor_slots)

        self._trace_precision_internal_tensor('text_o_before_history', text_o)
        self._trace_precision_internal_tensor('anchor_o', anchor_o)
        self._trace_precision_internal_tensor('anchor_states', anchor_states)
        if self._hsd_debug_verbose_enabled():
            self._record_debug_scalar('global_read_enabled', float(use_global_read))
            self._record_debug_scalar('null_read_gate_enabled', float(self.null_gate_proj is not None))
            self._record_debug_scalar(
                'history_out_grad_scale_schedule_enabled',
                float(self.hsd_history_out_grad_scale_schedule_enabled),
            )
            anchor_content_grad_scale = self._anchor_content_grad_scale_tensor(device=query.device)
            self._record_debug_scalar(
                'anchor_content_grad_scale',
                1.0 if anchor_content_grad_scale is None else anchor_content_grad_scale,
            )
            self._record_debug_scalar(
                'anchor_content_grad_scale_schedule_enabled',
                float(self.hsd_anchor_content_grad_scale_schedule_enabled),
            )
        if self._should_dispatch_history_read(use_global_read, int(anchor_states.shape[1])):
            if self.router_q_proj is None or anchor_k is None:
                raise RuntimeError('router_q_proj and anchor_k_proj are required when HSD global read is enabled.')
            if cp_size > 1:
                router_q_full = self._linear_heads_from_local_sequence(
                    self.router_q_proj, hidden_states, self.hsd_route_head_dim)
                router_q = self._select_bld_masked(router_q_full, text_mask)
                null_logits = self._null_gate_logits_from_local_sequence(hidden_states, text_mask)
            else:
                router_hidden = text_hidden
                router_q = self._linear_heads_bsh(self.router_q_proj, router_hidden, self.hsd_route_head_dim)
                null_logits = self._null_gate_logits(text_hidden)
            if router_q.shape[2] not in {1, query.shape[2]}:
                raise RuntimeError(
                    f'HSD router head count must be 1 or match value heads: '
                    f'query={query.shape[2]}, router_q={router_q.shape[2]}.')
            if null_logits is not None and null_logits.shape[2] != query.shape[2]:
                raise RuntimeError(
                    f'HSD null gate head count mismatch: query={query.shape[2]}, '
                    f'null_logits={null_logits.shape[2]}.')
            if self.hsd_route_qk_l2norm:
                router_q = l2norm(router_q.contiguous())
            if self.hsd_route_position_encoding != 'none':
                if packed_hsd:
                    text_positions = self._packed_local_positions_from_cu(
                        text_cu_seqlens,
                        text_len,
                        device=router_q.device,
                    )
                else:
                    text_positions = torch.arange(text_len, device=router_q.device, dtype=torch.long)
                router_q = self._apply_route_position_encoding(router_q, text_positions)
            read_q = self._build_history_read_query(query, text_hidden)
            self._trace_precision_internal_tensor('router_q', router_q)
            self._trace_precision_internal_tensor('read_q', read_q)
            self._trace_precision_internal_tensor('null_gate_logits', null_logits)
            self._register_debug_grad('null_gate_logits', null_logits)
            with self._profile_cuda_timer('history_read'):
                if packed_hsd:
                    history_out = self._history_read_packed(
                        read_q=read_q,
                        router_q=router_q,
                        states=anchor_states,
                        anchor_k=anchor_k,
                        boundary_counts=boundary_counts,
                        text_cu_seqlens=text_cu_seqlens,
                        anchor_cu_seqlens=anchor_cu_seqlens,
                        null_logits=null_logits,
                        persistent_anchor_slots=persistent_anchor_slots,
                    )
                else:
                    history_out = self._history_read(
                        read_q=read_q,
                        router_q=router_q,
                        states=anchor_states,
                        anchor_k=anchor_k,
                        boundary_counts=boundary_counts,
                        text_len=text_len,
                        null_logits=null_logits,
                        persistent_anchor_slots=persistent_anchor_slots,
                    )
            history_scale = self._history_scale_tensor(device=history_out.device)
            self._trace_precision_internal_tensor('history_out_text', history_out)
            history_out = self._apply_history_out_rmsnorm(history_out)
            if self.hsd_history_out_rmsnorm_weight is not None:
                self._trace_precision_internal_tensor('history_out_rmsnorm_text', history_out)
            self._trace_precision_internal_tensor(
                'history_scaled_text',
                (history_out * history_scale.to(dtype=history_out.dtype)).to(text_o.dtype),
            )
            self._record_hsd_history_debug_stats(
                text_o=text_o,
                history_out=history_out,
                history_scale=history_scale,
                anchor_states=anchor_states,
                router_q=router_q,
                anchor_k=anchor_k,
            )
            self._register_debug_grad('history_out', history_out)
            self._register_debug_grad('text_o_before_history', text_o)
            self._register_debug_grad('anchor_states', anchor_states)
            self._register_debug_grad('router_q', router_q)
            self._register_debug_grad('read_q', read_q)
            history_out_for_residual = self._apply_history_out_grad_scale(history_out)
            text_o = text_o + (
                history_out_for_residual * history_scale.to(dtype=history_out_for_residual.dtype)
            ).to(text_o.dtype)

        self._trace_precision_internal_tensor('delta_o_text', text_o)
        self._trace_precision_internal_backward_tensor('delta_o_text_grad', text_o)
        self._register_debug_grad('delta_o_text', text_o)

        value_heads = query.shape[2]
        core_attn_out = qkvzba.new_zeros(batch, seq_len, value_heads, self.value_head_dim)
        core_attn_out[text_mask] = text_o.reshape(-1, value_heads, self.value_head_dim)
        core_attn_out[anchor_mask] = anchor_o.reshape(-1, value_heads, self.value_head_dim)

        nvtx_range_push(suffix='gated_norm')
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix='gated_norm')

        norm_out = norm_out.reshape(batch, seq_len, -1)
        self._trace_precision_internal_tensor('gated_norm_text', self._select_bld_masked(norm_out, text_mask))
        self._trace_precision_internal_backward_tensor(
            'gated_norm_text_grad',
            norm_out,
            transform=lambda grad: self._select_bld_masked(grad, text_mask),
        )
        self._register_debug_grad(
            'gated_norm_text',
            norm_out,
            transform=lambda grad: self._select_bld_masked(grad, text_mask),
        )
        norm_out = norm_out.transpose(0, 1).contiguous()
        if cp_size > 1:
            norm_out = tensor_a2a_hp2cp(
                norm_out,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.pg_collection.cp,
            )
        text_seq_idx = torch.nonzero(text_mask[0].to(device=norm_out.device, dtype=torch.bool), as_tuple=False).flatten()
        self._trace_precision_internal_backward_tensor(
            'out_proj_input_grad',
            norm_out,
            transform=(
                None if cp_size > 1
                else lambda grad: grad.index_select(0, text_seq_idx.to(device=grad.device))
            ),
        )
        self._register_debug_grad(
            'out_proj_input_text',
            norm_out,
            transform=(
                None if cp_size > 1
                else lambda grad: grad.index_select(0, text_seq_idx.to(device=grad.device))
            ),
        )

        nvtx_range_push(suffix='out_proj')
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix='out_proj')
        local_anchor_mask, _ = self._local_anchor_mask(anchor_mask, out.shape[0], device=out.device)
        local_text_idx = torch.nonzero(~local_anchor_mask[0].to(dtype=torch.bool), as_tuple=False).flatten()
        local_out_text = out.index_select(0, local_text_idx.to(device=out.device))
        self._trace_precision_internal_tensor('out_proj_text', local_out_text)
        self._trace_precision_internal_tensor('attention_out_text', local_out_text)
        self._trace_precision_internal_backward_tensor(
            'attention_out_text_grad',
            out,
            transform=lambda grad: grad.index_select(0, local_text_idx.to(device=grad.device)),
        )
        self._register_debug_grad(
            'attention_out_text',
            out,
            transform=lambda grad: grad.index_select(0, local_text_idx.to(device=grad.device)),
        )
        local_anchor_idx = torch.nonzero(local_anchor_mask[0].to(dtype=torch.bool), as_tuple=False).flatten()
        self._trace_precision_internal_backward_tensor(
            'anchor_attention_out_grad',
            out,
            transform=lambda grad: grad.index_select(0, local_anchor_idx.to(device=grad.device)),
        )
        out = self._apply_anchor_content_grad_scale(out, local_anchor_mask)

        return out, out_bias
