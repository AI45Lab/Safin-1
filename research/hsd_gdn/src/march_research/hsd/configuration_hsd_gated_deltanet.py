from __future__ import annotations

import math
import warnings

from transformers.configuration_utils import PretrainedConfig

from ..runtime_policy import sanitize_research_config


class HSDGatedDeltaNetConfig(PretrainedConfig):
    """Gated DeltaNet config with summary-token state routing controls.

    This mirrors the FLA GatedDeltaNet config fields without importing FLA at
    config-import time, so HSD data utilities can be used in CPU-only workers.
    """

    model_type = "hsd_gated_deltanet"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_v: float = 2.0,
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int | None = None,
        max_text_length: int | None = None,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 21,
        norm_eps: float = 1e-6,
        attn: dict | None = None,
        hsd_hybrid_summary_mode: str = "all_tokens",
        use_cache: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        initializer_type: str = "normal",
        initializer_cutoff_factor: float = 3.0,
        hsd_packed_document_isolation: bool = False,
        hsd_require_packed_seq_params: bool = False,
        hsd_fixed_summary_count: int = 0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        summary_chunk_size: int = 256,
        summary_token_num: int = 1,
        summary_placeholder_token_id: int = 0,
        summary_independent_parameters: bool = True,
        summary_embedding_type: str = "learned",
        summary_embedding_init_from: str = "random",
        summary_position_ids_type: str = "zeros",
        hsd_summary_q_mode: str = "independent",
        hsd_summary_q_lora_rank: int = 64,
        hsd_history_read_q_mode: str = "shared_local",
        hsd_history_read_q_lora_rank: int = 32,
        hsd_summary_key_source: str = "summary_hidden",
        hsd_summary_key_content_mode: str = "none",
        hsd_summary_key_content_scale_init: float = 0.0,
        hsd_read_block_size: int = 1024,
        hsd_read_chunk_block_size: int = 16,
        hsd_use_global_read: bool = True,
        hsd_use_route_scale: bool = True,
        hsd_route_logit_scale: float = 1.0,
        hsd_route_head_dim: int | None = None,
        hsd_route_query_heads: int | None = None,
        hsd_route_key_heads: int | None = None,
        hsd_router_q_local_lora_rank: int = 0,
        hsd_router_q_local_lora_source: str = "local_q",
        hsd_route_qk_l2norm: bool = False,
        hsd_route_position_encoding: str = "none",
        hsd_route_position_mode: str = "chunk_end",
        hsd_route_rope_fraction: float = 0.5,
        hsd_route_rope_theta: float = 10000.0,
        hsd_read_topk: int = 0,
        hsd_topk_backend: str = "reference",
        hsd_route_weight_momentum_gamma: float = 0.0,
        hsd_decode_momentum_init_mode: str = "carry",
        hsd_read_layers: list[int] | None = None,
        hsd_share_router_across_read_layers: bool = False,
        hsd_null_read_gate: bool = False,
        hsd_null_read_gate_init_bias: float = 0.0,
        hsd_null_logm_scale: float = 0.0,
        hsd_history_mix_mode: str = "additive",
        hsd_history_gate_floor_init: float = 0.0,
        hsd_history_gate_floor_warmup_steps: int = 0,
        hsd_history_aux_loss_weight: float = 0.0,
        hsd_history_head_scale: bool = False,
        hsd_local_core: str = "gdn",
        hsd_kda_gate_rank: int | None = None,
        hsd_state_decay_mode: str = "gated",
        hsd_history_state_mode: str = "continuous",
        hsd_history_candidate_mode: str = "snapshot",
        hsd_chunk_beta_mode: str = "shared",
        hsd_train_selective_scan: bool = True,
        hsd_train_scan_backend: str = "triton",
        hsd_train_scan_strict: bool = False,
        hsd_train_selective_scan_packed: bool | None = None,
        hsd_train_selective_scan_packed_strict: bool = False,
        hsd_flashqla_tile_bwd_backend: str = "optimized",
        hsd_fused_global_read_backend: str = "auto",
        hsd_fused_global_read_strict: bool = False,
        hsd_fused_global_read_profile: str | None = None,
        **kwargs,
    ):
        kwargs = sanitize_research_config({**kwargs, "vocab_size": vocab_size})
        kwargs.pop("vocab_size")
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.max_text_length = int(
            max_position_embeddings if max_text_length is None else max_text_length
        )
        self.max_position_embeddings = max_position_embeddings
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.hsd_hybrid_summary_mode = str(hsd_hybrid_summary_mode).strip().lower()
        self.use_cache = use_cache
        self.initializer_range = float(initializer_range)
        self.initializer_type = str(initializer_type).strip().lower()
        self.initializer_cutoff_factor = float(initializer_cutoff_factor)
        self.hsd_packed_document_isolation = bool(hsd_packed_document_isolation)
        self.hsd_require_packed_seq_params = bool(hsd_require_packed_seq_params)
        self.hsd_fixed_summary_count = int(hsd_fixed_summary_count)
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.attention_dropout = float(attention_dropout)
        self.hidden_dropout = float(hidden_dropout)
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.allow_neg_eigval = allow_neg_eigval

        self.summary_chunk_size = int(summary_chunk_size)
        self.summary_token_num = int(summary_token_num)
        self.summary_placeholder_token_id = int(summary_placeholder_token_id)
        self.summary_independent_parameters = bool(summary_independent_parameters)
        self.summary_embedding_type = summary_embedding_type
        self.summary_embedding_init_from = str(summary_embedding_init_from).strip().lower()
        self.summary_position_ids_type = summary_position_ids_type
        self.hsd_summary_q_mode = str(hsd_summary_q_mode).strip().lower()
        self.hsd_summary_q_lora_rank = int(hsd_summary_q_lora_rank)
        self.hsd_history_read_q_mode = str(hsd_history_read_q_mode).strip().lower()
        self.hsd_history_read_q_lora_rank = int(hsd_history_read_q_lora_rank)
        self.hsd_summary_key_source = str(hsd_summary_key_source).strip().lower()
        self.hsd_summary_key_content_mode = str(hsd_summary_key_content_mode).strip().lower()
        self.hsd_summary_key_content_scale_init = float(hsd_summary_key_content_scale_init)
        self.hsd_read_block_size = int(hsd_read_block_size)
        self.hsd_read_chunk_block_size = int(hsd_read_chunk_block_size)
        self.hsd_use_global_read = bool(hsd_use_global_read)
        self.hsd_use_route_scale = bool(hsd_use_route_scale)
        self.hsd_route_logit_scale = float(hsd_route_logit_scale)
        self.hsd_route_head_dim = None if hsd_route_head_dim is None else int(hsd_route_head_dim)
        self.hsd_route_query_heads = None if hsd_route_query_heads is None else int(hsd_route_query_heads)
        self.hsd_route_key_heads = None if hsd_route_key_heads is None else int(hsd_route_key_heads)
        self.hsd_router_q_local_lora_rank = int(hsd_router_q_local_lora_rank)
        self.hsd_router_q_local_lora_source = str(hsd_router_q_local_lora_source).strip().lower()
        self.hsd_route_qk_l2norm = bool(hsd_route_qk_l2norm)
        self.hsd_route_position_encoding = str(hsd_route_position_encoding).strip().lower()
        self.hsd_route_position_mode = str(hsd_route_position_mode).strip().lower()
        self.hsd_route_rope_fraction = float(hsd_route_rope_fraction)
        self.hsd_route_rope_theta = float(hsd_route_rope_theta)
        self.hsd_read_topk = int(hsd_read_topk)
        self.hsd_topk_backend = str(hsd_topk_backend).strip().lower()
        self.hsd_route_weight_momentum_gamma = float(hsd_route_weight_momentum_gamma)
        self.hsd_decode_momentum_init_mode = str(hsd_decode_momentum_init_mode).strip().lower()
        self.hsd_share_router_across_read_layers = bool(hsd_share_router_across_read_layers)
        self.hsd_null_read_gate = bool(hsd_null_read_gate)
        self.hsd_null_read_gate_init_bias = float(hsd_null_read_gate_init_bias)
        self.hsd_null_logm_scale = float(hsd_null_logm_scale)
        self.hsd_history_mix_mode = str(hsd_history_mix_mode).strip().lower()
        self.hsd_history_gate_floor_init = float(hsd_history_gate_floor_init)
        self.hsd_history_gate_floor_warmup_steps = int(hsd_history_gate_floor_warmup_steps)
        self.hsd_history_aux_loss_weight = float(hsd_history_aux_loss_weight)
        self.hsd_history_head_scale = bool(hsd_history_head_scale)
        self.hsd_local_core = str(hsd_local_core).strip().lower()
        self.hsd_kda_gate_rank = (
            int(self.head_dim * self.expand_v)
            if hsd_kda_gate_rank is None else int(hsd_kda_gate_rank)
        )
        self.hsd_state_decay_mode = str(hsd_state_decay_mode).strip().lower()
        self.hsd_history_state_mode = str(hsd_history_state_mode).strip().lower()
        self.hsd_history_candidate_mode = str(hsd_history_candidate_mode).strip().lower()
        self.hsd_chunk_beta_mode = str(hsd_chunk_beta_mode).strip().lower()
        self.hsd_train_selective_scan = bool(hsd_train_selective_scan)
        self.hsd_train_scan_backend = str(hsd_train_scan_backend).strip().lower()
        self.hsd_train_scan_strict = bool(hsd_train_scan_strict)
        self.hsd_train_selective_scan_packed = (
            bool(hsd_train_selective_scan)
            if hsd_train_selective_scan_packed is None else bool(hsd_train_selective_scan_packed)
        )
        self.hsd_train_selective_scan_packed_strict = bool(hsd_train_selective_scan_packed_strict)
        self.hsd_flashqla_tile_bwd_backend = str(hsd_flashqla_tile_bwd_backend).strip().lower()
        self.hsd_fused_global_read_backend = str(hsd_fused_global_read_backend).strip().lower()
        self.hsd_fused_global_read_strict = bool(hsd_fused_global_read_strict)
        self.hsd_fused_global_read_profile = (
            None
            if hsd_fused_global_read_profile is None
            else str(hsd_fused_global_read_profile).strip().lower()
        )
        self.hsd_read_layers = None
        if hsd_read_layers is not None:
            if not isinstance(hsd_read_layers, (list, tuple, set)):
                raise ValueError("hsd_read_layers must be a list, tuple, set, or None.")
            read_layers = sorted({int(layer_idx) for layer_idx in hsd_read_layers})
            invalid_layers = [layer_idx for layer_idx in read_layers if layer_idx < 0 or layer_idx >= self.num_hidden_layers]
            if invalid_layers:
                raise ValueError(
                    f"hsd_read_layers must be in [0, {self.num_hidden_layers}); got {invalid_layers}."
                )
            self.hsd_read_layers = read_layers
        if self.hsd_share_router_across_read_layers:
            if not self.hsd_use_global_read:
                raise ValueError(
                    "hsd_share_router_across_read_layers=True requires "
                    "hsd_use_global_read=True."
                )
            configured_read_layer_count = (
                self.num_hidden_layers
                if self.hsd_read_layers is None
                else len(self.hsd_read_layers)
            )
            if configured_read_layer_count < 2:
                raise ValueError(
                    "hsd_share_router_across_read_layers=True requires at least "
                    "two configured HSD read layers."
                )

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError("`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot both be True.")
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which improves memory efficiency "
                "at the potential cost of reduced precision.",
            )
        if attn is not None:
            if not isinstance(attn, dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['qkv_bias'] = attn.get('qkv_bias', False)
            attn['window_size'] = attn.get('window_size', None)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)
            attn['use_rope'] = bool(attn.get('use_rope', True))
        if self.hsd_hybrid_summary_mode not in {"all_tokens", "text_only"}:
            raise ValueError(
                "hsd_hybrid_summary_mode must be 'all_tokens' or 'text_only'."
            )

        if self.summary_chunk_size < 0:
            raise ValueError("summary_chunk_size must be non-negative.")
        if self.summary_token_num < 0:
            raise ValueError("summary_token_num must be non-negative.")
        if self.summary_embedding_type not in {"learned", "chunk_end"}:
            raise ValueError("summary_embedding_type must be either 'learned' or 'chunk_end'.")
        if self.summary_embedding_init_from != "random":
            raise ValueError(
                "summary_embedding_init_from currently only supports 'random'; "
                "BOS-based initialization is invalid for the OLMo tokenizer."
            )
        if self.summary_position_ids_type not in {"zeros", "chunk_end"}:
            raise ValueError("summary_position_ids_type must be either 'zeros' or 'chunk_end'.")
        if not self.summary_independent_parameters:
            raise NotImplementedError("HSD V1 requires summary_independent_parameters=True.")
        if self.hsd_summary_q_mode not in {"independent", "shared_read_q_lora"}:
            raise ValueError(
                "hsd_summary_q_mode must be either 'independent' or 'shared_read_q_lora'."
            )
        if self.hsd_summary_q_lora_rank <= 0:
            raise ValueError("hsd_summary_q_lora_rank must be positive.")
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
        if self.hsd_history_read_q_lora_rank <= 0:
            raise ValueError("hsd_history_read_q_lora_rank must be positive.")
        if self.hsd_summary_key_source not in {"summary_hidden", "state_probe"}:
            raise ValueError(
                "hsd_summary_key_source must be 'summary_hidden' or 'state_probe'."
            )
        if self.hsd_summary_key_content_mode not in {
            "none",
            "chunk_mean_residual",
            "chunk_mean_only",
        }:
            raise ValueError(
                "hsd_summary_key_content_mode must be 'none', "
                "'chunk_mean_residual', or 'chunk_mean_only'."
            )
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
        resolved_num_v_heads = self.num_heads if self.num_v_heads is None else self.num_v_heads
        if self.hsd_summary_q_mode == "shared_read_q_lora" and resolved_num_v_heads != self.num_heads:
            raise ValueError(
                "hsd_summary_q_mode='shared_read_q_lora' requires num_v_heads == num_heads "
                "because the shared q_proj output is indexed by GDN query heads."
            )
        if self.hsd_read_block_size <= 0:
            raise ValueError("hsd_read_block_size must be positive.")
        if self.hsd_read_chunk_block_size <= 0:
            raise ValueError("hsd_read_chunk_block_size must be positive.")
        if self.hsd_route_logit_scale <= 0:
            raise ValueError("hsd_route_logit_scale must be positive.")
        if self.hsd_route_head_dim is not None and self.hsd_route_head_dim <= 0:
            raise ValueError("hsd_route_head_dim must be positive or None.")
        if self.hsd_route_query_heads is not None and self.hsd_route_query_heads <= 0:
            raise ValueError("hsd_route_query_heads must be positive or None.")
        if self.hsd_route_key_heads is not None and self.hsd_route_key_heads <= 0:
            raise ValueError("hsd_route_key_heads must be positive or None.")
        if self.hsd_router_q_local_lora_rank < 0:
            raise ValueError("hsd_router_q_local_lora_rank must be non-negative.")
        if self.hsd_router_q_local_lora_source not in {"local_q", "history_read_q"}:
            raise ValueError(
                "hsd_router_q_local_lora_source must be 'local_q' or 'history_read_q'."
            )
        if self.hsd_route_query_heads is not None and self.hsd_route_query_heads not in {
            1, resolved_num_v_heads
        }:
            raise ValueError("hsd_route_query_heads must be 1, num_v_heads, or None.")
        resolved_route_query_heads = (
            resolved_num_v_heads
            if self.hsd_route_query_heads is None
            else self.hsd_route_query_heads
        )
        if (
            self.hsd_use_global_read
            and self.hsd_router_q_local_lora_rank > 0
            and resolved_route_query_heads != self.num_heads
        ):
            raise ValueError(
                "hsd_router_q_local_lora_rank > 0 requires the router query-head "
                "count to equal num_heads, because it adapts each local-Q head."
            )
        if (
            self.hsd_route_key_heads is not None
            and resolved_num_v_heads % self.hsd_route_key_heads != 0
        ):
            raise ValueError(
                "hsd_route_key_heads must divide num_v_heads so each route key is shared "
                "by an equal query-head group."
            )
        self._validate_hsd_route_position()
        if self.hsd_read_topk < 0:
            raise ValueError("hsd_read_topk must be non-negative.")
        if self.hsd_topk_backend not in {"reference", "sparse"}:
            raise ValueError("hsd_topk_backend must be either 'reference' or 'sparse'.")
        if self.hsd_topk_backend == "sparse" and self.hsd_read_topk not in {0, 1, 2, 4, 8, 16}:
            raise ValueError("Sparse HSD top-k supports hsd_read_topk in {1, 2, 4, 8, 16} (or 0 when dormant).")
        if (
            not math.isfinite(self.hsd_route_weight_momentum_gamma)
            or not 0.0 <= self.hsd_route_weight_momentum_gamma < 1.0
        ):
            raise ValueError("hsd_route_weight_momentum_gamma must be finite and in [0, 1).")
        if self.hsd_route_weight_momentum_gamma > 0.0 and self.hsd_read_topk != 0:
            raise ValueError("HSD route-weight momentum requires hsd_read_topk=0.")
        if self.hsd_decode_momentum_init_mode not in {"carry", "last_prompt_raw"}:
            raise ValueError(
                "hsd_decode_momentum_init_mode must be either 'carry' or 'last_prompt_raw'."
            )
        if not math.isfinite(self.hsd_null_read_gate_init_bias):
            raise ValueError("hsd_null_read_gate_init_bias must be finite.")
        if not math.isfinite(self.hsd_null_logm_scale) or self.hsd_null_logm_scale < 0.0:
            raise ValueError("hsd_null_logm_scale must be a non-negative finite value.")
        valid_history_mix_modes = {"additive", "convex", "joint_softmax_logm", "gated_additive"}
        if self.hsd_history_mix_mode not in valid_history_mix_modes:
            raise ValueError(
                "hsd_history_mix_mode must be 'additive', 'convex', "
                "'joint_softmax_logm', or 'gated_additive'."
            )
        if self.hsd_history_mix_mode in {"convex", "joint_softmax_logm", "gated_additive"} \
                and not self.hsd_null_read_gate:
            raise ValueError(f"hsd_history_mix_mode={self.hsd_history_mix_mode!r} requires hsd_null_read_gate=True.")
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
        if (
            not math.isfinite(self.hsd_history_aux_loss_weight)
            or self.hsd_history_aux_loss_weight < 0.0
        ):
            raise ValueError("hsd_history_aux_loss_weight must be a non-negative finite value.")
        if self.hsd_history_aux_loss_weight > 0.0:
            if not self.hsd_use_global_read:
                raise ValueError(
                    "A positive hsd_history_aux_loss_weight requires hsd_use_global_read=True."
                )
            if not self.hsd_null_read_gate or self.hsd_history_mix_mode != "gated_additive":
                raise ValueError(
                    "A positive hsd_history_aux_loss_weight requires hsd_null_read_gate=True "
                    "and hsd_history_mix_mode='gated_additive'."
                )
        if self.hsd_history_head_scale and self.hsd_history_mix_mode not in {
            "additive",
            "gated_additive",
        }:
            raise ValueError(
                "hsd_history_head_scale=True requires hsd_history_mix_mode to be "
                "'additive' or 'gated_additive'."
            )
        if self.hsd_local_core not in {"gdn", "kda", "gdn2"}:
            raise ValueError("hsd_local_core must be 'gdn', 'kda', or 'gdn2'.")
        if self.hsd_kda_gate_rank <= 0:
            raise ValueError("hsd_kda_gate_rank must be positive.")
        if self.hsd_local_core == "kda":
            if self.head_dim > 256:
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
                raise ValueError(
                    "HSD-GDN2 V1 requires num_heads=num_v_heads=6."
                )
            if self.head_dim > 256:
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
        if self.hsd_state_decay_mode not in {"gated", "none"}:
            raise ValueError("hsd_state_decay_mode must be 'gated' or 'none'.")
        if self.hsd_history_state_mode not in {"continuous", "chunk_reset_no_decay"}:
            raise ValueError(
                "hsd_history_state_mode must be 'continuous' or 'chunk_reset_no_decay'."
            )
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
        if self.hsd_chunk_beta_mode not in {"shared", "independent"}:
            raise ValueError("hsd_chunk_beta_mode must be 'shared' or 'independent'.")
        if self.hsd_chunk_beta_mode == "independent" and self.hsd_history_state_mode != "chunk_reset_no_decay":
            raise ValueError(
                "hsd_chunk_beta_mode='independent' requires "
                "hsd_history_state_mode='chunk_reset_no_decay'."
            )
        if self.hsd_train_scan_backend not in {"triton", "flashqla_tile"}:
            raise ValueError(
                "hsd_train_scan_backend must be 'triton' or 'flashqla_tile'."
            )
        if self.hsd_train_scan_backend == "flashqla_tile":
            if self.hsd_local_core != "gdn":
                raise ValueError(
                    "hsd_train_scan_backend='flashqla_tile' currently requires "
                    "hsd_local_core='gdn'."
                )
            if self.hsd_history_state_mode != "continuous":
                raise ValueError(
                    "hsd_train_scan_backend='flashqla_tile' currently requires "
                    "hsd_history_state_mode='continuous'."
                )
        valid_flashqla_bwd_backends = {
            "optimized", "reference", "native", "split", "direct_a"
        }
        if self.hsd_flashqla_tile_bwd_backend not in valid_flashqla_bwd_backends:
            raise ValueError(
                "hsd_flashqla_tile_bwd_backend must be one of 'optimized', "
                "'reference', 'native', 'split', or 'direct_a'."
            )
        valid_fused_global_read_backends = {
            "auto", "reference", "triton", "tile", "tile_speed", "tile_online", "flashqla_tile"
        }
        if self.hsd_fused_global_read_backend not in valid_fused_global_read_backends:
            raise ValueError(
                "hsd_fused_global_read_backend must be one of 'auto', 'reference', "
                "'triton', 'tile', 'tile_speed', 'tile_online', or 'flashqla_tile'."
            )
        if self.hsd_fused_global_read_profile == "":
            raise ValueError("hsd_fused_global_read_profile must be non-empty when provided.")

        if self.initializer_type not in {"normal", "truncated_normal"}:
            raise ValueError("initializer_type must be 'normal' or 'truncated_normal'.")
        if not math.isfinite(self.initializer_range) or self.initializer_range <= 0.0:
            raise ValueError("initializer_range must be a positive finite value.")
        if not math.isfinite(self.initializer_cutoff_factor) or self.initializer_cutoff_factor <= 0.0:
            raise ValueError("initializer_cutoff_factor must be a positive finite value.")
        if self.max_text_length <= 0:
            raise ValueError("max_text_length must be positive.")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive.")
        if self.max_text_length > self.max_position_embeddings:
            raise ValueError(
                "max_text_length cannot exceed max_position_embeddings: "
                f"{self.max_text_length} > {self.max_position_embeddings}."
            )
        for dropout_name, dropout_value in (
            ("attention_dropout", self.attention_dropout),
            ("hidden_dropout", self.hidden_dropout),
        ):
            if not math.isfinite(dropout_value) or not 0.0 <= dropout_value < 1.0:
                raise ValueError(f"{dropout_name} must be finite and in [0, 1).")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def summary_enabled(self) -> bool:
        return self.summary_chunk_size > 0 and self.summary_token_num > 0


    @property
    def hsd_route_rope_dim(self) -> int:
        route_head_dim = self.head_dim if self.hsd_route_head_dim is None else self.hsd_route_head_dim
        return 2 * math.floor(route_head_dim * self.hsd_route_rope_fraction / 2)

    def _validate_hsd_route_position(self) -> None:
        if self.hsd_route_position_encoding not in {"none", "rope"}:
            raise ValueError("hsd_route_position_encoding must be either 'none' or 'rope'.")
        if self.hsd_route_position_mode != "chunk_end":
            raise ValueError("hsd_route_position_mode currently only supports 'chunk_end'.")
        if not math.isfinite(self.hsd_route_rope_fraction) or not 0 < self.hsd_route_rope_fraction <= 1:
            raise ValueError("hsd_route_rope_fraction must be finite and in (0, 1].")
        if not math.isfinite(self.hsd_route_rope_theta) or self.hsd_route_rope_theta <= 0:
            raise ValueError("hsd_route_rope_theta must be a positive finite value.")
        if self.hsd_route_position_encoding == "rope":
            route_head_dim = self.head_dim if self.hsd_route_head_dim is None else self.hsd_route_head_dim
            rope_dim = self.hsd_route_rope_dim
            if rope_dim < 2 or rope_dim > route_head_dim:
                raise ValueError(
                    "Enabling route RoPE requires 2 <= hsd_route_rope_dim <= route head dim; "
                    f"got rope_dim={rope_dim}, route_head_dim={route_head_dim}."
                )


__all__ = ["HSDGatedDeltaNetConfig"]
