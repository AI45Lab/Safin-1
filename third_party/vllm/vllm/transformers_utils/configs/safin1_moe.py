# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Safin-1-MoE configuration for HSD checkpoints.

This mirrors the HF remote-code config in ``march/safin1`` while
reusing vLLM's built-in Qwen3.5-MoE config implementation.
"""

from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)


def _reject_legacy_anchor_keys(values: dict) -> None:
    legacy = sorted(
        key for key in values
        if key.startswith("summary_") or key.startswith("hsd_summary_")
    )
    if legacy:
        raise ValueError(
            f"Unsupported pre-anchor Safin-1 config fields: {legacy}. "
            "Migrate the checkpoint before loading it with vLLM."
        )


class Safin1MoeTextConfig(Qwen3_5MoeTextConfig):
    model_type = "safin1_35b_a3b_text"

    base_model_tp_plan = Qwen3_5MoeTextConfig.base_model_tp_plan | {
        "layers.*.linear_attn.anchor_q_proj": "colwise",
        "layers.*.linear_attn.anchor_k_proj": "colwise",
        "layers.*.linear_attn.router_q_proj": "colwise",
        "layers.*.linear_attn.null_gate_proj": "colwise",
        "layers.*.hsd_anchor_mlp.linear_fc1": "colwise",
        "layers.*.hsd_anchor_mlp.linear_fc2": "rowwise",
    }

    def __init__(
        self,
        hist_chunk_size: int = 512,
        hist_read_block_size: int = 16384,
        hist_read_chunk_block_size: int = 16,
        hist_use_history_read: bool = True,
        hist_disable_cache: bool = False,
        hist_replace_linear_attn: bool = True,
        anchor_chunk_size: int = 512,
        anchor_token_num: int = 1,
        anchor_placeholder_token_id: int = 0,
        anchor_bos_token_id: int | None = 248045,
        anchor_position_ids_type: str = "zeros",
        anchor_embedding_type: str = "learned",
        anchor_embedding_init_from: str = "bos",
        anchor_embedding_init_scale: float | None = 0.5,
        hsd_read_block_size: int = 32768,
        hsd_read_chunk_block_size: int = 16,
        hsd_use_global_read: bool = True,
        hsd_global_read_layers: str | list[int] | tuple[int, ...] | None = "10-30",
        hsd_use_route_scale: bool = True,
        hsd_route_logit_scale: float = 1.0,
        hsd_route_qk_l2norm: bool = False,
        hsd_route_position_encoding: str = "none",
        hsd_route_position_mode: str = "chunk_end",
        hsd_read_topk: int = 4,
        hsd_null_read_gate: bool = True,
        hsd_null_read_gate_init_bias: float = 0.0,
        hsd_history_scale: float = 1.0,
        hsd_history_out_rmsnorm: bool = False,
        hsd_history_out_rmsnorm_eps: float = 1e-6,
        hsd_learnable_history_scale: bool | None = None,
        hsd_learnable_history_scale_init: float | None = None,
        hsd_learnable_history_scale_max: float | None = None,
        hsd_history_gate_schedule_start: float | None = None,
        hsd_history_gate_schedule_end: float | None = None,
        hsd_history_gate_schedule_steps: int = 0,
        hsd_history_out_grad_scale: float | None = None,
        hsd_history_out_grad_scale_start: float | None = None,
        hsd_history_out_grad_scale_end: float | None = None,
        hsd_history_out_grad_scale_steps: int = 0,
        hsd_anchor_content_grad_scale: float | None = None,
        hsd_anchor_content_grad_scale_start: float | None = None,
        hsd_anchor_content_grad_scale_end: float | None = None,
        hsd_anchor_content_grad_scale_steps: int = 0,
        hsd_anchor_embedding_grad_scale: float | None = None,
        hsd_anchor_embedding_grad_scale_start: float | None = None,
        hsd_anchor_embedding_grad_scale_end: float | None = None,
        hsd_anchor_embedding_grad_scale_steps: int = 0,
        hsd_anchor_embedding_lr_mult: float | None = None,
        hsd_local_state_mode: str = "full",
        hsd_text_only_mlp: bool = False,
        hsd_text_only_in_proj: bool = False,
        hsd_anchor_isolated_in_proj: bool = False,
        hsd_text_single_pass_delta: bool = False,
        hsd_anchor_mlp: bool = False,
        hsd_anchor_mlp_hidden_size: int | None = None,
        **kwargs,
    ):
        _reject_legacy_anchor_keys(kwargs)
        super().__init__(**kwargs)
        if isinstance(self.rope_parameters, dict):
            self.rope_parameters = dict(self.rope_parameters)
            self.rope_parameters.pop("mrope_interleaved", None)
            self.rope_parameters.pop("mrope_section", None)

        self.hist_chunk_size = hist_chunk_size
        self.hist_read_block_size = hist_read_block_size
        self.hist_read_chunk_block_size = hist_read_chunk_block_size
        self.hist_use_history_read = hist_use_history_read
        self.hist_disable_cache = hist_disable_cache
        self.hist_replace_linear_attn = hist_replace_linear_attn

        self.anchor_chunk_size = anchor_chunk_size
        self.anchor_token_num = anchor_token_num
        self.anchor_placeholder_token_id = anchor_placeholder_token_id
        self.anchor_bos_token_id = anchor_bos_token_id
        self.anchor_position_ids_type = anchor_position_ids_type
        self.anchor_embedding_type = anchor_embedding_type
        self.anchor_embedding_init_from = anchor_embedding_init_from
        self.anchor_embedding_init_scale = anchor_embedding_init_scale

        self.hsd_read_block_size = hsd_read_block_size
        self.hsd_read_chunk_block_size = hsd_read_chunk_block_size
        self.hsd_use_global_read = hsd_use_global_read
        self.hsd_global_read_layers = hsd_global_read_layers
        self.hsd_use_route_scale = hsd_use_route_scale
        self.hsd_route_logit_scale = hsd_route_logit_scale
        self.hsd_route_qk_l2norm = hsd_route_qk_l2norm
        self.hsd_route_position_encoding = hsd_route_position_encoding
        self.hsd_route_position_mode = hsd_route_position_mode
        self.hsd_read_topk = hsd_read_topk
        self.hsd_null_read_gate = hsd_null_read_gate
        self.hsd_null_read_gate_init_bias = hsd_null_read_gate_init_bias
        self.hsd_history_scale = hsd_history_scale
        self.hsd_history_out_rmsnorm = hsd_history_out_rmsnorm
        self.hsd_history_out_rmsnorm_eps = hsd_history_out_rmsnorm_eps
        self.hsd_learnable_history_scale = hsd_learnable_history_scale
        self.hsd_learnable_history_scale_init = hsd_learnable_history_scale_init
        self.hsd_learnable_history_scale_max = hsd_learnable_history_scale_max
        self.hsd_history_gate_schedule_start = hsd_history_gate_schedule_start
        self.hsd_history_gate_schedule_end = hsd_history_gate_schedule_end
        self.hsd_history_gate_schedule_steps = hsd_history_gate_schedule_steps

        self.hsd_history_out_grad_scale = hsd_history_out_grad_scale
        self.hsd_history_out_grad_scale_start = hsd_history_out_grad_scale_start
        self.hsd_history_out_grad_scale_end = hsd_history_out_grad_scale_end
        self.hsd_history_out_grad_scale_steps = hsd_history_out_grad_scale_steps
        self.hsd_anchor_content_grad_scale = hsd_anchor_content_grad_scale
        self.hsd_anchor_content_grad_scale_start = hsd_anchor_content_grad_scale_start
        self.hsd_anchor_content_grad_scale_end = hsd_anchor_content_grad_scale_end
        self.hsd_anchor_content_grad_scale_steps = hsd_anchor_content_grad_scale_steps
        self.hsd_anchor_embedding_grad_scale = hsd_anchor_embedding_grad_scale
        self.hsd_anchor_embedding_grad_scale_start = hsd_anchor_embedding_grad_scale_start
        self.hsd_anchor_embedding_grad_scale_end = hsd_anchor_embedding_grad_scale_end
        self.hsd_anchor_embedding_grad_scale_steps = hsd_anchor_embedding_grad_scale_steps
        self.hsd_anchor_embedding_lr_mult = hsd_anchor_embedding_lr_mult

        self.hsd_local_state_mode = hsd_local_state_mode
        self.hsd_text_only_mlp = hsd_text_only_mlp
        self.hsd_text_only_in_proj = hsd_text_only_in_proj
        self.hsd_anchor_isolated_in_proj = hsd_anchor_isolated_in_proj
        self.hsd_text_single_pass_delta = hsd_text_single_pass_delta
        self.hsd_anchor_mlp = hsd_anchor_mlp
        self.hsd_anchor_mlp_hidden_size = hsd_anchor_mlp_hidden_size


class Safin1MoeConfig(Qwen3_5MoeConfig):
    model_type = "safin1_35b_a3b"
    sub_configs = {
        "vision_config": Qwen3_5MoeVisionConfig,
        "text_config": Safin1MoeTextConfig,
    }

    def __init__(self, *args, **kwargs):
        _reject_legacy_anchor_keys(kwargs)
        if isinstance(kwargs.get("text_config"), dict):
            _reject_legacy_anchor_keys(kwargs["text_config"])
        super().__init__(*args, **kwargs)
        if isinstance(self.text_config, dict):
            self.text_config = Safin1MoeTextConfig(**self.text_config)
        if isinstance(getattr(self, "vision_config", None), dict):
            self.vision_config = Qwen3_5MoeVisionConfig(**self.vision_config)

    def get_text_config(self, decoder: bool = False):
        return self.text_config


__all__ = ["Safin1MoeConfig", "Safin1MoeTextConfig"]
