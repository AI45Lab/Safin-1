"""Shared Safin-1 runtime fields and validation for standalone HF remote code.

The anchor_/hsd_* fields own layout/history dimensions. Legacy dimension aliases,
retired options and the old query LoRA rank remain serialized. The three HF
control switches keep their historical names and remain active. Saved values
are never rewritten to the shared defaults.
"""
from dataclasses import dataclass
from typing import ClassVar

from .runtime_validation import validate_safin1_runtime


def _reject_legacy_anchor_fields(config) -> None:
    legacy = sorted(name for name in vars(config)
                    if name.startswith("summary_") or name.startswith("hsd_summary_"))
    if legacy:
        raise ValueError(
            f"Unsupported pre-anchor Safin-1 config fields: {legacy}. "
            "Run scripts/march/migrate_checkpoint.py on a checkpoint copy."
        )


class Safin1ConfigValidationMixin:
    def validate_anchor_schema(self) -> None:
        _reject_legacy_anchor_fields(self)
        validate_safin1_runtime(getattr(self, "text_config", self))


@dataclass(kw_only=True)
class Safin1RuntimeFields(Safin1ConfigValidationMixin):
    """Identical MARCH fields inherited by dense and MoE text configurations."""


    # Compatibility aliases. Explicit checkpoint values are preserved.
    hist_chunk_size: int = 512
    hist_read_block_size: int = 32768
    hist_read_chunk_block_size: int = 16
    hsd_history_read_q_lora_rank: int = 32  # Independent query LoRA is retired.

    # Active HF controls with historical names, distinct from dimension aliases.
    hist_use_history_read: bool = True
    hist_disable_cache: bool = False
    hist_replace_linear_attn: bool = True

    # Memory-anchor layout and embeddings.
    anchor_chunk_size: int = 512
    anchor_token_num: int = 1
    anchor_placeholder_token_id: int = 0
    anchor_bos_token_id: int | None = 248045
    anchor_position_ids_type: str = "zeros"
    anchor_embedding_type: str = "learned"
    anchor_embedding_init_from: str = "bos"
    anchor_embedding_init_scale: float | None = 0.5

    # Active history-read, routing and persistent-state runtime.
    hsd_read_block_size: int = 32768
    hsd_read_chunk_block_size: int = 16
    hsd_use_global_read: bool = True
    hsd_global_read_layers: str | list[int] | tuple[int, ...] | None = "10-30"
    hsd_history_read_q_mode: str = "shared_local"
    hsd_full_attention_anchor_tokens: bool = False
    hsd_use_route_scale: bool = True
    hsd_route_logit_scale: float = 1.0
    hsd_route_head_dim: int | None = None
    hsd_route_query_heads: int | None = None
    hsd_route_key_heads: int | None = None
    hsd_route_qk_l2norm: bool = False
    hsd_route_position_encoding: str = "none"
    hsd_route_position_mode: str = "chunk_end"
    hsd_read_topk: int = 4
    hsd_null_read_gate: bool = True
    hsd_null_read_gate_init_bias: float = 0.0
    hsd_persistent_memory_enabled: bool = False
    hsd_persistent_memory_route_key_enabled: bool = False
    hsd_persistent_memory_route_key_mode: str = "legacy"
    hsd_history_scale: float = 1.0
    hsd_history_out_rmsnorm: bool = False
    hsd_history_out_rmsnorm_eps: float = 1e-6
    hsd_learnable_history_scale: bool | None = None
    hsd_learnable_history_scale_init: float | None = None
    hsd_learnable_history_scale_max: float | None = None
    hsd_history_gate_schedule_start: float | None = None
    hsd_history_gate_schedule_end: float | None = None
    hsd_history_gate_schedule_steps: int = 0

    # Training metadata retained for reproducibility.
    hsd_history_out_grad_scale: float | None = None
    hsd_history_out_grad_scale_start: float | None = None
    hsd_history_out_grad_scale_end: float | None = None
    hsd_history_out_grad_scale_steps: int = 0
    hsd_anchor_content_grad_scale: float | None = None
    hsd_anchor_content_grad_scale_start: float | None = None
    hsd_anchor_content_grad_scale_end: float | None = None
    hsd_anchor_content_grad_scale_steps: int = 0
    hsd_anchor_embedding_grad_scale: float | None = None
    hsd_anchor_embedding_grad_scale_start: float | None = None
    hsd_anchor_embedding_grad_scale_end: float | None = None
    hsd_anchor_embedding_grad_scale_steps: int = 0
    hsd_anchor_embedding_lr_mult: float | None = None

    # Fixed supported modes and inactive retired architecture metadata.
    hsd_local_state_mode: str = "full"
    hsd_text_only_mlp: bool = False
    hsd_text_only_in_proj: bool = False
    hsd_anchor_isolated_in_proj: bool = False
    hsd_text_single_pass_delta: bool = False
    hsd_anchor_mlp: bool = False
    hsd_anchor_mlp_hidden_size: int | None = None
    HSD_TRAINING_ONLY_FIELDS: ClassVar[tuple[str, ...]] = (
        "hsd_history_out_grad_scale",
        "hsd_history_out_grad_scale_start",
        "hsd_history_out_grad_scale_end",
        "hsd_history_out_grad_scale_steps",
        "hsd_anchor_content_grad_scale",
        "hsd_anchor_content_grad_scale_start",
        "hsd_anchor_content_grad_scale_end",
        "hsd_anchor_content_grad_scale_steps",
        "hsd_anchor_embedding_grad_scale",
        "hsd_anchor_embedding_grad_scale_start",
        "hsd_anchor_embedding_grad_scale_end",
        "hsd_anchor_embedding_grad_scale_steps",
        "hsd_anchor_embedding_lr_mult",
    )

    HSD_COMPATIBILITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "hist_chunk_size", "hist_read_block_size", "hist_read_chunk_block_size",
        "hsd_history_read_q_lora_rank",
    )
    HSD_HF_CONTROL_FIELDS: ClassVar[tuple[str, ...]] = (
        "hist_use_history_read", "hist_disable_cache", "hist_replace_linear_attn",
    )
    HSD_RETIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "hsd_text_only_mlp", "hsd_text_only_in_proj", "hsd_anchor_isolated_in_proj",
        "hsd_text_single_pass_delta", "hsd_anchor_mlp", "hsd_anchor_mlp_hidden_size",
        "hsd_full_attention_anchor_tokens",
    )
