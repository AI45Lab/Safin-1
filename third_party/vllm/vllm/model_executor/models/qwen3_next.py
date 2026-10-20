# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

from collections.abc import Iterable
from itertools import islice
import os

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.utils.torch_utils import direct_register_custom_op

from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]


def hsd_graph_v3_sidecar_copy_row0(
    source: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Copy one selected diagnostic row outside Inductor fusion."""
    requested_row = int(
        os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_ROW", "0")
    )
    row = min(max(requested_row, 0), int(source.shape[0]) - 1)
    target.copy_(source[row])


def hsd_graph_v3_sidecar_copy_row0_fake(
    source: torch.Tensor,
    target: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="hsd_graph_v3_sidecar_copy_row0",
    op_func=hsd_graph_v3_sidecar_copy_row0,
    mutates_args=["target"],
    fake_impl=hsd_graph_v3_sidecar_copy_row0_fake,
)


def capture_hsd_graph_v3_sidecar_row0(
    source: torch.Tensor,
    target: torch.Tensor,
) -> None:
    torch.ops.vllm.hsd_graph_v3_sidecar_copy_row0(source, target)


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None
        for projection in (
            self.gate,
            self.shared_expert_gate,
            getattr(self.shared_expert, "gate_up_proj", None),
            getattr(self.shared_expert, "down_proj", None),
        ):
            if projection is not None:
                projection._hsd_moe_batch_invariant_linear = True
                projection._hsd_moe_batch_invariant_component = "moe_projection"

        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(orig_shape)


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.qkv_proj._hsd_moe_batch_invariant_linear = True
        self.o_proj._hsd_moe_batch_invariant_linear = True
        self.qkv_proj._hsd_moe_batch_invariant_component = (
            "attention_projection"
        )
        self.o_proj._hsd_moe_batch_invariant_component = "attention_projection"

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._hsd_graph_v3_tensor_sidecars: dict[str, torch.Tensor] = {}
        self._hsd_prefill_passive_sidecars: dict[str, torch.Tensor] = {}
        self._hsd_prefill_passive_meta: dict[str, int] = {}
        self._hsd_prefill_passive_layer_num = extract_layer_index(prefix) + 1

    def _capture_hsd_prefill_passive_sidecar(
        self,
        stage: str,
        tensor: torch.Tensor,
    ) -> None:
        if not os.environ.get("VLLM_HSD_PREFILL_PASSIVE_SIDECAR_DIR"):
            return
        target_tokens = os.environ.get(
            "VLLM_HSD_PREFILL_PASSIVE_SIDECAR_TARGET_TOKENS"
        )
        if target_tokens is not None and int(tensor.shape[0]) != int(
            target_tokens
        ):
            return
        selected_layers = os.environ.get(
            "VLLM_HSD_PREFILL_PASSIVE_SIDECAR_LAYERS"
        )
        if selected_layers:
            layer_set = {
                int(part)
                for part in selected_layers.split(",")
                if part.strip().isdigit()
            }
            if layer_set and self._hsd_prefill_passive_layer_num not in layer_set:
                return
        selected_stages = os.environ.get(
            "VLLM_HSD_PREFILL_PASSIVE_SIDECAR_STAGES"
        )
        if selected_stages:
            stage_set = {
                part.strip()
                for part in selected_stages.split(",")
                if part.strip()
            }
            if stage not in stage_set:
                return
        from vllm.distributed import get_tensor_model_parallel_rank

        tp_rank = int(get_tensor_model_parallel_rank())
        selected_ranks = os.environ.get(
            "VLLM_HSD_PREFILL_PASSIVE_SIDECAR_TP_RANKS"
        )
        if selected_ranks:
            rank_set = {
                int(part)
                for part in selected_ranks.split(",")
                if part.strip()
            }
            if rank_set and tp_rank not in rank_set:
                return
        self._hsd_prefill_passive_sidecars[stage] = tensor.detach().clone()
        self._hsd_prefill_passive_meta = {
            "past_text_len": 0,
            "text_len": int(tensor.shape[0]),
        }

    def prepare_hsd_graph_v3_tensor_sidecars(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Allocate diagnostic destinations before Dynamo/CUDA graph capture."""
        if not os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_DIR"):
            return
        selected_layers = os.environ.get(
            "VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_LAYERS"
        )
        if selected_layers:
            layer_set = {
                int(part)
                for part in selected_layers.split(",")
                if part.strip().isdigit()
            }
            if layer_set and self._hsd_prefill_passive_layer_num not in layer_set:
                return
        selected_stages = os.environ.get(
            "VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_STAGES"
        )
        stage_set = (
            {
                part.strip()
                for part in selected_stages.split(",")
                if part.strip()
            }
            if selected_stages
            else None
        )
        qkv_size = self.q_size * (2 if self.attn_output_gate else 1) + 2 * self.kv_size
        widths = {
            "full_attn_qkv": qkv_size,
            "full_attn_q_raw": self.q_size,
            "full_attn_k_raw": self.kv_size,
            "full_attn_q_norm": self.q_size,
            "full_attn_k_norm": self.kv_size,
            "full_attn_gate": self.q_size,
            "full_attn_q_rope": self.q_size,
            "full_attn_k_rope": self.kv_size,
            "full_attn_v": self.kv_size,
            "full_attn_core_out": self.q_size,
            "full_attn_pre_o": self.q_size,
            "full_attn_o_proj": self.hidden_size,
        }
        for stage, width in widths.items():
            if stage_set is not None and stage not in stage_set:
                continue
            shape = (int(width),)
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

    def _capture_hsd_graph_v3_tensor_sidecar(
        self,
        stage: str,
        tensor: torch.Tensor,
    ) -> None:
        if not os.environ.get("VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_DIR"):
            return
        selected_layers = os.environ.get(
            "VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_LAYERS"
        )
        if selected_layers:
            layer_set = {
                int(part)
                for part in selected_layers.split(",")
                if part.strip().isdigit()
            }
            if layer_set and self._hsd_prefill_passive_layer_num not in layer_set:
                return
        selected_stages = os.environ.get(
            "VLLM_HSD_GRAPH_V3_TENSOR_SIDECAR_STAGES"
        )
        if selected_stages:
            stage_set = {
                part.strip()
                for part in selected_stages.split(",")
                if part.strip()
            }
            if stage not in stage_set:
                return
        target = self._hsd_graph_v3_tensor_sidecars.get(stage)
        if target is None:
            if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "hsd_graph_v3_full_attention_sidecar_missing_before_capture: "
                    f"stage={stage} shape={tuple(tensor.shape)}"
                )
            target = torch.empty_like(tensor[0])
            self.register_buffer(
                f"_hsd_graph_v3_tensor_sidecar_{stage}",
                target,
                persistent=False,
            )
            self._hsd_graph_v3_tensor_sidecars[stage] = target
        capture_hsd_graph_v3_sidecar_row0(tensor, target)

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        capture_hsd_graph_v3_sidecar: bool = False,
        hsd_graph_v3_layer_name: object | None = None,
        hsd_prefix_pad_rows: int = 0,
    ):
        prefix_pad_rows = int(hsd_prefix_pad_rows)
        if prefix_pad_rows < 0:
            raise ValueError(
                "hsd_prefix_pad_rows must be non-negative, got "
                f"{prefix_pad_rows}."
            )
        pad_prefix_shape = prefix_pad_rows > 0
        projection_hidden_states = hidden_states
        if pad_prefix_shape:
            projection_hidden_states = torch.cat(
                [
                    hidden_states.new_zeros(
                        (prefix_pad_rows, hidden_states.shape[-1])
                    ),
                    hidden_states,
                ],
                dim=0,
            )
        qkv, _ = self.qkv_proj(projection_hidden_states)
        current_rows = slice(prefix_pad_rows, None) if pad_prefix_shape else slice(None)
        self._capture_hsd_prefill_passive_sidecar(
            "full_attn_qkv", qkv[current_rows]
        )
        if capture_hsd_graph_v3_sidecar:
            self._capture_hsd_graph_v3_tensor_sidecar("full_attn_qkv", qkv)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        self._capture_hsd_prefill_passive_sidecar(
            "full_attn_q_raw", q[current_rows]
        )
        self._capture_hsd_prefill_passive_sidecar(
            "full_attn_k_raw", k[current_rows]
        )
        self._capture_hsd_prefill_passive_sidecar(
            "full_attn_v", v[current_rows]
        )
        if capture_hsd_graph_v3_sidecar:
            self._capture_hsd_graph_v3_tensor_sidecar("full_attn_q_raw", q)
            self._capture_hsd_graph_v3_tensor_sidecar("full_attn_k_raw", k)
            self._capture_hsd_graph_v3_tensor_sidecar("full_attn_v", v)
            if self.attn_output_gate:
                self._capture_hsd_graph_v3_tensor_sidecar(
                    "full_attn_gate", gate
                )

        q_view = q.view(-1, self.num_heads, self.head_dim)
        k_view = k.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q_view).view(-1, self.num_heads * self.head_dim)
        k = self.k_norm(k_view).view(-1, self.num_kv_heads * self.head_dim)
        self._capture_hsd_prefill_passive_sidecar(
            "full_attn_q_norm", q[current_rows]
        )
        self._capture_hsd_prefill_passive_sidecar(
            "full_attn_k_norm", k[current_rows]
        )
        if capture_hsd_graph_v3_sidecar:
            self._capture_hsd_graph_v3_tensor_sidecar("full_attn_q_norm", q)
            self._capture_hsd_graph_v3_tensor_sidecar("full_attn_k_norm", k)

        rotary_positions = positions
        if pad_prefix_shape:
            rotary_positions = torch.cat(
                [
                    positions.new_zeros(
                        (*positions.shape[:-1], prefix_pad_rows)
                    ),
                    positions,
                ],
                dim=-1,
            )
        q, k = self.rotary_emb(rotary_positions, q, k)
        q = q[current_rows]
        k = k[current_rows]
        v = v[current_rows]
        if self.attn_output_gate:
            gate = gate[current_rows]
        self._capture_hsd_prefill_passive_sidecar("full_attn_q_rope", q)
        self._capture_hsd_prefill_passive_sidecar("full_attn_k_rope", k)
        if capture_hsd_graph_v3_sidecar:
            self._capture_hsd_graph_v3_tensor_sidecar("full_attn_q_rope", q)
            self._capture_hsd_graph_v3_tensor_sidecar("full_attn_k_rope", k)

        attn_output = self.attn(q, k, v)
        self._capture_hsd_prefill_passive_sidecar(
            "full_attn_core_out", attn_output
        )
        if capture_hsd_graph_v3_sidecar:
            self._capture_hsd_graph_v3_tensor_sidecar(
                "full_attn_core_out", attn_output
            )

        if self.attn_output_gate:
            if pad_prefix_shape:
                full_attn_output = torch.cat(
                    [
                        attn_output.new_zeros(
                            (prefix_pad_rows, attn_output.shape[-1])
                        ),
                        attn_output,
                    ],
                    dim=0,
                )
                full_gate = torch.cat(
                    [
                        gate.new_zeros((prefix_pad_rows, gate.shape[-1])),
                        gate,
                    ],
                    dim=0,
                )
                attn_output = (
                    full_attn_output * torch.sigmoid(full_gate)
                )[current_rows]
            else:
                gate = torch.sigmoid(gate)
                attn_output = attn_output * gate
        self._capture_hsd_prefill_passive_sidecar(
            "full_attn_pre_o", attn_output
        )
        if capture_hsd_graph_v3_sidecar:
            self._capture_hsd_graph_v3_tensor_sidecar(
                "full_attn_pre_o", attn_output
            )

        if pad_prefix_shape:
            full_attn_output = torch.cat(
                [
                    attn_output.new_zeros(
                        (prefix_pad_rows, attn_output.shape[-1])
                    ),
                    attn_output,
                ],
                dim=0,
            )
            full_output, _ = self.o_proj(full_attn_output)
            output.copy_(full_output[current_rows])
        else:
            output[:], _ = self.o_proj(attn_output)
        self._capture_hsd_prefill_passive_sidecar(
            "full_attn_o_proj", output
        )
        if capture_hsd_graph_v3_sidecar:
            self._capture_hsd_graph_v3_tensor_sidecar(
                "full_attn_o_proj", output
            )


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=getattr(self.config, "num_experts", 0),
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class QwenNextMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkvz"],
        "in_proj_ba": ["in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

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

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
