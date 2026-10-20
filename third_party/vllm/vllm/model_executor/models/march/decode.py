# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request cached recurrence and shared decode projections/admission.

The attention object owns all parameters, fused-weight caches and state tensors.
This mixin has no constructor. Batched decode routing and anchor/cache lifecycle
remain in the model; these methods retain the original statement order.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)

from .diagnostics import _hsd_trace_requested
from .layout import HistoricalHSDLayout
from .runtime import _env_truthy, _is_valid_hsd_state_index


class CachedDecodeMixin:
    def _apply_hsd_decode_gate_norm(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if _env_truthy("VLLM_HSD_BATCH_INVARIANT_GATE_NORM"):
            return self.norm.forward_cuda_batch_invariant(x, gate)
        return self.norm(x, gate)

    def _fused_decode_projections(
        self,
        hidden_states: torch.Tensor,
        *,
        include_history: bool,
        add_null_bias: bool = True,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
        | None
    ):
        if (
            not self._hsd_decode_projection_fusion_supported
            or not _env_truthy("VLLM_HSD_ENABLE_DECODE_PROJECTION_FUSION")
            or _env_truthy("VLLM_HSD_DISABLE_DECODE_PROJECTION_FUSION")
        ):
            return None

        modules: list[nn.Module] = [self.in_proj_qkvz, self.in_proj_ba]
        if self.router_q_proj is not None:
            modules.append(self.router_q_proj)
            if self.null_gate_proj is not None:
                modules.append(self.null_gate_proj)
        weights: list[torch.Tensor] = []
        for module in modules:
            weight = getattr(module, "weight", None)
            if (
                not isinstance(weight, torch.Tensor)
                or weight.ndim != 2
                or weight.shape[1] != hidden_states.shape[-1]
                or weight.device != hidden_states.device
                or weight.dtype != hidden_states.dtype
                or getattr(module, "bias", None) is not None
            ):
                return None
            weights.append(weight)

        sizes = tuple(int(weight.shape[0]) for weight in weights)
        fused_weight = self._hsd_decode_fused_proj_weight
        if (
            fused_weight is None
            or self._hsd_decode_fused_proj_sizes != sizes
            or fused_weight.device != hidden_states.device
            or fused_weight.dtype != hidden_states.dtype
        ):
            fused_weight = torch.cat([weight.detach() for weight in weights], dim=0)
            self._hsd_decode_fused_proj_weight = fused_weight
            self._hsd_decode_fused_proj_sizes = sizes

        output_sizes = sizes if include_history else sizes[:2]
        fused_weight = fused_weight[: sum(output_sizes)]
        if _env_truthy("VLLM_HSD_BATCH_INVARIANT_PROJECTIONS"):
            projected = linear_batch_invariant(
                hidden_states,
                fused_weight,
            )
        else:
            projected = F.linear(hidden_states, fused_weight)
        pieces = projected.split(output_sizes, dim=-1)
        router_q = pieces[2] if include_history and len(pieces) > 2 else None
        null_logits = pieces[3] if include_history and len(pieces) > 3 else None
        if null_logits is not None and add_null_bias:
            if self.hsd_null_read_gate_bias is None:
                raise RuntimeError(
                    "hsd_null_read_gate_bias is required for fused null logits."
                )
            null_logits = null_logits.float() + self.hsd_null_read_gate_bias.float().to(
                device=null_logits.device
            ).view(1, -1)
        return pieces[0], pieces[1], router_q, null_logits

    def _layout_is_cached_decode(self, layout: HistoricalHSDLayout) -> bool:
        if not layout.req_expanded_lens:
            return False
        if layout.padded_static_decode:
            if layout.decode_valid_mask is None:
                return False
            if int(layout.decode_valid_mask.numel()) != len(layout.req_expanded_lens):
                return False
        for req_idx, expanded_len in enumerate(layout.req_expanded_lens):
            if int(expanded_len) <= 0:
                return False
            if int(layout.req_text_lens[req_idx]) != 1:
                return False
            if int(layout.past_text_lens[req_idx]) <= 0:
                return False
        return True

    def _has_spec_decode_metadata(self) -> bool:
        metadata_raw = get_forward_context().attn_metadata
        if not isinstance(metadata_raw, dict):
            return False
        for metadata in metadata_raw.values():
            if getattr(metadata, "spec_sequence_masks", None) is not None:
                return True
        return False

    def _run_hsd_decode_vllm(
        self,
        mixed_qkv_text: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int,
        state_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, text_len, heads, key_dim = query.shape
        value_dim = self.head_v_dim
        anchor_count = int(anchor_q.shape[1])
        if batch_size != 1 or text_len != 1:
            raise NotImplementedError(
                "Safin-1 HSD vLLM cached decode supports one text token per request."
            )
        if not _is_valid_hsd_state_index(state_idx):
            raise RuntimeError(
                "Safin-1 HSD vLLM decode requires a non-null state index."
            )

        ssm_state = self._ssm_cache_for_kernel(state_idx + 1)
        if ssm_state is None:
            raise RuntimeError(
                "Safin-1 HSD vLLM decode requires the native GDN SSM cache."
            )
        state_indices = self._single_state_index_tensor(
            state_idx, mixed_qkv_text.device
        )
        text_o = mixed_qkv_text.new_empty(batch_size, text_len, heads, value_dim)
        trace_requested = _hsd_trace_requested()
        if trace_requested:
            self._trace_internal(
                "tail_initial_recurrent_state",
                ssm_state.index_select(0, state_indices.long())
                .transpose(-1, -2)
                .contiguous(),
            )
        if self.hsd_decode_backend == "recurrent":
            text_o_recurrent, _ = fused_recurrent_gated_delta_rule(
                q=query.contiguous(),
                k=key.contiguous(),
                v=value.contiguous(),
                g=g.contiguous(),
                beta=beta.contiguous(),
                scale=self.head_k_dim**-0.5,
                initial_state=ssm_state,
                inplace_final_state=True,
                ssm_state_indices=state_indices,
                use_qk_l2norm_in_kernel=False,
            )
            text_o.copy_(text_o_recurrent.to(dtype=text_o.dtype))
        else:
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv_text.squeeze(0).contiguous(),
                a=a.contiguous(),
                b=b.contiguous(),
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                scale=self.head_k_dim**-0.5,
                initial_state=ssm_state,
                out=text_o,
                ssm_state_indices=state_indices,
                use_qk_l2norm_in_kernel=True,
            )
        state = (
            ssm_state.index_select(0, state_indices.long())
            .transpose(-1, -2)
            .contiguous()
        )
        if trace_requested:
            self._trace_internal("tail_final_recurrent_state", state)

        if anchor_count == 0:
            anchor_o = query.new_zeros(batch_size, 0, heads, value_dim)
            states = query.new_zeros(batch_size, 0, heads, key_dim, value_dim)
            return text_o, anchor_o, states, state

        expected_boundary = int(boundary_start) + 1
        expected = torch.full_like(boundary_counts, expected_boundary)
        if not torch.equal(boundary_counts, expected):
            raise ValueError(
                "Safin-1 HSD vLLM cached decode anchor tokens must "
                "follow the current text token: "
                f"got={boundary_counts.detach().cpu().tolist()} "
                f"expected={expected.detach().cpu().tolist()}"
            )

        states = (
            state.unsqueeze(1)
            .expand(batch_size, anchor_count, heads, key_dim, value_dim)
            .contiguous()
        )
        anchor_o = (
            torch.einsum(
                "bmhk,bmhkv->bmhv",
                anchor_q.to(states.dtype),
                states,
            )
            * (self.head_k_dim**-0.5)
        ).to(text_o.dtype)
        return text_o, anchor_o, states, state
