# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape Graph V3 decode execution, including split and tensor-only cores.

Request ownership, fixed-buffer allocation/restoration, custom-op registration
and anchor-boundary fallback remain in the model/runner. Execution reads and
updates attention-owned buffers in place; no parameter or state is moved here.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.fla.ops import (
    fused_post_conv_read_q,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.utils.torch_utils import _encode_layer_name

from . import backends
from .runtime import _env_truthy, _l2norm


class GraphDecodeMixin:
    def _graph_v3_history_read_query(
        self,
        mixed_qkv: torch.Tensor,
        *,
        local_k_heads: int,
        local_v_heads: int,
    ) -> tuple[torch.Tensor, bool, int | None]:
        if not self.hsd_graph_v3_exact_read_q:
            return mixed_qkv, True, local_k_heads

        if self.hsd_post_conv_backend == "fused":
            read_q = fused_post_conv_read_q(
                conv_output=mixed_qkv,
                num_k_heads=local_k_heads,
                num_v_heads=local_v_heads,
                head_k_dim=self.head_k_dim,
            )
        else:
            read_q = mixed_qkv[:, : local_k_heads * self.head_k_dim].reshape(
                -1, local_k_heads, self.head_k_dim
            )
            read_q = _l2norm(read_q.contiguous())
            head_repeat = local_v_heads // local_k_heads
            if head_repeat > 1:
                read_q = read_q.repeat_interleave(head_repeat, dim=1)
        return read_q.contiguous(), False, None

    def forward_hsd_graph_v3(
        self,
        *,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        state_indices: torch.Tensor,
        history_state_indices: torch.Tensor,
        past_lens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        conv_cache, recurrent_cache = self._hsd_graph_v3_state_tensors()
        if _env_truthy("VLLM_HSD_GRAPH_V3_SPLIT_CORE"):
            self._forward_hsd_graph_v3_split_core(
                hidden_states=hidden_states,
                output=output,
                state_indices=state_indices,
                history_state_indices=history_state_indices,
                past_lens=past_lens,
                valid_mask=valid_mask,
                conv_cache=conv_cache,
                recurrent_cache=recurrent_cache,
            )
            return
        torch.ops.vllm.hsd_static_decode_attention(
            hidden_states,
            output,
            state_indices,
            history_state_indices,
            past_lens,
            valid_mask,
            conv_cache,
            recurrent_cache,
            _encode_layer_name(self.prefix),
        )

    def _hsd_graph_v3_project_inputs(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        local_key_dim = self.key_dim // self.tp_size
        local_value_dim = self.value_dim // self.tp_size
        qkv_size = local_key_dim * 2 + local_value_dim
        fused_projection = self._fused_decode_projections(
            hidden_states,
            include_history=self.hsd_layer_uses_global_read,
        )
        fused_router_q = None
        fused_null_logits = None
        if fused_projection is None:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
        else:
            mixed_qkvz, ba, fused_router_q, fused_null_logits = fused_projection
        mixed_qkv, z = mixed_qkvz.split([qkv_size, local_value_dim], dim=-1)

        if self.hsd_layer_uses_global_read:
            if _env_truthy("VLLM_HSD_GRAPH_V3_FUSED_HISTORY_PREP"):
                raise RuntimeError(
                    "hsd_graph_v3_split_core_fused_history_prep_unsupported"
                )
            if self.router_q_proj is None:
                raise RuntimeError("router_q_proj is required for Graph V3.")
            if fused_router_q is None:
                router_q, _ = self.router_q_proj(hidden_states)
            else:
                router_q = fused_router_q
            if fused_null_logits is None:
                null_logits = self._project_null_gate_logits(hidden_states)
            else:
                null_logits = fused_null_logits
            if null_logits is None:
                null_logits = hidden_states.new_empty(hidden_states.shape[0], 0)
        else:
            router_q = hidden_states.new_empty(hidden_states.shape[0], 0)
            null_logits = hidden_states.new_empty(hidden_states.shape[0], 0)
        return mixed_qkv, z, ba, router_q, null_logits

    def _forward_hsd_graph_v3_split_core(
        self,
        *,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        state_indices: torch.Tensor,
        history_state_indices: torch.Tensor,
        past_lens: torch.Tensor,
        valid_mask: torch.Tensor,
        conv_cache: torch.Tensor,
        recurrent_cache: torch.Tensor,
    ) -> None:
        req_count = hidden_states.shape[0]
        local_v_heads = self.num_v_heads // self.tp_size
        mixed_qkv, z, ba, router_q, null_logits = self._hsd_graph_v3_project_inputs(
            hidden_states
        )
        core_out = hidden_states.new_empty(
            req_count,
            local_v_heads,
            self.head_v_dim,
        )
        torch.ops.vllm.hsd_static_decode_core(
            mixed_qkv,
            ba,
            router_q,
            null_logits,
            core_out,
            state_indices,
            history_state_indices,
            past_lens,
            valid_mask,
            conv_cache,
            recurrent_cache,
            _encode_layer_name(self.prefix),
        )

        norm_input = core_out.reshape(-1, self.head_v_dim)
        norm_gate = z.reshape(-1, self.head_v_dim)
        projected_input = self._apply_hsd_decode_gate_norm(
            norm_input,
            norm_gate,
        ).reshape(req_count, -1)
        projected_out, _ = self.out_proj(projected_input)
        if _env_truthy("VLLM_HSD_GRAPH_V3_FUSED_OUTPUT_MASK"):
            fused_mask_copy = backends._load_hsd_valid_mask_copy()
            if fused_mask_copy(output, projected_out, valid_mask):
                return
        output.copy_(projected_out.masked_fill(~valid_mask.view(-1, 1), 0))

    def _forward_hsd_graph_v3_core_tensor_only(
        self,
        *,
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
    ) -> None:
        req_count = mixed_qkv.shape[0]
        local_v_heads = self.num_v_heads // self.tp_size
        local_k_heads = self.num_k_heads // self.tp_size
        b, a = ba.chunk(2, dim=-1)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0),
            self.conv1d.weight.size(2),
        )
        strided_qkv = _env_truthy("VLLM_HSD_GRAPH_V3_STRIDED_QKV")
        conv_input = mixed_qkv if strided_qkv else mixed_qkv.contiguous()
        mixed_qkv = causal_conv1d_update(
            conv_input,
            conv_cache,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=state_indices,
            validate_data=False,
        )
        if not strided_qkv:
            mixed_qkv = mixed_qkv.contiguous()

        if _env_truthy("VLLM_HSD_GRAPH_V3_STRIDED_BA"):
            recurrent_a = a
            recurrent_b = b
        else:
            recurrent_a = a.contiguous()
            recurrent_b = b.contiguous()
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=recurrent_a,
            b=recurrent_b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=recurrent_cache,
            out=output.unsqueeze(1),
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )

        if not self.hsd_layer_uses_global_read:
            return
        page_cache = self.hsd_anchor_page_cache
        if page_cache is None:
            raise RuntimeError("hsd_graph_v3_paged_history_cache_missing")
        if self.hsd_history_out_rmsnorm_weight is not None:
            raise RuntimeError(
                "hsd_graph_v3_history_out_rmsnorm_unsupported: the fused "
                "static history read currently implements scale-add only."
            )
        router_q = router_q.view(
            req_count,
            self.hsd_local_route_query_heads,
            self.hsd_route_head_dim,
        )
        if self.hsd_route_qk_l2norm:
            router_q = _l2norm(router_q if strided_qkv else router_q.contiguous())
        null_logits_or_none = null_logits if null_logits.numel() else None
        route_scale = (
            self.hsd_route_head_dim**-0.5 if self.hsd_use_route_scale else 1.0
        ) * self.hsd_route_logit_scale
        history_read_q, history_read_q_is_raw, history_read_q_num_heads = (
            self._graph_v3_history_read_query(
                mixed_qkv,
                local_k_heads=local_k_heads,
                local_v_heads=local_v_heads,
            )
        )
        validate_static_history = _env_truthy("VLLM_HSD_VALIDATE_STATIC_HISTORY_READ")
        reference_base = output.detach().clone() if validate_static_history else None
        static_history_read = backends._load_hsd_paged_static_history_read()
        static_history_read(
            cache=page_cache,
            read_q=history_read_q,
            router_q=router_q,
            state_indices=history_state_indices,
            start_indices=past_lens,
            valid_mask=valid_mask,
            route_scale=float(route_scale),
            read_scale=float(self.head_k_dim**-0.5),
            base_out=output,
            history_scale=self._history_scale_tensor(output.device),
            null_logits=null_logits_or_none,
            null_bias=None,
            read_q_is_raw=history_read_q_is_raw,
            read_q_num_heads=history_read_q_num_heads,
            normalize_router_q=False,
            read_topk=int(self.hsd_read_topk),
            persistent_memory_enabled=(self.hsd_persistent_memory_mode != "removed"),
            memory_metrics=self._paged_persistent_memory_metric_sink(history_read_q),
        )
        if validate_static_history:
            assert reference_base is not None
            loaded_history_read = backends._load_hsd_paged_history_read()
            if loaded_history_read is None:
                raise RuntimeError("hsd_paged_history_read_unavailable")
            _, _, paged_history_read = loaded_history_read
            reference_out = paged_history_read(
                cache=page_cache,
                read_q=history_read_q,
                router_q=router_q,
                state_indices=history_state_indices,
                start_indices=past_lens,
                route_scale=float(route_scale),
                read_scale=float(self.head_k_dim**-0.5),
                null_logits=null_logits_or_none,
                valid_mask=valid_mask,
                read_q_is_raw=history_read_q_is_raw,
                read_q_num_heads=history_read_q_num_heads,
                base_out=reference_base,
                history_scale=self._history_scale_tensor(output.device),
                read_topk=int(self.hsd_read_topk),
                persistent_memory_enabled=(
                    self.hsd_persistent_memory_mode != "removed"
                ),
                memory_metrics=self._paged_persistent_memory_metric_sink(
                    history_read_q
                ),
            )
            if reference_out is None:
                self._record_hsd_varlen_event(
                    "hsd_static_history_read_validate",
                    valid=0,
                    reason="reference_none",
                    requests=req_count,
                )
            else:
                diff = (output.float() - reference_out.float()).abs()
                self._record_hsd_varlen_event(
                    "hsd_static_history_read_validate",
                    valid=1,
                    requests=req_count,
                    max_abs=float(diff.max().item()) if diff.numel() else 0.0,
                    mean_abs=float(diff.mean().item()) if diff.numel() else 0.0,
                )

    def _forward_hsd_graph_v3_tensor_only(
        self,
        *,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        state_indices: torch.Tensor,
        history_state_indices: torch.Tensor,
        past_lens: torch.Tensor,
        valid_mask: torch.Tensor,
        conv_cache: torch.Tensor,
        recurrent_cache: torch.Tensor,
    ) -> None:
        req_count = hidden_states.shape[0]
        local_v_heads = self.num_v_heads // self.tp_size
        local_k_heads = self.num_k_heads // self.tp_size
        local_key_dim = self.key_dim // self.tp_size
        local_value_dim = self.value_dim // self.tp_size
        qkv_size = local_key_dim * 2 + local_value_dim

        fuse_history_prep = _env_truthy("VLLM_HSD_GRAPH_V3_FUSED_HISTORY_PREP")
        fused_projection = self._fused_decode_projections(
            hidden_states,
            include_history=self.hsd_layer_uses_global_read,
            add_null_bias=not fuse_history_prep,
        )
        fused_router_q = None
        fused_null_logits = None
        if fused_projection is None:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
        else:
            mixed_qkvz, ba, fused_router_q, fused_null_logits = fused_projection
        mixed_qkv, z = mixed_qkvz.split([qkv_size, local_value_dim], dim=-1)
        b, a = ba.chunk(2, dim=-1)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0),
            self.conv1d.weight.size(2),
        )
        strided_qkv = _env_truthy("VLLM_HSD_GRAPH_V3_STRIDED_QKV")
        conv_input = mixed_qkv if strided_qkv else mixed_qkv.contiguous()
        mixed_qkv = causal_conv1d_update(
            conv_input,
            conv_cache,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=state_indices,
            validate_data=False,
        )
        if not strided_qkv:
            mixed_qkv = mixed_qkv.contiguous()

        text_o = hidden_states.new_empty(
            req_count,
            local_v_heads,
            self.head_v_dim,
        )
        if _env_truthy("VLLM_HSD_GRAPH_V3_STRIDED_BA"):
            recurrent_a = a
            recurrent_b = b
        else:
            recurrent_a = a.contiguous()
            recurrent_b = b.contiguous()
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=recurrent_a,
            b=recurrent_b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=recurrent_cache,
            out=text_o.unsqueeze(1),
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        self._capture_hsd_graph_v3_tensor_sidecar(
            "hsd_recurrent_out",
            text_o,
        )

        if self.hsd_layer_uses_global_read:
            page_cache = self.hsd_anchor_page_cache
            if page_cache is None:
                raise RuntimeError("hsd_graph_v3_paged_history_cache_missing")
            if self.hsd_history_out_rmsnorm_weight is not None:
                raise RuntimeError(
                    "hsd_graph_v3_history_out_rmsnorm_unsupported: the fused "
                    "static history read currently implements scale-add only."
                )
            if self.router_q_proj is None:
                raise RuntimeError("router_q_proj is required for Graph V3.")
            if fused_router_q is None:
                router_q, _ = self.router_q_proj(hidden_states)
            else:
                router_q = fused_router_q
            router_q = router_q.view(
                req_count,
                self.hsd_local_route_query_heads,
                self.hsd_route_head_dim,
            )
            if self.hsd_route_qk_l2norm and not fuse_history_prep:
                router_q = _l2norm(router_q if strided_qkv else router_q.contiguous())
            null_bias = None
            if fused_null_logits is not None:
                null_logits = fused_null_logits
                if fuse_history_prep and self.null_gate_proj is not None:
                    if self.hsd_null_read_gate_bias is None:
                        raise RuntimeError(
                            "hsd_null_read_gate_bias is required when the fused "
                            "Graph V3 history prep is enabled."
                        )
                    null_bias = self.hsd_null_read_gate_bias
            elif fuse_history_prep and self.null_gate_proj is not None:
                null_logits, _ = self.null_gate_proj(hidden_states)
                if self.hsd_null_read_gate_bias is None:
                    raise RuntimeError(
                        "hsd_null_read_gate_bias is required when the fused "
                        "Graph V3 history prep is enabled."
                    )
                null_bias = self.hsd_null_read_gate_bias
            else:
                null_logits = self._project_null_gate_logits(hidden_states)
            route_scale = (
                self.hsd_route_head_dim**-0.5 if self.hsd_use_route_scale else 1.0
            ) * self.hsd_route_logit_scale
            history_read_q, history_read_q_is_raw, history_read_q_num_heads = (
                self._graph_v3_history_read_query(
                    mixed_qkv,
                    local_k_heads=local_k_heads,
                    local_v_heads=local_v_heads,
                )
            )
            validate_static_history = _env_truthy(
                "VLLM_HSD_VALIDATE_STATIC_HISTORY_READ"
            )
            reference_base = (
                text_o.detach().clone() if validate_static_history else None
            )
            static_history_read = backends._load_hsd_paged_static_history_read()
            separate_history_add = _env_truthy("VLLM_HSD_GRAPH_V3_SEPARATE_HISTORY_ADD")
            static_history_out = (
                torch.empty_like(text_o) if separate_history_add else text_o
            )
            static_history_read(
                cache=page_cache,
                read_q=history_read_q,
                router_q=router_q,
                state_indices=history_state_indices,
                start_indices=past_lens,
                valid_mask=valid_mask,
                route_scale=float(route_scale),
                read_scale=float(self.head_k_dim**-0.5),
                base_out=static_history_out,
                history_scale=self._history_scale_tensor(text_o.device),
                null_logits=null_logits,
                null_bias=null_bias,
                read_q_is_raw=history_read_q_is_raw,
                read_q_num_heads=history_read_q_num_heads,
                normalize_router_q=(fuse_history_prep and self.hsd_route_qk_l2norm),
                fuse_scale_add=not separate_history_add,
                read_topk=int(self.hsd_read_topk),
                persistent_memory_enabled=(
                    self.hsd_persistent_memory_mode != "removed"
                ),
                memory_metrics=self._paged_persistent_memory_metric_sink(
                    history_read_q
                ),
            )
            if separate_history_add:
                self._capture_hsd_graph_v3_tensor_sidecar(
                    "hsd_history_raw",
                    static_history_out,
                )
                scaled_history = (
                    static_history_out
                    * self._history_scale_tensor(static_history_out.device).to(
                        dtype=static_history_out.dtype,
                    )
                ).to(text_o.dtype)
                text_o.add_(scaled_history)
            self._capture_hsd_graph_v3_tensor_sidecar(
                "hsd_history_out",
                text_o,
            )
            if validate_static_history:
                assert reference_base is not None
                reference_router_q = router_q
                if fuse_history_prep and self.hsd_route_qk_l2norm:
                    reference_router_q = _l2norm(
                        router_q if strided_qkv else router_q.contiguous()
                    )
                reference_null_logits = null_logits
                if reference_null_logits is not None and null_bias is not None:
                    reference_null_logits = (
                        reference_null_logits.float()
                        + null_bias.float().to(reference_null_logits.device).view(1, -1)
                    )
                loaded_history_read = backends._load_hsd_paged_history_read()
                if loaded_history_read is None:
                    raise RuntimeError("hsd_paged_history_read_unavailable")
                _, _, paged_history_read = loaded_history_read
                reference_out = paged_history_read(
                    cache=page_cache,
                    read_q=history_read_q,
                    router_q=reference_router_q,
                    state_indices=history_state_indices,
                    start_indices=past_lens,
                    route_scale=float(route_scale),
                    read_scale=float(self.head_k_dim**-0.5),
                    null_logits=reference_null_logits,
                    valid_mask=valid_mask,
                    read_q_is_raw=history_read_q_is_raw,
                    read_q_num_heads=history_read_q_num_heads,
                    base_out=reference_base,
                    history_scale=self._history_scale_tensor(text_o.device),
                    read_topk=int(self.hsd_read_topk),
                    persistent_memory_enabled=(
                        self.hsd_persistent_memory_mode != "removed"
                    ),
                    memory_metrics=self._paged_persistent_memory_metric_sink(
                        history_read_q
                    ),
                )
                if reference_out is None:
                    self._record_hsd_varlen_event(
                        "hsd_static_history_read_validate",
                        valid=0,
                        reason="reference_none",
                        requests=req_count,
                    )
                else:
                    diff = (text_o.float() - reference_out.float()).abs()
                    self._record_hsd_varlen_event(
                        "hsd_static_history_read_validate",
                        valid=1,
                        requests=req_count,
                        max_abs=float(diff.max().item()) if diff.numel() else 0.0,
                        mean_abs=float(diff.mean().item()) if diff.numel() else 0.0,
                    )

        strided_gate_norm = None
        batch_invariant_gate_norm = _env_truthy("VLLM_HSD_BATCH_INVARIANT_GATE_NORM")
        if not batch_invariant_gate_norm and _env_truthy(
            "VLLM_HSD_GRAPH_V3_STRIDED_GATE_NORM"
        ):
            strided_gate_norm_fn = backends._load_hsd_strided_gate_rmsnorm()
            strided_gate_norm = strided_gate_norm_fn(
                text_o,
                z.view(req_count, local_v_heads, self.head_v_dim),
                self.norm.weight,
                eps=float(self.norm.eps),
                activation=str(self.norm.activation),
            )
        if strided_gate_norm is not None:
            core_out = strided_gate_norm.reshape(req_count, -1)
        else:
            norm_input = text_o.reshape(-1, self.head_v_dim)
            norm_gate = z.reshape(-1, self.head_v_dim)
            if batch_invariant_gate_norm:
                core_out = self._apply_hsd_decode_gate_norm(
                    norm_input,
                    norm_gate,
                ).reshape(req_count, -1)
            elif _env_truthy("VLLM_HSD_GRAPH_V3_FUSED_RMSNORM"):
                core_out = self.norm.forward_cuda(norm_input, norm_gate).reshape(
                    req_count,
                    -1,
                )
            else:
                core_out = self.norm(norm_input, norm_gate).reshape(req_count, -1)
        self._capture_hsd_graph_v3_tensor_sidecar(
            "hsd_gate_norm_out",
            core_out,
        )
        projected_out, _ = self.out_proj(core_out)
        self._capture_hsd_graph_v3_tensor_sidecar(
            "hsd_projected_out",
            projected_out,
        )
        if _env_truthy("VLLM_HSD_GRAPH_V3_FUSED_OUTPUT_MASK"):
            fused_mask_copy = backends._load_hsd_valid_mask_copy()
            if not fused_mask_copy(output, projected_out, valid_mask):
                output.copy_(projected_out)
                output.mul_(valid_mask.view(-1, 1))
        else:
            output.copy_(projected_out.masked_fill(~valid_mask.view(-1, 1), 0))
