"""MARCH GDN text recurrence, cached decode and anchor-state production."""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache
import torch.nn.functional as F
from .fused_march_segments import (
    FLA_CHUNK_SIZE, chunk_gated_delta_rule_with_chunk_states_forward,
    chunk_gated_delta_rule_with_anchor_states_forward,
)


class Safin1GDNRecurrenceMixin:
    """Shared methods; the model entrypoint owns parameters and dispatch."""

    def _run_text_conv(
        self,
        mixed_qkv_text: torch.Tensor,
        cache_params: Cache | None,
        conv_state: torch.Tensor | None,
        use_precomputed_states: bool,
        seq_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_len = mixed_qkv_text.shape[1]
        mixed_qkv = mixed_qkv_text.transpose(1, 2)
        if use_precomputed_states and text_len == 1 and conv_state is not None:
            if self.causal_conv1d_update is not None:
                mixed_qkv = self.causal_conv1d_update(
                    mixed_qkv,
                    conv_state,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.activation,
                )
                return mixed_qkv.transpose(1, 2)
            conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)
            self._update_conv_state(cache_params, F.pad(conv_input, (self.conv_kernel_size - conv_input.shape[-1], 0)))
            mixed_qkv = F.silu(self.conv1d(conv_input)[:, :, -text_len:])
            return mixed_qkv.transpose(1, 2)

        conv_input = mixed_qkv
        if use_precomputed_states and conv_state is not None:
            conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)
        if cache_params is not None:
            self._update_conv_state(cache_params, F.pad(conv_input, (self.conv_kernel_size - conv_input.shape[-1], 0)))
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=conv_input,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(conv_input)[:, :, : conv_input.shape[-1]])
        if use_precomputed_states and conv_state is not None:
            mixed_qkv = mixed_qkv[:, :, -text_len:]
        return mixed_qkv.transpose(1, 2)


    def _run_hsd_decode_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, text_len, heads, key_dim = query.shape
        value_dim = value.shape[-1]
        anchor_count = anchor_q.shape[1]
        if text_len != 1:
            raise NotImplementedError("Safin-1 HSD cached decode supports one text token per step.")

        self._trace_internal("tail_initial_recurrent_state", initial_state)
        text_o, state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        self._trace_internal("tail_final_recurrent_state", state)

        if anchor_count == 0:
            anchor_o = query.new_zeros(batch_size, 0, heads, value_dim)
            states = query.new_zeros(batch_size, 0, heads, key_dim, value_dim)
            return text_o, anchor_o, states, state

        expected_boundary = int(boundary_start) + 1
        expected = torch.full_like(boundary_counts, expected_boundary)
        if not torch.equal(boundary_counts, expected):
            raise ValueError(
                "Safin-1 HSD cached decode anchor tokens must follow the current text token: "
                f"got={boundary_counts.detach().cpu().tolist()} expected={expected.detach().cpu().tolist()}"
            )

        states = state.unsqueeze(1).expand(batch_size, anchor_count, heads, key_dim, value_dim).contiguous()
        anchor_o = (
            torch.einsum(
                "bmhk,bmhkv->bmhv",
                anchor_q.to(states.dtype),
                states,
            )
            * (self.head_k_dim**-0.5)
        ).to(text_o.dtype)
        return text_o, anchor_o, states, state


    def _run_hsd_segments(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int = 0,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, text_len, heads, key_dim = query.shape
        value_dim = value.shape[-1]
        fast = self._run_hsd_segments_fastpath(
            query=query,
            key=key,
            value=value,
            beta=beta,
            g=g,
            anchor_q=anchor_q,
            boundary_counts=boundary_counts,
            boundary_start=boundary_start,
            initial_state=initial_state,
        )
        if fast is not None:
            return fast

        text_outputs = []
        anchor_outputs = []
        anchor_states = []
        state = initial_state
        cursor = 0
        zero_state = query.new_zeros(batch_size, heads, key_dim, value_dim)

        for anchor_idx, boundary in enumerate(boundary_counts.tolist()):
            local_boundary = int(boundary) - int(boundary_start)
            if local_boundary < cursor:
                raise ValueError("anchor positions must be ordered after non-decreasing text prefixes.")
            if local_boundary > text_len:
                raise ValueError("anchor positions must not exceed current text length.")
            if local_boundary > cursor:
                out, state = self._run_delta_segment(query, key, value, beta, g, cursor, local_boundary, state)
                text_outputs.append(out)
                cursor = local_boundary

            state_for_anchor = zero_state if state is None else state
            anchor_states.append(state_for_anchor)
            anchor_outputs.append(
                torch.einsum(
                    "bhk,bhkv->bhv",
                    anchor_q[:, anchor_idx].to(state_for_anchor.dtype),
                    state_for_anchor,
                )
                * (self.head_k_dim**-0.5)
            )

        if cursor < text_len:
            initial_state = (state)
            self._trace_internal("tail_initial_recurrent_state", initial_state)
            out, state = self._run_delta_segment(query, key, value, beta, g, cursor, text_len, initial_state)
            self._trace_internal("tail_final_recurrent_state", state)
            text_outputs.append(out)

        text_o = torch.cat(text_outputs, dim=1) if text_outputs else query.new_zeros(
            batch_size, 0, heads, value_dim
        )
        anchor_o = torch.stack(anchor_outputs, dim=1).to(text_o.dtype) if anchor_outputs else query.new_zeros(
            batch_size, 0, heads, value_dim
        )
        states = torch.stack(anchor_states, dim=1) if anchor_states else query.new_zeros(
            batch_size, 0, heads, key_dim, value_dim
        )
        if state is None:
            state = zero_state
        return text_o, anchor_o, states, state


    def _run_hsd_segments_fastpath(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int = 0,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if anchor_q.ndim != 4 or boundary_counts.ndim != 1:
            return None
        batch_size, text_len, heads, key_dim = query.shape
        anchor_count = int(anchor_q.shape[1])
        value_dim = int(value.shape[-1])
        if anchor_count <= 0 or int(boundary_counts.shape[0]) != anchor_count:
            return None
        if anchor_q.shape != (batch_size, anchor_count, heads, key_dim):
            return None
        if torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (query, key, value, beta, g, anchor_q, boundary_counts, initial_state)
            if torch.is_tensor(tensor)
        ):
            return None

        local_boundaries = boundary_counts.to(device=query.device, dtype=torch.long) - int(boundary_start)
        if local_boundaries.numel() != anchor_count:
            return None
        if bool((local_boundaries < 0).any().item()):
            return None
        if bool((local_boundaries > int(text_len)).any().item()):
            return None
        if anchor_count > 1 and bool((local_boundaries[1:] < local_boundaries[:-1]).any().item()):
            return None
        is_final_boundary = local_boundaries == int(text_len)
        aligned_or_final = is_final_boundary | ((local_boundaries % FLA_CHUNK_SIZE) == 0)
        if not bool(aligned_or_final.all().item()):
            return None

        anchor_chunk_size = int(getattr(self, "anchor_chunk_size", 0) or getattr(self, "hist_chunk_size", 0) or 0)
        if anchor_chunk_size > 0 and anchor_chunk_size % FLA_CHUNK_SIZE == 0:
            expected_boundaries = (
                torch.arange(1, anchor_count + 1, device=query.device, dtype=torch.long) * int(anchor_chunk_size)
            )
            if torch.equal(local_boundaries, expected_boundaries):
                selective = chunk_gated_delta_rule_with_anchor_states_forward(
                    query=query,
                    key=key,
                    value=value,
                    g=g,
                    beta=beta,
                    anchor_count=anchor_count,
                    anchor_chunk_size=anchor_chunk_size,
                    initial_state=initial_state,
                    scale=self.head_k_dim**-0.5,
                    anchor_q=anchor_q,
                )
                if selective is not None:
                    text_o, fused_anchor_o, states, final_state = selective
                    if bool(is_final_boundary.any().item()):
                        final_expanded = final_state.unsqueeze(1).expand(
                            batch_size, anchor_count, heads, key_dim, value_dim
                        )
                        states = torch.where(
                            is_final_boundary.view(1, anchor_count, 1, 1, 1),
                            final_expanded,
                            states,
                        )
                    if fused_anchor_o is not None:
                        anchor_o = fused_anchor_o.to(text_o.dtype)
                    else:
                        anchor_outputs = []
                        scale = self.head_k_dim**-0.5
                        for anchor_idx in range(anchor_count):
                            state_for_anchor = states[:, anchor_idx]
                            anchor_outputs.append(
                                torch.einsum(
                                    "bhk,bhkv->bhv",
                                    anchor_q[:, anchor_idx].to(state_for_anchor.dtype),
                                    state_for_anchor,
                                )
                                * scale
                            )
                        anchor_o = torch.stack(anchor_outputs, dim=1).to(text_o.dtype)
                    self._trace_internal("tail_final_recurrent_state", final_state)
                    return text_o, anchor_o, states, final_state

        fast = chunk_gated_delta_rule_with_chunk_states_forward(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            scale=self.head_k_dim**-0.5,
        )
        if fast is None:
            return None
        text_o, chunk_states, final_state = fast
        if final_state is None:
            return None
        if chunk_states.ndim != 5 or int(chunk_states.shape[1]) <= 0:
            return None

        chunk_indices = torch.div(local_boundaries, FLA_CHUNK_SIZE, rounding_mode="floor")
        max_chunk_index = int(chunk_states.shape[1]) - 1
        if bool((chunk_indices[~is_final_boundary] > max_chunk_index).any().item()):
            return None
        gather_indices = chunk_indices.clamp(min=0, max=max_chunk_index)
        states = chunk_states.index_select(1, gather_indices).to(final_state.dtype)
        if bool(is_final_boundary.any().item()):
            final_expanded = final_state.unsqueeze(1).expand(batch_size, anchor_count, heads, key_dim, value_dim)
            states = torch.where(is_final_boundary.view(1, anchor_count, 1, 1, 1), final_expanded, states)

        anchor_outputs = []
        scale = self.head_k_dim**-0.5
        for anchor_idx in range(anchor_count):
            state_for_anchor = states[:, anchor_idx]
            anchor_outputs.append(
                torch.einsum(
                    "bhk,bhkv->bhv",
                    anchor_q[:, anchor_idx].to(state_for_anchor.dtype),
                    state_for_anchor,
                )
                * scale
            )
        anchor_o = torch.stack(anchor_outputs, dim=1).to(text_o.dtype)
        self._trace_internal("tail_final_recurrent_state", final_state)
        return text_o, anchor_o, states, final_state


    def _run_delta_segment(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        start: int,
        end: int,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if initial_state is not None and end - start == 1:
            return self.recurrent_gated_delta_rule(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                g=g[:, start:end],
                beta=beta[:, start:end],
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
        return self.chunk_gated_delta_rule(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )

