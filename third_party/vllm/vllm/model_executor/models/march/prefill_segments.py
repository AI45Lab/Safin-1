# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference segments and grouped parity execution; owns autotune warmup keys."""

from __future__ import annotations

import os

import torch

from . import backends
from .diagnostics import _hsd_trace_requested
from .runtime import _env_truthy

_HSD_VARLEN_SEGMENT_AUTOTUNE_WARMED: set[tuple[object, ...]] = set()


class SegmentPrefillMixin:
    def _run_delta_segment(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        start: int,
        end: int,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cu_seqlens = torch.tensor(
            [0, end - start], dtype=torch.int32, device=query.device
        )
        # HSD stores recurrent states as [B, H, K, V], while the generic
        # FLA chunk op consumes and returns [B, H, V, K].
        initial_state_fla = (
            initial_state.transpose(-1, -2).contiguous()
            if initial_state is not None
            else None
        )
        out, state_fla = self.chunk_gated_delta_rule(
            q=query[:, start:end],
            k=key[:, start:end],
            v=value[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            initial_state=initial_state_fla,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )
        state = state_fla.transpose(-1, -2).contiguous()
        return out, state

    def _maybe_warm_varlen_segment_autotune(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        first_text_len: int,
    ) -> None:
        warmup_override = os.environ.get("VLLM_HSD_VARLEN_SEGMENT_AUTOTUNE_WARMUP")
        warmup_enabled = (
            _env_truthy("VLLM_HSD_VARLEN_SEGMENT_AUTOTUNE_WARMUP")
            if warmup_override is not None
            else self._is_hsd_moe_model()
        )
        if not warmup_enabled:
            return
        warm_len = min(
            int(getattr(self.config, "anchor_chunk_size", 512) or 512),
            int(first_text_len),
        )
        if warm_len <= 0:
            return
        warm_key = (
            query.device.type,
            query.device.index,
            str(query.dtype),
            str(g.dtype),
            int(query.shape[1]),
            int(query.shape[2]),
            int(value.shape[2]),
            bool(self._use_deterministic_varlen_kkt()),
        )
        if warm_key in _HSD_VARLEN_SEGMENT_AUTOTUNE_WARMED:
            return

        # First select the exact FLA configs used by the accepted 512-token
        # segments path. Then execute the packed reference wrapper at the same
        # length so its varlen-only setup and anchor extraction are also warm
        # before the real, longer packed request reaches them.
        self._run_delta_segment(
            query[:warm_len].detach().clone().unsqueeze(0),
            key[:warm_len].detach().clone().unsqueeze(0),
            value[:warm_len].detach().clone().unsqueeze(0),
            g[:warm_len].detach().clone().unsqueeze(0),
            beta[:warm_len].detach().clone().unsqueeze(0),
            0,
            warm_len,
            initial_state[:1].detach().clone(),
        )
        varlen_forward = backends._load_hsd_varlen_prefill()
        if varlen_forward is None:
            raise RuntimeError(
                "HSD segment-parity autotune warmup requires the varlen helper"
            )
        anchor_chunk_size = int(getattr(self.config, "anchor_chunk_size", 512) or 512)
        anchor_count = warm_len // anchor_chunk_size
        warm_cu_seqlens = torch.tensor(
            [0, warm_len],
            dtype=torch.int32,
            device=query.device,
        )
        warm_anchor_offsets = torch.tensor(
            [0, anchor_count],
            dtype=torch.int32,
            device=query.device,
        )
        warm_result = varlen_forward(
            query=query[:warm_len].detach().clone().unsqueeze(0),
            key=key[:warm_len].detach().clone().unsqueeze(0),
            value=value[:warm_len].detach().clone().unsqueeze(0),
            g=g[:warm_len].detach().clone().unsqueeze(0),
            beta=beta[:warm_len].detach().clone().unsqueeze(0),
            cu_seqlens=warm_cu_seqlens,
            anchor_offsets=warm_anchor_offsets,
            anchor_chunk_size=anchor_chunk_size,
            initial_state=initial_state[:1].detach().clone().float(),
            scale=self.head_k_dim**-0.5,
            use_vllm_reference_kernel=True,
            use_deterministic_kkt=self._use_deterministic_varlen_kkt(),
        )
        if warm_result is None:
            raise RuntimeError(
                "HSD segment-parity autotune warmup varlen helper returned None"
            )
        # Autotune may benchmark several stateful Triton kernels. Do not let
        # the first real prefill overlap any trailing benchmark/JIT work; the
        # trace path synchronizes implicitly while saving tensors, and cold
        # no-trace runs must have the same ordering guarantee.
        torch.cuda.synchronize(query.device)
        _HSD_VARLEN_SEGMENT_AUTOTUNE_WARMED.add(warm_key)
        self._record_hsd_varlen_event(
            "hsd_varlen_segment_autotune_warmup_hit",
            layer=int(self.layer_idx) + 1,
            text_tokens=int(warm_len),
        )

    def _run_hsd_segments(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, text_len, heads, key_dim = query.shape
        value_dim = value.shape[-1]
        anchor_count = int(anchor_q.shape[1])
        deterministic_anchor_read = self._use_deterministic_anchor_read(anchor_count)
        cursor = 0
        state = initial_state
        text_outputs: list[torch.Tensor] = []
        anchor_outputs: list[torch.Tensor] = []
        anchor_states: list[torch.Tensor] = []

        for anchor_idx, boundary in enumerate(boundary_counts.detach().cpu().tolist()):
            local_boundary = int(boundary) - int(boundary_start)
            if local_boundary < cursor or local_boundary > text_len:
                raise ValueError(
                    "Invalid HSD anchor boundary for current vLLM step: "
                    f"boundary={boundary}, start={boundary_start}, text_len={text_len}."
                )
            if local_boundary > cursor:
                out, state = self._run_delta_segment(
                    query, key, value, g, beta, cursor, local_boundary, state
                )
                text_outputs.append(out)
                cursor = local_boundary

            # Boundary zero is a pre-read virtual row: take q/k from anchor
            # hidden, but do not run or seed the text recurrence with memory.
            if int(boundary) == 0 and int(boundary_start) == 0:
                snapshot_state = state.new_zeros(state.shape)
            else:
                snapshot_state = state
            if not deterministic_anchor_read:
                anchor_outputs.append(
                    torch.einsum(
                        "bmhk,bmhkv->bmhv",
                        anchor_q[:, anchor_idx : anchor_idx + 1].to(state.dtype),
                        snapshot_state.unsqueeze(1),
                    )
                    * (self.head_k_dim**-0.5)
                )
            anchor_states.append(snapshot_state)

        if cursor < text_len:
            if _hsd_trace_requested():
                self._trace_internal("tail_initial_recurrent_state", state)
            out, state = self._run_delta_segment(
                query, key, value, g, beta, cursor, text_len, state
            )
            if _hsd_trace_requested():
                self._trace_internal("tail_final_recurrent_state", state)
            text_outputs.append(out)

        text_o = (
            torch.cat(text_outputs, dim=1)
            if text_outputs
            else query.new_zeros(batch_size, 0, heads, value_dim)
        )
        states = (
            torch.cat([item.unsqueeze(1) for item in anchor_states], dim=1)
            if anchor_states
            else query.new_zeros(batch_size, 0, heads, key_dim, value_dim)
        )
        if deterministic_anchor_read:
            anchor_read = backends._load_hsd_deterministic_anchor_read()
            anchor_o_flat = (
                None
                if anchor_read is None
                else anchor_read(
                    anchor_q.reshape(-1, heads, key_dim),
                    states.reshape(-1, heads, key_dim, value_dim),
                    scale=self.head_k_dim**-0.5,
                )
            )
            if anchor_o_flat is None:
                raise RuntimeError(
                    "VLLM_HSD_DETERMINISTIC_ANCHOR_READ was requested, but "
                    "the fixed-reduction Triton kernel is unavailable."
                )
            anchor_o = anchor_o_flat.view(
                batch_size,
                anchor_count,
                heads,
                value_dim,
            ).to(text_o.dtype)
        else:
            anchor_o = (
                torch.cat(anchor_outputs, dim=1).to(text_o.dtype)
                if anchor_outputs
                else query.new_zeros(batch_size, 0, heads, value_dim)
            )
        return text_o, anchor_o, states, state

    def _run_hsd_grouped_segment_parity(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        text_offsets: list[int],
        anchor_offsets: list[int],
        boundary_counts: list[torch.Tensor],
        past_text_lens: list[int],
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run exact 512-boundary segments while batching the same round.

        The one-shot packed recurrence changes FP32 state accumulation at
        chunk boundaries by a few ulps. Dense models tolerate that difference,
        but MoE top-k routing can amplify it into a token branch. This path
        keeps each request's segment boundaries identical to ``segments`` and
        only packs independent requests from the same segment round.
        """
        req_count = len(text_offsets) - 1
        if req_count <= 0 or len(anchor_offsets) != req_count + 1:
            raise ValueError("Invalid HSD grouped segment-parity offsets.")
        if len(boundary_counts) != req_count or len(past_text_lens) != req_count:
            raise ValueError("Invalid HSD grouped segment-parity metadata.")
        if int(initial_state.shape[0]) != req_count:
            raise ValueError("Invalid HSD grouped segment-parity initial state.")

        local_boundaries: list[list[int]] = []
        for req_idx in range(req_count):
            text_len = int(text_offsets[req_idx + 1] - text_offsets[req_idx])
            positions = [
                int(boundary) - int(past_text_lens[req_idx])
                for boundary in boundary_counts[req_idx].detach().cpu().tolist()
            ]
            if any(
                position <= 0
                or position > text_len
                or (idx > 0 and position <= positions[idx - 1])
                for idx, position in enumerate(positions)
            ):
                raise ValueError(
                    "Invalid HSD grouped segment-parity anchor positions: "
                    f"request={req_idx}, positions={positions}, text_len={text_len}."
                )
            expected = int(anchor_offsets[req_idx + 1] - anchor_offsets[req_idx])
            if len(positions) != expected:
                raise ValueError(
                    "HSD grouped segment-parity anchor count mismatch: "
                    f"request={req_idx}, positions={len(positions)}, "
                    f"expected={expected}."
                )
            local_boundaries.append(positions)

        total_text = int(text_offsets[-1])
        total_anchor = int(anchor_offsets[-1])
        text_out = value.new_empty(
            total_text,
            int(value.shape[1]),
            int(value.shape[2]),
        )
        anchor_states = initial_state.new_empty(
            total_anchor,
            *initial_state.shape[1:],
        )
        states = [
            initial_state[req_idx : req_idx + 1].contiguous()
            for req_idx in range(req_count)
        ]
        cursors = [0] * req_count
        anchor_cursors = [0] * req_count

        while True:
            active = [
                req_idx
                for req_idx in range(req_count)
                if cursors[req_idx]
                < int(text_offsets[req_idx + 1] - text_offsets[req_idx])
            ]
            if not active:
                break

            segment_ranges: list[tuple[int, int, int, bool]] = []
            segment_lengths: list[int] = []
            for req_idx in active:
                local_start = cursors[req_idx]
                text_len = int(text_offsets[req_idx + 1] - text_offsets[req_idx])
                anchor_cursor = anchor_cursors[req_idx]
                is_anchor_boundary = anchor_cursor < len(local_boundaries[req_idx])
                local_end = (
                    local_boundaries[req_idx][anchor_cursor]
                    if is_anchor_boundary
                    else text_len
                )
                if local_end <= local_start:
                    raise RuntimeError(
                        "HSD grouped segment-parity made no progress: "
                        f"request={req_idx}, start={local_start}, end={local_end}."
                    )
                global_start = int(text_offsets[req_idx]) + local_start
                global_end = int(text_offsets[req_idx]) + local_end
                segment_ranges.append(
                    (req_idx, global_start, global_end, is_anchor_boundary)
                )
                segment_lengths.append(global_end - global_start)

            if len(segment_ranges) == 1:
                _, segment_start, segment_end, _ = segment_ranges[0]
                # Preserve the exact view strides used by _run_delta_segment.
                # Materializing a one-row torch.cat selects a different FLA
                # specialization and changes the FP32 state by a few ulps.
                packed_query = query[segment_start:segment_end].unsqueeze(0)
                packed_key = key[segment_start:segment_end].unsqueeze(0)
                packed_value = value[segment_start:segment_end].unsqueeze(0)
                packed_g = g[segment_start:segment_end].unsqueeze(0)
                packed_beta = beta[segment_start:segment_end].unsqueeze(0)
            else:
                packed_query = torch.cat(
                    [query[start:end] for _, start, end, _ in segment_ranges],
                    dim=0,
                ).unsqueeze(0)
                packed_key = torch.cat(
                    [key[start:end] for _, start, end, _ in segment_ranges],
                    dim=0,
                ).unsqueeze(0)
                packed_value = torch.cat(
                    [value[start:end] for _, start, end, _ in segment_ranges],
                    dim=0,
                ).unsqueeze(0)
                packed_g = torch.cat(
                    [g[start:end] for _, start, end, _ in segment_ranges],
                    dim=0,
                ).unsqueeze(0)
                packed_beta = torch.cat(
                    [beta[start:end] for _, start, end, _ in segment_ranges],
                    dim=0,
                ).unsqueeze(0)
            cu_seqlens = torch.tensor(
                [0, *torch.tensor(segment_lengths).cumsum(0).tolist()],
                dtype=torch.int32,
                device=query.device,
            )
            initial_state_fla = (
                torch.cat(
                    [states[req_idx] for req_idx in active],
                    dim=0,
                )
                .transpose(-1, -2)
                .contiguous()
            )
            packed_out, state_fla = self.chunk_gated_delta_rule(
                q=packed_query,
                k=packed_key,
                v=packed_value,
                g=packed_g,
                beta=packed_beta,
                initial_state=initial_state_fla,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )
            if state_fla is None:
                raise RuntimeError(
                    "HSD grouped segment-parity recurrent state is missing."
                )
            next_states = state_fla.transpose(-1, -2).contiguous()

            packed_cursor = 0
            for active_idx, (
                req_idx,
                global_start,
                global_end,
                is_anchor_boundary,
            ) in enumerate(segment_ranges):
                segment_len = global_end - global_start
                text_out[global_start:global_end].copy_(
                    packed_out[
                        0,
                        packed_cursor : packed_cursor + segment_len,
                    ]
                )
                packed_cursor += segment_len
                states[req_idx] = next_states[active_idx : active_idx + 1].contiguous()
                cursors[req_idx] += segment_len
                if is_anchor_boundary:
                    anchor_idx = int(anchor_offsets[req_idx]) + anchor_cursors[req_idx]
                    anchor_states[anchor_idx].copy_(states[req_idx][0])
                    anchor_cursors[req_idx] += 1

        final_state = torch.cat(states, dim=0)
        return text_out.unsqueeze(0), anchor_states, final_state
