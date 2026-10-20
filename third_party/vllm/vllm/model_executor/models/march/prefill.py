# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request prefill admission and FlashQLA / varlen / segments dispatch.

Return None only for an optional backend miss. Strict FlashQLA errors propagate.
This mixin owns no parameters or cache tensors."""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

from . import backends
from .diagnostics import _hsd_trace_requested
from .runtime import _env_truthy

logger = init_logger("vllm.model_executor.models.safin1")


class PrefillMixin:
    def _hsd_varlen_prefill_fallback_reason(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int,
        initial_state: torch.Tensor,
    ) -> str | None:
        if self.hsd_prefill_backend != "varlen":
            return "backend_segments"
        if torch.is_grad_enabled():
            return "grad_enabled"
        if not query.is_cuda:
            return "not_cuda"
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return "bad_qkv_rank"
        if g.ndim != 3 or beta.ndim != 3:
            return "bad_gate_rank"
        batch_size, text_len, heads, key_dim = query.shape
        if batch_size != 1:
            return "batch_size_not_one"
        if text_len <= 0:
            return "empty_text"
        if key.shape != query.shape or value.shape[:3] != query.shape[:3]:
            return "bad_qkv_shape"
        if g.shape != query.shape[:3] or beta.shape != query.shape[:3]:
            return "bad_gate_shape"
        if initial_state.shape != (batch_size, heads, key_dim, value.shape[-1]):
            return "bad_initial_state_shape"
        anchor_chunk_size = int(getattr(self.config, "anchor_chunk_size", 0) or 0)
        anchor_token_num = int(getattr(self.config, "anchor_token_num", 0) or 0)
        short_prefill_threshold = int(
            os.environ.get(
                "VLLM_HSD_VARLEN_MIN_FRESH_PREFILL_TOKENS",
                str(anchor_chunk_size),
            )
        )
        if (
            short_prefill_threshold > 0
            and int(boundary_start) <= 0
            and text_len < short_prefill_threshold
        ):
            return "short_prefill_segments_dispatch"
        arbitrary_anchor_kernel = _env_truthy("VLLM_HSD_ENABLE_ARBITRARY_ANCHOR_KERNEL")
        if anchor_chunk_size <= 0 or anchor_chunk_size % 64 != 0:
            return "bad_anchor_chunk_size"
        if anchor_token_num != 1:
            return "anchor_token_num_not_one"
        if int(boundary_start) % anchor_chunk_size != 0 and not arbitrary_anchor_kernel:
            return "boundary_start_not_aligned"
        anchor_count = int(anchor_q.shape[1])
        if anchor_count != int(boundary_counts.numel()):
            return "anchor_count_mismatch"
        # The current packed helper assumes every anchor follows a positive
        # text segment.  Keep the boundary-zero memory row on the exact eager
        # segments path until its dedicated kernel contract is selected.
        if anchor_count and bool((boundary_counts == 0).any().item()):
            return "persistent_memory_boundary_zero"
        expected_count = (int(boundary_start) + text_len) // anchor_chunk_size - int(
            boundary_start
        ) // anchor_chunk_size
        if anchor_count != expected_count:
            return "anchor_boundary_count_not_regular"
        if anchor_count:
            first_boundary = (
                (int(boundary_start) // anchor_chunk_size) + 1
            ) * anchor_chunk_size
            expected = torch.arange(
                first_boundary,
                first_boundary + anchor_count * anchor_chunk_size,
                anchor_chunk_size,
                dtype=boundary_counts.dtype,
                device=boundary_counts.device,
            )
            if not torch.equal(boundary_counts, expected):
                return "anchor_boundaries_not_regular"
        return None

    def _run_hsd_prefill_varlen(
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        reason = self._hsd_varlen_prefill_fallback_reason(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            anchor_q=anchor_q,
            boundary_counts=boundary_counts,
            boundary_start=boundary_start,
            initial_state=initial_state,
        )
        if reason is not None:
            if reason != "backend_segments":
                self._record_hsd_varlen_fallback(reason)
            return None

        varlen_forward = backends._load_hsd_varlen_prefill()
        if varlen_forward is None:
            self._record_hsd_varlen_fallback("import_failed")
            if backends._HSD_VARLEN_PREFILL_IMPORT_ERROR is not None:
                logger.warning_once(
                    "HSD varlen prefill backend is selected but the helper "
                    "could not be imported: %s. Falling back to segments.",
                    backends._HSD_VARLEN_PREFILL_IMPORT_ERROR,
                )
            return None

        text_len = int(query.shape[1])
        anchor_count = int(anchor_q.shape[1])
        anchor_chunk_size = int(getattr(self.config, "anchor_chunk_size"))
        arbitrary_anchor_kernel = _env_truthy("VLLM_HSD_ENABLE_ARBITRARY_ANCHOR_KERNEL")
        cu_seqlens = torch.tensor([0, text_len], dtype=torch.int32, device=query.device)
        anchor_offsets = torch.tensor(
            [0, anchor_count], dtype=torch.long, device=query.device
        )
        local_anchor_positions = None
        if (
            arbitrary_anchor_kernel
            and anchor_count
            and int(boundary_start) % anchor_chunk_size != 0
        ):
            local_anchor_positions = (
                boundary_counts.to(device=query.device, dtype=torch.long)
                - int(boundary_start)
            ).contiguous()
        initial_state_for_varlen = initial_state.float().contiguous()
        if _hsd_trace_requested():
            self._trace_internal("tail_initial_recurrent_state", initial_state)
        result = varlen_forward(
            query=query.contiguous(),
            key=key.contiguous(),
            value=value.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            cu_seqlens=cu_seqlens,
            anchor_offsets=anchor_offsets,
            anchor_chunk_size=anchor_chunk_size,
            initial_state=initial_state_for_varlen,
            scale=self.head_k_dim**-0.5,
            local_anchor_positions=local_anchor_positions,
            use_deterministic_kkt=self._use_deterministic_varlen_kkt(),
        )
        if result is None:
            self._record_hsd_varlen_fallback("kernel_returned_none")
            return None

        logger.info_once("Using HSD varlen packed prefill helper")
        text_o, anchor_states_flat, final_state_flat = result
        # The helper already returns HSD-native [H, K, V] states.
        anchor_states_flat = anchor_states_flat.contiguous()
        final_state_flat = final_state_flat.contiguous()
        recurrent_state = final_state_flat.contiguous()
        if _hsd_trace_requested():
            self._trace_internal("tail_final_recurrent_state", recurrent_state)
        anchor_states = anchor_states_flat.unsqueeze(0).contiguous()
        if anchor_count:
            anchor_o = (
                torch.einsum(
                    "bmhk,bmhkv->bmhv",
                    anchor_q.to(anchor_states.dtype),
                    anchor_states,
                )
                * (self.head_k_dim**-0.5)
            ).to(text_o.dtype)
        else:
            anchor_o = query.new_zeros(1, 0, int(query.shape[2]), int(value.shape[-1]))
        return text_o.to(query.dtype), anchor_o, anchor_states, recurrent_state

    def _run_hsd_prefill_flashqla(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        anchor_q: torch.Tensor,
        boundary_counts: torch.Tensor,
        boundary_start: int,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Run the shared Hopper local-state producer used by MARCH training."""

        if self.hsd_local_prefill_producer_backend != "flashqla_tile":
            return None

        def miss(reason: str):
            if self.hsd_local_prefill_producer_strict:
                import_error = backends._HSD_FLASHQLA_PREFILL_IMPORT_ERROR
                detail = (
                    f"; import_error={type(import_error).__name__}: {import_error}"
                    if import_error is not None
                    else ""
                )
                raise RuntimeError(
                    f"Strict MARCH FlashQLA prefill producer missed: {reason}{detail}"
                )
            return None

        if torch.is_grad_enabled():
            return miss("grad_enabled")
        if not query.is_cuda:
            return miss("not_cuda")
        if int(boundary_start) != 0:
            return miss(f"non_fresh_prefill_start={boundary_start}")
        if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
            return miss("invalid_qkv_shape")
        batch_size, text_len, heads, key_dim = query.shape
        value_dim = int(value.shape[-1])
        if batch_size != 1 or text_len <= 0:
            return miss(f"unsupported_batch_or_length={batch_size}x{text_len}")
        if value.shape[:3] != (batch_size, text_len, heads):
            return miss("invalid_value_shape")
        if g.shape != query.shape[:3] or beta.shape != query.shape[:3]:
            return miss("invalid_gate_shape")
        if initial_state.shape != (batch_size, heads, key_dim, value_dim):
            return miss("invalid_initial_state_shape")
        anchor_chunk_size = int(getattr(self.config, "anchor_chunk_size", 0) or 0)
        if anchor_chunk_size != 512:
            return miss(f"unsupported_anchor_chunk_size={anchor_chunk_size}")
        anchor_count = int(anchor_q.shape[1])
        if anchor_q.shape != (batch_size, anchor_count, heads, key_dim):
            return miss("invalid_anchor_query_shape")
        if int(boundary_counts.numel()) != anchor_count:
            return miss("anchor_boundary_count_mismatch")
        expected_count = text_len // anchor_chunk_size
        if anchor_count != expected_count:
            return miss(
                f"irregular_anchor_count={anchor_count},expected={expected_count}"
            )
        if anchor_count:
            expected_boundaries = torch.arange(
                anchor_chunk_size,
                (anchor_count + 1) * anchor_chunk_size,
                anchor_chunk_size,
                dtype=boundary_counts.dtype,
                device=boundary_counts.device,
            )
            if not torch.equal(boundary_counts, expected_boundaries):
                return miss("irregular_anchor_boundaries")

        producer = backends._load_hsd_flashqla_prefill()
        if producer is None:
            return miss("import_failed")
        text_cu_seqlens = torch.tensor(
            [0, text_len], dtype=torch.int32, device=query.device
        )
        anchor_cu_seqlens = torch.tensor(
            [0, anchor_count], dtype=torch.int32, device=query.device
        )
        result = producer(
            query=query.contiguous(),
            key=key.contiguous(),
            value=value.contiguous(),
            # The MCore/HF training producer receives the sigmoid gate in the
            # activation dtype and the log-decay gate in FP32.
            beta=beta.to(dtype=query.dtype).contiguous(),
            g=g.float().contiguous(),
            boundary_counts=boundary_counts.contiguous(),
            text_cu_seqlens=text_cu_seqlens,
            anchor_cu_seqlens=anchor_cu_seqlens,
            # This branch is admitted only for a fresh request
            # (boundary_start == 0).  The Stage4 training producer starts
            # every packed sequence with ``initial_state=None``; passing an
            # explicit all-zero tensor is mathematically equivalent but uses
            # a different public launch contract.  Keep serving on the exact
            # fresh-sequence contract exercised by training.
            initial_state=None,
            scale=self.head_k_dim**-0.5,
        )
        if result is None:
            return miss("kernel_returned_none")
        text_o, anchor_states, recurrent_state = result
        expected_text_shape = (batch_size, text_len, heads, value_dim)
        expected_anchor_shape = (
            batch_size,
            anchor_count,
            heads,
            key_dim,
            value_dim,
        )
        expected_state_shape = (batch_size, heads, key_dim, value_dim)
        if tuple(text_o.shape) != expected_text_shape:
            return miss(f"invalid_text_output_shape={tuple(text_o.shape)}")
        if tuple(anchor_states.shape) != expected_anchor_shape:
            return miss(f"invalid_anchor_state_shape={tuple(anchor_states.shape)}")
        if tuple(recurrent_state.shape) != expected_state_shape:
            return miss(f"invalid_final_state_shape={tuple(recurrent_state.shape)}")
        if anchor_count:
            anchor_o = (
                torch.einsum(
                    "bmhk,bmhkv->bmhv",
                    anchor_q.to(anchor_states.dtype),
                    anchor_states,
                )
                * (self.head_k_dim**-0.5)
            ).to(text_o.dtype)
        else:
            anchor_o = query.new_zeros(batch_size, 0, heads, value_dim)
        logger.info_once("Using Stage4 training-parity FlashQLA GDN prefill producer")
        self._record_hsd_varlen_event(
            "flashqla_tile_prefill_hit",
            requests=1,
            text_tokens=text_len,
            anchor_tokens=anchor_count,
        )
        return (
            text_o.to(query.dtype),
            anchor_o,
            anchor_states.contiguous(),
            recurrent_state.contiguous(),
        )

    def _run_hsd_prefill_vllm(
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
        flashqla_result = self._run_hsd_prefill_flashqla(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            anchor_q=anchor_q,
            boundary_counts=boundary_counts,
            boundary_start=boundary_start,
            initial_state=initial_state,
        )
        if flashqla_result is not None:
            return flashqla_result
        if not getattr(self, "_hsd_force_segments_fallback", False):
            varlen_result = self._run_hsd_prefill_varlen(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                anchor_q=anchor_q,
                boundary_counts=boundary_counts,
                boundary_start=boundary_start,
                initial_state=initial_state,
            )
            if varlen_result is not None:
                return varlen_result
        return self._run_hsd_segments(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            anchor_q=anchor_q,
            boundary_counts=boundary_counts,
            boundary_start=boundary_start,
            initial_state=initial_state,
        )
