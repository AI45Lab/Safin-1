"""MARCH paged history paged prefill."""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU reference does not require Triton.
    triton = None
    tl = None

if triton is not None:
    @triton.jit
    def _paged_prefill_top4_blocked_read_kernel(
        read_q,
        states_pages,
        page_table,
        anchor_lengths,
        request_state_indices,
        text_offsets,
        null_logits,
        top_indices,
        top_scores,
        history_scale,
        out,
        heads: tl.constexpr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        read_scale: tl.constexpr,
        read_q_stride_b: tl.constexpr,
        read_q_stride_h: tl.constexpr,
        read_q_stride_k: tl.constexpr,
        states_stride_p: tl.constexpr,
        states_stride_s: tl.constexpr,
        states_stride_h: tl.constexpr,
        states_stride_v: tl.constexpr,
        states_stride_k: tl.constexpr,
        page_table_stride_b: tl.constexpr,
        page_table_stride_p: tl.constexpr,
        null_logits_stride_b: tl.constexpr,
        null_logits_stride_h: tl.constexpr,
        top_indices_stride_b: tl.constexpr,
        top_indices_stride_h: tl.constexpr,
        top_indices_stride_t: tl.constexpr,
        top_scores_stride_b: tl.constexpr,
        top_scores_stride_h: tl.constexpr,
        top_scores_stride_t: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_v: tl.constexpr,
        max_pages: tl.constexpr,
        state_capacity: tl.constexpr,
        HAS_NULL: tl.constexpr,
        PAGE_M: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        UNIFORM_MODE: tl.constexpr,
    ):
        request_head_idx = tl.program_id(0)
        request_idx = request_head_idx // heads
        head_idx = request_head_idx - request_idx * heads
        token_block_idx = tl.program_id(1)
        value_block_idx = tl.program_id(2)
        offs_t = tl.arange(0, BLOCK_T)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = value_block_idx * BLOCK_V + tl.arange(0, BLOCK_V)

        text_begin = tl.load(text_offsets + request_idx).to(tl.int64)
        text_end = tl.load(text_offsets + request_idx + 1).to(tl.int64)
        token_idx = text_begin + token_block_idx * BLOCK_T + offs_t
        token_valid = token_idx < text_end
        state_idx = tl.load(request_state_indices + request_idx).to(tl.int64)
        state_valid = (state_idx >= 0) & (state_idx < state_capacity)
        metadata_row = tl.where(state_valid, state_idx, 0)
        anchor_len = tl.load(
            anchor_lengths + metadata_row,
            mask=state_valid,
            other=0,
        )

        top_idx_base = (
            top_indices
            + token_idx * top_indices_stride_b
            + head_idx * top_indices_stride_h
        )
        top_score_base = (
            top_scores
            + token_idx * top_scores_stride_b
            + head_idx * top_scores_stride_h
        )
        best_idx0 = tl.load(
            top_idx_base + 0 * top_indices_stride_t,
            mask=token_valid,
            other=0,
        )
        best_idx1 = tl.load(
            top_idx_base + 1 * top_indices_stride_t,
            mask=token_valid,
            other=0,
        )
        best_idx2 = tl.load(
            top_idx_base + 2 * top_indices_stride_t,
            mask=token_valid,
            other=0,
        )
        best_idx3 = tl.load(
            top_idx_base + 3 * top_indices_stride_t,
            mask=token_valid,
            other=0,
        )
        best_val0 = tl.load(
            top_score_base + 0 * top_scores_stride_t,
            mask=token_valid,
            other=-float("inf"),
        ).to(tl.float32)
        best_val1 = tl.load(
            top_score_base + 1 * top_scores_stride_t,
            mask=token_valid,
            other=-float("inf"),
        ).to(tl.float32)
        best_val2 = tl.load(
            top_score_base + 2 * top_scores_stride_t,
            mask=token_valid,
            other=-float("inf"),
        ).to(tl.float32)
        best_val3 = tl.load(
            top_score_base + 3 * top_scores_stride_t,
            mask=token_valid,
            other=-float("inf"),
        ).to(tl.float32)

        has0 = best_val0 != -float("inf")
        has1 = best_val1 != -float("inf")
        has2 = best_val2 != -float("inf")
        has3 = best_val3 != -float("inf")
        max_val = tl.maximum(
            tl.maximum(best_val0, best_val1),
            tl.maximum(best_val2, best_val3),
        )
        if HAS_NULL:
            null_val = tl.load(
                null_logits
                + token_idx * null_logits_stride_b
                + head_idx * null_logits_stride_h,
                mask=token_valid,
                other=-float("inf"),
            ).to(tl.float32)
            max_val = tl.maximum(max_val, null_val)
            exp_null = tl.exp(null_val - max_val)
        else:
            null_val = tl.full((BLOCK_T,), -float("inf"), tl.float32)
            has_any = has0 | has1 | has2 | has3
            max_val = tl.where(has_any, max_val, 0.0)
            exp_null = 0.0
        exp0 = tl.where(has0, tl.exp(best_val0 - max_val), 0.0)
        exp1 = tl.where(has1, tl.exp(best_val1 - max_val), 0.0)
        exp2 = tl.where(has2, tl.exp(best_val2 - max_val), 0.0)
        exp3 = tl.where(has3, tl.exp(best_val3 - max_val), 0.0)
        denom = exp0 + exp1 + exp2 + exp3 + exp_null
        inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
        w0 = exp0 * inv_denom
        w1 = exp1 * inv_denom
        w2 = exp2 * inv_denom
        w3 = exp3 * inv_denom

        read = tl.load(
            read_q
            + token_idx[:, None] * read_q_stride_b
            + head_idx * read_q_stride_h
            + offs_k[None, :] * read_q_stride_k,
            mask=token_valid[:, None] & (offs_k[None, :] < key_dim),
            other=0.0,
        ).to(tl.float32)

        page_offset0 = 0 // PAGE_M
        page_offset1 = 1 // PAGE_M
        page_offset2 = 2 // PAGE_M
        page_offset3 = 3 // PAGE_M
        page_id0 = tl.load(
            page_table + metadata_row * page_table_stride_b + page_offset0 * page_table_stride_p,
            mask=state_valid & (anchor_len > 0) & (page_offset0 < max_pages),
            other=0,
        )
        page_id1 = tl.load(
            page_table + metadata_row * page_table_stride_b + page_offset1 * page_table_stride_p,
            mask=state_valid & (anchor_len > 1) & (page_offset1 < max_pages),
            other=0,
        )
        page_id2 = tl.load(
            page_table + metadata_row * page_table_stride_b + page_offset2 * page_table_stride_p,
            mask=state_valid & (anchor_len > 2) & (page_offset2 < max_pages),
            other=0,
        )
        page_id3 = tl.load(
            page_table + metadata_row * page_table_stride_b + page_offset3 * page_table_stride_p,
            mask=state_valid & (anchor_len > 3) & (page_offset3 < max_pages),
            other=0,
        )

        state0 = tl.load(
            states_pages
            + page_id0 * states_stride_p
            + (0 % PAGE_M) * states_stride_s
            + head_idx * states_stride_h
            + offs_v[None, :] * states_stride_v
            + offs_k[:, None] * states_stride_k,
            mask=(anchor_len > 0) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state1 = tl.load(
            states_pages
            + page_id1 * states_stride_p
            + (1 % PAGE_M) * states_stride_s
            + head_idx * states_stride_h
            + offs_v[None, :] * states_stride_v
            + offs_k[:, None] * states_stride_k,
            mask=(anchor_len > 1) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state2 = tl.load(
            states_pages
            + page_id2 * states_stride_p
            + (2 % PAGE_M) * states_stride_s
            + head_idx * states_stride_h
            + offs_v[None, :] * states_stride_v
            + offs_k[:, None] * states_stride_k,
            mask=(anchor_len > 2) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state3 = tl.load(
            states_pages
            + page_id3 * states_stride_p
            + (3 % PAGE_M) * states_stride_s
            + head_idx * states_stride_h
            + offs_v[None, :] * states_stride_v
            + offs_k[:, None] * states_stride_k,
            mask=(anchor_len > 3) & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        read0 = tl.sum(read[:, :, None] * state0[None, :, :], axis=1) * read_scale
        read1 = tl.sum(read[:, :, None] * state1[None, :, :], axis=1) * read_scale
        read2 = tl.sum(read[:, :, None] * state2[None, :, :], axis=1) * read_scale
        read3 = tl.sum(read[:, :, None] * state3[None, :, :], axis=1) * read_scale

        selected0 = tl.where(
            best_idx0[:, None] == 0,
            read0,
            tl.where(
                best_idx0[:, None] == 1,
                read1,
                tl.where(best_idx0[:, None] == 2, read2, read3),
            ),
        )
        selected1 = tl.where(
            best_idx1[:, None] == 0,
            read0,
            tl.where(
                best_idx1[:, None] == 1,
                read1,
                tl.where(best_idx1[:, None] == 2, read2, read3),
            ),
        )
        selected2 = tl.where(
            best_idx2[:, None] == 0,
            read0,
            tl.where(
                best_idx2[:, None] == 1,
                read1,
                tl.where(best_idx2[:, None] == 2, read2, read3),
            ),
        )
        selected3 = tl.where(
            best_idx3[:, None] == 0,
            read0,
            tl.where(
                best_idx3[:, None] == 1,
                read1,
                tl.where(best_idx3[:, None] == 2, read2, read3),
            ),
        )
        result = (
            selected0 * w0[:, None]
            + selected1 * w1[:, None]
            + selected2 * w2[:, None]
            + selected3 * w3[:, None]
        )
        out_ptr = (
            out
            + token_idx[:, None] * out_stride_b
            + head_idx * out_stride_h
            + offs_v[None, :] * out_stride_v
        )
        base = tl.load(
            out_ptr,
            mask=token_valid[:, None] & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        rounded = result.to(out.dtype.element_ty).to(tl.float32)
        scale_value = tl.load(history_scale).to(tl.float32)
        result = base + rounded * scale_value
        tl.store(
            out_ptr,
            result,
            mask=token_valid[:, None] & (offs_v[None, :] < value_dim),
        )


    @triton.jit
    def _paged_prefill_top4_selected_blocked_read_kernel(
        read_q,
        states_pages,
        page_table,
        anchor_lengths,
        request_state_indices,
        text_offsets,
        null_logits,
        top_indices,
        top_scores,
        history_scale,
        out,
        heads: tl.constexpr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        read_scale: tl.constexpr,
        read_q_stride_b: tl.constexpr,
        read_q_stride_h: tl.constexpr,
        read_q_stride_k: tl.constexpr,
        states_stride_p: tl.constexpr,
        states_stride_s: tl.constexpr,
        states_stride_h: tl.constexpr,
        states_stride_v: tl.constexpr,
        states_stride_k: tl.constexpr,
        page_table_stride_b: tl.constexpr,
        page_table_stride_p: tl.constexpr,
        null_logits_stride_b: tl.constexpr,
        null_logits_stride_h: tl.constexpr,
        top_indices_stride_b: tl.constexpr,
        top_indices_stride_h: tl.constexpr,
        top_indices_stride_t: tl.constexpr,
        top_scores_stride_b: tl.constexpr,
        top_scores_stride_h: tl.constexpr,
        top_scores_stride_t: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_v: tl.constexpr,
        max_pages: tl.constexpr,
        state_capacity: tl.constexpr,
        HAS_NULL: tl.constexpr,
        PAGE_M: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        UNIFORM_MODE: tl.constexpr,
    ):
        request_head_idx = tl.program_id(0)
        request_idx = request_head_idx // heads
        head_idx = request_head_idx - request_idx * heads
        token_block_idx = tl.program_id(1)
        value_block_idx = tl.program_id(2)
        offs_t = tl.arange(0, BLOCK_T)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = value_block_idx * BLOCK_V + tl.arange(0, BLOCK_V)

        text_begin = tl.load(text_offsets + request_idx).to(tl.int64)
        text_end = tl.load(text_offsets + request_idx + 1).to(tl.int64)
        token_idx = text_begin + token_block_idx * BLOCK_T + offs_t
        token_valid = token_idx < text_end
        state_idx = tl.load(request_state_indices + request_idx).to(tl.int64)
        state_valid = (state_idx >= 0) & (state_idx < state_capacity)
        metadata_row = tl.where(state_valid, state_idx, 0)

        top_idx_base = (
            top_indices
            + token_idx * top_indices_stride_b
            + head_idx * top_indices_stride_h
        )
        top_score_base = (
            top_scores
            + token_idx * top_scores_stride_b
            + head_idx * top_scores_stride_h
        )
        best_idx0 = tl.load(
            top_idx_base + 0 * top_indices_stride_t,
            mask=token_valid,
            other=0,
        )
        best_idx1 = tl.load(
            top_idx_base + 1 * top_indices_stride_t,
            mask=token_valid,
            other=0,
        )
        best_idx2 = tl.load(
            top_idx_base + 2 * top_indices_stride_t,
            mask=token_valid,
            other=0,
        )
        best_idx3 = tl.load(
            top_idx_base + 3 * top_indices_stride_t,
            mask=token_valid,
            other=0,
        )
        best_val0 = tl.load(
            top_score_base + 0 * top_scores_stride_t,
            mask=token_valid,
            other=-float("inf"),
        ).to(tl.float32)
        best_val1 = tl.load(
            top_score_base + 1 * top_scores_stride_t,
            mask=token_valid,
            other=-float("inf"),
        ).to(tl.float32)
        best_val2 = tl.load(
            top_score_base + 2 * top_scores_stride_t,
            mask=token_valid,
            other=-float("inf"),
        ).to(tl.float32)
        best_val3 = tl.load(
            top_score_base + 3 * top_scores_stride_t,
            mask=token_valid,
            other=-float("inf"),
        ).to(tl.float32)

        has0 = token_valid & (best_val0 != -float("inf"))
        has1 = token_valid & (best_val1 != -float("inf"))
        has2 = token_valid & (best_val2 != -float("inf"))
        has3 = token_valid & (best_val3 != -float("inf"))
        if UNIFORM_MODE != 0:
            first_token_idx = text_begin + token_block_idx * BLOCK_T
            first_idx_base = (
                top_indices
                + first_token_idx * top_indices_stride_b
                + head_idx * top_indices_stride_h
            )
            first_score_base = (
                top_scores
                + first_token_idx * top_scores_stride_b
                + head_idx * top_scores_stride_h
            )
            ref_idx0 = tl.load(first_idx_base + 0 * top_indices_stride_t)
            ref_idx1 = tl.load(first_idx_base + 1 * top_indices_stride_t)
            ref_idx2 = tl.load(first_idx_base + 2 * top_indices_stride_t)
            ref_idx3 = tl.load(first_idx_base + 3 * top_indices_stride_t)
            ref_has0 = (
                tl.load(first_score_base + 0 * top_scores_stride_t).to(tl.float32)
                != -float("inf")
            )
            ref_has1 = (
                tl.load(first_score_base + 1 * top_scores_stride_t).to(tl.float32)
                != -float("inf")
            )
            ref_has2 = (
                tl.load(first_score_base + 2 * top_scores_stride_t).to(tl.float32)
                != -float("inf")
            )
            ref_has3 = (
                tl.load(first_score_base + 3 * top_scores_stride_t).to(tl.float32)
                != -float("inf")
            )
            uniform0 = tl.sum(
                ((~token_valid) | ((has0 == ref_has0) & ((~has0) | (best_idx0 == ref_idx0)))).to(tl.int32),
                axis=0,
            ) == BLOCK_T
            uniform1 = tl.sum(
                ((~token_valid) | ((has1 == ref_has1) & ((~has1) | (best_idx1 == ref_idx1)))).to(tl.int32),
                axis=0,
            ) == BLOCK_T
            uniform2 = tl.sum(
                ((~token_valid) | ((has2 == ref_has2) & ((~has2) | (best_idx2 == ref_idx2)))).to(tl.int32),
                axis=0,
            ) == BLOCK_T
            uniform3 = tl.sum(
                ((~token_valid) | ((has3 == ref_has3) & ((~has3) | (best_idx3 == ref_idx3)))).to(tl.int32),
                axis=0,
            ) == BLOCK_T
            uniform = uniform0 & uniform1 & uniform2 & uniform3
            if UNIFORM_MODE == 1:
                token_valid = token_valid & uniform
            else:
                token_valid = token_valid & (~uniform)
            has0 = has0 & token_valid
            has1 = has1 & token_valid
            has2 = has2 & token_valid
            has3 = has3 & token_valid
        max_val = tl.maximum(
            tl.maximum(best_val0, best_val1),
            tl.maximum(best_val2, best_val3),
        )
        if HAS_NULL:
            null_val = tl.load(
                null_logits
                + token_idx * null_logits_stride_b
                + head_idx * null_logits_stride_h,
                mask=token_valid,
                other=-float("inf"),
            ).to(tl.float32)
            max_val = tl.maximum(max_val, null_val)
            exp_null = tl.exp(null_val - max_val)
        else:
            has_any = has0 | has1 | has2 | has3
            max_val = tl.where(has_any, max_val, 0.0)
            exp_null = 0.0
        exp0 = tl.where(has0, tl.exp(best_val0 - max_val), 0.0)
        exp1 = tl.where(has1, tl.exp(best_val1 - max_val), 0.0)
        exp2 = tl.where(has2, tl.exp(best_val2 - max_val), 0.0)
        exp3 = tl.where(has3, tl.exp(best_val3 - max_val), 0.0)
        denom = exp0 + exp1 + exp2 + exp3 + exp_null
        inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
        w0 = exp0 * inv_denom
        w1 = exp1 * inv_denom
        w2 = exp2 * inv_denom
        w3 = exp3 * inv_denom

        read = tl.load(
            read_q
            + token_idx[:, None] * read_q_stride_b
            + head_idx * read_q_stride_h
            + offs_k[None, :] * read_q_stride_k,
            mask=token_valid[:, None] & (offs_k[None, :] < key_dim),
            other=0.0,
        ).to(tl.float32)

        if UNIFORM_MODE == 1:
            safe_idx0 = tl.where(ref_has0, ref_idx0, 0)
            safe_idx1 = tl.where(ref_has1, ref_idx1, 0)
            safe_idx2 = tl.where(ref_has2, ref_idx2, 0)
            safe_idx3 = tl.where(ref_has3, ref_idx3, 0)
        else:
            safe_idx0 = tl.where(has0, best_idx0, 0)
            safe_idx1 = tl.where(has1, best_idx1, 0)
            safe_idx2 = tl.where(has2, best_idx2, 0)
            safe_idx3 = tl.where(has3, best_idx3, 0)
        page0 = safe_idx0 // PAGE_M
        page1 = safe_idx1 // PAGE_M
        page2 = safe_idx2 // PAGE_M
        page3 = safe_idx3 // PAGE_M
        slot0 = safe_idx0 - page0 * PAGE_M
        slot1 = safe_idx1 - page1 * PAGE_M
        slot2 = safe_idx2 - page2 * PAGE_M
        slot3 = safe_idx3 - page3 * PAGE_M
        if UNIFORM_MODE == 1:
            page_id0 = tl.load(
                page_table
                + metadata_row * page_table_stride_b
                + page0 * page_table_stride_p,
                mask=state_valid & ref_has0 & (page0 < max_pages),
                other=0,
            )
            page_id1 = tl.load(
                page_table
                + metadata_row * page_table_stride_b
                + page1 * page_table_stride_p,
                mask=state_valid & ref_has1 & (page1 < max_pages),
                other=0,
            )
            page_id2 = tl.load(
                page_table
                + metadata_row * page_table_stride_b
                + page2 * page_table_stride_p,
                mask=state_valid & ref_has2 & (page2 < max_pages),
                other=0,
            )
            page_id3 = tl.load(
                page_table
                + metadata_row * page_table_stride_b
                + page3 * page_table_stride_p,
                mask=state_valid & ref_has3 & (page3 < max_pages),
                other=0,
            )
            state0 = tl.load(
                states_pages
                + page_id0 * states_stride_p
                + slot0 * states_stride_s
                + head_idx * states_stride_h
                + offs_v[None, :] * states_stride_v
                + offs_k[:, None] * states_stride_k,
                mask=ref_has0
                & (offs_k[:, None] < key_dim)
                & (offs_v[None, :] < value_dim),
                other=0.0,
            ).to(tl.float32)
            read0 = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
            for token_offset in tl.static_range(0, BLOCK_T):
                read_token = tl.load(
                    read_q
                    + (first_token_idx + token_offset) * read_q_stride_b
                    + head_idx * read_q_stride_h
                    + offs_k * read_q_stride_k,
                    mask=(first_token_idx + token_offset < text_end)
                    & (offs_k < key_dim),
                    other=0.0,
                ).to(tl.float32)
                read_value = (
                    tl.sum(read_token[:, None] * state0, axis=0) * read_scale
                )
                read0 += tl.where(
                    offs_t[:, None] == token_offset,
                    read_value[None, :],
                    0.0,
                )
            state1 = tl.load(
                states_pages
                + page_id1 * states_stride_p
                + slot1 * states_stride_s
                + head_idx * states_stride_h
                + offs_v[None, :] * states_stride_v
                + offs_k[:, None] * states_stride_k,
                mask=ref_has1
                & (offs_k[:, None] < key_dim)
                & (offs_v[None, :] < value_dim),
                other=0.0,
            ).to(tl.float32)
            read1 = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
            for token_offset in tl.static_range(0, BLOCK_T):
                read_token = tl.load(
                    read_q
                    + (first_token_idx + token_offset) * read_q_stride_b
                    + head_idx * read_q_stride_h
                    + offs_k * read_q_stride_k,
                    mask=(first_token_idx + token_offset < text_end)
                    & (offs_k < key_dim),
                    other=0.0,
                ).to(tl.float32)
                read_value = (
                    tl.sum(read_token[:, None] * state1, axis=0) * read_scale
                )
                read1 += tl.where(
                    offs_t[:, None] == token_offset,
                    read_value[None, :],
                    0.0,
                )
            state2 = tl.load(
                states_pages
                + page_id2 * states_stride_p
                + slot2 * states_stride_s
                + head_idx * states_stride_h
                + offs_v[None, :] * states_stride_v
                + offs_k[:, None] * states_stride_k,
                mask=ref_has2
                & (offs_k[:, None] < key_dim)
                & (offs_v[None, :] < value_dim),
                other=0.0,
            ).to(tl.float32)
            read2 = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
            for token_offset in tl.static_range(0, BLOCK_T):
                read_token = tl.load(
                    read_q
                    + (first_token_idx + token_offset) * read_q_stride_b
                    + head_idx * read_q_stride_h
                    + offs_k * read_q_stride_k,
                    mask=(first_token_idx + token_offset < text_end)
                    & (offs_k < key_dim),
                    other=0.0,
                ).to(tl.float32)
                read_value = (
                    tl.sum(read_token[:, None] * state2, axis=0) * read_scale
                )
                read2 += tl.where(
                    offs_t[:, None] == token_offset,
                    read_value[None, :],
                    0.0,
                )
            state3 = tl.load(
                states_pages
                + page_id3 * states_stride_p
                + slot3 * states_stride_s
                + head_idx * states_stride_h
                + offs_v[None, :] * states_stride_v
                + offs_k[:, None] * states_stride_k,
                mask=ref_has3
                & (offs_k[:, None] < key_dim)
                & (offs_v[None, :] < value_dim),
                other=0.0,
            ).to(tl.float32)
            read3 = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
            for token_offset in tl.static_range(0, BLOCK_T):
                read_token = tl.load(
                    read_q
                    + (first_token_idx + token_offset) * read_q_stride_b
                    + head_idx * read_q_stride_h
                    + offs_k * read_q_stride_k,
                    mask=(first_token_idx + token_offset < text_end)
                    & (offs_k < key_dim),
                    other=0.0,
                ).to(tl.float32)
                read_value = (
                    tl.sum(read_token[:, None] * state3, axis=0) * read_scale
                )
                read3 += tl.where(
                    offs_t[:, None] == token_offset,
                    read_value[None, :],
                    0.0,
                )
        else:
            page_id0 = tl.load(
                page_table
                + metadata_row * page_table_stride_b
                + page0 * page_table_stride_p,
                mask=state_valid & has0 & (page0 < max_pages),
                other=0,
            )
            page_id1 = tl.load(
                page_table
                + metadata_row * page_table_stride_b
                + page1 * page_table_stride_p,
                mask=state_valid & has1 & (page1 < max_pages),
                other=0,
            )
            page_id2 = tl.load(
                page_table
                + metadata_row * page_table_stride_b
                + page2 * page_table_stride_p,
                mask=state_valid & has2 & (page2 < max_pages),
                other=0,
            )
            page_id3 = tl.load(
                page_table
                + metadata_row * page_table_stride_b
                + page3 * page_table_stride_p,
                mask=state_valid & has3 & (page3 < max_pages),
                other=0,
            )
            state0 = tl.load(
                states_pages
                + page_id0[:, None, None] * states_stride_p
                + slot0[:, None, None] * states_stride_s
                + head_idx * states_stride_h
                + offs_v[None, None, :] * states_stride_v
                + offs_k[None, :, None] * states_stride_k,
                mask=has0[:, None, None]
                & (offs_k[None, :, None] < key_dim)
                & (offs_v[None, None, :] < value_dim),
                other=0.0,
            ).to(tl.float32)
            read0 = tl.sum(read[:, :, None] * state0, axis=1) * read_scale
            state1 = tl.load(
                states_pages
                + page_id1[:, None, None] * states_stride_p
                + slot1[:, None, None] * states_stride_s
                + head_idx * states_stride_h
                + offs_v[None, None, :] * states_stride_v
                + offs_k[None, :, None] * states_stride_k,
                mask=has1[:, None, None]
                & (offs_k[None, :, None] < key_dim)
                & (offs_v[None, None, :] < value_dim),
                other=0.0,
            ).to(tl.float32)
            read1 = tl.sum(read[:, :, None] * state1, axis=1) * read_scale
            state2 = tl.load(
                states_pages
                + page_id2[:, None, None] * states_stride_p
                + slot2[:, None, None] * states_stride_s
                + head_idx * states_stride_h
                + offs_v[None, None, :] * states_stride_v
                + offs_k[None, :, None] * states_stride_k,
                mask=has2[:, None, None]
                & (offs_k[None, :, None] < key_dim)
                & (offs_v[None, None, :] < value_dim),
                other=0.0,
            ).to(tl.float32)
            read2 = tl.sum(read[:, :, None] * state2, axis=1) * read_scale
            state3 = tl.load(
                states_pages
                + page_id3[:, None, None] * states_stride_p
                + slot3[:, None, None] * states_stride_s
                + head_idx * states_stride_h
                + offs_v[None, None, :] * states_stride_v
                + offs_k[None, :, None] * states_stride_k,
                mask=has3[:, None, None]
                & (offs_k[None, :, None] < key_dim)
                & (offs_v[None, None, :] < value_dim),
                other=0.0,
            ).to(tl.float32)
            read3 = tl.sum(read[:, :, None] * state3, axis=1) * read_scale
        result = (
            read0 * w0[:, None]
            + read1 * w1[:, None]
            + read2 * w2[:, None]
            + read3 * w3[:, None]
        )
        out_ptr = (
            out
            + token_idx[:, None] * out_stride_b
            + head_idx * out_stride_h
            + offs_v[None, :] * out_stride_v
        )
        base = tl.load(
            out_ptr,
            mask=token_valid[:, None] & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        rounded = result.to(out.dtype.element_ty).to(tl.float32)
        scale_value = tl.load(history_scale).to(tl.float32)
        result = base + rounded * scale_value
        tl.store(
            out_ptr,
            result,
            mask=token_valid[:, None] & (offs_v[None, :] < value_dim),
        )

