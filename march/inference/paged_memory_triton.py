"""MARCH paged history paged memory."""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU reference does not require Triton.
    triton = None
    tl = None

if triton is not None:
    @triton.jit
    def _paged_decode_memory_top4_select_kernel(
        router_q,
        keys_pages,
        boundary_pages,
        page_table,
        anchor_lengths,
        start_indices,
        state_indices,
        valid_mask,
        top_indices,
        top_scores,
        key_dim: tl.constexpr,
        route_scale: tl.constexpr,
        router_q_stride_b: tl.constexpr,
        router_q_stride_h: tl.constexpr,
        router_q_stride_k: tl.constexpr,
        keys_stride_p: tl.constexpr,
        keys_stride_s: tl.constexpr,
        keys_stride_h: tl.constexpr,
        keys_stride_k: tl.constexpr,
        boundary_stride_p: tl.constexpr,
        boundary_stride_s: tl.constexpr,
        page_table_stride_b: tl.constexpr,
        page_table_stride_p: tl.constexpr,
        top_indices_stride_b: tl.constexpr,
        top_indices_stride_h: tl.constexpr,
        top_indices_stride_t: tl.constexpr,
        top_scores_stride_b: tl.constexpr,
        top_scores_stride_h: tl.constexpr,
        top_scores_stride_t: tl.constexpr,
        max_pages: tl.constexpr,
        state_capacity: tl.constexpr,
        STATIC_STATE_METADATA: tl.constexpr,
        IGNORE_BOUNDARY_START: tl.constexpr,
        NORMALIZE_ROUTER_Q: tl.constexpr,
        PAGE_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Select one boundary-zero lane plus four non-memory lanes."""
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)
        offs_s = tl.arange(0, PAGE_M)
        offs_c = tl.arange(0, BLOCK_C)

        router = tl.load(
            router_q
            + batch_idx * router_q_stride_b
            + head_idx * router_q_stride_h
            + offs_k * router_q_stride_k,
            mask=offs_k < key_dim,
            other=0.0,
        ).to(tl.float32)
        if NORMALIZE_ROUTER_Q:
            router *= 1.0 / tl.sqrt(tl.sum(router * router, axis=0) + 1e-6)
            router = router.to(router_q.dtype.element_ty).to(tl.float32)

        memory_val = -float("inf")
        memory_idx = 0
        best_val0 = -float("inf")
        best_val1 = -float("inf")
        best_val2 = -float("inf")
        best_val3 = -float("inf")
        best_idx0 = 0
        best_idx1 = 0
        best_idx2 = 0
        best_idx3 = 0
        if STATIC_STATE_METADATA:
            state_idx = tl.load(state_indices + batch_idx).to(tl.int64)
            row_valid = tl.load(valid_mask + batch_idx)
            metadata_valid = row_valid & (state_idx >= 0) & (
                state_idx < state_capacity
            )
            metadata_row = tl.where(metadata_valid, state_idx, 0)
            anchor_len = tl.load(
                anchor_lengths + metadata_row,
                mask=metadata_valid,
                other=0,
            )
        else:
            metadata_row = batch_idx
            anchor_len = tl.load(anchor_lengths + batch_idx)
        if not IGNORE_BOUNDARY_START:
            start_index = tl.load(start_indices + batch_idx)

        for page_pos in tl.range(0, max_pages, loop_unroll_factor=1):
            page_id = tl.load(
                page_table
                + metadata_row * page_table_stride_b
                + page_pos * page_table_stride_p,
                mask=page_pos < max_pages,
                other=-1,
            )
            page_id_safe = tl.maximum(page_id, 0)
            global_m = page_pos * PAGE_M + offs_s
            base_valid = (page_id >= 0) & (global_m < anchor_len)
            boundaries = tl.load(
                boundary_pages
                + page_id_safe * boundary_stride_p
                + offs_s * boundary_stride_s,
                mask=base_valid,
                other=1,
            )
            valid = base_valid
            if not IGNORE_BOUNDARY_START:
                valid = valid & (boundaries <= start_index)
            keys = tl.load(
                keys_pages
                + page_id_safe * keys_stride_p
                + offs_s[:, None] * keys_stride_s
                + head_idx * keys_stride_h
                + offs_k[None, :] * keys_stride_k,
                mask=valid[:, None] & (offs_k[None, :] < key_dim),
                other=0.0,
            ).to(tl.float32)
            logits = tl.sum(keys * router[None, :], axis=1) * route_scale

            memory_scores = tl.where(
                valid & (boundaries == 0), logits, -float("inf")
            )
            page_memory_sel = tl.argmax(
                memory_scores, axis=0, tie_break_left=True
            )
            page_memory_val = tl.max(memory_scores, axis=0)
            take_page_memory = page_memory_val > memory_val
            memory_val = tl.where(
                take_page_memory, page_memory_val, memory_val
            )
            memory_idx = tl.where(
                take_page_memory,
                (page_pos * PAGE_M + page_memory_sel).to(tl.int32),
                memory_idx,
            )

            # Boundary zero is never eligible for a dynamic lane.
            page_scores = tl.where(
                valid & (boundaries != 0), logits, -float("inf")
            )
            page_sel0 = tl.argmax(page_scores, axis=0, tie_break_left=True)
            page_val0 = tl.max(page_scores, axis=0)
            page_scores1 = tl.where(
                offs_s == page_sel0, -float("inf"), page_scores
            )
            page_sel1 = tl.argmax(page_scores1, axis=0, tie_break_left=True)
            page_val1 = tl.max(page_scores1, axis=0)
            page_scores2 = tl.where(
                offs_s == page_sel1, -float("inf"), page_scores1
            )
            page_sel2 = tl.argmax(page_scores2, axis=0, tie_break_left=True)
            page_val2 = tl.max(page_scores2, axis=0)
            page_scores3 = tl.where(
                offs_s == page_sel2, -float("inf"), page_scores2
            )
            page_sel3 = tl.argmax(page_scores3, axis=0, tie_break_left=True)
            page_val3 = tl.max(page_scores3, axis=0)

            cand_scores = tl.full((BLOCK_C,), -float("inf"), tl.float32)
            cand_indices = tl.full((BLOCK_C,), 0, tl.int32)
            cand_scores = tl.where(offs_c == 0, best_val0, cand_scores)
            cand_scores = tl.where(offs_c == 1, best_val1, cand_scores)
            cand_scores = tl.where(offs_c == 2, best_val2, cand_scores)
            cand_scores = tl.where(offs_c == 3, best_val3, cand_scores)
            cand_scores = tl.where(offs_c == 4, page_val0, cand_scores)
            cand_scores = tl.where(offs_c == 5, page_val1, cand_scores)
            cand_scores = tl.where(offs_c == 6, page_val2, cand_scores)
            cand_scores = tl.where(offs_c == 7, page_val3, cand_scores)
            cand_indices = tl.where(offs_c == 0, best_idx0, cand_indices)
            cand_indices = tl.where(offs_c == 1, best_idx1, cand_indices)
            cand_indices = tl.where(offs_c == 2, best_idx2, cand_indices)
            cand_indices = tl.where(offs_c == 3, best_idx3, cand_indices)
            cand_indices = tl.where(
                offs_c == 4,
                (page_pos * PAGE_M + page_sel0).to(tl.int32),
                cand_indices,
            )
            cand_indices = tl.where(
                offs_c == 5,
                (page_pos * PAGE_M + page_sel1).to(tl.int32),
                cand_indices,
            )
            cand_indices = tl.where(
                offs_c == 6,
                (page_pos * PAGE_M + page_sel2).to(tl.int32),
                cand_indices,
            )
            cand_indices = tl.where(
                offs_c == 7,
                (page_pos * PAGE_M + page_sel3).to(tl.int32),
                cand_indices,
            )

            sel0 = tl.argmax(cand_scores, axis=0, tie_break_left=True)
            val0 = tl.max(cand_scores, axis=0)
            idx0 = tl.max(
                tl.where(offs_c == sel0, cand_indices, 0), axis=0
            )
            scores1 = tl.where(offs_c == sel0, -float("inf"), cand_scores)
            sel1 = tl.argmax(scores1, axis=0, tie_break_left=True)
            val1 = tl.max(scores1, axis=0)
            idx1 = tl.max(
                tl.where(offs_c == sel1, cand_indices, 0), axis=0
            )
            scores2 = tl.where(offs_c == sel1, -float("inf"), scores1)
            sel2 = tl.argmax(scores2, axis=0, tie_break_left=True)
            val2 = tl.max(scores2, axis=0)
            idx2 = tl.max(
                tl.where(offs_c == sel2, cand_indices, 0), axis=0
            )
            scores3 = tl.where(offs_c == sel2, -float("inf"), scores2)
            sel3 = tl.argmax(scores3, axis=0, tie_break_left=True)
            val3 = tl.max(scores3, axis=0)
            idx3 = tl.max(
                tl.where(offs_c == sel3, cand_indices, 0), axis=0
            )
            best_val0 = val0
            best_val1 = val1
            best_val2 = val2
            best_val3 = val3
            best_idx0 = idx0
            best_idx1 = idx1
            best_idx2 = idx2
            best_idx3 = idx3

        top_idx_base = (
            top_indices
            + batch_idx * top_indices_stride_b
            + head_idx * top_indices_stride_h
        )
        top_score_base = (
            top_scores
            + batch_idx * top_scores_stride_b
            + head_idx * top_scores_stride_h
        )
        tl.store(top_idx_base + 0 * top_indices_stride_t, memory_idx)
        tl.store(top_idx_base + 1 * top_indices_stride_t, best_idx0)
        tl.store(top_idx_base + 2 * top_indices_stride_t, best_idx1)
        tl.store(top_idx_base + 3 * top_indices_stride_t, best_idx2)
        tl.store(top_idx_base + 4 * top_indices_stride_t, best_idx3)
        tl.store(top_score_base + 0 * top_scores_stride_t, memory_val)
        tl.store(top_score_base + 1 * top_scores_stride_t, best_val0)
        tl.store(top_score_base + 2 * top_scores_stride_t, best_val1)
        tl.store(top_score_base + 3 * top_scores_stride_t, best_val2)
        tl.store(top_score_base + 4 * top_scores_stride_t, best_val3)


    @triton.jit
    def _paged_decode_memory_top4_read_kernel(
        read_q,
        states_pages,
        page_table,
        state_indices,
        valid_mask,
        null_logits,
        null_bias,
        top_indices,
        top_scores,
        history_scale,
        out,
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
        ADD_NULL_BIAS: tl.constexpr,
        FUSE_SCALE_ADD: tl.constexpr,
        STATIC_STATE_METADATA: tl.constexpr,
        RAW_READ_Q: tl.constexpr,
        READ_Q_HEAD_REPEAT: tl.constexpr,
        PAGE_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Read five real candidates and normalize them jointly with null."""
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        v_block_idx = tl.program_id(2)
        offs_t = tl.arange(0, BLOCK_T)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = v_block_idx * BLOCK_V + tl.arange(0, BLOCK_V)

        if STATIC_STATE_METADATA:
            state_idx = tl.load(state_indices + batch_idx).to(tl.int64)
            row_valid = tl.load(valid_mask + batch_idx)
            metadata_valid = row_valid & (state_idx >= 0) & (
                state_idx < state_capacity
            )
            metadata_row = tl.where(metadata_valid, state_idx, 0)
        else:
            metadata_row = batch_idx

        top_idx_base = (
            top_indices
            + batch_idx * top_indices_stride_b
            + head_idx * top_indices_stride_h
        )
        top_score_base = (
            top_scores
            + batch_idx * top_scores_stride_b
            + head_idx * top_scores_stride_h
        )
        candidate_mask = offs_t < 5
        candidate_indices = tl.load(
            top_idx_base + offs_t * top_indices_stride_t,
            mask=candidate_mask,
            other=0,
        )
        candidate_scores = tl.load(
            top_score_base + offs_t * top_scores_stride_t,
            mask=candidate_mask,
            other=-float("inf"),
        ).to(tl.float32)
        candidate_valid = candidate_mask & (
            candidate_scores != -float("inf")
        )
        max_val = tl.max(candidate_scores, axis=0)
        if HAS_NULL:
            null_val = tl.load(
                null_logits
                + batch_idx * null_logits_stride_b
                + head_idx * null_logits_stride_h
            ).to(tl.float32)
            if ADD_NULL_BIAS:
                null_val += tl.load(null_bias + head_idx).to(tl.float32)
            max_val = tl.maximum(max_val, null_val)
            exp_null = tl.exp(null_val - max_val)
        else:
            has_any = tl.sum(candidate_valid.to(tl.int32), axis=0) > 0
            max_val = tl.where(has_any, max_val, 0.0)
            exp_null = 0.0
        exp_candidates = tl.where(
            candidate_valid,
            tl.exp(candidate_scores - max_val),
            0.0,
        )
        denom = tl.sum(exp_candidates, axis=0) + exp_null
        weights = exp_candidates * tl.where(
            denom > 0.0, 1.0 / denom, 0.0
        )

        if RAW_READ_Q:
            read_head_idx = head_idx // READ_Q_HEAD_REPEAT
            read = tl.load(
                read_q
                + batch_idx * read_q_stride_b
                + read_head_idx * read_q_stride_h
                + offs_k * read_q_stride_k,
                mask=offs_k < key_dim,
                other=0.0,
            ).to(tl.float32)
            read *= 1.0 / tl.sqrt(tl.sum(read * read, axis=0) + 1e-6)
            read = read.to(read_q.dtype.element_ty).to(tl.float32)
        else:
            read = tl.load(
                read_q
                + batch_idx * read_q_stride_b
                + head_idx * read_q_stride_h
                + offs_k * read_q_stride_k,
                mask=offs_k < key_dim,
                other=0.0,
            ).to(tl.float32)

        page_offsets = candidate_indices // PAGE_M
        slots = candidate_indices - page_offsets * PAGE_M
        page_ids = tl.load(
            page_table
            + metadata_row * page_table_stride_b
            + page_offsets * page_table_stride_p,
            mask=candidate_valid & (page_offsets < max_pages),
            other=0,
        )
        states = tl.load(
            states_pages
            + page_ids[:, None, None] * states_stride_p
            + slots[:, None, None] * states_stride_s
            + head_idx * states_stride_h
            + offs_k[None, :, None] * states_stride_k
            + offs_v[None, None, :] * states_stride_v,
            mask=candidate_valid[:, None, None]
            & (offs_k[None, :, None] < key_dim)
            & (offs_v[None, None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        reads = tl.sum(states * read[None, :, None], axis=1) * read_scale
        result = tl.sum(reads * weights[:, None], axis=0)
        out_ptr = (
            out
            + batch_idx * out_stride_b
            + head_idx * out_stride_h
            + offs_v * out_stride_v
        )
        if FUSE_SCALE_ADD:
            base = tl.load(
                out_ptr, mask=offs_v < value_dim, other=0.0
            ).to(tl.float32)
            rounded = result.to(out.dtype.element_ty).to(tl.float32)
            scale_value = tl.load(history_scale).to(tl.float32)
            result = base + rounded * scale_value
        tl.store(out_ptr, result, mask=offs_v < value_dim)

