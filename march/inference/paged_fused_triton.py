"""MARCH paged history paged fused."""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU reference does not require Triton.
    triton = None
    tl = None

if triton is not None:
    @triton.jit
    def _paged_decode_top4_fused_read_kernel(
        read_q,
        router_q,
        states_pages,
        keys_pages,
        boundary_pages,
        page_table,
        anchor_lengths,
        start_indices,
        state_indices,
        valid_mask,
        null_logits,
        null_bias,
        history_scale,
        out,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        route_scale: tl.constexpr,
        read_scale: tl.constexpr,
        read_q_stride_b: tl.constexpr,
        read_q_stride_h: tl.constexpr,
        read_q_stride_k: tl.constexpr,
        router_q_stride_b: tl.constexpr,
        router_q_stride_h: tl.constexpr,
        router_q_stride_k: tl.constexpr,
        states_stride_p: tl.constexpr,
        states_stride_s: tl.constexpr,
        states_stride_h: tl.constexpr,
        states_stride_v: tl.constexpr,
        states_stride_k: tl.constexpr,
        keys_stride_p: tl.constexpr,
        keys_stride_s: tl.constexpr,
        keys_stride_h: tl.constexpr,
        keys_stride_k: tl.constexpr,
        boundary_stride_p: tl.constexpr,
        boundary_stride_s: tl.constexpr,
        page_table_stride_b: tl.constexpr,
        page_table_stride_p: tl.constexpr,
        null_logits_stride_b: tl.constexpr,
        null_logits_stride_h: tl.constexpr,
        null_bias_stride_h: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_v: tl.constexpr,
        max_pages: tl.constexpr,
        state_capacity: tl.constexpr,
        HAS_NULL: tl.constexpr,
        FUSE_SCALE_ADD: tl.constexpr,
        STATIC_STATE_METADATA: tl.constexpr,
        IGNORE_BOUNDARY_START: tl.constexpr,
        NORMALIZE_ROUTER_Q: tl.constexpr,
        ADD_NULL_BIAS: tl.constexpr,
        RAW_READ_Q: tl.constexpr,
        READ_Q_HEAD_REPEAT: tl.constexpr,
        PAGE_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        v_block_idx = tl.program_id(2)
        offs_k = tl.arange(0, BLOCK_K)
        offs_s = tl.arange(0, PAGE_M)
        offs_c = tl.arange(0, BLOCK_C)
        offs_v = v_block_idx * BLOCK_V + tl.arange(0, BLOCK_V)

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
            metadata_valid = row_valid & (state_idx >= 0) & (state_idx < state_capacity)
            metadata_row = tl.where(metadata_valid, state_idx, 0)
            anchor_len = tl.load(
                anchor_lengths + metadata_row,
                mask=metadata_valid,
                other=0,
            )
            if not IGNORE_BOUNDARY_START:
                start_index = tl.load(
                    start_indices + batch_idx,
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
                page_table + metadata_row * page_table_stride_b + page_pos * page_table_stride_p,
                mask=page_pos < max_pages,
                other=-1,
            )
            page_id_safe = tl.maximum(page_id, 0)
            global_m = page_pos * PAGE_M + offs_s
            keys = tl.load(
                keys_pages
                + page_id_safe * keys_stride_p
                + offs_s[:, None] * keys_stride_s
                + head_idx * keys_stride_h
                + offs_k[None, :] * keys_stride_k,
                mask=(page_id >= 0) & (global_m[:, None] < anchor_len) & (offs_k[None, :] < key_dim),
                other=0.0,
            ).to(tl.float32)
            logits = tl.sum(keys * router[None, :], axis=1) * route_scale
            valid = (page_id >= 0) & (global_m < anchor_len)
            if not IGNORE_BOUNDARY_START:
                boundaries = tl.load(
                    boundary_pages
                    + page_id_safe * boundary_stride_p
                    + offs_s * boundary_stride_s,
                    mask=valid,
                    other=start_index + 1,
                )
                valid = valid & (boundaries <= start_index)
            page_scores = tl.where(valid, logits, -float("inf"))

            page_sel0 = tl.argmax(page_scores, axis=0, tie_break_left=True)
            page_val0 = tl.max(page_scores, axis=0)
            page_scores1 = tl.where(offs_s == page_sel0, -float("inf"), page_scores)
            page_sel1 = tl.argmax(page_scores1, axis=0, tie_break_left=True)
            page_val1 = tl.max(page_scores1, axis=0)
            page_scores2 = tl.where(offs_s == page_sel1, -float("inf"), page_scores1)
            page_sel2 = tl.argmax(page_scores2, axis=0, tie_break_left=True)
            page_val2 = tl.max(page_scores2, axis=0)
            page_scores3 = tl.where(offs_s == page_sel2, -float("inf"), page_scores2)
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
            cand_indices = tl.where(offs_c == 4, (page_pos * PAGE_M + page_sel0).to(tl.int32), cand_indices)
            cand_indices = tl.where(offs_c == 5, (page_pos * PAGE_M + page_sel1).to(tl.int32), cand_indices)
            cand_indices = tl.where(offs_c == 6, (page_pos * PAGE_M + page_sel2).to(tl.int32), cand_indices)
            cand_indices = tl.where(offs_c == 7, (page_pos * PAGE_M + page_sel3).to(tl.int32), cand_indices)

            sel0 = tl.argmax(cand_scores, axis=0, tie_break_left=True)
            val0 = tl.max(cand_scores, axis=0)
            idx0 = tl.max(tl.where(offs_c == sel0, cand_indices, 0), axis=0)
            scores1 = tl.where(offs_c == sel0, -float("inf"), cand_scores)
            sel1 = tl.argmax(scores1, axis=0, tie_break_left=True)
            val1 = tl.max(scores1, axis=0)
            idx1 = tl.max(tl.where(offs_c == sel1, cand_indices, 0), axis=0)
            scores2 = tl.where(offs_c == sel1, -float("inf"), scores1)
            sel2 = tl.argmax(scores2, axis=0, tie_break_left=True)
            val2 = tl.max(scores2, axis=0)
            idx2 = tl.max(tl.where(offs_c == sel2, cand_indices, 0), axis=0)
            scores3 = tl.where(offs_c == sel2, -float("inf"), scores2)
            sel3 = tl.argmax(scores3, axis=0, tie_break_left=True)
            val3 = tl.max(scores3, axis=0)
            idx3 = tl.max(tl.where(offs_c == sel3, cand_indices, 0), axis=0)

            best_val0 = val0
            best_val1 = val1
            best_val2 = val2
            best_val3 = val3
            best_idx0 = idx0
            best_idx1 = idx1
            best_idx2 = idx2
            best_idx3 = idx3

        has0 = best_val0 != -float("inf")
        has1 = best_val1 != -float("inf")
        has2 = best_val2 != -float("inf")
        has3 = best_val3 != -float("inf")
        max_val = tl.maximum(tl.maximum(best_val0, best_val1), tl.maximum(best_val2, best_val3))
        if HAS_NULL:
            null_val = tl.load(
                null_logits
                + batch_idx * null_logits_stride_b
                + head_idx * null_logits_stride_h
            ).to(tl.float32)
            if ADD_NULL_BIAS:
                null_val += tl.load(
                    null_bias + head_idx * null_bias_stride_h
                ).to(tl.float32)
            max_val = tl.maximum(max_val, null_val)
        else:
            null_val = -float("inf")
            has_any = has0 | has1 | has2 | has3
            max_val = tl.where(has_any, max_val, 0.0)
        if HAS_NULL:
            exp_null = tl.exp(null_val - max_val)
        else:
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
            read *= (
                1.0 / tl.sqrt(tl.sum(read * read, axis=0) + 1e-6)
            )
            # Match the standalone post-conv path, which stores normalized Q
            # in BF16 before the history-read kernel reloads it in FP32.
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

        page0 = best_idx0 // PAGE_M
        slot0 = best_idx0 - page0 * PAGE_M
        page1 = best_idx1 // PAGE_M
        slot1 = best_idx1 - page1 * PAGE_M
        page2 = best_idx2 // PAGE_M
        slot2 = best_idx2 - page2 * PAGE_M
        page3 = best_idx3 // PAGE_M
        slot3 = best_idx3 - page3 * PAGE_M
        page_id0 = tl.load(page_table + metadata_row * page_table_stride_b + page0 * page_table_stride_p, mask=has0 & (page0 < max_pages), other=0)
        page_id1 = tl.load(page_table + metadata_row * page_table_stride_b + page1 * page_table_stride_p, mask=has1 & (page1 < max_pages), other=0)
        page_id2 = tl.load(page_table + metadata_row * page_table_stride_b + page2 * page_table_stride_p, mask=has2 & (page2 < max_pages), other=0)
        page_id3 = tl.load(page_table + metadata_row * page_table_stride_b + page3 * page_table_stride_p, mask=has3 & (page3 < max_pages), other=0)

        state0 = tl.load(
            states_pages
            + page_id0 * states_stride_p
            + slot0 * states_stride_s
            + head_idx * states_stride_h
            + offs_v[None, :] * states_stride_v
            + offs_k[:, None] * states_stride_k,
            mask=has0 & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state1 = tl.load(
            states_pages
            + page_id1 * states_stride_p
            + slot1 * states_stride_s
            + head_idx * states_stride_h
            + offs_v[None, :] * states_stride_v
            + offs_k[:, None] * states_stride_k,
            mask=has1 & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state2 = tl.load(
            states_pages
            + page_id2 * states_stride_p
            + slot2 * states_stride_s
            + head_idx * states_stride_h
            + offs_v[None, :] * states_stride_v
            + offs_k[:, None] * states_stride_k,
            mask=has2 & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        state3 = tl.load(
            states_pages
            + page_id3 * states_stride_p
            + slot3 * states_stride_s
            + head_idx * states_stride_h
            + offs_v[None, :] * states_stride_v
            + offs_k[:, None] * states_stride_k,
            mask=has3 & (offs_k[:, None] < key_dim) & (offs_v[None, :] < value_dim),
            other=0.0,
        ).to(tl.float32)
        read0 = tl.sum(read[:, None] * state0, axis=0) * read_scale
        read1 = tl.sum(read[:, None] * state1, axis=0) * read_scale
        read2 = tl.sum(read[:, None] * state2, axis=0) * read_scale
        read3 = tl.sum(read[:, None] * state3, axis=0) * read_scale
        result = read0 * w0 + read1 * w1 + read2 * w2 + read3 * w3
        if FUSE_SCALE_ADD:
            out_ptr = (
                out
                + batch_idx * out_stride_b
                + head_idx * out_stride_h
                + offs_v * out_stride_v
            )
            base = tl.load(
                out_ptr,
                mask=offs_v < value_dim,
                other=0.0,
            ).to(tl.float32)
            # Match eager decode: round the paged read to the output dtype,
            # then apply the FP32 scale and residual addition before the store.
            rounded = result.to(out.dtype.element_ty).to(tl.float32)
            scale_value = tl.load(history_scale).to(tl.float32)
            result = base + rounded * scale_value
        tl.store(
            out + batch_idx * out_stride_b + head_idx * out_stride_h + offs_v * out_stride_v,
            result,
            mask=offs_v < value_dim,
        )


    @triton.jit
    def _paged_decode_read_all_fused_kernel(
        read_q,
        router_q,
        states_pages,
        keys_pages,
        boundary_pages,
        page_table,
        anchor_lengths,
        start_indices,
        state_indices,
        valid_mask,
        null_logits,
        null_bias,
        history_scale,
        out,
        state_key_dim: tl.constexpr,
        route_dim: tl.constexpr,
        value_dim: tl.constexpr,
        route_scale: tl.constexpr,
        read_scale: tl.constexpr,
        read_q_stride_b: tl.constexpr,
        read_q_stride_h: tl.constexpr,
        read_q_stride_k: tl.constexpr,
        router_q_stride_b: tl.constexpr,
        router_q_stride_h: tl.constexpr,
        router_q_stride_k: tl.constexpr,
        states_stride_p: tl.constexpr,
        states_stride_s: tl.constexpr,
        states_stride_h: tl.constexpr,
        states_stride_v: tl.constexpr,
        states_stride_k: tl.constexpr,
        keys_stride_p: tl.constexpr,
        keys_stride_s: tl.constexpr,
        keys_stride_h: tl.constexpr,
        keys_stride_k: tl.constexpr,
        boundary_stride_p: tl.constexpr,
        boundary_stride_s: tl.constexpr,
        page_table_stride_b: tl.constexpr,
        page_table_stride_p: tl.constexpr,
        null_logits_stride_b: tl.constexpr,
        null_logits_stride_h: tl.constexpr,
        null_bias_stride_h: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_v: tl.constexpr,
        max_pages: tl.constexpr,
        state_capacity: tl.constexpr,
        HAS_NULL: tl.constexpr,
        HAS_VALID_MASK: tl.constexpr,
        FUSE_SCALE_ADD: tl.constexpr,
        STATIC_STATE_METADATA: tl.constexpr,
        IGNORE_BOUNDARY_START: tl.constexpr,
        NORMALIZE_ROUTER_Q: tl.constexpr,
        ADD_NULL_BIAS: tl.constexpr,
        RAW_READ_Q: tl.constexpr,
        READ_Q_HEAD_REPEAT: tl.constexpr,
        ROUTE_HEADS: tl.constexpr,
        PAGE_M: tl.constexpr,
        BLOCK_STATE_K: tl.constexpr,
        BLOCK_ROUTE_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        value_block_idx = tl.program_id(2)
        offs_state_k = tl.arange(0, BLOCK_STATE_K)
        offs_route_k = tl.arange(0, BLOCK_ROUTE_K)
        offs_anchor = tl.arange(0, PAGE_M)
        offs_v = value_block_idx * BLOCK_V + tl.arange(0, BLOCK_V)

        row_valid = True
        metadata_row = batch_idx
        if STATIC_STATE_METADATA:
            state_idx = tl.load(state_indices + batch_idx).to(tl.int64)
            row_valid = tl.load(valid_mask + batch_idx)
            metadata_valid = row_valid & (state_idx >= 0) & (
                state_idx < state_capacity
            )
            metadata_row = tl.where(metadata_valid, state_idx, 0)
            row_valid = metadata_valid
        elif HAS_VALID_MASK:
            row_valid = tl.load(valid_mask + batch_idx)
        anchor_len = tl.load(
            anchor_lengths + metadata_row, mask=row_valid, other=0
        )
        if not IGNORE_BOUNDARY_START:
            start_index = tl.load(
                start_indices + batch_idx, mask=row_valid, other=0
            )
        read_head_idx = head_idx // READ_Q_HEAD_REPEAT if RAW_READ_Q else head_idx
        read = tl.load(
            read_q + batch_idx * read_q_stride_b + read_head_idx * read_q_stride_h
            + offs_state_k * read_q_stride_k,
            mask=row_valid & (offs_state_k < state_key_dim), other=0.0,
        ).to(tl.float32)
        if RAW_READ_Q:
            read *= 1.0 / tl.sqrt(tl.sum(read * read, axis=0) + 1e-6)
            read = read.to(read_q.dtype.element_ty).to(tl.float32)
        router = tl.load(
            router_q + batch_idx * router_q_stride_b + head_idx * router_q_stride_h
            + offs_route_k * router_q_stride_k,
            mask=row_valid & (offs_route_k < route_dim), other=0.0,
        ).to(tl.float32)
        if NORMALIZE_ROUTER_Q:
            router *= 1.0 / tl.sqrt(tl.sum(router * router, axis=0) + 1e-6)
            router = router.to(router_q.dtype.element_ty).to(tl.float32)

        running_max = -float("inf")
        running_denom = 0.0
        running_out = tl.zeros((BLOCK_V,), tl.float32)
        if HAS_NULL:
            running_max = tl.load(
                null_logits + batch_idx * null_logits_stride_b
                + head_idx * null_logits_stride_h,
                mask=row_valid, other=-float("inf"),
            ).to(tl.float32)
            if ADD_NULL_BIAS:
                running_max += tl.load(
                    null_bias + head_idx * null_bias_stride_h,
                    mask=row_valid,
                    other=0.0,
                ).to(tl.float32)
            running_denom = tl.where(row_valid, 1.0, 0.0)

        route_head_idx = 0 if ROUTE_HEADS == 1 else head_idx
        for page_pos in tl.range(0, max_pages, loop_unroll_factor=1):
            page_id = tl.load(
                page_table + metadata_row * page_table_stride_b
                + page_pos * page_table_stride_p,
                mask=row_valid, other=-1,
            )
            safe_page_id = tl.maximum(page_id, 0)
            anchor_idx = page_pos * PAGE_M + offs_anchor
            valid = row_valid & (page_id >= 0) & (anchor_idx < anchor_len)
            if not IGNORE_BOUNDARY_START:
                boundaries = tl.load(
                    boundary_pages + safe_page_id * boundary_stride_p
                    + offs_anchor * boundary_stride_s,
                    mask=valid, other=start_index + 1,
                )
                valid = valid & (boundaries <= start_index)
            keys = tl.load(
                keys_pages + safe_page_id * keys_stride_p
                + offs_anchor[:, None] * keys_stride_s
                + route_head_idx * keys_stride_h
                + offs_route_k[None, :] * keys_stride_k,
                mask=valid[:, None] & (offs_route_k[None, :] < route_dim),
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * router[None, :], axis=1) * route_scale
            scores = tl.where(valid, scores, -float("inf"))
            page_has_values = tl.sum(valid.to(tl.int32), axis=0) > 0
            page_max = tl.max(scores, axis=0)
            next_max = tl.where(
                page_has_values, tl.maximum(running_max, page_max), running_max
            )
            old_scale = tl.where(
                running_denom > 0, tl.exp(running_max - next_max), 0.0
            )
            weights = tl.exp(tl.where(valid, scores - next_max, -float("inf")))
            states = tl.load(
                states_pages + safe_page_id * states_stride_p
                + offs_anchor[:, None, None] * states_stride_s
                + head_idx * states_stride_h
                + offs_v[None, None, :] * states_stride_v
                + offs_state_k[None, :, None] * states_stride_k,
                mask=valid[:, None, None]
                & (offs_state_k[None, :, None] < state_key_dim)
                & (offs_v[None, None, :] < value_dim), other=0.0,
            ).to(tl.float32)
            reads = tl.sum(states * read[None, :, None], axis=1)
            running_out = (
                running_out * old_scale
                + tl.sum(reads * weights[:, None], axis=0)
            )
            running_denom = running_denom * old_scale + tl.sum(weights, axis=0)
            running_max = next_max

        history = tl.where(
            running_denom > 0,
            running_out / running_denom * read_scale,
            0.0,
        )
        out_ptr = out + batch_idx * out_stride_b + head_idx * out_stride_h + offs_v * out_stride_v
        if FUSE_SCALE_ADD:
            base = tl.load(out_ptr, mask=row_valid & (offs_v < value_dim), other=0.0).to(tl.float32)
            history = base + history.to(out.dtype.element_ty).to(tl.float32) * tl.load(history_scale)
            tl.store(out_ptr, history, mask=row_valid & (offs_v < value_dim))
        else:
            tl.store(
                out_ptr,
                tl.where(row_valid, history, 0.0),
                mask=offs_v < value_dim,
            )

