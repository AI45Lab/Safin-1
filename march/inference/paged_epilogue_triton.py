"""MARCH paged history paged epilogue."""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU reference does not require Triton.
    triton = None
    tl = None

if triton is not None:
    @triton.jit
    def _history_scale_add_kernel(
        base_out,
        history_out,
        scale,
        value_dim: tl.constexpr,
        base_stride_b: tl.constexpr,
        base_stride_h: tl.constexpr,
        base_stride_v: tl.constexpr,
        history_stride_b: tl.constexpr,
        history_stride_h: tl.constexpr,
        history_stride_v: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        offs_v = tl.arange(0, BLOCK_V)
        mask = offs_v < value_dim
        base_ptr = (
            base_out
            + batch_idx * base_stride_b
            + head_idx * base_stride_h
            + offs_v * base_stride_v
        )
        history_ptr = (
            history_out
            + batch_idx * history_stride_b
            + head_idx * history_stride_h
            + offs_v * history_stride_v
        )
        base = tl.load(base_ptr, mask=mask, other=0.0).to(tl.float32)
        history = tl.load(history_ptr, mask=mask, other=0.0).to(tl.float32)
        scale_value = tl.load(scale).to(tl.float32)
        tl.store(base_ptr, base + history * scale_value, mask=mask)


    @triton.jit
    def _valid_mask_copy_kernel(
        source,
        valid_mask,
        output,
        numel: tl.constexpr,
        hidden_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < numel
        rows = offsets // hidden_dim
        values = tl.load(source + offsets, mask=active, other=0.0)
        row_valid = tl.load(valid_mask + rows, mask=active, other=False)
        tl.store(output + offsets, tl.where(row_valid, values, 0.0), mask=active)


    @triton.jit
    def _strided_gate_rmsnorm_kernel(
        x,
        gate,
        weight,
        output,
        heads: tl.constexpr,
        value_dim: tl.constexpr,
        eps: tl.constexpr,
        x_stride_b: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_v: tl.constexpr,
        gate_stride_b: tl.constexpr,
        gate_stride_h: tl.constexpr,
        gate_stride_v: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_v: tl.constexpr,
        SIGMOID_GATE: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch_idx = row // heads
        head_idx = row - batch_idx * heads
        offsets = tl.arange(0, BLOCK_V)
        active = offsets < value_dim
        values = tl.load(
            x
            + batch_idx * x_stride_b
            + head_idx * x_stride_h
            + offsets * x_stride_v,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        gate_values = tl.load(
            gate
            + batch_idx * gate_stride_b
            + head_idx * gate_stride_h
            + offsets * gate_stride_v,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        weights = tl.load(weight + offsets, mask=active, other=0.0).to(
            tl.float32
        )
        variance = tl.sum(values * values, axis=0) / value_dim
        normalized = values * tl.rsqrt(variance + eps) * weights
        if SIGMOID_GATE:
            normalized *= tl.sigmoid(gate_values)
        else:
            normalized *= gate_values * tl.sigmoid(gate_values)
        tl.store(
            output
            + batch_idx * out_stride_b
            + head_idx * out_stride_h
            + offsets * out_stride_v,
            normalized,
            mask=active,
        )

