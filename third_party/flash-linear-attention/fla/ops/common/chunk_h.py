# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_offsets
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs, check_shared_mem

BKV_LIST = [32, 64] if check_shared_mem() else [16, 32]


def _validate_varlen_boundary_layout(
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
    boundary_offsets: torch.Tensor,
    boundary_offsets_cpu: torch.Tensor | None,
    boundary_stride: int,
    total_length: int,
) -> int:
    """Return the exact flattened boundary count for a varlen scan.

    The CPU sidecars are the production path and avoid a device
    synchronization.  Direct low-level callers may omit them; validation then
    copies the small layout tensors once rather than trusting a self-described
    boundary-bank leading dimension.
    """
    N = len(cu_seqlens) - 1
    if boundary_offsets.ndim != 1 or boundary_offsets.shape[0] != N + 1:
        raise ValueError(
            f"boundary_offsets must have shape [{N + 1}], got "
            f"{tuple(boundary_offsets.shape)}"
        )
    if boundary_offsets.device != cu_seqlens.device:
        raise ValueError("boundary_offsets and cu_seqlens must share a device")
    if boundary_offsets.dtype not in {torch.int32, torch.int64}:
        raise ValueError("boundary_offsets must have int32 or int64 dtype")

    if cu_seqlens_cpu is None:
        cu_cpu = cu_seqlens.detach().to(device="cpu", dtype=torch.int64)
    else:
        if cu_seqlens_cpu.device.type != "cpu":
            raise ValueError("cu_seqlens_cpu must be a CPU tensor")
        if tuple(cu_seqlens_cpu.shape) != tuple(cu_seqlens.shape):
            raise ValueError("cu_seqlens_cpu must have the same shape as cu_seqlens")
        if cu_seqlens_cpu.dtype not in {torch.int32, torch.int64}:
            raise ValueError("cu_seqlens_cpu must have int32 or int64 dtype")
        cu_cpu = cu_seqlens_cpu.detach().to(torch.int64)
    lengths = cu_cpu[1:] - cu_cpu[:-1]
    if int(cu_cpu[0]) != 0 or int(cu_cpu[-1]) != total_length:
        raise ValueError("cu_seqlens must span the flattened token dimension")
    if bool(torch.any(lengths <= 0)):
        raise ValueError("cu_seqlens must describe strictly positive lengths")
    counts = torch.div(lengths, boundary_stride, rounding_mode="floor")
    expected_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))

    if boundary_offsets_cpu is None:
        offsets_cpu = boundary_offsets.detach().to(device="cpu", dtype=torch.int64)
    else:
        if boundary_offsets_cpu.device.type != "cpu":
            raise ValueError("boundary_offsets_cpu must be a CPU tensor")
        if tuple(boundary_offsets_cpu.shape) != tuple(boundary_offsets.shape):
            raise ValueError(
                "boundary_offsets_cpu must have the same shape as boundary_offsets"
            )
        if boundary_offsets_cpu.dtype not in {torch.int32, torch.int64}:
            raise ValueError("boundary_offsets_cpu must have int32 or int64 dtype")
        offsets_cpu = boundary_offsets_cpu.detach().to(torch.int64)
    if not torch.equal(offsets_cpu, expected_offsets):
        raise ValueError(
            "boundary offsets must start at zero, be monotonic, and equal the "
            "per-document floor(length / boundary_stride) prefix sums"
        )

    # A trusted CPU sidecar must not allow a stale or corrupted device copy to
    # steer the packed kernel.  Check adjacent counts asynchronously on CUDA,
    # matching the split-offset contract below without a host synchronization.
    device_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    expected_counts_device = torch.div(
        device_lengths,
        boundary_stride,
        rounding_mode="floor",
    )
    equal = (
        (boundary_offsets[0] == 0)
        & (boundary_offsets[-1] == int(expected_offsets[-1]))
        & torch.all(
            boundary_offsets[1:] - boundary_offsets[:-1]
            == expected_counts_device
        )
    )
    if boundary_offsets.device.type == "cpu":
        if not bool(equal):
            raise ValueError(
                "boundary_offsets device values do not match its CPU copy"
            )
    elif hasattr(torch, "_assert_async"):
        torch._assert_async(equal)
    elif not bool(equal):  # pragma: no cover - old supported torch builds
        raise ValueError("boundary_offsets device values do not match its CPU copy")
    return int(expected_offsets[-1])


def _validate_varlen_split_layout(
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
    split_offsets: torch.Tensor,
    split_offsets_cpu: torch.Tensor | None,
    split_size: int,
    total_length: int,
) -> int:
    """Validate reusable varlen split offsets and return total split count."""
    N = len(cu_seqlens) - 1
    if split_offsets.ndim != 1 or split_offsets.shape[0] != N + 1:
        raise ValueError(
            f"split_offsets must have shape [{N + 1}], got "
            f"{tuple(split_offsets.shape)}"
        )
    if split_offsets.device != cu_seqlens.device:
        raise ValueError("split_offsets and cu_seqlens must share a device")
    if split_offsets.dtype not in {torch.int32, torch.int64}:
        raise ValueError("split_offsets must have int32 or int64 dtype")

    if cu_seqlens_cpu is None:
        cu_cpu = cu_seqlens.detach().to(device="cpu", dtype=torch.int64)
    else:
        if cu_seqlens_cpu.device.type != "cpu":
            raise ValueError("cu_seqlens_cpu must be a CPU tensor")
        if tuple(cu_seqlens_cpu.shape) != tuple(cu_seqlens.shape):
            raise ValueError("cu_seqlens_cpu must have the same shape as cu_seqlens")
        if cu_seqlens_cpu.dtype not in {torch.int32, torch.int64}:
            raise ValueError("cu_seqlens_cpu must have int32 or int64 dtype")
        cu_cpu = cu_seqlens_cpu.detach().to(torch.int64)
    lengths = cu_cpu[1:] - cu_cpu[:-1]
    if int(cu_cpu[0]) != 0 or int(cu_cpu[-1]) != total_length:
        raise ValueError("cu_seqlens must span the flattened token dimension")
    if bool(torch.any(lengths <= 0)):
        raise ValueError("cu_seqlens must describe strictly positive lengths")
    counts = torch.div(
        lengths + split_size - 1,
        split_size,
        rounding_mode="floor",
    )
    expected_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))

    if split_offsets_cpu is None:
        offsets_cpu = split_offsets.detach().to(device="cpu", dtype=torch.int64)
    else:
        if split_offsets_cpu.device.type != "cpu":
            raise ValueError("split_offsets_cpu must be a CPU tensor")
        if tuple(split_offsets_cpu.shape) != tuple(split_offsets.shape):
            raise ValueError(
                "split_offsets_cpu must have the same shape as split_offsets"
            )
        if split_offsets_cpu.dtype not in {torch.int32, torch.int64}:
            raise ValueError("split_offsets_cpu must have int32 or int64 dtype")
        offsets_cpu = split_offsets_cpu.detach().to(torch.int64)
    if not torch.equal(offsets_cpu, expected_offsets):
        raise ValueError(
            "split offsets must equal the per-document ceil(length / split_size) "
            "prefix sums"
        )

    # Validate the device copy from device-resident ``cu_seqlens``.  This
    # deliberately checks adjacent counts rather than rebuilding/copying the
    # full CPU prefix-sum tensor on every key/value/recompute path.
    device_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    expected_counts_device = torch.div(
        device_lengths + split_size - 1,
        split_size,
        rounding_mode="floor",
    )
    equal = (
        (split_offsets[0] == 0)
        & (split_offsets[-1] == int(expected_offsets[-1]))
        & torch.all(split_offsets[1:] - split_offsets[:-1] == expected_counts_device)
    )
    if split_offsets.device.type == "cpu":
        if not bool(equal):
            raise ValueError("split_offsets device values do not match its CPU copy")
    elif hasattr(torch, "_assert_async"):
        torch._assert_async(equal)
    elif not bool(equal):  # pragma: no cover - old supported torch builds
        raise ValueError("split_offsets device values do not match its CPU copy")
    return int(expected_offsets[-1])


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'STORE_BOUNDARY_STATES': lambda args: args['boundary_states'] is not None,
    'STORE_CONTINUOUS_BOUNDARY_STATES': lambda args: args['continuous_boundary_states'] is not None,
    'STORE_LOCAL_SPLIT_STATES': lambda args: args['local_split_states'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=[
        'BT',
        'USE_G',
        'USE_GK',
        'USE_GV',
        'RESET_AT_BOUNDARY',
        'SHARED_ROUNDED_UPDATE_EXPERIMENTAL',
        'STORE_BOUNDARY_STATES',
        'STORE_CONTINUOUS_BOUNDARY_STATES',
        'STORE_LOCAL_SPLIT_STATES',
    ],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_h(
    k,
    v,
    h,
    g,
    g_gamma,
    gk,
    gv,
    h0,
    ht,
    boundary_states,
    continuous_boundary_states,
    local_split_states,
    boundary_offsets,
    cu_seqlens,
    split_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_BOUNDARY_STATES: tl.constexpr,
    STORE_CONTINUOUS_BOUNDARY_STATES: tl.constexpr,
    STORE_LOCAL_SPLIT_STATES: tl.constexpr,
    RESET_AT_BOUNDARY: tl.constexpr,
    SHARED_ROUNDED_UPDATE_EXPERIMENTAL: tl.constexpr,
    BOUNDARY_CHUNK_STRIDE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = i_n * NS
    NTS = BS // BT

    if USE_G_GAMMA:
        # decay rate given the head index
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if RESET_AT_BOUNDARY:
        # The chunk-local recurrence is independent of h0.  It shares every
        # token update with the continuous recurrence, but is reset after each
        # exported boundary state.
        b_h_local = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        i_s = i_t // NTS
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
        p_h = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        if i_t % NTS == 0:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
            if RESET_AT_BOUNDARY and STORE_LOCAL_SPLIT_STATES:
                p_h_local = tl.make_block_ptr(
                    local_split_states + o_h,
                    (K, V),
                    (V, 1),
                    (i_k * BK, i_v * BV),
                    (BK, BV),
                    (1, 0),
                )
                tl.store(
                    p_h_local,
                    b_h_local.to(p_h_local.dtype.element_ty),
                    boundary_check=(0, 1),
                )
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1

        # scalar decay
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_h *= exp(b_g_last)
            if RESET_AT_BOUNDARY:
                b_h_local *= exp(b_g_last)
            b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_h *= exp(b_g_last)
            if RESET_AT_BOUNDARY:
                b_h_local *= exp(b_g_last)
            b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        # vector decay, h = Diag(gk) @ h
        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_h *= exp(b_gk_last)[:, None]
            if RESET_AT_BOUNDARY:
                b_h_local *= exp(b_gk_last)[:, None]

            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_k = (b_k * exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

        # vector decay, h = h @ Diag(gv)
        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_h *= exp(b_gv_last)[None, :]
            if RESET_AT_BOUNDARY:
                b_h_local *= exp(b_gv_last)[None, :]

            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_v = (b_v * exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

        if RESET_AT_BOUNDARY:
            if SHARED_ROUNDED_UPDATE_EXPERIMENTAL:
                # Experimental throughput path: round the shared update once,
                # then add it to both recurrent accumulators.  This is a
                # numerically bounded approximation, not the strict backend:
                # the legacy recurrence fuses the old state as the MMA
                # accumulator and therefore rounds in a different order.
                b_update = tl.dot(b_k, b_v, input_precision="ieee")
                b_h += b_update
                b_h_local += b_update
            else:
                # Strict dual recurrence.  K/V loads, decay, boundary tests,
                # and layout are shared, but each state must be the direct MMA
                # accumulator to reproduce the two legacy scans exactly.
                b_h += tl.dot(b_k, b_v)
                b_h_local += tl.dot(b_k, b_v)
        else:
            b_h += tl.dot(b_k, b_v)

        # Raven-HiLS compact producer: expose the native fp32 recurrent
        # accumulator after every fixed group of internal chunks.  The normal
        # model-dtype ``h`` remains unchanged and is still written at every
        # split start for the token-output kernels.  Keeping this side output
        # in the scan avoids restarting GSA at each 512-token landmark and,
        # unlike ``h.float()``, does not quantize the snapshot through BF16.
        if STORE_BOUNDARY_STATES or STORE_CONTINUOUS_BOUNDARY_STATES or RESET_AT_BOUNDARY:
            if (i_t + 1) % BOUNDARY_CHUNK_STRIDE == 0:
                boundary_end = (i_t + 1) * BT
                if boundary_end <= T:
                    if STORE_BOUNDARY_STATES:
                        i_boundary = (i_t + 1) // BOUNDARY_CHUNK_STRIDE - 1
                        if IS_VARLEN:
                            boundary_base = tl.load(boundary_offsets + i_n).to(tl.int64)
                        else:
                            n_boundaries = T // (BT * BOUNDARY_CHUNK_STRIDE)
                            boundary_base = i_n * n_boundaries
                        o_boundary = (
                            ((boundary_base + i_boundary) * H + i_h).to(tl.int64)
                            * K * V
                        )
                        p_boundary = tl.make_block_ptr(
                            boundary_states + o_boundary,
                            (K, V),
                            (V, 1),
                            (i_k * BK, i_v * BV),
                            (BK, BV),
                            (1, 0),
                        )
                        if RESET_AT_BOUNDARY:
                            tl.store(
                                p_boundary,
                                b_h_local,
                                boundary_check=(0, 1),
                            )
                        else:
                            tl.store(
                                p_boundary,
                                b_h,
                                boundary_check=(0, 1),
                            )
                    if STORE_CONTINUOUS_BOUNDARY_STATES:
                        i_boundary = (i_t + 1) // BOUNDARY_CHUNK_STRIDE - 1
                        if IS_VARLEN:
                            boundary_base = tl.load(boundary_offsets + i_n).to(tl.int64)
                        else:
                            n_boundaries = T // (BT * BOUNDARY_CHUNK_STRIDE)
                            boundary_base = i_n * n_boundaries
                        o_boundary = (
                            ((boundary_base + i_boundary) * H + i_h).to(tl.int64)
                            * K * V
                        )
                        p_continuous_boundary = tl.make_block_ptr(
                            continuous_boundary_states + o_boundary,
                            (K, V),
                            (V, 1),
                            (i_k * BK, i_v * BV),
                            (BK, BV),
                            (1, 0),
                        )
                        tl.store(
                            p_continuous_boundary,
                            b_h,
                            boundary_check=(0, 1),
                        )
                    if RESET_AT_BOUNDARY:
                        b_h_local = tl.zeros([BK, BV], dtype=tl.float32)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_BOUNDARY_STATE_GRADIENTS': lambda args: args['boundary_state_grads'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV', 'RESET_AT_BOUNDARY'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dh(
    q,
    g,
    g_gamma,
    gk,
    gv,
    do,
    dh,
    dht,
    dh0,
    boundary_state_grads,
    local_adjoint_states,
    boundary_offsets,
    cu_seqlens,
    split_offsets,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_BOUNDARY_STATE_GRADIENTS: tl.constexpr,
    RESET_AT_BOUNDARY: tl.constexpr,
    BOUNDARY_CHUNK_STRIDE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = i_n * NS

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if RESET_AT_BOUNDARY:
        b_dh_local = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        # A boundary snapshot is the post-state of this internal chunk.  Its
        # cotangent must be added before storing/propagating ``b_dh`` so that
        # it affects this chunk and every earlier token, but never a later
        # token.  If the final state is also a boundary, this naturally adds
        # the snapshot and final-state cotangents.
        if USE_BOUNDARY_STATE_GRADIENTS:
            if (i_t + 1) % BOUNDARY_CHUNK_STRIDE == 0:
                boundary_end = (i_t + 1) * BT
                if boundary_end <= T:
                    i_boundary = (i_t + 1) // BOUNDARY_CHUNK_STRIDE - 1
                    if IS_VARLEN:
                        boundary_base = tl.load(boundary_offsets + i_n).to(tl.int64)
                    else:
                        n_boundaries = T // (BT * BOUNDARY_CHUNK_STRIDE)
                        boundary_base = i_n * n_boundaries
                    o_boundary = (
                        ((boundary_base + i_boundary) * HQ + i_hq).to(tl.int64)
                        * K * V
                    )
                    p_boundary_grad = tl.make_block_ptr(
                        boundary_state_grads + o_boundary,
                        (K, V),
                        (V, 1),
                        (i_k * BK, i_v * BV),
                        (BK, BV),
                        (1, 0),
                    )
                    b_boundary_grad = tl.load(
                        p_boundary_grad,
                        boundary_check=(0, 1),
                    ).to(tl.float32)
                    if RESET_AT_BOUNDARY:
                        # Replace the later chunk's adjoint at a reset.  The
                        # newly loaded cotangent belongs only to this chunk.
                        b_dh_local = b_boundary_grad
                    else:
                        b_dh += b_boundary_grad
        i_s = i_t // (BS // BT)
        o_dh = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
        p_dh = tl.make_block_ptr(dh + o_dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        if i_t % (BS // BT) == 0:
            if RESET_AT_BOUNDARY:
                p_dh_local = tl.make_block_ptr(
                    local_adjoint_states + o_dh,
                    (K, V),
                    (V, 1),
                    (i_k * BK, i_v * BV),
                    (BK, BV),
                    (1, 0),
                )
                tl.store(
                    p_dh,
                    b_dh.to(p_dh.dtype.element_ty),
                    boundary_check=(0, 1),
                )
                tl.store(
                    p_dh_local,
                    b_dh_local.to(p_dh_local.dtype.element_ty),
                    boundary_check=(0, 1),
                )
            else:
                tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        # [BK, BT]
        p_q = tl.make_block_ptr(q + (bos*HQ + i_hq) * K, (K, T), (1, HQ*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + (bos*HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        if USE_G:
            p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_q = (b_q * exp(b_g)[None, :]).to(b_q.dtype)
            b_dh *= exp(b_g_last)
            if RESET_AT_BOUNDARY:
                b_dh_local *= exp(b_g_last)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_q = (b_q * exp(b_g)[None, :]).to(b_q.dtype)
            b_dh *= exp(b_g_last)
            if RESET_AT_BOUNDARY:
                b_dh_local *= exp(b_g_last)

        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_q = (b_q * exp(b_gk)).to(b_q.dtype)
            b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
            b_dh *= exp(b_gk_last)[:, None]
            if RESET_AT_BOUNDARY:
                b_dh_local *= exp(b_gk_last)[:, None]

        if USE_GV:
            p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_do = (b_do * exp(b_gv))

            b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
            b_dh *= exp(b_gv_last)[None, :]
            if RESET_AT_BOUNDARY:
                b_dh_local *= exp(b_gv_last)[None, :]

        b_dh += tl.dot(b_q, b_do.to(b_q.dtype))

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
    boundary_states: torch.Tensor | None = None,
    continuous_boundary_states: torch.Tensor | None = None,
    boundary_stride: int = 1,
    boundary_offsets: torch.Tensor | None = None,
    boundary_offsets_cpu: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    split_offsets: torch.Tensor | None = None,
    split_offsets_cpu: torch.Tensor | None = None,
    local_split_states: torch.Tensor | None = None,
    reset_at_boundary: bool = False,
    shared_rounded_update_experimental: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    if (
        boundary_states is not None
        or continuous_boundary_states is not None
        or reset_at_boundary
    ):
        if boundary_stride <= 0 or boundary_stride % BT != 0:
            raise ValueError("boundary_stride must be a positive multiple of chunk_size")
        if cu_seqlens is None:
            if boundary_offsets is not None:
                raise ValueError("boundary_offsets is only valid with cu_seqlens")
            expected = (B, T // boundary_stride, H, K, V)
        else:
            if boundary_offsets is None:
                raise ValueError(
                    "boundary_offsets is required for variable-length boundary states"
                )
            boundary_count = _validate_varlen_boundary_layout(
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                boundary_offsets=boundary_offsets,
                boundary_offsets_cpu=boundary_offsets_cpu,
                boundary_stride=boundary_stride,
                total_length=T,
            )
            expected = (boundary_count, H, K, V)
        if boundary_states is not None:
            if tuple(boundary_states.shape) != expected:
                raise ValueError(
                    f"boundary_states must have shape {expected}, got {tuple(boundary_states.shape)}"
                )
            if boundary_states.dtype != torch.float32:
                raise ValueError("boundary_states must be fp32")
        if continuous_boundary_states is not None:
            if tuple(continuous_boundary_states.shape) != expected:
                raise ValueError(
                    "continuous_boundary_states must have shape "
                    f"{expected}, got {tuple(continuous_boundary_states.shape)}"
                )
            if continuous_boundary_states.dtype != torch.float32:
                raise ValueError("continuous_boundary_states must be fp32")
    if local_split_states is not None and not reset_at_boundary:
        raise ValueError("local_split_states requires reset_at_boundary")
    if shared_rounded_update_experimental and not reset_at_boundary:
        raise ValueError(
            "shared_rounded_update_experimental requires reset_at_boundary"
        )
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        if split_offsets is not None or split_offsets_cpu is not None:
            raise ValueError("split offsets require cu_seqlens")
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        N = len(cu_seqlens) - 1
        if split_offsets is not None:
            NS = _validate_varlen_split_layout(
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                split_offsets=split_offsets,
                split_offsets_cpu=split_offsets_cpu,
                split_size=BS,
                total_length=T,
            )
        else:
            if split_offsets_cpu is not None:
                raise ValueError("split_offsets_cpu requires split_offsets")
            if cu_seqlens_cpu is None:
                split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
                NS = split_offsets[-1].item()
            else:
                generated_offsets_cpu = prepare_chunk_offsets(cu_seqlens_cpu, BS)
                NS = int(generated_offsets_cpu[-1])
                split_offsets = generated_offsets_cpu.to(cu_seqlens)

    h = k.new_empty(B, NS, H, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    if local_split_states is not None:
        expected_local = (B, NS, H, K, V)
        if tuple(local_split_states.shape) != expected_local:
            raise ValueError(
                f"local_split_states must have shape {expected_local}, got "
                f"{tuple(local_split_states.shape)}"
            )
        if local_split_states.dtype != h.dtype:
            raise ValueError("local_split_states must have the recurrent-state dtype")
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_fwd_kernel_h[grid](
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        h0=h0,
        ht=ht,
        boundary_states=boundary_states,
        continuous_boundary_states=continuous_boundary_states,
        local_split_states=local_split_states,
        boundary_offsets=boundary_offsets,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BOUNDARY_CHUNK_STRIDE=max(1, boundary_stride // BT),
        RESET_AT_BOUNDARY=bool(reset_at_boundary),
        SHARED_ROUNDED_UPDATE_EXPERIMENTAL=bool(
            shared_rounded_update_experimental
        ),
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    return h, ht


def chunk_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
    boundary_state_grads: torch.Tensor | None = None,
    boundary_stride: int = 1,
    boundary_offsets: torch.Tensor | None = None,
    boundary_offsets_cpu: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    split_offsets: torch.Tensor | None = None,
    split_offsets_cpu: torch.Tensor | None = None,
    local_adjoint_states: torch.Tensor | None = None,
    reset_at_boundary: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    if boundary_state_grads is not None:
        if boundary_stride <= 0 or boundary_stride % BT != 0:
            raise ValueError("boundary_stride must be a positive multiple of chunk_size")
        if cu_seqlens is None:
            if boundary_offsets is not None:
                raise ValueError("boundary_offsets is only valid with cu_seqlens")
            expected = (B, T // boundary_stride, HQ, K, V)
        else:
            if boundary_offsets is None:
                raise ValueError(
                    "boundary_offsets is required for variable-length boundary gradients"
                )
            boundary_count = _validate_varlen_boundary_layout(
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                boundary_offsets=boundary_offsets,
                boundary_offsets_cpu=boundary_offsets_cpu,
                boundary_stride=boundary_stride,
                total_length=T,
            )
            expected = (boundary_count, HQ, K, V)
        if tuple(boundary_state_grads.shape) != expected:
            raise ValueError(
                "boundary_state_grads must have shape "
                f"{expected}, got {tuple(boundary_state_grads.shape)}"
            )
        if boundary_state_grads.dtype != torch.float32:
            raise ValueError("boundary_state_grads must be fp32")
    if reset_at_boundary:
        if boundary_state_grads is None:
            raise ValueError("reset_at_boundary requires boundary_state_grads")
        if local_adjoint_states is None:
            raise ValueError("reset_at_boundary requires local_adjoint_states")
    elif local_adjoint_states is not None:
        raise ValueError("local_adjoint_states requires reset_at_boundary")
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NG: number of groups in GQA
    if cu_seqlens is None:
        if split_offsets is not None or split_offsets_cpu is not None:
            raise ValueError("split offsets require cu_seqlens")
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        N = len(cu_seqlens) - 1
        if split_offsets is not None:
            NS = _validate_varlen_split_layout(
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                split_offsets=split_offsets,
                split_offsets_cpu=split_offsets_cpu,
                split_size=BS,
                total_length=T,
            )
        else:
            if split_offsets_cpu is not None:
                raise ValueError("split_offsets_cpu requires split_offsets")
            if cu_seqlens_cpu is None:
                split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
                NS = split_offsets[-1].item()
            else:
                generated_offsets_cpu = prepare_chunk_offsets(cu_seqlens_cpu, BS)
                NS = int(generated_offsets_cpu[-1])
                split_offsets = generated_offsets_cpu.to(cu_seqlens)
    NG = HQ // H

    dh = k.new_empty(B, NS, HQ, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    if local_adjoint_states is not None:
        expected_local = (B, NS, HQ, K, V)
        if tuple(local_adjoint_states.shape) != expected_local:
            raise ValueError(
                f"local_adjoint_states must have shape {expected_local}, got "
                f"{tuple(local_adjoint_states.shape)}"
            )
        if local_adjoint_states.dtype != dh.dtype:
            raise ValueError(
                "local_adjoint_states must have the adjoint-state dtype"
            )
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
    chunk_bwd_kernel_dh[grid](
        q=q,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        do=do,
        dh=dh,
        dht=dht,
        dh0=dh0,
        boundary_state_grads=boundary_state_grads,
        local_adjoint_states=local_adjoint_states,
        boundary_offsets=boundary_offsets,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        scale=scale,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BOUNDARY_CHUNK_STRIDE=max(1, boundary_stride // BT),
        RESET_AT_BOUNDARY=bool(reset_at_boundary),
        NG=NG,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    return dh, dh0
