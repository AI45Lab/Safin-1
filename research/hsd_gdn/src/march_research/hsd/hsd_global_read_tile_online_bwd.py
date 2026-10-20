"""Low-memory TileLang backward for HSD's online grouped global read.

This module is intentionally independent from :mod:`hsd_global_read_tile`.
It is the correctness/performance candidate that replaces the current
backward oracle which materializes both ``prob[T, H, M]`` and
``dz[T, H, M]``.

The implementation has four explicit Hopper geometry profiles:

* LC50B: six heads with state K/V 192/384 and route dimension 64;
* LC50B Transformer-aligned: sixteen heads with state K/V 96/96 and route
  dimension 64;
* OLMo3-1B Dense: sixteen heads with state K/V 128/128 and route dimension 128;
* Qwen3.5-4B TP2 Dense: sixteen heads with state K/V 128/128 and route
  dimension 64;
* BF16 queries, keys and output gradients;
* FP32 summary states.

Routing probabilities are recomputed from ``router_q``, ``summary_k`` and
the forward ``lse`` inside every gradient kernel.  Forward additionally saves
the pre-cast FP32 history accumulator ``[T,H,V]`` so the softmax-Jacobian
common term is one cheap dot product rather than another state/summary scan.
No tensor proportional to ``T * H * M`` is allocated.  The default backward
workspace is ``delta[T, H]``.  An opt-in fused dRead/route path additionally
uses ``K / BK`` FP32 ``[T,H,R]`` route partials, one per key tile, plus the
compact FP32 ``[M,H,R]`` dSummaryK atomic accumulator.

The four main gradients deliberately use complementary ownership:

* dReadQ: query-block owned, scanning visible summaries;
* dState: summary owned, scanning packed tokens;
* dRouterQ: query-block owned, scanning visible summaries;
* dSummaryK: summary owned, scanning packed tokens.

This remains a deliberately allowlisted Hopper implementation.  New shapes
must be added as named profiles and validated rather than admitted by a broad
dimension guard.
"""

import math
import os
from collections.abc import Sequence

import torch

try:
    import tilelang
    import tilelang.language as T
except Exception:  # pragma: no cover - optional CUDA/TileLang dependency.
    tilelang = None
    T = None

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional CUDA/Triton dependency.
    triton = None
    tl = None


_PROFILE_GEOMETRIES = {
    "lc50b_h6_k192_v384_r64": (6, 192, 384, 64),
    "lc50b_h16_k96_v96_r64": (16, 96, 96, 64),
    "olmo3_h16_k128_v128_r128": (16, 128, 128, 128),
    "qwen35_4b_tp2_h16_k128_v128_r64": (16, 128, 128, 64),
}
_PROFILE_ALIASES = {
    "lc50b": "lc50b_h6_k192_v384_r64",
    "h6_k192_v384_r64": "lc50b_h6_k192_v384_r64",
    "h16_k96_v96_r64": "lc50b_h16_k96_v96_r64",
    "olmo3_1b": "olmo3_h16_k128_v128_r128",
    "olmo3_1b_dense": "olmo3_h16_k128_v128_r128",
    "olmo3_1b_h16_k128_v128_r128": "olmo3_h16_k128_v128_r128",
    "h16_k128_v128_r128": "olmo3_h16_k128_v128_r128",
    "qwen35_4b": "qwen35_4b_tp2_h16_k128_v128_r64",
    "qwen35_4b_dense": "qwen35_4b_tp2_h16_k128_v128_r64",
    "qwen35_4b_tp2": "qwen35_4b_tp2_h16_k128_v128_r64",
    "h16_k128_v128_r64": "qwen35_4b_tp2_h16_k128_v128_r64",
}
_LOG2E = 1.4426950408889634


def _canonical_profile(profile: str | None) -> str | None:
    normalized = "" if profile is None else str(profile).strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return None
    normalized = _PROFILE_ALIASES.get(normalized, normalized)
    if normalized not in _PROFILE_GEOMETRIES:
        supported = ", ".join(sorted(_PROFILE_GEOMETRIES))
        raise ValueError(
            f"Unknown HSD tile-online backward profile {profile!r}; "
            f"supported profiles: {supported}."
        )
    return normalized


def _resolve_profile(
    profile: str | None,
    geometry: tuple[int, int, int, int],
) -> str | None:
    normalized = _canonical_profile(profile)
    if normalized is not None:
        return normalized if _PROFILE_GEOMETRIES[normalized] == geometry else None
    matches = [name for name, expected in _PROFILE_GEOMETRIES.items() if expected == geometry]
    return matches[0] if len(matches) == 1 else None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


if triton is not None:

    @triton.jit
    def _online_bwd_delta_kernel(
        dout,
        out,
        p_null,
        dlocal_prob,
        delta,
        dnull,
        T_LEN: tl.constexpr,
        H: tl.constexpr,
        V: tl.constexpr,
        HAS_NULL: tl.constexpr,
        HAS_DLOCAL: tl.constexpr,
        NEED_DNULL: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
    ):
        i_t = tl.program_id(0)
        i_h = tl.program_id(1)
        offs_t = i_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_v = tl.arange(0, BLOCK_V)
        t_mask = offs_t < T_LEN
        common = tl.zeros([BLOCK_T], dtype=tl.float32)

        # The route-softmax common term is dO dot O.  If the forward exposed
        # the null probability as a second differentiable output, its
        # contribution is dP_null * P_null.
        for i_v in range(NUM_V_BLOCKS):
            offs_v_cur = i_v * BLOCK_V + offs_v
            mask = t_mask[:, None] & (offs_v_cur[None, :] < V)
            do = tl.load(
                dout + ((offs_t[:, None] * H + i_h) * V + offs_v_cur[None, :]),
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            o = tl.load(
                out + ((offs_t[:, None] * H + i_h) * V + offs_v_cur[None, :]),
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            common += tl.sum(do * o, axis=1)

        pn = (
            tl.load(
                p_null + offs_t * H + i_h,
                mask=t_mask,
                other=0.0,
            ).to(tl.float32)
            if HAS_NULL
            else 0.0
        )
        gp = (
            tl.load(
                dlocal_prob + offs_t * H + i_h,
                mask=t_mask,
                other=0.0,
            ).to(tl.float32)
            if HAS_DLOCAL
            else 0.0
        )
        common += gp * pn
        tl.store(delta + offs_t * H + i_h, common, mask=t_mask)

        if HAS_NULL and NEED_DNULL:
            # d z_null = p_null * (d p_null - common_softmax_term).
            tl.store(
                dnull + offs_t * H + i_h,
                pn * (gp - common),
                mask=t_mask,
            )

    @triton.jit
    def _online_bwd_dnull_kernel(
        p_null,
        dlocal_prob,
        delta,
        dnull,
        T_LEN: tl.constexpr,
        H: tl.constexpr,
        HAS_DLOCAL: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Finish the null-logit Jacobian from an exact precomputed delta."""

        i_t = tl.program_id(0)
        i_h = tl.program_id(1)
        offs_t = i_t * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = offs_t < T_LEN
        pn = tl.load(
            p_null + offs_t * H + i_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gp = (
            tl.load(
                dlocal_prob + offs_t * H + i_h,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            if HAS_DLOCAL
            else 0.0
        )
        common = tl.load(
            delta + offs_t * H + i_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            dnull + offs_t * H + i_h,
            pn * (gp - common),
            mask=mask,
        )

    @triton.jit
    def _online_bwd_cast_compact_kernel(
        src,
        dst,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Cast the compact FP32 atomic accumulator to the BF16 input dtype."""

        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        values = tl.load(src + offsets, mask=mask, other=0.0)
        tl.store(dst + offsets, values, mask=mask)

    @triton.jit
    def _online_bwd_reduce_drouter_partial_kernel(
        partial,
        drouter,
        N: tl.constexpr,
        NUMEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Reduce N FP32 K-tile route partials and cast to BF16."""

        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < NUMEL
        values = tl.zeros([BLOCK], dtype=tl.float32)
        for i_n in range(N):
            values += tl.load(
                partial + i_n * NUMEL + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(drouter + offsets, values, mask=mask)

else:
    _online_bwd_delta_kernel = None
    _online_bwd_dnull_kernel = None
    _online_bwd_cast_compact_kernel = None
    _online_bwd_reduce_drouter_partial_kernel = None


if tilelang is not None:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        }
    )
    def _build_online_exact_delta_kernel(
        H,
        K,
        V,
        R,
        M,
        BT,
        BV,
        has_dlocal,
        num_warps=4,
    ):
        """Debug oracle for recomputing the FP32 softmax common term.

        The production dispatch no longer calls this kernel: forward saves its
        pre-cast FP32 history accumulator and ``_online_bwd_delta_kernel``
        evaluates the same common term with one O(T*H*V) dot product.  Keep
        this private implementation temporarily as a no-dense-route
        correctness oracle while the new backward path is being tuned.

        The forward output is BF16, so ``dout dot out`` is not accurate enough
        for the route-softmax Jacobian: it loses the pre-cast FP32 weighted
        mean, and that error is amplified by the subtraction in ``dz``.  This
        query-owned kernel instead evaluates

            delta = sum_m p_m * (dout dot read_m) + dP_null * p_null

        directly while scanning the summaries.  Its only output is the compact
        ``[T, H]`` delta tensor.
        """

        _H, _K, _V, _R = H, K, V, R
        _M, _BT, _BV = M, BT, BV
        _has_dlocal = has_dlocal
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        read_q_s = (1, T_d, _H, _K)
        router_q_s = (1, T_d, _H, _R)
        states_s = (1, _M, _H, _K, _V)
        summary_k_s = (1, _M, _H, _R)
        dout_s = (1, T_d, _H, _V)
        boundary_s = (_M,)
        token_range_s = (T_d,)
        stat_s = (T_d, _H)

        @T.prim_func
        def kernel(
            read_q: T.Tensor(read_q_s, T.bfloat16),
            router_q: T.Tensor(router_q_s, T.bfloat16),
            states: T.Tensor(states_s, T.float32),
            summary_k: T.Tensor(summary_k_s, T.bfloat16),
            boundary_counts: T.Tensor(boundary_s, T.int64),
            token_summary_start: T.Tensor(token_range_s, T.int64),
            token_summary_end: T.Tensor(token_range_s, T.int64),
            lse: T.Tensor(stat_s, T.float32),
            p_null: T.Tensor(stat_s, T.float32),
            dlocal_prob: T.Tensor(stat_s, T.float32),
            dout: T.Tensor(dout_s, T.bfloat16),
            delta: T.Tensor(stat_s, T.float32),
            route_scale: T.float32,
            read_scale: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(T_d, _BT), _H, threads=_threads
            ) as (i_t, i_h):
                t_s = i_t * _BT
                q_shared = T.alloc_shared((_BT, _K), T.bfloat16)
                router_shared = T.alloc_shared((_BT, _R), T.bfloat16)
                key_route_shared = T.alloc_shared((_R,), T.bfloat16)
                state_shared = T.alloc_shared((_K, _BV), T.bfloat16)
                dout_shared = T.alloc_shared((_BT, _BV), T.bfloat16)
                route_acc = T.alloc_fragment((_BT,), T.float32)
                read_acc = T.alloc_fragment((_BT, _BV), T.float32)
                prod = T.alloc_fragment((_BT, _BV), T.float32)
                dot_vec = T.alloc_fragment((_BT,), T.float32)
                dot_vec_shared = T.alloc_shared((_BT,), T.float32)
                dot_read_shared = T.alloc_shared((_BT,), T.float32)
                prob_shared = T.alloc_shared((_BT,), T.float32)
                common_shared = T.alloc_shared((_BT,), T.float32)
                for i in T.Parallel(_BT):
                    common_shared[i] = 0.0
                T.copy(
                    read_q[0, t_s : t_s + _BT, i_h, :],
                    q_shared,
                )
                T.copy(
                    router_q[0, t_s : t_s + _BT, i_h, :],
                    router_shared,
                )

                block_last_t = T.min(t_s + _BT - 1, T_d - 1)
                for i_m in T.serial(_M):
                    block_probe_t = T.min(
                        T.max(t_s, boundary_counts[i_m]),
                        block_last_t,
                    )
                    block_active = (
                        (boundary_counts[i_m] <= block_last_t)
                        & (
                            i_m
                            >= token_summary_start[block_probe_t]
                        )
                        & (
                            i_m
                            < token_summary_end[block_probe_t]
                        )
                    )
                    if block_active:
                        T.copy(
                            summary_k[0, i_m, i_h, :],
                            key_route_shared,
                        )
                        for i in T.Parallel(_BT):
                            route_acc[i] = 0.0
                        for j in T.serial(_R):
                            for i in T.Parallel(_BT):
                                route_acc[i] += (
                                    T.cast(router_shared[i, j], T.float32)
                                    * T.cast(key_route_shared[j], T.float32)
                                )
                        for i in T.Parallel(_BT):
                            dot_read_shared[i] = 0.0
                        for i_v in T.serial(T.ceildiv(_V, _BV)):
                            v_s = i_v * _BV
                            T.copy(
                                states[
                                    0,
                                    i_m,
                                    i_h,
                                    :,
                                    v_s : v_s + _BV,
                                ],
                                state_shared,
                            )
                            T.copy(
                                dout[
                                    0,
                                    t_s : t_s + _BT,
                                    i_h,
                                    v_s : v_s + _BV,
                                ],
                                dout_shared,
                            )
                            T.clear(read_acc)
                            T.gemm(
                                q_shared,
                                state_shared,
                                read_acc,
                                policy=T.GemmWarpPolicy.FullRow,
                            )
                            for i, j in T.Parallel(_BT, _BV):
                                prod[i, j] = (
                                    read_acc[i, j]
                                    * T.cast(
                                        dout_shared[i, j],
                                        T.float32,
                                    )
                                )
                            T.reduce_sum(prod, dot_vec, dim=1)
                            T.copy(dot_vec, dot_vec_shared)
                            for i in T.Parallel(_BT):
                                dot_read_shared[i] += (
                                    dot_vec_shared[i] * read_scale
                                )

                        for i in T.Parallel(_BT):
                            valid = (
                                (t_s + i < T_d)
                                & (
                                    i_m
                                    >= token_summary_start[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    i_m
                                    < token_summary_end[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    boundary_counts[i_m]
                                    <= T.min(t_s + i, T_d - 1)
                                )
                            )
                            prob_shared[i] = T.if_then_else(
                                valid,
                                T.exp2(
                                    (
                                        route_acc[i] * route_scale
                                        - lse[
                                            T.min(t_s + i, T_d - 1),
                                            i_h,
                                        ]
                                    )
                                    * _LOG2E
                                ),
                                0.0,
                            )
                            common_shared[i] += (
                                prob_shared[i] * dot_read_shared[i]
                            )

                for i in T.Parallel(_BT):
                    if t_s + i < T_d:
                        if _has_dlocal:
                            common_shared[i] += (
                                dlocal_prob[t_s + i, i_h]
                                * p_null[t_s + i, i_h]
                            )
                        delta[t_s + i, i_h] = common_shared[i]

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        }
    )
    def _build_online_dread_kernel(H, K, V, R, M, BT, BK, BV, num_warps=4):
        """Query-owned dReadQ; probabilities are recomputed from forward LSE."""

        _H, _K, _V, _R = H, K, V, R
        _M, _BT, _BK, _BV = M, BT, BK, BV
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        read_q_s = (1, T_d, _H, _K)
        router_q_s = (1, T_d, _H, _R)
        states_s = (1, _M, _H, _K, _V)
        summary_k_s = (1, _M, _H, _R)
        dout_s = (1, T_d, _H, _V)
        boundary_s = (_M,)
        token_range_s = (T_d,)
        stat_s = (T_d, _H)

        @T.prim_func
        def kernel(
            router_q: T.Tensor(router_q_s, T.bfloat16),
            states: T.Tensor(states_s, T.float32),
            summary_k: T.Tensor(summary_k_s, T.bfloat16),
            boundary_counts: T.Tensor(boundary_s, T.int64),
            token_summary_start: T.Tensor(token_range_s, T.int64),
            token_summary_end: T.Tensor(token_range_s, T.int64),
            lse: T.Tensor(stat_s, T.float32),
            dout: T.Tensor(dout_s, T.bfloat16),
            dread: T.Tensor(read_q_s, T.bfloat16),
            route_scale: T.float32,
            read_scale: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(_K, _BK),
                T.ceildiv(T_d, _BT),
                _H,
                threads=_threads,
            ) as (i_k, i_t, i_h):
                k_s = i_k * _BK
                t_s = i_t * _BT
                router_shared = T.alloc_shared((_BT, _R), T.bfloat16)
                key_route_shared = T.alloc_shared((_R,), T.bfloat16)
                dout_shared = T.alloc_shared((_BT, _BV), T.bfloat16)
                weighted_dout_shared = T.alloc_shared(
                    (_BT, _BV), T.bfloat16
                )
                state_shared = T.alloc_shared((_BK, _BV), T.bfloat16)
                route_acc = T.alloc_fragment((_BT,), T.float32)
                grad_acc = T.alloc_fragment((_BT, _BK), T.float32)
                prob_shared = T.alloc_shared((_BT,), T.float32)
                T.clear(grad_acc)
                T.copy(
                    router_q[0, t_s : t_s + _BT, i_h, :],
                    router_shared,
                )

                block_last_t = T.min(t_s + _BT - 1, T_d - 1)

                for i_m in T.serial(_M):
                    # CTA-uniform exact visibility test.  Probing the first
                    # potentially causal row rejects a summary emitted at the
                    # end of a packed document, because that row belongs to
                    # the next document.  Thus an all-invalid candidate never
                    # loads its state or enters a tensor-core GEMM.
                    block_probe_t = T.min(
                        T.max(t_s, boundary_counts[i_m]),
                        block_last_t,
                    )
                    if (
                        (boundary_counts[i_m] <= block_last_t)
                        & (
                            i_m
                            >= token_summary_start[block_probe_t]
                        )
                        & (
                            i_m
                            < token_summary_end[block_probe_t]
                        )
                    ):
                        T.copy(summary_k[0, i_m, i_h, :], key_route_shared)
                        for i in T.Parallel(_BT):
                            route_acc[i] = 0.0
                        for j in T.serial(_R):
                            for i in T.Parallel(_BT):
                                route_acc[i] += (
                                    T.cast(router_shared[i, j], T.float32)
                                    * T.cast(key_route_shared[j], T.float32)
                                )
                        for i in T.Parallel(_BT):
                            valid = (
                                (t_s + i < T_d)
                                & (
                                    i_m
                                    >= token_summary_start[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    i_m
                                    < token_summary_end[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    boundary_counts[i_m]
                                    <= T.min(t_s + i, T_d - 1)
                                )
                            )
                            prob_shared[i] = T.if_then_else(
                                valid,
                                T.exp2(
                                    (
                                        route_acc[i] * route_scale
                                        - lse[
                                            T.min(t_s + i, T_d - 1),
                                            i_h,
                                        ]
                                    )
                                    * _LOG2E
                                ),
                                0.0,
                            )

                        for i_v in T.serial(T.ceildiv(_V, _BV)):
                            v_s = i_v * _BV
                            T.copy(
                                dout[
                                    0,
                                    t_s : t_s + _BT,
                                    i_h,
                                    v_s : v_s + _BV,
                                ],
                                dout_shared,
                            )
                            T.copy(
                                states[
                                    0,
                                    i_m,
                                    i_h,
                                    k_s : k_s + _BK,
                                    v_s : v_s + _BV,
                                ],
                                state_shared,
                            )
                            for i, j in T.Parallel(_BT, _BV):
                                weighted_dout_shared[i, j] = T.cast(
                                    T.cast(
                                        dout_shared[i, j],
                                        T.float32,
                                    )
                                    * prob_shared[i]
                                    * read_scale,
                                    T.bfloat16,
                                )
                            T.gemm(
                                weighted_dout_shared,
                                state_shared,
                                grad_acc,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                            )

                out_shared = T.alloc_shared((_BT, _BK), T.bfloat16)
                T.copy(grad_acc, out_shared)
                for i, j in T.Parallel(_BT, _BK):
                    if (t_s + i < T_d) & (k_s + j < _K):
                        dread[0, t_s + i, i_h, k_s + j] = out_shared[i, j]

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        }
    )
    def _build_online_dread_route_kernel(H, K, V, R, M, BT, BK, BV, num_warps=4):
        """Fuse dReadQ, one K-tile dRouterQ partial, and dSummaryK.

        Each CTA owns one ``(K tile, query tile, head)``.  The raw
        ``dO @ state[K-tile].T`` product supplies both dReadQ and this K
        tile's contribution to the route-logit gradient.  The ``K / BK``
        partials are reduced by a compact Triton kernel after this
        kernel.  dSummaryK is linear in those same partials, so each CTA
        reduces its query tile in FP32 and atomically adds directly into the
        compact ``[M,H,R]`` accumulator.

        The full softmax common term ``delta`` is subtracted only by K tile
        zero.  Summing all route partials is therefore exactly the
        original route-logit derivative, including the null candidate.
        """

        _H, _K, _V, _R = H, K, V, R
        _M, _BT, _BK, _BV = M, BT, BK, BV
        _NK = _K // _BK
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        read_q_s = (1, T_d, _H, _K)
        router_q_s = (1, T_d, _H, _R)
        states_s = (1, _M, _H, _K, _V)
        summary_k_s = (1, _M, _H, _R)
        dout_s = (1, T_d, _H, _V)
        drouter_partial_s = (_NK, T_d, _H, _R)
        boundary_s = (_M,)
        token_range_s = (T_d,)
        stat_s = (T_d, _H)

        @T.prim_func
        def kernel(
            read_q: T.Tensor(read_q_s, T.bfloat16),
            router_q: T.Tensor(router_q_s, T.bfloat16),
            states: T.Tensor(states_s, T.float32),
            summary_k: T.Tensor(summary_k_s, T.bfloat16),
            boundary_counts: T.Tensor(boundary_s, T.int64),
            token_summary_start: T.Tensor(token_range_s, T.int64),
            token_summary_end: T.Tensor(token_range_s, T.int64),
            lse: T.Tensor(stat_s, T.float32),
            delta: T.Tensor(stat_s, T.float32),
            dout: T.Tensor(dout_s, T.bfloat16),
            dread: T.Tensor(read_q_s, T.bfloat16),
            drouter_partial: T.Tensor(drouter_partial_s, T.float32),
            dsummary_k_acc: T.Tensor(summary_k_s, T.float32),
            route_scale: T.float32,
            read_scale: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(_K, _BK),
                T.ceildiv(T_d, _BT),
                _H,
                threads=_threads,
            ) as (i_k, i_t, i_h):
                k_s = i_k * _BK
                t_s = i_t * _BT
                q_shared = T.alloc_shared((_BT, _BK), T.bfloat16)
                router_shared = T.alloc_shared((_BT, _R), T.bfloat16)
                key_route_shared = T.alloc_shared((_R,), T.bfloat16)
                dout_shared = T.alloc_shared((_BT, _BV), T.bfloat16)
                state_shared = T.alloc_shared((_BK, _BV), T.bfloat16)
                prob_shared = T.alloc_shared((_BT,), T.float32)
                dot_read_shared = T.alloc_shared((_BT,), T.float32)
                dz_partial_shared = T.alloc_shared((_BT,), T.float32)
                route_grad_shared = T.alloc_shared((_BT, _R), T.float32)
                summary_partial = T.alloc_shared((_R,), T.float32)
                route_acc = T.alloc_fragment((_BT,), T.float32)
                summary_grad_acc = T.alloc_fragment(
                    (_BT, _BK),
                    T.float32,
                )
                dread_acc = T.alloc_fragment((_BT, _BK), T.float32)
                dot_vec = T.alloc_fragment((_BT,), T.float32)

                T.clear(dread_acc)
                for i, j in T.Parallel(_BT, _R):
                    route_grad_shared[i, j] = 0.0
                T.copy(
                    read_q[
                        0,
                        t_s : t_s + _BT,
                        i_h,
                        k_s : k_s + _BK,
                    ],
                    q_shared,
                )
                T.copy(
                    router_q[0, t_s : t_s + _BT, i_h, :],
                    router_shared,
                )

                block_last_t = T.min(t_s + _BT - 1, T_d - 1)

                for i_m in T.serial(_M):
                    block_probe_t = T.min(
                        T.max(t_s, boundary_counts[i_m]),
                        block_last_t,
                    )
                    block_active = (
                        (boundary_counts[i_m] <= block_last_t)
                        & (
                            i_m
                            >= token_summary_start[block_probe_t]
                        )
                        & (
                            i_m
                            < token_summary_end[block_probe_t]
                        )
                    )
                    if block_active:
                        T.copy(
                            summary_k[0, i_m, i_h, :],
                            key_route_shared,
                        )
                        for i in T.Parallel(_BT):
                            route_acc[i] = 0.0
                        for j in T.serial(_R):
                            for i in T.Parallel(_BT):
                                route_acc[i] += (
                                    T.cast(
                                        router_shared[i, j],
                                        T.float32,
                                    )
                                    * T.cast(
                                        key_route_shared[j],
                                        T.float32,
                                    )
                                )
                        for i in T.Parallel(_BT):
                            valid = (
                                (t_s + i < T_d)
                                & (
                                    i_m
                                    >= token_summary_start[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    i_m
                                    < token_summary_end[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    boundary_counts[i_m]
                                    <= T.min(t_s + i, T_d - 1)
                                )
                            )
                            prob_shared[i] = T.if_then_else(
                                valid,
                                T.exp2(
                                    (
                                        route_acc[i] * route_scale
                                        - lse[
                                            T.min(t_s + i, T_d - 1),
                                            i_h,
                                        ]
                                    )
                                    * _LOG2E
                                ),
                                0.0,
                            )

                        # This unweighted K-tile derivative is shared by
                        # dReadQ and the route derivative q dot dReadQ.
                        T.clear(summary_grad_acc)
                        for i_v in T.serial(T.ceildiv(_V, _BV)):
                            v_s = i_v * _BV
                            T.copy(
                                dout[
                                    0,
                                    t_s : t_s + _BT,
                                    i_h,
                                    v_s : v_s + _BV,
                                ],
                                dout_shared,
                            )
                            T.copy(
                                states[
                                    0,
                                    i_m,
                                    i_h,
                                    k_s : k_s + _BK,
                                    v_s : v_s + _BV,
                                ],
                                state_shared,
                            )
                            T.gemm(
                                dout_shared,
                                state_shared,
                                summary_grad_acc,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                            )

                        for i, j in T.Parallel(_BT, _BK):
                            dread_acc[i, j] += (
                                summary_grad_acc[i, j]
                                * prob_shared[i]
                                * read_scale
                            )
                            # The raw derivative is no longer needed after
                            # dRead accumulation, so reuse its fragment for
                            # the per-row dot instead of allocating a third
                            # BT*BK FP32 fragment.
                            summary_grad_acc[i, j] *= T.cast(
                                q_shared[i, j],
                                T.float32,
                            )
                        T.reduce_sum(
                            summary_grad_acc,
                            dot_vec,
                            dim=1,
                        )
                        T.copy(dot_vec, dot_read_shared)

                        for i in T.Parallel(_BT):
                            dz_partial_shared[i] = prob_shared[i] * (
                                dot_read_shared[i] * read_scale
                                - T.if_then_else(
                                    i_k == 0,
                                    delta[
                                        T.min(t_s + i, T_d - 1),
                                        i_h,
                                    ],
                                    0.0,
                                )
                            )
                        for i, j in T.Parallel(_BT, _R):
                            route_grad_shared[i, j] += (
                                dz_partial_shared[i]
                                * T.cast(
                                    key_route_shared[j],
                                    T.float32,
                                )
                                * route_scale
                            )

                        # dSummaryK is linear in dz.  Reduce this query tile
                        # before one FP32 atomic per route dimension.
                        for j in T.Parallel(_R):
                            summary_partial[j] = 0.0
                        for i in T.serial(_BT):
                            for j in T.Parallel(_R):
                                summary_partial[j] += (
                                    dz_partial_shared[i]
                                    * T.cast(
                                        router_shared[i, j],
                                        T.float32,
                                    )
                                    * route_scale
                                )
                        for j in T.Parallel(_R):
                            T.atomic_add(
                                dsummary_k_acc[0, i_m, i_h, j],
                                summary_partial[j],
                            )

                dread_shared = T.alloc_shared(
                    (_BT, _BK),
                    T.bfloat16,
                )
                T.copy(dread_acc, dread_shared)
                for i, j in T.Parallel(_BT, _BK):
                    if (t_s + i < T_d) & (k_s + j < _K):
                        dread[0, t_s + i, i_h, k_s + j] = (
                            dread_shared[i, j]
                        )
                for i, j in T.Parallel(_BT, _R):
                    if t_s + i < T_d:
                        drouter_partial[i_k, t_s + i, i_h, j] = (
                            route_grad_shared[i, j]
                        )

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        }
    )
    def _build_online_dstates_kernel(H, K, V, R, M, BT, BK, BV, num_warps=4):
        """Summary-owned dState; no probability tensor is saved or rebuilt."""

        _H, _K, _V, _R = H, K, V, R
        _M, _BT, _BK, _BV = M, BT, BK, BV
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        read_q_s = (1, T_d, _H, _K)
        router_q_s = (1, T_d, _H, _R)
        summary_k_s = (1, _M, _H, _R)
        dout_s = (1, T_d, _H, _V)
        dstates_s = (1, _M, _H, _K, _V)
        boundary_s = (_M,)
        token_range_s = (T_d,)
        stat_s = (T_d, _H)

        @T.prim_func
        def kernel(
            read_q: T.Tensor(read_q_s, T.bfloat16),
            router_q: T.Tensor(router_q_s, T.bfloat16),
            summary_k: T.Tensor(summary_k_s, T.bfloat16),
            boundary_counts: T.Tensor(boundary_s, T.int64),
            token_summary_start: T.Tensor(token_range_s, T.int64),
            token_summary_end: T.Tensor(token_range_s, T.int64),
            lse: T.Tensor(stat_s, T.float32),
            dout: T.Tensor(dout_s, T.bfloat16),
            dstates: T.Tensor(dstates_s, T.float32),
            route_scale: T.float32,
            read_scale: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(_V, _BV),
                T.ceildiv(_K, _BK),
                _M * _H,
                threads=_threads,
            ) as (i_v, i_k, i_mh):
                i_m = i_mh % _M
                i_h = i_mh // _M
                v_s = i_v * _BV
                k_s = i_k * _BK
                q_shared = T.alloc_shared((_BT, _BK), T.bfloat16)
                router_shared = T.alloc_shared((_BT, _R), T.bfloat16)
                key_route_shared = T.alloc_shared((_R,), T.bfloat16)
                dout_shared = T.alloc_shared((_BT, _BV), T.bfloat16)
                weighted_dout_shared = T.alloc_shared(
                    (_BT, _BV), T.bfloat16
                )
                route_acc = T.alloc_fragment((_BT,), T.float32)
                grad_acc = T.alloc_fragment((_BK, _BV), T.float32)
                prob_shared = T.alloc_shared((_BT,), T.float32)
                T.clear(grad_acc)

                T.copy(summary_k[0, i_m, i_h, :], key_route_shared)

                # TileLang 0.1.9's Hopper warp-specialized pipeline can leave
                # producer/consumer barriers unmatched on the dynamic final
                # T tile.  A single-stage pipeline offers no useful overlap
                # here, so keep the dynamic scan explicitly serial.
                for i_tb in T.serial(T.ceildiv(T_d, _BT)):
                    t_s = i_tb * _BT
                    block_last_t = T.min(t_s + _BT - 1, T_d - 1)
                    block_probe_t = T.min(
                        T.max(t_s, boundary_counts[i_m]),
                        block_last_t,
                    )
                    if (
                        (boundary_counts[i_m] <= block_last_t)
                        & (
                            i_m
                            >= token_summary_start[block_probe_t]
                        )
                        & (
                            i_m
                            < token_summary_end[block_probe_t]
                        )
                    ):
                        T.copy(
                            read_q[
                                0,
                                t_s : t_s + _BT,
                                i_h,
                                k_s : k_s + _BK,
                            ],
                            q_shared,
                        )
                        T.copy(
                            router_q[0, t_s : t_s + _BT, i_h, :],
                            router_shared,
                        )
                        T.copy(
                            dout[
                                0,
                                t_s : t_s + _BT,
                                i_h,
                                v_s : v_s + _BV,
                            ],
                            dout_shared,
                        )
                        for i in T.Parallel(_BT):
                            route_acc[i] = 0.0
                        for j in T.serial(_R):
                            for i in T.Parallel(_BT):
                                route_acc[i] += (
                                    T.cast(router_shared[i, j], T.float32)
                                    * T.cast(key_route_shared[j], T.float32)
                                )
                        for i in T.Parallel(_BT):
                            valid = (
                                (t_s + i < T_d)
                                & (
                                    i_m
                                    >= token_summary_start[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    i_m
                                    < token_summary_end[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    boundary_counts[i_m]
                                    <= T.min(t_s + i, T_d - 1)
                                )
                            )
                            prob_shared[i] = T.if_then_else(
                                valid,
                                T.exp2(
                                    (
                                        route_acc[i] * route_scale
                                        - lse[
                                            T.min(t_s + i, T_d - 1),
                                            i_h,
                                        ]
                                    )
                                    * _LOG2E
                                ),
                                0.0,
                            )
                        for i, j in T.Parallel(_BT, _BV):
                            weighted_dout_shared[i, j] = T.cast(
                                T.cast(
                                    dout_shared[i, j],
                                    T.float32,
                                )
                                * prob_shared[i]
                                * read_scale,
                                T.bfloat16,
                            )
                        T.gemm(
                            q_shared,
                            weighted_dout_shared,
                            grad_acc,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                out_shared = T.alloc_shared((_BK, _BV), T.float32)
                T.copy(grad_acc, out_shared)
                for i, j in T.Parallel(_BK, _BV):
                    if (k_s + i < _K) & (v_s + j < _V):
                        dstates[0, i_m, i_h, k_s + i, v_s + j] = (
                            out_shared[i, j]
                        )

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        }
    )
    def _build_online_drouter_kernel(
        H,
        K,
        V,
        R,
        M,
        BT,
        BV,
        fuse_summary_k=False,
        num_warps=4,
    ):
        """Query-owned route gradients with one route/read recomputation.

        ``dRouterQ`` is CTA-owned.  When ``fuse_summary_k`` is true, the same
        FP32 ``dz`` values are reduced over the CTA's query tile and atomically
        accumulated into a compact FP32 ``[M,H,R]`` dSummaryK buffer.  This
        removes the summary-owned third q@state scan without introducing any
        workspace proportional to ``T*H*M``.
        """

        _H, _K, _V, _R = H, K, V, R
        _M, _BT, _BV = M, BT, BV
        _fuse_summary_k = fuse_summary_k
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        read_q_s = (1, T_d, _H, _K)
        router_q_s = (1, T_d, _H, _R)
        states_s = (1, _M, _H, _K, _V)
        summary_k_s = (1, _M, _H, _R)
        dout_s = (1, T_d, _H, _V)
        boundary_s = (_M,)
        token_range_s = (T_d,)
        stat_s = (T_d, _H)

        @T.prim_func
        def kernel(
            read_q: T.Tensor(read_q_s, T.bfloat16),
            router_q: T.Tensor(router_q_s, T.bfloat16),
            states: T.Tensor(states_s, T.float32),
            summary_k: T.Tensor(summary_k_s, T.bfloat16),
            boundary_counts: T.Tensor(boundary_s, T.int64),
            token_summary_start: T.Tensor(token_range_s, T.int64),
            token_summary_end: T.Tensor(token_range_s, T.int64),
            lse: T.Tensor(stat_s, T.float32),
            delta: T.Tensor(stat_s, T.float32),
            dout: T.Tensor(dout_s, T.bfloat16),
            drouter: T.Tensor(router_q_s, T.bfloat16),
            dsummary_k_acc: T.Tensor(summary_k_s, T.float32),
            route_scale: T.float32,
            read_scale: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(T_d, _BT), _H, threads=_threads
            ) as (i_t, i_h):
                t_s = i_t * _BT
                q_shared = T.alloc_shared((_BT, _K), T.bfloat16)
                router_shared = T.alloc_shared((_BT, _R), T.bfloat16)
                key_route_shared = T.alloc_shared((_R,), T.bfloat16)
                state_shared = T.alloc_shared((_K, _BV), T.bfloat16)
                dout_shared = T.alloc_shared((_BT, _BV), T.bfloat16)
                route_acc = T.alloc_fragment((_BT,), T.float32)
                read_acc = T.alloc_fragment((_BT, _BV), T.float32)
                prod = T.alloc_fragment((_BT, _BV), T.float32)
                dot_vec = T.alloc_fragment((_BT,), T.float32)
                dot_vec_shared = T.alloc_shared((_BT,), T.float32)
                dot_read_shared = T.alloc_shared((_BT,), T.float32)
                prob_shared = T.alloc_shared((_BT,), T.float32)
                dz_value_shared = T.alloc_shared((_BT,), T.float32)
                grad_shared = T.alloc_shared((_BT, _R), T.float32)
                summary_partial = T.alloc_shared((_R,), T.float32)
                for i, j in T.Parallel(_BT, _R):
                    grad_shared[i, j] = 0.0
                T.copy(
                    read_q[0, t_s : t_s + _BT, i_h, :],
                    q_shared,
                )
                T.copy(
                    router_q[0, t_s : t_s + _BT, i_h, :],
                    router_shared,
                )
                T.clear(dout_shared)
                block_last_t = T.min(t_s + _BT - 1, T_d - 1)

                for i_m in T.serial(_M):
                    block_probe_t = T.min(
                        T.max(t_s, boundary_counts[i_m]),
                        block_last_t,
                    )
                    block_active = (
                        (boundary_counts[i_m] <= block_last_t)
                        & (
                            i_m
                            >= token_summary_start[block_probe_t]
                        )
                        & (
                            i_m
                            < token_summary_end[block_probe_t]
                        )
                    )
                    if block_active:
                        T.copy(
                            summary_k[0, i_m, i_h, :],
                            key_route_shared,
                        )
                        for i in T.Parallel(_BT):
                            route_acc[i] = 0.0
                        for j in T.serial(_R):
                            for i in T.Parallel(_BT):
                                route_acc[i] += (
                                    T.cast(
                                        router_shared[i, j],
                                        T.float32,
                                    )
                                    * T.cast(
                                        key_route_shared[j],
                                        T.float32,
                                    )
                                )
                        for i in T.Parallel(_BT):
                            dot_read_shared[i] = 0.0
                        for i_v in T.serial(T.ceildiv(_V, _BV)):
                            v_s = i_v * _BV
                            T.copy(
                                dout[
                                    0,
                                    t_s : t_s + _BT,
                                    i_h,
                                    v_s : v_s + _BV,
                                ],
                                dout_shared,
                            )
                            T.clear(read_acc)
                            T.copy(
                                states[
                                    0,
                                    i_m,
                                    i_h,
                                    :,
                                    v_s : v_s + _BV,
                                ],
                                state_shared,
                            )
                            T.gemm(
                                q_shared,
                                state_shared,
                                read_acc,
                                policy=T.GemmWarpPolicy.FullRow,
                            )
                            for i, j in T.Parallel(_BT, _BV):
                                prod[i, j] = (
                                    read_acc[i, j]
                                    * T.cast(
                                        dout_shared[i, j],
                                        T.float32,
                                    )
                                )
                            T.reduce_sum(prod, dot_vec, dim=1)
                            T.copy(dot_vec, dot_vec_shared)
                            for i in T.Parallel(_BT):
                                dot_read_shared[i] += (
                                    dot_vec_shared[i] * read_scale
                                )

                        for i in T.Parallel(_BT):
                            valid = (
                                (t_s + i < T_d)
                                & (
                                    i_m
                                    >= token_summary_start[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    i_m
                                    < token_summary_end[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    boundary_counts[i_m]
                                    <= T.min(t_s + i, T_d - 1)
                                )
                            )
                            prob_shared[i] = T.if_then_else(
                                valid,
                                T.exp2(
                                    (
                                        route_acc[i] * route_scale
                                        - lse[
                                            T.min(t_s + i, T_d - 1),
                                            i_h,
                                        ]
                                    )
                                    * _LOG2E
                                ),
                                0.0,
                            )
                            dz_value_shared[i] = prob_shared[i] * (
                                dot_read_shared[i]
                                - delta[
                                    T.min(t_s + i, T_d - 1),
                                    i_h,
                                ]
                            )
                        # dz may be much smaller than the route inputs.
                        # Preserve the true outer product in FP32.
                        for i, j in T.Parallel(_BT, _R):
                            grad_shared[i, j] += (
                                dz_value_shared[i]
                                * T.cast(
                                    key_route_shared[j],
                                    T.float32,
                                )
                                * route_scale
                            )
                        if _fuse_summary_k:
                            # Reduce BT rows before one FP32 atomic per route
                            # dimension into the compact summary-key gradient.
                            for j in T.Parallel(_R):
                                summary_partial[j] = 0.0
                            for i in T.serial(_BT):
                                for j in T.Parallel(_R):
                                    summary_partial[j] += (
                                        dz_value_shared[i]
                                        * T.cast(
                                            router_shared[i, j],
                                            T.float32,
                                        )
                                        * route_scale
                                    )
                            for j in T.Parallel(_R):
                                T.atomic_add(
                                    dsummary_k_acc[0, i_m, i_h, j],
                                    summary_partial[j],
                                )

                for i, j in T.Parallel(_BT, _R):
                    if t_s + i < T_d:
                        drouter[0, t_s + i, i_h, j] = T.cast(
                            grad_shared[i, j],
                            T.bfloat16,
                        )

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        }
    )
    def _build_online_dsummary_k_kernel(H, K, V, R, M, BT, BV, num_warps=4):
        """Summary-owned dSummaryK with route/read recomputation."""

        _H, _K, _V, _R = H, K, V, R
        _M, _BT, _BV = M, BT, BV
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        read_q_s = (1, T_d, _H, _K)
        router_q_s = (1, T_d, _H, _R)
        states_s = (1, _M, _H, _K, _V)
        summary_k_s = (1, _M, _H, _R)
        dout_s = (1, T_d, _H, _V)
        boundary_s = (_M,)
        token_range_s = (T_d,)
        stat_s = (T_d, _H)

        @T.prim_func
        def kernel(
            read_q: T.Tensor(read_q_s, T.bfloat16),
            router_q: T.Tensor(router_q_s, T.bfloat16),
            states: T.Tensor(states_s, T.float32),
            summary_k: T.Tensor(summary_k_s, T.bfloat16),
            boundary_counts: T.Tensor(boundary_s, T.int64),
            token_summary_start: T.Tensor(token_range_s, T.int64),
            token_summary_end: T.Tensor(token_range_s, T.int64),
            lse: T.Tensor(stat_s, T.float32),
            delta: T.Tensor(stat_s, T.float32),
            dout: T.Tensor(dout_s, T.bfloat16),
            dsummary_k: T.Tensor(summary_k_s, T.bfloat16),
            route_scale: T.float32,
            read_scale: T.float32,
        ):
            with T.Kernel(_M, _H, threads=_threads) as (i_m, i_h):
                q_shared = T.alloc_shared((_BT, _K), T.bfloat16)
                router_shared = T.alloc_shared((_BT, _R), T.bfloat16)
                key_route_shared = T.alloc_shared((_R,), T.bfloat16)
                state_shared = T.alloc_shared((_K, _BV), T.bfloat16)
                dout_shared = T.alloc_shared((_BT, _BV), T.bfloat16)
                route_acc = T.alloc_fragment((_BT,), T.float32)
                read_acc = T.alloc_fragment((_BT, _BV), T.float32)
                prod = T.alloc_fragment((_BT, _BV), T.float32)
                dot_vec = T.alloc_fragment((_BT,), T.float32)
                dot_vec_shared = T.alloc_shared((_BT,), T.float32)
                dot_read_shared = T.alloc_shared((_BT,), T.float32)
                prob_shared = T.alloc_shared((_BT,), T.float32)
                dz_value_shared = T.alloc_shared((_BT,), T.float32)
                grad_shared = T.alloc_shared((_R,), T.float32)
                for j in T.Parallel(_R):
                    grad_shared[j] = 0.0

                T.copy(summary_k[0, i_m, i_h, :], key_route_shared)

                # See dState above: the dynamic packed tail must not enter the
                # one-stage warp-specialized pipeline on TileLang 0.1.9.
                for i_tb in T.serial(T.ceildiv(T_d, _BT)):
                    t_s = i_tb * _BT
                    block_last_t = T.min(t_s + _BT - 1, T_d - 1)
                    block_probe_t = T.min(
                        T.max(t_s, boundary_counts[i_m]),
                        block_last_t,
                    )
                    block_active = (
                        (boundary_counts[i_m] <= block_last_t)
                        & (
                            i_m
                            >= token_summary_start[block_probe_t]
                        )
                        & (
                            i_m
                            < token_summary_end[block_probe_t]
                        )
                    )
                    if block_active:
                        T.copy(
                            read_q[0, t_s : t_s + _BT, i_h, :],
                            q_shared,
                        )
                        T.copy(
                            router_q[0, t_s : t_s + _BT, i_h, :],
                            router_shared,
                        )
                        for i in T.Parallel(_BT):
                            route_acc[i] = 0.0
                        for j in T.serial(_R):
                            for i in T.Parallel(_BT):
                                route_acc[i] += (
                                    T.cast(
                                        router_shared[i, j],
                                        T.float32,
                                    )
                                    * T.cast(
                                        key_route_shared[j],
                                        T.float32,
                                    )
                                )
                        for i in T.Parallel(_BT):
                            dot_read_shared[i] = 0.0
                        for i_v in T.serial(T.ceildiv(_V, _BV)):
                            v_s = i_v * _BV
                            T.copy(
                                dout[
                                    0,
                                    t_s : t_s + _BT,
                                    i_h,
                                    v_s : v_s + _BV,
                                ],
                                dout_shared,
                            )
                            T.clear(read_acc)
                            T.copy(
                                states[
                                    0,
                                    i_m,
                                    i_h,
                                    :,
                                    v_s : v_s + _BV,
                                ],
                                state_shared,
                            )
                            T.gemm(
                                q_shared,
                                state_shared,
                                read_acc,
                                policy=T.GemmWarpPolicy.FullRow,
                            )
                            for i, j in T.Parallel(_BT, _BV):
                                prod[i, j] = (
                                    read_acc[i, j]
                                    * T.cast(
                                        dout_shared[i, j],
                                        T.float32,
                                    )
                                )
                            T.reduce_sum(prod, dot_vec, dim=1)
                            T.copy(dot_vec, dot_vec_shared)
                            for i in T.Parallel(_BT):
                                dot_read_shared[i] += (
                                    dot_vec_shared[i] * read_scale
                                )

                        for i in T.Parallel(_BT):
                            valid = (
                                (t_s + i < T_d)
                                & (
                                    i_m
                                    >= token_summary_start[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    i_m
                                    < token_summary_end[
                                        T.min(t_s + i, T_d - 1)
                                    ]
                                )
                                & (
                                    boundary_counts[i_m]
                                    <= T.min(t_s + i, T_d - 1)
                                )
                            )
                            prob_shared[i] = T.if_then_else(
                                valid,
                                T.exp2(
                                    (
                                        route_acc[i] * route_scale
                                        - lse[
                                            T.min(t_s + i, T_d - 1),
                                            i_h,
                                        ]
                                    )
                                    * _LOG2E
                                ),
                                0.0,
                            )
                            dz_value_shared[i] = prob_shared[i] * (
                                dot_read_shared[i]
                                - delta[
                                    T.min(t_s + i, T_d - 1),
                                    i_h,
                                ]
                            )
                        # Preserve dz in FP32 for the summary-key reduction.
                        for i in T.serial(_BT):
                            for j in T.Parallel(_R):
                                grad_shared[j] += (
                                    dz_value_shared[i]
                                    * T.cast(
                                        router_shared[i, j],
                                        T.float32,
                                    )
                                    * route_scale
                                )

                for i in T.Parallel(_R):
                    dsummary_k[0, i_m, i_h, i] = T.cast(
                        grad_shared[i],
                        T.bfloat16,
                    )

        return kernel

else:
    _build_online_exact_delta_kernel = None
    _build_online_dread_kernel = None
    _build_online_dread_route_kernel = None
    _build_online_dstates_kernel = None
    _build_online_drouter_kernel = None
    _build_online_dsummary_k_kernel = None


def _decode_needs(needs: Sequence[bool]) -> tuple[bool, bool, bool, bool, bool]:
    """Accept either five direct flags or ``ctx.needs_input_grad``."""

    flags = tuple(bool(value) for value in needs)
    if len(flags) == 5:
        return flags
    if len(flags) >= 8:
        return flags[0], flags[1], flags[2], flags[3], flags[7]
    raise ValueError(
        "needs must contain either five direct gradient flags or at least "
        "eight autograd input-gradient flags"
    )


def _should_fuse_dread_route(
    *,
    need_dread: bool,
    need_drouter: bool,
    need_dsummary_k: bool,
) -> bool:
    """Select the opt-in three-gradient fusion only for its exact contract."""

    return bool(
        _env_int(
            "HSD_FUSED_GLOBAL_READ_ONLINE_BWD_FUSE_DREAD_ROUTE",
            0,
        )
    ) and bool(need_dread and need_drouter and need_dsummary_k)


def _online_backward_launch_spec(
    profile: str | None,
) -> tuple[int, int, int, int, bool, bool, int, int, int]:
    """Resolve the online backward LaunchSpec.

    Returns ``BT, BK, BV, warps, fuse_route, fuse_dread_route,
    route_BT, route_BV, route_warps``.  Explicit model profiles are immutable;
    the no-profile HF compatibility path retains its legacy environment knobs.
    """
    if profile is not None:
        profile = _canonical_profile(profile)
    if profile == "lc50b_h16_k96_v96_r64":
        # K=V=96 is exactly three 32-wide tensor-core tiles.  Keep the compact
        # dRouter/dSummaryK fusion and use three BK=32 dRead/route partials.
        # This is a separate immutable profile; the established K=64 launch
        # contracts for LC50B H6 and OLMo remain unchanged.
        return 128, 32, 32, 4, True, True, 64, 32, 4
    if profile is not None:
        return 128, 64, 128, 4, True, True, 64, 64, 4
    return (
        max(16, _env_int("HSD_FUSED_GLOBAL_READ_ONLINE_BWD_BT", 128)),
        max(16, _env_int("HSD_FUSED_GLOBAL_READ_ONLINE_BWD_BK", 64)),
        max(16, _env_int("HSD_FUSED_GLOBAL_READ_ONLINE_BWD_BV", 128)),
        _env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4),
        bool(_env_int("HSD_FUSED_GLOBAL_READ_ONLINE_BWD_FUSE_ROUTE_GRADS", 1)),
        bool(_env_int("HSD_FUSED_GLOBAL_READ_ONLINE_BWD_FUSE_DREAD_ROUTE", 0)),
        max(16, _env_int("HSD_FUSED_GLOBAL_READ_ONLINE_BWD_ROUTE_BT", 32)),
        max(16, _env_int("HSD_FUSED_GLOBAL_READ_ONLINE_BWD_ROUTE_BV", 64)),
        _env_int("HSD_FUSED_GLOBAL_READ_ONLINE_BWD_ROUTE_WARPS", 4),
    )


def _validate_inputs(
    *,
    dout: torch.Tensor,
    history_accumulator: torch.Tensor,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    token_summary_start: torch.Tensor,
    token_summary_end: torch.Tensor,
    lse: torch.Tensor,
    p_null: torch.Tensor,
    dlocal_prob: torch.Tensor | None,
    null_logits: torch.Tensor | None,
    has_null: bool,
    route_scale: float,
    read_scale: float,
    profile: str | None,
) -> tuple[int, int, int, int, int, int, str]:
    if tilelang is None or triton is None:
        raise RuntimeError(
            "HSD tile-online backward requires both TileLang and Triton"
        )
    if not read_q.is_cuda:
        raise ValueError("HSD tile-online backward requires CUDA tensors")
    capability = torch.cuda.get_device_capability(read_q.device)
    if capability[0] != 9:
        raise ValueError(
            "HSD tile-online backward is specialized to Hopper (SM90), "
            f"but device capability is {capability[0]}.{capability[1]}"
        )

    if read_q.ndim != 4:
        raise ValueError(f"read_q must be rank 4, got {read_q.shape}")
    batch, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1]) if states.ndim == 5 else -1
    geometry = (
        int(heads),
        int(key_dim),
        int(states.shape[-1]) if states.ndim == 5 else -1,
        int(router_q.shape[-1]) if router_q.ndim == 4 else -1,
    )
    resolved_profile = _resolve_profile(profile, geometry)
    if batch != 1 or resolved_profile is None:
        raise ValueError(
            "HSD tile-online backward requires B1 and an exact allowlisted "
            f"profile; requested={profile!r}, got "
            f"B{batch}/H{geometry[0]}/K{geometry[1]}/"
            f"V{geometry[2]}/R{geometry[3]}."
        )
    value_dim = geometry[2]
    route_dim = geometry[3]
    if text_len <= 0:
        raise ValueError("text length must be positive")
    if router_q.shape != (1, text_len, heads, route_dim):
        raise ValueError(f"unexpected router_q shape: {router_q.shape}")
    if states.shape != (1, summary_count, heads, key_dim, value_dim):
        raise ValueError(f"unexpected states shape: {states.shape}")
    if summary_k.shape != (1, summary_count, heads, route_dim):
        raise ValueError(f"unexpected summary_k shape: {summary_k.shape}")
    if dout.shape != (1, text_len, heads, value_dim):
        raise ValueError(f"unexpected dout shape: {dout.shape}")
    if history_accumulator.shape != dout.shape:
        raise ValueError(
            "history_accumulator shape "
            f"{history_accumulator.shape} must equal dout shape {dout.shape}"
        )
    if boundary_counts.shape != (summary_count,):
        raise ValueError(
            "boundary_counts must have one entry per summary; got "
            f"{boundary_counts.shape} for M={summary_count}"
        )
    if token_summary_start.shape != (text_len,):
        raise ValueError(
            f"token_summary_start must have shape ({text_len},)"
        )
    if token_summary_end.shape != (text_len,):
        raise ValueError(f"token_summary_end must have shape ({text_len},)")
    if lse.shape != (text_len, heads):
        raise ValueError(f"lse must have shape ({text_len}, {heads})")
    if p_null.shape != (text_len, heads):
        raise ValueError(f"p_null must have shape ({text_len}, {heads})")
    if dlocal_prob is not None and dlocal_prob.numel() > 0:
        if dlocal_prob.shape != (1, text_len, heads, 1):
            raise ValueError(
                "dlocal_prob must have shape "
                f"(1, {text_len}, {heads}, 1), got {dlocal_prob.shape}"
            )
        if dlocal_prob.dtype != torch.float32:
            raise TypeError(
                "dlocal_prob must be torch.float32, got "
                f"{dlocal_prob.dtype}"
            )
    if has_null:
        if null_logits is None or null_logits.shape != (
            1,
            text_len,
            heads,
            1,
        ):
            raise ValueError(
                "has_null=True requires null_logits with shape "
                f"(1, {text_len}, {heads}, 1)"
            )

    bf16_inputs = {
        "read_q": read_q,
        "router_q": router_q,
        "summary_k": summary_k,
        "dout": dout,
    }
    if has_null and null_logits is not None:
        bf16_inputs["null_logits"] = null_logits
    for name, tensor in bf16_inputs.items():
        if tensor.dtype != torch.bfloat16:
            raise TypeError(
                f"{name} must be torch.bfloat16, got {tensor.dtype}"
            )
    if states.dtype != torch.float32:
        raise TypeError(f"states must be torch.float32, got {states.dtype}")
    if history_accumulator.dtype != torch.float32:
        raise TypeError(
            "history_accumulator must be torch.float32, got "
            f"{history_accumulator.dtype}"
        )
    if lse.dtype != torch.float32 or p_null.dtype != torch.float32:
        raise TypeError("lse and p_null must be torch.float32")
    for name, tensor in {
        "boundary_counts": boundary_counts,
        "token_summary_start": token_summary_start,
        "token_summary_end": token_summary_end,
    }.items():
        if tensor.dtype != torch.int64:
            raise TypeError(f"{name} must be torch.int64, got {tensor.dtype}")

    device = read_q.device
    tensors = {
        "router_q": router_q,
        "states": states,
        "summary_k": summary_k,
        "boundary_counts": boundary_counts,
        "token_summary_start": token_summary_start,
        "token_summary_end": token_summary_end,
        "lse": lse,
        "p_null": p_null,
        "dout": dout,
        "history_accumulator": history_accumulator,
    }
    if null_logits is not None:
        tensors["null_logits"] = null_logits
    if dlocal_prob is not None:
        tensors["dlocal_prob"] = dlocal_prob
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(
                f"{name} is on {tensor.device}, expected {device}"
            )

    if not math.isfinite(float(route_scale)):
        raise ValueError(f"route_scale must be finite, got {route_scale}")
    if not math.isfinite(float(read_scale)):
        raise ValueError(f"read_scale must be finite, got {read_scale}")
    return (
        int(text_len),
        int(summary_count),
        int(heads),
        int(key_dim),
        int(value_dim),
        int(route_dim),
        resolved_profile,
    )


def hsd_tile_online_backward(
    *,
    dout: torch.Tensor,
    history_accumulator: torch.Tensor,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    token_summary_start: torch.Tensor,
    token_summary_end: torch.Tensor,
    lse: torch.Tensor,
    p_null: torch.Tensor,
    dlocal_prob: torch.Tensor | None,
    null_logits: torch.Tensor | None,
    has_null: bool,
    route_scale: float,
    read_scale: float,
    needs: Sequence[bool],
    profile: str | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run an allowlisted low-memory online-read backward specialization.

    ``needs`` may be either five direct flags in return order
    ``(read_q, router_q, states, summary_k, null_logits)`` or the complete
    ``ctx.needs_input_grad`` tuple from
    ``_HSDOnlineGroupedGlobalReadFunction``.

    Returns gradients in the same five-field order.  This function never
    constructs routing probabilities or route-logit gradients with shape
    ``[T, H, M]``.
    """

    (
        text_len,
        summary_count,
        heads,
        key_dim,
        value_dim,
        route_dim,
        resolved_profile,
    ) = _validate_inputs(
        dout=dout,
        history_accumulator=history_accumulator,
        read_q=read_q,
        router_q=router_q,
        states=states,
        summary_k=summary_k,
        boundary_counts=boundary_counts,
        token_summary_start=token_summary_start,
        token_summary_end=token_summary_end,
        lse=lse,
        p_null=p_null,
        dlocal_prob=dlocal_prob,
        null_logits=null_logits,
        has_null=bool(has_null),
        route_scale=float(route_scale),
        read_scale=float(read_scale),
        profile=profile,
    )
    need_dread, need_drouter, need_dstates, need_dsummary_k, need_dnull = (
        _decode_needs(needs)
    )
    need_dnull = bool(need_dnull and has_null)

    if not (
        need_dread
        or need_drouter
        or need_dstates
        or need_dsummary_k
        or need_dnull
    ):
        return None, None, None, None, None

    # Contiguous copies are at most input-sized; unlike prob/dz they do not
    # introduce any T*H*M workspace.  Normal HSD callers already satisfy this.
    read_q = read_q.contiguous()
    router_q = router_q.contiguous()
    states = states.contiguous()
    summary_k = summary_k.contiguous()
    boundary_counts = boundary_counts.contiguous()
    token_summary_start = token_summary_start.contiguous()
    token_summary_end = token_summary_end.contiguous()
    lse = lse.contiguous()
    p_null = p_null.contiguous()
    dout = dout.contiguous()
    history_accumulator = history_accumulator.contiguous()

    dread = torch.zeros_like(read_q) if need_dread else None
    drouter = torch.zeros_like(router_q) if need_drouter else None
    dstates = torch.zeros_like(states) if need_dstates else None
    dsummary_k = torch.zeros_like(summary_k) if need_dsummary_k else None
    dnull = (
        torch.zeros_like(null_logits)
        if need_dnull and null_logits is not None
        else None
    )

    # With no summary candidate the output is independent of all four main
    # inputs.  If null is the sole candidate, its probability is exactly one,
    # so its softmax-logit gradient is also zero.
    if summary_count == 0:
        return dread, drouter, dstates, dsummary_k, dnull

    (
        block_t,
        block_k,
        block_v,
        num_warps,
        fuse_route_requested,
        fuse_dread_route_requested,
        route_block_t,
        route_block_v,
        route_num_warps,
    ) = _online_backward_launch_spec(resolved_profile)
    if (
        not _is_power_of_two(block_t)
        or not _is_power_of_two(block_k)
        or not _is_power_of_two(block_v)
        or key_dim % block_k != 0
        or value_dim % block_v != 0
    ):
        raise ValueError(
            "online backward tiles must be powers of two and divide the "
            f"selected profile K/V; profile={resolved_profile}, "
            f"K={key_dim}, V={value_dim}, "
            f"BT={block_t}, BK={block_k}, BV={block_v}"
        )
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError(
            "HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS must be one of "
            f"1, 2, 4, or 8; got {num_warps}"
        )
    fuse_dread_route = bool(
        fuse_dread_route_requested
        and need_dread
        and need_drouter
        and need_dsummary_k
    )
    fused_dread_route_tile_is_allowlisted = (
        (resolved_profile == "lc50b_h16_k96_v96_r64" and block_k == 32)
        or (
            resolved_profile
            in {
                "lc50b_h6_k192_v384_r64",
                "olmo3_h16_k128_v128_r128",
                "qwen35_4b_tp2_h16_k128_v128_r64",
            }
            and block_k == 64
        )
    )
    if fuse_dread_route and not fused_dread_route_tile_is_allowlisted:
        raise ValueError(
            "fused online dRead/route backward requires the exact BK "
            "allowlisted for the selected profile; "
            f"profile={resolved_profile}, "
            f"K={key_dim}, BK={block_k}"
        )
    fuse_route_grads = bool(
        fuse_route_requested
        and need_drouter
        and need_dsummary_k
        and not fuse_dread_route
    )
    if (
        not _is_power_of_two(route_block_t)
        or not _is_power_of_two(route_block_v)
        or value_dim % route_block_v != 0
        or route_num_warps not in {1, 2, 4, 8}
    ):
        raise ValueError(
            "fused route-gradient tiles require power-of-two BT/BV, BV "
            "dividing the selected profile V, and 1/2/4/8 warps; got "
            f"profile={resolved_profile}, V={value_dim}, "
            f"BT={route_block_t}, BV={route_block_v}, "
            f"warps={route_num_warps}"
        )
    # The fused kernel atomically reduces only into this compact M*H*R FP32
    # tensor.  When fusion is disabled the dRouter prim_func retains one
    # stable signature, but the uninitialized tensor is compile-time unused.
    dsummary_k_acc = (
        torch.zeros(
            summary_k.shape,
            device=summary_k.device,
            dtype=torch.float32,
        )
        if fuse_route_grads or fuse_dread_route
        else torch.empty(
            summary_k.shape,
            device=summary_k.device,
            dtype=torch.float32,
        )
        if need_drouter
        else None
    )
    drouter_partial = (
        torch.empty(
            (key_dim // block_k, text_len, heads, route_dim),
            device=router_q.device,
            dtype=torch.float32,
        )
        if fuse_dread_route
        else None
    )

    need_delta = need_drouter or need_dsummary_k or need_dnull
    delta = (
        torch.empty(
            (text_len, heads),
            device=read_q.device,
            dtype=torch.float32,
        )
        if need_delta
        else None
    )
    if need_delta:
        has_dlocal = bool(
            dlocal_prob is not None and dlocal_prob.numel() > 0
        )
        dlocal_arg = (
            dlocal_prob.contiguous().view(text_len, heads)
            if has_dlocal
            # The compact delta kernel keeps a fixed [T,H] signature.  This
            # tensor is never read when the compile-time flag is false.
            else p_null
        )
        assert _online_bwd_delta_kernel is not None
        null_grad_arg = (
            dnull.view(text_len, heads)
            if dnull is not None
            # Compile-time NEED_DNULL=False makes this pointer unused.
            else delta
        )
        with torch.cuda.device(read_q.device):
            _online_bwd_delta_kernel[
                (triton.cdiv(text_len, block_t), heads)
            ](
                dout,
                history_accumulator,
                p_null,
                dlocal_arg,
                delta,
                null_grad_arg,
                T_LEN=text_len,
                H=heads,
                V=value_dim,
                HAS_NULL=bool(has_null),
                HAS_DLOCAL=has_dlocal,
                NEED_DNULL=bool(need_dnull),
                BLOCK_T=block_t,
                BLOCK_V=block_v,
                NUM_V_BLOCKS=triton.cdiv(value_dim, block_v),
                num_warps=4,
                num_stages=1,
            )

    with torch.cuda.device(read_q.device):
        if need_dread:
            if fuse_dread_route:
                assert delta is not None
                assert drouter is not None
                assert drouter_partial is not None
                assert dsummary_k is not None
                assert dsummary_k_acc is not None
                kernel = _build_online_dread_route_kernel(
                    heads,
                    key_dim,
                    value_dim,
                    route_dim,
                    summary_count,
                    block_t,
                    block_k,
                    block_v,
                    num_warps=num_warps,
                )
                kernel(
                    read_q,
                    router_q,
                    states,
                    summary_k,
                    boundary_counts,
                    token_summary_start,
                    token_summary_end,
                    lse,
                    delta,
                    dout,
                    dread,
                    drouter_partial,
                    dsummary_k_acc,
                    float(route_scale),
                    float(read_scale),
                )

                # Every allowlisted K tile owns one exact route partial.
                # Reducing all compact [T,H,R] slabs recovers dRouterQ
                # without a third state scan (three slabs for LC50B H6 and
                # H16/K96, two for OLMo3-1B).
                assert (
                    _online_bwd_reduce_drouter_partial_kernel is not None
                )
                drouter_numel = drouter.numel()
                _online_bwd_reduce_drouter_partial_kernel[
                    (triton.cdiv(drouter_numel, 256),)
                ](
                    drouter_partial,
                    drouter,
                    N=key_dim // block_k,
                    NUMEL=drouter_numel,
                    BLOCK=256,
                    num_warps=4,
                    num_stages=1,
                )

                assert _online_bwd_cast_compact_kernel is not None
                dsummary_numel = dsummary_k_acc.numel()
                _online_bwd_cast_compact_kernel[
                    (triton.cdiv(dsummary_numel, 256),)
                ](
                    dsummary_k_acc,
                    dsummary_k,
                    N=dsummary_numel,
                    BLOCK=256,
                    num_warps=4,
                    num_stages=1,
                )
            else:
                kernel = _build_online_dread_kernel(
                    heads,
                    key_dim,
                    value_dim,
                    route_dim,
                    summary_count,
                    block_t,
                    block_k,
                    block_v,
                    num_warps=num_warps,
                )
                kernel(
                    router_q,
                    states,
                    summary_k,
                    boundary_counts,
                    token_summary_start,
                    token_summary_end,
                    lse,
                    dout,
                    dread,
                    float(route_scale),
                    float(read_scale),
                )

        if need_dstates:
            kernel = _build_online_dstates_kernel(
                heads,
                key_dim,
                value_dim,
                route_dim,
                summary_count,
                block_t,
                block_k,
                block_v,
                num_warps=num_warps,
            )
            kernel(
                read_q,
                router_q,
                summary_k,
                boundary_counts,
                token_summary_start,
                token_summary_end,
                lse,
                dout,
                dstates,
                float(route_scale),
                float(read_scale),
            )

        if need_drouter and not fuse_dread_route:
            assert delta is not None
            assert dsummary_k_acc is not None
            kernel = _build_online_drouter_kernel(
                heads,
                key_dim,
                value_dim,
                route_dim,
                summary_count,
                route_block_t if fuse_route_grads else block_t,
                route_block_v if fuse_route_grads else block_v,
                fuse_route_grads,
                num_warps=(
                    route_num_warps
                    if fuse_route_grads
                    else num_warps
                ),
            )
            kernel(
                read_q,
                router_q,
                states,
                summary_k,
                boundary_counts,
                token_summary_start,
                token_summary_end,
                lse,
                delta,
                dout,
                drouter,
                dsummary_k_acc,
                float(route_scale),
                float(read_scale),
            )
            if fuse_route_grads:
                assert dsummary_k is not None
                assert _online_bwd_cast_compact_kernel is not None
                numel = dsummary_k_acc.numel()
                _online_bwd_cast_compact_kernel[
                    (triton.cdiv(numel, 256),)
                ](
                    dsummary_k_acc,
                    dsummary_k,
                    N=numel,
                    BLOCK=256,
                    num_warps=4,
                    num_stages=1,
                )

        if (
            need_dsummary_k
            and not fuse_route_grads
            and not fuse_dread_route
        ):
            assert delta is not None
            kernel = _build_online_dsummary_k_kernel(
                heads,
                key_dim,
                value_dim,
                route_dim,
                summary_count,
                block_t,
                block_v,
                num_warps=num_warps,
            )
            kernel(
                read_q,
                router_q,
                states,
                summary_k,
                boundary_counts,
                token_summary_start,
                token_summary_end,
                lse,
                delta,
                dout,
                dsummary_k,
                float(route_scale),
                float(read_scale),
            )

    return dread, drouter, dstates, dsummary_k, dnull


__all__ = ["hsd_tile_online_backward"]
