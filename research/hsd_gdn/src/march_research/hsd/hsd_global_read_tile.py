import os
from typing import Callable, Optional

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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


_TILE_ONLINE_PROFILE_GEOMETRIES = {
    "lc50b_h6_k192_v384_r64": (6, 192, 384, 64),
    "lc50b_h16_k96_v96_r64": (16, 96, 96, 64),
    "olmo3_h16_k128_v128_r128": (16, 128, 128, 128),
    "qwen35_4b_tp2_h16_k128_v128_r64": (16, 128, 128, 64),
}
_TILE_ONLINE_PROFILE_ALIASES = {
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

# Production route monitoring is deliberately a separate contract from the
# training kernel. ``uniform_adjacent_pairs_v1`` selects uniformly spaced
# two-token runs with integer arithmetic. It includes both sequence endpoints
# when S > 1 and, unlike isolated uniform samples, provides real adjacent
# token pairs for the route-switch metric. Each selected block is reduced to
# scalars before the next block is visited; no [T,H,M] or [sampled_T,H,M]
# tensor survives the block.
_ONLINE_DEBUG_SAMPLING_POLICY = "uniform_adjacent_pairs_v1"
_ONLINE_DEBUG_SAMPLING_POLICY_ID = 1
_ONLINE_DEBUG_CONTRACT_VERSION = 1
_ONLINE_DEBUG_DEFAULT_SAMPLE_TOKENS = 4096
_ONLINE_DEBUG_DEFAULT_BLOCK_TOKENS = 256


def _backend_name(backend: str | None = None) -> str:
    if backend is not None:
        return str(backend).strip().lower()
    return os.environ.get("HSD_FUSED_GLOBAL_READ_BACKEND", "").strip().lower()


def _canonical_tile_online_profile(profile: str | None) -> str | None:
    if profile is None:
        profile = os.environ.get("HSD_FUSED_GLOBAL_READ_PROFILE", "")
    normalized = str(profile).strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return None
    normalized = _TILE_ONLINE_PROFILE_ALIASES.get(normalized, normalized)
    if normalized not in _TILE_ONLINE_PROFILE_GEOMETRIES:
        supported = ", ".join(sorted(_TILE_ONLINE_PROFILE_GEOMETRIES))
        raise ValueError(
            f"Unknown HSD tile-online profile {profile!r}; supported profiles: {supported}."
        )
    return normalized


def _resolve_tile_online_profile(
    profile: str | None,
    *,
    heads: int,
    key_dim: int,
    value_dim: int,
    route_dim: int,
) -> str | None:
    """Resolve an exact, allowlisted Hopper geometry.

    An omitted profile preserves the legacy HF API by inferring the profile
    from tensor geometry.  An explicit profile must match exactly; it never
    widens a kernel guard or falls back to another specialization.
    """

    requested = _canonical_tile_online_profile(profile)
    actual = (int(heads), int(key_dim), int(value_dim), int(route_dim))
    if requested is not None:
        return (
            requested
            if _TILE_ONLINE_PROFILE_GEOMETRIES[requested] == actual
            else None
        )
    matches = [
        name
        for name, geometry in _TILE_ONLINE_PROFILE_GEOMETRIES.items()
        if geometry == actual
    ]
    return matches[0] if len(matches) == 1 else None


def _online_grouped_enabled(backend: str | None = None) -> bool:
    return _backend_name(backend) in {
        "tile_online",
        "tile-online",
        "tile_grouped_online",
        "flashqla_tile",
    }


def _online_forward_launch_spec(profile: str | None) -> tuple[int, int, int, int]:
    """Resolve (BT, group-M, BV, warps) without env drift for profiles."""
    # Public and private callers may supply a documented alias.  Resolve it
    # before selecting an immutable LaunchSpec so aliases cannot accidentally
    # take the legacy environment-controlled path or another profile's tiles.
    if profile is not None:
        profile = _canonical_tile_online_profile(profile)
    if profile == "lc50b_h16_k96_v96_r64":
        # The Transformer-aligned HSD geometry has V=96.  Use three exact
        # tensor-core value tiles instead of widening/padding the persistent
        # state to the BV=128 launch used by the existing production shapes.
        return 64, 4, 32, 4
    if profile is not None:
        return 64, 4, 128, 4
    return (
        max(16, _env_int("HSD_FUSED_GLOBAL_READ_ONLINE_BT", 64)),
        max(1, _env_int("HSD_FUSED_GLOBAL_READ_ONLINE_GROUP_M", 4)),
        max(16, _env_int("HSD_FUSED_GLOBAL_READ_ONLINE_BV", 128)),
        int(_env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4)),
    )


def _online_summary_bucket_count(summary_count: int, block_m: int) -> int:
    """Return a stable, bounded family of M specializations for online kernels.

    TileLang 0.1.9 specializes the online forward and backward builders on
    ``M``.  Packed training produces a different number of real summaries in
    many batches, which otherwise causes several new JIT compilations per
    optimizer step.  Small validation cases use power-of-two buckets; larger
    production shapes use 32-summary buckets.  For Qwen's MBS4/T=131072
    contract this limits M to at most five compiled shapes while adding fewer
    than 32 unreachable entries, instead of padding every batch to 256.
    Padded entries receive an unreachable boundary and therefore are
    mathematically absent from both the online softmax and its backward.
    """

    summary_count = int(summary_count)
    block_m = max(1, int(block_m))
    if summary_count <= 0:
        return summary_count
    if summary_count <= 32:
        return _padded_summary_count(
            max(block_m, _next_power_of_2(summary_count)),
            block_m,
        )
    production_quantum = _padded_summary_count(32, block_m)
    return _padded_summary_count(summary_count, production_quantum)


def _tile_speed_enabled() -> bool:
    backend = _backend_name()
    if backend in {"tile_speed", "tilespeed"}:
        return True
    if backend in {"tile_lowmem", "tile-lowmem", "tile_low_memory", "tile"}:
        return False
    return _env_bool("HSD_FUSED_GLOBAL_READ_TILE_SPEED", False)


def _tile_speed_strict() -> bool:
    return _backend_name() in {"tile_speed", "tilespeed"} or _env_bool(
        "HSD_FUSED_GLOBAL_READ_TILE_SPEED_STRICT", False
    )


def _tile_speed_forward_enabled() -> bool:
    if not _tile_speed_enabled():
        return False
    raw = os.environ.get("HSD_FUSED_GLOBAL_READ_TILE_SPEED_FORWARD", "").strip()
    if raw:
        return _env_bool("HSD_FUSED_GLOBAL_READ_TILE_SPEED_FORWARD", False)
    return _backend_name() in {"tile_speed", "tilespeed"}


def _tile_speed_backward_enabled() -> bool:
    return _tile_speed_enabled() and _env_bool("HSD_FUSED_GLOBAL_READ_TILE_SPEED_BACKWARD", True)


def _supports_tile_global_read(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: torch.Tensor | None,
) -> bool:
    if not read_q.is_cuda:
        return False
    if read_q.ndim != 4 or router_q.ndim != 4:
        return False
    if states.ndim != 5 or summary_k.ndim != 4:
        return False
    batch, text_len, heads, state_key_dim = read_q.shape
    route_batch, route_text_len, route_heads, route_key_dim = router_q.shape
    summary_count = int(states.shape[1])
    if batch != 1 or text_len <= 0:
        return False
    if (route_batch, route_text_len, route_heads) != (batch, text_len, heads):
        return False
    if states.shape[:4] != (batch, summary_count, heads, state_key_dim):
        return False
    if summary_k.shape != (batch, summary_count, heads, route_key_dim):
        return False
    if states.device != read_q.device or router_q.device != read_q.device or summary_k.device != read_q.device:
        return False
    if null_logits is not None and (null_logits.device != read_q.device or tuple(null_logits.shape) != (batch, text_len, heads, 1)):
        return False
    if text_cu_seqlens.ndim != 1 or summary_cu_seqlens.ndim != 1:
        return False
    if int(text_cu_seqlens.numel()) != int(summary_cu_seqlens.numel()) or int(text_cu_seqlens.numel()) < 2:
        return False
    if boundary_counts.ndim != 1 or int(boundary_counts.numel()) != summary_count:
        return False
    if state_key_dim > 256 or route_key_dim > 256:
        return False
    return True


def _packed_valid_mask(
    *,
    text_len: int,
    summary_count: int,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    valid = torch.zeros(text_len, summary_count, device=device, dtype=torch.bool)
    if summary_count <= 0:
        return valid

    text_cu_cpu = text_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    summary_cu_cpu = summary_cu_seqlens.detach().to(device="cpu", dtype=torch.long)
    boundary_counts = boundary_counts.to(device=device, dtype=torch.long)
    for doc_idx in range(int(text_cu_cpu.numel()) - 1):
        text_start = int(text_cu_cpu[doc_idx].item())
        text_end = int(text_cu_cpu[doc_idx + 1].item())
        summary_start = int(summary_cu_cpu[doc_idx].item())
        summary_end = int(summary_cu_cpu[doc_idx + 1].item())
        if text_end <= text_start or summary_end <= summary_start:
            continue
        token_index = torch.arange(text_start, text_end, device=device, dtype=torch.long)
        boundaries = boundary_counts[summary_start:summary_end]
        valid[text_start:text_end, summary_start:summary_end] = boundaries.view(1, -1) <= token_index.view(-1, 1)
    return valid


def _route_probabilities(
    *,
    router_q: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: torch.Tensor | None,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    _, text_len, heads, _ = router_q.shape
    summary_count = int(summary_k.shape[1])
    device = router_q.device

    if summary_count <= 0:
        prob = torch.zeros(text_len, heads, 0, device=device, dtype=torch.float32)
        lse = torch.zeros(text_len, heads, device=device, dtype=torch.float32)
        if null_logits is None:
            return prob, lse, None
        p_null = torch.ones(text_len, heads, device=device, dtype=torch.float32)
        return prob, null_logits[0, ..., 0].float(), p_null

    valid = _packed_valid_mask(
        text_len=text_len,
        summary_count=summary_count,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu_seqlens,
        summary_cu_seqlens=summary_cu_seqlens,
        device=device,
    )
    logits = torch.einsum("thk,mhk->thm", router_q[0].float(), summary_k[0].float()) * float(route_scale)
    logits = logits.masked_fill(~valid[:, None, :], -torch.inf)

    if null_logits is not None:
        null_value = null_logits[0, ..., 0].float()
        max_logit = torch.maximum(logits.amax(dim=-1), null_value)
        max_safe = torch.where(torch.isfinite(max_logit), max_logit, torch.zeros_like(max_logit))
        summary_num = torch.where(valid[:, None, :], torch.exp(logits - max_safe[..., None]), torch.zeros_like(logits))
        null_num = torch.exp(null_value - max_safe)
        denom = summary_num.sum(dim=-1) + null_num
        prob = summary_num / denom[..., None]
        p_null = null_num / denom
        lse = max_safe + torch.log(denom)
        return prob, lse, p_null

    has_valid = valid.any(dim=-1)[:, None]
    max_logit = logits.amax(dim=-1)
    max_safe = torch.where(has_valid, max_logit, torch.zeros_like(max_logit))
    summary_num = torch.where(valid[:, None, :], torch.exp(logits - max_safe[..., None]), torch.zeros_like(logits))
    denom = summary_num.sum(dim=-1)
    prob = torch.where(denom[..., None] > 0.0, summary_num / denom.clamp_min(1e-20)[..., None], torch.zeros_like(summary_num))
    lse = torch.where(denom > 0.0, max_safe + torch.log(denom), torch.zeros_like(denom))
    return prob, lse, None


def _tile_block_m(text_len: int, heads: int, value_dim: int, dtype: torch.dtype, *, backward: bool) -> int:
    default = 2 if backward else 4
    block_m = max(1, _env_int("HSD_FUSED_GLOBAL_READ_TILE_BM", default))
    # Keep the largest temporary [T,H,BM,V] below roughly 1.25 GiB by default.
    bytes_per = torch.tensor([], dtype=torch.float32).element_size()
    max_tmp_bytes = max(1, _env_int("HSD_FUSED_GLOBAL_READ_TILE_MAX_TMP_MB", 1280)) * 1024 * 1024
    max_block_m = max(1, max_tmp_bytes // max(1, text_len * heads * value_dim * bytes_per))
    return int(max(1, min(block_m, max_block_m)))


def _tile_forward_impl(
    *,
    read_q: torch.Tensor,
    states: torch.Tensor,
    prob: torch.Tensor,
    read_scale: float,
) -> torch.Tensor:
    _, text_len, heads, _ = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    out_acc = torch.zeros(text_len, heads, value_dim, device=read_q.device, dtype=torch.float32)
    if summary_count <= 0:
        return out_acc.to(read_q.dtype).unsqueeze(0)

    read_q_f = read_q[0].float()
    block_m = _tile_block_m(text_len, heads, value_dim, read_q.dtype, backward=False)
    for summary_start in range(0, summary_count, block_m):
        summary_end = min(summary_start + block_m, summary_count)
        state_block = states[0, summary_start:summary_end].float()
        read_block = torch.einsum("thk,bhkv->thbv", read_q_f, state_block)
        out_acc.add_(
            torch.einsum(
                "thb,thbv->thv",
                prob[:, :, summary_start:summary_end],
                read_block,
            ),
            alpha=float(read_scale),
        )
    return out_acc.to(read_q.dtype).unsqueeze(0)


if tilelang is not None:

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _build_read_out_tilelang_kernel(H, K, V, M, BT, BV, dtype_str, num_warps=4):
        dtype_map = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}
        _dtype = dtype_map[dtype_str]
        _H, _K_orig, _V_orig, _M = H, K, V, M
        _BT, _BV = BT, BV
        _K = (K + 15) // 16 * 16
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        q_s = (1, T_d, _H, _K_orig)
        states_s = (1, _M, _H, _K_orig, _V_orig)
        prob_s = (T_d, _H, _M)
        out_s = (1, T_d, _H, _V_orig)

        @T.prim_func
        def kernel(
            read_q: T.Tensor(q_s, _dtype),
            states: T.Tensor(states_s, T.float32),
            prob: T.Tensor(prob_s, T.float32),
            out: T.Tensor(out_s, _dtype),
            read_scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(_V_orig, _BV), T.ceildiv(T_d, _BT), _H, threads=_threads) as (i_v, i_t, i_h):
                t_s = i_t * _BT
                v_s = i_v * _BV
                q_shared = T.alloc_shared((_BT, _K), _dtype)
                state_shared = T.alloc_shared((_K, _BV), _dtype)
                read_acc = T.alloc_fragment((_BT, _BV), T.float32)
                out_acc = T.alloc_fragment((_BT, _BV), T.float32)
                T.clear(out_acc)
                T.copy(read_q[0, t_s:t_s + _BT, i_h, :], q_shared)

                for i_m in T.Pipelined(_M, num_stages=1):
                    T.copy(states[0, i_m, i_h, :, v_s:v_s + _BV], state_shared)
                    T.clear(read_acc)
                    T.gemm(q_shared, state_shared, read_acc, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(_BT, _BV):
                        if (t_s + i < T_d) & (v_s + j < _V_orig):
                            out_acc[i, j] += read_acc[i, j] * prob[t_s + i, i_h, i_m] * read_scale

                out_shared = T.alloc_shared((_BT, _BV), _dtype)
                out_cast = T.alloc_fragment((_BT, _BV), _dtype)
                for i, j in T.Parallel(_BT, _BV):
                    out_cast[i, j] = T.cast(out_acc[i, j], _dtype)
                T.copy(out_cast, out_shared)
                for i, j in T.Parallel(_BT, _BV):
                    if (t_s + i < T_d) & (v_s + j < _V_orig):
                        out[0, t_s + i, i_h, v_s + j] = out_shared[i, j]

        return kernel

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _build_dread_tilelang_kernel(H, K, V, M, BT, BK, BV, dtype_str, num_warps=4):
        dtype_map = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}
        _dtype = dtype_map[dtype_str]
        _H, _K_orig, _V_orig, _M = H, K, V, M
        _BT, _BK, _BV = BT, BK, BV
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        dout_s = (1, T_d, _H, _V_orig)
        states_s = (1, _M, _H, _K_orig, _V_orig)
        prob_s = (T_d, _H, _M)
        dread_s = (1, T_d, _H, _K_orig)

        @T.prim_func
        def kernel(
            dout: T.Tensor(dout_s, _dtype),
            states: T.Tensor(states_s, T.float32),
            prob: T.Tensor(prob_s, T.float32),
            dread: T.Tensor(dread_s, _dtype),
            read_scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(_K_orig, _BK), T.ceildiv(T_d, _BT), _H, threads=_threads) as (i_k, i_t, i_h):
                t_s = i_t * _BT
                k_s = i_k * _BK
                do_shared = T.alloc_shared((_BT, _BV), _dtype)
                weighted_shared = T.alloc_shared((_BT, _BV), _dtype)
                state_shared = T.alloc_shared((_BK, _BV), _dtype)
                acc = T.alloc_fragment((_BT, _BK), T.float32)
                T.clear(acc)

                for i_v in T.Pipelined(T.ceildiv(_V_orig, _BV), num_stages=1):
                    v_s = i_v * _BV
                    T.copy(dout[0, t_s:t_s + _BT, i_h, v_s:v_s + _BV], do_shared)
                    for i_m in T.serial(_M):
                        T.copy(states[0, i_m, i_h, k_s:k_s + _BK, v_s:v_s + _BV], state_shared)
                        for i, j in T.Parallel(_BT, _BV):
                            if (t_s + i < T_d) & (v_s + j < _V_orig):
                                weighted_shared[i, j] = T.cast(
                                    T.cast(do_shared[i, j], T.float32) * prob[t_s + i, i_h, i_m] * read_scale,
                                    _dtype,
                                )
                        T.gemm(weighted_shared, state_shared, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                out_shared = T.alloc_shared((_BT, _BK), _dtype)
                out_cast = T.alloc_fragment((_BT, _BK), _dtype)
                for i, j in T.Parallel(_BT, _BK):
                    out_cast[i, j] = T.cast(acc[i, j], _dtype)
                T.copy(out_cast, out_shared)
                for i, j in T.Parallel(_BT, _BK):
                    if (t_s + i < T_d) & (k_s + j < _K_orig):
                        dread[0, t_s + i, i_h, k_s + j] = out_shared[i, j]

        return kernel

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _build_dstates_tilelang_kernel(H, K, V, M, BT, BK, BV, dtype_str, num_warps=4):
        dtype_map = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}
        _dtype = dtype_map[dtype_str]
        _H, _K_orig, _V_orig, _M = H, K, V, M
        _BT, _BK, _BV = BT, BK, BV
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        q_s = (1, T_d, _H, _K_orig)
        dout_s = (1, T_d, _H, _V_orig)
        prob_s = (T_d, _H, _M)
        dstates_s = (1, _M, _H, _K_orig, _V_orig)

        @T.prim_func
        def kernel(
            read_q: T.Tensor(q_s, _dtype),
            dout: T.Tensor(dout_s, _dtype),
            prob: T.Tensor(prob_s, T.float32),
            dstates: T.Tensor(dstates_s, T.float32),
            read_scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(_V_orig, _BV), T.ceildiv(_K_orig, _BK), _M * _H, threads=_threads) as (i_v, i_k, i_mh):
                i_m = i_mh // _H
                i_h = i_mh - i_m * _H
                k_s = i_k * _BK
                v_s = i_v * _BV
                q_shared = T.alloc_shared((_BT, _BK), _dtype)
                do_shared = T.alloc_shared((_BT, _BV), _dtype)
                weighted_shared = T.alloc_shared((_BT, _BV), _dtype)
                acc = T.alloc_fragment((_BK, _BV), T.float32)
                T.clear(acc)

                for i_t in T.Pipelined(T.ceildiv(T_d, _BT), num_stages=1):
                    t_s = i_t * _BT
                    T.copy(read_q[0, t_s:t_s + _BT, i_h, k_s:k_s + _BK], q_shared)
                    T.copy(dout[0, t_s:t_s + _BT, i_h, v_s:v_s + _BV], do_shared)
                    for i, j in T.Parallel(_BT, _BV):
                        if (t_s + i < T_d) & (v_s + j < _V_orig):
                            weighted_shared[i, j] = T.cast(
                                T.cast(do_shared[i, j], T.float32) * prob[t_s + i, i_h, i_m] * read_scale,
                                _dtype,
                            )
                    T.gemm(q_shared, weighted_shared, acc, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                out_shared = T.alloc_shared((_BK, _BV), T.float32)
                T.copy(acc, out_shared)
                for i, j in T.Parallel(_BK, _BV):
                    if (k_s + i < _K_orig) & (v_s + j < _V_orig):
                        dstates[0, i_m, i_h, k_s + i, v_s + j] = out_shared[i, j]

        return kernel

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _build_read_out_speed_tilelang_kernel(H, K, V, M, BT, BM, BV, dtype_str, num_warps=4):
        dtype_map = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}
        _dtype = dtype_map[dtype_str]
        _H, _K_orig, _V_orig, _M = H, K, V, M
        _BT, _BM, _BV = BT, BM, BV
        _K = (K + 15) // 16 * 16
        _NMB = (M + BM - 1) // BM
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        q_s = (1, T_d, _H, _K_orig)
        states_s = (1, _M, _H, _K_orig, _V_orig)
        prob_s = (T_d, _H, _M)
        partial_s = (_NMB, T_d, _H, _V_orig)

        @T.prim_func
        def kernel(
            read_q: T.Tensor(q_s, _dtype),
            states: T.Tensor(states_s, T.float32),
            prob: T.Tensor(prob_s, T.float32),
            partial_out: T.Tensor(partial_s, T.float32),
            read_scale: T.float32,
        ):
            with T.Kernel(_NMB, T.ceildiv(_V_orig, _BV), T.ceildiv(T_d, _BT) * _H, threads=_threads) as (i_mb, i_v, i_th):
                i_t = i_th // _H
                i_h = i_th - i_t * _H
                t_s = i_t * _BT
                v_s = i_v * _BV
                m_s = i_mb * _BM
                q_shared = T.alloc_shared((_BT, _K), _dtype)
                state_shared = T.alloc_shared((_K, _BV), _dtype)
                read_acc = T.alloc_fragment((_BT, _BV), T.float32)
                out_acc = T.alloc_fragment((_BT, _BV), T.float32)
                T.clear(out_acc)
                T.copy(read_q[0, t_s:t_s + _BT, i_h, :], q_shared)

                for i_m_local in T.serial(_BM):
                    cur_m = m_s + i_m_local
                    T.copy(states[0, cur_m, i_h, :, v_s:v_s + _BV], state_shared)
                    T.clear(read_acc)
                    T.gemm(q_shared, state_shared, read_acc, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(_BT, _BV):
                        if (t_s + i < T_d) & (v_s + j < _V_orig) & (cur_m < _M):
                            out_acc[i, j] += read_acc[i, j] * prob[t_s + i, i_h, cur_m] * read_scale

                out_shared = T.alloc_shared((_BT, _BV), T.float32)
                T.copy(out_acc, out_shared)
                for i, j in T.Parallel(_BT, _BV):
                    if (t_s + i < T_d) & (v_s + j < _V_orig):
                        partial_out[i_mb, t_s + i, i_h, v_s + j] = out_shared[i, j]

        return kernel

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _build_dread_speed_tilelang_kernel(H, K, V, M, BT, BM, BK, BV, dtype_str, num_warps=4):
        dtype_map = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}
        _dtype = dtype_map[dtype_str]
        _H, _K_orig, _V_orig, _M = H, K, V, M
        _BT, _BM, _BK, _BV = BT, BM, BK, BV
        _NMB = (M + BM - 1) // BM
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        dout_s = (1, T_d, _H, _V_orig)
        states_s = (1, _M, _H, _K_orig, _V_orig)
        prob_s = (T_d, _H, _M)
        partial_s = (_NMB, T_d, _H, _K_orig)

        @T.prim_func
        def kernel(
            dout: T.Tensor(dout_s, _dtype),
            states: T.Tensor(states_s, T.float32),
            prob: T.Tensor(prob_s, T.float32),
            partial_dread: T.Tensor(partial_s, T.float32),
            read_scale: T.float32,
        ):
            with T.Kernel(_NMB, T.ceildiv(_K_orig, _BK), T.ceildiv(T_d, _BT) * _H, threads=_threads) as (i_mb, i_k, i_th):
                i_t = i_th // _H
                i_h = i_th - i_t * _H
                t_s = i_t * _BT
                k_s = i_k * _BK
                m_s = i_mb * _BM
                do_shared = T.alloc_shared((_BT, _BV), _dtype)
                weighted_shared = T.alloc_shared((_BT, _BV), _dtype)
                state_shared = T.alloc_shared((_BK, _BV), _dtype)
                acc = T.alloc_fragment((_BT, _BK), T.float32)
                T.clear(acc)

                for i_v in T.Pipelined(T.ceildiv(_V_orig, _BV), num_stages=1):
                    v_s = i_v * _BV
                    T.copy(dout[0, t_s:t_s + _BT, i_h, v_s:v_s + _BV], do_shared)
                    for i_m_local in T.serial(_BM):
                        cur_m = m_s + i_m_local
                        T.copy(states[0, cur_m, i_h, k_s:k_s + _BK, v_s:v_s + _BV], state_shared)
                        for i, j in T.Parallel(_BT, _BV):
                            weighted_shared[i, j] = T.cast(
                                T.if_then_else(
                                    (t_s + i < T_d) & (v_s + j < _V_orig) & (cur_m < _M),
                                    T.cast(do_shared[i, j], T.float32) * prob[t_s + i, i_h, cur_m] * read_scale,
                                    0.0,
                                ),
                                _dtype,
                            )
                        T.gemm(weighted_shared, state_shared, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                out_shared = T.alloc_shared((_BT, _BK), T.float32)
                T.copy(acc, out_shared)
                for i, j in T.Parallel(_BT, _BK):
                    if (t_s + i < T_d) & (k_s + j < _K_orig):
                        partial_dread[i_mb, t_s + i, i_h, k_s + j] = out_shared[i, j]

        return kernel

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _build_dstates_speed_tilelang_kernel(H, K, V, M, BT, BK, BV, NS, dtype_str, num_warps=4):
        dtype_map = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}
        _dtype = dtype_map[dtype_str]
        _H, _K_orig, _V_orig, _M, _NS = H, K, V, M, NS
        _BT, _BK, _BV = BT, BK, BV
        _NKB = (K + BK - 1) // BK
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        q_s = (1, T_d, _H, _K_orig)
        dout_s = (1, T_d, _H, _V_orig)
        prob_s = (T_d, _H, _M)
        partial_s = (_NS, _M, _H, _K_orig, _V_orig)

        @T.prim_func
        def kernel(
            read_q: T.Tensor(q_s, _dtype),
            dout: T.Tensor(dout_s, _dtype),
            prob: T.Tensor(prob_s, T.float32),
            partial_dstates: T.Tensor(partial_s, T.float32),
            read_scale: T.float32,
        ):
            with T.Kernel(_NS, T.ceildiv(_V_orig, _BV), _NKB * _M * _H, threads=_threads) as (i_split, i_v, i_kmh):
                i_k = i_kmh % _NKB
                i_mh = i_kmh // _NKB
                i_m = i_mh // _H
                i_h = i_mh - i_m * _H
                k_s = i_k * _BK
                v_s = i_v * _BV
                q_shared = T.alloc_shared((_BT, _BK), _dtype)
                do_shared = T.alloc_shared((_BT, _BV), _dtype)
                weighted_shared = T.alloc_shared((_BT, _BV), _dtype)
                acc = T.alloc_fragment((_BK, _BV), T.float32)
                T.clear(acc)

                num_t_blocks = T.ceildiv(T_d, _BT)
                blocks_per_split = T.ceildiv(num_t_blocks, _NS)
                t_block_start = i_split * blocks_per_split
                t_block_end = T.min((i_split + 1) * blocks_per_split, num_t_blocks)
                for i_rel in T.Pipelined(blocks_per_split, num_stages=1):
                    i_t = t_block_start + i_rel
                    t_s = i_t * _BT
                    if i_t < t_block_end:
                        T.copy(read_q[0, t_s:t_s + _BT, i_h, k_s:k_s + _BK], q_shared)
                        T.copy(dout[0, t_s:t_s + _BT, i_h, v_s:v_s + _BV], do_shared)
                        for i, j in T.Parallel(_BT, _BV):
                            weighted_shared[i, j] = T.cast(
                                T.if_then_else(
                                    (t_s + i < T_d) & (v_s + j < _V_orig) & (i_m < _M),
                                    T.cast(do_shared[i, j], T.float32) * prob[t_s + i, i_h, i_m] * read_scale,
                                    0.0,
                                ),
                                _dtype,
                            )
                        T.gemm(q_shared, weighted_shared, acc, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                out_shared = T.alloc_shared((_BK, _BV), T.float32)
                T.copy(acc, out_shared)
                for i, j in T.Parallel(_BK, _BV):
                    if (i_m < _M) & (k_s + i < _K_orig) & (v_s + j < _V_orig):
                        partial_dstates[i_split, i_m, i_h, k_s + i, v_s + j] = out_shared[i, j]

        return kernel

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _build_dz_tilelang_kernel(
        H,
        K,
        V,
        M,
        BT,
        BM,
        BK,
        BV,
        dtype_str,
        apply_softmax_jacobian=True,
        num_warps=4,
    ):
        dtype_map = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}
        _dtype = dtype_map[dtype_str]
        _H, _K_orig, _V_orig, _M = H, K, V, M
        _apply_softmax_jacobian = bool(apply_softmax_jacobian)
        _BT, _BM, _BK, _BV = BT, BM, BK, BV
        _K = (K + 15) // 16 * 16
        _RESIDUAL_K = _K if not _apply_softmax_jacobian else 1
        _RESIDUAL_V = _BV if not _apply_softmax_jacobian else 1
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        q_s = (1, T_d, _H, _K_orig)
        states_s = (1, _M, _H, _K_orig, _V_orig)
        dout_s = (1, T_d, _H, _V_orig)
        prob_s = (T_d, _H, _M)
        delta_s = (T_d, _H)
        dz_s = (T_d, _H, _M)

        @T.prim_func
        def kernel(
            read_q: T.Tensor(q_s, _dtype),
            states: T.Tensor(states_s, T.float32),
            dout: T.Tensor(dout_s, _dtype),
            prob: T.Tensor(prob_s, T.float32),
            delta: T.Tensor(delta_s, T.float32),
            dz: T.Tensor(dz_s, T.float32),
            read_scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(_M, _BM), T.ceildiv(T_d, _BT), _H, threads=_threads) as (i_m_block, i_t, i_h):
                t_s = i_t * _BT
                m_s = i_m_block * _BM
                q_shared = T.alloc_shared((_BT, _K), _dtype)
                do_shared = T.alloc_shared((_BT, _BV), _dtype)
                state_shared = T.alloc_shared((_K, _BV), _dtype)
                # External momentum weights need dL/dw itself, so preserve the
                # old tf32x3 kernel's FP32-state accuracy.  A BF16 high part
                # plus BF16 residual represents the FP32 state much more
                # accurately while retaining tensor-core GEMMs.  The ordinary
                # route-softmax dz path remains byte-for-byte on its original
                # single-GEMM implementation.
                state_residual_shared = T.alloc_shared((_RESIDUAL_K, _RESIDUAL_V), _dtype)
                read_acc = T.alloc_fragment((_BT, _BV), T.float32)
                prod = T.alloc_fragment((_BT, _BV), T.float32)
                dot_vec = T.alloc_fragment((_BT,), T.float32)
                dot_acc = T.alloc_fragment((_BT, _BM), T.float32)
                T.clear(dot_acc)
                T.copy(read_q[0, t_s:t_s + _BT, i_h, :], q_shared)

                for i_v in T.Pipelined(T.ceildiv(_V_orig, _BV), num_stages=1):
                    v_s = i_v * _BV
                    T.copy(dout[0, t_s:t_s + _BT, i_h, v_s:v_s + _BV], do_shared)
                    for i_m_local in T.serial(_BM):
                        cur_m = m_s + i_m_local
                        if _apply_softmax_jacobian:
                            T.copy(states[0, cur_m, i_h, :, v_s:v_s + _BV], state_shared)
                        else:
                            for i, j in T.Parallel(_K, _BV):
                                state_value = T.if_then_else(
                                    (i < _K_orig) & (v_s + j < _V_orig) & (cur_m < _M),
                                    states[0, cur_m, i_h, i, v_s + j],
                                    0.0,
                                )
                                state_hi = T.cast(state_value, _dtype)
                                state_shared[i, j] = state_hi
                                state_residual_shared[i, j] = T.cast(
                                    state_value - T.cast(state_hi, T.float32),
                                    _dtype,
                                )
                        T.clear(read_acc)
                        T.gemm(q_shared, state_shared, read_acc, policy=T.GemmWarpPolicy.FullRow)
                        if not _apply_softmax_jacobian:
                            T.gemm(q_shared, state_residual_shared, read_acc, policy=T.GemmWarpPolicy.FullRow)
                        for i, j in T.Parallel(_BT, _BV):
                            prod[i, j] = T.if_then_else(
                                (t_s + i < T_d) & (v_s + j < _V_orig) & (cur_m < _M),
                                read_acc[i, j] * T.cast(do_shared[i, j], T.float32),
                                0.0,
                            )
                        T.reduce_sum(prod, dot_vec, dim=1)
                        for i in T.Parallel(_BT):
                            if (t_s + i < T_d) & (cur_m < _M):
                                dot_acc[i, i_m_local] += dot_vec[i] * read_scale

                for i, j in T.Parallel(_BT, _BM):
                    cur_m = m_s + j
                    if (t_s + i < T_d) & (cur_m < _M):
                        if _apply_softmax_jacobian:
                            dz[t_s + i, i_h, cur_m] = (
                                prob[t_s + i, i_h, cur_m]
                                * (dot_acc[i, j] - delta[t_s + i, i_h])
                            )
                        else:
                            dz[t_s + i, i_h, cur_m] = dot_acc[i, j]

        return kernel

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _build_drouter_tilelang_kernel(H, K, M, BT, BM, BK, dtype_str, num_warps=4):
        dtype_map = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}
        _dtype = dtype_map[dtype_str]
        _H, _K_orig, _M = H, K, M
        _BT, _BM, _BK = BT, BM, BK
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        dz_s = (T_d, _H, _M)
        summary_k_s = (1, _M, _H, _K_orig)
        drouter_s = (1, T_d, _H, _K_orig)

        @T.prim_func
        def kernel(
            dz: T.Tensor(dz_s, T.float32),
            summary_k: T.Tensor(summary_k_s, _dtype),
            drouter: T.Tensor(drouter_s, _dtype),
            route_scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(_K_orig, _BK), T.ceildiv(T_d, _BT), _H, threads=_threads) as (i_k, i_t, i_h):
                t_s = i_t * _BT
                k_s = i_k * _BK
                dz_shared = T.alloc_shared((_BT, _BM), _dtype)
                key_shared = T.alloc_shared((_BM, _BK), _dtype)
                acc = T.alloc_fragment((_BT, _BK), T.float32)
                T.clear(acc)

                for i_m_block in T.Pipelined(T.ceildiv(_M, _BM), num_stages=1):
                    m_s = i_m_block * _BM
                    for i, j in T.Parallel(_BT, _BM):
                        dz_shared[i, j] = T.cast(
                            T.if_then_else(
                                (t_s + i < T_d) & (m_s + j < _M),
                                dz[t_s + i, i_h, m_s + j],
                                0.0,
                            ),
                            _dtype,
                        )
                    T.copy(summary_k[0, m_s:m_s + _BM, i_h, k_s:k_s + _BK], key_shared)
                    T.gemm(dz_shared, key_shared, acc, policy=T.GemmWarpPolicy.FullRow)

                out_shared = T.alloc_shared((_BT, _BK), _dtype)
                out_cast = T.alloc_fragment((_BT, _BK), _dtype)
                for i, j in T.Parallel(_BT, _BK):
                    out_cast[i, j] = T.cast(acc[i, j] * route_scale, _dtype)
                T.copy(out_cast, out_shared)
                for i, j in T.Parallel(_BT, _BK):
                    if (t_s + i < T_d) & (k_s + j < _K_orig):
                        drouter[0, t_s + i, i_h, k_s + j] = out_shared[i, j]

        return kernel

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _build_dsummary_k_tilelang_kernel(H, K, M, BT, BM, BK, dtype_str, num_warps=4):
        dtype_map = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}
        _dtype = dtype_map[dtype_str]
        _H, _K_orig, _M = H, K, M
        _BT, _BM, _BK = BT, BM, BK
        _threads = num_warps * 32
        T_d = T.dynamic("T")

        router_q_s = (1, T_d, _H, _K_orig)
        dz_s = (T_d, _H, _M)
        dsummary_k_s = (1, _M, _H, _K_orig)

        @T.prim_func
        def kernel(
            router_q: T.Tensor(router_q_s, _dtype),
            dz: T.Tensor(dz_s, T.float32),
            dsummary_k: T.Tensor(dsummary_k_s, _dtype),
            route_scale: T.float32,
        ):
            with T.Kernel(T.ceildiv(_K_orig, _BK), T.ceildiv(_M, _BM), _H, threads=_threads) as (i_k, i_m_block, i_h):
                k_s = i_k * _BK
                m_s = i_m_block * _BM
                dz_shared = T.alloc_shared((_BT, _BM), _dtype)
                router_shared = T.alloc_shared((_BT, _BK), _dtype)
                acc = T.alloc_fragment((_BM, _BK), T.float32)
                T.clear(acc)

                for i_t in T.Pipelined(T.ceildiv(T_d, _BT), num_stages=1):
                    t_s = i_t * _BT
                    T.copy(router_q[0, t_s:t_s + _BT, i_h, k_s:k_s + _BK], router_shared)
                    for i, j in T.Parallel(_BT, _BM):
                        dz_shared[i, j] = T.cast(
                            T.if_then_else(
                                (t_s + i < T_d) & (m_s + j < _M),
                                dz[t_s + i, i_h, m_s + j],
                                0.0,
                            ),
                            _dtype,
                        )
                    T.gemm(dz_shared, router_shared, acc, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                out_shared = T.alloc_shared((_BM, _BK), _dtype)
                out_cast = T.alloc_fragment((_BM, _BK), _dtype)
                for i, j in T.Parallel(_BM, _BK):
                    out_cast[i, j] = T.cast(acc[i, j] * route_scale, _dtype)
                T.copy(out_cast, out_shared)
                for i, j in T.Parallel(_BM, _BK):
                    if (m_s + i < _M) & (k_s + j < _K_orig):
                        dsummary_k[0, m_s + i, i_h, k_s + j] = out_shared[i, j]

        return kernel

    @tilelang.jit(pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _build_online_grouped_read_tilelang_kernel(
        H,
        K,
        V,
        R,
        M,
        BT,
        BM,
        BV,
        dtype_str,
        has_null,
        save_history_accumulator,
        num_warps=4,
    ):
        """Route and read summaries without materializing [T,H,M] weights.

        Each CTA owns one (query block, value tile, head).  It scans summaries
        in groups of BM, updates an online softmax, and accumulates the state
        reads using the same rescaling factors.  Different query blocks remain
        independent GPU work, unlike a single sequence-wide mega-kernel.
        """

        dtype_map = {
            "float16": T.float16,
            "bfloat16": T.bfloat16,
            "float32": T.float32,
        }
        _dtype = dtype_map[dtype_str]
        _H, _K, _V, _R, _M = H, K, V, R, M
        _BT, _BM, _BV = BT, BM, BV
        _threads = num_warps * 32
        _T = T.dynamic("T")
        _LOG2E = 1.4426950408889634

        read_q_s = (1, _T, _H, _K)
        router_q_s = (1, _T, _H, _R)
        states_s = (1, _M, _H, _K, _V)
        summary_k_s = (1, _M, _H, _R)
        boundary_s = (_M,)
        token_range_s = (_T,)
        null_s = (1, _T, _H, 1)
        out_s = (1, _T, _H, _V)
        history_accumulator_s = out_s if save_history_accumulator else (1,)
        stat_s = (_T, _H)

        @T.prim_func
        def kernel(
            read_q: T.Tensor(read_q_s, _dtype),
            router_q: T.Tensor(router_q_s, _dtype),
            states: T.Tensor(states_s, T.float32),
            summary_k: T.Tensor(summary_k_s, _dtype),
            boundary_counts: T.Tensor(boundary_s, T.int64),
            token_summary_start: T.Tensor(token_range_s, T.int64),
            token_summary_end: T.Tensor(token_range_s, T.int64),
            null_logits: T.Tensor(null_s, _dtype),
            out: T.Tensor(out_s, _dtype),
            history_accumulator: T.Tensor(
                history_accumulator_s,
                T.float32,
            ),
            lse: T.Tensor(stat_s, T.float32),
            p_null: T.Tensor(stat_s, T.float32),
            route_scale: T.float32,
            read_scale: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(_V, _BV),
                T.ceildiv(_T, _BT),
                _H,
                threads=_threads,
            ) as (i_v, i_t, i_h):
                t_s = i_t * _BT
                v_s = i_v * _BV
                q_shared = T.alloc_shared((_BT, _K), _dtype)
                router_shared = T.alloc_shared((_BT, _R), _dtype)
                state_shared = T.alloc_shared((_K, _BV), _dtype)
                # Route logits have N=1.  Forcing that dot product through a
                # tensor-core GEMM either leaves too few column tiles for four
                # warps or trips TileLang 0.1.9's affine-map inversion with
                # fewer warps.  Keep the large q @ state read on tensor cores,
                # but evaluate this small R=64 dot product with SIMT and a
                # row-wise reduction.
                key_shared = T.alloc_shared((_R,), _dtype)
                read_acc = T.alloc_fragment((_BT, _BV), T.float32)
                route_acc = T.alloc_fragment((_BT,), T.float32)
                out_acc = T.alloc_fragment((_BT, _BV), T.float32)
                running_max = T.alloc_fragment((_BT,), T.float32)
                running_denom = T.alloc_fragment((_BT,), T.float32)
                null_value = T.alloc_fragment((_BT,), T.float32)
                old_factor_buf = T.alloc_fragment((_BT,), T.float32)
                new_factor_buf = T.alloc_fragment((_BT,), T.float32)
                T.clear(out_acc)
                T.copy(read_q[0, t_s:t_s + _BT, i_h, :], q_shared)
                T.copy(router_q[0, t_s:t_s + _BT, i_h, :], router_shared)

                # Decide whether a summary is visible to at least one row in
                # this CTA before loading its (large) state.  ``boundary`` is
                # the first causal token for the summary.  Therefore the
                # earliest row which can possibly read it is
                # max(tile_start, boundary).  Probing that row's document-local
                # summary range makes the CTA predicate exact, including a
                # summary emitted at document end: in that case the probe lands
                # in the next document and rejects the summary.  Keeping this
                # condition CTA-uniform is important for TileLang barriers and
                # also prevents an unread terminal NaN state from entering the
                # q @ state GEMM.
                block_last_t = T.min(t_s + _BT - 1, _T - 1)

                for i in T.Parallel(_BT):
                    null_value[i] = T.if_then_else(
                        (t_s + i < _T) & has_null,
                        T.cast(
                            null_logits[
                                0,
                                T.min(t_s + i, _T - 1),
                                i_h,
                                0,
                            ],
                            T.float32,
                        ),
                        -1.0e30,
                    )
                    running_max[i] = null_value[i]
                    running_denom[i] = T.if_then_else(
                        (t_s + i < _T) & has_null,
                        1.0,
                        0.0,
                    )

                # TileLang 0.1.9 cannot pipeline this grouped loop because
                # its body contains the partial-final-group guard plus a
                # nested serial loop.  With one stage, serial scheduling has
                # the same dependency structure and compiles reliably.
                for i_mb in T.serial(T.ceildiv(_M, _BM)):
                    for i_ml in T.serial(_BM):
                        i_m = i_mb * _BM + i_ml
                        # The final group may be partial.  M is a compile-time
                        # specialization, so TileLang removes these invalid
                        # iterations without requiring a padded copy of the
                        # (large) FP32 state tensor.
                        if i_m < _M:
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
                                T.copy(summary_k[0, i_m, i_h, :], key_shared)
                                T.copy(
                                    states[
                                        0,
                                        i_m,
                                        i_h,
                                        :,
                                        v_s:v_s + _BV,
                                    ],
                                    state_shared,
                                )
                                T.clear(read_acc)
                                for i in T.Parallel(_BT):
                                    route_acc[i] = 0.0
                                # Keep the reduction scalar in the route
                                # dimension.  A [BT,R] fragment plus
                                # T.reduce_sum currently has no legal layout
                                # in TileLang 0.1.9 for this CTA.
                                for j in T.serial(_R):
                                    for i in T.Parallel(_BT):
                                        route_acc[i] += (
                                            T.cast(
                                                router_shared[i, j],
                                                T.float32,
                                            )
                                            * T.cast(
                                                key_shared[j],
                                                T.float32,
                                            )
                                        )
                                T.gemm(
                                    q_shared,
                                    state_shared,
                                    read_acc,
                                    policy=T.GemmWarpPolicy.FullRow,
                                )
                                for i in T.Parallel(_BT):
                                    valid = (
                                        (t_s + i < _T)
                                        & (
                                            i_m
                                            >= token_summary_start[
                                                T.min(t_s + i, _T - 1)
                                            ]
                                        )
                                        & (
                                            i_m
                                            < token_summary_end[
                                                T.min(t_s + i, _T - 1)
                                            ]
                                        )
                                        & (
                                            boundary_counts[i_m]
                                            <= T.min(t_s + i, _T - 1)
                                        )
                                    )
                                    logit = route_acc[i] * route_scale
                                    new_max = T.if_then_else(
                                        valid & (logit > running_max[i]),
                                        logit,
                                        running_max[i],
                                    )
                                    old_factor_buf[i] = T.if_then_else(
                                        running_denom[i] > 0.0,
                                        T.exp2(
                                            (running_max[i] - new_max)
                                            * _LOG2E
                                        ),
                                        0.0,
                                    )
                                    new_factor_buf[i] = T.if_then_else(
                                        valid,
                                        T.exp2(
                                            (logit - new_max) * _LOG2E
                                        ),
                                        0.0,
                                    )
                                    running_denom[i] = (
                                        running_denom[i] * old_factor_buf[i]
                                        + new_factor_buf[i]
                                    )
                                    running_max[i] = new_max
                                for i, j in T.Parallel(_BT, _BV):
                                    out_acc[i, j] = (
                                        out_acc[i, j]
                                        * old_factor_buf[i]
                                        + read_acc[i, j]
                                        * read_scale
                                        * new_factor_buf[i]
                                    )

                # Normalize the online accumulator in place.  Training saves
                # this pre-cast FP32 value for the exact softmax-Jacobian
                # common term in backward.  Reusing ``out_acc`` avoids an
                # additional BT x BV register fragment.
                for i, j in T.Parallel(_BT, _BV):
                    out_acc[i, j] = T.if_then_else(
                        running_denom[i] > 0.0,
                        out_acc[i, j] / running_denom[i],
                        0.0,
                    )

                out_shared = T.alloc_shared((_BT, _BV), _dtype)
                out_cast = T.alloc_fragment((_BT, _BV), _dtype)
                for i, j in T.Parallel(_BT, _BV):
                    out_cast[i, j] = T.cast(out_acc[i, j], _dtype)
                T.copy(out_cast, out_shared)
                for i, j in T.Parallel(_BT, _BV):
                    if (t_s + i < _T) & (v_s + j < _V):
                        out[0, t_s + i, i_h, v_s + j] = out_shared[i, j]
                        if save_history_accumulator:
                            history_accumulator[
                                0,
                                t_s + i,
                                i_h,
                                v_s + j,
                            ] = out_acc[i, j]
                if i_v == 0:
                    for i in T.Parallel(_BT):
                        if t_s + i < _T:
                            lse[t_s + i, i_h] = T.if_then_else(
                                running_denom[i] > 0.0,
                                running_max[i] + T.log2(running_denom[i]) / _LOG2E,
                                0.0,
                            )
                            p_null[t_s + i, i_h] = T.if_then_else(
                                has_null & (running_denom[i] > 0.0),
                                T.exp2(
                                    (null_value[i] - running_max[i]) * _LOG2E
                                ) / running_denom[i],
                                0.0,
                            )

        return kernel

else:
    _build_read_out_tilelang_kernel = None
    _build_dread_tilelang_kernel = None
    _build_dstates_tilelang_kernel = None
    _build_read_out_speed_tilelang_kernel = None
    _build_dread_speed_tilelang_kernel = None
    _build_dstates_speed_tilelang_kernel = None
    _build_dz_tilelang_kernel = None
    _build_drouter_tilelang_kernel = None
    _build_dsummary_k_tilelang_kernel = None
    _build_online_grouped_read_tilelang_kernel = None


def _dtype_str(dtype: torch.dtype) -> str | None:
    return {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}.get(dtype)


def _run_read_out_tilelang(
    *,
    read_q: torch.Tensor,
    states: torch.Tensor,
    prob: torch.Tensor,
    read_scale: float,
    block_t: int,
    block_v: int,
) -> torch.Tensor | None:
    if tilelang is None or _build_read_out_tilelang_kernel is None:
        return None
    dtype_str = _dtype_str(read_q.dtype)
    if dtype_str is None:
        return None
    _, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    if key_dim < 16 or value_dim < 16:
        return None
    if summary_count <= 0:
        return torch.zeros_like(read_q.new_empty(1, text_len, heads, value_dim))
    out = torch.empty(1, text_len, heads, value_dim, device=read_q.device, dtype=read_q.dtype)
    kernel = _build_read_out_tilelang_kernel(
        int(heads), int(key_dim), int(value_dim), int(summary_count),
        int(block_t), int(block_v), dtype_str,
        num_warps=int(_env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4)),
    )
    with torch.cuda.device(read_q.device):
        kernel(read_q, states, prob, out, float(read_scale))
    return out


def _run_dread_tilelang(
    *,
    dout: torch.Tensor,
    states: torch.Tensor,
    prob: torch.Tensor,
    read_scale: float,
    key_dim: int,
    block_t: int,
    block_k: int,
    block_v: int,
) -> torch.Tensor | None:
    if tilelang is None or _build_dread_tilelang_kernel is None:
        return None
    dtype_str = _dtype_str(dout.dtype)
    if dtype_str is None:
        return None
    _, text_len, heads, value_dim = dout.shape
    summary_count = int(states.shape[1])
    if key_dim < 16 or value_dim < 16:
        return None
    if summary_count <= 0:
        return torch.zeros(1, text_len, heads, key_dim, device=dout.device, dtype=dout.dtype)
    dread = torch.empty(1, text_len, heads, key_dim, device=dout.device, dtype=dout.dtype)
    kernel = _build_dread_tilelang_kernel(
        int(heads), int(key_dim), int(value_dim), int(summary_count),
        int(block_t), int(block_k), int(block_v), dtype_str,
        num_warps=int(_env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4)),
    )
    with torch.cuda.device(dout.device):
        kernel(dout, states, prob, dread, float(read_scale))
    return dread


def _run_dstates_tilelang(
    *,
    read_q: torch.Tensor,
    dout: torch.Tensor,
    prob: torch.Tensor,
    states: torch.Tensor,
    read_scale: float,
    block_t: int,
    block_k: int,
    block_v: int,
) -> torch.Tensor | None:
    if tilelang is None or _build_dstates_tilelang_kernel is None:
        return None
    dtype_str = _dtype_str(read_q.dtype)
    if dtype_str is None:
        return None
    _, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    if key_dim < 16 or value_dim < 16:
        return None
    if summary_count <= 0:
        return torch.zeros_like(states)
    dstates = torch.empty_like(states)
    kernel = _build_dstates_tilelang_kernel(
        int(heads), int(key_dim), int(value_dim), int(summary_count),
        int(block_t), int(block_k), int(block_v), dtype_str,
        num_warps=int(_env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4)),
    )
    with torch.cuda.device(read_q.device):
        kernel(read_q, dout, prob, dstates, float(read_scale))
    return dstates


def _speed_max_tmp_bytes() -> int:
    return max(1, _env_int("HSD_FUSED_GLOBAL_READ_TILE_SPEED_MAX_TMP_MB", 4096)) * 1024 * 1024


def _speed_block_m(summary_count: int) -> int:
    block_m = max(16, _env_int("HSD_FUSED_GLOBAL_READ_TILE_SPEED_BM", 16))
    return min(block_m, max(1, int(summary_count)))


def _speed_block_t(fallback: int) -> int:
    return max(16, _env_int("HSD_FUSED_GLOBAL_READ_TILE_SPEED_BT", max(128, int(fallback))))


def _speed_num_t_splits(states: torch.Tensor) -> int:
    requested = max(1, _env_int("HSD_FUSED_GLOBAL_READ_TILE_SPEED_SPLIT_T", 8))
    summary_count = int(states.shape[1])
    heads = int(states.shape[2])
    key_dim = int(states.shape[3])
    value_dim = int(states.shape[4])
    per_split_bytes = max(1, summary_count * heads * key_dim * value_dim * torch.tensor([], dtype=torch.float32).element_size())
    max_splits = max(1, _speed_max_tmp_bytes() // per_split_bytes)
    return int(max(1, min(requested, max_splits)))


def _padded_summary_count(summary_count: int, block_m: int) -> int:
    if block_m <= 1:
        return int(summary_count)
    return ((int(summary_count) + int(block_m) - 1) // int(block_m)) * int(block_m)


def _speed_summary_bucket_count(summary_count: int, block_m: int, text_len: int) -> int:
    if summary_count <= 0:
        return int(summary_count)
    if not _tile_speed_enabled() or not _env_bool("HSD_FUSED_GLOBAL_READ_TILE_SPEED_FIXED_M", True):
        return _padded_summary_count(summary_count, block_m)
    explicit_bucket = _env_int("HSD_FUSED_GLOBAL_READ_TILE_SPEED_M_BUCKET", 0)
    if explicit_bucket > 0:
        bucket = explicit_bucket
    elif text_len <= 32768:
        bucket = 64
    elif text_len <= 65536:
        bucket = 128
    else:
        bucket = 256
    bucket = max(int(block_m), int(bucket))
    bucket = _padded_summary_count(bucket, block_m)
    return ((int(summary_count) + bucket - 1) // bucket) * bucket


def _speed_summary_partition(
    summary_count: int,
    text_len: int,
    heads: int,
    output_dim: int,
) -> tuple[int, int, int, int]:
    block_m = _speed_block_m(summary_count)
    element_size = torch.tensor([], dtype=torch.float32).element_size()
    max_tmp_bytes = _speed_max_tmp_bytes()
    for _ in range(8):
        padded_count = _speed_summary_bucket_count(summary_count, block_m, text_len)
        num_blocks = (padded_count + block_m - 1) // block_m
        partial_bytes = num_blocks * text_len * heads * output_dim * element_size
        if partial_bytes <= max_tmp_bytes or block_m >= padded_count:
            return block_m, padded_count, num_blocks, partial_bytes

        bytes_per_block = text_len * heads * output_dim * element_size
        max_blocks = max_tmp_bytes // max(1, bytes_per_block)
        if max_blocks <= 0:
            return block_m, padded_count, num_blocks, partial_bytes
        required_block_m = (padded_count + max_blocks - 1) // max_blocks
        next_block_m = min(padded_count, max(block_m + 1, _next_power_of_2(required_block_m)))
        if next_block_m == block_m:
            return block_m, padded_count, num_blocks, partial_bytes
        block_m = next_block_m

    return block_m, padded_count, num_blocks, partial_bytes


def _pad_summary_dim(value: torch.Tensor, padded_count: int, *, dim: int) -> torch.Tensor:
    current = int(value.shape[dim])
    if current == int(padded_count):
        return value
    padded_shape = list(value.shape)
    padded_shape[dim] = int(padded_count)
    padded = value.new_zeros(tuple(padded_shape))
    target = [slice(None)] * value.ndim
    target[dim] = slice(0, current)
    padded[tuple(target)].copy_(value)
    return padded


def _pad_boundary_counts(boundary_counts: torch.Tensor, padded_count: int, *, invalid_boundary: int) -> torch.Tensor:
    current = int(boundary_counts.shape[0])
    if current == int(padded_count):
        return boundary_counts
    padded = torch.full(
        (int(padded_count),),
        int(invalid_boundary),
        device=boundary_counts.device,
        dtype=boundary_counts.dtype,
    )
    padded[:current].copy_(boundary_counts)
    return padded


def _pad_summary_cu_last(summary_cu_seqlens: torch.Tensor, padded_count: int) -> torch.Tensor:
    if int(summary_cu_seqlens[-1].item()) == int(padded_count):
        return summary_cu_seqlens
    padded = summary_cu_seqlens.clone()
    padded[-1] = int(padded_count)
    return padded


def _packed_token_summary_ranges(
    *,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each packed token's [summary_start, summary_end) document range."""

    text_lengths = (text_cu_seqlens[1:] - text_cu_seqlens[:-1]).to(torch.long)
    doc_ids = torch.repeat_interleave(
        torch.arange(
            text_lengths.numel(),
            device=text_cu_seqlens.device,
            dtype=torch.long,
        ),
        text_lengths,
    )
    return (
        summary_cu_seqlens[:-1].to(torch.long).index_select(0, doc_ids).contiguous(),
        summary_cu_seqlens[1:].to(torch.long).index_select(0, doc_ids).contiguous(),
    )


def _run_online_grouped_tile_forward(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: torch.Tensor | None,
    route_scale: float,
    read_scale: float,
    save_history_accumulator: bool,
    profile: str | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
] | None:
    """Run the no-[T,H,M]-materialization TileLang forward."""

    if tilelang is None or _build_online_grouped_read_tilelang_kernel is None:
        return None
    if read_q.dtype != torch.bfloat16 or states.dtype != torch.float32:
        return None
    if not read_q.is_cuda or torch.cuda.get_device_capability(read_q.device)[0] != 9:
        return None
    dtype_str = _dtype_str(read_q.dtype)
    if dtype_str is None or router_q.dtype != read_q.dtype or summary_k.dtype != read_q.dtype:
        return None
    batch, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    route_dim = int(router_q.shape[-1])
    if batch != 1:
        return None
    resolved_profile = _resolve_tile_online_profile(
        profile,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        route_dim=route_dim,
    )
    if resolved_profile is None:
        return None
    if summary_count <= 0:
        out = read_q.new_zeros((1, text_len, heads, value_dim))
        history_accumulator = (
            torch.zeros_like(out, dtype=torch.float32)
            if save_history_accumulator
            else torch.empty(0, device=read_q.device, dtype=torch.float32)
        )
        lse = torch.zeros((text_len, heads), device=read_q.device, dtype=torch.float32)
        p_null = torch.ones_like(lse) if null_logits is not None else torch.zeros_like(lse)
        empty_ranges = torch.zeros(
            (text_len,),
            device=read_q.device,
            dtype=torch.long,
        )
        if null_logits is not None:
            lse.copy_(null_logits[0, ..., 0].float())
        return (
            out,
            history_accumulator,
            lse,
            p_null,
            empty_ranges,
            empty_ranges,
        )

    # An explicit profile is a closed production LaunchSpec.  It deliberately
    # ignores HSD_* so the immutable model contract, not ambient process state,
    # determines the actual kernel.  The omitted-profile path keeps legacy HF
    # tuning behavior.
    block_t, block_m, block_v, num_warps = _online_forward_launch_spec(
        resolved_profile
    )
    if (
        not _is_power_of_two(block_t)
        or not _is_power_of_two(block_v)
        or value_dim % block_v != 0
        or num_warps not in {1, 2, 4, 8}
    ):
        return None
    token_summary_start, token_summary_end = _packed_token_summary_ranges(
        text_cu_seqlens=text_cu_seqlens,
        summary_cu_seqlens=summary_cu_seqlens,
    )
    if token_summary_start.numel() != text_len:
        return None
    null_arg = (
        null_logits
        if null_logits is not None
        else read_q.new_zeros((1, text_len, heads, 1))
    )
    out = read_q.new_empty((1, text_len, heads, value_dim))
    history_accumulator = (
        torch.empty_like(out, dtype=torch.float32)
        if save_history_accumulator
        # The compile-time false branch never dereferences this pointer.  A
        # scalar keeps inference free of the O(T*H*V) training workspace.
        else torch.empty(1, device=read_q.device, dtype=torch.float32)
    )
    lse = torch.empty((text_len, heads), device=read_q.device, dtype=torch.float32)
    p_null = torch.empty_like(lse)
    kernel = _build_online_grouped_read_tilelang_kernel(
        int(heads),
        int(key_dim),
        int(value_dim),
        int(route_dim),
        int(summary_count),
        int(block_t),
        int(block_m),
        int(block_v),
        dtype_str,
        bool(null_logits is not None),
        bool(save_history_accumulator),
        num_warps=num_warps,
    )
    with torch.cuda.device(read_q.device):
        kernel(
            read_q,
            router_q,
            states,
            summary_k,
            boundary_counts,
            token_summary_start,
            token_summary_end,
            null_arg,
            out,
            history_accumulator,
            lse,
            p_null,
            float(route_scale),
            float(read_scale),
        )
    if not save_history_accumulator:
        history_accumulator = history_accumulator[:0]
    return (
        out,
        history_accumulator,
        lse,
        p_null,
        token_summary_start,
        token_summary_end,
    )


def _run_read_out_speed_tilelang(
    *,
    read_q: torch.Tensor,
    states: torch.Tensor,
    prob: torch.Tensor,
    read_scale: float,
    block_t: int,
    block_v: int,
) -> torch.Tensor | None:
    if (
        tilelang is None
        or triton is None
        or _build_read_out_speed_tilelang_kernel is None
        or _speed_partial_thd_reduce_kernel is None
    ):
        return None
    dtype_str = _dtype_str(read_q.dtype)
    if dtype_str is None:
        return None
    _, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    if key_dim < 16 or value_dim < 16 or summary_count <= 0:
        return None
    block_m, padded_summary_count, num_m_blocks, partial_bytes = _speed_summary_partition(
        summary_count,
        text_len,
        heads,
        value_dim,
    )
    if padded_summary_count != summary_count:
        states = _pad_summary_dim(states, padded_summary_count, dim=1)
        prob = _pad_summary_dim(prob, padded_summary_count, dim=2)
        summary_count = padded_summary_count
    if partial_bytes > _speed_max_tmp_bytes():
        if _tile_speed_strict():
            raise RuntimeError(
                'HSD TileSpeed read-out partial buffer exceeds its limit: '
                f'T={text_len}, H={heads}, V={value_dim}, M={summary_count}, BM={block_m}, '
                f'blocks={num_m_blocks}, bytes={partial_bytes}, limit={_speed_max_tmp_bytes()}.')
        return None
    partial = torch.empty(num_m_blocks, text_len, heads, value_dim, device=read_q.device, dtype=torch.float32)
    out = torch.empty(1, text_len, heads, value_dim, device=read_q.device, dtype=read_q.dtype)
    kernel = _build_read_out_speed_tilelang_kernel(
        int(heads), int(key_dim), int(value_dim), int(summary_count),
        int(block_t), int(block_m), int(block_v), dtype_str,
        num_warps=int(_env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4)),
    )
    with torch.cuda.device(read_q.device):
        kernel(read_q, states, prob, partial, float(read_scale))
        _speed_partial_thd_reduce_kernel[(triton.cdiv(value_dim, block_v), triton.cdiv(text_len, block_t), heads)](
            partial,
            out,
            N=int(num_m_blocks),
            T=int(text_len),
            H=int(heads),
            D=int(value_dim),
            BLOCK_T=int(block_t),
            BLOCK_D=int(block_v),
            num_warps=4,
            num_stages=1,
        )
    return out


def _run_dread_speed_tilelang(
    *,
    dout: torch.Tensor,
    states: torch.Tensor,
    prob: torch.Tensor,
    read_scale: float,
    key_dim: int,
    block_t: int,
    block_k: int,
    block_v: int,
) -> torch.Tensor | None:
    if (
        tilelang is None
        or triton is None
        or _build_dread_speed_tilelang_kernel is None
        or _speed_partial_thd_reduce_kernel is None
    ):
        return None
    dtype_str = _dtype_str(dout.dtype)
    if dtype_str is None:
        return None
    _, text_len, heads, value_dim = dout.shape
    summary_count = int(states.shape[1])
    if key_dim < 16 or value_dim < 16 or summary_count <= 0:
        return None
    block_m, padded_summary_count, num_m_blocks, partial_bytes = _speed_summary_partition(
        summary_count,
        text_len,
        heads,
        key_dim,
    )
    if padded_summary_count != summary_count:
        states = _pad_summary_dim(states, padded_summary_count, dim=1)
        prob = _pad_summary_dim(prob, padded_summary_count, dim=2)
        summary_count = padded_summary_count
    if partial_bytes > _speed_max_tmp_bytes():
        if _tile_speed_strict():
            raise RuntimeError(
                'HSD TileSpeed dread partial buffer exceeds its limit: '
                f'T={text_len}, H={heads}, K={key_dim}, M={summary_count}, BM={block_m}, '
                f'blocks={num_m_blocks}, bytes={partial_bytes}, limit={_speed_max_tmp_bytes()}.')
        return None
    partial = torch.empty(num_m_blocks, text_len, heads, key_dim, device=dout.device, dtype=torch.float32)
    dread = torch.empty(1, text_len, heads, key_dim, device=dout.device, dtype=dout.dtype)
    kernel = _build_dread_speed_tilelang_kernel(
        int(heads), int(key_dim), int(value_dim), int(summary_count),
        int(block_t), int(block_m), int(block_k), int(block_v), dtype_str,
        num_warps=int(_env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4)),
    )
    with torch.cuda.device(dout.device):
        kernel(dout, states, prob, partial, float(read_scale))
        _speed_partial_thd_reduce_kernel[(triton.cdiv(key_dim, block_k), triton.cdiv(text_len, block_t), heads)](
            partial,
            dread,
            N=int(num_m_blocks),
            T=int(text_len),
            H=int(heads),
            D=int(key_dim),
            BLOCK_T=int(block_t),
            BLOCK_D=int(block_k),
            num_warps=4,
            num_stages=1,
        )
    return dread


def _run_dstates_speed_tilelang(
    *,
    read_q: torch.Tensor,
    dout: torch.Tensor,
    prob: torch.Tensor,
    states: torch.Tensor,
    read_scale: float,
    block_t: int,
    block_k: int,
    block_v: int,
) -> torch.Tensor | None:
    if (
        tilelang is None
        or triton is None
        or _build_dstates_speed_tilelang_kernel is None
        or _speed_partial_dstates_reduce_kernel is None
    ):
        return None
    dtype_str = _dtype_str(read_q.dtype)
    if dtype_str is None or dout.dtype != read_q.dtype:
        return None
    _, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    if key_dim < 16 or value_dim < 16 or summary_count <= 0:
        return None
    original_summary_count = summary_count
    block_m = _speed_block_m(summary_count)
    padded_summary_count = _speed_summary_bucket_count(summary_count, block_m, text_len)
    if padded_summary_count != summary_count:
        states = _pad_summary_dim(states, padded_summary_count, dim=1)
        prob = _pad_summary_dim(prob, padded_summary_count, dim=2)
        summary_count = padded_summary_count
    num_splits = _speed_num_t_splits(states)
    if num_splits <= 1 and _env_bool("HSD_FUSED_GLOBAL_READ_TILE_SPEED_REQUIRE_SPLIT_T", False):
        return None
    partial_bytes = num_splits * summary_count * heads * key_dim * value_dim * torch.tensor([], dtype=torch.float32).element_size()
    if partial_bytes > _speed_max_tmp_bytes():
        return None
    partial = torch.empty(num_splits, summary_count, heads, key_dim, value_dim, device=read_q.device, dtype=torch.float32)
    dstates = torch.empty_like(states)
    kernel = _build_dstates_speed_tilelang_kernel(
        int(heads), int(key_dim), int(value_dim), int(summary_count),
        int(block_t), int(block_k), int(block_v), int(num_splits), dtype_str,
        num_warps=int(_env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4)),
    )
    with torch.cuda.device(read_q.device):
        kernel(read_q, dout, prob, partial, float(read_scale))
        _speed_partial_dstates_reduce_kernel[(triton.cdiv(value_dim, block_v), triton.cdiv(key_dim, block_k), summary_count * heads)](
            partial,
            dstates,
            N=int(num_splits),
            M=int(summary_count),
            H=int(heads),
            K=int(key_dim),
            V=int(value_dim),
            BLOCK_K=int(block_k),
            BLOCK_V=int(block_v),
            num_warps=4,
            num_stages=1,
        )
    if summary_count != original_summary_count:
        dstates = dstates[:, :original_summary_count].contiguous()
    return dstates


def _run_dz_tilelang(
    *,
    read_q: torch.Tensor,
    states: torch.Tensor,
    dout: torch.Tensor,
    prob: torch.Tensor,
    delta: torch.Tensor,
    read_scale: float,
    block_t: int,
    block_m: int,
    block_k: int,
    block_v: int,
) -> torch.Tensor | None:
    if tilelang is None or _build_dz_tilelang_kernel is None:
        return None
    dtype_str = _dtype_str(read_q.dtype)
    if dtype_str is None or dout.dtype != read_q.dtype:
        return None
    _, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    if key_dim < 16 or value_dim < 16 or summary_count <= 0:
        return None
    original_summary_count = summary_count
    padded_summary_count = _speed_summary_bucket_count(summary_count, block_m, text_len)
    if padded_summary_count != summary_count:
        states = _pad_summary_dim(states, padded_summary_count, dim=1)
        prob = _pad_summary_dim(prob, padded_summary_count, dim=2)
        summary_count = padded_summary_count
    dz = torch.empty(text_len, heads, summary_count, device=read_q.device, dtype=torch.float32)
    kernel = _build_dz_tilelang_kernel(
        int(heads), int(key_dim), int(value_dim), int(summary_count),
        int(block_t), int(block_m), int(block_k), int(block_v), dtype_str,
        num_warps=int(_env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4)),
    )
    with torch.cuda.device(read_q.device):
        kernel(read_q, states, dout, prob, delta, dz, float(read_scale))
    if summary_count != original_summary_count:
        dz = dz[:, :, :original_summary_count].contiguous()
    return dz


def _run_dweight_tilelang(
    *,
    read_q: torch.Tensor,
    states: torch.Tensor,
    dout: torch.Tensor,
    prob: torch.Tensor,
    read_scale: float,
    block_t: int,
    block_m: int,
    block_k: int,
    block_v: int,
) -> torch.Tensor | None:
    """Gradient of an externally supplied summary probability tensor.

    This is the candidate-read dot product used by the ordinary route-softmax
    backward before applying the softmax Jacobian.  Momentum needs that dot
    product directly so its reverse scan can run first.
    """
    if tilelang is None or _build_dz_tilelang_kernel is None:
        return None
    dtype_str = _dtype_str(read_q.dtype)
    if dtype_str is None or dout.dtype != read_q.dtype:
        return None
    _, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    if key_dim < 16 or value_dim < 16 or summary_count <= 0:
        return None
    original_summary_count = summary_count
    padded_summary_count = _speed_summary_bucket_count(summary_count, block_m, text_len)
    if padded_summary_count != summary_count:
        states = _pad_summary_dim(states, padded_summary_count, dim=1)
        prob = _pad_summary_dim(prob, padded_summary_count, dim=2)
        summary_count = padded_summary_count
    dweight = torch.empty(text_len, heads, summary_count, device=read_q.device, dtype=torch.float32)
    # These arguments are eliminated by TileLang when the compile-time mode is
    # false.  Keep correctly-shaped tensors so the prim_func signature remains
    # shared with the route-softmax dz kernel.
    unused_delta = torch.empty(text_len, heads, device=read_q.device, dtype=torch.float32)
    kernel = _build_dz_tilelang_kernel(
        int(heads), int(key_dim), int(value_dim), int(summary_count),
        int(block_t), int(block_m), int(block_k), int(block_v), dtype_str,
        apply_softmax_jacobian=False,
        num_warps=int(_env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4)),
    )
    with torch.cuda.device(read_q.device):
        kernel(read_q, states, dout, prob, unused_delta, dweight, float(read_scale))
    if summary_count != original_summary_count:
        dweight = dweight[:, :, :original_summary_count].contiguous()
    return dweight


def _run_drouter_tilelang(
    *,
    dz: torch.Tensor,
    summary_k: torch.Tensor,
    router_q: torch.Tensor,
    route_scale: float,
    block_t: int,
    block_m: int,
    block_k: int,
) -> torch.Tensor | None:
    if tilelang is None or _build_drouter_tilelang_kernel is None:
        return None
    dtype_str = _dtype_str(summary_k.dtype)
    if dtype_str is None or router_q.dtype != summary_k.dtype:
        return None
    _, text_len, heads, key_dim = router_q.shape
    summary_count = int(summary_k.shape[1])
    if key_dim < 16 or summary_count <= 0:
        return None
    padded_summary_count = _speed_summary_bucket_count(summary_count, block_m, text_len)
    if padded_summary_count != summary_count:
        summary_k = _pad_summary_dim(summary_k, padded_summary_count, dim=1)
        dz = _pad_summary_dim(dz, padded_summary_count, dim=2)
        summary_count = padded_summary_count
    drouter = torch.empty_like(router_q)
    kernel = _build_drouter_tilelang_kernel(
        int(heads), int(key_dim), int(summary_count),
        int(block_t), int(block_m), int(block_k), dtype_str,
        num_warps=int(_env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4)),
    )
    with torch.cuda.device(router_q.device):
        kernel(dz, summary_k, drouter, float(route_scale))
    return drouter


def _run_dsummary_k_tilelang(
    *,
    router_q: torch.Tensor,
    dz: torch.Tensor,
    summary_k: torch.Tensor,
    route_scale: float,
    block_t: int,
    block_m: int,
    block_k: int,
) -> torch.Tensor | None:
    if tilelang is None or _build_dsummary_k_tilelang_kernel is None:
        return None
    dtype_str = _dtype_str(router_q.dtype)
    if dtype_str is None or summary_k.dtype != router_q.dtype:
        return None
    _, text_len, heads, key_dim = router_q.shape
    summary_count = int(summary_k.shape[1])
    if key_dim < 16 or summary_count <= 0:
        return None
    original_summary_count = summary_count
    padded_summary_count = _speed_summary_bucket_count(summary_count, block_m, text_len)
    if padded_summary_count != summary_count:
        dz = _pad_summary_dim(dz, padded_summary_count, dim=2)
        summary_count = padded_summary_count
    dsummary_k = torch.empty_like(summary_k)
    if summary_count != original_summary_count:
        dsummary_k = _pad_summary_dim(dsummary_k, summary_count, dim=1)
    kernel = _build_dsummary_k_tilelang_kernel(
        int(heads), int(key_dim), int(summary_count),
        int(block_t), int(block_m), int(block_k), dtype_str,
        num_warps=int(_env_int("HSD_FUSED_GLOBAL_READ_TILE_TL_WARPS", 4)),
    )
    with torch.cuda.device(router_q.device):
        kernel(router_q, dz, dsummary_k, float(route_scale))
    if summary_count != original_summary_count:
        dsummary_k = dsummary_k[:, :original_summary_count].contiguous()
    return dsummary_k


if triton is not None:

    @triton.jit
    def _route_softmax_fwd_kernel(
        router_q,
        summary_k,
        boundary_counts,
        text_cu_seqlens,
        summary_cu_seqlens,
        null_logits,
        prob,
        lse,
        p_null,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        N_DOCS: tl.constexpr,
        ROUTE_SCALE: tl.constexpr,
        HAS_NULL: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_M_BLOCKS: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
    ):
        i_t = tl.program_id(0)
        i_h = tl.program_id(1)
        offs_t = i_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_m_base = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        t_mask = offs_t < T

        summary_start = tl.zeros([BLOCK_T], dtype=tl.int64)
        summary_end = tl.zeros([BLOCK_T], dtype=tl.int64)
        for i_doc in range(N_DOCS):
            doc_text_start = tl.load(text_cu_seqlens + i_doc)
            doc_text_end = tl.load(text_cu_seqlens + i_doc + 1)
            in_doc = (offs_t >= doc_text_start) & (offs_t < doc_text_end)
            summary_start = tl.where(in_doc, tl.load(summary_cu_seqlens + i_doc), summary_start)
            summary_end = tl.where(in_doc, tl.load(summary_cu_seqlens + i_doc + 1), summary_end)

        neg_inf = -3.4028234663852886e38
        max_logit = tl.full([BLOCK_T], neg_inf, dtype=tl.float32)

        for i_m_block in range(NUM_M_BLOCKS):
            offs_m = i_m_block * BLOCK_M + offs_m_base
            logits = tl.zeros([BLOCK_T, BLOCK_M], dtype=tl.float32)
            for i_k_block in range(NUM_K_BLOCKS):
                offs_k_cur = i_k_block * BLOCK_K + offs_k
                router = tl.load(
                    router_q + ((offs_t[:, None] * H + i_h) * K + offs_k_cur[None, :]),
                    mask=t_mask[:, None] & (offs_k_cur[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                keys = tl.load(
                    summary_k + ((offs_m[:, None] * H + i_h) * K + offs_k_cur[None, :]),
                    mask=(offs_m[:, None] < M) & (offs_k_cur[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.dot(router, tl.trans(keys), input_precision="tf32")

            boundaries = tl.load(boundary_counts + offs_m, mask=offs_m < M, other=T + 1)
            valid = (
                t_mask[:, None]
                & (offs_m[None, :] >= summary_start[:, None])
                & (offs_m[None, :] < summary_end[:, None])
                & (boundaries[None, :] <= offs_t[:, None])
            )
            logits = tl.where(valid, logits * ROUTE_SCALE, neg_inf)
            max_logit = tl.maximum(max_logit, tl.max(logits, axis=1))

        if HAS_NULL:
            null_value = tl.load(null_logits + offs_t * H + i_h, mask=t_mask, other=neg_inf).to(tl.float32)
            max_logit = tl.maximum(max_logit, null_value)

        max_safe = tl.where(max_logit == neg_inf, 0.0, max_logit)
        denom = tl.zeros([BLOCK_T], dtype=tl.float32)

        for i_m_block in range(NUM_M_BLOCKS):
            offs_m = i_m_block * BLOCK_M + offs_m_base
            logits = tl.zeros([BLOCK_T, BLOCK_M], dtype=tl.float32)
            for i_k_block in range(NUM_K_BLOCKS):
                offs_k_cur = i_k_block * BLOCK_K + offs_k
                router = tl.load(
                    router_q + ((offs_t[:, None] * H + i_h) * K + offs_k_cur[None, :]),
                    mask=t_mask[:, None] & (offs_k_cur[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                keys = tl.load(
                    summary_k + ((offs_m[:, None] * H + i_h) * K + offs_k_cur[None, :]),
                    mask=(offs_m[:, None] < M) & (offs_k_cur[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
                logits += tl.dot(router, tl.trans(keys), input_precision="tf32")

            boundaries = tl.load(boundary_counts + offs_m, mask=offs_m < M, other=T + 1)
            valid = (
                t_mask[:, None]
                & (offs_m[None, :] >= summary_start[:, None])
                & (offs_m[None, :] < summary_end[:, None])
                & (boundaries[None, :] <= offs_t[:, None])
            )
            logits = tl.where(valid, logits * ROUTE_SCALE, neg_inf)
            numerator = tl.where(valid, tl.exp(logits - max_safe[:, None]), 0.0)
            denom += tl.sum(numerator, axis=1)
            tl.store(
                prob + ((offs_t[:, None] * H + i_h) * M + offs_m[None, :]),
                numerator,
                mask=t_mask[:, None] & (offs_m[None, :] < M),
            )

        if HAS_NULL:
            null_value = tl.load(null_logits + offs_t * H + i_h, mask=t_mask, other=neg_inf).to(tl.float32)
            null_num = tl.exp(null_value - max_safe)
            denom += null_num
            tl.store(p_null + offs_t * H + i_h, null_num / denom, mask=t_mask)

        denom_safe = tl.where(denom > 0.0, denom, 1.0)
        for i_m_block in range(NUM_M_BLOCKS):
            offs_m = i_m_block * BLOCK_M + offs_m_base
            values = tl.load(
                prob + ((offs_t[:, None] * H + i_h) * M + offs_m[None, :]),
                mask=t_mask[:, None] & (offs_m[None, :] < M),
                other=0.0,
            )
            tl.store(
                prob + ((offs_t[:, None] * H + i_h) * M + offs_m[None, :]),
                values / denom_safe[:, None],
                mask=t_mask[:, None] & (offs_m[None, :] < M),
            )

        lse_value = tl.where(denom > 0.0, max_safe + tl.log(denom), 0.0)
        tl.store(lse + offs_t * H + i_h, lse_value, mask=t_mask)

    @triton.jit
    def _read_out_fwd_kernel(
        read_q,
        states,
        prob,
        out,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        READ_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_M_BLOCKS: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_h = tl.program_id(2)
        offs_t = i_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_k = tl.arange(0, BLOCK_K)
        t_mask = offs_t < T
        v_mask = offs_v < V
        acc = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)

        for i_m_block in range(NUM_M_BLOCKS):
            for i_m_local in tl.static_range(0, BLOCK_M):
                cur_m = i_m_block * BLOCK_M + i_m_local
                p = tl.load(
                    prob + (offs_t * H + i_h) * M + cur_m,
                    mask=t_mask & (cur_m < M),
                    other=0.0,
                ).to(tl.float32)
                read_value = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
                for i_k_block in range(NUM_K_BLOCKS):
                    offs_k_cur = i_k_block * BLOCK_K + offs_k
                    q = tl.load(
                        read_q + ((offs_t[:, None] * H + i_h) * K + offs_k_cur[None, :]),
                        mask=t_mask[:, None] & (offs_k_cur[None, :] < K),
                        other=0.0,
                    ).to(tl.float32)
                    state = tl.load(
                        states + (((cur_m * H + i_h) * K + offs_k_cur[:, None]) * V + offs_v[None, :]),
                        mask=(cur_m < M) & (offs_k_cur[:, None] < K) & v_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    read_value += tl.dot(q, state, input_precision="tf32")
                acc += p[:, None] * read_value * READ_SCALE

        tl.store(
            out + ((offs_t[:, None] * H + i_h) * V + offs_v[None, :]),
            acc,
            mask=t_mask[:, None] & v_mask[None, :],
        )

    @triton.jit
    def _bwd_delta_kernel(
        dout,
        out,
        p_null,
        dlocal_prob,
        delta,
        dnull,
        T: tl.constexpr,
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
        t_mask = offs_t < T
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)
        for i_v_block in range(NUM_V_BLOCKS):
            offs_v_cur = i_v_block * BLOCK_V + offs_v
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
            acc += tl.sum(do * o, axis=1)
        pn = tl.load(p_null + offs_t * H + i_h, mask=t_mask, other=0.0).to(tl.float32) if HAS_NULL else 0.0
        gp = tl.load(dlocal_prob + offs_t * H + i_h, mask=t_mask, other=0.0).to(tl.float32) if HAS_DLOCAL else 0.0
        acc += gp * pn
        tl.store(delta + offs_t * H + i_h, acc, mask=t_mask)
        if HAS_NULL and NEED_DNULL:
            tl.store(dnull + offs_t * H + i_h, pn * (gp - acc), mask=t_mask)

    @triton.jit
    def _bwd_dz_kernel(
        read_q,
        states,
        dout,
        prob,
        delta,
        dz,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        READ_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_K_BLOCKS: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
    ):
        i_m_block = tl.program_id(0)
        i_t = tl.program_id(1)
        i_h = tl.program_id(2)
        offs_t = i_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        t_mask = offs_t < T
        delta_t = tl.load(delta + offs_t * H + i_h, mask=t_mask, other=0.0).to(tl.float32)

        for i_m_local in tl.static_range(0, BLOCK_M):
            cur_m = i_m_block * BLOCK_M + i_m_local
            dot_read = tl.zeros([BLOCK_T], dtype=tl.float32)
            for i_v_block in range(NUM_V_BLOCKS):
                offs_v_cur = i_v_block * BLOCK_V + offs_v
                read_value = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
                for i_k_block in range(NUM_K_BLOCKS):
                    offs_k_cur = i_k_block * BLOCK_K + offs_k
                    q = tl.load(
                        read_q + ((offs_t[:, None] * H + i_h) * K + offs_k_cur[None, :]),
                        mask=t_mask[:, None] & (offs_k_cur[None, :] < K),
                        other=0.0,
                    ).to(tl.float32)
                    state = tl.load(
                        states + (((cur_m * H + i_h) * K + offs_k_cur[:, None]) * V + offs_v_cur[None, :]),
                        mask=(cur_m < M) & (offs_k_cur[:, None] < K) & (offs_v_cur[None, :] < V),
                        other=0.0,
                    ).to(tl.float32)
                    read_value += tl.dot(q, state, input_precision="tf32")
                do = tl.load(
                    dout + ((offs_t[:, None] * H + i_h) * V + offs_v_cur[None, :]),
                    mask=t_mask[:, None] & (offs_v_cur[None, :] < V),
                    other=0.0,
                ).to(tl.float32)
                dot_read += tl.sum(do * read_value, axis=1) * READ_SCALE

            p = tl.load(prob + (offs_t * H + i_h) * M + cur_m, mask=t_mask & (cur_m < M), other=0.0)
            tl.store(
                dz + (offs_t * H + i_h) * M + cur_m,
                p.to(tl.float32) * (dot_read - delta_t),
                mask=t_mask & (cur_m < M),
            )

    @triton.jit
    def _bwd_dread_kernel(
        dout,
        states,
        prob,
        dread,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        READ_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_M_BLOCKS: tl.constexpr,
        NUM_V_BLOCKS: tl.constexpr,
    ):
        i_k = tl.program_id(0)
        i_t = tl.program_id(1)
        i_h = tl.program_id(2)
        offs_t = i_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        t_mask = offs_t < T
        k_mask = offs_k < K
        acc = tl.zeros([BLOCK_T, BLOCK_K], dtype=tl.float32)

        for i_m_block in range(NUM_M_BLOCKS):
            for i_m_local in tl.static_range(0, BLOCK_M):
                cur_m = i_m_block * BLOCK_M + i_m_local
                p = tl.load(prob + (offs_t * H + i_h) * M + cur_m, mask=t_mask & (cur_m < M), other=0.0)
                for i_v_block in range(NUM_V_BLOCKS):
                    offs_v_cur = i_v_block * BLOCK_V + offs_v
                    do = tl.load(
                        dout + ((offs_t[:, None] * H + i_h) * V + offs_v_cur[None, :]),
                        mask=t_mask[:, None] & (offs_v_cur[None, :] < V),
                        other=0.0,
                    ).to(tl.float32)
                    state = tl.load(
                        states + (((cur_m * H + i_h) * K + offs_k[:, None]) * V + offs_v_cur[None, :]),
                        mask=(cur_m < M) & k_mask[:, None] & (offs_v_cur[None, :] < V),
                        other=0.0,
                    ).to(tl.float32)
                    weighted_do = do * (p[:, None].to(tl.float32) * READ_SCALE)
                    acc += tl.dot(weighted_do, tl.trans(state), input_precision="tf32")

        tl.store(
            dread + ((offs_t[:, None] * H + i_h) * K + offs_k[None, :]),
            acc,
            mask=t_mask[:, None] & k_mask[None, :],
        )

    @triton.jit
    def _bwd_dstates_kernel(
        read_q,
        dout,
        prob,
        dstates,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        READ_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_T_BLOCKS: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_mh = tl.program_id(2)
        i_m = i_mh % M
        i_h = i_mh // M
        offs_t_base = tl.arange(0, BLOCK_T)
        offs_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        k_mask = offs_k < K
        v_mask = offs_v < V
        acc = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)

        for i_t in range(NUM_T_BLOCKS):
            offs_t = i_t * BLOCK_T + offs_t_base
            t_mask = offs_t < T
            q = tl.load(
                read_q + ((offs_t[:, None] * H + i_h) * K + offs_k[None, :]),
                mask=t_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            do = tl.load(
                dout + ((offs_t[:, None] * H + i_h) * V + offs_v[None, :]),
                mask=t_mask[:, None] & v_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            p = tl.load(prob + (offs_t * H + i_h) * M + i_m, mask=t_mask & (i_m < M), other=0.0)
            weighted_do = do * (p[:, None].to(tl.float32) * READ_SCALE)
            acc += tl.dot(tl.trans(q), weighted_do, input_precision="tf32")

        tl.store(
            dstates + (((i_m * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :]),
            acc,
            mask=(i_m < M) & k_mask[:, None] & v_mask[None, :],
        )

    @triton.jit
    def _bwd_drouter_kernel(
        dz,
        summary_k,
        drouter,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        ROUTE_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_M_BLOCKS: tl.constexpr,
    ):
        i_k = tl.program_id(0)
        i_t = tl.program_id(1)
        i_h = tl.program_id(2)
        offs_t = i_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        t_mask = offs_t < T
        k_mask = offs_k < K
        acc = tl.zeros([BLOCK_T, BLOCK_K], dtype=tl.float32)

        for i_m_block in range(NUM_M_BLOCKS):
            offs_m_cur = i_m_block * BLOCK_M + offs_m
            dz_block = tl.load(
                dz + ((offs_t[:, None] * H + i_h) * M + offs_m_cur[None, :]),
                mask=t_mask[:, None] & (offs_m_cur[None, :] < M),
                other=0.0,
            ).to(tl.float32)
            keys = tl.load(
                summary_k + ((offs_m_cur[:, None] * H + i_h) * K + offs_k[None, :]),
                mask=(offs_m_cur[:, None] < M) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(dz_block, keys, input_precision="tf32") * ROUTE_SCALE

        tl.store(
            drouter + ((offs_t[:, None] * H + i_h) * K + offs_k[None, :]),
            acc,
            mask=t_mask[:, None] & k_mask[None, :],
        )

    @triton.jit
    def _bwd_dsummary_k_kernel(
        router_q,
        dz,
        dsummary_k,
        T: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        ROUTE_SCALE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_T_BLOCKS: tl.constexpr,
    ):
        i_k = tl.program_id(0)
        i_m = tl.program_id(1)
        i_h = tl.program_id(2)
        offs_t_base = tl.arange(0, BLOCK_T)
        offs_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)

        for i_t in range(NUM_T_BLOCKS):
            offs_t = i_t * BLOCK_T + offs_t_base
            t_mask = offs_t < T
            router = tl.load(
                router_q + ((offs_t[:, None] * H + i_h) * K + offs_k[None, :]),
                mask=t_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            dz_value = tl.load(
                dz + (offs_t * H + i_h) * M + i_m,
                mask=t_mask & (i_m < M),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(router * dz_value[:, None], axis=0) * ROUTE_SCALE

        tl.store(
            dsummary_k + ((i_m * H + i_h) * K + offs_k),
            acc,
            mask=(i_m < M) & k_mask,
        )

    @triton.jit
    def _speed_partial_thd_reduce_kernel(
        partial,
        out,
        N: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        i_d = tl.program_id(0)
        i_t = tl.program_id(1)
        i_h = tl.program_id(2)
        offs_t = i_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = i_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (offs_t[:, None] < T) & (offs_d[None, :] < D)
        acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        for i_n in range(N):
            acc += tl.load(
                partial + (((i_n * T + offs_t[:, None]) * H + i_h) * D + offs_d[None, :]),
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            out + ((offs_t[:, None] * H + i_h) * D + offs_d[None, :]),
            acc,
            mask=mask,
        )

    @triton.jit
    def _speed_partial_dstates_reduce_kernel(
        partial,
        out,
        N: tl.constexpr,
        M: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_mh = tl.program_id(2)
        i_m = i_mh // H
        i_h = i_mh - i_m * H
        offs_k = i_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = (i_m < M) & (offs_k[:, None] < K) & (offs_v[None, :] < V)
        acc = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
        for i_n in range(N):
            acc += tl.load(
                partial + ((((i_n * M + i_m) * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :]),
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            out + (((i_m * H + i_h) * K + offs_k[:, None]) * V + offs_v[None, :]),
            acc,
            mask=mask,
        )

else:
    _route_softmax_fwd_kernel = None
    _read_out_fwd_kernel = None
    _bwd_delta_kernel = None
    _bwd_dz_kernel = None
    _bwd_dread_kernel = None
    _bwd_dstates_kernel = None
    _bwd_drouter_kernel = None
    _bwd_dsummary_k_kernel = None
    _speed_partial_thd_reduce_kernel = None
    _speed_partial_dstates_reduce_kernel = None


def _supports_custom_tile_global_read(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: torch.Tensor | None,
) -> bool:
    if not _supports_tile_global_read(
        read_q=read_q,
        router_q=router_q,
        states=states,
        summary_k=summary_k,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu_seqlens,
        summary_cu_seqlens=summary_cu_seqlens,
        null_logits=null_logits,
    ):
        return False
    return (
        triton is not None
        and _route_softmax_fwd_kernel is not None
        and _read_out_fwd_kernel is not None
        and _bwd_delta_kernel is not None
        and _bwd_dz_kernel is not None
        and _bwd_dread_kernel is not None
        and _bwd_dstates_kernel is not None
        and _bwd_drouter_kernel is not None
        and _bwd_dsummary_k_kernel is not None
    )


def _tile_params(
    text_len: int,
    summary_count: int,
    state_key_dim: int,
    route_key_dim: int,
    value_dim: int,
) -> dict[str, int]:
    block_t = max(1, _env_int("HSD_FUSED_GLOBAL_READ_TILE_BT", 16))
    block_route_t = max(1, _env_int("HSD_FUSED_GLOBAL_READ_TILE_BT_ROUTE", 32))
    block_m = max(16, _env_int("HSD_FUSED_GLOBAL_READ_TILE_BM", 16))
    block_route_m = max(16, _env_int("HSD_FUSED_GLOBAL_READ_TILE_BM_ROUTE", 32))
    block_k = max(16, min(64, _next_power_of_2(max(1, state_key_dim))))
    block_k = max(16, _env_int("HSD_FUSED_GLOBAL_READ_TILE_BK", block_k))
    block_route_k = max(16, min(64, _next_power_of_2(max(1, route_key_dim))))
    block_route_k = max(16, _env_int("HSD_FUSED_GLOBAL_READ_TILE_BK_ROUTE", block_route_k))
    block_v = max(16, min(64, _next_power_of_2(max(1, value_dim))))
    block_v = max(16, _env_int("HSD_FUSED_GLOBAL_READ_TILE_BV", block_v))
    return {
        "block_t": block_t,
        "block_route_t": block_route_t,
        "block_m": block_m,
        "block_route_m": block_route_m,
        "block_k": block_k,
        "block_route_k": block_route_k,
        "block_v": block_v,
    }


def _run_custom_tile_forward(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: torch.Tensor | None,
    route_scale: float,
    read_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not _env_bool("HSD_FUSED_GLOBAL_READ_TILE_CUSTOM", True):
        return None
    if not _supports_custom_tile_global_read(
        read_q=read_q,
        router_q=router_q,
        states=states,
        summary_k=summary_k,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu_seqlens,
        summary_cu_seqlens=summary_cu_seqlens,
        null_logits=null_logits,
    ):
        return None

    batch, text_len, heads, state_key_dim = read_q.shape
    route_key_dim = int(router_q.shape[-1])
    summary_count = int(states.shape[1])
    original_summary_count = summary_count
    value_dim = int(states.shape[-1])
    if batch != 1:
        return None
    if summary_count <= 0:
        prob = torch.empty(text_len, heads, 0, device=read_q.device, dtype=torch.float32)
        lse = torch.empty(text_len, heads, device=read_q.device, dtype=torch.float32)
        p_null = torch.empty(text_len, heads, device=read_q.device, dtype=torch.float32) if null_logits is not None else torch.empty(0, device=read_q.device, dtype=torch.float32)
        out = torch.empty(batch, text_len, heads, value_dim, device=read_q.device, dtype=read_q.dtype)
        out.zero_()
        lse.zero_()
        if null_logits is not None:
            lse.copy_(null_logits[0, ..., 0].float())
            p_null.fill_(1.0)
        return out, lse, prob, p_null

    fixed_bucket_m = _speed_summary_bucket_count(
        summary_count,
        max(_speed_block_m(summary_count), _env_int("HSD_FUSED_GLOBAL_READ_TILE_BM_ROUTE", 32)),
        text_len,
    )
    if fixed_bucket_m != summary_count:
        states = _pad_summary_dim(states, fixed_bucket_m, dim=1)
        summary_k = _pad_summary_dim(summary_k, fixed_bucket_m, dim=1)
        boundary_counts = _pad_boundary_counts(boundary_counts, fixed_bucket_m, invalid_boundary=text_len + 1)
        summary_cu_seqlens = _pad_summary_cu_last(summary_cu_seqlens, fixed_bucket_m)
        summary_count = fixed_bucket_m

    prob = torch.empty(text_len, heads, summary_count, device=read_q.device, dtype=torch.float32)
    lse = torch.empty(text_len, heads, device=read_q.device, dtype=torch.float32)
    p_null = torch.empty(text_len, heads, device=read_q.device, dtype=torch.float32) if null_logits is not None else torch.empty(0, device=read_q.device, dtype=torch.float32)
    out = torch.empty(batch, text_len, heads, value_dim, device=read_q.device, dtype=read_q.dtype)
    params = _tile_params(text_len, summary_count, state_key_dim, route_key_dim, value_dim)
    null_arg = null_logits if null_logits is not None else read_q.new_empty((0,), dtype=read_q.dtype)
    with torch.cuda.device(read_q.device):
        _route_softmax_fwd_kernel[(triton.cdiv(text_len, params["block_route_t"]), heads)](
            router_q,
            summary_k,
            boundary_counts,
            text_cu_seqlens,
            summary_cu_seqlens,
            null_arg,
            prob,
            lse,
            p_null,
            T=int(text_len),
            M=int(summary_count),
            H=int(heads),
            K=int(route_key_dim),
            N_DOCS=int(text_cu_seqlens.numel()) - 1,
            ROUTE_SCALE=float(route_scale),
            HAS_NULL=bool(null_logits is not None),
            BLOCK_T=params["block_route_t"],
            BLOCK_M=params["block_route_m"],
            BLOCK_K=params["block_route_k"],
            NUM_M_BLOCKS=triton.cdiv(summary_count, params["block_route_m"]),
            NUM_K_BLOCKS=triton.cdiv(route_key_dim, params["block_route_k"]),
            num_warps=4,
            num_stages=1,
        )
    if _tile_speed_forward_enabled():
        try:
            speed_out = _run_read_out_speed_tilelang(
                read_q=read_q,
                states=states,
                prob=prob,
                read_scale=read_scale,
                block_t=_speed_block_t(params["block_t"]),
                block_v=params["block_v"],
            )
        except Exception:
            if _tile_speed_strict():
                raise
            speed_out = None
        if speed_out is not None:
            prob_to_save = prob[:, :, :original_summary_count].contiguous() if summary_count != original_summary_count else prob
            return speed_out, lse, prob_to_save, p_null
        if _tile_speed_strict():
            raise RuntimeError("HSD tile-speed global read forward kernel missed.")
    if _env_bool("HSD_FUSED_GLOBAL_READ_TILE_VALUE_TILELANG", True):
        try:
            tilelang_out = _run_read_out_tilelang(
                read_q=read_q,
                states=states,
                prob=prob,
                read_scale=read_scale,
                block_t=params["block_t"],
                block_v=params["block_v"],
            )
        except Exception:
            if _env_bool("HSD_FUSED_GLOBAL_READ_TILE_VALUE_TILELANG_STRICT", False):
                raise
            tilelang_out = None
        if tilelang_out is not None:
            prob_to_save = prob[:, :, :original_summary_count].contiguous() if summary_count != original_summary_count else prob
            return tilelang_out, lse, prob_to_save, p_null
        if _env_bool("HSD_FUSED_GLOBAL_READ_TILE_VALUE_TILELANG_STRICT", False):
            raise RuntimeError("HSD tile global read TileLang forward value kernel missed.")

    with torch.cuda.device(read_q.device):
        _read_out_fwd_kernel[(triton.cdiv(value_dim, params["block_v"]), triton.cdiv(text_len, params["block_t"]), heads)](
            read_q,
            states,
            prob,
            out,
            T=int(text_len),
            M=int(summary_count),
            H=int(heads),
            K=int(state_key_dim),
            V=int(value_dim),
            READ_SCALE=float(read_scale),
            BLOCK_T=params["block_t"],
            BLOCK_M=params["block_m"],
            BLOCK_K=params["block_k"],
            BLOCK_V=params["block_v"],
            NUM_M_BLOCKS=triton.cdiv(summary_count, params["block_m"]),
            NUM_K_BLOCKS=triton.cdiv(state_key_dim, params["block_k"]),
            num_warps=4,
            num_stages=1,
        )
    prob_to_save = prob[:, :, :original_summary_count].contiguous() if summary_count != original_summary_count else prob
    return out, lse, prob_to_save, p_null


def _run_custom_tile_backward(
    *,
    dout: torch.Tensor,
    out: torch.Tensor,
    prob: torch.Tensor,
    p_null: torch.Tensor,
    dlocal_prob: torch.Tensor | None,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    null_logits: torch.Tensor | None,
    route_scale: float,
    read_scale: float,
    needs: tuple[bool, ...],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None:
    if not _env_bool("HSD_FUSED_GLOBAL_READ_TILE_CUSTOM", True):
        return None
    if triton is None:
        return None
    batch, text_len, heads, state_key_dim = read_q.shape
    route_key_dim = int(router_q.shape[-1])
    summary_count = int(states.shape[1])
    value_dim = int(states.shape[-1])
    if batch != 1:
        return None
    need_dread = bool(needs[0])
    need_drouter = bool(needs[1])
    need_dstates = bool(needs[2])
    need_dsummary_k = bool(needs[3])
    need_dnull = bool(null_logits is not None and needs[7])
    if not (need_dread or need_drouter or need_dstates or need_dsummary_k or need_dnull):
        return None

    params = _tile_params(text_len, summary_count, state_key_dim, route_key_dim, value_dim)
    speed_block_t = _speed_block_t(params["block_t"]) if _tile_speed_enabled() else params["block_t"]
    num_t_blocks = triton.cdiv(text_len, params["block_t"])
    num_router_t_blocks = triton.cdiv(text_len, speed_block_t)
    num_m_blocks = triton.cdiv(max(1, summary_count), params["block_m"])
    num_k_blocks = triton.cdiv(state_key_dim, params["block_k"])
    num_route_k_blocks = triton.cdiv(route_key_dim, params["block_route_k"])
    num_v_blocks = triton.cdiv(value_dim, params["block_v"])

    delta = torch.empty(text_len, heads, device=read_q.device, dtype=torch.float32)
    dnull = torch.empty_like(null_logits) if need_dnull else None
    dread = None
    drouter = None
    dstates = None
    dsummary_k = None
    dz = None
    dz_from_tilelang = False

    null_grad_arg = dnull if dnull is not None else read_q.new_empty((0,), dtype=read_q.dtype)
    has_dlocal = dlocal_prob is not None and dlocal_prob.numel() > 0
    dlocal_arg = (
        dlocal_prob.contiguous().view(text_len, heads)
        if has_dlocal else read_q.new_empty((0,), dtype=torch.float32)
    )
    with torch.cuda.device(read_q.device):
        _bwd_delta_kernel[(num_router_t_blocks, heads)](
            dout,
            out,
            p_null,
            dlocal_arg,
            delta,
            null_grad_arg,
            T=int(text_len),
            H=int(heads),
            V=int(value_dim),
            HAS_NULL=bool(null_logits is not None),
            HAS_DLOCAL=bool(has_dlocal),
            NEED_DNULL=bool(need_dnull),
            BLOCK_T=speed_block_t,
            BLOCK_V=params["block_v"],
            NUM_V_BLOCKS=num_v_blocks,
            num_warps=4,
            num_stages=1,
        )
        if summary_count > 0 and (need_drouter or need_dsummary_k):
            if _env_bool("HSD_FUSED_GLOBAL_READ_TILE_ROUTER_TILELANG", True):
                try:
                    dz = _run_dz_tilelang(
                        read_q=read_q,
                        states=states,
                        dout=dout,
                        prob=prob,
                        delta=delta,
                        read_scale=read_scale,
                        block_t=speed_block_t,
                        block_m=params["block_m"],
                        block_k=params["block_k"],
                        block_v=params["block_v"],
                    )
                except Exception:
                    if _env_bool("HSD_FUSED_GLOBAL_READ_TILE_ROUTER_TILELANG_STRICT", False):
                        raise
                    dz = None
                dz_from_tilelang = dz is not None
                if dz is None and _env_bool("HSD_FUSED_GLOBAL_READ_TILE_ROUTER_TILELANG_STRICT", False):
                    raise RuntimeError("HSD tile global read TileLang dz kernel missed.")
            if dz is None:
                dz = torch.empty(text_len, heads, summary_count, device=read_q.device, dtype=torch.float32)
        if summary_count > 0 and dz is not None and not dz_from_tilelang:
            _bwd_dz_kernel[(num_m_blocks, num_router_t_blocks, heads)](
                read_q,
                states,
                dout,
                prob,
                delta,
                dz,
                T=int(text_len),
                M=int(summary_count),
                H=int(heads),
                K=int(state_key_dim),
                V=int(value_dim),
                READ_SCALE=float(read_scale),
                BLOCK_T=speed_block_t,
                BLOCK_M=params["block_m"],
                BLOCK_K=params["block_k"],
                BLOCK_V=params["block_v"],
                NUM_K_BLOCKS=num_k_blocks,
                NUM_V_BLOCKS=num_v_blocks,
                num_warps=4,
                num_stages=1,
            )
        if summary_count > 0 and need_dread:
            if _tile_speed_backward_enabled():
                try:
                    dread = _run_dread_speed_tilelang(
                        dout=dout,
                        states=states,
                        prob=prob,
                        read_scale=read_scale,
                        key_dim=state_key_dim,
                        block_t=speed_block_t,
                        block_k=params["block_k"],
                        block_v=params["block_v"],
                    )
                except Exception:
                    if _tile_speed_strict():
                        raise
                    dread = None
                if dread is None and _tile_speed_strict():
                    raise RuntimeError("HSD tile-speed global read dread kernel missed.")
            if dread is None and _env_bool("HSD_FUSED_GLOBAL_READ_TILE_VALUE_TILELANG", True):
                try:
                    dread = _run_dread_tilelang(
                        dout=dout,
                        states=states,
                        prob=prob,
                        read_scale=read_scale,
                        key_dim=state_key_dim,
                        block_t=params["block_t"],
                        block_k=params["block_k"],
                        block_v=params["block_v"],
                    )
                except Exception:
                    if _env_bool("HSD_FUSED_GLOBAL_READ_TILE_VALUE_TILELANG_STRICT", False):
                        raise
                    dread = None
                if dread is None and _env_bool("HSD_FUSED_GLOBAL_READ_TILE_VALUE_TILELANG_STRICT", False):
                    raise RuntimeError("HSD tile global read TileLang dread kernel missed.")
            if dread is None:
                dread = torch.empty_like(read_q)
                _bwd_dread_kernel[(num_k_blocks, num_t_blocks, heads)](
                    dout,
                    states,
                    prob,
                    dread,
                    T=int(text_len),
                    M=int(summary_count),
                    H=int(heads),
                    K=int(state_key_dim),
                    V=int(value_dim),
                    READ_SCALE=float(read_scale),
                    BLOCK_T=params["block_t"],
                    BLOCK_M=params["block_m"],
                    BLOCK_K=params["block_k"],
                    BLOCK_V=params["block_v"],
                    NUM_M_BLOCKS=num_m_blocks,
                    NUM_V_BLOCKS=num_v_blocks,
                    num_warps=4,
                    num_stages=1,
                        )
        if summary_count > 0 and need_dstates:
            if _tile_speed_backward_enabled():
                try:
                    dstates = _run_dstates_speed_tilelang(
                        read_q=read_q,
                        dout=dout,
                        prob=prob,
                        states=states,
                        read_scale=read_scale,
                        block_t=speed_block_t,
                        block_k=params["block_k"],
                        block_v=params["block_v"],
                    )
                except Exception:
                    if _tile_speed_strict():
                        raise
                    dstates = None
                if dstates is None and _tile_speed_strict():
                    raise RuntimeError("HSD tile-speed global read dstates kernel missed.")
            if dstates is None and _env_bool("HSD_FUSED_GLOBAL_READ_TILE_VALUE_TILELANG", True):
                try:
                    dstates = _run_dstates_tilelang(
                        read_q=read_q,
                        dout=dout,
                        prob=prob,
                        states=states,
                        read_scale=read_scale,
                        block_t=params["block_t"],
                        block_k=params["block_k"],
                        block_v=params["block_v"],
                    )
                except Exception:
                    if _env_bool("HSD_FUSED_GLOBAL_READ_TILE_VALUE_TILELANG_STRICT", False):
                        raise
                    dstates = None
                if dstates is None and _env_bool("HSD_FUSED_GLOBAL_READ_TILE_VALUE_TILELANG_STRICT", False):
                    raise RuntimeError("HSD tile global read TileLang dstates kernel missed.")
            if dstates is None:
                dstates = torch.empty_like(states)
                _bwd_dstates_kernel[(num_v_blocks, num_k_blocks, summary_count * heads)](
                    read_q,
                    dout,
                    prob,
                    dstates,
                    T=int(text_len),
                    M=int(summary_count),
                    H=int(heads),
                    K=int(state_key_dim),
                    V=int(value_dim),
                    READ_SCALE=float(read_scale),
                    BLOCK_T=params["block_t"],
                    BLOCK_K=params["block_k"],
                    BLOCK_V=params["block_v"],
                    NUM_T_BLOCKS=num_t_blocks,
                    num_warps=4,
                    num_stages=1,
                )
        if summary_count > 0 and need_drouter:
            if _env_bool("HSD_FUSED_GLOBAL_READ_TILE_ROUTER_TILELANG", True):
                try:
                    drouter = _run_drouter_tilelang(
                        dz=dz,
                        summary_k=summary_k,
                        router_q=router_q,
                        route_scale=route_scale,
                        block_t=speed_block_t,
                        block_m=params["block_m"],
                        block_k=params["block_route_k"],
                    )
                except Exception:
                    if _env_bool("HSD_FUSED_GLOBAL_READ_TILE_ROUTER_TILELANG_STRICT", False):
                        raise
                    drouter = None
                if drouter is None and _env_bool("HSD_FUSED_GLOBAL_READ_TILE_ROUTER_TILELANG_STRICT", False):
                    raise RuntimeError("HSD tile global read TileLang drouter kernel missed.")
            if drouter is None:
                drouter = torch.empty_like(router_q)
                _bwd_drouter_kernel[(num_route_k_blocks, num_router_t_blocks, heads)](
                    dz,
                    summary_k,
                    drouter,
                    T=int(text_len),
                    M=int(summary_count),
                    H=int(heads),
                    K=int(route_key_dim),
                    ROUTE_SCALE=float(route_scale),
                    BLOCK_T=speed_block_t,
                    BLOCK_M=params["block_m"],
                    BLOCK_K=params["block_route_k"],
                    NUM_M_BLOCKS=num_m_blocks,
                    num_warps=4,
                    num_stages=1,
                )
        if summary_count > 0 and need_dsummary_k:
            if _env_bool("HSD_FUSED_GLOBAL_READ_TILE_ROUTER_TILELANG", True):
                try:
                    dsummary_k = _run_dsummary_k_tilelang(
                        router_q=router_q,
                        dz=dz,
                        summary_k=summary_k,
                        route_scale=route_scale,
                        block_t=speed_block_t,
                        block_m=params["block_m"],
                        block_k=params["block_route_k"],
                    )
                except Exception:
                    if _env_bool("HSD_FUSED_GLOBAL_READ_TILE_ROUTER_TILELANG_STRICT", False):
                        raise
                    dsummary_k = None
                if dsummary_k is None and _env_bool("HSD_FUSED_GLOBAL_READ_TILE_ROUTER_TILELANG_STRICT", False):
                    raise RuntimeError("HSD tile global read TileLang dsummary_k kernel missed.")
            if dsummary_k is None:
                dsummary_k = torch.empty_like(summary_k)
                _bwd_dsummary_k_kernel[(num_route_k_blocks, summary_count, heads)](
                    router_q,
                    dz,
                    dsummary_k,
                    T=int(text_len),
                    M=int(summary_count),
                    H=int(heads),
                    K=int(route_key_dim),
                    ROUTE_SCALE=float(route_scale),
                    BLOCK_T=speed_block_t,
                    BLOCK_K=params["block_route_k"],
                    NUM_T_BLOCKS=num_router_t_blocks,
                    num_warps=4,
                    num_stages=1,
                )

    if summary_count <= 0:
        if need_dread and dread is None:
            dread = torch.zeros_like(read_q)
        if need_drouter and drouter is None:
            drouter = torch.zeros_like(router_q)
        if need_dstates and dstates is None:
            dstates = torch.zeros_like(states)
        if need_dsummary_k and dsummary_k is None:
            dsummary_k = torch.zeros_like(summary_k)
    return dread, drouter, dstates, dsummary_k, dnull


def _online_debug_sample_indices(
    text_len: int,
    sample_tokens: int,
    *,
    device: torch.device,
    policy: str = _ONLINE_DEBUG_SAMPLING_POLICY,
) -> torch.Tensor:
    """Return the deterministic token sample for scalar-only route metrics."""

    text_len = int(text_len)
    sample_tokens = min(text_len, max(1, int(sample_tokens)))
    if policy != _ONLINE_DEBUG_SAMPLING_POLICY:
        raise RuntimeError(
            "Unsupported HSD online route-monitor sampling policy "
            f"{policy!r}; expected {_ONLINE_DEBUG_SAMPLING_POLICY!r}."
        )
    if text_len <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if sample_tokens >= text_len:
        return torch.arange(text_len, device=device, dtype=torch.long)
    if sample_tokens == 1:
        return torch.tensor([text_len // 2], device=device, dtype=torch.long)
    # Integer-only pair anchors are stable across devices/PyTorch versions.
    # At least one adjacent pair is always present.  Uniform anchors include
    # 0 and T-2, so their successors include both sequence endpoints.
    pair_count = sample_tokens // 2
    if pair_count == 1:
        anchors = torch.tensor([0], device=device, dtype=torch.long)
    else:
        ordinal = torch.arange(pair_count, device=device, dtype=torch.long)
        anchors = torch.div(
            ordinal * (text_len - 2),
            pair_count - 1,
            rounding_mode="floor",
        )
    indices = torch.stack((anchors, anchors + 1), dim=-1).flatten()
    if sample_tokens % 2:
        indices = torch.cat((indices, indices.new_tensor([text_len // 2])))
    return indices.unique(sorted=True)


def _record_online_blockwise_debug_stats(
    *,
    router_q: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    token_summary_start: torch.Tensor,
    token_summary_end: torch.Tensor,
    lse: torch.Tensor,
    p_null: torch.Tensor,
    has_null: bool,
    route_scale: float,
    debug_stats_callback: Callable[..., None] | None,
    sample_tokens: int = _ONLINE_DEBUG_DEFAULT_SAMPLE_TOKENS,
    block_tokens: int = _ONLINE_DEBUG_DEFAULT_BLOCK_TOKENS,
    sampling_policy: str = _ONLINE_DEBUG_SAMPLING_POLICY,
) -> None:
    """Collect Dense route metrics with O(block_T * H * M) scratch.

    The training forward already produced ``lse`` and ``p_null``.  Monitoring
    recomputes logits only for a deterministic token sample and immediately
    reduces every block into scalar sums plus two O(M) usage histograms.  The
    callback therefore receives scalars only and cannot retain route weights.
    This function is under ``no_grad`` and never changes the autograd graph or
    the strict tile-online dispatch path.
    """

    if debug_stats_callback is None:
        return
    if router_q.ndim != 4 or summary_k.ndim != 4:
        raise RuntimeError(
            "HSD scalar route monitor expects router_q/summary_k as "
            f"[1,T,H,R]/[1,M,H,R], got {router_q.shape}/{summary_k.shape}."
        )
    batch, text_len, heads, route_dim = router_q.shape
    summary_count = int(summary_k.shape[1])
    if batch != 1 or tuple(summary_k.shape) != (1, summary_count, heads, route_dim):
        raise RuntimeError("HSD scalar route monitor requires batch=1 and matching route heads/dimensions.")
    if tuple(lse.shape) != (text_len, heads) or tuple(p_null.shape) != (text_len, heads):
        raise RuntimeError(
            "HSD scalar route monitor lse/p_null shape mismatch: "
            f"lse={tuple(lse.shape)} p_null={tuple(p_null.shape)} expected={(text_len, heads)}."
        )
    if tuple(token_summary_start.shape) != (text_len,) or tuple(token_summary_end.shape) != (text_len,):
        raise RuntimeError("HSD scalar route monitor token-summary ranges must be [T].")
    if boundary_counts.ndim != 1 or int(boundary_counts.numel()) != summary_count:
        raise RuntimeError("HSD scalar route monitor boundary count must match M.")
    if summary_count <= 0 or text_len <= 0:
        return
    sample_tokens = min(text_len, max(1, int(sample_tokens)))
    block_tokens = min(sample_tokens, max(1, int(block_tokens)))
    sample_index = _online_debug_sample_indices(
        text_len,
        sample_tokens,
        device=router_q.device,
        policy=sampling_policy,
    )
    sampled_count = int(sample_index.numel())
    if sampled_count <= 0:
        return

    with torch.no_grad():
        router = router_q.detach()
        keys = summary_k[0].detach().float()
        boundaries = boundary_counts.detach().to(device=router.device, dtype=torch.long)
        summary_index = torch.arange(summary_count, device=router.device, dtype=torch.long)
        zero = torch.zeros((), device=router.device, dtype=torch.float32)
        row_count = zero.clone()
        token_count = zero.clone()
        null_sum = zero.clone()
        null_max = zero.clone()
        summary_mass_sum = zero.clone()
        history_head_gt_sum = zero.clone()
        history_token_any_gt_sum = zero.clone()
        history_token_mean_gt_sum = zero.clone()
        null_top1_sum = zero.clone()
        entropy_sum = zero.clone()
        effective_count_sum = zero.clone()
        top1_prob_sum = zero.clone()
        top4_mass_sum = zero.clone()
        valid_summary_count_sum = zero.clone()
        weighted_age_sum = zero.clone()
        margin_sum = zero.clone()
        margin_count = zero.clone()
        oldest_sum = zero.clone()
        latest_sum = zero.clone()
        all_heads_same_sum = zero.clone()
        switch_sum = zero.clone()
        switch_count = zero.clone()
        top1_usage_count = torch.zeros(summary_count, device=router.device, dtype=torch.float32)
        mass_usage_sum = torch.zeros_like(top1_usage_count)
        visible_summary_mask = torch.zeros(summary_count, device=router.device, dtype=torch.bool)
        previous_index: torch.Tensor | None = None
        previous_top1: torch.Tensor | None = None
        previous_valid: torch.Tensor | None = None

        for block_start in range(0, sampled_count, block_tokens):
            indices = sample_index[block_start:block_start + block_tokens]
            block_size = int(indices.numel())
            route_block = router[0].index_select(0, indices).float()
            logits = torch.einsum("thr,mhr->thm", route_block, keys) * float(route_scale)
            summary_starts = token_summary_start.index_select(0, indices).to(torch.long)
            summary_ends = token_summary_end.index_select(0, indices).to(torch.long)
            valid_prefix = (
                (summary_index.view(1, -1) >= summary_starts.view(-1, 1))
                & (summary_index.view(1, -1) < summary_ends.view(-1, 1))
                & (boundaries.view(1, -1) <= indices.view(-1, 1))
            )
            token_valid = valid_prefix.any(dim=-1)
            visible_summary_mask |= valid_prefix.any(dim=0)
            row_valid = token_valid.view(-1, 1).expand(block_size, heads)
            row_valid_f = row_valid.float()
            block_rows = row_valid_f.sum()
            block_tokens_valid = token_valid.float().sum()
            row_count += block_rows
            token_count += block_tokens_valid

            masked_logits = logits.masked_fill(~valid_prefix[:, None, :], -torch.inf)
            # Do not recover the summary-conditional distribution by dividing
            # joint probabilities by (1-p_null): a dominant null logit can
            # underflow every joint summary probability. Softmax the masked
            # summary logits directly, then zero rows with no visible summary.
            safe_summary_logits = torch.where(
                token_valid.view(-1, 1, 1),
                masked_logits,
                torch.zeros_like(masked_logits),
            )
            conditional_weights = torch.softmax(safe_summary_logits, dim=-1)
            conditional_weights = torch.where(
                valid_prefix[:, None, :],
                conditional_weights,
                torch.zeros_like(conditional_weights),
            )
            if has_null:
                block_null = p_null.index_select(0, indices).detach().float()
                # Use the actual probability emitted by the production kernel
                # for null/summary mass health rather than a monitor recompute.
                block_summary_mass = (1.0 - block_null).clamp(0.0, 1.0)
                null_sum += (block_null * row_valid_f).sum()
                null_max = torch.maximum(
                    null_max,
                    block_null.masked_fill(~row_valid, 0.0).max(),
                )
                summary_mass_sum += (block_summary_mass * row_valid_f).sum()
                history_head_gt_sum += (
                    (block_summary_mass > 0.10) & row_valid
                ).float().sum()
                if block_size:
                    token_max_mass = block_summary_mass.masked_fill(~row_valid, 0.0).max(dim=-1).values
                    token_mean_mass = (
                        block_summary_mass * row_valid_f
                    ).sum(dim=-1) / row_valid_f.sum(dim=-1).clamp_min(1.0)
                    history_token_any_gt_sum += (
                        (token_max_mass > 0.10) & token_valid
                    ).float().sum()
                    history_token_mean_gt_sum += (
                        (token_mean_mass > 0.10) & token_valid
                    ).float().sum()
                summary_top1_joint = conditional_weights.max(dim=-1).values * block_summary_mass
                null_top1_sum += ((block_null >= summary_top1_joint) & row_valid).float().sum()
            else:
                block_summary_mass = row_valid_f
                summary_mass_sum += block_rows

            eps = torch.finfo(torch.float32).tiny
            log_weights = conditional_weights.clamp_min(eps).log()
            entropy = -(conditional_weights * log_weights).sum(dim=-1)
            topk = min(4, summary_count)
            top_values, top_indices = conditional_weights.topk(k=topk, dim=-1)
            top1_prob = top_values[..., 0]
            top1_index = top_indices[..., 0]
            entropy_sum += (entropy * row_valid_f).sum()
            effective_count_sum += (entropy.exp() * row_valid_f).sum()
            top1_prob_sum += (top1_prob * row_valid_f).sum()
            top4_mass_sum += (top_values.sum(dim=-1) * row_valid_f).sum()
            valid_counts = valid_prefix.sum(dim=-1).float()
            valid_summary_count_sum += (valid_counts * token_valid.float()).sum() * float(heads)
            weighted_boundary = (
                conditional_weights * boundaries.float().view(1, 1, -1)
            ).sum(dim=-1)
            weighted_age = indices.float().view(-1, 1) - weighted_boundary
            weighted_age_sum += (weighted_age * row_valid_f).sum()

            top2_logits = masked_logits.topk(k=min(2, summary_count), dim=-1).values
            if top2_logits.shape[-1] >= 2:
                margin_mask = row_valid & (valid_counts.view(-1, 1) >= 2)
                margin = torch.nan_to_num(
                    top2_logits[..., 0] - top2_logits[..., 1],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                margin_sum += (margin * margin_mask.float()).sum()
                margin_count += margin_mask.float().sum()

            selected = top1_index[row_valid].long()
            if selected.numel() > 0:
                top1_usage_count += torch.bincount(
                    selected,
                    minlength=summary_count,
                ).to(dtype=torch.float32)
            mass_usage_sum += conditional_weights.sum(dim=(0, 1))
            oldest_index = summary_starts.view(-1, 1).expand_as(top1_index)
            latest_index = torch.where(
                valid_prefix,
                summary_index.view(1, -1),
                torch.full_like(summary_index.view(1, -1), -1),
            ).max(dim=-1).values.view(-1, 1).expand_as(top1_index)
            oldest_sum += ((top1_index == oldest_index) & row_valid).float().sum()
            latest_sum += ((top1_index == latest_index) & row_valid).float().sum()
            all_heads_same = (top1_index == top1_index[..., :1]).all(dim=-1)
            all_heads_same_sum += (all_heads_same & token_valid).float().sum()

            if block_size > 1:
                adjacent = indices[1:] == indices[:-1] + 1
                adjacent_rows = (
                    adjacent.view(-1, 1)
                    & row_valid[1:]
                    & row_valid[:-1]
                    & (summary_starts[1:] == summary_starts[:-1]).view(-1, 1)
                )
                switch_sum += (
                    (top1_index[1:] != top1_index[:-1]) & adjacent_rows
                ).float().sum()
                switch_count += adjacent_rows.float().sum()
            if previous_index is not None and previous_top1 is not None and previous_valid is not None:
                cross_adjacent = indices[0] == previous_index + 1
                cross_rows = previous_valid & row_valid[0] & cross_adjacent
                switch_sum += ((top1_index[0] != previous_top1) & cross_rows).float().sum()
                switch_count += cross_rows.float().sum()
            previous_index = indices[-1]
            previous_top1 = top1_index[-1]
            previous_valid = row_valid[-1]

            # Explicitly drop O(block_T*H*M) scratch before the next block.
            del route_block, logits, masked_logits, conditional_weights

        monitor_contract = {
            "contract_version": _ONLINE_DEBUG_CONTRACT_VERSION,
            "sampling_policy": _ONLINE_DEBUG_SAMPLING_POLICY,
            "sampling_policy_id": _ONLINE_DEBUG_SAMPLING_POLICY_ID,
            "sample_tokens": sampled_count,
            "block_tokens": block_tokens,
            "total_tokens": text_len,
            "scalar_only": True,
        }
        monitor_stats = {
            "router_monitor_contract_version": zero.new_tensor(
                float(_ONLINE_DEBUG_CONTRACT_VERSION)
            ),
            "router_monitor_sampling_policy_id": zero.new_tensor(
                float(_ONLINE_DEBUG_SAMPLING_POLICY_ID)
            ),
            "router_monitor_scalar_only": zero.new_tensor(1.0),
            "router_monitor_total_tokens": zero.new_tensor(float(text_len)),
            "router_monitor_sample_tokens": zero.new_tensor(float(sampled_count)),
            "router_monitor_sample_fraction": zero.new_tensor(
                float(sampled_count) / float(text_len)
            ),
            "router_monitor_block_tokens": zero.new_tensor(float(block_tokens)),
            "router_monitor_max_block_route_elements": zero.new_tensor(
                float(min(block_tokens, sampled_count) * heads * summary_count)
            ),
            "router_monitor_valid_rows": row_count,
        }
        if not bool((row_count > 0).item()):
            # The deterministic token sample can legitimately contain no
            # readable summary (for example, a TP pair whose packed documents
            # are all shorter than one summary chunk).  Publish the monitor
            # contract and an explicit zero valid-row count so the production
            # health gate can distinguish this data-dependent case from a
            # broken or disabled callback.  Distribution metrics remain
            # absent because entropy/usage are undefined for an empty sample.
            debug_stats_callback(
                scalar_stats=monitor_stats,
                monitor_contract=monitor_contract,
                record_debug_stats=True,
            )
            return

        row_denom = row_count
        token_denom = token_count.clamp_min(1.0)
        top1_usage = top1_usage_count / row_denom
        mass_usage = mass_usage_sum / mass_usage_sum.sum().clamp_min(1.0)
        eps = torch.finfo(torch.float32).tiny
        top1_usage_entropy = -(top1_usage * top1_usage.clamp_min(eps).log()).sum()
        mass_usage_entropy = -(mass_usage * mass_usage.clamp_min(eps).log()).sum()
        summary_k_norm = torch.linalg.vector_norm(keys, dim=-1).mean(dim=1)
        # Fixed-M Stage-3 padding has boundary=text_len and therefore receives
        # no route mass. Restrict key-health statistics to summaries that were
        # actually visible in this deterministic sample.
        visible_summary_norm = summary_k_norm[visible_summary_mask]
        scalar_stats = {
            "router_null_prob": null_sum / row_denom if has_null else zero,
            "router_null_prob_max": null_max if has_null else zero,
            "router_summary_mass": summary_mass_sum / row_denom,
            "router_history_head_share_gt_0p10": history_head_gt_sum / row_denom if has_null else zero,
            "router_history_token_share_any_head_gt_0p10": (
                history_token_any_gt_sum / token_denom if has_null else zero
            ),
            "router_history_token_share_mean_head_gt_0p10": (
                history_token_mean_gt_sum / token_denom if has_null else zero
            ),
            "router_null_top1_share": null_top1_sum / row_denom if has_null else zero,
            "router_entropy": entropy_sum / row_denom,
            "router_top1_prob": top1_prob_sum / row_denom,
            "router_valid_summary_count": valid_summary_count_sum / row_denom,
            "router_effective_summary_count": effective_count_sum / row_denom,
            "router_top4_mass": top4_mass_sum / row_denom,
            # Dense routing has no post-softmax TopK truncation, so pre/post
            # metrics are identical by contract.
            "router_pre_topk_entropy": entropy_sum / row_denom,
            "router_pre_topk_effective_summary_count": effective_count_sum / row_denom,
            "router_pre_topk_top1_prob": top1_prob_sum / row_denom,
            "router_weighted_age": weighted_age_sum / row_denom,
            "router_top1_margin": margin_sum / margin_count.clamp_min(1.0),
            "router_top1_usage_effective_summary_count": top1_usage_entropy.exp(),
            "router_top1_max_share": top1_usage.max(),
            "router_mass_usage_effective_summary_count": mass_usage_entropy.exp(),
            "router_mass_max_share": mass_usage.max(),
            "router_top1_oldest_share": oldest_sum / row_denom,
            "router_top1_latest_share": latest_sum / row_denom,
            "router_head_all_same_share": all_heads_same_sum / token_denom,
            "router_top1_switch_rate": switch_sum / switch_count.clamp_min(1.0),
            "router_summary_k_norm_oldest": visible_summary_norm[0],
            "router_summary_k_norm_latest": visible_summary_norm[-1],
            "router_summary_k_norm_max_over_mean": (
                visible_summary_norm.max() / visible_summary_norm.mean().clamp_min(eps)
            ),
            **monitor_stats,
        }
        debug_stats_callback(
            scalar_stats=scalar_stats,
            monitor_contract=monitor_contract,
            record_debug_stats=True,
        )


def _record_tile_debug_stats(
    *,
    router_q: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    prob: torch.Tensor,
    p_null: torch.Tensor,
    has_null: bool,
    route_scale: float,
    debug_stats_callback: Callable[..., None] | None,
) -> None:
    if debug_stats_callback is None:
        return
    if prob.numel() == 0 or prob.shape[-1] == 0:
        return
    with torch.no_grad():
        text_len = int(prob.shape[0])
        heads = int(prob.shape[1])
        summary_count = int(prob.shape[2])
        device = prob.device
        token_index = torch.arange(text_len, device=device, dtype=torch.long)
        summary_index = torch.arange(summary_count, device=device, dtype=torch.long)
        text_ends = text_cu_seqlens[1:].to(device=device, dtype=torch.long)
        summary_ends = summary_cu_seqlens[1:].to(device=device, dtype=torch.long)
        token_doc = torch.searchsorted(text_ends, token_index, right=False)
        summary_doc = torch.searchsorted(summary_ends, summary_index, right=False)
        boundaries = boundary_counts[:summary_count].to(device=device, dtype=torch.long)
        valid_prefix = (
            (token_doc.view(text_len, 1) == summary_doc.view(1, summary_count))
            & (boundaries.view(1, summary_count) <= token_index.view(text_len, 1))
        )
        valid = valid_prefix.view(1, text_len, 1, summary_count).expand(1, text_len, heads, summary_count)
        has_valid = valid_prefix.any(dim=-1).view(1, text_len, 1, 1).expand(1, text_len, heads, 1)
        logits = torch.einsum(
            "bthk,bmhk->bthm",
            router_q.float(),
            summary_k[:, :summary_count].float(),
        ) * float(route_scale)
        router_debug_logits = logits.masked_fill(~valid, -torch.inf)
        weights = prob[:, :, :summary_count].float().unsqueeze(0)
        null_prob = None
        summary_mass = None
        debug_weights = weights
        if has_null:
            summary_mass = weights.sum(dim=-1, keepdim=True)
            debug_weights = torch.where(
                summary_mass > 0,
                weights / summary_mass.clamp_min(torch.finfo(torch.float32).tiny),
                torch.zeros_like(weights),
            )
            if p_null.numel() > 0:
                null_prob = p_null[:text_len, :heads].float().unsqueeze(0)
        debug_stats_callback(
            weights=debug_weights,
            logits=router_debug_logits,
            summary_k=summary_k[:, :summary_count],
            valid_prefix=valid_prefix,
            has_valid=has_valid,
            boundaries=boundaries,
            token_index=token_index,
            null_prob=null_prob,
            summary_mass=summary_mass,
            record_debug_stats=True,
        )


def _supports_tile_weighted_global_read(
    read_q: torch.Tensor,
    states: torch.Tensor,
    summary_weights: torch.Tensor,
) -> bool:
    if tilelang is None or triton is None or not read_q.is_cuda:
        return False
    if read_q.ndim != 4 or states.ndim != 5 or summary_weights.ndim != 4:
        return False
    batch, text_len, heads, key_dim = read_q.shape
    summary_count = int(states.shape[1])
    if batch != 1 or text_len <= 1 or summary_count <= 0:
        return False
    if tuple(states.shape[:4]) != (batch, summary_count, heads, key_dim):
        return False
    if tuple(summary_weights.shape) != (batch, text_len, heads, summary_count):
        return False
    if read_q.device != states.device or read_q.device != summary_weights.device:
        return False
    if states.dtype != torch.float32 or summary_weights.dtype != torch.float32:
        return False
    # Keep FP32 on the existing tf32x3 Triton implementation so this
    # optimization cannot weaken its stricter 1e-4/1e-5 accuracy contract.
    if read_q.dtype not in (torch.float16, torch.bfloat16):
        return False
    # The dState and dWeight tensor-core GEMMs split their output-column tile
    # across four warps.  TileLang requires at least one 8-column MMA tile per
    # warp, so BV=16 is not a valid speed-kernel shape (BV>=32 is).  Keep small
    # synthetic/model shapes on the precise Triton implementation instead of
    # letting strict mode reach a TileLang compile-time assertion.
    if key_dim < 16 or key_dim > 256 or int(states.shape[-1]) < 32:
        return False
    return True


class _HSDTileWeightedGlobalReadFunction(torch.autograd.Function):
    """TileSpeed value read for probabilities produced outside the kernel."""

    @staticmethod
    def forward(
        ctx,
        read_q: torch.Tensor,
        states: torch.Tensor,
        summary_weights: torch.Tensor,
        read_scale: float,
    ) -> torch.Tensor:
        _, text_len, _, key_dim = read_q.shape
        summary_count = int(states.shape[1])
        value_dim = int(states.shape[-1])
        params = _tile_params(text_len, summary_count, key_dim, key_dim, value_dim)
        block_t = _speed_block_t(params["block_t"])
        prob = summary_weights[0].float().contiguous()
        original_summary_count = summary_count
        padded_summary_count = _speed_summary_bucket_count(
            summary_count,
            _speed_block_m(summary_count),
            text_len,
        )
        if padded_summary_count != summary_count:
            states = _pad_summary_dim(states, padded_summary_count, dim=1)
            prob = _pad_summary_dim(prob, padded_summary_count, dim=2)
            summary_count = padded_summary_count
        read_q_c = read_q.contiguous()
        states_c = states.contiguous()
        out = _run_read_out_speed_tilelang(
            read_q=read_q_c,
            states=states_c,
            prob=prob,
            read_scale=float(read_scale),
            block_t=block_t,
            block_v=params["block_v"],
        )
        if out is None:
            raise RuntimeError("HSD momentum TileSpeed weighted-read forward kernel missed.")
        ctx.read_scale = float(read_scale)
        ctx.original_summary_count = original_summary_count
        ctx.save_for_backward(read_q_c, states_c, prob)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        read_q, states, prob = ctx.saved_tensors
        _, text_len, _, key_dim = read_q.shape
        summary_count = int(states.shape[1])
        value_dim = int(states.shape[-1])
        params = _tile_params(text_len, summary_count, key_dim, key_dim, value_dim)
        block_t = _speed_block_t(params["block_t"])
        needs = ctx.needs_input_grad
        dout = dout.contiguous()

        dread = None
        if needs[0]:
            dread = _run_dread_speed_tilelang(
                dout=dout,
                states=states,
                prob=prob,
                read_scale=ctx.read_scale,
                key_dim=key_dim,
                block_t=block_t,
                block_k=params["block_k"],
                block_v=params["block_v"],
            )
            if dread is None:
                raise RuntimeError("HSD momentum TileSpeed weighted-read dQ kernel missed.")

        dstates = None
        if needs[1]:
            dstates = _run_dstates_speed_tilelang(
                read_q=read_q,
                dout=dout,
                prob=prob,
                states=states,
                read_scale=ctx.read_scale,
                block_t=block_t,
                block_k=params["block_k"],
                block_v=params["block_v"],
            )
            if dstates is None:
                raise RuntimeError("HSD momentum TileSpeed weighted-read dState kernel missed.")
            if summary_count != ctx.original_summary_count:
                dstates = dstates[:, :ctx.original_summary_count].contiguous()

        dweights = None
        if needs[2]:
            dweight = _run_dweight_tilelang(
                read_q=read_q,
                states=states,
                dout=dout,
                prob=prob,
                read_scale=ctx.read_scale,
                block_t=block_t,
                block_m=params["block_m"],
                block_k=params["block_k"],
                block_v=params["block_v"],
            )
            if dweight is None:
                raise RuntimeError("HSD momentum TileSpeed weighted-read dWeight kernel missed.")
            if summary_count != ctx.original_summary_count:
                dweight = dweight[:, :, :ctx.original_summary_count].contiguous()
            dweights = dweight.unsqueeze(0)
        return dread, dstates, dweights, None


def hsd_tile_weighted_global_read(
    *,
    read_q: torch.Tensor,
    states: torch.Tensor,
    summary_weights: torch.Tensor,
    read_scale: float,
) -> torch.Tensor | None:
    """Consume externally smoothed FP32 summary weights with TileSpeed kernels."""
    if not _tile_speed_enabled() or not _tile_speed_forward_enabled() or not _tile_speed_backward_enabled():
        return None
    if not _supports_tile_weighted_global_read(read_q, states, summary_weights):
        return None
    return _HSDTileWeightedGlobalReadFunction.apply(
        read_q,
        states,
        summary_weights,
        float(read_scale),
    )


class _HSDOnlineGroupedGlobalReadFunction(torch.autograd.Function):
    """Low-memory TileLang online-softmax forward and recompute backward.

    Neither direction allocates dense ``[T,H,M]`` route probabilities or
    logit gradients.  Backward uses the forward log-sum-exp to recompute each
    visible probability inside query- or summary-owned TileLang kernels, and
    the saved pre-cast FP32 history accumulator to evaluate the exact
    softmax-Jacobian common term without another state scan.
    """

    @staticmethod
    def forward(
        ctx,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        summary_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        summary_cu_seqlens: torch.Tensor,
        null_logits: torch.Tensor,
        route_scale: float,
        read_scale: float,
        has_null: bool,
        debug_stats_callback: Callable[..., None] | None,
        profile: str | None,
    ):
        null_arg = null_logits if bool(has_null) else None
        # The route-softmax Jacobian needs the history output before its final
        # BF16 cast.  Save it only for autograd-enabled calls; inference keeps
        # the original BF16 output-only memory footprint.
        save_history_accumulator = any(
            bool(ctx.needs_input_grad[index]) for index in (0, 1, 2, 3, 7)
        )
        outputs = _run_online_grouped_tile_forward(
            read_q=read_q,
            router_q=router_q,
            states=states,
            summary_k=summary_k,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
            null_logits=null_arg,
            route_scale=float(route_scale),
            read_scale=float(read_scale),
            save_history_accumulator=save_history_accumulator,
            profile=profile,
        )
        if outputs is None:
            raise RuntimeError(
                "HSD TileLang online grouped-read backend was requested, "
                "but no exact allowlisted Hopper profile matched the inputs "
                f"(requested profile={profile!r})."
            )
        (
            out,
            history_accumulator,
            lse,
            p_null,
            token_summary_start,
            token_summary_end,
        ) = outputs
        scalar_debug_enabled = _env_bool(
            "HSD_FUSED_GLOBAL_READ_ONLINE_DEBUG_SCALARS",
            # Legacy callers that still set DEBUG_PROB are routed through the
            # scalar collector too.  The online production path never
            # materializes full probabilities, regardless of that old name.
            _env_bool("HSD_FUSED_GLOBAL_READ_ONLINE_DEBUG_PROB", False),
        )
        if debug_stats_callback is not None and scalar_debug_enabled:
            sampling_policy = os.environ.get(
                "HSD_FUSED_GLOBAL_READ_ONLINE_DEBUG_SAMPLING_POLICY",
                _ONLINE_DEBUG_SAMPLING_POLICY,
            ).strip()
            _record_online_blockwise_debug_stats(
                router_q=router_q,
                summary_k=summary_k,
                boundary_counts=boundary_counts,
                token_summary_start=token_summary_start,
                token_summary_end=token_summary_end,
                lse=lse,
                p_null=p_null,
                has_null=bool(has_null),
                route_scale=float(route_scale),
                debug_stats_callback=debug_stats_callback,
                sample_tokens=_env_int(
                    "HSD_FUSED_GLOBAL_READ_ONLINE_DEBUG_SAMPLE_TOKENS",
                    _ONLINE_DEBUG_DEFAULT_SAMPLE_TOKENS,
                ),
                block_tokens=_env_int(
                    "HSD_FUSED_GLOBAL_READ_ONLINE_DEBUG_BLOCK_TOKENS",
                    _ONLINE_DEBUG_DEFAULT_BLOCK_TOKENS,
                ),
                sampling_policy=sampling_policy,
            )
        ctx.route_scale = float(route_scale)
        ctx.read_scale = float(read_scale)
        ctx.has_null = bool(has_null)
        ctx.profile = profile
        ctx.save_for_backward(
            read_q,
            router_q,
            states,
            summary_k,
            boundary_counts,
            text_cu_seqlens,
            summary_cu_seqlens,
            null_logits,
            history_accumulator,
            lse,
            p_null,
            token_summary_start,
            token_summary_end,
        )
        local_prob = (
            p_null.unsqueeze(0).unsqueeze(-1)
            if bool(has_null)
            else out.new_empty((0,), dtype=torch.float32)
        )
        return out, local_prob

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlocal_prob: torch.Tensor | None):
        (
            read_q,
            router_q,
            states,
            summary_k,
            boundary_counts,
            text_cu_seqlens,
            summary_cu_seqlens,
            null_logits,
            history_accumulator,
            lse,
            p_null,
            token_summary_start,
            token_summary_end,
        ) = ctx.saved_tensors
        null_arg = null_logits if ctx.has_null else None
        from .hsd_global_read_tile_online_bwd import (
            hsd_tile_online_backward,
        )

        custom_grads = hsd_tile_online_backward(
            dout=dout.contiguous(),
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
            null_logits=null_arg,
            has_null=ctx.has_null,
            route_scale=ctx.route_scale,
            read_scale=ctx.read_scale,
            needs=ctx.needs_input_grad,
            profile=ctx.profile,
        )
        dread, drouter, dstates, dsummary_k, dnull = custom_grads
        return (
            dread,
            drouter,
            dstates,
            dsummary_k,
            None,
            None,
            None,
            dnull if ctx.has_null else None,
            None,
            None,
            None,
            None,
            None,
        )


class _HSDTileGlobalReadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        read_q: torch.Tensor,
        router_q: torch.Tensor,
        states: torch.Tensor,
        summary_k: torch.Tensor,
        boundary_counts: torch.Tensor,
        text_cu_seqlens: torch.Tensor,
        summary_cu_seqlens: torch.Tensor,
        null_logits: torch.Tensor,
        route_scale: float,
        read_scale: float,
        has_null: bool,
        debug_stats_callback: Callable[..., None] | None,
    ):
        null_arg = null_logits if bool(has_null) else None
        outputs = _run_custom_tile_forward(
            read_q=read_q,
            router_q=router_q,
            states=states,
            summary_k=summary_k,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
            null_logits=null_arg,
            route_scale=float(route_scale),
            read_scale=float(read_scale),
        )
        if outputs is None:
            if not _env_bool("HSD_FUSED_GLOBAL_READ_TILE_TORCH_FALLBACK", False):
                raise RuntimeError("HSD custom tile global read was called for an unsupported input.")
            prob, lse, p_null = _route_probabilities(
                router_q=router_q,
                summary_k=summary_k,
                boundary_counts=boundary_counts,
                text_cu_seqlens=text_cu_seqlens,
                summary_cu_seqlens=summary_cu_seqlens,
                null_logits=null_arg,
                route_scale=float(route_scale),
            )
            out = _tile_forward_impl(read_q=read_q, states=states, prob=prob, read_scale=float(read_scale))
            ctx.used_custom_tile = False
        else:
            out, lse, prob, p_null = outputs
            ctx.used_custom_tile = True
        _record_tile_debug_stats(
            router_q=router_q,
            summary_k=summary_k,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
            prob=prob,
            p_null=p_null if p_null is not None else prob.new_empty(0),
            has_null=bool(has_null),
            route_scale=float(route_scale),
            debug_stats_callback=debug_stats_callback,
        )
        ctx.route_scale = float(route_scale)
        ctx.read_scale = float(read_scale)
        ctx.has_null = bool(has_null)
        ctx.save_for_backward(read_q, router_q, states, summary_k, null_logits, out, prob, lse, p_null if p_null is not None else prob.new_empty(0))
        local_prob = (
            p_null.unsqueeze(0).unsqueeze(-1)
            if bool(has_null) else out.new_empty((0,), dtype=torch.float32)
        )
        return out, local_prob

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlocal_prob: torch.Tensor | None):
        read_q, router_q, states, summary_k, null_logits, out, prob, _lse, p_null = ctx.saved_tensors
        needs = ctx.needs_input_grad
        need_dread = bool(needs[0])
        need_drouter = bool(needs[1])
        need_dstates = bool(needs[2])
        need_dsummary_k = bool(needs[3])
        need_dnull = bool(ctx.has_null and needs[7])
        if not (need_dread or need_drouter or need_dstates or need_dsummary_k or need_dnull):
            return (None, None, None, None, None, None, None, None, None, None, None, None)

        null_arg = null_logits if ctx.has_null else None
        if getattr(ctx, "used_custom_tile", False):
            custom_grads = _run_custom_tile_backward(
                dout=dout.contiguous(),
                out=out,
                prob=prob,
                p_null=p_null,
                dlocal_prob=dlocal_prob,
                read_q=read_q,
                router_q=router_q,
                states=states,
                summary_k=summary_k,
                null_logits=null_arg,
                route_scale=ctx.route_scale,
                read_scale=ctx.read_scale,
                needs=needs,
            )
            if custom_grads is None:
                raise RuntimeError("HSD custom tile global read backward missed after a custom forward hit.")
            dread, drouter, dstates, dsummary_k, dnull = custom_grads
            return (
                dread,
                drouter,
                dstates,
                dsummary_k,
                None,
                None,
                None,
                dnull if ctx.has_null else None,
                None,
                None,
                None,
                None,
            )

        _, text_len, heads, key_dim = read_q.shape
        summary_count = int(states.shape[1])
        value_dim = int(states.shape[-1])

        dout_f = dout[0].float()
        out_f = out[0].float()
        delta = (dout_f * out_f).sum(dim=-1)
        gp = (
            torch.zeros_like(delta)
            if dlocal_prob is None or dlocal_prob.numel() == 0 else dlocal_prob[0, ..., 0].float()
        )
        if ctx.has_null:
            delta = delta + gp * p_null

        dnull = None
        if need_dnull:
            dnull = (p_null * (gp - delta)).to(null_logits.dtype).unsqueeze(0).unsqueeze(-1)

        dread = torch.zeros_like(read_q, dtype=torch.float32) if need_dread else None
        drouter = torch.zeros_like(router_q, dtype=torch.float32) if need_drouter else None
        dstates = torch.zeros_like(states, dtype=torch.float32) if need_dstates else None
        dsummary_k = torch.zeros_like(summary_k, dtype=torch.float32) if need_dsummary_k else None

        if summary_count > 0 and (need_dread or need_drouter or need_dstates or need_dsummary_k):
            read_q_f = read_q[0].float()
            router_q_f = router_q[0].float()
            block_m = _tile_block_m(text_len, heads, value_dim, read_q.dtype, backward=True)
            for summary_start in range(0, summary_count, block_m):
                summary_end = min(summary_start + block_m, summary_count)
                prob_block = prob[:, :, summary_start:summary_end]
                state_block = states[0, summary_start:summary_end].float()

                dz_block = None
                if need_drouter or need_dsummary_k:
                    read_block = torch.einsum("thk,bhkv->thbv", read_q_f, state_block)
                    dot_read = (dout_f[:, :, None, :] * read_block).sum(dim=-1) * float(ctx.read_scale)
                    dz_block = prob_block * (dot_read - delta[:, :, None])
                    del read_block, dot_read

                    if need_drouter:
                        drouter[0].add_(
                            torch.einsum(
                                "thb,bhk->thk",
                                dz_block,
                                summary_k[0, summary_start:summary_end].float(),
                            ),
                            alpha=float(ctx.route_scale),
                        )
                    if need_dsummary_k:
                        dsummary_k[0, summary_start:summary_end].copy_(
                            torch.einsum("thb,thk->bhk", dz_block, router_q_f) * float(ctx.route_scale)
                        )

                if need_dread or need_dstates:
                    weighted_dout = dout_f[:, :, None, :] * (prob_block[..., None] * float(ctx.read_scale))
                    if need_dread:
                        dread[0].add_(torch.einsum("thbv,bhkv->thk", weighted_dout, state_block))
                    if need_dstates:
                        dstates[0, summary_start:summary_end].copy_(
                            torch.einsum("thk,thbv->bhkv", read_q_f, weighted_dout)
                        )
                    del weighted_dout
                del dz_block

        return (
            None if dread is None else dread.to(read_q.dtype),
            None if drouter is None else drouter.to(router_q.dtype),
            dstates,
            None if dsummary_k is None else dsummary_k.to(summary_k.dtype),
            None,
            None,
            None,
            dnull,
            None,
            None,
            None,
            None,
        )


def hsd_tile_global_read(
    *,
    read_q: torch.Tensor,
    router_q: torch.Tensor,
    states: torch.Tensor,
    summary_k: torch.Tensor,
    boundary_counts: torch.Tensor,
    text_cu_seqlens: torch.Tensor,
    summary_cu_seqlens: torch.Tensor,
    null_logits: Optional[torch.Tensor] = None,
    route_scale: float,
    read_scale: float,
    debug_stats_callback: Optional[Callable[..., None]] = None,
    return_local_prob: bool = False,
    backend: str | None = None,
    profile: str | None = None,
    strict: bool | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    backend_name = _backend_name(backend)
    strict_mode = bool(strict) if strict is not None else False
    online_grouped = _online_grouped_enabled(backend_name)
    explicit_online_profile = bool(
        online_grouped and backend is not None and profile is not None
    )
    if backend_name in {"reference", "gemm"}:
        return None
    if not _env_bool("HSD_FUSED_GLOBAL_READ_TILE", True) and backend_name not in {
        "tile_speed",
        "tilespeed",
        "tile_lowmem",
        "tile-lowmem",
        "tile",
        "tile_online",
        "tile-online",
        "tile_grouped_online",
        "flashqla_tile",
    }:
        if strict_mode:
            raise RuntimeError("HSD tile global-read strict mode is disabled by configuration.")
        return None
    if not _supports_tile_global_read(
        read_q=read_q,
        router_q=router_q,
        states=states,
        summary_k=summary_k,
        boundary_counts=boundary_counts,
        text_cu_seqlens=text_cu_seqlens,
        summary_cu_seqlens=summary_cu_seqlens,
        null_logits=null_logits,
    ):
        if strict_mode:
            raise RuntimeError("HSD tile global-read strict mode rejected the input tensor contract.")
        return None
    if (
        not explicit_online_profile
        and any(t.requires_grad for t in (read_q, router_q, states, summary_k))
        and not _env_bool("HSD_FUSED_GLOBAL_READ_BWD", True)
    ):
        if strict_mode:
            raise RuntimeError("HSD tile global-read strict mode requires the custom backward path.")
        return None
    if (
        not explicit_online_profile
        and null_logits is not None
        and null_logits.requires_grad
        and not _env_bool("HSD_FUSED_GLOBAL_READ_BWD", True)
    ):
        if strict_mode:
            raise RuntimeError("HSD tile global-read strict mode requires null-logit backward support.")
        return None
    if (
        not explicit_online_profile
        and not _env_bool("HSD_FUSED_GLOBAL_READ_TILE_CUSTOM", True)
        and not _env_bool("HSD_FUSED_GLOBAL_READ_TILE_TORCH_FALLBACK", False)
    ):
        if strict_mode:
            raise RuntimeError("HSD tile global-read strict mode has no enabled custom implementation.")
        return None
    if (
        not explicit_online_profile
        and not _supports_custom_tile_global_read(
            read_q=read_q,
            router_q=router_q,
            states=states,
            summary_k=summary_k,
            boundary_counts=boundary_counts,
            text_cu_seqlens=text_cu_seqlens,
            summary_cu_seqlens=summary_cu_seqlens,
            null_logits=null_logits,
        )
        and not _env_bool("HSD_FUSED_GLOBAL_READ_TILE_TORCH_FALLBACK", False)
    ):
        if strict_mode:
            raise RuntimeError("HSD tile global-read strict mode missed the custom kernel prerequisites.")
        return None

    read_q = read_q.contiguous()
    router_q = router_q.contiguous()
    states = states.contiguous()
    summary_k = summary_k.contiguous()
    boundary_counts = boundary_counts.to(device=read_q.device, dtype=torch.long).contiguous()
    text_cu_seqlens = text_cu_seqlens.to(device=read_q.device, dtype=torch.long).contiguous()
    summary_cu_seqlens = summary_cu_seqlens.to(device=read_q.device, dtype=torch.long).contiguous()
    if online_grouped:
        resolved_profile = _resolve_tile_online_profile(
            profile,
            heads=int(read_q.shape[2]),
            key_dim=int(read_q.shape[3]),
            value_dim=int(states.shape[-1]),
            route_dim=int(router_q.shape[-1]),
        )
        if resolved_profile is None:
            if strict_mode:
                raise RuntimeError(
                    "HSD tile-online strict mode missed an exact allowlisted "
                    f"profile (requested={profile!r}, actual="
                    f"H{read_q.shape[2]}/K{read_q.shape[3]}/"
                    f"V{states.shape[-1]}/R{router_q.shape[-1]})."
                )
            return None
        profile = resolved_profile
        # Keep the padding outside the custom autograd Function so CopySlices
        # remains visible to autograd and returns gradients only for the real
        # summaries.  Invalid padded summaries are attached to the last packed
        # document but have boundary T+1, so every forward/backward validity
        # predicate rejects them.
        _, block_m, _, _ = _online_forward_launch_spec(profile)
        summary_count = int(states.shape[1])
        padded_summary_count = _online_summary_bucket_count(
            summary_count,
            block_m,
        )
        if padded_summary_count != summary_count:
            states = _pad_summary_dim(states, padded_summary_count, dim=1)
            summary_k = _pad_summary_dim(
                summary_k,
                padded_summary_count,
                dim=1,
            )
            boundary_counts = _pad_boundary_counts(
                boundary_counts,
                padded_summary_count,
                invalid_boundary=int(read_q.shape[1]) + 1,
            )
            summary_cu_seqlens = _pad_summary_cu_last(
                summary_cu_seqlens,
                padded_summary_count,
            )
    global_read_function = (
        _HSDOnlineGroupedGlobalReadFunction
        if online_grouped
        else _HSDTileGlobalReadFunction
    )
    if null_logits is None:
        null_arg = read_q.new_empty((0,), dtype=read_q.dtype)
    elif global_read_function is _HSDOnlineGroupedGlobalReadFunction:
        # The allowlisted Hopper online profiles and their low-memory backward
        # specialize all route inputs, including the optional null/local
        # candidate, to the query dtype.  The model intentionally forms the
        # gate logit in FP32.  Keep that model-side precision and insert the
        # differentiable cast only at this backend boundary; autograd casts
        # dNull back to the original FP32 tensor.
        null_arg = null_logits.to(dtype=read_q.dtype).contiguous()
    else:
        null_arg = null_logits.contiguous()
    apply_args = (
        read_q,
        router_q,
        states,
        summary_k,
        boundary_counts,
        text_cu_seqlens,
        summary_cu_seqlens,
        null_arg,
        float(route_scale),
        float(read_scale),
        null_logits is not None,
        debug_stats_callback,
    )
    if global_read_function is _HSDOnlineGroupedGlobalReadFunction:
        out, local_prob = global_read_function.apply(*apply_args, profile)
    else:
        out, local_prob = global_read_function.apply(*apply_args)
    if return_local_prob:
        if null_logits is None:
            raise RuntimeError("HSD Tile local probability requires local/null logits.")
        return out, local_prob
    return out


__all__ = ["hsd_tile_global_read", "hsd_tile_weighted_global_read"]
