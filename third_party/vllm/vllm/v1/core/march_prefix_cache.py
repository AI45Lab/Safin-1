# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-safe policy and identity for experimental MARCH prefix reuse."""

import hashlib
import json
import os


def prefix_cache_namespace(config: object, dtype: object, epoch: int) -> str:
    """Bind cached state to the complete text runtime and reset generation."""
    values = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    payload = json.dumps(
        {"text_config": values, "dtype": str(dtype)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return (
        f"march-prefix-v1:{int(epoch)}:{hashlib.sha256(payload.encode()).hexdigest()}"
    )


def validate_march_prefix_cache(config: object, vllm_config: object) -> None:
    """Reject unimplemented state combinations before the model is built."""
    cache = vllm_config.cache_config
    if not cache.enable_prefix_caching:
        return
    enabled = os.environ.get("VLLM_HSD_ENABLE_PREFIX_CACHE", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        raise NotImplementedError(
            "Safin-1 prefix caching is experimental; use --no-enable-prefix-caching "
            "or opt in through --experimental-prefix-cache with the MARCH profile."
        )
    compact = ("USE_PERSISTENT_STATE_SIDECAR", "ENABLE_COMPACT_STATE_SLOTS",
               "ENABLE_COMPACT_STATE_KV_SPEC", "SERIALIZE_PREFIX_PREFILL")
    if any(os.environ.get("VLLM_HSD_" + key, "").lower() not in
           {"1", "true", "yes", "on"} for key in compact):
        raise NotImplementedError("MARCH prefix reuse requires the compact state/Graph V3 prefix profile.")
    if bool(getattr(config, "hsd_persistent_memory_enabled", False)):
        raise NotImplementedError(
            "Mutable persistent state requires prefix caching to be disabled."
        )
    if not vllm_config.model_config.enforce_eager:
        required = (
            "ENABLE_GRAPH_PREFIX_CACHE", "ENABLE_GRAPH_V3",
            "GRAPH_V3_FULL_CUDAGRAPH", "ENABLE_COMPACT_STATE_SLOTS",
            "USE_PERSISTENT_STATE_SIDECAR",
        )
        if any(os.environ.get("VLLM_HSD_" + key, "").lower() not in
               {"1", "true", "yes", "on"} for key in required):
            raise NotImplementedError(
                "MARCH prefix reuse requires eager or the complete Graph V3 prefix profile."
            )
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise NotImplementedError(
            "Experimental MARCH prefix reuse requires pipeline parallel size 1."
        )
    if bool(getattr(vllm_config.scheduler_config, "async_scheduling", False)):
        raise NotImplementedError(
            "Experimental MARCH prefix reuse requires synchronous scheduling."
        )
    if cache.mamba_cache_mode != "none":
        raise NotImplementedError("MARCH snapshots require mamba_cache_mode=none.")
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        raise NotImplementedError(
            "MARCH snapshots cannot be transferred through a KV connector."
        )
    if getattr(vllm_config.model_config, "quantization", None) is not None:
        raise NotImplementedError(
            "Quantized MARCH prefix reuse has not been validated."
        )
