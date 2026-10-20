# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Safin-1 checkpoint mapping, TP slicing and model weight-loading methods.

Mixins have no constructor or parameters. The inner model still delegates native
weights through its original superclass; the outer model retains strict first
loads and prefix invalidation before subsequent partial updates.
"""

from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.configs.safin1 import Safin1TextConfig
from vllm.transformers_utils.configs.safin1_moe import Safin1MoeTextConfig

from ..utils import AutoWeightsLoader, WeightsMapper, is_pp_missing_parameter

# Preserve the established model logger identity after extracting the methods.
logger = init_logger("vllm.model_executor.models.safin1")

_HISTORICAL_HF_TO_VLLM_MAPPER = WeightsMapper(
    orig_to_new_substr={
        ".linear_attn.hsd_anchor_mlp.": ".hsd_anchor_mlp.",
        ".linear_attn.anchor_mlp.": ".hsd_anchor_mlp.",
        ".linear_attention.": ".linear_attn.",
        ".route_q_proj.": ".router_q_proj.",
        ".route_k_proj.": ".anchor_k_proj.",
        ".anchor_mlp.": ".hsd_anchor_mlp.",
        ".history_gate_logit": ".hsd_history_gate_logit",
        ".history_out_rmsnorm_weight": ".hsd_history_out_rmsnorm_weight",
    },
    orig_to_new_prefix={
        "model.language_model.model.": "model.",
        "model.language_model.": "model.",
        "language_model.model.": "model.",
        "language_model.": "model.",
        "layers.": "model.layers.",
        "embed_tokens.": "model.embed_tokens.",
        "norm.": "model.norm.",
        "anchor_embedding.": "model.anchor_embedding.",
        "model.visual.": None,
        "visual.": None,
    },
)


_HISTORICAL_HSD_PARAM_FRAGMENTS = (
    "anchor_embedding.",
    ".anchor_q_proj.",
    ".history_read_q_down_proj.",
    ".history_read_q_up_proj.",
    ".anchor_k_proj.",
    ".router_q_proj.",
    ".null_gate_proj.",
    ".hsd_anchor_mlp.",
    ".hsd_history_gate_logit",
    ".hsd_history_out_rmsnorm_weight",
    ".hsd_null_read_gate_bias",
    ".persistent_memory_state",
    ".persistent_memory_route_key",
)


def _is_historical_hsd_param(name: str) -> bool:
    return any(fragment in name for fragment in _HISTORICAL_HSD_PARAM_FRAGMENTS)


def _is_optional_unmaterialized_hsd_param(name: str) -> bool:
    return name.endswith(
        (
            ".linear_attn.router_q_proj.weight",
            ".linear_attn.anchor_k_proj.weight",
            ".linear_attn.null_gate_proj.weight",
            ".linear_attn.hsd_null_read_gate_bias",
            ".self_attn.router_q_proj.weight",
            ".self_attn.anchor_k_proj.weight",
            ".self_attn.null_gate_proj.weight",
            ".self_attn.hsd_null_read_gate_bias",
        )
    )


def _is_optional_missing_hsd_param(name: str) -> bool:
    return name.endswith(
        (
            ".linear_attn.null_gate_proj.weight",
            ".linear_attn.hsd_null_read_gate_bias",
            ".self_attn.null_gate_proj.weight",
            ".self_attn.hsd_null_read_gate_bias",
        )
    )


def _maybe_shard_hsd_null_gate_bias(
    name: str,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> torch.Tensor:
    if not name.endswith(
        (
            ".linear_attn.hsd_null_read_gate_bias",
            ".self_attn.hsd_null_read_gate_bias",
        )
    ):
        return loaded_weight
    if tuple(loaded_weight.shape) == tuple(param.shape):
        return loaded_weight
    if loaded_weight.ndim != 1 or param.ndim != 1:
        return loaded_weight
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_size = int(get_tensor_model_parallel_world_size())
        tp_rank = int(get_tensor_model_parallel_rank())
    except Exception:
        tp_size = 1
        tp_rank = 0
    if tp_size <= 1 or loaded_weight.numel() != param.numel() * tp_size:
        return loaded_weight
    return loaded_weight.chunk(tp_size, dim=0)[tp_rank].contiguous()


def _load_hsd_persistent_memory_state(
    name: str,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> torch.Tensor:
    """Strictly load one value-head shard and return its expected CPU value.

    The returned tensor is the exact rank-local slice selected from the
    unsharded checkpoint tensor.  Keeping that value separate from ``param``
    lets the model-level load audit prove that the destination parameter was
    populated with the intended TP shard instead of merely proving that every
    worker saw the same full checkpoint tensor.
    """

    if not name.endswith(".persistent_memory_state"):
        raise ValueError(f"not a persistent memory parameter: {name}")
    if loaded_weight.dtype != torch.float32:
        raise RuntimeError(
            f"Persistent memory {name} must be FP32, got {loaded_weight.dtype}."
        )
    if loaded_weight.ndim != 3:
        raise RuntimeError(
            f"Persistent memory {name} must be rank 3 [H,K,V], got "
            f"{tuple(loaded_weight.shape)}."
        )
    if not bool(torch.isfinite(loaded_weight).all().item()):
        raise RuntimeError(f"Persistent memory {name} contains non-finite values.")
    if param.dtype != torch.float32:
        raise RuntimeError(
            f"Persistent memory destination {name} must remain FP32, got {param.dtype}."
        )
    if param.ndim != 3:
        raise RuntimeError(
            f"Persistent memory destination {name} must be rank 3 [H,K,V], "
            f"got {tuple(param.shape)}."
        )

    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_size = int(get_tensor_model_parallel_world_size())
        tp_rank = int(get_tensor_model_parallel_rank())
    except Exception:
        tp_size = 1
        tp_rank = 0
    if tp_size <= 0 or tp_rank < 0 or tp_rank >= tp_size:
        raise RuntimeError(
            f"Invalid tensor-parallel coordinates for {name}: "
            f"rank={tp_rank}, size={tp_size}."
        )
    expected_global_shape = (
        int(param.shape[0]) * tp_size,
        int(param.shape[1]),
        int(param.shape[2]),
    )
    if tuple(loaded_weight.shape) != expected_global_shape:
        raise RuntimeError(
            f"Persistent memory {name} shape mismatch: checkpoint="
            f"{tuple(loaded_weight.shape)}, expected global={expected_global_shape}, "
            f"local={tuple(param.shape)}, tp={tp_size}."
        )
    if expected_global_shape[0] % tp_size:
        raise RuntimeError(
            f"Persistent memory {name} value heads are not divisible by TP={tp_size}."
        )
    local = loaded_weight.narrow(
        0,
        tp_rank * int(param.shape[0]),
        int(param.shape[0]),
    ).contiguous()
    default_weight_loader(param, local)
    actual = param.detach()
    if tuple(actual.shape) != tuple(local.shape):
        raise RuntimeError(
            f"Persistent memory {name} local shape changed while loading: "
            f"destination={tuple(actual.shape)}, expected={tuple(local.shape)}."
        )
    if actual.dtype != torch.float32:
        raise RuntimeError(
            f"Persistent memory {name} local shard must remain FP32, "
            f"got {actual.dtype}."
        )
    if not bool(torch.isfinite(actual).all().item()):
        raise RuntimeError(
            f"Persistent memory {name} local shard contains non-finite values."
        )
    expected_on_device = local.to(device=actual.device)
    if not torch.equal(actual, expected_on_device):
        raise RuntimeError(
            f"Persistent memory {name} local TP shard differs from the exact "
            f"checkpoint value-head slice for rank {tp_rank}/{tp_size}."
        )
    return local.detach().to(device="cpu").contiguous()


def _load_hsd_persistent_memory_route_key(
    name: str,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> torch.Tensor:
    """Strictly load the route-key value-head shard and return its CPU value."""

    if not name.endswith(".persistent_memory_route_key"):
        raise ValueError(f"not a persistent memory route key: {name}")
    if loaded_weight.dtype != torch.float32 or loaded_weight.ndim != 2:
        raise RuntimeError(
            f"Persistent route key {name} must be FP32 [H,K], got "
            f"dtype={loaded_weight.dtype}, shape={tuple(loaded_weight.shape)}."
        )
    if param.dtype != torch.float32 or param.ndim != 2:
        raise RuntimeError(
            f"Persistent route-key destination {name} must be FP32 [H,K], got "
            f"dtype={param.dtype}, shape={tuple(param.shape)}."
        )
    if not bool(torch.isfinite(loaded_weight).all().item()):
        raise RuntimeError(f"Persistent route key {name} contains non-finite values.")
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_size = int(get_tensor_model_parallel_world_size())
        tp_rank = int(get_tensor_model_parallel_rank())
    except Exception:
        tp_size = 1
        tp_rank = 0
    expected_global_shape = (
        int(param.shape[0]) * tp_size,
        int(param.shape[1]),
    )
    if tuple(loaded_weight.shape) != expected_global_shape:
        raise RuntimeError(
            f"Persistent route key {name} shape mismatch: checkpoint="
            f"{tuple(loaded_weight.shape)}, expected={expected_global_shape}."
        )
    local = loaded_weight.narrow(
        0,
        tp_rank * int(param.shape[0]),
        int(param.shape[0]),
    ).contiguous()
    default_weight_loader(param, local)
    actual = param.detach()
    if actual.dtype != torch.float32 or not torch.equal(
        actual, local.to(device=actual.device)
    ):
        raise RuntimeError(
            f"Persistent route key {name} did not load the exact TP shard."
        )
    return local.detach().to(device="cpu").contiguous()


def _is_hsd_conv1d_weight(name: str) -> bool:
    return name.endswith(
        (
            ".linear_attn.conv1d.weight",
            ".self_attn.conv1d.weight",
        )
    )


def _maybe_shard_hsd_conv1d_weight(
    config: Safin1TextConfig | Safin1MoeTextConfig,
    name: str,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> torch.Tensor:
    if not _is_hsd_conv1d_weight(name):
        return loaded_weight
    if tuple(loaded_weight.shape) == tuple(param.shape):
        return loaded_weight
    if loaded_weight.ndim < 1 or param.ndim < 1:
        return loaded_weight
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_size = int(get_tensor_model_parallel_world_size())
        tp_rank = int(get_tensor_model_parallel_rank())
    except Exception:
        tp_size = 1
        tp_rank = 0
    if tp_size <= 1:
        return loaded_weight

    key_dim = int(getattr(config, "linear_num_key_heads", 0) or 0) * int(
        getattr(config, "linear_key_head_dim", 0) or 0
    )
    value_dim = int(getattr(config, "linear_num_value_heads", 0) or 0) * int(
        getattr(config, "linear_value_head_dim", 0) or 0
    )
    full_dim = key_dim * 2 + value_dim
    local_dim = int(param.shape[0])
    if (
        full_dim <= 0
        or int(loaded_weight.shape[0]) != full_dim
        or local_dim * tp_size != full_dim
        or key_dim % tp_size != 0
        or value_dim % tp_size != 0
    ):
        return loaded_weight

    key_local = key_dim // tp_size
    value_local = value_dim // tp_size
    q_start = tp_rank * key_local
    k_start = key_dim + tp_rank * key_local
    v_start = key_dim * 2 + tp_rank * value_local
    return torch.cat(
        [
            loaded_weight.narrow(0, q_start, key_local),
            loaded_weight.narrow(0, k_start, key_local),
            loaded_weight.narrow(0, v_start, value_local),
        ],
        dim=0,
    ).contiguous()


class ModelWeightLoadingMixin:
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        remaining_weights: list[tuple[str, torch.Tensor]] = []
        hsd_passthrough = (
            ".anchor_q_proj.",
            ".history_read_q_down_proj.",
            ".history_read_q_up_proj.",
            ".anchor_k_proj.",
            ".router_q_proj.",
            ".null_gate_proj.",
            ".hsd_anchor_mlp.",
            ".hsd_history_gate_logit",
            ".hsd_history_out_rmsnorm_weight",
            ".hsd_null_read_gate_bias",
            ".persistent_memory_state",
            ".persistent_memory_route_key",
            "anchor_embedding.",
        )
        for name, loaded_weight in weights:
            if _is_hsd_conv1d_weight(name):
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    raise RuntimeError(
                        "HSD conv1d parameter "
                        f"{name} not found in Safin1Model."
                    )
                param = params_dict[name]
                loaded_weight = _maybe_shard_hsd_conv1d_weight(
                    self.config,
                    name,
                    param,
                    loaded_weight,
                )
                default_weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue
            if any(part in name for part in hsd_passthrough):
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    if _is_optional_unmaterialized_hsd_param(name):
                        logger.warning_once(
                            "Skipping optional unmaterialized HSD checkpoint "
                            "parameter %s",
                            name,
                        )
                        continue
                    raise RuntimeError(
                        "HSD parameter "
                        f"{name} not found in Safin1Model."
                    )
                param = params_dict[name]
                if name.endswith(".persistent_memory_state"):
                    _load_hsd_persistent_memory_state(
                        name,
                        param,
                        loaded_weight,
                    )
                    if self.hsd_persistent_memory_mode == "zero":
                        with torch.no_grad():
                            param.zero_()
                    loaded_params.add(name)
                    continue
                if name.endswith(".persistent_memory_route_key"):
                    _load_hsd_persistent_memory_route_key(
                        name,
                        param,
                        loaded_weight,
                    )
                    loaded_params.add(name)
                    continue
                loaded_weight = _maybe_shard_hsd_null_gate_bias(
                    name,
                    param,
                    loaded_weight,
                )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            else:
                remaining_weights.append((name, loaded_weight))
        loaded_params.update(super().load_weights(remaining_weights))
        return loaded_params


class CausalLMWeightLoadingMixin:
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if getattr(self, "_historical_hsd_weights_validated", False):
            self.invalidate_hsd_prefix_cache()
        mapped_weights = list(self.hf_to_vllm_mapper.apply(weights))
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        remaining_weights: list[tuple[str, torch.Tensor]] = []
        validate_required_hsd = not getattr(
            self, "_historical_hsd_weights_validated", False
        )
        checkpoint_memory: list[tuple[str, torch.Tensor]] = []
        expected_local_memory: list[tuple[str, torch.Tensor]] = []
        memory_names_seen: set[str] = set()
        checkpoint_route_keys: list[tuple[str, torch.Tensor]] = []
        expected_local_route_keys: list[tuple[str, torch.Tensor]] = []
        route_key_names_seen: set[str] = set()
        for name, loaded_weight in mapped_weights:
            if _is_hsd_conv1d_weight(name):
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    raise RuntimeError(
                        "HSD conv1d parameter "
                        f"{name} not found in {type(self).__name__}."
                    )
                param = params_dict[name]
                loaded_weight = _maybe_shard_hsd_conv1d_weight(
                    self.config,
                    name,
                    param,
                    loaded_weight,
                )
                default_weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue
            if _is_historical_hsd_param(name):
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    if _is_optional_unmaterialized_hsd_param(name):
                        logger.warning_once(
                            "Skipping optional unmaterialized HSD checkpoint "
                            "parameter %s",
                            name,
                        )
                        continue
                    raise RuntimeError(
                        "HSD parameter "
                        f"{name} not found in {type(self).__name__}."
                    )
                param = params_dict[name]
                if name.endswith(".persistent_memory_state"):
                    if name in memory_names_seen:
                        raise RuntimeError(
                            f"Duplicate persistent memory checkpoint tensor: {name}."
                        )
                    memory_names_seen.add(name)
                    # Validation and the global checksum intentionally use the
                    # unsharded source before selecting local value heads.
                    checkpoint_memory.append((name, loaded_weight))
                    expected_local = _load_hsd_persistent_memory_state(
                        name,
                        param,
                        loaded_weight,
                    )
                    expected_local_memory.append((name, expected_local))
                    loaded_params.add(name)
                    continue
                if name.endswith(".persistent_memory_route_key"):
                    if name in route_key_names_seen:
                        raise RuntimeError(
                            f"Duplicate persistent route-key checkpoint tensor: {name}."
                        )
                    route_key_names_seen.add(name)
                    checkpoint_route_keys.append((name, loaded_weight))
                    expected_local = _load_hsd_persistent_memory_route_key(
                        name,
                        param,
                        loaded_weight,
                    )
                    expected_local_route_keys.append((name, expected_local))
                    loaded_params.add(name)
                    continue
                loaded_weight = _maybe_shard_hsd_null_gate_bias(
                    name,
                    param,
                    loaded_weight,
                )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            else:
                remaining_weights.append((name, loaded_weight))

        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        loaded_params.update(loader.load_weights(remaining_weights))
        # vLLM also calls load_weights for runtime partial weight syncs from
        # Megatron. Keep checkpoint loading strict, then allow later updates to
        # omit unchanged HSD parameters.
        if validate_required_hsd:
            missing_hsd = sorted(
                name
                for name, _ in self.named_parameters()
                if _is_historical_hsd_param(name)
                and not is_pp_missing_parameter(name, self)
                and not _is_optional_missing_hsd_param(name)
                and name not in loaded_params
            )
            if missing_hsd:
                raise RuntimeError(
                    "Safin-1 HSD checkpoint is missing required "
                    f"HSD parameters: {missing_hsd[:20]}"
                )
            self._finalize_persistent_memory_load_audit(
                checkpoint_memory,
                expected_local_memory,
            )
            self._finalize_persistent_route_key_load_audit(
                checkpoint_route_keys,
                expected_local_route_keys,
            )
            self._historical_hsd_weights_validated = True
        return loaded_params
