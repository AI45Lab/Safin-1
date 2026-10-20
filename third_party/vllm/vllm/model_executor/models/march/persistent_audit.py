# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict persistent-state load verification and worker audit RPC.

The constructor-free mixin operates on the owning model's parameters and audit
attributes. It does not register modules, parameters or request/cache state.
"""

import hashlib

import torch

from ..utils import is_pp_missing_parameter

_HSD_PERSISTENT_MEMORY_METRIC_NAMES = (
    "memory_logit",
    "memory_probability",
    "dynamic_probability",
    "null_probability",
    "dynamic_rank1_probability",
    "dynamic_rank2_probability",
    "dynamic_rank3_probability",
    "dynamic_rank4_probability",
    "memory_read_rms",
    "dynamic_read_rms",
    "total_history_read_rms",
    "residual_rms",
    "state_rms",
    "dynamic_state_rms_median",
)


def _persistent_memory_tensor_checksum(
    tensors: list[tuple[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


class PersistentMemoryAuditMixin:
    @torch.inference_mode()
    def _finalize_persistent_route_key_load_audit(
        self,
        checkpoint_route_keys: list[tuple[str, torch.Tensor]],
        expected_local_route_keys: list[tuple[str, torch.Tensor]],
    ) -> None:
        declared = bool(
            getattr(self.config, "hsd_persistent_memory_route_key_enabled", False)
        )
        local_parameters = [
            (name, param)
            for name, param in self.named_parameters()
            if name.endswith(".persistent_memory_route_key")
            and not is_pp_missing_parameter(name, self)
        ]
        if not declared:
            if checkpoint_route_keys or local_parameters:
                raise RuntimeError(
                    "Checkpoint/config mismatch for dedicated persistent route keys."
                )
            self._hsd_persistent_route_key_checksum = None
            return
        additional_config = getattr(self.vllm_config, "additional_config", None) or {}
        expected_count = int(
            additional_config.get(
                "hsd_persistent_memory_route_key_expected_tensor_count", 16
            )
        )
        expected_numel = int(
            additional_config.get(
                "hsd_persistent_memory_route_key_expected_parameter_count", 65_536
            )
        )
        if not (
            len(checkpoint_route_keys)
            == len(expected_local_route_keys)
            == len(local_parameters)
            == expected_count
        ):
            raise RuntimeError(
                "Persistent route-key tensor count mismatch: "
                f"checkpoint={len(checkpoint_route_keys)}, "
                f"local={len(local_parameters)}, "
                f"expected={expected_count}."
            )
        if sum(int(t.numel()) for _, t in checkpoint_route_keys) != expected_numel:
            raise RuntimeError(
                "Persistent route-key global parameter count mismatch: "
                f"expected={expected_numel:,}."
            )
        required_names = {name for name, _ in local_parameters}
        if {name for name, _ in checkpoint_route_keys} != required_names:
            raise RuntimeError("Persistent route-key checkpoint key mismatch.")
        if {name for name, _ in expected_local_route_keys} != required_names:
            raise RuntimeError("Persistent route-key local shard key mismatch.")
        actual_by_name = dict(local_parameters)
        for name, expected_local in expected_local_route_keys:
            actual = actual_by_name[name].detach()
            if actual.dtype != torch.float32 or expected_local.dtype != torch.float32:
                raise RuntimeError("Persistent route keys must remain FP32.")
            if tuple(actual.shape) != tuple(expected_local.shape):
                raise RuntimeError(
                    f"Persistent route-key local shape mismatch for {name}."
                )
            if not bool(torch.isfinite(actual).all().item()) or not torch.equal(
                actual.to(device="cpu").contiguous(), expected_local.contiguous()
            ):
                raise RuntimeError(
                    f"Persistent route-key local shard mismatch for {name}."
                )
        self._hsd_persistent_route_key_checksum = _persistent_memory_tensor_checksum(
            checkpoint_route_keys
        )
        self._hsd_persistent_route_key_runtime_local_checksum = (
            _persistent_memory_tensor_checksum(local_parameters)
        )
        self._hsd_persistent_route_key_expected_tensor_count = expected_count
        self._hsd_persistent_route_key_expected_global_numel = expected_numel

    @torch.inference_mode()
    def _finalize_persistent_memory_load_audit(
        self,
        checkpoint_memory: list[tuple[str, torch.Tensor]],
        expected_local_memory: list[tuple[str, torch.Tensor]],
    ) -> None:
        declared = bool(
            getattr(self.config, "hsd_persistent_memory_enabled", False)
        )
        local_parameters = [
            (name, param)
            for name, param in self.named_parameters()
            if name.endswith(".persistent_memory_state")
            and not is_pp_missing_parameter(name, self)
        ]
        if not declared:
            if checkpoint_memory or local_parameters:
                raise RuntimeError(
                    "Checkpoint/config mismatch: persistent memory tensors exist "
                    "while hsd_persistent_memory_enabled=false."
                )
            self._hsd_persistent_memory_checksum = None
            self._hsd_persistent_memory_original_rms = None
            self._hsd_persistent_memory_expected_local_checksum = None
            self._hsd_persistent_memory_runtime_local_checksum = None
            return

        additional_config = getattr(self.vllm_config, "additional_config", None) or {}
        expected_count_raw = additional_config.get(
            "hsd_persistent_memory_expected_tensor_count"
        )
        expected_numel_raw = additional_config.get(
            "hsd_persistent_memory_expected_parameter_count"
        )
        target_dense_shape = (
            int(getattr(self.config, "linear_num_value_heads", 0) or 0),
            int(getattr(self.config, "linear_key_head_dim", 0) or 0),
            int(getattr(self.config, "linear_value_head_dim", 0) or 0),
        )
        is_target_dense = (
            str(getattr(self.config, "model_type", ""))
            == "safin1_4b_text"
            and target_dense_shape == (32, 128, 128)
        )
        compact_dedicated = bool(
            getattr(self.config, "hsd_persistent_memory_route_key_enabled", False)
        )
        expected_count = int(
            expected_count_raw
            if expected_count_raw is not None
            else 16
            if is_target_dense and compact_dedicated
            else 23
            if is_target_dense
            else len(local_parameters)
        )
        expected_numel = int(
            expected_numel_raw
            if expected_numel_raw is not None
            else 8_388_608
            if is_target_dense and compact_dedicated
            else 12_058_624
            if is_target_dense
            else sum(int(t.numel()) for _, t in checkpoint_memory)
        )
        if len(checkpoint_memory) != expected_count:
            raise RuntimeError(
                "Persistent memory checkpoint tensor count mismatch: "
                f"got {len(checkpoint_memory)}, expected {expected_count}."
            )
        if len(local_parameters) != expected_count:
            raise RuntimeError(
                "Persistent memory model tensor count mismatch: "
                f"got {len(local_parameters)}, expected {expected_count}."
            )
        if len(expected_local_memory) != expected_count:
            raise RuntimeError(
                "Persistent memory expected local tensor count mismatch: "
                f"got {len(expected_local_memory)}, expected {expected_count}."
            )
        global_numel = sum(int(t.numel()) for _, t in checkpoint_memory)
        if global_numel != expected_numel:
            raise RuntimeError(
                "Persistent memory checkpoint parameter count mismatch: "
                f"got {global_numel:,}, expected {expected_numel:,}."
            )
        required_names = {name for name, _ in local_parameters}
        loaded_names = {name for name, _ in checkpoint_memory}
        if loaded_names != required_names:
            raise RuntimeError(
                "Persistent memory checkpoint key mismatch: missing="
                f"{sorted(required_names - loaded_names)}, unexpected="
                f"{sorted(loaded_names - required_names)}."
            )
        expected_local_names = {name for name, _ in expected_local_memory}
        if expected_local_names != required_names:
            raise RuntimeError(
                "Persistent memory expected local key mismatch: missing="
                f"{sorted(required_names - expected_local_names)}, unexpected="
                f"{sorted(expected_local_names - required_names)}."
            )
        if any(t.dtype != torch.float32 for _, t in checkpoint_memory):
            raise RuntimeError("All persistent memory checkpoint tensors must be FP32.")
        if any(not bool(torch.isfinite(t).all().item()) for _, t in checkpoint_memory):
            raise RuntimeError(
                "Persistent memory checkpoint contains non-finite values."
            )

        try:
            from vllm.distributed import get_tensor_model_parallel_world_size

            tp_size = int(get_tensor_model_parallel_world_size())
        except Exception:
            tp_size = int(
                getattr(self.vllm_config.parallel_config, "tensor_parallel_size", 1)
                or 1
            )
        if tp_size <= 0 or expected_numel % tp_size:
            raise RuntimeError(
                "Persistent memory global parameter count is not divisible by "
                f"tensor parallel size {tp_size}: {expected_numel}."
            )
        expected_local_numel = expected_numel // tp_size
        actual_local_numel = sum(int(t.numel()) for _, t in local_parameters)
        shard_local_numel = sum(int(t.numel()) for _, t in expected_local_memory)
        if actual_local_numel != expected_local_numel:
            raise RuntimeError(
                "Persistent memory local parameter count mismatch: "
                f"got {actual_local_numel:,}, expected {expected_local_numel:,}."
            )
        if shard_local_numel != expected_local_numel:
            raise RuntimeError(
                "Persistent memory expected local shard count mismatch: "
                f"got {shard_local_numel:,}, expected {expected_local_numel:,}."
            )

        actual_by_name = dict(local_parameters)
        for name, expected_local in sorted(
            expected_local_memory,
            key=lambda item: item[0],
        ):
            actual = actual_by_name[name].detach()
            if expected_local.dtype != torch.float32:
                raise RuntimeError(
                    f"Persistent memory expected local shard {name} must be FP32, "
                    f"got {expected_local.dtype}."
                )
            if tuple(actual.shape) != tuple(expected_local.shape):
                raise RuntimeError(
                    f"Persistent memory local shape mismatch for {name}: "
                    f"actual={tuple(actual.shape)}, "
                    f"expected={tuple(expected_local.shape)}."
                )
            if actual.dtype != torch.float32:
                raise RuntimeError(
                    f"Persistent memory local parameter {name} must be FP32, "
                    f"got {actual.dtype}."
                )
            if not bool(torch.isfinite(expected_local).all().item()):
                raise RuntimeError(
                    f"Persistent memory expected local shard {name} contains "
                    "non-finite values."
                )
            if not bool(torch.isfinite(actual).all().item()):
                raise RuntimeError(
                    f"Persistent memory local parameter {name} contains "
                    "non-finite values."
                )
            if not torch.equal(
                actual.to(device="cpu").contiguous(),
                expected_local.contiguous(),
            ):
                raise RuntimeError(
                    f"Persistent memory local TP shard mismatch for {name}; "
                    "the loaded Parameter does not equal the exact checkpoint "
                    "value-head slice."
                )

        expected_local_checksum = _persistent_memory_tensor_checksum(
            expected_local_memory
        )
        actual_local_checksum = _persistent_memory_tensor_checksum(local_parameters)
        if actual_local_checksum != expected_local_checksum:
            raise RuntimeError(
                "Persistent memory local TP checksum mismatch after load: "
                f"actual={actual_local_checksum}, "
                f"expected={expected_local_checksum}."
            )

        sumsq = sum(
            float(t.detach().float().square().sum().cpu().item())
            for _, t in checkpoint_memory
        )
        self._hsd_persistent_memory_checksum = (
            _persistent_memory_tensor_checksum(checkpoint_memory)
        )
        self._hsd_persistent_memory_original_rms = (
            (sumsq / global_numel) ** 0.5 if global_numel else 0.0
        )
        self._hsd_persistent_memory_expected_local_checksum = (
            expected_local_checksum
        )
        self._hsd_persistent_memory_expected_tensor_count = expected_count
        self._hsd_persistent_memory_expected_global_numel = expected_numel

        # Zero is a structural ablation: validate and checksum the exact same
        # checkpoint first, then clear local shards before any graph capture.
        if self.hsd_persistent_memory_mode == "zero":
            for _, parameter in local_parameters:
                parameter.zero_()
        self._hsd_persistent_memory_runtime_local_checksum = (
            _persistent_memory_tensor_checksum(local_parameters)
        )

    @torch.inference_mode()
    def get_persistent_memory_audit(
        self,
        reset_metrics: bool = True,
    ) -> dict[str, object]:
        """Return a JSON-safe checkpoint/routing audit for the named worker RPC."""

        # Resolve the owning attention class after the model module has loaded.
        from ..safin1 import HistoricalGatedDeltaNetAttention

        mode = str(self.hsd_persistent_memory_mode)
        local_parameters = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if name.endswith(".persistent_memory_state")
            and not is_pp_missing_parameter(name, self)
        ]
        parameters = [parameter for _, parameter in local_parameters]
        local_route_keys = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if name.endswith(".persistent_memory_route_key")
            and not is_pp_missing_parameter(name, self)
        ]
        tensor_count = len(parameters)
        local_numel = sum(int(parameter.numel()) for parameter in parameters)
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size

            tp_size = int(get_tensor_model_parallel_world_size())
        except Exception:
            tp_size = int(
                getattr(self.vllm_config.parallel_config, "tensor_parallel_size", 1)
                or 1
            )
        global_numel = local_numel * tp_size
        finite = all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in parameters
        )
        local_sumsq = sum(
            float(parameter.detach().float().square().sum().cpu().item())
            for parameter in parameters
        )
        local_parameter_rms = (
            (local_sumsq / local_numel) ** 0.5 if local_numel else None
        )
        local_checksum = (
            _persistent_memory_tensor_checksum(local_parameters)
            if local_parameters
            else None
        )
        route_key_local_checksum = (
            _persistent_memory_tensor_checksum(local_route_keys)
            if local_route_keys
            else None
        )
        expected_route_key_local_checksum = getattr(
            self,
            "_hsd_persistent_route_key_runtime_local_checksum",
            None,
        )
        if (
            expected_route_key_local_checksum is not None
            and route_key_local_checksum != expected_route_key_local_checksum
        ):
            raise RuntimeError(
                "Persistent route-key Parameters were mutated after checkpoint load."
            )
        expected_runtime_local_checksum = getattr(
            self,
            "_hsd_persistent_memory_runtime_local_checksum",
            None,
        )
        if (
            expected_runtime_local_checksum is not None
            and local_checksum != expected_runtime_local_checksum
        ):
            raise RuntimeError(
                "Persistent memory local Parameters were mutated after their "
                "validated checkpoint load: "
                f"actual={local_checksum}, "
                f"expected={expected_runtime_local_checksum}."
            )
        # `zero` must be exactly, not approximately, zero.
        if mode == "zero" and any(
            bool(torch.count_nonzero(parameter).item()) for parameter in parameters
        ):
            raise RuntimeError("Persistent memory zero mode was mutated after load.")
        parameter_rms = (
            0.0
            if mode == "zero" and parameters
            else getattr(
                self,
                "_hsd_persistent_memory_original_rms",
                local_parameter_rms,
            )
        )

        metric_values: dict[str, list[float]] = {
            name: [] for name in _HSD_PERSISTENT_MEMORY_METRIC_NAMES
        }
        if mode != "removed":
            for module in self.modules():
                if not isinstance(module, HistoricalGatedDeltaNetAttention):
                    continue
                values = module.get_persistent_memory_metrics(
                    cumulative=True,
                    reset=reset_metrics,
                )
                for name in _HSD_PERSISTENT_MEMORY_METRIC_NAMES:
                    if name in values:
                        metric_values[name].append(float(values[name]))
        metrics = {
            name: (
                sum(values) / len(values)
                if values
                else None
            )
            for name, values in metric_values.items()
        }
        return {
            "mode": mode,
            "tensor_count": tensor_count,
            "local_numel": local_numel,
            "global_numel": global_numel,
            "dtype": (
                "float32"
                if parameters and all(p.dtype == torch.float32 for p in parameters)
                else None
            ),
            "finite": finite,
            "checksum": getattr(
                self,
                "_hsd_persistent_memory_checksum",
                None,
            ),
            "local_checksum": local_checksum,
            "parameter_rms": parameter_rms,
            "local_parameter_rms": local_parameter_rms,
            "route_key": {
                "enabled": bool(
                    getattr(
                        self.config,
                        "hsd_persistent_memory_route_key_enabled",
                        False,
                    )
                ),
                "mode": str(
                    getattr(
                        self.config,
                        "hsd_persistent_memory_route_key_mode",
                        "legacy",
                    )
                ),
                "tensor_count": len(local_route_keys),
                "local_numel": sum(int(t.numel()) for _, t in local_route_keys),
                "global_numel": sum(int(t.numel()) for _, t in local_route_keys)
                * tp_size,
                "dtype": (
                    "float32"
                    if local_route_keys
                    and all(t.dtype == torch.float32 for _, t in local_route_keys)
                    else None
                ),
                "finite": all(
                    bool(torch.isfinite(t).all().item()) for _, t in local_route_keys
                ),
                "checksum": getattr(
                    self,
                    "_hsd_persistent_route_key_checksum",
                    None,
                ),
                "local_checksum": route_key_local_checksum,
            },
            "metrics": metrics,
        }
