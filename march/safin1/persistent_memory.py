"""Learned persistent memory artifacts and metrics for Safin-1."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any
from pathlib import Path
import torch
from torch import nn

from .config_utils import atomic_write_bytes


def _persistent_memory_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith("persistent_memory_state")
    }


def _persistent_memory_route_key_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith("persistent_memory_route_key")
    }


def save_persistent_memory_artifact(
    model: nn.Module,
    output_dir: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Save the compact persistent state and optional dedicated route keys."""

    from safetensors.torch import save

    state_parameters = _persistent_memory_parameters(model)
    route_key_parameters = _persistent_memory_route_key_parameters(model)
    parameters = {**state_parameters, **route_key_parameters}
    if not parameters:
        raise ValueError(
            "model has no persistent_memory_state parameters; construct it with "
            "hsd_persistent_memory_enabled=true."
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tensor_path = root / "persistent_memory.safetensors"
    config_path = root / "persistent_memory_config.json"
    tensors = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, parameter in parameters.items()
    }
    artifact_format = "hsd_persistent_memory_v2" if route_key_parameters else "hsd_persistent_memory_v1"
    tensor_bytes = save(tensors, metadata={"format": artifact_format})
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", model_config)
    layer_names = sorted(name.rsplit(".persistent_memory_state", 1)[0] for name in state_parameters)
    metadata = {
        "format": artifact_format,
        "model_type": getattr(model_config, "model_type", None),
        "tensor_sha256": hashlib.sha256(tensor_bytes).hexdigest(),
        "key_source": (
            "per_layer_dedicated_parameter"
            if route_key_parameters
            else "anchor_hidden_projection"
        ),
        "enabled": bool(getattr(text_config, "hsd_persistent_memory_enabled", True)),
        "layers": layer_names,
        "anchor_chunk_size": getattr(text_config, "anchor_chunk_size", None),
        "hsd_read_topk": getattr(text_config, "hsd_read_topk", None),
        "hsd_null_read_gate": getattr(text_config, "hsd_null_read_gate", None),
        "tensor_count": len(tensors),
        "state_tensor_count": len(state_parameters),
        "route_key_tensor_count": len(route_key_parameters),
        "parameter_count": sum(tensor.numel() for tensor in tensors.values()),
        "dtype": "float32",
        "tensors": {name: list(tensor.shape) for name, tensor in tensors.items()},
    }
    metadata_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(tensor_path, tensor_bytes)
    atomic_write_bytes(config_path, metadata_bytes)
    return tensor_path, config_path


def _validate_persistent_memory_metadata(tensor_path, tensor_bytes, loaded, model):
    # Read both tensor data and its header from one immutable file snapshot.
    header_length = int.from_bytes(tensor_bytes[:8], "little")
    header = json.loads(tensor_bytes[8:8 + header_length])
    tensor_format = (header.get("__metadata__") or {}).get("format")
    formats = {"hsd_persistent_memory_v1", "hsd_persistent_memory_v2"}
    if tensor_format is not None and tensor_format not in formats:
        raise RuntimeError(f"Unsupported persistent memory format: {tensor_format!r}.")
    route_keys = [name for name in loaded if name.endswith("persistent_memory_route_key")]
    if (tensor_format == "hsd_persistent_memory_v1" and route_keys
            or tensor_format == "hsd_persistent_memory_v2" and not route_keys):
        raise RuntimeError("Persistent memory tensor format disagrees with route-key contents.")

    config_path = tensor_path.with_name("persistent_memory_config.json")
    if tensor_path.name != "persistent_memory.safetensors" or not config_path.is_file():
        return  # Existing tensor-only artifacts remain supported.
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("Persistent memory metadata must be a JSON object.")
    metadata_format = metadata.get("format")
    if metadata_format is not None and metadata_format not in formats:
        raise RuntimeError(f"Unsupported persistent memory metadata format: {metadata_format!r}.")
    if metadata_format is not None and tensor_format is not None and metadata_format != tensor_format:
        raise RuntimeError("Persistent memory metadata/tensor format mismatch.")
    checksum = metadata.get("tensor_sha256")
    if checksum is not None and checksum != hashlib.sha256(tensor_bytes).hexdigest():
        raise RuntimeError("Persistent memory checksum mismatch; artifact files may be from different saves.")
    shapes = {name: list(value.shape) for name, value in loaded.items()}
    if "tensors" in metadata and metadata["tensors"] != shapes:
        raise RuntimeError("Persistent memory metadata/tensor shape mismatch.")
    if "tensor_count" in metadata and metadata["tensor_count"] != len(loaded):
        raise RuntimeError("Persistent memory metadata tensor count mismatch.")
    source_type = metadata.get("model_type")
    target_type = getattr(getattr(model, "config", None), "model_type", None)
    if source_type is not None and target_type is not None and source_type != target_type:
        raise RuntimeError(f"Persistent memory model type mismatch: artifact={source_type!r} model={target_type!r}.")


def load_persistent_memory_artifact(
    model: nn.Module,
    artifact: str | os.PathLike[str],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate and stage a compact state/key artifact before updating parameters.

    Key, shape, metadata, finite-value and device/dtype conversion failures leave
    the model unchanged. ``strict=False`` still permits partial matching keys.
    """

    from safetensors.torch import load

    artifact_path = Path(artifact)
    tensor_path = artifact_path / "persistent_memory.safetensors" if artifact_path.is_dir() else artifact_path
    if not tensor_path.is_file():
        raise FileNotFoundError(f"persistent memory tensor file not found: {tensor_path}")
    tensor_bytes = tensor_path.read_bytes()
    loaded = load(tensor_bytes)
    parameters = {
        **_persistent_memory_parameters(model),
        **_persistent_memory_route_key_parameters(model),
    }
    if not parameters:
        raise ValueError("model has no persistent memory parameters; construct a memory-enabled model first.")
    expected, actual = set(parameters), set(loaded)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "persistent memory artifact key mismatch: "
            f"missing={missing} unexpected={unexpected}."
        )
    _validate_persistent_memory_metadata(tensor_path, tensor_bytes, loaded, model)
    copied = sorted(expected & actual)
    for name in copied:
        source = loaded[name]
        target = parameters[name]
        if tuple(source.shape) != tuple(target.shape):
            raise RuntimeError(
                f"persistent memory shape mismatch for {name}: "
                f"artifact={tuple(source.shape)} model={tuple(target.shape)}."
            )
        if not source.is_floating_point() or not bool(torch.isfinite(source).all()):
            raise RuntimeError(f"Persistent memory tensor {name} must contain finite floating-point values.")
    staged = {}
    with torch.no_grad():
        for name in copied:
            target = parameters[name]
            converted = loaded[name].to(device=target.device, dtype=target.dtype)
            if not bool(torch.isfinite(converted).all()):
                raise RuntimeError(f"Persistent memory conversion produced non-finite values for {name}.")
            staged[name] = converted
        for name in copied:
            parameters[name].copy_(staged[name])
    return {
        "tensor_path": str(tensor_path),
        "loaded": copied,
        "missing": missing,
        "unexpected": unexpected,
        "parameter_count": sum(parameters[name].numel() for name in copied),
    }


def collect_persistent_memory_metrics(
    model: nn.Module,
    *,
    cumulative: bool = False,
    reset: bool = False,
    cpu: bool = True,
) -> dict[str, dict[str, torch.Tensor | float]]:
    """Collect per-layer persistent-memory routing metrics from an HF model."""

    result = {}
    for name, module in model.named_modules():
        getter = getattr(module, "get_persistent_memory_metrics", None)
        if not callable(getter):
            continue
        metrics = getter(cumulative=cumulative, reset=reset, cpu=cpu)
        if metrics:
            result[name] = metrics
    return result
