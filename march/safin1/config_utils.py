"""Checkpoint metadata and remote-code export for Safin-1 (MARCH).

This module deliberately has no torch/Transformers dependency. The `hsd_*` and
`anchor_*` runtime fields and matching parameter keys form the checkpoint schema.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import Any


SAFIN1_DENSE_MODEL_TYPE = "safin1_4b"
SAFIN1_DENSE_TEXT_MODEL_TYPE = "safin1_4b_text"
SAFIN1_MODEL_TYPE = "safin1_35b_a3b"
SAFIN1_TEXT_MODEL_TYPE = "safin1_35b_a3b_text"

_VARIANTS = {
    SAFIN1_DENSE_MODEL_TYPE: ("Safin1", "safin1", SAFIN1_DENSE_TEXT_MODEL_TYPE),
    SAFIN1_MODEL_TYPE: ("Safin1Moe", "safin1_moe", SAFIN1_TEXT_MODEL_TYPE),
}
_LEGACY_TYPES = {
    "qwen3_5": SAFIN1_DENSE_MODEL_TYPE,
    "qwen3_5_text": SAFIN1_DENSE_MODEL_TYPE,
    "historical_qwen3_5": SAFIN1_DENSE_MODEL_TYPE,
    "historical_qwen3_5_text": SAFIN1_DENSE_MODEL_TYPE,
    "qwen3_5_moe": SAFIN1_MODEL_TYPE,
    "qwen3_5_moe_text": SAFIN1_MODEL_TYPE,
    "historical_qwen3_5_moe": SAFIN1_MODEL_TYPE,
    "historical_qwen3_5_moe_text": SAFIN1_MODEL_TYPE,
    SAFIN1_DENSE_MODEL_TYPE: SAFIN1_DENSE_MODEL_TYPE,
    SAFIN1_DENSE_TEXT_MODEL_TYPE: SAFIN1_DENSE_MODEL_TYPE,
    SAFIN1_MODEL_TYPE: SAFIN1_MODEL_TYPE,
    SAFIN1_TEXT_MODEL_TYPE: SAFIN1_MODEL_TYPE,
}
_REMOTE_CODE_FILES = (
    "__init__.py",
    "config_utils.py",
    "runtime_validation.py",
    "configuration_common.py",
    "configuration_safin1.py",
    "configuration_safin1_moe.py",
    "modeling_safin1.py",
    "modeling_safin1_moe.py",
    "march_gdn.py",
    "gdn_cache.py",
    "gdn_recurrence.py",
    "persistent_memory.py",
    "anchor_layout.py",
    "cache_utils.py",
    "modeling_common.py",
    "fused_history_read.py",
    "fused_march_segments.py",
    "ops.py",
)


def _variant(model_type: str):
    try:
        return _VARIANTS[model_type]
    except KeyError:
        raise ValueError(f"Unsupported Safin-1 model_type: {model_type!r}") from None


def safin1_remote_code_files(model_type: str) -> tuple[str, ...]:
    _variant(model_type)
    return _REMOTE_CODE_FILES


def safin1_auto_map(model_type: str) -> dict[str, str]:
    class_prefix, module, _ = _variant(model_type)
    return {
        "AutoConfig": f"configuration_{module}.{class_prefix}Config",
        "AutoModelForCausalLM": f"modeling_{module}.{class_prefix}ForConditionalGeneration",
    }


def infer_safin1_target_model_type(config_dict: dict[str, Any]) -> str:
    text_config = config_dict.get("text_config") or {}
    if not isinstance(text_config, dict):
        raise ValueError("text_config must be an object")
    candidates = {
        _LEGACY_TYPES[value] for value in
        (config_dict.get("model_type"), text_config.get("model_type"))
        if value in _LEGACY_TYPES
    }
    if len(candidates) != 1:
        raise ValueError("Cannot infer a unique Safin-1 dense/MoE variant from the checkpoint model types")
    return candidates.pop()


def is_safin1_config(config_dict: dict[str, Any]) -> bool:
    model_type = config_dict.get("model_type")
    if model_type not in _VARIANTS:
        return False
    text_config = config_dict.get("text_config") or {}
    return isinstance(text_config, dict) and text_config.get("model_type") == _VARIANTS[model_type][2]


def normalize_safin1_config(
    config_dict: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Normalize loading metadata while preserving weights and runtime values.

    Serialized runtime fields take precedence over defaults; explicit training
    runtime overrides take precedence over serialized values. A mismatched
    runtime profile is rejected rather than silently changing dense into MoE.
    """
    inferred = infer_safin1_target_model_type(config_dict)
    model_type = model_type or inferred
    class_prefix, _, text_model_type = _variant(model_type)
    if model_type != inferred:
        raise ValueError("Requested Safin-1 variant disagrees with checkpoint model types")
    runtime_config = runtime_config or {}
    runtime_type = runtime_config.get("model_type")
    if runtime_type is not None and _LEGACY_TYPES.get(runtime_type) != model_type:
        raise ValueError("Runtime profile model_type disagrees with the checkpoint variant")
    normalized = deepcopy(config_dict)
    text_config = normalized.setdefault("text_config", {})
    if text_config is None:
        text_config = normalized["text_config"] = {}
    text_config.update({key: deepcopy(value) for key, value in runtime_config.items()
                        if value is not None and key not in {"model_type", "architectures", "auto_map"}})
    normalized["model_type"] = model_type
    normalized["architectures"] = [f"{class_prefix}ForConditionalGeneration"]
    # Keep unrelated processor mappings if present.
    auto_map = dict(normalized.get("auto_map") or {})
    for key in ("AutoModel", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        auto_map.pop(key, None)
    auto_map.update(safin1_auto_map(model_type))
    normalized["auto_map"] = auto_map
    text_config["model_type"] = text_model_type
    # TextConfig is nested under the top-level remote-code config.
    text_config.pop("auto_map", None)
    text_config.pop("architectures", None)
    return normalized


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Replace a file, detaching hardlinks instead of modifying their sources."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def copy_safin1_remote_code(output_dir: str | Path, model_type: str = SAFIN1_MODEL_TYPE) -> list[str]:
    """Copy the complete remote-code closure; reject incomplete source trees."""
    src_dir = Path(__file__).resolve().parent
    files = safin1_remote_code_files(model_type)
    missing = [name for name in files if not (src_dir / name).is_file() or (src_dir / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Incomplete Safin-1 remote-code source: {missing}")
    dst_dir = Path(output_dir)
    copied = []
    for name in files:
        src, dst = src_dir / name, dst_dir / name
        if src.resolve() != dst.resolve():
            atomic_write_bytes(dst, src.read_bytes())
        copied.append(name)
    return copied
