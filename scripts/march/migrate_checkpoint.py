#!/usr/bin/env python3
"""Migrate a local checkpoint copy to the current Safin-1 schema.

The migration updates model names, memory-anchor config fields and matching
safetensors parameter keys. Large weight files are staged before replacement,
so hardlinked source checkpoints are never edited through the destination.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from march.safin1.config_utils import (  # noqa: E402
    atomic_write_bytes,
    copy_safin1_remote_code,
    normalize_safin1_config,
    safin1_remote_code_files,
)


_RENAMES = (
    ("configuration_historical_qwen3_5_moe", "configuration_safin1_moe"),
    ("configuration_historical_qwen3_5", "configuration_safin1"),
    ("modeling_historical_qwen3_5_moe", "modeling_safin1_moe"),
    ("modeling_historical_qwen3_5", "modeling_safin1"),
    ("HistoricalQwen3_5Moe", "Safin1Moe"),
    ("HistoricalQwen3_5", "Safin1"),
    ("loglinear.qwen35_historical", "march.safin1"),
    ("loglinear.safin1", "march.safin1"),
    ("loglinear.march", "march.kernels"),
    ("loglinear.hsd", "march.kernels"),
    ("loglinear.ms_swift", "march.ms_swift"),
    ("historical_qwen3_5_moe_text", "safin1_35b_a3b_text"),
    ("historical_qwen3_5_moe", "safin1_35b_a3b"),
    ("historical_qwen3_5_text", "safin1_4b_text"),
    ("historical_qwen3_5", "safin1_4b"),
    ("historical_gdn", "march_gdn"),
    ("fused_hsd_varlen_segments", "fused_march_varlen_segments"),
    ("fused_hsd_segments", "fused_march_segments"),
)
_OBSOLETE_FILES = (
    "configuration_historical_qwen3_5.py",
    "configuration_historical_qwen3_5_moe.py",
    "modeling_historical_qwen3_5.py",
    "modeling_historical_qwen3_5_moe.py",
    "historical_gdn.py",
    "fused_hsd_segments.py",
    "fused_hsd_varlen_segments.py",
)
_ANCHOR_RENAMES = (
    ("SUMMARIES", "ANCHORS"),
    ("Summaries", "Anchors"),
    ("summaries", "anchors"),
    ("SUMMARY", "ANCHOR"),
    ("Summary", "Anchor"),
    ("summary", "anchor"),
)


def _rename_model_metadata(value):
    if isinstance(value, str):
        for old, new in _RENAMES:
            value = value.replace(old, new)
    elif isinstance(value, list):
        return [_rename_model_metadata(item) for item in value]
    elif isinstance(value, dict):
        return {key: _rename_model_metadata(item) for key, item in value.items()}
    return value


def _rename_anchor_text(value: str) -> str:
    for old, new in _ANCHOR_RENAMES:
        value = value.replace(old, new)
    return value


def _rename_anchor_schema(value):
    if isinstance(value, str):
        return _rename_anchor_text(value)
    if isinstance(value, list):
        return [_rename_anchor_schema(item) for item in value]
    if isinstance(value, dict):
        renamed = {}
        for key, item in value.items():
            new_key = _rename_anchor_text(key)
            if new_key in renamed:
                raise ValueError(f"Memory-anchor migration creates duplicate key: {new_key}")
            renamed[new_key] = _rename_anchor_schema(item)
        return renamed
    return value


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _read_safetensors_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Invalid safetensors length prefix: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"Truncated safetensors header: {path}")
    try:
        return header_length, json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid safetensors JSON header: {path}") from error


def _encode_safetensors_header(header: dict, header_length: int) -> bytes:
    payload = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > header_length:
        raise ValueError(
            f"Migrated safetensors header requires {len(payload)} bytes; only {header_length} are available")
    return payload + b" " * (header_length - len(payload))


def _weight_files(checkpoint: Path, index: dict | None) -> list[Path]:
    if index is not None:
        names = sorted(set(index.get("weight_map", {}).values()))
        if not names:
            raise ValueError("model.safetensors.index.json has no weight_map entries")
        files = [checkpoint / name for name in names]
    else:
        files = [checkpoint / "model.safetensors"]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing safetensors files: {missing}")
    return files


def _tensor_migration_plan(checkpoint: Path, index: dict | None) -> list[tuple[Path, int, dict, int]]:
    plan = []
    for path in _weight_files(checkpoint, index):
        header_length, header = _read_safetensors_header(path)
        migrated = _rename_anchor_schema(header)
        old_keys = {key for key in header if key != "__metadata__"}
        new_keys = {key for key in migrated if key != "__metadata__"}
        expected = {_rename_anchor_text(key) for key in old_keys}
        if new_keys != expected:
            raise ValueError(f"Tensor metadata changed unexpectedly: {path}")
        renamed_count = sum(key != _rename_anchor_text(key) for key in old_keys)
        if renamed_count:
            plan.append((path, header_length, migrated, renamed_count))
    return plan


def _stage_tensor_migration(plan: list[tuple[Path, int, dict, int]]) -> list[tuple[Path, Path]]:
    required = sum(path.stat().st_size for path, _, _, _ in plan)
    if plan and shutil.disk_usage(plan[0][0].parent).free < required:
        raise OSError(f"Safetensors migration needs {required} free bytes to stage hardlink-safe copies")
    staged = []
    try:
        for path, header_length, migrated, _ in plan:
            temp = path.with_name(f".{path.name}.anchor-migrating-{os.getpid()}")
            if temp.exists():
                temp.unlink()
            staged.append((temp, path))
            shutil.copy2(path, temp)
            with temp.open("r+b", buffering=0) as stream:
                stream.seek(8)
                stream.write(_encode_safetensors_header(migrated, header_length))
                stream.flush()
                os.fsync(stream.fileno())
            check_length, check_header = _read_safetensors_header(temp)
            if (temp.stat().st_size != path.stat().st_size or check_length != header_length
                    or check_header != migrated):
                raise RuntimeError(f"Staged safetensors validation failed: {path}")
        return staged
    except BaseException:
        for temp, _ in staged:
            temp.unlink(missing_ok=True)
        raise


def migrate_checkpoint(checkpoint: Path, *, dry_run: bool = False) -> dict:
    checkpoint = checkpoint.resolve()
    config_path = checkpoint / "config.json"
    config = normalize_safin1_config(_rename_anchor_schema(json.loads(config_path.read_text())))
    args_path = checkpoint / "args.json"
    args = _rename_anchor_schema(_rename_model_metadata(json.loads(args_path.read_text()))) if args_path.exists() else None
    index_path = checkpoint / "model.safetensors.index.json"
    index = _rename_anchor_schema(json.loads(index_path.read_text())) if index_path.exists() else None
    tensor_plan = _tensor_migration_plan(checkpoint, index)
    if args is not None:
        args["model_type"] = config["model_type"]
        args["model"] = str(checkpoint)
    result = {
        "checkpoint": str(checkpoint),
        "model_type": config["model_type"],
        "architectures": config["architectures"],
        "auto_map": config["auto_map"],
        "remote_code_files": list(safin1_remote_code_files(config["model_type"])),
        "rewritten_weight_shards": len(tensor_plan),
        "renamed_tensor_keys": sum(item[3] for item in tensor_plan),
        "dry_run": dry_run,
    }
    if dry_run:
        return result
    staged = _stage_tensor_migration(tensor_plan)
    try:
        # Publish new metadata only after all referenced modules and staged
        # weight files are complete.
        copy_safin1_remote_code(checkpoint, config["model_type"])
        for temp, path in staged:
            os.replace(temp, path)
        if index is not None:
            atomic_write_bytes(index_path, _json_bytes(index))
        if args is not None:
            atomic_write_bytes(args_path, _json_bytes(args))
        atomic_write_bytes(config_path, _json_bytes(config))
        for filename in _OBSOLETE_FILES:
            path = checkpoint / filename
            if path.is_file():
                path.unlink()
    finally:
        for temp, _ in staged:
            temp.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local checkpoint copy to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Print planned metadata without writing files")
    args = parser.parse_args()
    print(json.dumps(migrate_checkpoint(args.checkpoint, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
