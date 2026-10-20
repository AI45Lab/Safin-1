# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from safetensors.torch import safe_open, save_file


_MEMORY_SUFFIX = '.persistent_memory_state'
_ROUTE_KEY_SUFFIX = '.persistent_memory_route_key'
_MEMORY_KEY_RE = re.compile(
    r'(?:^|\.)layers\.(\d+)\..*\.(?:persistent_memory_state|persistent_memory_route_key)$')


def _is_memory_key(key: str) -> bool:
    return key in {'persistent_memory_state', 'persistent_memory_route_key'} or key.endswith(
        (_MEMORY_SUFFIX, _ROUTE_KEY_SUFFIX))


def _memory_weight_map(model_dir: Path) -> dict[str, str]:
    """Return only persistent-memory keys and their source shard names."""
    index_path = model_dir / 'model.safetensors.index.json'
    if index_path.is_file():
        with index_path.open('r', encoding='utf-8') as f:
            index = json.load(f)
        weight_map = index.get('weight_map') or {}
        if not isinstance(weight_map, dict):
            raise RuntimeError(f'Invalid safetensors weight_map in {index_path}.')
        result = {}
        for key, filename in weight_map.items():
            if not _is_memory_key(str(key)):
                continue
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise RuntimeError(f'Invalid safetensors shard name for {key!r}: {filename!r}.')
            result[str(key)] = filename
        return result

    single_file = model_dir / 'model.safetensors'
    shard_paths = [single_file] if single_file.is_file() else sorted(model_dir.glob('model-*.safetensors'))
    result = {}
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if _is_memory_key(key):
                    if key in result:
                        raise RuntimeError(f'Duplicate persistent-memory tensor {key!r}.')
                    result[key] = shard_path.name
    return result


def _load_memory_tensors(model_dir: Path, weight_map: dict[str, str]) -> dict[str, Any]:
    """Load matching tensors only; safetensors shards remain lazily mapped."""
    keys_by_shard = defaultdict(list)
    for key, filename in weight_map.items():
        keys_by_shard[filename].append(key)

    tensors = {}
    for filename, keys in sorted(keys_by_shard.items()):
        shard_path = model_dir / filename
        if not shard_path.is_file():
            raise FileNotFoundError(f'Persistent-memory source shard does not exist: {shard_path}')
        with safe_open(str(shard_path), framework='pt', device='cpu') as handle:
            available = set(handle.keys())
            for key in sorted(keys):
                if key not in available:
                    raise RuntimeError(f'Safetensors index maps {key!r} to {filename!r}, but it is absent.')
                # get_tensor is invoked only for the selected memory keys.  No
                # unrelated parameter from this potentially multi-GB shard is
                # materialized.
                tensors[key] = handle.get_tensor(key).contiguous()
    return tensors


def _layer_indices(keys: list[str]) -> list[int]:
    layers = []
    for key in keys:
        match = _MEMORY_KEY_RE.search(key)
        if match is None:
            raise RuntimeError(f'Cannot derive the HF layer index from persistent-memory key {key!r}.')
        layers.append(int(match.group(1)))
    return sorted(set(layers))


def _config_int(config: Any, name: str, fallback: Optional[str] = None) -> int:
    value = getattr(config, name, None)
    if value is None and fallback is not None:
        value = getattr(config, fallback, None)
    return int(value or 0)


def export_persistent_memory_artifact(output_dir: str, config: Any) -> Optional[dict[str, Any]]:
    """Extract exported HF memory states into a small standalone artifact.

    Returns the JSON metadata, or ``None`` when persistent memory is disabled.
    Dedicated checkpoints include one state/key pair for every active
    global-read layer.  Legacy state-only checkpoints retain their broader
    causal-path state allocation.
    """
    if not bool(getattr(config, 'hsd_persistent_memory_enabled', False)):
        return None

    model_dir = Path(output_dir)
    weight_map = _memory_weight_map(model_dir)
    if not weight_map:
        raise RuntimeError(
            f'Persistent HSD memory is enabled, but no *{_MEMORY_SUFFIX} tensors were exported to {model_dir}.')
    tensors = _load_memory_tensors(model_dir, weight_map)
    ordered_keys = sorted(tensors, key=lambda key: (_layer_indices([key])[0], key))
    tensors = {key: tensors[key] for key in ordered_keys}

    dtypes = {str(tensor.dtype).removeprefix('torch.') for tensor in tensors.values()}
    if dtypes != {'float32'}:
        raise RuntimeError(f'Persistent-memory artifacts require FP32 states, got dtypes={sorted(dtypes)}.')
    layers = _layer_indices(ordered_keys)
    state_tensors = {key: value for key, value in tensors.items() if key.endswith(_MEMORY_SUFFIX)}
    route_key_tensors = {key: value for key, value in tensors.items() if key.endswith(_ROUTE_KEY_SUFFIX)}
    state_shapes = {tuple(int(dim) for dim in tensor.shape) for tensor in state_tensors.values()}
    route_key_shapes = {tuple(int(dim) for dim in tensor.shape) for tensor in route_key_tensors.values()}
    route_key_enabled = bool(
        getattr(config, 'hsd_persistent_memory_route_key_enabled', False))
    if route_key_enabled and len(route_key_tensors) != len(state_tensors):
        raise RuntimeError(
            'Dedicated persistent memory requires one route key per state: '
            f'states={len(state_tensors)}, route_keys={len(route_key_tensors)}.')
    if not route_key_enabled and route_key_tensors:
        raise RuntimeError(
            'Route-key tensors were exported while hsd_persistent_memory_route_key_enabled=false.')
    parameter_count = sum(int(tensor.numel()) for tensor in tensors.values())
    metadata = {
        'schema': (
            'swift-hsd-persistent-memory-v2'
            if route_key_enabled
            else 'swift-hsd-persistent-memory-v1'
        ),
        'enabled': True,
        'layers': layers,
        'layer_index_base': 0,
        'layer_keys': ordered_keys,
        'chunk': _config_int(config, 'anchor_chunk_size', 'hist_chunk_size'),
        'topk': _config_int(config, 'hsd_read_topk'),
        'null': bool(getattr(config, 'hsd_null_read_gate', False)),
        'tensor_count': len(tensors),
        'state_tensor_count': len(state_tensors),
        'route_key_tensor_count': len(route_key_tensors),
        'parameter_count': parameter_count,
        'dtype': 'float32',
        'state_shape': (
            list(next(iter(state_shapes)))
            if len(state_shapes) == 1
            else [list(shape) for shape in sorted(state_shapes)]
        ),
        'route_key_shape': (
            list(next(iter(route_key_shapes)))
            if len(route_key_shapes) == 1
            else [list(shape) for shape in sorted(route_key_shapes)]
        ),
        'key_source': (
            'per_layer_dedicated_parameter'
            if route_key_enabled
            else 'anchor_hidden->anchor_k_proj'
        ),
        'route_key_mode': str(
            getattr(config, 'hsd_persistent_memory_route_key_mode', 'legacy')),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = model_dir / 'persistent_memory.safetensors'
    config_path = model_dir / 'persistent_memory_config.json'
    unique = f'{os.getpid()}'
    tensor_tmp = model_dir / f'.persistent_memory.safetensors.tmp.{unique}'
    config_tmp = model_dir / f'.persistent_memory_config.json.tmp.{unique}'
    try:
        save_file(
            tensors,
            str(tensor_tmp),
            metadata={'format': 'pt', 'artifact': 'hsd_persistent_memory_state'},
        )
        with config_tmp.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write('\n')
        os.replace(tensor_tmp, tensor_path)
        os.replace(config_tmp, config_path)
    finally:
        for tmp_path in (tensor_tmp, config_tmp):
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
    return metadata
