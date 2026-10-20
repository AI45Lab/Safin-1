# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
from typing import Any, Iterable


def _runtime_config_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        if os.path.isfile(value):
            with open(value, 'r', encoding='utf-8') as f:
                parsed = json.load(f)
        else:
            parsed = json.loads(value)
    except (OSError, json.JSONDecodeError):
        # Model construction validates this value separately and provides a
        # better user-facing error.  Checkpoint loading only needs a safe
        # feature-detection fallback.
        return {}
    return parsed if isinstance(parsed, dict) else {}


def hsd_persistent_memory_enabled(args: Any, models: Iterable[Any]) -> bool:
    """Resolve persistent-memory enablement after the MCore model is built.

    HSD launchers normally carry this option inside ``hsd_runtime_config``;
    it is therefore not guaranteed to exist as a direct Megatron argument.
    The resolved model config is authoritative, with direct/runtime args kept
    as fallbacks for wrappers that do not expose that config.
    """
    for model in models or ():
        language_model = getattr(model, 'language_model', None)
        for candidate in (model, language_model):
            config = getattr(candidate, 'config', None)
            if bool(getattr(config, 'hsd_persistent_memory_enabled', False)):
                return True
    if bool(getattr(args, 'hsd_persistent_memory_enabled', False)):
        return True
    runtime_config = _runtime_config_dict(getattr(args, 'hsd_runtime_config', None))
    return bool(runtime_config.get('hsd_persistent_memory_enabled', False))
