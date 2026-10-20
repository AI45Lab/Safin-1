"""Supported Safin-1 architecture; inactive historical metadata remains loadable."""

RETIRED_ARCHITECTURE_FLAGS = (
    "hsd_text_only_mlp", "hsd_text_only_in_proj", "hsd_anchor_isolated_in_proj",
    "hsd_text_single_pass_delta", "hsd_anchor_mlp", "hsd_full_attention_anchor_tokens",
)


def validate_safin1_runtime(config):
    get = config.get if isinstance(config, dict) else lambda name, default: getattr(config, name, default)
    enabled = [name for name in RETIRED_ARCHITECTURE_FLAGS if get(name, False)]
    if enabled:
        raise ValueError(f"Retired Safin-1 architecture options cannot be enabled: {enabled}")
    for name, expected in (("hsd_local_state_mode", "full"), ("hsd_history_read_q_mode", "shared_local")):
        value = get(name, expected)
        if value != expected:
            raise ValueError(f"Safin-1 requires {name}={expected!r}; retired variant {value!r} is unsupported")


def validate_safin1_training_runtime(config):
    """Reject training settings that have no implementation before model setup."""
    validate_safin1_runtime(config)
    get = config.get if isinstance(config, dict) else lambda name, default: getattr(config, name, default)
    for name, expected in (("hsd_history_mix_mode", "additive"),
                           ("hsd_route_weight_momentum_gamma", 0.0)):
        value = get(name, expected)
        if value != expected:
            raise ValueError(
                f"Safin-1 training requires {name}={expected!r}; "
                f"{value!r} is not implemented and would not change the computation."
            )


FORBIDDEN_TRAINING_ENV = (
    "HSD_DEBUG_DETACH_TEXT_O", "HSD_DEBUG_ZERO_TEXT_O",
    *(name.upper() for name in RETIRED_ARCHITECTURE_FLAGS),
)


def validate_training_environment(environ=None, *, config=None, require_march=False):
    """Reject inherited experimental switches before loading or allocating a model."""
    if environ is None:
        import os
        environ = os.environ
    if require_march or is_march_training_config(config) or str(environ.get("USE_HIST_LINEAR_ATTN", "")).strip().lower() in _TRUE:
        for name in ("USE_HIST_LINEAR_ATTN", "USE_MCORE_GDN", "SWIFT_USE_MCORE_GDN"):
            value = str(environ.get(name, "")).strip().lower()
            if value and value not in _TRUE:
                raise ValueError(f"Safin-1/MARCH training requires {name}=1; conflicting value {value!r}")
    if config is not None:
        validate_runtime_environment(config, environ)
    enabled = [name for name in FORBIDDEN_TRAINING_ENV
               if str(environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}]
    if enabled:
        raise ValueError(f"Unsupported training environment options: {enabled}; unset them before training")
    for name, expected in (("HSD_LOCAL_STATE_MODE", "full"),
                           ("HSD_HISTORY_READ_Q_MODE", "shared_local")):
        value = str(environ.get(name, "")).strip().lower()
        if value and value != expected:
            raise ValueError(f"Unsupported training environment {name}={value!r}; only {expected!r} is supported")


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def is_march_training_config(config):
    """Recognize the architecture independently of process-local switches."""
    if config is None:
        return False
    get = config.get if isinstance(config, dict) else lambda name, default=None: getattr(config, name, default)
    for name in ("model_type", "hf_model_type", "llm_model_type"):
        model_type = str(get(name, "") or "")
        if model_type.startswith(("safin1_", "historical_qwen3_5")):
            return True
    if int(get("anchor_chunk_size", 0) or 0) > 0 and int(get("anchor_token_num", 0) or 0) > 0:
        return True
    nested = get("hf_config")
    return nested is not None and nested is not config and is_march_training_config(nested)


def validate_runtime_environment(config, environ):
    """Legacy runtime environment values may confirm, but never override, config.

    Performance-only kernel tuning variables have no serialized config field
    and remain available. ModelConfig supplies all serialized runtime defaults.
    """
    import json
    import math

    values = config if isinstance(config, dict) else vars(config)
    for name, raw in environ.items():
        field = "hsd_read_block_size" if name == "HIST_READ_BLOCK_SIZE" else name.lower()
        if (not field.startswith("hsd_") or field == "hsd_runtime_config"
                or field not in values or not str(raw).strip()):
            continue
        expected = values[field]
        raw = str(raw).strip()
        try:
            if isinstance(expected, bool):
                if raw.lower() not in _TRUE | _FALSE:
                    raise ValueError("expected a boolean")
                actual = raw.lower() in _TRUE
            elif isinstance(expected, int):
                actual = int(raw)
            elif isinstance(expected, float):
                actual = float(raw)
                if not math.isfinite(actual):
                    raise ValueError("expected a finite number")
            else:
                try:
                    actual = json.loads(raw)
                except ValueError:
                    actual = None if raw.lower() in {"none", "null"} else raw
                if isinstance(expected, str):
                    actual = str(actual).strip().lower()
                    expected = expected.strip().lower()
                if isinstance(expected, tuple):
                    expected = list(expected)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid training environment {name}={raw!r}: {exc}") from exc
        if actual != expected:
            raise ValueError(
                f"Training environment {name}={raw!r} conflicts with serialized {field}={expected!r}. "
                "Unset the environment variable and use --hsd-runtime-config or an explicit CLI field; "
                "training and saved configuration must agree."
            )


def march_training_enabled(config, environ=None):
    if environ is None:
        import os
        environ = os.environ
    enabled = is_march_training_config(config) or str(environ.get("USE_HIST_LINEAR_ATTN", "")).strip().lower() in _TRUE
    validate_training_environment(environ, config=config, require_march=enabled)
    return enabled
