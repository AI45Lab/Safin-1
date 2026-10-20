"""Safin-1 models implementing MARCH on the Qwen3.5 backbone.

Configuration/export helpers are usable without torch or Transformers. Model
classes and their optional dependencies are imported only when requested.
"""

from importlib import import_module

_EXPORTS = {
    "Safin1Config": "configuration_safin1",
    "Safin1TextConfig": "configuration_safin1",
    "Safin1MoeConfig": "configuration_safin1_moe",
    "Safin1MoeTextConfig": "configuration_safin1_moe",
    "Safin1ForConditionalGeneration": "modeling_safin1",
    "Safin1MoeForConditionalGeneration": "modeling_safin1_moe",
}
__all__ = [*_EXPORTS, "register_safin1_models"]


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def register_safin1_models():
    """Register local Safin-1 classes for AutoConfig/AutoModel construction.

    Published checkpoints load via their auto_map and trust_remote_code=True.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    for prefix in ("Safin1", "Safin1Moe"):
        config = __getattr__(f"{prefix}Config")
        text_config = __getattr__(f"{prefix}TextConfig")
        model = __getattr__(f"{prefix}ForConditionalGeneration")
        AutoConfig.register(config.model_type, config, exist_ok=True)
        AutoConfig.register(text_config.model_type, text_config, exist_ok=True)
        AutoModelForCausalLM.register(config, model, exist_ok=True)
