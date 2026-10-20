"""Configuration wrappers for safin1_4b; shared MARCH fields live in configuration_common."""
from huggingface_hub.dataclasses import strict
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
)

from .configuration_common import Safin1ConfigValidationMixin, Safin1RuntimeFields
# Declare this dependency directly for fresh Transformers remote-code caches.
from .runtime_validation import validate_safin1_runtime as _remote_runtime_validation  # noqa: F401


@strict
class Safin1TextConfig(Safin1RuntimeFields, Qwen3_5TextConfig):
    """Qwen3.5 text configuration with the shared MARCH schema."""
    model_type = "safin1_4b_text"


@strict
class Safin1Config(Safin1ConfigValidationMixin, Qwen3_5Config):
    """Top-level configuration used by trust_remote_code checkpoints."""
    model_type = "safin1_4b"
    sub_configs = {
        "vision_config": Qwen3_5VisionConfig,
        "text_config": Safin1TextConfig,
    }


__all__ = ["Safin1Config", "Safin1TextConfig"]
