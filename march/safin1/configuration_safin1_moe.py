"""Configuration wrappers for safin1_35b_a3b; shared MARCH fields live in configuration_common."""
from huggingface_hub.dataclasses import strict
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig, Qwen3_5MoeTextConfig, Qwen3_5MoeVisionConfig,
)

from .configuration_common import Safin1ConfigValidationMixin, Safin1RuntimeFields
# Declare this dependency directly for fresh Transformers remote-code caches.
from .runtime_validation import validate_safin1_runtime as _remote_runtime_validation  # noqa: F401


@strict
class Safin1MoeTextConfig(Safin1RuntimeFields, Qwen3_5MoeTextConfig):
    """Qwen3.5 text configuration with the shared MARCH schema."""
    model_type = "safin1_35b_a3b_text"


@strict
class Safin1MoeConfig(Safin1ConfigValidationMixin, Qwen3_5MoeConfig):
    """Top-level configuration used by trust_remote_code checkpoints."""
    model_type = "safin1_35b_a3b"
    sub_configs = {
        "vision_config": Qwen3_5MoeVisionConfig,
        "text_config": Safin1MoeTextConfig,
    }


__all__ = ["Safin1MoeConfig", "Safin1MoeTextConfig"]
