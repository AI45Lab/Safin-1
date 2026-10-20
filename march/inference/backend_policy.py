"""Per-call paged decode policy; environment variables supply compatibility defaults."""

from __future__ import annotations

from dataclasses import dataclass
import os


def normalize_history_read_backend(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {
        "": "auto", "auto": "auto", "torch": "torch", "aten": "torch",
        "paged_torch": "torch", "tensor": "torch", "triton": "triton",
        "split_triton": "triton", "fused": "fused_triton", "fused_triton": "fused_triton",
    }
    if value not in aliases:
        raise ValueError(
            "Paged history backend (HSD_PAGED_HISTORY_READ_IMPL) must be one of "
            f"auto, torch, triton, or fused_triton; got {value!r}."
        )
    return aliases[value]


@dataclass(frozen=True)
class PagedHistoryReadPolicy:
    """Resolve backend and diagnostics once, then carry them through split batches."""

    backend: str
    validate_fused: bool
    record_backend: bool = False

    @classmethod
    def resolve(cls, *, backend: str | None = None, validate_fused: bool | None = None,
                record_backend: bool | None = None):
        if backend is None:
            backend = os.environ.get("HSD_PAGED_HISTORY_READ_IMPL", "auto")
        if validate_fused is None:
            validate_fused = os.environ.get("HSD_PAGED_HISTORY_VALIDATE_FUSED", "").strip().lower() in {
                "1", "true", "yes", "on",
            }
        if not isinstance(validate_fused, bool):
            raise TypeError("validate_fused must be a bool or None")
        if record_backend is None:
            record_backend = os.environ.get("HSD_PAGED_HISTORY_BACKEND_EVENTS", "").strip().lower() in {
                "1", "true", "yes", "on",
            }
        if not isinstance(record_backend, bool):
            raise TypeError("record_backend must be a bool or None")
        return cls(normalize_history_read_backend(backend), validate_fused, record_backend)
