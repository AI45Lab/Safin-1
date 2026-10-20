"""Logical page allocation and storage specification for MARCH history."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from .paged_utils import _normalize_state_format


@dataclass
class _AnchorEntry:
    page_ids: list[int]
    length: int


@dataclass(frozen=True)
class HSDHistoryCacheSpec:
    """Storage shape for one fixed HSD history anchor page pool."""

    page_m: int
    num_pages: int
    heads: int
    value_dim: int
    key_dim: int
    states_dtype: torch.dtype
    keys_dtype: torch.dtype
    route_heads: int | None = None
    route_dim: int | None = None
    state_format: str = "dense"

    def __post_init__(self) -> None:
        _normalize_state_format(self.state_format)


class HSDHistoryCacheManager:
    """CPU-side logical page allocator for HSD history anchors.

    The manager intentionally owns only request/state metadata. Tensor storage
    remains in ``HSDAnchorPageCache`` so scheduler accounting can evolve
    independently from the model-side storage layout.
    """

    def __init__(self, *, page_m: int = 16, num_pages: int) -> None:
        if page_m <= 0:
            raise ValueError(f"page_m must be positive, got {page_m}.")
        if num_pages < 0:
            raise ValueError(f"num_pages must be non-negative, got {num_pages}.")
        self.page_m = int(page_m)
        self.num_pages = int(num_pages)
        self.entries: dict[int, _AnchorEntry] = {}
        self.free_pages: list[int] = list(range(self.num_pages - 1, -1, -1))
        self.alloc_failures = 0
        self.scheduler_throttles = 0

    @property
    def total_pages(self) -> int:
        return self.num_pages

    @property
    def free_page_count(self) -> int:
        return len(self.free_pages)

    @property
    def used_pages(self) -> int:
        return self.num_pages - len(self.free_pages)

    def length(self, state_idx: int) -> int:
        entry = self.entries.get(int(state_idx))
        return 0 if entry is None else int(entry.length)

    def lengths(
        self,
        state_indices: torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        target_device = state_indices.device if device is None else device
        indices = [int(v) for v in state_indices.detach().cpu().tolist()]
        values = [self.length(state_idx) for state_idx in indices]
        return torch.tensor(values, device=target_device, dtype=torch.long)

    def page_ids(self, state_idx: int) -> tuple[int, ...]:
        entry = self.entries.get(int(state_idx))
        return () if entry is None else tuple(entry.page_ids)

    def pages_needed_for_append(self, state_idx: int, count: int) -> int:
        if count <= 0:
            return 0
        entry = self.entries.get(int(state_idx))
        old_length = 0 if entry is None else int(entry.length)
        old_pages = 0 if entry is None else len(entry.page_ids)
        new_length = old_length + int(count)
        new_pages = (new_length + self.page_m - 1) // self.page_m
        return max(0, new_pages - old_pages)

    def can_append(self, state_idx: int, count: int) -> bool:
        return self.pages_needed_for_append(state_idx, count) <= len(self.free_pages)

    def reserve_append(self, state_idx: int, count: int) -> bool:
        count = int(count)
        if count <= 0:
            return True
        needed = self.pages_needed_for_append(state_idx, count)
        if needed > len(self.free_pages):
            self.alloc_failures += 1
            self.scheduler_throttles += 1
            return False
        entry = self.entries.setdefault(int(state_idx), _AnchorEntry(page_ids=[], length=0))
        for _ in range(needed):
            entry.page_ids.append(self.free_pages.pop())
        entry.length += count
        return True

    def release(self, state_indices: set[int] | list[int] | tuple[int, ...] | None = None) -> int:
        if state_indices is None:
            count = len(self.entries)
            self.entries.clear()
            self.free_pages = list(range(self.num_pages - 1, -1, -1))
            return count
        count = 0
        for state_idx in state_indices:
            entry = self.entries.pop(int(state_idx), None)
            if entry is None:
                continue
            count += 1
            self.free_pages.extend(entry.page_ids)
        return count

    def stats(self) -> dict[str, int]:
        return {
            "history_pages_used": int(self.used_pages),
            "history_pages_free": int(self.free_page_count),
            "history_pages_total": int(self.total_pages),
            "history_page_alloc_failures": int(self.alloc_failures),
            "history_scheduler_throttles": int(self.scheduler_throttles),
        }

