"""MARCH paged history: history cache."""

from __future__ import annotations

from collections import OrderedDict
import os
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU reference remains available without Triton.
    triton = None
    tl = None
from .paged_utils import (
    _env_truthy,
    _export_cache_events_enabled,
    _normalize_state_format,
    _record_paged_history_event,
)


from .cache_metadata import _AnchorEntry, HSDHistoryCacheSpec, HSDHistoryCacheManager


class HSDAnchorPageCache:
    """Small paged cache for per-state HSD anchor states and route keys.

    The tensors use vLLM's history-read state layout:
    ``states_pages[page, slot, head, value, key]``.
    """

    def __init__(
        self,
        page_m: int = 16,
        *,
        state_format: str = "dense",
    ) -> None:
        if page_m <= 0:
            raise ValueError(f"page_m must be positive, got {page_m}.")
        self.page_m = int(page_m)
        self.entries: dict[int, _AnchorEntry] = {}
        self.free_pages: list[int] = []
        self.state_format = _normalize_state_format(state_format)
        self.states_pages: torch.Tensor | None = None
        self.keys_pages: torch.Tensor | None = None
        self.boundary_pages: torch.Tensor | None = None
        self.heads: int | None = None
        self.value_dim: int | None = None
        self.key_dim: int | None = None
        self.route_heads: int | None = None
        self.route_dim: int | None = None
        self.fixed_capacity: bool = False
        self._lengths_by_state: torch.Tensor | None = None
        self._page_table_by_state: torch.Tensor | None = None
        self._metadata_fixed_capacity: bool = False
        self._max_anchor_len: int = 0
        self._metadata_version: int = 0
        self._export_page_table_cache: OrderedDict[
            tuple[object, ...], tuple[torch.Tensor, torch.Tensor, int]
        ] = OrderedDict()
        self.history_read_time_us = 0
        self.materialize_fallback_count = 0
        # Boundary zero is reserved for the fixed persistent-memory row.  The
        # CPU set lets graph/replay entrypoints fail closed without a device
        # synchronization on every decode token.
        self._persistent_memory_states: set[int] = set()
        self._invalid_persistent_memory_states: set[int] = set()
        self._persistent_memory_validated_batches: set[
            tuple[int, tuple[int, ...], tuple[bool, ...]]
        ] = set()


    @property
    def _primary_state_pages(self) -> torch.Tensor | None:
        return self.states_pages

    @property
    def device(self) -> torch.device | None:
        pages = self._primary_state_pages
        return None if pages is None else pages.device

    @property
    def dtype(self) -> torch.dtype | None:
        pages = self._primary_state_pages
        return None if pages is None else pages.dtype

    @property
    def initialized(self) -> bool:
        pages = self._primary_state_pages
        return (
            pages is not None
            and self.keys_pages is not None
            and self.boundary_pages is not None
        )


    @property
    def allocated_page_bytes(self) -> int:
        if self.keys_pages is None or self.dtype is None:
            return 0
        assert self.heads is not None
        assert self.value_dim is not None and self.key_dim is not None
        state_elements = (
            self.page_m * self.heads * self.value_dim * self.key_dim
        )
        assert self.route_heads is not None and self.route_dim is not None
        key_elements = self.page_m * self.route_heads * self.route_dim
        boundary_elements = self.page_m
        return (
            state_elements * self.dtype.itemsize
            + key_elements * self.keys_pages.dtype.itemsize
            + boundary_elements * torch.tensor([], dtype=torch.long).element_size()
        )

    def stats(self) -> dict[str, object]:
        allocated_bytes = self.allocated_page_bytes
        return {
            "state_format": self.state_format,
            "state_heads": self.heads,
            "state_key_dim": self.key_dim,
            "route_heads": self.route_heads,
            "route_dim": self.route_dim,
            "allocated_page_bytes": allocated_bytes,
            "history_read_time_us": int(self.history_read_time_us),
            "materialize_fallback_count": int(self.materialize_fallback_count),
        }

    def clear(self, state_indices: set[int] | list[int] | tuple[int, ...] | None = None) -> int:
        if state_indices is None:
            count = len(self.entries)
            self.entries.clear()
            self.free_pages.clear()
            self.states_pages = None
            self.keys_pages = None
            self.boundary_pages = None
            self.heads = None
            self.value_dim = None
            self.key_dim = None
            self.route_heads = None
            self.route_dim = None
            self.fixed_capacity = False
            self._lengths_by_state = None
            self._page_table_by_state = None
            self._metadata_fixed_capacity = False
            self._max_anchor_len = 0
            self.history_read_time_us = 0
            self.materialize_fallback_count = 0
            self._persistent_memory_states.clear()
            self._invalid_persistent_memory_states.clear()
            self._bump_metadata_version()
            return count

        count = 0
        for state_idx in state_indices:
            entry = self.entries.pop(int(state_idx), None)
            if entry is None:
                continue
            count += 1
            self.free_pages.extend(entry.page_ids)
            self._persistent_memory_states.discard(int(state_idx))
            self._invalid_persistent_memory_states.discard(int(state_idx))
            self._clear_metadata_entry(int(state_idx))
        if count:
            self._refresh_metadata_max_anchor_len()
            self._bump_metadata_version()
        return count

    def reset_runtime_state(self) -> int:
        """Drop warmup/request ownership while preserving allocated tensors."""
        count = len(self.entries)
        self.entries.clear()
        self.free_pages = list(range(self.total_pages - 1, -1, -1))
        if self._lengths_by_state is not None:
            self._lengths_by_state.zero_()
        if self._page_table_by_state is not None:
            self._page_table_by_state.fill_(-1)
        if self.boundary_pages is not None:
            self.boundary_pages.fill_(2**62)
        self._max_anchor_len = 0
        self.history_read_time_us = 0
        self.materialize_fallback_count = 0
        self._persistent_memory_states.clear()
        self._invalid_persistent_memory_states.clear()
        self._bump_metadata_version()
        return count

    def remap_state_indices(self, mapping: dict[int, int]) -> int:
        """Move cache ownership between fixed state rows without moving pages."""
        normalized = {
            int(source): int(target)
            for source, target in mapping.items()
            if int(source) != int(target)
        }
        if not normalized:
            return 0
        sources = set(normalized)
        targets = set(normalized.values())
        if len(targets) != len(normalized):
            raise ValueError("HSD anchor state remap targets must be unique.")
        if min(sources | targets) < 0:
            raise ValueError("HSD anchor state remap indices must be non-negative.")
        occupied_targets = {
            target
            for target in targets
            if target in self.entries and target not in sources
        }
        if occupied_targets:
            raise RuntimeError(
                "HSD anchor state remap would overwrite live targets: "
                f"{sorted(occupied_targets)}"
            )
        if self._lengths_by_state is not None:
            capacity = int(self._lengths_by_state.shape[0])
            if max(sources | targets) >= capacity:
                if self._metadata_fixed_capacity:
                    raise RuntimeError(
                        "HSD anchor state remap exceeds preallocated metadata "
                        f"rows: capacity={capacity} mapping={normalized}"
                    )
                self._ensure_metadata_state_capacity(
                    max(sources | targets),
                    device=self._lengths_by_state.device,
                )

        moved_entries = {
            source: self.entries.get(source)
            for source in normalized
        }
        moved_memory = {
            source: source in self._persistent_memory_states
            for source in normalized
        }
        moved_invalid_memory = {
            source: source in self._invalid_persistent_memory_states
            for source in normalized
        }
        for source in sources:
            self.entries.pop(source, None)
            self._persistent_memory_states.discard(source)
            self._invalid_persistent_memory_states.discard(source)
        moved = 0
        for source, target in normalized.items():
            entry = moved_entries[source]
            if entry is not None:
                self.entries[target] = entry
                moved += 1
            if moved_memory[source]:
                self._persistent_memory_states.add(target)
            if moved_invalid_memory[source]:
                self._invalid_persistent_memory_states.add(target)

        if self._lengths_by_state is not None:
            device = self._lengths_by_state.device
            source_indices = torch.tensor(
                list(normalized),
                device=device,
                dtype=torch.long,
            )
            target_indices = torch.tensor(
                list(normalized.values()),
                device=device,
                dtype=torch.long,
            )
            moved_lengths = self._lengths_by_state.index_select(
                0, source_indices
            ).clone()
            affected_indices = torch.tensor(
                sorted(sources | targets),
                device=device,
                dtype=torch.long,
            )
            self._lengths_by_state.index_fill_(0, affected_indices, 0)
            self._lengths_by_state.index_copy_(
                0,
                target_indices,
                moved_lengths,
            )
            if self._page_table_by_state is not None:
                moved_pages = self._page_table_by_state.index_select(
                    0, source_indices
                ).clone()
                self._page_table_by_state.index_fill_(
                    0,
                    affected_indices,
                    -1,
                )
                self._page_table_by_state.index_copy_(
                    0,
                    target_indices,
                    moved_pages,
                )
        self._bump_metadata_version()
        return moved

    def _export_cache_limit(self) -> int:
        raw = os.environ.get("HSD_PAGED_HISTORY_EXPORT_CACHE_SIZE", "128").strip()
        try:
            value = int(raw)
        except ValueError:
            value = 128
        return max(0, value)

    def _export_cache_enabled(self) -> bool:
        return (
            not _env_truthy("HSD_DISABLE_PAGED_HISTORY_EXPORT_CACHE")
            and self._export_cache_limit() > 0
        )

    def _bump_metadata_version(self) -> None:
        self._metadata_version += 1
        self._export_page_table_cache.clear()
        self._persistent_memory_validated_batches.clear()

    def assert_persistent_memory_ready(
        self,
        state_indices: torch.Tensor | tuple[int, ...] | list[int],
        *,
        valid_mask: torch.Tensor | tuple[bool, ...] | list[bool] | None = None,
    ) -> None:
        """Fail closed unless every active state owns one boundary-zero row."""
        if torch.is_tensor(state_indices):
            indices = tuple(
                int(value)
                for value in state_indices.detach().cpu().tolist()
            )
        else:
            indices = tuple(int(value) for value in state_indices)
        if valid_mask is None:
            valid = tuple(True for _ in indices)
        elif torch.is_tensor(valid_mask):
            valid = tuple(
                bool(value) for value in valid_mask.detach().cpu().tolist()
            )
        else:
            valid = tuple(bool(value) for value in valid_mask)
        if len(valid) != len(indices):
            raise ValueError("Persistent-memory state and valid-mask sizes differ.")
        key = (self._metadata_version, indices, valid)
        if key in self._persistent_memory_validated_batches:
            return
        metadata_rows = (
            0
            if self._lengths_by_state is None
            else int(self._lengths_by_state.shape[0])
        )
        missing = sorted(
            {
                state_idx
                for state_idx, is_valid in zip(indices, valid, strict=True)
                if is_valid
                and (
                    state_idx < 0
                    or state_idx >= metadata_rows
                    or state_idx in self._invalid_persistent_memory_states
                    or state_idx not in self._persistent_memory_states
                )
            }
        )
        if missing:
            raise RuntimeError(
                "Persistent-memory cache requires exactly one boundary-zero "
                f"row for every active state; missing/invalid states={missing}."
            )
        self._persistent_memory_validated_batches.add(key)

    @property
    def total_pages(self) -> int:
        pages = self._primary_state_pages
        return 0 if pages is None else int(pages.shape[0])

    @property
    def used_pages(self) -> int:
        return self.total_pages - len(self.free_pages)

    def preallocate(
        self,
        *,
        spec: HSDHistoryCacheSpec,
        device: torch.device,
    ) -> None:
        """Allocate a fixed-size pool used by scheduler-budgeted decode."""

        if spec.num_pages <= 0:
            raise ValueError(f"num_pages must be positive, got {spec.num_pages}.")
        if spec.page_m != self.page_m:
            raise ValueError(
                f"page_m mismatch: cache={self.page_m}, spec={spec.page_m}."
            )
        self.clear(None)
        self.state_format = _normalize_state_format(spec.state_format)
        self.heads = int(spec.heads)
        self.value_dim = int(spec.value_dim)
        self.key_dim = int(spec.key_dim)
        self.route_heads = int(spec.route_heads or spec.heads)
        self.route_dim = int(spec.route_dim or spec.key_dim)
        if self.route_heads not in {1, self.heads}:
            raise ValueError(
                "route_heads must be one or match state heads, got "
                f"route_heads={self.route_heads}, heads={self.heads}."
            )
        self.states_pages = torch.empty(
            int(spec.num_pages),
            self.page_m,
            self.heads,
            self.value_dim,
            self.key_dim,
            device=device,
            dtype=spec.states_dtype,
        )
        self.keys_pages = torch.empty(
            int(spec.num_pages),
            self.page_m,
            self.route_heads,
            self.route_dim,
            device=device,
            dtype=spec.keys_dtype,
        )
        self.boundary_pages = torch.empty(
            int(spec.num_pages),
            self.page_m,
            device=device,
            dtype=torch.long,
        )
        self.boundary_pages.fill_(2**62)
        self.free_pages = list(range(int(spec.num_pages) - 1, -1, -1))
        self.fixed_capacity = True

    def preallocate_metadata(
        self,
        *,
        state_rows: int,
        max_pages_per_state: int,
        device: torch.device,
    ) -> None:
        """Keep page-table metadata pointers stable for CUDA graph replay."""
        if state_rows <= 0 or max_pages_per_state <= 0:
            raise ValueError(
                "state_rows and max_pages_per_state must both be positive, got "
                f"{state_rows=} {max_pages_per_state=}."
            )
        self._lengths_by_state = torch.zeros(
            int(state_rows),
            device=device,
            dtype=torch.long,
        )
        self._page_table_by_state = torch.full(
            (int(state_rows), int(max_pages_per_state)),
            -1,
            device=device,
            dtype=torch.long,
        )
        self._max_anchor_len = 0
        self._metadata_fixed_capacity = True
        self._bump_metadata_version()

    def has_state(self, state_idx: int) -> bool:
        entry = self.entries.get(int(state_idx))
        return entry is not None and entry.length > 0

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
        cached = self._lookup_metadata_lengths(
            state_indices.to(device=target_device, dtype=torch.long)
        )
        if cached is not None:
            return cached
        indices = [int(v) for v in state_indices.detach().cpu().tolist()]
        values = [self.length(state_idx) for state_idx in indices]
        return torch.tensor(values, device=target_device, dtype=torch.long)

    def _metadata_device(self, fallback: torch.device) -> torch.device:
        if self.device is not None:
            return self.device
        if self._lengths_by_state is not None:
            return self._lengths_by_state.device
        return fallback

    def _ensure_metadata_state_capacity(
        self,
        state_idx: int,
        *,
        device: torch.device,
    ) -> None:
        if state_idx < 0:
            raise ValueError(f"state_idx must be non-negative, got {state_idx}.")
        current_rows = (
            0
            if self._lengths_by_state is None
            else int(self._lengths_by_state.shape[0])
        )
        if current_rows > state_idx:
            return
        if self._metadata_fixed_capacity:
            raise RuntimeError(
                "Fixed HSD history metadata state capacity exceeded: "
                f"state_idx={state_idx}, state_rows={current_rows}."
            )
        new_rows = max(state_idx + 1, max(16, current_rows * 2))
        new_lengths = torch.zeros(new_rows, device=device, dtype=torch.long)
        if self._lengths_by_state is not None:
            new_lengths[:current_rows].copy_(self._lengths_by_state.to(device))
        self._lengths_by_state = new_lengths

        current_cols = (
            0
            if self._page_table_by_state is None
            else int(self._page_table_by_state.shape[1])
        )
        new_table = torch.full(
            (new_rows, current_cols),
            -1,
            device=device,
            dtype=torch.long,
        )
        if self._page_table_by_state is not None:
            new_table[:current_rows, :current_cols].copy_(self._page_table_by_state.to(device))
        self._page_table_by_state = new_table

    def _ensure_metadata_page_capacity(
        self,
        pages: int,
        *,
        device: torch.device,
    ) -> None:
        if pages <= 0:
            return
        if self._lengths_by_state is None:
            self._ensure_metadata_state_capacity(0, device=device)
        assert self._lengths_by_state is not None
        rows = int(self._lengths_by_state.shape[0])
        current_cols = (
            0
            if self._page_table_by_state is None
            else int(self._page_table_by_state.shape[1])
        )
        if current_cols >= pages:
            return
        if self._metadata_fixed_capacity:
            raise RuntimeError(
                "Fixed HSD history metadata page capacity exceeded: "
                f"required_pages={pages}, max_pages_per_state={current_cols}."
            )
        new_cols = max(pages, max(1, current_cols * 2))
        new_table = torch.full(
            (rows, new_cols),
            -1,
            device=device,
            dtype=torch.long,
        )
        if self._page_table_by_state is not None and current_cols > 0:
            new_table[:, :current_cols].copy_(self._page_table_by_state.to(device))
        self._page_table_by_state = new_table

    def _clear_metadata_entry(self, state_idx: int) -> None:
        if self._lengths_by_state is None or state_idx < 0:
            return
        if state_idx >= int(self._lengths_by_state.shape[0]):
            return
        self._lengths_by_state[state_idx].zero_()
        if self._page_table_by_state is not None:
            self._page_table_by_state[state_idx].fill_(-1)

    def _refresh_metadata_max_anchor_len(self) -> None:
        self._max_anchor_len = max((int(entry.length) for entry in self.entries.values()), default=0)

    def _sync_metadata_entry(self, state_idx: int) -> None:
        entry = self.entries.get(int(state_idx))
        if entry is None:
            self._clear_metadata_entry(int(state_idx))
            self._bump_metadata_version()
            return
        device = self._metadata_device(torch.device("cpu"))
        self._ensure_metadata_state_capacity(int(state_idx), device=device)
        self._ensure_metadata_page_capacity(len(entry.page_ids), device=device)
        assert self._lengths_by_state is not None and self._page_table_by_state is not None
        self._lengths_by_state[int(state_idx)] = int(entry.length)
        self._page_table_by_state[int(state_idx)].fill_(-1)
        if entry.page_ids:
            self._page_table_by_state[
                int(state_idx), : len(entry.page_ids)
            ] = torch.as_tensor(entry.page_ids, device=device, dtype=torch.long)
        self._max_anchor_len = max(self._max_anchor_len, int(entry.length))
        self._bump_metadata_version()

    def _lookup_metadata_lengths(self, state_indices: torch.Tensor) -> torch.Tensor | None:
        if self._lengths_by_state is None:
            return None
        lengths_by_state = self._lengths_by_state.to(state_indices.device)
        capacity = int(lengths_by_state.shape[0])
        in_range = (state_indices >= 0) & (state_indices < capacity)
        safe_indices = torch.where(in_range, state_indices, torch.zeros_like(state_indices))
        lengths = lengths_by_state.index_select(0, safe_indices)
        return torch.where(in_range, lengths, torch.zeros_like(lengths))

    def _check_or_init(self, states: torch.Tensor, keys: torch.Tensor, *, initialize: bool = True) -> None:
        if states.ndim != 5 or int(states.shape[0]) != 1:
            raise ValueError(f"states must have shape [1, M, H, V, K], got {tuple(states.shape)}.")
        if keys.ndim != 4 or int(keys.shape[0]) != 1:
            raise ValueError(f"keys must have shape [1, M, H, K], got {tuple(keys.shape)}.")
        if int(states.shape[1]) != int(keys.shape[1]):
            raise ValueError("states and keys anchor counts must match.")
        heads, value_dim, key_dim = int(states.shape[2]), int(states.shape[3]), int(states.shape[4])
        route_heads, route_dim = int(keys.shape[2]), int(keys.shape[3])
        if route_heads not in {1, heads}:
            raise ValueError(
                "route key heads must be one or match state heads, got "
                f"route_heads={route_heads}, state_heads={heads}."
            )

        if not self.initialized:
            if not initialize:
                return
            self.heads = heads
            self.value_dim = value_dim
            self.key_dim = key_dim
            self.route_heads = route_heads
            self.route_dim = route_dim
            return

        if (
            self.heads != heads
            or self.value_dim != value_dim
            or self.key_dim != key_dim
            or self.route_heads != route_heads
            or self.route_dim != route_dim
            or self.device != states.device
            or self.keys_pages is None
            or self.keys_pages.device != keys.device
        ):
            raise ValueError(
                "Paged HSD anchor cache shape/device changed; clear the cache before reusing it."
            )

    def _grow_pages(self, min_pages: int, *, device: torch.device, states_dtype: torch.dtype, keys_dtype: torch.dtype) -> None:
        assert self.heads is not None and self.value_dim is not None and self.key_dim is not None
        assert self.route_heads is not None and self.route_dim is not None
        old_pages = self.total_pages
        if old_pages >= min_pages:
            return
        new_pages = max(min_pages, max(1, old_pages * 2))

        new_states = None
        new_states = torch.empty(
            new_pages,
            self.page_m,
            self.heads,
            self.value_dim,
            self.key_dim,
            device=device,
            dtype=states_dtype,
        )
        new_keys = torch.empty(
            new_pages,
            self.page_m,
            self.route_heads,
            self.route_dim,
            device=device,
            dtype=keys_dtype,
        )
        new_boundaries = torch.empty(new_pages, self.page_m, device=device, dtype=torch.long)
        if old_pages:
            assert self.states_pages is not None and new_states is not None
            new_states[:old_pages].copy_(self.states_pages)
            assert self.keys_pages is not None and self.boundary_pages is not None
            new_keys[:old_pages].copy_(self.keys_pages)
            new_boundaries[:old_pages].copy_(self.boundary_pages)
        self.states_pages = new_states
        self.keys_pages = new_keys
        self.boundary_pages = new_boundaries
        self.free_pages.extend(range(old_pages, new_pages))

    def _alloc_page(self, *, device: torch.device, states_dtype: torch.dtype, keys_dtype: torch.dtype) -> int:
        if not self.free_pages:
            if self.fixed_capacity:
                raise RuntimeError("fixed HSD anchor page cache is full")
            old_pages = self.total_pages
            self._grow_pages(old_pages + 1, device=device, states_dtype=states_dtype, keys_dtype=keys_dtype)
        page_id = self.free_pages.pop()
        assert self.boundary_pages is not None
        self.boundary_pages[page_id].fill_(2**62)
        return page_id

    def _pages_needed_for_append(self, state_idx: int, count: int) -> int:
        if count <= 0:
            return 0
        entry = self.entries.get(int(state_idx))
        old_length = 0 if entry is None else int(entry.length)
        old_pages = 0 if entry is None else len(entry.page_ids)
        new_length = old_length + int(count)
        new_pages = (new_length + self.page_m - 1) // self.page_m
        return max(0, new_pages - old_pages)

    def _capacity_error(self, state_idx: int, count: int, *, replace: bool = False) -> str | None:
        """Check all fixed budgets without reserving pages or changing metadata."""
        state_idx, count = int(state_idx), int(count)
        if state_idx < 0 or count < 0:
            raise ValueError("state_idx and anchor count must be non-negative")
        if count == 0:
            return None
        entry = self.entries.get(state_idx)
        old_length = 0 if entry is None or replace else int(entry.length)
        old_pages = 0 if entry is None else len(entry.page_ids)
        required_pages = (old_length + count + self.page_m - 1) // self.page_m
        if self._metadata_fixed_capacity:
            rows = 0 if self._lengths_by_state is None else int(self._lengths_by_state.shape[0])
            cols = 0 if self._page_table_by_state is None else int(self._page_table_by_state.shape[1])
            if state_idx >= rows:
                return f"Fixed HSD history metadata state capacity exceeded: state_idx={state_idx}, state_rows={rows}."
            if required_pages > cols:
                return ("Fixed HSD history metadata page capacity exceeded: "
                        f"required_pages={required_pages}, max_pages_per_state={cols}.")
        if self.fixed_capacity:
            available = len(self.free_pages) + (old_pages if replace else 0)
            needed = required_pages if replace else max(0, required_pages - old_pages)
            if needed > available:
                return "fixed HSD anchor page cache is full"
        return None

    def can_append(self, state_idx: int, count: int) -> bool:
        return self._capacity_error(state_idx, count) is None

    def can_replace(self, state_idx: int, count: int) -> bool:
        """Check replacement against page-pool and graph-visible metadata budgets."""
        return self._capacity_error(state_idx, count, replace=True) is None

    @staticmethod
    def _boundary_zero_positions(boundaries: torch.Tensor) -> list[int]:
        """Synchronize only at anchor insertion, never in the decode path."""
        return [
            int(value)
            for value in torch.nonzero(
                boundaries.detach().eq(0), as_tuple=False
            ).flatten().cpu().tolist()
        ]

    def _track_memory_append(
        self,
        state_idx: int,
        boundaries: torch.Tensor,
        *,
        old_length: int,
    ) -> bool:
        zero_positions = self._boundary_zero_positions(boundaries)
        if not zero_positions:
            return False
        state_idx = int(state_idx)
        valid = (
            zero_positions == [0]
            and old_length == 0
            and state_idx not in self._persistent_memory_states
            and state_idx not in self._invalid_persistent_memory_states
        )
        if not valid:
            self._persistent_memory_states.discard(state_idx)
            self._invalid_persistent_memory_states.add(state_idx)
        return valid

    def append(
        self,
        state_idx: int,
        states: torch.Tensor,
        keys: torch.Tensor | None,
        boundaries: torch.Tensor,
    ) -> None:
        if keys is None:
            raise ValueError("Paged HSD anchor cache requires anchor route keys.")
        count = int(states.shape[1])
        if count == 0:
            return
        if boundaries.ndim != 1 or int(boundaries.shape[0]) != count:
            raise ValueError(f"boundaries must have shape [{count}], got {tuple(boundaries.shape)}.")
        self._check_or_init(states, keys, initialize=False)
        capacity_error = self._capacity_error(state_idx, count)
        if capacity_error is not None:
            raise RuntimeError(capacity_error)

        target_device = self.device if self.device is not None else states.device
        target_states_dtype = self.dtype if self.dtype is not None else states.dtype
        target_keys_dtype = self.keys_pages.dtype if self.keys_pages is not None else keys.dtype
        states = states.detach().to(device=target_device, dtype=target_states_dtype)
        keys = keys.detach().to(device=target_device, dtype=target_keys_dtype)
        boundaries = boundaries.detach().to(device=target_device, dtype=torch.long)
        # Validation, fixed-capacity checks and all input conversions finish
        # before ownership, payloads or persistent-memory markers are changed.
        self._check_or_init(states, keys)
        old_entry = self.entries.get(int(state_idx))
        old_length = 0 if old_entry is None else int(old_entry.length)
        total_pages = (old_length + count + self.page_m - 1) // self.page_m
        self._ensure_metadata_state_capacity(int(state_idx), device=target_device)
        self._ensure_metadata_page_capacity(total_pages, device=target_device)
        adds_memory = self._track_memory_append(int(state_idx), boundaries, old_length=old_length)
        entry = self.entries.setdefault(int(state_idx), _AnchorEntry(page_ids=[], length=0))
        state_rows = states[0]
        for anchor_idx in range(count):
            page_offset = entry.length // self.page_m
            slot = entry.length % self.page_m
            if page_offset == len(entry.page_ids):
                entry.page_ids.append(
                    self._alloc_page(device=states.device, states_dtype=states.dtype, keys_dtype=keys.dtype)
                )
                assert self._page_table_by_state is not None
                self._page_table_by_state[int(state_idx), page_offset] = entry.page_ids[page_offset]
            page_id = entry.page_ids[page_offset]
            assert self.keys_pages is not None and self.boundary_pages is not None
            assert self.states_pages is not None
            self.states_pages[page_id, slot].copy_(state_rows[anchor_idx])
            self.keys_pages[page_id, slot].copy_(keys[0, anchor_idx])
            self.boundary_pages[page_id, slot].copy_(boundaries[anchor_idx])
            entry.length += 1
        self._sync_metadata_entry(int(state_idx))
        if adds_memory:
            self._persistent_memory_states.add(int(state_idx))

    def try_append(
        self,
        state_idx: int,
        states: torch.Tensor,
        keys: torch.Tensor | None,
        boundaries: torch.Tensor,
    ) -> bool:
        if keys is None:
            raise ValueError("Paged HSD anchor cache requires anchor route keys.")
        count = int(states.shape[1])
        if count == 0:
            return True
        if boundaries.ndim != 1 or int(boundaries.shape[0]) != count:
            raise ValueError(f"boundaries must have shape [{count}], got {tuple(boundaries.shape)}.")
        self._check_or_init(states, keys, initialize=False)
        if not self.can_append(state_idx, count):
            return False
        self.append(state_idx, states, keys, boundaries)
        return True

    def try_append_batch_one(
        self,
        state_indices: list[int] | tuple[int, ...],
        states: torch.Tensor,
        keys: torch.Tensor | None,
        boundaries: torch.Tensor,
    ) -> bool:
        """Append one anchor per state with batched device writes.

        This is the high-throughput decode-boundary form of :meth:`try_append`.
        CPU ownership metadata is still updated per request, while anchor
        state/key/boundary tensors and fixed page-table metadata are written in
        a constant number of device launches.
        """
        state_indices = tuple(int(state_idx) for state_idx in state_indices)
        batch_size = len(state_indices)
        if batch_size == 0:
            return True
        if keys is None:
            raise ValueError("Paged HSD anchor cache requires anchor route keys.")
        if len(set(state_indices)) != batch_size or min(state_indices) < 0:
            return False
        if states.ndim != 4 or int(states.shape[0]) != batch_size:
            return False
        if keys.ndim != 3 or int(keys.shape[0]) != batch_size:
            return False
        if boundaries.ndim != 1 or int(boundaries.shape[0]) != batch_size:
            return False
        if (
            not self.initialized
            or self.keys_pages is None
            or self.boundary_pages is None
            or self._lengths_by_state is None
            or self._page_table_by_state is None
            or self.heads is None
            or self.value_dim is None
            or self.key_dim is None
            or self.route_heads is None
            or self.route_dim is None
        ):
            return False
        if tuple(states.shape[1:]) != (
            self.heads,
            self.value_dim,
            self.key_dim,
        ):
            return False
        if tuple(keys.shape[1:]) != (self.route_heads, self.route_dim):
            return False
        if max(state_indices) >= int(self._lengths_by_state.shape[0]):
            return False

        memory_flags = [bool(value) for value in boundaries.detach().eq(0).cpu().tolist()]
        valid_memory_states: list[int] = []
        invalid_memory_states: list[int] = []
        for state_idx, adds_memory in zip(
            state_indices, memory_flags, strict=True
        ):
            if not adds_memory:
                continue
            entry = self.entries.get(state_idx)
            old_length = 0 if entry is None else int(entry.length)
            if (
                old_length == 0
                and state_idx not in self._persistent_memory_states
                and state_idx not in self._invalid_persistent_memory_states
            ):
                valid_memory_states.append(state_idx)
            else:
                invalid_memory_states.append(state_idx)

        required_new_pages = sum(
            self._pages_needed_for_append(state_idx, 1)
            for state_idx in state_indices
        )
        if self.fixed_capacity and required_new_pages > len(self.free_pages):
            return False
        if not self.fixed_capacity and required_new_pages:
            return False

        assert self.device is not None and self.dtype is not None
        target_device = self.device
        page_ids: list[int] = []
        slots: list[int] = []
        page_offsets: list[int] = []
        new_page_flags: list[bool] = []
        new_lengths_list: list[int] = []
        planned_pages: dict[int, list[int]] = {}
        free_page_cursor = 0
        for state_idx in state_indices:
            entry = self.entries.get(state_idx)
            old_length = 0 if entry is None else int(entry.length)
            entry_pages = [] if entry is None else list(entry.page_ids)
            page_offset = old_length // self.page_m
            slot = old_length % self.page_m
            needs_new_page = page_offset == len(entry_pages)
            if needs_new_page:
                page_id = self.free_pages[-1 - free_page_cursor]
                free_page_cursor += 1
                entry_pages.append(page_id)
            else:
                page_id = entry_pages[page_offset]
            planned_pages[state_idx] = entry_pages
            page_ids.append(page_id)
            slots.append(slot)
            page_offsets.append(page_offset)
            new_page_flags.append(needs_new_page)
            new_lengths_list.append(old_length + 1)

        if max(page_offsets) >= int(self._page_table_by_state.shape[1]):
            return False

        page_ids_tensor = torch.tensor(
            page_ids, device=target_device, dtype=torch.long
        )
        slots_tensor = torch.tensor(
            slots, device=target_device, dtype=torch.long
        )
        state_indices_tensor = torch.tensor(
            state_indices, device=target_device, dtype=torch.long
        )
        page_offsets_tensor = torch.tensor(
            page_offsets, device=target_device, dtype=torch.long
        )
        states = states.detach().to(
            device=target_device, dtype=self.dtype
        )
        keys = keys.detach().to(
            device=target_device, dtype=self.keys_pages.dtype
        )
        boundaries = boundaries.detach().to(
            device=target_device, dtype=torch.long
        )

        if required_new_pages:
            new_page_ids = page_ids_tensor[
                torch.tensor(
                    new_page_flags,
                    device=target_device,
                    dtype=torch.bool,
                )
            ]
            self.boundary_pages.index_fill_(0, new_page_ids, 2**62)
        state_rows = states
        assert self.states_pages is not None
        self.states_pages.index_put_(
            (page_ids_tensor, slots_tensor), state_rows, accumulate=False
        )
        self.keys_pages.index_put_(
            (page_ids_tensor, slots_tensor), keys, accumulate=False
        )
        self.boundary_pages.index_put_(
            (page_ids_tensor, slots_tensor), boundaries, accumulate=False
        )
        self._page_table_by_state.index_put_(
            (state_indices_tensor, page_offsets_tensor),
            page_ids_tensor,
            accumulate=False,
        )
        new_lengths = torch.tensor(
            new_lengths_list,
            device=target_device,
            dtype=torch.long,
        )
        self._lengths_by_state.index_copy_(
            0, state_indices_tensor, new_lengths
        )

        if required_new_pages:
            del self.free_pages[-required_new_pages:]
        for state_idx, new_length in zip(
            state_indices, new_lengths_list, strict=True
        ):
            self.entries[state_idx] = _AnchorEntry(
                page_ids=planned_pages[state_idx],
                length=int(new_length),
            )
        self._persistent_memory_states.update(valid_memory_states)
        self._persistent_memory_states.difference_update(invalid_memory_states)
        self._invalid_persistent_memory_states.update(invalid_memory_states)
        self._max_anchor_len = max(
            self._max_anchor_len,
            max(new_lengths_list),
        )
        self._bump_metadata_version()
        return True

    def try_append_batch(
        self,
        state_indices: list[int] | tuple[int, ...],
        states: torch.Tensor,
        keys: torch.Tensor | None,
        boundaries: torch.Tensor,
        anchor_offsets: list[int] | tuple[int, ...],
    ) -> bool:
        """Append packed, variable-count anchors with batched device writes.

        ``anchor_offsets`` partitions the packed anchor tensors by request.
        The fixed-capacity scheduler cache lets us plan page ownership on the
        CPU, then update all page payloads and graph-visible metadata with a
        constant number of device launches.
        """
        state_indices = tuple(int(state_idx) for state_idx in state_indices)
        anchor_offsets = tuple(int(offset) for offset in anchor_offsets)
        batch_size = len(state_indices)
        if batch_size == 0:
            return int(states.shape[0]) == 0 and anchor_offsets in {(), (0,)}
        if keys is None:
            raise ValueError("Paged HSD anchor cache requires anchor route keys.")
        if len(set(state_indices)) != batch_size or min(state_indices) < 0:
            return False
        if len(anchor_offsets) != batch_size + 1:
            return False
        if anchor_offsets[0] != 0 or any(
            end < begin
            for begin, end in zip(
                anchor_offsets[:-1], anchor_offsets[1:], strict=True
            )
        ):
            return False
        total_anchors = anchor_offsets[-1]
        if states.ndim != 4 or int(states.shape[0]) != total_anchors:
            return False
        if keys.ndim != 3 or int(keys.shape[0]) != total_anchors:
            return False
        if boundaries.ndim != 1 or int(boundaries.shape[0]) != total_anchors:
            return False
        if total_anchors == 0:
            return True
        if (
            not self.initialized
            or self.keys_pages is None
            or self.boundary_pages is None
            or self._lengths_by_state is None
            or self._page_table_by_state is None
            or self.heads is None
            or self.value_dim is None
            or self.key_dim is None
            or self.route_heads is None
            or self.route_dim is None
        ):
            return False
        if tuple(states.shape[1:]) != (
            self.heads,
            self.value_dim,
            self.key_dim,
        ):
            return False
        if tuple(keys.shape[1:]) != (self.route_heads, self.route_dim):
            return False
        if max(state_indices) >= int(self._lengths_by_state.shape[0]):
            return False

        zero_positions = set(self._boundary_zero_positions(boundaries))
        memory_states: list[int] = []
        invalid_memory_states: list[int] = []
        for state_idx, begin, end in zip(
            state_indices,
            anchor_offsets[:-1],
            anchor_offsets[1:],
            strict=True,
        ):
            local_zero_positions = sorted(
                position - begin
                for position in zero_positions
                if begin <= position < end
            )
            if not local_zero_positions:
                continue
            entry = self.entries.get(state_idx)
            old_length = 0 if entry is None else int(entry.length)
            valid_memory = (
                local_zero_positions == [0]
                and old_length == 0
                and state_idx not in self._persistent_memory_states
                and state_idx not in self._invalid_persistent_memory_states
            )
            if valid_memory:
                memory_states.append(state_idx)
            else:
                invalid_memory_states.append(state_idx)

        anchor_counts = [
            anchor_offsets[idx + 1] - anchor_offsets[idx]
            for idx in range(batch_size)
        ]
        required_new_pages = sum(
            self._pages_needed_for_append(state_idx, count)
            for state_idx, count in zip(
                state_indices, anchor_counts, strict=True
            )
        )
        if self.fixed_capacity and required_new_pages > len(self.free_pages):
            return False
        if not self.fixed_capacity and required_new_pages:
            return False

        page_ids: list[int] = []
        slots: list[int] = []
        new_page_state_indices: list[int] = []
        new_page_offsets: list[int] = []
        new_page_ids: list[int] = []
        new_lengths: list[int] = []
        planned_pages: dict[int, list[int]] = {}
        free_page_cursor = 0
        max_page_offset = -1
        for state_idx, count in zip(
            state_indices, anchor_counts, strict=True
        ):
            entry = self.entries.get(state_idx)
            old_length = 0 if entry is None else int(entry.length)
            entry_pages = [] if entry is None else list(entry.page_ids)
            for anchor_offset in range(count):
                position = old_length + anchor_offset
                page_offset = position // self.page_m
                slot = position % self.page_m
                if page_offset == len(entry_pages):
                    page_id = self.free_pages[-1 - free_page_cursor]
                    free_page_cursor += 1
                    entry_pages.append(page_id)
                    new_page_state_indices.append(state_idx)
                    new_page_offsets.append(page_offset)
                    new_page_ids.append(page_id)
                else:
                    page_id = entry_pages[page_offset]
                page_ids.append(page_id)
                slots.append(slot)
                max_page_offset = max(max_page_offset, page_offset)
            planned_pages[state_idx] = entry_pages
            new_lengths.append(old_length + count)

        if max_page_offset >= int(self._page_table_by_state.shape[1]):
            return False

        assert self.device is not None and self.dtype is not None
        target_device = self.device
        page_ids_tensor = torch.tensor(
            page_ids, device=target_device, dtype=torch.long
        )
        slots_tensor = torch.tensor(
            slots, device=target_device, dtype=torch.long
        )
        state_indices_tensor = torch.tensor(
            state_indices, device=target_device, dtype=torch.long
        )
        states = states.detach().to(
            device=target_device, dtype=self.dtype
        )
        keys = keys.detach().to(
            device=target_device, dtype=self.keys_pages.dtype
        )
        boundaries = boundaries.detach().to(
            device=target_device, dtype=torch.long
        )

        if new_page_ids:
            new_page_ids_tensor = torch.tensor(
                new_page_ids, device=target_device, dtype=torch.long
            )
            self.boundary_pages.index_fill_(
                0, new_page_ids_tensor, 2**62
            )
            self._page_table_by_state.index_put_(
                (
                    torch.tensor(
                        new_page_state_indices,
                        device=target_device,
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        new_page_offsets,
                        device=target_device,
                        dtype=torch.long,
                    ),
                ),
                new_page_ids_tensor,
                accumulate=False,
            )
        state_rows = states
        assert self.states_pages is not None
        self.states_pages.index_put_(
            (page_ids_tensor, slots_tensor), state_rows, accumulate=False
        )
        self.keys_pages.index_put_(
            (page_ids_tensor, slots_tensor), keys, accumulate=False
        )
        self.boundary_pages.index_put_(
            (page_ids_tensor, slots_tensor), boundaries, accumulate=False
        )
        new_lengths_tensor = torch.tensor(
            new_lengths, device=target_device, dtype=torch.long
        )
        self._lengths_by_state.index_copy_(
            0, state_indices_tensor, new_lengths_tensor
        )

        if required_new_pages:
            del self.free_pages[-required_new_pages:]
        for state_idx, new_length in zip(
            state_indices, new_lengths, strict=True
        ):
            self.entries[state_idx] = _AnchorEntry(
                page_ids=planned_pages[state_idx],
                length=int(new_length),
            )
        self._persistent_memory_states.update(memory_states)
        self._persistent_memory_states.difference_update(invalid_memory_states)
        self._invalid_persistent_memory_states.update(invalid_memory_states)
        self._max_anchor_len = max(
            self._max_anchor_len,
            max(new_lengths),
        )
        self._bump_metadata_version()
        return True

    def migrate_from_legacy(self, state_idx: int, cached: dict[str, torch.Tensor] | None) -> bool:
        if cached is None or self.has_state(state_idx):
            return False
        states = cached.get("states")
        keys = cached.get("keys")
        boundaries = cached.get("boundaries")
        if not (torch.is_tensor(states) and torch.is_tensor(keys) and torch.is_tensor(boundaries)):
            return False
        if int(states.shape[1]) == 0:
            return False
        if not self.try_append(state_idx, states, keys, boundaries):
            return False
        return True

    def materialize(
        self,
        state_idx: int,
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        entry = self.entries.get(int(state_idx))
        if entry is None or entry.length == 0:
            return None
        assert (
            self.initialized
            and self.keys_pages is not None
            and self.boundary_pages is not None
            and self.heads is not None
            and self.value_dim is not None
            and self.key_dim is not None
            and self.route_heads is not None
            and self.route_dim is not None
        )
        assert self.device is not None and self.dtype is not None
        target_device = self.device if device is None else device
        states = torch.empty(
            1,
            entry.length,
            self.heads,
            self.value_dim,
            self.key_dim,
            device=target_device,
            dtype=self.dtype,
        )
        keys = torch.empty(
            1,
            entry.length,
            self.route_heads,
            self.route_dim,
            device=target_device,
            dtype=self.keys_pages.dtype,
        )
        boundaries = torch.empty(entry.length, device=target_device, dtype=torch.long)
        copied = 0
        for page_id in entry.page_ids:
            take = min(self.page_m, entry.length - copied)
            assert self.states_pages is not None
            states[0, copied : copied + take].copy_(
                self.states_pages[page_id, :take].to(target_device)
            )
            keys[0, copied : copied + take].copy_(self.keys_pages[page_id, :take].to(target_device))
            boundaries[copied : copied + take].copy_(self.boundary_pages[page_id, :take].to(target_device))
            copied += take
            if copied >= entry.length:
                break
        return states, keys, boundaries

    def export_page_table(
        self,
        state_indices: torch.Tensor,
        *,
        device: torch.device,
        state_indices_key: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self._lengths_by_state is not None and self._page_table_by_state is not None:
            state_indices = state_indices.to(device=device, dtype=torch.long)
            cache_key: tuple[object, ...] | None = None
            if state_indices_key is not None and self._export_cache_enabled():
                cache_key = (
                    int(self._metadata_version),
                    str(device),
                    tuple(int(v) for v in state_indices_key),
                )
                cached = self._export_page_table_cache.get(cache_key)
                if cached is not None:
                    self._export_page_table_cache.move_to_end(cache_key)
                    if _export_cache_events_enabled():
                        _record_paged_history_event(
                            "paged_history_export_cache_hit",
                            batch=int(state_indices.shape[0]),
                        )
                    return cached
            lengths_by_state = self._lengths_by_state.to(device)
            table_by_state = self._page_table_by_state.to(device)
            capacity = int(lengths_by_state.shape[0])
            in_range = (state_indices >= 0) & (state_indices < capacity)
            safe_indices = torch.where(in_range, state_indices, torch.zeros_like(state_indices))
            lengths = lengths_by_state.index_select(0, safe_indices)
            lengths = torch.where(in_range, lengths, torch.zeros_like(lengths))
            table = table_by_state.index_select(0, safe_indices)
            if int(table.shape[1]) > 0:
                table = torch.where(
                    in_range.view(-1, 1),
                    table,
                    torch.full_like(table, -1),
                )
            result = (table, lengths, int(self._max_anchor_len))
            if cache_key is not None:
                self._export_page_table_cache[cache_key] = result
                limit = self._export_cache_limit()
                while len(self._export_page_table_cache) > limit:
                    self._export_page_table_cache.popitem(last=False)
                if _export_cache_events_enabled():
                    _record_paged_history_event(
                        "paged_history_export_cache_miss",
                        batch=int(state_indices.shape[0]),
                    )
            return result

        indices = [int(v) for v in state_indices.detach().cpu().tolist()]
        lengths = [self.length(state_idx) for state_idx in indices]
        max_pages = 0
        for state_idx in indices:
            entry = self.entries.get(state_idx)
            if entry is not None:
                max_pages = max(max_pages, len(entry.page_ids))
        table = torch.full((len(indices), max_pages), -1, device=device, dtype=torch.long)
        for row, state_idx in enumerate(indices):
            entry = self.entries.get(state_idx)
            if entry is None:
                continue
            if entry.page_ids:
                table[row, : len(entry.page_ids)] = torch.tensor(entry.page_ids, device=device, dtype=torch.long)
        return table, torch.tensor(lengths, device=device, dtype=torch.long), max(lengths, default=0)
