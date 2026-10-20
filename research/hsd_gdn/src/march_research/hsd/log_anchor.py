from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class LogAnchorBlock:
    """One chunk-level weak-Fenwick block.

    ``start`` and ``end`` are inclusive, zero-based global chunk indices.
    ``anchor_index`` addresses the ephemeral state/key/boundary bank supplied by
    the caller.  A merge retains the right block's anchor index; this module
    never combines the values stored in that bank.
    """

    start: int
    end: int
    count: int
    level: int
    anchor_index: int

    def __post_init__(self) -> None:
        for name in ("start", "end", "count", "level", "anchor_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        if self.start < 0:
            raise ValueError("start must be non-negative.")
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start.")
        if self.level < 0:
            raise ValueError("level must be non-negative.")
        if self.anchor_index < 0:
            raise ValueError("anchor_index must be non-negative.")
        expected_count = self.end - self.start + 1
        if self.count != expected_count:
            raise ValueError(
                f"count must equal end - start + 1 ({expected_count}), got {self.count}."
            )
        if self.count != 1 << self.level:
            raise ValueError(
                f"count must be 2**level ({1 << self.level}), got {self.count}."
            )


@dataclass(frozen=True)
class LogAnchorFoldResult:
    """Result of folding new chunk-boundary events into an active cache.

    The count matrix has shape ``[tokens, bank_size]``.  Zero marks an inactive
    bank entry; a positive value is the represented chunk count used both as an
    active mask and as the input to ``log(count)``.
    """

    active_count_matrix: tuple[tuple[int, ...], ...]
    active_blocks_per_token: tuple[tuple[LogAnchorBlock, ...], ...]
    final_blocks: tuple[LogAnchorBlock, ...]
    bank_size: int
    old_total_chunks: int
    new_event_count: int


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _logical_signature(block: LogAnchorBlock) -> tuple[int, int, int, int]:
    return block.start, block.end, block.count, block.level


def plan_weak_fenwick_blocks(total_chunks: int) -> tuple[LogAnchorBlock, ...]:
    """Plan ``binary(total_chunks - 1) + newest singleton`` from a dense bank.

    This is the weak carry-before-insert layout.  For example, eight chunks are
    represented by ``[0, 3], [4, 5], [6, 6], [7, 7]``.  The dense-bank anchor
    index is the inclusive right endpoint of each block.
    """

    total_chunks = _require_nonnegative_int(total_chunks, "total_chunks")
    if total_chunks == 0:
        return ()

    blocks: list[LogAnchorBlock] = []
    prefix_chunks = total_chunks - 1
    cursor = 0
    for level in range(prefix_chunks.bit_length() - 1, -1, -1):
        count = 1 << level
        if not prefix_chunks & count:
            continue
        end = cursor + count - 1
        blocks.append(
            LogAnchorBlock(
                start=cursor,
                end=end,
                count=count,
                level=level,
                anchor_index=end,
            )
        )
        cursor = end + 1

    newest = total_chunks - 1
    blocks.append(
        LogAnchorBlock(
            start=newest,
            end=newest,
            count=1,
            level=0,
            anchor_index=newest,
        )
    )
    return tuple(blocks)


def validate_weak_fenwick_blocks(blocks: Sequence[LogAnchorBlock]) -> int:
    """Validate a weak layout and return its represented total chunk count.

    Anchor indices may refer either to a dense bank or to a compact/ephemeral
    bank.  Therefore validation compares logical intervals but deliberately
    does not require ``anchor_index == end``.
    """

    blocks = tuple(blocks)
    if not blocks:
        return 0
    if any(not isinstance(block, LogAnchorBlock) for block in blocks):
        raise TypeError("blocks must contain LogAnchorBlock instances.")
    total_chunks = blocks[-1].end + 1
    expected = plan_weak_fenwick_blocks(total_chunks)
    if tuple(map(_logical_signature, blocks)) != tuple(map(_logical_signature, expected)):
        raise ValueError("blocks are not the canonical weak carry-before-insert layout.")
    anchor_indices = tuple(block.anchor_index for block in blocks)
    if len(set(anchor_indices)) != len(anchor_indices):
        raise ValueError("active blocks must use distinct anchor indices.")
    return total_chunks


def _merge_blocks_keep_right(left: LogAnchorBlock, right: LogAnchorBlock) -> LogAnchorBlock:
    if left.end + 1 != right.start:
        raise ValueError("only contiguous blocks can be carried.")
    if left.level != right.level or left.count != right.count:
        raise ValueError("only equal-level blocks can be carried.")
    return LogAnchorBlock(
        start=left.start,
        end=right.end,
        count=left.count + right.count,
        level=left.level + 1,
        anchor_index=right.anchor_index,
    )


def weak_carry_before_insert(
    blocks: Sequence[LogAnchorBlock],
    *,
    singleton_anchor_index: int | None = None,
) -> tuple[LogAnchorBlock, ...]:
    """Carry the existing weak cache, then insert one new singleton event.

    Carries only update interval metadata and retain the newer/right anchor.
    ``singleton_anchor_index`` can point at a newly appended entry in an
    ephemeral bank.  When omitted, it defaults to the new event's dense index.
    """

    carried = list(blocks)
    total_chunks = validate_weak_fenwick_blocks(carried)
    if singleton_anchor_index is None:
        singleton_anchor_index = total_chunks
    singleton_anchor_index = _require_nonnegative_int(singleton_anchor_index, "singleton_anchor_index")
    if singleton_anchor_index in {block.anchor_index for block in carried}:
        raise ValueError("the singleton anchor index must not alias an active old anchor.")

    # The weak layout intentionally leaves the newest singleton uncarried.  On
    # the next event, normalize the old suffix first, then append the new one.
    while len(carried) >= 2 and carried[-2].level == carried[-1].level:
        right = carried.pop()
        left = carried.pop()
        carried.append(_merge_blocks_keep_right(left, right))

    carried.append(
        LogAnchorBlock(
            start=total_chunks,
            end=total_chunks,
            count=1,
            level=0,
            anchor_index=singleton_anchor_index,
        )
    )
    validate_weak_fenwick_blocks(carried)
    return tuple(carried)


def plan_recent_blocks(total_chunks: int, budget: int | None = None) -> tuple[LogAnchorBlock, ...]:
    """Plan a same-capacity recent-only control from a dense bank.

    With no explicit budget, the number of retained singleton snapshots equals
    the weak-Fenwick anchor count at the same ``total_chunks``.
    """

    total_chunks = _require_nonnegative_int(total_chunks, "total_chunks")
    if budget is None:
        budget = len(plan_weak_fenwick_blocks(total_chunks))
    budget = _require_nonnegative_int(budget, "budget")
    first = max(0, total_chunks - budget)
    return tuple(
        LogAnchorBlock(
            start=index,
            end=index,
            count=1,
            level=0,
            anchor_index=index,
        )
        for index in range(first, total_chunks)
    )


def validate_recent_blocks(blocks: Sequence[LogAnchorBlock], total_chunks: int) -> None:
    """Validate the default same-budget recent-only layout.

    As with weak blocks, anchor indices may address a dense or compact bank and
    are therefore validated independently from the logical chunk indices.
    """

    blocks = tuple(blocks)
    total_chunks = _require_nonnegative_int(total_chunks, "total_chunks")
    if any(not isinstance(block, LogAnchorBlock) for block in blocks):
        raise TypeError("blocks must contain LogAnchorBlock instances.")
    expected = plan_recent_blocks(total_chunks)
    if tuple(map(_logical_signature, blocks)) != tuple(map(_logical_signature, expected)):
        raise ValueError("blocks are not the same-budget recent-only layout.")
    anchor_indices = tuple(block.anchor_index for block in blocks)
    if len(set(anchor_indices)) != len(anchor_indices):
        raise ValueError("active blocks must use distinct anchor indices.")


def reindex_compacted_blocks(blocks: Sequence[LogAnchorBlock]) -> tuple[LogAnchorBlock, ...]:
    """Point blocks at their order in a newly compacted bank."""

    return tuple(replace(block, anchor_index=index) for index, block in enumerate(blocks))


def select_right_anchor_values(values: Sequence[T], blocks: Sequence[LogAnchorBlock]) -> tuple[T, ...]:
    """Select right-anchor values by reference, without reducing or adding them."""

    value_count = len(values)
    selected: list[T] = []
    for block in blocks:
        if block.anchor_index >= value_count:
            raise IndexError(
                f"anchor_index {block.anchor_index} is outside a bank of size {value_count}."
            )
        selected.append(values[block.anchor_index])
    return tuple(selected)


def _validate_visible_counts(
    visible_new_event_counts: Sequence[int],
    new_event_count: int,
) -> tuple[int, ...]:
    counts = tuple(
        _require_nonnegative_int(value, f"visible_new_event_counts[{index}]")
        for index, value in enumerate(visible_new_event_counts)
    )
    if any(value > new_event_count for value in counts):
        raise ValueError("visible new-event counts cannot exceed new_event_count.")
    if any(right < left for left, right in zip(counts, counts[1:])):
        raise ValueError("visible new-event counts must be non-decreasing within a sequence.")
    return counts


def _count_row(blocks: Sequence[LogAnchorBlock], bank_size: int) -> tuple[int, ...]:
    row = [0] * bank_size
    for block in blocks:
        if block.anchor_index >= bank_size:
            raise ValueError(
                f"anchor_index {block.anchor_index} is outside the ephemeral bank of size {bank_size}."
            )
        if row[block.anchor_index] != 0:
            raise ValueError("active blocks must not alias the same bank entry.")
        row[block.anchor_index] = block.count
    return tuple(row)


def fold_weak_fenwick_events(
    existing_blocks: Sequence[LogAnchorBlock],
    new_event_count: int,
    visible_new_event_counts: Sequence[int] = (),
    *,
    existing_bank_size: int | None = None,
) -> LogAnchorFoldResult:
    """Fold consecutive new summary events into an existing weak cache.

    The ephemeral bank is laid out as ``[old bank, new singleton events]``.
    Existing block anchor indices address the old portion; event ``i`` is placed
    at ``existing_bank_size + i``.  Per-token visible counts select the exact
    prefix state after that many new events, so prefill never uses future
    anchors.  The final blocks still address the ephemeral bank and can be
    gathered with :func:`select_right_anchor_values`, then reindexed with
    :func:`reindex_compacted_blocks` for persistent cache storage.
    """

    existing_blocks = tuple(existing_blocks)
    old_total_chunks = validate_weak_fenwick_blocks(existing_blocks)
    new_event_count = _require_nonnegative_int(new_event_count, "new_event_count")
    visible_counts = _validate_visible_counts(visible_new_event_counts, new_event_count)

    minimum_old_bank_size = 0
    if existing_blocks:
        minimum_old_bank_size = max(block.anchor_index for block in existing_blocks) + 1
    if existing_bank_size is None:
        existing_bank_size = minimum_old_bank_size
    existing_bank_size = _require_nonnegative_int(existing_bank_size, "existing_bank_size")
    if existing_bank_size < minimum_old_bank_size:
        raise ValueError(
            f"existing_bank_size must be at least {minimum_old_bank_size}, got {existing_bank_size}."
        )

    prefix_blocks: list[tuple[LogAnchorBlock, ...]] = [existing_blocks]
    blocks = existing_blocks
    for event_index in range(new_event_count):
        blocks = weak_carry_before_insert(
            blocks,
            singleton_anchor_index=existing_bank_size + event_index,
        )
        prefix_blocks.append(blocks)

    bank_size = existing_bank_size + new_event_count
    active_blocks_per_token = tuple(prefix_blocks[count] for count in visible_counts)
    active_count_matrix = tuple(_count_row(blocks, bank_size) for blocks in active_blocks_per_token)
    return LogAnchorFoldResult(
        active_count_matrix=active_count_matrix,
        active_blocks_per_token=active_blocks_per_token,
        final_blocks=prefix_blocks[-1],
        bank_size=bank_size,
        old_total_chunks=old_total_chunks,
        new_event_count=new_event_count,
    )


def fold_recent_events(
    existing_blocks: Sequence[LogAnchorBlock],
    old_total_chunks: int,
    new_event_count: int,
    visible_new_event_counts: Sequence[int] = (),
    *,
    existing_bank_size: int | None = None,
) -> LogAnchorFoldResult:
    """Fold new events into a compact same-budget recent-only cache.

    The ephemeral bank is ``[old recent bank, new singleton events]``.  The
    target recent window is replanned for every visible prefix and mapped back
    to that bank.  Its left edge never moves backward as total chunk count
    grows, so every required old entry is guaranteed to be present in a valid
    previous recent cache.
    """

    existing_blocks = tuple(existing_blocks)
    old_total_chunks = _require_nonnegative_int(old_total_chunks, "old_total_chunks")
    validate_recent_blocks(existing_blocks, old_total_chunks)
    new_event_count = _require_nonnegative_int(new_event_count, "new_event_count")
    visible_counts = _validate_visible_counts(visible_new_event_counts, new_event_count)

    minimum_old_bank_size = 0
    if existing_blocks:
        minimum_old_bank_size = max(block.anchor_index for block in existing_blocks) + 1
    if existing_bank_size is None:
        existing_bank_size = minimum_old_bank_size
    existing_bank_size = _require_nonnegative_int(existing_bank_size, "existing_bank_size")
    if existing_bank_size < minimum_old_bank_size:
        raise ValueError(
            f"existing_bank_size must be at least {minimum_old_bank_size}, got {existing_bank_size}."
        )

    old_sources = {block.end: block.anchor_index for block in existing_blocks}
    prefix_blocks: list[tuple[LogAnchorBlock, ...]] = []
    for visible_new_events in range(new_event_count + 1):
        dense_plan = plan_recent_blocks(old_total_chunks + visible_new_events)
        mapped: list[LogAnchorBlock] = []
        for block in dense_plan:
            if block.end < old_total_chunks:
                anchor_index = old_sources.get(block.end)
                if anchor_index is None:
                    raise ValueError(
                        "valid recent-cache update requires an old chunk that is missing from the old compact bank."
                    )
            else:
                anchor_index = existing_bank_size + block.end - old_total_chunks
            mapped.append(replace(block, anchor_index=anchor_index))
        prefix_blocks.append(tuple(mapped))

    bank_size = existing_bank_size + new_event_count
    active_blocks_per_token = tuple(prefix_blocks[count] for count in visible_counts)
    active_count_matrix = tuple(_count_row(blocks, bank_size) for blocks in active_blocks_per_token)
    return LogAnchorFoldResult(
        active_count_matrix=active_count_matrix,
        active_blocks_per_token=active_blocks_per_token,
        final_blocks=prefix_blocks[-1],
        bank_size=bank_size,
        old_total_chunks=old_total_chunks,
        new_event_count=new_event_count,
    )


def plan_active_log_anchors(
    visible_summary_counts: Sequence[int],
    *,
    mode: Literal["fenwick", "recent"] = "fenwick",
) -> LogAnchorFoldResult:
    """Build causal per-token plans against a dense summary bank from scratch."""

    counts = tuple(
        _require_nonnegative_int(value, f"visible_summary_counts[{index}]")
        for index, value in enumerate(visible_summary_counts)
    )
    if any(right < left for left, right in zip(counts, counts[1:])):
        raise ValueError("visible summary counts must be non-decreasing within a sequence.")
    total_chunks = counts[-1] if counts else 0
    if mode == "fenwick":
        return fold_weak_fenwick_events((), total_chunks, counts, existing_bank_size=0)
    if mode != "recent":
        raise ValueError(f"unsupported log-anchor mode: {mode!r}.")
    return fold_recent_events((), 0, total_chunks, counts, existing_bank_size=0)


__all__ = [
    "LogAnchorBlock",
    "LogAnchorFoldResult",
    "fold_recent_events",
    "fold_weak_fenwick_events",
    "plan_active_log_anchors",
    "plan_recent_blocks",
    "plan_weak_fenwick_blocks",
    "reindex_compacted_blocks",
    "select_right_anchor_values",
    "validate_recent_blocks",
    "validate_weak_fenwick_blocks",
    "weak_carry_before_insert",
]
