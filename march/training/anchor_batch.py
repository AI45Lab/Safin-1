from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass
class HSDAnchorBatch:
    """Runtime anchor layout passed through HSD models."""

    anchor_mask: torch.Tensor
    text_mask: torch.Tensor
    position_ids: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.anchor_mask.numel() and self.anchor_mask.any().item())


def _get_config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _get_config_bool(config: Any, name: str, default: bool) -> bool:
    value = _get_config_value(config, name, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _lcm(a: int, b: int) -> int:
    if a <= 0:
        return b
    if b <= 0:
        return a
    return abs(a * b) // math.gcd(a, b)


def _position_ids_2d(position_ids: torch.Tensor | None, input_shape: torch.Size) -> torch.Tensor | None:
    if position_ids is None:
        return None
    if tuple(position_ids.shape) == tuple(input_shape):
        return position_ids
    if position_ids.ndim >= 2 and tuple(position_ids.shape[-2:]) == tuple(input_shape):
        return position_ids.reshape(-1, *input_shape)[0]
    return None


def _packed_text_cu_from_position_ids(position_ids: torch.Tensor | None, input_shape: torch.Size) -> torch.Tensor | None:
    pos_2d = _position_ids_2d(position_ids, input_shape)
    if pos_2d is None:
        return None
    if pos_2d.shape[0] != 1:
        return None
    starts = torch.nonzero(pos_2d[0] == 0, as_tuple=False).flatten()
    if starts.numel() <= 1:
        return None
    if int(starts[0].item()) != 0:
        raise ValueError("Packed HSD position_ids must start the first sequence at position 0.")
    return torch.cat([
        starts.to(dtype=torch.long),
        starts.new_tensor([input_shape[-1]], dtype=torch.long),
    ])


def get_hsd_safe_padding_to(config: Any, padding_to: int | None) -> int | None:
    """Return a text padding multiple that keeps expanded HSD length SP-safe."""
    chunk_size = int(_get_config_value(config, "anchor_chunk_size", 0) or 0)
    anchor_token_num = int(_get_config_value(config, "anchor_token_num", 0) or 0)
    tp_size = int(_get_config_value(config, "tensor_model_parallel_size", 1) or 1)
    sequence_parallel = _get_config_bool(config, "sequence_parallel", False)
    if chunk_size <= 0 or anchor_token_num <= 0 or tp_size <= 1 or not sequence_parallel:
        return padding_to

    chunks_per_tp_period = tp_size // math.gcd(chunk_size + anchor_token_num, tp_size)
    hsd_padding_to = chunk_size * chunks_per_tp_period
    if padding_to is None:
        return hsd_padding_to
    return _lcm(int(padding_to), hsd_padding_to)


def build_anchor_embedding_ids(anchor_mask: torch.Tensor, anchor_token_num: int) -> torch.Tensor:
    """Return per-anchor embedding ids in expanded sequence order."""
    if anchor_token_num <= 0:
        raise ValueError("anchor_token_num must be positive.")
    if anchor_mask.ndim != 2:
        raise ValueError(f"anchor_mask must be [batch, seq], got {tuple(anchor_mask.shape)}")
    per_row = anchor_mask.sum(dim=1)
    if per_row.numel() == 0 or int(per_row.max().item()) == 0:
        return torch.empty(0, dtype=torch.long, device=anchor_mask.device)
    if not torch.equal(per_row, per_row[:1].expand_as(per_row)):
        raise NotImplementedError("HSD V1 requires the same anchor layout for every batch row.")
    ids = torch.arange(int(per_row[0].item()), device=anchor_mask.device, dtype=torch.long)
    return ids.remainder(int(anchor_token_num))


def build_anchor_indices(anchor_mask: torch.Tensor) -> torch.Tensor:
    """Return the preceding text-token index for each anchor position."""
    if anchor_mask.ndim != 2:
        raise ValueError(f"anchor_mask must be [batch, seq], got {tuple(anchor_mask.shape)}")
    anchor_mask = anchor_mask.to(dtype=torch.bool)
    per_row = anchor_mask.sum(dim=1)
    if per_row.numel() == 0 or int(per_row.max().item()) == 0:
        return torch.empty(anchor_mask.shape[0], 0, dtype=torch.long, device=anchor_mask.device)
    if not torch.equal(per_row, per_row[:1].expand_as(per_row)):
        raise NotImplementedError("HSD V1 requires the same anchor layout for every batch row.")

    batch_size, seq_len = anchor_mask.shape
    positions = torch.arange(seq_len, device=anchor_mask.device, dtype=torch.long).expand(batch_size, seq_len)
    text_positions = torch.where(anchor_mask, positions.new_full((), -1), positions)
    anchor_positions = text_positions.cummax(dim=1).values
    anchor_positions = anchor_positions[anchor_mask].view(batch_size, int(per_row[0].item()))
    if bool((anchor_positions < 0).any().item()):
        raise ValueError("anchor tokens must have a preceding text token for chunk_end embeddings.")
    return anchor_positions


def maybe_build_hsd_anchor_batch(
    batch: dict[str, torch.Tensor],
    config: Any,
) -> tuple[dict[str, torch.Tensor], HSDAnchorBatch | None]:
    """Insert anchor placeholder tokens into an already-collated text batch.

    Labels intentionally stay at the original text length. The HSD LM head filters
    anchor hidden states before computing logits/loss so the lengths line up.
    """
    chunk_size = int(_get_config_value(config, "anchor_chunk_size", 0))
    anchor_token_num = int(_get_config_value(config, "anchor_token_num", 0))
    if chunk_size <= 0 or anchor_token_num <= 0:
        return batch, None
    if "input_ids" not in batch:
        raise ValueError("HSD anchor insertion requires batch['input_ids'].")
    if "inputs_embeds" in batch:
        raise NotImplementedError("HSD V1 inserts anchor tokens for input_ids batches only.")

    input_ids = batch["input_ids"]
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")

    if "anchor_mask" in batch:
        anchor_mask = batch["anchor_mask"].to(device=input_ids.device, dtype=torch.bool)
        if anchor_mask.shape != input_ids.shape:
            raise ValueError(
                f"anchor_mask shape {tuple(anchor_mask.shape)} must match input_ids {tuple(input_ids.shape)}"
            )
        text_mask = ~anchor_mask
        labels = batch.get("labels")
        if labels is not None and labels.ndim == 2:
            per_row_text = text_mask.sum(dim=1)
            if not torch.equal(per_row_text, per_row_text[:1].expand_as(per_row_text)):
                raise NotImplementedError("HSD V1 requires the same anchor layout for every batch row.")
            if labels.shape[1] != int(per_row_text[0].item()):
                raise ValueError(
                    f"HSD labels length {labels.shape[1]} must match text token length {int(per_row_text[0].item())}."
                )
        if batch["anchor_mask"] is not anchor_mask:
            batch = dict(batch)
            batch["anchor_mask"] = anchor_mask
        return batch, HSDAnchorBatch(
            anchor_mask=anchor_mask,
            text_mask=text_mask,
            position_ids=batch.get("position_ids"),
        )

    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        if attention_mask.shape == input_ids.shape:
            if not bool(attention_mask.all().item()):
                raise NotImplementedError("HSD V1 does not support padded batches.")
        elif attention_mask.ndim >= 2 and attention_mask.shape[-2:] == (input_ids.shape[1], input_ids.shape[1]):
            # Text-only causal masks are kept as-is. Full-attention blocks run on
            # text-only hidden states, while HSD blocks ignore non-2D masks.
            pass
        else:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} must match input_ids {tuple(input_ids.shape)} "
                f"or be a text-only causal mask ending with {(input_ids.shape[1], input_ids.shape[1])}."
            )

    batch_size, seq_len = input_ids.shape
    position_ids = batch.get("position_ids")
    packed_text_cu = _packed_text_cu_from_position_ids(position_ids, input_ids.shape)
    if packed_text_cu is not None and batch_size != 1:
        raise NotImplementedError("Packed HSD anchor insertion requires flattened batch_size=1 inputs.")

    if packed_text_cu is None:
        doc_ranges = [(0, seq_len)]
        doc_chunk_counts = [seq_len // chunk_size]
    else:
        cu_cpu = packed_text_cu.detach().to(device="cpu", dtype=torch.long)
        doc_ranges = [
            (int(start), int(end))
            for start, end in zip(cu_cpu[:-1].tolist(), cu_cpu[1:].tolist())
        ]
        doc_chunk_counts = [(end - start) // chunk_size for start, end in doc_ranges]
    chunk_count = sum(doc_chunk_counts)
    if chunk_count == 0:
        anchor_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
        text_mask = ~anchor_mask
        batch["anchor_mask"] = anchor_mask
        if packed_text_cu is not None:
            batch = dict(batch)
            batch["anchor_mask"] = anchor_mask
            batch["text_cu_seqlens"] = packed_text_cu.to(device=input_ids.device, dtype=torch.long)
            batch["anchor_cu_seqlens"] = torch.zeros(
                len(doc_ranges) + 1, dtype=torch.long, device=input_ids.device
            )
            batch["cu_seqlens"] = batch["text_cu_seqlens"]
        return batch, HSDAnchorBatch(anchor_mask=anchor_mask, text_mask=text_mask, position_ids=batch.get("position_ids"))

    anchor_counts_by_doc = [count * anchor_token_num for count in doc_chunk_counts]
    new_seq_len = seq_len + chunk_count * anchor_token_num
    max_position_embeddings = _get_config_value(config, "max_position_embeddings", None)
    if max_position_embeddings is not None:
        max_position_embeddings = int(max_position_embeddings)
        if packed_text_cu is None:
            max_expanded_doc_len = new_seq_len
        else:
            max_expanded_doc_len = max(
                (doc_end - doc_start) + doc_anchor_count
                for (doc_start, doc_end), doc_anchor_count in zip(doc_ranges, anchor_counts_by_doc)
            )
        if max_expanded_doc_len > max_position_embeddings:
            raise ValueError(
                f"HSD expanded document length {max_expanded_doc_len} exceeds max_position_embeddings "
                f"{max_position_embeddings}."
            )
    placeholder_id = int(_get_config_value(config, "anchor_placeholder_token_id", 0))

    new_input_ids = torch.empty(batch_size, new_seq_len, dtype=input_ids.dtype, device=input_ids.device)
    anchor_mask = torch.zeros(batch_size, new_seq_len, dtype=torch.bool, device=input_ids.device)
    new_position_ids = None
    if position_ids is not None:
        if position_ids.shape == input_ids.shape:
            new_position_ids = torch.empty(batch_size, new_seq_len, dtype=position_ids.dtype, device=position_ids.device)
        elif position_ids.ndim >= 2 and tuple(position_ids.shape[-2:]) == tuple(input_ids.shape):
            new_position_ids = torch.empty(
                *position_ids.shape[:-1],
                new_seq_len,
                dtype=position_ids.dtype,
                device=position_ids.device,
            )
        else:
            raise ValueError(
                f"position_ids shape {tuple(position_ids.shape)} must match input_ids {tuple(input_ids.shape)} "
                f"or end with {tuple(input_ids.shape)}."
            )

    write = 0
    expanded_cu_values = [0]
    anchor_cu_values = [0]
    for (doc_start, doc_end), doc_chunk_count, doc_anchor_count in zip(
        doc_ranges, doc_chunk_counts, anchor_counts_by_doc
    ):
        read = doc_start
        for _ in range(doc_chunk_count):
            text_slice = slice(write, write + chunk_size)
            src_slice = slice(read, read + chunk_size)
            new_input_ids[:, text_slice] = input_ids[:, src_slice]
            if new_position_ids is not None:
                new_position_ids[..., text_slice] = position_ids[..., src_slice]

            anchor_slice = slice(text_slice.stop, text_slice.stop + anchor_token_num)
            new_input_ids[:, anchor_slice] = placeholder_id
            anchor_mask[:, anchor_slice] = True
            if new_position_ids is not None:
                if _get_config_value(config, "anchor_position_ids_type", "zeros") == "zeros":
                    new_position_ids[..., anchor_slice] = 0
                else:
                    new_position_ids[..., anchor_slice] = position_ids[..., src_slice.stop - 1:src_slice.stop]

            read += chunk_size
            write = anchor_slice.stop

        if read < doc_end:
            tail_slice = slice(write, write + (doc_end - read))
            src_slice = slice(read, doc_end)
            new_input_ids[:, tail_slice] = input_ids[:, src_slice]
            if new_position_ids is not None:
                new_position_ids[..., tail_slice] = position_ids[..., src_slice]
            write = tail_slice.stop
        expanded_cu_values.append(write)
        anchor_cu_values.append(anchor_cu_values[-1] + doc_anchor_count)

    new_batch = dict(batch)
    new_batch["input_ids"] = new_input_ids
    if attention_mask is None or attention_mask.shape == input_ids.shape:
        new_batch["attention_mask"] = torch.ones(batch_size, new_seq_len, dtype=torch.long, device=input_ids.device)
    else:
        new_batch["attention_mask"] = attention_mask
    new_batch["anchor_mask"] = anchor_mask
    if new_position_ids is not None:
        new_batch["position_ids"] = new_position_ids
    if packed_text_cu is not None:
        new_batch["text_cu_seqlens"] = packed_text_cu.to(device=input_ids.device, dtype=torch.long)
        new_batch["anchor_cu_seqlens"] = torch.tensor(
            anchor_cu_values, dtype=torch.long, device=input_ids.device
        )
        new_batch["cu_seqlens"] = torch.tensor(
            expanded_cu_values, dtype=torch.long, device=input_ids.device
        )

    text_mask = ~anchor_mask
    anchor_batch = HSDAnchorBatch(
        anchor_mask=anchor_mask,
        text_mask=text_mask,
        position_ids=new_batch.get("position_ids"),
    )
    return new_batch, anchor_batch


class HSDDataCollator:
    """Wrap an existing ms-swift data collator and insert HSD anchors."""

    def __init__(self, base_collator, config):
        self.base_collator = base_collator
        self.config = config

    def __call__(self, batch, *, padding_to=None):
        padding_to = get_hsd_safe_padding_to(self.config, padding_to)
        collated = self.base_collator(batch, padding_to=padding_to)
        collated, _ = maybe_build_hsd_anchor_batch(collated, self.config)
        return collated


__all__ = [
    "HSDAnchorBatch",
    "HSDDataCollator",
    "build_anchor_indices",
    "build_anchor_embedding_ids",
    "get_hsd_safe_padding_to",
    "maybe_build_hsd_anchor_batch",
]
