"""Shared Safin-1 anchor layout for dense and MoE remote code."""

from __future__ import annotations

from typing import Any

import torch


def _anchor_enabled(config: Any) -> bool:
    return (
        bool(getattr(config, "hist_replace_linear_attn", True))
        and int(getattr(config, "anchor_token_num", 0) or 0) > 0
        and (
            int(getattr(config, "anchor_chunk_size", 0) or 0) > 0
            or bool(getattr(config, "hsd_persistent_memory_enabled", False))
        )
    )


def _persistent_memory_enabled(config) -> bool:
    return bool(getattr(config, "hsd_persistent_memory_enabled", False))






def _build_anchor_layout(
    batch_size: int,
    text_len: int,
    config: Any,
    device: torch.device,
) -> torch.Tensor:
    anchor_mask, _ = _build_anchor_layout_masks(batch_size, text_len, config, device)
    return anchor_mask


def _build_anchor_layout_masks(
    batch_size: int,
    text_len: int,
    config: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_size = int(getattr(config, "anchor_chunk_size", 0) or 0)
    anchor_token_num = int(config.anchor_token_num)
    chunk_count = text_len // chunk_size if chunk_size > 0 else 0
    persistent_count = 1 if _persistent_memory_enabled(config) else 0
    expanded_len = text_len + persistent_count + chunk_count * anchor_token_num
    anchor_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    persistent_memory_mask = torch.zeros_like(anchor_mask)

    write = 0
    if persistent_count:
        anchor_mask[:, 0] = True
        persistent_memory_mask[:, 0] = True
        write = 1
    for _ in range(chunk_count):
        write += chunk_size
        anchor_mask[:, write : write + anchor_token_num] = True
        write += anchor_token_num
    return anchor_mask, persistent_memory_mask


def _is_left_padding_mask(attention_mask: torch.Tensor) -> bool:
    if attention_mask.ndim != 2:
        return False
    mask = attention_mask.to(dtype=torch.bool)
    if mask.shape[1] <= 1:
        return True
    return not bool((mask[:, :-1] & ~mask[:, 1:]).any().item())


def _expand_2d_with_anchor(
    tensor: torch.Tensor,
    anchor_mask: torch.Tensor,
    fill_value: int | float = 0,
    anchor_fill: bool = False,
) -> torch.Tensor:
    batch_size, text_len = tensor.shape
    expanded = torch.full(
        (batch_size, anchor_mask.shape[1]),
        fill_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    text_positions = (~anchor_mask[0]).nonzero(as_tuple=False).flatten().to(tensor.device)
    expanded.index_copy_(1, text_positions, tensor)
    if anchor_fill and bool(anchor_mask.any().item()):
        anchors = _anchor_text_indices(anchor_mask).to(tensor.device)
        expanded[anchor_mask] = tensor.gather(1, anchors).reshape(-1)
    return expanded


def _expand_embeds_with_anchor(inputs_embeds: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
    batch_size, _, hidden_size = inputs_embeds.shape
    expanded = inputs_embeds.new_zeros(batch_size, anchor_mask.shape[1], hidden_size)
    text_positions = (~anchor_mask[0]).nonzero(as_tuple=False).flatten().to(inputs_embeds.device)
    expanded.index_copy_(1, text_positions, inputs_embeds)
    return expanded


def _anchor_text_indices(anchor_mask: torch.Tensor) -> torch.Tensor:
    text_seen = (~anchor_mask).long().cumsum(dim=1).sub(1)
    anchors = text_seen[anchor_mask].view(anchor_mask.shape[0], -1)
    # A persistent-memory anchor intentionally precedes the first text token.
    # For chunk-end embedding/position compatibility, anchor that one virtual
    # token to text position zero.  Dynamic anchors retain their old anchors.
    return anchors.clamp_min(0)


def _expand_position_ids(
    position_ids: torch.Tensor | None,
    text_len: int,
    anchor_mask: torch.Tensor,
    config: Any,
    device: torch.device,
) -> torch.Tensor | None:
    if anchor_mask.shape[1] == text_len and not bool(anchor_mask.any().item()):
        return position_ids

    anchor_position_type = str(getattr(config, "anchor_position_ids_type", "zeros"))
    anchor_fill = anchor_position_type == "chunk_end"
    if anchor_position_type not in {"zeros", "chunk_end"}:
        raise ValueError("anchor_position_ids_type must be either 'zeros' or 'chunk_end'.")

    batch_size = anchor_mask.shape[0]
    if position_ids is None:
        base = torch.arange(text_len, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
        expanded = _expand_2d_with_anchor(base, anchor_mask, fill_value=0, anchor_fill=anchor_fill)
        return expanded.view(1, batch_size, -1).expand(4, batch_size, -1)

    if position_ids.ndim == 2:
        return _expand_2d_with_anchor(position_ids, anchor_mask, fill_value=0, anchor_fill=anchor_fill)
    if position_ids.ndim == 3:
        rows = []
        for row in position_ids:
            rows.append(_expand_2d_with_anchor(row, anchor_mask, fill_value=0, anchor_fill=anchor_fill))
        return torch.stack(rows, dim=0)
    raise ValueError(f"Unsupported position_ids shape for HSD anchor insertion: {tuple(position_ids.shape)}")


def _insert_anchor_inputs_with_layout(
    *,
    config: Any,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None = None,
    force_layout: bool = False,
) -> dict[str, torch.Tensor | None]:
    if input_ids is None and inputs_embeds is None:
        raise ValueError("HSD anchor insertion requires input_ids or inputs_embeds.")
    ref = input_ids if input_ids is not None else inputs_embeds
    batch_size, text_len = ref.shape[:2]
    if attention_mask is None:
        text_attention_mask = torch.ones(batch_size, text_len, dtype=torch.long, device=ref.device)
    elif attention_mask.ndim == 2 and attention_mask.shape[-1] == text_len:
        if not _is_left_padding_mask(attention_mask):
            raise NotImplementedError("Safin-1 HSD supports left-padded 2D attention masks only.")
        text_attention_mask = attention_mask.to(device=ref.device, dtype=torch.long)
    elif attention_mask is not None and attention_mask.ndim == 2:
        raise ValueError(f"attention_mask length {attention_mask.shape[-1]} must match text length {text_len}.")
    else:
        raise NotImplementedError("Safin-1 HSD anchor insertion supports 2D padding masks only.")

    text_valid_source = text_attention_mask.to(dtype=torch.bool)
    has_padding = not bool(text_valid_source.all().item())
    anchor_mask, persistent_memory_mask = _build_anchor_layout_masks(
        batch_size, text_len, config, ref.device
    )
    has_anchor_tokens = bool(anchor_mask.any().item())
    if not has_anchor_tokens and not force_layout:
        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "mm_token_type_ids": mm_token_type_ids,
            "anchor_mask": anchor_mask,
            "hsd_valid_mask": None,
            "text_valid_mask": None,
            "anchor_valid_mask": None,
            "persistent_memory_mask": persistent_memory_mask,
        }

    if has_padding:
        return _insert_left_padded_anchor_inputs(
            config=config,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=text_attention_mask,
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            text_valid_source=text_valid_source,
        )

    if input_ids is not None:
        input_ids = _expand_2d_with_anchor(
            input_ids,
            anchor_mask,
            fill_value=int(getattr(config, "anchor_placeholder_token_id", 0)),
        )
    if inputs_embeds is not None:
        inputs_embeds = _expand_embeds_with_anchor(inputs_embeds, anchor_mask)
    if mm_token_type_ids is not None:
        mm_token_type_ids = _expand_2d_with_anchor(mm_token_type_ids, anchor_mask, fill_value=0)
    position_ids = _expand_position_ids(position_ids, text_len, anchor_mask, config, ref.device)
    attention_mask = torch.ones(batch_size, anchor_mask.shape[1], dtype=torch.long, device=ref.device)
    return {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "mm_token_type_ids": mm_token_type_ids,
        "anchor_mask": anchor_mask,
        "hsd_valid_mask": attention_mask.to(dtype=torch.bool),
        "text_valid_mask": (~anchor_mask).to(dtype=torch.bool),
        "anchor_valid_mask": anchor_mask.to(dtype=torch.bool),
        "persistent_memory_mask": persistent_memory_mask,
    }


def _insert_left_padded_anchor_inputs(
    *,
    config: Any,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
    text_valid_source: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    ref = input_ids if input_ids is not None else inputs_embeds
    batch_size, text_len = ref.shape[:2]
    device = ref.device
    chunk_size = int(getattr(config, "anchor_chunk_size", 0) or 0)
    anchor_token_num = int(config.anchor_token_num)
    if chunk_size > 0:
        anchor_counts = (
            torch.div(text_valid_source.long().sum(dim=1), chunk_size, rounding_mode="floor")
            * anchor_token_num
        )
    else:
        anchor_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    if _persistent_memory_enabled(config):
        anchor_counts = anchor_counts + 1
    max_anchor_count = int(anchor_counts.max().item()) if anchor_counts.numel() else 0
    expanded_len = text_len + max_anchor_count
    placeholder_id = int(getattr(config, "anchor_placeholder_token_id", 0))

    anchor_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    hsd_valid_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    text_valid_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    anchor_valid_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    persistent_memory_mask = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=device)
    expanded_attention = torch.zeros(batch_size, expanded_len, dtype=attention_mask.dtype, device=device)

    expanded_ids = None
    if input_ids is not None:
        expanded_ids = torch.full(
            (batch_size, expanded_len),
            placeholder_id,
            dtype=input_ids.dtype,
            device=device,
        )
    expanded_embeds = None
    if inputs_embeds is not None:
        expanded_embeds = inputs_embeds.new_zeros(batch_size, expanded_len, inputs_embeds.shape[-1])
    expanded_mm_ids = None
    if mm_token_type_ids is not None:
        expanded_mm_ids = torch.zeros(batch_size, expanded_len, dtype=mm_token_type_ids.dtype, device=device)

    expanded_position_ids = None
    if position_ids is None:
        base_position_ids = attention_mask.to(device=device, dtype=torch.long).cumsum(dim=1) - 1
        base_position_ids = base_position_ids.masked_fill(attention_mask.to(device=device, dtype=torch.bool).logical_not(), 0)
        position_source = base_position_ids
    else:
        position_source = position_ids.to(device=device)
    if position_source.ndim == 2:
        expanded_position_ids = torch.zeros(batch_size, expanded_len, dtype=position_source.dtype, device=device)
    elif position_source.ndim == 3:
        expanded_position_ids = torch.zeros(
            position_source.shape[0],
            batch_size,
            expanded_len,
            dtype=position_source.dtype,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported position_ids shape for HSD anchor insertion: {tuple(position_source.shape)}")

    anchor_position_type = str(getattr(config, "anchor_position_ids_type", "zeros"))
    if anchor_position_type not in {"zeros", "chunk_end"}:
        raise ValueError("anchor_position_ids_type must be either 'zeros' or 'chunk_end'.")

    for batch_idx in range(batch_size):
        write = 0
        real_seen = 0
        anchor_seen = 0
        for text_idx in range(text_len):
            if (
                _persistent_memory_enabled(config)
                and anchor_seen == 0
                and bool(text_valid_source[batch_idx, text_idx].item())
            ):
                anchor_mask[batch_idx, write] = True
                anchor_valid_mask[batch_idx, write] = True
                persistent_memory_mask[batch_idx, write] = True
                hsd_valid_mask[batch_idx, write] = True
                expanded_attention[batch_idx, write] = 1
                write += 1
                anchor_seen += 1
            if expanded_ids is not None:
                expanded_ids[batch_idx, write] = input_ids[batch_idx, text_idx]
            if expanded_embeds is not None:
                expanded_embeds[batch_idx, write] = inputs_embeds[batch_idx, text_idx]
            if expanded_mm_ids is not None:
                expanded_mm_ids[batch_idx, write] = mm_token_type_ids[batch_idx, text_idx]
            if expanded_position_ids.ndim == 2:
                expanded_position_ids[batch_idx, write] = position_source[batch_idx, text_idx]
            else:
                expanded_position_ids[:, batch_idx, write] = position_source[:, batch_idx, text_idx]

            if bool(text_valid_source[batch_idx, text_idx].item()):
                text_valid_mask[batch_idx, write] = True
                hsd_valid_mask[batch_idx, write] = True
                expanded_attention[batch_idx, write] = 1
                real_seen += 1
            write += 1

            if (
                chunk_size > 0
                and real_seen > 0
                and real_seen % chunk_size == 0
                and bool(text_valid_source[batch_idx, text_idx].item())
            ):
                for _ in range(anchor_token_num):
                    anchor_mask[batch_idx, write] = True
                    anchor_valid_mask[batch_idx, write] = True
                    hsd_valid_mask[batch_idx, write] = True
                    expanded_attention[batch_idx, write] = 1
                    if expanded_position_ids.ndim == 2:
                        if anchor_position_type == "chunk_end":
                            expanded_position_ids[batch_idx, write] = position_source[batch_idx, text_idx]
                    else:
                        if anchor_position_type == "chunk_end":
                            expanded_position_ids[:, batch_idx, write] = position_source[:, batch_idx, text_idx]
                    write += 1
                    anchor_seen += 1

        while anchor_seen < max_anchor_count:
            anchor_mask[batch_idx, write] = True
            write += 1
            anchor_seen += 1
        if write != expanded_len:
            raise RuntimeError(f"internal HSD left-padding layout error: wrote {write} slots, expected {expanded_len}.")

    return {
        "input_ids": expanded_ids,
        "inputs_embeds": expanded_embeds,
        "attention_mask": expanded_attention,
        "position_ids": expanded_position_ids,
        "mm_token_type_ids": expanded_mm_ids,
        "anchor_mask": anchor_mask,
        "hsd_valid_mask": hsd_valid_mask,
        "text_valid_mask": text_valid_mask,
        "anchor_valid_mask": anchor_valid_mask,
        "persistent_memory_mask": persistent_memory_mask,
    }


def _insert_anchor_inputs(
    *,
    config: Any,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None = None,
    force_layout: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    layout = _insert_anchor_inputs_with_layout(
        config=config,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        mm_token_type_ids=mm_token_type_ids,
        force_layout=force_layout,
    )
    return (
        layout["input_ids"],
        layout["inputs_embeds"],
        layout["attention_mask"],
        layout["position_ids"],
        layout["mm_token_type_ids"],
        layout["anchor_mask"],
    )


def _filter_anchor_hidden(hidden_states: torch.Tensor, anchor_mask: torch.Tensor | None) -> torch.Tensor:
    if anchor_mask is None or not bool(anchor_mask.any().item()):
        return hidden_states
    anchor_mask = anchor_mask.to(device=hidden_states.device, dtype=torch.bool)
    text_mask = ~anchor_mask
    if not torch.equal(text_mask.sum(dim=1), text_mask.sum(dim=1)[:1].expand(text_mask.shape[0])):
        raise NotImplementedError("Safin-1 HSD V1 requires the same anchor layout for every batch row.")
    return hidden_states[text_mask].view(hidden_states.shape[0], int(text_mask[0].sum().item()), hidden_states.shape[-1])


def _anchor_has_tokens(anchor_mask: torch.Tensor | None) -> bool:
    return anchor_mask is not None and bool(anchor_mask.any().item())


def _text_positions_from_anchor_mask(anchor_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    anchor_mask = anchor_mask.to(device=device, dtype=torch.bool)
    if not torch.equal(anchor_mask, anchor_mask[:1].expand_as(anchor_mask)):
        raise NotImplementedError("Safin-1 HSD V1 requires the same anchor layout for every batch row.")
    return (~anchor_mask[0]).nonzero(as_tuple=False).flatten()


def _filter_batch_sequence_to_text(tensor: torch.Tensor | None, anchor_mask: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or not _anchor_has_tokens(anchor_mask):
        return tensor
    if tensor.ndim < 2:
        return tensor
    anchor_mask = anchor_mask.to(device=tensor.device, dtype=torch.bool)
    text_mask = ~anchor_mask
    if tensor.ndim >= 2 and tuple(tensor.shape[:2]) == tuple(anchor_mask.shape):
        if not torch.equal(text_mask.sum(dim=1), text_mask.sum(dim=1)[:1].expand(text_mask.shape[0])):
            raise NotImplementedError("Safin-1 HSD requires equal text slot counts across batch rows.")
        text_count = int(text_mask[0].sum().item())
        return tensor[text_mask].view(tensor.shape[0], text_count, *tensor.shape[2:])
    if tensor.ndim >= 3 and tuple(tensor.shape[1:3]) == tuple(anchor_mask.shape):
        if not torch.equal(text_mask.sum(dim=1), text_mask.sum(dim=1)[:1].expand(text_mask.shape[0])):
            raise NotImplementedError("Safin-1 HSD requires equal text slot counts across batch rows.")
        rows = []
        for item in tensor:
            rows.append(item[text_mask].view(anchor_mask.shape[0], int(text_mask[0].sum().item()), *item.shape[2:]))
        return torch.stack(rows, dim=0)
    return tensor


def _filter_position_embeddings_to_text(
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    anchor_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if position_embeddings is None or not _anchor_has_tokens(anchor_mask):
        return position_embeddings
    return tuple(_filter_batch_sequence_to_text(item, anchor_mask) for item in position_embeddings)


def _scatter_text_hidden(
    base_hidden_states: torch.Tensor,
    text_hidden_states: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    anchor_mask = anchor_mask.to(device=base_hidden_states.device, dtype=torch.bool)
    text_mask = ~anchor_mask
    text_counts = text_mask.sum(dim=1)
    if not torch.equal(text_counts, text_counts[:1].expand_as(text_counts)):
        raise NotImplementedError("Safin-1 HSD requires equal text slot counts across batch rows.")
    text_count = int(text_counts[0].item())
    expected_shape = (base_hidden_states.shape[0], text_count, base_hidden_states.shape[-1])
    if tuple(text_hidden_states.shape) != expected_shape:
        raise ValueError(
            "HSD full-attention text output shape mismatch: "
            f"got {tuple(text_hidden_states.shape)}, expected {expected_shape}."
        )
    output = base_hidden_states.clone()
    output[text_mask] = text_hidden_states.reshape(-1, base_hidden_states.shape[-1]).to(dtype=output.dtype)
    return output


def _full_attention_mask_source(attention_mask: torch.Tensor | None, anchor_mask: torch.Tensor | None) -> torch.Tensor | None:
    if anchor_mask is None:
        return attention_mask
    if attention_mask is None:
        return None
    if not bool(anchor_mask.any().item()):
        if attention_mask.ndim == 2 and bool(attention_mask.all().item()):
            return None
        return attention_mask
    if attention_mask.ndim == 2 and bool(attention_mask.all().item()):
        return None
    if attention_mask.ndim == 2 and tuple(attention_mask.shape) == tuple(anchor_mask.shape):
        return _filter_batch_sequence_to_text(attention_mask, anchor_mask)
    if attention_mask.ndim == 2:
        return attention_mask
    raise NotImplementedError("Safin-1 HSD anchor tokens do not support custom or padded attention masks.")
