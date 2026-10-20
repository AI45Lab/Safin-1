# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native/text-cache and static-sidecar synchronization; no tensor ownership."""

from __future__ import annotations

import torch

from .runtime import _env_truthy, _is_valid_hsd_state_index


class StateCacheMixin:
    def _clear_native_state_for_request(self, state_idx: int) -> None:
        if not hasattr(self, "kv_cache") or not _is_valid_hsd_state_index(state_idx):
            return
        # A compact slot is model-owned and has no positional relationship to
        # the scheduler-managed native GDN block row. Clearing native caches
        # with a compact slot can corrupt a different live request during a
        # mixed decode/prefill admission step.
        if not self.hsd_use_compact_state_slots:
            if len(self.kv_cache) > 0:
                conv_cache = self.kv_cache[0]
                if conv_cache.numel() > 0 and state_idx < conv_cache.shape[0]:
                    conv_cache[state_idx].zero_()
            if len(self.kv_cache) > 1:
                ssm_cache = self.kv_cache[1]
                if ssm_cache.numel() > 0 and state_idx < ssm_cache.shape[0]:
                    ssm_cache[state_idx].zero_()
        for sidecar in self.hsd_persistent_state_sidecar.values():
            if torch.is_tensor(sidecar) and state_idx < sidecar.shape[0]:
                sidecar[state_idx].zero_()

    def _use_hsd_static_text_state_sidecar(self) -> bool:
        return (
            _env_truthy("VLLM_HSD_ENABLE_PADDED_STATIC_DECODE")
            and _env_truthy("VLLM_HSD_ENABLE_UNSAFE_PADDED_STATIC_DECODE")
            and _env_truthy("VLLM_HSD_PADDED_STATIC_USE_SIDECAR")
            and not _env_truthy("VLLM_HSD_DISABLE_TEXT_STATE_CACHE")
            and not _env_truthy("VLLM_HSD_DISABLE_STATIC_TEXT_STATE_SIDECAR")
        )

    def _ensure_hsd_static_text_state_sidecar(
        self,
        row_count: int,
    ) -> bool:
        if not self._use_hsd_static_text_state_sidecar() or row_count <= 0:
            return False
        conv_cache = self._conv_cache_for_kernel()
        ssm_cache = self._ssm_cache_for_kernel()
        if conv_cache is None or ssm_cache is None:
            return False

        rows = int(row_count)

        def ensure_tensor(
            name: str,
            shape: tuple[int, ...],
            dtype: torch.dtype,
            device: torch.device,
        ) -> torch.Tensor:
            current = self.hsd_static_text_state_sidecar.get(name)
            if (
                torch.is_tensor(current)
                and tuple(current.shape) == shape
                and current.dtype == dtype
                and current.device == device
            ):
                return current
            new_tensor = torch.zeros(shape, dtype=dtype, device=device)
            if torch.is_tensor(current) and current.ndim == len(shape):
                copy_rows = min(int(current.shape[0]), int(shape[0]))
                if copy_rows > 0 and tuple(current.shape[1:]) == tuple(shape[1:]):
                    new_tensor[:copy_rows].copy_(
                        current[:copy_rows].to(device=device, dtype=dtype)
                    )
            self.hsd_static_text_state_sidecar[name] = new_tensor
            return new_tensor

        ensure_tensor(
            "conv_state",
            (rows, *tuple(conv_cache.shape[1:])),
            conv_cache.dtype,
            conv_cache.device,
        )
        ensure_tensor(
            "recurrent_state",
            (rows, *tuple(ssm_cache.shape[1:])),
            ssm_cache.dtype,
            ssm_cache.device,
        )
        if len(self.hsd_static_text_state_sidecar_state_indices) < rows:
            self.hsd_static_text_state_sidecar_state_indices.extend(
                [-1] * (rows - len(self.hsd_static_text_state_sidecar_state_indices))
            )
        if len(self.hsd_static_text_state_sidecar_past_lens) < rows:
            self.hsd_static_text_state_sidecar_past_lens.extend(
                [-1] * (rows - len(self.hsd_static_text_state_sidecar_past_lens))
            )
        return True

    def prepare_hsd_static_decode_sidecar(
        self,
        state_indices: torch.Tensor,
        valid_mask: torch.Tensor | None,
        past_lens: torch.Tensor | None = None,
    ) -> bool:
        row_count = int(state_indices.numel())
        if not self._ensure_hsd_static_text_state_sidecar(row_count):
            return False
        conv_sidecar = self.hsd_static_text_state_sidecar.get("conv_state")
        recurrent_sidecar = self.hsd_static_text_state_sidecar.get("recurrent_state")
        if not torch.is_tensor(conv_sidecar) or not torch.is_tensor(recurrent_sidecar):
            return False
        valid_values = (
            [True] * row_count
            if valid_mask is None
            else [bool(v) for v in valid_mask[:row_count].detach().cpu().tolist()]
        )
        state_values = [
            int(v) for v in state_indices[:row_count].detach().cpu().tolist()
        ]
        past_values = (
            [-1] * row_count
            if past_lens is None
            else [int(v) for v in past_lens[:row_count].detach().cpu().tolist()]
        )
        old_state_indices = list(
            self.hsd_static_text_state_sidecar_state_indices[:row_count]
        )
        old_past_lens = list(self.hsd_static_text_state_sidecar_past_lens[:row_count])
        old_row_by_state = {
            int(state_idx): int(row)
            for row, state_idx in enumerate(old_state_indices)
            if _is_valid_hsd_state_index(int(state_idx))
        }
        remap_needed = any(
            valid
            and _is_valid_hsd_state_index(state_idx)
            and (
                old_state_indices[row] != state_idx
                or (
                    past_lens is not None
                    and int(old_past_lens[row]) != int(past_values[row])
                )
            )
            and int(state_idx) in old_row_by_state
            and (
                past_lens is None
                or int(old_past_lens[old_row_by_state[int(state_idx)]])
                == int(past_values[row])
            )
            for row, (state_idx, valid) in enumerate(zip(state_values, valid_values))
        )
        old_conv_sidecar = (
            conv_sidecar[:row_count].detach().clone() if remap_needed else None
        )
        old_recurrent_sidecar = (
            recurrent_sidecar[:row_count].detach().clone() if remap_needed else None
        )
        initialized_rows = 0
        remapped_rows = 0
        stale_rows = 0
        for row, (state_idx, valid) in enumerate(zip(state_values, valid_values)):
            if not valid or not _is_valid_hsd_state_index(state_idx):
                self.hsd_static_text_state_sidecar_state_indices[row] = -1
                self.hsd_static_text_state_sidecar_past_lens[row] = -1
                conv_sidecar[row].zero_()
                recurrent_sidecar[row].zero_()
                initialized_rows += 1
                continue
            if self.hsd_static_text_state_sidecar_state_indices[row] == state_idx and (
                past_lens is None
                or int(self.hsd_static_text_state_sidecar_past_lens[row])
                == int(past_values[row])
            ):
                continue
            old_row = old_row_by_state.get(int(state_idx))
            if (
                old_row is not None
                and old_conv_sidecar is not None
                and old_recurrent_sidecar is not None
                and (
                    past_lens is None
                    or int(old_past_lens[old_row]) == int(past_values[row])
                )
            ):
                cached_conv = old_conv_sidecar[old_row]
                cached_recurrent = old_recurrent_sidecar[old_row]
                cached_recurrent_is_native = True
                remapped_rows += 1
            else:
                if (
                    self.hsd_static_text_state_sidecar_state_indices[row] == state_idx
                    and past_lens is not None
                ):
                    stale_rows += 1
                cached = self.hsd_text_state_cache.get(state_idx)
                cached_conv = (
                    cached.get("conv_state")
                    if cached is not None and torch.is_tensor(cached.get("conv_state"))
                    else self._conv_state_for_request(state_idx)
                )
                cached_recurrent = (
                    cached.get("recurrent_state")
                    if cached is not None
                    and torch.is_tensor(cached.get("recurrent_state"))
                    else self._stored_recurrent_state_for_request(state_idx)
                )
                cached_recurrent_is_native = False
            if torch.is_tensor(cached_conv):
                conv_sidecar[row].copy_(
                    cached_conv.squeeze(0).to(
                        device=conv_sidecar.device,
                        dtype=conv_sidecar.dtype,
                    )
                )
            else:
                conv_sidecar[row].zero_()
            if torch.is_tensor(cached_recurrent):
                recurrent_value = cached_recurrent.squeeze(0)
                if not cached_recurrent_is_native:
                    recurrent_value = recurrent_value.transpose(-1, -2).contiguous()
                recurrent_sidecar[row].copy_(
                    recurrent_value.to(
                        device=recurrent_sidecar.device,
                        dtype=recurrent_sidecar.dtype,
                    )
                )
            else:
                recurrent_sidecar[row].zero_()
            self.hsd_static_text_state_sidecar_state_indices[row] = state_idx
            self.hsd_static_text_state_sidecar_past_lens[row] = int(past_values[row])
            initialized_rows += 1
        if initialized_rows:
            self._record_hsd_varlen_event(
                "hsd_static_text_state_sidecar_prepare",
                requests=row_count,
                initialized_rows=initialized_rows,
                remapped_rows=remapped_rows,
                stale_rows=stale_rows,
            )
        return True

    def sync_hsd_static_decode_sidecar_to_text_cache(
        self,
        state_indices: torch.Tensor,
    ) -> None:
        if not self._use_hsd_static_text_state_sidecar():
            return
        conv_sidecar = self.hsd_static_text_state_sidecar.get("conv_state")
        recurrent_sidecar = self.hsd_static_text_state_sidecar.get("recurrent_state")
        if not torch.is_tensor(conv_sidecar) or not torch.is_tensor(recurrent_sidecar):
            return
        state_values = [int(v) for v in state_indices.detach().cpu().tolist()]
        synced = 0
        for state_idx in state_values:
            if not _is_valid_hsd_state_index(state_idx):
                continue
            try:
                row = self.hsd_static_text_state_sidecar_state_indices.index(state_idx)
            except ValueError:
                continue
            self._store_conv_state(state_idx, conv_sidecar[row : row + 1])
            self._store_recurrent_state(
                state_idx,
                recurrent_sidecar[row : row + 1].transpose(-1, -2).contiguous(),
            )
            self._update_hsd_text_state_cache(state_idx)
            synced += 1
        if synced:
            self._record_hsd_varlen_event(
                "hsd_static_text_state_sidecar_sync",
                requests=len(state_values),
                synced_rows=synced,
            )

    def _restore_hsd_static_text_state_sidecar_batched(
        self,
        state_indices: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> bool:
        row_count = int(state_indices.numel())
        if row_count <= 0 or not self._ensure_hsd_static_text_state_sidecar(row_count):
            return False
        conv_sidecar = self.hsd_static_text_state_sidecar.get("conv_state")
        recurrent_sidecar = self.hsd_static_text_state_sidecar.get("recurrent_state")
        conv_cache = self._conv_cache_for_kernel()
        ssm_cache = self._ssm_cache_for_kernel()
        if (
            not torch.is_tensor(conv_sidecar)
            or not torch.is_tensor(recurrent_sidecar)
            or conv_cache is None
            or ssm_cache is None
            or int(conv_sidecar.shape[0]) < row_count
            or int(recurrent_sidecar.shape[0]) < row_count
        ):
            return False
        indices = state_indices.to(device=conv_cache.device, dtype=torch.long)
        conv_values = conv_sidecar[:row_count].to(
            device=conv_cache.device,
            dtype=conv_cache.dtype,
        )
        recurrent_values = recurrent_sidecar[:row_count].to(
            device=ssm_cache.device,
            dtype=ssm_cache.dtype,
        )
        if valid_mask is not None:
            conv_valid = valid_mask.to(
                device=conv_values.device, dtype=torch.bool
            ).view(
                row_count,
                *([1] * (conv_values.ndim - 1)),
            )
            recurrent_valid = valid_mask.to(
                device=recurrent_values.device,
                dtype=torch.bool,
            ).view(row_count, *([1] * (recurrent_values.ndim - 1)))
            conv_values = torch.where(
                conv_valid, conv_values, torch.zeros_like(conv_values)
            )
            recurrent_values = torch.where(
                recurrent_valid,
                recurrent_values,
                torch.zeros_like(recurrent_values),
            )
        conv_cache.index_copy_(0, indices, conv_values)
        ssm_cache.index_copy_(0, indices.to(device=ssm_cache.device), recurrent_values)
        return True

    def _update_hsd_static_text_state_sidecar_batched(
        self,
        state_indices: torch.Tensor,
        valid_mask: torch.Tensor | None,
        past_lens: torch.Tensor | None = None,
    ) -> bool:
        row_count = int(state_indices.numel())
        if row_count <= 0 or not self._ensure_hsd_static_text_state_sidecar(row_count):
            return False
        conv_sidecar = self.hsd_static_text_state_sidecar.get("conv_state")
        recurrent_sidecar = self.hsd_static_text_state_sidecar.get("recurrent_state")
        conv_cache = self._conv_cache_for_kernel()
        ssm_cache = self._ssm_cache_for_kernel()
        if (
            not torch.is_tensor(conv_sidecar)
            or not torch.is_tensor(recurrent_sidecar)
            or conv_cache is None
            or ssm_cache is None
            or int(conv_sidecar.shape[0]) < row_count
            or int(recurrent_sidecar.shape[0]) < row_count
        ):
            return False
        indices = state_indices.to(device=conv_cache.device, dtype=torch.long)
        conv_values = conv_cache.index_select(0, indices).to(
            device=conv_sidecar.device,
            dtype=conv_sidecar.dtype,
        )
        recurrent_values = ssm_cache.index_select(
            0,
            indices.to(device=ssm_cache.device),
        ).to(device=recurrent_sidecar.device, dtype=recurrent_sidecar.dtype)
        if valid_mask is not None:
            conv_valid = valid_mask.to(
                device=conv_values.device, dtype=torch.bool
            ).view(
                row_count,
                *([1] * (conv_values.ndim - 1)),
            )
            recurrent_valid = valid_mask.to(
                device=recurrent_values.device,
                dtype=torch.bool,
            ).view(row_count, *([1] * (recurrent_values.ndim - 1)))
            conv_values = torch.where(
                conv_valid, conv_values, torch.zeros_like(conv_values)
            )
            recurrent_values = torch.where(
                recurrent_valid,
                recurrent_values,
                torch.zeros_like(recurrent_values),
            )
        conv_sidecar[:row_count].copy_(conv_values)
        recurrent_sidecar[:row_count].copy_(recurrent_values)
        if past_lens is not None:
            past_values = [
                int(v) for v in past_lens[:row_count].detach().cpu().tolist()
            ]
            valid_values = (
                [True] * row_count
                if valid_mask is None
                else [bool(v) for v in valid_mask[:row_count].detach().cpu().tolist()]
            )
            for row, valid in enumerate(valid_values):
                self.hsd_static_text_state_sidecar_past_lens[row] = (
                    int(past_values[row]) + 1 if valid else -1
                )
        return True

    def _restore_hsd_text_state_cache(
        self,
        state_idx: int,
        device: torch.device,
    ) -> None:
        if not _is_valid_hsd_state_index(state_idx):
            return
        if _env_truthy("VLLM_HSD_DISABLE_TEXT_STATE_CACHE"):
            self._record_hsd_text_state_cache_debug(
                "restore_disabled",
                state_idx,
                cache_hit=False,
                native_conv_state=self._conv_state_for_request(state_idx),
                native_recurrent_state=self._stored_recurrent_state_for_request(
                    state_idx
                ),
            )
            return
        cached = self.hsd_text_state_cache.get(state_idx)
        cached_conv = (
            cached.get("conv_state")
            if cached is not None and torch.is_tensor(cached.get("conv_state"))
            else None
        )
        cached_recurrent = (
            cached.get("recurrent_state")
            if cached is not None and torch.is_tensor(cached.get("recurrent_state"))
            else None
        )
        self._record_hsd_text_state_cache_debug(
            "restore_before",
            state_idx,
            cache_hit=cached is not None,
            cached_conv_state=cached_conv,
            cached_recurrent_state=cached_recurrent,
            native_conv_state=self._conv_state_for_request(state_idx),
            native_recurrent_state=self._stored_recurrent_state_for_request(state_idx),
        )
        if cached is None:
            return
        conv_state = cached.get("conv_state")
        if torch.is_tensor(conv_state):
            self._store_conv_state(
                state_idx,
                conv_state.to(device=device),
            )
        recurrent_state = cached.get("recurrent_state")
        if torch.is_tensor(recurrent_state):
            self._store_recurrent_state(
                state_idx,
                recurrent_state.to(device=device),
            )
        self._record_hsd_text_state_cache_debug(
            "restore_after",
            state_idx,
            cache_hit=True,
            cached_conv_state=cached_conv,
            cached_recurrent_state=cached_recurrent,
            native_conv_state=self._conv_state_for_request(state_idx),
            native_recurrent_state=self._stored_recurrent_state_for_request(state_idx),
        )
        if (
            _env_truthy("VLLM_HSD_PREFIX_CACHE_DEBUG")
            and torch.is_tensor(conv_state)
            and torch.is_tensor(recurrent_state)
        ):
            self._record_hsd_varlen_event(
                "hsd_prefix_text_state_cache_restore_debug",
                state_idx=int(state_idx),
                recurrent_norm=float(recurrent_state.detach().float().norm().item()),
                recurrent_sum=float(recurrent_state.detach().float().sum().item()),
                conv_norm=float(conv_state.detach().float().norm().item()),
                conv_sum=float(conv_state.detach().float().sum().item()),
            )

    def _restore_hsd_text_state_cache_batched(
        self,
        state_indices: torch.Tensor,
        device: torch.device,
    ) -> bool:
        if _env_truthy("VLLM_HSD_DISABLE_BATCHED_TEXT_STATE_CACHE"):
            return False
        if _env_truthy("VLLM_HSD_DISABLE_TEXT_STATE_CACHE"):
            return False
        conv_cache = self._conv_cache_for_kernel()
        ssm_cache = self._ssm_cache_for_kernel()
        if conv_cache is None or ssm_cache is None:
            return False
        if int(state_indices.numel()) <= 1:
            return False

        restore_indices: list[int] = []
        conv_values: list[torch.Tensor] = []
        recurrent_values: list[torch.Tensor] = []
        for raw_state_idx in state_indices.detach().cpu().tolist():
            state_idx = int(raw_state_idx)
            if not _is_valid_hsd_state_index(state_idx):
                continue
            cached = self.hsd_text_state_cache.get(state_idx)
            if cached is None:
                continue
            cached_conv = cached.get("conv_state")
            cached_recurrent = cached.get("recurrent_state")
            if not torch.is_tensor(cached_conv) or not torch.is_tensor(
                cached_recurrent
            ):
                continue
            restore_indices.append(state_idx)
            conv_values.append(cached_conv.squeeze(0))
            recurrent_values.append(cached_recurrent.squeeze(0))
        if not restore_indices:
            return True

        index_tensor = torch.tensor(
            restore_indices,
            dtype=torch.long,
            device=conv_cache.device,
        )
        conv_stacked = torch.stack(conv_values, dim=0).to(
            device=conv_cache.device,
            dtype=conv_cache.dtype,
        )
        conv_cache.index_copy_(0, index_tensor, conv_stacked)

        recurrent_stacked = (
            torch.stack(recurrent_values, dim=0).transpose(-1, -2).contiguous()
        )
        recurrent_stacked = recurrent_stacked.to(
            device=ssm_cache.device,
            dtype=ssm_cache.dtype,
        )
        ssm_cache.index_copy_(
            0,
            index_tensor.to(device=ssm_cache.device),
            recurrent_stacked,
        )
        self._record_hsd_varlen_event(
            "batched_text_state_cache_restore_hit",
            requests=int(state_indices.numel()),
            restored_rows=len(restore_indices),
        )
        return True

    def _update_hsd_text_state_cache(self, state_idx: int) -> None:
        if not _is_valid_hsd_state_index(state_idx):
            return
        conv_state = self._conv_state_for_request(state_idx)
        recurrent_state = self._stored_recurrent_state_for_request(state_idx)
        if _env_truthy("VLLM_HSD_DISABLE_TEXT_STATE_CACHE"):
            self.hsd_text_state_cache.pop(state_idx, None)
            self._record_hsd_text_state_cache_debug(
                "update_disabled",
                state_idx,
                cache_hit=False,
                native_conv_state=conv_state,
                native_recurrent_state=recurrent_state,
            )
            return
        cached = self.hsd_text_state_cache.get(state_idx)
        if cached is None:
            cached = {}
            self.hsd_text_state_cache[state_idx] = cached

        def update_cached_tensor(name: str, value: torch.Tensor | None) -> None:
            if not torch.is_tensor(value):
                cached.pop(name, None)
                return
            detached = value.detach()
            current = cached.get(name)
            if (
                torch.is_tensor(current)
                and tuple(current.shape) == tuple(detached.shape)
                and current.dtype == detached.dtype
                and current.device == detached.device
            ):
                current.copy_(detached)
            else:
                cached[name] = detached.clone()

        update_cached_tensor("conv_state", conv_state)
        update_cached_tensor("recurrent_state", recurrent_state)
        if not cached:
            self.hsd_text_state_cache.pop(state_idx, None)
            return
        self._record_hsd_text_state_cache_debug(
            "update",
            state_idx,
            cache_hit=True,
            cached_conv_state=cached.get("conv_state"),
            cached_recurrent_state=cached.get("recurrent_state"),
            native_conv_state=conv_state,
            native_recurrent_state=recurrent_state,
        )

    def _update_hsd_text_state_cache_batched(
        self,
        state_indices: torch.Tensor,
    ) -> bool:
        if _env_truthy("VLLM_HSD_DISABLE_BATCHED_TEXT_STATE_CACHE"):
            return False
        if _env_truthy("VLLM_HSD_DISABLE_TEXT_STATE_CACHE"):
            return False
        conv_cache = self._conv_cache_for_kernel()
        ssm_cache = self._ssm_cache_for_kernel()
        if conv_cache is None or ssm_cache is None:
            return False
        if int(state_indices.numel()) <= 1:
            return False

        state_values = [
            int(v)
            for v in state_indices.detach().cpu().tolist()
            if _is_valid_hsd_state_index(int(v))
        ]
        if not state_values:
            return True
        index_tensor = torch.tensor(
            state_values,
            dtype=torch.long,
            device=conv_cache.device,
        )
        conv_values = conv_cache.index_select(0, index_tensor)
        recurrent_values = (
            ssm_cache.index_select(
                0,
                index_tensor.to(device=ssm_cache.device),
            )
            .transpose(-1, -2)
            .contiguous()
        )

        for row, state_idx in enumerate(state_values):
            cached = self.hsd_text_state_cache.get(state_idx)
            if cached is None:
                cached = {}
                self.hsd_text_state_cache[state_idx] = cached

            conv_value = conv_values[row : row + 1]
            conv_value = conv_value.detach()
            current_conv = cached.get("conv_state")
            if (
                torch.is_tensor(current_conv)
                and tuple(current_conv.shape) == tuple(conv_value.shape)
                and current_conv.dtype == conv_value.dtype
                and current_conv.device == conv_value.device
            ):
                current_conv.copy_(conv_value)
            else:
                cached["conv_state"] = conv_value.clone()

            recurrent_value = recurrent_values[row : row + 1].detach()
            current_recurrent = cached.get("recurrent_state")
            if (
                torch.is_tensor(current_recurrent)
                and tuple(current_recurrent.shape) == tuple(recurrent_value.shape)
                and current_recurrent.dtype == recurrent_value.dtype
                and current_recurrent.device == recurrent_value.device
            ):
                current_recurrent.copy_(recurrent_value)
            else:
                cached["recurrent_state"] = recurrent_value.clone()

        self._record_hsd_varlen_event(
            "batched_text_state_cache_update_hit",
            requests=int(state_indices.numel()),
            updated_rows=len(state_values),
        )
        return True
