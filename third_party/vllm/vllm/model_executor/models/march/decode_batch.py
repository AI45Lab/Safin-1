# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered eager batched decode, including padded and anchor-boundary rows.

False remains the original outer fallback result. A no-anchor helper returning
None continues from the already-computed recurrence; it never replays that
update. Frozen stage objects retain the same tensor/cache references.
"""

from __future__ import annotations

import torch

from .decode_batch_core import BatchDecodeCoreMixin
from .decode_batch_no_anchor import BatchDecodeNoAnchorMixin
from .decode_batch_plan import BatchDecodePlanMixin
from .decode_batch_reference import BatchDecodeReferenceMixin
from .layout import HistoricalHSDLayout


class BatchedDecodeMixin(
    BatchDecodePlanMixin,
    BatchDecodeCoreMixin,
    BatchDecodeNoAnchorMixin,
    BatchDecodeReferenceMixin,
):
    def _forward_hsd_batched_decode_light(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layout: HistoricalHSDLayout,
    ) -> bool:
        prepared = self._prepare_hsd_batched_decode(hidden_states, output, layout)
        if prepared is False:
            return False
        core = self._prepare_hsd_batched_decode_core(prepared)
        boundary = self._append_hsd_batched_decode_boundaries(prepared, core)
        handled = self._try_hsd_batched_decode_no_anchor(prepared, core)
        if handled is not None:
            return handled
        reference = self._run_hsd_batched_decode_reference(prepared, core, boundary)
        if reference is False:
            return False
        return self._finalize_hsd_batched_decode_reference(prepared, core, reference)
